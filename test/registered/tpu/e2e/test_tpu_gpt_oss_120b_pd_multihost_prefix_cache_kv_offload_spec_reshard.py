"""GPT-OSS-120B 2P+2D prefix cache + KV offload + EAGLE3, with heterogeneous
attention TP so the KV crosses Raiden's push-reshard (issue #379).

Same cluster as test_tpu_gpt_oss_120b_pd_multihost_prefix_cache_kv_offload_spec
with one difference: the decode engine keeps tp=16 but splits into dp=2, so each
of its ranks owns twice the kv-head slice a prefill rank owns and is fed by two
of them. That 2:1 FAN_IN puts every transfer on Raiden's strided push-reshard
instead of a rank-for-rank landing.

Its own file rather than a flag on that gate: the gate runs nightly and its
GSM8K baseline is only meaningful for the shape it was measured on, so an edit
there would need re-validating on hardware before it could be trusted. Same
trade test_tpu_gpt_oss_120b_pd_multihost_tp_resharding.py already makes.

Needs #592 (push-resharding for hybrid SWA pools; before it an asymmetric
GPT-OSS run could only raise) and #583 (draft KV over the P/D transfer).

Two assertions make the row falsifiable, because through the mini-LB a resharded
cluster looks exactly like a symmetric one:
  * assert_decode_reshard_topology reads the decode engine's resolved ServerArgs
    rather than trusting the env handed to a launcher two hops upstream. An
    earlier iteration of this row ran green against a symmetric engine.
  * MIN_SPEC_ACCEPT_LENGTH is a real floor here. Losing draft KV through the
    reshard presents as acceptance collapsing toward 1.0 with everything else
    still green.
"""

import argparse
import concurrent.futures
import enum
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from typing import Dict, List, Mapping, Optional, Tuple

SGLANG_IMPORT_ERROR = None
try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.tpu import model_config_main
except ModuleNotFoundError as exc:
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def model_config_main(*_args, **_kwargs):
        return None


register_tpu_ci(
    est_time=10800,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "multihost",
        "gpt-oss-120b",
        "pd",
        "prefix-cache",
        "kv-offload",
        "spec",
        "reshard",
    ],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
WAIT_SECS = int(E("WAIT_SECS", "3600"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")


class Role(str, enum.Enum):
    """Execution roles in the P/D cluster."""

    PREFILL = "prefill"
    DECODE = "decode"


PREFILL_LOG = "/tmp/pd_prefill.log"
DECODE_LOG = "/tmp/pd_decode.log"
LB_LOG = "/tmp/pd_lb.log"

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
HF_REPO_ID = "openai/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"
DRAFT_MODEL_ENV = "GPT_OSS_DRAFT_MODEL"
DEFAULT_DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

PAGE_SIZE = 64
# EAGLE3 loads the target + draft model and captures verify/draft graphs on top,
# so give it the same headroom as the passing single-host P/D spec gate.
MEM_FRACTION_PREFILL = "0.6"
MEM_FRACTION_DECODE = "0.6"
QUANTIZATION = ""
# bf16 KV, matching the single-host P/D EAGLE3 gate (the spec verify path is
# validated against bf16, and the pool stays small via MAX_TOTAL_TOKENS).
KV_CACHE_DTYPE = "bfloat16"
MAX_TOTAL_TOKENS = 8192
MAX_RUNNING_REQUESTS = 16
DISAGG_TRANSFER_BACKEND = "raiden"
HICACHE_RATIO = "2"
CACHE_TARGET_WORDS = int(E("CACHE_PROBE_TARGET_WORDS", "2048"))
CACHE_FILLER_WORDS = int(E("CACHE_PROBE_FILLER_WORDS", "2048"))
CACHE_MAX_NEW_TOKENS = int(E("CACHE_PROBE_MAX_NEW_TOKENS", "64"))
# A floor of 1.0 cannot fail: accept_length is >= 1 by construction because the
# bonus token lands even when every draft token is rejected. Losing draft KV
# through the reshard -- the regression #583 fixed on the rank-for-rank path,
# and the likeliest to reappear here -- presents as acceptance collapsing toward
# 1.0 while everything else stays green, so this row needs a real one. Measured
# here: probe 3.2 (64 tokens / 20 verify steps), GSM8K 2.80 mean over 200,
# against 2.7786 standalone. 2.5 is ~78% of the probe figure.
MIN_SPEC_ACCEPT_LENGTH = float(E("MIN_SPEC_ACCEPT_LENGTH", "2.5"))
# Held under MAX_RUNNING_REQUESTS so the wave lands in one engine batch rather
# than queueing into separate ones, which is what makes it a batching test.
CONCURRENT_WAVE_SIZE = int(E("CONCURRENT_WAVE_SIZE", "8"))
EXTRA_SERVER_ARGS: Tuple[str, ...] = ("--reasoning-parser", "gpt-oss")
# Bound the compile grid + allow spec recompiles: the unbounded bs<=max_running
# verify-graph capture spikes host RAM and OOM-kills the decode. RESHARD off:
# symmetric prefill/decode TP needs no strided reshard.
EXTRA_LAUNCH_ENV: Dict[str, str] = {
    "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
    "TPU_COMPILE_BS": "1 2 4 16",
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
    # The feature under test. Raiden engages the strided reshard only when a
    # decode rank is fed by more than one prefill rank, and raises rather than
    # falling back when it cannot -- so there is no silent symmetric path.
    "SGLANG_RAIDEN_RESHARD": "1",
}

# launch_server.sh turns DP_SIZE>1 into --dp-size + --enable-dp-attention (plus
# --enable-dp-lm-head, which the TPU platform requires at attn_tp > 1).
DECODE_DP_SIZE = int(E("PD_DECODE_DP_SIZE", "2"))

# EAGLE3 speculative decoding, shared by both engines: the P/D handoff carries
# the spec bootstrap state, so the prefill loads the draft model too.
SPEC_NUM_STEPS = 3
SPEC_EAGLE_TOPK = 1
SPEC_NUM_DRAFT_TOKENS = 4

# Measured on THIS topology over three runs: 0.950, 0.965, 0.970. The symmetric
# gate lands in the same range, so gating here at the same level makes GSM8K a
# comparison against the non-resharded reference rather than a floor any working
# engine clears -- a subtle reshard numerical error would never move a 0.72
# floor. 0.92 with the default tolerance enforces 0.89, about four standard
# errors below the lowest observed score at n=200.
#
# The sibling files quote 0.785 from the single-host gate and set 0.75; that
# figure matches nothing this topology produces, so this row uses its own
# measurements, under its own env var -- tuning one row should not silently
# retune the others sharing GSM8K_THRESHOLD_GPT_OSS_120B_SPEC.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B_SPEC_RESHARD", "0.92"))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "200"))
# Match client concurrency to MAX_RUNNING_REQUESTS and the largest
# compiled batch.
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "16"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))


def resolve_draft_model_path(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    return values.get(DRAFT_MODEL_ENV) or DEFAULT_DRAFT_MODEL_PATH


def _spec_server_args(env: Optional[Mapping[str, str]] = None) -> List[str]:
    return [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        resolve_draft_model_path(env),
        "--speculative-num-steps",
        str(SPEC_NUM_STEPS),
        "--speculative-eagle-topk",
        str(SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens",
        str(SPEC_NUM_DRAFT_TOKENS),
        "--watchdog-timeout",
        "7200",
    ]


MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-pd-multihost-prefix-cache-kv-offload-spec-reshard-gsm8k",
    "variant": "PD-2host-TP16+EP16-raiden+EAGLE3+prefixcache+kvoffload+reshard(attn_tp16->8)",
    "launch": {
        "external": True,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-120b-pd-pc-offload-spec-reshard-gsm8k-e2e",
    },
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_BASELINE,
            "atol": ATOL,
            "num_examples": NUM_EXAMPLES,
            "num_threads": NUM_THREADS,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }
    ],
}


class PodTopology:
    """Topology parsed from slice-local environment."""

    def __init__(
        self,
        role: str,
        group_rank: int,
        group_hosts: int,
        group_hostnames: Tuple[str, ...],
    ) -> None:
        self.role = role
        self.group_rank = group_rank
        self.group_hosts = group_hosts
        self.group_hostnames = group_hostnames

    @property
    def is_prefill_leader(self) -> bool:
        return self.role == Role.PREFILL.value and self.group_rank == 0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "PodTopology":
        values = os.environ if env is None else env
        role = values.get("CI_PD_ROLE", "")
        valid_roles = {r.value for r in Role}
        if role not in valid_roles:
            raise ValueError(f"CI_PD_ROLE must be one of {valid_roles}, got {role!r}")
        hostnames = tuple(
            h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h
        )
        if not hostnames:
            hostnames = (socket.gethostname(),)
        return cls(
            role=role,
            group_rank=int(values.get("TPU_WORKER_ID", "0")),
            group_hosts=len(hostnames),
            group_hostnames=hostnames,
        )


def get_role(topology: PodTopology) -> Role:
    """Resolve strongly-typed Role from pod topology."""
    return Role(topology.role)


def peer_group_hostname(hostname: str, from_role: str, to_role: str) -> str:
    """Map a pod hostname of one replicated job onto its peer job pod."""
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


def resolve_model_path(env: Optional[Mapping[str, str]] = None) -> str:
    """Resolve model path from environment or default constant."""
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Validate model checkpoint directory exists when path is local."""
    if not os.path.isabs(model_path):
        return
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"GPT-OSS-120B checkpoint incomplete or missing at {model_path}; "
            f"provision it on the weights volume or set {MODEL_ENV}"
        )


def _common_launch_env(
    group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    return {
        "MODEL": resolve_model_path(values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "PAGE_SIZE": str(PAGE_SIZE),
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": QUANTIZATION,
        "KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        # Keep radix caching available to the prefill engine, where shared
        # prefixes are matched. The decode engine's disaggregation setting still
        # leaves its own radix cache off, avoiding the spec-incompatible path.
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "CONTEXT_LEN": values.get("CONTEXT_LEN", "8192"),
        "CHUNKED_PREFILL_SIZE": values.get("CHUNKED_PREFILL_SIZE", "4096"),
        "MAX_PREFILL_TOKENS": values.get("MAX_PREFILL_TOKENS", "4096"),
        "MAX_RUNNING": values.get("MAX_RUNNING", str(MAX_RUNNING_REQUESTS)),
        "TORCH_COMPILE_MAX_BS": values.get(
            "TORCH_COMPILE_MAX_BS", str(MAX_RUNNING_REQUESTS)
        ),
        "PYTHON": PYTHON,
        "SGLANG_RAIDEN_KV_PORT": values.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        **EXTRA_LAUNCH_ENV,
    }


def build_prefill_env(
    group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Construct environment dictionary for prefill engine."""
    values = os.environ if env is None else env
    served = ("--served-model-name", SERVED_MODEL_NAME)
    extra_args = [
        *served,
        *EXTRA_SERVER_ARGS,
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        values.get("HICACHE_RATIO", HICACHE_RATIO),
        "--hicache-io-backend",
        "raiden",
        "--hicache-mem-layout",
        "page_first_fused",
        "--enable-cache-report",
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
        *_spec_server_args(values),
    ]
    if MAX_TOTAL_TOKENS is not None:
        extra_args += ["--max-total-tokens", str(MAX_TOTAL_TOKENS)]
    if values.get("EXTRA_PREFILL_ARGS"):
        extra_args.append(values["EXTRA_PREFILL_ARGS"])

    return {
        **_common_launch_env(group_hosts, values),
        "MEM_FRACTION_STATIC": values.get("MEM_FRACTION_STATIC", MEM_FRACTION_PREFILL),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def build_decode_env(
    group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Construct environment dictionary for decode engine."""
    values = os.environ if env is None else env
    decode_args = ["--served-model-name", SERVED_MODEL_NAME]
    if MAX_TOTAL_TOKENS is not None:
        decode_args += ["--max-total-tokens", str(MAX_TOTAL_TOKENS)]

    extra_args = [
        *decode_args,
        *EXTRA_SERVER_ARGS,
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        # No raiden hicache retraction backup here: the single-host P/D EAGLE3
        # gate omits it (spec draft KV is not offloaded), so keep the decode
        # engine's memory profile matched to what is validated with spec.
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
        *_spec_server_args(values),
    ]
    if values.get("EXTRA_DECODE_ARGS"):
        extra_args.append(values["EXTRA_DECODE_ARGS"])

    return {
        **_common_launch_env(group_hosts, values),
        "MEM_FRACTION_STATIC": values.get("MEM_FRACTION_STATIC", MEM_FRACTION_DECODE),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
        # The asymmetry: tp stays 16, dp splits it so attn_tp is 8.
        "DP_SIZE": str(int(values.get("DP_SIZE", str(DECODE_DP_SIZE)))),
        "ENABLE_DP_ATTENTION": "1",
    }


def _tee_command(argv: Tuple[str, ...], log_path: str) -> Tuple[str, ...]:
    quoted = " ".join(f"'{a}'" for a in argv)
    return ("bash", "-c", f"set -o pipefail; {quoted} 2>&1 | tee -a '{log_path}'")


def _launch(
    cmd: Tuple[str, ...],
    log_path: str,
    extra_env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        _tee_command(cmd, log_path=log_path),
        env=env,
        start_new_session=True,
        cwd=cwd,
    )


def execute_prefill(
    worker_id: int = 0,
    group_hosts: int = 2,
    env: Optional[Mapping[str, str]] = None,
    log_path: str = PREFILL_LOG,
) -> subprocess.Popen:
    """Launch prefill engine on specified worker rank."""
    launch_env = build_prefill_env(group_hosts, env)
    launch_env["NODE_RANK"] = str(worker_id)
    launch_env["TPU_WORKER_ID"] = str(worker_id)
    return _launch(
        ("bash", os.path.join(BENCH, "launch_prefill.sh")),
        log_path=log_path,
        extra_env=launch_env,
        cwd=BENCH,
    )


def execute_decode(
    worker_id: int = 0,
    group_hosts: int = 2,
    env: Optional[Mapping[str, str]] = None,
    log_path: str = DECODE_LOG,
) -> subprocess.Popen:
    """Launch decode engine on specified worker rank."""
    launch_env = build_decode_env(group_hosts, env)
    launch_env["NODE_RANK"] = str(worker_id)
    launch_env["TPU_WORKER_ID"] = str(worker_id)
    return _launch(
        ("bash", os.path.join(BENCH, "launch_decode.sh")),
        log_path=log_path,
        extra_env=launch_env,
        cwd=BENCH,
    )


def execute_router(
    prefill_host: str,
    decode_host: str,
    port: int = LB_PORT,
    prefill_port: int = PREFILL_PORT,
    decode_port: int = DECODE_PORT,
    log_path: str = LB_LOG,
) -> subprocess.Popen:
    """Launch mini-LB router connecting prefill and decode engines."""
    cmd = (
        PYTHON,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill",
        f"http://{prefill_host}:{prefill_port}",
        "--decode",
        f"http://{decode_host}:{decode_port}",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    )
    return _launch(cmd, log_path=log_path)


def _kill(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _http_ok(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
        return False


def _wait_http_200(url: str, secs: int, procs: Tuple = ()) -> None:
    deadline = time.time() + secs
    while time.time() < deadline:
        for p in procs:
            if p is not None and p.poll() is not None:
                raise RuntimeError(f"process for {url} exited rc={p.returncode}")
        if _http_ok(url):
            return
        time.sleep(10)
    raise TimeoutError(f"{url} not ready after {secs}s")


def _probe_prompt(index: int, words: int) -> str:
    """Build a deterministic, unique prompt large enough to occupy many pages."""
    line = f"document-{index} contains marker-{index * 17 + 11}. "
    return (line * max(1, words // 4)) + "\nRepeat the marker exactly:\n"


def _generate_probe(base_url: str, prompt: str, max_new_tokens: int) -> Dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read())
    meta = payload.get("meta_info") or {}
    details = meta.get("cached_tokens_details") or {}
    return {
        "text": payload.get("text") or "",
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "completion_tokens": int(meta.get("completion_tokens") or 0),
        "device_cached_tokens": int(details.get("device") or 0),
        "host_cached_tokens": int(details.get("host") or 0),
        "spec_verify_ct": int(meta.get("spec_verify_ct") or 0),
    }


def _assert_spec_active(record: Dict, label: str) -> None:
    verify_count = record["spec_verify_ct"]
    if verify_count <= 0:
        raise AssertionError(
            f"{label}: spec_verify_ct={verify_count}; speculative decoding did not run"
        )
    accept_length = record["completion_tokens"] / verify_count
    if accept_length <= MIN_SPEC_ACCEPT_LENGTH:
        raise AssertionError(
            f"{label}: speculative accept length {accept_length:.3f} <= "
            f"{MIN_SPEC_ACCEPT_LENGTH}"
        )


def run_cache_offload_spec_probe(base_url: str, flush_url: str) -> None:
    """Prove prefix KV survives host offload while EAGLE3 remains active."""
    flush_endpoint = flush_url.rstrip("/") + "/flush_cache"
    with urllib.request.urlopen(flush_endpoint, timeout=60) as r:
        r.read()

    target = _probe_prompt(0, CACHE_TARGET_WORDS)
    cold = _generate_probe(base_url, target, CACHE_MAX_NEW_TOKENS)
    device_hit = _generate_probe(base_url, target, CACHE_MAX_NEW_TOKENS)
    if cold["device_cached_tokens"] or cold["host_cached_tokens"]:
        raise AssertionError(f"cold request unexpectedly hit cache: {cold}")
    expected = max(PAGE_SIZE, (cold["prompt_tokens"] // PAGE_SIZE - 1) * PAGE_SIZE)
    if device_hit["device_cached_tokens"] < expected:
        raise AssertionError(
            "immediate resend missed the device prefix cache: "
            f"{device_hit['device_cached_tokens']} < {expected}"
        )

    first_filler = _generate_probe(base_url, _probe_prompt(1000, CACHE_FILLER_WORDS), 1)
    filler_tokens = first_filler["prompt_tokens"]
    if filler_tokens <= 0:
        raise AssertionError("cannot size cache churn from a zero-token filler")
    # This spec profile deliberately uses a tiny 8K device pool. Churning 2x
    # that pool (as the non-spec test does with its much larger host tier) can
    # evict the target from both tiers before we probe it. Cross device
    # capacity by the minimum measured amount instead: enough to force the
    # target out of HBM while retaining its host-tier copy.
    target_cached_tokens = device_hit["device_cached_tokens"]
    tokens_needed = max(1, MAX_TOTAL_TOKENS - target_cached_tokens + 1)
    num_fillers = (tokens_needed + filler_tokens - 1) // filler_tokens
    for index in range(1, num_fillers):
        _generate_probe(base_url, _probe_prompt(1000 + index, CACHE_FILLER_WORDS), 1)

    host_reload = _generate_probe(base_url, target, CACHE_MAX_NEW_TOKENS)
    post_reload_hit = _generate_probe(base_url, target, CACHE_MAX_NEW_TOKENS)
    print(
        json.dumps(
            {
                "pd_multihost_cache_offload_spec_probe": {
                    "cold": cold,
                    "device_hit": device_hit,
                    "host_reload": host_reload,
                    "post_reload_hit": post_reload_hit,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if host_reload["host_cached_tokens"] <= 0:
        raise AssertionError(
            "target did not reload from host; KV offload never engaged"
        )
    recovered_tokens = (
        host_reload["device_cached_tokens"] + host_reload["host_cached_tokens"]
    )
    if recovered_tokens < expected:
        raise AssertionError("host reload did not recover the complete cached prefix")
    if host_reload["text"] != device_hit["text"]:
        raise AssertionError("host-reloaded KV produced different greedy output")
    if post_reload_hit["text"] != device_hit["text"]:
        raise AssertionError("post-reload device hit produced different greedy output")
    for label, record in (
        ("device-hit", device_hit),
        ("host-reload", host_reload),
        ("post-reload-hit", post_reload_hit),
    ):
        _assert_spec_active(record, label)

    _assert_concurrent_wave_is_deterministic(base_url, target, device_hit["text"])


def _serve_worker(topology: PodTopology, role: Role) -> int:
    """Serve prefill or decode worker shard until termination."""
    if role == Role.DECODE:
        proc = execute_decode(
            worker_id=topology.group_rank, group_hosts=topology.group_hosts
        )
    else:
        proc = execute_prefill(
            worker_id=topology.group_rank, group_hosts=topology.group_hosts
        )

    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            time.sleep(10)
    finally:
        _kill(proc)


def assert_decode_reshard_topology(decode_host: str) -> None:
    """Fail unless the decode engine really is running the resharded shape.

    Everything else here is observed through the mini-LB, which looks identical
    whether or not the KV was resharded, so nothing else would notice this row
    degenerating into the symmetric gate it derives from. Not hypothetical: an
    earlier iteration ran green with a symmetric decode engine and proved
    nothing. Before #592 an accidental symmetric run at least raised on hybrid
    SWA; now that resharding works, it would simply pass.
    """
    url = f"http://{decode_host}:{DECODE_PORT}/server_info"
    with urllib.request.urlopen(url, timeout=60) as resp:
        info = json.loads(resp.read())
    attn_dp_size = int(info.get("attn_dp_size") or 1)
    dp_attention = bool(info.get("enable_dp_attention"))
    tp_size = int(info.get("tp_size") or 0)
    attn_tp = tp_size // attn_dp_size
    print(
        f"decode engine: tp={tp_size} attn_dp={attn_dp_size} dp_attention={dp_attention} "
        f"attn_tp={attn_tp}",
        flush=True,
    )
    if attn_dp_size != DECODE_DP_SIZE or not dp_attention:
        raise AssertionError(
            f"decode engine reports attn_dp_size={attn_dp_size} "
            f"enable_dp_attention={dp_attention}; expected attn_dp_size={DECODE_DP_SIZE} "
            "with DP attention on. The KV would land rank-for-rank and this row "
            "would silently re-run the symmetric gate."
        )
    if attn_tp <= 0 or tp_size % attn_dp_size:
        raise AssertionError(
            f"decode tp={tp_size} is not divisible by attn_dp={attn_dp_size}"
        )


def _assert_concurrent_wave_is_deterministic(
    base_url: str, prompt: str, expected_text: str
) -> None:
    """Same prompt, CONCURRENT_WAVE_SIZE at once: every reply must be identical.

    The cache probe is strictly one request at a time, so it cannot see
    nondeterminism that only appears when requests share a batch -- the class of
    bug resharding is most exposed to, since a decode rank now assembles its KV
    from two prefill sources and ordering between them, and between requests
    interleaved in one batch, is newly load-bearing. Greedy decoding makes the
    expected output exact; the sequential result is the reference.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENT_WAVE_SIZE
    ) as pool:
        records = list(
            pool.map(
                lambda _: _generate_probe(base_url, prompt, CACHE_MAX_NEW_TOKENS),
                range(CONCURRENT_WAVE_SIZE),
            )
        )
    for index, record in enumerate(records):
        _assert_spec_active(record, f"concurrent-wave[{index}]")
        if record["text"] != expected_text:
            matched = 0
            for left, right in zip(expected_text, record["text"]):
                if left != right:
                    break
                matched += 1
            raise AssertionError(
                f"concurrent request {index} of {CONCURRENT_WAVE_SIZE} diverged "
                f"from the sequential greedy output after {matched} identical "
                f"characters.\n  sequential: {expected_text[:200]!r}\n  "
                f"concurrent: {record['text'][:200]!r}"
            )
    print(
        f"concurrent wave: {CONCURRENT_WAVE_SIZE}/{CONCURRENT_WAVE_SIZE} replies "
        "identical to the sequential greedy output",
        flush=True,
    )


def _serve_prefill_leader(topology: PodTopology) -> int:
    """Serve prefill, start the mini-LB, then run the cache probe and GSM8K."""
    # Run the router on the prefill leader to avoid requesting another pod.
    procs = []
    self_host = topology.group_hostnames[0]
    decode_host = peer_group_hostname(
        self_host, from_role=Role.PREFILL.value, to_role=Role.DECODE.value
    )

    try:
        procs.append(execute_prefill(worker_id=0, group_hosts=topology.group_hosts))
        _wait_http_200(
            f"http://127.0.0.1:{PREFILL_PORT}/health", WAIT_SECS, tuple(procs)
        )
        _wait_http_200(f"http://{decode_host}:{DECODE_PORT}/health", WAIT_SECS)
        # Before the LB hides which shape the decode engine actually took.
        assert_decode_reshard_topology(decode_host)

        procs.append(
            execute_router(
                prefill_host=self_host,
                decode_host=decode_host,
                port=LB_PORT,
                prefill_port=PREFILL_PORT,
                decode_port=DECODE_PORT,
            )
        )
        _wait_http_200(f"http://127.0.0.1:{LB_PORT}/health", 300, tuple(procs))

        base_url = f"http://127.0.0.1:{LB_PORT}"
        run_cache_offload_spec_probe(
            base_url, flush_url=f"http://127.0.0.1:{PREFILL_PORT}"
        )
        model_config_main(MODEL_CONFIG, base_url=base_url)
        return 0
    finally:
        if not KEEP_UP:
            for p in procs:
                _kill(p)


def serve_role(topology: PodTopology, role: Role) -> int:
    """Dispatch and execute service behavior according to role and rank."""
    # model_config_main invokes unittest, so remove the CLI mode first.
    sys.argv = [sys.argv[0]]
    if topology.is_prefill_leader:
        return _serve_prefill_leader(topology)
    return _serve_worker(topology, role)


class TestPdGptOss120bReshardConfiguration(unittest.TestCase):
    """Host-side configuration tests runnable without TPU hardware."""

    def test_both_engines_use_raiden_transfer_backend(self) -> None:
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            self.assertIn("--disaggregation-transfer-backend raiden", env["EXTRA_ARGS"])
            self.assertIn("--skip-server-warmup", env["EXTRA_ARGS"])
            self.assertIn("--sampling-backend pytorch", env["EXTRA_ARGS"])
            self.assertIn("--reasoning-parser gpt-oss", env["EXTRA_ARGS"])
            self.assertEqual(env["TP_SIZE"], "16")
            self.assertEqual(env["EP_SIZE"], "16")
            self.assertEqual(env["PAGE_SIZE"], "64")
            self.assertEqual(env["QUANTIZATION"], "")
            self.assertEqual(env["KV_CACHE_DTYPE"], "bfloat16")
            self.assertEqual(env["SGLANG_TPU_FUSED_MOE_SC_GATHER"], "0")

    def test_attention_tp_is_asymmetric(self) -> None:
        # The whole point of this file. Raiden reshards only when a decode rank
        # is fed by more than one prefill rank, so equal attn_tp on both sides
        # would leave this row silently re-running the symmetric gate it was
        # copied from, with every other assertion here still passing.
        prefill = build_prefill_env(2, env={})
        decode = build_decode_env(2, env={})
        self.assertNotIn("DP_SIZE", prefill)
        self.assertEqual(decode["DP_SIZE"], str(DECODE_DP_SIZE))
        self.assertEqual(decode["ENABLE_DP_ATTENTION"], "1")
        self.assertEqual(prefill["TP_SIZE"], decode["TP_SIZE"])
        attn_tp = int(decode["TP_SIZE"]) // int(decode["DP_SIZE"])
        self.assertEqual((int(prefill["TP_SIZE"]), attn_tp), (16, 8))
        for env in (prefill, decode):
            self.assertEqual(env["SGLANG_RAIDEN_RESHARD"], "1")

    def test_gates_are_stricter_than_the_symmetric_row(self) -> None:
        # Both gates this row inherits are unfailable at their default values:
        # accept_length >= 1 always, and a 0.72 GSM8K floor is cleared by any
        # working engine. Neither could see a subtle reshard numerical error.
        self.assertGreater(MIN_SPEC_ACCEPT_LENGTH, 1.0)
        self.assertGreaterEqual(GSM8K_BASELINE, 0.9)
        # ...but not so strict the row is flaky by construction: the floor stays
        # below the lowest score this topology has produced.
        self.assertLess(GSM8K_BASELINE - ATOL, 0.95)
        self.assertEqual(MODEL_CONFIG["gates"][0]["baseline"], GSM8K_BASELINE)

    def test_concurrent_wave_fits_one_engine_batch(self) -> None:
        # Sized above MAX_RUNNING_REQUESTS the wave would queue into separate
        # batches and stop exercising the interleaving it exists to check.
        self.assertGreater(CONCURRENT_WAVE_SIZE, 1)
        self.assertLessEqual(CONCURRENT_WAVE_SIZE, MAX_RUNNING_REQUESTS)

    def test_runtime_oracles_reject_what_they_are_meant_to(self) -> None:
        # Both runtime checks guard failure modes that look exactly like a pass
        # from outside: a symmetric decode engine, and a wave whose replies
        # disagree. Stub the engine and drive each one.
        stub_url = "http://stub-host:30000"
        served: dict = {}
        replies: list = []
        lock = threading.Lock()

        class _Resp:
            def read(self):
                with lock:
                    return json.dumps(replies.pop(0) if replies else served).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _reply(text):
            return {
                "text": text,
                "meta_info": {
                    "completion_tokens": 64,
                    "spec_verify_ct": 20,
                    "prompt_tokens": 4102,
                    "cached_tokens_details": {"device": 0, "host": 0},
                },
            }

        real_urlopen = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: _Resp()
        try:
            served.update(
                tp_size=16, dp_size=1, attn_dp_size=1, enable_dp_attention=False
            )
            with self.assertRaisesRegex(AssertionError, "attn_dp_size=1"):
                assert_decode_reshard_topology("decode-host")
            served.update(attn_dp_size=2, enable_dp_attention=False)
            with self.assertRaisesRegex(AssertionError, "enable_dp_attention=False"):
                assert_decode_reshard_topology("decode-host")
            served.update(attn_dp_size=2, enable_dp_attention=True)
            assert_decode_reshard_topology("decode-host")  # must not raise

            replies[:] = [_reply("same") for _ in range(CONCURRENT_WAVE_SIZE)]
            _assert_concurrent_wave_is_deterministic(stub_url, "prompt", "same")
            replies[:] = [_reply("same") for _ in range(CONCURRENT_WAVE_SIZE)]
            replies[-1] = _reply("same but drifted")
            with self.assertRaisesRegex(AssertionError, "diverged"):
                _assert_concurrent_wave_is_deterministic(stub_url, "prompt", "same")
        finally:
            urllib.request.urlopen = real_urlopen

    def test_both_engines_run_eagle3(self) -> None:
        # The P/D handoff carries spec bootstrap state, so BOTH engines run spec.
        for env_fn in (build_prefill_env, build_decode_env):
            args = env_fn(2, env={})["EXTRA_ARGS"]
            self.assertIn("--speculative-algorithm EAGLE3", args)
            self.assertIn(
                "--speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16", args
            )
            self.assertIn("--speculative-eagle-topk 1", args)

    def test_prefix_cache_and_kv_offload_with_bounded_spec_compile_grid(self) -> None:
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            # Prefix caching stays ON alongside spec (radix ⊥ spec only on the
            # decode's own cache, which defaults off).
            self.assertEqual(env["DISABLE_RADIX_CACHE"], "0")
            self.assertNotIn("--disable-radix-cache", env["EXTRA_ARGS"])
            # Bounded verify-graph compile grid guards against the host-RAM OOM.
            self.assertEqual(env["TPU_COMPILE_BS"], "1 2 4 16")
            self.assertEqual(env["SGLANG_TPU_ALLOW_SPEC_RECOMPILE"], "1")
        prefill_args = build_prefill_env(2, env={})["EXTRA_ARGS"]
        self.assertIn("--enable-hierarchical-cache", prefill_args)
        self.assertIn("--hicache-io-backend raiden", prefill_args)
        self.assertIn("--hicache-mem-layout page_first_fused", prefill_args)
        self.assertIn("--enable-cache-report", prefill_args)

    def test_decode_omits_raiden_hicache_retraction(self) -> None:
        # Spec decode does not offload draft KV, so no host-pool retraction flags.
        decode_args = build_decode_env(2, env={})["EXTRA_ARGS"]
        self.assertNotIn("hicache", decode_args)

    def test_spec_runtime_oracle_rejects_fallback_decode(self) -> None:
        fallback = {"spec_verify_ct": 0, "completion_tokens": 16}
        with self.assertRaisesRegex(AssertionError, "did not run"):
            _assert_spec_active(fallback, "fallback")

    def test_spec_runtime_oracle_accepts_active_drafting(self) -> None:
        _assert_spec_active({"spec_verify_ct": 4, "completion_tokens": 12}, "active")

    def test_raiden_port_does_not_collide_with_service_ports(self) -> None:
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            raiden_base = int(env["SGLANG_RAIDEN_KV_PORT"])
            for port in (PREFILL_PORT, DECODE_PORT, LB_PORT, BOOTSTRAP_PORT):
                self.assertNotIn(port, range(raiden_base, raiden_base + 4 * 16))

    def test_peer_hostname_maps_prefill_to_decode(self) -> None:
        prefill_host = "tpu-ci-pd-gpt-oss-120b-prefill-0-0"
        self.assertEqual(
            peer_group_hostname(
                prefill_host,
                from_role=Role.PREFILL.value,
                to_role=Role.DECODE.value,
            ),
            "tpu-ci-pd-gpt-oss-120b-decode-0-0",
        )

    def test_get_role_resolution(self) -> None:
        leader_topo = PodTopology.from_env(
            {
                "CI_PD_ROLE": "prefill",
                "TPU_WORKER_ID": "0",
                "TPU_WORKER_HOSTNAMES": "a-prefill-0-0,a-prefill-0-1",
            }
        )
        self.assertEqual(get_role(leader_topo), Role.PREFILL)
        self.assertTrue(leader_topo.is_prefill_leader)

        worker_topo = PodTopology.from_env(
            {
                "CI_PD_ROLE": "prefill",
                "TPU_WORKER_ID": "1",
                "TPU_WORKER_HOSTNAMES": "a-prefill-0-0,a-prefill-0-1",
            }
        )
        self.assertEqual(get_role(worker_topo), Role.PREFILL)
        self.assertFalse(worker_topo.is_prefill_leader)

        decode_topo = PodTopology.from_env(
            {
                "CI_PD_ROLE": "decode",
                "TPU_WORKER_ID": "0",
                "TPU_WORKER_HOSTNAMES": "a-decode-0-0,a-decode-0-1",
            }
        )
        self.assertEqual(get_role(decode_topo), Role.DECODE)
        self.assertFalse(decode_topo.is_prefill_leader)

    def test_benchmark_directory_resolves(self) -> None:
        self.assertTrue(
            os.path.isfile(os.path.join(BENCH, "launch_prefill.sh")),
            f"BENCH does not point at benchmark/tpu: {BENCH}",
        )
        self.assertTrue(
            os.path.isfile(os.path.join(BENCH, "launch_decode.sh")),
            f"BENCH does not point at benchmark/tpu: {BENCH}",
        )

    def test_local_model_path_validation(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, MODEL_ENV):
            validate_model_source("/weights/definitely-missing-gpt-oss-120b")
        validate_model_source(HF_REPO_ID)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P/D multihost prefix-cache KV-offload EAGLE3 test"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration"],
        help="Execution mode (default: unittest)",
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration":
        topology = PodTopology.from_env()
        role = get_role(topology)
        sys.exit(serve_role(topology, role))
    else:
        sys.argv = [sys.argv[0], "TestPdGptOss120bReshardConfiguration", *extra_args]
        unittest.main(verbosity=2)
