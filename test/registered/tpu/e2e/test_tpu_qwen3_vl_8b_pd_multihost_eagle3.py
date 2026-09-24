"""Qwen3-VL-8B P/D + EAGLE3 multihost MMMU-Pro (vision) gate.

Topology: a 2-host prefill engine and a 2-host decode engine (TP16 each, four
hosts total) across separate TPU slices, KV over raiden. The prefill engine owns
the ViT, so the handoff carries image-derived KV plus the spec bootstrap state;
this is the only gate where a vision prefill feeds a disaggregated draft/verify
decode loop.

The score is report-only -- the enforced threshold is the accept length, which
is what goes flat when the draft stops tracking the target across the handoff.
"""

import argparse
import enum
import os
import signal
import socket
import subprocess
import sys
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
    # Source-only checkouts can still run the launch-configuration tests.
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def model_config_main(*_args, **_kwargs):
        return None


register_tpu_ci(
    est_time=10800,
    suite="tpu-integration",
    nightly=True,
    labels=["multihost", "qwen3-vl-8b", "pd", "spec", "eagle3", "multimodal"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
WAIT_SECS = int(E("WAIT_SECS", "5400"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")


class Role(str, enum.Enum):
    """Execution roles in the P/D cluster."""

    PREFILL = "prefill"
    DECODE = "decode"


PREFILL_LOG = "/tmp/pd_prefill.log"
DECODE_LOG = "/tmp/pd_decode.log"
LB_LOG = "/tmp/pd_lb.log"

SERVED_MODEL_NAME = "qwen3-vl-8b"
DEFAULT_MODEL_PATH = "/weights/Qwen3-VL-8B-Instruct"
MODEL_ENV = "QWEN3_VL_MODEL"
DRAFT_MODEL_ENV = "QWEN3_VL_DRAFT_MODEL"
# On qwen3-spec-vl-rox alongside the target, not on model-weights-filestore.
DEFAULT_DRAFT_MODEL_PATH = "/weights/qwen3_8b_eagle3"

PAGE_SIZE = 64
# 8B bf16 is ~1 GB per TP16 rank, so the budget is dominated by graph
# temporaries: the ViT and the decoder each capture their own, and EAGLE3 adds
# the draft/verify captures on top.
MEM_FRACTION_PREFILL = "0.30"
MEM_FRACTION_DECODE = "0.30"
# bf16 checkpoint. "unquant" is the explicit opt-out; set here rather than
# inherited from the recipe because P/D drives launch_prefill.sh/launch_decode.sh,
# whose own default is fp8.
QUANTIZATION = "unquant"
KV_CACHE_DTYPE = "bfloat16"
MAX_TOTAL_TOKENS = 65536
# Doubles as the compile cap: reconcile_tpu_batch_limits raises
# torch_compile_max_bs back to max_running_requests, so the bs grid is
# [1, 2, 4, 8] only while concurrency stays here.
MAX_RUNNING_REQUESTS = 8
DISAGG_TRANSFER_BACKEND = "raiden"

# max_pixels caps one image at ~306 vision tokens (patch 16 + 2x2 merge = 1024
# px per token). Uncapped, a single MMMU-Pro page exceeds the whole 8k chunk.
MM_PROCESS_CONFIG = '{"image":{"min_pixels":65536,"max_pixels":313600}}'
EXTRA_SERVER_ARGS: Tuple[str, ...] = (
    "--trust-remote-code",
    "--mm-process-config",
    MM_PROCESS_CONFIG,
)
# The default PowerBucketer gives 8 token buckets (64..8192), and the VL and
# spec EXTEND grids each multiply by them, costing ~20 min of startup. Four
# buckets halve that; 8192 must stay to cover MAX_PREFILL_TOKENS. RESHARD off:
# symmetric prefill and decode TP needs no strided reshard.
EXTRA_LAUNCH_ENV: Dict[str, str] = {
    "TPU_COMPILE_TOKENS": "512 1024 2048 8192",
    # common_env.sh defaults this to +recompiles, and dynamo reprints the whole
    # cached guard list per recompile: 340k of the 371k log lines. Neither ""
    # nor "-recompiles" silences it -- the former loses to ${TORCH_LOGS:-...},
    # and a "-" on an artifact still enables it. Serving-time recompiles stay
    # fatal via the scheduler's fail_on_recompile stance.
    "TORCH_LOGS": "-dynamo",
    "SGLANG_RAIDEN_RESHARD": "0",
}

# EAGLE3 speculative decoding, shared by both engines: the P/D handoff carries
# the spec bootstrap state, so the prefill loads the draft model too.
SPEC_NUM_STEPS = 3
SPEC_EAGLE_TOPK = 1
SPEC_NUM_DRAFT_TOKENS = 4

# A floor on "spec is still accepting", not a tuned target: the draft is the
# text-only Qwen3-8B speculator, and no VL P/D night has been measured yet.
# Ratchet it up once the trend exists.
SPEC_ACCEPT_LENGTH_THRESHOLD = float(E("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.30"))
NUM_EXAMPLES = int(E("MMMU_NUM_EXAMPLES", "200"))
NUM_THREADS = int(E("MMMU_NUM_THREADS", str(MAX_RUNNING_REQUESTS)))
MAX_TOKENS = int(E("MMMU_MAX_TOKENS", "512"))
# Budget: the ViT and verify-graph captures. The single-host gate timed out at
# 2400s mid-capture.
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "5400"))


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
        # The ViT and EXTEND graph captures outlast the 300s default and trip
        # the watchdog into repeated stack dumps mid-compile.
        "--watchdog-timeout",
        "7200",
    ]


MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-vl-8b-pd-multihost-eagle3-mmmu-pro-vision",
    "variant": "PD-2host-TP16+EAGLE3",
    "launch": {
        "external": True,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-vl-pd-eagle3-mmmu-e2e",
    },
    "gates": [
        {
            "dataset": "mmmu_pro_vision",
            "label": "mmmu_pro_vision",
            "report_only": True,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
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
            f"Qwen3-VL-8B checkpoint incomplete or missing at {model_path}; "
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
        # Dense decoder: --ep-size is inert, but launch_prefill.sh always emits
        # it, so keep it consistent with TP rather than leaving the 8 default.
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "PAGE_SIZE": str(PAGE_SIZE),
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": QUANTIZATION,
        "KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        # Prefix caching ON. In PD it is effective at the prefill; the decode's
        # own radix cache stays off (disaggregation_decode_enable_radix_cache
        # defaults False), which is the one path that conflicts with spec.
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "CONTEXT_LEN": values.get("CONTEXT_LEN", "9216"),
        "CHUNKED_PREFILL_SIZE": values.get("CHUNKED_PREFILL_SIZE", "8192"),
        "MAX_PREFILL_TOKENS": values.get("MAX_PREFILL_TOKENS", "8192"),
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
    extra_args = [
        "--served-model-name",
        SERVED_MODEL_NAME,
        *EXTRA_SERVER_ARGS,
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
        "--max-total-tokens",
        str(MAX_TOTAL_TOKENS),
        *_spec_server_args(values),
    ]
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
    extra_args = [
        "--served-model-name",
        SERVED_MODEL_NAME,
        *EXTRA_SERVER_ARGS,
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
        "--max-total-tokens",
        str(MAX_TOTAL_TOKENS),
        *_spec_server_args(values),
    ]
    if values.get("EXTRA_DECODE_ARGS"):
        extra_args.append(values["EXTRA_DECODE_ARGS"])

    return {
        **_common_launch_env(group_hosts, values),
        "MEM_FRACTION_STATIC": values.get("MEM_FRACTION_STATIC", MEM_FRACTION_DECODE),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
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


def _serve_prefill_leader(topology: PodTopology) -> int:
    """Serve prefill engine, start mini-LB, and run the MMMU-Pro evaluation."""
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

        model_config_main(MODEL_CONFIG, base_url=f"http://127.0.0.1:{LB_PORT}")
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


class TestPdQwen3Vl8bConfiguration(unittest.TestCase):
    """Host-side configuration tests runnable without TPU hardware."""

    def test_both_engines_carry_the_multimodal_processor_config(self) -> None:
        # The decode engine builds its own tokenizer/processor for the handed-off
        # request, so an image cap set only on the prefill desynchronizes the two.
        for env_fn in (build_prefill_env, build_decode_env):
            args = env_fn(2, env={})["EXTRA_ARGS"]
            self.assertIn(f"--mm-process-config {MM_PROCESS_CONFIG}", args)
            self.assertIn("--trust-remote-code", args)

    def test_mm_process_config_survives_word_splitting(self) -> None:
        # launch_server.sh word-splits EXTRA_ARGS, so a space anywhere in the
        # JSON would hand the server a truncated value.
        self.assertNotIn(" ", MM_PROCESS_CONFIG)

    def test_both_engines_run_eagle3(self) -> None:
        # The P/D handoff carries spec bootstrap state, so BOTH engines run spec.
        for env_fn in (build_prefill_env, build_decode_env):
            args = env_fn(2, env={})["EXTRA_ARGS"]
            self.assertIn("--speculative-algorithm EAGLE3", args)
            self.assertIn(
                f"--speculative-draft-model-path {DEFAULT_DRAFT_MODEL_PATH}", args
            )
            self.assertIn("--speculative-eagle-topk 1", args)

    def test_bf16_checkpoint_is_not_requantized(self) -> None:
        # An unset QUANTIZATION lets launch_server.sh apply its fp8 default,
        # which would try to re-quantize the bf16 weights.
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            self.assertEqual(env["QUANTIZATION"], "unquant")
            self.assertEqual(env["KV_CACHE_DTYPE"], "bfloat16")

    def test_prefill_chunk_admits_a_full_mmmu_page(self) -> None:
        # A capped MMMU-Pro page is ~306 vision tokens on top of the prompt; the
        # chunk must stay under the context so the ViT output is not split.
        env = build_prefill_env(2, env={})
        self.assertLessEqual(int(env["CHUNKED_PREFILL_SIZE"]), int(env["CONTEXT_LEN"]))
        self.assertEqual(env["MAX_PREFILL_TOKENS"], env["CHUNKED_PREFILL_SIZE"])

    def test_raiden_port_does_not_collide_with_service_ports(self) -> None:
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            raiden_base = int(env["SGLANG_RAIDEN_KV_PORT"])
            for port in (PREFILL_PORT, DECODE_PORT, LB_PORT, BOOTSTRAP_PORT):
                self.assertNotIn(port, range(raiden_base, raiden_base + 4 * 16))

    def test_peer_hostname_maps_prefill_to_decode(self) -> None:
        self.assertEqual(
            peer_group_hostname(
                "tpu-ci-pd-mh-qwen3-vl-8b-spec-prefill-0-0",
                from_role=Role.PREFILL.value,
                to_role=Role.DECODE.value,
            ),
            "tpu-ci-pd-mh-qwen3-vl-8b-spec-decode-0-0",
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
            validate_model_source("/weights/definitely-missing-qwen3-vl-8b")
        # HF ids are resolved by the server, so they skip the on-disk check.
        validate_model_source("Qwen/Qwen3-VL-8B-Instruct")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL-8B P/D multihost test")
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
        sys.argv = [sys.argv[0], "TestPdQwen3Vl8bConfiguration", *extra_args]
        unittest.main(verbosity=2)
