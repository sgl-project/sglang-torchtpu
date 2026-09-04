"""GPT-OSS-120B P/D + EAGLE3 + prefix caching multihost GSM8K accuracy test.

Topology: a 2-host prefill engine and a 2-host decode engine (TP16/EP16 each,
four hosts total) across separate TPU slices, KV over raiden. Both engines run
EAGLE3 speculative decoding (the P/D handoff carries the spec bootstrap state)
AND keep prefix caching on at the prefill. Prefill rank 0 launches the prefill
engine, waits for decode, starts the mini-LB, and runs GSM8K through it.

Fills the gap between the no-spec 2-host P/D gate (pd_multihost) and the
single-host P/D EAGLE3 gate (pd_singlehost_spec): this is the only gate that
exercises 2P+2D with spec AND prefix caching together.
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
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def model_config_main(*_args, **_kwargs):
        return None


register_tpu_ci(
    est_time=7200,
    suite="tpu-integration",
    nightly=True,
    labels=["multihost", "gpt-oss-120b", "pd", "spec"],
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
EXTRA_SERVER_ARGS: Tuple[str, ...] = ("--reasoning-parser", "gpt-oss")
# Bound the compile grid + allow spec recompiles: the unbounded bs<=max_running
# verify-graph capture spikes host RAM and OOM-kills the decode. RESHARD off:
# symmetric prefill/decode TP needs no strided reshard.
EXTRA_LAUNCH_ENV: Dict[str, str] = {
    "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
    "TPU_COMPILE_BS": "1 2 4 16",
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "SGLANG_RAIDEN_RESHARD": "0",
}

# EAGLE3 speculative decoding, shared by both engines: the P/D handoff carries
# the spec bootstrap state, so the prefill loads the draft model too.
SPEC_NUM_STEPS = 3
SPEC_EAGLE_TOPK = 1
SPEC_NUM_DRAFT_TOKENS = 4

# The single-host GPT-OSS EAGLE3 gate measured ~0.785 over 200 examples; keep the
# spec threshold rather than the no-spec 0.72.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B_SPEC", "0.70"))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "200"))
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
    "test_name": "gpt-oss-120b-pd-multihost-spec-gsm8k",
    "variant": "PD-2host-TP16+EP16-raiden+EAGLE3+prefixcache",
    "launch": {
        "external": True,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-120b-pd-spec-gsm8k-e2e",
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
        # Prefix caching ON. In PD it is effective at the prefill (where the
        # shared prompt prefix is matched); the decode's own radix cache stays
        # off (disaggregation_decode_enable_radix_cache defaults False), which is
        # the one path that conflicts with speculative decoding.
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
    """Serve prefill engine, start mini-LB, and run GSM8K accuracy evaluation."""
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


class TestPdGptOss120bConfiguration(unittest.TestCase):
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

    def test_both_engines_run_eagle3(self) -> None:
        # The P/D handoff carries spec bootstrap state, so BOTH engines run spec.
        for env_fn in (build_prefill_env, build_decode_env):
            args = env_fn(2, env={})["EXTRA_ARGS"]
            self.assertIn("--speculative-algorithm EAGLE3", args)
            self.assertIn(
                "--speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16", args
            )
            self.assertIn("--speculative-eagle-topk 1", args)

    def test_prefix_caching_on_with_bounded_spec_compile_grid(self) -> None:
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            # Prefix caching stays ON alongside spec (radix ⊥ spec only on the
            # decode's own cache, which defaults off).
            self.assertEqual(env["DISABLE_RADIX_CACHE"], "0")
            self.assertNotIn("--disable-radix-cache", env["EXTRA_ARGS"])
            # Bounded verify-graph compile grid guards against the host-RAM OOM.
            self.assertEqual(env["TPU_COMPILE_BS"], "1 2 4 16")

    def test_decode_omits_raiden_hicache_retraction(self) -> None:
        # Spec decode does not offload draft KV, so no host-pool retraction flags.
        decode_args = build_decode_env(2, env={})["EXTRA_ARGS"]
        self.assertNotIn("hicache", decode_args)

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
    parser = argparse.ArgumentParser(description="P/D multihost test")
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
        sys.argv = [sys.argv[0], "TestPdGptOss120bConfiguration", *extra_args]
        unittest.main(verbosity=2)
