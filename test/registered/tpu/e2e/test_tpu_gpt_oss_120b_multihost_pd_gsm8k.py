"""GPT-OSS-120B P/D disaggregation multihost GSM8K accuracy test.

Topology: a multihost PREFILL engine and a multihost DECODE engine, each spanning
its own TPU slice, connected to a dedicated CPU ROUTER pod with raiden transfer backend.
The CI JobSet (.github/tpu-ci/jobset-tpu-pd.yaml) runs this file on every pod across
all three replicated jobs (router, prefill, decode); roles are dispatched from CI_PD_ROLE.

Roles:
  router (CPU pod): waits for prefill & decode engines, starts mini-LB, runs GSM8K.
  prefill (TPU slice): launches/joins prefill engine, serves until teardown.
  decode (TPU slice): launches/joins decode engine, serves until teardown.
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
from typing import Dict, Mapping, Optional, Tuple

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


register_tpu_ci(est_time=7200, suite="tpu-integration", nightly=True)

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

    ROUTER = "router"
    PREFILL = "prefill"
    DECODE = "decode"


PREFILL_LOG = "/tmp/pd_prefill.log"
DECODE_LOG = "/tmp/pd_decode.log"
LB_LOG = "/tmp/pd_lb.log"

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
HF_REPO_ID = "openai/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"

PAGE_SIZE = 64
MEM_FRACTION_PREFILL = "0.35"
MEM_FRACTION_DECODE = "0.35"
QUANTIZATION = ""
KV_CACHE_DTYPE = "fp8_e4m3"
MAX_TOTAL_TOKENS = 786432
DISAGG_TRANSFER_BACKEND = "raiden"
EXTRA_SERVER_ARGS: Tuple[str, ...] = ("--reasoning-parser", "gpt-oss")
EXTRA_LAUNCH_ENV: Dict[str, str] = {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"}

GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.72")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-pd-multihost-gsm8k",
    "variant": "PD-2host-TP16+EP16-raiden",
    "launch": {
        "external": True,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-120b-pd-gsm8k-e2e",
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
    def is_router(self) -> bool:
        return self.role == Role.ROUTER.value

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
    """Resolve strongly-typed Role from pod topology.

    Args:
      topology: Pod topology parsed from environment.

    Returns:
      Resolved Role enum instance.
    """
    return Role(topology.role)


def peer_group_hostname(hostname: str, from_role: str, to_role: str) -> str:
    """Map a pod hostname of one replicated job onto its peer job pod.

    Args:
      hostname: Pod hostname with role segment.
      from_role: Current pod role segment to match.
      to_role: Target peer role segment to replace with.

    Returns:
      Peer pod hostname.

    Raises:
      ValueError: When hostname has no matching role segment.
    """
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


def resolve_model_path(env: Optional[Mapping[str, str]] = None) -> str:
    """Resolve model path from environment or default constant.

    Args:
      env: Optional environment mapping.

    Returns:
      Resolved model path or HF repo ID.
    """
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Validate model checkpoint directory exists when path is local.

    Args:
      model_path: Local directory path or HF repo ID.

    Raises:
      FileNotFoundError: When local checkpoint directory is incomplete.
    """
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
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "PYTHON": PYTHON,
        "SGLANG_RAIDEN_KV_PORT": values.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        **EXTRA_LAUNCH_ENV,
    }


def build_prefill_env(
    group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Construct environment dictionary for prefill engine.

    Args:
      group_hosts: Number of TPU hosts in the prefill group.
      env: Optional environment mapping.

    Returns:
      Environment dictionary for launch_prefill.sh.
    """
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
    """Construct environment dictionary for decode engine.

    Args:
      group_hosts: Number of TPU hosts in the decode group.
      env: Optional environment mapping.

    Returns:
      Environment dictionary for launch_decode.sh.
    """
    values = os.environ if env is None else env
    decode_args = ["--served-model-name", SERVED_MODEL_NAME]
    if MAX_TOTAL_TOKENS is not None:
        decode_args += ["--max-total-tokens", str(MAX_TOTAL_TOKENS)]

    extra_args = [
        *decode_args,
        *EXTRA_SERVER_ARGS,
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
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
    """Launch prefill engine on specified worker rank.

    Args:
      worker_id: Slice-local node rank (0 is leader).
      group_hosts: Total hosts in prefill slice.
      env: Optional environment mapping overrides.
      log_path: Path to write log output.

    Returns:
      Spawned process instance.
    """
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
    """Launch decode engine on specified worker rank.

    Args:
      worker_id: Slice-local node rank.
      group_hosts: Total hosts in decode slice.
      env: Optional environment mapping overrides.
      log_path: Path to write log output.

    Returns:
      Spawned process instance.
    """
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
    """Launch mini-LB router connecting prefill and decode engines.

    Args:
      prefill_host: Prefill leader hostname or IP.
      decode_host: Decode leader hostname or IP.
      port: Router listening port.
      prefill_port: Prefill engine HTTP port.
      decode_port: Decode engine HTTP port.
      log_path: Path to write log output.

    Returns:
      Spawned process instance.
    """
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
    """Serve prefill or decode worker until termination."""
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


def _serve_router(topology: PodTopology) -> int:
    """Serve router orchestrator and execute benchmark gate."""
    procs = []
    self_host = topology.group_hostnames[0]

    prefill_host = peer_group_hostname(
        self_host, from_role=Role.ROUTER.value, to_role=Role.PREFILL.value
    )
    decode_host = peer_group_hostname(
        self_host, from_role=Role.ROUTER.value, to_role=Role.DECODE.value
    )

    try:
        _wait_http_200(f"http://{prefill_host}:{PREFILL_PORT}/health", WAIT_SECS)
        _wait_http_200(f"http://{decode_host}:{DECODE_PORT}/health", WAIT_SECS)

        procs.append(
            execute_router(
                prefill_host=prefill_host,
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
    """Dispatch and execute service behavior according to role.

    Args:
      topology: Pod topology parsed from environment.
      role: Target role to execute.

    Returns:
      Exit code of the role lifecycle.
    """
    if role == Role.ROUTER:
        return _serve_router(topology)
    return _serve_worker(topology, role)


def render_jobset_yaml(
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights-rox",
    run_name: str = "gpt-oss-120b-pd-gsm8k",
) -> str:
    """Render the Kubernetes JobSet YAML for this test.

    Args:
      image: Container image tag.
      lane: CI lane (e.g. 'nightly').
      weights_pvc: Name of the weights PersistentVolumeClaim.
      run_name: Name of the JobSet run.

    Returns:
      Rendered YAML string.
    """
    rel_path = os.path.relpath(__file__, REPO)
    cmd = [
        PYTHON,
        os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py"),
        "--pd-prefill-hosts",
        "2",
        "--pd-decode-hosts",
        "2",
        "--lane",
        lane,
        "--image",
        image,
        "--weights-pvc",
        weights_pvc,
        "--name",
        run_name,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


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
            self.assertEqual(env["KV_CACHE_DTYPE"], "fp8_e4m3")
            self.assertEqual(env["SGLANG_TPU_FUSED_MOE_SC_GATHER"], "0")

    def test_no_hierarchical_cache_or_speculative_features(self) -> None:
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            self.assertNotIn("hicache", env["EXTRA_ARGS"])
            self.assertNotIn("hierarchical-cache", env["EXTRA_ARGS"])
            self.assertNotIn("speculative", env["EXTRA_ARGS"])

    def test_raiden_port_does_not_collide_with_service_ports(self) -> None:
        for env_fn in (build_prefill_env, build_decode_env):
            env = env_fn(2, env={})
            raiden_base = int(env["SGLANG_RAIDEN_KV_PORT"])
            for port in (PREFILL_PORT, DECODE_PORT, LB_PORT, BOOTSTRAP_PORT):
                self.assertNotIn(port, range(raiden_base, raiden_base + 4 * 16))

    def test_peer_hostname_maps_router_to_prefill_and_decode(self) -> None:
        router_host = "tpu-ci-pd-gpt-oss-120b-router-0-0"
        self.assertEqual(
            peer_group_hostname(
                router_host,
                from_role=Role.ROUTER.value,
                to_role=Role.PREFILL.value,
            ),
            "tpu-ci-pd-gpt-oss-120b-prefill-0-0",
        )
        self.assertEqual(
            peer_group_hostname(
                router_host,
                from_role=Role.ROUTER.value,
                to_role=Role.DECODE.value,
            ),
            "tpu-ci-pd-gpt-oss-120b-decode-0-0",
        )

    def test_get_role_resolution(self) -> None:
        router_topo = PodTopology.from_env(
            {
                "CI_PD_ROLE": "router",
                "TPU_WORKER_ID": "0",
                "TPU_WORKER_HOSTNAMES": "a-router-0-0",
            }
        )
        self.assertEqual(get_role(router_topo), Role.ROUTER)
        self.assertTrue(router_topo.is_router)

        prefill_topo = PodTopology.from_env(
            {
                "CI_PD_ROLE": "prefill",
                "TPU_WORKER_ID": "1",
                "TPU_WORKER_HOSTNAMES": "a-prefill-0-0,a-prefill-0-1",
            }
        )
        self.assertEqual(get_role(prefill_topo), Role.PREFILL)
        self.assertFalse(prefill_topo.is_router)

        decode_topo = PodTopology.from_env(
            {
                "CI_PD_ROLE": "decode",
                "TPU_WORKER_ID": "0",
                "TPU_WORKER_HOSTNAMES": "a-decode-0-0,a-decode-0-1",
            }
        )
        self.assertEqual(get_role(decode_topo), Role.DECODE)

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

    def test_dry_run_render_jobset_yaml(self) -> None:
        yaml_content = render_jobset_yaml()
        self.assertIn("generateName: tpu-ci-gpt-oss-120b-pd-gsm8k-", yaml_content)
        self.assertIn("targetReplicatedJobs:", yaml_content)
        self.assertIn("- router", yaml_content)
        self.assertIn("name: router", yaml_content)
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/test_tpu_gpt_oss_120b_multihost_pd_gsm8k.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P/D multihost test")
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
        help="Execution mode (default: unittest)",
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml())
        sys.exit(0)
    elif args.mode == "integration":
        topology = PodTopology.from_env()
        role = get_role(topology)
        sys.exit(serve_role(topology, role))
    else:
        sys.argv = [sys.argv[0], "TestPdGptOss120bConfiguration", *extra_args]
        unittest.main(verbosity=2)
