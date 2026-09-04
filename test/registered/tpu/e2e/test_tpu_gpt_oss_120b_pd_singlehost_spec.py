"""GPT-OSS-120B P/D + EAGLE3 gate, one TPU host per engine.

The JobSet gives prefill and decode one full v7x host each (TP8/EP8). Both
engines load the target and EAGLE3 draft model because the P/D handoff carries
speculative bootstrap state in addition to target KV. The prefill pod also
runs the mini router, verifies that requests really execute speculative
verification after the handoff, and then scores GSM8K through that router.
"""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    est_time=7200,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd", "spec"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
DEFAULT_DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"
MODEL_ENV = "GPT_OSS_MODEL"
DRAFT_MODEL_ENV = "GPT_OSS_DRAFT_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
ROLE_ENV = "CI_PD_ROLE"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

# A v7x host has four physical chips and eight addressable TPU devices.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = CHIPLETS_PER_HOST
EP_SIZE = CHIPLETS_PER_HOST

# The roles live in different pods, so their HTTP and Raiden ports may match.
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
RAIDEN_KV_PORT = int(E("SGLANG_RAIDEN_KV_PORT", "31000"))
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "5400"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")

# Keep the run name short enough for the generated Job and Pod suffixes. It
# also cannot contain the literal role names because peer discovery swaps the
# "-prefill-" and "-decode-" hostname segments.
RUN_NAME = "pd-sh-gpt-120b-spec"

PREFILL_LOG = "/tmp/pd_spec_prefill.log"
DECODE_LOG = "/tmp/pd_spec_decode.log"
LB_LOG = "/tmp/pd_spec_lb.log"

PAGE_SIZE = 64
MEM_FRACTION_STATIC = "0.6"
KV_CACHE_DTYPE = "bfloat16"
MAX_TOTAL_TOKENS = 8192
MAX_RUNNING_REQUESTS = 16
DISAGG_TRANSFER_BACKEND = "raiden"
RETRACTION_BACKUP = "cpu_tensor"

SPEC_NUM_STEPS = 3
SPEC_EAGLE_TOPK = 1
SPEC_NUM_DRAFT_TOKENS = 4
SPEC_ACCEPT_LENGTH_THRESHOLD = float(E("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90"))

# The non-P/D GPT-OSS EAGLE3 gate measured 0.785 over 200 examples. Keep the
# same floor and concurrency so this test isolates the P/D handoff.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B_SPEC", "0.70"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "200"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "16"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "512"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "5400"))


def resolve_model_path(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def resolve_draft_model_path(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    return (
        values.get(DRAFT_MODEL_ENV)
        or values.get("DRAFT_MODEL")
        or DEFAULT_DRAFT_MODEL_PATH
    )


def validate_model_source(model_path: str, *, label: str) -> None:
    """Fail clearly for a missing mounted checkpoint; HF/GCS ids are remote."""
    if not os.path.isabs(model_path):
        return
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"{label} checkpoint incomplete or missing at {model_path}; "
            f"provision it or set {MODEL_ENV if label == 'target' else DRAFT_MODEL_ENV}"
        )


def resolve_role(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    role = values.get(ROLE_ENV, "")
    if role not in (PREFILL_ROLE, DECODE_ROLE):
        raise ValueError(
            f"{ROLE_ENV} must be {PREFILL_ROLE!r} or {DECODE_ROLE!r}, got {role!r}"
        )
    return role


def self_hostname(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def peer_hostname(hostname: str, *, from_role: str, to_role: str) -> str:
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


def _spec_server_args(draft_model: str) -> List[str]:
    args = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        draft_model,
        "--speculative-num-steps",
        str(SPEC_NUM_STEPS),
        "--speculative-eagle-topk",
        str(SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens",
        str(SPEC_NUM_DRAFT_TOKENS),
    ]
    if draft_model.startswith("gs://"):
        args += ["--speculative-draft-load-format", "runai_streamer"]
    return args


def _shared_server_args(values: Mapping[str, str]) -> List[str]:
    model = resolve_model_path(values)
    args = [
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--reasoning-parser",
        "gpt-oss",
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        "--max-total-tokens",
        str(MAX_TOTAL_TOKENS),
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
        "--watchdog-timeout",
        "7200",
        *_spec_server_args(resolve_draft_model_path(values)),
    ]
    if model.startswith("gs://"):
        args += ["--load-format", "runai_streamer"]
    return args


def _common_launch_env(
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    return {
        "MODEL": resolve_model_path(values),
        "DRAFT_MODEL": resolve_draft_model_path(values),
        "TP_SIZE": str(TP_SIZE),
        "EP_SIZE": str(EP_SIZE),
        # Each role is an independent single-host engine even though both pods
        # belong to one P/D JobSet.
        "FORCE_SINGLE_HOST": "1",
        "SGLANG_USE_RAY": "0",
        "PAGE_SIZE": str(PAGE_SIZE),
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        # GPT-OSS carries checkpoint-provided MXFP4 weights.
        "QUANTIZATION": "",
        "KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        "DISABLE_RADIX_CACHE": "1",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "CONTEXT_LEN": values.get("CONTEXT_LEN", "8192"),
        "CHUNKED_PREFILL_SIZE": values.get("CHUNKED_PREFILL_SIZE", "4096"),
        "MAX_PREFILL_TOKENS": values.get("MAX_PREFILL_TOKENS", "4096"),
        "MAX_RUNNING": values.get("MAX_RUNNING", str(MAX_RUNNING_REQUESTS)),
        "TORCH_COMPILE_MAX_BS": values.get(
            "TORCH_COMPILE_MAX_BS", str(MAX_RUNNING_REQUESTS)
        ),
        # Match the established one-host GPT-OSS EAGLE3 GSM8K grid.
        "TPU_COMPILE_BS": values.get("TPU_COMPILE_BS", "1 2 4 16"),
        "TPU_COMPILE_TOKENS": values.get("TPU_COMPILE_TOKENS", "512 2048 4096"),
        "MEM_FRACTION_STATIC": values.get("MEM_FRACTION_STATIC", MEM_FRACTION_STATIC),
        "SGLANG_RAIDEN_KV_PORT": str(RAIDEN_KV_PORT),
        "SGLANG_RAIDEN_RESHARD": "0",
        "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
        "ENABLE_PROFILER_ENDPOINT": "0",
        "PYTHON": PYTHON,
    }


def build_prefill_env(
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    extra_args = _shared_server_args(values)
    if values.get("EXTRA_PREFILL_ARGS"):
        extra_args.append(values["EXTRA_PREFILL_ARGS"])
    return {
        **_common_launch_env(values),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def build_decode_env(
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    extra_args = _shared_server_args(values) + [
        # Keep this gate focused on P/D + spec. A host-pool backup would also
        # pull in HiCache layout and draft-pool behavior, which has its own test.
        "--disaggregation-decode-retraction-backup",
        RETRACTION_BACKUP,
    ]
    if values.get("EXTRA_DECODE_ARGS"):
        extra_args.append(values["EXTRA_DECODE_ARGS"])
    return {
        **_common_launch_env(values),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def _tee_command(argv: Sequence[str], log_path: str) -> Tuple[str, ...]:
    return (
        "bash",
        "-c",
        f"set -o pipefail; {shlex.join(argv)} 2>&1 | tee -a {shlex.quote(log_path)}",
    )


def _launch(
    cmd: Sequence[str],
    *,
    log_path: str,
    extra_env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        _tee_command(cmd, log_path), env=env, start_new_session=True, cwd=cwd
    )


def execute_prefill(log_path: str = PREFILL_LOG) -> subprocess.Popen:
    return _launch(
        ("bash", os.path.join(BENCH, "launch_prefill.sh")),
        log_path=log_path,
        extra_env=build_prefill_env(),
        cwd=BENCH,
    )


def execute_decode(log_path: str = DECODE_LOG) -> subprocess.Popen:
    return _launch(
        ("bash", os.path.join(BENCH, "launch_decode.sh")),
        log_path=log_path,
        extra_env=build_decode_env(),
        cwd=BENCH,
    )


def execute_router(
    *, prefill_host: str, decode_host: str, log_path: str = LB_LOG
) -> subprocess.Popen:
    return _launch(
        (
            PYTHON,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            f"http://{prefill_host}:{PREFILL_PORT}",
            "--decode",
            f"http://{decode_host}:{DECODE_PORT}",
            "--host",
            "0.0.0.0",
            "--port",
            str(LB_PORT),
        ),
        log_path=log_path,
    )


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
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
        return False


def _get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_http_200(
    *, url: str, secs: int, procs: Sequence[Tuple[str, subprocess.Popen]] = ()
) -> None:
    deadline = time.time() + secs
    waited = 0
    while time.time() < deadline:
        for name, proc in procs:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{name} exited rc={proc.returncode} while waiting on {url}"
                )
        if _http_ok(url):
            return
        if waited and waited % 60 == 0:
            print(f"waiting for {url} ({waited}s / {secs}s)", flush=True)
        time.sleep(10)
        waited += 10
    raise TimeoutError(f"{url} not ready after {secs}s")


def engine_configuration_failures(
    *, prefill_info: Mapping[str, Any], decode_info: Mapping[str, Any]
) -> List[str]:
    failures = []
    for label, info, expected_mode in (
        (PREFILL_ROLE, prefill_info, PREFILL_ROLE),
        (DECODE_ROLE, decode_info, DECODE_ROLE),
    ):
        if info.get("disaggregation_mode") != expected_mode:
            failures.append(
                f"{label} reports disaggregation_mode="
                f"{info.get('disaggregation_mode')!r}, expected {expected_mode!r}"
            )
        if info.get("disaggregation_transfer_backend") != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{label} reports disaggregation_transfer_backend="
                f"{info.get('disaggregation_transfer_backend')!r}, expected "
                f"{DISAGG_TRANSFER_BACKEND!r}"
            )
        if str(info.get("speculative_algorithm", "")).upper() != "EAGLE3":
            failures.append(
                f"{label} reports speculative_algorithm="
                f"{info.get('speculative_algorithm')!r}, expected 'EAGLE3'"
            )
        if info.get("tp_size") != TP_SIZE or info.get("ep_size") != EP_SIZE:
            failures.append(
                f"{label} reports tp/ep={info.get('tp_size')}/{info.get('ep_size')}, "
                f"expected {TP_SIZE}/{EP_SIZE}"
            )
        if info.get("nnodes") != HOSTS_PER_ENGINE:
            failures.append(
                f"{label} reports nnodes={info.get('nnodes')!r}, expected "
                f"{HOSTS_PER_ENGINE}"
            )
    if decode_info.get("disaggregation_decode_retraction_backup") != RETRACTION_BACKUP:
        failures.append(
            "decode reports disaggregation_decode_retraction_backup="
            f"{decode_info.get('disaggregation_decode_retraction_backup')!r}, "
            f"expected {RETRACTION_BACKUP!r}"
        )
    return failures


def build_model_config() -> Dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-spec-gsm8k",
        "variant": "PD-1P1D-TP8+EP8+Raiden+EAGLE3-c16",
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
                "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
                "num_examples": NUM_EXAMPLES,
                "num_threads": NUM_THREADS,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                # Match the other speculative GSM8K gates so their acceptance
                # measurements use the same request contract.
                "api": "chat",
            },
        ],
    }


def run_prefill_leader() -> int:
    this_host = self_hostname()
    # CI role pod names map by swapping their replicated-job segment. Retained
    # CDK debug pods use one generic segment, so they pass the peer explicitly.
    decode_host = E(DECODE_HOST_ENV) or peer_hostname(
        this_host, from_role=PREFILL_ROLE, to_role=DECODE_ROLE
    )
    procs: List[Tuple[str, subprocess.Popen]] = []
    try:
        procs.append((PREFILL_ROLE, execute_prefill()))
        _wait_http_200(
            url=f"http://{LOCAL_HOST}:{PREFILL_PORT}/health",
            secs=WAIT_SECS,
            procs=procs,
        )
        _wait_http_200(url=f"http://{decode_host}:{DECODE_PORT}/health", secs=WAIT_SECS)
        config_failures = engine_configuration_failures(
            prefill_info=_get_json(f"http://{LOCAL_HOST}:{PREFILL_PORT}/server_info"),
            decode_info=_get_json(f"http://{decode_host}:{DECODE_PORT}/server_info"),
        )
        if config_failures:
            raise RuntimeError("; ".join(config_failures))
        procs.append(
            (
                "router",
                execute_router(prefill_host=this_host, decode_host=decode_host),
            )
        )
        _wait_http_200(
            url=f"http://{LOCAL_HOST}:{LB_PORT}/health", secs=300, procs=procs
        )
        model_config_main(
            build_model_config(),
            base_url=f"http://{LOCAL_HOST}:{LB_PORT}",
        )
        return 0
    finally:
        if not KEEP_UP:
            for _, proc in procs:
                _kill(proc)


def run_decode_worker() -> int:
    proc = execute_decode()
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            time.sleep(10)
    finally:
        _kill(proc)


def run_integration() -> int:
    validate_model_source(resolve_model_path(), label="target")
    validate_model_source(resolve_draft_model_path(), label="draft")
    if resolve_role() == PREFILL_ROLE:
        return run_prefill_leader()
    return run_decode_worker()


def render_jobset_yaml(
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights-rox",
    run_name: str = RUN_NAME,
) -> str:
    """Render one full prefill host plus one full decode host."""
    rel_path = os.path.relpath(__file__, REPO)
    cmd = [
        PYTHON,
        RENDER_JOBSET,
        "--pd-prefill-hosts",
        str(HOSTS_PER_ENGINE),
        "--pd-decode-hosts",
        str(HOSTS_PER_ENGINE),
        "--lane",
        lane,
        "--image",
        image,
        "--weights-pvc",
        weights_pvc,
        "--name",
        run_name,
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


class TestPdSingleHostSpecGptOss120bConfiguration(unittest.TestCase):
    """Launch-contract tests that do not initialize TPU hardware."""

    def test_each_engine_owns_one_full_host_without_ray(self) -> None:
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            self.assertEqual(env["TP_SIZE"], str(CHIPLETS_PER_HOST))
            self.assertEqual(env["EP_SIZE"], str(CHIPLETS_PER_HOST))
            self.assertEqual(env["FORCE_SINGLE_HOST"], "1")
            self.assertEqual(env["SGLANG_USE_RAY"], "0")
            for name in (
                "TPU_VISIBLE_DEVICES",
                "TPU_VISIBLE_CHIPS",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
            ):
                self.assertNotIn(name, env)

    def test_both_engines_enable_the_same_eagle3_and_raiden_config(self) -> None:
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            args = env["EXTRA_ARGS"]
            self.assertIn("--disaggregation-transfer-backend raiden", args)
            self.assertIn("--speculative-algorithm EAGLE3", args)
            self.assertIn(
                f"--speculative-draft-model-path {DEFAULT_DRAFT_MODEL_PATH}", args
            )
            self.assertIn("--speculative-num-steps 3", args)
            self.assertIn("--speculative-eagle-topk 1", args)
            self.assertIn("--speculative-num-draft-tokens 4", args)

    def test_decode_uses_cpu_retraction_without_hicache_layout(self) -> None:
        prefill_args = build_prefill_env(env={})["EXTRA_ARGS"]
        decode_args = build_decode_env(env={})["EXTRA_ARGS"]
        self.assertNotIn("disaggregation-decode-retraction-backup", prefill_args)
        self.assertIn(
            "--disaggregation-decode-retraction-backup cpu_tensor", decode_args
        )
        for args in (prefill_args, decode_args):
            self.assertNotIn("hicache", args)
            self.assertNotIn("page_first", args)

    def test_peer_hostname_and_role_resolution(self) -> None:
        prefill = "tpu-ci-pd-sh-gpt-120b-spec-prefill-0-0"
        self.assertEqual(
            peer_hostname(prefill, from_role=PREFILL_ROLE, to_role=DECODE_ROLE),
            "tpu-ci-pd-sh-gpt-120b-spec-decode-0-0",
        )
        self.assertEqual(resolve_role({ROLE_ENV: PREFILL_ROLE}), PREFILL_ROLE)
        self.assertEqual(resolve_role({ROLE_ENV: DECODE_ROLE}), DECODE_ROLE)
        self.assertEqual(
            self_hostname({SELF_HOST_ENV: "retained-prefill.example"}),
            "retained-prefill.example",
        )
        with self.assertRaisesRegex(ValueError, ROLE_ENV):
            resolve_role({})

    def test_gsm8k_uses_chat_api_and_spec_acceptance_threshold(self) -> None:
        gsm8k_gate = build_model_config()["gates"][0]
        self.assertEqual(gsm8k_gate["api"], "chat")
        self.assertEqual(gsm8k_gate["max_tokens"], 512)
        self.assertEqual(
            gsm8k_gate["spec_accept_length_threshold"],
            SPEC_ACCEPT_LENGTH_THRESHOLD,
        )

    def test_local_model_path_validation(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, MODEL_ENV):
            validate_model_source("/weights/missing-target", label="target")
        validate_model_source("openai/gpt-oss-120b", label="target")
        validate_model_source(DEFAULT_DRAFT_MODEL_PATH, label="draft")

    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_renders_one_host_per_role(self) -> None:
        yaml_content = render_jobset_yaml()
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertIn("completions: 1", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/"
            "test_tpu_gpt_oss_120b_pd_singlehost_spec.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-host-per-engine GPT-OSS P/D + EAGLE3 GSM8K test"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml())
        sys.exit(0)
    if args.mode == "integration":
        # model_config_main delegates to unittest and must not see this mode.
        sys.argv = [sys.argv[0], *extra_args]
        sys.exit(run_integration())
    sys.argv = [
        sys.argv[0],
        "TestPdSingleHostSpecGptOss120bConfiguration",
        *extra_args,
    ]
    unittest.main(verbosity=2)
