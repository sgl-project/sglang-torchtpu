"""GPT-OSS-120B P/D disaggregation GSM8K quality gate, one host per engine."""

import argparse
import json
import os
import socket
import subprocess
import sys
import unittest
import urllib.request
from collections.abc import Mapping
from typing import Any

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase, peer_hostname

# Expected runtime: 45 minutes; CI uses this for scheduling and timeouts.
register_tpu_ci(
    est_time=2700,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"

# Each engine uses one eight-chiplet TPU host.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

# Separate pods may reuse the same service and Raiden ports.
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"

PYTHON = E("PYTHON", "python3")

# The runtime image excludes the JobSet renderer.
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# Used in generated pod names, which Kubernetes limits to 63 characters.
RUN_NAME = "gpt-oss-120b-pd-singlehost-gsm8k"

PREFILL_LOG = "/tmp/pd_singlehost_prefill.log"
DECODE_LOG = "/tmp/pd_singlehost_decode.log"
LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost.py"

DISAGG_TRANSFER_BACKEND = "raiden"

# Defaults to a 0.77 passing floor; this topology measured 0.97.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.80")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    """Resolve the checkpoint from the environment, else the mounted default."""
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint."""
    if not os.path.isabs(model_path):
        return  # HF repo ids resolve remotely
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"GPT-OSS-120B checkpoint incomplete or missing at {model_path}, "
            f"provision it on the weights volume or set {MODEL_ENV}"
        )


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    """This pod's own DNS name."""
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def transport_failures(*, prefill_info: dict[str, Any], decode_info: dict[str, Any]):
    """Reject a run where the two engines are not a raiden P/D pair."""
    failures = []
    for label, info, expected_mode in (
        (PREFILL_ROLE, prefill_info, PREFILL_ROLE),
        (DECODE_ROLE, decode_info, DECODE_ROLE),
    ):
        mode = info.get("disaggregation_mode")
        if mode != expected_mode:
            failures.append(
                f"{label} engine reports disaggregation_mode={mode!r}, expected "
                f"{expected_mode!r}: it is not serving a P/D role"
            )
        backend = info.get("disaggregation_transfer_backend")
        if backend != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{label} engine reports disaggregation_transfer_backend="
                f"{backend!r}, expected {DISAGG_TRANSFER_BACKEND!r}"
            )
    return failures


def pd_transport_runner(*, prefill_url: str, decode_url: str):
    """Verify that both servers report the expected P/D roles and backend."""

    def run(base_url: str) -> dict[str, Any]:
        prefill_info = _get_json(f"{prefill_url}/server_info")
        decode_info = _get_json(f"{decode_url}/server_info")
        failures = transport_failures(
            prefill_info=prefill_info, decode_info=decode_info
        )
        print(
            f"[pd-singlehost] lb={base_url} prefill="
            f"{prefill_info.get('disaggregation_mode')}/"
            f"{prefill_info.get('disaggregation_transfer_backend')} decode="
            f"{decode_info.get('disaggregation_mode')}/"
            f"{decode_info.get('disaggregation_transfer_backend')}",
            flush=True,
        )
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def build_model_config(*, decode_host: str) -> dict[str, Any]:
    """Assemble the gate list, which needs the decode pod's resolved name."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-gsm8k",
        "variant": f"PD-singlehost-TP{TP_SIZE}+EP{EP_SIZE}-raiden",
        # Keep this specialized P/D result off the standard accuracy dashboard.
        "gates": [
            {
                "dataset": "pd_transport",
                "label": "pd-transport",
                "runner": pd_transport_runner(
                    prefill_url=f"http://{LOCAL_HOST}:{PREFILL_PORT}",
                    decode_url=f"http://{decode_host}:{DECODE_PORT}",
                ),
            },
            {
                "dataset": "gsm8k",
                "label": "gsm8k",
                "baseline": GSM8K_BASELINE,
                "atol": ATOL,
                "num_examples": NUM_EXAMPLES,
                "num_threads": NUM_THREADS,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
        ],
    }


class TestPdSingleHostGptOss120bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gates."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config(decode_host=cls.decode_hostname())

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(resolve_model_path())
        super().setUpClass()

    def test_accuracy(self) -> None:
        self.run_evaluations()


def render_jobset_yaml(
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights",
    run_name: str = RUN_NAME,
) -> str:
    """Render the JobSet for this lane: one host of prefill, one of decode."""
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
        # Names the dashboard row and the TIMINGS record.
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


class TestPdSingleHostGptOss120bConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {}
        for role in (PREFILL_ROLE, DECODE_ROLE):
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "benchmark.tpu.launch_server",
                    "--config",
                    LAUNCH_CONFIG_MODULE + ":" + role.upper() + "_CONFIG",
                    "--runtime-envs",
                    json.dumps(
                        {"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}
                    ),
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
            )
            cls.plans[role] = json.loads(output)

    def test_each_engine_takes_one_whole_host(self) -> None:
        for plan in self.plans.values():
            self.assertEqual(plan["server_args"]["tp_size"], CHIPLETS_PER_HOST)
            self.assertEqual(plan["server_args"]["ep_size"], CHIPLETS_PER_HOST)
            # Do not mask chiplets when each engine owns its host.
            for var in (
                "TPU_VISIBLE_DEVICES",
                "TPU_VISIBLE_CHIPS",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
            ):
                self.assertNotIn(var, plan["runtime_env_keys"])

    def test_both_engines_select_the_raiden_transport(self) -> None:
        for plan in self.plans.values():
            self.assertEqual(
                plan["server_args"]["disaggregation_transfer_backend"],
                DISAGG_TRANSFER_BACKEND,
            )

    def test_peer_hostname_maps_prefill_to_decode(self) -> None:
        self.assertEqual(
            peer_hostname(
                "tpu-ci-gpt-oss-120b-pd-singlehost-gsm8k-prefill-0-0",
                from_role=PREFILL_ROLE,
                to_role=DECODE_ROLE,
            ),
            "tpu-ci-gpt-oss-120b-pd-singlehost-gsm8k-decode-0-0",
        )
        with self.assertRaises(ValueError):
            peer_hostname(
                "no-role-segment", from_role=PREFILL_ROLE, to_role=DECODE_ROLE
            )

    def test_self_hostname_falls_back_to_the_pod_name(self) -> None:
        # Single-host pods may omit TPU_WORKER_HOSTNAMES.
        self.assertEqual(
            self_hostname({"TPU_WORKER_HOSTNAMES": "a-prefill-0-0"}), "a-prefill-0-0"
        )
        self.assertEqual(self_hostname({}), socket.gethostname())

    def test_transport_gate_rejects_a_non_disaggregated_pair(self) -> None:
        raiden = {
            "disaggregation_mode": "null",
            "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
        }
        self.assertTrue(transport_failures(prefill_info=raiden, decode_info=raiden))
        native = {
            "disaggregation_mode": PREFILL_ROLE,
            "disaggregation_transfer_backend": "tpu",
        }
        self.assertTrue(
            transport_failures(
                prefill_info=native,
                decode_info={**native, "disaggregation_mode": DECODE_ROLE},
            )
        )
        self.assertEqual(
            transport_failures(
                prefill_info={
                    "disaggregation_mode": PREFILL_ROLE,
                    "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
                },
                decode_info={
                    "disaggregation_mode": DECODE_ROLE,
                    "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
                },
            ),
            [],
        )

    def test_local_model_path_validation(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, MODEL_ENV):
            validate_model_source("/weights/definitely-missing-gpt-oss-120b")
        validate_model_source("openai/gpt-oss-120b")

    def test_run_name_fits_the_generated_pod_name(self) -> None:
        # Kubernetes rejects generated pod names longer than 63 characters.
        pod_name = f"tpu-ci-{RUN_NAME}-xxxxx-prefill-0-0-xxxxx"
        self.assertLessEqual(len(pod_name), 63, pod_name)

    # Skip in runtime images, which do not contain the renderer.
    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_render_jobset_yaml(self) -> None:
        # Verify the workflow name, roles, and command remain aligned.
        yaml_content = render_jobset_yaml()
        self.assertIn(
            "generateName: tpu-ci-gpt-oss-120b-pd-singlehost-gsm8k-", yaml_content
        )
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/test_tpu_gpt_oss_120b_pd_singlehost.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-host-per-engine P/D GSM8K")
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

    test_case = (
        TestPdSingleHostGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
