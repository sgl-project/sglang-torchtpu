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
import socket
import subprocess
import sys
import unittest
import urllib.request
from collections.abc import Mapping
from typing import Any

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase, peer_hostname

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

LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_spec.py"

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
DEFAULT_DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"
MODEL_ENV = "GPT_OSS_MODEL"
DRAFT_MODEL_ENV = "GPT_OSS_DRAFT_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
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
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "5400"))
PYTHON = E("PYTHON", "python3")

# Keep the run name short enough for the generated Job and Pod suffixes. It
# also cannot contain the literal role names because peer discovery swaps the
# "-prefill-" and "-decode-" hostname segments.
RUN_NAME = "pd-sh-gpt-120b-spec"

PREFILL_LOG = "/tmp/pd_spec_prefill.log"
DECODE_LOG = "/tmp/pd_spec_decode.log"

DISAGG_TRANSFER_BACKEND = "raiden"
RETRACTION_BACKUP = "cpu_tensor"

SPEC_ACCEPT_LENGTH_THRESHOLD = float(E("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90"))

# The non-P/D GPT-OSS EAGLE3 gate measured 0.785 over 200 examples. Keep the
# same floor and concurrency so this test isolates the P/D handoff.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B_SPEC", "0.70"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "200"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "16"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "512"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "5400"))


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def resolve_draft_model_path(env: Mapping[str, str] | None = None) -> str:
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


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def engine_configuration_failures(
    *, prefill_info: Mapping[str, Any], decode_info: Mapping[str, Any]
) -> list[str]:
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


def build_model_config() -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-spec-gsm8k",
        "variant": "PD-1P1D-TP8+EP8+Raiden+EAGLE3-c16",
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


class TestPdSingleHostSpecGptOss120bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gates."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-spec"
    health_timeout = 2 * WAIT_SECS + 300
    health_generate_timeout = HEALTH_TIMEOUT

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config()

    @classmethod
    def check_server_configuration(cls) -> None:
        config_failures = engine_configuration_failures(
            prefill_info=_get_json(f"http://{LOCAL_HOST}:{PREFILL_PORT}/server_info"),
            decode_info=_get_json(
                f"http://{cls.decode_hostname()}:{DECODE_PORT}/server_info"
            ),
        )
        if config_failures:
            raise RuntimeError("; ".join(config_failures))

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(resolve_model_path(), label="target")
        validate_model_source(resolve_draft_model_path(), label="draft")
        super().setUpClass()

    def test_accuracy(self) -> None:
        self.run_evaluations()


def render_jobset_yaml(
    image: str = "sglang-tpu:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights",
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

    def test_each_engine_owns_one_full_host_without_ray(self) -> None:
        for plan in self.plans.values():
            self.assertEqual(plan["server_args"]["tp_size"], CHIPLETS_PER_HOST)
            self.assertEqual(plan["server_args"]["ep_size"], CHIPLETS_PER_HOST)
            self.assertEqual(plan["server_args"]["nnodes"], 1)
            self.assertFalse(plan["server_args"].get("use_ray", False))
            for name in (
                "TPU_VISIBLE_DEVICES",
                "TPU_VISIBLE_CHIPS",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
            ):
                self.assertNotIn(name, plan["runtime_env_keys"])

    def test_both_engines_enable_the_same_eagle3_and_raiden_config(self) -> None:
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["disaggregation_transfer_backend"], "raiden")
            self.assertEqual(args["speculative_algorithm"], "EAGLE3")
            self.assertEqual(
                args["speculative_draft_model_path"], resolve_draft_model_path()
            )
            self.assertEqual(args["speculative_num_steps"], 3)
            self.assertEqual(args["speculative_eagle_topk"], 1)
            self.assertEqual(args["speculative_num_draft_tokens"], 4)

    def test_decode_uses_cpu_retraction_without_hicache_layout(self) -> None:
        prefill_args = self.plans[PREFILL_ROLE]["server_args"]
        decode_args = self.plans[DECODE_ROLE]["server_args"]
        self.assertNotIn("disaggregation_decode_retraction_backup", prefill_args)
        self.assertEqual(
            decode_args["disaggregation_decode_retraction_backup"], "cpu_tensor"
        )
        for args in (prefill_args, decode_args):
            self.assertFalse(any(name.startswith("hicache") for name in args))

    def test_peer_hostname_resolution(self) -> None:
        prefill = "tpu-ci-pd-sh-gpt-120b-spec-prefill-0-0"
        self.assertEqual(
            peer_hostname(prefill, from_role=PREFILL_ROLE, to_role=DECODE_ROLE),
            "tpu-ci-pd-sh-gpt-120b-spec-decode-0-0",
        )
        self.assertEqual(
            self_hostname({SELF_HOST_ENV: "retained-prefill.example"}),
            "retained-prefill.example",
        )

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
    test_case = (
        TestPdSingleHostSpecGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostSpecGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
