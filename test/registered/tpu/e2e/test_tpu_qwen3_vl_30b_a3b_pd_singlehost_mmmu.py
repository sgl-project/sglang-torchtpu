"""Qwen3-VL-30B-A3B P/D MMMU-Pro (vision) gate, one TPU host per engine.

Prefill and decode each own one full v7x host (TP8/EP8) and exchange KV over
Raiden. The prefill engine runs the ViT, so the handoff carries image-derived KV
into a fused-MoE decode engine that never runs the ViT. The serving settings
and the gate match test_tpu_qwen3_vl_30b_a3b_singlehost_mmmu.py, so a score
that drops only here points at the handoff. The prefill pod also runs the mini
router and scores MMMU-Pro through it.
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
    # Both engines start in parallel (the single-host cold start was ~21 min
    # to ready), then ~10 min of dataset prep and ~5 min of eval.
    est_time=3600,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "qwen3-vl-30b-a3b", "pd", "multimodal", "compile-tracer"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

LAUNCH_CONFIG_MODULE = "recipe/configs/qwen3_vl_30b_a3b/pd_singlehost.py"

E = os.environ.get

SERVED_MODEL_NAME = "qwen3-vl-30b-a3b"
DEFAULT_MODEL_PATH = "/weights/Qwen3-VL-30B-A3B-Instruct"
MODEL_ENV = "QWEN3_VL_30B_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

# A v7x host has four physical chips and eight addressable TPU devices.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = CHIPLETS_PER_HOST
EP_SIZE = CHIPLETS_PER_HOST

# The roles live in different pods, so their HTTP ports may match.
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "3600"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))
PYTHON = E("PYTHON", "python3")

# Local dry-run name only; CI derives the run name from the file path. Peer
# discovery swaps the "-prefill-" and "-decode-" hostname segments, so the
# name cannot contain either role.
RUN_NAME = "pd-sh-qwen3-vl-30b"

PREFILL_LOG = "/tmp/pd_qwen3_vl_30b_prefill.log"
DECODE_LOG = "/tmp/pd_qwen3_vl_30b_decode.log"

DISAGG_TRANSFER_BACKEND = "raiden"
RETRACTION_BACKUP = "cpu_tensor"
# Must match the recipe: the decode engine processes each request's images
# again, and a different cap there changes the prompt length it expects.
MM_PROCESS_CONFIG = {"image": {"min_pixels": 65536, "max_pixels": 313600}}
COMPILE_TRACER = "direct"

# The single-host gate, unchanged, so the two scores compare directly.
MMMU_GATE = {
    "dataset": "mmmu_pro_vision",
    "label": "mmmu_pro_vision",
    "baseline": 0.45,
    "atol": 0.08,
    "num_examples": 200,
    "num_threads": 8,
    "max_tokens": 4096,
    "temperature": 0.0,
}


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Fail clearly for a missing mounted checkpoint; HF/GCS ids are remote."""
    if not os.path.isabs(model_path):
        return
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"checkpoint incomplete or missing at {model_path}; "
            f"provision it on the weights volume or set {MODEL_ENV}"
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
    for role, info in ((PREFILL_ROLE, prefill_info), (DECODE_ROLE, decode_info)):
        if info.get("disaggregation_mode") != role:
            failures.append(
                f"{role} reports disaggregation_mode="
                f"{info.get('disaggregation_mode')!r}, expected {role!r}"
            )
        if info.get("disaggregation_transfer_backend") != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{role} reports disaggregation_transfer_backend="
                f"{info.get('disaggregation_transfer_backend')!r}, expected "
                f"{DISAGG_TRANSFER_BACKEND!r}"
            )
        if info.get("tp_size") != TP_SIZE or info.get("ep_size") != EP_SIZE:
            failures.append(
                f"{role} reports tp/ep={info.get('tp_size')}/{info.get('ep_size')}, "
                f"expected {TP_SIZE}/{EP_SIZE}"
            )
        if info.get("nnodes") != HOSTS_PER_ENGINE:
            failures.append(
                f"{role} reports nnodes={info.get('nnodes')!r}, expected "
                f"{HOSTS_PER_ENGINE}"
            )
        if info.get("tpu_compile_tracer") != COMPILE_TRACER:
            failures.append(
                f"{role} reports tpu_compile_tracer="
                f"{info.get('tpu_compile_tracer')!r}, expected {COMPILE_TRACER!r}"
            )
        if not info.get("use_spmd_compile"):
            failures.append(
                f"{role} reports use_spmd_compile="
                f"{info.get('use_spmd_compile')!r}, expected True"
            )
        if info.get("mm_process_config") != MM_PROCESS_CONFIG:
            failures.append(
                f"{role} reports mm_process_config="
                f"{info.get('mm_process_config')!r}, expected {MM_PROCESS_CONFIG!r}"
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
        "test_name": "qwen3-vl-30b-a3b-pd-singlehost-mmmu-pro-vision",
        "variant": "PD-1P1D-TP8+EP8+Raiden",
        "gates": [dict(MMMU_GATE)],
    }


class TestPdSingleHostQwen3Vl30bA3bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gate."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-qwen3-vl-30b"
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


class TestPdSingleHostQwen3Vl30bA3bConfiguration(unittest.TestCase):
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

    def test_both_engines_share_raiden_and_the_image_cap(self) -> None:
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["disaggregation_transfer_backend"], "raiden")
            self.assertEqual(args["mm_process_config"], MM_PROCESS_CONFIG)
            self.assertEqual(args["served_model_name"], SERVED_MODEL_NAME)
            self.assertEqual(args["model_path"], resolve_model_path())
            # The dict reaches the server as one argv element; its JSON parser
            # accepts the spaces json.dumps adds.
            flag = plan["command"].index("--mm-process-config")
            self.assertEqual(json.loads(plan["command"][flag + 1]), MM_PROCESS_CONFIG)

    def test_both_engines_trace_with_direct_compile(self) -> None:
        # The gate must exercise the tracer and SPMD sharing the recipes serve
        # with; a dynamo or unshared engine would hide a compile regression.
        for plan in self.plans.values():
            self.assertEqual(plan["server_args"]["tpu_compile_tracer"], COMPILE_TRACER)
            self.assertIs(plan["server_args"]["use_spmd_compile"], True)
            self.assertIn("--use-spmd-compile", plan["command"])

    def test_bf16_checkpoint_is_served_unquantized(self) -> None:
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["quantization"], "unquant")
            self.assertEqual(args["kv_cache_dtype"], "bfloat16")
            self.assertEqual(args["dtype"], "bfloat16")

    def test_gate_fits_the_compiled_serving_shape(self) -> None:
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["torch_compile_max_bs"], args["max_running_requests"])
            self.assertLessEqual(MMMU_GATE["num_threads"], args["max_running_requests"])
            self.assertLess(MMMU_GATE["max_tokens"], args["context_length"])
            self.assertIn(args["max_prefill_tokens"], args["tpu_compile_tokens"])

    def test_decode_uses_cpu_retraction_without_hicache_layout(self) -> None:
        prefill_args = self.plans[PREFILL_ROLE]["server_args"]
        decode_args = self.plans[DECODE_ROLE]["server_args"]
        self.assertNotIn("disaggregation_decode_retraction_backup", prefill_args)
        self.assertEqual(
            decode_args["disaggregation_decode_retraction_backup"], RETRACTION_BACKUP
        )
        for args in (prefill_args, decode_args):
            self.assertFalse(any(name.startswith("hicache") for name in args))

    def test_peer_hostname_resolution(self) -> None:
        prefill = f"tpu-ci-{RUN_NAME}-prefill-0-0"
        self.assertEqual(
            peer_hostname(prefill, from_role=PREFILL_ROLE, to_role=DECODE_ROLE),
            f"tpu-ci-{RUN_NAME}-decode-0-0",
        )
        self.assertEqual(
            self_hostname({SELF_HOST_ENV: "retained-prefill.example"}),
            "retained-prefill.example",
        )

    def test_local_model_path_validation(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, MODEL_ENV):
            validate_model_source("/weights/missing-qwen3-vl-30b-a3b")
        validate_model_source("Qwen/Qwen3-VL-30B-A3B-Instruct")

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
            "test_tpu_qwen3_vl_30b_a3b_pd_singlehost_mmmu.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-host-per-engine Qwen3-VL-30B-A3B P/D MMMU-Pro test"
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
        TestPdSingleHostQwen3Vl30bA3bIntegration
        if args.mode == "integration"
        else TestPdSingleHostQwen3Vl30bA3bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
