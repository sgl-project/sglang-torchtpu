"""GPT-OSS-120B P/D gate for RAM KV offload on both engines with DFlash.

Reuses test_tpu_gpt_oss_120b_pd_singlehost_kv_offload_spec.py with DFlash on
the experimental spec worker. The decode engine's RAM pool holds full-attention
and DFlash draft KV; the probe and GSM8K gates are the EAGLE3 test's.
"""

import argparse
import json
import os
import subprocess
import sys
import unittest
from typing import Any

# Module import, not the classes: unittest would also collect them.
import test_tpu_gpt_oss_120b_pd_singlehost_kv_offload_spec as base

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=5400,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "singlehost",
        "gpt-oss-120b",
        "pd",
        "spec",
        "dflash",
        "prefix-cache",
        "kv-offload",
        "decode-radix",
    ],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_kv_offload_dflash.py"
SPEC_ALGORITHM = "DFLASH"
DEFAULT_DRAFT_MODEL_PATH = "z-lab/gpt-oss-120b-DFlash"
RUN_NAME = "pd-sh-gpt-120b-offload-dflash"

# Measured 3.40 over 200 GSM8K examples on tpu7x (GSM8K 0.84).
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "3.0")
)


def resolve_draft_model_path() -> str:
    return (
        os.environ.get(base.DRAFT_MODEL_ENV)
        or os.environ.get("DRAFT_MODEL")
        or DEFAULT_DRAFT_MODEL_PATH
    )


def build_model_config(*, decode_host: str) -> dict[str, Any]:
    config = base.build_model_config(decode_host=decode_host)
    gsm8k = next(g for g in config["gates"] if g["dataset"] == "gsm8k")
    gsm8k["spec_accept_length_threshold"] = SPEC_ACCEPT_LENGTH_THRESHOLD
    return {
        **config,
        "test_name": "gpt-oss-120b-pd-singlehost-kv-offload-dflash",
        "variant": "PD-1P1D-TP8+EP8+Raiden+DFLASH+offload-both",
    }


class TestPdSingleHostKvOffloadDflashGptOss120bIntegration(
    base.TestPdSingleHostKvOffloadSpecGptOss120bIntegration
):
    config_module = LAUNCH_CONFIG_MODULE
    tag = "pd-singlehost-kv-offload-dflash"

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config(decode_host=cls.decode_hostname())

    @classmethod
    def check_server_configuration(cls) -> None:
        url = f"http://{base.LOCAL_HOST}:{base.PREFILL_PORT}/server_info"
        failures = base.engine_configuration_failures(
            prefill_info=base._get_json(url),
            decode_info=base._get_json(
                f"http://{cls.decode_hostname()}:{base.DECODE_PORT}/server_info"
            ),
            speculative_algorithm=SPEC_ALGORITHM,
        )
        if failures:
            raise RuntimeError("; ".join(failures))

    @classmethod
    def setUpClass(cls) -> None:
        base.validate_model_source(base.resolve_model_path(), label="target")
        base.validate_model_source(resolve_draft_model_path(), label="draft")
        # Skip the EAGLE3 parent's draft check; run the shared P/D fixture's setup.
        super(base.TestPdSingleHostKvOffloadSpecGptOss120bIntegration, cls).setUpClass()


class TestPdSingleHostKvOffloadDflashGptOss120bConfiguration(unittest.TestCase):
    """Launch-contract tests that do not initialize TPU hardware."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {}
        for role in (base.PREFILL_ROLE, base.DECODE_ROLE):
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
                cwd=base.REPO,
                text=True,
            )
            cls.plans[role] = json.loads(output)

    def test_both_engines_run_dflash_on_the_experimental_worker(self) -> None:
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["speculative_algorithm"], SPEC_ALGORITHM)
            self.assertTrue(args["enable_hierarchical_cache"])
            self.assertEqual(args["hicache_io_backend"], "raiden")
            self.assertIn(
                "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER", plan["runtime_env_keys"]
            )

    def test_configuration_gate_requires_dflash(self) -> None:
        info = {
            "disaggregation_transfer_backend": base.DISAGG_TRANSFER_BACKEND,
            "enable_hierarchical_cache": True,
            "hicache_io_backend": "raiden",
            "tp_size": base.TP_SIZE,
            "ep_size": base.EP_SIZE,
            "speculative_algorithm": "EAGLE3",
            "disaggregation_decode_enable_radix_cache": True,
            "disaggregation_decode_retraction_backup": base.RETRACTION_BACKUP,
        }
        failures = base.engine_configuration_failures(
            prefill_info={**info, "disaggregation_mode": base.PREFILL_ROLE},
            decode_info={**info, "disaggregation_mode": base.DECODE_ROLE},
            speculative_algorithm=SPEC_ALGORITHM,
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("expected 'DFLASH'" in f for f in failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-host-per-engine GPT-OSS P/D RAM offload + DFlash test"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(base.render_jobset_yaml(run_name=RUN_NAME, test_path=__file__))
        sys.exit(0)
    test_case = (
        TestPdSingleHostKvOffloadDflashGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostKvOffloadDflashGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
