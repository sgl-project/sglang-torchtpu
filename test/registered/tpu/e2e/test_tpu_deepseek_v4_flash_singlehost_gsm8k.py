"""DeepSeek-V4-Flash single-host e2e quality gate on TPU v7x-8 (TP=8, native packed-FP4 W4A8 MoE).

Validates end-to-end model quality on GSM8K and MGSM-EN using the DSV4 hybrid
MLA + SWA attention backend and native packed-FP4 (W4A8) MoE path on TPU.

Serving config lives in recipe/configs/deepseek_v4_flash.py. To run against an
already-running server, set SGLANG_TPU_USE_LIVE_SERVER=1 and BASE_URL.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=1800,
    suite="stage-b-test-tpu-4-v7",
    disabled="Pending DeepSeek-V4-Flash checkpoint staging on shared CI PVC",
    labels=["singlehost", "deepseek-v4-flash", "accuracy"],
)

SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "deepseek-v4-flash")
MODEL_PATH = os.environ.get("MODEL", "/weights/deepseek-v4-flash")
CONFIG_MODULE = "recipe/configs/deepseek_v4_flash.py"

GATES = [
    {
        "dataset": "gsm8k",
        "label": "gsm8k",
        # Full 1,319-example GSM8K measured 0.9703 on TPU v7x-8 (TP=8, W4A8 MoE).
        "baseline": 0.95,
        "atol": 0.05,
        "num_examples": int(os.environ.get("GSM8K_NUM_EXAMPLES", "200")),
        "num_threads": 32,
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    {
        "dataset": "mgsm_en",
        "label": "mgsm_en",
        # Full 250-example MGSM-EN measured 0.9760 on TPU v7x-8 (TP=8, W4A8 MoE).
        "baseline": 0.95,
        "atol": 0.05,
        "num_examples": int(os.environ.get("MGSM_NUM_EXAMPLES", "250")),
        "num_threads": 32,
        "max_tokens": 1024,
        "temperature": 0.0,
    },
]


def build_model_config():
    """Assemble the DeepSeek-V4-Flash quality gates."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "deepseek-v4-flash-gsm8k",
        "variant": "TP8+W4A8-MoE",
        "gates": GATES,
    }


class TestDeepSeekV4FlashGSM8K(AccuracyTestCase):
    """Score DeepSeek-V4-Flash on GSM8K and MGSM-EN."""

    config_module = CONFIG_MODULE
    model_path = MODEL_PATH
    tag = "deepseek-v4-flash-gsm8k-e2e"

    @classmethod
    def get_runtime_envs(cls):
        return {**super().get_runtime_envs(), "MODEL": MODEL_PATH}

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestDeepSeekV4FlashConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_every_gate_fits_the_server_its_recipe_starts(self):
        for gate in GATES:
            with self.subTest(gate=gate["label"]):
                self.assertEqual(gate_server_failures(CONFIG_MODULE, gate), [])


if __name__ == "__main__":
    unittest.main()
