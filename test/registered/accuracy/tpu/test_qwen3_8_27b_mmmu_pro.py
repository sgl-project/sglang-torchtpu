"""Qwen3.8-27B TPU accuracy on all 1,730 MMMU-Pro standard-10 questions."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu import QWEN3_8_27B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

# CI measured 5490 s (14 min bring-up + 77 min eval); the JobSet deadline is
# 1.5 x est_time.
register_tpu_ci(
    est_time=6000,
    suite="nightly-tpu-accuracy-qwen3-8-27b-mmmupro",
    nightly=True,
    labels=["singlehost", "qwen3.8-27b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/qwen3_8_27b/mmmu_pro.py"
# The /weights mount when it carries the checkpoint, else the HF id.
MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3.8-27B")
# The card's sampling with a 32k thinking budget: the vision recipe serves a 40k
# context at 32 concurrent requests, not the text recipe's 86k at 64.
QWEN3_8_27B_MMMU = {
    **QWEN3_8_27B,
    "config_module": CONFIG_MODULE,
    "sampling": {**QWEN3_8_27B["sampling"], "max_tokens": 32768, "num_threads": 32},
}

GATE = {
    "dataset": "mmmu_pro",
    "label": "mmmu_pro",
    "baseline": 0.749,  # TPU BF16, full 1,730-example run on 2026-10-15.
    "atol": 0.1,
    "runner": make_runner(QWEN3_8_27B_MMMU, "mmmu_pro"),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        QWEN3_8_27B_MMMU,
        test_name="qwen3-8-27b-mmmupro",
        # The dashboard's series join key: never renamed, never shown.
        series_id="accuracy-qwen3-8-27b-mmmupro",
        gate=GATE,
    )


class TestQwen38MMMUPro(AccuracyTestCase):
    """Score Qwen3.8-27B mmmu_pro on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = MODEL_PATH
    tag = "qwen3-8-27b-mmmupro-e2e"
    health_timeout = 3600

    @classmethod
    def get_runtime_envs(cls):
        # The checkpoint may still come from HF, so pass the resolved path.
        return {**super().get_runtime_envs(), "MODEL": MODEL_PATH}

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestQwen38MMMUProConfiguration(CustomTestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, QWEN3_8_27B_MMMU["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
