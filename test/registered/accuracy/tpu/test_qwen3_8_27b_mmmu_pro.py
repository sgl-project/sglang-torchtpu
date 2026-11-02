"""Qwen3.8-27B TPU accuracy on the nightly MMMU-Pro question set (mmmu_full)."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import QWEN3_8_27B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=4500,
    suite="nightly-tpu-accuracy-qwen3-8-27b-mmmupro",
    nightly=True,
    labels=["singlehost", "qwen3.8-27b", "accuracy"],
)

CONFIG_MODULE = QWEN3_8_27B["config_module"]
# The /weights mount when it carries the checkpoint, else the HF id.
MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3.8-27B")

GATE = {
    "dataset": "mmmu_pro",
    "label": "mmmu_pro",
    "baseline": 0.749,  # TPU BF16, full 1,730-example run on 2026-10-15.
    "atol": 0.1,
    "runner": make_runner(QWEN3_8_27B, "mmmu_pro"),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        QWEN3_8_27B,
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


class TestQwen38MMMUProConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, QWEN3_8_27B["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
