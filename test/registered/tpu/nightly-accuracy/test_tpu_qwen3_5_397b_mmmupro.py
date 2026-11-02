"""Qwen3.5-397B-A17B-FP8 TPU MMMU-Pro score on the nightly question set (mmmu_full).

sgl-eval's own loader and grader, thinking on, with the model card's sampling.
"""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import QWEN3_5_397B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=6000,
    suite="nightly-tpu-accuracy-qwen3-5-397b-mmmupro",
    nightly=True,
    labels=["singlehost", "qwen3.5-397b", "accuracy"],
)

CONFIG_MODULE = QWEN3_5_397B["config_module"]

GATE = {
    "dataset": "mmmu_pro",
    "label": "mmmu_pro",
    "baseline": 0.786,  # measured on TPU
    "atol": 0.1,
    "runner": make_runner(QWEN3_5_397B, "mmmu_pro"),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        QWEN3_5_397B,
        test_name="qwen3-5-397b-mmmupro",
        # The dashboard's series join key: never renamed, never shown.
        series_id="accuracy-qwen3-5-397b-mmmupro",
        gate=GATE,
    )


class TestQwen35MMMUPro(AccuracyTestCase):
    """Score Qwen3.5-397B-FP8 mmmu_pro on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/Qwen3.5-397B-A17B-FP8"
    tag = "qwen3-5-mmmupro-e2e"
    # One host at tp8: the FP8 bring-up measured 42-44 min in CI.
    health_timeout = 3600

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestQwen35MMMUProConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, QWEN3_5_397B["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
