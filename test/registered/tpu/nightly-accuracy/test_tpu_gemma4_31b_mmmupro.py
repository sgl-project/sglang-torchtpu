"""Gemma-4-31B TPU MMMU-Pro score, all 1,730 standard-10 questions."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import GEMMA4_31B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=6060,
    suite="nightly-tpu-accuracy-gemma4-31b-mmmupro",
    nightly=True,
    labels=["singlehost", "gemma4-31b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/gemma4_31b/accuracy.py"
MODEL_PATH = GEMMA4_31B["served_model_name"]

GATE = {
    "dataset": "mmmu_pro",
    # sgl-eval's own grader: its own series, apart from the lenient manual re-grades.
    "label": "mmmu_pro_strict",
    "baseline": 0.448,  # measured on TPU
    "atol": 0.1,
    "runner": make_runner(GEMMA4_31B, "mmmu_pro"),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        GEMMA4_31B,
        test_name="gemma4-31b-mmmupro",
        series_id="accuracy-gemma4-31b-mmmupro",
        gate=GATE,
    )


class TestGemma31BMMMUPro(AccuracyTestCase):
    """Score Gemma-4-31B mmmu_pro_strict on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/gemma-4-31B-it"
    tag = "gemma4-31b-mmmupro-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGemma31BMMMUProConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, GEMMA4_31B["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
