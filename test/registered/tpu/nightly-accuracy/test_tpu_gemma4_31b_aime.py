"""Gemma-4-31B TPU AIME 2026 nightly score, mean of four passes."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import GEMMA4_31B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=2540,
    suite="nightly-tpu-accuracy-gemma4-31b-aime",
    nightly=True,
    labels=["singlehost", "gemma4-31b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/gemma4_31b/accuracy.py"
MODEL_PATH = GEMMA4_31B["served_model_name"]

GATE = {
    "dataset": "aime26",
    "label": "aime26",
    "baseline": 0.887,  # measured on TPU
    "atol": 0.12,
    "runner": make_runner(GEMMA4_31B, "aime26", n_repeats=4),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        GEMMA4_31B,
        test_name="gemma4-31b-aime",
        series_id="accuracy-gemma4-31b-aime",
        gate=GATE,
    )


class TestGemma31BAIME(AccuracyTestCase):
    """Score Gemma-4-31B aime26 on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/gemma-4-31B-it"
    tag = "gemma4-31b-aime-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGemma31BAIMEConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, GEMMA4_31B["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
