"""Gemma-4-26B-A4B TPU GPQA Diamond nightly score, mean of four passes."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import GEMMA4_26B_A4B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=4760,
    suite="nightly-tpu-accuracy-gemma4-26b-a4b-gpqa",
    nightly=True,
    labels=["singlehost", "gemma4-26b-a4b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/gemma4_26b_a4b/accuracy.py"
MODEL_PATH = GEMMA4_26B_A4B["served_model_name"]

GATE = {
    "dataset": "gpqa",
    "label": "gpqa_diamond",
    "baseline": 0.786,  # measured on TPU
    "atol": 0.1,
    "runner": make_runner(GEMMA4_26B_A4B, "gpqa", n_repeats=4),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        GEMMA4_26B_A4B,
        test_name="gemma4-26b-a4b-gpqa",
        series_id="accuracy-gemma4-26b-a4b-gpqa",
        gate=GATE,
    )


class TestGemma26BA4BGPQA(AccuracyTestCase):
    """Score Gemma-4-26B-A4B gpqa_diamond on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/gemma-4-26B-A4B-it"
    tag = "gemma4-26b-a4b-gpqa-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGemma26BA4BGPQAConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, GEMMA4_26B_A4B["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
