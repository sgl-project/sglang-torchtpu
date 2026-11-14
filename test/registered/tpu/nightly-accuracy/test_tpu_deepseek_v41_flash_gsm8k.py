"""DeepSeek-V4.1-Flash TPU full-dataset GSM8K nightly score."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import DEEPSEEK_V41_FLASH, make_model_config
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=5400,
    suite="nightly-tpu-accuracy-deepseek-v41-flash-gsm8k",
    nightly=True,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy"],
)

CONFIG_MODULE = DEEPSEEK_V41_FLASH["config_module"]

GATE = {
    "dataset": "gsm8k",
    "label": "gsm8k_full",
    "baseline": 0.971,  # measured on TPU
    "atol": 0.1,
    "num_examples": 1319,
    "num_threads": 128,
    "max_tokens": 1024,
    "temperature": 0.0,
    "thinking": False,
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        DEEPSEEK_V41_FLASH,
        test_name="deepseek-v41-flash-gsm8k",
        # The dashboard's series join key: never renamed, never shown.
        series_id="accuracy-deepseek-v41-flash-gsm8k",
        gate=GATE,
    )


class TestDeepSeekV41FlashGSM8K(AccuracyTestCase):
    """Score DeepSeek-V4.1-Flash gsm8k on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/DeepSeek-V4.1-Flash"
    tag = "deepseek-v41-flash-gsm8k-e2e"
    # Boot measured 67 min: weight and Engram load ~30 min, then eight graphs.
    health_timeout = 5400

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestDeepSeekV41FlashGSM8KConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(gate_server_failures(CONFIG_MODULE, GATE), [])


if __name__ == "__main__":
    unittest.main()
