"""GPT-OSS 120B TPU GPQA Diamond nightly score."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    GPT_OSS_120B,
    gpqa_openai_runner,
    make_model_config,
)
from sglang.test.tpu.utils.accuracy_base import (
    AccuracyTestCase,
    gate_server_failures,
)

register_tpu_ci(
    est_time=2690,
    suite="nightly-tpu-accuracy-gpt-oss-120b-gpqa",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

MAX_TOKENS = 65536

GATE = {
    "dataset": "gpqa",
    "label": "gpqa_diamond_high",
    "baseline": 0.801,  # model card
    "atol": 0.1,
    "runner": gpqa_openai_runner(
        model=GPT_OSS_120B["served_model_name"],
        n_repeats=1,
        reasoning_effort="high",
        max_tokens=MAX_TOKENS,
        n_threads=16,
        num_examples=198,
    ),
}

CONFIG_MODULE = "recipe/configs/gpt_oss_120b/accuracy_gpqa.py"


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        GPT_OSS_120B,
        test_name="gpt-oss-120b-gpqa",
        series_id="accuracy-gpt-oss-120b-gpqa",
        gate=GATE,
    )


class TestGptOss120bGPQA(AccuracyTestCase):
    """Score GPT-OSS-120B gpqa on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/gpt-oss-120b"
    tag = "gpt-oss-120b-gpqa-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGptOss120bGPQAConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(gate_server_failures(CONFIG_MODULE, GATE), [])


if __name__ == "__main__":
    unittest.main()
