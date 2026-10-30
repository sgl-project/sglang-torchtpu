"""Muse Spark NVFP4 TPU GSM8K nightly quality evaluation."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import MUSE_SPARK, make_model_config, sgl_eval_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase

register_tpu_ci(
    est_time=10800,
    suite="nightly-tpu-accuracy-muse-spark-gsm8k",
    nightly=True,
    labels=["singlehost", "muse-spark-open-816b-a42b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/muse_spark/accuracy.py"

# Full TPU calibration, 2026-10-24: 97.6497% on all 1,319 questions.
# https://github.com/sgl-project/sglang-torchtpu/actions/runs/40070907182
GATE = {
    "dataset": "gsm8k",
    "label": "gsm8k",
    "baseline": 0.9765,
    "atol": 0.10,
    "runner": sgl_eval_runner(
        "gsm8k",
        model=MUSE_SPARK["served_model_name"],
        max_tokens=4096,
        num_threads=8,
        temperature=0.0,
        top_p=1.0,
        chat_template_kwargs={"reasoning_effort": "low"},
    ),
}


def build_model_config():
    return make_model_config(
        MUSE_SPARK,
        test_name="muse-spark-gsm8k",
        series_id="accuracy-muse-spark-gsm8k",
        gate=GATE,
    )


class TestMuseSparkGSM8K(AccuracyTestCase):
    """Score this benchmark on the backbone recipe's server."""

    config_module = CONFIG_MODULE
    model_path = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
    tag = "muse-spark-gsm8k-e2e"
    health_timeout = 5400

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
