"""Muse Spark NVFP4 TPU GPQA nightly quality evaluation."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import MUSE_SPARK, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase

register_tpu_ci(
    est_time=10800,
    suite="nightly-tpu-accuracy-muse-spark-gpqa",
    nightly=True,
    labels=["singlehost", "muse-spark-open-816b-a42b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/muse_spark/accuracy.py"

GATE = {
    "dataset": "gpqa",
    "label": "gpqa",
    "baseline": 0.89,  # Muse Spark Open model card, xhigh reasoning.
    "atol": 0.10,
    "runner": make_runner(MUSE_SPARK, "gpqa"),
}


def build_model_config():
    return make_model_config(
        MUSE_SPARK,
        test_name="muse-spark-gpqa",
        series_id="accuracy-muse-spark-gpqa",
        gate=GATE,
    )


class TestMuseSparkGPQA(AccuracyTestCase):
    """Score this benchmark on the backbone recipe's server."""

    config_module = CONFIG_MODULE
    model_path = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
    tag = "muse-spark-gpqa-e2e"
    health_timeout = 5400

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
