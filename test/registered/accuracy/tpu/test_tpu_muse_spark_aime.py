"""Muse Spark NVFP4 TPU AIME nightly quality evaluation."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import MUSE_SPARK, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase

register_tpu_ci(
    # Leave queueing time behind the other Muse quality jobs.
    est_time=9000,
    suite="nightly-tpu-accuracy-muse-spark-aime",
    nightly=True,
    labels=["singlehost", "muse-spark-open-816b-a42b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/muse_spark/accuracy.py"

# Full TPU calibration, 2026-10-24: 90/90 requests across three repeats.
# https://github.com/sgl-project/sglang-torchtpu/actions/runs/40075252040
GATE = {
    "dataset": "aime25",
    "label": "aime25",
    "baseline": 1.00,
    "atol": 0.12,
    "runner": make_runner(MUSE_SPARK, "aime25", n_repeats=3),
}


def build_model_config():
    return make_model_config(
        MUSE_SPARK,
        test_name="muse-spark-aime",
        series_id="accuracy-muse-spark-aime",
        gate=GATE,
    )


class TestMuseSparkAIME(AccuracyTestCase):
    """Score this benchmark on the backbone recipe's server."""

    config_module = CONFIG_MODULE
    model_path = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
    tag = "muse-spark-aime-e2e"
    health_timeout = 5400

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
