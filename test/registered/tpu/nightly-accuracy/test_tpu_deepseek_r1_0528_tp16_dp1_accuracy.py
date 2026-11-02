"""Original DeepSeek-R1-0528 TP16/EP16 FP8-KV evaluation, no DP attention.

One pass each of GPQA, AIME25 and 250 MMLU questions at concurrency 128.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu.utils.accuracy_base import (
    AccuracyTestCase,
    live_server_only,
)

register_tpu_ci(
    est_time=4130,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "deepseek-r1", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/deepseek_r1_0528/accuracy.py",
    "weights_pvc": "model-weights",
    # Which accuracy-test-tpu.yml lane runs this file.
    "schedule": "nightly",
    "timeout_per_file": 4130,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
CONFIG_MODULE = "recipe/configs/deepseek_r1_0528/accuracy.py"


def build_model_config():
    """Assemble this file's gates against the model's identity."""
    return {
        "id": "accuracy-deepseek-r1-0528",
        "display_name": "DeepSeek R1-0528",
        "dashboard": "accuracy",
        "dashboard_model": "deepseek-r1-0528",
        "model": os.environ.get("MODEL", "/weights/DeepSeek-R1-0528"),
        "test_name": "DeepSeek-R1-0528-multihost-fp8kv",
        "variant": "2host-TP16+EP16+DP1-FP8KV",
        "gates": [
            {
                "dataset": "gpqa",
                "label": "gpqa_diamond",
                "baseline": 0.810,  # model card
                "atol": 0.10,
                "repeat": 1,
                "num_threads": 128,
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 32768,
            },
            {
                "dataset": "aime25",
                "label": "aime25",
                "baseline": 0.875,  # model card
                # 30 questions x 1 pass: 0.12 would false-alarm ~1 night in 8.
                "atol": 0.20,
                "repeat": 1,
                "num_threads": 128,
                "temperature": 0.6,
                "top_p": 0.95,
                # 32k truncated 3-7 of 30 answers per run (each scored 0); 64k truncates none.
                "max_tokens": 65536,
            },
            {
                "dataset": "mmlu",
                "label": "mmlu",
                "baseline": 0.91,  # measured on TPU
                "atol": 0.1,
                "repeat": 1,
                # The same seeded 250 rows every model and every night: sgl-eval's
                # mmlu pins the archive revision and samples with sample_seed=0.
                "num_examples": 250,
                "num_threads": 128,
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 16384,
            },
        ],
    }


@live_server_only
class TestDeepSeekR1Accuracy(AccuracyTestCase):
    """Score this file's gates on the server the JobSet started."""

    config_module = CONFIG_MODULE
    tag = "r1-fp8kv-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestDeepSeekR1AccuracyConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_the_jobset_launches_this_files_recipe(self):
        self.assertEqual(MULTIHOST_LAUNCH["launch_script"], CONFIG_MODULE)


if __name__ == "__main__":
    unittest.main()
