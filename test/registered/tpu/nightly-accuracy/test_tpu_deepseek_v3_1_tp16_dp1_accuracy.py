"""DeepSeek-V3.1 TP16/EP16 FP8-KV evaluation, no DP attention.

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
    est_time=7200,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "deepseek-v3-1", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/deepseek_v3_1/accuracy.py",
    "weights_pvc": "deepseek-v3-1-rox",
    # Which accuracy-test-tpu.yml lane runs this file.
    "schedule": "nightly",
    "timeout_per_file": 10800,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
CONFIG_MODULE = "recipe/configs/deepseek_v3_1/accuracy.py"


def build_model_config():
    """Assemble this file's gates against the model's identity."""
    return {
        "id": "accuracy-deepseek-v3-1",
        "display_name": "DeepSeek V3.1",
        "dashboard": "accuracy",
        "dashboard_model": "deepseek-v3-1",
        "model": os.environ.get("MODEL", "/weights/DeepSeek-V3.1"),
        "test_name": "DeepSeek-V3.1-multihost-fp8kv",
        "variant": "2host-TP16+EP16+DP1-FP8KV",
        "gates": [
            {
                "dataset": "gpqa",
                "label": "gpqa_diamond",
                "baseline": 0.801,  # model card
                "atol": 0.1,
                "thinking": True,
                "repeat": 1,
                "num_threads": 128,
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 32768,
            },
            {
                "dataset": "aime25",
                "label": "aime25",
                "baseline": 0.884,  # model card
                "atol": 0.12,
                "thinking": True,
                # Mean pass@1 over four passes: one pass of 30 problems swings by 2-3 problems.
                "repeat": 4,
                "num_threads": 128,
                "temperature": 0.6,
                "top_p": 0.95,
                # 32k truncated 3-7 of 30 answers per run (each scored 0); 64k truncates none.
                "max_tokens": 65536,
            },
            {
                "dataset": "mmlu",
                "label": "mmlu",
                "baseline": 0.918,  # measured on TPU
                "atol": 0.1,
                "thinking": True,
                "repeat": 1,
                # Match the non-DPA test: 250 sampled questions per repeat.
                "num_examples": 250,
                "num_threads": 128,
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 16384,
            },
        ],
    }


@live_server_only
class TestDeepSeekV31Accuracy(AccuracyTestCase):
    """Score this file's gates on the server the JobSet started."""

    config_module = CONFIG_MODULE
    tag = "v3-1-fp8kv-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestDeepSeekV31AccuracyConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_the_jobset_launches_this_files_recipe(self):
        self.assertEqual(MULTIHOST_LAUNCH["launch_script"], CONFIG_MODULE)


if __name__ == "__main__":
    unittest.main()
