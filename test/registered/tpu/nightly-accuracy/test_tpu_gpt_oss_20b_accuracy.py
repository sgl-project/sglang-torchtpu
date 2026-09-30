"""GPT-OSS-20B: four concurrent AIME25 repeats; GPQA and MMLU-250 once.

Serving config lives in recipe/configs/gpt_oss_20b/accuracy.py, which CI runs verbatim.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.launcher_test_case import TPULauncherTestCase
from sglang.test.tpu.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-accuracy-gpt-oss-20b",
    nightly=True,
    labels=["singlehost", "gpt-oss-20b", "accuracy"],
)

SERVED_MODEL_NAME = "gpt-oss-20b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3000"))
SAMPLING = {
    "reasoning_effort": "high",
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 131072,
    "num_threads": 128,
    "report_only": True,
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-accuracy",
    "id": "accuracy-gpt-oss-20b",
    "display_name": "GPT-OSS 20B",
    "variant": "TP8+EP8",
    "dashboard": "accuracy",
    "dashboard_model": SERVED_MODEL_NAME,
    "gates": [
        {"dataset": "gpqa", "label": "gpqa_diamond_high", "repeat": 1, **SAMPLING},
        # sgl-eval submits all 30 x 4 samples to one shared worker pool.
        {"dataset": "aime25", "label": "aime25_high", "repeat": 4, **SAMPLING},
        {
            "dataset": "mmlu",
            "label": "mmlu_high",
            "repeat": 1,
            "num_examples": 250,
            **SAMPLING,
        },
    ],
}


class TestGptOss20bAccuracy(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_20b/accuracy.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gpt-oss-20b-accuracy"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        # The /weights mount when it carries the checkpoint, else the HF id.
        "MODEL": resolve_tpu_model_path("openai/gpt-oss-20b"),
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
    }

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
