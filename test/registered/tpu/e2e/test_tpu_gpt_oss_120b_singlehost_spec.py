"""GPT-OSS-120B EAGLE3 single-host GSM8K quality gate.

The quality lane launches the same speculative-decoding configuration as the
functional integration test, then scores 200 GSM8K examples through run_eval.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "spec"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70  # measured 0.7850 over 200 examples
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90")
)


MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-spec-decoding-gsm8k",
    "variant": "TP8+EP8+EAGLE3",
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


class TestGptOssSingleHostSpec(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_120b/singlehost_spec.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gpt-oss-spec-gsm8k-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        "MODEL": None,
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
    }

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
