"""GPT-OSS 120B TPU full-dataset GSM8K nightly score."""

import unittest
from typing import ClassVar

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    BASE_URL,
    GPT_OSS_120B,
    HEALTH_TIMEOUT,
    fixture_runtime_envs,
    make_model_config,
    run_model_config,
)
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=2700,
    suite="nightly-tpu-accuracy-gpt-oss-120b-gsm8k",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

GATE = {
    "dataset": "gsm8k",
    "label": "gsm8k_full",
    "report_only": True,
    "num_examples": 1319,
    "num_threads": 32,
    "max_tokens": 4096,
    "temperature": 0.0,
    "reasoning_effort": "low",
}

MODEL_CONFIG = make_model_config(
    GPT_OSS_120B,
    test_name="gpt-oss-120b-gsm8k",
    series_id="accuracy-gpt-oss-120b-gsm8k",
    gate=GATE,
)


class TestGptOss120bGSM8K(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_120b/accuracy_gsm8k.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gpt-oss-120b-gsm8k-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = fixture_runtime_envs()

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
