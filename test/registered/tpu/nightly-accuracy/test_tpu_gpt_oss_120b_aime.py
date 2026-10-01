"""GPT-OSS 120B TPU AIME 2025 nightly score."""

import unittest
from typing import ClassVar

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    BASE_URL,
    GPT_OSS_120B,
    HEALTH_TIMEOUT,
    aime25_openai_runner,
    fixture_runtime_envs,
    make_model_config,
    run_model_config,
)
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=4500,
    suite="nightly-tpu-accuracy-gpt-oss-120b-aime",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

MAX_TOKENS = 65536

GATE = {
    "dataset": "aime25",
    "label": "aime25_high",
    "runner": aime25_openai_runner(
        model=GPT_OSS_120B["served_model_name"],
        n_repeats=3,
        reasoning_effort="high",
        max_tokens=MAX_TOKENS,
        n_threads=16,
    ),
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    GPT_OSS_120B,
    test_name="gpt-oss-120b-aime",
    series_id="accuracy-gpt-oss-120b-aime",
    gate=GATE,
)


class TestGptOss120bAIME(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_120b/accuracy_aime.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gpt-oss-120b-aime-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = fixture_runtime_envs()

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
