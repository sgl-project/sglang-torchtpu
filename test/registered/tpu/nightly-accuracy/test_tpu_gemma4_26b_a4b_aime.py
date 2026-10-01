"""Gemma-4-26B-A4B TPU AIME 2026 nightly score, mean of four passes."""

import unittest
from typing import ClassVar

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    BASE_URL,
    GEMMA4_26B_A4B,
    HEALTH_TIMEOUT,
    fixture_runtime_envs,
    make_model_config,
    make_runner,
    run_model_config,
)
from sglang.test.tpu.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=4500,
    suite="nightly-tpu-accuracy-gemma4-26b-a4b-aime",
    nightly=True,
    labels=["singlehost", "gemma4-26b-a4b", "accuracy"],
)

GATE = {
    "dataset": "aime26",
    "label": "aime26",
    "runner": make_runner(GEMMA4_26B_A4B, "aime26", n_repeats=4),
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    GEMMA4_26B_A4B,
    test_name="gemma4-26b-a4b-aime",
    series_id="accuracy-gemma4-26b-a4b-aime",
    gate=GATE,
)


class TestGemma26BA4BAIME(TPULauncherTestCase):
    config_module = GEMMA4_26B_A4B["config_module"]
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gemma4-26b-a4b-aime-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = fixture_runtime_envs()

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
