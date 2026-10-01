"""Gemma-4-31B TPU MMMU-Pro score, all 1,730 standard-10 questions."""

import unittest
from typing import ClassVar

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    BASE_URL,
    GEMMA4_31B,
    HEALTH_TIMEOUT,
    fixture_runtime_envs,
    make_model_config,
    make_runner,
    run_model_config,
)
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-accuracy-gemma4-31b-mmmupro",
    nightly=True,
    labels=["singlehost", "gemma4-31b", "accuracy"],
)

GATE = {
    "dataset": "mmmu_pro",
    # sgl-eval's own grader: its own series, apart from the lenient manual re-grades.
    "label": "mmmu_pro_strict",
    "runner": make_runner(GEMMA4_31B, "mmmu_pro"),
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    GEMMA4_31B,
    test_name="gemma4-31b-mmmupro",
    series_id="accuracy-gemma4-31b-mmmupro",
    gate=GATE,
)


class TestGemma31BMMMUPro(TPULauncherTestCase):
    config_module = GEMMA4_31B["config_module"]
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gemma4-31b-mmmupro-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = fixture_runtime_envs()

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
