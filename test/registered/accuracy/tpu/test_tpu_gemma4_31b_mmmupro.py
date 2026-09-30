"""Gemma-4-31B TPU MMMU-Pro nightly score, all 1,730 standard-10 questions."""

import unittest
from typing import ClassVar

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.gemma4_31b_eval_common import (
    BASE_URL,
    CONFIG_MODULE,
    HEALTH_TIMEOUT,
    RUNTIME_ENVS,
    make_model_config,
    make_runner,
)
from sglang.test.tpu.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-accuracy-gemma4-31b-mmmupro",
    nightly=True,
    labels=["singlehost", "gemma4-31b", "accuracy"],
)

GATE = {
    "dataset": "mmmu_pro",
    # sgl-eval's own grader; 31B answers "Answer: $X", which it misses (0.46 vs 0.76 lenient).
    "label": "mmmu_pro_strict",
    "runner": make_runner("mmmu_pro"),
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    test_name="gemma4-31b-mmmupro",
    series_id="accuracy-gemma4-31b-mmmupro",
    gate=GATE,
)


class TestGemma4MMMUPro(TPULauncherTestCase):
    config_module = CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gemma4-mmmupro-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = RUNTIME_ENVS

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
