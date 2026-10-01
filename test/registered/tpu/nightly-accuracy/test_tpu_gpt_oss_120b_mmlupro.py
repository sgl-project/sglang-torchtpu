"""GPT-OSS 120B TPU MMLU-Pro nightly score."""

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
    est_time=4500,
    suite="nightly-tpu-accuracy-gpt-oss-120b-mmlupro",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

GATE = {
    "dataset": "mmlu_pro",
    "label": "mmlu_pro_high",
    "lm_eval_config": "test/lm_eval_configs/GPT-OSS-120B-TPU-MMLU-Pro.yaml",
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    GPT_OSS_120B,
    test_name="gpt-oss-120b-mmlupro",
    series_id="accuracy-gpt-oss-120b-mmlupro",
    gate=GATE,
)


class TestGptOss120bMMLUPro(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_120b/accuracy_mmlupro.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gpt-oss-120b-mmlupro-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = fixture_runtime_envs()

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
