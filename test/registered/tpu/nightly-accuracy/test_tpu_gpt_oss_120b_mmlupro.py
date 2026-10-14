"""GPT-OSS 120B TPU MMLU-Pro nightly score."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    GPT_OSS_120B,
    make_model_config,
)
from sglang.test.tpu.utils.accuracy_base import (
    AccuracyTestCase,
    gate_server_failures,
)

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
}

CONFIG_MODULE = "recipe/configs/gpt_oss_120b/accuracy_mmlupro.py"


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        GPT_OSS_120B,
        test_name="gpt-oss-120b-mmlupro",
        series_id="accuracy-gpt-oss-120b-mmlupro",
        gate=GATE,
    )


class TestGptOss120bMMLUPro(AccuracyTestCase):
    """Score GPT-OSS-120B mmlupro on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/gpt-oss-120b"
    tag = "gpt-oss-120b-mmlupro-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGptOss120bMMLUProConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(gate_server_failures(CONFIG_MODULE, GATE), [])


if __name__ == "__main__":
    unittest.main()
