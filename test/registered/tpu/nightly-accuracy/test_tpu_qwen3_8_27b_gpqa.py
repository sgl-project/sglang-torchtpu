"""Qwen3.8-27B TPU GPQA Diamond nightly score, thinking on, mean of four passes."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import QWEN3_8_27B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=5700,
    suite="nightly-tpu-accuracy-qwen3-8-27b-gpqa",
    nightly=True,
    labels=["singlehost", "qwen3.8-27b", "accuracy"],
)

CONFIG_MODULE = QWEN3_8_27B["config_module"]
# The /weights mount when it carries the checkpoint, else the HF id.
MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3.8-27B")

GATE = {
    "dataset": "gpqa",
    "label": "gpqa_diamond",
    "baseline": 0.893,  # measured on TPU
    "atol": 0.1,
    "runner": make_runner(QWEN3_8_27B, "gpqa", n_repeats=4),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        QWEN3_8_27B,
        test_name="qwen3-8-27b-gpqa",
        # The dashboard's series join key: never renamed, never shown.
        series_id="accuracy-qwen3-8-27b-gpqa",
        gate=GATE,
    )


class TestQwen38GPQA(AccuracyTestCase):
    """Score Qwen3.8-27B gpqa_diamond on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = MODEL_PATH
    tag = "qwen3-8-27b-gpqa-e2e"
    health_timeout = 3600

    @classmethod
    def get_runtime_envs(cls):
        # The checkpoint may still come from HF, so pass the resolved path.
        return {**super().get_runtime_envs(), "MODEL": MODEL_PATH}

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestQwen38GPQAConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, QWEN3_8_27B["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
