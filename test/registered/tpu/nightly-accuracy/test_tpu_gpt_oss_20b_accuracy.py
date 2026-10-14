"""GPT-OSS-20B: four concurrent AIME25 repeats; GPQA and MMLU-250 once.

Serving config lives in recipe/configs/gpt_oss_20b/accuracy.py, which CI runs verbatim.
"""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-accuracy-gpt-oss-20b",
    nightly=True,
    labels=["singlehost", "gpt-oss-20b", "accuracy"],
)

SERVED_MODEL_NAME = "gpt-oss-20b"
CONFIG_MODULE = "recipe/configs/gpt_oss_20b/accuracy.py"
# The /weights mount when it carries the checkpoint, else the HF id.
MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")

SAMPLING = {
    "reasoning_effort": "high",
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 131072,
    "num_threads": 128,
}

# Baselines: the gpt-oss-20b model card (reasoning high).
GATES = [
    {
        "dataset": "gpqa",
        "label": "gpqa_diamond_high",
        "baseline": 0.715,
        "atol": 0.1,
        "repeat": 1,
        **SAMPLING,
    },
    # sgl-eval submits all 30 x 4 samples to one shared worker pool.
    {
        "dataset": "aime25",
        "label": "aime25_high",
        "baseline": 0.917,
        "atol": 0.12,
        "repeat": 4,
        **SAMPLING,
    },
    {
        "dataset": "mmlu",
        "label": "mmlu_high",
        "baseline": 0.853,  # model card
        "atol": 0.1,
        "repeat": 1,
        "num_examples": 250,
        **SAMPLING,
    },
]


def build_model_config():
    """Assemble every gate against this model's shared identity."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-20b-accuracy",
        # The dashboard's series join key: never renamed, never shown.
        "id": "accuracy-gpt-oss-20b",
        "display_name": "GPT-OSS 20B",
        "variant": "TP8+EP8",
        "dashboard": "accuracy",
        "dashboard_model": SERVED_MODEL_NAME,
        "gates": GATES,
    }


class TestGptOss20bAccuracy(AccuracyTestCase):
    """Score GPQA, AIME25 and MMLU-250 on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = MODEL_PATH
    tag = "gpt-oss-20b-accuracy"

    @classmethod
    def get_runtime_envs(cls):
        # This checkpoint may still come from HF, so pass the resolved path.
        return {**super().get_runtime_envs(), "MODEL": MODEL_PATH}

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGptOss20bAccuracyConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_every_gate_fits_the_server_its_recipe_starts(self):
        for gate in GATES:
            with self.subTest(gate=gate["label"]):
                self.assertEqual(gate_server_failures(CONFIG_MODULE, gate), [])


if __name__ == "__main__":
    unittest.main()
