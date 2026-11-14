"""DeepSeek-V4.1-Flash single-host serving gate on TPU v7x-8 (attention DP8, EP8).

Boots recipe/configs/deepseek_v41_flash.py, then runs one of:
  * TestDeepSeekV41Flash.test_smoke   greedy chat, "17*23" must answer 391
  * TestDeepSeekV41Flash.test_gsm8k   GSM8K against a lenient floor

Dispatch-only: .github/workflows/dsv41-flash-test-tpu.yml runs it at any ref.
It is registered disabled so nightly and PR discovery never schedule it.
To run against an already-running server, set SGLANG_TPU_USE_LIVE_SERVER=1 and BASE_URL.
"""

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=5400,
    suite="nightly-tpu-4-v7",
    nightly=True,
    disabled="manual: dispatched by dsv41-flash-test-tpu.yml",
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy"],
    weights_pvc="model-weights",
)

SERVED_MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_PATH = os.environ.get("MODEL", "/weights/DeepSeek-V4.1-Flash")
CONFIG_MODULE = "recipe/configs/deepseek_v41_flash.py"

GSM8K_GATE = {
    "dataset": "gsm8k",
    "label": "gsm8k",
    # #1743 measured 1281/1319 = 0.971 on TPU v7x-8 with the same layout.
    "baseline": 0.95,
    "atol": 0.05,
    "num_examples": int(os.environ.get("GSM8K_NUM_EXAMPLES", "1319")),
    "num_threads": 8,
    "max_tokens": 1024,
    "temperature": 0.0,
}


class TestDeepSeekV41Flash(AccuracyTestCase):
    """Serve DeepSeek-V4.1-Flash with its default launch, then score it."""

    config_module = CONFIG_MODULE
    model_path = MODEL_PATH
    tag = "deepseek-v41-flash-e2e"

    @classmethod
    def get_runtime_envs(cls):
        return {**super().get_runtime_envs(), "MODEL": MODEL_PATH}

    def test_smoke(self):
        resp = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": SERVED_MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 17*23? Answer with the number only.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 64,
                "chat_template_kwargs": {"thinking": False},
            },
            timeout=600,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"] or ""
        print(f"[smoke] 17*23 -> {text!r}", flush=True)
        self.assertIn("391", text)

    def test_gsm8k(self):
        from sglang.test.tpu import run_model_config

        run_model_config(
            {
                "model": SERVED_MODEL_NAME,
                "test_name": "deepseek-v41-flash-gsm8k",
                "variant": "DP8+EP8",
                "gates": [GSM8K_GATE],
            },
            base_url=self.base_url,
        )


class TestDeepSeekV41FlashConfiguration(unittest.TestCase):
    """Host-side checks, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(gate_server_failures(CONFIG_MODULE, GSM8K_GATE), [])


if __name__ == "__main__":
    unittest.main()
