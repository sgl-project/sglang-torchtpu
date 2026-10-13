"""GPT-OSS-20B single-host prefill context parallelism PR gate.

PCP8 with MoE DP8 on the batched RPA backend, radix cache on. A greedy needle
probe checks that prompts landing in the smallest and in a larger local token
bucket, alone and in shared context-parallel batches, keep every code word. The
shared batches also carry prompts shorter than the zigzag split and the two
prompts sent first, which come back with a cached prefix. Few-shot GSM8K, whose
prompts share their examples, checks answers end to end.
"""

import os
import tempfile
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.needle_probe import needle_runner
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

# PCP is an attention feature; the checkpoint only has to be good enough for the
# probe's assertions. test_tpu_qwen3_30b_a3b_singlehost_pcp.py keeps the 30B
# FP8 configuration on the nightly.
register_tpu_ci(
    est_time=600,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "pcp"],
)

MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
SERVER_LOG = os.path.join(tempfile.mkdtemp(prefix="gpt-oss-20b-pcp-"), "server.log")
# gpt-oss-20b scores 0.52-0.59 on these 200 questions (PP4xPCP2, TP8, PP8, and
# 0.54 here); broken output scores near zero. Same floor as pp4_pcp2.
GSM8K_ACCURACY = 0.45
# GPT-OSS reasons before answering; 48 tokens cut a found word off mid-answer.
NEEDLE_MAX_NEW_TOKENS = 128

MODEL_CONFIG = {
    "model": "gpt-oss-20b",
    "test_name": "gpt-oss-20b-pcp",
    "variant": "PCP8+MoE-DP8",
    "gates": [
        {
            "dataset": "needle",
            "label": "needle-recall",
            "runner": lambda base_url: needle_runner(
                base_url, max_new_tokens=NEEDLE_MAX_NEW_TOKENS
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 32,
            "max_tokens": 512,
            "temperature": 0.0,
            # Few-shot completion prompts: the chat template would spend the
            # token budget on reasoning.
            "api": "completion",
        },
    ],
}


class TestGptOss20bSingleHostPcp(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_20b/pcp.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    log_path = SERVER_LOG
    tag = "gpt-oss-20b-pcp-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        "MODEL": MODEL_PATH,
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
    }

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
