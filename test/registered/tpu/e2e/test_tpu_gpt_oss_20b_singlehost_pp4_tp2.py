"""GPT-OSS-20B single-host PP4 x TP2 PR gate.

Four pipeline stages, each a two-device tensor-parallel pair. Every stage must
make its tensor-parallel group standalone at startup, a code word in the middle
of a prompt longer than two prefill chunks must be recalled, and few-shot GSM8K
checks answers end to end.
"""

import os
import re
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=1800,
    suite="stage-b-test-tpu-4-v7",
    disabled="needs a torch_tpu with standalone process subgroups (torch_tpu #4387)",
    labels=["singlehost", "gpt-oss-20b", "pp", "tp"],
)

MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
SERVER_LOG = os.path.join(tempfile.mkdtemp(prefix="pp4-tp2-"), "server.log")
PP_SIZE = 4
# gpt-oss-20b scores 0.52-0.59 on these 200 questions under TP8, PP8 and this layout;
# broken output scores near zero.
GSM8K_ACCURACY = 0.45

FILLER = "The quick brown fox jumps over the lazy dog. "
QUESTION = "Question: what is the secret code word? Answer: The secret code word is"
# About 12K tokens: more than two 4096-token prefill chunks.
NEEDLE = ("TANGERINE", 600)


def standalone_groups_runner(base_url: str) -> dict:
    del base_url
    stages = {}
    for line in Path(SERVER_LOG).read_text(errors="replace").splitlines():
        match = re.search(r"pipeline stage (\d+): standalone process groups (.*)", line)
        if match:
            stages[int(match.group(1))] = match.group(2)
    failures = [
        f"pipeline stage {stage} logged no standalone groups: {stages.get(stage)}"
        for stage in range(PP_SIZE)
        if stages.get(stage, "[]").strip() == "[]"
    ]
    return {
        "score": 1.0 - len(failures) / PP_SIZE,
        "_passed": not failures,
        "_failures": failures,
    }


def needle_runner(base_url: str) -> dict:
    word, repeats = NEEDLE
    side = FILLER * repeats
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "text": f"{side}The secret code word is {word}. Remember it. {side}{QUESTION}",
            # gpt-oss-20b restates the question before it answers.
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 128},
        },
        timeout=600,
    )
    response.raise_for_status()
    out = response.json()
    print(f"needle: {out['meta_info']['prompt_tokens']} tokens -> {out['text']!r}")
    failures = []
    if word not in out["text"]:
        failures.append(
            f"{word} ({out['meta_info']['prompt_tokens']} tokens) not recalled: "
            f"{out['text']!r}"
        )
    return {
        "score": float(not failures),
        "_passed": not failures,
        "_failures": failures,
    }


MODEL_CONFIG = {
    "model": "gpt-oss-20b",
    "test_name": "gpt-oss-20b-pp4-tp2",
    "variant": "PP4xTP2",
    "gates": [
        {
            "dataset": "standalone_groups",
            "label": "standalone-stage-groups",
            "runner": standalone_groups_runner,
        },
        {"dataset": "needle", "label": "needle-recall", "runner": needle_runner},
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


class TestGptOss20bSingleHostPp4Tp2(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_20b/pp4_tp2.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    log_path = SERVER_LOG
    tag = "gpt-oss-20b-pp4-tp2-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        "MODEL": MODEL_PATH,
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
    }

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
