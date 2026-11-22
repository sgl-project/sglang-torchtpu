"""GPT-OSS-20B single-host PP8 nightly test with the buffered pipeline hand-off.

Eight single-device stages with uneven free memory (embedding on the first,
logits on the last) and a profiled KV pool. Every stage must agree on one token
count while the stages with more free memory size their budget on it, the first
stage must reserve less for the hand-off than a receiving stage, concurrent
multi-chunk prefills must fill the send queue and all complete, a code word in
the middle of a long prompt must be recalled, and few-shot GSM8K checks answers.
"""

import os
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
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
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "gpt-oss-20b", "pp"],
)

MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
SERVER_LOG = os.path.join(tempfile.mkdtemp(prefix="pp8-buffered-"), "server.log")
PP_SIZE = 8
# gpt-oss-20b scores 0.52-0.59 on these 200 questions under TP8, PP8 and PP4 x TP2;
# broken output scores near zero.
GSM8K_ACCURACY = 0.45

FILLER = "The quick brown fox jumps over the lazy dog. "
QUESTION = "Question: what is the secret code word? Answer: The secret code word is"
# About 12K tokens: more than two 4096-token prefill chunks.
NEEDLE = ("TANGERINE", 600)
PREFILL_TOKENS = 4096
# Concurrent prefills of about 6K tokens (two chunks each; FILLER is about ten
# tokens): enough forwards in flight to fill every stage's send queue.
LOAD_REQUESTS, LOAD_REPEATS = 32, 600


def _stage_values(pattern: str) -> dict[int, list[float]]:
    values: dict[int, list[float]] = {}
    for line in Path(SERVER_LOG).read_text(errors="replace").splitlines():
        match = re.search(r"PP(\d+)\b.*?" + pattern, line)
        if match:
            values.setdefault(int(match.group(1)), []).append(float(match.group(2)))
    return values


def kv_budget_runner(base_url: str) -> dict:
    del base_url
    tokens = _stage_values(r"max_total_num_tokens=(\d+)")
    headroom = _stage_values(
        r"Sizing the KV budget on this pipeline stage's free memory: [\d.]+ GB, ([\d.]+) GB above"
    )
    reserve = _stage_values(
        r"Reserving ([\d.]+) GB of the KV budget for the buffered PP hand-off"
    )
    failures = []
    # Core takes the minimum over stages after the hook; stages disagree only if
    # the stage budget moves past that minimum.
    if sorted(tokens) != list(range(PP_SIZE)):
        failures.append(f"max_total_num_tokens logged by stages {sorted(tokens)}")
    elif len({values[-1] for values in tokens.values()}) != 1:
        failures.append(f"stages disagree on the token count: {tokens}")
    if not headroom:
        failures.append("no stage sized its KV budget on its own free memory")
    if sorted(reserve) != list(range(PP_SIZE)):
        failures.append(f"hand-off reserve logged by stages {sorted(reserve)}")
    elif not reserve[0][-1] < reserve[1][-1]:
        failures.append(f"first stage reserve not below a receiving stage's: {reserve}")
    print(f"kv budget: tokens {tokens}, headroom {headroom}, reserve {reserve}")
    return {
        "score": float(not failures),
        "_passed": not failures,
        "_failures": failures,
    }


def _generate(base_url: str, text: str, max_new_tokens: int) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


def load_runner(base_url: str) -> dict:
    prompts = [f"Request {i}. " + FILLER * LOAD_REPEATS for i in range(LOAD_REQUESTS)]
    with ThreadPoolExecutor(LOAD_REQUESTS) as pool:
        futures = [pool.submit(_generate, base_url, p, 8) for p in prompts]
    failures = []
    for i, future in enumerate(futures):
        try:
            out = future.result()
            if out["meta_info"]["prompt_tokens"] <= PREFILL_TOKENS:
                failures.append(
                    f"request {i} fits one prefill chunk: "
                    f"{out['meta_info']['prompt_tokens']} tokens"
                )
            if not out["meta_info"]["completion_tokens"]:
                failures.append(f"request {i} returned no tokens")
        except Exception as exc:  # report every request, not the first error
            failures.append(f"request {i}: {exc}")
    return {
        "score": 1.0 - len(failures) / LOAD_REQUESTS,
        "_passed": not failures,
        "_failures": failures,
    }


def needle_runner(base_url: str) -> dict:
    word, repeats = NEEDLE
    side = FILLER * repeats
    # gpt-oss-20b restates the question before it answers.
    out = _generate(
        base_url,
        f"{side}The secret code word is {word}. Remember it. {side}{QUESTION}",
        128,
    )
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
    "test_name": "gpt-oss-20b-pp8-buffered",
    "variant": "PP8-buffered",
    "gates": [
        {
            "dataset": "kv_budget",
            "label": "stage-kv-budget",
            "runner": kv_budget_runner,
        },
        {"dataset": "load", "label": "concurrent-prefill", "runner": load_runner},
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


class TestGptOss20bSingleHostPp8Buffered(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_20b/pp8_buffered.py"
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    log_path = SERVER_LOG
    tag = "gpt-oss-20b-pp8-buffered-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        "MODEL": MODEL_PATH,
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
    }

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
