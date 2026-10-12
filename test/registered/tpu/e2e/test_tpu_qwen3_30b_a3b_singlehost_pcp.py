"""Qwen3-30B-A3B-FP8 single-host prefill context parallelism PR gate.

PCP8 with MoE DP8 on the batched RPA backend, radix cache on. A greedy needle
probe checks that prompts landing in the smallest and in a larger local token
bucket, alone and in shared context-parallel batches, keep every code word. The
shared batches also carry prompts shorter than the zigzag split and the two
prompts sent first, which come back with a cached prefix. Few-shot GSM8K, whose
prompts share their examples, checks answers end to end.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=1200,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-30b-a3b", "pcp"],
)

SERVED_MODEL_NAME = "qwen3-30b-a3b"
MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3-30B-A3B-FP8")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
# PCP8 + MoE DP8 scored 0.91 on these 200 questions.
GSM8K_ACCURACY = 0.86

FILLER = "The quick brown fox jumps over the lazy dog. "
QUESTION = "Question: what is the secret code word? Answer: The secret code word is"
# Prompt k repeats the filler 40 + 5k times on each side of its code word:
# 0.8K to 2.3K tokens, one 512-row local bucket each at attention CP 8.
CODE_WORDS = (
    "MARIGOLD",
    "TANGERINE",
    "LAVENDER",
    "OBSIDIAN",
    "PELICAN",
    "SAPPHIRE",
    "WALNUT",
    "CINNAMON",
    "GLACIER",
    "HARBOR",
    "JUNIPER",
    "LANTERN",
    "MEADOW",
    "NUTMEG",
    "ORCHID",
    "QUARTZ",
)
# About 5K tokens: alone in a batch it takes the 1024-row local bucket.
LONG_NEEDLE = ("VERMILION", 250)
# Each has fewer tokens than the 16 zigzag blocks of attention CP 8.
SHORT_PROMPTS = (
    ("Paris", "The capital of France is"),
    ("Tokyo", "The capital of Japan is"),
    ("Rome", "The capital of Italy is"),
    ("Berlin", "The capital of Germany is"),
)


def _recall(base_url: str, word: str, filler_repeats: int) -> dict:
    side = FILLER * filler_repeats
    text = f"{side}The secret code word is {word}. Remember it. {side}{QUESTION}"
    return _generate(base_url, word, text)


def _generate(base_url: str, word: str, text: str) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 12},
        },
        timeout=600,
    )
    response.raise_for_status()
    out = response.json()
    return {
        "word": word,
        "prompt_tokens": out["meta_info"]["prompt_tokens"],
        "text": out["text"],
    }


def needle_runner(base_url: str) -> dict:
    cases = [(word, 40 + 5 * k) for k, word in enumerate(CODE_WORDS)]
    results = [_recall(base_url, *cases[0]), _recall(base_url, *cases[-1])]
    with ThreadPoolExecutor(len(cases) + len(SHORT_PROMPTS)) as pool:
        # Each short prompt follows long ones, so it joins their batch instead of
        # running alone on an idle server.
        futures = []
        for k, case in enumerate(cases):
            futures.append(pool.submit(_recall, base_url, *case))
            if k % 4 == 3:
                short = SHORT_PROMPTS[k // 4]
                futures.append(pool.submit(_generate, base_url, *short))
        results += [future.result() for future in futures]
    results.append(_recall(base_url, *LONG_NEEDLE))
    failures = []
    for result in results:
        recalled = result["word"] in result["text"]
        print(f"needle {'ok' if recalled else 'MISSED'}: {result}")
        if not recalled:
            failures.append(
                f"{result['word']} ({result['prompt_tokens']} tokens) not recalled: "
                f"{result['text']!r}"
            )
    return {
        "score": 1.0 - len(failures) / len(results),
        "_passed": not failures,
        "_failures": failures,
    }


LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    # CP batches bucket their local rows (1/8 of the tokens) here too: 512 and
    # 1024 hold the needles, 2048 a full chunk; 8192 covers non-CP batches.
    "TPU_COMPILE_TOKENS": "512 1024 2048 8192",
    "TPU_COMPILE_BS": "8 16 32",
    "MAX_RUNNING": "32",
    "TORCH_COMPILE_MAX_BS": "32",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "DISABLE_RADIX_CACHE": "0",
    "EXTRA_ARGS": f"--served-model-name {SERVED_MODEL_NAME}",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-30b-a3b-pcp",
    "variant": "PCP8+MoE-DP8",
    "launch": {
        "recipe": "recipe/qwen3_30b_a3b_fp8_pcp.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-30b-a3b-pcp-e2e",
    },
    "gates": [
        {
            "dataset": "needle",
            "label": "needle-recall",
            "runner": needle_runner,
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 32,
            "max_tokens": 512,
            "temperature": 0.0,
            # Few-shot completion prompts: the checkpoint's chat template would
            # spend the token budget on thinking.
            "api": "completion",
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
