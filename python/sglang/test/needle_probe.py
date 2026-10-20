"""Needle-recall probe for prefill context parallelism.

A greedy probe that checks prompts landing in the smallest and in a larger local
token bucket, alone and in shared context-parallel batches, keep every code
word. The shared batches also carry prompts shorter than the zigzag split and
the two prompts sent first, which come back with a cached prefix.

Shared so the gate and the nightly run the identical probe against different
checkpoints: the geometry it exercises is a function of the CP degree and the
compile buckets, not of the model.
"""

from concurrent.futures import ThreadPoolExecutor

import requests

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


def _recall(
    base_url: str,
    word: str,
    filler_repeats: int,
    max_new_tokens: int,
    prompt_template: str,
) -> dict:
    side = FILLER * filler_repeats
    text = f"{side}The secret code word is {word}. Remember it. {side}{QUESTION}"
    return _generate(
        base_url, word, prompt_template.format(prompt=text), max_new_tokens
    )


def _generate(base_url: str, word: str, text: str, max_new_tokens: int) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
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


def needle_runner(
    base_url: str, max_new_tokens: int = 12, *, prompt_template: str = "{prompt}"
) -> dict:
    """Format long recall requests with ``{prompt}``; keep short completion probes."""
    cases = [(word, 40 + 5 * k) for k, word in enumerate(CODE_WORDS)]
    results = [
        _recall(base_url, *cases[0], max_new_tokens, prompt_template),
        _recall(base_url, *cases[-1], max_new_tokens, prompt_template),
    ]
    with ThreadPoolExecutor(len(cases) + len(SHORT_PROMPTS)) as pool:
        # Each short prompt follows long ones, so it joins their batch instead of
        # running alone on an idle server.
        futures = []
        for k, case in enumerate(cases):
            futures.append(
                pool.submit(_recall, base_url, *case, max_new_tokens, prompt_template)
            )
            if k % 4 == 3:
                short = SHORT_PROMPTS[k // 4]
                futures.append(pool.submit(_generate, base_url, *short, max_new_tokens))
        results += [future.result() for future in futures]
    results.append(_recall(base_url, *LONG_NEEDLE, max_new_tokens, prompt_template))
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
