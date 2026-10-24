"""GLM-5.3-Flash with the KDA radix prefix cache, one tpu7x host.

The KDA layers write extra_buffer snapshots at prefill and decode track
boundaries; the MLA/DSA pages ride the full-attention radix tree. The gate checks
that a second turn reuses state snapshotted during the first turn's decode and
still recalls a fact from its start, that a batch with shared prefixes reuses
them, and GSM8K.
"""

import os
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=5400,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "glm-5.3-flash", "prefix-cache"],
)

SERVED_MODEL_NAME = "zai-org/GLM-5.3-Flash"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
CONCURRENCY = 8
GENERATE_TIMEOUT = 900
# Long enough that the first turn's decode crosses several track boundaries.
TURN1_NEW_TOKENS = 640

# Measured 0.990-0.995 with the radix cache off (200 problems, greedy, thinking).
GSM8K_BASELINE = float(os.environ.get("GSM8K_THRESHOLD_GLM_5_3_FLASH", "0.93"))


def _post(path: str, payload: dict) -> dict:
    response = requests.post(
        f"{BASE_URL}{path}", json=payload, timeout=GENERATE_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _tokenize(text: str) -> list[int]:
    reply = _post(
        "/tokenize",
        {"model": SERVED_MODEL_NAME, "prompt": text, "add_special_tokens": False},
    )
    return list(reply["tokens"])


def _generate(input_ids: list[int], max_new_tokens: int, *, ignore_eos: bool) -> dict:
    return _post(
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": ignore_eos,
            },
        },
    )


def _shared_prefix_request(index: int) -> dict:
    preamble = "You are a careful counting assistant. " * 60
    reply = _post(
        "/generate",
        {
            "text": f"{preamble}Request {index}: count upward from {index * 7} in words.",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 32,
                "ignore_eos": True,
            },
        },
    )
    return reply["meta_info"]


class TestGlm53FlashRadix(TPULauncherTestCase):
    config_module = "recipe/configs/glm_5_3_flash/prefix_cache.py"
    base_url = BASE_URL
    tag = "glm-5-3-flash-radix-e2e"
    log_path = str(Path(tempfile.gettempdir()) / "glm_5_3_flash_radix.log")

    def test_second_turn_reuses_decode_snapshots(self):
        secret = 50021
        filler = "".join(
            f"Ledger line {j} books transfer number {1000 + j}. " for j in range(120)
        )
        turn1_ids = _tokenize(
            f"The secret number is {secret}. Remember it. {filler}"
            "Now tell a long story about a harbor."
        )
        turn1 = _generate(turn1_ids, TURN1_NEW_TOKENS, ignore_eos=True)
        question_ids = _tokenize(
            "\n\nQuestion: what was the secret number stated at the beginning? "
            "Answer with just the number.\nAnswer:"
        )
        turn2_ids = turn1_ids + list(turn1["output_ids"]) + question_ids
        turn2 = _generate(turn2_ids, 16, ignore_eos=False)

        # Reuse past the first prompt needs a snapshot taken during decode.
        cached = turn2["meta_info"]["cached_tokens"]
        self.assertGreater(
            cached,
            len(turn1_ids),
            f"turn 2 reused {cached} of {len(turn1_ids)} prompt tokens",
        )
        self.assertIn(str(secret), re.findall(r"\d{5}", turn2["text"]), turn2["text"])

    def test_concurrent_shared_prefix_is_reused(self):
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            metas = list(pool.map(_shared_prefix_request, range(2 * CONCURRENCY)))
        self.assertGreater(
            sum(meta["cached_tokens"] for meta in metas), 0, "no prefix was reused"
        )

    def test_gsm8k(self):
        run_model_config(
            {
                "model": SERVED_MODEL_NAME,
                "test_name": "glm-5.3-flash-radix",
                "variant": "TP8-DSA+radix",
                "gates": [
                    {
                        "dataset": "gsm8k",
                        "label": "gsm8k",
                        "baseline": GSM8K_BASELINE,
                        "num_examples": 200,
                        "num_threads": CONCURRENCY,
                        "max_tokens": 8192,
                        "temperature": 0.0,
                        "api": "chat",
                    }
                ],
            },
            base_url=self.base_url,
        )


if __name__ == "__main__":
    unittest.main()
