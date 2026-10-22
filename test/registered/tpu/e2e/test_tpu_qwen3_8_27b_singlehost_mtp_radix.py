"""Qwen3.8-27B MTP with the GDN radix prefix cache on the experimental spec worker.

Speculative verifies keep their checkpoints in per-request windows; radix
snapshots are taken at extend and after each verify that crosses a track
boundary. The gate checks that a second turn reuses state snapshotted during
the first turn's decode and still recalls a fact from its start, that a draining
batch with shared prefixes compiles nothing while serving, and GSM8K.
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
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=1800,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3.8-27b", "spec", "compile", "prefix-cache"],
)

MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3.8-27B")
SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
CONCURRENCY = 16
SERVING_EVENT = "at serving time"
GENERATE_TIMEOUT = 900
# Long enough that the first turn's decode crosses several track boundaries.
TURN1_NEW_TOKENS = 384

# Measured 0.975 with radix on and off (chat, 2048 max tokens, 200 examples);
# Qwen3.8 always thinks first, and a 512-token cap cut most answers off.
GSM8K_BASELINE = float(os.environ.get("GSM8K_THRESHOLD_QWEN3_8_27B_MTP", "0.90"))
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.5")
)


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


def _drain_request(index: int) -> dict:
    # A shared preamble lets later requests hit the radix cache mid-drain.
    preamble = "You are a careful counting assistant. " * 40
    reply = _post(
        "/generate",
        {
            "text": f"{preamble}Request {index}: count upward from {index * 7} in words.",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 16 + 24 * index,
                "ignore_eos": True,
            },
        },
    )
    return reply["meta_info"]


class TestQwen38MtpRadix(TPULauncherTestCase):
    config_module = "recipe/configs/qwen3_8_27b/mtp_radix_gate.py"
    base_url = BASE_URL
    tag = "qwen3-8-27b-mtp-radix-e2e"
    log_path = str(Path(tempfile.gettempdir()) / "qwen3_8_27b_mtp_radix.log")

    @classmethod
    def get_runtime_envs(cls):
        # CI-only FutureMap debug asserts scatter into live rows eagerly, one
        # program per live bs; this server measures production serving.
        return {
            **super().get_runtime_envs(),
            "MODEL": MODEL_PATH,
            "SGLANG_IS_IN_CI": None,
        }

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

        meta = turn2["meta_info"]
        self.assertGreater(meta.get("spec_verify_ct", 0), 0, "spec decode never ran")
        # Reuse past the first prompt needs a snapshot taken after a verify.
        self.assertGreater(
            meta["cached_tokens"],
            len(turn1_ids),
            f"turn 2 reused {meta['cached_tokens']} of {len(turn1_ids)} prompt tokens",
        )
        self.assertIn(str(secret), re.findall(r"\d{5}", turn2["text"]), turn2["text"])

    def test_draining_batch_compiles_nothing(self):
        log = Path(self.log_path)
        served_from = log.stat().st_size
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            metas = list(pool.map(_drain_request, range(CONCURRENCY)))

        verify_count = sum(meta.get("spec_verify_ct", 0) for meta in metas)
        completion_tokens = sum(meta["completion_tokens"] for meta in metas)
        self.assertGreater(verify_count, 0, "speculative verification never ran")
        self.assertGreater(completion_tokens, verify_count, "no draft was accepted")
        self.assertGreater(
            sum(meta["cached_tokens"] for meta in metas), 0, "no prefix was reused"
        )

        with log.open(errors="replace") as f:
            f.seek(served_from)
            events = [line.strip() for line in f if SERVING_EVENT in line]
        self.assertEqual(events, [], "XLA programs compiled or loaded while serving")

    def test_gsm8k(self):
        run_model_config(
            {
                "model": SERVED_MODEL_NAME,
                "test_name": "qwen3.8-27b-mtp-radix",
                "variant": "TP8+MTP+radix",
                "gates": [
                    {
                        "dataset": "gsm8k",
                        "label": "gsm8k",
                        "baseline": GSM8K_BASELINE,
                        "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
                        "num_examples": 200,
                        "num_threads": CONCURRENCY,
                        "max_tokens": 2048,
                        "temperature": 0.0,
                        "api": "chat",
                    }
                ],
            },
            base_url=self.base_url,
        )


if __name__ == "__main__":
    unittest.main()
