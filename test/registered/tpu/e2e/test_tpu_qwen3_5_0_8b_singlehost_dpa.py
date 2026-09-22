"""Qwen3.5 (mamba-hybrid) under full DP attention, tp_size == dp_size, one host.

Each DP rank owns its own requests, so a rank holding decode work is relabelled
EXTEND whenever a peer prefills. On a hybrid model that step used to take the
fabricated idle-rank conversion and kill the scheduler. Served through the VL
wrapper with --language-model-only, which drops the vision tower.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(
    est_time=900,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3.5-0.8b", "dpa", "hybrid"],
)

MODEL = "Qwen/Qwen3.5-0.8B"
DP_SIZE = 8
CHUNK_TOKENS_PER_RANK = 256
RUNNING_PER_RANK = 2
# A cold DP8 compile grid does not fit the default launch timeout.
LAUNCH_TIMEOUT_S = 900
PROMPT = "The capital of France is"
ANSWER = "Paris"


class TestTpuQwen3_5SinglehostDpa(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=LAUNCH_TIMEOUT_S,
            other_args=[
                "--dtype",
                "bfloat16",
                "--device",
                "tpu",
                "--grammar-backend",
                "none",
                "--enable-torch-compile",
                "--tp-size",
                str(DP_SIZE),
                "--dp-size",
                str(DP_SIZE),
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--language-model-only",
                "--attention-backend",
                "pallas_tpu_rpa_v3",
                "--page-size",
                "256",
                "--disable-radix-cache",
                "--context-length",
                "1024",
                # Global budgets, split across DP ranks by the server.
                "--chunked-prefill-size",
                str(CHUNK_TOKENS_PER_RANK * DP_SIZE),
                "--max-running-requests",
                str(RUNNING_PER_RANK * DP_SIZE),
                # Per-rank compile grid.
                "--torch-compile-max-bs",
                str(RUNNING_PER_RANK),
                "--tpu-compile-tokens",
                *("64", "128", str(CHUNK_TOKENS_PER_RANK)),
                "--tpu-compile-bs",
                *("1", str(RUNNING_PER_RANK)),
                "--mem-fraction-static",
                "0.3",
            ],
            device="tpu",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid, wait_timeout=30)

    def _complete(self, prompt: str, max_tokens: int = 8) -> str:
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["choices"][0]["text"]

    def test_every_rank_answers_correctly(self):
        # Round-robin dispatch: 4x dp_size requests in flight cannot all land
        # on one rank, so a rank with wrong logits or a misaligned gather slot
        # shows up even when rank 0 is fine.
        with ThreadPoolExecutor(max_workers=4 * DP_SIZE) as pool:
            texts = list(pool.map(lambda _: self._complete(PROMPT), range(4 * DP_SIZE)))

        for i, text in enumerate(texts):
            self.assertIn(ANSWER, text, f"request {i}: {text!r}")
        self.assertEqual(len(set(texts)), 1, f"ranks disagreed: {set(texts)}")

    def test_prefill_on_one_rank_while_others_decode(self):
        # One wave never mixes modes (every rank prefills, then every rank
        # decodes); a second wave staggered into the first one's decode does.
        # The relabelled decode rows run the prefill kernel, so the long wave
        # is only required to keep decoding, not to match an unmixed run.
        long_prompt = "Count: 1, 2, 3,"
        with ThreadPoolExecutor(max_workers=2 * DP_SIZE) as pool:
            long_wave = [
                pool.submit(self._complete, long_prompt, 128) for _ in range(DP_SIZE)
            ]
            time.sleep(0.5)
            short_wave = list(pool.map(self._complete, [PROMPT] * DP_SIZE))
            long_texts = [f.result() for f in long_wave]

        for text in short_wave:
            self.assertIn(ANSWER, text)
        for text in long_texts:
            self.assertTrue(text.strip(), "decode stopped while a rank prefilled")

    def test_mixed_prompt_wave(self):
        # Different prompt lengths in one wave: ranks pad to different real
        # token counts inside one cross-rank bucket, and some run the idle batch.
        # Padding buckets differ per rank, so bf16 greedy outputs may diverge;
        # only the anchored prompt is checked for content.
        prompts = [
            PROMPT,
            "2 + 2 =",
            "Water boils at",
            "One kilogram equals how many grams?",
        ] * DP_SIZE
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            texts = list(pool.map(self._complete, prompts))

        for prompt, text in zip(prompts, texts):
            self.assertTrue(text.strip(), f"empty completion for {prompt!r}")
            if prompt == PROMPT:
                self.assertIn(ANSWER, text)


if __name__ == "__main__":
    unittest.main()
