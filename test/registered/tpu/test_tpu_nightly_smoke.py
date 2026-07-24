"""Nightly smoke test for TPU serving.

Launches a small model (Qwen3-0.6B) on a single TPU host via the standard
`popen_launch_server` harness (the same pattern the GPU tests use) and runs a few
factual completions plus a small concurrent batch -- a quick end-to-end check that
inference is correct and the server is stable, without a full accuracy eval.
"""

import concurrent.futures
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(est_time=600, suite="nightly-tpu-4-v7", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"

# (prompt, expected substring) -- small factual smoke set.
_FACTS = (
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The largest planet in the solar system is", "Jupiter"),
)


class TestTpuNightlySmoke(CustomTestCase):
    """Quick nightly serving smoke on a single TPU host."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            model=MODEL,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dtype", "bfloat16",
                "--device", "tpu",
                # torch.compile auto-disables the tc_piecewise prefill graph
                # (unimplemented on TPU); small shapes keep warmup fast.
                "--enable-torch-compile",
                "--grammar-backend", "none",
                "--tp-size", "1",
                "--page-size", "64",
                "--chunked-prefill-size", "512",
                "--context-length", "1024",
                "--max-running-requests", "8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _complete(self, prompt, max_tokens=8):
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["choices"][0]["text"]

    def test_factual_completions(self):
        for prompt, expected in _FACTS:
            with self.subTest(prompt=prompt):
                self.assertIn(expected, self._complete(prompt=prompt))

    def test_small_concurrent_batch(self):
        prompts = ["Count to three: 1, 2," for _ in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(self._complete, prompts))
        self.assertEqual(len(results), 8)
        for text in results:
            self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
