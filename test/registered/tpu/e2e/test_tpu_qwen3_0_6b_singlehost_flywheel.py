"""--attention-backend flywheel serves Qwen3-0.6B as RPA v3 does.

Serves the same greedy requests with RPA v3 and then with flywheel, both on a
bf16 KV cache. Flywheel's tokens must match RPA's up to the first position
where RPA's top-2 logprobs are nearly tied; bf16 rounding can flip those.
The prompts run each kernel branch the server reaches at batch size 1: ragged
prefill (the 64-token bucket), dense prefill (a 512-token chunk), paged
extend (the later chunks and a radix-cache hit) and decode.
"""

import os
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

register_tpu_ci(
    est_time=600,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "accuracy", "flywheel"],
)

MODEL = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 32
# Largest RPA top-2 logprob gap at which the backends may pick different
# tokens. The one divergence seen on v6e sat at a gap of 0.125.
TIE_GAP = 0.25

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "List the first ten prime numbers:",
    "def fibonacci(n):",
    # About 1300 tokens: one dense 512-token chunk, then paged extend.
    "Summarize the following text.\n"
    + "The quick brown fox jumps over the lazy dog. " * 130,
]


def _skip_if_no_tpu():
    import importlib.util

    return importlib.util.find_spec("torch_tpu") is None


def _serve(attention_backend, kv_cache_dtype):
    """Greedy (output ids, RPA-style top-2 logprobs) per prompt, then the long
    prompt again, which hits the radix cache."""
    process = popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=[
            "--dtype",
            "bfloat16",
            "--device",
            "tpu",
            "--grammar-backend",
            "none",
            "--enable-torch-compile",
            "--tp-size",
            "1",
            "--attention-backend",
            attention_backend,
            "--kv-cache-dtype",
            kv_cache_dtype,
            "--page-size",
            "128",
            "--chunked-prefill-size",
            "512",
            "--context-length",
            "2048",
            # Requests run one at a time, so one batch size covers them.
            "--torch-compile-max-bs",
            "1",
            "--tpu-compile-bs",
            "1",
            "--tpu-compile-tokens",
            "64",
            "512",
        ],
    )
    try:
        return [_generate(p) for p in PROMPTS + PROMPTS[-1:]]
    finally:
        kill_process_tree(process.pid)


def _generate(prompt):
    resp = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0},
            "return_logprob": True,
            "top_logprobs_num": 2,
        },
        timeout=300,
    )
    resp.raise_for_status()
    meta = resp.json()["meta_info"]
    ids = [token_id for _, token_id, _ in meta["output_token_logprobs"]]
    gaps = [top[0][0] - top[1][0] for top in meta["output_top_logprobs"]]
    return ids, gaps


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostFlywheel(CustomTestCase):
    KV_CACHE_DTYPE = "bfloat16"

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls.rpa = _serve("pallas_tpu_rpa_v3", cls.KV_CACHE_DTYPE)
        cls.flywheel = _serve("flywheel", cls.KV_CACHE_DTYPE)

    def test_greedy_tokens_match_rpa_up_to_a_tie(self):
        for i, ((rpa_ids, rpa_gaps), (fw_ids, _)) in enumerate(
            zip(self.rpa, self.flywheel)
        ):
            with self.subTest(request=i):
                self.assertEqual(len(fw_ids), len(rpa_ids))
                diverged = next(
                    (j for j, (a, b) in enumerate(zip(rpa_ids, fw_ids)) if a != b),
                    None,
                )
                if diverged is not None:
                    self.assertLess(
                        rpa_gaps[diverged],
                        TIE_GAP,
                        f"token {diverged} differs where RPA is not near a tie",
                    )


class TestTpuQwen306bSinglehostFlywheelFp8(TestTpuQwen306bSinglehostFlywheel):
    KV_CACHE_DTYPE = "fp8_e4m3"


if __name__ == "__main__":
    unittest.main()
