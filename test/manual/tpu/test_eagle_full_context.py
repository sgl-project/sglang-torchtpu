"""Eager (no torch.compile) E2E test: EAGLE3 speculative decoding at (near)
full context length on TPU, with page_size > 1.

A prompt is built that fills the KV cache to within a few tokens of
``--context-length``, then decoded to the very edge. This exercises the
page_size>1 paths at the context boundary that are otherwise rarely hit:
paged slot allocation for the draft/verify tokens of the last partial page,
and the ``align_evict_mask_to_page_size`` partial-page eviction in verify.

Checks:
  * no crash -- both a baseline (no spec) and an EAGLE3 server serve a
    near-full-context request and return non-empty output;
  * the full context is actually exercised (prompt + generated tokens land
    within a few tokens of ``--context-length``);
  * lossless -- the greedy EAGLE3 output equals the greedy baseline output
    (spec decode is distribution-preserving; greedy => target argmax);
  * EAGLE3 still accepts drafts at the context boundary (accept length > 1).

Runs in eager mode (no ``--enable-torch-compile``), so it validates the
spec/attention/eviction logic independently of the compilation path.

Requires TPU hardware + torch_tpu and the EAGLE3 draft checkpoint.
Usage:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_eagle_full_context.py
"""

import importlib.util
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = os.environ.get("TPU_TEST_MODEL", "Qwen/Qwen3-1.7B")
DRAFT_MODEL = os.environ.get("TPU_TEST_DRAFT_MODEL", "AngelSlim/Qwen3-1.7B_eagle3")
TP_SIZE = os.environ.get("TPU_TEST_TP_SIZE", "4")
CONTEXT_LEN = 1024
PAGE_SIZE = 64
MAX_NEW_TOKENS = 16  # short greedy decode keeps the lossless check robust to
# bf16 ordering differences between batched target-verify and plain decode.
PROMPT_TOKENS = CONTEXT_LEN - MAX_NEW_TOKENS - 8  # land a few tokens shy of the edge


def _skip_if_no_tpu():
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _build_full_context_input_ids(n_tokens: int):
    """Return exactly ``n_tokens`` token ids using the model's tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    seed = (
        "In a faraway kingdom, a curious scholar catalogued the tides, the "
        "stars, and the migrations of birds, writing each finding in a long "
        "ledger that grew heavier with every passing season. "
    )
    ids = tok(seed, add_special_tokens=False).input_ids
    out = (ids * (n_tokens // len(ids) + 1))[:n_tokens]
    return out


_BASE_ARGS = [
    "--dtype",
    "bfloat16",
    "--device",
    "tpu",
    "--tp-size",
    TP_SIZE,
    "--attention-backend",
    "pallas_tpu_rpa_v3",
    "--page-size",
    str(PAGE_SIZE),
    "--context-length",
    str(CONTEXT_LEN),
    "--disable-overlap-schedule",
    "--grammar-backend",
    "none",
]

_EAGLE_ARGS = [
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    DRAFT_MODEL,
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestEagleFullContext(CustomTestCase):
    """EAGLE3 spec decode at the context boundary, eager, page_size>1."""

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.input_ids = _build_full_context_input_ids(PROMPT_TOKENS)

    def _generate(self):
        """POST a near-full-context greedy generation, return (text, meta)."""
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": self.input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            },
            timeout=300,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        return data["text"], data["meta_info"]

    def _run_server(self, extra_args):
        """Launch a server, run one near-full-context generation, tear down."""
        process = popen_launch_server(
            MODEL,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_BASE_ARGS + extra_args,
        )
        try:
            return self._generate()
        finally:
            kill_process_tree(process.pid)

    def test_eagle_full_context_lossless(self):
        # Baseline (no speculative decoding).
        baseline_text, baseline_meta = self._run_server([])
        # EAGLE3 speculative decoding at page_size>1.
        eagle_text, eagle_meta = self._run_server(_EAGLE_ARGS)

        # No crash: both produced output.
        self.assertGreater(len(baseline_text), 0)
        self.assertGreater(len(eagle_text), 0)

        # The full context was actually exercised (landed near the edge).
        used = eagle_meta["prompt_tokens"] + eagle_meta["completion_tokens"]
        self.assertLessEqual(used, CONTEXT_LEN)
        self.assertGreaterEqual(used, int(0.9 * CONTEXT_LEN))

        # Lossless: greedy spec decode reproduces the greedy baseline output.
        self.assertEqual(eagle_text, baseline_text)

        # Spec decode is still accepting drafts at the context boundary.
        verify_ct = eagle_meta.get("spec_verify_ct") or 0
        self.assertGreater(verify_ct, 0)
        accept_length = eagle_meta["completion_tokens"] / verify_ct
        self.assertGreater(accept_length, 1.0)


if __name__ == "__main__":
    unittest.main()
