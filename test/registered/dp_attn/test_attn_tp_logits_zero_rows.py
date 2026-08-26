# SPDX-License-Identifier: Apache-2.0
"""The attention-TP logits all-gather must tolerate a rank with no rows.

Under DP attention an idle DP rank reaches the LM head with zero selected
tokens. XLA rejects a collective whose operand has a zero-length dimension
("dimension size of operand at 'all_gather_dim' cannot be zero"), so the gather
has to be skipped rather than issued with an empty operand. Skipping stays
collective-safe because the row count is a property of the DP rank: every
member of an attention-TP group shards the same tokens over the vocabulary and
so sees zero together.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

# Pure tensor arithmetic: it exercises a TPU-only code path but needs no
# accelerator, so it runs on the cheap CPU runners.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.logits_processor import LogitsProcessor


class _StubLogitsProcessor:
    """Only the state _gather_attn_tp_logits touches."""

    def __init__(self, vocab_size: int, attn_tp_size: int):
        self.vocab_size = vocab_size
        self.attn_tp_size = attn_tp_size

    _gather_attn_tp_logits = LogitsProcessor._gather_attn_tp_logits


class TestAttnTpLogitsZeroRows(unittest.TestCase):
    def test_zero_rows_skips_the_collective(self):
        # A real all-gather here would need a process group; reaching one at
        # all is the failure. Returning early means we never do.
        proc = _StubLogitsProcessor(vocab_size=248320, attn_tp_size=2)
        shard = torch.empty((0, proc.vocab_size // proc.attn_tp_size))

        out = proc._gather_attn_tp_logits(shard)

        self.assertEqual(tuple(out.shape), (0, proc.vocab_size))
        self.assertEqual(out.dtype, shard.dtype)

    def test_zero_rows_when_vocab_not_divisible(self):
        # The other branch of the function must be short-circuited too.
        proc = _StubLogitsProcessor(vocab_size=1001, attn_tp_size=2)
        out = proc._gather_attn_tp_logits(torch.empty((0, 500)))
        self.assertEqual(tuple(out.shape), (0, 1001))


if __name__ == "__main__":
    unittest.main()
