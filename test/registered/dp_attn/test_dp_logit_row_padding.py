# SPDX-License-Identifier: Apache-2.0
"""Every DP rank must feed the LM head the same number of rows.

Under DP attention with a DP LM head and attention TP > 1, the vocab-parallel
gather is a collective sized by the DP rank's REQUEST count. That axis is not
MAX_LEN padded the way the token axis is, so a rank that drains to zero requests
would issue a differently sized -- at zero rows, illegal -- collective while its
peers issue a full one, and the step deadlocks. The row count must therefore be
equalised across DP ranks with a floor of 1.
"""

from __future__ import annotations

import unittest

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.test.ci.ci_register import register_cpu_ci

# Pure arithmetic on the padding decision; no accelerator needed.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stub:
    """Only the state _dp_padded_logit_rows touches."""

    def __init__(self, use_attn_tp_group: bool, attn_tp_size: int):
        self.use_attn_tp_group = use_attn_tp_group
        self.attn_tp_size = attn_tp_size

    _dp_padded_logit_rows = LogitsProcessor._dp_padded_logit_rows


class _Mode:
    def __init__(self, decode_or_idle: bool):
        self._d = decode_or_idle

    def is_decode_or_idle(self) -> bool:
        return self._d


def _meta(counts, decode_or_idle: bool = True):
    m = LogitsMetadata.__new__(LogitsMetadata)
    m.global_num_tokens_for_logprob_cpu = counts
    m.forward_mode = _Mode(decode_or_idle)
    return m


class TestDpLogitRowPadding(unittest.TestCase):
    def test_idle_rank_is_padded_up_to_the_busiest(self):
        # The case that deadlocks: this rank has 0, peers have 1.
        proc = _Stub(True, 2)
        self.assertEqual(proc._dp_padded_logit_rows(0, _meta([0, 1, 1, 1])), 1)

    def test_all_ranks_agree_on_the_same_padded_count(self):
        proc, counts = _Stub(True, 2), [0, 3, 1, 2]
        pads = {proc._dp_padded_logit_rows(c, _meta(counts)) for c in counts}
        self.assertEqual(pads, {3}, "every rank must pick the same row count")

    def test_floor_of_one_when_every_rank_is_idle(self):
        # A zero-row collective is illegal even if no rank has work.
        proc = _Stub(True, 4)
        self.assertEqual(proc._dp_padded_logit_rows(0, _meta([0, 0, 0, 0])), 1)

    def test_never_truncates_below_the_real_rows(self):
        proc = _Stub(True, 2)
        self.assertEqual(proc._dp_padded_logit_rows(5, _meta([1, 1])), 5)

    def test_untouched_when_attn_tp_is_one(self):
        # Pure DP does no LM-head collective, so padding would be dead cost.
        proc = _Stub(True, 1)
        self.assertEqual(proc._dp_padded_logit_rows(0, _meta([0, 1, 1, 1])), 0)

    def test_untouched_without_dp_lm_head(self):
        proc = _Stub(False, 2)
        self.assertEqual(proc._dp_padded_logit_rows(0, _meta([0, 1])), 0)

    def test_extend_is_left_alone(self):
        # On an extend step the rows are per TOKEN, so the per-request counts
        # are the wrong unit; padding to them corrupts the next batch.
        proc = _Stub(True, 2)
        self.assertEqual(
            proc._dp_padded_logit_rows(0, _meta([0, 8, 8, 8], decode_or_idle=False)), 0
        )

    def test_no_metadata_is_a_no_op(self):
        proc = _Stub(True, 2)
        self.assertEqual(proc._dp_padded_logit_rows(4, _meta(None)), 4)


if __name__ == "__main__":
    unittest.main()
