"""Attention-TP ranks must extend over the same prefix (no TPU, no cluster).

Regression guard: when the ranks of a DP group matched different prefix lengths,
the batch was built to rank 0's shape and the ranks that matched less crashed
padding it by a negative amount. These cases pin the two properties of the
reconciliation that are silent if broken -- the extend range keeps its length,
and only a rank that matched more than the agreement is touched.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Range:
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

    @property
    def length(self) -> int:
        return self.end - self.start


class _Req:
    """Only the fields the reconciliation reads and writes."""

    def __init__(self, rid: str, prefix_len: int, extend_len: int) -> None:
        self.rid = rid
        self.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
        self.kv = SimpleNamespace(cache_protected_len=prefix_len)
        self.extend_range = _Range(prefix_len, prefix_len + extend_len)

    def set_extend_range(self, start: int, end: int) -> None:
        self.extend_range = _Range(start, end)


def _align(reqs, agreed):
    """Run the reconciliation with the collective stubbed to `agreed`."""
    scheduler = SimpleNamespace(
        enable_dp_attention=True,
        ps=SimpleNamespace(attn_tp_size=8),
        attn_tp_cpu_group=object(),
    )
    original = torch.distributed.all_reduce

    def fake_all_reduce(tensor, op=None, group=None):
        tensor.copy_(torch.tensor(agreed, dtype=tensor.dtype))

    torch.distributed.all_reduce = fake_all_reduce
    try:
        Scheduler._align_prefix_across_attn_tp(scheduler, reqs)
    finally:
        torch.distributed.all_reduce = original


class TestAttnTpPrefixAgreement(CustomTestCase):
    def test_shorter_peer_truncates_this_rank(self):
        # Numbers from the failing run: 8307-token request, 128-token pages.
        req = _Req("r0", prefix_len=7296, extend_len=1011)
        _align([req], agreed=[7168])

        self.assertEqual(len(req.prefix_indices), 7168)
        self.assertEqual(req.kv.cache_protected_len, 7168)
        self.assertEqual(req.extend_range.start, 7168)

    def test_extend_length_is_preserved_not_grown(self):
        # Growing the chunk by the 128 reclaimed tokens would overrun the budget
        # the adder reserved; the displaced tokens go to the next chunk instead.
        req = _Req("r0", prefix_len=7296, extend_len=1011)
        _align([req], agreed=[7168])

        self.assertEqual(req.extend_range.length, 1011)
        self.assertEqual(req.extend_range.end, 7168 + 1011)

    def test_agreement_equal_to_local_is_a_no_op(self):
        # Every rank matched the same prefix: nothing may move, or ordinary
        # traffic pays a recompute it does not owe.
        req = _Req("r0", prefix_len=7296, extend_len=1011)
        _align([req], agreed=[7296])

        self.assertEqual(len(req.prefix_indices), 7296)
        self.assertEqual(req.kv.cache_protected_len, 7296)
        self.assertEqual(req.extend_range.start, 7296)
        self.assertEqual(req.extend_range.length, 1011)

    def test_only_the_longer_ranks_are_truncated(self):
        # The min-reduce is per request: one already at the agreement must not
        # be disturbed by a neighbour that is not.
        short = _Req("short", prefix_len=7168, extend_len=1139)
        long_ = _Req("long", prefix_len=7296, extend_len=1011)
        _align([short, long_], agreed=[7168, 7168])

        self.assertEqual(len(short.prefix_indices), 7168)
        self.assertEqual(short.extend_range.length, 1139)
        self.assertEqual(len(long_.prefix_indices), 7168)
        self.assertEqual(long_.extend_range.length, 1011)

    def test_skipped_without_dp_attention(self):
        # Non-DPA deployments stay bit-for-bit unchanged and never enter the
        # collective, so nothing new can hang there.
        req = _Req("r0", prefix_len=7296, extend_len=1011)
        scheduler = SimpleNamespace(
            enable_dp_attention=False,
            ps=SimpleNamespace(attn_tp_size=8),
            attn_tp_cpu_group=object(),
        )

        def explode(*_args, **_kwargs):
            raise AssertionError("all_reduce must not run without DP attention")

        original = torch.distributed.all_reduce
        torch.distributed.all_reduce = explode
        try:
            Scheduler._align_prefix_across_attn_tp(scheduler, [req])
        finally:
            torch.distributed.all_reduce = original

        self.assertEqual(len(req.prefix_indices), 7296)


if __name__ == "__main__":
    unittest.main()
