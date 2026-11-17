"""Byte spans of the Raiden pipeline transfer's request-state pools.

A TP=1 pipeline stage sends each decode rank its shard of a request's GDN and
PLE state rows (``pool_spans.state_slice_spans``). These tests apply the spans
to a random source row on the host and compare every destination rank's bytes
with the shard the decode's own model would hold, so they need no TPU.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang_tpu.disaggregation.raiden.pool_spans import (
    is_state_pool,
    state_destination_blocks,
    state_page_shape,
    state_slice_spans,
)

TP = 8
ROW = 3
# Qwen3.8-Flash-Next GDN: 16 key heads x 128 (q, k), 48 value heads x 128.
GROUPS = [2048, 2048, 6144]


def _pool(tag, tensor):
    return dict(tag=tag, block_stride_bytes=tensor[0].numel() * tensor.element_size())


def _apply(spans, src_row, rank, nbytes):
    """Destination bytes of ``rank`` and how often each byte was written."""
    src = src_row.contiguous().view(torch.uint8).reshape(-1)
    dst = torch.zeros(nbytes, dtype=torch.uint8)
    hits = torch.zeros(nbytes, dtype=torch.int32)
    for span in spans:
        if getattr(span, "dst_unit_ordinal", rank) != rank:
            continue
        for c in range(span.count):
            s = span.src_offset_bytes + c * span.src_stride_bytes
            d = span.dst_offset_bytes + c * span.dst_stride_bytes
            dst[d : d + span.size_bytes] = src[s : s + span.size_bytes]
            hits[d : d + span.size_bytes] += 1
    return dst, hits


def _shard(tag, row, rank):
    if tag.startswith("gdn.conv"):
        q, k, v = row[..., :2048], row[..., 2048:4096], row[..., 4096:]
        w = 2048 // TP
        u = 6144 // TP
        return torch.cat(
            [
                q[..., w * rank : w * (rank + 1)],
                k[..., w * rank : w * (rank + 1)],
                v[..., u * rank : u * (rank + 1)],
            ],
            -1,
        )
    if tag.startswith("gdn.ssm"):
        heads = row.shape[0] // TP
        return row[heads * rank : heads * (rank + 1)]
    return row


class TestStateSliceSpans(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.src = {
            "gdn.conv.l0": torch.randn(5, 3, 1, 10240),
            "gdn.ssm.l0": torch.randn(5, 48, 128, 128),
            "ple.sc.l1": torch.randn(5, 10240, 9).to(torch.bfloat16),
            "ple.ngram.l0": torch.randint(0, 1000, (5, 2)),
        }
        self.pools = [_pool(t, x) for t, x in self.src.items()]

    def test_every_rank_receives_exactly_its_shard(self):
        spans = state_slice_spans(
            self.pools, list(self.src.values()), [ROW], TP, GROUPS
        )
        for pool_spans, (tag, x) in zip(spans, self.src.items()):
            self.assertEqual(pool_spans.tag, tag)
            self.assertEqual(
                pool_spans.declared_bytes,
                sum(s.size_bytes * s.count for s in pool_spans.spans),
            )
            for rank in range(TP):
                want = _shard(tag, x[ROW], rank).contiguous().view(torch.uint8).reshape(-1)
                got, hits = _apply(pool_spans.spans, x[ROW], rank, want.numel())
                self.assertTrue(torch.equal(got, want), f"{tag} rank {rank}")
                self.assertTrue(bool((hits == 1).all()), f"{tag} rank {rank} tiling")

    def test_tp1_moves_whole_rows(self):
        spans = state_slice_spans(self.pools, list(self.src.values()), [ROW], 1, GROUPS)
        for pool_spans, pool in zip(spans, self.pools):
            (span,) = pool_spans.spans
            self.assertFalse(hasattr(span, "dst_unit_ordinal"))
            self.assertEqual(span.size_bytes, pool["block_stride_bytes"])

    def test_conv_needs_its_qkv_widths(self):
        with self.assertRaises(ValueError):
            state_slice_spans(self.pools[:1], [self.src["gdn.conv.l0"]], [ROW], TP, None)

    def test_fingerprint_shape_drops_the_sharded_axis(self):
        conv = self.src["gdn.conv.l0"]
        self.assertEqual(state_page_shape("gdn.conv.l0", conv), [3, 1])
        self.assertEqual(state_page_shape("gdn.conv.l0", conv[..., :1280]), [3, 1])
        self.assertEqual(state_page_shape("gdn.ssm.l0", self.src["gdn.ssm.l0"]), [128, 128])
        self.assertEqual(state_page_shape("ple.ngram.l0", self.src["ple.ngram.l0"]), [2])
        self.assertTrue(is_state_pool("ple.sc.l1"))
        self.assertFalse(is_state_pool("fa.l3"))

    def test_destination_is_the_decode_row_in_every_pool(self):
        dst = state_destination_blocks(self.pools, [7], only_tags={"gdn.ssm.l0"})
        self.assertEqual(dst["transfer_pool_tags"], ["gdn.ssm.l0"])
        self.assertEqual(dst["dst_device_block_ids"], [7])
        self.assertEqual(dst["dst_block_counts"], [1])
        self.assertEqual(dst["dst_skip_bytes"], [0])


if __name__ == "__main__":
    unittest.main()
