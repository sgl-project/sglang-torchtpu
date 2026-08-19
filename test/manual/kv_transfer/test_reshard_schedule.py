"""Unit tests for the asymmetric-TP KV resharding byte-span declarations.

Pure byte-math, so these run anywhere -- no TPU / torch_tpu / tpu_raiden
required. The on-device transfer is validated separately on a TPU VM.

Run: python -m pytest test/manual/kv_transfer/test_reshard_schedule.py
"""

import unittest

from sglang.srt.disaggregation.raiden import reshard_schedule as rs


# ---- mock fused tensor [num_blocks, page, kv_heads_packed, kv_packing, hd] ----
class _Tensor:
    def __init__(self, shape, itemsize):
        self.shape = shape
        self._itemsize = itemsize

    def element_size(self):
        return self._itemsize


def _covered_intervals(registrations, dst_block_index):
    """Every byte interval the given declarations write to one destination page."""
    intervals = []
    for registration in registrations:
        for span in registration.spans:
            if span.dst_block_index != dst_block_index:
                continue
            for repeat in range(span.count):
                start = span.dst_offset_bytes + repeat * span.dst_stride_bytes
                intervals.append((start, start + span.size_bytes))
    return sorted(intervals)


def _declare_all_sources(geom, decode_rank, src_block_ids, live_tokens):
    """Declarations from every prefill rank feeding one decode rank."""
    return [
        rs.build_span_registration(
            tag="kv",
            geom=geom,
            prefill_rank=prefill_rank,
            decode_rank=decode_rank,
            src_block_ids=src_block_ids,
            live_tokens=live_tokens,
        )
        for prefill_rank in rs.source_ranks_for(decode_rank, geom.ratio)
    ]


class ReshardGeometryTest(unittest.TestCase):
    def test_head_pair_bytes_matches_both_sides(self):
        # hd=128 (bf16, kv_packing=2): head-pair = 2*128*2 = 512B. The granule
        # must be independent of the per-rank head count, since the two sides of
        # a reshard hold different head counts.
        prefill_t = _Tensor((100, 1024, 1, 2, 128), 2)
        decode_t = _Tensor((100, 1024, 2, 2, 128), 2)
        self.assertEqual(rs.head_pair_bytes(prefill_t), 512)
        self.assertEqual(rs.head_pair_bytes(decode_t), 512)
        self.assertEqual(rs.per_rank_head_count(prefill_t), 1)
        self.assertEqual(rs.per_rank_head_count(decode_t), 2)

    def test_geometry_8_to_4_concat(self):
        # total_kv_heads=8 >= src_tp=8: distinct heads, decode rank concatenates
        # its two prefill ranks at offsets 0 and 512, dst per-token width 1024.
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        self.assertEqual(geom.ratio, 2)
        # prefill 0 -> decode 0, writer 0 (offset 0); prefill 1 -> decode 0 (offset 512).
        for p, r, exp_dst_off in [(0, 0, 0), (1, 0, 512), (6, 3, 0), (7, 3, 512)]:
            src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(p, r)
            self.assertEqual((src_off, size, src_stride), (0, 512, 512))
            self.assertEqual(dst_off, exp_dst_off)
            self.assertEqual(dst_stride, 1024)  # 2 heads/token on decode

    def test_geometry_8_to_4_replication_qwen3(self):
        # total_kv_heads=4 < src_tp=8: heads replicated across prefill ranks. Both
        # of decode rank r's sources hold the SAME head, landing at offset 0 with
        # dst per-token width 512 (1 head/token).
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=4)
        for p, r in [(0, 0), (1, 0), (6, 3), (7, 3)]:
            src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(p, r)
            self.assertEqual(src_off, 0)
            self.assertEqual(size, 512)  # 1 head
            self.assertEqual(src_stride, 512)  # src 1 head/token
            self.assertEqual(dst_off, 0)  # both replicas -> offset 0
            self.assertEqual(dst_stride, 512)  # decode 1 head/token

    def test_rank_mapping(self):
        self.assertEqual(rs.reshard_ratio(8, 4), 2)
        self.assertEqual(rs.source_ranks_for(0, 2), [0, 1])
        self.assertEqual(rs.source_ranks_for(3, 2), [6, 7])
        self.assertEqual(rs.decode_rank_for(6, 2), 3)
        self.assertEqual(rs.writer_index(7, 2), 1)

    def test_non_divisible_ratio_raises(self):
        with self.assertRaises(ValueError):
            rs.reshard_ratio(8, 3)


class DeclaringRankTest(unittest.TestCase):
    def test_every_rank_declares_when_heads_are_distinct(self):
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        self.assertTrue(all(geom.declares_spans(rank) for rank in range(8)))

    def test_only_the_first_replica_declares(self):
        # 4 kv-heads over 8 prefill ranks: each head lives on two ranks. The
        # destination bytes may be claimed only once, so the odd ranks stay out.
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=4)
        self.assertEqual(
            [rank for rank in range(8) if geom.declares_spans(rank)], [0, 2, 4, 6]
        )
        silent = rs.build_span_registration(
            tag="kv",
            geom=geom,
            prefill_rank=1,
            decode_rank=0,
            src_block_ids=[5, 6],
            live_tokens=[1024, 1024],
        )
        self.assertEqual(silent.spans, ())
        self.assertEqual(silent.declared_bytes, 0)
        # The block ids still ride along: the controller admits a source set only
        # when every one of its ranks has registered.
        self.assertEqual(silent.block_ids, (5, 6))


class SpanCoverageTest(unittest.TestCase):
    """The rule the controller enforces: the sources of one decode rank must
    tile each destination page exactly once -- no gap, no overlap."""

    def _assert_tiles(self, registrations, num_pages, live_tokens, dst_token_bytes):
        for page in range(num_pages):
            intervals = _covered_intervals(registrations, page)
            self.assertTrue(intervals, f"page {page} has no coverage")
            covered_until = 0
            for start, end in intervals:
                self.assertEqual(start, covered_until, f"page {page} is not a tiling")
                covered_until = end
            self.assertEqual(covered_until, live_tokens[page] * dst_token_bytes)

    def test_concat_sources_tile_each_page(self):
        # 8 kv-heads, 8TP -> 4TP: two writers of one head each fill a 2-head
        # decode token.
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        live_tokens = [1024, 1024, 300]
        registrations = _declare_all_sources(
            geom, decode_rank=0, src_block_ids=[5, 6, 7], live_tokens=live_tokens
        )
        self._assert_tiles(registrations, 3, live_tokens, dst_token_bytes=1024)

    def test_replicated_sources_tile_each_page(self):
        # 4 kv-heads, 8TP -> 4TP: only the first replica declares, so the single
        # head still tiles the 1-head decode token exactly once.
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=4)
        live_tokens = [1024, 700]
        registrations = _declare_all_sources(
            geom, decode_rank=2, src_block_ids=[1, 2], live_tokens=live_tokens
        )
        self._assert_tiles(registrations, 2, live_tokens, dst_token_bytes=512)

    def test_declared_bytes_matches_spans(self):
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        registration = rs.build_span_registration(
            tag="kv",
            geom=geom,
            prefill_rank=1,
            decode_rank=0,
            src_block_ids=[5, 6, 7],
            live_tokens=[1024, 1024, 300],
        )
        self.assertEqual(registration.declared_bytes, 512 * (1024 + 1024 + 300))
        first = registration.spans[0]
        self.assertEqual((first.src_block_ordinal, first.dst_block_index), (0, 0))
        self.assertEqual(first.dst_offset_bytes, 512)
        self.assertEqual(first.dst_stride_bytes, 1024)
        self.assertEqual(registration.spans[2].count, 300)  # tail page

    def test_empty_page_is_not_declared(self):
        # A page with no live token must claim no destination bytes, or the
        # controller sees an overlap against the rank that does own them.
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        registration = rs.build_span_registration(
            tag="kv",
            geom=geom,
            prefill_rank=0,
            decode_rank=0,
            src_block_ids=[5, 6],
            live_tokens=[1024, 0],
        )
        self.assertEqual([span.dst_block_index for span in registration.spans], [0])

    def test_length_mismatch_raises(self):
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        with self.assertRaises(ValueError):
            rs.build_span_registration(
                tag="kv",
                geom=geom,
                prefill_rank=0,
                decode_rank=0,
                src_block_ids=[1, 2],
                live_tokens=[1024],
            )


if __name__ == "__main__":
    unittest.main()
