"""Unit tests for the asymmetric-TP KV resharding schedule builder.

Pure byte-math and protobuf construction (the proto is mocked), so these run
anywhere -- no TPU / torch_tpu / tpu_sync required. The on-device push path is
validated separately on a TPU VM.

Run: python3 test/registered/unit/disaggregation/test_reshard_schedule.py
"""

import unittest

from sglang.srt.disaggregation.raiden import reshard_schedule as rs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# ---- mock protobuf StartTransferRequest --------------------------------------
class _Entry:
    pass


class _EntryList(list):
    def add(self):
        e = _Entry()
        self.append(e)
        return e


class _Schedule:
    def __init__(self):
        self.entries = _EntryList()


class _ScheduleMap(dict):
    def __getitem__(self, k):
        if k not in self:
            self[k] = _Schedule()
        return dict.__getitem__(self, k)


class _Req:
    def __init__(self):
        self.shard_push_schedules = _ScheduleMap()


class _Pb2:
    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return _Req()


# ---- mock fused tensor [num_blocks, page, kv_heads_packed, kv_packing, hd] ----
class _Tensor:
    def __init__(self, shape, itemsize):
        self.shape = shape
        self._itemsize = itemsize

    def element_size(self):
        return self._itemsize


class ReshardGeometryTest(unittest.TestCase):
    def test_head_pair_bytes_matches_both_sides(self):
        # hd=128 (bf16, kv_packing=2): head-pair = 2*128*2 = 512B.
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
        # dst per-token width 512 (1 head/token). Whole-block, idempotent.
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


class SenderRequestTest(unittest.TestCase):
    def test_sender_concat_dst_placement(self):
        # 8 kv-heads: prefill rank 1 -> decode rank 0, second writer at offset 512.
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        req = rs.build_sender_request(
            _Pb2,
            uuid=42,
            req_id="42",
            peer="10.0.0.9:30001",
            prefill_rank=1,
            decode_rank=0,
            geom=geom,
            src_block_ids=[5, 6, 7],
            dst_block_ids=[50, 60, 70],
            live_tokens=[1024, 1024, 300],
        )
        self.assertTrue(req.is_sender)
        self.assertEqual(req.dst_mem_type, _Pb2.MEMORY_TYPE_DRAM)
        entries = req.shard_push_schedules[0].entries
        self.assertEqual(len(entries), 3)
        e = entries[0]
        self.assertEqual((e.src_block_id, e.dst_block_id), (5, 50))
        self.assertEqual(
            (e.src_offset_bytes, e.size_bytes, e.src_stride_bytes), (0, 512, 512)
        )
        self.assertEqual(e.dst_offset_bytes, 512)
        self.assertEqual(e.dst_stride_bytes, 1024)
        self.assertEqual(e.count, 1024)
        self.assertEqual(entries[2].count, 300)  # tail page

    def test_length_mismatch_raises(self):
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        with self.assertRaises(ValueError):
            rs.build_sender_request(
                _Pb2,
                uuid=1,
                req_id="1",
                peer="p",
                prefill_rank=0,
                decode_rank=0,
                geom=geom,
                src_block_ids=[1, 2],
                dst_block_ids=[1],
                live_tokens=[1024, 1024],
            )


class ReceiverRequestTest(unittest.TestCase):
    def test_receiver_concat_two_writers_8heads(self):
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)
        req = rs.build_receiver_request(
            _Pb2,
            uuid=42,
            req_id="42",
            peer="10.0.0.5:30001",
            decode_rank=0,
            geom=geom,
            sources=[(0, [5, 6, 7]), (1, [5, 6, 7])],
            dst_block_ids=[50, 60, 70],
            live_tokens=[1024, 1024, 300],
        )
        self.assertFalse(req.is_sender)
        self.assertEqual(req.dst_mem_type, _Pb2.MEMORY_TYPE_HBM)
        self.assertEqual(req.expected_block_count, 6)  # 2 writers * 3 pages
        self.assertEqual(req.shard_push_schedules[0].entries[0].dst_offset_bytes, 0)
        self.assertEqual(req.shard_push_schedules[1].entries[0].dst_offset_bytes, 512)
        self.assertEqual(req.shard_push_schedules[0].entries[0].dst_stride_bytes, 1024)

    def test_receiver_replication_two_writers_qwen3(self):
        # 4 kv-heads: both writers land the same head at offset 0 (idempotent).
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=4)
        req = rs.build_receiver_request(
            _Pb2,
            uuid=7,
            req_id="7",
            peer="10.0.0.5:30001",
            decode_rank=0,
            geom=geom,
            sources=[(0, [5]), (1, [5])],
            dst_block_ids=[50],
            live_tokens=[1024],
        )
        self.assertEqual(req.expected_block_count, 2)  # 2 writers * 1 page
        self.assertEqual(req.shard_push_schedules[0].entries[0].dst_offset_bytes, 0)
        self.assertEqual(req.shard_push_schedules[1].entries[0].dst_offset_bytes, 0)
        self.assertEqual(req.shard_push_schedules[0].entries[0].dst_stride_bytes, 512)

    def test_receiver_schedule_keyed_by_source_node_id(self):
        # Decode rank 2 is fed by prefill ranks 4 and 5. The receiver picks a
        # source's schedule by the pushing node's id, so the schedules must be
        # keyed by prefill rank (4, 5), not by writer position (0, 1).
        geom = rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=4)
        req = rs.build_receiver_request(
            _Pb2,
            uuid=9,
            req_id="9",
            peer="10.0.0.5:30001",
            decode_rank=2,
            geom=geom,
            sources=[(4, [1]), (5, [1])],
            dst_block_ids=[1],
            live_tokens=[1024],
        )
        self.assertEqual(set(req.shard_push_schedules.keys()), {4, 5})
        self.assertEqual(req.shard_push_schedules[4].entries[0].src_block_id, 1)
        self.assertEqual(req.shard_push_schedules[5].entries[0].dst_block_id, 1)


class BlockLayoutTest(unittest.TestCase):
    """The two KV layouts, and the padding the token-major one can carry."""

    # bf16 token-major, 1 kv-head/rank: 2 K/V slots exactly fill kv_packing=2,
    # so the token row is unpadded and equals heads * head_pair.
    NHD_BF16 = _Tensor((100, 1024, 1, 2, 128), 2)
    # fp8 token-major, same rank: 2 slots in a group of 4 -- HALF the row is pad.
    NHD_FP8 = _Tensor((100, 256, 1, 4, 128), 1)
    # fp8 seq-along-lane, 1 kv-head/rank: [blocks, 2*heads, hd//pack, pack, page].
    HND_FP8_1H = _Tensor((100, 2, 32, 4, 256), 1)
    HND_FP8_2H = _Tensor((100, 4, 32, 4, 256), 1)

    def test_head_pair_bytes_is_the_logical_unit_not_an_address(self):
        # One head's K+V is 2*head_dim whatever the dtype. Under fp8 that is
        # only HALF a packing group, and half a group is not a byte range --
        # see the head_span tests. head_pair_bytes stays logical so prefill and
        # decode can compare it.
        self.assertEqual(rs.head_pair_bytes(self.NHD_BF16), 512)
        self.assertEqual(rs.head_pair_bytes(self.NHD_FP8), 256)

    def test_head_pair_seq_along_lane_spans_the_page(self):
        # A head's K (and V) each cover every token of the page: 32*4*256 = 32KiB.
        hp = rs.head_pair_bytes(self.HND_FP8_1H, rs.LAYOUT_SEQ_ALONG_LANE)
        self.assertEqual(hp, 2 * 32 * 4 * 256)
        # Independent of how many heads the rank holds -- it must match on both sides.
        self.assertEqual(
            rs.head_pair_bytes(self.HND_FP8_2H, rs.LAYOUT_SEQ_ALONG_LANE), hp
        )

    def test_per_rank_head_count(self):
        self.assertEqual(
            rs.per_rank_head_count(self.HND_FP8_2H, rs.LAYOUT_SEQ_ALONG_LANE), 2
        )

    def test_row_bytes_rounds_up_to_the_packing_group(self):
        nhd = rs.BlockLayout.from_tensor(self.NHD_FP8, rs.LAYOUT_HEAD_ALONG_SUBLANE)
        # 1 head = 2 slots, rounded up to a 4-slot group: 4*128 = 512B per token.
        self.assertEqual(nhd.row_bytes(1), 512)
        # 2 heads = 4 slots: exactly one group, no pad.
        self.assertEqual(nhd.row_bytes(2), 512)
        self.assertEqual(nhd.row_bytes(4), 1024)

    def test_row_bytes_seq_along_lane_is_the_head_slab(self):
        hnd = rs.BlockLayout.from_tensor(self.HND_FP8_1H, rs.LAYOUT_SEQ_ALONG_LANE)
        self.assertEqual(hnd.row_bytes(1), 65536)
        self.assertEqual(hnd.row_bytes(2), 131072)

    def test_count_collapses_to_one_under_seq_along_lane(self):
        nhd = rs.BlockLayout.from_tensor(self.NHD_FP8, rs.LAYOUT_HEAD_ALONG_SUBLANE)
        hnd = rs.BlockLayout.from_tensor(self.HND_FP8_1H, rs.LAYOUT_SEQ_ALONG_LANE)
        self.assertEqual(nhd.count_for(300), 300)
        self.assertEqual(hnd.count_for(300), 1)

    def test_granule_is_the_packing_group_under_nhd(self):
        bf16 = rs.BlockLayout.from_tensor(self.NHD_BF16, rs.LAYOUT_HEAD_ALONG_SUBLANE)
        fp8 = rs.BlockLayout.from_tensor(self.NHD_FP8, rs.LAYOUT_HEAD_ALONG_SUBLANE)
        # bf16 packs 2 slots per 32-bit word group = exactly one head.
        self.assertEqual(bf16.heads_per_granule, 1)
        self.assertEqual(bf16.granule_bytes, 512)
        # fp8 packs 4 = two heads share their words and cannot be separated.
        self.assertEqual(fp8.heads_per_granule, 2)
        self.assertEqual(fp8.granule_bytes, 512)

    def test_granule_is_one_head_under_hnd(self):
        hnd = rs.BlockLayout.from_tensor(self.HND_FP8_2H, rs.LAYOUT_SEQ_ALONG_LANE)
        # The head axis carries no packing here, so fp8 still slices per head.
        self.assertEqual(hnd.heads_per_granule, 1)
        self.assertEqual(hnd.granule_bytes, hnd.hp_bytes)

    def test_head_span_pads_a_lone_fp8_head_out_to_its_group(self):
        fp8 = rs.BlockLayout.from_tensor(self.NHD_FP8, rs.LAYOUT_HEAD_ALONG_SUBLANE)
        # A 1-head rank owns 2 of the group's 4 interleaved slots, but its
        # bytes are spread over every word of the group, so the copy is all
        # 512B. Moving the nominal 256B instead is what corrupted decode.
        self.assertEqual(fp8.head_span(0, 1, 1), (0, 512))
        # A 2-head rank fills the group exactly.
        self.assertEqual(fp8.head_span(0, 2, 2), (0, 512))
        # 4 heads = two groups, and the second is addressable on its own.
        self.assertEqual(fp8.head_span(2, 2, 4), (512, 512))

    def test_head_span_rejects_a_slice_inside_an_fp8_group(self):
        fp8 = rs.BlockLayout.from_tensor(self.NHD_FP8, rs.LAYOUT_HEAD_ALONG_SUBLANE)
        # Starts mid-group: head 1's bytes are interleaved with head 0's.
        with self.assertRaises(ValueError):
            fp8.head_span(1, 1, 2)
        # Ends mid-group while head 1 of the same rank owns the remainder.
        with self.assertRaises(ValueError):
            fp8.head_span(0, 1, 2)

    def test_head_span_is_exact_for_bf16_and_for_hnd(self):
        bf16 = rs.BlockLayout.from_tensor(self.NHD_BF16, rs.LAYOUT_HEAD_ALONG_SUBLANE)
        hnd = rs.BlockLayout.from_tensor(self.HND_FP8_2H, rs.LAYOUT_SEQ_ALONG_LANE)
        self.assertEqual(bf16.head_span(1, 1, 2), (512, 512))
        self.assertEqual(hnd.head_span(1, 1, 2), (hnd.hp_bytes, hnd.hp_bytes))

    def test_unknown_layout_rejected(self):
        with self.assertRaises(ValueError):
            rs.BlockLayout("nchw", hp_bytes=1)
        with self.assertRaises(ValueError):
            rs.head_pair_bytes(self.NHD_BF16, "nchw")


class SeqAlongLaneGeometryTest(unittest.TestCase):
    """8TP prefill -> 4TP decode, 8 kv-heads, fp8, page 256, head_dim 128."""

    SLAB = 32 * 4 * 256  # one axis-1 entry = one head's K (or V) over the page
    HP = 2 * SLAB  # 64KiB

    def _geom(self, total_kv_heads=8):
        block = rs.BlockLayout.from_tensor(
            _Tensor((100, 2, 32, 4, 256), 1), rs.LAYOUT_SEQ_ALONG_LANE
        )
        return rs.ReshardGeometry(8, 4, block.hp_bytes, total_kv_heads, block)

    def test_geometry_is_a_single_contiguous_run(self):
        geom = self._geom()
        self.assertTrue(geom.is_seq_along_lane)
        for p, r, exp_dst_off in [(0, 0, 0), (1, 0, self.HP), (7, 3, self.HP)]:
            src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(p, r)
            self.assertEqual(src_off, 0)
            self.assertEqual(size, self.HP)  # 1 head = the whole source block
            # The source block IS the run, so the stride is never re-applied.
            self.assertEqual(src_stride, self.HP)
            self.assertEqual(dst_off, exp_dst_off)
            self.assertEqual(dst_stride, 2 * self.HP)  # decode holds 2 heads
        # One run per page instead of one per token: 256x fewer descriptors.
        self.assertEqual(geom.count_for(256), 1)
        # The regroup fp8 NHD has to refuse is routine here.
        geom.validate()

    def test_writers_tile_the_decode_block_exactly(self):
        # The two writers' [dst_off, dst_off+size) must partition the decode block.
        geom = self._geom()
        spans = []
        for p in rs.source_ranks_for(0, geom.ratio):
            _, size, _, dst_off, dst_stride = geom.entry_bytes(p, 0)
            spans.append((dst_off, dst_off + size))
        self.assertEqual(spans, [(0, self.HP), (self.HP, 2 * self.HP)])
        self.assertEqual(spans[-1][1], dst_stride)  # no gap, no overrun

    def test_sender_emits_one_entry_per_page(self):
        geom = self._geom()
        req = rs.build_sender_request(
            _Pb2,
            uuid=42,
            req_id="42",
            peer="10.0.0.9:30001",
            prefill_rank=1,
            decode_rank=0,
            geom=geom,
            src_block_ids=[5, 6, 7],
            dst_block_ids=[50, 60, 70],
            live_tokens=[256, 256, 30],
        )
        entries = req.shard_push_schedules[0].entries
        self.assertEqual(len(entries), 3)
        for e in entries:
            self.assertEqual(e.count, 1)
            self.assertEqual(e.size_bytes, self.HP)
            self.assertEqual(e.dst_offset_bytes, self.HP)
        # A partially filled page still moves whole: the dead tail lanes are
        # overwritten before they are read.
        self.assertEqual(entries[2].count, 1)

    def test_receiver_mirrors_the_sender(self):
        geom = self._geom()
        req = rs.build_receiver_request(
            _Pb2,
            uuid=42,
            req_id="42",
            peer="10.0.0.5:30001",
            decode_rank=0,
            geom=geom,
            sources=[(0, [5, 6]), (1, [5, 6])],
            dst_block_ids=[50, 60],
            live_tokens=[256, 30],
        )
        self.assertEqual(req.expected_block_count, 4)
        self.assertEqual(req.shard_push_schedules[0].entries[0].dst_offset_bytes, 0)
        self.assertEqual(
            req.shard_push_schedules[1].entries[0].dst_offset_bytes, self.HP
        )
        for sched in req.shard_push_schedules.values():
            for e in sched.entries:
                self.assertEqual(e.count, 1)

    def test_replicated_heads_land_idempotently(self):
        # 4 kv-heads over 8 prefill ranks: both writers carry the same head and
        # must target the same bytes.
        geom = self._geom(total_kv_heads=4)
        a = geom.entry_bytes(0, 0)
        b = geom.entry_bytes(1, 0)
        self.assertEqual(a, b)
        self.assertEqual(a[3], 0)  # dst_offset
        self.assertEqual(a[4], self.HP)  # decode holds 1 head


class Fp8TokenMajorGeometryTest(unittest.TestCase):
    """fp8 NHD: the packed group holds two heads, which bounds what can move."""

    def _geom(self, src_tp=8, dst_tp=4, total_kv_heads=4):
        block = rs.BlockLayout.from_tensor(
            _Tensor((100, 256, 1, 4, 128), 1), rs.LAYOUT_HEAD_ALONG_SUBLANE
        )
        return rs.ReshardGeometry(src_tp, dst_tp, block.hp_bytes, total_kv_heads, block)

    def test_replicated_single_head_moves_the_whole_group(self):
        # Qwen3-Coder-30B, 8TP prefill -> 4TP decode: 4 kv-heads replicated, so
        # 1 head per rank on both sides, each sitting in a half-padded group.
        # The transfer is the whole 512B group -- what the pre-HND code did,
        # and what the e2e control arm proved is required.
        geom = self._geom()
        for p, r in [(0, 0), (1, 0), (6, 3), (7, 3)]:
            self.assertEqual(geom.entry_bytes(p, r), (0, 512, 512, 0, 512))
        geom.validate()

    def test_count_is_still_per_token(self):
        self.assertEqual(self._geom().count_for(30), 30)

    def test_regrouping_one_head_into_two_is_rejected(self):
        # 8 kv-heads, 8TP -> 4TP: each decode rank must CONCATENATE two distinct
        # heads into one group. Under fp8 the second head's slots interleave
        # with the first's, so no strided copy can place it. Refuse loudly
        # rather than write plausible-looking garbage.
        geom = self._geom(total_kv_heads=8)
        # Neither writer works: writer 0 would stop halfway through the group
        # and writer 1 would start halfway into it.
        with self.assertRaises(ValueError):
            geom.entry_bytes(0, 0)
        with self.assertRaises(ValueError):
            geom.entry_bytes(1, 0)
        with self.assertRaises(ValueError):
            geom.validate()

    def test_bf16_can_regroup_where_fp8_cannot(self):
        # The same reshard at kv_packing=2, where each head is its own group.
        block = rs.BlockLayout.from_tensor(
            _Tensor((100, 1024, 1, 2, 128), 2), rs.LAYOUT_HEAD_ALONG_SUBLANE
        )
        geom = rs.ReshardGeometry(8, 4, block.hp_bytes, 8, block)
        geom.validate()
        self.assertEqual(geom.entry_bytes(1, 0), (0, 512, 512, 512, 1024))


class LegacyGeometryTest(unittest.TestCase):
    def test_geometry_without_a_block_keeps_the_old_behaviour(self):
        geom = rs.ReshardGeometry(8, 4, 512, 8)
        self.assertFalse(geom.is_seq_along_lane)
        self.assertEqual(geom.entry_bytes(1, 0), (0, 512, 512, 512, 1024))
        self.assertEqual(geom.count_for(300), 300)

    def test_block_must_agree_with_hp_bytes(self):
        block = rs.BlockLayout.from_tensor(
            _Tensor((100, 1024, 1, 2, 128), 2), rs.LAYOUT_HEAD_ALONG_SUBLANE
        )
        rs.ReshardGeometry(8, 4, 512, 8, block)  # consistent
        with self.assertRaises(ValueError):
            rs.ReshardGeometry(8, 4, 1024, 8, block)


if __name__ == "__main__":
    unittest.main()
