"""Unit tests for the asymmetric-TP KV resharding schedule builder.

Pure byte-math and protobuf construction (the proto is mocked), so these run
anywhere -- no TPU / torch_tpu / tpu_raiden required. The on-device push path is
validated separately on a TPU VM.

Run: python -m pytest test/manual/kv_transfer/test_reshard_schedule.py
"""

import unittest

from sglang.srt.disaggregation.raiden import reshard_schedule as rs


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


if __name__ == "__main__":
    unittest.main()
