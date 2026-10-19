"""Unit tests for the asymmetric-TP KV resharding schedule builder.

Pure byte-math and protobuf construction (the proto is mocked), so these run
anywhere -- no TPU / torch_tpu / tpu_sync required. The on-device push path is
validated separately on a TPU VM.

Run: python -m pytest test/manual/kv_transfer/test_reshard_schedule.py
"""

import unittest

from sglang_tpu.disaggregation.raiden import reshard_schedule as rs


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
    def test_layout_geometry_matches_both_sides(self):
        # hd=128 bf16 (kv_packing=2): one head's K+V per 512B granule.
        prefill = rs.KVLayoutGeometry.from_tensor(
            (100, 1024, 1, 2, 128), 2, 1, seq_along_lane=False
        )
        decode = rs.KVLayoutGeometry.from_tensor(
            (100, 1024, 2, 2, 128), 2, 2, seq_along_lane=False
        )
        self.assertEqual(prefill.unit(), decode.unit())
        self.assertEqual(prefill.granule_bytes, 512)
        self.assertEqual((prefill.row_bytes, decode.row_bytes), (512, 1024))

    def test_fp8_granule_holds_two_heads(self):
        # fp8 (kv_packing=4): a 512B granule is two heads, 256B each; one and
        # two heads per rank share a shape, so the count comes from the caller.
        one = rs.KVLayoutGeometry.from_tensor(
            (100, 256, 1, 4, 128), 1, 1, seq_along_lane=False
        )
        two = one.with_heads(2)
        self.assertEqual(one.granule_bytes / one.heads_per_granule, 256)
        self.assertEqual(one.row_bytes, two.row_bytes)
        geom = rs.ReshardGeometry.from_layout(4, 2, one, 8)
        # Prefill rank 1 holds heads 2-3: the second granule of decode rank 0.
        self.assertEqual(geom.entry_bytes(1, 0), (0, 512, 512, 512, 1024))

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


class LandRequestTest(unittest.TestCase):
    """fp8, 8 kv heads at TP8 -> TP4: odd prefill ranks land in scratch."""

    def setUp(self):
        layout = rs.KVLayoutGeometry(rs.TOKEN_MAJOR, 4, 1, 2, 128, 64)
        self.geom = rs.ReshardGeometry.from_layout(8, 4, layout, 8, land=True)

    def sender(self, rank, **kw):
        return rs.build_sender_request(
            _Pb2, uuid=1, req_id="1", peer="p", prefill_rank=rank,
            decode_rank=rank // 2, geom=self.geom, src_block_ids=[1, 2],
            dst_block_ids=[20, 21], live_tokens=[64, 9], **kw,
        )  # fmt: skip

    def test_landing_sources_target_scratch(self):
        req = rs.build_receiver_request(
            _Pb2, uuid=1, req_id="1", peer="p", decode_rank=1, geom=self.geom,
            sources=[(2, [1, 2]), (3, [1, 2])], dst_block_ids=[20, 21],
            live_tokens=[64, 9], scratch_block_ids=[30, 31],
        )  # fmt: skip
        self.assertEqual(req.expected_block_count, 4)
        recv = {
            k: [(e.dst_block_id, e.src_block_id, e.count) for e in v.entries]
            for k, v in req.shard_push_schedules.items()
        }
        self.assertEqual(
            recv, {2: [(20, 1, 64), (21, 2, 9)], 3: [(30, 1, 64), (31, 2, 9)]}
        )
        # Each sender names the blocks its receiver schedule expects.
        for rank in (2, 3):
            sent = self.sender(rank, scratch_block_ids=[30, 31])
            entries = sent.shard_push_schedules[0].entries
            self.assertEqual(
                [(e.dst_block_id, e.src_block_id, e.count) for e in entries], recv[rank]
            )
            self.assertEqual(
                [e.dst_offset_bytes for e in entries],
                [e.dst_offset_bytes for e in req.shard_push_schedules[rank].entries],
            )

    def test_scratch_ids_required_for_landing_only(self):
        with self.assertRaisesRegex(ValueError, "scratch_block_ids"):
            self.sender(3)
        with self.assertRaisesRegex(ValueError, "equal length"):
            self.sender(3, scratch_block_ids=[30])
        direct = self.sender(2)
        self.assertEqual(
            [e.dst_block_id for e in direct.shard_push_schedules[0].entries], [20, 21]
        )


class PackedKVScheduleTest(unittest.TestCase):
    @staticmethod
    def packed_page(heads, packing, itemsize, page, dim, block=None):
        result = bytearray()
        lanes = [value for head in heads for value in [(head, 0), (head, 1)]]
        lanes += [None] * (-len(lanes) % packing)
        for token in range(page):
            for start in range(0, len(lanes), packing):
                for column in range(dim):
                    for lane in lanes[start : start + packing]:
                        for byte in range(itemsize):
                            if block is None or lane is None:
                                value = 253
                            else:
                                head, kv = lane
                                value = (
                                    block * 17
                                    + token * 29
                                    + head * 37
                                    + kv * 43
                                    + column * 61
                                    + byte * 73
                                ) % 251
                            result.append(value)
        return result

    def check_transfer(self, src_tp, dst_tp, packing, itemsize, page, dim):
        geom = rs.ReshardGeometry(
            src_tp,
            dst_tp,
            2 * dim * itemsize,
            8,
            kv_packing=packing,
            element_size=itemsize,
        )
        src_heads, dst_heads = 8 // src_tp, 8 // dst_tp
        source_ids, destination_ids, live = [1, 3, 4], [2, 5, 6], [page, page - 3, 0]
        for decode_rank in range(dst_tp):
            wanted_heads = list(
                range(decode_rank * dst_heads, (decode_rank + 1) * dst_heads)
            )
            senders = [
                rank
                for rank in range(src_tp)
                if set(range(rank * src_heads, (rank + 1) * src_heads))
                & set(wanted_heads)
            ]
            blank = self.packed_page(wanted_heads, packing, itemsize, page, dim)
            destination = [bytearray(blank) for _ in range(7)]
            expected = [bytearray(blank) for _ in range(7)]
            receiver = rs.build_receiver_request(
                _Pb2,
                uuid=42,
                req_id="42",
                peer="localhost:31000",
                decode_rank=decode_rank,
                geom=geom,
                sources=[(rank + src_tp, source_ids) for rank in senders],
                dst_block_ids=destination_ids,
                live_tokens=live,
            )
            self.assertEqual(receiver.expected_block_count, len(senders) * 2)
            for rank in senders:
                heads = list(range(rank * src_heads, (rank + 1) * src_heads))
                sources = [
                    self.packed_page(heads, packing, itemsize, page, dim, block)
                    for block in range(5)
                ]
                sender = rs.build_sender_request(
                    _Pb2,
                    uuid=42,
                    req_id="42",
                    peer="localhost:31000",
                    prefill_rank=rank + src_tp,
                    decode_rank=decode_rank,
                    geom=geom,
                    src_block_ids=source_ids,
                    dst_block_ids=destination_ids,
                    live_tokens=live,
                )
                sent = sender.shard_push_schedules[0].entries
                received = receiver.shard_push_schedules[rank + src_tp].entries
                self.assertEqual(
                    [vars(entry) for entry in sent], [vars(entry) for entry in received]
                )
                for entry in received:
                    source = sources[entry.src_block_id]
                    target = destination[entry.dst_block_id]
                    for repeat in range(entry.count):
                        src = entry.src_offset_bytes + repeat * entry.src_stride_bytes
                        dst = entry.dst_offset_bytes + repeat * entry.dst_stride_bytes
                        size = entry.size_bytes
                        self.assertLessEqual(src + size, len(source))
                        self.assertLessEqual(dst + size, len(target))
                        target[dst : dst + size] = source[src : src + size]
            for source_id, destination_id, tokens in zip(
                source_ids, destination_ids, live
            ):
                filled = self.packed_page(
                    wanted_heads, packing, itemsize, page, dim, source_id
                )
                end = len(blank) // page * tokens
                # Padding lanes retain their original bytes in both buffers.
                expected[destination_id][:end] = filled[:end]
            self.assertEqual(destination, expected)

    def test_bf16_full_and_partial_pages(self):
        for src_tp, dst_tp in [(8, 1), (8, 4), (4, 1), (8, 2), (4, 2), (8, 8)]:
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                self.check_transfer(src_tp, dst_tp, 2, 2, 128, 64)

    def test_fan_out_takes_the_decode_ranks_head_slice(self):
        geom = rs.ReshardGeometry(1, 8, 512, 8, kv_packing=2, element_size=2)
        self.assertEqual(geom.ratio, 1)
        self.assertEqual(geom.entry_bytes(0, 3), (1536, 512, 4096, 0, 512))
        # Consecutive decode ranks share a kv-head when the model has fewer
        # kv-heads than decode ranks.
        geom = rs.ReshardGeometry(1, 8, 512, 4, kv_packing=2, element_size=2)
        self.assertEqual(geom.entry_bytes(0, 5), (1024, 512, 2048, 0, 512))
        self.assertEqual(rs.head_slice_params(1, 8, 0, 5, 4), (2, 1, 0, 4, 1))
        geom = rs.ReshardGeometry(1, 8, 256, 8, kv_packing=4, element_size=1)
        with self.assertRaises(ValueError):
            geom.entry_bytes(0, 3)
        with self.assertRaises(ValueError):
            rs.reshard_ratio(3, 8)


if __name__ == "__main__":
    unittest.main()
