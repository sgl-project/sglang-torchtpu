"""Compare Raiden reshard byte ranges with the actual fused KV pool layout."""

import collections
import threading
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import reshard_schedule as rs
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer
from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver
from sglang_tpu.kv_pool import (
    TpuBlockMajorMHATokenToKVPool,
    TpuFusedMHATokenToKVPool,
    TpuSeqAlongLaneKVPool,
)
from sglang_tpu.unified_kv_pool import AttnLayerSpec, TpuUnifiedSWAKVPool

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _layer(dtype, heads):
    pool = TpuFusedMHATokenToKVPool(
        size=128,
        page_size=64,
        dtype=dtype,
        head_num=heads,
        head_dim=128,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return pool.get_fused_kv_buffer(0)


class TestTpuReshardLayout(unittest.TestCase):
    def test_bf16_writer_ranges_cover_each_destination_token(self):
        for total_heads in (4, 8, 16, 32):
            for swa in (False, True):
                with self.subTest(total_heads=total_heads, swa=swa):
                    self._check_layout(torch.bfloat16, total_heads, swa=swa)

    def _check_layout(self, dtype, total_heads, *, swa=False):
        src_tp, dst_tp = 8, 4
        source = _layer(dtype, max(1, total_heads // src_tp))
        destination = _layer(dtype, max(1, total_heads // dst_tp))
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr.attn_tp_size = dst_tp
        mgr._kv_caches = [destination]
        mgr._swa_kv_caches = [destination] if swa else []
        mgr.kv_args = SimpleNamespace(total_kv_head_num=total_heads, page_size=64)
        geometry = mgr._make_geometry(src_tp, dst_tp, swa=swa)
        source_width = source.stride(1) * source.element_size()
        destination_width = destination.stride(1) * destination.element_size()
        for dst_rank in range(dst_tp):
            written = set()
            for src_rank in range(dst_rank * 2, dst_rank * 2 + 2):
                src_off, size, src_stride, dst_off, dst_stride = geometry.entry_bytes(
                    src_rank, dst_rank
                )
                self.assertEqual(src_stride, source_width)
                self.assertEqual(dst_stride, destination_width)
                self.assertLessEqual(src_off + size, source_width)
                self.assertLessEqual(dst_off + size, destination_width)
                written.update(range(dst_off, dst_off + size))
            self.assertEqual(written, set(range(destination_width)))

    def test_block_major_bundle_refuses_unequal_tp(self):
        """A bundle's axis 2 is page_size, not the kv-head axis the reshard
        geometry reads, so resharding a bundle must fail instead of computing
        wrong byte ranges."""
        pool = TpuBlockMajorMHATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
        )
        mgr = object.__new__(RaidenKVManager)
        mgr._kv_caches = pool.page_buffers()
        mgr._swa_kv_caches = pool.page_buffers()
        mgr.kv_args = SimpleNamespace(total_kv_head_num=8)
        for swa in (False, True):
            with self.subTest(swa=swa):
                with self.assertRaisesRegex(
                    ValueError, "SGLANG_TPU_USE_BLOCK_MAJOR_KV.*TP 8 on prefill and 4"
                ):
                    mgr._make_geometry(8, 4, swa=swa)

    def test_non_token_major_5d_layouts_refuse_unequal_tp(self):
        with patch(
            "sglang_tpu.kv_pool._seq_along_lane_head_dim",
            side_effect=lambda *, head_dim, kv_packing: head_dim,
        ):
            lane_pool = TpuUnifiedSWAKVPool(
                size=32,
                page_size=4,
                dtype=torch.bfloat16,
                full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=(0, 2)),
                swa=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=(1, 3)),
                device="cpu",
                token_to_kv_pool_class=TpuSeqAlongLaneKVPool,
            )
        lane_grp = get_raiden_kv_caches(lane_pool)
        self.assertTrue(lane_grp.seq_along_lane)

        for label, bufs, seq_lane in (
            ("mxint8_3d", [torch.zeros(9, 576, 128, dtype=torch.int8)], False),
            ("seq_along_lane_5d", lane_grp.full, lane_grp.seq_along_lane),
        ):
            with self.subTest(layout=label):
                mgr = object.__new__(RaidenKVManager)
                mgr._kv_caches = bufs
                mgr._unified_swa_pool = lane_pool
                mgr._seq_along_lane = seq_lane
                mgr.kv_args = SimpleNamespace(total_kv_head_num=8)
                with self.assertRaisesRegex(
                    ValueError, "5D token-major fused KV tensors"
                ):
                    mgr._make_geometry(8, 4)


TOTAL_KV_HEADS = 8  # GPT-OSS-120B
HEAD_DIM = 64  # GPT-OSS-120B; the pool pads it to 128 lanes
PAGE_SIZE = 64
NUM_PAGES = 6
PAD = 0xFD  # a byte no KV value takes (values are < 251)

# Byte layout of one token row inside the buffer Raiden copies:
#   "packed": 32-bit words hold ``kv_packing`` slots of one head-dim lane
#             (row = [pack][lane d][slot]), slot s in bits [8s*itemsize, ...).
#             The native TPU layout: the RPA kernel bitcasts a pack row to
#             uint32 lanes and shifts K/V out of each word, and the raw D2H/H2D
#             host copies keep the device bytes.
#   "contiguous": the torch-stride order (row = [pack][slot][lane d]); kept
#             only to show which results depend on the packing.
PHYSICAL_LAYOUT = "packed"


class _Entry:
    pass


class _EntryList(list):
    def add(self):
        entry = _Entry()
        self.append(entry)
        return entry


class _Schedule:
    def __init__(self):
        self.entries = _EntryList()


class _ScheduleMap(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = _Schedule()
        return dict.__getitem__(self, key)


class _Req:
    def __init__(self):
        self.shard_push_schedules = _ScheduleMap()


class _Pb2:
    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return _Req()


def _fused_layer(dtype, heads):
    pool = TpuFusedMHATokenToKVPool(
        size=PAGE_SIZE * (NUM_PAGES - 1),
        page_size=PAGE_SIZE,
        dtype=dtype,
        head_num=heads,
        head_dim=HEAD_DIM,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return pool.get_fused_kv_buffer(0)


def _manager(tensor, local_tp):
    mgr = object.__new__(RaidenKVManager)
    mgr.attn_tp_rank = 0
    mgr.attn_tp_size = local_tp
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = False
    mgr._kv_caches = [tensor]
    mgr._swa_kv_caches = []
    mgr.kv_args = SimpleNamespace(total_kv_head_num=TOTAL_KV_HEADS, page_size=PAGE_SIZE)
    return mgr


def _heads_of_rank(tp, rank):
    """Global kv-heads a rank holds, in local slot order (sglang head sharding:
    heads split evenly, replicated when the model has fewer heads than ranks)."""
    per_rank = max(1, TOTAL_KV_HEADS // tp)
    replicas = max(1, tp // TOTAL_KV_HEADS)
    first = (rank % tp) // replicas * per_rank
    return list(range(first, first + per_rank))


def _byte_offsets(tensor, layout):
    """Byte offset within one page of every (token, slot, lane, byte)."""
    _, page, packs, packing, lanes = tensor.shape
    itemsize = tensor.element_size()
    token_width = tensor.stride(1) * itemsize
    assert packing * itemsize == 4
    t = np.arange(page)[:, None, None, None]
    slot = np.arange(packs * packing)[None, :, None, None]
    d = np.arange(lanes)[None, None, :, None]
    b = np.arange(itemsize)[None, None, None, :]
    pack, lane_slot = slot // packing, slot % packing
    if layout == "packed":
        inner = pack * (lanes * 4) + d * 4 + lane_slot * itemsize + b
    elif layout == "contiguous":
        inner = (pack * packing + lane_slot) * (lanes * itemsize) + d * itemsize + b
    else:
        raise AssertionError(layout)
    return t * token_width + inner  # [page, slots, lanes, itemsize]


def _values(block, heads, kv_slots, itemsize):
    """Expected byte of (token, slot, lane, byte) for global ``heads`` in page
    ``block``; slot 2i is head i's K, 2i+1 its V, the rest padding."""
    t = np.arange(PAGE_SIZE)[:, None, None, None]
    d = np.arange(128)[None, None, :, None]
    b = np.arange(itemsize)[None, None, None, :]
    out = np.full((PAGE_SIZE, kv_slots, 128, itemsize), PAD, dtype=np.int64)
    for local, head in enumerate(heads):
        for kv in (0, 1):
            value = (block * 17 + t * 29 + head * 37 + kv * 43 + d * 61 + b * 73) % 251
            out[:, 2 * local + kv] = value[:, 0]
    return out


def _page_image(tensor, block, heads, layout):
    page_bytes = tensor.stride(0) * tensor.element_size()
    image = np.full(page_bytes, PAD, dtype=np.uint8)
    offsets = _byte_offsets(tensor, layout)
    vals = _values(block, heads, offsets.shape[1], tensor.element_size())
    image[offsets.reshape(-1)] = vals.reshape(-1)
    return image


def _apply(entry, source_pages, dest_pages):
    src = source_pages[entry.src_block_id]
    dst = dest_pages[entry.dst_block_id]
    rep = np.arange(entry.count, dtype=np.int64)[:, None]
    byte = np.arange(entry.size_bytes, dtype=np.int64)[None, :]
    src_idx = entry.src_offset_bytes + rep * entry.src_stride_bytes + byte
    dst_idx = entry.dst_offset_bytes + rep * entry.dst_stride_bytes + byte
    assert src_idx.min() >= 0 and src_idx.max() < src.size, "src out of block"
    assert dst_idx.min() >= 0 and dst_idx.max() < dst.size, "dst out of block"
    dst[dst_idx.reshape(-1)] = src[src_idx.reshape(-1)]


class TestTpuReshardFp8Packing(unittest.TestCase):
    """Decode rank j must hold, at every token of every page, exactly the K/V
    of its own kv-heads as written by the prefill ranks."""

    # Paired pages: prefill page ids differ from decode page ids; the last page
    # is partial (35 live tokens) so tokens 31/32 cross the half-page mark and
    # 63 is the page's last slot on the full pages.
    SRC_BLOCKS = [1, 2, 4]
    DST_BLOCKS = [3, 5, 2]
    LIVE = [PAGE_SIZE, PAGE_SIZE, 35]

    def _transfer(self, dtype, src_tp, dst_tp, layout=PHYSICAL_LAYOUT):
        src_heads = max(1, TOTAL_KV_HEADS // src_tp)
        dst_heads = max(1, TOTAL_KV_HEADS // dst_tp)
        prefill_t = _fused_layer(dtype, src_heads)
        decode_t = _fused_layer(dtype, dst_heads)
        sender_geom = _manager(prefill_t, src_tp)._make_geometry(src_tp, dst_tp)
        receiver_geom = _manager(decode_t, dst_tp)._make_geometry(src_tp, dst_tp)
        ratio = rs.reshard_ratio(src_tp, dst_tp)
        results = {}
        for decode_rank in range(dst_tp):
            if src_tp >= dst_tp:
                senders = rs.source_ranks_for(decode_rank, ratio)
            else:
                senders = [decode_rank * src_tp // dst_tp]
            receiver = rs.build_receiver_request(
                _Pb2,
                uuid=7,
                req_id="7",
                peer="decode:1",
                decode_rank=decode_rank,
                geom=receiver_geom,
                sources=[(rank, self.SRC_BLOCKS) for rank in senders],
                dst_block_ids=self.DST_BLOCKS,
                live_tokens=self.LIVE,
            )
            dest_pages = {
                b: np.full(
                    decode_t.stride(0) * decode_t.element_size(), PAD, dtype=np.uint8
                )
                for b in range(NUM_PAGES)
            }
            for rank in senders:
                sender = rs.build_sender_request(
                    _Pb2,
                    uuid=7,
                    req_id="7",
                    peer="decode:1",
                    prefill_rank=rank,
                    decode_rank=decode_rank,
                    geom=sender_geom,
                    src_block_ids=self.SRC_BLOCKS,
                    dst_block_ids=self.DST_BLOCKS,
                    live_tokens=self.LIVE,
                )
                sent = [vars(e) for e in sender.shard_push_schedules[0].entries]
                recv = [vars(e) for e in receiver.shard_push_schedules[rank].entries]
                for e in sent + recv:
                    e.pop("dst_peer", None)
                self.assertEqual(sent, recv, f"sender/receiver disagree, rank {rank}")
                source_pages = {
                    b: _page_image(prefill_t, b, _heads_of_rank(src_tp, rank), layout)
                    for b in range(NUM_PAGES)
                }
                for entry in receiver.shard_push_schedules[rank].entries:
                    _apply(entry, source_pages, dest_pages)
            results[decode_rank] = (dest_pages, decode_t)
        return results

    def _check(self, dtype, src_tp, dst_tp, layout=PHYSICAL_LAYOUT):
        results = self._transfer(dtype, src_tp, dst_tp, layout)
        for decode_rank, (dest_pages, decode_t) in results.items():
            heads = _heads_of_rank(dst_tp, decode_rank)
            offsets = _byte_offsets(decode_t, layout)
            itemsize = decode_t.element_size()
            for src_b, dst_b, live in zip(self.SRC_BLOCKS, self.DST_BLOCKS, self.LIVE):
                got = dest_pages[dst_b][offsets]  # [page, slots, lanes, itemsize]
                want = _values(src_b, heads, offsets.shape[1], itemsize)
                for local, head in enumerate(heads):
                    for kv in (0, 1):
                        slot = 2 * local + kv
                        bad = np.argwhere(got[:live, slot] != want[:live, slot])
                        if bad.size:
                            token = int(bad[0][0])
                            self.fail(
                                f"{dtype} {src_tp}->{dst_tp} decode rank "
                                f"{decode_rank} page {dst_b} token {token} "
                                f"head {head} {'KV'[kv]}: got "
                                f"{got[token, slot, :4, 0].tolist()} want "
                                f"{want[token, slot, :4, 0].tolist()} "
                                f"({len(bad)} bad bytes)"
                            )
                # Spot-check the page-boundary tokens explicitly.
                for token in (0, 31, 32, 63):
                    if token < live:
                        np.testing.assert_array_equal(
                            got[token, : 2 * len(heads)], want[token, : 2 * len(heads)]
                        )

    def test_fp8_fan_in_8_to_4(self):
        """GPT-OSS-120B fp8, prefill TP8 -> decode TP8 DP2 (attention TP4).
        One fp8 head per prefill rank cannot be placed into a decode pack it
        shares with another rank's head by any strided copy: refuse it."""
        with self.assertRaisesRegex(
            ValueError, "Partial FP8 head packs are unsupported"
        ):
            self._check(torch.float8_e4m3fn, 8, 4)

    def test_fp8_fan_in_8_to_2(self):
        with self.assertRaisesRegex(
            ValueError, "Partial FP8 head packs are unsupported"
        ):
            self._check(torch.float8_e4m3fn, 8, 2)

    def test_partial_pack_message_tp8_to_attn_tp4(self):
        decode_t = _fused_layer(torch.float8_e4m3fn, 2)
        geometry = _manager(decode_t, 4)._make_geometry(8, 4)
        with self.assertRaisesRegex(ValueError, "splits a 4-slot pack"):
            geometry.check_decode_rank(0)

    def test_register_reshard_request_refuses_partial_pack_without_sending(self):
        mgr = _manager(
            tensor=_fused_layer(dtype=torch.float8_e4m3fn, heads=2), local_tp=4
        )
        mgr.kv_args.engine_rank = 0
        mgr._state_lock = threading.Lock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr._failed = set()
        mgr._ensure_engine = mock.Mock()
        mgr._send_multipart_locked = mock.Mock()
        infos = [{"rank_ip": "10.0.0.1", "rank_port": p} for p in (1, 2)]
        mgr._register_reshard_request(
            room=42,
            local_blocks=[1, 2],
            local_swa_blocks=[],
            aux_index=None,
            bootstrap_infos=infos,
        )
        self.assertIn("42", mgr._failed)
        self.assertIn("Partial FP8 head packs are unsupported", mgr.failure_records[42])
        mgr._ensure_engine.assert_not_called()
        mgr._send_multipart_locked.assert_not_called()

    def test_fp8_equal_8_to_8(self):
        self._check(torch.float8_e4m3fn, 8, 8)

    def test_fp8_replicated_16_to_8(self):
        """Multihost TP16 prefill -> attention-TP8 decode (heads replicated)."""
        self._check(torch.float8_e4m3fn, 16, 8)

    def test_bf16_all_shapes(self):
        for src_tp, dst_tp in ((8, 4), (8, 8), (8, 2), (16, 8)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                self._check(torch.bfloat16, src_tp, dst_tp)

    def test_geometry_requires_layout(self):
        """kv_packing / element_size have no bf16 defaults any more."""
        with self.assertRaises(TypeError):
            rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)

    def test_token_width_checked_against_strides(self):
        """A geometry whose computed per-token width disagrees with the local
        tensor's real row is refused -- here the old bf16-style sizing, a whole fp8
        pack (512 B) taken as one head."""
        decode_t = _fused_layer(torch.float8_e4m3fn, 2)
        row = decode_t.stride(1) * decode_t.element_size()
        geometry = rs.ReshardGeometry(
            src_tp=8,
            dst_tp=4,
            hp_bytes=256,
            total_kv_heads=8,
            kv_packing=4,
            element_size=1,
        )
        geometry.check_token_bytes(tp=4, actual=row, what="decode")
        wrong = rs.ReshardGeometry(
            src_tp=8,
            dst_tp=4,
            hp_bytes=512,
            total_kv_heads=8,
            kv_packing=4,
            element_size=1,
        )
        with self.assertRaisesRegex(ValueError, "per-token KV row"):
            wrong.check_token_bytes(tp=4, actual=row, what="decode")

    def test_fp8_replicated_entry_unchanged(self):
        """TP16 -> attention TP8 fp8 (1 head -> 1 head) keeps the whole padded
        pack move it had before."""
        decode_t = _fused_layer(torch.float8_e4m3fn, 1)
        geometry = _manager(decode_t, 8)._make_geometry(16, 8)
        for prefill_rank in (0, 1, 14, 15):
            self.assertEqual(
                geometry.entry_bytes(prefill_rank, prefill_rank // 2),
                (0, 512, 512, 0, 512),
            )


class TestTpuFinerDecodeRefusal(unittest.TestCase):
    """A decode at a larger attention TP than its prefill pulls one prefill
    rank's whole pages (no head slicing), so it is refused unless both hold
    the same kv-heads."""

    def test_finer_decode_refusal(self):
        decode_t = _fused_layer(torch.float8_e4m3fn, 1)
        for total, prefill_tp, decode_tp, refused in (
            (8, 4, 8, True),  # 2 heads/prefill rank, 1/decode rank
            (8, 2, 8, True),
            (4, 4, 8, False),  # replicated: both hold 1 head
            (8, 8, 8, False),  # equal TP: pull
            (8, 16, 8, False),  # fan-in: reshard
        ):
            with self.subTest(total=total, prefill_tp=prefill_tp, decode_tp=decode_tp):
                mgr = _manager(decode_t, decode_tp)
                mgr.kv_args.total_kv_head_num = total
                mgr.is_mla_backend = False
                reason = mgr.finer_decode_refusal(prefill_tp)
                self.assertEqual(reason is not None, refused, reason)
                if refused:
                    self.assertIn("Cannot serve decode attention TP", reason)
        mgr = _manager(decode_t, 8)
        mgr.is_mla_backend = True
        self.assertIsNone(mgr.finer_decode_refusal(4))

    def test_receiver_fails_before_registering(self):
        mgr = _manager(_fused_layer(torch.float8_e4m3fn, 1), 8)
        mgr.is_mla_backend = False
        mgr.record_failure = mock.Mock()
        mgr.update_status = mock.Mock()
        mgr.register_decode_request = mock.Mock()
        receiver = object.__new__(RaidenKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 11
        receiver.conclude_state = None
        receiver.prefill_info = SimpleNamespace(attn_tp_size=4)
        receiver.abort_notified = False
        receiver.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 1}]
        receiver._send_abort_notification = mock.Mock()
        receiver.send_metadata(np.array([1, 2], dtype=np.int32))
        mgr.register_decode_request.assert_not_called()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        mgr.update_status.assert_called_once_with(11, KVPoll.Failed)
        self.assertIn("attention TP 8", mgr.record_failure.call_args[0][1])
        # The prefill is told right away instead of waiting for its timeout.
        receiver._send_abort_notification.assert_called_once_with()
        self.assertTrue(receiver.abort_notified)

    def test_pipeline_transfer_allows_finer_decode_tp(self):
        """Pipeline transfers slice KV heads per decode rank on the PP producer,
        so finer decode attention TP must register instead of failing the room."""
        mgr = _manager(
            tensor=_fused_layer(dtype=torch.float8_e4m3fn, heads=2), local_tp=8
        )
        mgr.kv_args = SimpleNamespace(
            total_kv_head_num=16,
            page_size=256,
            engine_rank=0,
            state_types=[],
        )
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.local_ip = "10.0.0.2"
        mgr.rank_port = 31000
        mgr._timeout_s = 120
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.request_status = {}
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._aborted = {}
        mgr.addr_to_rooms_tracker = collections.defaultdict(set)
        mgr.defer_decode_allocation = False
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._staging_outstanding = {}
        sent = []
        pipe = PipelineTransfer.__new__(PipelineTransfer)
        pipe.manager = mgr
        pipe.producer = False
        pipe.lock = threading.RLock()
        pipe.receives = {}
        pipe._send = lambda peer, frames: sent.append((peer, frames))
        mgr._pipeline_enabled = True
        mgr._pipeline = pipe

        receiver = RaidenKVReceiver(
            mgr=mgr, bootstrap_addr="10.0.0.1:8998", bootstrap_room=77
        )
        receiver.prefill_info = SimpleNamespace(attn_tp_size=1, pp_size=2)
        receiver.bootstrap_infos = [
            {"pp_rank": 0, "rank_ip": "10.0.0.1", "rank_port": 30001},
            {"pp_rank": 1, "rank_ip": "10.0.0.1", "rank_port": 30002},
        ]
        receiver._send_abort_notification = mock.Mock()

        receiver.send_metadata(
            kv_indices=np.array([10, 11], dtype=np.int32), state_indices=[]
        )

        self.assertEqual(mgr.failure_records, {})
        self.assertIsNone(receiver.conclude_state)
        self.assertEqual(mgr.request_status[77], KVPoll.WaitingForInput)
        self.assertIn(77, pipe.receives)
        self.assertEqual(len(sent), 2)
        receiver._send_abort_notification.assert_not_called()


if __name__ == "__main__":
    unittest.main()
