"""Compare Raiden reshard byte ranges with the actual fused KV pool layout."""

import collections
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import reshard_schedule as rs
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer
from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.reshard_merge import (
    bucket_pages,
    merge_staged_pages,
)
from sglang_tpu.disaggregation.raiden.serde import UNIFIED_SWA_KV_LAYOUT
from sglang_tpu.disaggregation.raiden.state import DecodeState
from sglang_tpu.kv_pool import (
    TpuBlockMajorMHATokenToKVPool,
    TpuFusedMHATokenToKVPool,
    TpuSeqAlongLaneKVPool,
)
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


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
NUM_PAGES = 10
PAD = 0xFD  # a byte no KV value takes (values are < 251)
FP8 = torch.float8_e4m3fn

# Native RPA token-row layout is [pack][lane][slot] in 32-bit lane words,
# preserved by raw D2H/H2D host copies.
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


def _fused_layer(dtype, heads, num_pages=NUM_PAGES):
    pool = TpuFusedMHATokenToKVPool(
        size=PAGE_SIZE * (num_pages - 1),
        page_size=PAGE_SIZE,
        dtype=dtype,
        head_num=heads,
        head_dim=HEAD_DIM,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return pool.get_fused_kv_buffer(0)


def _manager(tensor, local_tp, total_heads=TOTAL_KV_HEADS):
    mgr = object.__new__(RaidenKVManager)
    mgr.attn_tp_rank = 0
    mgr.attn_tp_size = local_tp
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = False
    mgr._kv_caches = [tensor]
    mgr._swa_kv_caches = []
    mgr.kv_args = SimpleNamespace(total_kv_head_num=total_heads, page_size=PAGE_SIZE)
    return mgr


def _heads_of_rank(tp, rank, total_heads=TOTAL_KV_HEADS):
    per_rank = max(1, total_heads // tp)
    replicas = max(1, tp // total_heads)
    first = (rank % tp) // replicas * per_rank
    return list(range(first, first + per_rank))


def _byte_offsets(tensor, layout):
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


def _apply_merge(dest_pages, decode_t, layout, real_pages, staging_pages, merges):
    """The decode's merge, defined on logical slots, realized on the bytes."""
    offsets = _byte_offsets(decode_t, layout)
    for real, stage in zip(real_pages, staging_pages):
        for stage_slot, decode_slot, num_slots in merges:
            for i in range(num_slots):
                dest_pages[real][offsets[:, decode_slot + i].reshape(-1)] = dest_pages[
                    stage
                ][offsets[:, stage_slot + i].reshape(-1)]


class TestTpuReshardFp8Packing(unittest.TestCase):
    """Raiden push resharding must place each prefill kv-head into its decode slot."""

    # Distinct src/dst page ids; the 35-token tail page crosses the half-page mark.
    # Staging pages pair with DST_BLOCKS one-to-one.
    SRC_BLOCKS = [1, 2, 4]
    DST_BLOCKS = [3, 5, 2]
    STAGING = [8, 6, 9]
    LIVE = [PAGE_SIZE, PAGE_SIZE, 35]

    def _transfer(self, dtype, src_tp, dst_tp, total_heads, layout, replica=0):
        """``replica`` offsets the prefill engine ids by whole attention-TP
        groups, as a prefill with DP attention (or several DP replicas)
        numbers them; the geometry uses ``engine_id % src_tp``."""
        src_heads = max(1, total_heads // src_tp)
        dst_heads = max(1, total_heads // dst_tp)
        prefill_t = _fused_layer(dtype, src_heads)
        decode_t = _fused_layer(dtype, dst_heads)
        sender_geom = _manager(prefill_t, src_tp, total_heads)._make_geometry(
            src_tp, dst_tp
        )
        receiver_geom = _manager(decode_t, dst_tp, total_heads)._make_geometry(
            src_tp, dst_tp
        )
        results = {}
        for decode_rank in range(dst_tp):
            senders = receiver_geom.sources_of(decode_rank)
            self.assertEqual(
                sender_geom.decode_rank_plan(decode_rank),
                receiver_geom.decode_rank_plan(decode_rank),
                "prefill and decode disagree on the writer plan",
            )
            engine = {rank: rank + replica * src_tp for rank in senders}
            receiver = rs.build_receiver_request(
                _Pb2,
                uuid=7,
                req_id="7",
                peer="decode:1",
                decode_rank=decode_rank,
                geom=receiver_geom,
                sources=[(engine[rank], self.SRC_BLOCKS) for rank in senders],
                dst_block_ids=self.DST_BLOCKS,
                live_tokens=self.LIVE,
                staging_block_ids=self.STAGING,
            )
            dest_pages = {
                b: np.full(
                    decode_t.stride(0) * decode_t.element_size(), PAD, dtype=np.uint8
                )
                for b in range(NUM_PAGES)
            }
            plans = sender_geom.decode_rank_plan(decode_rank)
            for rank in senders:
                staged = plans[engine[rank] % sender_geom.src_tp].staged
                sender = rs.build_sender_request(
                    _Pb2,
                    uuid=7,
                    req_id="7",
                    peer="decode:1",
                    prefill_rank=engine[rank],
                    decode_rank=decode_rank,
                    geom=sender_geom,
                    src_block_ids=self.SRC_BLOCKS,
                    # What the decode put in this source's RESHARD_REQ.
                    dst_block_ids=self.STAGING if staged else self.DST_BLOCKS,
                    live_tokens=self.LIVE,
                )
                sent = [vars(e) for e in sender.shard_push_schedules[0].entries]
                recv = [
                    vars(e) for e in receiver.shard_push_schedules[engine[rank]].entries
                ]
                for e in sent + recv:
                    e.pop("dst_peer", None)
                self.assertEqual(sent, recv, f"sender/receiver disagree, rank {rank}")
                source_pages = {
                    b: _page_image(
                        prefill_t, b, _heads_of_rank(src_tp, rank, total_heads), layout
                    )
                    for b in range(NUM_PAGES)
                }
                for entry in receiver.shard_push_schedules[engine[rank]].entries:
                    _apply(entry, source_pages, dest_pages)
            merges = receiver_geom.merge_spec(decode_rank)
            _apply_merge(
                dest_pages, decode_t, layout, self.DST_BLOCKS, self.STAGING, merges
            )
            results[decode_rank] = (dest_pages, decode_t)
        return results

    def _check(
        self,
        dtype,
        src_tp,
        dst_tp,
        total_heads=TOTAL_KV_HEADS,
        layout=PHYSICAL_LAYOUT,
        replica=0,
    ):
        results = self._transfer(dtype, src_tp, dst_tp, total_heads, layout, replica)
        for decode_rank, (dest_pages, decode_t) in results.items():
            heads = _heads_of_rank(dst_tp, decode_rank, total_heads)
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
                                f"{dtype} {src_tp}->{dst_tp} ({total_heads} heads) "
                                f"decode rank {decode_rank} page {dst_b} token "
                                f"{token} head {head} {'KV'[kv]}: got "
                                f"{got[token, slot, :4, 0].tolist()} want "
                                f"{want[token, slot, :4, 0].tolist()} "
                                f"({len(bad)} bad bytes)"
                            )
                for token in (0, 31, 32, 63):
                    if token < live:
                        np.testing.assert_array_equal(
                            got[token, : 2 * len(heads)], want[token, : 2 * len(heads)]
                        )

    # ---- byte simulation ---------------------------------------------------

    def test_fp8_fan_in_8_to_4(self):
        """GPT-OSS-120B fp8, prefill TP8 -> decode TP8 DP2 (attention TP4)."""
        self._check(FP8, 8, 4)

    def test_fp8_fan_in_8_to_2(self):
        self._check(FP8, 8, 2)

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
        self._check(FP8, 8, 8)

    def test_fp8_replicated_16_to_8(self):
        """Multihost TP16 prefill -> attention-TP8 decode (heads replicated)."""
        self._check(FP8, 16, 8)

    def test_bf16_all_shapes(self):
        for src_tp, dst_tp in ((8, 4), (8, 8), (8, 2), (16, 8)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                self._check(torch.bfloat16, src_tp, dst_tp)
        decode_t = _fused_layer(torch.bfloat16, 2)
        geometry = _manager(decode_t, 4)._make_geometry(8, 4)
        for decode_rank in range(4):
            plans = geometry.decode_rank_plan(decode_rank)
            self.assertFalse(any(p.staged for p in plans.values()))
            for rank, plan in plans.items():
                self.assertEqual(plan.entry, geometry.entry_bytes(rank, decode_rank))

    def test_general_head_counts(self):
        """Other kv-head counts and TP pairs, staged or not, both dtypes."""
        cases = [
            (16, 16, 8),  # fp8: 1 head -> 2 heads, staged
            (16, 16, 4),  # 1 -> 4 heads, two staged writers per staging row
            (16, 8, 4),  # 2 -> 4 heads: whole fp8 packs, direct
            (32, 16, 8),  # 2 -> 4
            (32, 32, 16),  # 1 -> 2
            (4, 8, 4),  # replicated prefill heads, 1 -> 1
            (4, 8, 2),  # replicated prefill heads, 1 -> 2: staged replicas
            (8, 8, 1),  # 1 -> 8 heads, four staged writers
            (64, 32, 8),  # 2 -> 8
        ]
        for total, src_tp, dst_tp in cases:
            for dtype in (FP8, torch.bfloat16):
                with self.subTest(
                    total=total, src_tp=src_tp, dst_tp=dst_tp, dtype=dtype
                ):
                    self._check(dtype, src_tp, dst_tp, total_heads=total)

    def test_prefill_dp_replica_engine_ids(self):
        """Fan-in from a prefill whose engine ids are offset by whole
        attention-TP groups (prefill DP attention / DP replicas): the geometry
        keys on ``engine_id % src_tp`` and places the same bytes."""
        for dtype in (FP8, torch.bfloat16):
            for replica in (1, 3):
                with self.subTest(dtype=dtype, replica=replica):
                    self._check(dtype, 8, 4, replica=replica)

    def test_fan_out_geometry_places_or_refuses(self):
        """Decode attention TP > prefill attention TP. At runtime Raiden maps
        each such decode rank to one prefill rank and pulls whole pages (see
        test_finer_decode_*), so this path is refused unless heads are
        replicated; the reshard geometry itself either places correctly or
        raises an actionable error -- never a silent misplacement."""
        cases = {
            (FP8, 4, 8): False,  # one staged head per decode rank
            (torch.bfloat16, 4, 8): False,
            (FP8, 2, 8): True,  # a 4-head row does not fit a 1-head staging row
            (torch.bfloat16, 2, 8): False,
        }
        for (dtype, src_tp, dst_tp), refused in cases.items():
            with self.subTest(dtype=dtype, src_tp=src_tp, dst_tp=dst_tp):
                if not refused:
                    self._check(dtype, src_tp, dst_tp)
                else:
                    with self.assertRaisesRegex(ValueError, "exceeds decode row"):
                        self._check(dtype, src_tp, dst_tp)
        geometry = rs.ReshardGeometry(
            src_tp=2,
            dst_tp=8,
            hp_bytes=256,
            total_kv_heads=8,
            kv_packing=4,
            element_size=1,
        )
        with self.assertRaisesRegex(ValueError, "exceeds decode row"):
            geometry.decode_rank_plan(1)

    def test_swa_receiver_registers_plan(self):
        """The decode registers its sliding-window receive plan with the SWA
        engine (tpu_sync is stubbed so this runs on CPU-only CI)."""
        mgr = _manager(_fused_layer(FP8, 2), 4)
        mgr._swa_kv_caches = [_fused_layer(FP8, 2)]
        mgr._engine = mock.Mock()
        mgr._swa_engine = mock.Mock(transfer_address="")
        mgr._decode = DecodeState()
        mgr._aborted = {}
        mgr._state_lock = threading.Lock()
        mgr._reshard_plans = {}
        rpc = SimpleNamespace(raiden_service_pb2=_Pb2)
        stub = {
            "tpu_sync": SimpleNamespace(rpc=rpc),
            "tpu_sync.rpc": rpc,
            "tpu_sync.rpc.raiden_service_pb2": _Pb2,
        }
        with mock.patch.dict(sys.modules, stub):
            mgr._register_reshard_swa_receiver(
                room=1, local_swa_blocks=[0], swa_sources_map={0: [0]}, src_attn_tp=8
            )
        mgr._swa_engine.register_active_plan.assert_called_once_with(
            1, mock.ANY, is_sender=False
        )

    def test_strided_copy_alone_cannot_place_shared_fp8_packs(self):
        """Without the staging merge the packed-word layout makes the old
        strided placement impossible: the strict entry builder refuses it."""
        decode_t = _fused_layer(FP8, 2)
        geometry = _manager(decode_t, 4)._make_geometry(8, 4)
        with self.assertRaisesRegex(ValueError, "splits a 4-slot pack"):
            geometry.check_decode_rank(0)
        plan = geometry.decode_rank_plan(0)
        self.assertFalse(plan[0].staged)
        self.assertTrue(plan[1].staged)
        self.assertEqual(plan[1].entry, (0, 512, 512, 0, 512))
        self.assertEqual(geometry.merge_spec(0), ((0, 2, 2),))

    # ---- geometry invariants -----------------------------------------------

    def test_geometry_requires_layout(self):
        with self.assertRaises(TypeError):
            rs.ReshardGeometry(src_tp=8, dst_tp=4, hp_bytes=512, total_kv_heads=8)

    def test_token_width_checked_against_strides(self):
        """Refuse a geometry whose token byte width differs from the tensor stride."""
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
        pack move it had before, and nothing is staged."""
        decode_t = _fused_layer(FP8, 1)
        geometry = _manager(decode_t, 8)._make_geometry(16, 8)
        for prefill_rank in (0, 1, 14, 15):
            plan = geometry.writer_plan(prefill_rank, prefill_rank // 2)
            self.assertFalse(plan.staged)
            self.assertEqual(plan.entry, (0, 512, 512, 0, 512))
            self.assertEqual(
                geometry.entry_bytes(prefill_rank, prefill_rank // 2), plan.entry
            )

    def test_receiver_needs_staging_pages_for_staged_sources(self):
        decode_t = _fused_layer(FP8, 2)
        geometry = _manager(decode_t, 4)._make_geometry(8, 4)
        with self.assertRaisesRegex(ValueError, "lands in staging pages"):
            rs.build_receiver_request(
                _Pb2,
                uuid=1,
                req_id="1",
                peer="p",
                decode_rank=0,
                geom=geometry,
                sources=[(0, [1]), (1, [1])],
                dst_block_ids=[2],
                live_tokens=[PAGE_SIZE],
            )


class TestTpuFinerDecodeRefusal(unittest.TestCase):
    """Whole-page pulls at finer decode attention TP cannot slice heads,
    so refuse them unless prefill and decode hold the same kv-heads."""

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


class TestTpuReshardMerge(unittest.TestCase):
    """The decode merge on real (CPU) fused pool tensors."""

    def _logical(self, buf):
        b, page, packs, packing, lanes = buf.shape
        return buf.view(b, page, packs * packing, lanes)

    def test_merge_on_logical_tensor(self):
        for heads, merges in ((2, ((0, 2, 2),)), (4, ((0, 2, 2), (4, 6, 2)))):
            with self.subTest(heads=heads):
                buf = _fused_layer(FP8, heads)
                torch.manual_seed(0)
                buf.copy_(torch.randn(buf.shape).to(FP8))
                before = buf.clone()
                real, stage = [3, 5], [7, 1]
                merge_staged_pages([buf], real, stage, merges)
                logical, old = self._logical(buf), self._logical(before)
                expected = old.clone()
                for r, s in zip(real, stage):
                    for stage_slot, decode_slot, n in merges:
                        expected[r, :, decode_slot : decode_slot + n] = old[
                            s, :, stage_slot : stage_slot + n
                        ]
                self.assertTrue(
                    torch.equal(logical.view(torch.uint8), expected.view(torch.uint8))
                )

    def test_bucket_pads_with_dummy_page(self):
        self.assertEqual(
            bucket_pages([3, 4, 5], [6, 7, 8]), ([3, 4, 5, 0], [6, 7, 8, 0])
        )
        self.assertEqual(bucket_pages([3], [6]), ([3], [6]))
        with self.assertRaises(ValueError):
            bucket_pages([1, 2], [3])

    def test_merge_writes_back_through_the_pool_page_write(self):
        """The merged pages go back through kv_page_write.write_pages_torch
        (the in-place aliased page write load_cpu_copy uses), once per layer,
        into the live buffer's storage -- never an eager in-place op on it."""
        import sglang_tpu.kv_page_write as kv_page_write

        bufs = [_fused_layer(FP8, 2), _fused_layer(FP8, 2)]
        calls = []
        real_write = kv_page_write.write_pages_torch

        def recorder(block_ids, pages, kv):
            calls.append((block_ids.tolist(), kv.untyped_storage().data_ptr()))
            real_write(block_ids, pages, kv)

        with mock.patch.object(kv_page_write, "write_pages_torch", recorder):
            merge_staged_pages(bufs, [3, 5, 2], [7, 1, 8], ((0, 2, 2),))
        self.assertEqual(
            calls,
            [([3, 5, 2, 0], b.untyped_storage().data_ptr()) for b in bufs],
        )


class _FakeAllocator:
    """Paged token allocator stand-in that records every alloc and free."""

    def __init__(self, pages):
        self.free_pages = list(pages)
        self.live = {}
        self.allocs = 0
        self.frees = 0

    def alloc(self, need_size):
        n = need_size // PAGE_SIZE
        if n > len(self.free_pages):
            return None
        pages, self.free_pages = self.free_pages[:n], self.free_pages[n:]
        idx = torch.cat(
            [torch.arange(p * PAGE_SIZE, (p + 1) * PAGE_SIZE) for p in pages]
        )
        self.live[id(idx)] = pages
        self.allocs += 1
        return idx

    def free(self, idx):
        pages = self.live.pop(id(idx))
        self.free_pages.extend(pages)
        self.frees += 1


class TestTpuReshardStagingLifecycle(unittest.TestCase):
    """Staging pages: allocated once per staged room, handed to the staged
    sources, merged once, returned to the allocator exactly once."""

    def _decode_manager(self, allocator, src_tp=8, dst_tp=4):
        decode_t = _fused_layer(FP8, max(1, TOTAL_KV_HEADS // dst_tp), num_pages=16)
        mgr = _manager(decode_t, dst_tp)
        mgr.kv_args.engine_rank = 0
        mgr._engine = mock.Mock(
            transfer_address="",
            poll_stats_with_details=lambda: SimpleNamespace(
                sent=[], received=[], failed=[]
            ),
        )
        mgr._swa_engine = None
        mgr._state_engine = None
        mgr._engine_poll_failing_since = {}
        mgr._pipeline = None
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr._failed, mgr._native_failed = set(), set()
        mgr.chunked_transfer = False
        mgr._completed_send, mgr._completed_recv = set(), set()
        mgr._send_done_at = {}
        mgr._pulling = None
        mgr._cleared_plans = {}
        mgr._reshard_plans = {}
        mgr._stale_plans = []
        mgr._aborted = {}
        mgr.waiting_timeout = 300
        mgr.local_ip, mgr.rank_port = "127.0.0.1", 1
        mgr.sent = []
        mgr.addr_to_rooms_tracker = collections.defaultdict(set)
        mgr.defer_decode_allocation = False
        mgr._dynamic_host_staging = True
        mgr.kv_args.state_types = []
        mgr.request_status = {}
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._staging_outstanding = {}
        mgr._ensure_engine = lambda: SimpleNamespace(transfer_address="decode:1")
        mgr._send_multipart_locked = lambda ep, frames: mgr.sent.append((ep, frames))
        mgr._is_room_transfer = lambda key, room: str(key).split("#")[0] == str(room)
        if allocator is not None:
            mgr.attach_token_allocator(allocator)
        return mgr, decode_t

    @staticmethod
    def _infos(n):
        return [{"rank_ip": "10.0.0.1", "rank_port": 100 + i} for i in range(n)]

    def _register(self, mgr, room, blocks):
        mgr._register_reshard_request(room, blocks, [], None, self._infos(2))

    def test_staged_room_allocates_sends_merges_and_frees_once(self):
        from sglang_tpu.disaggregation.raiden.serde import ReshardReq

        allocator = _FakeAllocator(range(10, 16))
        mgr, decode_t = self._decode_manager(allocator)
        self._register(mgr, 42, [3, 4, 5])
        self.assertEqual(allocator.allocs, 1)
        staging = mgr._decode.reshard_staging[42]
        self.assertEqual(len(staging.pages), 3)
        self.assertFalse(set(staging.pages) & {3, 4, 5})
        reqs = [ReshardReq.from_frames(frames) for _, frames in mgr.sent]
        self.assertEqual(
            [list(r.decode_blocks) for r in reqs], [[3, 4, 5], staging.pages]
        )
        # The merge runs once.
        torch.manual_seed(2)
        decode_t.copy_(torch.randn(decode_t.shape).to(FP8))
        before = decode_t.clone()
        mgr.merge_reshard_staging(42)
        after_first = decode_t.clone()
        self.assertFalse(
            torch.equal(before.view(torch.uint8), after_first.view(torch.uint8))
        )
        mgr.merge_reshard_staging(42)
        self.assertTrue(
            torch.equal(after_first.view(torch.uint8), decode_t.view(torch.uint8))
        )
        # Freed exactly once, even if the room is cleared twice.
        mgr.clear_room(42)
        mgr.clear_room(42)
        self.assertEqual((allocator.allocs, allocator.frees), (1, 1))
        self.assertEqual(sorted(allocator.free_pages), list(range(10, 16)))

    def test_failed_rooms_and_aborts_do_not_leak(self):
        allocator = _FakeAllocator(range(10, 16))
        mgr, _ = self._decode_manager(allocator)
        # Room cleared (abort / timeout / failure all end in clear_room)
        # before its transfer ever completes: no merge, pages returned.
        self._register(mgr, 1, [3, 4])
        self._register(mgr, 2, [5, 6])
        self.assertEqual(len(allocator.free_pages), 2)
        # Not enough pages for a third room: it fails with the reason and
        # allocates nothing.
        self._register(mgr, 3, [7, 8, 9])
        self.assertIn("3", mgr._failed)
        self.assertIn("free KV pages", mgr.failure_records[3])
        self.assertNotIn(3, mgr._decode.reshard_staging)
        for room in (1, 2, 3):
            mgr.clear_room(room)
        self.assertEqual((allocator.allocs, allocator.frees), (2, 2))
        self.assertEqual(sorted(allocator.free_pages), list(range(10, 16)))

    def test_without_allocator_staged_reshard_is_refused(self):
        mgr, _ = self._decode_manager(None)
        self._register(mgr, 5, [3])
        self.assertIn("5", mgr._failed)
        self.assertIn("Partial FP8 head packs are unsupported", mgr.failure_records[5])
        self.assertEqual(mgr.sent, [])

    def test_unstaged_room_allocates_nothing(self):
        allocator = _FakeAllocator(range(10, 16))
        mgr, _ = self._decode_manager(allocator, src_tp=16, dst_tp=8)
        self._register(mgr, 7, [3, 4])
        self.assertEqual(allocator.allocs, 0)
        self.assertNotIn(7, mgr._decode.reshard_staging)
        mgr.merge_reshard_staging(7)  # no-op
        mgr.clear_room(7)
        self.assertEqual(allocator.frees, 0)

    def _hybrid_swa_allocator(self, full_pages, swa_pages):
        """The real SWATokenToKVPoolAllocator class (its alloc() asserts
        page_size == 1) over real paged sub-allocators, as on a paged
        GPT-OSS decode; only its constructor's pool wiring is skipped."""
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

        def paged(num_pages):
            return PagedTokenToKVPoolAllocator(
                size=num_pages * PAGE_SIZE,
                page_size=PAGE_SIZE,
                dtype=torch.int64,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )

        allocator = object.__new__(SWATokenToKVPoolAllocator)
        allocator.page_size = PAGE_SIZE
        allocator.full_attn_allocator = paged(full_pages)
        allocator.swa_attn_allocator = paged(swa_pages)
        with self.assertRaises(AssertionError):
            allocator.alloc(PAGE_SIZE)  # the F1 failure, j-a0bd1397
        return allocator

    def test_paged_hybrid_swa_allocator_stages_both_pools(self):
        """GPT-OSS decodes run the hybrid SWA allocator, whose alloc() only
        serves page_size 1 (F1 j-a0bd1397: ``assert self.page_size == 1``).
        Staging takes plain pages from its full and SWA sub-allocators, sends
        both to the staged source, merges both pools and frees both."""
        from sglang_tpu.disaggregation.raiden.serde import ReshardReq

        allocator = self._hybrid_swa_allocator(15, 15)
        full, swa = allocator.full_attn_allocator, allocator.swa_attn_allocator
        full_free, swa_free = full.available_size(), swa.available_size()
        mgr, decode_t = self._decode_manager(allocator)
        swa_t = _fused_layer(FP8, 2, num_pages=16)
        mgr._swa_kv_caches = [swa_t]
        mgr._register_reshard_request(42, [3, 4, 5], [6, 7], None, self._infos(2))
        staging = mgr._decode.reshard_staging[42]
        self.assertEqual((len(staging.pages), len(staging.swa_pages)), (3, 2))
        self.assertEqual(full.available_size(), full_free - 3 * PAGE_SIZE)
        self.assertEqual(swa.available_size(), swa_free - 2 * PAGE_SIZE)
        self.assertTrue(all(0 < p < 16 for p in staging.pages + staging.swa_pages))
        reqs = [ReshardReq.from_frames(frames) for _, frames in mgr.sent]
        self.assertEqual(
            [list(r.decode_swa_blocks) for r in reqs], [[6, 7], staging.swa_pages]
        )
        torch.manual_seed(3)
        swa_t.copy_(torch.randn(swa_t.shape).to(FP8))
        before = swa_t.clone().view(torch.uint8)
        mgr.merge_reshard_staging(42)
        logical = swa_t.view(torch.uint8).view(16, PAGE_SIZE, 4, 128)
        old = before.view(16, PAGE_SIZE, 4, 128)
        for real, stage in zip([6, 7], staging.swa_pages):
            self.assertTrue(torch.equal(logical[real, :, 2:4], old[stage, :, 0:2]))
            self.assertTrue(torch.equal(logical[real, :, 0:2], old[real, :, 0:2]))
        mgr.clear_room(42)
        mgr.clear_room(42)
        self.assertEqual(full.available_size(), full_free)
        self.assertEqual(swa.available_size(), swa_free)

    def test_paged_hybrid_swa_allocator_rolls_back_partial_allocation(self):
        allocator = self._hybrid_swa_allocator(15, 1)
        full, swa = allocator.full_attn_allocator, allocator.swa_attn_allocator
        full_free, swa_free = full.available_size(), swa.available_size()
        mgr, _ = self._decode_manager(allocator)
        mgr._swa_kv_caches = [_fused_layer(FP8, 2, num_pages=16)]
        from sglang.srt.disaggregation.base.conn import StateType

        mgr.kv_args.state_types = [StateType.SWA]

        req = RaidenKVReceiver(mgr, "10.0.0.1", 9)
        req.bootstrap_infos = self._infos(2)
        req._send_abort_notification = lambda: setattr(req, "abort_notified", True)
        req.abort_notified = False
        req.send_metadata(np.array([3, 4], dtype=np.int32), state_indices=[[6, 7]])
        self.assertIn("free sliding-window KV pages", mgr.failure_records[9])
        self.assertEqual(req.poll(), KVPoll.Failed)
        self.assertTrue(req.abort_notified)
        self.assertNotIn(9, mgr._decode.reshard_staging)
        self.assertEqual(full.available_size(), full_free)
        self.assertEqual(swa.available_size(), swa_free)

    def test_staging_allocation_failure_aborts_cleanly(self):
        allocator = _FakeAllocator(range(10, 14))
        mgr, _ = self._decode_manager(allocator)

        req = RaidenKVReceiver(mgr, "10.0.0.1", 42)
        req.bootstrap_infos = self._infos(2)

        req._send_abort_notification = lambda: setattr(req, "abort_notified", True)
        req.abort_notified = False
        req.send_metadata(np.array([1, 2, 3], dtype=np.int32), state_indices=[])
        self.assertEqual(allocator.allocs, 1)
        self.assertEqual(req.poll(), KVPoll.WaitingForInput)

        req2 = RaidenKVReceiver(mgr, "10.0.0.1", 43)
        req2.bootstrap_infos = self._infos(2)
        req2._send_abort_notification = lambda: setattr(req2, "abort_notified", True)
        req2.abort_notified = False
        req2.send_metadata(np.array([4, 5, 6], dtype=np.int32), state_indices=[])
        self.assertEqual(allocator.allocs, 1)
        self.assertEqual(req2.poll(), KVPoll.Failed)
        self.assertTrue(req2.abort_notified)
        mgr.clear_room(43)
        self.assertEqual(allocator.frees, 0)
        mgr.clear_room(42)
        self.assertEqual(allocator.frees, 1)
        req3 = RaidenKVReceiver(mgr, "10.0.0.1", 44)
        req3.bootstrap_infos = self._infos(2)
        req3.send_metadata(np.array([7, 8, 9], dtype=np.int32), state_indices=[])
        self.assertEqual(req3.poll(), KVPoll.WaitingForInput)
        self.assertEqual(allocator.allocs, 2)

    def test_staging_missing_swa_pool_aborts_cleanly(self):
        allocator = _FakeAllocator(range(10, 20))
        mgr, _ = self._decode_manager(allocator)
        mgr._swa_kv_caches = [_fused_layer(FP8, 2, num_pages=16)]
        from sglang.srt.disaggregation.base.conn import StateType

        mgr.kv_args.state_types = [StateType.SWA]

        req = RaidenKVReceiver(mgr, "10.0.0.1", 42)
        req.bootstrap_infos = self._infos(2)
        req._send_abort_notification = lambda: setattr(req, "abort_notified", True)
        req.abort_notified = False

        req.send_metadata(np.array([1, 2, 3], dtype=np.int32), state_indices=[[4, 5]])
        self.assertEqual(allocator.allocs, 1)
        self.assertEqual(allocator.frees, 1)
        self.assertEqual(req.poll(), KVPoll.Failed)
        self.assertTrue(req.abort_notified)
        self.assertIn("42", mgr._failed)
        self.assertIn("has no sliding-window pool", mgr.failure_records[42])


class TestTpuUnifiedSwaReshardStaging(unittest.TestCase):
    PAGE_SIZE = 16
    INFOS = (
        {"rank_ip": "10.0.0.1", "rank_port": 20001},
        {"rank_ip": "10.0.0.1", "rank_port": 20002},
    )

    def _build_unified_pool_and_allocator(self, *, num_phys_pages: int):
        # 1 full layer per 6 SWA layers gives num_swa_groups == 6.
        pool = TpuUnifiedSWAKVPool(
            size=num_phys_pages * self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=FP8,
            full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=[5]),
            swa=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=[0, 1, 2, 3, 4, 6]),
            device="cpu",
        )
        self.assertEqual(pool.layout.num_swa_groups, 6)
        allocator = TpuUnifiedSWATokenToKVPoolAllocator(
            size=num_phys_pages * self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=FP8,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        return pool, allocator

    def _build_decode_mgr(
        self,
        *,
        pool: TpuUnifiedSWAKVPool,
        allocator: TpuUnifiedSWATokenToKVPoolAllocator,
        chunked: bool = True,
    ) -> RaidenKVManager:
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr.attn_tp_size = 4
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr._kv_caches = pool.kv_pool.page_buffers()
        mgr._swa_kv_caches = []
        mgr._unified_swa_pool = pool
        mgr.kv_layout = UNIFIED_SWA_KV_LAYOUT
        mgr._seq_along_lane = False
        mgr._bounded_staging = False
        mgr._dynamic_host_staging = True
        mgr._engine = mock.Mock(
            transfer_address="decode:1",
            poll_stats_with_details=lambda: SimpleNamespace(
                sent=[], received=[], failed=[]
            ),
        )
        mgr._swa_engine = None
        mgr._state_engine = None
        mgr._engine_poll_failing_since = {}
        mgr._pipeline = None
        mgr._pipeline_enabled = False
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr._failed, mgr._native_failed = set(), set()
        mgr.chunked_transfer = chunked
        mgr._completed_send, mgr._completed_recv = set(), set()
        mgr._send_done_at = {}
        mgr._pulling = None
        mgr._cleared_plans = {}
        mgr._reshard_plans = {}
        mgr._stale_plans = []
        mgr._aborted = {}
        mgr.waiting_timeout = 300
        mgr.local_ip, mgr.rank_port = "127.0.0.1", 31001
        mgr.sent = []
        mgr.addr_to_rooms_tracker = collections.defaultdict(set)
        mgr.defer_decode_allocation = False
        mgr.request_status = {}
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._staging_outstanding = {}
        mgr.dram_offload_channel = None
        mgr._ensure_engine = lambda: mgr._engine
        mgr._send_multipart_locked = lambda ep, frames: mgr.sent.append((ep, frames))
        mgr._is_room_transfer = lambda key, room: str(key).split("#")[0] == str(room)
        mgr.kv_args = SimpleNamespace(
            total_kv_head_num=8,
            page_size=self.PAGE_SIZE,
            engine_rank=0,
            state_types=[StateType.SWA],
        )
        mgr.attach_token_allocator(allocator)
        return mgr

    def _send_unified_metadata(
        self, *, mgr: RaidenKVManager, room: int, f_page: int, s_page: int
    ) -> RaidenKVReceiver:
        recv = RaidenKVReceiver(
            mgr=mgr, bootstrap_addr="10.0.0.1:8998", bootstrap_room=room
        )
        recv.prefill_info = SimpleNamespace(attn_tp_size=8, pp_size=1)
        recv.bootstrap_infos = list(self.INFOS)
        recv._send_abort_notification = lambda: setattr(recv, "abort_notified", True)
        recv.send_metadata(
            kv_indices=np.array([f_page], dtype=np.int32), state_indices=[[s_page]]
        )
        return recv

    def test_unified_swa_staging_allocates_exact_physical_pages(self):
        """Unified SWA staging must allocate one physical page per expanded full and SWA page,
        never multiplying expanded SWA pages by num_swa_groups again."""
        pool, alloc = self._build_unified_pool_and_allocator(num_phys_pages=64)
        mgr = self._build_decode_mgr(pool=pool, allocator=alloc)
        f_page = (
            int(alloc.full_attn_allocator.alloc(self.PAGE_SIZE)[0].item())
            // self.PAGE_SIZE
        )
        s_page = (
            int(alloc.swa_attn_allocator.alloc(self.PAGE_SIZE)[0].item())
            // self.PAGE_SIZE
        )
        free_before = alloc._free_page_count()

        self._send_unified_metadata(mgr=mgr, room=51, f_page=f_page, s_page=s_page)
        taken = free_before - alloc._free_page_count()
        self.assertEqual(taken, 7, f"Expected 1 + 6 = 7 physical pages, got {taken}")

    def test_unified_swa_staging_fits_tight_shared_pool(self):
        """Unified SWA staging must succeed whenever enough shared physical pages remain,
        instead of failing the room with spurious out-of-pages."""
        pool, alloc = self._build_unified_pool_and_allocator(num_phys_pages=24)
        mgr = self._build_decode_mgr(pool=pool, allocator=alloc)
        f_page = (
            int(alloc.full_attn_allocator.alloc(self.PAGE_SIZE)[0].item())
            // self.PAGE_SIZE
        )
        s_page = (
            int(alloc.swa_attn_allocator.alloc(self.PAGE_SIZE)[0].item())
            // self.PAGE_SIZE
        )
        self.assertEqual(alloc._free_page_count(), 17)

        recv = self._send_unified_metadata(
            mgr=mgr, room=52, f_page=f_page, s_page=s_page
        )
        self.assertNotEqual(
            recv.poll(),
            KVPoll.Failed,
            f"Spurious OOM with 17 free physical pages: {mgr.failure_records.get(52)}",
        )

    def test_unified_swa_staging_merges_swa_pages(self):
        """Unified SWA staging must merge staged SWA slots in the shared pool buffer,
        and include staged SWA pages on non-chunked receive plans."""
        pool, alloc = self._build_unified_pool_and_allocator(num_phys_pages=64)
        mgr = self._build_decode_mgr(pool=pool, allocator=alloc, chunked=False)
        f_page = (
            int(alloc.full_attn_allocator.alloc(self.PAGE_SIZE)[0].item())
            // self.PAGE_SIZE
        )
        s_page = (
            int(alloc.swa_attn_allocator.alloc(self.PAGE_SIZE)[0].item())
            // self.PAGE_SIZE
        )
        self._send_unified_metadata(mgr=mgr, room=53, f_page=f_page, s_page=s_page)

        mgr._decode.reshard_sources[53] = {0: [1], 1: [1]}
        mgr._decode.reshard_swa_sources[53] = {0: [2] * 6, 1: [2] * 6}
        mgr._decode.reshard_meta[53]["ready_endpoints"] = []
        rpc = SimpleNamespace(raiden_service_pb2=_Pb2)
        stub = {
            "tpu_sync": SimpleNamespace(rpc=rpc),
            "tpu_sync.rpc": rpc,
            "tpu_sync.rpc.raiden_service_pb2": _Pb2,
        }
        with mock.patch.dict(sys.modules, stub):
            mgr._register_receiver_and_signal(53)
        self.assertEqual(mgr.failure_records, {})
        mgr._engine.register_active_plan.assert_called_once()

        buf = mgr._kv_caches[0]
        torch.manual_seed(7)
        buf.copy_(torch.randn(buf.shape).to(FP8))
        before = buf.clone().view(torch.uint8)
        mgr.merge_reshard_staging(53)
        after = buf.view(torch.uint8)

        staging = mgr._decode.reshard_staging[53]
        local_swa = mgr._decode.reshard_meta[53]["local_swa_blocks"]
        for real_p, stage_p in zip(local_swa, staging.swa_pages):
            self.assertTrue(
                torch.equal(after[real_p, :, 0, 2:4], before[stage_p, :, 0, 0:2]),
                f"Unified SWA page {real_p} was not merged from staging page {stage_p}",
            )


if __name__ == "__main__":
    unittest.main()
