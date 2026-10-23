"""Raiden reshard landing (#829): a misaligned fp8 source lands in a scratch
block that the decode merges into the final page once its plan settles.

The decode side runs the real receiver, drain, and pre-pass over CPU fp8
tensors; the push is replayed from the registered receive schedule byte for
byte, as tpu-sync's receiver would write it.
"""

import functools
import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import land as land_mod
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.land import (
    LAND_VERSION,
    LandState,
    ScratchArena,
    merge_bucket,
)
from sglang_tpu.disaggregation.raiden.reshard_schedule import source_ranks_for
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    ReshardBlocks,
    ReshardChunk,
    ReshardChunkReady,
    ReshardLandChunkReady,
    ReshardLandReady,
    ReshardReady,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

ROOM = 41
PAGE = 128
FP8 = torch.float8_e4m3fn
HEAD_DIM = 128
# Decode pool: block 0 pads, 1..16 belong to the allocator, the rest are the
# arena.
ALLOC_BLOCKS = 16
FINAL = [1, 2, 3, 4]
PREFILL_BLOCKS = [4, 5, 6, 7]
PAD, GARBAGE = 0xFF, 0xFD


# ---- fakes -----------------------------------------------------------------


class _Entries(list):
    def add(self):
        self.append(SimpleNamespace())
        return self[-1]


class _Schedules(dict):
    def __missing__(self, key):
        self[key] = SimpleNamespace(entries=_Entries())
        return self[key]


class _Pb2:
    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return SimpleNamespace(shard_push_schedules=_Schedules())


def _pb2_modules():
    rpc = types.ModuleType("tpu_sync.rpc")
    rpc.raiden_service_pb2 = _Pb2
    pkg = types.ModuleType("tpu_sync")
    pkg.rpc = rpc
    return {
        "tpu_sync": pkg,
        "tpu_sync.rpc": rpc,
        "tpu_sync.rpc.raiden_service_pb2": _Pb2,
    }


def _settled(ids):
    return [
        SimpleNamespace(
            req_id=r,
            completed_ns=None,
            elapsed_ns=None,
            peer=None,
            code=None,
            reason=None,
            phase=None,
        )
        for r in ids
    ]


def _polled(received=(), failed=()):
    return SimpleNamespace(
        sent=[], received=_settled(received), failed=_settled(failed)
    )


class _InlineExecutor:
    def submit(self, fn, *args):
        fn(*args)


class _ReadyFuture:
    def is_ready(self):
        return True

    def ok(self):
        return True


# ---- tensors ---------------------------------------------------------------


def _heads_of(rank, tp, total):
    if total >= tp:
        return [rank * (total // tp) + i for i in range(total // tp)]
    return [rank // (tp // total)]


def _label(g, kv, layer):
    """uint8 [page, head_dim] bytes of global head ``g``'s K (0) or V (1)."""
    t = torch.arange(PAGE).view(-1, 1)
    c = torch.arange(HEAD_DIM).view(1, -1)
    return (((g * 2 + kv) * 37 + layer * 11 + t * 7 + c * 3) % 251 + 1).to(torch.uint8)


def _pool(blocks, heads):
    rows = -(-2 * heads // 4)
    t = torch.empty((blocks, PAGE, rows, 4, HEAD_DIM), dtype=FP8)
    t.view(torch.uint8).fill_(GARBAGE)
    return t


def _fill_page(kv, block, heads, layer, pad=True):
    """Write ``heads`` (global ids) into page ``block``; slots past them pad."""
    u = kv.view(torch.uint8)[block].reshape(PAGE, -1, HEAD_DIM)
    if pad:
        u.fill_(PAD)
    for local, g in enumerate(heads):
        for kv_ in (0, 1):
            u[:, 2 * local + kv_, :] = _label(g, kv_, layer)


def _expected_page(heads, layer):
    rows = -(-2 * len(heads) // 4)
    out = torch.full((PAGE, rows * 4, HEAD_DIM), PAD, dtype=torch.uint8)
    for local, g in enumerate(heads):
        for kv_ in (0, 1):
            out[:, 2 * local + kv_, :] = _label(g, kv_, layer)
    return out


def _replay(rreq, sources, decode_kv):
    """Write every receive entry's bytes, as tpu-sync's receiver would."""
    for p_rank, schedule in rreq.shard_push_schedules.items():
        for layer, dst in enumerate(decode_kv):
            src = sources[p_rank][layer].view(torch.uint8)
            d = dst.view(torch.uint8)
            for e in schedule.entries:
                n = e.count
                idx_s = (
                    e.src_offset_bytes
                    + e.src_stride_bytes * torch.arange(n).view(-1, 1)
                    + torch.arange(e.size_bytes)
                ).reshape(-1)
                idx_d = (
                    e.dst_offset_bytes
                    + e.dst_stride_bytes * torch.arange(n).view(-1, 1)
                    + torch.arange(e.size_bytes)
                ).reshape(-1)
                d[e.dst_block_id].reshape(-1)[idx_d] = src[e.src_block_id].reshape(-1)[
                    idx_s
                ]


# ---- managers --------------------------------------------------------------


def _decode(
    total_heads=8,
    src_tp=8,
    dst_tp=4,
    rank=0,
    cap=8,
    pages=4,
    final=FINAL,
    alloc=ALLOC_BLOCKS,
    src_blocks=PREFILL_BLOCKS,
):
    """A decode RaidenKVManager at attention-TP ``rank`` with real fp8 KV
    tensors whose last ``cap`` blocks are the landing arena; its room's
    final pages are ``final[:pages]``, its sources' pages ``src_blocks``."""
    heads = max(1, total_heads // dst_tp)
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr.disaggregation_mode = DisaggregationMode.DECODE
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.chunked_transfer = True
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._engine = Mock(transfer_address="10.0.0.2:31000")
    mgr._swa_engine = None
    mgr._ensure_engine = lambda: mgr._engine
    mgr.kv_args = SimpleNamespace(
        page_size=PAGE, total_kv_head_num=total_heads, total_swa_kv_head_num=0
    )
    mgr._kv_caches = [_pool(1 + alloc + cap, heads) for _ in range(2)]
    mgr._kv_head_nums = (heads,)
    mgr._swa_kv_caches = []
    mgr._seq_along_lane = False
    mgr._land = LandState(ScratchArena(1 + alloc, cap))
    mgr.waiting_timeout = 300
    mgr.attn_tp_rank = rank
    mgr.attn_tp_size = dst_tp
    mgr._local_rank = rank
    mgr._uuid_seq = 0
    mgr._check_reshard_bounds = lambda *args, **kwargs: None
    srcs = source_ranks_for(rank, src_tp // dst_tp)
    mgr._decode.reshard_meta = {
        ROOM: dict(
            local_blocks=list(final[:pages]),
            local_swa_blocks=[],
            source_ranks=srcs,
            src_attn_tp=src_tp,
            aux_index=None,
            ready_endpoints=[f"tcp://10.0.0.{10 + p}:1" for p in srcs],
        )
    }
    mgr._decode.reshard_sources = {ROOM: {}}
    mgr._decode.reshard_swa_sources = {ROOM: {}}
    mgr._land.started[ROOM] = time.time()
    mgr._reshard_plans = {}
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr._pulling = None
    mgr._aborted = {}
    mgr._send_reads = {}
    mgr._pulling_cleared = set()
    mgr._decode.announced_at = {}
    mgr._timeout_s = 120.0
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._superseded_sender_reads = set()
    mgr._send_multipart_locked = Mock()
    mgr._reshard_slot_of = {}
    mgr._reshard_slot_rooms = [None]
    mgr._engine.poll_stats_with_details = Mock(return_value=_polled())
    # The sources' prefill tensors, filled with their own heads.
    mgr.test_sources = {}
    for p in srcs:
        layers = []
        for layer in range(2):
            t = _pool(max(src_blocks) + 1, 1)
            for b in src_blocks:
                _fill_page(t, b, _heads_of(p, src_tp, total_heads), layer)
            layers.append(t)
        mgr.test_sources[p] = layers
    mgr.test_heads = _heads_of(rank, dst_tp, total_heads)
    return mgr


def _sent(mgr, cls):
    """Distinct ``cls`` messages sent (each goes to every source endpoint)."""
    out = []
    for c in mgr._send_multipart_locked.call_args_list:
        if c.args[1][0] == cls.TAG:
            msg = cls.from_frames(c.args[1])
            if msg not in out:
                out.append(msg)
    return out


def _registered(mgr):
    """{req_id: StartTransferRequest} of every receive plan registered."""
    return {
        c.args[1].req_id: c.args[1]
        for c in mgr._engine.register_active_plan.call_args_list
    }


def _chunk(mgr, index, start, length, total=4):
    with patch.dict(sys.modules, _pb2_modules()):
        for p in mgr._decode.reshard_meta[ROOM]["source_ranks"]:
            mgr._on_reshard_chunk(
                ReshardChunk(
                    ROOM, index, start, total, PREFILL_BLOCKS[start : start + length], p
                ).to_frames()
            )


def _closing(mgr):
    with patch.dict(sys.modules, _pb2_modules()):
        for p in mgr._decode.reshard_meta[ROOM]["source_ranks"]:
            mgr._on_reshard_blocks(
                ReshardBlocks(ROOM, list(PREFILL_BLOCKS), p, b"", []).to_frames()
            )


def _settle(mgr, received=(), failed=()):
    mgr._engine.poll_stats_with_details = Mock(return_value=_polled(received, failed))


def _prefill(rank=1, total_heads=8, src_tp=8, dst_tp=4, announce_pages=16):
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = PrefillState(announce_pages=announce_pages)
    mgr._decode = None
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr._aborted = {}
    mgr._send_reads = {}
    mgr._superseded_sender_reads = set()
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.request_status = {}
    mgr.chunked_transfer = True
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._reshard_push_lock = threading.Lock()
    mgr._transfer_parallelism = 4
    mgr._engine = Mock()
    mgr._engine.d2h = Mock(return_value=_ReadyFuture())
    mgr._engine.plan_host_blocks = Mock(side_effect=lambda uuid, blocks: blocks)
    mgr._ensure_engine = lambda: mgr._engine
    mgr._swa_engine = None
    mgr.kv_args = SimpleNamespace(
        page_size=PAGE, total_kv_head_num=total_heads, total_swa_kv_head_num=0
    )
    mgr._kv_caches = [_pool(8, max(1, total_heads // src_tp)) for _ in range(2)]
    mgr._kv_head_nums = (max(1, total_heads // src_tp),)
    mgr._swa_kv_caches = []
    mgr._seq_along_lane = False
    mgr.attn_tp_rank = rank
    mgr.attn_tp_size = src_tp
    mgr._local_rank = rank
    mgr._timeout_s = 5
    mgr._push_executor = _InlineExecutor()
    mgr._check_reshard_bounds = lambda *args, **kwargs: None
    mgr._read_leases = {}
    mgr._uuid_seq = 0
    mgr._reshard_pushes_pending = {}
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._pulling = None
    mgr._pulling_cleared = set()
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr._send_multipart_locked = Mock()
    mgr._reshard_slot_of = {}
    mgr._reshard_slot_rooms = [None]
    mgr._prefill.reshard_req[ROOM] = dict(
        reply_addr=("10.0.0.2", 5555),
        decode_blocks=list(FINAL),
        dst_attn_tp=dst_tp,
        push_endpoint="10.0.0.2:31000",
    )
    return mgr


# ---- decode lifecycle ------------------------------------------------------


class TestDecodeLandsAndMerges(unittest.TestCase):
    """The receive plan, the replayed push, the settle and the merge leave
    every final page exactly the decode rank's heads."""

    def _check_pages(self, mgr, pages):
        for layer, kv in enumerate(mgr._kv_caches):
            want = _expected_page(mgr.test_heads, layer)
            for b in pages:
                got = kv.view(torch.uint8)[b].reshape(want.shape)
                self.assertTrue(torch.equal(got, want), f"layer {layer} block {b}")

    def _run(self, total_heads, dst_tp, rank, chunked):
        mgr = _decode(total_heads=total_heads, dst_tp=dst_tp, rank=rank)
        if chunked:
            _chunk(mgr, 0, 0, 2)
            (cr,) = _sent(mgr, ReshardLandChunkReady)
            self.assertEqual((cr.index, cr.land_version), (0, LAND_VERSION))
            self.assertEqual(len(cr.scratch), 2)
            self.assertEqual(_sent(mgr, ReshardChunkReady), [])
        _closing(mgr)
        (ready,) = _sent(mgr, ReshardLandReady)
        self.assertEqual(ready.pages, [2, 3] if chunked else [0, 1, 2, 3])
        self.assertEqual(len(ready.scratch), len(ready.pages))
        self.assertEqual(_sent(mgr, ReshardReady), [])
        plans = _registered(mgr)
        for req in plans.values():
            _replay(req, mgr.test_sources, mgr._kv_caches)
        # Scratch is held until the merge; Success waits for it.
        self.assertFalse(mgr.land_merged(ROOM))
        self.assertGreater(mgr._land.arena.held, 0)
        _settle(mgr, received=list(plans))
        mgr.land_prepass()
        self._check_pages(mgr, FINAL)
        self.assertTrue(mgr.land_merged(ROOM))
        self.assertEqual(mgr._land.arena.held, 0)
        self.assertFalse(mgr.is_failed(ROOM))
        return mgr

    def test_8kv_tp8_to_tp4(self):
        for chunked in (False, True):
            for rank in (0, 3):
                with self.subTest(chunked=chunked, rank=rank):
                    self._run(8, 4, rank, chunked)

    def test_4kv_tp8_to_tp2(self):
        for chunked in (False, True):
            for rank in (0, 1):
                with self.subTest(chunked=chunked, rank=rank):
                    self._run(4, 2, rank, chunked)

    def test_unmerged_replay_is_wrong(self):
        # The merge is load-bearing: without it the final pages keep the
        # direct source's padding where the landed head belongs.
        mgr = _decode()
        _closing(mgr)
        for req in _registered(mgr).values():
            _replay(req, mgr.test_sources, mgr._kv_caches)
        want = _expected_page(mgr.test_heads, 0)
        got = mgr._kv_caches[0].view(torch.uint8)[FINAL[0]].reshape(want.shape)
        self.assertFalse(torch.equal(got, want))

    def test_merge_pads_to_a_bucket_with_the_last_pair(self):
        self.assertEqual(merge_bucket(1, 128), 8)
        self.assertEqual(merge_bucket(9, 128), 16)
        self.assertEqual(merge_bucket(100, 128), 128)
        with self.assertRaises(ValueError):
            merge_bucket(129, 128)
        mgr = _decode()
        seen = []
        import sglang_tpu.kv_page_write as kpw

        orig = kpw.write_pages_torch

        def spy(ids, pages, kv):
            seen.append(ids.tolist())
            return orig(ids, pages, kv)

        with patch.object(kpw, "write_pages_torch", spy):
            land_mod.merge_pages(
                mgr._kv_caches, [(1, 17), (2, 18)], 8, mgr._land_layout
            )
        self.assertEqual(seen, [[1, 2, 2, 2, 2, 2, 2, 2]] * 2)


class TestScratchFreePaths(unittest.TestCase):
    """Every scratch id goes back exactly once, on every terminal path."""

    def test_hold_then_retry_after_merge(self):
        # A 4-block arena: the first chunk's plan holds all of it, so the
        # closing plan waits; the merge frees it and the pre-pass retries.
        mgr = _decode(cap=4)
        _chunk(mgr, 0, 0, 4)
        self.assertEqual(mgr._land.arena.available, 0)
        # A second room's closing take queues behind nothing but the arena.
        mgr._decode.reshard_covered[ROOM] = []  # chunk 0 not yet acknowledged
        with patch.dict(sys.modules, _pb2_modules()):
            ids = mgr._land_take(ROOM, "other", 2, retry=Mock(), expire=Mock())
        self.assertIsNone(ids)
        self.assertEqual(len(mgr._land.holds), 1)
        hold = mgr._land.holds[0]
        plans = _registered(mgr)
        _settle(mgr, received=list(plans))
        mgr.land_prepass()
        hold.retry.assert_called_once_with()
        hold.expire.assert_not_called()
        self.assertEqual(len(mgr._land.holds), 0)
        self.assertEqual(mgr._land.arena.held, 0)

    def test_held_closing_registers_on_retry(self):
        mgr = _decode(cap=4)
        mgr._land.take("blocker", 99, 4)
        _closing(mgr)
        self.assertEqual(_sent(mgr, ReshardLandReady), [])
        self.assertEqual(len(mgr._land.holds), 1)
        mgr._land.free("blocker")
        with patch.dict(sys.modules, _pb2_modules()):
            mgr.land_prepass()
        (ready,) = _sent(mgr, ReshardLandReady)
        self.assertEqual(ready.pages, [0, 1, 2, 3])
        self.assertEqual(mgr._land.arena.held, 4)
        self.assertEqual(len(mgr._land.holds), 0)

    def test_held_chunk_blocks_the_closing_plan_until_retried(self):
        mgr = _decode(cap=4)
        mgr._land.take("blocker", 99, 4)
        _chunk(mgr, 0, 0, 2)
        self.assertIn(0, mgr._decode.reshard_chunk_registering[ROOM])
        _closing(mgr)
        # The closing plan waits for the held chunk.
        self.assertEqual(_sent(mgr, ReshardLandReady), [])
        mgr._land.free("blocker")
        with patch.dict(sys.modules, _pb2_modules()):
            mgr.land_prepass()
        (cr,) = _sent(mgr, ReshardLandChunkReady)
        (ready,) = _sent(mgr, ReshardLandReady)
        self.assertEqual(ready.pages, [2, 3])
        self.assertEqual(mgr._land.arena.held, 4)
        self.assertNotIn(ROOM, mgr._decode.reshard_chunk_registering)

    def test_hold_expiry_fails_the_room(self):
        mgr = _decode(cap=4)
        mgr._land.take("blocker", 99, 4)
        with envs.SGLANG_RAIDEN_LAND_HOLD_TIMEOUT_S.override(0.0):
            _closing(mgr)
            mgr.land_prepass()
        self.assertTrue(mgr.is_failed(ROOM))
        self.assertIn("SGLANG_RAIDEN_LAND_HOLD_TIMEOUT_S", mgr.failure_records[ROOM])
        self.assertEqual(len(mgr._land.holds), 0)
        self.assertEqual(mgr._land.arena.held, 4)  # only the blocker
        self.assertEqual(_sent(mgr, ReshardLandReady), [])

    def test_hold_deadline_ends_before_the_waiting_timeout(self):
        mgr = _decode(cap=4)
        mgr._land.take("blocker", 99, 4)
        mgr._land.started[ROOM] = time.time() - 295  # waiting_timeout 300
        _closing(mgr)
        self.assertLess(mgr._land.holds[0].deadline, time.time())

    def test_plan_larger_than_the_arena_fails_the_room(self):
        mgr = _decode(cap=2)
        _closing(mgr)
        self.assertTrue(mgr.is_failed(ROOM))
        self.assertIn("exceeds", mgr.failure_records[ROOM])
        self.assertEqual(mgr._land.arena.held, 0)
        self.assertEqual(_sent(mgr, ReshardLandReady), [])

    def test_register_failure_frees(self):
        mgr = _decode()
        mgr._engine.register_active_plan = Mock(side_effect=RuntimeError("boom"))
        _closing(mgr)
        self.assertTrue(mgr.is_failed(ROOM))
        self.assertEqual(mgr._land.arena.held, 0)
        self.assertEqual(mgr._land.plans, {})

    def test_registered_after_abort_frees(self):
        mgr = _decode()

        def clears(*args, **kwargs):
            mgr.clear_room(ROOM)

        mgr._engine.register_active_plan = Mock(side_effect=clears)
        _closing(mgr)
        self.assertEqual(_sent(mgr, ReshardLandReady), [])
        self.assertEqual(mgr._land.arena.held, 0)

    def test_failed_settle_frees_without_merge(self):
        mgr = _decode()
        _closing(mgr)
        merge = Mock()
        with patch.object(land_mod, "merge_pages", merge):
            _settle(mgr, failed=list(_registered(mgr)))
            mgr.land_prepass()
        merge.assert_not_called()
        self.assertEqual(mgr._land.arena.held, 0)
        self.assertTrue(mgr.land_merged(ROOM))

    def test_cleared_while_unsettled_frees_at_the_settle(self):
        mgr = _decode()
        _closing(mgr)
        plans = _registered(mgr)
        mgr.clear_room(ROOM)
        # The engine still owns the scratch: nothing is freed yet.
        self.assertEqual(mgr._land.arena.held, 4)
        merge = Mock()
        with patch.object(land_mod, "merge_pages", merge):
            _settle(mgr, received=list(plans))
            mgr.land_prepass()
        merge.assert_not_called()
        self.assertEqual(mgr._land.arena.held, 0)

    def test_cleared_after_settle_frees_without_merge(self):
        mgr = _decode()
        _closing(mgr)
        _settle(mgr, received=list(_registered(mgr)))
        mgr._drain_engine()
        self.assertTrue(all(p.settled for p in mgr._land.plans.values()))
        mgr.clear_room(ROOM)
        self.assertEqual(mgr._land.arena.held, 0)

    def test_room_failed_before_the_prepass_is_not_merged(self):
        # Settled while live, failed before the pre-pass: only the pre-pass
        # re-check stands between the merge and pages the room no longer
        # owns. The engine reports a plan once, so the pre-pass's own drain
        # sees nothing and cannot free it first.
        mgr = _decode()
        _closing(mgr)
        _settle(mgr, received=list(_registered(mgr)))
        mgr._drain_engine()
        _settle(mgr)
        self.assertTrue(all(p.settled for p in mgr._land.plans.values()))
        mgr._fail_reshard_room(ROOM, RuntimeError("peer died"))
        self.assertEqual(mgr._land.arena.held, 4)  # no other path freed it
        merge = Mock()
        with patch.object(land_mod, "merge_pages", merge):
            mgr.land_prepass()
        merge.assert_not_called()
        self.assertEqual(mgr._land.arena.held, 0)

    def test_room_failed_before_the_settle_frees_at_the_settle(self):
        mgr = _decode()
        _closing(mgr)
        _settle(mgr, received=list(_registered(mgr)))
        mgr._fail_reshard_room(ROOM, RuntimeError("peer died"))
        mgr._drain_engine()
        self.assertEqual((mgr._land.arena.held, mgr._land.plans), (0, {}))
        land = LandState(ScratchArena(5, 4))
        land.take("a", 1, 2)
        land.publish("a", [1, 2])
        land.settle(["a"], [], live=lambda room: False)
        self.assertEqual((land.arena.held, land.plans), (0, {}))

    def test_take_never_jumps_the_hold_queue(self):
        mgr = _decode(cap=4)
        mgr._land.take("blocker", 99, 4)
        retry = Mock()
        self.assertIsNone(mgr._land_take(ROOM, "head", 4, retry=retry, expire=Mock()))
        mgr._land.free("blocker")
        # The arena could serve a 1-block take now, but "head" is queued.
        self.assertIsNone(mgr._land_take(ROOM, "late", 1, retry=Mock(), expire=Mock()))
        self.assertEqual([h.key for h in mgr._land.holds], ["head", "late"])
        self.assertEqual(mgr._land.arena.held, 0)
        # The head's own retry takes, and the queue moves on.
        self.assertEqual(len(mgr._land.take("head", ROOM, 4)), 4)
        self.assertEqual([h.key for h in mgr._land.holds], ["late"])

    def test_merge_error_fails_the_room_and_frees(self):
        mgr = _decode()
        _closing(mgr)
        _settle(mgr, received=list(_registered(mgr)))
        with patch.object(
            land_mod, "merge_pages", Mock(side_effect=RuntimeError("xla"))
        ):
            mgr.land_prepass()
        self.assertTrue(mgr.is_failed(ROOM))
        self.assertIn("xla", mgr.failure_records[ROOM])
        self.assertEqual(mgr._land.arena.held, 0)

    def test_merge_error_fails_only_the_room_that_raised(self):
        other = ROOM + 1
        mgr = _decode()
        mgr._decode.reshard_meta[other] = dict(mgr._decode.reshard_meta[ROOM])
        _closing(mgr)
        plans = _registered(mgr)
        for req in plans.values():
            _replay(req, mgr.test_sources, mgr._kv_caches)
        mgr._land.take("bad", other, 2)
        mgr._land.publish("bad", [9, 10])
        _settle(mgr, received=[*plans, "bad"])
        real = land_mod.merge_pages
        calls = []

        def merge(kv, pairs, *rest):
            calls.append([f for f, _ in pairs])
            if any(f in (9, 10) for f, _ in pairs):
                raise RuntimeError("xla")
            real(kv, pairs, *rest)

        with patch.object(land_mod, "merge_pages", merge):
            mgr.land_prepass()
        # The batch fails, then each room re-merges alone.
        self.assertEqual(calls, [FINAL + [9, 10], FINAL, [9, 10]])
        self.assertTrue(mgr.is_failed(other))
        self.assertIn("xla", mgr.failure_records[other])
        self.assertFalse(mgr.is_failed(ROOM))
        self.assertTrue(mgr.land_merged(ROOM))
        want = _expected_page(mgr.test_heads, 1)
        got = mgr._kv_caches[1].view(torch.uint8)[FINAL[2]].reshape(want.shape)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(mgr._land.arena.held, 0)

    def test_settle_seen_by_another_drain_still_merges(self):
        # The real _drain_engine records the settle; a later pre-pass whose
        # own drain sees nothing still merges.
        mgr = _decode()
        _closing(mgr)
        for req in _registered(mgr).values():
            _replay(req, mgr.test_sources, mgr._kv_caches)
        _settle(mgr, received=list(_registered(mgr)))
        mgr._drain_engine()
        _settle(mgr)
        mgr.land_prepass()
        self.assertTrue(mgr.land_merged(ROOM))
        want = _expected_page(mgr.test_heads, 1)
        got = mgr._kv_caches[1].view(torch.uint8)[FINAL[3]].reshape(want.shape)
        self.assertTrue(torch.equal(got, want))

    def test_land_state_free_is_idempotent(self):
        land = LandState(ScratchArena(5, 4))
        ids = land.take("a", 1, 3)
        land.free("a")
        land.free("a")
        self.assertEqual((land.arena.held, sorted(ids)), (0, [5, 6, 7]))
        with self.assertRaises(ValueError):
            land.take("b", 1, 5)
        land.take("b", 1, 1)
        with self.assertRaises(ValueError):
            land.take("b", 1, 1)


class TestPromptLongerThanTheArena(unittest.TestCase):
    """A prompt of four arenas (A=8 blocks) at C=4 pages per chunk: every
    source cuts the same spans, the last step's tail included, so each plan
    fits and the held scratch never passes A. The prefill side is real down
    to its frames; the push is the decode plan's replay, as elsewhere."""

    PAGES, CAP, C, MID = 32, 8, 4, 14  # MID: pages of the one offered step

    def test_four_arenas_land_byte_exact(self):
        n = self.PAGES
        final = list(range(1, n + 1))
        src_blocks = list(range(1, n + 1))
        mgr = _decode(
            cap=self.CAP, pages=n, final=final, alloc=n, src_blocks=src_blocks
        )
        srcs = mgr._decode.reshard_meta[ROOM]["source_ranks"]
        peak = []
        take = mgr._land.take

        def spy(*args):
            ids = take(*args)
            peak.append(mgr._land.arena.held)
            return ids

        mgr._land.take = spy
        frames = {}
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            for p in srcs:
                pre = _prefill(rank=p, announce_pages=self.C)
                pre._prefill.reshard_req[ROOM]["decode_blocks"] = list(final)
                # A mid step is offered; the last step goes straight to the
                # closing reply, as RaidenKVSender.send does.
                self.assertEqual(
                    pre.append_send_blocks(ROOM, src_blocks[: self.MID]), 0
                )
                pre.offer_chunk(ROOM, 0, src_blocks[: self.MID], n)
                pre.append_send_blocks(ROOM, src_blocks[self.MID :])
                pre._prefill.reshard_finalized.add(ROOM)
                pre._maybe_reshard_reply(ROOM)
                frames[p] = [
                    c.args[1] for c in pre._send_multipart_locked.call_args_list
                ]
        spans = {
            p: [
                (c.start, len(c.blocks)) for c in map(ReshardChunk.from_frames, fs[:-1])
            ]
            for p, fs in frames.items()
        }
        want = [
            (0, 4),
            (4, 4),
            (8, 4),
            (12, 2),
            (14, 4),
            (18, 4),
            (22, 4),
            (26, 4),
            (30, 2),
        ]
        self.assertEqual(spans, {p: want for p in srcs})
        handlers = {
            ReshardChunk.TAG: mgr._on_reshard_chunk,
            ReshardBlocks.TAG: mgr._on_reshard_blocks,
        }
        done = set()
        with patch.dict(sys.modules, _pb2_modules()):
            # One source's frames, then the other's: chunks queue as holds
            # behind a full arena until merges free it.
            for p in srcs:
                for f in frames[p]:
                    handlers[f[0]](f)
            for _ in range(4 * len(want)):
                new = {k: r for k, r in _registered(mgr).items() if k not in done}
                if not new:
                    break
                for req in new.values():
                    _replay(req, mgr.test_sources, mgr._kv_caches)
                _settle(mgr, received=list(new))
                done |= set(new)
                mgr.land_prepass()
        self.assertFalse(mgr.is_failed(ROOM), mgr.failure_records.get(ROOM))
        self.assertEqual(len(_sent(mgr, ReshardLandChunkReady)), len(want))
        # Every page went by chunk: the closing READY is empty.
        self.assertEqual(_sent(mgr, ReshardLandReady), [])
        self.assertEqual([r.pages for r in _sent(mgr, ReshardReady)], [[]])
        self.assertLessEqual(max(peak), self.CAP)
        self.assertEqual(max(peak), self.CAP)  # the arena did fill
        self.assertEqual((mgr._land.arena.held, len(mgr._land.holds)), (0, 0))
        self.assertTrue(mgr.land_merged(ROOM))
        for layer, kv in enumerate(mgr._kv_caches):
            want_page = _expected_page(mgr.test_heads, layer)
            for b in final:
                got = kv.view(torch.uint8)[b].reshape(want_page.shape)
                self.assertTrue(torch.equal(got, want_page), f"layer {layer} block {b}")


class _Fabric:
    """Decode rank 0 and its sources wired end to end: RaidenKVSender.send
    drives each prefill, every frame is routed to the peer's handler, and
    each prefill push_registered_plan writes the bytes its sender plan names,
    at the offsets the decode's receive plan (same uuid, same (dst, src)
    order) names, as tpu-sync does. A receive plan settles once every source
    pushed every layer."""

    def __init__(self, pages, cap, announce):
        n = pages
        self.final = list(range(1, n + 1))
        self.src_blocks = list(range(1, n + 1))
        self.dec = _decode(
            cap=cap, pages=n, final=self.final, alloc=n, src_blocks=self.src_blocks
        )
        self.srcs = self.dec._decode.reshard_meta[ROOM]["source_ranks"]
        self.peak = 0
        take = self.dec._land.take

        def spy(*args):
            ids = take(*args)
            self.peak = max(self.peak, self.dec._land.arena.held)
            return ids

        self.dec._land.take = spy
        self.pre = {}
        self.sender = {}
        self.pushed = {}  # uuid -> {(p, layer)}
        self.settled = set()
        for p in self.srcs:
            pre = _prefill(rank=p, announce_pages=announce)
            pre._prefill.reshard_req[ROOM]["decode_blocks"] = list(self.final)
            pre._kv_caches = self.dec.test_sources[p]
            pre._pipeline = None
            pre.is_dummy_cp_rank = False
            pre.request_status[ROOM] = KVPoll.WaitingForInput
            plans = {}
            pre._engine.register_active_plan = Mock(
                side_effect=lambda uuid, req, is_sender, plans=plans: plans.update(
                    {uuid: req}
                )
            )
            pre._engine.push_registered_plan = Mock(
                side_effect=functools.partial(self._push, p, plans)
            )
            self.pre[p] = pre
            snd = object.__new__(RaidenKVSender)
            vars(snd).update(
                kv_mgr=pre,
                bootstrap_room=ROOM,
                num_kv_indices=n,
                aux_index=None,
                _transfer_start_time=None,
                _record_transfer_indices=lambda *args: None,
                _started=False,
                _last_chunk=False,
            )
            self.sender[p] = snd
        self.cursor = {id(m): 0 for m in self._mgrs()}

    def _mgrs(self):
        return [self.dec, *self.pre.values()]

    def _push(self, p, plans, uuid, endpoint, src_ids, dst_ids, layer, **kwargs):
        sreq = plans[uuid]
        (rreq,) = [
            c.args[1]
            for c in self.dec._engine.register_active_plan.call_args_list
            if c.args[0] == uuid
        ]
        sent = sreq.shard_push_schedules[0].entries
        recv = rreq.shard_push_schedules[p].entries
        pairs = [(e.dst_block_id, e.src_block_id) for e in sent]
        assert pairs == [(e.dst_block_id, e.src_block_id) for e in recv]
        # tpu-sync contract: the push's dst list is the entries' dst ids.
        assert list(dst_ids) == [e.dst_block_id for e in sent], (dst_ids, pairs)
        src = self.pre[p]._kv_caches[layer].view(torch.uint8)
        dst = self.dec._kv_caches[layer].view(torch.uint8)
        for se, re in zip(sent, recv):
            idx_s = (
                se.src_offset_bytes
                + se.src_stride_bytes * torch.arange(se.count).view(-1, 1)
                + torch.arange(se.size_bytes)
            ).reshape(-1)
            idx_d = (
                re.dst_offset_bytes
                + re.dst_stride_bytes * torch.arange(re.count).view(-1, 1)
                + torch.arange(re.size_bytes)
            ).reshape(-1)
            dst[re.dst_block_id].reshape(-1)[idx_d] = src[se.src_block_id].reshape(-1)[
                idx_s
            ]
        self.pushed.setdefault(uuid, set()).add((p, layer))

    def step(self, p, blocks, last):
        snd = self.sender[p]
        snd.should_send_kv_chunk(len(blocks), last)
        snd.send(np.array(blocks, dtype=np.int32))

    def _pump(self):
        """Deliver every queued frame once; True if any moved."""
        moved = False
        for mgr in self._mgrs():
            calls = mgr._send_multipart_locked.call_args_list
            start = self.cursor[id(mgr)]
            self.cursor[id(mgr)] = len(calls)
            for c in calls[start:]:
                endpoint, frames = c.args[0], c.args[1]
                moved = True
                if mgr is self.dec:
                    p = int(endpoint.split(".")[3].split(":")[0]) - 10
                    target = self.pre[p]
                else:
                    target = self.dec
                handler = {
                    ReshardChunk.TAG: "_on_reshard_chunk",
                    ReshardBlocks.TAG: "_on_reshard_blocks",
                    ReshardChunkReady.TAG: "_on_reshard_chunk_ready",
                    ReshardLandChunkReady.TAG: "_on_reshard_land_chunk_ready",
                    ReshardReady.TAG: "_on_reshard_ready",
                    ReshardLandReady.TAG: "_on_reshard_land_ready",
                }[frames[0]]
                getattr(target, handler)(frames)
        return moved

    def run(self):
        layers = len(self.dec._kv_caches)
        want = {(p, layer) for p in self.srcs for layer in range(layers)}
        with patch.dict(sys.modules, _pb2_modules()):
            for _ in range(200):
                moved = self._pump()
                uuids = {
                    c.args[0]: c.args[1].req_id
                    for c in self.dec._engine.register_active_plan.call_args_list
                }
                done = [
                    rid
                    for u, rid in uuids.items()
                    if self.pushed.get(u) == want and rid not in self.settled
                ]
                self.settled.update(done)
                _settle(self.dec, received=done)
                self.dec.land_prepass()
                if not moved and not done:
                    break


class TestOneStepLandingRoom(unittest.TestCase):
    """The real send path, prefill push included: a landing room whose last
    step (or only step) is longer than C pages is pushed whole. send() never
    offers the last step, so the chunks the closing reply announces carry no
    send_total; their pushes must window against the room's page count, not
    the chunk's (#829)."""

    CAP, C = 8, 4

    def _run(self, steps):
        n = sum(steps)
        fab = _Fabric(n, self.CAP, self.C)
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            for p in fab.srcs:
                lo = 0
                for i, k in enumerate(steps):
                    fab.step(p, fab.src_blocks[lo : lo + k], i == len(steps) - 1)
                    lo += k
            fab.run()
        dec = fab.dec
        self.assertFalse(dec.is_failed(ROOM), dec.failure_records.get(ROOM))
        # Every receive plan got every source's every layer, and settled.
        registered = {
            c.args[1].req_id for c in dec._engine.register_active_plan.call_args_list
        }
        self.assertEqual(fab.settled, registered)
        self.assertTrue(dec.land_merged(ROOM))
        self.assertEqual((dec._land.arena.held, len(dec._land.holds)), (0, 0))
        self.assertLessEqual(fab.peak, self.CAP)
        for pre in fab.pre.values():
            self.assertFalse(pre.is_failed(ROOM), pre.failure_records.get(ROOM))
            self.assertEqual(pre._reshard_pushes_pending, {})
            self.assertIn(str(ROOM), pre._completed_send)
        for layer, kv in enumerate(dec._kv_caches):
            want = _expected_page(dec.test_heads, layer)
            for b in fab.final:
                got = kv.view(torch.uint8)[b].reshape(want.shape)
                self.assertTrue(torch.equal(got, want), f"layer {layer} block {b}")

    def test_one_step_longer_than_c(self):
        self._run([3 * self.C])

    def test_one_step_of_exactly_c(self):
        self._run([self.C])

    def test_one_step_of_c_plus_one(self):
        self._run([self.C + 1])

    def test_last_step_alone_longer_than_c(self):
        # The first step is offered (send_total set), the last is not.
        self._run([self.C, 2 * self.C + 1])
        self._run([self.C - 1, 2 * self.C + 1])


class TestSuccessGate(unittest.TestCase):
    def _receiver(self, merged):
        mgr = object.__new__(RaidenKVManager)
        vars(mgr).update(
            _drain_engine=lambda: None,
            is_failed=lambda room: False,
            is_recv_done=lambda room: True,
            land_merged=lambda room: merged,
            room_has_active_io=lambda room: False,
            last_announcement_age=lambda room: None,
            maybe_start_pull=lambda room: False,
            request_status={ROOM: KVPoll.WaitingForInput},
            failure_records={},
            failure_lock=threading.Lock(),
            waiting_timeout=300,
            _state_lock=threading.Lock(),
            _aborted={},
            _prefill=None,
            _failed=set(),
            _native_failed=set(),
            connection_lock=threading.Lock(),
            connection_pool={},
        )
        mgr.check_status = lambda room: mgr.request_status[room]
        recv = RaidenKVReceiver.__new__(RaidenKVReceiver)
        recv.kv_mgr = mgr
        recv.bootstrap_room = ROOM
        recv.conclude_state = None
        recv.abort_notified = False
        recv.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 4444}]
        recv._connection_pool_entries = {}
        recv._started = False
        recv.init_time = time.time()
        recv._handshake_start = time.time()
        recv._handshake_timeout = 300
        return recv

    def test_success_and_drained_wait_for_the_merge(self):
        recv = self._receiver(merged=False)
        self.assertNotEqual(recv.poll(), KVPoll.Success)
        self.assertFalse(recv.drained())
        recv = self._receiver(merged=True)
        self.assertEqual(recv.poll(), KVPoll.Success)
        self.assertTrue(recv.drained())

    def test_no_arena_is_always_merged(self):
        mgr = object.__new__(RaidenKVManager)
        self.assertTrue(mgr.land_merged(ROOM))
        mgr.land_prepass()  # no arena: nothing to do


class TestRefusal(unittest.TestCase):
    """reshard_refusal caches per manager, so every verdict uses a fresh one."""

    def _mgr(self, cap, total_heads=8):
        mgr = _decode(cap=8, total_heads=total_heads)
        mgr._land = LandState(ScratchArena(17, cap)) if cap else None
        return mgr

    def test_landing_cell_needs_env_and_a_whole_chunk(self):
        # C = 8192 / 128 = 64 pages.
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(False):
            reason = self._mgr(128).reshard_refusal(8, 4)
            self.assertIn("SGLANG_RAIDEN_RESHARD_LAND_ARENA=1", reason)
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            self.assertIsNone(self._mgr(128).reshard_refusal(8, 4))
            self.assertIsNone(self._mgr(64).reshard_refusal(8, 4))
            self.assertIn("63 blocks", self._mgr(63).reshard_refusal(8, 4))
            self.assertIn("no landing arena", self._mgr(0).reshard_refusal(8, 4))

    def test_aligned_cell_stays_allowed(self):
        # 16 kv heads TP8 -> TP4: two whole-granule sources per decode rank.
        for env in (False, True):
            with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(env):
                for cap in (0, 64):
                    with self.subTest(env=env, cap=cap):
                        mgr = self._mgr(cap, total_heads=16)
                        self.assertIsNone(mgr.reshard_refusal(8, 4))

    def test_hint_probe_error_keeps_the_refusal(self):
        from sglang_tpu.disaggregation.raiden import reshard_schedule as rs

        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(False):
            plain = self._mgr(128).reshard_refusal(8, 4)
            real = rs.push_reshard_refusal

            def probe_raises(*args, land_ok=False, **kwargs):
                if land_ok:
                    raise ValueError("probe")
                return real(*args, land_ok=land_ok, **kwargs)

            with patch.object(rs, "push_reshard_refusal", probe_raises):
                got = self._mgr(128).reshard_refusal(8, 4)
        self.assertNotIn("probe", got)
        self.assertNotIn("SGLANG_RAIDEN_RESHARD_LAND_ARENA=1", got)
        self.assertTrue(plain.startswith(got), (plain, got))

    def test_prefill_needs_only_the_env(self):
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(False):
            reason = _prefill().reshard_refusal(8, 4)
            self.assertIn("SGLANG_RAIDEN_RESHARD_LAND_ARENA=1", reason)
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            self.assertIsNone(_prefill().reshard_refusal(8, 4))


# ---- prefill ---------------------------------------------------------------


class TestPrefillValidatesTheReady(unittest.TestCase):
    """A READY whose kind or pairing disagrees with this side's
    classification fails the room instead of pushing."""

    def _chunk_ready(self, mgr, msg):
        mgr._prefill.reshard_chunks[ROOM] = {0: (0, PREFILL_BLOCKS[:2])}
        mgr._push_executor = Mock()
        mgr.request_status[ROOM] = KVPoll.Transferring
        handler = (
            mgr._on_reshard_land_chunk_ready
            if isinstance(msg, ReshardLandChunkReady)
            else mgr._on_reshard_chunk_ready
        )
        handler(msg.to_frames())
        return mgr._push_executor.submit

    def test_chunk_ready_checks(self):
        cases = [
            # (rank, message, failed)
            (1, ReshardLandChunkReady(ROOM, 0, 7, [17, 18], LAND_VERSION), False),
            (1, ReshardLandChunkReady(ROOM, 0, 7, [17, 18], LAND_VERSION + 1), True),
            (1, ReshardLandChunkReady(ROOM, 0, 7, [17], LAND_VERSION), True),
            (1, ReshardChunkReady(ROOM, 0, 7), True),
            # Rank 0 and rank 1 share decode rank 0, whose plan lands.
            (0, ReshardLandChunkReady(ROOM, 0, 7, [17, 18], LAND_VERSION), False),
            (0, ReshardChunkReady(ROOM, 0, 7), True),
        ]
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            for rank, msg, failed in cases:
                with self.subTest(rank=rank, msg=msg):
                    mgr = _prefill(rank=rank)
                    submit = self._chunk_ready(mgr, msg)
                    self.assertEqual(mgr.is_failed(ROOM), failed)
                    self.assertEqual(submit.call_count, 0 if failed else 1)
                    if not failed and isinstance(msg, ReshardLandChunkReady):
                        self.assertEqual(submit.call_args.args[-1], [17, 18])
                    if failed:
                        # Tombstoned like a refused RESHARD_REQ: a later
                        # well-formed READY pushes nothing.
                        self.assertTrue(mgr.is_aborted(ROOM))
                        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
                        good = ReshardLandChunkReady(ROOM, 0, 7, [17, 18], LAND_VERSION)
                        mgr._on_reshard_land_chunk_ready(good.to_frames())
                        submit.assert_not_called()

    def test_land_ready_for_a_direct_plan_fails(self):
        # 8 kv heads TP4 -> TP2 never lands.
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            mgr = _prefill(rank=1, src_tp=4, dst_tp=2)
            submit = self._chunk_ready(
                mgr, ReshardLandChunkReady(ROOM, 0, 7, [17, 18], LAND_VERSION)
            )
            self.assertTrue(mgr.is_failed(ROOM))
            submit.assert_not_called()

    def test_closing_ready_checks(self):
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            # An empty plain READY (nothing left to land) is fine.
            mgr = _prefill()
            mgr._maybe_reshard_push = Mock()
            mgr._on_reshard_ready(ReshardReady(ROOM, []).to_frames())
            self.assertFalse(mgr.is_failed(ROOM))
            mgr._maybe_reshard_push.assert_called_once_with(ROOM)
            # Pages without scratch for a landing plan fail.
            mgr = _prefill()
            mgr.request_status[ROOM] = KVPoll.Transferring
            mgr._maybe_reshard_push = Mock()
            mgr._on_reshard_ready(ReshardReady(ROOM, [0, 1]).to_frames())
            self.assertTrue(mgr.is_failed(ROOM))
            self.assertTrue(mgr.is_aborted(ROOM))
            self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
            mgr._on_reshard_land_ready(
                ReshardLandReady(ROOM, [0, 1], [17, 18], LAND_VERSION).to_frames()
            )
            mgr._maybe_reshard_push.assert_not_called()
            self.assertNotIn(ROOM, mgr._prefill.reshard_ready_scratch)
            # A matching landing READY records its scratch.
            mgr = _prefill()
            mgr._maybe_reshard_push = Mock()
            mgr._on_reshard_land_ready(
                ReshardLandReady(ROOM, [0, 1], [17, 18], LAND_VERSION).to_frames()
            )
            self.assertFalse(mgr.is_failed(ROOM))
            self.assertEqual(mgr._prefill.reshard_ready_scratch[ROOM], [17, 18])

    def test_env_off_keeps_todays_ready(self):
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(False):
            mgr = _prefill()
            mgr._maybe_reshard_push = Mock()
            mgr._on_reshard_ready(ReshardReady(ROOM, [0, 1]).to_frames())
            self.assertFalse(mgr.is_failed(ROOM))


class TestPrefillChunks(unittest.TestCase):
    def _announce(self, env, pages=150):
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(env):
            mgr = _prefill(announce_pages=64)
            mgr._prefill.reshard_req[ROOM]["decode_blocks"] = list(range(pages))
            mgr._announce_reshard_chunk(ROOM, 0, list(range(1000, 1000 + pages)), pages)
        return [
            ReshardChunk.from_frames(c.args[1])
            for c in mgr._send_multipart_locked.call_args_list
        ]

    def test_landing_room_cuts_chunks_to_one_announcement(self):
        chunks = self._announce(True)
        self.assertEqual([len(c.blocks) for c in chunks], [64, 64, 22])
        self.assertEqual([c.start for c in chunks], [0, 64, 128])
        self.assertEqual([c.index for c in chunks], [0, 1, 2])
        self.assertEqual(chunks[1].blocks[0], 1064)

    def test_env_off_announces_one_chunk(self):
        self.assertEqual([len(c.blocks) for c in self._announce(False)], [150])

    def _reply(self, env, pending=None, offered_end=None):
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(env):
            mgr = _prefill(announce_pages=64)
            mgr._prefill.send_blocks[ROOM] = list(PREFILL_BLOCKS)
            if pending is not None or offered_end is not None:
                mgr._prefill.send_total[ROOM] = 4
            if pending is not None:
                mgr._prefill.pending_offer[ROOM] = (pending, PREFILL_BLOCKS[pending:3])
            if offered_end is not None:
                mgr._prefill.offered_end[ROOM] = offered_end
            mgr._prefill.reshard_finalized.add(ROOM)
            mgr._maybe_reshard_reply(ROOM)
        frames = [c.args[1] for c in mgr._send_multipart_locked.call_args_list]
        chunks = [
            (c.start, c.blocks, c.total_pages)
            for c in map(ReshardChunk.from_frames, frames[:-1])
        ]
        self.assertEqual(frames[-1][0], MessageTag.RESHARD_BLOCKS)
        return mgr, chunks

    def test_landing_room_announces_every_unannounced_page_before_the_reply(self):
        # Pending remainder plus the last step (page 3), which send() never
        # offers; the last step alone; and a room no step offered.
        for kwargs, start in (
            (dict(pending=1, offered_end=3), 1),
            (dict(offered_end=2), 2),
            (dict(), 0),
        ):
            with self.subTest(**kwargs):
                mgr, chunks = self._reply(True, **kwargs)
                self.assertEqual(chunks, [(start, PREFILL_BLOCKS[start:], 4)])
                self.assertNotIn(ROOM, mgr._prefill.pending_offer)
        _, chunks = self._reply(False, pending=1, offered_end=3)
        self.assertEqual(chunks, [])
        # Everything already offered: nothing left to announce.
        _, chunks = self._reply(True, offered_end=4)
        self.assertEqual(chunks, [])


class TestPrefillPushTargets(unittest.TestCase):
    def test_landing_rank_pushes_into_scratch(self):
        with envs.SGLANG_RAIDEN_RESHARD_LAND_ARENA.override(True):
            for rank, want in ((0, FINAL[:2]), (1, [17, 18])):
                with self.subTest(rank=rank):
                    mgr = _prefill(rank=rank)
                    mgr._prefill.send_total[ROOM] = 4
                    mgr._prefill.reshard_chunks[ROOM] = {0: (0, PREFILL_BLOCKS[:2])}
                    with patch.dict(sys.modules, _pb2_modules()):
                        mgr._run_reshard_push(ROOM, 0, 7, [17, 18])
                    calls = mgr._engine.push_registered_plan.call_args_list
                    self.assertEqual(len(calls), 2)  # one per layer
                    for c in calls:
                        self.assertEqual(c.args[2], PREFILL_BLOCKS[:2])
                        self.assertEqual(c.args[3], want)
                    _, req = mgr._engine.register_active_plan.call_args.args
                    dst = [e.dst_block_id for e in req.shard_push_schedules[0].entries]
                    self.assertEqual(dst, want)


class TestSerde(unittest.TestCase):
    def test_round_trip(self):
        for msg in (
            ReshardLandReady(ROOM, [0, 3], [17, 18], LAND_VERSION),
            ReshardLandChunkReady(ROOM, 2, (1 << 63) - 1, [17], LAND_VERSION),
        ):
            with self.subTest(msg=type(msg).__name__):
                self.assertEqual(type(msg).from_frames(msg.to_frames()), msg)
        self.assertNotEqual(ReshardLandReady.TAG, ReshardReady.TAG)
        self.assertNotEqual(ReshardLandChunkReady.TAG, ReshardChunkReady.TAG)


if __name__ == "__main__":
    unittest.main()
