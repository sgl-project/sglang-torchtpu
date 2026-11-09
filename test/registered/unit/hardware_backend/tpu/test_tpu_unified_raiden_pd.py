"""CPU unit tests for TPU unified KV pool P/D disaggregation via Raiden."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric, StateType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import (
    PER_LAYER_KV_LAYOUT,
    UNIFIED_SWA_KV_LAYOUT,
    ReshardBlocks,
    ReshardChunk,
)
from sglang_tpu.unified_kv_pool import AttnLayerSpec, TpuUnifiedSWAKVPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "_disagg_raiden_transfer_harness",
    pathlib.Path(__file__).resolve().parents[3]
    / "tpu"
    / "unit"
    / "test_disagg_raiden_transfer.py",
)
_harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _harness
_spec.loader.exec_module(_harness)
RaidenRendezvousTestCase = _harness.RaidenRendezvousTestCase
_fake_tpu_sync = _harness._fake_tpu_sync
_FakeRaidenEngine = _harness._FakeRaidenEngine


def _pool(swa_layers=(1, 2, 4, 5, 6), *, head_num=2, dtype=torch.bfloat16):
    return TpuUnifiedSWAKVPool(
        size=64,
        page_size=4,
        dtype=dtype,
        full=AttnLayerSpec(head_num=head_num, head_dim=128, layer_ids=(0, 3)),
        swa=AttnLayerSpec(head_num=head_num, head_dim=128, layer_ids=swa_layers),
        device="cpu",
    )


def _fake_pb2_modules(engine_cls=None) -> dict[str, types.ModuleType]:
    class _Entries(list):
        def add(self):
            e = SimpleNamespace()
            self.append(e)
            return e

    pb2 = types.ModuleType("tpu_sync.rpc.raiden_service_pb2")
    pb2.MEMORY_TYPE_DRAM, pb2.MEMORY_TYPE_HBM = 1, 2
    pb2.StartTransferRequest = lambda: SimpleNamespace(
        shard_push_schedules=defaultdict(lambda: SimpleNamespace(entries=_Entries()))
    )
    mods = _fake_tpu_sync(engine_cls or _FakeRaidenEngine)
    mods["tpu_sync"].rpc = rpc = types.ModuleType("tpu_sync.rpc")
    rpc.raiden_service_pb2 = pb2
    mods["tpu_sync.rpc"] = rpc
    mods["tpu_sync.rpc.raiden_service_pb2"] = pb2
    return mods


class TestUnifiedRaidenPD(RaidenRendezvousTestCase):
    def _unified(self, mgr, pool: TpuUnifiedSWAKVPool, *, chunked: bool = False):
        mgr._unified_swa_pool, mgr.kv_layout = pool, UNIFIED_SWA_KV_LAYOUT
        mgr._kv_caches = pool.kv_pool.page_buffers()
        mgr._deferred_ack_targets, mgr._staging_outstanding = {}, {}
        mgr._deferred_ack_poisoned_rooms, mgr.deferred_bootstrap = set(), None
        mgr.chunked_transfer = chunked
        mgr._listener_start_read = False
        mgr._engine.transfer_address = mgr.raiden_endpoint
        mgr._engine.plan_host_blocks = lambda uuid, blks: list(range(len(blks)))
        mgr._engine.d2h = lambda *_: SimpleNamespace(
            is_ready=lambda: True, ok=lambda: True, error_message=str
        )
        if hasattr(mgr, "_push_executor"):
            mgr._push_executor.shutdown(wait=True)
            mgr._push_executor = SimpleNamespace(
                submit=lambda fn, *a, **k: fn(*a, **k), shutdown=lambda **_: None
            )
        return mgr

    def _pump(self, *mgrs):
        progressed = True
        while progressed:
            progressed = False
            for m in mgrs:
                frames = self.inboxes[m.endpoint].frames
                while frames:
                    m._dispatch(frames.pop(0))
                    progressed = True

    def test_single_engine_sizing_metrics_and_validation(self):
        for dtype, swa_ids in (
            (torch.bfloat16, (1, 2, 4, 5, 6)),
            (torch.bfloat16, (1, 2)),
            (torch.float8_e4m3fn, (1, 2, 4, 5, 6)),
        ):
            with self.subTest(dtype=dtype, swa_ids=swa_ids):
                pool = _pool(swa_ids, dtype=dtype)
                m = self._prefill()
                m._engine = None
                m.kv_args.state_types = [StateType.SWA]
                m.kv_args.kv_item_lens = pool.kv_pool.get_contiguous_buf_infos()[2]
                m.kv_item_lens_sum = sum(m.kv_args.kv_item_lens)
                m._max_request_blocks, m._swa_max_request_blocks = lambda: 8, lambda: 2
                m._size_host_staging = lambda engines, layers: [(6, n) for n in layers]
                built = []
                groups = get_raiden_kv_caches(pool)
                self.assertEqual(groups.swa, [])
                with patch.dict(
                    sys.modules, _fake_tpu_sync(lambda **kw: built.append(kw) or kw)
                ):
                    m.set_kv_caches(groups)
                m_swa = pool.layout.num_swa_groups
                self.assertEqual(len(built), 1)
                self.assertIsNone(m._swa_engine)
                self.assertEqual(m.kv_layout, UNIFIED_SWA_KV_LAYOUT)
                self.assertEqual(built[0]["max_blocks"], 8 + 2 * m_swa)
                self.assertEqual(m.state_item_lens_sum, m_swa * m.kv_item_lens_sum)

                sender = object.__new__(RaidenKVSender)
                sender.kv_mgr = m
                sender._transfer_metric = KVTransferMetric()
                sender._transfer_num_kv_indices = sender._transfer_num_state_indices = 0
                m.get_kv_replica_factor = lambda: 1
                sender._record_transfer_indices(np.array([1, 2, 3]), [np.array([4, 5])])
                self.assertEqual(
                    sender.get_transfer_metric().transfer_total_bytes,
                    (3 + 2 * m_swa) * m.kv_item_lens_sum,
                )

        pool = _pool()
        m = self._prefill()
        m.kv_args.state_types = [StateType.SWA]
        m._pipeline_enabled = True
        with self.assertRaisesRegex(NotImplementedError, "enable-unified-memory"):
            m.set_kv_caches(get_raiden_kv_caches(pool))
        m._unified_swa_pool = pool
        for bad in (None, [], [None]):
            with self.assertRaisesRegex(RuntimeError, "missing its page indices"):
                m._swa_blocks_from_state_indices(bad)

    def test_pull_block_pairing_prefix_hits_and_layout_mismatch(self):
        pool = _pool()  # M_swa = 3
        p_swa, d_swa = pool.expand_swa_pages([5, 6, 7]), pool.expand_swa_pages([8, 9])

        # 1) Chunked pull (normal 2+2 Full + SWA, and cover_all_full where missing == [])
        for cover_all in (False, True):
            with self.subTest(chunked=True, cover_all=cover_all):
                pre = self._unified(self._prefill(), pool, chunked=True)
                dec = self._unified(self._decode(), pool, chunked=True)
                room, n_full = 8100 + int(cover_all), 2 if cover_all else 4
                dec.register_decode_request(
                    room,
                    list(range(20, 20 + n_full)),
                    d_swa,
                    None,
                    self._bootstrap_info(pre),
                )
                self._pump(pre, dec)
                pre.append_send_blocks(
                    room, [1, 2], swa_block_ids=p_swa[-6:] if cover_all else None
                )
                pre._announce_chunk(room, 0, [1, 2], total_pages=n_full)
                if not cover_all:
                    pre.append_send_blocks(room, [3, 4], swa_block_ids=p_swa[-6:])
                pre.finalize_send(room, aux_index=None)
                self._pump(pre, dec)

                self.assertTrue(dec.maybe_start_pull(room))
                dec._engine._stats = ([], [f"{room}#0"], [])
                dec._drain_engine()
                self.assertTrue(dec.maybe_start_pull(room))
                dec._engine._stats = ([], [str(room)], [])
                pre._engine._stats = ([str(room)], [], [])
                dec._drain_engine()
                pre._drain_engine()
                self.assertTrue(pre.is_send_done(room) and dec.is_recv_done(room))
                expected_tail = p_swa[-6:] if cover_all else [3, 4] + p_swa[-6:]
                expected_local = d_swa if cover_all else [22, 23] + d_swa
                self.assertEqual(
                    [(a[0], a[3], a[4]) for a, _ in dec._engine.started],
                    [
                        (f"{room}#0", [1, 2], [20, 21]),
                        (str(room), expected_tail, expected_local),
                    ],
                )

        # 2) Whole pull partial prefix hit (4->2 Full tail, 9->6 SWA tail), 100% hit, SWA-only
        for d_full, d_s, exp_r in (
            ([20, 21], d_swa, [([3, 4] + p_swa[-6:], [20, 21] + d_swa)]),
            ([], d_swa, [(p_swa[-6:], d_swa)]),
            ([], [], []),
        ):
            with self.subTest(d_full=d_full, swa=bool(d_s)):
                pre = self._unified(self._prefill(), pool, chunked=False)
                dec = self._unified(self._decode(), pool, chunked=False)
                room = 8110 + len(d_full) + len(d_s)
                dec.register_decode_request(
                    room, d_full, d_s, None, self._bootstrap_info(pre)
                )
                self._pump(pre, dec)
                pre.append_send_blocks(room, [1, 2, 3, 4], swa_block_ids=p_swa)
                pre.finalize_send(room, aux_index=None)
                self._pump(pre, dec)
                self.assertTrue(dec.maybe_start_pull(room))
                if exp_r:
                    dec._engine._stats = ([], [str(room)], [])
                    pre._engine._stats = ([str(room)], [], [])
                    dec._drain_engine()
                    pre._drain_engine()
                self.assertTrue(pre.is_send_done(room) and dec.is_recv_done(room))
                self.assertEqual([(a[3], a[4]) for a, _ in dec._engine.started], exp_r)

        # 3) Layout mismatch on CHUNK and BLOCKS
        for via_chunk in (False, True):
            pre = self._unified(self._prefill(), pool, chunked=via_chunk)
            dec = self._unified(self._decode(), pool, chunked=via_chunk)
            pre.kv_layout = PER_LAYER_KV_LAYOUT
            room = 8130 + int(via_chunk)
            dec.request_status[room] = KVPoll.WaitingForInput
            dec.register_decode_request(
                room, [1, 2], d_swa, None, self._bootstrap_info(pre)
            )
            self._pump(pre, dec)
            pre.append_send_blocks(room, [1, 2], swa_block_ids=p_swa[-6:])
            if via_chunk:
                pre._announce_chunk(room, 0, [1], total_pages=2)
            else:
                pre.finalize_send(room, aux_index=None)
            self._pump(pre, dec)
            self.assertEqual(dec.request_status[room], KVPoll.Failed)

    def test_reshard_push_block_pairing_swa_only_and_layout_mismatch(self):
        p_pool, d_pool = _pool(head_num=1), _pool(head_num=2)
        p_swa, d_swa = p_pool.expand_swa_pages([5, 6]), d_pool.expand_swa_pages([8, 9])

        def _trio(chunked: bool):
            ps = [
                self._unified(
                    self._prefill(attn_tp_rank=r, attn_tp_size=4),
                    p_pool,
                    chunked=chunked,
                )
                for r in (0, 1)
            ]
            dec = self._unified(
                self._decode(attn_tp_rank=0, attn_tp_size=2), d_pool, chunked=chunked
            )
            return ps, dec

        with patch.dict(sys.modules, _fake_pb2_modules()):
            for idx, (chunked, cover_all, swa_only) in enumerate(
                (
                    (True, False, False),
                    (True, True, False),
                    (False, False, False),
                    (False, False, True),  # regression: decode_blocks == []
                    (True, False, True),
                )
            ):
                with self.subTest(
                    chunked=chunked, cover_all=cover_all, swa_only=swa_only
                ):
                    ps, dec = _trio(chunked)
                    room = 8200 + idx
                    dec.register_decode_request(
                        room,
                        [] if swa_only else [20, 21],
                        d_swa,
                        None,
                        [
                            {"rank_ip": "127.0.0.1", "rank_port": p.rank_port}
                            for p in ps
                        ],
                    )
                    self._pump(*ps, dec)
                    for p in ps:
                        if chunked and not swa_only:
                            nc = 2 if cover_all else 1
                            p.append_send_blocks(
                                room,
                                [1, 2][:nc],
                                swa_block_ids=p_swa if cover_all else None,
                            )
                            p._announce_reshard_chunk(
                                room, 0, [1, 2][:nc], total_pages=2
                            )
                    self._pump(*ps, dec)
                    for p in ps:
                        if chunked and not swa_only and not cover_all:
                            p.append_send_blocks(room, [2], swa_block_ids=p_swa)
                        elif not (chunked and not swa_only):
                            p.append_send_blocks(room, [1, 2], swa_block_ids=p_swa)
                        p.finalize_send(room, aux_index=None)
                    self._pump(*ps, dec)

                    # Verify sender pushed exact paired Full + SWA blocks on layer 0
                    l0_pushes = [
                        (a[2], a[3]) for a, _ in ps[0]._engine.pushes if a[4] == 0
                    ]
                    if swa_only:
                        self.assertEqual(l0_pushes, [(p_swa, d_swa)])
                    elif chunked and cover_all:
                        self.assertEqual(
                            l0_pushes, [([1, 2], [20, 21]), (p_swa, d_swa)]
                        )
                    elif chunked:
                        self.assertEqual(
                            l0_pushes, [([1], [20]), ([2] + p_swa, [21] + d_swa)]
                        )
                    else:
                        self.assertEqual(
                            l0_pushes, [([1, 2] + p_swa, [20, 21] + d_swa)]
                        )

            for via_chunk in (False, True):
                ps, dec = _trio(via_chunk)
                room = 8250 + int(via_chunk)
                dec.register_decode_request(
                    room,
                    [20],
                    d_swa,
                    None,
                    [{"rank_ip": "127.0.0.1", "rank_port": p.rank_port} for p in ps],
                )
                self._pump(*ps, dec)
                bad = (
                    ReshardChunk(room, 0, 0, 1, [1], 0, kv_layout=PER_LAYER_KV_LAYOUT)
                    if via_chunk
                    else ReshardBlocks(
                        room, [1], 0, b"", p_swa, kv_layout=PER_LAYER_KV_LAYOUT
                    )
                )
                dec._dispatch(bad.to_frames())
                self.assertEqual(dec.request_status[room], KVPoll.Failed)

            for via_chunk in (False, True):
                with self.subTest(seq_along_lane_reshard=True, via_chunk=via_chunk):
                    ps, dec = _trio(via_chunk)
                    dec._seq_along_lane = True
                    room = 8260 + int(via_chunk)
                    dec.register_decode_request(
                        room,
                        [20],
                        d_swa,
                        None,
                        [
                            {"rank_ip": "127.0.0.1", "rank_port": p.rank_port}
                            for p in ps
                        ],
                    )
                    self._pump(*ps, dec)
                    for p in ps:
                        p.append_send_blocks(room, [1], swa_block_ids=p_swa)
                        if via_chunk:
                            p._announce_reshard_chunk(room, 0, [1], total_pages=1)
                        p.finalize_send(room, aux_index=None)
                    self._pump(*ps, dec)
                    self.assertTrue(dec.is_failed(room))
                    self.assertIn("5D token-major", dec.failure_records[room])
                    self.assertEqual(ps[0]._engine.pushes, [])


if __name__ == "__main__":
    unittest.main()
