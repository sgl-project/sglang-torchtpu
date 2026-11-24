# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for TPU unified KV pool speculative draft equal-TP Raiden P/D."""

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

from sglang.srt.disaggregation.base.conn import (
  KVPoll,
  KVTransferMetric,
  StateType,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import unified_kv_pool as ukp
from sglang_tpu.disaggregation.raiden import common as r_common
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import (
  UNIFIED_SWA_KV_LAYOUT,
  ReshardReq,
  RoomFailed,
)
from sglang_tpu.kv_pool import TpuBlockMajorMHATokenToKVPool

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _load(path: pathlib.Path):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[path.stem] = mod
  spec.loader.exec_module(mod)
  return mod


_HERE = pathlib.Path(__file__).resolve().parent
_raiden_ut = _load(_HERE.parents[2] / "tpu/unit/test_disagg_raiden_transfer.py")
_hc_ut = _load(_HERE / "test_tpu_unified_spec_hicache.py")
_draft_ut = _hc_ut._draft_ut


def _alloc_pages(a):
  ft, st = a.full_attn_allocator.alloc(8), a.swa_attn_allocator.alloc(8)
  a.set_full_to_swa_mapping(ft, st)
  return (ft[::4] // 4).tolist(), (st[::4] // 4).tolist()


def _settle(pre, dec, tag: str, *, draft: bool = False) -> None:
  ep, ed = (
    (pre._draft_engine, dec._draft_engine)
    if draft
    else (pre._engine, dec._engine)
  )
  ep._stats, ed._stats = ([tag], [], []), ([], [tag], [])
  pre._drain_engine()
  dec._drain_engine()


def _fake_pb2_modules() -> dict[str, types.ModuleType]:
  class _Entries(list):
    def add(self):
      e = SimpleNamespace()
      self.append(e)
      return e

  pb2 = types.ModuleType("tpu_sync.rpc.raiden_service_pb2")
  pb2.MEMORY_TYPE_DRAM, pb2.MEMORY_TYPE_HBM = 1, 2
  pb2.StartTransferRequest = lambda: SimpleNamespace(
    shard_push_schedules=defaultdict(
      lambda: SimpleNamespace(entries=_Entries())
    )
  )
  tpu_sync = types.ModuleType("tpu_sync")
  tpu_sync.rpc = rpc = types.ModuleType("tpu_sync.rpc")
  rpc.raiden_service_pb2 = pb2
  return {
    "tpu_sync": tpu_sync,
    "tpu_sync.rpc": rpc,
    "tpu_sync.rpc.raiden_service_pb2": pb2,
  }


def _make_reshard_engine(addr: str):
  eng = _raiden_ut._FakeRaidenEngine()
  eng.transfer_address = addr
  eng.unregistered: list[int] = []
  eng.registered: list[tuple] = []

  def reg(uuid, req, *, is_sender=False):
    eng.registered.append((uuid, req, is_sender))
    eng.plans.append(((uuid, req), {"is_sender": is_sender}))

  def unreg(uuid):
    eng.unregistered.append(uuid)
    eng.plans.append(((uuid,), {}))

  eng.register_active_plan = reg
  eng.unregister_active_plan = unreg
  eng.plan_host_blocks = lambda uuid, blks: list(range(len(blks)))
  eng.d2h = lambda *_: SimpleNamespace(
    is_ready=lambda: True, ok=lambda: True, error_message=str
  )
  return eng


class TestUnifiedSpecRaidenPD(_raiden_ut.RaidenRendezvousTestCase):
  def setUp(self) -> None:
    super().setUp()
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="scheduler")
    self.addCleanup(reset_context)

  def _pair(self, *, swa_draft: bool, chunked: bool = False):
    fl, sl = ((), (0, 1)) if swa_draft else ((0, 1), ())
    p_t, p_a = _hc_ut.make_target(64)
    d_t, d_a = _hc_ut.make_target(64)
    p_d, d_d = (
      _hc_ut.attach_draft(p_a, fl, sl),
      _hc_ut.attach_draft(d_a, fl, sl),
    )
    pre, dec = self._prefill(), self._decode()
    for m, t_p, d_p, port in ((pre, p_t, p_d, 42000), (dec, d_t, d_d, 42008)):
      m.is_dummy_cp_rank, m._listener_start_read = False, False
      m.kv_args.state_types = [StateType.SWA]
      m.set_kv_caches(r_common.get_raiden_kv_caches(t_p, draft_pool=d_p))
      m._engine, m._draft_engine = (
        _raiden_ut._FakeRaidenEngine(),
        _raiden_ut._FakeRaidenEngine(),
      )
      m._full_engine_max_blocks = m._draft_engine_max_blocks = 64
      m._pulling = StagingAdmission(64, draft_capacity=64)
      m.raiden_endpoint = f"127.0.0.1:{port}"
      m.draft_raiden_endpoint = f"127.0.0.1:{port + r_common.DRAFT_PORT_OFFSET}"
      m.chunked_transfer = chunked
    return pre, dec, p_a, d_a

  def _pump(self, *mgrs) -> None:
    progressed = True
    while progressed:
      progressed = False
      for m in mgrs:
        while self.inboxes[m.endpoint].frames:
          m._dispatch(self.inboxes[m.endpoint].frames.pop(0))
          progressed = True

  def test_draft_engine_rendezvous_whole_and_chunked(self):
    self.assertEqual(r_common.PORTS_PER_RANK, 8)
    bm_t, bm_a = _hc_ut.make_target()
    bm_d = _hc_ut.attach_draft(
      bm_a, (), (0, 1), pool_cls=TpuBlockMajorMHATokenToKVPool
    )
    self.assertEqual(
      len(r_common.get_raiden_kv_caches(bm_t, draft_pool=bm_d).draft), 1
    )
    i32 = lambda xs: np.asarray(xs, dtype=np.int32)

    for swa_draft in (False, True):
      for chunked in (False, True):
        pre, dec, p_a, d_a = self._pair(swa_draft=swa_draft, chunked=chunked)
        self.assertEqual(pre.kv_layout, UNIFIED_SWA_KV_LAYOUT)
        room = 9100 + int(swa_draft) * 10 + int(chunked)
        _ = d_a.full_attn_allocator.alloc(4)
        (p_fp, p_sp), (d_fp, d_sp) = _alloc_pages(p_a), _alloc_pages(d_a)

        pre.request_status[room] = dec.request_status[room] = (
          KVPoll.Bootstrapping
        )
        rv = object.__new__(RaidenKVReceiver)
        rv.kv_mgr, rv.bootstrap_room, rv.conclude_state, rv.bootstrap_infos = (
          dec,
          room,
          None,
          self._bootstrap_info(pre),
        )
        rv.send_metadata(i32(d_fp), aux_index=None, state_indices=[i32(d_sp)])
        self._pump(pre, dec)

        sd = object.__new__(RaidenKVSender)
        sd.kv_mgr, sd.bootstrap_room, sd._started = pre, room, False
        sd.aux_index = sd._transfer_start_time = sd.conclude_state = None
        sd._transfer_metric = KVTransferMetric()
        sd._transfer_num_kv_indices = sd._transfer_num_state_indices = 0
        pre.get_kv_replica_factor = lambda: 1
        if chunked:
          pre._prefill.announce_pages, sd.num_kv_indices, sd._last_chunk = (
            1,
            2,
            False,
          )
          sd.send(i32(p_fp[:1]), state_indices=None)
          self._pump(pre, dec)
          self.assertTrue(dec.maybe_start_pull(room))
          _settle(pre, dec, f"{room}#0")
          sd._last_chunk = True
          sd.send(i32(p_fp[1:]), state_indices=[i32(p_sp)])
        else:
          sd.num_kv_indices, sd._last_chunk = 2, True
          sd.send(i32(p_fp), state_indices=[i32(p_sp)])

        self._pump(pre, dec)
        self.assertTrue(dec.maybe_start_pull(room))
        tr = lambda a, sp, fp: (
          a.draft_swa_sidecar.translate_pages(torch.tensor(sp)).tolist()
          if swa_draft
          else fp
        )
        exp_src, exp_dst = tr(p_a, p_sp, p_fp), tr(d_a, d_sp, d_fp)
        self.assertEqual(
          pre._draft_engine.reads, [(f"draft:{room}", room, exp_src)]
        )
        self.assertEqual(
          [a for a, _ in dec._draft_engine.started],
          [
            (
              f"draft:{room}",
              room,
              pre.draft_raiden_endpoint,
              exp_src,
              exp_dst,
            )
          ],
        )

        _settle(pre, dec, str(room))
        self.assertFalse(pre.is_send_done(room) or dec.is_recv_done(room))
        _settle(pre, dec, f"draft:{room}", draft=True)
        self.assertTrue(pre.is_send_done(room) and dec.is_recv_done(room))
        self.assertEqual(len(dec._pulling), 0)

  def _reshard_cluster(
    self,
    *,
    swa_draft: bool,
    chunked: bool = False,
    src_tp: int = 2,
    dst_tp: int = 1,
  ):
    fl, sl = ((), (0, 1)) if swa_draft else ((0, 1), ())
    prefills, p_allocs = [], []
    for rank in range(src_tp):
      p_t, p_a = _hc_ut.make_target(64)
      p_d = _hc_ut.attach_draft(p_a, fl, sl)
      m = self._prefill(attn_tp_rank=rank, attn_tp_size=src_tp)
      m.is_dummy_cp_rank, m._listener_start_read = False, False
      m.kv_args.state_types = [StateType.SWA]
      m.kv_args.total_kv_head_num = 2 * src_tp
      m.kv_args.total_swa_kv_head_num = 2 * src_tp
      m.set_kv_caches(r_common.get_raiden_kv_caches(p_t, draft_pool=p_d))
      m._engine = _make_reshard_engine(f"127.0.0.1:{43000 + rank * 8}")
      m._draft_engine = _make_reshard_engine(
        f"127.0.0.1:{43000 + rank * 8 + r_common.DRAFT_PORT_OFFSET}"
      )
      m._full_engine_max_blocks = m._draft_engine_max_blocks = 64
      m.raiden_endpoint = m._engine.transfer_address
      m.draft_raiden_endpoint = m._draft_engine.transfer_address
      m.chunked_transfer = chunked
      m._push_executor.shutdown(wait=True)
      m._push_executor = SimpleNamespace(
        submit=lambda fn, *a, **k: fn(*a, **k), shutdown=lambda **_: None
      )
      prefills.append(m)
      p_allocs.append(p_a)

    ratio = src_tp // dst_tp
    d_t_wide = ukp.TpuUnifiedSWAKVPool(
      size=64,
      page_size=4,
      dtype=torch.bfloat16,
      full=ukp.AttnLayerSpec(2 * ratio, 128, (0, 3)),
      swa=ukp.AttnLayerSpec(2 * ratio, 128, (1, 2, 4, 5, 6)),
      device="cpu",
    )
    d_a = ukp.TpuUnifiedSWATokenToKVPoolAllocator(
      kvcache=d_t_wide,
      size=64,
      page_size=4,
      dtype=torch.bfloat16,
      device="cpu",
      need_sort=False,
      sliding_window_size=8,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    draft_kw = dict(
      page_size=4,
      dtype=torch.bfloat16,
      head_num=4 * ratio,
      head_dim=64,
      device="cpu",
      enable_memory_saver=False,
    )
    d_d = (
      SWAKVPool(
        size=64,
        size_swa=32,
        swa_attention_layer_ids=list(sl),
        full_attention_layer_ids=list(fl),
        token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        **draft_kw,
      )
      if sl
      else TpuFusedMHATokenToKVPool(size=64, layer_num=len(fl), **draft_kw)
    )
    d_a.attach_draft_kv_pool(
      SimpleNamespace(
        is_hybrid_swa=bool(sl),
        sliding_window_size=8 if sl else None,
        model_config=SimpleNamespace(sliding_window_size=None),
        layer_info=SimpleNamespace(
          full_attention_layer_ids=list(fl), swa_attention_layer_ids=list(sl)
        ),
      ),
      d_d,
    )
    dec = self._decode(attn_tp_rank=0, attn_tp_size=dst_tp)
    dec.is_dummy_cp_rank, dec._listener_start_read = False, False
    dec.kv_args.state_types = [StateType.SWA]
    dec.kv_args.total_kv_head_num = 2 * src_tp
    dec.kv_args.total_swa_kv_head_num = 2 * src_tp
    dec.set_kv_caches(r_common.get_raiden_kv_caches(d_t_wide, draft_pool=d_d))
    dec._engine = _make_reshard_engine("127.0.0.1:44000")
    dec._draft_engine = _make_reshard_engine(
      f"127.0.0.1:{44000 + r_common.DRAFT_PORT_OFFSET}"
    )
    dec._full_engine_max_blocks = dec._draft_engine_max_blocks = 64
    dec.raiden_endpoint = dec._engine.transfer_address
    dec.draft_raiden_endpoint = dec._draft_engine.transfer_address
    dec.chunked_transfer = chunked
    return prefills, dec, p_allocs, d_a

  def test_asymmetric_tp_and_gate(self):
    i32 = lambda xs: np.asarray(xs, dtype=np.int32)
    with patch.dict(sys.modules, _fake_pb2_modules()):
      for swa_draft in (False, True):
        for chunked in (False, True):
          with self.subTest(swa_draft=swa_draft, chunked=chunked):
            prefills, dec, p_allocs, d_a = self._reshard_cluster(
              swa_draft=swa_draft, chunked=chunked
            )
            room = 9900 + int(swa_draft) * 10 + int(chunked)
            _ = d_a.full_attn_allocator.alloc(4)
            p_pages = [_alloc_pages(a) for a in p_allocs]
            d_fp, d_sp = _alloc_pages(d_a)
            infos = [self._bootstrap_info(p)[0] for p in prefills]

            dec.request_status[room] = KVPoll.Bootstrapping
            for p in prefills:
              p.request_status[room] = KVPoll.Bootstrapping

            rv = object.__new__(RaidenKVReceiver)
            rv.kv_mgr, rv.bootstrap_room, rv.conclude_state = (
              dec,
              room,
              None,
            )
            rv.bootstrap_infos = infos
            rv.send_metadata(
              i32(d_fp), aux_index=None, state_indices=[i32(d_sp)]
            )
            self._pump(*prefills, dec)

            for p, (p_fp, p_sp) in zip(prefills, p_pages, strict=True):
              sd = object.__new__(RaidenKVSender)
              sd.kv_mgr, sd.bootstrap_room, sd._started = p, room, False
              sd.aux_index = sd._transfer_start_time = sd.conclude_state = None
              sd._transfer_metric = KVTransferMetric()
              sd._transfer_num_kv_indices = sd._transfer_num_state_indices = 0
              p.get_kv_replica_factor = lambda: 1
              if chunked:
                p._prefill.announce_pages = 1
                sd.num_kv_indices, sd._last_chunk = 2, False
                sd.send(i32(p_fp[:1]), state_indices=None)
                self._pump(*prefills, dec)
                sd._last_chunk = True
                sd.send(i32(p_fp[1:]), state_indices=[i32(p_sp)])
              else:
                sd.num_kv_indices, sd._last_chunk = 2, True
                sd.send(i32(p_fp), state_indices=[i32(p_sp)])
            self._pump(*prefills, dec)

            tr = lambda a, sp, fp: (
              a.draft_swa_sidecar.translate_pages(torch.tensor(sp)).tolist()
              if swa_draft
              else fp
            )
            exp_dst = tr(d_a, d_sp, d_fp)
            self.assertEqual(len(dec._draft_engine.registered), 1)
            self.assertEqual(dec._draft_engine.registered[0][0], room)
            for p, p_a, (p_fp, p_sp) in zip(
              prefills, p_allocs, p_pages, strict=True
            ):
              exp_src = tr(p_a, p_sp, p_fp)
              self.assertEqual(
                len(p._draft_engine.pushes), len(p._draft_kv_caches)
              )
              for args, _ in p._draft_engine.pushes:
                self.assertEqual(args[0], room)
                self.assertEqual(args[1], dec.draft_raiden_endpoint)
                self.assertEqual(args[2], exp_src)
                self.assertEqual(args[3], exp_dst)
              self.assertTrue(p.is_send_done(room))

            # Receiver settles only after both full/chunk and draft plans complete.
            if chunked:
              dec._engine._stats = ([], [f"{room}#0", str(room)], [])
            else:
              dec._engine._stats = ([], [str(room)], [])
            dec._drain_engine()
            self.assertFalse(dec.is_recv_done(room))
            dec._draft_engine._stats = ([], [f"draft:{room}"], [])
            dec._drain_engine()
            self.assertTrue(dec.is_recv_done(room))

    ctx = _draft_ut.TestDraftCoPricingAndValidation()._ctx
    kvc = _draft_ut._kvc(
      _draft_ut._mc([0, 1], [2, 3]), draft_mc=_draft_ut._mc([0], [])
    )
    for mode in ("prefill", "decode"):
      with ctx(
        disaggregation_mode=mode, disaggregation_transfer_backend="raiden"
      ):
        self.assertTrue(ukp.unified_kv_pool_enabled(kvc))

    # Thread #3: public draft_swa_sidecar property & typed RaidenKVCacheGroups.
    tp_s, ta_s = _hc_ut.make_target(size=64)
    dp_s = _hc_ut.attach_draft(ta_s, (), (0, 1))
    self.assertIs(tp_s.draft_swa_sidecar, ta_s.draft_swa_sidecar)
    grps = r_common.get_raiden_kv_caches(tp_s, draft_pool=dp_s)
    self.assertIs(grps.draft_swa_sidecar, ta_s.draft_swa_sidecar)
    self.assertTrue(all(isinstance(t, torch.Tensor) for t in grps.draft))

    # Thread #1: non-spec rooms (_draft_engine is None) skip empty closing reads.
    _, dec_ns, *_ = self._pair(swa_draft=False)
    dec_ns._draft_engine = None
    dec_ns._unified_swa_pool = None
    room_ns = 9903
    dec_ns._decode.local_blocks[room_ns] = [20]
    dec_ns._decode.remote_blocks[room_ns] = [10, 11]
    dec_ns._decode.remote_total[room_ns] = 2
    dec_ns._decode.chunk_queue[room_ns] = [
      (str(room_ns), room_ns, [0], [10], "tcp://p")
    ]
    self.assertFalse(dec_ns._start_chunked_pull(room_ns))
    self.assertEqual(dec_ns._engine.started, [])
    self.assertEqual(dec_ns._decode.chunk_queue[room_ns], [])

  def test_reshard_full_prefix_hit_settles_draft_and_final_seen(self):
    for swa_draft in (False, True):
      for chunked in (False, True):
        with self.subTest(swa_draft=swa_draft, chunked=chunked):
          prefills, dec, *_ = self._reshard_cluster(
            swa_draft=swa_draft, chunked=chunked
          )
          infos = [self._bootstrap_info(p)[0] for p in prefills]
          room = 9920 + int(swa_draft) * 10 + int(chunked)
          dec.register_decode_request(
            room, [], [], None, infos, local_draft_swa_blocks=[]
          )
          self.assertTrue(
            all(not self.inboxes[p.endpoint].frames for p in prefills)
          )
          self.assertIn(room, dec._decode.final_seen)
          self.assertIn(f"draft:{room}", dec._completed_recv)
          self.assertTrue(dec.is_recv_done(room))
          if swa_draft:
            room_dswa = room + 100
            dec.request_status[room_dswa] = KVPoll.Bootstrapping
            dec.register_decode_request(
              room_dswa, [], [], None, infos, local_draft_swa_blocks=[3]
            )
            self.assertTrue(
              all(len(self.inboxes[p.endpoint].frames) == 1 for p in prefills)
            )

  def test_reshard_abort_mid_push_unregisters_only_settled_engine(self):
    prefills, dec, *_ = self._reshard_cluster(swa_draft=False, chunked=False)
    room = 9950
    dec.request_status[room] = KVPoll.WaitingForInput
    dec._reshard_plans[room] = {str(room): room, f"draft:{room}": room}
    dec._completed_recv.add(str(room))

    # str(room) settled while draft:{room} is still in flight on _draft_engine.
    dec.clear_room(room)
    self.assertEqual(dec._engine.unregistered, [room])
    self.assertEqual(dec._draft_engine.unregistered, [])
    self.assertEqual(dec._cleared_plans, {f"draft:{room}": room})

    # Once draft:{room} settles, only _draft_engine unregisters it.
    dec._draft_engine._stats = ([], [f"draft:{room}"], [])
    dec._drain_engine()
    self.assertEqual(dec._engine.unregistered, [room])
    self.assertEqual(dec._draft_engine.unregistered, [room])
    self.assertEqual(dec._cleared_plans, {})

    # If _publish_reshard_plan fails on _engine, the armed draft plan rolls back.
    room_rb = 9951
    dec._engine.unregistered.clear()
    dec._draft_engine.unregistered.clear()
    dec.request_status[room_rb] = KVPoll.WaitingForInput
    dec._decode.reshard_meta[room_rb] = dict(
      local_blocks=[1, 2],
      local_swa_blocks=[3, 4],
      local_draft_swa_blocks=[],
      source_ranks=[0, 1],
      src_attn_tp=2,
      aux_index=None,
      ready_endpoints=[p.endpoint for p in prefills],
    )
    dec._decode.reshard_sources[room_rb] = {0: [1, 2], 1: [1, 2]}
    dec._decode.reshard_swa_sources[room_rb] = {0: [3, 4], 1: [3, 4]}
    dec._decode.reshard_draft_swa_sources[room_rb] = {0: [], 1: []}

    def fail_reg(*_a, **_k):
      raise RuntimeError("staging full")

    dec._engine.register_active_plan = fail_reg
    with patch.dict(sys.modules, _fake_pb2_modules()):
      dec._register_receiver_and_signal(room_rb)
    self.assertEqual(dec._draft_engine.unregistered, [room_rb])
    self.assertFalse(dec.room_has_active_io(room_rb))
    self.assertTrue(dec.is_failed(room_rb))

  def test_reshard_draft_mismatch_fails_room_via_room_failed(self):
    cases = (
      ("missing_endpoint", dict(draft_push_endpoint="")),
      ("layout_mismatch", dict(decode_draft_seq_along_lane=True)),
      ("swa_mode_mismatch", dict(decode_draft_is_swa=True)),
      ("zero_draft_heads", dict(decode_draft_kv_head_num=0)),
      ("unexpected_endpoint", dict(strip_prefill_draft=True)),
    )
    for idx, (name, overrides) in enumerate(cases):
      with self.subTest(case=name):
        prefills, dec, *_ = self._reshard_cluster(
          swa_draft=False, chunked=False
        )
        pre = prefills[0]
        if overrides.get("strip_prefill_draft"):
          pre._draft_kv_caches = ()
          pre._draft_engine = None
        room = 9960 + idx
        pre.request_status[room] = dec.request_status[room] = (
          KVPoll.WaitingForInput
        )
        dec._decode.reshard_meta[room] = dict(
          local_blocks=[1],
          local_swa_blocks=[2],
          local_draft_swa_blocks=[],
          source_ranks=[0, 1],
          src_attn_tp=2,
          aux_index=None,
          ready_endpoints=[pre.endpoint],
        )
        fields = dict(
          room=room,
          decode_ip=dec.local_ip,
          decode_port=dec.rank_port,
          push_endpoint=dec.raiden_endpoint,
          decode_blocks=[1],
          dst_attn_tp=dec.attn_tp_size,
          decode_swa_blocks=[2],
          swa_push_endpoint="",
          draft_push_endpoint=dec.draft_raiden_endpoint,
          decode_draft_swa_blocks=[],
          decode_draft_kv_head_num=dec._draft_kv_head_num,
          decode_draft_seq_along_lane=bool(dec._draft_seq_along_lane),
          decode_draft_is_swa=bool(dec._draft_is_swa),
        )
        for k, v in overrides.items():
          if k != "strip_prefill_draft":
            fields[k] = v
        pre._on_reshard_req(ReshardReq(**fields).to_frames())
        self.assertEqual(pre.check_status(room), KVPoll.Failed)
        self.assertNotIn(room, pre._prefill.reshard_req)
        frames = self.inboxes[dec.endpoint].take(RoomFailed.TAG)
        dec._dispatch(frames)
        self.assertTrue(dec.is_failed(room))
        self.assertIn(room, dec.failure_records)


if __name__ == "__main__":
  unittest.main()
