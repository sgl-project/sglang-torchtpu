# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for TPU unified KV pool speculative draft equal-TP Raiden P/D."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric, StateType
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import unified_kv_pool as ukp
from sglang_tpu.disaggregation.raiden import common as r_common
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import UNIFIED_SWA_KV_LAYOUT, ReshardReq
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
        (pre._draft_engine, dec._draft_engine) if draft else (pre._engine, dec._engine)
    )
    ep._stats, ed._stats = ([tag], [], []), ([], [tag], [])
    pre._drain_engine()
    dec._drain_engine()


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
        p_d, d_d = _hc_ut.attach_draft(p_a, fl, sl), _hc_ut.attach_draft(d_a, fl, sl)
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

    def test_asymmetric_tp_and_gate(self):
        pre, dec, *_ = self._pair(swa_draft=True)
        with self.assertRaisesRegex(NotImplementedError, "equal attention TP"):
            dec.register_decode_request(
                9901, [1], [2], None, self._bootstrap_info(pre) * 2
            )
        req = ReshardReq(9902, "127.0.0.1", dec.rank_port, "tcp://p", [1], 1, [], "")
        with self.assertRaisesRegex(NotImplementedError, "equal attention TP"):
            pre._on_reshard_req(req.to_frames())

        ctx = _draft_ut.TestDraftCoPricingAndValidation()._ctx
        kvc = _draft_ut._kvc(
            _draft_ut._mc([0, 1], [2, 3]), draft_mc=_draft_ut._mc([0], [])
        )
        for mode in ("prefill", "decode"):
            with ctx(
                disaggregation_mode=mode, disaggregation_transfer_backend="raiden"
            ):
                self.assertTrue(ukp.unified_kv_pool_enabled(kvc))


if __name__ == "__main__":
    unittest.main()
