# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for TPU unified KV pool speculative draft HiCache sidecars."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache import hicache_auto_size
from sglang.srt.mem_cache.hicache_storage import PoolName as PN
from sglang.srt.mem_cache.hicache_storage import PoolTransfer as PT
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import CacheOperation
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import base_spec_worker as bsw
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as SA
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import kv_pool_host as kph
from sglang_tpu import unified_kv_pool as ukp
from sglang_tpu.hooks import draft_kv_pool as dkp_hooks
from sglang_tpu.kv_pool import TpuBlockMajorMHATokenToKVPool, TpuFusedMHATokenToKVPool
from sglang_tpu.tpu_dma_hicache import page_major_tensors as pm_tensors
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _load(rel: str):
    p = pathlib.Path(__file__).resolve().parent / rel
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[p.stem] = mod
    spec.loader.exec_module(mod)
    return mod


_draft_ut = _load("test_tpu_unified_spec_draft.py")


class _InMemoryDMA:
    def __init__(self, io_backend: str, device_pool, page_num: int, *, draft_pools=()):
        self.dev = [t for p in (device_pool, *draft_pools) for t in pm_tensors(p)]
        self.host = [
            torch.zeros((page_num, *t.shape[1:]), dtype=t.dtype) for t in self.dev
        ]

    def backup(self, dp, hp):
        for d, h in zip(self.dev, self.host, strict=True):
            h[list(hp)] = d[list(dp)]
        return kph.NOOP_EVENT

    def load(self, hp, dp):
        for d, h in zip(self.dev, self.host, strict=True):
            d[list(dp)] = h[list(hp)]
        return kph.NOOP_EVENT

    destroy = lambda self: None


def make_target(size=64, full_ids=(0, 3), swa_ids=(1, 2, 4, 5, 6)):
    sp = ukp.AttnLayerSpec
    kw = {"size": size, "page_size": 4, "dtype": torch.bfloat16, "device": "cpu"}
    pool = ukp.TpuUnifiedSWAKVPool(
        full=sp(2, 128, full_ids), swa=sp(2, 128, swa_ids), **kw
    )
    alloc = ukp.TpuUnifiedSWATokenToKVPoolAllocator(
        kvcache=pool, need_sort=False, sliding_window_size=8, **kw
    )
    return pool, alloc


def attach_draft(
    alloc, full_layers=(), swa_layers=(), head_dim=64, pool_cls=TpuFusedMHATokenToKVPool
):
    fl, sl = list(full_layers), list(swa_layers)
    ft, ds = alloc.size_full, max(32, alloc.size_full // 3 // 32 * 32)
    kw = {
        "page_size": 4,
        "dtype": torch.bfloat16,
        "head_num": 4,
        "head_dim": head_dim,
        "device": "cpu",
        "enable_memory_saver": False,
    }
    pool = (
        SWAKVPool(
            size=ft,
            size_swa=ds,
            swa_attention_layer_ids=sl,
            full_attention_layer_ids=fl,
            token_to_kv_pool_class=pool_cls,
            **kw,
        )
        if sl
        else pool_cls(size=ft, layer_num=len(fl), **kw)
    )
    kvc = NS(
        is_hybrid_swa=bool(sl),
        sliding_window_size=8 if sl else None,
        model_config=NS(sliding_window_size=None),
        layer_info=NS(full_attention_layer_ids=fl, swa_attention_layer_ids=sl),
    )
    alloc.attach_draft_kv_pool(kvc, pool)
    return pool


def _fill(bufs, pages: list[int]) -> list[torch.Tensor]:
    out = []
    for b in bufs:
        v = torch.randn((len(pages), *b.shape[1:]), dtype=b.dtype)
        b[pages] = v
        out.append(v.clone())
    return out


def _pair(pool, host_size: float = 0.0):
    return kph.TpuDMAUnifiedHostKVPool.build_pair(
        pool,
        io_backend="raiden",
        host_to_device_ratio=2.0,
        host_size=host_size,
        page_size=4,
    )


def _entry(nm, hp, dp, lm, mx=7, **kw):
    return hpa.build_pool_entry(
        name=nm,
        host_pool=hp,
        device_pool=dp,
        layer_mapping=lm,
        transfer_layer_id_max=mx,
        **kw,
    )


class TestUnifiedSpecHiCacheSidecars(unittest.TestCase):
    def setUp(self) -> None:
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.addCleanup(reset_context)
        kph.register_unified_hicache_strategy()
        op, os_fn = (
            bsw.BaseSpecWorker._build_hicache_draft_plan,
            hpa.build_swa_draft_pools,
        )
        mem = NS(
            hicache_io_backend="raiden",
            hicache_mem_layout="page_first",
            hicache_ratio=2.0,
            hicache_size=0.0,
        )
        wrap_plan = lambda w: dkp_hooks.build_unified_hicache_draft_plan(op, w)
        wrap_swa = lambda *a, **kw: dkp_hooks.bind_unified_swa_draft_host_pool(
            os_fn(*a, **kw), *a, **kw
        )
        host_cls = lambda *_, **__: kph.tpu_dma_host_kv_pool_cls("raiden")
        for p in (
            patch.object(bsw.BaseSpecWorker, "_build_hicache_draft_plan", wrap_plan),
            patch.object(hpa, "build_swa_draft_pools", wrap_swa),
            patch.object(kph, "create_tpu_dma_kv_transfer", side_effect=_InMemoryDMA),
            patch.object(kph, "host_memory_budget_bytes", return_value=1 << 40),
            patch(
                "sglang.srt.platforms.current_platform.get_mha_kv_pool_host_cls",
                side_effect=host_cls,
            ),
            patch.object(hpa, "get_memory", return_value=mem),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _roundtrip(self, d_full, d_swa, swa_ids=(1, 2, 4, 5, 6)) -> None:
        tp, ta = make_target(size=128, swa_ids=swa_ids)
        dp = attach_draft(ta, d_full, d_swa)
        kv_h, swa_h = _pair(tp)
        fm, sm = kph._unified_swa_layer_mappings(tp)
        mx, sa = max(fm.keys() | sm.keys()) + 1, ta.swa_attn_allocator
        grp = HostPoolGroup(
            [
                _entry(PN.KV, kv_h, tp, fm, mx, is_anchor=True),
                _entry(
                    PN.SWA,
                    swa_h,
                    tp,
                    sm,
                    mx,
                    device_alloc_fn=sa.alloc,
                    device_free_fn=sa.free,
                ),
            ]
        )
        ctrl = object.__new__(TpuDMAHybridCacheController)
        ctrl.mem_pool_device_allocator, ctrl.mem_pool_host = ta, grp
        ctrl.page_size, ctrl.io_backend, ctrl.device = 4, "raiden", "cpu"
        ctrl.write_queue, ctrl.load_queue, ctrl.ack_write_queue, ctrl.ack_load_queue = (
            [],
            [],
            [],
            [],
        )
        ctrl.layer_done_counter = BatchDoneCounter()
        specs, entries = hpa.build_hicache_draft_sidecars(
            draft_device_pools=(dp,), tree_cache=NS(cache_controller=ctrl)
        )
        for e in entries:
            grp.add_entry(e)

        df, ds = ta.full_attn_allocator.alloc(8), sa.alloc(8)
        ta.set_full_to_swa_mapping(df, ds)
        fp, sp = (df[::4] // 4).tolist(), (ds[::4] // 4).tolist()
        t_bufs, d_sub, sc = (
            tp.kv_pool.page_buffers(),
            (dp.swa_kv_pool if d_swa else dp),
            ta.draft_swa_sidecar,
        )
        exp_tf, exp_ts = _fill(t_bufs, fp), _fill(t_bufs, tp.expand_swa_pages(sp))
        exp_d = _fill(
            pm_tensors(d_sub),
            sc.translate_pages(torch.tensor(sp)).tolist() if d_swa else fp,
        )

        hf = grp.alloc(8, pool=PN.KV)
        xfers = [PT(name=PN.SWA, device_indices=ds)] + [
            PT(name=s.pool_name, indices_from_pool=s.indices_from_pool) for s in specs
        ]
        res = grp.resolve_host_transfers(
            xfers, primary_device_indices=df, primary_host_indices=hf
        )
        hs = next(t.host_indices for t in res if t.name == PN.SWA)
        ctrl.write_queue.append(CacheOperation(hf, df, node_id=1, pool_transfers=res))
        ctrl.start_writing()
        for b in (*t_bufs, *pm_tensors(d_sub)):
            b.zero_()

        nf, ns = ta.full_attn_allocator.alloc(8), sa.alloc(8)
        ta.set_full_to_swa_mapping(nf, ns)
        r_xf = [PT(name=PN.SWA, host_indices=hs, device_indices=ns)] + [
            PT(
                name=s.pool_name,
                indices_from_pool=s.indices_from_pool,
                host_indices=hf if s.indices_from_pool == PN.KV else hs,
                device_indices=nf if s.indices_from_pool == PN.KV else ns,
            )
            for s in specs
        ]
        ctrl.load_queue.append(CacheOperation(hf, nf, node_id=2, pool_transfers=r_xf))
        ctrl.start_loading()

        nfp, nsp = (nf[::4] // 4).tolist(), (ns[::4] // 4).tolist()
        self.assertNotEqual(nfp, fp)
        ndp = sc.translate_pages(torch.tensor(nsp)).tolist() if d_swa else nfp
        for bufs, idx, exps in (
            (t_bufs, nfp, exp_tf),
            (t_bufs, tp.expand_swa_pages(nsp), exp_ts),
            (pm_tensors(d_sub), ndp, exp_d),
        ):
            for b, e in zip(bufs, exps):
                self.assertTrue(torch.equal(b[idx], e))

        st = swa_h._state
        grp.release_transfers(res)
        grp.free(hf, pool=PN.KV)
        self.assertEqual(
            (len(st._compact_swa_free), st._host_to_compact_swa),
            (st.page_num // st.num_swa_groups, {}),
        )

    def test_backup_evict_reload_and_host_free(self):
        self._roundtrip((0, 1), ())
        self._roundtrip((), (0, 1))
        self._roundtrip((), (0, 1), swa_ids=(1, 2))  # M_T == 1 compact device sidecar

    def test_copricing_plan_and_gates(self):
        with patch.object(kph, "sync_fixed_hicache_size", side_effect=lambda t, _: t):
            kv0, _ = _pair(make_target()[0], host_size=0.01)
            for fl, sl in (((), (0, 1, 2, 3)), ((0, 1), ())):
                p, a = make_target()
                d = attach_draft(a, fl, sl, head_dim=128)
                self.assertLess(_pair(p, host_size=0.01)[0].page_num, kv0.page_num)
                exp = hicache_auto_size._pool_bytes(d.swa_kv_pool if sl else d) * (
                    1 if sl else p.size // d.size
                )
                self.assertEqual(hicache_auto_size._draft_bytes(p, d), exp)

        tp, ta = make_target()
        dp = attach_draft(ta, (0,))
        r1 = NS(
            token_to_kv_pool=dp,
            model_config=NS(
                num_nextn_predict_layers=1,
                hf_config=NS(architectures=["EagleLlamaForCausalLM"]),
            ),
        )
        w = object.__new__(bsw.BaseSpecWorker)
        w._target_worker = NS(
            model_runner=NS(
                token_to_kv_pool=tp, mtp_draft_device_pools=(), spec_algorithm=SA.EAGLE
            )
        )
        w._draft_worker = NS(draft_runners=[r1])
        mem_on = NS(
            enable_hierarchical_cache=True, enable_unified_cache_external_linker=False
        )
        with patch("sglang.srt.runtime_context.get_memory", return_value=mem_on):
            plan = w._build_hicache_draft_plan()
            self.assertEqual(
                (plan.mode, plan.device_pools), (bsw.HiCacheDraftMode.SIDECAR, (dp,))
            )
            w._draft_worker = NS(draft_runners=[r1, r1])
            with self.assertRaisesRegex(NotImplementedError, "Multi-runner MTP"):
                w._build_hicache_draft_plan()

        empty_tc = NS(cache_controller=NS(mem_pool_host=NS(entry_map={})))
        self.assertEqual(
            hpa.build_swa_draft_pools(
                draft_kv_pool=attach_draft(make_target()[1], (), (0, 1)),
                tree_cache=empty_tc,
            ),
            ([], []),
        )
        bm = attach_draft(
            make_target()[1], (), (0, 1), pool_cls=TpuBlockMajorMHATokenToKVPool
        )
        self.assertEqual(bm.get_contiguous_buf_infos(), ([], [], []))

        ctx = _draft_ut.TestDraftCoPricingAndValidation()._ctx
        t_mc = _draft_ut._mc([0, 1], [2, 3, 4, 5, 6, 7])
        kvc_swa = _draft_ut._kvc(t_mc, draft_mc=_draft_ut._mc([], [0, 1], window=64))
        with ctx(max_running_requests=2, page_size=16):
            dev_only = ukp.TpuUnifiedSWAPoolConfigurator(kvc_swa).calculate_pool_sizes(
                64 << 20, page_size=16
            )
        with ctx(
            enable_hierarchical_cache=True,
            hicache_io_backend="raiden",
            max_running_requests=2,
            page_size=16,
        ):
            with_hc = ukp.TpuUnifiedSWAPoolConfigurator(kvc_swa).calculate_pool_sizes(
                64 << 20, page_size=16
            )
            self.assertGreater(
                with_hc.full_max_total_num_tokens, dev_only.full_max_total_num_tokens
            )
        kvc = _draft_ut._kvc(
            _draft_ut._mc([0, 1], [2, 3]), draft_mc=_draft_ut._mc([0], [])
        )
        with ctx(enable_hierarchical_cache=True, hicache_io_backend="raiden"):
            self.assertTrue(ukp.unified_kv_pool_enabled(kvc))


if __name__ == "__main__":
    unittest.main()
