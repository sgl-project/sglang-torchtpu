# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EAGLE/EAGLE3/MTP/DFLASH/DSPARK draft KV caches on TpuUnifiedSWAKVPool."""

from __future__ import annotations

import types
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.mem_cache.allocator import swa
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as SA
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import unified_kv_pool as ukp
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.dflash_draft_kv_write import _scatter_layer_kv
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.speculative.dflash.dflash_draft_worker import (
    TpuDflashDraftWorker,
)
from sglang_tpu.hooks import draft_kv_pool as dkp_hooks
from sglang_tpu.kv_free_fence import install_kv_free_fence
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _mc(
    full_ids,
    swa_ids,
    *,
    h=2,
    d=128,
    window=128,
    arch=AttentionArch.MHA,
    asym=False,
    nextn=None,
):
    win = window if swa_ids else None
    n_layers = len(full_ids) + len(swa_ids)
    hf = types.SimpleNamespace(architectures=["Dummy"], sliding_window=win)
    hf.get_text_config = lambda: hf
    return types.SimpleNamespace(
        dtype=torch.bfloat16,
        attention_arch=arch,
        has_asymmetric_kv=asym,
        is_hybrid_swa=bool(swa_ids),
        is_deepseek_v4_arch=False,
        linear_attn_registry_result=None,
        num_hidden_layers=n_layers,
        num_attention_layers=n_layers,
        num_nextn_predict_layers=nextn,
        full_attention_layer_ids=list(full_ids),
        swa_attention_layer_ids=list(swa_ids),
        head_dim=d,
        v_head_dim=d,
        swa_head_dim=d,
        swa_v_head_dim=d,
        sliding_window_size=win,
        hf_config=hf,
        hf_text_config=hf,
        get_num_kv_heads=lambda *_: h,
        get_swa_num_kv_heads=lambda *_: h,
    )


def _kvc(
    t_mc,
    *,
    algo=SA.EAGLE,
    draft_mc=None,
    runners=1,
    ps=16,
    dt="bf16",
    pool_cls=TpuFusedMHATokenToKVPool,
    draft_backend=None,
):
    n = 0 if draft_mc is None else runners * draft_mc.num_hidden_layers
    aux = (
        None
        if draft_mc is None
        else types.SimpleNamespace(
            draft_model_config=draft_mc,
            dflash_draft_num_layers=n if algo.is_dflash_family() else 0,
            eagle_draft_num_layers=0 if algo.is_dflash_family() else n,
            eagle_draft_swa_num_layers=runners * len(draft_mc.swa_attention_layer_ids),
        )
    )
    li = types.SimpleNamespace(
        full_attention_layer_ids=list(t_mc.full_attention_layer_ids),
        swa_attention_layer_ids=list(t_mc.swa_attention_layer_ids),
    )
    return types.SimpleNamespace(
        device="cpu",
        server_args=types.SimpleNamespace(speculative_draft_model_path="d"),
        is_hybrid_swa=True,
        use_mla_backend=False,
        mambaish_config=None,
        hybrid_gdn_config=None,
        hybrid_kda_config=None,
        is_draft_worker=False,
        spec_algorithm=algo,
        spec_aux_config=aux,
        pp_size=1,
        speculative_draft_attention_backend=draft_backend,
        _tpu_mha_pool_class=lambda: pool_cls,
        kv_cache_dtype_str=dt,
        kv_cache_dtype=torch.int8 if dt == "mxint8" else torch.bfloat16,
        kv_group_size=32 if dt == "mxint8" else None,
        pool_page_size=ps,
        page_size=ps,
        sliding_window_size=127,
        model_config=t_mc,
        layer_info=li,
    )


class TestDraftCoPricingAndValidation(unittest.TestCase):
    @contextmanager
    def _ctx(self, *, exp_spec=False, **overrides):
        args = {
            "attention_backend": "pallas_tpu_rpa_v3",
            "tp_size": 1,
            "pp_size": 1,
            "dcp_size": 1,
            "attn_cp_size": 1,
            "disaggregation_mode": "null",
            "enable_hierarchical_cache": False,
            "page_size": 16,
            "enable_unified_cache_external_linker": False,
            "kv_cache_dtype": "auto",
            "speculative_draft_kv_cache_dtype": "auto",
            "speculative_algorithm": "EAGLE",
            **overrides,
        }
        envs = (
            (tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL, True),
            (tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER, exp_spec),
            (tpu_envs.SGLANG_TPU_HOST_PADDING_NO_D2H, exp_spec),
        )
        try:
            with ExitStack() as st:
                for ev, val in envs:
                    st.enter_context(ev.override(val))
                st.enter_context(get_context().override_server_args(**args))
                st.enter_context(
                    get_parallel().override(
                        attn_tp_size=1, attn_dcp_size=1, dcp_enabled=False
                    )
                )
                yield
        finally:
            for ev, _ in envs:
                ev.clear()

    def test_copricing_and_model_config_resolution(self):
        from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
            resolve_spec_aux_hidden_state_config,
        )

        t_mc = _mc([0, 1], [2, 3, 4, 5, 6, 7])  # M_T = 3
        budget = 64 * 1024 * 1024

        # Non-spec target with sliding_window_size=None keeps None (does not fall back
        # to model_config.sliding_window_size for min-token / max_request_tokens).
        with self._ctx(speculative_algorithm=None):
            ns_kvc = _kvc(t_mc, algo=SA.NONE)
            ns_kvc.sliding_window_size = None
            ns_conf = ukp.TpuUnifiedSWAPoolConfigurator(ns_kvc)
            self.assertIsNone(ns_conf._sliding_window_size)

        for algo_s, d_path, srv_kw, nextn in (
            ("EAGLE3", "mock/e3", {}, None),
            ("EAGLE", None, {"model_path": "mock/step3p5", "revision": "r1"}, 1),
        ):
            t_cur = _mc([0, 1], [2, 3, 4, 5, 6, 7], nextn=nextn)
            d_srv = _mc([], [0], h=1, d=64, window=64, nextn=nextn)
            with self._ctx(
                speculative_algorithm=algo_s, speculative_draft_model_path=d_path
            ):
                with patch.object(
                    ModelConfig, "from_server_args", return_value=d_srv
                ) as mf:
                    kvc = _kvc(t_cur, algo=SA[algo_s])
                    if srv_kw:
                        kvc.server_args = types.SimpleNamespace(**srv_kw)
                    kvc.spec_aux_config = resolve_spec_aux_hidden_state_config(
                        server_args=kvc.server_args,
                        model_config=t_cur,
                        spec_algorithm=SA[algo_s],
                        is_draft_worker=False,
                    )
                    self.assertIs(kvc.spec_aux_config.draft_model_config, d_srv)
                    self.assertTrue(ukp.unified_kv_pool_enabled(kvc))
                    self.assertGreater(
                        ukp.resolve_unified_draft_kv_pricing(kvc).swa_compact_per_token,
                        0,
                    )
                    mf.assert_called_once()
                    self.assertTrue(mf.call_args.kwargs["is_draft_model"])

        for algo, df, ds, dh, dd, ps, dt, runners in (
            (SA.EAGLE, [0, 1], [], 1, 64, 16, "bf16", 1),
            (SA.EAGLE, [], [0], 2, 64, 16, "bf16", 2),
            (SA.EAGLE, [], [0, 1], 2, 128, 128, "mxint8", 1),
            (SA.DFLASH, [0, 1, 2, 3], [], 2, 128, 16, "bf16", 1),
            (SA.DSPARK, [], [0, 1], 2, 128, 16, "bf16", 1),
        ):
            kv_arg = dt if dt == "mxint8" else "auto"
            with self._ctx(
                exp_spec=algo.is_dflash_family(),
                speculative_algorithm=algo.name,
                kv_cache_dtype=kv_arg,
                speculative_draft_kv_cache_dtype=kv_arg,
                page_size=ps,
            ):
                d_mc = _mc(df, ds, h=dh, d=dd, window=64)
                kvc = _kvc(
                    t_mc, algo=algo, draft_mc=d_mc, runners=runners, ps=ps, dt=dt
                )
                conf = ukp.TpuUnifiedSWAPoolConfigurator(kvc)
                self.assertEqual(conf._align_pages, 8)
                n_tok = conf.calculate_pool_sizes(
                    budget, page_size=ps
                ).full_max_total_num_tokens
                self.assertEqual((n_tok // ps) % conf._align_pages, 0)
                n_pg = n_tok // ps + 1
                d_pg = (
                    ukp._compute_draft_swa_pages(
                        n_tok // ps,
                        num_swa_groups=3,
                    )
                    + 1
                    if ds
                    else 0
                )
                self.assertLessEqual(
                    n_pg * conf._page_bytes + d_pg * conf._draft_swa_page_bytes, budget
                )

        try:
            with self._ctx(
                exp_spec=True,
                speculative_algorithm="DFLASH",
                page_size=128,
                attention_backend="pallas_tpu_rpa_batched",
                speculative_draft_attention_backend="pallas_tpu_rpa_v3",
            ):
                with tpu_envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(True):
                    with patch.object(
                        TpuSeqAlongLaneKVPool,
                        "kv_bytes_per_token_per_layer",
                        return_value=512,
                    ):
                        ks = _kvc(
                            t_mc,
                            algo=SA.DFLASH,
                            draft_mc=_mc([], [0, 1], window=64),
                            ps=128,
                            pool_cls=TpuSeqAlongLaneKVPool,
                            draft_backend="pallas_tpu_rpa_v3",
                        )
                        cs = ukp.TpuUnifiedSWAPoolConfigurator(ks)
                        self.assertEqual(
                            (cs._full_per_token, cs._align_pages), (512, 8)
                        )
        finally:
            tpu_envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.clear()

    def test_unsupported_draft_configs_rejected(self):
        t_mc = _mc([0, 1], [2, 3])
        d_fa, d_swa = _mc([0, 1], [], d=64), _mc([], [0, 1], window=64)
        with self._ctx():
            for bad in (
                _mc([0], [], arch=AttentionArch.MLA),
                _mc([0], [], asym=True),
                _mc([0], [1], window=64),
                _mc([], [0], window=512),
                _mc([], [0], window=None),
            ):
                with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
                    ukp.unified_kv_pool_enabled(_kvc(t_mc, draft_mc=bad))
            for algo in (SA.FROZEN_KV_MTP, SA.STANDALONE, SA.NGRAM):
                bk = _kvc(t_mc, algo=algo, draft_mc=d_fa)
                self.assertEqual(
                    ukp.resolve_unified_draft_kv_pricing(bk), ukp.DraftKVPricing()
                )
                with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
                    ukp.unified_kv_pool_enabled(bk)

        with self._ctx(
            speculative_algorithm="EAGLE3",
            speculative_draft_attention_backend="pallas_tpu_flash",
        ):
            kf = _kvc(
                t_mc, algo=SA.EAGLE3, draft_mc=d_fa, draft_backend="pallas_tpu_flash"
            )
            self.assertEqual(
                ukp.resolve_unified_draft_kv_pricing(kf).full_per_token, 1024
            )
            with self.assertRaisesRegex(
                ValueError, "TpuFusedMHATokenToKVPool draft pool subclass"
            ):
                ukp.unified_kv_pool_enabled(kf)

        def _chk(
            algo, dm, pat, *, env=None, pool_cls=TpuFusedMHATokenToKVPool, ps=16, **kw
        ):
            try:
                with ExitStack() as st:
                    st.enter_context(
                        self._ctx(speculative_algorithm=algo.name, page_size=ps, **kw)
                    )
                    if env is not None:
                        st.enter_context(env.override(True))
                    with self.assertRaisesRegex(ValueError, pat):
                        ukp.unified_kv_pool_enabled(
                            _kvc(t_mc, algo=algo, draft_mc=dm, ps=ps, pool_cls=pool_cls)
                        )
            finally:
                if env is not None:
                    env.clear()

        _chk(SA.DSPARK, d_fa, "EXPERIMENTAL_SPEC_WORKER", exp_spec=False)
        _chk(SA.DFLASH, d_swa, "EXPERIMENTAL_SPEC_WORKER", exp_spec=False)
        _chk(
            SA.DFLASH,
            _mc([0], []),
            "non-MXINT8",
            exp_spec=True,
            ps=128,
            speculative_draft_kv_cache_dtype="mxint8",
        )
        _chk(
            SA.DFLASH,
            d_fa,
            "TpuFusedMHATokenToKVPool draft pool",
            exp_spec=True,
            env=tpu_envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV,
            pool_cls=TpuSeqAlongLaneKVPool,
            attention_backend="pallas_tpu_rpa_batched",
        )
        _chk(
            SA.DFLASH,
            d_fa,
            "--speculative-draft-window-size",
            exp_spec=True,
            speculative_draft_window_size=64,
        )
        _chk(
            SA.DSPARK,
            d_fa,
            "SGLANG_TPU_PD_DECODE_DRAFT_PREFILL",
            exp_spec=True,
            env=tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL,
        )


class TestDraftSwaSidecarAndFence(unittest.TestCase):
    def setUp(self):
        for p in (
            patch.object(swa, "_is_tpu", True),
            patch.object(
                swa,
                "TpuPagedTokenToKVPoolAllocator",
                TpuPagedTokenToKVPoolAllocator,
                create=True,
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _build_target(self, num_pages=24, m_swa=3):
        sz = num_pages * 8
        spec = ukp.AttnLayerSpec
        tp = ukp.TpuUnifiedSWAKVPool(
            size=sz,
            page_size=8,
            dtype=torch.bfloat16,
            full=spec(1, 64, [0, 1]),
            swa=spec(1, 64, list(range(2, 2 + 2 * m_swa))),
            device="cpu",
        )
        rp = ReqToTokenPool(
            size=8, max_context_len=256, device="cpu", enable_memory_saver=False
        )
        ta = ukp.TpuUnifiedSWATokenToKVPoolAllocator(
            size=sz,
            page_size=8,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=tp,
            need_sort=False,
            req_to_token_pool=rp,
            sliding_window_size=16,
        )
        return ta, rp

    def _attach(self, ta, rp, *, full_ids=(), swa_ids=(0, 1), window=8):
        li = types.SimpleNamespace(
            full_attention_layer_ids=list(full_ids),
            swa_attention_layer_ids=list(swa_ids),
        )
        dk = types.SimpleNamespace(
            is_draft_worker=True,
            is_hybrid_swa=bool(swa_ids),
            sliding_window_size=window,
            token_to_kv_pool_allocator=ta,
            layer_info=li,
            model_config=types.SimpleNamespace(sliding_window_size=None),
        )
        cfg = dkp_hooks.derive_unified_draft_pool_sizes(
            lambda _k, *, config: config, dk, config=MemoryPoolConfig(1, 1, 1)
        )
        dp = SWAKVPool(
            size=cfg.full_max_total_num_tokens,
            size_swa=cfg.swa_max_total_num_tokens,
            page_size=8,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=list(swa_ids),
            full_attention_layer_ids=list(full_ids),
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        res = dkp_hooks.attach_unified_draft_kv_pool(
            lambda *_, **__: None,
            dk,
            token_to_kv_pool=dp,
            req_to_token_pool=rp,
            token_to_kv_pool_allocator=ta,
        )
        self.assertIs(res, ta)
        return dp

    def test_sidecar_lifecycle_fence_and_block_draft_scatter(self):
        ta, rp = self._build_target(32, 3)
        dp = self._attach(ta, rp)
        sc = ta.draft_swa_sidecar
        self.assertEqual((dp.size, dp.size_swa), (256, 64))

        fence = install_kv_free_fence(
            ta,
            synchronize_device=lambda: None,
            spec_algorithm=SA.EAGLE,
            copies_draft_tail=True,
        )
        fence.attach(sc.allocator)
        res = types.SimpleNamespace(copy_done=None)
        fence.on_batch_formed()
        fence.on_forward_launched(res)
        z0, neg1, s16, s20 = (
            torch.tensor([0]),
            torch.tensor([-1]),
            torch.tensor([16]),
            torch.tensor([20]),
        )
        with fence.forward_write_target():
            ext = ta.alloc_extend(z0, z0, s20, s20, neg1, 20)
            rp.req_to_token[0, :20] = ext
            self.assertEqual(
                len(set(dp.translate_loc_from_full_to_swa(ext).tolist())), 20
            )
            self.assertEqual(int(dp.translate_loc_from_full_to_swa(neg1).item()), -1)
            ta.free_swa(ext[:16])
            self.assertTrue(torch.all(dp.full_to_swa_index_mapping[ext[:16]] == 0))

        self.assertEqual(sc.allocator.held_pages.num_pages, 2)
        ext2 = ta.alloc_extend(z0, z0, s16, s16, neg1, 16)
        self.assertEqual(fence.stats.device_syncs, 0)
        fence.on_forward_done(res)
        self.assertEqual(sc.allocator.held_pages.num_pages, 0)

        for pos in range(20, 25):
            st = torch.tensor([pos + 1])
            dec = ta.alloc_decode(st, st, rp.req_to_token[0, pos - 1 : pos])
            rp.req_to_token[0, pos] = dec[0]
            self.assertTrue(8 <= int(dp.translate_loc_from_full_to_swa(dec)[0]) < 72)

        ta.free(rp.req_to_token[0, :25])
        ta.free(ext2)
        self.assertEqual(
            (sc.allocator.available_size() // 8, ta.full_available_size()), (8, 256)
        )

        bw = object.__new__(TpuDflashDraftWorker)
        bw.model_runner = types.SimpleNamespace(
            token_to_kv_pool=dp, token_to_kv_pool_allocator=ta
        )
        ext48 = torch.cat(
            [ta.alloc_extend(z0, z0, s16, s16, neg1, 16) for _ in range(3)]
        )
        self.assertGreater(int(ta.full_to_swa_index_mapping[ext48].max()), 64)
        locs = bw.full_to_swa[ext48]
        k, v = (
            torch.randn((48, 1, 64), dtype=torch.bfloat16),
            torch.randn((48, 1, 64), dtype=torch.bfloat16),
        )
        _scatter_layer_kv(kv_pool=dp.swa_kv_pool, layer_idx=0, loc=locs, k=k, v=v)
        flat = dp.swa_kv_pool.get_fused_kv_buffer(0).flatten(0, 1)
        self.assertTrue(
            torch.equal(flat[locs, 0, 0, :64], k[:, 0])
            and torch.equal(flat[locs, 0, 1, :64], v[:, 0])
        )
        ta.free(ext48)

    def test_edge_cases_and_hook_passthrough(self):
        base_cfg = MemoryPoolConfig(max_total_num_tokens=16)
        self.assertIs(
            dkp_hooks.derive_unified_draft_pool_sizes(
                lambda _k, *, config: config,
                types.SimpleNamespace(is_draft_worker=True),
                config=base_cfg,
            ),
            base_cfg,
        )
        ta, rp = self._build_target(16, 2)
        with self.assertRaisesRegex(ValueError, "Mixed full and sliding-window"):
            self._attach(ta, rp, full_ids=(0,), swa_ids=(1,))
        with self.assertRaisesRegex(ValueError, "Draft sliding window must not exceed"):
            self._attach(ta, rp, window=128)
        self.assertIs(
            self._attach(ta, rp).full_to_swa_index_mapping,
            self._attach(ta, rp).full_to_swa_index_mapping,
        )
        ta1, rp1 = self._build_target(16, 1)
        dp1 = self._attach(ta1, rp1, window=16)
        self.assertEqual((dp1.size, dp1.size_swa), (128, 64))
        self.assertIsNotNone(ta1.draft_swa_sidecar)
        z0, neg1, s16 = torch.tensor([0]), torch.tensor([-1]), torch.tensor([16])
        ext1 = ta1.alloc_extend(z0, z0, s16, s16, neg1, 16)
        self.assertTrue(8 <= int(dp1.translate_loc_from_full_to_swa(ext1).max()) < 72)
        s17 = torch.tensor([17])
        dec1 = ta1.alloc_decode(s17, s17, ext1[-1:])
        self.assertIsNotNone(dec1)
        self.assertTrue(8 <= int(dp1.translate_loc_from_full_to_swa(dec1).item()) < 72)
        ta1.free(torch.cat([ext1, dec1]))

    def test_sidecar_capacity_eviction(self):
        # Sidecar smaller than shared SWA capacity gates swa_available_size / can_reserve
        # and triggers SWA eviction via evict_to_free_tokens.
        ta, rp = self._build_target(32, 1)  # 32 shared pages, M_T = 1
        dp = SWAKVPool(
            size=256,
            size_swa=16,  # 2 compact draft SWA pages
            page_size=8,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        dk = types.SimpleNamespace(
            is_draft_worker=True,
            is_hybrid_swa=True,
            sliding_window_size=8,
            layer_info=types.SimpleNamespace(
                full_attention_layer_ids=[], swa_attention_layer_ids=[0]
            ),
            model_config=types.SimpleNamespace(sliding_window_size=None),
        )
        ta.attach_draft_kv_pool(dk, dp)
        self.assertEqual(ta.swa_available_size(), 16)
        z0, neg1, s16 = torch.tensor([0]), torch.tensor([-1]), torch.tensor([16])
        held = ta.alloc_extend(z0, z0, s16, s16, neg1, 16)
        self.assertEqual((ta.swa_available_size(), ta.full_available_size()), (0, 224))
        self.assertFalse(ta.new_pages_available(1, 1))
        self.assertFalse(ta.can_reserve(8, 8))
        self.assertTrue(ta.can_reserve(8, 8, swa_evictable_tokens=8))
        with self.assertRaisesRegex(RuntimeError, "Draft SWA sidecar ran out of pages"):
            ta.draft_swa_sidecar.bind_target_pages(torch.tensor([31]))
        self.assertIsNone(
            ta.alloc_extend(z0, z0, torch.tensor([8]), torch.tensor([8]), neg1, 8)
        )

        evicted = []
        fake_tree = types.SimpleNamespace(
            supports_prefix_sharing=lambda: True,
            evict_for_alloc=lambda p: (
                evicted.append((p.num_tokens, p.swa_num_tokens)),
                ta.free_swa(held[:8]),
            ),
        )
        ta.evict_to_free_tokens(fake_tree, 8)
        self.assertEqual(evicted, [(0, 8)])
        self.assertEqual(ta.swa_available_size(), 8)
        ext2 = ta.alloc_extend(z0, z0, torch.tensor([8]), torch.tensor([8]), neg1, 8)
        self.assertIsNotNone(ext2)
        ta.free(held)
        ta.free(ext2)

    def test_sidecar_reclaim_filters_to_draft_owning_swa_windows(self):
        from array import array

        from sglang.srt.mem_cache.base_prefix_cache import (
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        # End-to-end UnifiedRadixCache reproduction of the W2-nopt prefill OOM:
        # 3 older SWA windows hold no draft sidecar pages, 2 newer windows hold
        # 1 sidecar page each (2 total), and the shared pool has extra room so
        # sidecar reclamation alone decides which SWA windows get evicted.
        ta, rp = self._build_target(24, 1)  # 24 shared pages, page_size=8, M_T=1
        dp = SWAKVPool(
            size=192,
            size_swa=16,  # 2 sidecar pages total
            page_size=8,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        dk = types.SimpleNamespace(
            is_draft_worker=True,
            is_hybrid_swa=True,
            sliding_window_size=8,
            layer_info=types.SimpleNamespace(
                full_attention_layer_ids=[], swa_attention_layer_ids=[0]
            ),
            model_config=types.SimpleNamespace(sliding_window_size=None),
        )
        ta.attach_draft_kv_pool(dk, dp)
        cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=rp,
                token_to_kv_pool_allocator=ta,
                page_size=8,
                disable=False,
                sliding_window_size=8,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                eviction_policy="lru",
                tree_core_backend="python",
            )
        )
        z0, neg1 = torch.tensor([0]), torch.tensor([-1])

        def _insert_seq(base_tok: int, length: int = 16):
            s_t = torch.tensor([length])
            idx = ta.alloc_extend(z0, z0, s_t, s_t, neg1, length)
            self.assertIsNotNone(idx)
            toks = list(range(base_tok, base_tok + length))
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", toks), None),
                    value=idx.to(torch.int64),
                    prev_prefix_len=0,
                )
            )
            return idx

        # Three 16-token windows (12 shared pages) whose draft sidecar pages were lost.
        for base in (100, 200, 300):
            seq_idx = _insert_seq(base, 16)
            swa_pages = torch.unique(ta.full_to_swa_index_mapping[seq_idx] // 8)
            ta.draft_swa_sidecar.release_target_pages(swa_pages)
            ta.draft_swa_sidecar.clear_mapping(seq_idx)

        # Two 8-token windows (2 shared pages each = 4 shared pages) owning 1 sidecar page each.
        _insert_seq(400, 8)
        _insert_seq(500, 8)
        self.assertEqual(ta._free_page_count(), 8)
        self.assertEqual(ta.draft_swa_sidecar.allocator.available_size(), 0)

        # Reclaiming 8 prefill tokens (1 sidecar page) skips all 3 draft-less windows
        # (100, 200, 300), evicts ONLY 400 (freeing exactly 1 sidecar page), and stops
        # before touching 500 (zero over-eviction).
        s8 = torch.tensor([8])
        self.assertTrue(ta.evict_to_free_tokens(cache, 8))
        self.assertEqual(ta.draft_swa_sidecar.allocator.available_size(), 8)
        for kept_base, kept_len in ((100, 16), (200, 16), (300, 16), (500, 8)):
            hit = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        array("q", list(range(kept_base, kept_base + kept_len))), None
                    )
                )
            )
            self.assertEqual(len(hit.device_indices), kept_len)
        evicted_400 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", list(range(400, 408))), None))
        )
        self.assertEqual(len(evicted_400.device_indices), 0)

        prefill = ta.alloc_extend(z0, z0, s8, s8, neg1, 8)
        self.assertIsNotNone(prefill)
        ta.free(prefill)
        cache.sanity_check()

    def test_internal_swa_eviction_and_shared_host_reclaim_policy(self):
        from array import array

        from sglang.srt.managers.cache_controller import BatchDoneCounter
        from sglang.srt.mem_cache.base_prefix_cache import (
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.hicache_storage import PoolName as PN
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer as PT
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
        from sglang.srt.mem_cache.pool_host.group import HostPoolGroup
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang_tpu import kv_pool_host as kph
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.tpu_dma_hybrid_cache_controller import (
            TpuDMAHybridCacheController,
        )

        # 1. Internal-SWA device eviction ahead of older Full D-leaves.
        ta, rp = self._build_target(24, 3)
        dp = self._attach(ta, rp, window=8)
        cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=rp,
                token_to_kv_pool_allocator=ta,
                page_size=8,
                disable=False,
                sliding_window_size=8,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                eviction_policy="lru",
                tree_core_backend="python",
            )
        )
        neg1 = torch.tensor([-1])

        def _alloc_and_insert(toks, prev_len=0, prefix_idx=None):
            ext = len(toks) - prev_len
            p_t = torch.tensor([prev_len])
            s_t = torch.tensor([len(toks)])
            last = neg1 if prefix_idx is None else prefix_idx[-1:]
            new_idx = ta.alloc_extend(p_t, p_t, s_t, s_t, last, ext)
            self.assertIsNotNone(new_idx)
            full_idx = (
                new_idx
                if prefix_idx is None
                else torch.cat([prefix_idx.to(new_idx.dtype), new_idx])
            )
            res = cache.insert(
                InsertParams(
                    key=RadixKey(array("q", toks), None),
                    value=full_idx.to(torch.int64),
                    prev_prefix_len=prev_len,
                )
            )
            cache.components[ComponentType.SWA]._demote_stale_windows(
                last_node=cache.tree_core.node_by_id(res.last_device_node)
            )
            return full_idx

        toks_a = list(range(100, 116))
        _alloc_and_insert(toks_a)
        toks_b = list(range(200, 232))
        idx_b1 = _alloc_and_insert(toks_b[:16])
        _alloc_and_insert(toks_b, prev_len=16, prefix_idx=idx_b1)
        self.assertEqual(ta._free_page_count(), 0)

        self.assertTrue(ta.evict_to_free_tokens(cache, 8))
        self.assertEqual(ta._free_page_count(), 6)
        self.assertEqual(dp.size_swa // 8, 6)
        hit_a = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", toks_a + [999]), None))
        )
        hit_b = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", toks_b + [999]), None))
        )
        self.assertEqual(len(hit_a.device_indices), 16)
        self.assertEqual(len(hit_b.device_indices), 32)
        cache.sanity_check()

        # 2. TPU unified shared-host reclaim policy (_UnifiedHostSide + TpuDMAHybridCacheController).
        ta_h, _ = self._build_target(10, 3)  # page_size=8, M_T=3
        tp_h = ta_h.get_kvcache()
        fm, sm = kph._unified_swa_layer_mappings(tp_h)
        mx = max(fm.keys() | sm.keys()) + 1

        def _make_stack(share: float):
            evict_calls: list[tuple[str, int]] = []
            held_full: list[torch.Tensor] = []
            held_swa: list[torch.Tensor] = []
            with (
                tpu_envs.SGLANG_TPU_UNIFIED_HICACHE_SWA_SHARE.override(share),
                patch.object(
                    kph,
                    "create_tpu_dma_kv_transfer",
                    side_effect=lambda *_, **__: types.SimpleNamespace(
                        destroy=lambda: None
                    ),
                ),
                patch.object(kph, "host_memory_budget_bytes", return_value=1 << 40),
            ):
                kv_h, swa_h = kph.TpuDMAUnifiedHostKVPool.build_pair(
                    tp_h,
                    io_backend="raiden",
                    host_to_device_ratio=2.0,  # 21 physical pages
                    host_size=0.0,
                    page_size=8,
                )

            def _evict_full(tokens: int) -> int:
                evict_calls.append(("full", tokens))
                freed = 0
                while freed < tokens and held_full:
                    freed += kv_h.free(held_full.pop(0))
                return freed

            def _evict_swa(tokens: int) -> int:
                evict_calls.append(("swa", tokens))
                freed = 0
                while freed < tokens and held_swa:
                    freed += swa_h.free(held_swa.pop(0))
                return freed

            c = object.__new__(TpuDMAHybridCacheController)
            c.mem_pool_host = HostPoolGroup(
                [
                    hpa.build_pool_entry(
                        name=PN.KV,
                        host_pool=kv_h,
                        device_pool=tp_h,
                        layer_mapping=fm,
                        transfer_layer_id_max=mx,
                        is_anchor=True,
                        host_evict_fn=_evict_full,
                    ),
                    hpa.build_pool_entry(
                        name=PN.SWA,
                        host_pool=swa_h,
                        device_pool=tp_h,
                        layer_mapping=sm,
                        transfer_layer_id_max=mx,
                        host_evict_fn=_evict_swa,
                    ),
                ]
            )
            c.page_size = 8
            c.layer_done_counter = BatchDoneCounter()
            return c, kv_h, swa_h, evict_calls, held_full, held_swa

        ctrl, kv_h, swa_h, evict_calls, held_full, held_swa = _make_stack(0.3)
        domain = kv_h.shared_allocation_domain
        self.assertEqual(domain.total_bytes(), 21 * domain.bytes_per_page)

        # Case A (c16 + large incoming SWA request): 15 Full + 2 SWA (6/21 = 28.6% <= 30%).
        # Even when incoming SWA is 2 pages (resident + req = 12/21 > 30%), resident SWA is
        # <= 30%, so Full KV reclaims first without flushing resident SWA windows.
        for _ in range(15):
            held_full.append(kv_h.alloc(8))
        for _ in range(2):
            held_swa.append(swa_h.alloc(8))
        self.assertEqual(domain.free_bytes(), 0)
        res_a = ctrl.allocate_shared_host_transfers(
            torch.arange(8), [PT(name=PN.SWA, device_indices=torch.arange(16))]
        )
        self.assertIsNotNone(res_a)
        self.assertEqual(evict_calls, [("full", 56)])
        held_full.append(res_a[0])
        held_swa.append(res_a[1][0].host_indices)

        # Case B (c22 + resident excess trim): SWA now holds 4 SWA pages (12/21 = 57.1% > 30%;
        # cap = int(0.3 * 21) = 6 phys pages -> 2 excess SWA pages = 16 SWA tokens).
        # Requesting 7 Full pages (56 tokens) trims ONLY the 16 excess SWA tokens first, then
        # falls through in the same pass to Full for the remaining 1 page.
        evict_calls.clear()
        res_b = ctrl.allocate_shared_host_transfers(torch.arange(56))
        self.assertIsNotNone(res_b)
        self.assertEqual(evict_calls, [("swa", 16), ("full", 8)])
        held_full.append(res_b[0])

        # Case C (opt-out <= 0.0): restores legacy reversed(requests) SWA-first order.
        ctrl_l, kv_l, swa_l, calls_l, full_l, swa_list_l = _make_stack(0.0)
        for _ in range(18):
            full_l.append(kv_l.alloc(8))
        swa_list_l.append(swa_l.alloc(8))
        res_c = ctrl_l.allocate_shared_host_transfers(
            torch.arange(8), [PT(name=PN.SWA, device_indices=torch.arange(8))]
        )
        self.assertIsNotNone(res_c)
        self.assertEqual(calls_l[0][0], "swa")


if __name__ == "__main__":
    unittest.main()
