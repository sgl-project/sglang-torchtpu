# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EAGLE/MTP/Standalone speculative draft KV caches over TpuUnifiedSWAKVPool."""

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
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks.draft_kv_pool import (
    attach_unified_draft_kv_pool,
    derive_unified_draft_pool_sizes,
)
from sglang_tpu.kv_free_fence import install_kv_free_fence
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWAPoolConfigurator,
    TpuUnifiedSWATokenToKVPoolAllocator,
    resolve_unified_draft_kv_pricing,
    unified_kv_pool_enabled,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftCoPricingAndValidation(unittest.TestCase):
    @contextmanager
    def _ctx(self, **overrides):
        args = {
            "attention_backend": "pallas_tpu_rpa_v3",
            "tp_size": 1,
            "pp_size": 1,
            "dcp_size": 1,
            "attn_cp_size": 1,
            "disaggregation_mode": "null",
            "enable_hierarchical_cache": False,
            "enable_unified_cache_external_linker": False,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "speculative_draft_kv_cache_dtype": "auto",
            "speculative_algorithm": "EAGLE",
            **overrides,
        }
        try:
            with ExitStack() as stack:
                stack.enter_context(tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL.override(True))
                stack.enter_context(get_context().override_server_args(**args))
                stack.enter_context(
                    get_parallel().override(
                        attn_tp_size=1, attn_dcp_size=1, dcp_enabled=False
                    )
                )
                yield
        finally:
            tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL.clear()

    def _mc(
        self,
        full_ids: list[int],
        swa_ids: list[int],
        *,
        h: int = 2,
        d: int = 128,
        sh: int | None = None,
        sd: int | None = None,
        window: int | None = 128,
        arch=AttentionArch.MHA,
        asym: bool = False,
    ):
        sh, sd = (h if sh is None else sh), (d if sd is None else sd)
        win = window if swa_ids else None
        hf = types.SimpleNamespace(architectures=["Dummy"], sliding_window=win)
        hf.get_text_config = lambda: hf
        return types.SimpleNamespace(
            dtype=torch.bfloat16,
            attention_arch=arch,
            has_asymmetric_kv=asym,
            is_hybrid_swa=bool(swa_ids),
            linear_attn_registry_result=None,
            mambaish_config=None,
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            num_hidden_layers=len(full_ids) + len(swa_ids),
            full_attention_layer_ids=list(full_ids),
            swa_attention_layer_ids=list(swa_ids),
            head_dim=d,
            v_head_dim=d,
            swa_head_dim=sd,
            swa_v_head_dim=sd,
            sliding_window_size=win,
            hf_config=hf,
            hf_text_config=hf,
            get_num_kv_heads=lambda tp=1, dcp=1: h,
            get_swa_num_kv_heads=lambda tp=1, dcp=1: sh,
            get_hybrid_layer_ids=lambda s, e: (list(full_ids), list(swa_ids)),
        )

    def _kvc(
        self,
        target_mc,
        *,
        algo=SpeculativeAlgorithm.EAGLE,
        draft_mc=None,
        runners: int = 1,
        page_size: int = 16,
        dtype=torch.bfloat16,
        dtype_str: str = "bf16",
    ):
        f_ids, s_ids = target_mc.get_hybrid_layer_ids(0, target_mc.num_hidden_layers)
        spec_aux = (
            None
            if draft_mc is None
            else types.SimpleNamespace(
                draft_model_config=draft_mc,
                eagle_draft_num_layers=runners * draft_mc.num_hidden_layers,
                eagle_draft_swa_num_layers=runners
                * len(draft_mc.swa_attention_layer_ids),
            )
        )
        return types.SimpleNamespace(
            device="cpu",
            server_args=types.SimpleNamespace(
                speculative_draft_model_path="dummy-draft",
                speculative_draft_model_revision=None,
            ),
            is_hybrid_swa=True,
            use_mla_backend=False,
            mambaish_config=None,
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            is_draft_worker=False,
            spec_algorithm=algo,
            spec_aux_config=spec_aux,
            pp_size=1,
            attn_dp_size=1,
            _tpu_mha_pool_class=lambda: TpuFusedMHATokenToKVPool,
            kv_cache_dtype_str=dtype_str,
            kv_cache_dtype=dtype,
            kv_group_size=32 if dtype_str == "mxint8" else None,
            pool_page_size=page_size,
            page_size=page_size,
            sliding_window_size=127,
            model_config=target_mc,
            layer_info=types.SimpleNamespace(
                start_layer=0,
                end_layer=target_mc.num_hidden_layers,
                full_attention_layer_ids=f_ids,
                swa_attention_layer_ids=s_ids,
            ),
        )

    def test_closed_form_copricing_and_server_args_resolution(self):
        t_mc = self._mc([0, 1], [2, 3, 4, 5, 6, 7])  # M_T = 3
        budget = 64 * 1024 * 1024

        # 1. Resolve draft ModelConfig via ServerArgs when spec_aux lacks draft_model_config.
        d_mc_srv = self._mc([0], [1], h=1, d=64, window=64)
        with (
            self._ctx(
                speculative_algorithm="EAGLE3",
                speculative_draft_model_path="mock/eagle3",
            ),
            patch.object(
                ModelConfig, "from_server_args", return_value=d_mc_srv
            ) as mock_from,
        ):
            kvc = self._kvc(t_mc, algo=SpeculativeAlgorithm.EAGLE3)
            kvc.spec_aux_config = types.SimpleNamespace(
                eagle_draft_num_layers=2, eagle_draft_swa_num_layers=1
            )
            self.assertTrue(unified_kv_pool_enabled(kvc))
            p = resolve_unified_draft_kv_pricing(kvc)
            self.assertGreater(p.full_per_token, 0)
            self.assertGreater(p.swa_compact_per_token, 0)
            mock_from.assert_called_once()

        # 2. Closed-form co-pricing + 8*M_T alignment + HBM <= budget across layouts & mxint8.
        cases = (
            ("fa_eagle", SpeculativeAlgorithm.EAGLE, [0, 1], [], 1, 64, 16, "bf16", 1),
            (
                "swa_eagle",
                SpeculativeAlgorithm.EAGLE,
                [],
                [0, 1],
                1,
                64,
                16,
                "bf16",
                1,
            ),
            (
                "hetero_mtp",
                SpeculativeAlgorithm.STANDALONE,
                [0],
                [1, 2],
                2,
                64,
                16,
                "bf16",
                2,
            ),
            (
                "mxint8",
                SpeculativeAlgorithm.EAGLE,
                [0],
                [1],
                2,
                128,
                128,
                "mxint8",
                1,
            ),
        )
        for name, algo, df, ds, dh, dd, ps, dt_str, runners in cases:
            dt = torch.int8 if dt_str == "mxint8" else torch.bfloat16
            with (
                self.subTest(case=name),
                self._ctx(
                    speculative_algorithm=algo.name,
                    kv_cache_dtype=dt_str if dt_str == "mxint8" else "auto",
                    speculative_draft_kv_cache_dtype=(
                        dt_str if dt_str == "mxint8" else "auto"
                    ),
                    page_size=ps,
                ),
            ):
                d_mc = self._mc(df, ds, h=dh, d=dd, window=64 if ds else None)
                kvc = self._kvc(
                    t_mc,
                    algo=algo,
                    draft_mc=d_mc,
                    runners=runners,
                    page_size=ps,
                    dtype=dt,
                    dtype_str=dt_str,
                )
                conf = TpuUnifiedSWAPoolConfigurator(kvc)
                self.assertEqual(conf._align_pages, 24 if ds else 8)
                n_tok = conf.calculate_pool_sizes(
                    budget, page_size=ps
                ).full_max_total_num_tokens
                self.assertGreater(n_tok, 0)
                self.assertEqual((n_tok // ps) % conf._align_pages, 0)

                if dt_str != "mxint8":
                    t_pool = TpuUnifiedSWAKVPool(
                        size=n_tok,
                        page_size=ps,
                        dtype=dt,
                        full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=[0, 1]),
                        swa=AttnLayerSpec(
                            head_num=2, head_dim=128, layer_ids=[2, 3, 4, 5, 6, 7]
                        ),
                        device="cpu",
                    )
                    d_pool = (
                        SWAKVPool(
                            size=n_tok,
                            size_swa=((n_tok // ps) // 3) * ps,
                            page_size=ps,
                            dtype=dt,
                            head_num=dh,
                            head_dim=dd,
                            swa_attention_layer_ids=ds,
                            full_attention_layer_ids=df,
                            device="cpu",
                            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
                        )
                        if ds
                        else TpuFusedMHATokenToKVPool(
                            size=n_tok,
                            page_size=ps,
                            dtype=dt,
                            head_num=dh,
                            head_dim=dd,
                            layer_num=len(df),
                            device="cpu",
                            enable_memory_saver=False,
                        )
                    )
                    total = sum(t_pool.get_kv_size_bytes()) + runners * sum(
                        d_pool.get_kv_size_bytes()
                    )
                    self.assertLessEqual(total, budget)

    def test_unsupported_draft_configs_rejected(self):
        t_mc = self._mc([0, 1], [2, 3])
        bad_drafts = (
            self._mc([0], [], arch=AttentionArch.MLA),
            self._mc([0], [], asym=True),
            self._mc([0], [1], window=512),  # W_D > W_T
            self._mc([0], [1], window=None),  # unwindowed SWA draft
        )
        with self._ctx():
            for bad in bad_drafts:
                with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
                    unified_kv_pool_enabled(self._kvc(t_mc, draft_mc=bad))


class TestDraftSwaSidecarAndFence(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(swa, "_is_tpu", True))
        self.enterContext(
            patch.object(
                swa,
                "TpuPagedTokenToKVPoolAllocator",
                TpuPagedTokenToKVPoolAllocator,
                create=True,
            )
        )

    def _build_target(self, num_pages: int = 24, page_size: int = 8, m_swa: int = 3):
        size = num_pages * page_size
        t_pool = TpuUnifiedSWAKVPool(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=[0, 1]),
            swa=AttnLayerSpec(
                head_num=1, head_dim=64, layer_ids=list(range(2, 2 + 2 * m_swa))
            ),
            device="cpu",
        )
        req_pool = ReqToTokenPool(
            size=8, max_context_len=256, device="cpu", enable_memory_saver=False
        )
        t_alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=t_pool,
            need_sort=False,
            req_to_token_pool=req_pool,
            sliding_window_size=16,
        )
        return t_pool, t_alloc, req_pool

    def _attach_via_hooks(self, t_alloc, req_pool, *, swa_ids=(1,), window: int = 8):
        d_kvc = types.SimpleNamespace(
            is_draft_worker=True,
            is_hybrid_swa=bool(swa_ids),
            sliding_window_size=window,
            token_to_kv_pool_allocator=t_alloc,
            layer_info=types.SimpleNamespace(
                full_attention_layer_ids=[0], swa_attention_layer_ids=list(swa_ids)
            ),
            model_config=types.SimpleNamespace(sliding_window_size=None),
        )
        cfg = derive_unified_draft_pool_sizes(
            lambda _kvc, *, config: config,
            d_kvc,
            config=MemoryPoolConfig(
                max_total_num_tokens=1,
                full_max_total_num_tokens=1,
                swa_max_total_num_tokens=1,
            ),
        )
        d_pool = SWAKVPool(
            size=cfg.full_max_total_num_tokens,
            size_swa=cfg.swa_max_total_num_tokens,
            page_size=t_alloc.page_size,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=list(swa_ids),
            full_attention_layer_ids=[0],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        returned = attach_unified_draft_kv_pool(
            lambda *_a, **_k: AssertionError("unexpected fallback"),
            d_kvc,
            token_to_kv_pool=d_pool,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=t_alloc,
        )
        self.assertIs(returned, t_alloc)
        return d_pool

    def test_sidecar_extend_decode_evict_and_fence(self):
        _, t_alloc, req_pool = self._build_target(num_pages=24, page_size=8, m_swa=3)
        d_pool = self._attach_via_hooks(t_alloc, req_pool)
        sidecar = t_alloc.draft_swa_sidecar
        self.assertEqual((d_pool.size, d_pool.size_swa), (192, 64))
        self.assertIs(d_pool.full_to_swa_index_mapping, sidecar.mapping)

        fence = install_kv_free_fence(
            t_alloc,
            synchronize_device=lambda: None,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            copies_draft_tail=True,
        )
        fence.attach(sidecar.allocator)
        self.assertEqual(len(fence._allocators), 3)

        res = types.SimpleNamespace(copy_done=None)
        fence.on_batch_formed()
        fence.on_forward_launched(res)
        z0 = torch.tensor([0], dtype=torch.int64)
        neg1 = torch.tensor([-1], dtype=torch.int64)
        with fence.forward_write_target():
            s20 = torch.tensor([20], dtype=torch.int64)
            ext = t_alloc.alloc_extend(z0, z0, s20, s20, neg1, 20)
            req_pool.req_to_token[0, :20] = ext
            d_slots = d_pool.translate_loc_from_full_to_swa(ext)
            self.assertEqual(len(set(d_slots.tolist())), 20)
            self.assertTrue(torch.all((d_slots >= 8) & (d_slots < 72)))
            self.assertEqual(
                int(d_pool.translate_loc_from_full_to_swa(neg1).item()), -1
            )
            t_alloc.free_swa(ext[:16])
            self.assertTrue(torch.all(d_pool.full_to_swa_index_mapping[ext[:16]] == 0))

        self.assertEqual(sidecar.allocator.held_pages.num_pages, 2)
        s16 = torch.tensor([16], dtype=torch.int64)
        ext2 = t_alloc.alloc_extend(z0, z0, s16, s16, neg1, 16)
        self.assertEqual(fence.stats.device_syncs, 0)
        fence.on_forward_done(res)
        self.assertEqual(sidecar.allocator.held_pages.num_pages, 0)

        for pos in (20, 21, 22, 23, 24):
            seq_t = torch.tensor([pos + 1], dtype=torch.int64)
            dec = t_alloc.alloc_decode(
                seq_t, seq_t, req_pool.req_to_token[0, pos - 1 : pos]
            )
            req_pool.req_to_token[0, pos] = dec[0]
            self.assertTrue(
                8 <= int(d_pool.translate_loc_from_full_to_swa(dec)[0]) < 72
            )

        t_alloc.free(req_pool.req_to_token[0, :25])
        t_alloc.free(ext2)
        self.assertEqual(sidecar.allocator.available_size() // 8, 8)
        self.assertEqual(t_alloc.full_available_size(), 192)

    def test_edge_cases_and_hook_passthrough(self):
        # Bare __new__ configurator without token_to_kv_pool_allocator falls through.
        base_cfg = MemoryPoolConfig(max_total_num_tokens=16)
        self.assertIs(
            derive_unified_draft_pool_sizes(
                lambda _kvc, *, config: config,
                types.SimpleNamespace(is_draft_worker=True),
                config=base_cfg,
            ),
            base_cfg,
        )

        # W_D > W_T rejected in allocator sizing and attachment.
        _, t_alloc, req_pool = self._build_target(num_pages=16, page_size=8, m_swa=2)
        with self.assertRaisesRegex(ValueError, "Draft sliding window must not exceed"):
            self._attach_via_hooks(t_alloc, req_pool, window=128)

        # Multi-runner MTP shares a single _DraftSwaSidecar.
        d1 = self._attach_via_hooks(t_alloc, req_pool)
        d2 = self._attach_via_hooks(t_alloc, req_pool)
        self.assertIs(d1.full_to_swa_index_mapping, d2.full_to_swa_index_mapping)

        # M_T == 1 reuses target full_to_swa_index_mapping without allocating a sidecar.
        _, t_alloc1, req_pool1 = self._build_target(num_pages=16, page_size=8, m_swa=1)
        d_m1 = self._attach_via_hooks(t_alloc1, req_pool1, window=16)
        self.assertIsNone(t_alloc1.draft_swa_sidecar)
        self.assertIs(
            d_m1.full_to_swa_index_mapping, t_alloc1.full_to_swa_index_mapping
        )


if __name__ == "__main__":
    unittest.main()
