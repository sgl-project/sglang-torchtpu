# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import copy
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.unified_kv_pool import (
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWAPoolConfigurator,
    TpuUnifiedSWATokenToKVPoolAllocator,
    merge_swa_kv_heads,
    num_swa_groups,
    split_swa_output,
    unified_kv_pool_enabled,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuUnifiedSWAKVPoolStorage(unittest.TestCase):
    """Tests for PR 1: shared physical buffers, group page mapping, and head merging."""

    def test_equal_full_and_swa_layers_share_buffers(self):
        pool = TpuUnifiedSWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=4,
            head_dim=128,
            swa_attention_layer_ids=[0, 2],
            full_attention_layer_ids=[1, 3],
            device="cpu",
        )
        self.assertEqual((pool.size, pool.size_swa), (64, 64))
        self.assertEqual((pool.num_swa_groups, pool.num_shared_buffers), (1, 2))
        self.assertFalse(pool.swa_kv_heads_merged)
        self.assertEqual(len(pool.get_state()), 2)
        for full_lid, swa_lid in ((1, 0), (3, 2)):
            self.assertEqual(
                pool.get_fused_kv_buffer(full_lid).data_ptr(),
                pool.get_fused_kv_buffer(swa_lid).data_ptr(),
            )
            self.assertEqual(pool.swa_page_group(swa_lid), 0)
        self.assertEqual(
            pool.get_kv_size_bytes(), pool.full_kv_pool.get_kv_size_bytes()
        )
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_gemma4_ratio_and_head_merging_bf16_and_fp8(self):
        full_ids = [5, 11]
        swa_ids = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = TpuUnifiedSWAKVPool(
                    size=80,
                    size_swa=40,
                    page_size=4,
                    dtype=dtype,
                    head_num=1,
                    head_dim=512,
                    swa_head_num=2,
                    swa_head_dim=256,
                    swa_attention_layer_ids=swa_ids,
                    full_attention_layer_ids=full_ids,
                    device="cpu",
                )
                self.assertTrue(pool.swa_kv_heads_merged)
                self.assertEqual((pool.num_swa_groups, pool.num_shared_buffers), (5, 2))
                self.assertEqual(pool.full_kv_pool.num_blocks, 21)
                self.assertEqual(pool.swa_group_pages.shape, (4, 21))
                for j, lid in enumerate(swa_ids):
                    self.assertEqual(pool.swa_page_group(lid), j // 2)
                    self.assertEqual(
                        pool.get_fused_kv_buffer(lid).data_ptr(),
                        pool.get_state()[j % 2].data_ptr(),
                    )
                new_state = [torch.zeros_like(buf) for buf in pool.get_state()]
                pool.set_state(new_state)
                for j, lid in enumerate(swa_ids):
                    self.assertEqual(
                        pool.get_fused_kv_buffer(lid).data_ptr(),
                        new_state[j % 2].data_ptr(),
                    )

    def test_incompatible_page_shapes_rejected(self):
        with self.assertRaises(ValueError):
            TpuUnifiedSWAKVPool(
                size=32,
                page_size=4,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=512,
                swa_head_num=3,
                swa_head_dim=128,
                swa_attention_layer_ids=[0, 1],
                full_attention_layer_ids=[2],
                device="cpu",
            )

    def test_merge_swa_kv_heads_and_split_output_exact(self):
        torch.manual_seed(42)
        T, Hq, Hkv, D = 7, 8, 2, 64
        q = torch.randn(T, Hq, D)
        k = torch.randn(T, Hkv, D)
        v = torch.randn(T, Hkv, D)
        scale = D**-0.5

        qg = q.view(T, Hkv, Hq // Hkv, D)
        scores = torch.einsum("thgd,shd->thgs", qg, k) * scale
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        probs = torch.softmax(scores + mask[:, None, None, :], dim=-1)
        out_ref = torch.einsum("thgs,shd->thgd", probs, v).reshape(T, Hq, D)

        q_m, k_m, v_m = merge_swa_kv_heads(q, k, v)
        self.assertEqual(
            (q_m.shape, k_m.shape, v_m.shape), ((T, Hq, 128), (T, 1, 128), (T, 1, 128))
        )
        scores_m = torch.einsum("tqd,skd->tqks", q_m, k_m).squeeze(2) * scale
        probs_m = torch.softmax(scores_m + mask[:, None, :], dim=-1)
        out_m = torch.einsum("tqs,skd->tqd", probs_m, v_m)
        torch.testing.assert_close(
            split_swa_output(out_m, Hkv), out_ref, rtol=1e-5, atol=1e-5
        )

    def test_non_divisible_swa_groups_and_cpu_copy_restore(self):
        self.assertEqual(num_swa_groups(8, 20), 3)
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=48,
            size_swa=48,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=list(range(8, 28)),
            full_attention_layer_ids=list(range(8)),
            device="cpu",
        )
        self.assertEqual((pool.num_shared_buffers, pool.num_swa_groups), (8, 3))
        # SWA scattered pages [3, 9, 11] -> [6, 2, 10] across M_swa = 3 groups.
        pool.swa_group_pages[:, 3] = torch.tensor([9, 11], dtype=torch.int32)
        for j in range(20):
            p = 3 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 3])
            pool.get_state()[j % 8][p].fill_(float(10 + j))
        swa_copy = pool.swa_kv_pool.get_cpu_copy(
            torch.arange(3 * page_size, 4 * page_size)
        )
        pool.swa_group_pages[:, 6] = torch.tensor([2, 10], dtype=torch.int32)
        pool.swa_kv_pool.load_cpu_copy(
            swa_copy, torch.arange(6 * page_size, 7 * page_size)
        )
        for j in range(20):
            p = 6 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 6])
            self.assertTrue(torch.all(pool.get_state()[j % 8][p] == float(10 + j)))

        # Full page 2 -> 5 across all 8 shared buffers directly.
        for i in range(8):
            pool.get_state()[i][2].fill_(float(100 + i))
        full_copy = pool.full_kv_pool.get_cpu_copy(
            torch.arange(2 * page_size, 3 * page_size)
        )
        pool.full_kv_pool.load_cpu_copy(
            full_copy, torch.arange(5 * page_size, 6 * page_size)
        )
        for i in range(8):
            self.assertTrue(torch.all(pool.get_state()[i][5] == float(100 + i)))


class TestTpuUnifiedSWAKVPoolAllocator(unittest.TestCase):
    """Tests for PR 2: zero-fragmentation Logical-to-Physical unified allocator."""

    def _make_pool_and_allocator(
        self,
        total_pages=24,
        page_size=4,
        full_ids=(4, 9),
        swa_ids=(0, 1, 2, 3, 5, 6, 7, 8),
    ):
        size = total_pages * page_size
        pool = TpuUnifiedSWAKVPool(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=512,
            swa_head_num=2,
            swa_head_dim=256,
            swa_attention_layer_ids=list(swa_ids),
            full_attention_layer_ids=list(full_ids),
            device="cpu",
        )
        alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        return pool, alloc

    def test_zero_fragmentation_and_companion_pages(self):
        # 2 Full + 8 SWA -> G=2, M_swa=4; 24 physical pages (96 tokens).
        pool, alloc = self._make_pool_and_allocator(total_pages=24, page_size=4)
        self.assertEqual(
            (alloc.full_available_size(), alloc.swa_available_size(), alloc.max_extend_tokens()),
            (96, 24, 16),
        )
        fa, sa = alloc.full_attn_allocator, alloc.swa_attn_allocator
        # Allocate 2 Full pages (2 phys IDs) + 3 SWA pages (12 phys IDs) -> 10 free.
        full_a = fa.alloc(8)
        swa_a = sa.alloc(12)
        self.assertEqual(alloc._free_page_count(), 10)
        swa_primary = (swa_a[::4] // 4).tolist()
        companions = pool.swa_group_pages[:, swa_primary].reshape(-1).tolist()
        all_claimed = (full_a[::4] // 4).tolist() + swa_primary + companions
        self.assertEqual(len(set(all_claimed)), 14)

        # Fragment free list by alternating Full allocations/frees, then allocate 2 SWA pages (8 IDs).
        f1, f2, f3, f4 = (
            fa.alloc(4),
            fa.alloc(4),
            fa.alloc(4),
            fa.alloc(4),
        )
        fa.free(f1)
        fa.free(f3)
        self.assertEqual(alloc._free_page_count(), 8)
        swa_b = sa.alloc(8)
        self.assertIsNotNone(swa_b)
        self.assertEqual(alloc._free_page_count(), 0)

        for t, fn in (
            (full_a, fa.free),
            (f2, fa.free),
            (f4, fa.free),
            (swa_a, sa.free),
            (swa_b, sa.free),
        ):
            fn(t)
        self.assertEqual(alloc._free_page_count(), 24)

    def test_non_divisible_swa_groups_alloc_and_free(self):
        # 8 Full + 20 SWA -> G=8, M_swa=ceil(20/8)=3; 20 physical pages (80 tokens).
        pool, alloc = self._make_pool_and_allocator(
            total_pages=20,
            page_size=4,
            full_ids=range(8),
            swa_ids=range(8, 28),
        )
        self.assertEqual(
            (alloc.full_available_size(), alloc.swa_available_size(), alloc.max_extend_tokens()),
            (80, 24, 20),
        )
        full_tok = alloc.full_attn_allocator.alloc(12)  # 3 * 1 = 3 phys IDs
        swa_tok = alloc.swa_attn_allocator.alloc(8)  # 2 * 3 = 6 phys IDs
        self.assertEqual(alloc._free_page_count(), 11)
        f_p = (full_tok[::4] // 4).tolist()
        s_p = (swa_tok[::4] // 4).tolist()
        claimed = (
            f_p
            + s_p
            + pool.swa_group_pages[:, s_p].reshape(-1).tolist()
        )
        self.assertEqual(len(set(claimed)), 9)
        alloc.full_attn_allocator.free(full_tok)
        alloc.swa_attn_allocator.free(swa_tok)
        self.assertEqual(alloc._free_page_count(), 20)

    def test_alloc_extend_decode_and_eviction(self):
        _, alloc = self._make_pool_and_allocator(total_pages=20, page_size=4)
        # 1+4=5 phys IDs per joint page -> 4 joint pages max (16 tokens).
        out = alloc.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            seq_lens_cpu=torch.tensor([8]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=8,
        )
        self.assertIsNotNone(out)
        self.assertEqual(alloc._free_page_count(), 10)
        dec = alloc.alloc_decode(
            seq_lens=torch.tensor([9]),
            seq_lens_cpu=torch.tensor([9]),
            last_loc=out[-1:],
        )
        self.assertIsNotNone(dec)
        self.assertEqual(alloc._free_page_count(), 5)

        # Request 2 more joint pages (10 phys IDs): shortfall = 5 phys IDs.
        self.assertEqual(
            alloc.reclaim_plan(8, 8, full_evictable_tokens=8, swa_evictable_tokens=8),
            (8, 4),
        )
        self.assertIsNone(
            alloc.reclaim_plan(8, 8, full_evictable_tokens=4, swa_evictable_tokens=0)
        )

        class _TreeCache:
            def supports_prefix_sharing(self):
                return True

            def full_evictable_size(self):
                return 8

            def swa_evictable_size(self):
                return 8

            def evict(self, params: EvictParams):
                if params.swa_num_tokens > 0:
                    alloc.free_swa(out[: params.swa_num_tokens])
                if params.num_tokens > 0:
                    alloc.free_full(out[: params.num_tokens])

        self.assertIsNone(alloc.reclaim_for_prealloc(_TreeCache(), 8, 8))
        self.assertGreaterEqual(alloc._free_page_count(), 10)


class TestTpuUnifiedSWAKVPoolWiring(unittest.TestCase):
    """Tests for PR 3: configurator sizing, RPA page-table selection, and CLI guards."""

    def _cfg_patches(self):
        from contextlib import ExitStack

        stack = ExitStack()
        for mod, val in (
            (
                "sglang.srt.runtime_context.get_exec",
                types.SimpleNamespace(
                    kernel=types.SimpleNamespace(attention_backend="pallas_tpu_rpa_v3")
                ),
            ),
            (
                "sglang.srt.model_executor.pool_configurator.get_parallel",
                types.SimpleNamespace(
                    tp_size=1,
                    attn_tp_size=1,
                    pp_size=1,
                    attn_cp_size=1,
                    attn_dcp_size=1,
                    dcp_enabled=False,
                ),
            ),
            (
                "sglang.srt.model_executor.pool_configurator.get_disagg",
                types.SimpleNamespace(disaggregation_mode="null"),
            ),
            (
                "sglang.srt.model_executor.pool_configurator.get_memory",
                types.SimpleNamespace(
                    kv_cache_dtype="auto",
                    swa_kv_cache_ratio=None,
                    enable_hierarchical_cache=False,
                    hicache_io_backend="raiden",
                    enable_unified_cache_external_linker=False,
                    enable_unified_memory=False,
                    disable_radix_cache=False,
                ),
            ),
            (
                "sglang.srt.model_executor.pool_configurator.get_schedule",
                types.SimpleNamespace(
                    swa_full_tokens_ratio=0.5,
                    chunked_prefill_size=4096,
                    disable_overlap_schedule=False,
                    max_running_requests=None,
                ),
            ),
        ):
            stack.enter_context(patch(mod, return_value=val))
        return stack

    def _make_mock_kvc(
        self,
        full_ids=None,
        swa_ids=None,
        head_num=1,
        head_dim=512,
        swa_head_num=2,
        swa_head_dim=256,
        dtype=torch.bfloat16,
    ):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        full_ids = list(range(10)) if full_ids is None else list(full_ids)
        swa_ids = list(range(10, 60)) if swa_ids is None else list(swa_ids)
        mc = types.SimpleNamespace(
            attention_arch=None,
            head_dim=head_dim,
            v_head_dim=head_dim,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_head_dim,
            get_num_kv_heads=lambda tp=1, dcp=1: head_num,
            get_swa_num_kv_heads=lambda tp=1, dcp=1: swa_head_num,
        )
        elem = torch.empty((), dtype=dtype).element_size()
        full_row = TpuFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
            num_kv_heads=head_num,
            head_dim=head_dim,
            v_head_dim=head_dim,
            store_dtype=dtype,
        )
        swa_row = TpuFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
            num_kv_heads=swa_head_num,
            head_dim=swa_head_dim,
            v_head_dim=swa_head_dim,
            store_dtype=dtype,
        )
        return types.SimpleNamespace(
            is_hybrid_swa=True,
            use_mla_backend=False,
            is_hybrid_gdn=False,
            has_mamba_state=False,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            pp_size=1,
            attn_dp_size=1,
            dcp_enabled=False,
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
            kv_cache_dtype_str="fp8_e4m3" if elem == 1 else "bf16",
            kv_cache_dtype=dtype,
            pool_page_size=128,
            page_size=128,
            custom_mem_pool_head_num=None,
            custom_mem_pool_head_dim=None,
            sliding_window_size=1024,
            max_Prefill_tokens=8192,
            chunked_prefill_size=4096,
            context_len=32768,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            swa_kv_cache_ratio=None,
            model_config=mc,
            layer_info=types.SimpleNamespace(
                full_attention_layer_ids=full_ids,
                swa_attention_layer_ids=swa_ids,
            ),
            _full_attention_cell_size=full_row * len(full_ids),
            _swa_attention_cell_size=swa_row * len(swa_ids),
            _cell_size=full_row * len(full_ids) + swa_row * len(swa_ids),
        )

    def test_configurator_sizes_from_shared_buffers_bf16_and_fp8(self):
        with self._cfg_patches():
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(dtype=dtype):
                    kvc = self._make_mock_kvc(dtype=dtype)
                    conf = TpuUnifiedSWAPoolConfigurator(kvc)
                    self.assertEqual(conf._num_swa_groups, 5)
                    budget = 2 * 1024**3
                    cfg = conf.calculate_pool_sizes(budget, page_size=128)
                    self.assertEqual(
                        cfg.full_max_total_num_tokens, cfg.swa_max_total_num_tokens
                    )
                    pool = TpuUnifiedSWAKVPool(
                        size=cfg.full_max_total_num_tokens,
                        page_size=128,
                        dtype=dtype,
                        head_num=1,
                        head_dim=512,
                        swa_head_num=2,
                        swa_head_dim=256,
                        swa_attention_layer_ids=list(range(10, 60)),
                        full_attention_layer_ids=list(range(10)),
                        device="cpu",
                    )
                    self.assertLessEqual(sum(pool.get_kv_size_bytes()), budget)

    def test_unsupported_configurations_rejected(self):
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.server_args import ServerArgs
        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool
        from sglang_tpu.tpu_platform import _resolve_unified_kv_pool

        base = self._make_mock_kvc()
        with self._cfg_patches():
            for attr, val, exc in (
                ("is_hybrid_swa", False, ValueError),
                ("use_mla_backend", True, ValueError),
                ("is_hybrid_gdn", True, ValueError),
                ("is_draft_worker", True, ValueError),
                ("pp_size", 2, ValueError),
                ("dcp_enabled", True, ValueError),
                ("pool_page_size", 1, ValueError),
            ):
                kvc = copy.copy(base)
                setattr(kvc, attr, val)
                with self.assertRaises(exc):
                    TpuUnifiedSWAPoolConfigurator(kvc)
            kvc_along = copy.copy(base)
            kvc_along.token_to_kv_pool_class = TpuSeqAlongLaneKVPool
            TpuUnifiedSWAPoolConfigurator(kvc_along)
            kvc_mx = copy.copy(base)
            kvc_mx.kv_cache_dtype_str = "mxint8"
            TpuUnifiedSWAPoolConfigurator(kvc_mx)

        with patch(
            "sglang_tpu.unified_kv_pool.get_memory",
            return_value=types.SimpleNamespace(
                tpu_unified_kv_pool=True,
                enable_hierarchical_cache=False,
                enable_unified_cache_external_linker=False,
            ),
        ):
            self.assertTrue(unified_kv_pool_enabled(base))

        ok_args = ServerArgs(
            model_path="dummy",
            enable_unified_memory=True,
            attention_backend="pallas_tpu_rpa_v3",
            page_size=128,
        )
        _resolve_unified_kv_pool(ok_args)
        view = resolving_view(ok_args)
        self.assertFalse(view.enable_unified_memory)
        self.assertTrue(view.tpu_unified_kv_pool)

        for kwargs in (
            dict(attention_backend="pallas_tpu_flash"),
            dict(pp_size=2),
            dict(dcp_size=2),
            dict(attn_cp_size=2),
            dict(disaggregation_mode="prefill"),
            dict(speculative_algorithm="EAGLE"),
            dict(enable_hierarchical_cache=True, hicache_io_backend="kernel"),
            dict(page_size=1),
        ):
            bad_args = ServerArgs(
                model_path="dummy",
                enable_unified_memory=True,
                **{"attention_backend": "pallas_tpu_rpa_v3", "page_size": 128, **kwargs},
            )
            with self.assertRaises(ValueError):
                _resolve_unified_kv_pool(bad_args)

    def test_rpa_backend_layer_page_indices_selects_full_and_swa_groups(self):
        from sglang_tpu.attention_pallas_rpa_base import (
            BaseTpuPallasRpaAttnBackend,
            PallasForwardMetadata,
        )

        backend = object.__new__(BaseTpuPallasRpaAttnBackend)
        backend.token_to_kv_pool = types.SimpleNamespace(swa_raw_head_num=2)
        backend._swa_kv_heads_merged = True
        backend._swa_layer_groups = {2: 0, 3: 1, 4: 2}
        f0 = torch.tensor([1, 2], dtype=torch.int32)
        s0 = torch.tensor([3, 4], dtype=torch.int32)
        sg = torch.tensor([[13, 14], [23, 24]], dtype=torch.int32)
        backend.forward_metadata = PallasForwardMetadata(
            kv_lens=torch.tensor([8], dtype=torch.int32),
            page_indices=f0,
            cu_q_lens=torch.tensor([0, 8], dtype=torch.int32),
            cu_kv_lens=torch.tensor([0, 8], dtype=torch.int32),
            distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
            active_num_tokens=torch.tensor([8], dtype=torch.int32),
            swa_page_indices=s0,
            swa_group_page_indices=sg,
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(0, is_swa_layer=False)[0], f0)
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(1, is_swa_layer=False)[0], f0)
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(2, is_swa_layer=True)[0], s0)
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(3, is_swa_layer=True)[0], sg[0])
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(4, is_swa_layer=True)[0], sg[1])
        )

    def test_pingpong_two_window_swa_prefill_and_admission(self):
        with self._cfg_patches():
            kvc = self._make_mock_kvc()
            conf = TpuUnifiedSWAPoolConfigurator(kvc)
            budget = 256 * 1024**2
            phys_pages = conf.calculate_total_physical_pages(budget, page_size=128)
            self.assertGreater(phys_pages, 0)
            self.assertEqual(phys_pages % 8, 0)

        _, alloc = TestTpuUnifiedSWAKVPoolAllocator()._make_pool_and_allocator(
            total_pages=64, page_size=128
        )
        self.assertTrue(
            alloc.can_reserve_pingpong_prefill(
                extend_tokens=1024, sliding_window_size=256, has_prev_window=False
            )
        )
        # 3 consecutive 512-token chunks with W = 256 (2 pages per window, M_swa = 4 => 8 phys pages per window)
        last_loc = torch.tensor([-1], dtype=torch.int64)
        for chunk_idx in range(3):
            prefix_len = chunk_idx * 512
            seq_len = (chunk_idx + 1) * 512
            out = alloc.alloc_extend_pingpong_swa(
                req_id="req-pingpong",
                prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
                seq_lens=torch.tensor([seq_len], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=512,
                sliding_window_size=256,
            )
            self.assertIsNotNone(out)
            self.assertEqual(out.numel(), 512)
            # Out-of-window first 256 tokens of the chunk map to dummy page 0
            mapped_head = alloc.translate_loc_from_full_to_swa(out[:256])
            mapped_tail = alloc.translate_loc_from_full_to_swa(out[256:])
            self.assertTrue(bool((mapped_head == 0).all()))
            self.assertTrue(bool((mapped_tail > 0).all()))
            # At most 2 windows (2 * 2 pages * M_swa=4 = 16 physical pages) of SWA are ever live
            self.assertLessEqual(alloc.swa_attn_allocator.allocated_pages, 16)
            last_loc = out[-1:]
        freed = alloc.advance_pingpong_swa_window("req-pingpong")
        self.assertEqual(freed, 256)
        # After advancing, only 1 window (2 pages * M_swa=4 = 8 physical pages) remains
        self.assertEqual(alloc.swa_attn_allocator.allocated_pages, 8)
        alloc.advance_pingpong_swa_window("req-pingpong", release_all=True)
        self.assertEqual(alloc.swa_attn_allocator.allocated_pages, 0)

    def test_radix_core_cascade_reclaim_tombstone_and_prefix_validation(self):
        _, alloc = TestTpuUnifiedSWAKVPoolAllocator()._make_pool_and_allocator(
            total_pages=20, page_size=128
        )
        # Allocate 2 pages (256 tokens): costs 2 * (1 + M_swa=4) = 10 physical pages, leaving 10 free
        idx = alloc.alloc_extend(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([256], dtype=torch.int64),
            torch.tensor([256], dtype=torch.int64),
            torch.tensor([-1], dtype=torch.int64),
            256,
        )
        self.assertIsNotNone(idx)
        # Request 384 tokens (3 pages = 15 physical pages, shortfall = 5 physical pages):
        # With credit_swa_cascade=True, evicting 1 full leaf page cascades to free 1 + 4 = 5 physical pages!
        plan_cascade = alloc.reclaim_plan(
            384,
            384,
            full_evictable_tokens=256,
            swa_evictable_tokens=256,
            credit_swa_cascade=True,
            policy="full_first",
        )
        self.assertEqual(plan_cascade, (128, 0))
        plan_swa_first = alloc.reclaim_plan(
            384,
            384,
            full_evictable_tokens=256,
            swa_evictable_tokens=256,
            policy="swa_first",
        )
        self.assertEqual(plan_swa_first, (0, 256))

        # Validate SWA prefix window before and after tombstoning the tail page
        self.assertEqual(alloc.validate_swa_prefix_window(idx, 128), 256)
        freed_phys = alloc.tombstone_swa_pages(idx[128:])
        self.assertEqual(freed_phys, 4)
        self.assertEqual(alloc.validate_swa_prefix_window(idx, 128), 128)

    def test_timestamp_interleaved_eviction_policy(self):
        from sglang_tpu.unified_kv_pool import EvictionCandidate

        _, alloc = TestTpuUnifiedSWAKVPoolAllocator()._make_pool_and_allocator(
            total_pages=20, page_size=128
        )
        alloc.eviction_policy = "timestamp_interleaved"
        # Candidate list:
        # - Node A SWA window at t=10.0 (128 tokens = 1 page => frees M_swa=4 phys pages)
        # - Node B Full leaf at t=20.0 (128 tokens = 1 page, coupled SWA => frees 1+4=5 phys pages)
        # - Node A Full leaf at t=30.0 (128 tokens = 1 page, SWA already reclaimed at t=10.0 => frees 1 phys page)
        cands = [
            EvictionCandidate("node_B", "full", 128, last_access_time=20.0, has_coupled_swa=True),
            EvictionCandidate("node_A", "swa", 128, last_access_time=10.0),
            EvictionCandidate("node_A", "full", 128, last_access_time=30.0, has_coupled_swa=True),
        ]
        # Need 9 physical pages: t=10.0 (node_A SWA -> 4 phys) + t=20.0 (node_B Full+SWA -> 5 phys) = 9 phys!
        res = alloc.plan_interleaved_eviction(cands, short_physical_pages=9)
        self.assertIsNotNone(res)
        f_tok, s_tok, chosen = res
        self.assertEqual((f_tok, s_tok), (128, 128))
        self.assertEqual([c.node_id for c in chosen], ["node_A", "node_B"])
        # Need 10 physical pages: also pops node_A Full at t=30.0 (only 1 extra phys page since SWA already reclaimed)
        res10 = alloc.plan_interleaved_eviction(cands, short_physical_pages=10)
        self.assertIsNotNone(res10)
        self.assertEqual((res10[0], res10[1]), (256, 128))

    def test_stale_swa_then_whole_branch_protects_forks_and_sweeps_zombies(self):
        from sglang_tpu.unified_kv_pool import EvictionCandidate

        _, alloc = TestTpuUnifiedSWAKVPoolAllocator()._make_pool_and_allocator(
            total_pages=20, page_size=128
        )
        alloc.eviction_policy = "stale_swa_then_whole_branch"
        # Tree:
        # - fork_root (num_children=2, is_stale_swa=True): branched fork point, MUST NOT be evicted in Phase 1!
        # - mid_chain (parent=fork_root, num_children=1, is_stale_swa=True): unbranched interior stale SWA (128 tok -> 4 phys pages)
        # - leaf_child (parent=mid_chain, num_children=0, has_coupled_swa=True): live leaf (128 tok -> 1+4=5 phys pages)
        # When leaf_child is evicted in Phase 3, mid_chain (whose SWA was reclaimed in Phase 1) becomes a spotlight-less Zombie leaf and is swept automatically (+1 phys page)!
        cands = [
            EvictionCandidate(
                "fork_root", "swa", 128, last_access_time=1.0, num_children=2, is_stale_swa=True
            ),
            EvictionCandidate(
                "mid_chain",
                "swa",
                128,
                last_access_time=5.0,
                num_children=1,
                is_stale_swa=True,
            ),
            EvictionCandidate(
                "mid_chain",
                "full",
                128,
                last_access_time=5.0,
                has_coupled_swa=True,
                num_children=1,
                parent_id="fork_root",
            ),
            EvictionCandidate(
                "leaf_child",
                "full",
                128,
                last_access_time=15.0,
                has_coupled_swa=True,
                num_children=0,
                parent_id="mid_chain",
            ),
        ]
        # Shortfall = 4 physical pages: satisfied purely by mid_chain's unbranched stale SWA window; fork_root SWA is untouched!
        res4 = alloc.plan_stale_swa_then_whole_branch_eviction(cands, short_physical_pages=4)
        self.assertIsNotNone(res4)
        self.assertEqual((res4[0], res4[1]), (0, 128))
        self.assertEqual([ (c.node_id, c.kind) for c in res4[2] ], [("mid_chain", "swa")])

        # Shortfall = 9 physical pages: Phase 1 reclaims mid_chain SWA (4 phys), Phase 3 evicts leaf_child (5 phys) AND sweeps mid_chain Zombie Full leaf (+1 phys = 10 phys total)!
        res9 = alloc.plan_stale_swa_then_whole_branch_eviction(cands, short_physical_pages=9)
        self.assertIsNotNone(res9)
        self.assertEqual((res9[0], res9[1]), (256, 128))
        self.assertEqual(
            [(c.node_id, c.kind) for c in res9[2]],
            [("mid_chain", "swa"), ("leaf_child", "full"), ("mid_chain", "full")],
        )

    def test_cost_benefit_weighted_eviction_protects_long_prompts(self):
        from sglang_tpu.unified_kv_pool import EvictionCandidate

        _, alloc = TestTpuUnifiedSWAKVPoolAllocator()._make_pool_and_allocator(
            total_pages=40, page_size=128
        )
        alloc.eviction_policy = "cost_benefit_weighted"
        # Compare:
        # - long_prompt_32k: older (t=1.0), 4096 tokens at prefix depth 32768 (expensive quadratic recompute!)
        # - short_prompt_1: newer (t=50.0), 256 tokens at prefix depth 256 (cheap recompute, frees 2*(1+4)=10 phys pages)
        # - stale_swa_mid: t=100.0, unbranched stale SWA (128 tokens, 0 recompute cost, frees 4 phys pages)
        cands = [
            EvictionCandidate(
                "long_prompt_32k",
                "full",
                4096,
                last_access_time=1.0,
                has_coupled_swa=True,
                prefix_tokens=32768,
            ),
            EvictionCandidate(
                "short_prompt_1",
                "full",
                256,
                last_access_time=50.0,
                has_coupled_swa=True,
                prefix_tokens=256,
            ),
            EvictionCandidate(
                "stale_swa_mid",
                "swa",
                128,
                last_access_time=100.0,
                num_children=1,
                is_stale_swa=True,
            ),
        ]
        # Need 14 physical pages: cost_benefit_weighted picks stale_swa_mid (score=0, 4 phys) + short_prompt_1 (10 phys)
        # even though long_prompt_32k has the oldest timestamp (t=1.0)!
        res = alloc.plan_cost_benefit_eviction(cands, short_physical_pages=14)
        self.assertIsNotNone(res)
        self.assertEqual((res[0], res[1]), (256, 128))
        self.assertEqual([c.node_id for c in res[2]], ["stale_swa_mid", "short_prompt_1"])

    def test_observability_metrics_physical_page_breakdown_and_hit_rates(self):
        _, alloc = TestTpuUnifiedSWAKVPoolAllocator()._make_pool_and_allocator(
            total_pages=20, page_size=128
        )
        # Allocate 2 pages (256 tokens): 2 Full physical pages + 2 * 4 = 8 SWA physical pages = 10 used, 10 free
        alloc.alloc_extend(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([256], dtype=torch.int64),
            torch.tensor([256], dtype=torch.int64),
            torch.tensor([-1], dtype=torch.int64),
            256,
        )
        alloc.record_prefix_cache_lookup(384, 512, policy="full_first")
        alloc.record_prefix_cache_lookup(480, 512, policy="cost_benefit_weighted")
        metrics = alloc.get_observability_metrics(
            full_evictable_tokens=128, swa_evictable_tokens=128
        )
        self.assertEqual(metrics.total_usable_physical_pages, 20)
        self.assertEqual(metrics.full_locked_physical_pages, 1)
        self.assertEqual(metrics.full_cached_evictable_physical_pages, 1)
        self.assertEqual(metrics.swa_locked_physical_pages, 4)
        self.assertEqual(metrics.swa_cached_evictable_physical_pages, 4)
        self.assertEqual(metrics.fenced_physical_pages, 0)
        self.assertEqual(metrics.free_physical_pages, 10)
        self.assertAlmostEqual(metrics.kv_page_utilization_ratio, 0.5)
        self.assertAlmostEqual(metrics.prefix_cache_hit_rate, 0.75)
        self.assertAlmostEqual(
            metrics.prefix_cache_hit_rates_by_policy["cost_benefit_weighted"], 0.9375
        )

    def test_kv_geometry_models_tp_shardings_and_seq_along_lane_fp8_savings(self):
        from sglang_tpu.unified_kv_heads import merge_swa_kv_heads, split_swa_output
        from sglang_tpu.unified_kv_pool import (
            TpuUnifiedSWAKVPool,
            resolve_unified_page_shape,
        )

        # 1. Muse Spark Open across TP=1,2,4,8 (full_dim=512, swa_dim=256, swa_heads = 2 * full_heads)
        for full_heads, swa_heads in [(8, 16), (4, 8), (2, 4), (1, 2)]:
            shape, merged = resolve_unified_page_shape(
                page_size=128,
                head_num=full_heads,
                head_dim=512,
                v_head_dim=512,
                swa_head_num=swa_heads,
                swa_head_dim=256,
                swa_v_head_dim=256,
                store_dtype=torch.bfloat16,
            )
            self.assertTrue(merged)
            q = torch.randn(4, swa_heads * 2, 256)
            k = torch.randn(4, swa_heads, 256)
            v = torch.randn(4, swa_heads, 256)
            qm, km, vm = merge_swa_kv_heads(q, k, v, target_kv_heads=full_heads)
            self.assertEqual(km.shape, (4, full_heads, 512))
            self.assertEqual(vm.shape, (4, full_heads, 512))
            restored = split_swa_output(qm, swa_heads, target_kv_heads=full_heads)
            self.assertTrue(torch.allclose(restored, q))

        # 2. MiMo (asymmetric v_head_dim=128 < head_dim=192) & GPT-OSS-120B / Gemma
        mimo_shape, mimo_merged = resolve_unified_page_shape(
            page_size=128,
            head_num=2,
            head_dim=192,
            v_head_dim=128,
            swa_head_num=2,
            swa_head_dim=192,
            swa_v_head_dim=128,
            store_dtype=torch.bfloat16,
        )
        self.assertFalse(mimo_merged)
        self.assertEqual(mimo_shape, (128, 2, 2, 256))

        # 3. SEQ_ALONG_LANE vs SEQ_ACROSS_LANE at 1 KV head per rank in fp8:
        # SEQ_ACROSS_LANE rounds 2*1=2 up to kv_packing=4 (50% waste), whereas SEQ_ALONG_LANE uses 2*1=2 directly!
        with patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU v6e",
        ):
            pool_across = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=1,
                head_dim=128,
                swa_attention_layer_ids=[1, 2],
                full_attention_layer_ids=[0],
                device="cpu",
                seq_along_lane=False,
            )
            pool_along = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=1,
                head_dim=128,
                swa_attention_layer_ids=[1, 2],
                full_attention_layer_ids=[0],
                device="cpu",
                seq_along_lane=True,
            )
        # SEQ_ALONG_LANE buffer is [num_blocks=3, head_num*2=2, hd_words=32, packing=4, page_size=128]
        # which is exactly 50% of SEQ_ACROSS_LANE [num_blocks=3, page_size=128, kv_heads_packed=1, packing=4, head_dim=128]!
        self.assertEqual(
            pool_along.full_kv_pool.kv_buffer[0].numel() * 2,
            pool_across.full_kv_pool.kv_buffer[0].numel(),
        )

    def test_quantized_kv_scale_buffers_mxint8_mxfp8_nvfp4(self):
        from sglang_tpu.kv_mxint8 import EXPONENT_PAD_BYTE
        from sglang_tpu.unified_kv_pool import TpuUnifiedSWAKVPool

        for fmt in ("mxint8", "mxfp8", "nvfp4"):
            pool = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.int8 if fmt == "mxint8" else torch.float8_e4m3fn,
                head_num=2,
                head_dim=128,
                swa_attention_layer_ids=[1, 2, 3, 4],
                full_attention_layer_ids=[0, 5],
                device="cpu",
                quant_kv_format=fmt,
            )
            self.assertIsNotNone(pool.shared_kv_scale_buffer)
            self.assertEqual(len(pool.shared_kv_scale_buffer), 2)
            s0 = pool.get_kv_scale_buffer(0)
            s1 = pool.get_kv_scale_buffer(1)
            s3 = pool.get_kv_scale_buffer(3)
            # SWA layer 1 (idx 0) and SWA layer 3 (idx 2) share slab 0's scale buffer with Full layer 0 (idx 0)!
            self.assertEqual(s0.data_ptr(), s1.data_ptr())
            self.assertEqual(s0.data_ptr(), s3.data_ptr())
            self.assertEqual(s0.shape, (3, 32, 128))
            self.assertTrue(bool((s0 == EXPONENT_PAD_BYTE).all()))


if __name__ == "__main__":
    unittest.main()




