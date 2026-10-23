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
                ("token_to_kv_pool_class", TpuSeqAlongLaneKVPool, ValueError),
                ("kv_cache_dtype_str", "mxint8", NotImplementedError),
                ("pool_page_size", 1, ValueError),
            ):
                kvc = copy.copy(base)
                setattr(kvc, attr, val)
                with self.assertRaises(exc):
                    TpuUnifiedSWAPoolConfigurator(kvc)

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


class TestTpuRaidenUnifiedHostKVCache(unittest.TestCase):
    """Tests for PR 4: unified Raiden host KV cache offload and shared eviction."""

    class _FakeRaidenKVTransfer:
        def __init__(self, device_pool, num_host_pages):
            self.device_pool = device_pool
            self.num_host_pages = num_host_pages
            first = device_pool.get_fused_kv_buffer(device_pool.start_layer)
            self.host_buffers = [
                torch.zeros((num_host_pages, *first.shape[1:]), dtype=first.dtype)
                for _ in range(device_pool.layer_num)
            ]

        def backup(self, device_pages, host_pages):
            for l, hbuf in enumerate(self.host_buffers):
                dbuf = self.device_pool.get_fused_kv_buffer(
                    self.device_pool.start_layer + l
                )
                for dp, hp in zip(device_pages, host_pages):
                    hbuf[hp].copy_(dbuf[dp])
            return "fake_backup_event"

        def load(self, host_pages, device_pages):
            for l, hbuf in enumerate(self.host_buffers):
                dbuf = self.device_pool.get_fused_kv_buffer(
                    self.device_pool.start_layer + l
                )
                for hp, dp in zip(host_pages, device_pages):
                    dbuf[dp].copy_(hbuf[hp])
            return "fake_load_event"

    def test_shared_raiden_host_pool_roundtrip_and_platform_selection(self):
        from sglang_tpu.kv_pool_host import TpuRaidenUnifiedHostKVCache
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        # 8 Full + 20 SWA -> G=8, M_swa=ceil(20/8)=3; 20 physical pages (80 tokens).
        pool = TpuUnifiedSWAKVPool(
            size=80,
            size_swa=80,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=512,
            swa_head_num=2,
            swa_head_dim=256,
            swa_attention_layer_ids=list(range(8, 28)),
            full_attention_layer_ids=list(range(8)),
            device="cpu",
        )
        with patch(
            "sglang.srt.runtime_context.get_memory",
            return_value=types.SimpleNamespace(hicache_io_backend="raiden"),
        ):
            self.assertIs(
                TpuSRTPlatform().get_mha_kv_pool_host_cls(pool.full_kv_pool),
                TpuRaidenUnifiedHostKVCache,
            )

        with patch(
            "sglang_tpu.raiden_transfer.RaidenKVTransfer",
            self._FakeRaidenKVTransfer,
        ), patch(
            "sglang_tpu.kv_pool_host.host_memory_budget_bytes",
            return_value=10 * 1024**3,
        ):
            full_host = TpuRaidenUnifiedHostKVCache(
                device_pool=pool.full_kv_pool,
                host_to_device_ratio=1.0,
                host_size=0,
                page_size=4,
                layout="page_first",
            )
            swa_host = TpuRaidenUnifiedHostKVCache(
                device_pool=pool.swa_kv_pool,
                host_to_device_ratio=1.0,
                host_size=0,
                page_size=4,
                layout="page_first",
            )

        self.assertIs(full_host.transfer, swa_host.transfer)
        self.assertEqual((full_host.available_size(), swa_host.available_size()), (80, 24))

        # Full backup/restore across all 8 shared buffers (1 physical page).
        for i in range(8):
            pool.get_state()[i][2].fill_(float(50 + i))
        h_full = full_host.alloc(4)
        full_host.backup_from_device_all_layer(
            pool.full_kv_pool, h_full, torch.arange(8, 12), "raiden"
        )

        # SWA backup/restore across M_swa = 3 groups (3 physical pages).
        pool.swa_group_pages[:, 3] = torch.tensor([7, 11], dtype=torch.int32)
        for j in range(20):
            p = 3 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 3])
            pool.get_state()[j % 8][p].fill_(float(10 + j))
        h_swa = swa_host.alloc(4)
        swa_host.backup_from_device_all_layer(
            pool.swa_kv_pool, h_swa, torch.arange(12, 16), "raiden"
        )
        self.assertEqual(len(full_host._state.free_pages), 16)

        for b in pool.get_state():
            b.zero_()
        full_host.load_all_layer(pool.full_kv_pool, h_full, torch.arange(20, 24))
        for i in range(8):
            self.assertTrue(torch.all(pool.get_state()[i][5] == float(50 + i)))
        pool.swa_group_pages[:, 6] = torch.tensor([2, 8], dtype=torch.int32)
        swa_host.load_all_layer(pool.swa_kv_pool, h_swa, torch.arange(24, 28))
        for j in range(20):
            p = 6 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 6])
            self.assertTrue(torch.all(pool.get_state()[j % 8][p] == float(10 + j)))

        full_host.free(h_full)
        swa_host.free(h_swa)
        self.assertEqual(len(full_host._state.free_pages), 20)

    def test_shared_host_eviction_fallback_frees_sibling_component(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            build_hybrid_swa_group,
        )

        # 8 Full + 20 SWA -> G=8, M_swa=3; 12 physical pages (48 tokens).
        pool = TpuUnifiedSWAKVPool(
            size=48,
            size_swa=48,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=512,
            swa_head_num=2,
            swa_head_dim=256,
            swa_attention_layer_ids=list(range(8, 28)),
            full_attention_layer_ids=list(range(8)),
            device="cpu",
        )
        allocator = TpuUnifiedSWATokenToKVPoolAllocator(
            size=48,
            size_swa=48,
            page_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        mem_mock = types.SimpleNamespace(
            enable_hierarchical_cache=True,
            hicache_ratio=1.0,
            hicache_size=0,
            hicache_mem_layout="page_first",
            hicache_io_backend="raiden",
            hicache_write_policy="write_through",
            hicache_storage_backend=None,
            hicache_storage_prefetch_policy="best_effort",
            hicache_storage_backend_extra_config=None,
            enable_metrics=False,
            enable_unified_memory=False,
        )
        parallel_mock = types.SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            pp_rank=0,
            pp_size=1,
            dcp_enabled=False,
        )

        full_host_holders = []
        calls = []

        def _host_swa_evict(num_tokens):
            calls.append(("swa", num_tokens))
            return 0

        def _host_full_evict(num_tokens):
            calls.append(("full", num_tokens))
            pages = (num_tokens + 3) // 4
            freed = 0
            while full_host_holders and pages > 0:
                h = full_host_holders.pop(0)
                full_entry.host_pool.free(h)
                pages -= 1
                freed += int(h.numel())
            return freed

        with patch(
            "sglang_tpu.raiden_transfer.RaidenKVTransfer",
            self._FakeRaidenKVTransfer,
        ), patch(
            "sglang_tpu.kv_pool_host.host_memory_budget_bytes",
            return_value=10 * 1024**3,
        ), patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_memory",
            return_value=mem_mock,
        ), patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_parallel",
            return_value=parallel_mock,
        ), patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler._get_allocator_type",
            return_value="default",
        ), patch(
            "sglang.srt.runtime_context.get_memory",
            return_value=mem_mock,
        ):
            group = build_hybrid_swa_group(
                page_size=4,
                full_kv_pool=pool.full_kv_pool,
                swa_kv_pool=pool.swa_kv_pool,
                full_layer_mapping={i: i for i in range(8)},
                swa_layer_mapping={8 + j: j for j in range(20)},
                use_mla=False,
                host_full_evict_fn=_host_full_evict,
                host_swa_evict_fn=_host_swa_evict,
                swa_attn_allocator=allocator.swa_attn_allocator,
            )

        full_entry = group.entry_map[PoolName.KV]
        swa_entry = group.entry_map[PoolName.SWA]
        self.assertIs(full_entry.host_pool._state, swa_entry.host_pool._state)

        # Exhaust all 12 Full logical pages (= 12 physical pages -> 0 free).
        for _ in range(12):
            h = full_entry.host_pool.alloc(4)
            self.assertIsNotNone(h)
            full_host_holders.append(h)
        self.assertEqual(swa_entry.host_pool.available_size(), 0)

        # Trigger SWA host eviction callback: SWA has nothing to evict (returns 0),
        # so the shared-storage fallback invokes _host_full_evict(12) and frees 3 Full
        # logical pages (= 3 physical pages needed for 1 SWA page).
        swa_entry.host_evict_fn(4)
        self.assertEqual(calls, [("swa", 4), ("full", 12)])
        self.assertEqual(swa_entry.host_pool.available_size(), 4)


if __name__ == "__main__":
    unittest.main()



