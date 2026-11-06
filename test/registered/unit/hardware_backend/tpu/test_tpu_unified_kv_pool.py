# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import copy
import types
import unittest
from array import array
from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import (
    get_kv_transfer_buf_infos,
    setup_state_kv_args,
)
from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache.allocator import swa
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pool_host import base as pool_host_base
from sglang.srt.mem_cache.pool_host.base import uses_shared_host_layout
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import hooks, tpu_dma_hicache
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks.hicache_swa_metrics import used_tokens
from sglang_tpu.kv_mxint8 import TOKEN_BLOCK, blocked_page_rows, cache_bytes_per_token
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool
from sglang_tpu.kv_pool_host import (
    _TPU_UNIFIED_SWA_STRATEGY,
    TpuDMAUnifiedHostKVPool,
    register_unified_hicache_strategy,
)
from sglang_tpu.tpu_dma_hicache import NOOP_EVENT
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController
from sglang_tpu.tpu_platform import TpuSRTPlatform
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWAPoolConfigurator,
    TpuUnifiedSWATokenToKVPoolAllocator,
    UnifiedPageLayout,
    num_swa_groups,
    unified_kv_pool_enabled,
    unified_kv_pool_requested,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuUnifiedSWAKVPoolStorage(unittest.TestCase):
    """Tests for PR 1: shared physical buffers and SWA group page mapping."""

    def test_equal_full_and_swa_layers_share_buffers(self):
        pool = TpuUnifiedSWAKVPool(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=4, head_dim=128, layer_ids=[1, 3]),
            swa=AttnLayerSpec(head_num=4, head_dim=128, layer_ids=[0, 2]),
            device="cpu",
        )
        self.assertIsInstance(pool, BaseSWAKVPool)
        self.assertNotIsInstance(pool, SWAKVPool)
        self.assertEqual(pool.size, 64)
        self.assertEqual(
            pool.layout,
            UnifiedPageLayout(
                page_head_num=4,
                page_head_dim=128,
                num_shared_buffers=2,
                num_swa_groups=1,
            ),
        )
        self.assertEqual(len(pool.get_state()), 2)
        for full_lid, swa_lid in ((1, 0), (3, 2)):
            self.assertEqual(
                pool.get_fused_kv_buffer(full_lid).data_ptr(),
                pool.get_fused_kv_buffer(swa_lid).data_ptr(),
            )
            self.assertEqual(pool.swa_page_group(swa_lid), 0)
        self.assertEqual(pool.get_kv_size_bytes(), pool.kv_pool.get_kv_size_bytes())
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_multi_group_swa_ratio_bf16_and_fp8(self):
        full_ids = [5, 11]
        swa_ids = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = TpuUnifiedSWAKVPool(
                    size=80,
                    page_size=4,
                    dtype=dtype,
                    full=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=full_ids),
                    swa=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=swa_ids),
                    device="cpu",
                )
                self.assertEqual(
                    (pool.layout.num_swa_groups, pool.layout.num_shared_buffers),
                    (5, 2),
                )
                self.assertEqual(pool.kv_pool.num_blocks, 21)
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

    def test_mismatched_page_shapes_rejected(self):
        for swa_heads, swa_dim in ((1, 256), (4, 256), (2, 128), (2, 512)):
            with self.subTest(swa_heads=swa_heads, swa_dim=swa_dim):
                with self.assertRaises(ValueError):
                    TpuUnifiedSWAKVPool(
                        size=32,
                        page_size=4,
                        dtype=torch.bfloat16,
                        full=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=[2]),
                        swa=AttnLayerSpec(
                            head_num=swa_heads, head_dim=swa_dim, layer_ids=[0, 1]
                        ),
                        device="cpu",
                    )

    def test_non_divisible_swa_groups_and_cpu_copy_restore(self):
        self.assertEqual(num_swa_groups(8, 20), 3)
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=48,
            page_size=page_size,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=range(8)),
            swa=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=range(8, 28)),
            device="cpu",
        )
        self.assertEqual(
            (pool.layout.num_shared_buffers, pool.layout.num_swa_groups), (8, 3)
        )
        # Prefix Full page 1 (unmapped in SWA) + tail Full page 2 -> SWA primary page 3
        # with companion pages [9, 11], restored into Full pages [4, 5] and SWA [6, 2, 10].
        mapping = torch.zeros(48 + page_size + 1, dtype=torch.int64)
        mapping[-1] = -1
        mapping[2 * page_size : 3 * page_size] = torch.arange(
            3 * page_size, 4 * page_size
        )
        pool.register_mapping(mapping)
        self.assertTrue(
            torch.equal(
                pool.translate_loc_from_full_to_swa(torch.tensor([8, 9, -1])),
                torch.tensor([12, 13, -1], dtype=torch.int32),
            )
        )
        pool.swa_group_pages[:, 3] = torch.tensor([9, 11], dtype=torch.int32)
        for i in range(8):
            pool.get_state()[i][1].fill_(float(50 + i))
            pool.get_state()[i][2].fill_(float(100 + i))
        for j in range(20):
            p = 3 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 3])
            pool.get_state()[j % 8][p].fill_(float(10 + j))

        src_full = torch.arange(1 * page_size, 3 * page_size)
        cpu_copy = pool.get_cpu_copy(src_full)

        mapping.zero_()
        mapping[-1] = -1
        mapping[5 * page_size : 6 * page_size] = torch.arange(
            6 * page_size, 7 * page_size
        )
        pool.swa_group_pages[:, 6] = torch.tensor([2, 10], dtype=torch.int32)
        dst_full = torch.arange(4 * page_size, 6 * page_size)
        pool.load_cpu_copy(cpu_copy, dst_full)

        for i in range(8):
            self.assertTrue(torch.all(pool.get_state()[i][4] == float(50 + i)))
            self.assertTrue(torch.all(pool.get_state()[i][5] == float(100 + i)))
        for j in range(20):
            p = 6 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 6])
            self.assertTrue(torch.all(pool.get_state()[j % 8][p] == float(10 + j)))

        # Verify Raiden retraction backup stages Full pages via handle while SWA uses row-copy
        from sglang_tpu.raiden_kv_backup import RaidenKVBackupHandle

        saved_indices = []
        restored = []

        class _FakeRaidenBackup:
            def _free_pages(self, pages):
                pass

            def try_save(self, idx):
                saved_indices.append(idx.tolist())
                return RaidenKVBackupHandle(self, [0, 1], 0, idx.numel())

            def restore(self, handle, idx):
                restored.append((list(handle.host_pages), idx.tolist()))

        pool.kv_pool._raiden_backup_obj = _FakeRaidenBackup()
        raiden_copy = pool.get_cpu_copy(dst_full)
        self.assertIsInstance(raiden_copy["full"], RaidenKVBackupHandle)
        self.assertEqual(len(saved_indices), 1)
        self.assertIsInstance(raiden_copy["swa"], list)
        # Shrink SWA mapping on restore to exercise filter_cpu_copy alongside Raiden Full restore
        mapping[5 * page_size] = 0
        pool.load_cpu_copy(raiden_copy, dst_full)
        self.assertEqual(restored, [([0, 1], dst_full.tolist())])

        # Verify TpuSeqAlongLaneKVPool saves 50% HBM vs TpuFusedMHATokenToKVPool at head_num=1 in fp8
        with patch(
            "sglang_tpu.kv_pool._seq_along_lane_head_dim",
            side_effect=lambda *, head_dim, kv_packing: (
                ((head_dim + 8 * kv_packing - 1) // (8 * kv_packing)) * (8 * kv_packing)
            ),
        ):
            pool_across = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                full=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[0]),
                swa=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[1, 2]),
                device="cpu",
                token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
            )
            pool_along = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                full=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[0]),
                swa=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[1, 2]),
                device="cpu",
                token_to_kv_pool_class=TpuSeqAlongLaneKVPool,
            )
        self.assertEqual(
            pool_along.kv_pool.kv_buffer[0].numel() * 2,
            pool_across.kv_pool.kv_buffer[0].numel(),
        )


class TestTpuUnifiedSWAKVPoolAllocator(unittest.TestCase):
    """Tests for PR 2: zero-fragmentation Logical-to-Physical unified allocator."""

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
            page_size=page_size,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=full_ids),
            swa=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=swa_ids),
            device="cpu",
        )
        alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=size,
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
            (
                alloc.full_available_size(),
                alloc.swa_available_size(),
                alloc.max_extend_tokens(),
            ),
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
            (
                alloc.full_available_size(),
                alloc.swa_available_size(),
                alloc.max_extend_tokens(),
            ),
            (80, 24, 20),
        )
        full_tok = alloc.full_attn_allocator.alloc(12)  # 3 * 1 = 3 phys IDs
        swa_tok = alloc.swa_attn_allocator.alloc(8)  # 2 * 3 = 6 phys IDs
        self.assertEqual(alloc._free_page_count(), 11)
        f_p = (full_tok[::4] // 4).tolist()
        s_p = (swa_tok[::4] // 4).tolist()
        claimed = f_p + s_p + pool.swa_group_pages[:, s_p].reshape(-1).tolist()
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
        self.assertTrue(
            alloc.can_reserve(8, 8, full_evictable_tokens=8, swa_evictable_tokens=8)
        )
        self.assertFalse(
            alloc.can_reserve(8, 8, full_evictable_tokens=4, swa_evictable_tokens=0)
        )

        # 1) Contract check: verify sequential Full -> SWA shortfall passes (short=5, ps=4, M_swa=4).
        mock_cache = MagicMock(spec=BasePrefixCache)
        mock_cache.supports_prefix_sharing.return_value = True
        self.assertFalse(alloc.evict_to_free_tokens(mock_cache, 8, swa_num_tokens=8))
        self.assertEqual(
            mock_cache.evict_for_alloc.call_args_list,
            [
                call(EvictParams(num_tokens=20, swa_num_tokens=0)),
                call(EvictParams(num_tokens=0, swa_num_tokens=8)),
            ],
        )

        # 2) End-to-end check on CPU with a real UnifiedRadixCache.
        cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=ReqToTokenPool(
                    size=4, max_context_len=64, device="cpu", enable_memory_saver=False
                ),
                token_to_kv_pool_allocator=alloc,
                page_size=4,
                disable=False,
                sliding_window_size=8,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                eviction_policy="lru",
                tree_core_backend="python",
            )
        )
        cache.insert(InsertParams(key=RadixKey(array("q", range(1, 9))), value=out))
        self.assertEqual(
            (cache.full_evictable_size(), cache.swa_evictable_size()), (8, 8)
        )

        self.assertIsNone(alloc.reclaim_for_prealloc(cache, 8, 8))
        self.assertEqual(alloc._free_page_count(), 15)
        self.assertIsNotNone(alloc.reclaim_for_prealloc(cache, 16, 16))
        self.assertTrue(alloc.prealloc_fits_assumes_reclaim())
        self.assertTrue(alloc.prealloc_ceiling_fits(8, 8))
        # 15 free phys pages (ps=4, M_swa=4): full_available=60, swa_available=12, evictable=0.
        self.assertTrue(
            alloc.prealloc_fits(
                cache, 28, 8, full_budget_tokens=60, swa_budget_tokens=12
            )
        )
        # Individually 48 <= 60 and 8 <= 12, but jointly 12 + 4*2 = 20 > 15 phys pages -> False.
        self.assertFalse(
            alloc.prealloc_fits(
                cache, 48, 8, full_budget_tokens=60, swa_budget_tokens=12
            )
        )
        # Deducting 8 reserved Full tokens (full_budget=52) leaves 13 phys pages -> 28+8 needs 15 -> False.
        self.assertFalse(
            alloc.prealloc_fits(
                cache, 28, 8, full_budget_tokens=52, swa_budget_tokens=12
            )
        )
        # Pool size_full=80, ps=4, M_swa=4: window=8 reserves 4*(8+4)=48 -> max_request_tokens=32; None -> 16.
        self.assertEqual(alloc.max_request_tokens(8), 32)
        self.assertEqual(alloc.max_request_tokens(None), 16)

    def test_kv_free_fence_shared_held_pages_and_stats(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.kv_free_fence import install_kv_free_fence

        # M_swa = 4, page_size = 4: 1 joint page = 1 Full + 4 SWA = 5 physical pages.
        _, alloc = self._make_pool_and_allocator(total_pages=10, page_size=4)
        fence = install_kv_free_fence(
            alloc,
            synchronize_device=lambda: None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            copies_draft_tail=False,
        )
        out = alloc.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([4]),
            seq_lens_cpu=torch.tensor([4]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=4,
        )
        self.assertIsNotNone(out)

        result = SimpleNamespace(copy_done=None)
        fence.on_batch_formed()
        fence.on_forward_launched(result)
        alloc.free(out)
        # Full and SWA sub-allocators share one HeldPages queue (5 physical pages total).
        self.assertEqual(alloc.full_attn_allocator.held_pages.num_pages, 5)
        self.assertIs(
            alloc.full_attn_allocator.held_pages,
            alloc.swa_attn_allocator.held_pages,
        )

        # Stats logging must deduplicate the shared HeldPages instance (held now = 5, not 10).
        fence._stats_since -= 120.0
        with self.assertLogs("sglang_tpu.kv_free_fence", level="INFO") as cm:
            fence._maybe_log_stats()
        self.assertTrue(
            any("pages held 5" in msg and "held now 5;" in msg for msg in cm.output),
            cm.output,
        )

        fence.on_forward_done(result)
        self.assertEqual(alloc.full_attn_allocator.held_pages.num_pages, 0)
        self.assertEqual(alloc._free_page_count(), 10)

    def test_evict_to_free_tokens_no_cross_component_rounding_bugs(self):
        def _setup(total_pages: int):
            # M_swa = 4, page_size = 4, sliding_window_size = 4.
            _, alloc = self._make_pool_and_allocator(
                total_pages=total_pages, page_size=4
            )
            cache = UnifiedRadixCache(
                params=CacheInitParams(
                    req_to_token_pool=ReqToTokenPool(
                        size=4,
                        max_context_len=64,
                        device="cpu",
                        enable_memory_saver=False,
                    ),
                    token_to_kv_pool_allocator=alloc,
                    page_size=4,
                    disable=False,
                    sliding_window_size=4,
                    tree_components=(ComponentType.FULL, ComponentType.SWA),
                    eviction_policy="lru",
                    tree_core_backend="python",
                )
            )
            # 1) 4-token prompt whose SWA was already evicted before insert (1 Full phys page).
            #    Allocated first so its transient 5-page alloc_extend fits even in tight pools.
            out_b = alloc.alloc_extend(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([4]),
                seq_lens_cpu=torch.tensor([4]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=4,
            )
            self.assertIsNotNone(out_b)
            alloc.free_swa(out_b)
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", range(200, 204))),
                    value=out_b,
                    component_evicted_seqlens={ComponentType.SWA: 4},
                )
            )

            # 2) 8-token prompt (10 phys pages): locking tip with window=4 locks FULL on
            #    both pages [100..104] & [104..108], but locks SWA only on [104..108],
            #    leaving interior [100..104] SWA-evictable (4 phys pages).
            out_a = alloc.alloc_extend(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([8]),
                seq_lens_cpu=torch.tensor([8]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=8,
            )
            self.assertIsNotNone(out_a)
            key_a = RadixKey(array("q", range(100, 108)))
            cache.insert(InsertParams(key=key_a, value=out_a))
            match_a = cache.match_prefix(MatchPrefixParams(key=key_a))
            cache.inc_lock_ref(match_a.last_device_node)
            self.assertEqual(
                (cache.full_evictable_size(), cache.swa_evictable_size()), (4, 4)
            )
            return alloc, cache

        # Mode A (under-eviction regression): 14 total - 10 (A) - 1 (B) = 3 free pages.
        # Need (3 Full, 1 SWA) = 7 phys pages (short = 4). Pass 1 frees B (+1 -> 4 free,
        # crossing floor(free / 4) == 1); Pass 2 must still tombstone A's interior SWA (+4 -> 8 free).
        alloc, cache = _setup(total_pages=14)
        self.assertEqual(alloc._free_page_count(), 3)
        self.assertTrue(alloc.evict_to_free_tokens(cache, 12, swa_num_tokens=4))
        self.assertEqual(alloc._free_page_count(), 8)
        self.assertEqual(
            (cache.full_evictable_size(), cache.swa_evictable_size()), (0, 0)
        )

        # Mode B (over-eviction regression): 11 total - 10 (A) - 1 (B) = 0 free pages.
        # Need (1 Full, 0 SWA) = 1 phys page (short = 1). Pass 1 frees B (+1 -> 1 free) and
        # satisfies short; Pass 2 must not run and must preserve A's interior SWA.
        alloc, cache = _setup(total_pages=11)
        self.assertEqual(alloc._free_page_count(), 0)
        self.assertTrue(alloc.evict_to_free_tokens(cache, 4, swa_num_tokens=0))
        self.assertEqual(alloc._free_page_count(), 1)
        self.assertEqual(
            (cache.full_evictable_size(), cache.swa_evictable_size()), (0, 4)
        )


class TestTpuUnifiedSWAKVPoolWiring(unittest.TestCase):
    """Tests for PR 3: configurator sizing, RPA page-table selection, and CLI guards."""

    def _cfg_patches(self, **server_arg_overrides):
        from contextlib import ExitStack

        from sglang.srt.runtime_context import get_context, get_parallel

        stack = ExitStack()
        tpu_unified_kv_pool = server_arg_overrides.pop("tpu_unified_kv_pool", True)
        defaults = dict(
            attention_backend="pallas_tpu_rpa_v3",
            tp_size=1,
            pp_size=1,
            dcp_size=1,
            attn_cp_size=1,
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
            hicache_io_backend="raiden",
            enable_unified_cache_external_linker=False,
            enable_unified_memory=False,
            page_size=128,
            disable_radix_cache=False,
            swa_full_tokens_ratio=0.5,
            chunked_prefill_size=4096,
            disable_overlap_schedule=False,
            max_running_requests=None,
        )
        defaults.update(server_arg_overrides)
        stack.enter_context(
            tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL.override(tpu_unified_kv_pool)
        )
        stack.enter_context(get_context().override_server_args(**defaults))
        stack.enter_context(
            get_parallel().override(
                attn_tp_size=defaults["tp_size"] // defaults["attn_cp_size"],
                attn_dcp_size=defaults["dcp_size"],
                dcp_enabled=defaults["dcp_size"] > 1,
            )
        )
        return stack

    def _make_mock_kvc(
        self,
        full_ids=None,
        swa_ids=None,
        head_num=2,
        head_dim=256,
        swa_head_num=2,
        swa_head_dim=256,
        dtype=torch.bfloat16,
        pool_cls=TpuFusedMHATokenToKVPool,
    ):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        full_ids = list(range(10)) if full_ids is None else list(full_ids)
        swa_ids = list(range(10, 60)) if swa_ids is None else list(swa_ids)
        mc = types.SimpleNamespace(
            attention_arch=None,
            has_asymmetric_kv=False,
            head_dim=head_dim,
            v_head_dim=head_dim,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_head_dim,
            get_num_kv_heads=lambda tp=1, dcp=1: head_num,
            get_swa_num_kv_heads=lambda tp=1, dcp=1: swa_head_num,
        )
        elem = torch.empty((), dtype=dtype).element_size()
        return types.SimpleNamespace(
            device="cpu",
            server_args=None,
            is_hybrid_swa=True,
            use_mla_backend=False,
            mambaish_config=None,
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            pp_size=1,
            attn_dp_size=1,
            _tpu_mha_pool_class=lambda: pool_cls,
            kv_cache_dtype_str="fp8_e4m3" if elem == 1 else "bf16",
            kv_cache_dtype=dtype,
            kv_group_size=None,
            pool_page_size=128,
            page_size=128,
            sliding_window_size=1024,
            model_config=mc,
            layer_info=types.SimpleNamespace(
                start_layer=0,
                full_attention_layer_ids=full_ids,
                swa_attention_layer_ids=swa_ids,
            ),
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
                    self.assertEqual(
                        cfg.max_total_num_tokens,
                        conf.max_request_tokens(
                            cfg.full_max_total_num_tokens, page_size=128
                        ),
                    )
                    self.assertGreater(
                        conf.calculate_total_physical_pages(budget, page_size=128), 0
                    )
                    self.assertEqual(
                        conf.max_request_tokens(300_000, page_size=128),
                        300_000 - 5 * (1024 + 128),
                    )
                    # --max-total-tokens clamping stays monotonically <= profiled shared_tokens.
                    capped = conf.calculate_pool_sizes_from_max_tokens(
                        cfg.max_total_num_tokens // 2, page_size=128
                    )
                    self.assertLessEqual(
                        capped.full_max_total_num_tokens, cfg.full_max_total_num_tokens
                    )
                    self.assertEqual(
                        capped.max_total_num_tokens,
                        ((cfg.max_total_num_tokens // 2) // 128) * 128,
                    )
                    with self.assertRaises(ValueError):
                        conf.calculate_pool_sizes_from_max_tokens(128, page_size=128)
                    pool = TpuUnifiedSWAKVPool(
                        size=cfg.full_max_total_num_tokens,
                        page_size=128,
                        dtype=dtype,
                        full=AttnLayerSpec(
                            head_num=2, head_dim=256, layer_ids=range(10)
                        ),
                        swa=AttnLayerSpec(
                            head_num=2, head_dim=256, layer_ids=range(10, 60)
                        ),
                        device="cpu",
                    )
                    self.assertLessEqual(sum(pool.get_kv_size_bytes()), budget)

            # Mismatched H_swa != H_full or D_swa != D_full is rejected
            for bad_kwargs in (
                dict(head_num=4, head_dim=256, swa_head_num=4, swa_head_dim=128),
                dict(head_num=1, head_dim=256, swa_head_num=2, swa_head_dim=256),
                dict(head_num=4, head_dim=128, swa_head_num=1, swa_head_dim=128),
            ):
                with self.assertRaises(ValueError):
                    TpuUnifiedSWAPoolConfigurator(self._make_mock_kvc(**bad_kwargs))

        # Tight HBM under chunk-cap flags (disable_radix_cache + max_running_requests)
        # sizes directly from shared buffers instead of raising RuntimeError.
        with self._cfg_patches(disable_radix_cache=True, max_running_requests=256):
            kvc_tight = self._make_mock_kvc(dtype=torch.bfloat16)
            conf_tight = TpuUnifiedSWAPoolConfigurator(kvc_tight)
            tight_budget = 64 * 128 * conf_tight._full_per_token * 10
            tight_cfg = conf_tight.calculate_pool_sizes(tight_budget, page_size=128)
            self.assertEqual(
                tight_cfg.max_total_num_tokens,
                conf_tight.max_request_tokens(
                    tight_cfg.full_max_total_num_tokens, page_size=128
                ),
            )

    def test_unsupported_configurations_rejected(self):
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.tpu_platform import _resolve_unified_kv_pool

        base = self._make_mock_kvc()
        with self._cfg_patches():
            self.assertTrue(unified_kv_pool_enabled(base))
            for attr, val in (
                ("is_hybrid_swa", False),
                ("use_mla_backend", True),
                ("mambaish_config", object()),
                ("hybrid_gdn_config", object()),
                ("spec_algorithm", SpeculativeAlgorithm.UNO),
                ("spec_algorithm", SpeculativeAlgorithm.FROZEN_KV_MTP),
                ("spec_algorithm", SpeculativeAlgorithm.STANDALONE),
                ("spec_algorithm", SpeculativeAlgorithm.NGRAM),
                ("pp_size", 2),
                ("_tpu_mha_pool_class", lambda: MHATokenToKVPool),
                ("_tpu_mha_pool_class", lambda: object),
                ("pool_page_size", 1),
            ):
                kvc = copy.copy(base)
                setattr(kvc, attr, val)
                with self.assertRaises(ValueError):
                    unified_kv_pool_enabled(kvc)
            kvc_draft = copy.copy(base)
            kvc_draft.is_draft_worker = True
            self.assertFalse(unified_kv_pool_enabled(kvc_draft))
            for supported_spec in (
                SpeculativeAlgorithm.EAGLE,
                SpeculativeAlgorithm.EAGLE3,
                SpeculativeAlgorithm.DFLASH,
                SpeculativeAlgorithm.DSPARK,
            ):
                kvc_spec = copy.copy(base)
                kvc_spec.spec_algorithm = supported_spec
                self.assertTrue(unified_kv_pool_enabled(kvc_spec))
            kvc_asym = self._make_mock_kvc()
            kvc_asym.model_config.has_asymmetric_kv = True
            with self.assertRaises(ValueError):
                unified_kv_pool_enabled(kvc_asym)

            # TpuSeqAlongLaneKVPool is supported
            kvc_along = self._make_mock_kvc(pool_cls=TpuSeqAlongLaneKVPool)
            with patch(
                "sglang_tpu.kv_pool._seq_along_lane_head_dim",
                side_effect=lambda *, head_dim, kv_packing: (
                    ((head_dim + 8 * kv_packing - 1) // (8 * kv_packing))
                    * (8 * kv_packing)
                ),
            ):
                self.assertIsNotNone(TpuUnifiedSWAPoolConfigurator(kvc_along))
            # SGLANG_TPU_KV_PACK_HEADS is rejected
            with patch(
                "sglang_tpu.envs.envs.SGLANG_TPU_KV_PACK_HEADS.get", return_value=True
            ):
                with self.assertRaises(ValueError):
                    unified_kv_pool_enabled(base)

        with self._cfg_patches(
            enable_hierarchical_cache=True, hicache_io_backend="raiden"
        ):
            self.assertTrue(unified_kv_pool_enabled(base))

        for mode in ("prefill", "decode"):
            with self._cfg_patches(
                disaggregation_mode=mode,
                disaggregation_transfer_backend="raiden",
            ):
                self.assertTrue(unified_kv_pool_enabled(base))
            with self._cfg_patches(
                disaggregation_mode=mode,
                disaggregation_transfer_backend="raiden",
                enable_hierarchical_cache=True,
                hicache_io_backend="raiden",
            ):
                self.assertTrue(unified_kv_pool_enabled(base))

        with (
            self._cfg_patches(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="raiden",
            ),
            tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.override(True),
        ):
            with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
                unified_kv_pool_enabled(base)

        for ctx_overrides in (
            dict(pp_size=2),
            dict(dcp_size=2, tp_size=2),
            dict(attn_cp_size=2, tp_size=2),
            dict(disaggregation_mode="prefill"),
            dict(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="mooncake",
            ),
            dict(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="raiden",
                enable_pd_role_switch=True,
            ),
            dict(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="raiden",
                disaggregation_decode_enable_offload_kvcache=True,
            ),
            dict(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="raiden",
                disaggregation_decode_retraction_backup="host_pool",
            ),
            dict(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="raiden",
                disaggregation_decode_host_receive_threshold=0.5,
            ),
            dict(enable_hierarchical_cache=True, hicache_io_backend="kernel"),
            dict(enable_hierarchical_cache=True, hicache_io_backend="torch_tpu"),
            dict(
                enable_hierarchical_cache=True,
                hicache_io_backend="raiden",
                hicache_storage_backend="file",
            ),
            dict(
                enable_hierarchical_cache=True,
                hicache_io_backend="raiden",
                hicache_host_memory_mode="buffer_only",
            ),
            dict(enable_unified_cache_external_linker=True),
        ):
            with self._cfg_patches(**ctx_overrides):
                with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
                    unified_kv_pool_enabled(base)

        with self._cfg_patches(tpu_unified_kv_pool=False):
            self.assertFalse(unified_kv_pool_enabled(base))

        import pickle

        from sglang_tpu.tpu_platform import TpuSRTPlatform

        ok_args = ServerArgs(model_path="dummy", enable_unified_memory=True)
        _resolve_unified_kv_pool(ok_args)
        view = resolving_view(ok_args)
        self.assertFalse(view.enable_unified_memory)
        self.assertTrue(unified_kv_pool_requested(ok_args))
        self.assertTrue(unified_kv_pool_requested(pickle.loads(pickle.dumps(ok_args))))

        off_args = ServerArgs(model_path="dummy", enable_unified_memory=False)
        _resolve_unified_kv_pool(off_args)
        self.assertFalse(unified_kv_pool_requested(off_args))

        platform = TpuSRTPlatform()
        with self._cfg_patches(tpu_unified_kv_pool=False):
            self.assertIsNone(platform.create_memory_pool_configurator(base))
            self.assertIsNone(
                platform.build_swa_kv_pool(
                    base,
                    full_max_total_num_tokens=2048,
                    swa_max_total_num_tokens=2048,
                )
            )
            self.assertIsNone(
                platform.build_swa_kv_pool_allocator(
                    base,
                    sizes=types.SimpleNamespace(full_max_total_num_tokens=2048),
                    token_to_kv_pool=None,
                    req_to_token_pool=None,
                    need_sort=False,
                )
            )

        with self._cfg_patches():
            conf = platform.create_memory_pool_configurator(base)
            self.assertIsInstance(conf, TpuUnifiedSWAPoolConfigurator)
            sizes = conf.calculate_pool_sizes_from_max_tokens(1024, page_size=128)
            pool = platform.build_swa_kv_pool(
                base,
                full_max_total_num_tokens=sizes.full_max_total_num_tokens,
                swa_max_total_num_tokens=sizes.swa_max_total_num_tokens,
            )
            self.assertIsInstance(pool, TpuUnifiedSWAKVPool)
            req_pool = ReqToTokenPool(
                size=4,
                max_context_len=1024,
                device="cpu",
                enable_memory_saver=False,
            )
            alloc = platform.build_swa_kv_pool_allocator(
                base,
                sizes=sizes,
                token_to_kv_pool=pool,
                req_to_token_pool=req_pool,
                need_sort=False,
            )
            self.assertIsInstance(alloc, TpuUnifiedSWATokenToKVPoolAllocator)
            expected_capacity = conf.max_request_tokens(
                sizes.full_max_total_num_tokens, page_size=128
            )
            self.assertEqual(
                alloc.max_request_tokens(base.sliding_window_size),
                expected_capacity,
            )
            from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

            self.assertEqual(
                KVCacheConfigurator.hybrid_swa_token_capacity(
                    base,
                    allocator=alloc,
                    full_capacity=sizes.full_max_total_num_tokens,
                    swa_capacity=sizes.swa_max_total_num_tokens,
                ),
                expected_capacity,
            )

    def test_rpa_backend_layer_page_indices_selects_full_and_swa_groups(self):
        from sglang_tpu.attention_pallas_rpa_base import (
            BaseTpuPallasRpaAttnBackend,
            PallasForwardMetadata,
        )

        backend = object.__new__(BaseTpuPallasRpaAttnBackend)
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
            torch.equal(backend._layer_page_indices(0, is_swa_layer=False), f0)
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(1, is_swa_layer=False), f0)
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(2, is_swa_layer=True), s0)
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(3, is_swa_layer=True), sg[0])
        )
        self.assertTrue(
            torch.equal(backend._layer_page_indices(4, is_swa_layer=True), sg[1])
        )

    def test_mxint8_unified_blocked_sizing_and_pool(self):
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.server_args import ServerArgs
        from sglang_tpu import tpu_platform
        from sglang_tpu.kv_mxint8 import (
            TOKEN_BLOCK,
            blocked_page_rows,
            cache_bytes_per_token,
        )
        from sglang_tpu.tpu_platform import (
            _apply_page_size_default,
            _resolve_unified_kv_pool,
        )

        class _FakeMXINT8Pool(TpuFusedMHATokenToKVPool):
            def __init__(
                self,
                *args,
                kv_cache_dtype_str=None,
                kv_group_size=32,
                kv_layout="merged",
                **kw,
            ):
                self.kv_cache_dtype_str = kv_cache_dtype_str
                self.kv_group_size = kv_group_size
                self.kv_layout = kv_layout
                super().__init__(*args, **kw)

            def kv_bytes_per_token_per_layer(self, **_):
                return cache_bytes_per_token(
                    self.head_num,
                    self.head_dim,
                    self.page_size,
                    group_size=self.kv_group_size,
                    layout=self.kv_layout,
                )

            def _create_buffers(self):
                shape = (
                    self.num_blocks,
                    blocked_page_rows(
                        self.head_num, self.head_dim, self.page_size, self.kv_group_size
                    ),
                    TOKEN_BLOCK,
                )
                self.kv_buffer = [
                    torch.zeros(shape, dtype=torch.int8, device=self.device)
                    for _ in range(self.layer_num)
                ]

            def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
                pages = (indices.cpu() // self.page_size).long()
                return [b[pages].clone() for b in self.kv_buffer]

            def load_cpu_copy(
                self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
            ):
                pages = (indices.cpu() // self.page_size).long()
                for b, src in zip(self.kv_buffer, kv_cache_cpu):
                    b[pages] = src

        for ps, expected_bpt in ((128, 160), (256, 144), (512, 136)):
            kvc = self._make_mock_kvc(
                full_ids=[0, 1],
                swa_ids=[2, 3, 4, 5],
                head_num=1,
                head_dim=64,
                swa_head_num=1,
                swa_head_dim=64,
                dtype=torch.int8,
                pool_cls=_FakeMXINT8Pool,
            )
            kvc.kv_cache_dtype_str = "mxint8"
            kvc.kv_group_size = 32
            kvc.pool_page_size = ps
            kvc.page_size = ps

            with self._cfg_patches():
                cfg = TpuUnifiedSWAPoolConfigurator(kvc)
                self.assertEqual(cfg._full_per_token, expected_bpt)
                avail = 64 * ps * expected_bpt * 2
                sized = cfg.calculate_pool_sizes(avail, ps)
                self.assertEqual(
                    sized.max_total_num_tokens,
                    cfg.max_request_tokens(sized.full_max_total_num_tokens, ps),
                )

            pool = TpuUnifiedSWAKVPool(
                size=sized.full_max_total_num_tokens,
                page_size=ps,
                dtype=torch.int8,
                full=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=[0, 1]),
                swa=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=[2, 3, 4, 5]),
                device="cpu",
                token_to_kv_pool_class=_FakeMXINT8Pool,
                pool_kwargs={"kv_cache_dtype_str": "mxint8", "kv_group_size": 32},
            )
            self.assertEqual(pool.kv_layout, "blocked")
            self.assertEqual(
                sum(pool.get_kv_size_bytes()),
                (sized.full_max_total_num_tokens // ps + 1) * ps * expected_bpt * 2,
            )
            self.assertLessEqual(sum(pool.get_kv_size_bytes()), avail)
            self.assertEqual(
                pool.get_fused_kv_buffer(0).data_ptr(),
                pool.get_fused_kv_buffer(2).data_ptr(),
            )

            # Verify get_cpu_copy / load_cpu_copy works with an OOT pool overriding get_cpu_copy.
            mapping = torch.zeros(
                sized.full_max_total_num_tokens + ps + 1, dtype=torch.int64
            )
            mapping[-1] = -1
            mapping[ps : 2 * ps] = torch.arange(2 * ps, 3 * ps)
            pool.register_mapping(mapping)
            pool.swa_group_pages[:, 2] = torch.tensor([5], dtype=torch.int32)
            pool.get_fused_kv_buffer(0)[1].fill_(7)
            pool.get_fused_kv_buffer(2)[2].fill_(11)
            pool.get_fused_kv_buffer(4)[5].fill_(13)
            saved = pool.get_cpu_copy(torch.tensor([ps]))
            mapping[3 * ps : 4 * ps] = torch.arange(4 * ps, 5 * ps)
            pool.swa_group_pages[:, 4] = torch.tensor([6], dtype=torch.int32)
            pool.load_cpu_copy(saved, torch.tensor([3 * ps]))
            self.assertTrue(torch.all(pool.get_fused_kv_buffer(0)[3] == 7))
            self.assertTrue(torch.all(pool.get_fused_kv_buffer(2)[4] == 11))
            self.assertTrue(torch.all(pool.get_fused_kv_buffer(4)[6] == 13))

        # Invalid blocked page sizes (64, 192, 1024) are rejected in validation.
        for bad_ps in (64, 192, 1024):
            bad_kvc = self._make_mock_kvc(
                full_ids=[0, 1],
                swa_ids=[2, 3],
                head_num=1,
                head_dim=64,
                swa_head_num=1,
                swa_head_dim=64,
                dtype=torch.int8,
                pool_cls=_FakeMXINT8Pool,
            )
            bad_kvc.kv_cache_dtype_str = "mxint8"
            bad_kvc.pool_page_size = bad_ps
            bad_kvc.page_size = bad_ps
            with self._cfg_patches(), self.assertRaises(ValueError):
                unified_kv_pool_enabled(bad_kvc)

        # Platform defaults promote page_size=None to spec.min_page_size=128 before unified resolution.
        with patch.object(
            tpu_platform,
            "_rpa_kernel_spec",
            side_effect=lambda b: (
                types.SimpleNamespace(min_page_size=128)
                if b == "oot_mxint8_rpa"
                else None
            ),
        ):
            mx_args = ServerArgs(
                model_path="dummy",
                enable_unified_memory=True,
                attention_backend="oot_mxint8_rpa",
                kv_cache_dtype="mxint8",
                page_size=None,
            )
            _apply_page_size_default(mx_args)
            _resolve_unified_kv_pool(mx_args)
            mx_view = resolving_view(mx_args)
            self.assertFalse(mx_view.enable_unified_memory)
            self.assertTrue(unified_kv_pool_requested(mx_args))
            self.assertEqual(mx_view.page_size, 128)


class _InMemoryDMAKVTransfer:
    """CPU tensor-copy ``TpuDMAKVTransfer`` double for unified HiCache tests."""

    def __init__(self, device_pool, num_host_pages: int):
        self._noop = NOOP_EVENT
        self.device_pool = device_pool
        self.num_host_pages = num_host_pages
        self.device_buffers = device_pool.page_buffers()
        self.host_buffers = [
            torch.zeros((num_host_pages, *buf.shape[1:]), dtype=buf.dtype)
            for buf in self.device_buffers
        ]
        self.backups: list[tuple[list[int], list[int]]] = []
        self.loads: list[tuple[list[int], list[int]]] = []
        self.destroyed = False

    def backup(self, device_pages: list[int], host_pages: list[int]):
        self.backups.append((list(device_pages), list(host_pages)))
        for dev_buf, host_buf in zip(self.device_buffers, self.host_buffers):
            host_buf[host_pages] = dev_buf[device_pages]
        return self._noop

    def load(self, host_pages: list[int], device_pages: list[int]):
        self.loads.append((list(host_pages), list(device_pages)))
        for dev_buf, host_buf in zip(self.device_buffers, self.host_buffers):
            dev_buf[device_pages] = host_buf[host_pages]
        return self._noop

    def destroy(self) -> None:
        self.destroyed = True


class TestTpuUnifiedDMAHiCache(unittest.TestCase):
    """Tests for PR 4: unified TPU DMA host KV pool, multi-group SWA transfers, and shared host eviction."""

    BACKEND_NAME = "_test_unified_dma"

    def setUp(self):
        register_unified_hicache_strategy()
        self.created_engines: list[_InMemoryDMAKVTransfer] = []
        self._saved_engines = dict(tpu_dma_hicache._ENGINES)

        def _factory(device_pool, num_host_pages: int):
            eng = _InMemoryDMAKVTransfer(device_pool, num_host_pages)
            self.created_engines.append(eng)
            return eng

        tpu_dma_hicache._ENGINES[self.BACKEND_NAME] = _factory

    def tearDown(self):
        tpu_dma_hicache._ENGINES.clear()
        tpu_dma_hicache._ENGINES.update(self._saved_engines)

    def _hicache_ctx(self, **overrides):
        stack = ExitStack()
        budget_bytes = overrides.pop("budget_bytes", 4 * 1024**3)
        defaults = dict(
            attention_backend="pallas_tpu_rpa_v3",
            tp_size=1,
            pp_size=1,
            dcp_size=1,
            attn_cp_size=1,
            enable_hierarchical_cache=True,
            hicache_io_backend=self.BACKEND_NAME,
            hicache_write_policy="write_through",
            hicache_mem_layout="page_first_fused",
            hicache_ratio=2.0,
            hicache_size=0,
            hicache_storage_backend=None,
            hicache_host_memory_mode="cache",
            enable_unified_memory=False,
            page_size=4,
        )
        defaults.update(overrides)
        stack.enter_context(get_context().override_server_args(**defaults))
        stack.enter_context(
            get_parallel().override(
                attn_tp_size=1,
                attn_dcp_size=1,
                dcp_enabled=False,
            )
        )
        platform = TpuSRTPlatform()
        stack.enter_context(patch("sglang.srt.platforms.current_platform", platform))
        stack.enter_context(
            patch.object(
                pool_host_base,
                "available_host_memory_bytes",
                return_value=budget_bytes
                + pool_host_base.HICACHE_HOST_MEMORY_RESERVE_BYTES,
            )
        )
        stack.enter_context(platform.get_hicache_explicit_size_context())
        return stack

    def test_single_dma_engine_budget_and_swa_metrics(self):
        # L_full=2, L_swa=10 -> G=2 shared buffers, M_swa=5 SWA groups.
        pool = TpuUnifiedSWAKVPool(
            size=32,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[0, 1]),
            swa=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=list(range(2, 12))),
            device="cpu",
        )
        req_pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=32,
            page_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
            req_to_token_pool=req_pool,
        )
        params = CacheInitParams(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=alloc,
            page_size=4,
            disable=False,
            sliding_window_size=8,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            eviction_policy="lru",
            tree_core_backend="python",
        )
        self.assertEqual(
            sizing._estimate_hicache_bytes(params, None), sum(pool.get_kv_size_bytes())
        )
        bytes_per_page = sum(buf[0].nbytes for buf in pool.kv_pool.page_buffers())
        expected_host_pages = int(32 * 2.0) // 4 + 1  # 17 physical host pages
        # Tight RAM budget: fits 17 pages once, would fail if deducted twice!
        tight_budget = int(expected_host_pages * bytes_per_page * 1.2)

        with self._hicache_ctx(budget_bytes=tight_budget, hicache_ratio=2.0):
            cache = UnifiedRadixCache(params=params)
            server_args = ServerArgs(
                model_path="dummy",
                enable_hierarchical_cache=True,
                hicache_io_backend=self.BACKEND_NAME,
                page_size=4,
            )
            cache.init_hicache(server_args, params)
            group = cache.host_pool_group

        self.assertEqual(len(self.created_engines), 1)
        self.assertEqual(self.created_engines[0].num_host_pages, expected_host_pages)
        kv_host = group.get_pool(PoolName.KV)
        swa_host = group.get_pool(PoolName.SWA)
        self.assertIsInstance(kv_host, TpuDMAUnifiedHostKVPool)
        self.assertIsInstance(swa_host, TpuDMAUnifiedHostKVPool)
        self.assertTrue(uses_shared_host_layout(kv_host))
        self.assertTrue(uses_shared_host_layout(swa_host))
        self.assertIs(
            kv_host.shared_allocation_domain, swa_host.shared_allocation_domain
        )
        self.assertIsNotNone(group.entry_map[PoolName.KV].host_evict_fn)
        self.assertEqual(swa_host.size_per_token, 5 * kv_host.size_per_token)
        self.assertEqual(kv_host.size, expected_host_pages * 4)
        self.assertEqual(swa_host.size, (expected_host_pages // 5) * 4)

        # Allocate 8 Full tokens (2 phys pages) and 4 SWA tokens (1 logical = 5 phys pages).
        full_idx = kv_host.alloc(8)
        swa_idx = swa_host.alloc(4)
        self.assertIsNotNone(full_idx)
        self.assertIsNotNone(swa_idx)
        # SWA metrics report only SWA's live logical tokens (4), unaffected by Full's 8 tokens.
        self.assertEqual(used_tokens(swa_host), 4)
        self.assertEqual(kv_host.available_size(), (expected_host_pages - 7) * 4)
        self.assertEqual(
            swa_host.available_size(), ((expected_host_pages - 7) // 5) * 4
        )

        group.clear()
        self.assertEqual(used_tokens(swa_host), 0)
        self.assertEqual(
            list(kv_host.shared_allocation_domain.free_pages),
            list(range(expected_host_pages)),
        )
        kv_host.destroy()
        self.assertFalse(self.created_engines[0].destroyed)
        swa_host.destroy()
        self.assertTrue(self.created_engines[0].destroyed)

    def test_multi_group_d2h_h2d_roundtrip_node_split_and_mxint8(self):
        class _FakeMXINT8Pool(TpuFusedMHATokenToKVPool):
            def __init__(
                self,
                *args,
                kv_cache_dtype_str=None,
                kv_group_size=32,
                kv_layout="merged",
                **kw,
            ):
                self.kv_cache_dtype_str = kv_cache_dtype_str
                self.kv_group_size = kv_group_size
                self.kv_layout = kv_layout
                super().__init__(*args, **kw)

            def kv_bytes_per_token_per_layer(self, **_):
                return cache_bytes_per_token(
                    self.head_num,
                    self.head_dim,
                    self.page_size,
                    group_size=self.kv_group_size,
                    layout=self.kv_layout,
                )

            def _create_buffers(self):
                shape = (
                    self.num_blocks,
                    blocked_page_rows(
                        self.head_num, self.head_dim, self.page_size, self.kv_group_size
                    ),
                    TOKEN_BLOCK,
                )
                self.kv_buffer = [
                    torch.zeros(shape, dtype=torch.int8, device=self.device)
                    for _ in range(self.layer_num)
                ]

        cases = (
            ("bf16", 4, torch.bfloat16, TpuFusedMHATokenToKVPool, None),
            (
                "mxint8",
                128,
                torch.int8,
                _FakeMXINT8Pool,
                {"kv_cache_dtype_str": "mxint8", "kv_group_size": 32},
            ),
        )
        for label, ps, dtype, pool_cls, pool_kwargs in cases:
            with self.subTest(case=label):
                # L_full=2, L_swa=6 -> G=2 shared buffers, M_swa=3 groups.
                pool = TpuUnifiedSWAKVPool(
                    size=16 * ps,
                    page_size=ps,
                    dtype=dtype,
                    full=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=[0, 1]),
                    swa=AttnLayerSpec(
                        head_num=1, head_dim=64, layer_ids=[2, 3, 4, 5, 6, 7]
                    ),
                    device="cpu",
                    token_to_kv_pool_class=pool_cls,
                    pool_kwargs=pool_kwargs,
                )
                req_pool = ReqToTokenPool(
                    size=4,
                    max_context_len=16 * ps,
                    device="cpu",
                    enable_memory_saver=False,
                )
                alloc = TpuUnifiedSWATokenToKVPoolAllocator(
                    size=16 * ps,
                    page_size=ps,
                    dtype=dtype,
                    device="cpu",
                    kvcache=pool,
                    need_sort=False,
                    req_to_token_pool=req_pool,
                )
                with self._hicache_ctx(page_size=ps):
                    kv_host, swa_host = TpuDMAUnifiedHostKVPool.build_pair(
                        pool,
                        io_backend=self.BACKEND_NAME,
                        host_to_device_ratio=2.0,
                        host_size=0,
                        page_size=ps,
                    )

                # Allocate 2 joint pages (2 Full + 2*3 SWA = 8 physical pages).
                dev_full = alloc.alloc_extend(
                    prefix_lens=torch.tensor([0]),
                    prefix_lens_cpu=torch.tensor([0]),
                    seq_lens=torch.tensor([2 * ps]),
                    seq_lens_cpu=torch.tensor([2 * ps]),
                    last_loc=torch.tensor([-1]),
                    extend_num_tokens=2 * ps,
                )
                dev_swa = pool.translate_loc_from_full_to_swa(dev_full).to(torch.int64)
                full_pages = (dev_full[::ps] // ps).tolist()
                swa_prim_pages = (dev_swa[::ps] // ps).tolist()

                # Stamp distinct values on every Full buffer and all 3 SWA groups.
                expected_full = []
                for b in range(2):
                    val = 10 + b
                    pool.kv_pool.kv_buffer[b][full_pages[1]].fill_(val)
                    expected_full.append(
                        pool.kv_pool.kv_buffer[b][full_pages[1]].clone()
                    )
                expected_swa = []
                for g in range(3):
                    phys_p = (
                        swa_prim_pages[1]
                        if g == 0
                        else int(pool.swa_group_pages[g - 1, swa_prim_pages[1]].item())
                    )
                    group_expected = []
                    for b in range(2):
                        val = 30 + g * 10 + b
                        pool.kv_pool.kv_buffer[b][phys_p].fill_(val)
                        group_expected.append(pool.kv_pool.kv_buffer[b][phys_p].clone())
                    expected_swa.append(group_expected)

                host_full, host_swa = kv_host.shared_allocation_domain.alloc_many(
                    [("full", 2 * ps), ("swa", 2 * ps)]
                )
                kv_host.backup_from_device_all_layer(
                    pool, host_full, dev_full, self.BACKEND_NAME
                )
                swa_host.backup_from_device_all_layer(
                    pool, host_swa, dev_swa, self.BACKEND_NAME
                )

                # Free & zero device buffers, scramble free page order by holding a dummy page,
                # then reload ONLY the 2nd logical page (node-split / partial window slice).
                alloc.free(dev_full)
                for buf in pool.kv_pool.kv_buffer:
                    buf.zero_()
                dummy_full = alloc.full_attn_allocator.alloc(ps)
                new_dev_full = alloc.full_attn_allocator.alloc(ps)
                new_dev_swa = alloc.swa_attn_allocator.alloc(ps)
                self.assertNotEqual(int(new_dev_swa[0] // ps), swa_prim_pages[1])

                kv_host.load_all_layer(pool, host_full[ps:], new_dev_full)
                swa_host.load_all_layer(pool, host_swa[ps:], new_dev_swa)

                new_fp = int(new_dev_full[0] // ps)
                new_sp = int(new_dev_swa[0] // ps)
                for b in range(2):
                    self.assertTrue(
                        torch.equal(pool.kv_pool.kv_buffer[b][new_fp], expected_full[b])
                    )
                for g in range(3):
                    phys_p = (
                        new_sp
                        if g == 0
                        else int(pool.swa_group_pages[g - 1, new_sp].item())
                    )
                    for b in range(2):
                        self.assertTrue(
                            torch.equal(
                                pool.kv_pool.kv_buffer[b][phys_p], expected_swa[g][b]
                            )
                        )

                # Freeing split halves independently returns all primary + companion host pages.
                kv_host.free(host_full[:ps])
                kv_host.free(host_full[ps:])
                swa_host.free(host_swa[:ps])
                swa_host.free(host_swa[ps:])
                domain = kv_host.shared_allocation_domain
                self.assertEqual(
                    domain.free_bytes(), domain.page_num * domain.bytes_per_page
                )
                alloc.full_attn_allocator.free(dummy_full)
                alloc.full_attn_allocator.free(new_dev_full)
                alloc.swa_attn_allocator.free(new_dev_swa)

    def test_unified_radix_cache_end_to_end_offload_and_cross_component_host_eviction(
        self,
    ):
        # L_full=2, L_swa=6 -> G=2 shared buffers, M_swa=3, page_size=4, window=4.
        # Device size=128 -> 32 usable physical pages.
        # Host hicache_ratio=0.22 -> int(128 * 0.22) // 4 + 1 = 28 // 4 + 1 = 8 physical host pages.
        ps = 4
        pool = TpuUnifiedSWAKVPool(
            size=128,
            page_size=ps,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[0, 1]),
            swa=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[2, 3, 4, 5, 6, 7]),
            device="cpu",
        )
        req_pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=128,
            page_size=ps,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
            req_to_token_pool=req_pool,
        )
        params = CacheInitParams(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=alloc,
            page_size=ps,
            disable=False,
            sliding_window_size=ps,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            eviction_policy="lru",
            tree_core_backend="python",
        )
        with self._hicache_ctx(
            page_size=ps,
            hicache_ratio=0.22,
            enable_unified_memory=True,
        ):
            cache = UnifiedRadixCache(params=params)
            server_args = ServerArgs(
                model_path="dummy",
                enable_hierarchical_cache=True,
                hicache_io_backend=self.BACKEND_NAME,
                page_size=ps,
            )
            cache.init_hicache(server_args, params)
            ctrl = cache.cache_controller
            self.assertIsInstance(ctrl, TpuDMAHybridCacheController)
            self.assertIsInstance(pool.layer_transfer_counter, BatchDoneCounter)
            self.assertIsNone(pool.kv_pool.layer_transfer_counter)
            # Must NOT wire CUDA virtual-ID bind_swa_for_loaded_rows even with enable_unified_memory=True.
            self.assertIsNone(
                cache.host_pool_group.entry_map[
                    PoolName.SWA
                ].device_indices_from_anchor_fn
            )
            domain = cache.host_pool_group.get_pool(
                PoolName.KV
            ).shared_allocation_domain
            self.assertEqual(domain.page_num, 8)

            # 1) Insert 4-token prompt A (1 Full + 1 SWA = 4 phys pages on device & host).
            out_a = alloc.alloc_extend(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([4]),
                seq_lens_cpu=torch.tensor([4]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=4,
            )
            fp_a = int(out_a[0] // ps)
            sp_a = int(pool.translate_loc_from_full_to_swa(out_a)[0].item() // ps)
            pool.kv_pool.kv_buffer[0][fp_a].fill_(11.0)
            pool.kv_pool.kv_buffer[1][fp_a].fill_(12.0)
            for g in range(3):
                phys_s = (
                    sp_a if g == 0 else int(pool.swa_group_pages[g - 1, sp_a].item())
                )
                pool.kv_pool.kv_buffer[0][phys_s].fill_(21.0 + g)
                pool.kv_pool.kv_buffer[1][phys_s].fill_(31.0 + g)

            key_a = RadixKey(array("q", [10, 11, 12, 13]))
            cache.insert(InsertParams(key=key_a, value=out_a))
            ctrl.start_writing()
            cache.writing_check()
            # Host now uses 1 (Full) + 3 (SWA) = 4 of 8 physical pages (4 free).
            self.assertEqual(len(domain.free_pages), 4)

            # Evict A completely from device and zero device buffers.
            cache.evict_for_alloc(EvictParams(num_tokens=4, swa_num_tokens=4))
            self.assertEqual(alloc._free_page_count(), 32)
            for buf in pool.kv_pool.kv_buffer:
                buf.zero_()

            # Match A -> hits on host -> H2D reload restores Full AND all 3 SWA groups!
            match_a = cache.match_prefix(MatchPrefixParams(key=key_a))
            self.assertEqual(match_a.host_hit_length, 4)
            self.assertEqual(match_a.swa_host_hit_length, 4)
            self.assertTrue(cache.load_back(match_a.best_match_node))
            ctrl.start_loading()
            cache.loading_check()
            loaded_full = cache.match_prefix(
                MatchPrefixParams(key=key_a)
            ).device_indices
            self.assertEqual(len(loaded_full), 4)

            reloaded_fp = int(loaded_full[0] // ps)
            reloaded_sp = int(
                pool.translate_loc_from_full_to_swa(loaded_full)[0].item() // ps
            )
            self.assertGreater(reloaded_sp, 0)
            self.assertTrue(torch.all(pool.kv_pool.kv_buffer[0][reloaded_fp] == 11.0))
            self.assertTrue(torch.all(pool.kv_pool.kv_buffer[1][reloaded_fp] == 12.0))
            for g in range(3):
                phys_s = (
                    reloaded_sp
                    if g == 0
                    else int(pool.swa_group_pages[g - 1, reloaded_sp].item())
                )
                self.assertTrue(
                    torch.all(pool.kv_pool.kv_buffer[0][phys_s] == 21.0 + g)
                )
                self.assertTrue(
                    torch.all(pool.kv_pool.kv_buffer[1][phys_s] == 31.0 + g)
                )

            # 2) Cross-component host eviction (Direction 1: Full-only backup evicts interior SWA on host
            #    when Full is host-locked!).
            # Extend A ([10..14]) with child E ([10..18], Full+SWA): A uses 4 host pages and E uses 4 host
            # pages -> 8/8 host pages used (0 free). Locking E on host locks FULL on both A & E, while SWA
            # only locks within window=4 (E), leaving A's interior SWA (3 phys pages) evictable on host!
            out_e = alloc.alloc_extend(
                prefix_lens=torch.tensor([4]),
                prefix_lens_cpu=torch.tensor([4]),
                seq_lens=torch.tensor([8]),
                seq_lens_cpu=torch.tensor([8]),
                last_loc=loaded_full[-1:],
                extend_num_tokens=4,
            )
            key_e = RadixKey(array("q", [10, 11, 12, 13, 14, 15, 16, 17]))
            cache.insert(
                InsertParams(
                    key=key_e,
                    value=torch.cat([loaded_full, out_e]),
                    prev_prefix_len=4,
                )
            )
            ctrl.start_writing()
            cache.writing_check()
            self.assertEqual(len(domain.free_pages), 0)
            self.assertEqual(
                cache.host_pool_group.get_pool(PoolName.SWA).live_tokens(), 8
            )
            match_e = cache.match_prefix(MatchPrefixParams(key=key_e))
            self.assertEqual(len(match_e.device_indices), 8)
            host_lock_e = cache.inc_host_lock_ref(
                match_e.last_host_node
            ).to_dec_params()
            cache.evict_for_alloc(EvictParams(num_tokens=8, swa_num_tokens=8))

            # Insert 8-token Full-only prompt B (needs 2 Full host pages, 0 free, Full host is locked ->
            # candidates[0]=KV evicts 0, candidates[1]=SWA tombstones A's 3 SWA host pages -> 3-2=1 free!).
            out_b = alloc.alloc_extend(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([8]),
                seq_lens_cpu=torch.tensor([8]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=8,
            )
            alloc.free_swa(out_b)
            key_b = RadixKey(array("q", range(20, 28)))
            cache.insert(
                InsertParams(
                    key=key_b,
                    value=out_b,
                    component_evicted_seqlens={ComponentType.SWA: 8},
                )
            )
            ctrl.start_writing()
            cache.writing_check()
            self.assertEqual(len(domain.free_pages), 1)
            self.assertEqual(
                cache.host_pool_group.get_pool(PoolName.SWA).live_tokens(), 4
            )
            self.assertEqual(
                cache.host_pool_group.get_pool(PoolName.KV).live_tokens(), 16
            )
            cache.dec_host_lock_ref(match_e.last_host_node, host_lock_e)
            cache.evict_for_alloc(EvictParams(num_tokens=8))

            # Replace host contents with 32-token Full-only prompt C (8 Full host pages) -> 0 free host
            # pages and ZERO SWA pages on host (all 8 physical host pages are Full-only!).
            out_c = alloc.alloc_extend(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([32]),
                seq_lens_cpu=torch.tensor([32]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=32,
            )
            alloc.free_swa(out_c)
            key_c = RadixKey(array("q", range(100, 132)))
            cache.insert(
                InsertParams(
                    key=key_c,
                    value=out_c,
                    component_evicted_seqlens={ComponentType.SWA: 32},
                )
            )
            ctrl.start_writing()
            cache.writing_check()
            self.assertEqual(len(domain.free_pages), 0)
            self.assertEqual(
                cache.host_pool_group.get_pool(PoolName.SWA).live_tokens(), 0
            )
            self.assertEqual(
                cache.host_pool_group.get_pool(PoolName.KV).live_tokens(), 32
            )

            # 3) Cross-component host eviction (Direction 2: Full+SWA backup when host is 100% Full!).
            # Evict C from device so host holds 8 Full-only pages and 0 SWA pages; insert 4-token
            # prompt D (needs 1 Full + 1 SWA = 4 phys host pages). SWA has 0 host pages to evict, so
            # _alloc_shared_host_requests_with_reclaim MUST call host_full_evict_fn to evict Full!
            cache.evict_for_alloc(EvictParams(num_tokens=32))
            out_d = alloc.alloc_extend(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([4]),
                seq_lens_cpu=torch.tensor([4]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=4,
            )
            key_d = RadixKey(array("q", [40, 41, 42, 43]))
            cache.insert(InsertParams(key=key_d, value=out_d))
            ctrl.start_writing()
            cache.writing_check()
            self.assertEqual(
                cache.host_pool_group.get_pool(PoolName.SWA).live_tokens(), 4
            )

    def test_hooks_register_installs_unified_hicache_strategy_on_tpu_only(self):
        saved_strategies = list(hybrid_pool_assembler._STRATEGIES)
        saved_sizeable = sizing._SIZEABLE_POOLS
        try:
            hybrid_pool_assembler._STRATEGIES[:] = [
                s
                for s in hybrid_pool_assembler._STRATEGIES
                if s is not _TPU_UNIFIED_SWA_STRATEGY
            ]
            sizing._SIZEABLE_POOLS = tuple(
                p for p in sizing._SIZEABLE_POOLS if p is not TpuUnifiedSWAKVPool
            )
            HookRegistry.reset()
            self.addCleanup(HookRegistry.reset)

            non_tpu = types.SimpleNamespace(is_tpu=lambda: False)
            with (
                patch.object(hooks, "HOOKS", [non_tpu]),
                patch.object(hooks, "_registered", False),
                patch("sglang.srt.platforms.current_platform", non_tpu),
            ):
                hooks.register()
            self.assertNotIn(
                _TPU_UNIFIED_SWA_STRATEGY, hybrid_pool_assembler._STRATEGIES
            )
            self.assertNotIn(TpuUnifiedSWAKVPool, sizing._SIZEABLE_POOLS)

            tpu = types.SimpleNamespace(is_tpu=lambda: True)
            noop_hook = (
                "sglang.srt.utils.common.get_device_name",
                lambda r, *a, **k: r,
                hooks.HookType.AFTER,
            )
            with (
                patch.object(hooks, "HOOKS", [noop_hook]),
                patch.object(hooks, "_registered", False),
                patch("sglang.srt.platforms.current_platform", tpu),
                patch(
                    "sglang_tpu.models.muse_spark_hf_processor.register_muse_spark_processor"
                ),
            ):
                hooks.register()
                hooks.register()
            self.assertEqual(
                hybrid_pool_assembler._STRATEGIES.count(_TPU_UNIFIED_SWA_STRATEGY), 1
            )
            self.assertIn(TpuUnifiedSWAKVPool, sizing._SIZEABLE_POOLS)
        finally:
            hybrid_pool_assembler._STRATEGIES[:] = saved_strategies
            sizing._SIZEABLE_POOLS = saved_sizeable

    def test_pd_decode_prealloc_buf_infos_and_expand_swa_pages(self):
        pool = TpuUnifiedSWAKVPool(
            size=128,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=[0, 3]),
            swa=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=[1, 2, 4, 5, 6]),
            device="cpu",
        )
        req_pool = ReqToTokenPool(
            size=4, max_context_len=128, device="cpu", enable_memory_saver=False
        )
        alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=128,
            page_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
            req_to_token_pool=req_pool,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool = pool
        queue.token_to_kv_pool_allocator = alloc
        self.assertTrue(queue._uses_swa_tail_prealloc())

        kv_args = KVArgs()
        (
            kv_args.kv_data_ptrs,
            kv_args.kv_data_lens,
            kv_args.kv_item_lens,
        ) = get_kv_transfer_buf_infos(pool)
        setup_state_kv_args(kv_args, pool)
        self.assertEqual(
            (kv_args.kv_data_ptrs, kv_args.kv_data_lens, kv_args.kv_item_lens),
            pool.kv_pool.get_contiguous_buf_infos(),
        )
        self.assertEqual(kv_args.state_types, [StateType.SWA])
        self.assertEqual(kv_args.state_item_lens, [[]])

        self.assertEqual(pool.expand_swa_pages([]), [])
        swa_toks = alloc.swa_attn_allocator.alloc(8)
        swa_pages = (swa_toks[::4] // 4).tolist()
        self.assertEqual(
            pool.expand_swa_pages(swa_pages),
            [
                swa_pages[0],
                int(pool.swa_group_pages[0, swa_pages[0]]),
                int(pool.swa_group_pages[1, swa_pages[0]]),
                swa_pages[1],
                int(pool.swa_group_pages[0, swa_pages[1]]),
                int(pool.swa_group_pages[1, swa_pages[1]]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
