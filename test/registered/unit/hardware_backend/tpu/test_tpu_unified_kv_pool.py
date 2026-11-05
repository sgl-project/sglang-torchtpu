# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import unittest
from array import array
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.mem_cache.allocator import swa
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
    UnifiedPageLayout,
    num_swa_groups,
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


if __name__ == "__main__":
    unittest.main()
