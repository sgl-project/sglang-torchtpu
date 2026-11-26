"""Host-index reload contracts, including allocation ownership and fallback.

Meta-device allocations make an accidental device-to-host read fail while
exercising the real allocator and controller without TPU hardware.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import kv_pool_host
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.kv_free_fence import KVFreeFence
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _allocator(device="meta", size=32, page_size=4):
    return TpuPagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=device,
        kvcache=None,
        need_sort=False,
    )


def _controller(allocator, cls=TpuDMAHybridCacheController):
    controller = object.__new__(cls)
    controller.mem_pool_device_allocator = allocator
    controller.device = allocator.device
    controller.kv_storage = None
    controller.load_queue = []
    return controller


def _hybrid_controller():
    # CPU buffers with meta allocation results make either index readback fail.
    pool = TpuUnifiedSWAKVPool(
        size=128,
        page_size=4,
        dtype=torch.bfloat16,
        full=AttnLayerSpec(1, 128, [0, 3]),
        swa=AttnLayerSpec(1, 128, [1, 2, 4, 5, 6]),
        device="cpu",
    )
    with (
        patch("sglang.srt.mem_cache.allocator.swa._is_tpu", True),
        patch(
            "sglang.srt.mem_cache.allocator.swa.current_platform",
            SimpleNamespace(
                get_paged_allocator_cls=lambda: TpuPagedTokenToKVPoolAllocator
            ),
        ),
    ):
        allocator = TpuUnifiedSWATokenToKVPoolAllocator(
            size=128,
            page_size=4,
            dtype=torch.bfloat16,
            device="meta",
            kvcache=pool,
            need_sort=False,
        )
    controller = _controller(allocator)
    params = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        mtp_draft_device_pools=None,
        page_size=4,
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )
    memory = SimpleNamespace(
        hicache_io_backend="raiden",
        hicache_ratio=2.0,
        hicache_size=0,
        hicache_write_policy="write_through",
        hicache_host_memory_mode="cache",
    )
    with (
        patch.object(kv_pool_host, "get_memory", return_value=memory),
        patch.object(
            kv_pool_host.TpuDMAUnifiedHostKVPool,
            "build_pair",
            return_value=(Mock(), Mock()),
        ),
        patch.object(
            HybridCacheController, "create_for_backend", return_value=controller
        ),
    ):
        result = kv_pool_host._TPU_UNIFIED_SWA_STRATEGY.build(
            cache=SimpleNamespace(
                token_to_kv_pool_allocator=allocator, evict=Mock(), evict_host=Mock()
            ),
            kvcache=pool,
            params=params,
            server_args=None,
            load_cache_event=None,
        )
    controller.mem_pool_host = result.host_pool_group
    controller.mem_pool_device = pool
    # Skip storage workers but exercise the TPU constructor's callback wiring.
    with patch.object(HybridCacheController, "__init__", return_value=None):
        controller.__init__()
    return controller


class TestHiCacheHostIndices(CustomTestCase):
    def test_default_still_returns_device_indices(self):
        self.assertEqual(_allocator().alloc(4).device.type, "meta")

    def test_host_values_ownership_oom_and_reuse(self):
        for page_size in (4, 32):
            with self.subTest(page_size=page_size):
                allocator = _allocator(size=8 * page_size, page_size=page_size)
                indices = allocator.alloc(8 * page_size, return_host=True)
                torch.testing.assert_close(
                    indices, torch.arange(page_size, 9 * page_size, dtype=torch.int32)
                )
                self.assertEqual(allocator.available_size(), 0)
                self.assertIsNone(allocator.alloc(page_size, return_host=True))
                allocator.free(indices)
                self.assertEqual(allocator.available_size(), allocator.size)
                reused = allocator.alloc(allocator.size, return_host=True)
                torch.testing.assert_close(indices, reused)
                self.assertEqual(torch.unique(reused).numel(), allocator.size)

    def test_load_queues_host_indices_without_upload(self):
        # A meta -> CPU readback would fail, so this checks placement through
        # the real load method as well as the returned values and queue entry.
        controller = _controller(_allocator())
        indices = controller.load(torch.arange(8), node_id=73)
        torch.testing.assert_close(indices, torch.arange(4, 12, dtype=torch.int32))
        self.assertIs(controller.load_queue[0].device_indices, indices)
        self.assertEqual(controller.load_queue[0].node_ids, [73])
        self.assertIsNone(controller.load(torch.arange(32)))
        self.assertEqual(len(controller.load_queue), 1)

    def test_empty_load_retains_dtype(self):
        indices = _controller(_allocator()).load(torch.empty(0, dtype=torch.int64))
        self.assertEqual(
            (indices.numel(), indices.dtype, indices.device.type),
            (0, torch.int64, "cpu"),
        )

    def test_generic_hook_keeps_default_device(self):
        allocator = _allocator()
        indices = HybridCacheController._alloc_load_kv_indices(
            _controller(allocator), allocator, 4, None
        )
        self.assertEqual(indices.device.type, "meta")

    def test_generic_load_accepts_allocator_without_host_option(self):
        class DeviceAllocator:
            device = "meta"

            def alloc(self, need_size):
                return torch.empty(need_size, dtype=torch.int32, device=self.device)

        controller = _controller(DeviceAllocator(), HybridCacheController)
        indices = controller.load(torch.arange(4), node_id=19)
        self.assertEqual(indices.device.type, "meta")
        self.assertIs(controller.load_queue[0].device_indices, indices)
        self.assertEqual(controller.load_queue[0].node_ids, [19])

    def test_generic_empty_load_does_not_call_allocator(self):
        allocator = SimpleNamespace(device="meta", alloc=Mock())
        controller = _controller(allocator, HybridCacheController)
        indices = controller.load(torch.empty(0, dtype=torch.int64))
        allocator.alloc.assert_not_called()
        self.assertEqual(
            (indices.numel(), indices.dtype, indices.device.type),
            (0, torch.int64, "meta"),
        )

    def test_caller_owned_destinations_and_anchor_are_preserved(self):
        controller = _controller(_allocator())
        controller._alloc_load_kv_indices = Mock(side_effect=AssertionError)
        controller.mem_pool_device_allocator.free = Mock(side_effect=AssertionError)
        owned = torch.arange(4, 8, dtype=torch.int32)
        anchor = torch.arange(12, 16, dtype=torch.int32)
        host = torch.arange(4)
        extra = [PoolTransfer(name=PoolName.SWA)]
        controller._resolve_device_transfers = Mock(return_value=None)
        self.assertIsNone(
            controller.load(
                host, extra_pools=extra, device_indices=owned, anchor_indices=anchor
            )
        )
        self.assertFalse(controller.load_queue)
        controller._resolve_device_transfers.assert_called_once_with(
            extra, kv_device_indices=anchor, kv_host_indices=host
        )
        controller._resolve_device_transfers.return_value = extra
        self.assertIs(
            controller.load(
                host, extra_pools=extra, device_indices=owned, anchor_indices=anchor
            ),
            owned,
        )
        self.assertIs(controller.load_queue[0].device_indices, owned)

    def test_storage_owner_allocation_and_rollback_bypass_plain_hook(self):
        controller = _controller(_allocator())
        controller._alloc_load_kv_indices = Mock(side_effect=AssertionError)
        owned = torch.arange(4, 8, dtype=torch.int32)
        failed = torch.arange(8, 12, dtype=torch.int32)
        take, give_back = Mock(side_effect=[owned, failed, None]), Mock()
        controller.kv_storage = SimpleNamespace(
            owner_load_alloc=Mock(return_value=(take, give_back))
        )
        self.assertIs(controller.load(torch.arange(4)), owned)
        take.assert_called_once_with(4)
        give_back.assert_not_called()
        controller._resolve_device_transfers = Mock(return_value=None)
        self.assertIsNone(
            controller.load(
                torch.arange(4), extra_pools=[PoolTransfer(name=PoolName.SWA)]
            )
        )
        give_back.assert_called_once_with(failed)
        self.assertIsNone(controller.load(torch.arange(4)))
        give_back.assert_called_once_with(failed)
        self.assertEqual(len(controller.load_queue), 1)

    def test_page_interleave_keeps_cyclic_placement(self):
        allocator = object.__new__(PageInterleavePoolAllocator)
        allocator.device, allocator.page_size = "meta", 4
        expected = torch.empty(8, dtype=torch.int32, device="meta")
        allocator.alloc_cyclic = Mock(return_value=expected)
        controller = _controller(allocator, HybridCacheController)
        controller._alloc_load_kv_indices = Mock(side_effect=AssertionError)
        self.assertIs(controller.load(torch.arange(20, 28)), expected)
        allocator.alloc_cyclic.assert_called_once_with(start_class=5, need_size=8)

    def test_wrapped_full_allocator_and_derived_sidecar_stay_on_host(self):
        allocator = _allocator()
        controller = _controller(allocator)
        extra = [PoolTransfer(name=PoolName.DRAFT_SWA, indices_from_pool=PoolName.KV)]
        self.assertEqual(
            controller._alloc_load_kv_indices(allocator, 4, extra).device.type,
            "cpu",
        )
        controller.mem_pool_device_allocator = SimpleNamespace(
            full_attn_allocator=allocator
        )
        self.assertEqual(
            controller._alloc_load_kv_indices(allocator, 4, extra).device.type,
            "cpu",
        )

    def test_unified_full_swa_and_derived_indices_stay_on_host(self):
        controller = _hybrid_controller()
        allocator = controller.mem_pool_device_allocator
        swa = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(4))
        draft = PoolTransfer(name=PoolName.DRAFT_SWA, indices_from_pool=PoolName.KV)
        full = controller.load(torch.arange(8), extra_pools=[swa, draft])
        self.assertEqual(full.device.type, "cpu")
        self.assertEqual(swa.device_indices.device.type, "cpu")
        self.assertIs(draft.device_indices, full)
        self.assertIs(controller.load_queue[0].pool_transfers[0], swa)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 8)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 4)
        # Five SWA layers over two shared buffers require three physical groups.
        self.assertEqual(allocator.full_available_size(), 128 - 8 - 3 * 4)
        allocator.swa_attn_allocator.free(swa.device_indices)
        allocator.full_attn_allocator.free(full)
        self.assertEqual(allocator.full_available_size(), 128)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)

    def test_unified_swa_failure_rolls_back_full_allocation(self):
        for missing_source in (False, True):
            with self.subTest(missing_source=missing_source):
                controller = _hybrid_controller()
                allocator = controller.mem_pool_device_allocator
                # Either SWA cannot fit, or a later sidecar cannot resolve its
                # source after both the Full and SWA allocations succeeded.
                swa = PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.arange(4 if missing_source else 44),
                )
                extra = [swa]
                if missing_source:
                    extra.append(
                        PoolTransfer(
                            name=PoolName.DRAFT_SWA,
                            indices_from_pool=PoolName.MAMBA,
                        )
                    )
                self.assertIsNone(controller.load(torch.arange(8), extra_pools=extra))
                self.assertIsNone(swa.device_indices)
                self.assertEqual(allocator.full_available_size(), 128)
                self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
                self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)
                self.assertFalse(controller.load_queue)

    def test_unified_default_allocation_keeps_device_contract(self):
        allocator = _hybrid_controller().mem_pool_device_allocator
        self.assertEqual(allocator.full_attn_allocator.alloc(4).device.type, "meta")
        self.assertEqual(allocator.swa_attn_allocator.alloc(4).device.type, "meta")

    def test_split_swa_reload_and_custom_callback_fallback(self):
        class LegacyAllocator(TpuPagedTokenToKVPoolAllocator):
            def alloc(self, need_size):
                return super().alloc(need_size)

        for kind in ("plain", "custom", "legacy"):
            with self.subTest(kind=kind):
                full, swa = _allocator(), _allocator()
                if kind == "legacy":
                    swa = LegacyAllocator(
                        size=32,
                        page_size=4,
                        dtype=torch.bfloat16,
                        device="meta",
                        kvcache=None,
                        need_sort=False,
                    )
                allocator = SimpleNamespace(
                    device="meta",
                    full_attn_allocator=full,
                    swa_attn_allocator=swa,
                    kv_free_fence=None,
                )
                controller = _controller(allocator)
                controller.mem_pool_device = Mock()
                allocate = (
                    Mock(side_effect=lambda n: swa.alloc(n))
                    if kind == "custom"
                    else swa.alloc
                )
                entry = PoolEntry(
                    PoolName.SWA,
                    Mock(),
                    Mock(),
                    lambda x: x,
                    device_alloc_fn=allocate,
                    device_free_fn=swa.free,
                )
                controller.mem_pool_host = HostPoolGroup(
                    [
                        PoolEntry(
                            PoolName.KV,
                            Mock(),
                            Mock(),
                            lambda x: x,
                            is_primary_index_anchor=True,
                        ),
                        entry,
                    ]
                )
                with patch.object(HybridCacheController, "__init__", return_value=None):
                    controller.__init__()
                if kind != "plain":
                    self.assertIs(entry.device_alloc_fn, allocate)
                    self.assertEqual(entry.device_alloc_fn(4).device.type, "meta")
                else:
                    transfer = PoolTransfer(
                        name=PoolName.SWA, host_indices=torch.arange(4)
                    )
                    indices = controller.load(torch.arange(8), extra_pools=[transfer])
                    self.assertEqual(indices.device.type, "cpu")
                    self.assertEqual(transfer.device_indices.device.type, "cpu")

    def test_reuse_of_inflight_pages_still_waits(self):
        allocator = _allocator(size=8)
        synchronize = Mock()
        fence = KVFreeFence(synchronize)
        fence.attach(allocator)
        indices = allocator.alloc(8, return_host=True)
        fence.on_batch_formed()
        fence.on_forward_launched(SimpleNamespace(copy_done=None))
        allocator.free(indices)
        self.assertEqual(allocator.held_pages.num_pages, 2)
        loaded = _controller(allocator).load(torch.arange(8))
        synchronize.assert_called_once()
        self.assertEqual(allocator.held_pages.num_pages, 0)
        torch.testing.assert_close(indices, loaded)

    def test_subclass_without_host_keyword_keeps_old_contract(self):
        class LegacyAllocator(TpuPagedTokenToKVPoolAllocator):
            def alloc(self, need_size):
                return super().alloc(need_size)

        allocator = LegacyAllocator(
            size=32,
            page_size=4,
            dtype=torch.bfloat16,
            device="meta",
            kvcache=None,
            need_sort=False,
        )
        self.assertEqual(
            _controller(allocator)
            ._alloc_load_kv_indices(allocator, 4, None)
            .device.type,
            "meta",
        )

    def test_load_refuses_forward_write_target(self):
        allocator = _allocator()
        fence = KVFreeFence(Mock())
        fence.attach(allocator)
        fence._forward_write_target = True
        with self.assertRaisesRegex(AssertionError, "DMA target"):
            _controller(allocator).load(torch.arange(4))


if __name__ == "__main__":
    unittest.main()
