"""Host-index reload contracts, including allocation ownership and fallback.

Meta-device allocations make an accidental device-to-host read fail while
exercising the real allocator and controller without TPU hardware.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.kv_free_fence import KVFreeFence
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

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
    controller.load_queue = []
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

    def test_side_pool_and_wrapped_allocators_keep_old_contract(self):
        allocator = _allocator()
        controller = _controller(allocator)
        extra = [PoolTransfer(name=PoolName.SWA)]
        self.assertEqual(
            controller._alloc_load_kv_indices(allocator, 4, extra).device.type,
            "meta",
        )
        controller.mem_pool_device_allocator = SimpleNamespace(
            full_attn_allocator=allocator
        )
        self.assertEqual(
            controller._alloc_load_kv_indices(allocator, 4, None).device.type,
            "meta",
        )

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
