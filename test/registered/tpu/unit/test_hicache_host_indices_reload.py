"""Real Raiden reloads with CPU destination indices and reused device pages.

These are byte-level correctness checks, not performance measurements. The
fixture bypasses storage workers, while using the real KV pools, allocator,
controller load/start_loading methods, DMA engine, and completion gate.
"""

import unittest

import torch
import torch_tpu  # noqa: F401

from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.kv_pool_host import TpuDMAHostKVPool
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)


class TestHiCacheHostIndicesReload(CustomTestCase):
    def _check_reload(self, dtype):
        device, page_size, source_tokens = "tpu:0", 64, 2048
        pool = TpuFusedMHATokenToKVPool(
            size=4096,
            page_size=page_size,
            dtype=dtype,
            head_num=1,
            head_dim=128,
            layer_num=4,
            device=device,
            enable_memory_saver=False,
        )
        generator = torch.Generator().manual_seed(9271)
        buffers, golden = pool.page_buffers(), []
        for buffer in buffers:
            values = (
                torch.randint(-15, 16, buffer.shape, generator=generator).float() / 4
            ).to(dtype)
            buffer.copy_(values)
            golden.append(values.view(torch.uint8))
        torch.accelerator.synchronize()

        host = TpuDMAHostKVPool(
            pool,
            host_to_device_ratio=1,
            host_size=0,
            page_size=page_size,
            layout="page_first_fused",
        )
        group = HostPoolGroup(
            [
                PoolEntry(
                    PoolName.KV, host, pool, lambda x: x, is_primary_index_anchor=True
                )
            ]
        )
        allocator = TpuPagedTokenToKVPoolAllocator(
            size=pool.size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        source = allocator.alloc(source_tokens, return_host=True)
        host_indices = host.alloc(source_tokens)
        host.backup_from_device_all_layer(
            pool, host_indices, source, "raiden"
        ).synchronize()

        controller = object.__new__(TpuDMAHybridCacheController)
        controller.mem_pool_device_allocator = allocator
        controller.kv_storage = None
        controller.mem_pool_device = pool
        controller.mem_pool_host = group
        controller.device = device
        controller.io_backend = "raiden"
        controller.page_size = page_size
        controller.load_queue, controller.write_queue = [], []
        controller.ack_load_queue, controller.ack_write_queue = [], []
        controller.layer_done_counter = BatchDoneCounter()
        available = allocator.available_size()

        for count in (64, 256, 1024):
            for repeat in range(4):
                with self.subTest(dtype=dtype, tokens=count, repeat=repeat):
                    # Change the source when reusing destination pages so stale
                    # contents cannot pass as a completed reload.
                    offset = (
                        (repeat * 7) % ((source_tokens - count) // page_size + 1)
                    ) * page_size
                    indices = controller.load(
                        host_indices[offset : offset + count], node_id=repeat
                    )
                    self.assertEqual(indices.device.type, "cpu")
                    self.assertEqual(indices.dtype, torch.int32)
                    self.assertEqual(indices.numel(), count)
                    self.assertFalse(torch.isin(indices, source).any().item())
                    producer = controller.start_loading()
                    controller.layer_done_counter.set_consumer(producer)
                    controller.layer_done_counter.wait()
                    self.assertTrue(
                        controller.ack_load_queue.pop(0).finish_event.query()
                    )
                    destination = (indices[::page_size] // page_size).long().to(device)
                    original = (
                        source[offset : offset + count : page_size] // page_size
                    ).long()
                    for buffer, expected in zip(buffers, golden):
                        actual = (
                            buffer.index_select(0, destination).cpu().view(torch.uint8)
                        )
                        self.assertTrue(
                            torch.equal(actual, expected.index_select(0, original))
                        )
                    torch.accelerator.synchronize()
                    allocator.free(indices)
                    allocator.merge_and_sort_free()
                    self.assertEqual(allocator.available_size(), available)
                    self.assertFalse(controller.load_queue)
                    self.assertFalse(controller.ack_load_queue)

        allocator.free(source)
        allocator.merge_and_sort_free()
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(host.free(host_indices), source_tokens)

    def test_bf16_reload_and_reuse(self):
        self._check_reload(torch.bfloat16)

    def test_fp8_reload_and_reuse(self):
        self._check_reload(torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
