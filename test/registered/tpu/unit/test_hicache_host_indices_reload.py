"""Real Raiden reloads with CPU destination indices and reused device pages.

These are byte-level correctness checks, not performance measurements. The
fixture bypasses storage workers, while using the real KV pools, allocator,
controller load/start_loading methods, DMA engine, and completion gate.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_tpu  # noqa: F401

from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.kv_pool_host import _TPU_UNIFIED_SWA_STRATEGY, TpuDMAHostKVPool
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
)

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "prefix-cache", "kv-offload"],
)


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

    def _check_hybrid_reload(self, dtype):
        device, page_size, size = "tpu:0", 64, 8192
        pool = TpuUnifiedSWAKVPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            full=AttnLayerSpec(1, 128, [0, 3]),
            swa=AttnLayerSpec(1, 128, [1, 2, 4, 5, 6]),
        )
        allocator = TpuUnifiedSWATokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
            sliding_window_size=256,
        )
        buffers, golden = pool.kv_pool.page_buffers(), []
        generator = torch.Generator().manual_seed(8731)
        for buffer in buffers:
            values = (
                torch.randint(-15, 16, buffer.shape, generator=generator).float() / 4
            ).to(dtype)
            buffer.copy_(values)
            golden.append(values.view(torch.uint8))
        torch.accelerator.synchronize()
        controller = object.__new__(TpuDMAHybridCacheController)
        controller.mem_pool_device_allocator = allocator
        controller.kv_storage = None
        controller.mem_pool_device = pool
        controller.device, controller.io_backend = device, "raiden"
        controller.page_size = page_size
        controller.load_queue, controller.write_queue = [], []
        controller.ack_load_queue, controller.ack_write_queue = [], []
        controller.layer_done_counter = BatchDoneCounter()
        params = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            mtp_draft_device_pools=None,
            page_size=page_size,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        with (
            get_context().override_server_args(
                hicache_io_backend="raiden",
                hicache_ratio=2.0,
                hicache_size=0,
                hicache_write_policy="write_through",
                hicache_host_memory_mode="cache",
            ),
            patch.object(
                HybridCacheController, "create_for_backend", return_value=controller
            ),
        ):
            result = _TPU_UNIFIED_SWA_STRATEGY.build(
                cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
                kvcache=pool,
                params=params,
                server_args=None,
                load_cache_event=None,
            )
        controller.mem_pool_host = result.host_pool_group
        with patch.object(HybridCacheController, "__init__", return_value=None):
            controller.__init__()
        hosts = [
            result.host_pool_group.get_pool(name)
            for name in (PoolName.KV, PoolName.SWA)
        ]
        for host in hosts:
            self.addCleanup(host.destroy)
        self.assertEqual(allocator.full_to_swa_index_mapping.device.type, "cpu")
        full_alloc, swa_alloc = (
            allocator.full_attn_allocator,
            allocator.swa_attn_allocator,
        )
        sources = [
            full_alloc.alloc(2048, return_host=True),
            swa_alloc.alloc(512, return_host=True),
        ]
        host_indices = [host.alloc(len(source)) for host, source in zip(hosts, sources)]
        for host, src, indices in zip(hosts, sources, host_indices):
            host.backup_from_device_all_layer(
                pool, indices, src, "raiden"
            ).synchronize()
        available = allocator.full_available_size()

        def physical_pages(indices, is_swa):
            primary = (indices[::page_size] // page_size).long()
            if not is_swa:
                return primary
            return (
                torch.cat([primary[None], pool.swa_group_pages[:, primary]], dim=0)
                .T.flatten()
                .long()
            )

        for count in (64, 256, 1024):
            for repeat in range(4):
                with self.subTest(dtype=dtype, full=count, repeat=repeat):
                    swa_count = min(count, 256)
                    offsets = [
                        ((repeat * 7) % ((2048 - count) // page_size + 1)) * page_size,
                        (repeat % ((512 - swa_count) // page_size + 1)) * page_size,
                    ]
                    swa = PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=host_indices[1][
                            offsets[1] : offsets[1] + swa_count
                        ],
                    )
                    full = controller.load(
                        host_indices[0][offsets[0] : offsets[0] + count],
                        node_id=repeat,
                        extra_pools=[swa],
                    )
                    destinations = [full, swa.device_indices]
                    for indices in destinations:
                        self.assertEqual(
                            (indices.device.type, indices.dtype), ("cpu", torch.int32)
                        )
                    producer = controller.start_loading()
                    controller.layer_done_counter.set_consumer(producer)
                    controller.layer_done_counter.wait()
                    self.assertTrue(
                        controller.ack_load_queue.pop(0).finish_event.query()
                    )
                    for side, (indices, offset, length) in enumerate(
                        zip(destinations, offsets, (count, swa_count))
                    ):
                        src_pages = physical_pages(
                            sources[side][offset : offset + length], bool(side)
                        )
                        dst_pages = physical_pages(indices, bool(side))
                        self.assertFalse(torch.isin(dst_pages, src_pages).any().item())
                        # SWA comparisons include all three physical page groups.
                        for buffer, expected in zip(buffers, golden):
                            actual = (
                                buffer.index_select(0, dst_pages.to(device))
                                .cpu()
                                .view(torch.uint8)
                            )
                            self.assertTrue(
                                torch.equal(
                                    actual, expected.index_select(0, src_pages)
                                ),
                                f"{dtype}: side={side}, count={count}, repeat={repeat}",
                            )
                    torch.accelerator.synchronize()
                    swa_alloc.free(swa.device_indices)
                    full_alloc.free(full)
                    full_alloc.merge_and_sort_free()
                    self.assertEqual(allocator.full_available_size(), available)
                    self.assertFalse(controller.load_queue)
                    self.assertFalse(controller.ack_load_queue)
        swa_alloc.free(sources[1])
        full_alloc.free(sources[0])
        self.assertEqual(allocator.full_available_size(), size)
        self.assertEqual(swa_alloc.allocated_count(), 0)
        self.assertEqual(full_alloc.allocated_count(), 0)
        for host, indices in zip(hosts, host_indices):
            self.assertEqual(host.free(indices), len(indices))

    def test_hybrid_bf16_reload_and_reuse(self):
        self._check_hybrid_reload(torch.bfloat16)

    def test_hybrid_fp8_reload_and_reuse(self):
        self._check_hybrid_reload(torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
