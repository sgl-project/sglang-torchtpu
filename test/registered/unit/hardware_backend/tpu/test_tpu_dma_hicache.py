"""CPU tests for the TPU DMA HiCache backend registry: a registered backend
selects the TPU controllers and does the host pool's copies."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import base as host_base
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_pool_host import tpu_dma_host_kv_pool_cls
from sglang_tpu.tpu_dma_hicache import (
    NOOP_EVENT,
    is_tpu_dma_hicache_backend,
    register_tpu_dma_hicache_backend,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ENGINE = "test_engine"


class _Engine:
    def __init__(self, device_pool, num_host_pages):
        self.num_host_pages = num_host_pages
        self.calls = []

    def backup(self, device_pages, host_pages):
        self.calls.append(("backup", device_pages, host_pages))
        return NOOP_EVENT

    def load(self, host_pages, device_pages):
        self.calls.append(("load", host_pages, device_pages))
        return NOOP_EVENT


def setUpModule():
    register_tpu_dma_hicache_backend(ENGINE, _Engine)


class TestTpuDMAKVTransfer(CustomTestCase):
    def test_registration(self):
        self.assertTrue(is_tpu_dma_hicache_backend("raiden"))
        self.assertFalse(is_tpu_dma_hicache_backend("torch_tpu"))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_tpu_dma_hicache_backend(ENGINE, _Engine)

    def test_engines_select_the_tpu_controllers(self):
        with (
            patch(
                "sglang_tpu.tpu_dma_hybrid_cache_controller.TpuDMAHybridCacheController"
            ) as hybrid,
            patch(
                "sglang_tpu.tpu_dma_cache_controller.TpuDMAHiCacheController"
            ) as single,
        ):
            HybridCacheController.create_for_backend(io_backend=ENGINE)
            HiCacheController.create_for_backend(io_backend=ENGINE)
        hybrid.assert_called_once_with(io_backend=ENGINE)
        single.assert_called_once_with(io_backend=ENGINE)

    def test_host_pool_copies_whole_pages_through_its_engine(self):
        device_pool = SimpleNamespace(
            store_dtype=torch.bfloat16,
            size=8,
            start_layer=0,
            end_layer=2,
            layer_num=2,
            page_buffers=Mock(
                return_value=[
                    torch.empty((3, 4, 1, 2, 128), dtype=torch.bfloat16)
                    for _ in range(2)
                ]
            ),
        )
        with (
            patch.object(host_base, "ranks_per_host", return_value=1),
            patch.object(
                host_base, "available_host_memory_bytes", return_value=1 << 40
            ),
        ):
            pool = tpu_dma_host_kv_pool_cls(ENGINE)(
                device_pool,
                host_to_device_ratio=2,
                host_size=0,
                page_size=4,
                layout="page_first_fused",
            )
        engine = pool.transfer
        self.assertEqual(engine.num_host_pages, pool.page_num)
        host = pool.alloc(8)  # host pages 0 and 1
        device = torch.arange(4, 12)  # device pages 1 and 2
        pool.backup_from_device_all_layer(None, host, device, ENGINE)
        pool.load_all_layer(None, host, device)
        self.assertEqual(
            engine.calls, [("backup", [1, 2], [0, 1]), ("load", [0, 1], [1, 2])]
        )

    def test_fixed_host_size_syncs_capacity_across_asymmetric_pp_stages(self):
        """Fixed --hicache-size (GB) yields differing raw token counts on PP
        stages that own different layer counts (e.g. 5 vs 6 SWA layers on Muse
        Spark 7,8,8,8,8,8,8,7); TpuDMAHostKVPool must sync to the PP minimum
        like HostKVCache, while leaving ratio-sized pools untouched."""
        pool_cls = tpu_dma_host_kv_pool_cls(ENGINE)

        def make_device_pool(layer_num: int):
            return SimpleNamespace(
                store_dtype=torch.bfloat16,
                size=1024,
                start_layer=0,
                end_layer=layer_num,
                layer_num=layer_num,
                page_buffers=Mock(
                    return_value=[
                        torch.empty((8, 128, 8, 2, 128), dtype=torch.bfloat16)
                        for _ in range(layer_num)
                    ]
                ),
            )

        with (
            patch.object(host_base, "ranks_per_host", return_value=8),
            patch.object(
                host_base, "available_host_memory_bytes", return_value=1 << 42
            ),
            patch(
                "sglang_tpu.kv_pool_host.sync_fixed_hicache_size",
                side_effect=lambda raw_size, host_size: min(raw_size, 1500),
            ) as sync_mock,
        ):
            stage_5_layers = pool_cls(
                make_device_pool(5),
                host_to_device_ratio=0.0,
                host_size=4.0,
                page_size=128,
                layout="page_first_fused",
            )
            stage_6_layers = pool_cls(
                make_device_pool(6),
                host_to_device_ratio=0.0,
                host_size=4.0,
                page_size=128,
                layout="page_first_fused",
            )
            ratio_pool = pool_cls(
                make_device_pool(5),
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=128,
                layout="page_first_fused",
            )

        self.assertEqual(sync_mock.call_count, 2)
        self.assertEqual(stage_5_layers.size, stage_6_layers.size)
        self.assertEqual(stage_5_layers.page_num, stage_6_layers.page_num)
        # 1500 // 128 + 1 == 12 pages -> 1536 tokens
        self.assertEqual(stage_5_layers.size, 12 * 128)
        self.assertEqual(ratio_pool.size, (2048 // 128 + 1) * 128)


if __name__ == "__main__":
    unittest.main()
