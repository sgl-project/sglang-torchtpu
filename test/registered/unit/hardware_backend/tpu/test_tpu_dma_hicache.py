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
            get_fused_kv_buffer=Mock(
                return_value=torch.empty((3, 4, 1, 2, 128), dtype=torch.bfloat16)
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


if __name__ == "__main__":
    unittest.main()
