"""CPU tests for the TPU DMA HiCache backend registry: a registered backend
selects the TPU controllers and does the host pool's copies."""

import threading
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
from sglang_tpu.disaggregation.decode_dram_offload.host_page_metadata import (
    HostPageMetadata,
)
from sglang_tpu.kv_pool_host import (
    TpuDMAHostKVPool,
    TpuDMAUnifiedHostKVPool,
    tpu_dma_host_kv_pool_cls,
)
from sglang_tpu.tpu_dma_hicache import (
    NOOP_EVENT,
    is_tpu_dma_hicache_backend,
    register_tpu_dma_hicache_backend,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ENGINE = "test_engine"


def _page_pool():
    pool = TpuDMAHostKVPool.__new__(TpuDMAHostKVPool)
    pool.page_size = 4
    pool.page_num = 4
    pool.lock = threading.RLock()
    pool.transfer = _Engine(None, pool.page_num)
    pool.clear()
    return pool


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
    def test_retained_metadata_is_cleared_when_pages_are_freed_and_reused(self):
        pool = _page_pool()
        table = HostPageMetadata(pool)
        first, kept = pool.alloc(4), pool.alloc(4)
        table.restore([0, 1], [b"old occupant", b"still owned"])
        pool.free(first)
        # Free invalidation precedes reuse, including a later failed landing.
        self.assertEqual(table.export([0, 1]), [b"", b"still owned"])
        pool.alloc(8)
        reused = pool.alloc(4)
        self.assertTrue(torch.equal(first, reused))
        pool.backup_from_device_all_layer(None, reused, kept, ENGINE)
        self.assertEqual(table.export([0]), [])
        table.restore([0], [b"new occupant"])
        self.assertEqual(table.export([0, 1]), [b"new occupant", b"still owned"])

    def test_rewriting_owned_pages_clears_only_their_metadata_before_backup(self):
        pool = _page_pool()
        table = HostPageMetadata(pool)
        indices = pool.alloc(8)
        table.restore([0, 1], [b"old", b"kept"])
        backup = Mock(side_effect=lambda *_: self.assertEqual(table.export([0]), []))
        pool.transfer.backup = backup
        pool.backup_from_device_all_layer(None, indices[:4], indices[4:], ENGINE)
        backup.assert_called_once_with([1], [0])
        self.assertEqual(table.export([1]), [b"kept"])

    def test_pool_reset_clears_metadata_for_every_table_bound_to_the_pool(self):
        pool = _page_pool()
        first, second = HostPageMetadata(pool), HostPageMetadata(pool)
        pool.alloc(8)
        first.restore([0, 1], [b"old0", b"old1"])
        self.assertEqual(second.export([0, 1]), [b"old0", b"old1"])
        pool.clear()
        self.assertEqual(first.export([0, 1]), [])
        self.assertEqual(second.export([0, 1]), [])
        self.assertEqual(pool.available_size(), 16)

    def test_translated_backup_invalidates_logical_metadata_before_writing(self):
        pool = _page_pool()
        table = HostPageMetadata(pool)
        indices = pool.alloc(8)
        table.restore([0, 1], [b"old", b"kept"])
        pool.set_page_translators(
            host=lambda pages: [p + 2 for p in pages],
            device=lambda pages: [p + 1 for p in pages],
        )
        pool.backup_from_device_all_layer(None, indices[:4], indices[4:], ENGINE)
        self.assertEqual(pool.transfer.calls, [("backup", [2], [2])])
        self.assertEqual(table.export([0, 1]), [b"", b"kept"])

    def test_unified_pool_free_and_clear_invalidate_its_logical_page_rows(self):
        for is_swa in (False, True):
            with self.subTest(swa=is_swa):
                pool = TpuDMAUnifiedHostKVPool.__new__(TpuDMAUnifiedHostKVPool)
                pool.page_size, pool.page_num = 4, 4
                pool.lock = threading.RLock()
                pool.pool_label = "swa" if is_swa else "full"
                pool._state = SimpleNamespace(free=Mock(return_value=4), clear=Mock())
                table = HostPageMetadata(pool)
                table.restore([0, 1], [b"old", b"kept"])
                indices = torch.arange(4)
                self.assertEqual(pool.free(indices), 4)
                pool._state.free.assert_called_once_with(pool.pool_label, indices)
                self.assertEqual(table.export([0, 1]), [b"", b"kept"])
                pool.clear()
                pool._state.clear.assert_called_once_with(pool.pool_label)
                self.assertEqual(table.export([0, 1]), [])

    def test_unified_backup_invalidates_logical_rows_for_expanded_physical_pages(self):
        for is_swa in (False, True):
            with self.subTest(swa=is_swa):
                pool = TpuDMAUnifiedHostKVPool.__new__(TpuDMAUnifiedHostKVPool)
                pool.page_size, pool.page_num = 4, 4
                pool._is_swa = is_swa
                groups = torch.tensor([[4, 5, 6, 7]])
                pool._unified_pool = SimpleNamespace(swa_group_pages=groups)
                pool._state = SimpleNamespace(swa_group_pages=groups)
                pool.transfer = _Engine(None, 8)
                table = HostPageMetadata(pool)
                table.restore([0, 1], [b"old", b"kept"])
                pool.backup_from_device_all_layer(
                    None, torch.arange(4), torch.arange(4, 8), ENGINE
                )
                expected = (
                    ("backup", [1, 5], [0, 4]) if is_swa else ("backup", [1], [0])
                )
                self.assertEqual(pool.transfer.calls, [expected])
                self.assertEqual(table.export([0, 1]), [b"", b"kept"])

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
