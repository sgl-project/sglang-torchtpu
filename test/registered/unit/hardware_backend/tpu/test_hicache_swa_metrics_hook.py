"""CPU checks of the HiCache host SWA gauges hook."""

import pkgutil
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS, hicache_swa_metrics

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _build_host_pair(page_size: int = 4):
    bundle = init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=2,
        head_dim=4,
        v_head_dim=4,
        swa_head_num=2,
        swa_head_dim=4,
        swa_v_head_dim=4,
        page_size=page_size,
        start_layer=0,
        end_layer=4,
        swa_attention_layer_ids=[3],
        full_attention_layer_ids=[0, 1, 2],
        total_bytes=4096,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
    )
    pool = bundle.token_to_kv_pool
    # The host pool keeps a fixed RAM reserve; low-memory CI runners fall below it.
    with host_memory_budget_scope(1 << 40):
        return UnifiedPageEnvelopeHostPool.build_hybrid_swa_pool_pair(
            device_pools=(pool.full_kv_pool, pool.swa_kv_pool),
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=pool.page_size,
            layout="layer_first",
            pin_memory=False,
            allocator_type="default",
        )


class _FakeGauge:
    def __init__(self, *, name, documentation, labelnames, multiprocess_mode):
        self.name = name
        self.value = None

    def labels(self, **labels):
        return self

    def set(self, value):
        self.value = value


class TestHicacheSwaMetricsHook(CustomTestCase):
    def setUp(self):
        self.full_pool, self.swa_pool = _build_host_pair()
        self.addCleanup(self.full_pool.destroy)
        self.addCleanup(self.swa_pool.destroy)

    def test_used_tokens_counts_only_this_sides_pages(self):
        """Free bytes are shared, so FULL allocations shrink SWA's available size;
        SWA's used tokens must still count only SWA pages."""
        swa_available = self.swa_pool.available_size()
        self.assertIsNotNone(
            self.full_pool.alloc(self.full_pool.available_size() // 8 * 4)
        )
        self.assertLess(self.swa_pool.available_size(), swa_available)
        self.assertEqual(hicache_swa_metrics.used_tokens(self.swa_pool), 0)
        self.assertIsNotNone(self.swa_pool.alloc(4))
        self.assertEqual(hicache_swa_metrics.used_tokens(self.swa_pool), 4)

    def test_hooks_export_swa_host_pool_occupancy(self):
        collector = SimpleNamespace(
            enable_hierarchical_cache=True,
            _gauge_cls=_FakeGauge,
            labels={"model_name": "m"},
            _log_gauge=lambda gauge, value: gauge.labels().set(value),
        )
        hicache_swa_metrics.add_swa_host_gauges(None, collector)
        tree_cache = mock.create_autospec(UnifiedRadixCache, instance=True)
        tree_cache.tree_core = SimpleNamespace(has_swa_host_pool=True)
        tree_cache.host_pool_group = mock.Mock()
        tree_cache.host_pool_group.get_pool.return_value = self.swa_pool
        reporter = SimpleNamespace(
            scheduler=SimpleNamespace(
                enable_hierarchical_cache=True, tree_cache=tree_cache
            ),
            metrics_collector=collector,
        )
        self.assertIsNotNone(self.swa_pool.alloc(4))

        hicache_swa_metrics.log_swa_host_stats(None, reporter)

        tree_cache.host_pool_group.get_pool.assert_called_once_with(PoolName.SWA)
        self.assertEqual(collector.hicache_host_swa_used_tokens.value, 4)
        self.assertEqual(
            collector.hicache_host_swa_total_tokens.value, self.swa_pool.logical_size
        )

    def test_no_swa_host_pool_leaves_gauges_unset(self):
        collector = SimpleNamespace(
            enable_hierarchical_cache=True,
            _gauge_cls=_FakeGauge,
            labels={},
            _log_gauge=lambda gauge, value: gauge.labels().set(value),
        )
        hicache_swa_metrics.add_swa_host_gauges(None, collector)
        tree_cache = mock.create_autospec(UnifiedRadixCache, instance=True)
        tree_cache.tree_core = SimpleNamespace(has_swa_host_pool=False)
        reporter = SimpleNamespace(
            scheduler=SimpleNamespace(
                enable_hierarchical_cache=True, tree_cache=tree_cache
            ),
            metrics_collector=collector,
        )

        hicache_swa_metrics.log_swa_host_stats(None, reporter)

        self.assertIsNone(collector.hicache_host_swa_used_tokens.value)
        self.assertIsNone(collector.hicache_host_swa_total_tokens.value)

    def test_hook_targets_resolve_and_are_registered(self):
        """A renamed target only logs at hook application, so pin both here."""
        for target, hook, hook_type in hicache_swa_metrics.hook_specs():
            self.assertIn((target, hook, hook_type), HOOKS)
            pkgutil.resolve_name(target)


if __name__ == "__main__":
    unittest.main()
