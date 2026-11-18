"""CPU checks of the eviction-cause counters hook."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS, evict_cause_metrics

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeCounter:
    def __init__(self, *, name, documentation, labelnames):
        self.labelnames, self.values = labelnames, {}

    def labels(self, **labels):
        assert sorted(labels) == sorted(self.labelnames)
        key = labels["cause"]
        counter = self

        class _Child:
            def inc(self, n):
                counter.values[key] = counter.values.get(key, 0) + n

        return _Child()


class TestEvictCauseMetricsHook(CustomTestCase):
    def _collector(self):
        collector = SimpleNamespace(labels={"tp_rank": "0"}, _counter_cls=_FakeCounter)
        evict_cause_metrics.add_cause_counters(None, collector)
        return collector

    def test_counts_freed_tokens_and_states_by_the_short_pool(self):
        collector = self._collector()
        cache = SimpleNamespace(metrics_collector=collector)
        calls = (
            (EvictParams(num_tokens=256), EvictResult(num_tokens_evicted=384)),
            (
                EvictParams(mamba_num=1),
                EvictResult(num_tokens_evicted=128, mamba_num_evicted=2),
            ),
            (EvictParams(num_tokens=1, mamba_num=1), EvictResult(mamba_num_evicted=1)),
            (EvictParams(), EvictResult(num_tokens_evicted=7)),  # asked for nothing
        )
        for params, result in calls:
            self.assertIs(
                evict_cause_metrics.count_eviction(result, cache, params), result
            )
        self.assertEqual(
            collector.evicted_tokens_by_cause.values, {"kv": 384, "state": 128}
        )
        self.assertEqual(
            collector.evicted_states_by_cause.values, {"state": 2, "kv_and_state": 1}
        )

    def test_a_cache_without_metrics_is_left_alone(self):
        result = EvictResult(num_tokens_evicted=5)
        cache = SimpleNamespace(metrics_collector=None)
        self.assertIs(
            evict_cause_metrics.count_eviction(
                result, cache, EvictParams(num_tokens=5)
            ),
            result,
        )

    def test_hooks_are_registered(self):
        targets = {target for target, _, _ in HOOKS}
        for name in ("evict", "evict_for_alloc"):
            self.assertIn(
                f"sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.{name}",
                targets,
            )


if __name__ == "__main__":
    unittest.main()
