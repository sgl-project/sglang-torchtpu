"""CPU checks of the eviction-age hook and the tree core's wall-clock recency stamps."""

import contextlib
import pkgutil
import unittest
from array import array
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client
import test_unified_radix_cache_unittest as radix_fixture
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.plugins.hook_registry import HookType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS, eviction_age_metrics

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextlib.contextmanager
def _hooks_applied():
    """Apply the module's hooks for one test only, without the global registry."""
    with contextlib.ExitStack() as stack:
        for target, hook, hook_type in eviction_age_metrics.hook_specs():
            owner_path, name = target.rsplit(".", 1)
            owner = pkgutil.resolve_name(owner_path)
            original = getattr(owner, name)
            if hook_type is HookType.AROUND:

                def wrapper(*args, _orig=original, _hook=hook, **kwargs):
                    return _hook(_orig, *args, **kwargs)

            else:

                def wrapper(*args, _orig=original, _hook=hook, **kwargs):
                    result = _orig(*args, **kwargs)
                    _hook(result, *args, **kwargs)
                    return result

            stack.enter_context(mock.patch.object(owner, name, wrapper))
        yield


def _collector():
    registry = prometheus_client.CollectorRegistry()
    # The cache calls other collector methods during eviction; only these counters are real.
    collector = mock.MagicMock()
    collector._counter_cls = partial(prometheus_client.Counter, registry=registry)
    collector.labels = {"model_name": "m"}
    eviction_age_metrics.add_eviction_age_counters(None, collector)

    def sample(pool, last_copy=True):
        labels = {"model_name": "m", "pool": pool, "last_copy": str(last_copy).lower()}
        tokens = registry.get_sample_value("sglang:eviction_age_tokens_total", labels)
        seconds = registry.get_sample_value(
            "sglang:eviction_age_token_seconds_total", labels
        )
        return tokens, seconds

    return collector, sample


class TestDeviceEvictionAge(CustomTestCase):
    def _cache(self):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=4,
        )
        # The fixture asks for an accelerator; the CPU CI runner has none.
        with mock.patch.object(radix_fixture, "get_device", return_value="cpu"):
            cache, allocator, _ = build_fixture(cfg, tree_core_backend="python")
        cache.metrics_collector, sample = _collector()
        return cache, allocator, sample

    def _insert(self, cache, allocator, tokens):
        value = allocator.alloc(len(tokens))
        cache.insert(
            InsertParams(key=RadixKey(array("q", tokens)), value=value[: len(tokens)])
        )

    def test_each_pool_reports_only_its_own_walk(self):
        """A pool's walk reports the age of what it evicted, from that pool's own
        last refresh; cascade frees of the other pool are not attributed to it."""
        seq = list(range(1, 17))
        for evict_params, now, pool in (
            (EvictParams(swa_num_tokens=len(seq)), 130.0, "swa"),
            (EvictParams(num_tokens=len(seq)), 150.0, "kv"),
        ):
            with self.subTest(pool=pool), _hooks_applied():
                cache, allocator, sample = self._cache()
                with mock.patch("time.monotonic", return_value=100.0):
                    self._insert(cache, allocator, seq)
                with mock.patch("time.monotonic", return_value=now):
                    cache.evict(evict_params)
                tokens, seconds = sample(pool)
                self.assertGreater(tokens, 0)
                self.assertAlmostEqual(seconds, (now - 100.0) * tokens)
                other = "kv" if pool == "swa" else "swa"
                self.assertEqual(sample(other), (None, None))

    def test_match_that_splits_a_node_restarts_full_kv_age(self):
        """A match diverging inside a node refreshes both halves, so their age
        counts from the split, not from the insert."""
        with _hooks_applied():
            cache, allocator, sample = self._cache()
            with mock.patch("time.monotonic", return_value=100.0):
                self._insert(cache, allocator, list(range(1, 9)))
            with mock.patch("time.monotonic", return_value=200.0):
                cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 99])))
                )
            with mock.patch("time.monotonic", return_value=300.0):
                cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(sample("kv"), (8, 8 * 100.0))


class _Core:
    """Hashable, weak-referenceable stand-in; the hook keys its state by tree core."""

    def __init__(self, lru_lists):
        self.lru_lists = lru_lists


class TestHostEvictionAge(CustomTestCase):
    """Host victims through the recording hook, with stand-ins: the HiCache
    fixture needs an accelerator."""

    def setUp(self):
        self.core = _Core({ComponentType.SWA: SimpleNamespace(_pt=ComponentType.SWA)})
        self.addCleanup(eviction_age_metrics._walks.pop, self.core, None)
        self.addCleanup(eviction_age_metrics._pending.pop, self.core, None)

    def _node(self, *, device_value, host_value=None):
        data = [SimpleNamespace(value=None, host_value=None) for _ in ComponentType]
        data[ComponentType.SWA] = SimpleNamespace(
            value=device_value, host_value=host_value
        )
        data[ComponentType.FULL] = SimpleNamespace(
            value=device_value, host_value=host_value
        )
        # Device SWA slot last used at 40; entering the host LRU at 90 is no use.
        lru_wall = [0.0] * (2 * len(ComponentType))
        lru_wall[ComponentType.SWA] = 40.0
        lru_wall[ComponentType.SWA + len(ComponentType)] = 90.0
        return SimpleNamespace(
            component_data=data, last_access_wall=10.0, lru_wall=lru_wall
        )

    def _record(self, walk, comp_type, node, freed):
        eviction_age_metrics._walks[self.core] = walk
        with mock.patch("time.monotonic", return_value=100.0):
            eviction_age_metrics.record_eviction(
                freed, self.core, node, SimpleNamespace(component_type=comp_type)
            )
        return eviction_age_metrics._pending.get(self.core, {})

    def test_host_swa_victim_is_aged_from_its_device_slot_and_labeled_by_copy(self):
        dropped_duplicate = self._record(
            (ComponentType.SWA, True),
            ComponentType.SWA,
            self._node(device_value=[1]),
            (0, 4),
        )
        self.assertEqual(dropped_duplicate, {("host_swa", False): [4, 4 * 60.0]})
        eviction_age_metrics._pending.pop(self.core)
        forgotten = self._record(
            (ComponentType.SWA, True),
            ComponentType.SWA,
            self._node(device_value=None),
            (0, 4),
        )
        self.assertEqual(forgotten, {("host_swa", True): [4, 4 * 60.0]})

    def test_host_kv_leaf_is_aged_from_last_access(self):
        recorded = self._record(
            (ComponentType.FULL, True),
            ComponentType.FULL,
            self._node(device_value=None),
            (0, 8),
        )
        self.assertEqual(recorded, {("host_kv", True): [8, 8 * 90.0]})

    def test_cascades_and_other_layers_are_not_recorded(self):
        """A host walk's cascade into another pool, and device frees during a host
        walk, are not that pool's own evictions."""
        node = self._node(device_value=None)
        self.assertEqual(
            self._record((ComponentType.FULL, True), ComponentType.SWA, node, (0, 4)),
            {},
        )
        self.assertEqual(
            self._record((ComponentType.SWA, True), ComponentType.SWA, node, (4, 0)),
            {},
        )

    def test_hook_targets_resolve_and_are_registered(self):
        """A renamed target only logs at hook application, so pin every target here."""
        for target, hook, hook_type in eviction_age_metrics.hook_specs():
            self.assertIn((target, hook, hook_type), HOOKS)
            pkgutil.resolve_name(target)


if __name__ == "__main__":
    unittest.main()
