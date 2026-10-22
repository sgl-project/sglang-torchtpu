"""CPU checks of the SWA-miss counter hook."""

import pkgutil
import unittest
from array import array
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client
import test_unified_radix_cache_unittest as radix_fixture
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.session.streaming_session import _VirtualNode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS, swa_miss_metrics
from sglang_tpu.hooks.swa_miss_metrics import SwaPrefixMatch, swa_miss_range

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Req:
    """Hashable stand-in; the hook keys its state by request identity."""

    def __init__(self, start=0, end=0, retracted=False):
        self.session = None
        self.retracted_stain = retracted
        self.extend_range = SimpleNamespace(start=start, end=end)


class _Adder:
    def __init__(self):
        self.can_run_list = []


class _Stats:
    """Weak-referenceable like PrefillStats, which the hook snapshots by id."""


class TestSwaMissMatch(CustomTestCase):
    def setUp(self):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=4,
        )
        # The fixture asks for an accelerator; the CPU CI runner has none.
        with mock.patch.object(radix_fixture, "get_device", return_value="cpu"):
            self.cache, self.allocator, _ = build_fixture(
                cfg, tree_core_backend="python"
            )

    def _insert(self, tokens, swa_evicted=0):
        value = self.allocator.alloc(len(tokens))
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value[: len(tokens)],
                component_evicted_seqlens=(
                    {ComponentType.SWA: swa_evicted} if swa_evicted else {}
                ),
            )
        )

    def _recorded(self, tokens) -> SwaPrefixMatch:
        req = _Req()
        params = MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        swa_miss_metrics.record_match(
            self.cache.match_prefix(params), self.cache, params
        )
        return swa_miss_metrics._matches[req]

    def test_swa_tombstones_under_cached_full_kv_are_a_miss(self):
        seq = list(range(1, 17))
        self._insert(seq, swa_evicted=len(seq))
        match = self._recorded(seq)
        start, end = swa_miss_range(match=match, prefix_len=0)
        self.assertEqual((match.full_kv_hit_length, max(0, end - start)), (16, 16))

    def test_reusable_prefix_is_the_best_match_depth_across_a_split(self):
        """The hook derives the validators' best match from best_match_node's depth;
        it must equal the tokens matched, including at a node a split created."""
        self._insert(list(range(1, 17)))
        self._insert(list(range(1, 9)) + list(range(50, 58)))
        match = self._recorded(list(range(1, 9)) + [99])
        self.assertEqual(match, SwaPrefixMatch(full_kv_hit_length=8, reusable_length=8))
        match = self._recorded(list(range(1, 17)) + [99])
        self.assertEqual(match.reusable_length, 16)

    def test_streaming_session_match_is_not_resolved_as_a_tree_node(self):
        """A streaming session's next turn returns a sentinel best_match_node; the
        hook must neither look it up in the tree nor count the turn as a miss."""
        req = _Req()
        swa_miss_metrics._matches[req] = SwaPrefixMatch(full_kv_hit_length=16)
        params = MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])), req=req)
        session_result = self.cache.match_prefix(params)._replace(
            best_match_node=_VirtualNode(), full_kv_hit_length=0
        )
        swa_miss_metrics.record_match(session_result, self.cache, params)
        self.assertNotIn(req, swa_miss_metrics._matches)


class TestSwaMissCharging(CustomTestCase):
    def test_swa_miss_range_excludes_restored_and_declined_load_back(self):
        """Load-back that restores a missing window extends the admitted prefix, so
        those tokens are reused, not missed; declining a restorable load-back is a
        capacity loss, not an SWA miss."""
        cases = [
            # reusable_length, prefix_len, swa_miss
            (16, 32, 0),
            (16, 24, 8),
            (16, 16, 16),
            (16, 8, 16),
            (16, 40, 0),
            (32, 16, 0),
        ]
        for reusable_length, prefix_len, expected in cases:
            with self.subTest(reusable_length=reusable_length, prefix_len=prefix_len):
                match = SwaPrefixMatch(
                    full_kv_hit_length=32, reusable_length=reusable_length
                )
                start, end = swa_miss_range(match=match, prefix_len=prefix_len)
                self.assertEqual(max(0, end - start), expected)

    def test_swa_miss_counted_per_scheduled_chunk(self):
        """SWA misses are charged as their chunks are scheduled, like input tokens,
        so a prefill aborted between chunks counts only what it recomputed; a
        retracted re-admission is not counted again."""
        adder = _Adder()
        req = _Req(0, 8)
        swa_miss_metrics._matches[req] = SwaPrefixMatch(full_kv_hit_length=24)
        swa_miss_metrics.charge_admission(None, adder, req, 0)
        self.assertEqual(swa_miss_metrics._adder_tokens[adder], 8)

        def schedule(adder, req):
            req.extend_range = SimpleNamespace(start=8, end=32)
            adder.can_run_list.append(req)

        swa_miss_metrics.charge_continuation_chunk(schedule, adder, req)
        self.assertEqual(swa_miss_metrics._adder_tokens[adder], 24)
        swa_miss_metrics.charge_continuation_chunk(lambda adder, req: req, adder, req)
        self.assertEqual(swa_miss_metrics._adder_tokens[adder], 24)

        retracted = _Req(0, 16, retracted=True)
        swa_miss_metrics._matches[retracted] = SwaPrefixMatch(full_kv_hit_length=16)
        swa_miss_metrics.charge_admission(None, adder, retracted, 0)
        self.assertEqual(swa_miss_metrics._adder_tokens[adder], 24)

    def test_batch_total_reaches_the_counter_and_zero_still_exports(self):
        """Dashboard ratios mask replicas by series presence, so a zero batch must
        still create the series; a charged batch adds its snapshot exactly once."""
        registry = prometheus_client.CollectorRegistry()
        collector = SimpleNamespace(
            _counter_cls=partial(prometheus_client.Counter, registry=registry),
            labels={"model_name": "m"},
        )
        swa_miss_metrics.add_swa_miss_counter(None, collector)
        reporter = SimpleNamespace(
            current_scheduler_metrics_enabled=True,
            scheduler=SimpleNamespace(tree_cache=object()),
            metrics_collector=collector,
        )

        def sample():
            return registry.get_sample_value(
                "sglang:prefill_swa_miss_tokens_total", {"model_name": "m"}
            )

        with mock.patch.object(
            swa_miss_metrics, "counts_swa_misses", return_value=True
        ):
            empty = _Stats()
            swa_miss_metrics.snapshot_batch(empty, None, _Adder())
            swa_miss_metrics.report_swa_miss_tokens(None, reporter, prefill_stats=empty)
            self.assertEqual(sample(), 0)

            adder = _Adder()
            swa_miss_metrics._adder_tokens[adder] = 12
            stats = _Stats()
            swa_miss_metrics.snapshot_batch(stats, None, adder)
            swa_miss_metrics.report_swa_miss_tokens(None, reporter, prefill_stats=stats)
            swa_miss_metrics.report_swa_miss_tokens(None, reporter, prefill_stats=stats)
            self.assertEqual(sample(), 12)

    def test_hook_targets_resolve_and_are_registered(self):
        """A renamed target only logs at hook application, so pin every target here."""
        for target, hook, hook_type in swa_miss_metrics.hook_specs():
            self.assertIn((target, hook, hook_type), HOOKS)
            pkgutil.resolve_name(target)


if __name__ == "__main__":
    unittest.main()
