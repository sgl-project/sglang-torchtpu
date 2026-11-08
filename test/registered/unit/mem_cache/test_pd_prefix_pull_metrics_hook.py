"""CPU checks of the P/D prefix-pull metrics: the pull-point search and its hooks."""

import pkgutil
import unittest
from array import array
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client
import test_unified_radix_cache_unittest as radix_fixture
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.disaggregation.decode_hicache_mixin import DecodePrefixMatch
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.prefix_pull import (
    DecodePrefixReport,
    DecodeWindowCase,
    PathSegment,
    PrefixGapCase,
    classify_prefix_gap,
    decode_match_cap,
    iter_path_segments,
    longest_pullable_len,
)
from sglang_tpu.hooks import HOOKS, pd_prefix_pull_metrics
from sglang_tpu.hooks.pd_prefix_pull_metrics import DecodeWindowStats, PrefillGapStats

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _segs(*spans):
    return [PathSegment(start=s, end=e, swa_resident=r) for s, e, r in spans]


class TestLongestPullableLen(CustomTestCase):
    def test_resident_window_at_the_limit(self):
        segs = _segs((0, 512, False), (512, 1024, True))
        self.assertEqual(longest_pullable_len(segs, 1024, 256, 256), 1024)

    def test_window_must_fit_inside_one_resident_run(self):
        # Only 256 resident tokens before 1024, the window needs 512.
        segs = _segs((0, 768, False), (768, 1024, True))
        self.assertEqual(longest_pullable_len(segs, 1024, 512, 256), 0)

    def test_falls_back_to_a_shallower_run(self):
        segs = _segs(
            (0, 256, False), (256, 1024, True), (1024, 1280, False), (1280, 1536, True)
        )
        self.assertEqual(longest_pullable_len(segs, 1536, 512, 256), 1024)

    def test_limit_clips_the_run_to_a_page(self):
        segs = _segs((0, 2048, True))
        self.assertEqual(longest_pullable_len(segs, 1000, 512, 256), 768)

    def test_adjacent_resident_nodes_form_one_run(self):
        segs = _segs((0, 256, False), (256, 512, True), (512, 768, True))
        self.assertEqual(longest_pullable_len(segs, 768, 512, 256), 768)

    def test_window_longer_than_the_prefix_needs_it_from_zero(self):
        self.assertEqual(
            longest_pullable_len(_segs((0, 256, True)), 256, 512, 256), 256
        )
        segs = _segs((0, 256, False), (256, 512, True))
        self.assertEqual(longest_pullable_len(segs, 512, 1024, 256), 0)

    def test_no_prefix(self):
        self.assertEqual(longest_pullable_len([], 0, 512, 256), 0)


class TestClassify(CustomTestCase):
    def test_prefix_gap_cases(self):
        cases = [
            # decode, prefill, expected case, gap
            (0, 0, PrefixGapCase.DECODE_MISS, 0),
            (0, 512, PrefixGapCase.DECODE_MISS, 0),
            (512, 1024, PrefixGapCase.PREFILL_COVERS, 0),
            (1024, 1024, PrefixGapCase.PREFILL_COVERS, 0),
            (2048, 1024, PrefixGapCase.DECODE_AHEAD_SMALL, 1024),
            (8192, 1024, PrefixGapCase.DECODE_AHEAD_PULL, 7168),
        ]
        for decode, prefill, case, gap in cases:
            with self.subTest(decode=decode, prefill=prefill):
                self.assertEqual(
                    classify_prefix_gap(
                        decode_prefix_len=decode,
                        prefill_prefix_len=prefill,
                        min_pull_tokens=4096,
                    ),
                    (case, gap),
                )

    def test_decode_match_cap_is_the_page_aligned_window_start(self):
        self.assertEqual(decode_match_cap(10000, 2048, 256), 7936)
        self.assertEqual(decode_match_cap(1000, 2048, 256), 0)


class TestDecodeTreePath(CustomTestCase):
    def setUp(self):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=4,
        )
        with mock.patch.object(radix_fixture, "get_device", return_value="cpu"):
            self.cache, self.allocator, _ = build_fixture(
                cfg, tree_core_backend="python"
            )

    def _insert(self, tokens, swa_evicted):
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=self.allocator.alloc(len(tokens))[: len(tokens)],
                component_evicted_seqlens={ComponentType.SWA: swa_evicted},
            )
        )

    def _match(self, tokens) -> DecodePrefixMatch:
        result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        return DecodePrefixMatch(
            prefix_indices=result.device_indices,
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=result.last_device_node,
        )

    def test_segments_mark_the_tombstoned_prefix(self):
        seq = list(range(1, 17))
        self._insert(seq, swa_evicted=8)
        match = self._match(seq)
        segs = list(
            iter_path_segments(
                self.cache.tree_core, match.last_device_node, ComponentType.SWA
            )
        )
        self.assertEqual(segs[-1].end, 16)
        resident = [t for s in segs for t in range(s.start, s.end) if s.swa_resident]
        self.assertEqual(resident, list(range(8, 16)))


def _collector():
    registry = prometheus_client.CollectorRegistry()
    collector = SimpleNamespace(
        labels={"pp_rank": "0"},
        _counter_cls=partial(prometheus_client.Counter, registry=registry),
    )
    pd_prefix_pull_metrics.add_prefix_pull_counters(None, collector)
    return collector, registry


def _reporter(collector):
    return SimpleNamespace(
        current_scheduler_metrics_enabled=True,
        is_stats_logging_rank=True,
        metrics_collector=collector,
    )


class _Req:
    """Hashable stand-in; the decode hook keys its state by request identity."""

    def __init__(self, decode_prefix_len=0, prompt_len=0):
        self.retracted_stain = False
        self.pending_bootstrap = False
        self.bootstrap_host = "10.0.0.1"
        self.disagg_decode_prefix_len = decode_prefix_len
        self.bootstrap_room = None
        self.origin_input_ids = [0] * prompt_len


class TestPrefillHooks(CustomTestCase):
    def setUp(self):
        self.stats = PrefillGapStats()
        for target, value in (
            ("_prefill_stats", self.stats),
            ("_is_counting_prefill_rank", lambda: True),
        ):
            patcher = mock.patch.object(pd_prefix_pull_metrics, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        tree_cache = SimpleNamespace(
            supports_swa=lambda: True, sliding_window_size=2048
        )
        self.adder = SimpleNamespace(page_size=256, tree_cache=tree_cache)

    def _admit(self, req, prefix_len):
        pd_prefix_pull_metrics.record_prefill_admission(
            None, self.adder, req, prefix_len
        )

    def test_counts_gap_by_case_and_flushes_once(self):
        collector, registry = _collector()
        # Decode reached its cap (window start 7936) and is 7936 - 1024 ahead.
        self._admit(_Req(decode_prefix_len=7936, prompt_len=10000), 1024)
        self._admit(_Req(decode_prefix_len=2048, prompt_len=10000), 2048)
        self._admit(_Req(decode_prefix_len=0, prompt_len=10000), 0)
        retracted = _Req(decode_prefix_len=7936, prompt_len=10000)
        retracted.retracted_stain = True
        self._admit(retracted, 0)
        reporter = _reporter(collector)
        pd_prefix_pull_metrics.flush_prefill_stats(None, reporter)
        pd_prefix_pull_metrics.flush_prefill_stats(None, reporter)

        def reqs(case, where):
            return registry.get_sample_value(
                "sglang:pd_prefill_prefix_gap_reqs_total",
                {"pp_rank": "0", "case": case.value, "decode_match": where},
            )

        self.assertEqual(reqs(PrefixGapCase.DECODE_AHEAD_PULL, "to_window"), 1)
        self.assertEqual(reqs(PrefixGapCase.PREFILL_COVERS, "interior"), 1)
        self.assertEqual(reqs(PrefixGapCase.DECODE_MISS, "interior"), 1)
        self.assertEqual(
            registry.get_sample_value(
                "sglang:pd_prefill_decode_ahead_tokens_total",
                {"pp_rank": "0", "case": PrefixGapCase.DECODE_AHEAD_PULL.value},
            ),
            6912,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:pd_prefill_decode_prefix_tokens_total", {"pp_rank": "0"}
            ),
            7936 + 2048,
        )
        self.assertIn("decode_ahead_pull=1", self.stats.log_line())

    def test_raiden_report_overrides_the_bootstrap_prefix(self):
        collector, registry = _collector()
        req = _Req(decode_prefix_len=0, prompt_len=10000)
        pd_prefix_pull_metrics._decode_reports[req] = DecodePrefixReport(
            prefix_len=7936, pullable_len=6144
        )
        self._admit(req, 1024)
        pd_prefix_pull_metrics.flush_prefill_stats(None, _reporter(collector))
        self.assertEqual(
            registry.get_sample_value(
                "sglang:pd_prefill_decode_pullable_ahead_tokens_total",
                {"pp_rank": "0"},
            ),
            5120,
        )
        self.assertNotIn(req, pd_prefix_pull_metrics._decode_reports)

    def test_pending_bootstrap_is_not_counted(self):
        req = _Req(decode_prefix_len=7936, prompt_len=10000)
        req.pending_bootstrap = True
        self._admit(req, 0)
        self.assertFalse(self.stats.reqs)


class TestDecodeHooks(CustomTestCase):
    def setUp(self):
        self.stats = DecodeWindowStats()
        patcher = mock.patch.object(pd_prefix_pull_metrics, "_decode_stats", self.stats)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _queue(self, collector):
        return SimpleNamespace(
            scheduler=SimpleNamespace(metrics_reporter=_reporter(collector))
        )

    def test_counted_only_when_the_prealloc_sends_the_matched_prefix(self):
        collector, registry = _collector()
        queue = self._queue(collector)
        sent, host_received, unmatched = _Req(), _Req(), _Req()
        for req in (sent, host_received):
            pd_prefix_pull_metrics.note_decode_match(
                req, decode_prefix_len=1024, pullable_len=512
            )
        pd_prefix_pull_metrics.count_decode_prealloc(
            None, queue, SimpleNamespace(req=sent), decode_prefix_len=1024
        )
        pd_prefix_pull_metrics.count_decode_prealloc(
            None, queue, SimpleNamespace(req=host_received), decode_prefix_len=0
        )
        pd_prefix_pull_metrics.count_decode_prealloc(
            None, queue, SimpleNamespace(req=unmatched), decode_prefix_len=0
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:pd_decode_prefix_window_reqs_total",
                {"pp_rank": "0", "case": DecodeWindowCase.PARTIAL.value},
            ),
            1,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:pd_decode_pullable_tokens_total", {"pp_rank": "0"}
            ),
            512,
        )
        self.assertEqual(sum(self.stats.reqs.values()), 1)
        self.assertIn("pullable_tokens=512 (50.0%)", self.stats.log_line())


class TestHookRegistration(CustomTestCase):
    def test_hooks_registered_and_targets_resolve(self):
        for target, hook, hook_type in pd_prefix_pull_metrics.hook_specs():
            self.assertIn((target, hook, hook_type), HOOKS)
            pkgutil.resolve_name(target)


if __name__ == "__main__":
    unittest.main()
