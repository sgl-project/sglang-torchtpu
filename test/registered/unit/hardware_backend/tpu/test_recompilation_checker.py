"""Unit tests for the TPU serving-time recompilation checker."""

import datetime
import random
import re
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import recompilation_checker
from sglang_tpu.compilation.recompilation_checker import (
    RecompilationChecker,
    _build_event,
    _format_event,
    _shared_recompilation_checker,
    install_recompilation_checker,
    maybe_enable_site_tracing,
    snapshot_recompilation_baseline,
)

# Pure logic over injected cache readers, no torch_tpu needed.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOGGER = recompilation_checker.logger.name

# The shape log-based checks parse: a bracketed context followed by the
# durations list. Drift here breaks them silently.
_PARSER_RE = re.compile(
    r".*\[recompilation checker\] \d+ (?:new|cheap) XLA (?:program|compile)\(s\)"
    r".*\[(?P<context>[^\]]*)\]: durations_ms=\[(?P<durations>[0-9, ]*)\]"
)


class _FakeCache:
    """Stand-in for the torch.tpu cache-stats pybind."""

    def __init__(self, durations_ms):
        self.durations_ms = list(durations_ms)
        self.misses = len(self.durations_ms)
        self.stats_reads = 0
        self.syncs = 0

    def sync(self):
        self.syncs += 1

    def compile(self, *durations_ms):
        self.durations_ms.extend(durations_ms)
        self.misses += len(durations_ms)

    def read_misses(self):
        return self.misses

    def read_stats(self):
        self.stats_reads += 1
        entries = [
            SimpleNamespace(compilation_duration=datetime.timedelta(milliseconds=ms))
            for ms in self.durations_ms
        ]
        return SimpleNamespace(per_entry_stats=entries)


def _checker(cache, warn_ms=1000, clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    return RecompilationChecker(
        read_misses=cache.read_misses,
        read_stats=cache.read_stats,
        sync_device=cache.sync,
        warn_ms=warn_ms,
        **kwargs,
    )


def _context():
    return "Model DECODE"


class TestRecompilationChecker(CustomTestCase):
    def test_site_keeps_plugin_and_core_frames(self):
        frames = [
            traceback.FrameSummary("/venv/torch/dispatch.py", 1, "dispatch"),
            traceback.FrameSummary("/repo/python/sglang/srt/runner.py", 20, "run"),
            traceback.FrameSummary("/repo/sglang_tpu/sampler.py", 30, "sample"),
            traceback.FrameSummary(recompilation_checker.__file__, 40, "trace"),
        ]
        with patch.object(
            recompilation_checker.traceback, "extract_stack", return_value=frames
        ):
            site = recompilation_checker._describe_site("aten.add", (), {})
        self.assertEqual(
            site,
            "aten.add() @ sglang_tpu/sampler.py:30 <- sglang/srt/runner.py:20",
        )

    def test_baseline_entries_are_not_reported(self):
        cache = _FakeCache([500, 500, 30000])
        checker = _checker(cache)
        checker.snapshot_baseline()
        self.assertIsNone(checker.check(context_fn=_context))

    def test_check_before_baseline_is_noop(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        cache.compile(30000)
        self.assertIsNone(checker.check(context_fn=_context))
        self.assertEqual(cache.stats_reads, 0)

    def test_steady_state_reads_only_the_miss_counter(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        reads, syncs = cache.stats_reads, cache.syncs
        contexts = []
        for _ in range(100):
            checker.check(context_fn=lambda: contexts.append(1))
        self.assertEqual(cache.stats_reads, reads)
        self.assertEqual(cache.syncs, syncs)
        self.assertEqual(contexts, [])

    def test_miss_without_new_entry_rearms_the_gate(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        # A tier-1 miss whose entry was evicted before the check.
        cache.misses += 1
        self.assertIsNone(checker.check(context_fn=_context))
        reads = cache.stats_reads
        checker.check(context_fn=_context)
        self.assertEqual(cache.stats_reads, reads)

    def test_below_threshold_event_is_cheap(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(12, 45)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        self.assertFalse(event.heavy)
        self.assertEqual(event.durations_ms, (45, 12))
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])

    def test_sum_over_threshold_is_heavy(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(600, 600)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        self.assertTrue(event.heavy)
        self.assertEqual(event.total_ms, 1200)
        self.assertEqual([r.levelname for r in logs.records], ["WARNING"])

    def test_duplicate_duration_is_detected(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(500)
        event = checker.check(context_fn=_context)
        self.assertEqual(event.durations_ms, (500,))

    def test_entry_order_does_not_matter(self):
        cache = _FakeCache([1, 2, 3, 4, 5])
        checker = _checker(cache)
        checker.snapshot_baseline()
        random.Random(0).shuffle(cache.durations_ms)
        cache.misses += 1
        self.assertIsNone(checker.check(context_fn=_context))

    def test_stats_are_read_only_after_a_device_sync(self):
        """torch_tpu fills the duration from a compile thread after the entry exists."""
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        self.assertEqual(cache.syncs, 1)
        cache.compile(30)
        order = []
        cache.sync = lambda: order.append("sync")
        real_read = cache.read_stats
        cache.read_stats = lambda: (order.append("read"), real_read())[1]
        checker._sync_device = cache.sync
        checker._read_stats = cache.read_stats
        checker.check(context_fn=_context)
        self.assertEqual(order, ["sync", "read"])

    def test_persistent_cache_loads_are_counted_not_timed(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(0, 0, 17400)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        self.assertEqual(event.load_count, 2)
        self.assertEqual(event.durations_ms, (17400,))
        self.assertTrue(event.heavy)
        self.assertIn("persistent compile cache", logs.output[0])
        self.assertEqual(checker.serving_compiles, 1)

    def test_load_only_event_is_not_a_compile_line(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(0)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        self.assertEqual(event.durations_ms, ())
        self.assertFalse(event.heavy)
        self.assertIsNone(_PARSER_RE.match(logs.output[0]))
        self.assertEqual(checker.worst_compile_ms, 0)

    def test_counters_accumulate_across_events(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(12, 45)
        checker.check(context_fn=_context)
        cache.compile(17400)
        checker.check(context_fn=_context)
        self.assertEqual(checker.serving_compiles, 3)
        self.assertEqual(checker.serving_compile_ms, 12 + 45 + 17400)
        self.assertEqual(checker.worst_compile_ms, 17400)

    def test_rebaseline_absorbs_deferred_draft_precompile(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(30000)
        checker.snapshot_baseline()
        self.assertIsNone(checker.check(context_fn=_context))

    def test_context_is_built_only_when_an_event_fires(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        cache.compile(7)
        event = checker.check(context_fn=lambda: "Model EXTEND real=q9,b1 pad=q16,b64")
        self.assertEqual(event.context, "Model EXTEND real=q9,b1 pad=q16,b64")


class TestWindowSummary(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Short window so the arithmetic in these cases stays readable.
        patcher = patch.object(recompilation_checker, "_SUMMARY_WINDOW_SEC", 60.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_summary_reports_compile_share_of_wall_once_per_window(self):
        """A trickle of sub-threshold compiles is only visible as a share of wall."""
        now = [1000.0]
        cache = _FakeCache([500])
        checker = _checker(cache, clock=lambda: now[0])
        checker.snapshot_baseline()
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            for _ in range(5):
                now[0] += 10
                cache.compile(700, 200)
                checker.check(context_fn=_context)
            now[0] += 10  # window closes at 60 s
            cache.compile(100)
            checker.check(context_fn=_context)
        summaries = [line for line in logs.output if "of wall" in line]
        self.assertEqual(len(summaries), 1)
        # 6 forwards, 5 * 900 + 100 = 4600 ms over 60 s of wall.
        self.assertIn("6 forward(s) compiled, 4600 ms", summaries[0])
        self.assertIn("(7.7% of wall)", summaries[0])
        self.assertTrue(summaries[0].startswith("INFO:"))
        # The window resets, so the next event alone does not summarize.
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            now[0] += 1
            cache.compile(50)
            checker.check(context_fn=_context)
        self.assertFalse([line for line in logs.output if "of wall" in line])

    def test_burst_is_summarized_even_after_compiles_stop(self):
        """Idle time after a burst must not dilute the burst's share."""
        now = [0.0]
        cache = _FakeCache([500])
        checker = _checker(cache, clock=lambda: now[0])
        checker.snapshot_baseline()
        for _ in range(10):
            now[0] += 1
            cache.compile(800)  # 8 s of compile inside the first 10 s
            checker.check(context_fn=_context)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            for _ in range(10):
                now[0] += 10  # quiet forwards carry the window past 60 s
                checker.check(context_fn=_context)
        summaries = [line for line in logs.output if "of wall" in line]
        self.assertEqual(len(summaries), 1)
        self.assertIn("10 forward(s) compiled, 8000 ms", summaries[0])
        self.assertIn("(13.3% of wall)", summaries[0])

    def test_compile_spanning_the_deadline_stays_in_its_own_window(self):
        """A compile is charged to the window it ran in, so never past 100%."""
        now = [0.0]
        cache = _FakeCache([500])
        checker = _checker(cache, clock=lambda: now[0])
        checker.snapshot_baseline()
        now[0] += 70  # one 70 s compile, so the forward returns past the deadline
        cache.compile(70000)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            checker.check(context_fn=_context)
        summaries = [line for line in logs.output if "of wall" in line]
        self.assertEqual(len(summaries), 1)
        self.assertIn("(100.0% of wall)", summaries[0])

    def test_quiet_server_logs_no_summary(self):
        now = [0.0]
        cache = _FakeCache([500])
        checker = _checker(cache, clock=lambda: now[0])
        checker.snapshot_baseline()
        now[0] += 3600
        with self.assertNoLogs(_LOGGER, level="INFO"):
            for _ in range(10):
                checker.check(context_fn=_context)


class TestEventFormat(CustomTestCase):
    def test_log_lines_keep_the_parsed_shape(self):
        for new_us in ((17_400_000, 0), (12_000, 45_000)):
            event = _build_event(
                new_us=new_us,
                context="Model EXTEND real=q9,b1 pad=q16,b64",
                warn_ms=1000,
            )
            match = _PARSER_RE.match(_format_event(event, warn_ms=1000))
            self.assertIsNotNone(match, _format_event(event, warn_ms=1000))
            self.assertEqual(match.group("context"), event.context)
            parsed = [int(d) for d in match.group("durations").split(",")]
            self.assertEqual(tuple(parsed), event.durations_ms)


class TestSharedChecker(CustomTestCase):
    def setUp(self):
        super().setUp()
        _shared_recompilation_checker.cache_clear()
        self.addCleanup(_shared_recompilation_checker.cache_clear)

    def test_returns_none_without_the_cache_stats_pybind(self):
        fake_torch = SimpleNamespace(tpu=SimpleNamespace())
        with patch.object(recompilation_checker, "torch", fake_torch):
            self.assertIsNone(_shared_recompilation_checker())

    def test_install_is_a_noop_unless_opted_in(self):
        policy = MagicMock()
        with envs.SGLANG_TPU_CHECK_RECOMPILATION.override(False):
            install_recompilation_checker(policies=(policy,), model_label="M")
        policy.set_recompilation_checker.assert_not_called()

    def test_install_baselines_model_warmup_and_detects_first_serving_compile(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        target = SimpleNamespace(set_recompilation_checker=MagicMock())
        draft = SimpleNamespace(set_recompilation_checker=MagicMock())
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True),
            patch.object(
                recompilation_checker,
                "_shared_recompilation_checker",
                return_value=checker,
            ),
        ):
            install_recompilation_checker(policies=(target,), model_label="target M")
            cache.compile(30000)  # Deferred draft warmup.
            install_recompilation_checker(policies=(draft,), model_label="draft M")
            for policy, label in ((target, "target M"), (draft, "draft M")):
                policy.set_recompilation_checker.assert_called_once_with(
                    checker, model_label=label
                )
            self.assertIsNone(checker.check(context_fn=_context))
            cache.compile(50)  # The very first serving forward is monitored.
            self.assertEqual(checker.check(context_fn=_context).durations_ms, (50,))
            self.assertEqual(checker.serving_compiles, 1)

    def _assert_late_warmup_is_excluded(self, warmup):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True),
            patch.object(
                recompilation_checker,
                "_shared_recompilation_checker",
                return_value=checker,
            ),
        ):
            warmup(cache)
        self.assertGreater(cache.misses, 1)
        self.assertIsNone(checker.check(context_fn=_context))
        self.assertEqual(checker.serving_compiles, 0)
        cache.compile(50)
        self.assertEqual(checker.check(context_fn=_context).durations_ms, (50,))

    def test_speculative_op_warmup_refreshes_shared_baseline(self):
        from sglang_tpu import spec_op_precompile

        def warmup(cache):
            op = SimpleNamespace(
                make_warmup_args=lambda context: (),
                op=lambda: cache.compile(200),
            )
            with (
                patch.object(spec_op_precompile, "get_compiled_ops", return_value=[op]),
                patch.object(
                    spec_op_precompile.torch.accelerator,
                    "synchronize",
                    cache.sync,
                ),
            ):
                spec_op_precompile.precompile_spec_v2_ops(
                    device=SimpleNamespace(type="tpu"),
                    physical_batch_buckets=(1, 2),
                    num_verify_tokens=3,
                    vocab_size=8,
                    draft_vocab_size=8,
                    hidden_size=4,
                    hidden_dtype=spec_op_precompile.torch.float32,
                )

        self._assert_late_warmup_is_excluded(warmup)

    def test_kv_offload_warmup_refreshes_shared_baseline(self):
        from sglang_tpu.compilation import compiled_forward_shim

        def warmup(cache):
            pool = SimpleNamespace(
                device_pool=SimpleNamespace(
                    kv_buffer=[compiled_forward_shim.torch.zeros(2)]
                ),
                page_size=1,
                load_to_device_all=lambda *args: cache.compile(200),
                backup_from_device_all_layer=lambda *args: cache.compile(400),
            )
            compiled_forward_shim.precompile_kv_offload(pool)

        self._assert_late_warmup_is_excluded(warmup)

    def test_install_skips_policies_without_monitoring_support(self):
        from sglang.srt.compilation.support_torch_compile import NoopCompilePolicy

        cache = _FakeCache([])
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True),
            patch.object(
                recompilation_checker,
                "_shared_recompilation_checker",
                return_value=_checker(cache),
            ),
        ):
            install_recompilation_checker(
                policies=(NoopCompilePolicy(),), model_label="vision"
            )
        self.assertEqual(cache.stats_reads, 0)

    def test_snapshot_is_a_noop_unless_opted_in(self):
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(False),
            patch.object(
                recompilation_checker, "_shared_recompilation_checker"
            ) as shared,
        ):
            snapshot_recompilation_baseline()
        shared.assert_not_called()

    def test_snapshot_is_a_noop_without_cache_stats(self):
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True),
            patch.object(
                recompilation_checker,
                "_shared_recompilation_checker",
                return_value=None,
            ),
        ):
            snapshot_recompilation_baseline()

    def test_target_and_draft_share_one_checker(self):
        """Two checkers over the one process cache report every compile twice."""
        cache = _FakeCache([500])
        fake_torch = SimpleNamespace(
            tpu=SimpleNamespace(
                _get_cache_misses=cache.read_misses,
                _get_cache_stats=cache.read_stats,
            ),
            accelerator=SimpleNamespace(
                synchronize=cache.sync,
            ),
        )
        with patch.object(recompilation_checker, "torch", fake_torch):
            checker = _shared_recompilation_checker()
            self.assertIs(checker, _shared_recompilation_checker())
            checker.snapshot_baseline()
            cache.compile(25)
            event = _shared_recompilation_checker().check(context_fn=_context)
            self.assertEqual(event.durations_ms, (25,))
            self.assertIsNone(checker.check(context_fn=_context))


# The cache an op "compiles" into, set per test.
_OP_CACHE: list[_FakeCache] = []


@torch.library.custom_op("sglang_test::compiles_once", mutates_args=())
def _compiles_once(x: torch.Tensor) -> torch.Tensor:
    _OP_CACHE[0].compile(40)
    return x.clone()


class TestSiteTracing(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.cache = _FakeCache([500])
        _OP_CACHE[:] = [self.cache]
        self.addCleanup(_OP_CACHE.clear)
        self.checker = _checker(self.cache, warn_ms=10)
        self.checker.snapshot_baseline()

    def _enable(self):
        self.checker.enable_site_tracing()
        self.addCleanup(self.checker._site_tracer.__exit__, None, None, None)

    def test_compiling_op_is_attached_to_the_event(self):
        self._enable()
        x = torch.ones(3, 2, dtype=torch.int32)
        x + 1  # cache hit, not a site
        _compiles_once(x)
        _compiles_once(x)
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            event = self.checker.check(context_fn=_context)
        self.assertEqual(len(event.sites), 1)
        site, n = event.sites[0]
        self.assertEqual(n, 2)
        self.assertIn("sglang_test.compiles_once.default((3, 2):int32)", site)
        self.assertIn("2x sglang_test.compiles_once", logs.output[0])
        self.assertIsNotNone(_PARSER_RE.match(logs.output[0]))

    def test_sites_do_not_leak_into_the_next_event(self):
        self._enable()
        _compiles_once(torch.ones(1))
        self.checker.check(context_fn=_context)
        self.cache.compile(40)  # e.g. a compiled graph, invisible to the tracer
        event = self.checker.check(context_fn=_context)
        self.assertEqual(event.sites, ())

    def test_sites_are_none_without_tracing(self):
        _compiles_once(torch.ones(1))
        event = self.checker.check(context_fn=_context)
        self.assertIsNone(event.sites)
        self.assertNotIn("sites", _format_event(event, warn_ms=10))

    def test_enable_requires_both_flags_and_an_installed_checker(self):
        cases = ((False, True, True), (True, False, True), (True, True, False))
        for check, sites, installed in cases:
            checker = MagicMock(has_baseline=installed)
            with (
                envs.SGLANG_TPU_CHECK_RECOMPILATION.override(check),
                envs.SGLANG_TPU_CHECK_RECOMPILATION_SITES.override(sites),
                patch.object(
                    recompilation_checker,
                    "_shared_recompilation_checker",
                    return_value=checker,
                ),
            ):
                maybe_enable_site_tracing()
            checker.enable_site_tracing.assert_not_called()


class _FakeRss:
    """Stand-in for this rank's /proc RSS reader."""

    def __init__(self, bytes_=10 * (1 << 30)):
        self.bytes = bytes_

    def read(self):
        return self.bytes

    def grow(self, mib):
        self.bytes += int(mib * (1 << 20))


def _memory_checker(cache, rss, clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    return RecompilationChecker(
        read_misses=cache.read_misses,
        read_stats=cache.read_stats,
        sync_device=cache.sync,
        warn_ms=1000,
        read_rss=rss.read,
        **kwargs,
    )


class TestHostMemoryReport(CustomTestCase):
    """Issue #1429: each rank's executables cost host RAM once per rank."""

    def test_baseline_logs_this_ranks_rss(self):
        cache = _FakeCache([500])
        rss = _FakeRss(bytes_=int(2.5 * (1 << 30)))
        checker = _memory_checker(cache, rss)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            checker.snapshot_baseline()
        self.assertIn("host RSS 2.50 GiB (this rank", logs.output[0])

    def test_event_reports_rss_growth_per_new_program(self):
        cache = _FakeCache([500])
        rss = _FakeRss()
        checker = _memory_checker(cache, rss)
        checker.snapshot_baseline()
        cache.compile(300, 300)
        rss.grow(1.0)  # two new programs, 0.5 MiB each on this rank
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        self.assertEqual(event.host_rss_delta_bytes, 1 << 20)
        line = logs.output[0]
        self.assertIn("host RSS +1.0 MiB since the previous report", line)
        self.assertIn("0.5 MiB per program", line)
        self.assertIn("Each TP rank process holds its own copy", line)
        # The compile line keeps the shape log-based checks parse.
        self.assertIsNotNone(_PARSER_RE.match(line))

    def test_rss_delta_spans_the_forwards_since_the_previous_report(self):
        cache = _FakeCache([500])
        rss = _FakeRss()
        checker = _memory_checker(cache, rss)
        checker.snapshot_baseline()
        rss.grow(3.0)  # growth with no compile is charged to the next event
        for _ in range(5):
            checker.check(context_fn=_context)
        cache.compile(300)
        rss.grow(1.0)
        event = checker.check(context_fn=_context)
        self.assertEqual(event.host_rss_delta_bytes, 4 << 20)
        # A second event measures from the first, not from the baseline.
        cache.compile(300)
        rss.grow(2.0)
        self.assertEqual(
            checker.check(context_fn=_context).host_rss_delta_bytes, 2 << 20
        )

    def test_unreadable_rss_leaves_the_line_unchanged(self):
        cache = _FakeCache([500])
        checker = _memory_checker(cache, _FakeRss(bytes_=0))
        checker.snapshot_baseline()
        cache.compile(300)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        self.assertEqual(event.host_rss_delta_bytes, 0)
        self.assertNotIn("host RSS", logs.output[0])
        self.assertNotIn("This rank", logs.output[0])

    def test_peak_compile_host_memory_of_new_entries_is_reported(self):
        cache = _FakeCache([500])
        rss = _FakeRss()
        checker = _memory_checker(cache, rss)
        checker.snapshot_baseline()

        def read_stats():
            stats = cache.read_stats()
            peaks = [0] * (len(stats.per_entry_stats) - 2) + [3 << 20, 7 << 20]
            for entry, peak in zip(stats.per_entry_stats, peaks):
                entry.peak_compilation_host_memory_bytes = peak
            return stats

        checker._read_stats = read_stats
        cache.compile(300, 300)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        self.assertEqual(event.peak_compile_host_bytes, 7 << 20)
        self.assertIn("peak compile host memory 7 MiB", logs.output[0])

    def test_entries_differing_only_in_peak_memory_are_distinct_programs(self):
        cache = _FakeCache([500])
        checker = _memory_checker(cache, _FakeRss())
        checker.snapshot_baseline()
        # A new program with the same duration as a cached one is only
        # distinguishable by its peak host memory.
        cache.compile(500)
        orig = cache.read_stats

        def read_stats():
            stats = orig()
            stats.per_entry_stats[-1].peak_compilation_host_memory_bytes = 1 << 20
            return stats

        checker._read_stats = read_stats
        event = checker.check(context_fn=_context)
        self.assertEqual(event.durations_ms, (500,))

    def test_window_summary_reports_growth_since_baseline_per_program(self):
        now = [0.0]
        cache = _FakeCache([500])
        rss = _FakeRss()
        checker = _memory_checker(cache, rss, clock=lambda: now[0])
        with patch.object(recompilation_checker, "_SUMMARY_WINDOW_SEC", 60.0):
            checker.snapshot_baseline()
            for _ in range(4):
                now[0] += 10
                cache.compile(100)
                cache.misses += 1  # one program loaded from the persistent cache
                cache.durations_ms.append(0)
                rss.grow(2.0)
                checker.check(context_fn=_context)
            now[0] += 30
            with self.assertLogs(_LOGGER, level="INFO") as logs:
                checker.check(context_fn=_context)
        summaries = [line for line in logs.output if "of wall" in line]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(checker.serving_programs, 8)
        self.assertIn(
            "Host RSS of this rank +8 MiB since the baseline for 8 program(s) "
            "(1.0 MiB per program)",
            summaries[0],
        )

    def test_default_rss_reader_reads_this_process(self):
        cache = _FakeCache([500])
        checker = _checker(cache)
        checker.snapshot_baseline()
        self.assertGreater(checker._baseline_rss, 0)


def _with_peaks(cache, peaks):
    """A stats reader that assigns ``peaks`` to the last entries, in order."""
    orig = cache.read_stats

    def read_stats():
        stats = orig()
        for entry, peak in zip(stats.per_entry_stats[-len(peaks) :], peaks):
            entry.peak_compilation_host_memory_bytes = peak
        return stats

    return read_stats


class TestHostMemoryReportEdgeCases(CustomTestCase):
    """Degenerate readings must degrade the report, never the forward."""

    _TIB = 1 << 40

    def _event(self, cache, rss, *durations_ms, grow_mib=0.0, peaks=None):
        checker = _memory_checker(cache, rss)
        checker.snapshot_baseline()
        if peaks is not None:
            checker._read_stats = _with_peaks(cache, peaks)
        cache.compile(*durations_ms)
        rss.grow(grow_mib)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            event = checker.check(context_fn=_context)
        return checker, event, logs.output[0]

    def test_first_event_measures_from_the_baseline_reading(self):
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = _memory_checker(cache, rss)
        checker.snapshot_baseline()
        self.assertEqual(checker._last_rss, checker._baseline_rss)
        cache.compile(300)
        rss.grow(2.0)
        self.assertEqual(
            checker.check(context_fn=_context).host_rss_delta_bytes, 2 << 20
        )

    def test_flat_rss_omits_the_host_part_but_keeps_the_peak(self):
        _, event, line = self._event(
            _FakeCache([500]), _FakeRss(), 300, peaks=[5 << 20]
        )
        self.assertEqual(event.host_rss_delta_bytes, 0)
        self.assertNotIn("host RSS", line)
        self.assertIn("This rank: peak compile host memory 5 MiB.", line)

    def test_shrinking_rss_is_signed_and_has_no_per_program_share(self):
        _, event, line = self._event(_FakeCache([500]), _FakeRss(), 300, grow_mib=-3.0)
        self.assertEqual(event.host_rss_delta_bytes, -(3 << 20))
        self.assertIn("host RSS -3.0 MiB since the previous report", line)
        self.assertNotIn("per program", line)
        self.assertIsNotNone(_PARSER_RE.match(line))

    def test_per_program_share_divides_by_compiled_plus_loaded_programs(self):
        # One compile and one persistent-cache load: 3 MiB over 2 programs.
        _, event, line = self._event(
            _FakeCache([500]), _FakeRss(), 300, 0, grow_mib=3.0
        )
        self.assertEqual(event.load_count, 1)
        self.assertIn("1.5 MiB per program", line)

    def test_load_only_event_still_reports_host_memory(self):
        _, event, line = self._event(_FakeCache([500]), _FakeRss(), 0, grow_mib=4.0)
        self.assertEqual(event.durations_ms, ())
        self.assertIn("loaded from the persistent compile cache", line)
        self.assertIn(
            "host RSS +4.0 MiB since the previous report, 4.0 MiB per program", line
        )
        self.assertIsNone(_PARSER_RE.match(line))

    def test_small_share_rounds_to_one_decimal(self):
        _, _, line = self._event(_FakeCache([500]), _FakeRss(), 1, 2, 3, grow_mib=1.0)
        self.assertIn("0.3 MiB per program", line)

    def test_rss_unreadable_at_the_event_keeps_the_last_reading(self):
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = _memory_checker(cache, rss)
        checker.snapshot_baseline()
        base = rss.bytes
        rss.bytes = 0  # /proc read failed this time
        cache.compile(300)
        self.assertEqual(checker.check(context_fn=_context).host_rss_delta_bytes, 0)
        self.assertEqual(checker._last_rss, base)
        rss.bytes = base + (2 << 20)  # readable again: delta from the last good read
        cache.compile(300)
        self.assertEqual(
            checker.check(context_fn=_context).host_rss_delta_bytes, 2 << 20
        )

    def test_rss_unreadable_at_the_baseline_reports_no_delta_until_two_reads(self):
        cache, rss = _FakeCache([500]), _FakeRss(bytes_=0)
        checker = _memory_checker(cache, rss)
        checker.snapshot_baseline()
        rss.bytes = 10 << 30
        cache.compile(300)
        # First reading after an unreadable baseline: nothing to diff against.
        self.assertEqual(checker.check(context_fn=_context).host_rss_delta_bytes, 0)
        rss.grow(1.0)
        cache.compile(300)
        self.assertEqual(
            checker.check(context_fn=_context).host_rss_delta_bytes, 1 << 20
        )

    def test_baseline_with_an_empty_cache_and_no_rss_logs_zeros(self):
        checker = _memory_checker(_FakeCache([]), _FakeRss(bytes_=0))
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            checker.snapshot_baseline()
        self.assertIn("0 cached programs, 0.0s total compile time", logs.output[0])
        self.assertIn("host RSS 0.00 GiB", logs.output[0])

    def test_entries_without_peak_memory_key_on_duration_only(self):
        cache = _FakeCache([500])
        for peaks in ([None], []):  # None value, or attribute absent
            with self.subTest(peaks=peaks):
                checker, event, line = self._event(
                    cache, _FakeRss(), 300, peaks=peaks or None
                )
                self.assertEqual(event.durations_ms, (300,))
                self.assertEqual(event.peak_compile_host_bytes, 0)
                self.assertNotIn("peak compile host memory", line)

    def test_unusable_peak_memory_degrades_to_zero(self):
        for peak in ("n/a", object(), 3.9, "7"):
            with self.subTest(peak=peak):
                _, event, _ = self._event(
                    _FakeCache([500]), _FakeRss(), 300, peaks=[peak]
                )
                self.assertEqual(event.durations_ms, (300,))
                expected = {3.9: 3, "7": 7}.get(peak, 0)
                self.assertEqual(event.peak_compile_host_bytes, expected)

    def test_largest_peak_among_the_new_entries_wins(self):
        _, event, line = self._event(
            _FakeCache([500]), _FakeRss(), 300, 300, 300, peaks=[9 << 20, 2 << 20, 0]
        )
        self.assertEqual(event.peak_compile_host_bytes, 9 << 20)
        self.assertIn("peak compile host memory 9 MiB", line)

    def test_duplicate_keys_count_every_new_program(self):
        cache = _FakeCache([500, 500])
        checker = _memory_checker(cache, _FakeRss())
        checker._read_stats = _with_peaks(cache, [1 << 20] * 4)
        checker.snapshot_baseline()
        cache.compile(500, 500)  # same duration and peak as the baseline pair
        event = checker.check(context_fn=_context)
        self.assertEqual(event.durations_ms, (500, 500))
        self.assertEqual(checker.serving_programs, 2)

    def test_evicted_entry_replaced_by_a_same_key_entry_is_invisible(self):
        # Documented limit of the multiset identity: no stable entry id upstream.
        cache = _FakeCache([500])
        checker = _memory_checker(cache, _FakeRss())
        checker.snapshot_baseline()
        cache.misses += 1  # a miss whose entry evicted a same-key one
        self.assertIsNone(checker.check(context_fn=_context))

    def test_terabyte_values_format_without_overflow(self):
        rss = _FakeRss(bytes_=self._TIB)
        checker = _memory_checker(_FakeCache([500]), rss)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            checker.snapshot_baseline()
        self.assertIn("host RSS 1024.00 GiB", logs.output[0])
        event = _build_event(
            new_us=(300_000,),
            context="c",
            warn_ms=1000,
            host_rss_delta_bytes=2 * self._TIB,
            peak_compile_host_bytes=self._TIB + (self._TIB >> 1),
        )
        line = _format_event(event, warn_ms=1000)
        self.assertIn("host RSS +2097152.0 MiB", line)
        self.assertIn("peak compile host memory 1572864 MiB", line)

    def test_event_builder_defaults_keep_older_call_sites_working(self):
        event = _build_event(new_us=(300_000,), context="c", warn_ms=1000)
        self.assertEqual(
            (event.host_rss_delta_bytes, event.peak_compile_host_bytes), (0, 0)
        )
        self.assertEqual(recompilation_checker._format_host_memory(event), "")

    def test_host_memory_sits_between_the_compile_sentence_and_the_sites(self):
        event = _build_event(
            new_us=(300_000,),
            context="Model EXTEND real=q9,b1 pad=q16,b64",
            warn_ms=1000,
            sites=(("aten.add() @ sglang_tpu/x.py:1", 2),),
            host_rss_delta_bytes=1 << 20,
            peak_compile_host_bytes=1 << 20,
        )
        line = _format_event(event, warn_ms=1000)
        match = _PARSER_RE.match(line)
        self.assertEqual(match.group("durations"), "300")
        self.assertEqual(match.group("context"), event.context)
        compile_end = line.index("threshold).")
        host = line.index(" This rank: ")
        sites = line.index("aten.add()")
        self.assertLess(compile_end, host)
        self.assertLess(host, sites)
        self.assertIn("durations_ms=[300] (total 300 ms", line)

    def test_heavy_line_keeps_its_shape_with_host_memory(self):
        _, event, line = self._event(
            _FakeCache([500]), _FakeRss(), 5000, grow_mib=1.0, peaks=[1 << 20]
        )
        self.assertTrue(event.heavy)
        self.assertTrue(line.startswith("WARNING:"))
        match = _PARSER_RE.match(line)
        self.assertEqual(match.group("durations"), "5000")
        self.assertIn("covered by precompile. This rank:", line)


class TestHostMemorySummaryEdgeCases(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(recompilation_checker, "_SUMMARY_WINDOW_SEC", 60.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.now = [0.0]

    def _checker(self, cache, rss):
        checker = _memory_checker(cache, rss, clock=lambda: self.now[0])
        checker.snapshot_baseline()
        return checker

    def _summary(self, checker):
        self.now[0] += 61
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            checker.check(context_fn=_context)
        summaries = [line for line in logs.output if "of wall" in line]
        self.assertEqual(len(summaries), 1)
        return summaries[0]

    def test_shrinking_rss_since_baseline_has_no_per_program_share(self):
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = self._checker(cache, rss)
        cache.compile(100)
        rss.grow(-5.0)
        checker.check(context_fn=_context)
        summary = self._summary(checker)
        self.assertIn(
            "Host RSS of this rank -5 MiB since the baseline for 1 program(s);", summary
        )
        self.assertNotIn("per program", summary)

    def test_growth_with_zero_programs_divides_by_nothing(self):
        # A miss whose entry was evicted counts as a forward with no program.
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = self._checker(cache, rss)
        cache.misses += 1
        rss.grow(7.0)
        checker.check(context_fn=_context)
        # The window closes on a later event, so force one program-less event
        # into the window via the counters: no programs, but a window event.
        checker._window_events = 1
        summary = self._summary(checker)
        self.assertIn("+7 MiB since the baseline for 0 program(s);", summary)
        self.assertNotIn("per program", summary)

    def test_unreadable_rss_at_baseline_omits_host_growth(self):
        cache, rss = _FakeCache([500]), _FakeRss(bytes_=0)
        checker = self._checker(cache, rss)
        rss.bytes = 10 << 30
        cache.compile(100)
        checker.check(context_fn=_context)
        summary = self._summary(checker)
        self.assertIn("of wall", summary)
        self.assertNotIn("Host RSS", summary)

    def test_rss_becoming_unreadable_omits_host_growth(self):
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = self._checker(cache, rss)
        cache.compile(100)
        checker.check(context_fn=_context)
        rss.bytes = 0
        summary = self._summary(checker)
        self.assertNotIn("Host RSS", summary)
        self.assertTrue(summary.endswith("worst 100 ms."))

    def test_rss_growth_without_any_compile_is_not_summarized(self):
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = self._checker(cache, rss)
        rss.grow(100.0)
        self.now[0] += 61
        with self.assertNoLogs(_LOGGER, level="INFO"):
            checker.check(context_fn=_context)

    def test_summary_counts_programs_across_windows(self):
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = self._checker(cache, rss)
        cache.compile(100, 0)
        rss.grow(2.0)
        checker.check(context_fn=_context)
        self._summary(checker)
        cache.compile(100)
        rss.grow(1.0)
        checker.check(context_fn=_context)
        summary = self._summary(checker)
        self.assertIn(
            "+3 MiB since the baseline for 3 program(s) (1.0 MiB per program)", summary
        )

    def test_rebaseline_resets_the_growth_origin(self):
        cache, rss = _FakeCache([500]), _FakeRss()
        checker = self._checker(cache, rss)
        cache.compile(100)
        rss.grow(50.0)
        checker.check(context_fn=_context)
        checker.snapshot_baseline()  # a later warmup phase
        cache.compile(100)
        rss.grow(1.0)
        event = checker.check(context_fn=_context)
        self.assertEqual(event.host_rss_delta_bytes, 1 << 20)
        summary = self._summary(checker)
        self.assertIn("+1 MiB since the baseline", summary)


class TestDisabledChecker(CustomTestCase):
    def test_opt_out_reads_neither_cache_stats_nor_rss(self):
        with (
            envs.SGLANG_TPU_CHECK_RECOMPILATION.override(False),
            patch.object(recompilation_checker, "rss_bytes") as rss,
            patch.object(
                recompilation_checker, "_shared_recompilation_checker"
            ) as shared,
            self.assertNoLogs(_LOGGER, level="INFO"),
        ):
            install_recompilation_checker(policies=(MagicMock(),), model_label="M")
            snapshot_recompilation_baseline()
        rss.assert_not_called()
        shared.assert_not_called()


if __name__ == "__main__":
    unittest.main()
