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

    def test_arm_and_get_recompilation_counters_flush(self):
        from sglang_tpu.compilation.recompilation_checker import (
            MAX_RECORDED_EVENTS,
            arm_recompilation_checker,
            get_recompilation_counters,
        )

        cache = _FakeCache([500])
        checker = _checker(cache)
        with patch.object(
            recompilation_checker,
            "_shared_recompilation_checker",
            return_value=checker,
        ):
            # Before baseline, counters are all zero.
            self.assertEqual(
                get_recompilation_counters(flush=True)["tpu_serving_compiles"], 0
            )
            self.assertEqual(
                get_recompilation_counters(flush=True)["tpu_heavy_serving_compiles"], 0
            )
            self.assertEqual(
                get_recompilation_counters(flush=True)["tpu_eager_serving_compiles"], 0
            )
            armed = arm_recompilation_checker(reset_counters=True)
            self.assertIs(armed, checker)
            cache.compile(0, 120, 45)
            # Without flush, counters have not yet read the new misses.
            self.assertEqual(
                get_recompilation_counters(flush=False)["tpu_serving_compiles"], 0
            )
            snapshot = get_recompilation_counters(
                flush=True, context="Model EXTEND real=q9,b1 pad=q16,b64"
            )
            self.assertEqual(snapshot["tpu_serving_compiles"], 2)
            # Event total_ms is 165 < warn_ms (1000 in _checker), so heavy is False.
            self.assertEqual(snapshot["tpu_heavy_serving_compiles"], 0)
            self.assertEqual(snapshot["tpu_eager_serving_compiles"], 2)
            self.assertEqual(snapshot["tpu_serving_cache_loads"], 1)
            self.assertEqual(snapshot["tpu_serving_compile_ms"], 165)
            self.assertEqual(snapshot["tpu_worst_compile_ms"], 120)
            self.assertEqual(len(snapshot["tpu_recompile_events"]), 1)
            self.assertEqual(
                snapshot["tpu_recompile_events"][0]["context"],
                "Model EXTEND real=q9,b1 pad=q16,b64",
            )
            self.assertEqual(
                snapshot["tpu_recompile_events"][0]["durations_ms"], [120, 45]
            )
            self.assertEqual(snapshot["tpu_recompile_events"][0]["load_count"], 1)

            # Multiple cheap compiles whose sum >= warn_ms (600 + 600 >= 1000)
            # mark event.heavy=True for logging, and increment tpu_eager_serving_compiles
            # (4 total) while per-program tpu_heavy_serving_compiles stays 0.
            cache.compile(600, 600)
            ev = checker.check(context_fn=_context)
            self.assertTrue(ev.heavy)
            self.assertEqual(
                get_recompilation_counters(flush=False)["tpu_serving_compiles"], 4
            )
            self.assertEqual(
                get_recompilation_counters(flush=False)["tpu_heavy_serving_compiles"], 0
            )
            self.assertEqual(
                get_recompilation_counters(flush=False)["tpu_eager_serving_compiles"], 4
            )

            # An individual compile >= warn_ms=1000 increments tpu_heavy_serving_compiles.
            cache.compile(1500)
            checker.check(context_fn=_context)
            self.assertEqual(
                get_recompilation_counters(flush=False)["tpu_heavy_serving_compiles"], 1
            )
            self.assertEqual(
                get_recompilation_counters(flush=False)["tpu_eager_serving_compiles"], 4
            )

            # overwrite_baseline=False preserves existing baseline and counters.
            arm_recompilation_checker(reset_counters=False, overwrite_baseline=False)
            self.assertEqual(
                get_recompilation_counters(flush=False)["tpu_serving_compiles"], 5
            )

            # Bounded events list caps at MAX_RECORDED_EVENTS.
            for _ in range(MAX_RECORDED_EVENTS + 10):
                cache.compile(10)
                checker.check(context_fn=_context)
            self.assertEqual(len(checker.events), MAX_RECORDED_EVENTS)

            # reset_counters clears all serving totals and events.
            checker.reset_counters()
            self.assertEqual(checker.serving_compiles, 0)
            self.assertEqual(checker.heavy_serving_compiles, 0)
            self.assertEqual(checker.eager_serving_compiles, 0)
            self.assertEqual(checker.serving_cache_loads, 0)
            self.assertEqual(checker.serving_compile_ms, 0)
            self.assertEqual(checker.worst_compile_ms, 0)
            self.assertEqual(checker.events, [])


class TestRecompilationGateHelpers(CustomTestCase):
    def test_extract_and_diff_snapshots(self):
        from sglang.test.recompilation_gate import (
            diff_recompilation_snapshots,
            extract_recompilation_states,
        )

        before_payload = {
            "internal_state": [
                {
                    "tpu_serving_compiles": 1,
                    "tpu_heavy_serving_compiles": 0,
                    "tpu_eager_serving_compiles": 1,
                    "tpu_serving_cache_loads": 0,
                    "tpu_serving_compile_ms": 50,
                    "tpu_worst_compile_ms": 50,
                    "tpu_recompile_events": [
                        {
                            "context": "warmup",
                            "compile_count": 1,
                            "load_count": 0,
                            "durations_ms": [50],
                        }
                    ],
                }
            ]
        }
        after_payload = {
            "internal_state": [
                {
                    "tpu_serving_compiles": 4,
                    "tpu_heavy_serving_compiles": 2,
                    "tpu_eager_serving_compiles": 2,
                    "tpu_serving_cache_loads": 1,
                    "tpu_serving_compile_ms": 365,
                    "tpu_worst_compile_ms": 200,
                    "tpu_recompile_events": [
                        {
                            "context": "warmup",
                            "compile_count": 1,
                            "load_count": 0,
                            "durations_ms": [50],
                        },
                        {
                            "context": "Model DECODE pad=q4,b4",
                            "compile_count": 3,
                            "load_count": 1,
                            "durations_ms": [200, 100, 15],
                        },
                    ],
                }
            ]
        }
        before = extract_recompilation_states(before_payload)
        after = extract_recompilation_states(after_payload)
        deltas = diff_recompilation_snapshots(before, after)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["tpu_serving_compiles"], 3)
        self.assertEqual(deltas[0]["tpu_heavy_serving_compiles"], 2)
        self.assertEqual(deltas[0]["tpu_eager_serving_compiles"], 1)
        self.assertEqual(deltas[0]["tpu_serving_cache_loads"], 1)
        self.assertEqual(deltas[0]["tpu_serving_compile_ms"], 315)
        self.assertEqual(deltas[0]["tpu_worst_compile_ms"], 200)
        self.assertEqual(len(deltas[0]["tpu_recompile_events"]), 1)
        self.assertEqual(
            deltas[0]["tpu_recompile_events"][0]["context"],
            "Model DECODE pad=q4,b4",
        )

    def test_assert_no_serving_recompilation_and_gate_runner(self):
        from sglang.test import recompilation_gate
        from sglang.test.recompilation_gate import (
            assert_no_serving_recompilation,
            recompilation_gate_runner,
        )

        clean_state = {
            "internal_state": [
                {
                    "tpu_serving_compiles": 0,
                    "tpu_heavy_serving_compiles": 0,
                    "tpu_eager_serving_compiles": 0,
                    "tpu_serving_cache_loads": 0,
                    "tpu_serving_compile_ms": 0,
                    "tpu_worst_compile_ms": 0,
                    "tpu_recompile_events": [],
                }
            ]
        }
        eager_only_state = {
            "internal_state": [
                {
                    "tpu_serving_compiles": 1,
                    "tpu_heavy_serving_compiles": 0,
                    "tpu_eager_serving_compiles": 1,
                    "tpu_serving_cache_loads": 0,
                    "tpu_serving_compile_ms": 8,
                    "tpu_worst_compile_ms": 8,
                    "tpu_recompile_events": [
                        {
                            "context": "Model DECODE pad=q4,b4",
                            "compile_count": 1,
                            "load_count": 0,
                            "durations_ms": [8],
                            "heavy": False,
                            "sites": [
                                {
                                    "site": "aten.slice.Tensor((4, 10):float32) @ sglang_tpu/sampler.py:42",
                                    "count": 1,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        dirty_state = {
            "internal_state": [
                {
                    "tpu_serving_compiles": 1,
                    "tpu_heavy_serving_compiles": 1,
                    "tpu_eager_serving_compiles": 0,
                    "tpu_serving_cache_loads": 0,
                    "tpu_serving_compile_ms": 1400,
                    "tpu_worst_compile_ms": 1400,
                    "tpu_recompile_events": [
                        {
                            "context": "Model EXTEND pad=q128,b1",
                            "compile_count": 1,
                            "load_count": 0,
                            "durations_ms": [1400],
                            "heavy": True,
                        }
                    ],
                }
            ]
        }

        with patch.object(recompilation_gate, "_get_json", return_value=clean_state):
            assert_no_serving_recompilation("http://127.0.0.1:30000")
            runner = recompilation_gate_runner(run_workload=False)
            res = runner("http://127.0.0.1:30000")
            self.assertTrue(res["_passed"])
            self.assertEqual(res["score"], 1.0)

        with patch.object(
            recompilation_gate, "_get_json", return_value=eager_only_state
        ):
            # Default only_heavy=False fails on eager op recompilations and surfaces sites.
            with self.assertRaisesRegex(
                AssertionError,
                r"tpu_eager_serving_compiles=1.*sites=\[1x aten\.slice\.Tensor",
            ):
                assert_no_serving_recompilation("http://127.0.0.1:30000")
            runner = recompilation_gate_runner(run_workload=False)
            res = runner("http://127.0.0.1:30000")
            self.assertFalse(res["_passed"])
            self.assertEqual(res["score"], 0.0)
            self.assertTrue(
                any("tpu_eager_serving_compiles=1" in f for f in res["_failures"])
            )
            # Explicit only_heavy=True opt-in ignores sub-threshold eager compiles.
            assert_no_serving_recompilation("http://127.0.0.1:30000", only_heavy=True)

        with patch.object(recompilation_gate, "_get_json", return_value=dirty_state):
            with self.assertRaisesRegex(AssertionError, "Model EXTEND pad=q128,b1"):
                assert_no_serving_recompilation("http://127.0.0.1:30000")
            with self.assertRaisesRegex(AssertionError, "tpu_heavy_serving_compiles=1"):
                assert_no_serving_recompilation(
                    "http://127.0.0.1:30000", only_heavy=True
                )
            runner = recompilation_gate_runner(run_workload=False)
            res = runner("http://127.0.0.1:30000")
            self.assertFalse(res["_passed"])
            self.assertEqual(res["score"], 0.0)
            self.assertTrue(
                any("tpu_serving_compiles=1" in f for f in res["_failures"])
            )

    def test_model_config_builds_fail_on_recompile_gate(self):
        from sglang.test.tpu.evals.model_config import build_gates

        gates = build_gates(
            {
                "model": "openai/gpt-oss-120b",
                "test_name": "gpt-oss-120b",
                "gates": [
                    {
                        "dataset": "gsm8k",
                        "baseline": 0.8,
                        "atol": 0.03,
                        "fail_on_recompile": True,
                    }
                ],
            }
        )
        self.assertEqual(len(gates), 1)
        self.assertTrue(gates[0].fail_on_recompile)

    def test_platform_and_internal_state_expose_recompilation_counters(self):
        from sglang.srt.managers.io_struct import GetInternalStateReq
        from sglang.srt.managers.scheduler_components.internal_state import (
            SchedulerInternalStateController,
        )
        from sglang.srt.runtime_context import get_context
        from sglang_tpu.compilation.recompilation_checker import (
            arm_recompilation_checker,
        )
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        cache = _FakeCache([400])
        checker = _checker(cache)

        scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(
                last_gen_throughput=1.0,
                spec_total_num_forward_ct=0,
                spec_total_num_accept_tokens=0,
                step_time_dict={},
            ),
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(weight_load_mem_usage=1.0, device="tpu"),
                graph_memory_usage=None,
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(mem_usage=3.0)
            ),
            startup_available_gpu_memory_gb=4.0,
            startup_time=1.0,
            max_total_num_tokens=100,
            swa_tokens_per_layer=None,
            max_running_requests=8,
            spec_algorithm=SimpleNamespace(
                is_none=lambda: True,
                is_dspark=lambda: False,
            ),
            draft_worker=None,
            rust_server=None,
        )
        controller = SchedulerInternalStateController(scheduler, record_step_time=False)
        tpu_platform = TpuSRTPlatform()

        with (
            patch.object(
                recompilation_checker,
                "_shared_recompilation_checker",
                return_value=checker,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.internal_state.current_platform",
                tpu_platform,
            ),
            patch("sglang_tpu.tpu_platform.torch.compiler.set_stance"),
            get_context().override_server_args(
                enable_torch_compile=True, speculative_algorithm=None
            ) as server_args,
        ):
            scheduler.server_args = server_args
            arm_recompilation_checker(reset_counters=True, overwrite_baseline=True)
            # on_warmup_complete must not overwrite an existing baseline.
            tpu_platform.on_warmup_complete()
            cache.compile(1250)
            # Bracketed forward check records the serving compile.
            checker.check(context_fn=_context)
            # Post-forward unbracketed eager op (e.g. TpuSampler or FutureMap after
            # the final forward step) is flushed when /get_internal_state is queried.
            cache.compile(15)
            with envs.SGLANG_TPU_CHECK_RECOMPILATION.override(True):
                state = controller.get_internal_state(
                    recv_req=GetInternalStateReq()
                ).internal_state
            self.assertEqual(state["tpu_serving_compiles"], 2)
            self.assertEqual(state["tpu_heavy_serving_compiles"], 1)
            self.assertEqual(state["tpu_eager_serving_compiles"], 1)
            self.assertEqual(state["tpu_serving_cache_loads"], 0)
            self.assertEqual(state["tpu_serving_compile_ms"], 1265)
            self.assertEqual(state["tpu_worst_compile_ms"], 1250)
            self.assertEqual(len(state["tpu_recompile_events"]), 2)
            self.assertEqual(
                state["tpu_recompile_events"][1]["context"], "get_internal_state"
            )
            self.assertEqual(state["tpu_recompile_events"][1]["durations_ms"], [15])

            cache.compile(1500)
            with envs.SGLANG_TPU_CHECK_RECOMPILATION.override(False):
                state_flushed = controller.get_internal_state(
                    recv_req=GetInternalStateReq()
                ).internal_state
            self.assertEqual(state_flushed["tpu_serving_compiles"], 3)
            self.assertEqual(state_flushed["tpu_heavy_serving_compiles"], 2)
            self.assertEqual(state_flushed["tpu_eager_serving_compiles"], 1)

    def test_accuracy_test_runner_fail_on_recompile(self):
        import contextlib

        from sglang.test.accuracy_test_runner import (
            AccuracyTestParams,
            run_accuracy_test,
        )

        params = AccuracyTestParams(
            dataset="gsm8k",
            baseline_accuracy=0.80,
            atol=0.03,
            fail_on_recompile=True,
            runner=lambda _url: {"accuracy": 0.90},
        )
        with (
            patch(
                "sglang.test.accuracy_test_runner.served",
                return_value=contextlib.nullcontext("http://127.0.0.1:30000"),
            ),
            patch(
                "sglang.test.recompilation_gate.check_no_serving_recompilation",
                return_value=["shard 0: tpu_serving_compiles=1 (expected 0)"],
            ),
        ):
            res = run_accuracy_test(
                SimpleNamespace(
                    model_path="openai/gpt-oss-120b",
                    variant="bf16",
                    launcher=None,
                ),
                params,
                base_url="http://127.0.0.1:30000",
            )
        self.assertFalse(res.passed)
        self.assertIn("tpu_serving_compiles=1", res.error)


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


if __name__ == "__main__":
    unittest.main()
