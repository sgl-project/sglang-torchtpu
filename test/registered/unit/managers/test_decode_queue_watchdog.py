"""A blocked decode queue poll is watchdog work even without a model batch."""

import signal
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import decode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import invariant_checker
from sglang.srt.utils import watchdog as watchdog_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _EndWindow(Exception):
    pass


class _Clock:
    def __init__(self):
        self.elapsed = 0.0

    def perf_counter(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds
        if self.elapsed > 10:
            raise _EndWindow()


class TestDecodeQueueWatchdog(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.is_initializing = False
        self.scheduler.cur_batch_for_debug = None
        self.scheduler.forward_ct = 12
        self.scheduler._decode_queue_watchdog_active = False
        self.scheduler._decode_queue_watchdog_counter = 0
        self.scheduler.enable_decode_hicache = False
        self.scheduler.enable_hisparse = False
        self.scheduler.scheduler_stage_metrics = None
        self.scheduler.waiting_queue = []
        self.scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            resume_retracted_reqs=lambda: [],
            retracted_queue=[],
            pop_preallocated=lambda: ([], []),
        )
        self.scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            resolve_deferred_releases=lambda: None,
            pop_transferred=lambda: [],
            extend=lambda reqs: None,
        )
        self.config = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
            disaggregation_decode_host_receive_threshold=0,
        )
        self.scheduler.invariant_checker = Mock()
        self.scheduler.pool_stats_observer = Mock()
        with patch.object(
            invariant_checker,
            "WatchdogRaw",
            side_effect=lambda **kw: SimpleNamespace(**kw),
        ):
            self.watchdog = invariant_checker.create_scheduler_watchdog(
                self.scheduler, watchdog_timeout=2.0
            )
        self.watchdog.parent_process = SimpleNamespace(send_signal=Mock())

    def poll(self):
        with patch.object(decode, "get_disagg", return_value=self.config):
            self.scheduler.process_decode_queue()

    def test_blocked_poll_triggers_without_a_model_batch(self):
        def blocked_poll():
            clock = _Clock()
            with (
                patch.object(watchdog_module.time, "perf_counter", clock.perf_counter),
                patch.object(watchdog_module.time, "sleep", clock.sleep),
                patch.object(watchdog_module, "pyspy_dump_schedulers"),
            ):
                try:
                    watchdog_module.WatchdogRaw._watchdog_once(self.watchdog)
                except _EndWindow:
                    pass

        self.scheduler.disagg_decode_transfer_queue.resolve_deferred_releases = (
            blocked_poll
        )
        self.poll()

        self.watchdog.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        self.scheduler.pool_stats_observer.get_pool_stats.assert_not_called()
        self.assertEqual(self.scheduler.forward_ct, 12)
        self.assertFalse(self.watchdog.is_active())

    def test_returning_polls_make_progress_without_model_forwards(self):
        counters = []

        def returning_poll():
            self.assertTrue(self.watchdog.is_active())
            counters.append(self.watchdog.get_counter())

        self.scheduler.disagg_decode_transfer_queue.resolve_deferred_releases = (
            returning_poll
        )
        for _ in range(20):
            self.poll()
            self.assertFalse(self.watchdog.is_active())
        self.assertEqual(len(set(counters)), 20)
        self.assertGreater(self.watchdog.get_counter(), counters[-1])
        self.assertEqual(self.scheduler.forward_ct, 12)

    def test_poll_entry_advances_before_arming(self):
        prior = self.watchdog.get_counter()

        def entering_poll():
            self.assertTrue(self.watchdog.is_active())
            self.assertGreater(self.watchdog.get_counter(), prior)

        self.scheduler.disagg_decode_transfer_queue.resolve_deferred_releases = (
            entering_poll
        )
        self.poll()

    def test_poll_exception_is_preserved_and_disarms_watchdog(self):
        def failed_poll():
            self.assertTrue(self.watchdog.is_active())
            raise RuntimeError("peer failure")

        self.scheduler.disagg_decode_transfer_queue.resolve_deferred_releases = (
            failed_poll
        )
        with self.assertRaisesRegex(RuntimeError, "peer failure"):
            self.poll()
        self.assertFalse(self.watchdog.is_active())

    def test_idle_scheduler_remains_unmonitored(self):
        self.assertFalse(self.watchdog.is_active())
        self.poll()
        self.assertFalse(self.watchdog.is_active())

    def test_model_forward_and_initialization_monitoring_are_preserved(self):
        self.scheduler.cur_batch_for_debug = object()
        self.assertTrue(self.watchdog.is_active())
        prior = self.watchdog.get_counter()
        self.scheduler.forward_ct += 1
        self.assertGreater(self.watchdog.get_counter(), prior)
        self.scheduler.cur_batch_for_debug = None
        self.scheduler.is_initializing = True
        self.assertTrue(self.watchdog.is_active())

    def test_diagnostic_tolerates_batch_disappearing(self):
        self.assertIn("no active model batch", self.watchdog.dump_info())
        self.scheduler.pool_stats_observer.get_pool_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
