"""on_idle's stalled-path load publish is wall-clock bounded.

A non-storage no-batch stall spins on_idle without sleeping, so the gate must
cap the O(queue) get_loads for both the DP-balancing writer and the load
socket. CPU-only: builds a bare Scheduler with mocked collaborators, like
test_scheduler_flush_cache.
"""

import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def _stalled_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.enable_hicache_storage = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_lmcache = False
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=False)  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s._last_stall_publish_ts = float("-inf")
        return s

    def test_spinning_stall_publishes_once_within_the_floor(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(100):
                s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)
        self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)

    def test_publishes_again_after_the_floor_elapses(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            mono.return_value = 100.0
            s.on_idle()
            mono.return_value = 100.10  # > LOAD_STALL_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)

    def test_stall_offers_the_metrics_gauges_a_publish(self):
        """A no-batch stall must reach the /metrics gauges too, or a stuck P/D
        engine reads exactly like an idle one (#1303). The reporter paces the
        publish itself; on_idle only has to offer it on the stalled path."""
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            s.on_idle()
        s.metrics_reporter.maybe_log_stall_metrics.assert_called_once_with()

    def _idle_scheduler(self):
        s = self._stalled_scheduler()
        s.enable_overlap = True
        s.batch_record_buf = [None, None]
        s.batch_record_ct = 0
        s.device_module = MagicMock()
        s.forward_ct = 0
        s.enable_unified_memory = False
        s.enable_hisparse = False
        s.scheduler_stage_metrics = MagicMock()
        s.invariant_checker = MagicMock()
        s.invariant_checker._check_all_pools.return_value = (False, [])
        s.pool_stats_observer = MagicMock()
        s.token_to_kv_pool_allocator = MagicMock()
        s.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []
        s.kv_events_publisher = MagicMock()
        s.new_token_ratio_tracker = MagicMock()
        s.maybe_sleep_on_idle = MagicMock()
        return s

    def test_overlap_records_survive_stalls_and_release_after_idle_sync(self):
        s = self._idle_scheduler()
        batch = ScheduleBatch(reqs=[], input_ids=torch.tensor([1]))
        batch_ref, tensor_ref = weakref.ref(batch), weakref.ref(batch.input_ids)
        s.record_batch_in_overlap(batch)
        del batch

        def synchronize():
            self.assertIsNotNone(batch_ref())
            self.assertIsNotNone(tensor_ref())

        s.device_module.synchronize.side_effect = synchronize
        with patch("sglang.srt.managers.scheduler.use_mlx", return_value=False):
            s.on_idle()
            self.assertIsNotNone(batch_ref())
            s.device_module.synchronize.assert_not_called()

            s.is_fully_idle.return_value = True
            s.on_idle()
            self.assertIsNone(batch_ref())
            self.assertIsNone(tensor_ref())
            s.on_idle()
        s.device_module.synchronize.assert_called_once_with()

    def test_failed_sync_preserves_overlap_records(self):
        s = self._idle_scheduler()
        s.is_fully_idle.return_value = True
        batch = ScheduleBatch(reqs=[])
        batch_ref = weakref.ref(batch)
        s.record_batch_in_overlap(batch)
        del batch
        s.device_module.synchronize.side_effect = RuntimeError("sync failed")
        with (
            patch("sglang.srt.managers.scheduler.use_mlx", return_value=False),
            self.assertRaisesRegex(RuntimeError, "sync failed"),
        ):
            s.on_idle()
        self.assertIsNotNone(batch_ref())

    def test_backends_without_overlap_records_do_not_synchronize(self):
        for enable_overlap, mlx in ((False, False), (True, True)):
            with self.subTest(enable_overlap=enable_overlap, mlx=mlx):
                s = self._idle_scheduler()
                s.enable_overlap = enable_overlap
                del s.batch_record_buf
                s.is_fully_idle.return_value = True
                with patch("sglang.srt.managers.scheduler.use_mlx", return_value=mlx):
                    s.on_idle()
                s.device_module.synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
