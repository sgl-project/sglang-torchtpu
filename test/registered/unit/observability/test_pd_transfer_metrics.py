"""CPU coverage for backend-independent PD transfer-queue age reporting."""

import functools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import prometheus_client

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import (
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

OLDEST = "sglang:kv_transfer_oldest_inflight_age_seconds"
REPORTER = "sglang.srt.managers.scheduler_components.metrics_reporter"


class TestPDTransferMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(enable_metrics=True)
        self.server_args = override.install()
        self.addCleanup(override.restore)
        self.reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        self.reporter.stats = SchedulerStats()
        self.reporter.scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_inflight_queue=[],
            disagg_decode_transfer_queue=SimpleNamespace(
                queue=[], num_pending_deferred_releases=lambda: 0
            ),
        )

    def set_queue(self, mode, times):
        self.reporter.scheduler.disaggregation_mode = mode
        requests = [
            SimpleNamespace(
                time_stats=SimpleNamespace(
                    prefill_transfer_queue_entry_time=(
                        ts if mode == DisaggregationMode.PREFILL else 0
                    ),
                    decode_transfer_queue_entry_time=(
                        ts if mode == DisaggregationMode.DECODE else 0
                    ),
                )
            )
            for ts in times
        ]
        if mode == DisaggregationMode.PREFILL:
            self.reporter.scheduler.disagg_prefill_inflight_queue = requests
        else:
            # No transport poller or status is needed to measure waiting rooms.
            self.reporter.scheduler.disagg_decode_transfer_queue.queue = [
                SimpleNamespace(req=req) for req in requests
            ]

    def age_at(self, now):
        with patch(f"{REPORTER}.time.perf_counter", return_value=now):
            self.reporter._update_kv_transfer_age()
        return self.reporter.stats.kv_transfer_oldest_inflight_age_seconds

    def test_waiting_requests_age_on_both_sides_without_backend_status(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                self.set_queue(mode, [80, 60, 90])
                self.assertEqual(self.age_at(100), 40)
                self.assertEqual(self.age_at(120), 60)

    def test_oldest_removal_and_empty_queue_reset(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                self.set_queue(mode, [60, 80])
                self.assertEqual(self.age_at(100), 40)
                self.set_queue(mode, [80])
                self.assertEqual(self.age_at(110), 30)
                self.set_queue(mode, [])
                self.assertEqual(self.age_at(120), 0)

    def test_unset_timestamps_are_excluded_and_age_is_nonnegative(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            for times, expected in (([0, -1, 80], 20), ([0, -1], 0), ([110], 0)):
                with self.subTest(mode=mode, times=times):
                    self.set_queue(mode, times)
                    self.assertEqual(self.age_at(100), expected)

    def test_non_pd_mode_resets_without_reading_transfer_queues(self):
        self.reporter.stats.kv_transfer_oldest_inflight_age_seconds = 10
        self.reporter.scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL
        )
        self.assertEqual(self.age_at(100), 0)

    def prepare_reporting(self):
        reporter = self.reporter
        scheduler = reporter.scheduler
        scheduler.running_batch = SimpleNamespace(
            reqs=[],
            seq_lens_cpu=None,
            forward_iter=None,
            dp_cooperation_info=None,
            batch_size=lambda: 0,
        )
        scheduler.waiting_queue = []
        scheduler.grammar_manager = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hierarchical_cache = False
        scheduler.forward_ct = 0
        scheduler.max_total_num_tokens = 1000
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[], num_tokens_pre_allocated=0
        )
        pool = Mock()
        pool.get_prefill_usage_msg_parts.return_value = []
        pool.get_decode_usage_msg_parts.return_value = []
        scheduler.pool_stats_observer = Mock()
        scheduler.pool_stats_observer.get_pool_stats.return_value = pool
        scheduler.pool_stats_observer.streaming_session_count.return_value = 0
        scheduler.pool_stats_observer.session_held_tokens.return_value = 0
        scheduler.kv_events_publisher = Mock()
        scheduler.spec_algorithm = Mock()
        scheduler.spec_algorithm.is_none.return_value = True
        reporter.metrics_collector = Mock(last_log_time=0)
        reporter.current_scheduler_metrics_enabled = True
        reporter.is_stats_logging_rank = False
        reporter.enable_mfu_metrics = False
        reporter.last_prefill_stats_tic = 0
        reporter.last_decode_stats_tic = 0
        reporter.forward_ct_decode = 0
        reporter.decode_log_interval = 1
        reporter.num_generated_tokens = 0
        reporter._graph_backend_label = "test"
        reporter.fwd_occupancy = float("nan")
        reporter.scheduler_status_logger = None
        reporter.kv_transfer_speed_gb_s = 0
        reporter.kv_transfer_latency_ms = 0
        reporter.cache_hit_rate_window = Mock()
        reporter._calculate_utilization = Mock()
        reporter._update_lora_metrics = Mock()
        reporter._log_hicache_stats = Mock()

    def test_prefill_decode_and_idle_reporting_publish_and_reset_age(self):
        self.prepare_reporting()
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            for path in ("prefill", "decode", "idle"):
                for times, expected in (([60, 80], 40), ([], 0)):
                    with self.subTest(mode=mode, path=path, times=times):
                        self.set_queue(mode, times)
                        reporter = self.reporter
                        reporter.last_decode_stats_tic = 0
                        reporter.metrics_collector.log_stats.reset_mock()
                        with (
                            patch(f"{REPORTER}.time.perf_counter", return_value=100),
                            patch(f"{REPORTER}.ENABLE_METRICS_DEVICE_TIMER", False),
                            patch(f"{REPORTER}.RECORD_STEP_TIME", False),
                        ):
                            if path == "prefill":
                                reporter.report_prefill_stats(
                                    reporter.scheduler.running_batch,
                                    PrefillStats(0, 0, 0, QueueCount(), 0),
                                    False,
                                )
                            elif path == "decode":
                                reporter.report_decode_stats(False)
                            else:
                                reporter._maybe_log_idle_metrics()
                        reporter.metrics_collector.log_stats.assert_called_once_with(
                            reporter.stats
                        )
                        self.assertEqual(
                            reporter.stats.kv_transfer_oldest_inflight_age_seconds,
                            expected,
                        )

    def test_disabled_reporting_does_not_compute_age(self):
        self.prepare_reporting()
        self.reporter.current_scheduler_metrics_enabled = False
        with patch.object(self.reporter, "_update_kv_transfer_age") as update:
            self.reporter.report_prefill_stats(
                self.reporter.scheduler.running_batch,
                PrefillStats(0, 0, 0, QueueCount(), 0),
                False,
            )
            self.reporter.report_decode_stats(False)
            with patch(f"{REPORTER}.ENABLE_METRICS_DEVICE_TIMER", False):
                self.reporter._maybe_log_idle_metrics()
            update.assert_not_called()
        self.reporter.metrics_collector.log_stats.assert_not_called()

    def stall_publish(self, mode, times, last_log_time, now=100):
        self.set_queue(mode, times)
        self.reporter.metrics_collector.last_log_time = last_log_time
        self.reporter.metrics_collector.log_stats.reset_mock()
        with (
            patch(f"{REPORTER}.time.perf_counter", return_value=now),
            patch(f"{REPORTER}.ENABLE_METRICS_DEVICE_TIMER", False),
        ):
            self.reporter.maybe_log_stall_metrics()

    def test_stall_publishes_queues_age_and_zero_throughput(self):
        """A P/D engine whose queued requests never form a batch must not keep
        its last batch's gauges (#1303): after 30 s without a stats log, the
        queues, the growing oldest-transfer age and zero throughput go out."""
        self.prepare_reporting()
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                self.reporter.stats.gen_throughput = 123.0
                self.stall_publish(mode, [60, 80], last_log_time=0)
                self.reporter.metrics_collector.log_stats.assert_called_once_with(
                    self.reporter.stats
                )
                self.assertEqual(
                    self.reporter.stats.kv_transfer_oldest_inflight_age_seconds, 40
                )
                self.assertEqual(self.reporter.stats.gen_throughput, 0)
                if mode == DisaggregationMode.PREFILL:
                    queued = self.reporter.stats.num_prefill_inflight_queue_reqs
                else:
                    queued = self.reporter.stats.num_decode_transfer_queue_reqs
                self.assertEqual(queued.total, 2)

    def test_stall_waits_out_the_window_after_a_stats_log(self):
        """An engine that is only between batches logged stats recently; it must
        not report a zero throughput in that gap."""
        self.prepare_reporting()
        self.stall_publish(DisaggregationMode.DECODE, [60], last_log_time=90)
        self.reporter.metrics_collector.log_stats.assert_not_called()

    def test_stall_publish_respects_disabled_metrics(self):
        self.prepare_reporting()
        self.reporter.current_scheduler_metrics_enabled = False
        self.stall_publish(DisaggregationMode.DECODE, [60], last_log_time=0)
        self.reporter.metrics_collector.log_stats.assert_not_called()

    def test_collector_exports_stats_with_standard_labels(self):
        registry = prometheus_client.CollectorRegistry()
        for kind in ("Counter", "Gauge", "Histogram", "Summary"):
            factory = functools.partial(
                getattr(prometheus_client, kind), registry=registry
            )
            mocked = patch.object(prometheus_client, kind, factory)
            mocked.start()
            self.addCleanup(mocked.stop)
        labels = {
            "engine_type": "decode",
            "model_name": "test",
            "tp_rank": "0",
            "moe_ep_rank": "0",
        }
        collector = SchedulerMetricsCollector(labels, self.server_args)
        for age in (12.5, 0.0):
            collector.log_stats(
                SchedulerStats(kv_transfer_oldest_inflight_age_seconds=age)
            )
            self.assertEqual(registry.get_sample_value(OLDEST, labels), age)


if __name__ == "__main__":
    unittest.main()
