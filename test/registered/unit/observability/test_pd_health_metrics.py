"""Decode readiness state exported to Prometheus (sglang-torchtpu #1276).

Contract: alerts can read the readiness state and its inputs from /metrics;
exactly one state label carries 1, and a disabled tracker publishes nothing.

Usage:
    python3 -m pytest test/registered/unit/observability/test_pd_health_metrics.py -v
"""

import functools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import prometheus_client

from sglang.srt.disaggregation.health import PDHealthConfig, PDTransferHealth
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

STATE = "sglang:pd_health_state"
KNOWN = "sglang:pd_health_known_peers"
OPEN = "sglang:pd_health_open_peers"
PEER_OPEN = "sglang:pd_health_peer_open"
WAIT = "sglang:pd_health_probation_wait_seconds"
LABELS = {
    "engine_type": "decode",
    "model_name": "test",
    "tp_rank": "0",
    "moe_ep_rank": "0",
}


class TestPDHealthMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(enable_metrics=True)
        self.server_args = override.install()
        self.addCleanup(override.restore)
        self.registry = prometheus_client.CollectorRegistry()
        for kind in ("Counter", "Gauge", "Histogram", "Summary"):
            factory = functools.partial(
                getattr(prometheus_client, kind), registry=self.registry
            )
            mocked = patch.object(prometheus_client, kind, factory)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.collector = SchedulerMetricsCollector(LABELS, self.server_args)

    def _state_values(self):
        return {
            s: self.registry.get_sample_value(STATE, {**LABELS, "state": s})
            for s in ("ready", "probation", "unhealthy")
        }

    def _peer_open(self, peer):
        return self.registry.get_sample_value(PEER_OPEN, {**LABELS, "peer": peer})

    def test_exactly_one_state_label_is_set_and_follows_the_tracker(self):
        clock = SimpleNamespace(t=0.0)
        tracker = PDTransferHealth(
            config=PDHealthConfig(
                failure_threshold=2,
                failure_window_s=10.0,
                probation_output_timeout_s=10.0,
            ),
            clock=lambda: clock.t,
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_health = tracker
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = self.collector

        scheduler.publish_pd_health_metrics()
        self.assertEqual(
            self._state_values(), {"ready": 1, "probation": 0, "unhealthy": 0}
        )
        self.assertEqual(self.registry.get_sample_value(KNOWN, LABELS), 0)

        # p1 opens (2 failures spanning 10 s); p2 is healthy: 1/2, degraded, ready.
        tracker.record_failure(peer="p1")
        tracker.record_success(peer="p2")
        clock.t = 10.0
        tracker.record_failure(peer="p1")
        scheduler.publish_pd_health_metrics()
        self.assertEqual(
            self._state_values(), {"ready": 1, "probation": 0, "unhealthy": 0}
        )
        self.assertEqual(self.registry.get_sample_value(KNOWN, LABELS), 2)
        self.assertEqual(self.registry.get_sample_value(OPEN, LABELS), 1)
        self.assertEqual((self._peer_open("p1"), self._peer_open("p2")), (1, 0))

        # p2 opens too: 2/2 trips the engine.
        tracker.record_failure(peer="p2")
        clock.t = 20.0
        tracker.record_failure(peer="p2")
        scheduler.publish_pd_health_metrics()
        self.assertEqual(
            self._state_values(), {"ready": 0, "probation": 0, "unhealthy": 1}
        )
        self.assertEqual(self.registry.get_sample_value(OPEN, LABELS), 2)
        self.assertEqual(self.registry.get_sample_value(WAIT, LABELS), 10.0)

        clock.t = 40.0
        scheduler.publish_pd_health_metrics()
        self.assertEqual(
            self._state_values(), {"ready": 0, "probation": 1, "unhealthy": 0}
        )

        # A success on p2 closes its circuit: the per-peer series must drop to 0,
        # not linger at 1 from the previous publish.
        tracker.record_success(peer="p2")
        scheduler.publish_pd_health_metrics()
        self.assertEqual((self._peer_open("p1"), self._peer_open("p2")), (1, 0))
        self.assertEqual(
            self._state_values(), {"ready": 1, "probation": 0, "unhealthy": 0}
        )

    def test_nothing_is_published_without_a_tracker_or_with_metrics_off(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.metrics_collector = Mock()
        scheduler.pd_health = None
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.publish_pd_health_metrics()
        scheduler.pd_health = PDTransferHealth()
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.publish_pd_health_metrics()
        scheduler.metrics_collector.set_pd_health.assert_not_called()


if __name__ == "__main__":
    unittest.main()
