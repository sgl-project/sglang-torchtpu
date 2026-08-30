"""CPU contract tests for Raiden PD metrics."""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from prometheus_client import CollectorRegistry

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.raiden.conn import (
    RaidenKVManager,
    RaidenKVReceiver,
    RaidenKVSender,
)
from sglang.srt.disaggregation.raiden.metric_collector import (
    NullTPURaidenMetricsCollector,
    TPURaidenMetricsCollector,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ACTIVE = "sglang:tpu_raiden_kv_active_transfers"
TRANSFERS = "sglang:tpu_raiden_kv_transfers_total"


class TestRaidenMetricCollector(CustomTestCase):
    """Verify exported samples and lifecycle semantics with real Prometheus."""

    def test_exports_core_families_and_config_labels(self):
        registry = CollectorRegistry()
        with get_context().override_server_args(
            enable_metrics=True,
            served_model_name="test-model",
            disaggregation_mode="decode",
            extra_metric_labels={"fleet": "test"},
        ):
            metrics = TPURaidenMetricsCollector.init_new(
                tp_rank=0, dp_rank=1, registry=registry
            )
        metrics.transfer_started(1)
        metrics.transfer_started(1)
        metrics.transfer_started(2)
        metrics.transfer_finished(3, "failure")
        metrics.transfer_finished(1, "success")
        metrics.transfer_finished(1, "failure")

        self.assertEqual(
            {metric.name for metric in registry.collect()},
            {ACTIVE, TRANSFERS.removesuffix("_total")},
        )
        labels = {
            "model_name": "test-model",
            "engine_type": "decode",
            "tp_rank": "0",
            "dp_rank": "1",
            "fleet": "test",
            "transfer_mode": "pull",
            "transfer_type": "whole",
        }
        self.assertEqual(registry.get_sample_value(ACTIVE, labels), 1)
        self.assertEqual(
            registry.get_sample_value(TRANSFERS, {**labels, "outcome": "success"}),
            1,
        )

    def test_transfer_types_have_independent_lifecycles(self):
        registry = CollectorRegistry()
        metrics = TPURaidenMetricsCollector(labels={}, registry=registry)

        def value(name, transfer_type, **extra):
            return registry.get_sample_value(
                name,
                {
                    "transfer_mode": "pull",
                    "transfer_type": transfer_type,
                    **extra,
                },
            )

        for transfer_type in ("chunked", "whole"):
            self.assertEqual(value(ACTIVE, transfer_type), 0)
            for outcome in ("success", "failure", "cancelled"):
                self.assertEqual(value(TRANSFERS, transfer_type, outcome=outcome), 0)

        metrics.transfer_started(1, chunked=True)
        metrics.transfer_started(1, chunked=True)
        metrics.transfer_started(1, chunked=False)
        metrics.transfer_started(2, chunked=False)
        self.assertEqual(value(ACTIVE, "chunked"), 1)
        self.assertEqual(value(ACTIVE, "whole"), 1)

        metrics.transfer_finished(1, "success")
        metrics.transfer_finished(1, "failure")
        self.assertEqual(value(ACTIVE, "chunked"), 0)
        self.assertEqual(value(ACTIVE, "whole"), 1)
        self.assertEqual(value(TRANSFERS, "chunked", outcome="success"), 1)
        self.assertEqual(value(TRANSFERS, "chunked", outcome="failure"), 0)

        metrics.transfer_finished(2, "failure")
        self.assertEqual(value(ACTIVE, "whole"), 0)
        self.assertEqual(value(TRANSFERS, "whole", outcome="failure"), 1)

        metrics.transfer_started(3, chunked=True)
        metrics.transfer_finished(3, "cancelled")
        metrics.transfer_finished(3, "cancelled")
        self.assertEqual(value(ACTIVE, "chunked"), 0)
        self.assertEqual(value(TRANSFERS, "chunked", outcome="cancelled"), 1)

    def test_disabled_and_nonprimary_ranks_register_nothing(self):
        for enabled, rank in ((False, 0), (True, 1)):
            with self.subTest(enabled=enabled, rank=rank):
                registry = CollectorRegistry()
                with get_context().override_server_args(enable_metrics=enabled):
                    metrics = TPURaidenMetricsCollector.init_new(
                        tp_rank=rank, dp_rank=0, registry=registry
                    )
                self.assertIs(type(metrics), NullTPURaidenMetricsCollector)
                metrics.transfer_started(1)
                metrics.transfer_finished(1, "success")
                self.assertEqual(list(registry.collect()), [])
                with self.assertRaises(AttributeError):
                    metrics.transfer_strated(1)

    def test_concurrent_duplicate_hooks_count_once(self):
        registry = CollectorRegistry()
        metrics = TPURaidenMetricsCollector(labels={}, registry=registry)
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(metrics.transfer_started, [1] * 32))
            list(
                executor.map(
                    lambda _: metrics.transfer_finished(1, "success"), range(32)
                )
            )
        labels = {"transfer_mode": "pull", "transfer_type": "whole"}
        self.assertEqual(registry.get_sample_value(ACTIVE, labels), 0)
        self.assertEqual(
            registry.get_sample_value(TRANSFERS, {**labels, "outcome": "success"}), 1
        )

    def test_prefill_labels_without_extra_labels(self):
        registry = CollectorRegistry()
        with get_context().override_server_args(
            enable_metrics=True,
            served_model_name="prefill-model",
            disaggregation_mode="prefill",
            extra_metric_labels=None,
        ):
            metrics = TPURaidenMetricsCollector.init_new(
                tp_rank=0, dp_rank=2, registry=registry
            )
        metrics.transfer_started(1, chunked=True)
        labels = {
            "model_name": "prefill-model",
            "engine_type": "prefill",
            "tp_rank": "0",
            "dp_rank": "2",
            "transfer_mode": "pull",
            "transfer_type": "chunked",
        }
        self.assertEqual(registry.get_sample_value(ACTIVE, labels), 1)

    def test_finishing_unknown_rooms_leaves_all_series_at_zero(self):
        registry = CollectorRegistry()
        metrics = TPURaidenMetricsCollector(labels={}, registry=registry)
        for outcome in ("success", "failure", "cancelled"):
            metrics.transfer_finished(42, outcome)
        for transfer_type in ("chunked", "whole"):
            labels = {"transfer_mode": "pull", "transfer_type": transfer_type}
            self.assertEqual(registry.get_sample_value(ACTIVE, labels), 0)
            for outcome in ("success", "failure", "cancelled"):
                self.assertEqual(
                    registry.get_sample_value(
                        TRANSFERS, {**labels, "outcome": outcome}
                    ),
                    0,
                )


class TestRaidenTransferMetricHooks(CustomTestCase):
    """Drive Raiden terminal paths against a real collector and mocked I/O."""

    def _transfer(self, transfer_class, *, started=True, chunked=False):
        registry = CollectorRegistry()
        manager = Mock(spec=RaidenKVManager)
        manager.metrics = TPURaidenMetricsCollector(labels={}, registry=registry)
        manager.chunked_transfer = chunked
        manager.is_dummy_cp_rank = False
        manager._state_lock = threading.Lock()
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        manager.required_prefill_response_num_table = {}
        manager.prefill_response_tracker = {}
        manager.bootstrap_timeout = 300
        manager.check_status.return_value = KVPoll.Transferring
        manager.is_failed.return_value = False
        manager.is_send_done.return_value = False
        manager.is_recv_done.return_value = False
        manager.sender_has_active_io.return_value = False
        manager.room_has_active_io.return_value = False
        manager.maybe_start_pull.return_value = False
        manager.fail_decode_silence.return_value = False

        def update_status(room, state):
            manager.check_status.return_value = state

        def fail_room(room, reason, **kwargs):
            manager.is_failed.return_value = True
            manager.failure_records[room] = reason

        manager.update_status.side_effect = update_status
        manager._fail_room_locked.side_effect = fail_room
        transfer = object.__new__(transfer_class)
        transfer.kv_mgr = manager
        transfer.bootstrap_room = 71
        transfer.conclude_state = None
        transfer._started = False
        transfer._transfer_start_time = None
        transfer.init_time = None
        transfer.abort_notified = True
        if started:
            # Seed an admitted transfer. Tests must terminalize it via Raiden.
            manager.metrics.transfer_started(71, chunked=chunked)
        return transfer, registry

    def _assert_metrics(self, transfer, registry, *, active, outcome=None):
        labels = {
            "transfer_mode": "pull",
            "transfer_type": (
                "chunked" if transfer.kv_mgr.chunked_transfer else "whole"
            ),
        }
        self.assertEqual(registry.get_sample_value(ACTIVE, labels), active)
        for candidate in ("success", "failure", "cancelled"):
            self.assertEqual(
                registry.get_sample_value(TRANSFERS, {**labels, "outcome": candidate}),
                int(candidate == outcome),
            )

    def test_success_is_recorded_before_cleanup(self):
        for transfer_class in (RaidenKVSender, RaidenKVReceiver):
            for chunked in (False, True):
                with self.subTest(transfer=transfer_class.__name__, chunked=chunked):
                    transfer, registry = self._transfer(transfer_class, chunked=chunked)
                    manager = transfer.kv_mgr
                    self.assertEqual(transfer.poll(), KVPoll.Transferring)
                    self._assert_metrics(transfer, registry, active=1)
                    manager.is_send_done.return_value = True
                    manager.is_recv_done.return_value = True
                    self.assertEqual(transfer.poll(), KVPoll.Success)
                    self._assert_metrics(
                        transfer, registry, active=0, outcome="success"
                    )
                    transfer.poll()
                    transfer.clear()
                    transfer.clear()
                    self._assert_metrics(
                        transfer, registry, active=0, outcome="success"
                    )

    def test_sender_status_success_is_recorded(self):
        sender, registry = self._transfer(RaidenKVSender)
        sender.kv_mgr.check_status.return_value = KVPoll.Success
        self.assertEqual(sender.poll(), KVPoll.Success)
        self._assert_metrics(sender, registry, active=0, outcome="success")

    def test_receiver_waits_for_chunks_and_swa(self):
        receiver, registry = self._transfer(RaidenKVReceiver, chunked=True)
        manager = receiver.kv_mgr
        room = receiver.bootstrap_room
        manager._final_seen = {room}
        manager._chunk_queue = {}
        manager._chunk_reads = {room: [f"{room}#0", str(room)]}
        manager._completed_recv = {str(room)}
        manager._swa_engine = Mock()
        manager._swa_req_id.return_value = f"swa:{room}"
        manager.is_recv_done.side_effect = lambda room: RaidenKVManager.is_recv_done(
            manager, room
        )
        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self._assert_metrics(receiver, registry, active=1)
        manager._completed_recv.add(f"{room}#0")
        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self._assert_metrics(receiver, registry, active=1)
        manager._completed_recv.add(f"swa:{room}")
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self._assert_metrics(receiver, registry, active=0, outcome="success")

    def test_failure_is_recorded_only_after_native_io_drains(self):
        for transfer_class in (RaidenKVSender, RaidenKVReceiver):
            with self.subTest(transfer=transfer_class.__name__):
                transfer, registry = self._transfer(transfer_class)
                manager = transfer.kv_mgr
                manager.is_failed.return_value = True
                manager.sender_has_active_io.return_value = True
                manager.room_has_active_io.return_value = True
                self.assertEqual(transfer.poll(), KVPoll.Transferring)
                self._assert_metrics(transfer, registry, active=1)
                manager.sender_has_active_io.return_value = False
                manager.room_has_active_io.return_value = False
                self.assertEqual(transfer.poll(), KVPoll.Failed)
                self._assert_metrics(transfer, registry, active=0, outcome="failure")
                with self.assertRaisesRegex(Exception, "Raiden"):
                    transfer.failure_exception()
                transfer.clear()
                self._assert_metrics(transfer, registry, active=0, outcome="failure")

    def test_abort_is_recorded_without_changing_native_io_drain(self):
        for transfer_class in (RaidenKVSender, RaidenKVReceiver):
            for pending_io in (False, True):
                with self.subTest(
                    transfer=transfer_class.__name__, pending_io=pending_io
                ):
                    transfer, registry = self._transfer(transfer_class)
                    manager = transfer.kv_mgr
                    manager.sender_has_active_io.return_value = pending_io
                    manager.room_has_active_io.return_value = pending_io
                    transfer.abort()
                    self._assert_metrics(
                        transfer, registry, active=0, outcome="cancelled"
                    )
                    if pending_io:
                        self.assertIsNone(transfer.conclude_state)
                        self.assertEqual(transfer.poll(), KVPoll.Transferring)
                        self._assert_metrics(
                            transfer, registry, active=0, outcome="cancelled"
                        )
                        manager.sender_has_active_io.return_value = False
                        manager.room_has_active_io.return_value = False
                    self.assertEqual(transfer.poll(), KVPoll.Failed)
                    self._assert_metrics(
                        transfer, registry, active=0, outcome="cancelled"
                    )
                    with self.assertRaisesRegex(Exception, "Aborted by AbortReq"):
                        transfer.failure_exception()
                    transfer.clear()
                    self._assert_metrics(
                        transfer, registry, active=0, outcome="cancelled"
                    )

    def test_failure_exception_records_settled_failure_without_another_poll(self):
        for transfer_class in (RaidenKVSender, RaidenKVReceiver):
            with self.subTest(transfer=transfer_class.__name__):
                transfer, registry = self._transfer(transfer_class)
                with self.assertRaisesRegex(Exception, "Raiden"):
                    transfer.failure_exception()
                self._assert_metrics(transfer, registry, active=0, outcome="failure")

    def test_later_group_failure_does_not_reclassify_local_success(self):
        for transfer_class in (RaidenKVSender, RaidenKVReceiver):
            with self.subTest(transfer=transfer_class.__name__):
                transfer, registry = self._transfer(transfer_class)
                transfer.kv_mgr.is_send_done.return_value = True
                transfer.kv_mgr.is_recv_done.return_value = True
                self.assertEqual(transfer.poll(), KVPoll.Success)
                transfer.quiesce_group_failure()
                self.assertEqual(transfer.poll(), KVPoll.Failed)
                transfer.clear()
                self._assert_metrics(transfer, registry, active=0, outcome="success")

    def test_clear_has_no_metric_side_effects(self):
        for transfer_class in (RaidenKVSender, RaidenKVReceiver):
            for state in (None, KVPoll.Success, KVPoll.Failed):
                with self.subTest(transfer=transfer_class.__name__, state=state):
                    transfer, registry = self._transfer(transfer_class)
                    transfer.conclude_state = state
                    transfer.clear()
                    transfer.clear()
                    self._assert_metrics(transfer, registry, active=1)

    def test_failure_before_transfer_start_is_not_counted(self):
        for transfer_class in (RaidenKVSender, RaidenKVReceiver):
            with self.subTest(transfer=transfer_class.__name__):
                transfer, registry = self._transfer(transfer_class, started=False)
                transfer.abort()
                self.assertEqual(transfer.poll(), KVPoll.Failed)
                transfer.clear()
                self._assert_metrics(transfer, registry, active=0)

    def test_sender_registration_counts_only_accepted_reads(self):
        for accepted in (False, True):
            for chunked in (False, True):
                with self.subTest(accepted=accepted, chunked=chunked):
                    sender, registry = self._transfer(
                        RaidenKVSender, started=False, chunked=chunked
                    )
                    manager = sender.kv_mgr
                    manager._aborted = {}
                    manager._send_reads = {}
                    engine = Mock()
                    engine.register_read.return_value = accepted
                    RaidenKVManager._register_sender_read(
                        manager, sender.bootstrap_room, engine, "71", 71, [1]
                    )
                    self._assert_metrics(sender, registry, active=int(accepted))


if __name__ == "__main__":
    unittest.main()
