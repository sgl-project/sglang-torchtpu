"""CPU tests for Raiden KV transfer metrics."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import sglang.srt.observability.req_time_stats as rts
from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.disaggregation.raiden.conn import RaidenKVSender

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRaidenKVTransferMetrics(CustomTestCase):
    def _make_sender(self):
        manager = Mock()
        manager.is_dummy_cp_rank = False
        manager.chunked_transfer = True
        manager.kv_item_lens_sum = 1 << 30
        manager.state_item_lens_sum = 0
        manager.get_kv_replica_factor.return_value = 1
        manager._swa_blocks_from_state_indices.return_value = []
        manager.append_send_blocks.side_effect = [0, 1]
        manager.is_aborted.return_value = False
        manager.is_failed.return_value = False
        manager.is_send_done.return_value = True
        # No raiden finish time: falls back to the poll time.
        manager.send_done_time.return_value = None
        manager.check_status.return_value = KVPoll.Transferring

        sender = object.__new__(RaidenKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7
        sender.aux_index = None
        sender.conclude_state = None
        sender._transfer_metric = KVTransferMetric()
        sender._transfer_num_kv_indices = 0
        sender._transfer_num_state_indices = 0
        sender.num_kv_indices = 2
        sender._last_chunk = False
        sender._started = False
        sender._transfer_start_time = None
        return sender

    def test_latency_spans_first_chunk_through_request_completion(self):
        sender = self._make_sender()

        sender.should_send_kv_chunk(1, last_chunk=False)
        with patch(
            "sglang_tpu.disaggregation.raiden.conn.time.perf_counter",
            return_value=10.0,
        ):
            sender.send([3])

        sender.should_send_kv_chunk(1, last_chunk=True)
        sender.send([4])
        with patch(
            "sglang_tpu.disaggregation.raiden.conn.time.perf_counter",
            return_value=14.0,
        ):
            self.assertEqual(sender.poll(), KVPoll.Success)

        metric = sender.get_transfer_metric()
        self.assertEqual(metric.transfer_total_bytes, 2 << 30)
        self.assertEqual(metric.transfer_latency_s, 4.0)
        self.assertEqual(
            metric.transfer_total_bytes / (1 << 30) / metric.transfer_latency_s,
            0.5,
        )

    def test_status_success_also_records_transfer_latency(self):
        sender = self._make_sender()
        sender._transfer_start_time = 10.0
        sender.kv_mgr.is_send_done.return_value = False
        sender.kv_mgr.check_status.return_value = KVPoll.Success

        with patch(
            "sglang_tpu.disaggregation.raiden.conn.time.perf_counter",
            return_value=14.0,
        ):
            self.assertEqual(sender.poll(), KVPoll.Success)

        self.assertEqual(sender.get_transfer_metric().transfer_latency_s, 4.0)

    def test_terminal_poll_does_not_overwrite_transfer_latency(self):
        sender = self._make_sender()
        sender._transfer_start_time = 10.0

        with patch(
            "sglang_tpu.disaggregation.raiden.conn.time.perf_counter",
            return_value=14.0,
        ):
            self.assertEqual(sender.poll(), KVPoll.Success)

        with patch(
            "sglang_tpu.disaggregation.raiden.conn.time.perf_counter"
        ) as perf_counter:
            self.assertEqual(sender.poll(), KVPoll.Success)
            perf_counter.assert_not_called()

        self.assertEqual(sender.get_transfer_metric().transfer_latency_s, 4.0)


class TestPrefillTransferStageEndsAtRaidenFinish(CustomTestCase):
    """The TPU patch ends prefill_transfer_kv_cache at raiden's finish time."""

    ENTRY, POLL_TIME = 10.0, 14.0

    def setUp(self):
        mixin = SchedulerDisaggregationPrefillMixin
        orig_poll = mixin.process_disagg_prefill_inflight_queue
        self.shared_poll = orig_poll
        orig_set = rts.SchedulerReqTimeStats.set_prefill_kv_transfer_finish_time

        def restore():
            mixin.process_disagg_prefill_inflight_queue = orig_poll
            rts.SchedulerReqTimeStats.set_prefill_kv_transfer_finish_time = orig_set
            patches._prefill_transfer_finish_from_raiden_patched = False

        self.addCleanup(restore)

        # Minimal stand-in for the shared poll: stamps each finished request.
        def poll(self, rids_to_check=None):
            for req in self.disagg_prefill_inflight_queue:
                req.time_stats.set_prefill_kv_transfer_finish_time()
            return list(self.disagg_prefill_inflight_queue)

        mixin.process_disagg_prefill_inflight_queue = poll
        patches._prefill_transfer_finish_from_raiden_patched = False
        patches._patch_prefill_transfer_finish_from_raiden()

    def _observed_stage(self, sender):
        stats = rts.SchedulerReqTimeStats(disagg_mode=rts.DisaggregationMode.PREFILL)
        stats.enable_metrics = True
        stats.metrics_collector = MagicMock()
        stats.set_prefill_transfer_queue_entry_time(self.ENTRY)
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[
                SimpleNamespace(time_stats=stats, disagg_kv_sender=sender)
            ]
        )
        with patch.object(rts.time, "perf_counter", return_value=self.POLL_TIME):
            SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )
        observe = stats.metrics_collector.observe_per_stage_req_latency
        observe.assert_called_once()
        stage, latency = observe.call_args.args
        self.assertEqual(stage, "prefill_transfer_kv_cache")
        return latency

    @staticmethod
    def _raiden_sender(finish):
        sender = object.__new__(RaidenKVSender)
        sender.transfer_finish_time = finish
        return sender

    def test_stage_ends_at_raidens_finish(self):
        self.assertAlmostEqual(self._observed_stage(self._raiden_sender(11.5)), 1.5)

    def test_other_backends_and_unrecorded_finishes_keep_the_poll_time(self):
        for sender in (Mock(), self._raiden_sender(None)):
            with self.subTest(sender=sender):
                self.assertAlmostEqual(self._observed_stage(sender), 4.0)

    def test_finish_before_the_queue_entry_observes_zero(self):
        self.assertEqual(self._observed_stage(self._raiden_sender(9.9)), 0.0)

    def test_shared_poll_still_stamps_without_a_time(self):
        """The patch relies on this call; if it changes, the stage silently
        falls back to the poll time."""
        source = inspect.getsource(self.shared_poll)
        self.assertIn("req.time_stats.set_prefill_kv_transfer_finish_time()", source)


if __name__ == "__main__":
    unittest.main()
