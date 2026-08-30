"""CPU tests for Raiden KV transfer metrics."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.raiden.conn import RaidenKVSender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=10.0,
        ):
            sender.send([3])

        sender.should_send_kv_chunk(1, last_chunk=True)
        sender.send([4])
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
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
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=14.0,
        ):
            self.assertEqual(sender.poll(), KVPoll.Success)

        self.assertEqual(sender.get_transfer_metric().transfer_latency_s, 4.0)

    def test_terminal_poll_does_not_overwrite_transfer_latency(self):
        sender = self._make_sender()
        sender._transfer_start_time = 10.0

        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=14.0,
        ):
            self.assertEqual(sender.poll(), KVPoll.Success)

        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter"
        ) as perf_counter:
            self.assertEqual(sender.poll(), KVPoll.Success)
            perf_counter.assert_not_called()

        self.assertEqual(sender.get_transfer_metric().transfer_latency_s, 4.0)


if __name__ == "__main__":
    unittest.main()
