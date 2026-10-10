import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.tpu.conn import TpuKVSender

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GIB = 1024**3


def _make_sender(statuses):
    mgr = SimpleNamespace(
        bootstrap_timeout=300,
        check_status=Mock(side_effect=statuses),
        gather_and_push=Mock(),
        get_kv_replica_factor=lambda: 1,
        kv_item_lens_sum=GIB,
        state_item_lens_sum=0,
    )
    sender = TpuKVSender.__new__(TpuKVSender)
    sender.kv_mgr = mgr
    sender.bootstrap_room = 7
    sender.conclude_state = None
    sender.init_time = time.time()
    sender.curr_idx = 0
    sender.num_kv_indices = 3
    sender.aux_index = None
    sender._transfer_metric = KVTransferMetric()
    sender._transfer_num_kv_indices = 0
    sender._transfer_num_state_indices = 0
    sender._transfer_start_time = None
    return sender


class TestTpuTransferMetrics(CustomTestCase):
    @patch(
        "sglang_tpu.disaggregation.tpu.base.time.perf_counter",
        side_effect=[10.0, 14.0],
    )
    def test_chunked_transfer_uses_full_wall_span(self, _perf_counter):
        sender = _make_sender([KVPoll.Transferring, KVPoll.Success])

        sender.send(np.array([1], dtype=np.int32))
        sender.send(np.array([2, 3], dtype=np.int32))
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertEqual(sender.poll(), KVPoll.Success)

        metric = sender.get_transfer_metric()
        self.assertEqual(metric.transfer_total_bytes, 3 * GIB)
        self.assertEqual(metric.transfer_latency_s, 4.0)
        self.assertEqual(
            metric.transfer_total_bytes / GIB / metric.transfer_latency_s, 0.75
        )

        # Terminal polls must not advance the end timestamp or change the metric.
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(metric.transfer_latency_s, 4.0)
        self.assertEqual(_perf_counter.call_count, 2)

    @patch("sglang_tpu.disaggregation.tpu.base.time.perf_counter", return_value=10.0)
    def test_failed_transfer_does_not_report_latency(self, _perf_counter):
        sender = _make_sender([KVPoll.Failed])

        sender.send(np.array([1, 2, 3], dtype=np.int32))

        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertIsNone(sender.get_transfer_metric().transfer_latency_s)

    @patch(
        "sglang_tpu.disaggregation.tpu.base.time.perf_counter",
        side_effect=[10.0, 12.0],
    )
    def test_state_only_send_starts_timer(self, _perf_counter):
        sender = _make_sender([KVPoll.Success])
        sender.num_kv_indices = 0

        sender.send(
            np.array([], dtype=np.int32),
            state_indices=[np.array([4], dtype=np.int32)],
        )

        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(sender.get_transfer_metric().transfer_latency_s, 2.0)
        self.assertEqual(_perf_counter.call_count, 2)


if __name__ == "__main__":
    unittest.main()
