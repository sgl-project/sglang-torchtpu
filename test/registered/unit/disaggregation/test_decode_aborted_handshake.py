"""Aborting a decode request that waits for its KV also fails its KV receiver.
The handshake poll must then keep the abort instead of reporting a handshake
failure (500, ERROR log, bootstrap failure count)."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

LOGGER = "sglang.srt.disaggregation.decode"


def _prealloc_queue():
    """A prealloc queue holding one request whose receiver has not failed."""
    req = SimpleNamespace(
        rid="req-0", bootstrap_room=7, finished_reason=None, return_logprob=False
    )
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.queue = [SimpleNamespace(req=req, kv_receiver=FakeKVReceiver(None, ""))]
    queue.retracted_queue = []
    queue.pp_size = 1
    queue.tp_rank = 0
    queue.gloo_group = None
    queue.scheduler = SimpleNamespace(
        metrics_reporter=SimpleNamespace(enable_metrics=True),
        metrics_collector=MagicMock(),
    )
    return queue, req


def _decode_scheduler(prealloc_queue):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = None
    scheduler.mm_receiver = None
    scheduler.waiting_queue = []
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.disagg_decode_prealloc_queue = prealloc_queue
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
    scheduler.collect_inflight_reqs = MagicMock(return_value=set())
    return scheduler


@patch(
    "sglang.srt.disaggregation.decode.poll_and_all_reduce",
    new=lambda receivers, group, **kwargs: [r.poll() for r in receivers],
)
class TestDecodeAbortedHandshake(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_aborted_request_keeps_its_abort(self):
        queue, req = _prealloc_queue()
        _decode_scheduler(queue).abort_request(AbortReq(rid=req.rid))

        with self.assertNoLogs(LOGGER, level="ERROR"):
            queue._update_handshake_waiters()

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.message, "Aborted by AbortReq.")
        self.assertIsNone(req.finished_reason.status_code)
        counted = queue.scheduler.metrics_collector.increment_bootstrap_failed_reqs
        counted.assert_not_called()

    def test_handshake_failure_is_a_counted_500(self):
        queue, req = _prealloc_queue()
        # The decode gives up on the prefill without any AbortReq, as
        # _ensure_prefill_info does.
        queue.queue[0].kv_receiver.abort()

        with self.assertLogs(LOGGER, level="ERROR"):
            queue._update_handshake_waiters()

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(
            req.finished_reason.status_code, HTTPStatus.INTERNAL_SERVER_ERROR
        )
        counted = queue.scheduler.metrics_collector.increment_bootstrap_failed_reqs
        counted.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
