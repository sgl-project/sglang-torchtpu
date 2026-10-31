"""Unit tests for the TPU patch that decides --max-queued-requests at the PD decode's door."""

import threading
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_decode_door_admission as tpu
from sglang_tpu.disaggregation.raiden.sender import RaidenSenderMixin

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _req(rid="r"):
    return SimpleNamespace(
        rid=rid, bootstrap_room=7, return_logprob=False, finished_reason=None
    )


def _receiver(bootstrap_infos=({"rank_ip": "10.0.0.1", "rank_port": 9000},)):
    sock = MagicMock()
    receiver = MagicMock()
    receiver.bootstrap_room = 7
    receiver.conclude_state = None
    receiver.bootstrap_infos = list(bootstrap_infos)
    receiver.kv_mgr = MagicMock(local_ip="10.0.0.2", rank_port=9100)
    receiver._connect_to_bootstrap_server.return_value = (sock, threading.Lock())
    receiver._sock = sock
    return receiver


def _queue(
    *, limit, running=8, max_running=8, pending=0, transfer=0, waiting=0, retracted=0
):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.queue = [object()] * pending
    queue.retracted_queue = [object()] * retracted
    queue.scheduler = SimpleNamespace(
        max_queued_requests=limit,
        disagg_decode_transfer_queue=SimpleNamespace(queue=[object()] * transfer),
        waiting_queue=[object()] * waiting,
        req_to_token_pool=SimpleNamespace(size=max_running),
        max_running_requests=max_running,
        running_batch=MagicMock(**{"batch_size.return_value": running}),
        retire_unadmitted_request=MagicMock(),
    )
    return queue


def _decode_req(queue, receiver=None):
    decode_req = SimpleNamespace(
        req=_req(), kv_receiver=receiver or _receiver(), is_rebootstrap=False
    )
    queue.queue.append(decode_req)  # _create_receiver_and_enqueue did this already
    return decode_req


class TestDecodeDoor(CustomTestCase):
    def setUp(self):
        patcher = patch.object(tpu, "_upstream_init_receiver")
        self.upstream = patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_limit_passes_through(self):
        queue = _queue(limit=None, waiting=50)
        dr = _decode_req(queue)
        tpu._init_receiver(queue, dr, 0)
        self.upstream.assert_called_once_with(queue, dr, 0)
        dr.kv_receiver.init.assert_not_called()

    def test_below_the_limit_passes_through(self):
        queue = _queue(limit=4, waiting=3)
        dr = _decode_req(queue)
        tpu._init_receiver(queue, dr, 0)
        self.upstream.assert_called_once()
        queue.scheduler.retire_unadmitted_request.assert_not_called()

    def test_over_the_limit_refuses_both_sides(self):
        queue = _queue(limit=1, waiting=1)
        dr = _decode_req(queue)
        tpu._init_receiver(queue, dr, 3)
        self.upstream.assert_not_called()
        # The prefill ranks of the room are resolved (nothing pulled, so the prefill stays idle) ...
        dr.kv_receiver.init.assert_called_once_with(3)
        # ... and told, with the reason, so the twin ends with the same 503.
        frames = dr.kv_receiver._sock.send_multipart.call_args[0][0]
        self.assertEqual(frames[:2], [b"ABORT", b"7"])
        self.assertEqual(frames[-1], tpu.REASON_FRAME)
        dr.kv_receiver.kv_mgr.update_status.assert_called_once_with(7, KVPoll.Failed)
        self.assertIs(dr.kv_receiver.conclude_state, KVPoll.Failed)
        # The client gets the decode's 503 and the request never reaches pre-allocation.
        self.assertNotIn(dr, queue.queue)
        self.assertIsInstance(dr.req.finished_reason, FINISH_ABORT)
        self.assertEqual(
            dr.req.finished_reason.status_code, HTTPStatus.SERVICE_UNAVAILABLE
        )
        self.assertEqual(dr.req.finished_reason.message, tpu.QUEUE_FULL)
        queue.scheduler.retire_unadmitted_request.assert_called_once_with(dr.req)

    def test_init_failure_skips_the_notification(self):
        queue = _queue(limit=1, waiting=1)
        receiver = _receiver()
        receiver.init.side_effect = lambda rank: setattr(
            receiver, "conclude_state", KVPoll.Failed
        )
        dr = _decode_req(queue, receiver)
        tpu._init_receiver(queue, dr, 0)
        receiver._sock.send_multipart.assert_not_called()
        queue.scheduler.retire_unadmitted_request.assert_called_once_with(dr.req)

    def test_free_running_slots_offset_the_backlog(self):
        # 6 free slots absorb 6 pending requests (the newcomer among them).
        queue = _queue(limit=1, running=2, pending=3, transfer=2, waiting=1)
        tpu._init_receiver(queue, _decode_req(queue), 0)
        self.upstream.assert_called_once()
        self.upstream.reset_mock()
        queue = _queue(limit=1, running=2, pending=3, transfer=2, waiting=2)
        tpu._init_receiver(queue, _decode_req(queue), 0)
        self.upstream.assert_not_called()

    def test_retracted_requests_hold_a_slot(self):
        queue = _queue(limit=1, running=6, retracted=2, waiting=1)
        tpu._init_receiver(queue, _decode_req(queue), 0)
        self.upstream.assert_not_called()

    def test_rebootstrap_bypasses_the_gate(self):
        queue = _queue(limit=0, waiting=5)
        dr = _decode_req(queue)
        dr.is_rebootstrap = True
        tpu._init_receiver(queue, dr, 0)
        self.upstream.assert_called_once()


class TestPrefillSide(CustomTestCase):
    def _scheduler(self):
        return SimpleNamespace(
            tree_cache=MagicMock(),
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            output_streamer=MagicMock(),
        )

    def _req(self, reason):
        kv_mgr = SimpleNamespace(failure_records={7: reason} if reason else {})
        return SimpleNamespace(
            rid="r",
            bootstrap_room=7,
            return_logprob=False,
            finished_reason=None,
            pending_bootstrap=True,
            cache_request_handle=object(),
            disagg_kv_sender=SimpleNamespace(kv_mgr=kv_mgr),
            kv=SimpleNamespace(holds_kv=False, holds_mamba=False),
            time_stats=MagicMock(),
        )

    def test_refused_room_answers_503_without_the_failure_path(self):
        scheduler, req = self._scheduler(), self._req(tpu.QUEUE_FULL)
        with (
            patch.object(tpu, "_upstream_handle_bootstrap_failure") as upstream,
            patch.object(tpu, "maybe_release_metadata_buffer") as release_md,
        ):
            tpu.handle_bootstrap_failure(scheduler, req)
        upstream.assert_not_called()
        release_md.assert_called_once()
        self.assertFalse(req.pending_bootstrap)
        self.assertEqual(
            req.finished_reason.status_code, HTTPStatus.SERVICE_UNAVAILABLE
        )
        self.assertEqual(req.finished_reason.message, tpu.QUEUE_FULL)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        scheduler.tree_cache.finish.assert_called_once()

    def test_other_failures_take_the_upstream_path(self):
        for reason in (None, "Aborted by decode-side abort notification."):
            with (
                self.subTest(reason=reason),
                patch.object(tpu, "_upstream_handle_bootstrap_failure") as upstream,
            ):
                scheduler, req = self._scheduler(), self._req(reason)
                tpu.handle_bootstrap_failure(scheduler, req)
                upstream.assert_called_once_with(scheduler, req)


class _FakeManager:
    """Just enough of RaidenKVManager for _on_abort."""

    def __init__(self, status=None):
        self._state_lock = threading.Lock()
        self._status = status
        self.aborted, self.failed, self.updates, self.dropped = [], [], [], []

    def check_status(self, room):
        if self._status is None:
            raise KeyError(room)
        return self._status

    def _mark_aborted_locked(self, room):
        self.aborted.append(room)

    def _fail_room_locked(self, room, reason):
        self.failed.append((room, reason))

    def update_status(self, room, status):
        self.updates.append((room, status))

    def _drop_room_state(self, room):
        self.dropped.append(room)


class TestRaidenAbortReason(CustomTestCase):
    def test_reason_frame_fails_an_unregistered_room(self):
        mgr = _FakeManager(status=None)
        RaidenSenderMixin._on_abort(
            mgr, [b"ABORT", b"7", b"10.0.0.2", b"9100", tpu.REASON_FRAME]
        )
        self.assertEqual(mgr.aborted, [7])
        self.assertEqual(mgr.failed, [(7, tpu.QUEUE_FULL)])
        self.assertEqual(mgr.updates, [])

    def test_plain_abort_only_tombstones_an_unregistered_room(self):
        mgr = _FakeManager(status=None)
        RaidenSenderMixin._on_abort(mgr, [b"ABORT", b"7", b"10.0.0.2", b"9100"])
        self.assertEqual(mgr.aborted, [7])
        self.assertEqual(mgr.failed, [])

    def test_registered_room_fails_with_the_reason(self):
        mgr = _FakeManager(status=KVPoll.Bootstrapping)
        RaidenSenderMixin._on_abort(mgr, [b"ABORT", b"7", tpu.REASON_FRAME])
        self.assertEqual(mgr.failed, [(7, tpu.QUEUE_FULL)])
        self.assertEqual(mgr.updates, [(7, KVPoll.Failed)])
        self.assertEqual(mgr.dropped, [7])


class TestInstall(CustomTestCase):
    def test_install_replaces_both_methods(self):
        with (
            patch.object(DecodePreallocQueue, "_init_receiver"),
            patch.object(
                SchedulerDisaggregationPrefillMixin, "handle_bootstrap_failure"
            ),
            patch.object(tpu, "_is_installed", False),
        ):
            tpu.install()
            self.assertIs(DecodePreallocQueue._init_receiver, tpu._init_receiver)
            self.assertIs(
                SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure,
                tpu.handle_bootstrap_failure,
            )


if __name__ == "__main__":
    unittest.main()
