"""Unit tests for the raiden transport's failure handling between prefill and decode."""

import errno
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.conn import (
    RaidenKVManager,
    RaidenKVReceiver,
    RaidenKVSender,
)
from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    PullReq,
    ReshardReq,
    RoomFailed,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 7
OTHER_ROOM = 8


def _polled(sent=(), received=(), failed=(), completed_ns=None):
    """Fake poll_stats_with_details() result for the given req ids."""

    def settled(ids):
        return [SimpleNamespace(req_id=r, completed_ns=completed_ns) for r in ids]

    return SimpleNamespace(
        sent=settled(sent), received=settled(received), failed=settled(failed)
    )


def _abort_frame(room):
    return [MessageTag.ABORT, str(room).encode("ascii"), b"10.0.0.2", b"5555"]


def _prefill_manager():
    """A prefill-mode RaidenKVManager with its room tables, no sockets or engine."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr.attn_tp_rank = 0
    mgr._prefill = PrefillState(announce_pages=1)
    mgr._decode = None
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._swa_engine = None
    mgr._reshard_slot_rooms = [None]
    mgr._reshard_slot_of = {}
    mgr._aborted = {}
    mgr._pulling = None
    # Settle-aware cleanup state shared by every manager mode.
    mgr._pulling_cleared = set()
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr.chunked_transfer = False
    mgr._send_reads = {}
    mgr._superseded_sender_reads = set()
    mgr._timeout_s = 120.0
    mgr._reshard_pushes_pending = {}
    mgr.attn_tp_rank = 0
    mgr._send_multipart_locked = Mock()
    mgr._message_handlers = {
        PullReq.TAG: mgr._on_pull_req,
        ReshardReq.TAG: mgr._on_reshard_req,
    }
    return mgr


def _decode_manager():
    """A decode-mode RaidenKVManager with its room tables, no sockets or engine."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr._engine = None
    mgr._swa_engine = None
    mgr._pulling = None
    mgr.attn_tp_rank = 0
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._aborted = {}
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._completed_recv = set()
    mgr.chunked_transfer = False
    mgr.waiting_timeout = 300
    mgr._message_handlers = {RoomFailed.TAG: mgr._on_room_failed}
    return mgr


def _sent_frames(mgr):
    """(endpoint, frames) of every message the manager sent."""
    return [call.args[:2] for call in mgr._send_multipart_locked.call_args_list]


def _populate_send_state(mgr, room):
    mgr.request_status[room] = KVPoll.Transferring
    mgr._prefill.send_blocks[room] = [1, 2, 3]
    mgr._prefill.send_swa_blocks[room] = [4]
    mgr._prefill.send_aux[room] = b"aux"
    mgr._prefill.decode_addr[room] = ("10.0.0.2", 5555)
    mgr._prefill.replied.add(room)
    mgr._prefill.reshard_finalized.add(room)


class TestPrefillAbortHandling(unittest.TestCase):
    def test_dispatch_rejects_missing_or_invalid_tags(self):
        mgr = _prefill_manager()

        with patch("sglang_tpu.disaggregation.raiden.conn.logger.error") as log_error:
            mgr._dispatch([])
            mgr._dispatch([b""])
            mgr._dispatch([None])
            mgr._dispatch([b"UNKNOWN"])

        self.assertEqual(log_error.call_count, 4)

    def test_abort_marks_room_failed_and_drops_send_state(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)
        _populate_send_state(mgr, OTHER_ROOM)

        mgr._dispatch(_abort_frame(ROOM))

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertIn(ROOM, mgr.failure_records)
        for table in (
            mgr._prefill.send_blocks,
            mgr._prefill.send_swa_blocks,
            mgr._prefill.send_aux,
            mgr._prefill.decode_addr,
            mgr._prefill.replied,
            mgr._prefill.reshard_finalized,
        ):
            self.assertNotIn(ROOM, table)
            self.assertIn(OTHER_ROOM, table)
        self.assertEqual(mgr.check_status(OTHER_ROOM), KVPoll.Transferring)

    def test_abort_is_noop_for_succeeded_room(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)
        mgr.request_status[ROOM] = KVPoll.Success

        mgr._dispatch(_abort_frame(ROOM))

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Success)
        self.assertNotIn(ROOM, mgr.failure_records)
        self.assertIn(ROOM, mgr._prefill.send_blocks)

    def test_abort_for_unknown_room_tombstones_it(self):
        """ABORT and PULL_REQ travel on different sockets, so the abort can
        arrive first; it must still win, or the prefill would admit and
        compute a request the decode already gave up on, holding its pages
        until the waiting timeout."""
        mgr = _prefill_manager()

        mgr._dispatch(_abort_frame(ROOM))

        self.assertEqual(mgr.request_status, {})
        self.assertEqual(mgr.failure_records, {})
        self.assertTrue(mgr.is_aborted(ROOM))

        # The delayed pull request finds the tombstone and is answered with
        # ROOM_FAILED instead of admitting the room.
        mgr._dispatch(PullReq(ROOM, "10.0.0.2", 5555, 4).to_frames())
        self.assertNotIn(ROOM, mgr._prefill.decode_addr)
        [(endpoint, frames)] = _sent_frames(mgr)
        self.assertEqual(endpoint, "tcp://10.0.0.2:5555")
        self.assertEqual(RoomFailed.from_frames(frames).room, ROOM)

    def test_malformed_abort_is_ignored(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)

        with patch("sglang_tpu.disaggregation.raiden.conn.logger.error") as log_error:
            mgr._dispatch([MessageTag.ABORT, b"not-a-room", b"10.0.0.2", b"5555"])
            mgr._dispatch([MessageTag.ABORT])

        self.assertEqual(log_error.call_count, 2)

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Transferring)
        self.assertIn(ROOM, mgr._prefill.send_blocks)

    def test_abort_without_sender_address_is_still_honored(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)

        mgr._dispatch([MessageTag.ABORT, str(ROOM).encode("ascii")])

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertNotIn(ROOM, mgr._prefill.send_blocks)

    def test_abort_propagates_to_sender_poll_and_clear(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)
        mgr._drain_engine = lambda: None
        mgr.is_dummy_cp_rank = False
        mgr.bootstrap_timeout = 300
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = ROOM
        sender.conclude_state = None
        sender.init_time = time.time()

        mgr._dispatch(_abort_frame(ROOM))

        self.assertEqual(sender.poll(), KVPoll.Failed)
        with self.assertRaises(Exception) as cm:
            sender.failure_exception()
        self.assertIn("abort", str(cm.exception).lower())
        self.assertNotIn(ROOM, mgr.request_status)
        self.assertNotIn(ROOM, mgr.failure_records)
        # The tombstone outlives the room: frames still on the wire for it
        # must stay rejected. It ages out on its own.
        self.assertTrue(mgr.is_aborted(ROOM))

    def test_abort_is_terminal_for_a_prefill_still_computing(self):
        """A send that lands after the abort must not revive the room."""
        mgr = _prefill_manager()
        mgr._ensure_engine = lambda: None
        mgr.is_dummy_cp_rank = False
        mgr.request_status[ROOM] = KVPoll.WaitingForInput
        mgr._prefill.decode_addr[ROOM] = ("10.0.0.2", 5555)
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = ROOM
        sender.conclude_state = None
        sender.init_time = time.time()
        sender.aux_index = None
        sender._last_chunk = True
        sender._started = False
        sender._record_transfer_indices = lambda *a: None

        mgr._dispatch(_abort_frame(ROOM))
        sender.send(np.asarray([1, 2], dtype=np.int32))

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertNotIn(ROOM, mgr._prefill.send_blocks)
        self.assertFalse(sender._started)


def _decode_manager_stub(**overrides):
    fields = dict(
        _drain_engine=lambda: None,
        is_failed=lambda room: False,
        is_recv_done=lambda room: False,
        last_announcement_age=lambda room: None,
        maybe_start_pull=lambda room: False,
        request_status={ROOM: KVPoll.WaitingForInput},
        failure_records={},
        failure_lock=threading.Lock(),
        waiting_timeout=300,
        _state_lock=threading.Lock(),
        _aborted={},
        _failed=set(),
        _native_failed=set(),
        room_has_active_io=lambda room: False,
        # _check_waiting_timeout takes the manager's connection lock.
        connection_lock=threading.Lock(),
        connection_pool={},
    )
    fields.update(overrides)
    mgr = SimpleNamespace(**fields)
    mgr._mark_aborted_locked = types.MethodType(
        RaidenKVManager._mark_aborted_locked, mgr
    )
    mgr._fail_room_locked = types.MethodType(RaidenKVManager._fail_room_locked, mgr)
    mgr.check_status = lambda room: mgr.request_status[room]
    mgr.record_failure = lambda room, reason: mgr.failure_records.__setitem__(
        room, reason
    )

    def update_status(room, status):
        mgr.request_status[room] = status

    mgr.update_status = update_status
    return mgr


def _receiver(mgr):
    recv = RaidenKVReceiver.__new__(RaidenKVReceiver)
    recv.kv_mgr = mgr
    recv.bootstrap_room = ROOM
    recv.conclude_state = None
    recv.abort_notified = False
    recv.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 4444}]
    recv._connection_pool_entries = {}
    recv._started = False
    recv.init_time = None
    recv._handshake_start = time.time()
    recv._handshake_timeout = 300
    return recv


class TestReceiverFailurePaths(unittest.TestCase):
    def test_failed_status_is_sticky_against_late_progress(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.request_status = {ROOM: KVPoll.Failed}

        mgr.update_status(ROOM, KVPoll.Transferring)

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)

    def test_abort_while_start_read_blocks_cannot_be_overwritten(self):
        from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission

        started = threading.Event()
        resume = threading.Event()
        engine = Mock()

        def blocking_start(*_args):
            started.set()
            self.assertTrue(resume.wait(5))

        engine.start_read.side_effect = blocking_start
        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr._aborted = {}
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._decode.remote_blocks = {ROOM: [11, 12]}
        mgr._decode.remote_endpoint = {ROOM: "peer"}
        mgr._decode.local_blocks = {ROOM: [21, 22]}
        mgr._swa_engine = None
        mgr.chunked_transfer = False
        mgr._full_engine_max_blocks = 4
        mgr._swa_engine_max_blocks = 0
        mgr._dynamic_host_staging = True
        mgr._pulling = StagingAdmission(4)
        mgr._timeout_s = 120.0
        mgr._ensure_engine = lambda: engine
        mgr._drain_engine = lambda: None
        mgr.is_failed = lambda room: False
        mgr.is_recv_done = lambda room: False
        mgr.room_has_active_io = lambda room: str(room) in mgr._pulling
        recv = _receiver(mgr)
        polls = []
        thread = threading.Thread(target=lambda: polls.append(recv.poll()))

        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                recv.abort()
                self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
            finally:
                resume.set()
                thread.join(5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(polls, [KVPoll.Transferring])
        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)

    def test_poll_fails_after_waiting_timeout_and_notifies_prefill(self):
        mgr = _decode_manager_stub(waiting_timeout=1)
        recv = _receiver(mgr)
        recv.init_time = time.time() - 2
        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            self.assertEqual(recv.poll(), KVPoll.Failed)
            self.assertEqual(recv.poll(), KVPoll.Failed)
        send_abort.assert_called_once()
        self.assertTrue(recv.abort_notified)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn(ROOM, mgr.failure_records)

    def test_poll_keeps_waiting_before_timeout(self):
        mgr = _decode_manager_stub(waiting_timeout=300)
        recv = _receiver(mgr)
        recv.init_time = time.time()
        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            self.assertEqual(recv.poll(), KVPoll.WaitingForInput)
        send_abort.assert_not_called()

    def test_poll_without_init_time_never_times_out(self):
        mgr = _decode_manager_stub(waiting_timeout=0)
        recv = _receiver(mgr)
        self.assertEqual(recv.poll(), KVPoll.WaitingForInput)

    def test_poll_fails_when_stuck_bootstrapping(self):
        mgr = _decode_manager_stub(request_status={ROOM: KVPoll.Bootstrapping})
        recv = _receiver(mgr)
        del recv.bootstrap_infos  # init() never ran
        recv._handshake_timeout = 1
        recv._handshake_start = time.time() - 2
        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            self.assertEqual(recv.poll(), KVPoll.Failed)
        send_abort.assert_not_called()
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn("KVPoll.Bootstrapping", mgr.failure_records[ROOM])

    def test_handshake_deadline_fails_room_still_waiting_for_kv_pages(self):
        """A room still waiting for decode KV pages past the bootstrap timeout
        fails, since the prefill has given up on it by then."""
        mgr = _decode_manager_stub()
        recv = _receiver(mgr)
        recv._handshake_timeout = 1
        recv._handshake_start = time.time() - 2
        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            self.assertEqual(recv.poll(), KVPoll.Failed)
        send_abort.assert_called_once()
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn("KVPoll.WaitingForInput", mgr.failure_records[ROOM])

    def test_handshake_deadline_stops_once_metadata_is_sent(self):
        mgr = _decode_manager_stub()
        recv = _receiver(mgr)
        recv._handshake_timeout = 1
        recv._handshake_start = time.time() - 2
        recv.init_time = time.time()
        self.assertEqual(recv.poll(), KVPoll.WaitingForInput)

    def test_handshake_keeps_waiting_before_deadline(self):
        mgr = _decode_manager_stub(request_status={ROOM: KVPoll.Bootstrapping})
        recv = _receiver(mgr)
        self.assertEqual(recv.poll(), KVPoll.Bootstrapping)

    def test_engine_failure_notifies_prefill_once(self):
        mgr = _decode_manager_stub(is_failed=lambda room: True)
        recv = _receiver(mgr)
        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            self.assertEqual(recv.poll(), KVPoll.Failed)
            self.assertEqual(recv.poll(), KVPoll.Failed)
        send_abort.assert_called_once()

    def test_node_failure_status_concludes_receiver(self):
        mgr = _decode_manager_stub(request_status={ROOM: KVPoll.Failed})
        recv = _receiver(mgr)
        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            self.assertEqual(recv.poll(), KVPoll.Failed)
        self.assertEqual(recv.conclude_state, KVPoll.Failed)
        send_abort.assert_not_called()


class TestSenderBootstrapTimeout(unittest.TestCase):
    def _sender(self, has_decode_request, bootstrap_timeout, age):
        mgr = SimpleNamespace(
            _drain_engine=lambda: None,
            is_dummy_cp_rank=False,
            is_failed=lambda room: False,
            is_send_done=lambda room: False,
            request_status={ROOM: KVPoll.Bootstrapping},
            failure_records={},
            failure_lock=threading.Lock(),
            bootstrap_timeout=bootstrap_timeout,
            waiting_timeout=300,
            _state_lock=threading.Lock(),
            _aborted={},
            _prefill=PrefillState(
                announce_pages=1,
                decode_addr=({ROOM: ("10.0.0.2", 5555)} if has_decode_request else {}),
            ),
            _failed=set(),
            _native_failed=set(),
            _pipeline=None,
            sender_has_active_io=lambda room: False,
        )
        mgr._fail_room_locked = types.MethodType(RaidenKVManager._fail_room_locked, mgr)
        mgr._decode_asked_locked = types.MethodType(
            RaidenKVManager._decode_asked_locked, mgr
        )
        mgr._decode_requested_locked = types.MethodType(
            RaidenKVManager._decode_requested_locked, mgr
        )
        mgr.finish_bootstrap_if_decode_requested = types.MethodType(
            RaidenKVManager.finish_bootstrap_if_decode_requested, mgr
        )
        mgr.fail_decode_silence = types.MethodType(
            RaidenKVManager.fail_decode_silence, mgr
        )
        mgr._mark_aborted_locked = types.MethodType(
            RaidenKVManager._mark_aborted_locked, mgr
        )
        mgr.check_status = lambda room: mgr.request_status[room]
        mgr.record_failure = lambda room, reason: mgr.failure_records.__setitem__(
            room, reason
        )
        mgr.update_status = lambda room, status: mgr.request_status.__setitem__(
            room, status
        )
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = ROOM
        sender.conclude_state = None
        sender.init_time = time.time() - age
        return sender, mgr

    def test_poll_fails_when_decode_never_asks_for_blocks(self):
        sender, mgr = self._sender(has_decode_request=False, bootstrap_timeout=1, age=2)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn(ROOM, mgr.failure_records)

    def test_poll_keeps_waiting_once_decode_has_asked(self):
        sender, _ = self._sender(has_decode_request=True, bootstrap_timeout=1, age=2)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_poll_stays_bootstrapping_before_timeout(self):
        sender, _ = self._sender(has_decode_request=False, bootstrap_timeout=300, age=0)
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)


def _gate_manager():
    """A prefill manager a real RaidenKVSender can be constructed on."""
    mgr = _prefill_manager()
    mgr.is_dummy_cp_rank = False
    mgr._engine = None
    mgr.bootstrap_timeout = 300
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = {}
    mgr.req_to_decode_prefix_len = {}
    return mgr


def _gate_sender(mgr, room):
    with get_context().override_server_args(dp_size=1):
        return RaidenKVSender(mgr, "unused", room)


def _producer_pipeline(mgr):
    """A prefill-side PipelineTransfer that records requests but never replies."""
    pipe = PipelineTransfer.__new__(PipelineTransfer)
    pipe.manager = mgr
    pipe.producer = True
    pipe.lock = threading.RLock()
    pipe.replies = {}
    pipe.sent = {}
    pipe.failed = set()
    pipe.poll = lambda: (set(), set(), set())
    return pipe


class TestSenderBootstrapGate(unittest.TestCase):
    """A sender must report Bootstrapping until the decode's first request for
    the room arrives. Reporting WaitingForInput earlier lets the bootstrap
    queue compute a prefill whose decode has not reserved KV pages yet."""

    def _sender(self):
        mgr = _gate_manager()
        return _gate_sender(mgr, ROOM), mgr

    def test_sender_stays_bootstrapping_without_a_decode_request(self):
        sender, mgr = self._sender()
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)

    def test_pull_request_ends_bootstrapping(self):
        sender, mgr = self._sender()
        mgr._dispatch(PullReq(ROOM, "10.0.0.2", 5555, 4).to_frames())
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.WaitingForInput)

    def test_reshard_request_ends_bootstrapping(self):
        # A resharding decode sends RESHARD_READY only after the prefill has
        # announced its blocks, so the request itself must open the gate.
        sender, mgr = self._sender()
        mgr._dispatch(
            ReshardReq(ROOM, "10.0.0.2", 5555, "push", [1, 2], 2, [], "").to_frames()
        )
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_pipeline_request_ends_bootstrapping(self):
        sender, mgr = self._sender()
        mgr._pipeline = _producer_pipeline(mgr)
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        mgr._pipeline.on_message(
            [
                MessageTag.PIPELINE_REQ,
                str(ROOM).encode(),
                b"tcp://10.0.0.2:31000",
                b'{"attn_tp_size": 1, "attn_tp_rank": 0}',
            ]
        )
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_tombstoned_room_is_not_admitted_on_a_late_request(self):
        sender, mgr = self._sender()
        with mgr._state_lock:
            mgr._mark_aborted_locked(ROOM)
        mgr._prefill.decode_addr[ROOM] = ("10.0.0.2", 5555)
        self.assertFalse(mgr.finish_bootstrap_if_decode_requested(ROOM))
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)


class TestBootstrapQueueAdmission(unittest.TestCase):
    """The prefill bootstrap queue releases a raiden request only after its
    decode asked for it, whatever the arrival order: an earlier request the
    decode has not reserved pages for keeps waiting behind a later one it has."""

    def _request(self, mgr, room):
        return SimpleNamespace(
            rid=f"req-{room}",
            bootstrap_room=room,
            disagg_kv_sender=_gate_sender(mgr, room),
            origin_input_ids=list(range(64)),
            metadata_buffer_index=0,
            prefill_attempt_count=0,
            is_retracted=False,
            swa_branching_seqlen=None,
            kv=SimpleNamespace(cache_protected_len=0),
            pending_bootstrap=True,
            time_stats=Mock(),
        )

    def _queue(self, reqs):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler_stage_metrics = None
        queue.pp_size = 1
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            tree_cache=None,
            processed_tokens_counter=0,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=32),
        )
        queue.ensure_metadata_buffer = Mock(return_value=True)
        queue.queue = list(reqs)
        return queue

    def _pop(self, queue):
        # Single rank: the group reduction is the senders' own polls.
        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=lambda senders, *_: [s.poll() for s in senders],
            ),
            patch(
                "sglang.srt.disaggregation.prefill.get_disagg",
                return_value=SimpleNamespace(
                    optimistic_prefill_attempts=0,
                    disaggregation_decode_allocation_policy="early",
                ),
            ),
        ):
            return queue.pop_bootstrapped()

    def test_only_the_decode_requested_request_is_released(self):
        mgr = _gate_manager()
        first = self._request(mgr, ROOM)
        second = self._request(mgr, OTHER_ROOM)
        queue = self._queue([first, second])

        self.assertEqual(self._pop(queue), [])

        mgr._dispatch(PullReq(OTHER_ROOM, "10.0.0.2", 5555, 2).to_frames())
        self.assertEqual(self._pop(queue), [second])
        self.assertEqual(queue.queue, [first])
        self.assertTrue(first.pending_bootstrap)
        self.assertFalse(second.pending_bootstrap)

        mgr._dispatch(PullReq(ROOM, "10.0.0.2", 5555, 2).to_frames())
        self.assertEqual(self._pop(queue), [first])
        self.assertEqual(queue.queue, [])
        self.assertFalse(first.pending_bootstrap)


class TestReshardHandshakeWindow(unittest.TestCase):
    """The prefill times its wait on the decode only while the decode owes a
    message: after blocks are announced, until RESHARD_READY."""

    def test_window_spans_request_to_ready(self):
        mgr = _prefill_manager()
        start = 1000.0
        self.assertEqual(mgr.awaiting_decode_since(ROOM, start), start)

        # The decode asked for the room; until the prefill announces its
        # blocks the wait is on local compute, which has no deadline here.
        mgr._prefill.reshard_req[ROOM] = {}
        self.assertIsNone(mgr.awaiting_decode_since(ROOM, start))

        # Blocks announced: the decode now owes RESHARD_READY, timed from here.
        mgr._prefill.reshard_replied.add(ROOM)
        mgr._prefill.reshard_replied_at[ROOM] = 1500.0
        self.assertEqual(mgr.awaiting_decode_since(ROOM, start), 1500.0)

        mgr._prefill.reshard_ready.add(ROOM)
        self.assertIsNone(mgr.awaiting_decode_since(ROOM, start))

    def test_pull_request_closes_the_window(self):
        mgr = _prefill_manager()
        mgr._prefill.decode_addr[ROOM] = ("10.0.0.2", 5555)
        self.assertIsNone(mgr.awaiting_decode_since(ROOM, 1000.0))


class TestAbortIsTerminal(unittest.TestCase):
    def _mgr(self):
        mgr = _prefill_manager()
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        mgr._ensure_engine = Mock()
        mgr._send_multipart_locked = Mock()
        return mgr

    def test_abort_after_finalize_stays_failed(self):
        mgr = self._mgr()
        self.assertTrue(mgr.finalize_send(ROOM, None))
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Transferring)
        mgr._dispatch(_abort_frame(ROOM))
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)

    def test_finalize_after_abort_cannot_revive_the_room(self):
        mgr = self._mgr()
        mgr._dispatch(_abort_frame(ROOM))
        self.assertFalse(mgr.finalize_send(ROOM, None))
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)

    def test_reply_is_never_built_for_an_aborted_room(self):
        mgr = self._mgr()
        mgr._prefill.send_blocks[ROOM] = [1, 2]
        mgr._prefill.reshard_finalized.add(ROOM)
        mgr._prefill.decode_addr[ROOM] = ("10.0.0.2", 5555)
        mgr._dispatch(_abort_frame(ROOM))
        mgr._maybe_reply(ROOM)
        mgr._ensure_engine.assert_not_called()
        mgr._send_multipart_locked.assert_not_called()

    def test_decode_silence_failure_answers_a_racing_pull_request(self):
        mgr = self._mgr()
        self.assertTrue(mgr.fail_decode_silence(ROOM, time.time() - 10, 1))
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        # The decode's address is unknown until it asks, so nothing is sent yet.
        mgr._send_multipart_locked.assert_not_called()
        # The pull request that raced the deadline finds the tombstone.
        mgr._dispatch(PullReq(ROOM, "10.0.0.2", 5555, 4).to_frames())
        self.assertNotIn(ROOM, mgr._prefill.decode_addr)
        mgr._ensure_engine.assert_not_called()
        [(endpoint, frames)] = _sent_frames(mgr)
        self.assertEqual(endpoint, "tcp://10.0.0.2:5555")
        self.assertEqual(RoomFailed.from_frames(frames).room, ROOM)

    def test_frames_after_clear_do_not_rebuild_the_room(self):
        mgr = self._mgr()
        mgr._dispatch(_abort_frame(ROOM))
        mgr.clear_room(ROOM)
        mgr._dispatch(PullReq(ROOM, "10.0.0.2", 5555, 4).to_frames())
        mgr._dispatch(
            ReshardReq(
                ROOM, "10.0.0.2", 5555, "10.0.0.2:31000", [], 4, [], ""
            ).to_frames()
        )
        self.assertNotIn(ROOM, mgr._prefill.decode_addr)
        self.assertNotIn(ROOM, mgr._prefill.reshard_req)
        # Each request is answered with ROOM_FAILED and nothing else.
        sent = _sent_frames(mgr)
        self.assertEqual(len(sent), 2)
        for _, frames in sent:
            self.assertEqual(RoomFailed.from_frames(frames).room, ROOM)


class TestDecodeHonorsRoomFailed(unittest.TestCase):
    def test_pull_for_a_room_the_prefill_dropped_fails_the_decode_room(self):
        """A decode that asks for a room its prefill already failed must fail
        the room on its next poll, not hold its KV pages until the waiting
        timeout."""
        prefill = _prefill_manager()
        prefill.request_status[ROOM] = KVPoll.Bootstrapping
        self.assertTrue(prefill.fail_decode_silence(ROOM, time.time() - 10, 1))
        prefill._dispatch(PullReq(ROOM, "10.0.0.2", 5555, 4).to_frames())
        [(_, reply)] = _sent_frames(prefill)

        decode = _decode_manager()
        decode.request_status[ROOM] = KVPoll.WaitingForInput
        recv = _receiver(decode)
        recv.init_time = time.time()
        self.assertEqual(recv.poll(), KVPoll.WaitingForInput)

        decode._dispatch(reply)
        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            self.assertEqual(recv.poll(), KVPoll.Failed)
        with self.assertRaisesRegex(Exception, "had already failed room"):
            recv.failure_exception()

    def test_notice_for_a_cleared_room_recreates_nothing(self):
        decode = _decode_manager()
        decode._dispatch(RoomFailed(room=ROOM, reason="gone").to_frames())
        self.assertEqual(decode.request_status, {})
        self.assertEqual(decode.failure_records, {})
        self.assertFalse(decode.is_failed(ROOM))


class TestSenderTerminalPrecedence(unittest.TestCase):
    def _sender(self, **overrides):
        fields = dict(
            _drain_engine=lambda: None,
            is_dummy_cp_rank=False,
            is_failed=lambda room: False,
            is_send_done=lambda room: False,
            sender_has_active_io=lambda room: False,
            request_status={ROOM: KVPoll.Failed},
            failure_records={},
            failure_lock=threading.Lock(),
        )
        fields.update(overrides)
        mgr = SimpleNamespace(**fields)
        mgr.check_status = lambda room: mgr.request_status[room]
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = ROOM
        sender.conclude_state = None
        sender.init_time = time.time()
        return sender

    def test_late_native_completion_does_not_outrank_failure(self):
        sender = self._sender(is_send_done=lambda room: True)
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_failure_is_withheld_while_a_push_is_active(self):
        sender = self._sender(sender_has_active_io=lambda room: room == ROOM)
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertIsNone(sender.conclude_state)
        sender.kv_mgr.sender_has_active_io = lambda room: False
        self.assertEqual(sender.poll(), KVPoll.Failed)


class TestOffersRespectTheTombstone(unittest.TestCase):
    def test_offer_after_abort_recreates_nothing(self):
        mgr = _prefill_manager()
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        mgr._prefill.announce_pages = 1
        mgr._dispatch(_abort_frame(ROOM))
        mgr.offer_chunk(ROOM, 0, [1, 2, 3], 8)
        self.assertNotIn(ROOM, mgr._prefill.send_total)
        self.assertNotIn(ROOM, mgr._prefill.pending_offer)

    def test_announce_after_abort_is_dropped(self):
        # The room fails between the offer's snapshot and the announcement.
        mgr = _prefill_manager()
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        mgr._send_multipart_locked = Mock()
        mgr._dispatch(_abort_frame(ROOM))
        mgr._announce_chunk(ROOM, 0, [1, 2, 3], 8)
        mgr._announce_reshard_chunk(ROOM, 0, [1, 2, 3], 8)
        mgr._send_multipart_locked.assert_not_called()


class TestExplicitAbortQuiesces(unittest.TestCase):
    def test_local_abort_withholds_conclusion_while_io_is_active(self):
        mgr = _decode_manager_stub(room_has_active_io=lambda room: room == ROOM)
        recv = _receiver(mgr)
        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            recv.abort()
        self.assertIsNone(recv.conclude_state)
        self.assertIn(ROOM, mgr._aborted)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)

        mgr.room_has_active_io = lambda room: False
        mgr.is_failed = lambda room: True
        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            self.assertEqual(recv.poll(), KVPoll.Failed)


class TestGroupFailureDrainGate(unittest.TestCase):
    """The MIN reduction must not turn one rank's Failed into page release
    on a rank whose transfer still runs."""

    def _gate(self, polls, pollers):
        from unittest.mock import patch as _patch

        from sglang.srt.disaggregation.utils import _gate_group_failure_on_drain

        # Single-process stand-in: the all_reduce over the drain bits is an
        # identity, so the local bits are the group verdict.
        with _patch("sglang.srt.disaggregation.utils.dist") as dist:
            dist.all_reduce = lambda t, op=None, group=None: None
            return _gate_group_failure_on_drain(polls, pollers, [object()])

    def test_failed_is_withheld_while_any_rank_is_undrained(self):
        poller = SimpleNamespace(
            group_failure_requires_drain=True,
            quiesce_group_failure=Mock(),
            drained=lambda: False,
        )
        self.assertEqual(self._gate([KVPoll.Failed], [poller]), [KVPoll.Transferring])
        poller.quiesce_group_failure.assert_called_once_with()

    def test_failed_lands_once_drained(self):
        poller = SimpleNamespace(
            group_failure_requires_drain=True,
            quiesce_group_failure=Mock(),
            drained=lambda: True,
        )
        self.assertEqual(self._gate([KVPoll.Failed], [poller]), [KVPoll.Failed])

    def test_every_rank_quiesces_before_its_drain_bit_is_sampled(self):
        events = []
        poller = SimpleNamespace(
            group_failure_requires_drain=True,
            quiesce_group_failure=lambda: events.append("quiesce"),
            drained=lambda: events.append("drained") or True,
        )

        self._gate([KVPoll.Failed], [poller])

        self.assertEqual(events, ["quiesce", "drained"])

    def test_tp_and_cp_status_reduce_before_quiesce(self):
        from sglang.srt.disaggregation.utils import (
            poll_and_all_reduce_attn_cp_tp_group,
        )

        tp_group, cp_group = object(), object()
        events = []
        poller = SimpleNamespace(
            group_failure_requires_drain=True,
            poll=lambda: KVPoll.Success,
            quiesce_group_failure=lambda: events.append("quiesce"),
            drained=lambda: events.append("drained") or True,
        )
        reduce_count = 0

        def all_reduce(tensor, op=None, group=None):
            nonlocal reduce_count
            reduce_count += 1
            events.append(("reduce", group))
            # A CP peer contributes the failure on the second status leg.
            if reduce_count == 2:
                tensor.fill_(int(KVPoll.Failed))

        with (
            patch("sglang.srt.disaggregation.utils.dist") as dist,
            patch(
                "sglang.srt.disaggregation.utils._poll_with_failure_injection",
                return_value=[int(KVPoll.Success)],
            ),
        ):
            dist.all_reduce = all_reduce
            polls = poll_and_all_reduce_attn_cp_tp_group([poller], cp_group, tp_group)

        self.assertEqual(polls, [KVPoll.Failed])
        self.assertEqual(
            events,
            [
                ("reduce", tp_group),
                ("reduce", cp_group),
                "quiesce",
                "drained",
                ("reduce", tp_group),
                ("reduce", cp_group),
            ],
        )

    def test_decode_metadata_gate_precedes_tp_cp_status_and_drain_votes(self):
        from sglang.srt.disaggregation.utils import poll_and_all_reduce

        tp_group, cp_group = object(), object()
        events = []
        poller = SimpleNamespace(
            group_failure_requires_drain=True,
            poll=lambda: KVPoll.Success,
            quiesce_group_failure=lambda: events.append("quiesce"),
            drained=lambda: events.append("drained") or True,
        )
        reduce_count = 0

        def all_reduce(tensor, op=None, group=None):
            nonlocal reduce_count
            reduce_count += 1
            events.append(("reduce", group))
            if reduce_count == 2:
                tensor.fill_(int(KVPoll.Failed))

        with (
            patch("sglang.srt.disaggregation.utils.dist") as dist,
            patch(
                "sglang.srt.disaggregation.utils._poll_with_failure_injection",
                return_value=[int(KVPoll.Success)],
            ),
            patch(
                "sglang.srt.disaggregation.utils._apply_metadata_gate",
                side_effect=lambda *args: events.append("metadata"),
            ),
        ):
            dist.all_reduce = all_reduce
            polls = poll_and_all_reduce(
                [poller],
                tp_group,
                decode_reqs=[object()],
                metadata_buffers=object(),
                additional_groups=[cp_group],
            )

        self.assertEqual(polls, [KVPoll.Failed])
        self.assertEqual(
            events,
            [
                "metadata",
                ("reduce", tp_group),
                ("reduce", cp_group),
                "quiesce",
                "drained",
                ("reduce", tp_group),
                ("reduce", cp_group),
            ],
        )

    def test_pollers_without_the_capability_keep_todays_behavior(self):
        poller = SimpleNamespace()
        self.assertEqual(self._gate([KVPoll.Failed], [poller]), [KVPoll.Failed])

    def test_raiden_pollers_declare_the_capability(self):
        self.assertTrue(RaidenKVSender.group_failure_requires_drain)
        self.assertTrue(RaidenKVReceiver.group_failure_requires_drain)
        self.assertTrue(callable(RaidenKVSender.quiesce_group_failure))
        self.assertTrue(callable(RaidenKVReceiver.quiesce_group_failure))


class TestGroupFailureQuiescesRaiden(unittest.TestCase):
    def test_sender_tombstones_and_retains_accepted_read_lease(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)
        mgr._send_reads[ROOM] = {f"{ROOM}#0"}
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = ROOM
        sender.conclude_state = KVPoll.Success

        sender.quiesce_group_failure()

        self.assertIn(ROOM, mgr._aborted)
        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertIsNone(sender.conclude_state)
        self.assertNotIn(ROOM, mgr._prefill.send_blocks)
        self.assertEqual(set(mgr._send_reads[ROOM]), {f"{ROOM}#0"})
        self.assertFalse(sender.drained())

    def test_receiver_tombstones_and_overrides_cached_success(self):
        mgr = _decode_manager_stub()
        recv = _receiver(mgr)
        recv.conclude_state = KVPoll.Success

        with patch.object(RaidenKVReceiver, "_send_abort_notification") as notify:
            recv.quiesce_group_failure()

        self.assertIn(ROOM, mgr._aborted)
        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertEqual(recv.conclude_state, KVPoll.Failed)
        notify.assert_called_once_with()


class TestAdmissionRespectsTheTombstone(unittest.TestCase):
    def test_a_tombstoned_room_cannot_admit_a_read(self):
        from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission

        mgr = _prefill_manager()  # provides locks and the tombstone dict
        mgr.disaggregation_mode = DisaggregationMode.DECODE
        mgr._prefill = None
        mgr._decode = DecodeState()
        mgr.chunked_transfer = False
        mgr._engine = Mock()
        mgr._swa_engine = None
        mgr._dynamic_host_staging = True
        mgr._pulling = StagingAdmission(32)
        mgr._full_engine_max_blocks = 8
        mgr._swa_engine_max_blocks = 0
        mgr._decode.local_blocks = {ROOM: [8]}
        mgr._decode.local_swa_blocks = {}
        mgr._decode.remote_blocks = {ROOM: [1]}
        mgr._decode.remote_swa_blocks = {}
        mgr._decode.remote_endpoint = {ROOM: "127.0.0.1:31020"}
        mgr._decode.remote_swa_endpoint = {}
        with mgr._state_lock:
            mgr._mark_aborted_locked(ROOM)
        self.assertFalse(mgr.maybe_start_pull(ROOM))
        mgr._engine.start_read.assert_not_called()
        self.assertEqual(len(mgr._pulling), 0)


class TestSenderLeases(unittest.TestCase):
    def test_registered_read_holds_room_until_poll_stats_settles_it(self):
        mgr = _prefill_manager()
        mgr._engine = Mock()
        mgr._engine.register_read.return_value = True

        mgr._register_sender_read(ROOM, mgr._engine, "7#0", 17, [1, 2])
        self.assertTrue(mgr.sender_has_active_io(ROOM))

        with mgr._state_lock:
            mgr._completed_send.add("7#0")
            mgr._settle_sender_reads_locked()
        self.assertFalse(mgr.sender_has_active_io(ROOM))

    def test_register_read_exception_terminalizes_the_room(self):
        mgr = _prefill_manager()
        mgr.request_status[ROOM] = KVPoll.Transferring
        mgr._engine = Mock()
        mgr._engine.register_read.side_effect = RuntimeError("registration broke")

        with self.assertRaisesRegex(RuntimeError, "registration broke"):
            mgr._register_sender_read(ROOM, mgr._engine, str(ROOM), ROOM, [1, 2])

        self.assertIn(ROOM, mgr._aborted)
        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertIn("register_read failed", mgr.failure_records[ROOM])
        self.assertNotIn(ROOM, mgr._send_reads)
        self.assertFalse(mgr.sender_has_active_io(ROOM))

    def test_registered_read_lease_survives_room_cleanup(self):
        mgr = _prefill_manager()
        mgr._send_reads[ROOM] = {"7#0"}
        mgr._drop_room_state(ROOM)
        self.assertEqual(set(mgr._send_reads[ROOM]), {"7#0"})
        self.assertTrue(mgr.sender_has_active_io(ROOM))

    def test_drop_keeps_the_push_lease_until_the_worker_returns_it(self):
        mgr = _prefill_manager()
        mgr._reshard_pushes_pending[ROOM] = 1
        mgr._drop_room_state(ROOM)
        self.assertTrue(mgr.sender_has_active_io(ROOM))
        # The worker's own unwind hands the lease back.
        with mgr._state_lock:
            left = mgr._reshard_pushes_pending.get(ROOM, 1) - 1
            if left <= 0:
                mgr._reshard_pushes_pending.pop(ROOM, None)
        self.assertFalse(mgr.sender_has_active_io(ROOM))

    def test_a_parked_staging_copy_holds_the_room(self):
        mgr = _prefill_manager()
        mgr._stale_plans = [(7, ROOM, object())]
        self.assertTrue(mgr.sender_has_active_io(ROOM))
        self.assertFalse(mgr.sender_has_active_io(ROOM + 1))

    def test_explicit_sender_abort_waits_for_native_read(self):
        mgr = _prefill_manager()
        mgr.request_status[ROOM] = KVPoll.Transferring
        mgr._send_reads[ROOM] = {"7#0"}
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = ROOM
        sender.conclude_state = None

        sender.abort()

        self.assertIn(ROOM, mgr._aborted)
        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertIsNone(sender.conclude_state)


class TestSenderLeaseSettlement(unittest.TestCase):
    def test_unserved_read_holds_until_native_settlement(self):
        # A read the decode never comes for is expired by raiden at its own
        # deadline; no local timer may drop the lease while the read could
        # still pull the room's pages.
        mgr = _prefill_manager()
        mgr._timeout_s = 0.0
        mgr._send_reads[ROOM] = {str(ROOM)}
        self.assertTrue(mgr.sender_has_active_io(ROOM))
        # raiden reports the read settled: the lease is released.
        mgr._failed.add(str(ROOM))
        mgr._native_failed.add(str(ROOM))
        with mgr._state_lock:
            mgr._settle_sender_reads_locked()
        self.assertFalse(mgr.sender_has_active_io(ROOM))
        self.assertNotIn(ROOM, mgr._send_reads)


class TestPollStatsFailure(unittest.TestCase):
    def test_raising_poll_is_retried_instead_of_killing_the_scheduler(self):
        """A poll_stats raise must not escape the scheduler's poll; raiden
        raises before taking any completion, so the next poll delivers it."""
        mgr = _prefill_manager()
        mgr._engine = Mock()
        mgr.attn_tp_rank = 0
        mgr._engine.poll_stats_with_details.side_effect = [
            RuntimeError("injected complete_read"),
            _polled([str(ROOM)], [], []),
        ]
        mgr._send_reads[ROOM] = {str(ROOM)}

        mgr._drain_engine()
        self.assertTrue(mgr.sender_has_active_io(ROOM))

        mgr._drain_engine()
        self.assertTrue(mgr.is_send_done(ROOM))
        self.assertFalse(mgr.sender_has_active_io(ROOM))

    def test_polls_failing_past_the_transfer_deadline_raise(self):
        # No transfer can settle once polls fail for a whole deadline; a
        # success in between starts the count over.
        mgr = _prefill_manager()
        mgr._engine = Mock()
        mgr.attn_tp_rank = 0
        broken = RuntimeError("injected complete_read")
        mgr._engine.poll_stats_with_details.side_effect = [
            broken,
            _polled([], [], []),
            broken,
            broken,
        ]
        mgr._monotonic = Mock(side_effect=[0.0, 200.0, 321.0])

        mgr._drain_engine()
        mgr._drain_engine()
        mgr._drain_engine()
        with self.assertRaisesRegex(RuntimeError, "transfer deadline"):
            mgr._drain_engine()


class TestSendFinishTime(unittest.TestCase):
    """A room's send finish time comes from raiden, not from the poll."""

    def _manager(self, completed_ns):
        mgr = _prefill_manager()
        mgr._engine = Mock()
        mgr.attn_tp_rank = 0
        mgr.is_dummy_cp_rank = False
        mgr.request_status[ROOM] = KVPoll.Transferring
        mgr._send_reads[ROOM] = {str(ROOM)}
        mgr._engine.poll_stats_with_details.return_value = _polled(
            [str(ROOM)], completed_ns=completed_ns
        )
        return mgr

    def _sender(self, mgr, started_at):
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = ROOM
        sender.conclude_state = None
        sender.init_time = time.time()
        sender._transfer_start_time = started_at
        sender._transfer_metric = KVTransferMetric()
        return sender

    def test_sender_keeps_raidens_finish_time_past_clear(self):
        mgr = self._manager(completed_ns=1_500_000_000)
        sender = self._sender(mgr, started_at=1.0)

        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertAlmostEqual(sender._transfer_metric.transfer_latency_s, 0.5)

        # clear() runs before the stage is stamped; the sender keeps the time.
        sender.clear()
        self.assertEqual(mgr._send_done_at, {})
        self.assertEqual(sender.transfer_finish_time, 1.5)

    def test_hybrid_room_finishes_with_its_later_engine(self):
        mgr = self._manager(completed_ns=2_000_000_000)
        mgr._swa_engine = Mock()
        mgr._swa_engine.poll_stats_with_details.return_value = _polled()
        mgr._drain_engine()
        self.assertIsNone(mgr.send_done_time(ROOM))

        mgr._engine.poll_stats_with_details.return_value = _polled()
        mgr._swa_engine.poll_stats_with_details.return_value = _polled(
            [f"swa:{ROOM}"], completed_ns=3_000_000_000
        )
        mgr._drain_engine()
        self.assertEqual(mgr.send_done_time(ROOM), 3.0)


class TestFailureWaitsForActiveIo(unittest.TestCase):
    def test_failed_room_is_not_concluded_while_its_read_is_in_flight(self):
        from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission

        mgr = _decode_manager_stub(is_failed=lambda room: True)
        mgr._swa_engine = None
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._pipeline = None
        mgr._pulling = StagingAdmission(4)
        mgr._pulling.admit(str(ROOM), 1)
        mgr._swa_req_id = lambda room: f"swa:{room}"
        mgr._pull_settled_locked = types.MethodType(
            RaidenKVManager._pull_settled_locked, mgr
        )
        mgr._paired_swa_read = types.MethodType(RaidenKVManager._paired_swa_read, mgr)
        mgr.room_has_active_io = types.MethodType(
            RaidenKVManager.room_has_active_io, mgr
        )
        recv = _receiver(mgr)
        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            # The read is still in flight: the room's pages must stay owned,
            # so the poll reports Transferring, not Failed.
            self.assertEqual(recv.poll(), KVPoll.Transferring)
            self.assertIsNone(recv.conclude_state)
            send_abort.assert_called_once()

            # The read settles: the failure is delivered.
            mgr._failed.add(str(ROOM))
            mgr._native_failed.add(str(ROOM))
            self.assertEqual(recv.poll(), KVPoll.Failed)
            self.assertEqual(recv.conclude_state, KVPoll.Failed)


class TestReceiverFailureRecords(unittest.TestCase):
    def test_failure_reason_is_consumed_and_cleared(self):
        mgr = _prefill_manager()
        mgr.record_failure(ROOM, "first request's reason")
        receiver = SimpleNamespace(
            kv_mgr=mgr,
            bootstrap_room=ROOM,
            failure_exception=RaidenKVReceiver.failure_exception,
        )
        with self.assertRaisesRegex(Exception, "first request's reason"):
            receiver.failure_exception(receiver)
        self.assertNotIn(ROOM, mgr.failure_records)

        # An unconsumed record leaves with the room, so a later room with
        # the same id cannot surface a stale reason.
        mgr.record_failure(ROOM, "unconsumed reason")
        mgr.clear_room(ROOM)
        self.assertNotIn(ROOM, mgr.failure_records)

    def test_clear_removes_the_failure_record_under_the_room_state_lock(self):
        mgr = _prefill_manager()
        observed = []

        class FailureRecords(dict):
            def pop(records, key, default=None):
                observed.append(mgr._state_lock.locked())
                return super().pop(key, default)

        mgr.failure_records = FailureRecords({ROOM: "unconsumed reason"})
        mgr.clear_room(ROOM)

        self.assertEqual(observed, [True])
        self.assertNotIn(ROOM, mgr.failure_records)


class TestSendRevalidatesTheSocket(unittest.TestCase):
    def test_sender_that_lost_the_eviction_race_uses_a_fresh_socket(self):
        endpoint = "tcp://10.0.0.1:4444"
        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._socket_lock = threading.Lock()
        old_sock, new_sock = Mock(), Mock()
        mgr._socket_cache = {endpoint: old_sock}
        mgr._connect = lambda ep, is_ipv6=False: mgr._socket_cache.setdefault(
            ep, new_sock
        )

        class EvictOnFirstAcquire:
            """The eviction wins the race just as the sender acquires the
            send lock: the cached socket is popped and closed."""

            fired = False

            def __enter__(self):
                if not self.fired:
                    self.fired = True
                    mgr._socket_cache.pop(endpoint).close(linger=0)

            def __exit__(self, *exc):
                return False

        mgr._socket_send_locks = {endpoint: EvictOnFirstAcquire()}
        mgr._send_multipart_locked(endpoint, [b"PING"])

        old_sock.send_multipart.assert_not_called()
        old_sock.close.assert_called_once()
        new_sock.send_multipart.assert_called_once_with([b"PING"])


class TestConnectDefersStaleSocketClose(unittest.TestCase):
    def test_stale_socket_close_waits_for_the_send_lock(self):
        """_connect used to close a monitor-flagged socket under _socket_lock
        alone, which could close it under a sender inside send_multipart on
        another thread; the close must wait on the endpoint send lock."""

        class FakeSocket:
            def __init__(self):
                self.closed = False

            def close(self, linger=None):
                self.closed = True

        class FakeMonitor(FakeSocket):
            def recv_multipart(self, *args):
                return [b"disconnected"]

        endpoint = "tcp://10.0.0.1:4444"
        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._socket_lock = threading.Lock()
        stale, monitor = FakeSocket(), FakeMonitor()
        mgr._socket_cache = {endpoint: stale}
        mgr._monitor_cache = {endpoint: monitor}
        send_lock = threading.Lock()
        mgr._socket_send_locks = {endpoint: send_lock}
        mgr._zmq_ctx = Mock()

        # A send in progress holds the endpoint lock; the close waits for it,
        # while the fresh socket is already published for new senders.
        with send_lock:
            t = threading.Thread(target=mgr._connect, args=(endpoint,))
            t.start()
            t.join(0.2)
            self.assertTrue(t.is_alive())
            self.assertFalse(stale.closed)
            self.assertIs(mgr._socket_cache[endpoint], mgr._zmq_ctx.socket.return_value)
        t.join(2)

        self.assertFalse(t.is_alive())
        self.assertTrue(stale.closed)
        self.assertTrue(monitor.closed)


class TestHeartbeatPrunesRoomTracker(unittest.TestCase):
    def test_heartbeat_success_drops_settled_rooms_from_the_tracker(self):
        """The tracker gains one entry per request and is only consumed when
        a prefill dies; without the heartbeat prune it grows for the process
        lifetime."""
        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr.addr_to_rooms_tracker = {"10.0.0.1:8998": {ROOM, OTHER_ROOM}}
        mgr.request_status = {ROOM: KVPoll.Transferring}

        mgr._on_heartbeat_success("10.0.0.1:8998")

        self.assertEqual(mgr.addr_to_rooms_tracker["10.0.0.1:8998"], {ROOM})


class TestNodeFailureSkipsClearedRooms(unittest.TestCase):
    def test_cleared_room_is_not_recorded_when_check_status_defaults(self):
        """NIXL's check_status returns WaitingForInput for a missing room
        instead of raising, so the node-failure sweep must test membership
        itself or every historical room of the dead prefill would gain a
        stale failure record."""

        class DefaultingManager(CommonKVManager):
            def check_status(self, bootstrap_room):
                return self.request_status.get(bootstrap_room, KVPoll.WaitingForInput)

        mgr = DefaultingManager.__new__(DefaultingManager)
        mgr.connection_lock = threading.Lock()
        mgr.connection_pool = {}
        mgr.prefill_info_table = {}
        mgr.addr_to_rooms_tracker = {"10.0.0.1:8998": {ROOM, OTHER_ROOM}}
        mgr.request_status = {ROOM: KVPoll.Transferring}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()

        mgr._handle_node_failure("10.0.0.1:8998")

        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn(ROOM, mgr.failure_records)
        self.assertNotIn(OTHER_ROOM, mgr.failure_records)
        self.assertNotIn(OTHER_ROOM, mgr.request_status)


class TestNodeFailureEvictsSockets(unittest.TestCase):
    def test_sockets_to_the_dead_prefill_are_closed_under_the_send_lock(self):
        class FakeSocket:
            def __init__(self):
                self.closed = False

            def close(self, linger=None):
                self.closed = True

        endpoint = "tcp://10.0.0.1:4444"
        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr.connection_lock = threading.Lock()
        mgr.connection_pool = {
            "10.0.0.1:8998_0_0_0": [{"rank_ip": "10.0.0.1", "rank_port": 4444}],
            "10.0.0.9:8998_0_0_0": [{"rank_ip": "10.0.0.9", "rank_port": 4444}],
        }
        mgr._socket_lock = threading.Lock()
        send_lock = threading.Lock()
        mgr._socket_send_locks = {endpoint: send_lock}
        sock, monitor = FakeSocket(), FakeSocket()
        other = FakeSocket()
        mgr._socket_cache = {endpoint: sock, "tcp://10.0.0.9:4444": other}
        mgr._monitor_cache = {endpoint: monitor}
        seen = {}

        def base_handler(self_, addr):
            seen["addr"] = addr

        with patch.object(CommonKVManager, "_handle_node_failure", base_handler):
            # A send in progress holds the endpoint lock; eviction waits for it.
            with send_lock:
                t = threading.Thread(
                    target=mgr._handle_node_failure, args=("10.0.0.1:8998",)
                )
                t.start()
                t.join(0.2)
                self.assertTrue(t.is_alive())
                self.assertFalse(sock.closed)
            t.join(2)

        self.assertEqual(seen["addr"], "10.0.0.1:8998")
        self.assertTrue(sock.closed)
        self.assertTrue(monitor.closed)
        self.assertFalse(other.closed)
        self.assertNotIn(endpoint, mgr._socket_cache)
        self.assertNotIn(endpoint, mgr._monitor_cache)

    def test_heartbeat_failure_serializes_with_native_progress_publication(self):
        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._state_lock = threading.Lock()
        mgr._status_lock = threading.Lock()
        mgr.connection_lock = threading.Lock()
        mgr.connection_pool = {}
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}

        failure_published = threading.Event()
        release_failure = threading.Event()

        def base_handler(self_, addr):
            self_.update_status(ROOM, KVPoll.Failed)
            failure_published.set()
            self.assertTrue(release_failure.wait(2))

        with patch.object(CommonKVManager, "_handle_node_failure", base_handler):
            failure = threading.Thread(
                target=mgr._handle_node_failure, args=("10.0.0.1:8998",)
            )
            failure.start()
            self.assertTrue(failure_published.wait(2))

            def publish_progress():
                with mgr._state_lock:
                    mgr.update_status(ROOM, KVPoll.Transferring)

            progress = threading.Thread(target=publish_progress)
            progress.start()
            progress.join(0.2)
            self.assertTrue(progress.is_alive())
            release_failure.set()
            failure.join(2)
            progress.join(2)

        self.assertFalse(failure.is_alive())
        self.assertFalse(progress.is_alive())
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)

    def test_status_transaction_serializes_unlocked_waiting_with_failure(self):
        inspected_waiting = threading.Event()
        release_waiting = threading.Event()
        test_case = self

        class PausingStatus(dict):
            paused = False

            def get(self, room, default=None):
                value = super().get(room, default)
                if threading.current_thread().name == "waiting" and not self.paused:
                    self.paused = True
                    inspected_waiting.set()
                    test_case.assertTrue(release_waiting.wait(2))
                return value

        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._status_lock = threading.Lock()
        mgr.request_status = PausingStatus({ROOM: KVPoll.Bootstrapping})

        waiting = threading.Thread(
            name="waiting",
            target=mgr.update_status,
            args=(ROOM, KVPoll.WaitingForInput),
        )
        waiting.start()
        self.assertTrue(inspected_waiting.wait(2))

        failure = threading.Thread(target=mgr.update_status, args=(ROOM, KVPoll.Failed))
        failure.start()
        failure.join(0.2)
        self.assertTrue(failure.is_alive())
        release_waiting.set()
        waiting.join(2)
        failure.join(2)

        self.assertFalse(waiting.is_alive())
        self.assertFalse(failure.is_alive())
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)

    def test_clear_status_joins_the_status_transaction(self):
        inspected_waiting = threading.Event()
        release_waiting = threading.Event()
        test_case = self

        class PausingStatus(dict):
            paused = False

            def get(self, room, default=None):
                value = super().get(room, default)
                if threading.current_thread().name == "waiting" and not self.paused:
                    self.paused = True
                    inspected_waiting.set()
                    test_case.assertTrue(release_waiting.wait(2))
                return value

        mgr = RaidenKVManager.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr._status_lock = threading.Lock()
        mgr.request_status = PausingStatus({ROOM: KVPoll.Bootstrapping})

        waiting = threading.Thread(
            name="waiting",
            target=mgr.update_status,
            args=(ROOM, KVPoll.WaitingForInput),
        )
        waiting.start()
        self.assertTrue(inspected_waiting.wait(2))

        clear = threading.Thread(target=mgr.clear_status, args=(ROOM,))
        clear.start()
        clear.join(0.2)
        self.assertTrue(clear.is_alive())

        release_waiting.set()
        waiting.join(2)
        clear.join(2)

        self.assertFalse(waiting.is_alive())
        self.assertFalse(clear.is_alive())
        self.assertNotIn(ROOM, mgr.request_status)


ANNOUNCE_ROOM = 21
ANNOUNCE_PAGES = 64


def _armed_for_announce(mgr, room):
    """Top up the prefill fixture with what _announce_chunk needs to reach its send."""
    mgr.chunked_transfer = True
    mgr.attn_tp_rank = 0
    mgr.raiden_endpoint = "tcp://10.0.0.1:40000"
    mgr._ensure_engine = Mock(return_value=Mock())
    mgr._register_sender_read = Mock()
    mgr._prefill.decode_addr[room] = ("10.0.0.9", 5555)
    mgr._prefill.decode_pages[room] = ANNOUNCE_PAGES
    mgr.request_status[room] = KVPoll.WaitingForInput


def _announce(mgr, room=ANNOUNCE_ROOM):
    mgr._announce_chunk(
        room, 0, list(range(1000, 1000 + ANNOUNCE_PAGES)), ANNOUNCE_PAGES
    )


class TestAnnounceSendFailureIsContained(unittest.TestCase):
    """An undeliverable CHUNK announce must fail its own room, not the engine.

    Unguarded, the zmq error propagates out of _announce_chunk to the scheduler
    event loop, whose catch-all handler takes down every TP rank.
    """

    @staticmethod
    def _dead_send(endpoint, parts, is_ipv6=False):
        raise zmq.Again("Resource temporarily unavailable")

    def test_undeliverable_chunk_fails_room_not_engine(self):
        mgr = _prefill_manager()
        _armed_for_announce(mgr, ANNOUNCE_ROOM)
        mgr._send_multipart_locked = self._dead_send

        try:
            _announce(mgr)
        except zmq.ZMQError as exc:
            self.fail(f"zmq error escaped _announce_chunk to the scheduler: {exc!r}")

        self.assertEqual(mgr.check_status(ANNOUNCE_ROOM), KVPoll.Failed)
        self.assertIn(ANNOUNCE_ROOM, mgr.failure_records)
        self.assertIn(ANNOUNCE_ROOM, mgr._aborted)

    def test_failed_announce_leaves_other_rooms_alone(self):
        mgr = _prefill_manager()
        other = ANNOUNCE_ROOM + 1
        for room in (ANNOUNCE_ROOM, other):
            _armed_for_announce(mgr, room)
        mgr.request_status[other] = KVPoll.Transferring
        mgr._send_multipart_locked = self._dead_send

        _announce(mgr)

        self.assertEqual(mgr.check_status(other), KVPoll.Transferring)
        self.assertNotIn(other, mgr._aborted)

    def test_engine_wide_zmq_errors_still_reach_the_scheduler(self):
        for bad in (zmq.ETERM, errno.ENOMEM, errno.ENOTSOCK):
            with self.subTest(errno=bad):
                mgr = _prefill_manager()
                _armed_for_announce(mgr, ANNOUNCE_ROOM)

                def engine_wide(endpoint, parts, is_ipv6=False, _e=bad):
                    raise zmq.ZMQError(_e)

                mgr._send_multipart_locked = engine_wide
                with self.assertRaises(zmq.ZMQError):
                    _announce(mgr)
                self.assertNotEqual(mgr.check_status(ANNOUNCE_ROOM), KVPoll.Failed)

    def test_peer_scoped_errnos_are_absorbed(self):
        for ok in (zmq.EAGAIN, errno.EHOSTUNREACH, errno.EINVAL, errno.EINTR):
            with self.subTest(errno=ok):
                mgr = _prefill_manager()
                _armed_for_announce(mgr, ANNOUNCE_ROOM)

                def peer_scoped(endpoint, parts, is_ipv6=False, _e=ok):
                    raise zmq.ZMQError(_e)

                mgr._send_multipart_locked = peer_scoped
                _announce(mgr)
                self.assertEqual(mgr.check_status(ANNOUNCE_ROOM), KVPoll.Failed)

    def test_absorbed_failure_is_logged(self):
        mgr = _prefill_manager()
        _armed_for_announce(mgr, ANNOUNCE_ROOM)
        mgr._send_multipart_locked = self._dead_send

        with patch("sglang_tpu.disaggregation.raiden.sender.logger.error") as log_error:
            _announce(mgr)
        self.assertEqual(log_error.call_count, 1)
        self.assertIn("CHUNK announce", log_error.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
