"""Unit tests for the raiden transport's failure handling between prefill and decode."""

import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.raiden.conn import (
    RaidenKVManager,
    RaidenKVReceiver,
    RaidenKVSender,
)
from sglang.srt.disaggregation.raiden.serde import MessageTag
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 7
OTHER_ROOM = 8


def _abort_frame(room):
    return [MessageTag.ABORT, str(room).encode("ascii"), b"10.0.0.2", b"5555"]


def _prefill_manager():
    """A prefill-mode RaidenKVManager with its room tables, no sockets or engine."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._swa_engine = None
    mgr._send_blocks = {}
    mgr._send_swa_blocks = {}
    mgr._send_aux = {}
    mgr._decode_addr = {}
    mgr._replied = set()
    mgr._reshard_req = {}
    mgr._reshard_finalized = set()
    mgr._reshard_replied = set()
    mgr._reshard_ready = set()
    mgr._reshard_pushed = set()
    mgr._reshard_replied_at = {}
    mgr._reshard_slot_rooms = [None]
    mgr._reshard_slot_of = {}
    mgr._aborted = {}
    mgr._pulling = None
    # Settle-aware cleanup state shared by every manager mode.
    mgr._pulling_cleared = set()
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr.chunked_transfer = False
    # Chunked-transfer send state swept by _drop_room_state.
    mgr._held_chunks = {}
    mgr._send_total = {}
    mgr._decode_pages = {}
    mgr._reshard_chunks = {}
    mgr._reshard_announced = {}
    mgr._reshard_pushes_pending = {}
    mgr._reshard_final_pushed = set()
    mgr._pending_offer = {}
    mgr._chunk_seq = {}
    return mgr


def _populate_send_state(mgr, room):
    mgr.request_status[room] = KVPoll.Transferring
    mgr._send_blocks[room] = [1, 2, 3]
    mgr._send_swa_blocks[room] = [4]
    mgr._send_aux[room] = b"aux"
    mgr._decode_addr[room] = ("10.0.0.2", 5555)
    mgr._replied.add(room)
    mgr._reshard_finalized.add(room)


class TestPrefillAbortHandling(unittest.TestCase):
    def test_abort_marks_room_failed_and_drops_send_state(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)
        _populate_send_state(mgr, OTHER_ROOM)

        mgr._dispatch(_abort_frame(ROOM))

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertIn(ROOM, mgr.failure_records)
        for table in (
            mgr._send_blocks,
            mgr._send_swa_blocks,
            mgr._send_aux,
            mgr._decode_addr,
            mgr._replied,
            mgr._reshard_finalized,
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
        self.assertIn(ROOM, mgr._send_blocks)

    def test_abort_for_unknown_room_is_ignored(self):
        mgr = _prefill_manager()

        mgr._dispatch(_abort_frame(ROOM))

        self.assertEqual(mgr.request_status, {})
        self.assertEqual(mgr.failure_records, {})

    def test_malformed_abort_is_ignored(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)

        mgr._dispatch([MessageTag.ABORT, b"not-a-room", b"10.0.0.2", b"5555"])
        mgr._dispatch([MessageTag.ABORT])

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Transferring)
        self.assertIn(ROOM, mgr._send_blocks)

    def test_abort_without_sender_address_is_still_honored(self):
        mgr = _prefill_manager()
        _populate_send_state(mgr, ROOM)

        mgr._dispatch([MessageTag.ABORT, str(ROOM).encode("ascii")])

        self.assertEqual(mgr.check_status(ROOM), KVPoll.Failed)
        self.assertNotIn(ROOM, mgr._send_blocks)

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
        mgr._decode_addr[ROOM] = ("10.0.0.2", 5555)
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
        self.assertNotIn(ROOM, mgr._send_blocks)
        self.assertFalse(sender._started)


def _decode_manager_stub(**overrides):
    fields = dict(
        _drain_engine=lambda: None,
        is_failed=lambda room: False,
        is_recv_done=lambda room: False,
        maybe_start_pull=lambda room: False,
        request_status={ROOM: KVPoll.WaitingForInput},
        failure_records={},
        failure_lock=threading.Lock(),
        waiting_timeout=300,
        _state_lock=threading.Lock(),
        _aborted={},
        room_has_active_io=lambda room: False,
    )
    fields.update(overrides)
    mgr = SimpleNamespace(**fields)
    mgr._mark_aborted_locked = types.MethodType(
        RaidenKVManager._mark_aborted_locked, mgr
    )
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
    recv._started = False
    recv.init_time = None
    return recv


class TestReceiverFailurePaths(unittest.TestCase):
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
            request_status={ROOM: KVPoll.WaitingForInput},
            failure_records={},
            failure_lock=threading.Lock(),
            bootstrap_timeout=bootstrap_timeout,
            waiting_timeout=300,
            _state_lock=threading.Lock(),
            _aborted={},
            _decode_addr=({ROOM: ("10.0.0.2", 5555)} if has_decode_request else {}),
            _reshard_ready=set(),
            _reshard_req={},
            _reshard_replied_at={},
            sender_has_active_io=lambda room: False,
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

    def test_poll_keeps_waiting_before_timeout(self):
        sender, _ = self._sender(has_decode_request=False, bootstrap_timeout=300, age=0)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)


class TestReshardHandshakeWindow(unittest.TestCase):
    """The prefill times its wait on the decode only while the decode owes a
    message: after blocks are announced, until RESHARD_READY."""

    def test_window_spans_request_to_ready(self):
        mgr = _prefill_manager()
        start = 1000.0
        self.assertEqual(mgr.awaiting_decode_since(ROOM, start), start)

        # The decode asked for the room; until the prefill announces its
        # blocks the wait is on local compute, which has no deadline here.
        mgr._reshard_req[ROOM] = {}
        self.assertIsNone(mgr.awaiting_decode_since(ROOM, start))

        # Blocks announced: the decode now owes RESHARD_READY, timed from here.
        mgr._reshard_replied.add(ROOM)
        mgr._reshard_replied_at[ROOM] = 1500.0
        self.assertEqual(mgr.awaiting_decode_since(ROOM, start), 1500.0)

        mgr._reshard_ready.add(ROOM)
        self.assertIsNone(mgr.awaiting_decode_since(ROOM, start))

    def test_pull_request_closes_the_window(self):
        mgr = _prefill_manager()
        mgr._decode_addr[ROOM] = ("10.0.0.2", 5555)
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
        mgr._send_blocks[ROOM] = [1, 2]
        mgr._reshard_finalized.add(ROOM)
        mgr._decode_addr[ROOM] = ("10.0.0.2", 5555)
        mgr._dispatch(_abort_frame(ROOM))
        mgr._maybe_reply(ROOM)
        mgr._ensure_engine.assert_not_called()
        mgr._send_multipart_locked.assert_not_called()

    def test_decode_silence_failure_drops_a_racing_pull_request(self):
        mgr = self._mgr()
        self.assertTrue(mgr.fail_decode_silence(ROOM, time.time() - 10, 1))
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        # The pull request that raced the deadline finds the tombstone.
        mgr._dispatch(
            [MessageTag.PULL_REQ, str(ROOM).encode(), b"10.0.0.2", b"5555", b"4"]
        )
        self.assertNotIn(ROOM, mgr._decode_addr)
        mgr._ensure_engine.assert_not_called()

    def test_frames_after_clear_do_not_rebuild_the_room(self):
        mgr = self._mgr()
        mgr._dispatch(_abort_frame(ROOM))
        mgr.clear_room(ROOM)
        mgr._dispatch(
            [MessageTag.PULL_REQ, str(ROOM).encode(), b"10.0.0.2", b"5555", b"4"]
        )
        mgr._dispatch(
            [
                MessageTag.RESHARD_REQ,
                str(ROOM).encode(),
                b"10.0.0.2",
                b"5555",
                b"10.0.0.2:31000",
                b"",
                b"4",
            ]
        )
        self.assertNotIn(ROOM, mgr._decode_addr)
        self.assertNotIn(ROOM, mgr._reshard_req)
        mgr._send_multipart_locked.assert_not_called()


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
        mgr._announce_pages = 1
        mgr._dispatch(_abort_frame(ROOM))
        mgr.offer_chunk(ROOM, 0, [1, 2, 3], 8)
        self.assertNotIn(ROOM, mgr._send_total)
        self.assertNotIn(ROOM, mgr._pending_offer)

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
            group_failure_requires_drain=True, drained=lambda: False
        )
        self.assertEqual(self._gate([KVPoll.Failed], [poller]), [KVPoll.Transferring])

    def test_failed_lands_once_drained(self):
        poller = SimpleNamespace(
            group_failure_requires_drain=True, drained=lambda: True
        )
        self.assertEqual(self._gate([KVPoll.Failed], [poller]), [KVPoll.Failed])

    def test_pollers_without_the_capability_keep_todays_behavior(self):
        poller = SimpleNamespace()
        self.assertEqual(self._gate([KVPoll.Failed], [poller]), [KVPoll.Failed])

    def test_raiden_pollers_declare_the_capability(self):
        self.assertTrue(RaidenKVSender.group_failure_requires_drain)
        self.assertTrue(RaidenKVReceiver.group_failure_requires_drain)


class TestAdmissionRespectsTheTombstone(unittest.TestCase):
    def test_a_tombstoned_room_cannot_admit_a_read(self):
        from sglang.srt.disaggregation.raiden.host_pool import SlotAdmission

        mgr = _prefill_manager()  # provides locks and the tombstone dict
        mgr.chunked_transfer = False
        mgr._engine = Mock()
        mgr._swa_engine = None
        mgr._pulling = SlotAdmission(4)
        mgr._engine_max_blocks = 8
        mgr._local_blocks = {ROOM: [8]}
        mgr._local_swa_blocks = {}
        mgr._remote_blocks = {ROOM: [1]}
        mgr._remote_swa_blocks = {}
        mgr._remote_endpoint = {ROOM: "127.0.0.1:31020"}
        mgr._remote_swa_endpoint = {}
        with mgr._state_lock:
            mgr._mark_aborted_locked(ROOM)
        self.assertFalse(mgr.maybe_start_pull(ROOM))
        mgr._engine.start_read.assert_not_called()
        self.assertEqual(len(mgr._pulling), 0)


class TestSenderLeases(unittest.TestCase):
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


class TestFailureWaitsForActiveIo(unittest.TestCase):
    def test_failed_room_is_not_concluded_while_its_read_is_in_flight(self):
        from sglang.srt.disaggregation.raiden.host_pool import SlotAdmission

        mgr = _decode_manager_stub(is_failed=lambda room: True)
        mgr._swa_engine = None
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._pulling = SlotAdmission(4)
        mgr._pulling.admit(str(ROOM))
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


class TestSendRevalidatesTheSocket(unittest.TestCase):
    def test_sender_that_lost_the_eviction_race_uses_a_fresh_socket(self):
        endpoint = "tcp://10.0.0.1:4444"
        mgr = RaidenKVManager.__new__(RaidenKVManager)
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


class TestNodeFailureEvictsSockets(unittest.TestCase):
    def test_sockets_to_the_dead_prefill_are_closed_under_the_send_lock(self):
        class FakeSocket:
            def __init__(self):
                self.closed = False

            def close(self, linger=None):
                self.closed = True

        endpoint = "tcp://10.0.0.1:4444"
        mgr = RaidenKVManager.__new__(RaidenKVManager)
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


if __name__ == "__main__":
    unittest.main()
