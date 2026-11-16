"""A raiden start_read that raises fails its room, not the decode scheduler;
with SGLANG_RAIDEN_LISTENER_START_READ an announcement starts its read, and
a room that dies cancels its unsettled reads."""

import ast
import inspect
import threading
import time
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden import receiver as receiver_module
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.conn import logger as raiden_logger
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.serde import Blocks, Chunk, RoomFailed
from sglang_tpu.disaggregation.raiden.state import DecodeState

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 53
OTHER_ROOM = 54
# SettledTransfer's failure fields, unset.
_UNREPORTED = dict(elapsed_ns=None, peer=None, code=None, reason=None, phase=None)
_CANCELLED = dict(code="CANCELLED", reason="receive plan unregistered")


def _polled(sent=(), received=(), failed=(), completed_ns=None, details=None):
    """Fake poll_stats_with_details() result for the given req ids; ``details``
    maps a req id to the failure fields raiden reports for it."""

    def settled(ids):
        return [
            SimpleNamespace(
                req_id=r,
                completed_ns=completed_ns,
                **{**_UNREPORTED, **(details or {}).get(r, {})},
            )
            for r in ids
        ]

    return SimpleNamespace(
        sent=settled(sent), received=settled(received), failed=settled(failed)
    )


def _blocks(hybrid, *, state=False, room=ROOM, endpoint="127.0.0.1:31020"):
    """The prefill's closing reply for the room's three pages."""
    return Blocks(
        room=room,
        blocks=[0, 1, 2],
        endpoint=endpoint,
        swa_blocks=[3] if hybrid else [],
        swa_endpoint="127.0.0.1:31022" if hybrid else "",
        aux_blob=b"",
        state_blocks=[5] if state else [],
        state_endpoint="127.0.0.1:31024" if state else "",
    ).to_frames()


def _chunk(index, *, uuid=None):
    """The announcement of chunk ``index``, which holds page ``index``."""
    return Chunk(
        room=ROOM,
        req_id=f"{ROOM}#{index}",
        uuid=(900 + index) if uuid is None else uuid,
        start=index,
        total_pages=3,
        blocks=[index],
        endpoint="127.0.0.1:31020",
    ).to_frames()


def _started(engine):
    return [call.args[0] for call in engine.start_read.call_args_list]


def _decode_manager(*, hybrid, chunked, listener=False, state=False):
    """A decode manager with one registered room, mock engines and no sockets;
    nothing has been announced for the room yet."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr.attn_tp_rank = 0
    mgr.chunked_transfer = chunked
    mgr._listener_start_read = listener
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr.connection_lock = threading.Lock()
    mgr.connection_pool = {}
    mgr.prefill_info_table = {}
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._engine = Mock()
    mgr._swa_engine = Mock() if hybrid else None
    mgr._state_engine = Mock() if state else None
    for engine in (mgr._engine, mgr._swa_engine, mgr._state_engine):
        if engine is not None:
            engine.poll_stats_with_details.return_value = _polled([], [], [])
    mgr._dynamic_host_staging = True
    mgr._pulling = StagingAdmission(64, 8 if hybrid else None, 4 if state else None)
    mgr._pulling_cleared = set()
    mgr._full_engine_max_blocks = 16
    mgr._swa_engine_max_blocks = 4
    if state:
        mgr._state_engine_max_blocks = 1
    mgr._aborted = {}
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._superseded_sender_reads = set()
    mgr._send_reads = {}
    mgr._pull_sessions = {}
    mgr._reshard_plans = {}
    mgr._cancelled_recvs = set()
    mgr._pending_unregisters = {}
    mgr._cleared_plans = set()
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr._timeout_s = 120.0
    mgr._decode.local_blocks[ROOM] = [21, 22, 23]
    mgr._decode.local_swa_blocks[ROOM] = [24] if hybrid else []
    mgr._decode.local_state_rows[ROOM] = [7] if state else []
    return mgr


def _receiver(mgr, *, room=ROOM):
    recv = RaidenKVReceiver.__new__(RaidenKVReceiver)
    recv.kv_mgr = mgr
    recv.bootstrap_room = room
    recv.conclude_state = None
    recv.abort_notified = False
    recv.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 4444}]
    recv._started = False
    recv.init_time = None
    recv._handshake_start = time.time()
    recv._handshake_timeout = 300
    return recv


class TestStartReadFailure(unittest.TestCase):
    def test_raising_read_fails_the_room_instead_of_the_poll(self):
        mgr = _decode_manager(hybrid=False, chunked=False)
        mgr._on_blocks(_blocks(hybrid=False))
        mgr._engine.start_read.side_effect = RuntimeError("injected start_read")
        recv = _receiver(mgr)

        with patch.object(RaidenKVReceiver, "_send_abort_notification") as abort:
            recv.poll()
            self.assertEqual(recv.poll(), KVPoll.Failed)

        abort.assert_called_once()
        self.assertEqual(len(mgr._pulling), 0)
        with self.assertRaisesRegex(Exception, "injected start_read"):
            recv.failure_exception()

    def test_read_raiden_failed_names_its_cause(self):
        """A read raiden failed must name raiden's cause in the room's failure."""
        mgr = _decode_manager(hybrid=False, chunked=False)
        mgr._on_blocks(_blocks(hybrid=False))
        recv = _receiver(mgr)
        rejection = "Remote producer rejected Hybrid Bridge read request"

        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            recv.poll()
            mgr._engine.poll_stats_with_details.return_value = _polled(
                failed=[str(ROOM)],
                details={
                    str(ROOM): dict(
                        code="FAILED_PRECONDITION",
                        phase="connect",
                        peer="127.0.0.1:31020",
                        elapsed_ns=2_000_000,
                        reason=rejection,
                    )
                },
            )
            self.assertEqual(recv.poll(), KVPoll.Failed)

        with self.assertRaises(KVTransferError) as cm:
            recv.failure_exception()
        self.assertEqual(
            cm.exception.failure_reason,
            f"Raiden receive failed: req_id={ROOM} code=FAILED_PRECONDITION "
            f"phase=connect peer=127.0.0.1:31020 elapsed_ms=2.0 reason={rejection}",
        )

    def test_window_read_failure_holds_the_room_while_the_full_read_runs(self):
        mgr = _decode_manager(hybrid=True, chunked=False)
        mgr._on_blocks(_blocks(hybrid=True))
        mgr._swa_engine.start_read.side_effect = RuntimeError("injected")
        recv = _receiver(mgr)

        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            recv.poll()
            # The full-attention read was issued and still writes the room's
            # pages, so the failure is withheld and the staging stays claimed.
            self.assertEqual(recv.poll(), KVPoll.Transferring)
            self.assertEqual(len(mgr._pulling), 1)

            mgr._engine.poll_stats_with_details.return_value = _polled(
                [], [str(ROOM)], []
            )
            self.assertEqual(recv.poll(), KVPoll.Failed)

        self.assertEqual(len(mgr._pulling), 0)

    def test_raising_closing_chunk_read_never_issues_the_window_read(self):
        for listener in (False, True):
            with self.subTest(listener=listener):
                mgr = _decode_manager(hybrid=True, chunked=True, listener=listener)
                mgr._engine.start_read.side_effect = RuntimeError("injected")
                mgr._on_blocks(_blocks(hybrid=True))
                self.assertEqual(mgr._engine.start_read.call_count, int(listener))
                mgr.maybe_start_pull(ROOM)

                mgr._engine.start_read.assert_called_once()
                mgr._swa_engine.start_read.assert_not_called()
                self.assertTrue(mgr.is_failed(ROOM))
                self.assertFalse(mgr.room_has_active_io(ROOM))
                self.assertEqual(len(mgr._pulling), 0)

    def test_every_receiver_start_read_is_guarded(self):
        # A bare engine.start_read in the decode poll path brings back the
        # scheduler crash; every call must go through _start_read_or_fail.
        tree = ast.parse(inspect.getsource(receiver_module))
        guard = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_start_read_or_fail"
        )
        guarded = {id(node) for node in ast.walk(guard)}
        bare = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start_read"
            and id(node) not in guarded
        ]
        self.assertEqual(bare, [])


class TestListenerStartsReads(unittest.TestCase):
    def test_announcement_starts_the_read_only_when_enabled(self):
        for listener in (False, True):
            with self.subTest(listener=listener):
                mgr = _decode_manager(hybrid=True, chunked=True, listener=listener)
                recv = _receiver(mgr)

                mgr._on_chunk(_chunk(0))
                # One read in flight per room: chunk 1 waits for chunk 0.
                mgr._on_chunk(_chunk(1))
                self.assertEqual(
                    _started(mgr._engine), [f"{ROOM}#0"] if listener else []
                )
                recv.poll()
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    received=[f"{ROOM}#0"]
                )
                recv.poll()
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    received=[f"{ROOM}#0", f"{ROOM}#1"]
                )
                recv.poll()
                mgr._on_blocks(_blocks(hybrid=True))
                self.assertEqual(
                    _started(mgr._swa_engine), [f"swa:{ROOM}"] if listener else []
                )
                recv.poll()
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    received=[f"{ROOM}#0", f"{ROOM}#1", str(ROOM)]
                )
                mgr._swa_engine.poll_stats_with_details.return_value = _polled(
                    received=[f"swa:{ROOM}"]
                )
                self.assertEqual(recv.poll(), KVPoll.Success)

                self.assertEqual(
                    _started(mgr._engine), [f"{ROOM}#0", f"{ROOM}#1", str(ROOM)]
                )
                self.assertEqual(_started(mgr._swa_engine), [f"swa:{ROOM}"])

    def test_failed_room_starts_no_read_from_either_thread(self):
        mgr = _decode_manager(hybrid=False, chunked=True, listener=True)
        # A failure the poll has not acted on yet.
        mgr._failed.add(f"{ROOM}#0")

        mgr._on_blocks(_blocks(hybrid=False))
        self.assertFalse(mgr.maybe_start_pull(ROOM))

        mgr._engine.start_read.assert_not_called()
        self.assertEqual(len(mgr._pulling), 0)

    def test_poll_during_a_listener_start_does_not_issue_it_again(self):
        mgr = _decode_manager(hybrid=False, chunked=True, listener=True)
        recv = _receiver(mgr)
        entered, resume = threading.Event(), threading.Event()

        def blocking_start(*_args):
            entered.set()
            resume.wait(5)

        mgr._engine.start_read.side_effect = blocking_start
        listener = threading.Thread(
            target=mgr._on_blocks, args=(_blocks(hybrid=False),)
        )
        listener.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(recv.poll(), KVPoll.Transferring)
        finally:
            resume.set()
            listener.join(5)

        mgr._engine.start_read.assert_called_once()


class TestRaidenDecodeReceiveCancellation(CustomTestCase):
    def test_pull_abort_cancels_all_engines_once_and_fences_until_settled(self):
        """Aborting an in-flight hybrid+state pull cancels all three engines once,
        fences pages until settled, and suppresses only CANCELLED logs at DEBUG."""
        mgr = _decode_manager(hybrid=True, chunked=False, state=True)
        mgr._on_blocks(_blocks(hybrid=True, state=True))
        # Older bindings surface NotFound as "not registered"; swallow without warning.
        mgr._state_engine.unregister_active_plan.side_effect = RuntimeError(
            f"Plan with UUID {ROOM} is not registered"
        )
        recv = _receiver(mgr)

        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            self.assertEqual(recv.poll(), KVPoll.Transferring)
            self.assertEqual(len(mgr._pulling), 1)

            with patch.object(raiden_logger, "warning") as log_warn:
                recv.abort()
            log_warn.assert_not_called()
            for eng in (mgr._engine, mgr._swa_engine, mgr._state_engine):
                eng.unregister_active_plan.assert_called_once_with(ROOM)
            self.assertTrue(mgr.room_has_active_io(ROOM))
            self.assertFalse(recv.drained())
            self.assertEqual(recv.poll(), KVPoll.Transferring)
            for eng in (mgr._engine, mgr._swa_engine, mgr._state_engine):
                eng.unregister_active_plan.assert_called_once_with(ROOM)

            mgr._engine.poll_stats_with_details.return_value = _polled(
                failed=[str(ROOM)], details={str(ROOM): _CANCELLED}
            )
            mgr._swa_engine.poll_stats_with_details.return_value = _polled(
                failed=[f"swa:{ROOM}"], details={f"swa:{ROOM}": _CANCELLED}
            )
            with (
                patch.object(raiden_logger, "error") as log_err,
                patch.object(raiden_logger, "debug") as log_dbg,
            ):
                self.assertEqual(recv.poll(), KVPoll.Transferring)
            log_err.assert_not_called()
            self.assertTrue(
                any(
                    "receive plan unregistered" in str(c.args)
                    for c in log_dbg.call_args_list
                )
            )
            self.assertTrue(mgr.room_has_active_io(ROOM))
            self.assertEqual(len(mgr._pulling), 1)

            mgr._engine.poll_stats_with_details.return_value = _polled()
            mgr._swa_engine.poll_stats_with_details.return_value = _polled()
            mgr._state_engine.poll_stats_with_details.return_value = _polled(
                failed=[f"state:{ROOM}"],
                details={
                    f"state:{ROOM}": dict(code="DEADLINE_EXCEEDED", reason="timed out")
                },
            )
            with patch.object(raiden_logger, "error") as log_err:
                self.assertEqual(recv.poll(), KVPoll.Failed)
            log_err.assert_called_once()
            self.assertTrue(recv.drained())
            self.assertEqual(len(mgr._pulling), 0)

        with self.assertRaises(KVTransferError) as cm:
            recv.failure_exception()
        self.assertEqual(cm.exception.failure_reason, "Aborted by AbortReq.")
        mgr.clear_room(ROOM)
        for eng in (mgr._engine, mgr._swa_engine, mgr._state_engine):
            eng.unregister_active_plan.assert_called_once_with(ROOM)
        self.assertEqual(mgr._cancelled_recvs, set())
        self.assertEqual(mgr._pull_sessions, {})

    def test_chunked_pull_and_state_only_prefix_hit_cancel_exact_uuids(self):
        """Chunked pulls cancel their random chunk uuid, and state-only prefix-hit
        pulls cancel only the state engine."""
        with self.subTest("chunked_pull"):
            mgr = _decode_manager(hybrid=False, chunked=True)
            chunk_uuid = 9876543210123
            mgr._on_chunk(_chunk(0, uuid=chunk_uuid))
            recv = _receiver(mgr)
            with patch.object(RaidenKVReceiver, "_send_abort_notification"):
                self.assertEqual(recv.poll(), KVPoll.Transferring)
                recv.abort()
                mgr._engine.unregister_active_plan.assert_called_once_with(chunk_uuid)
                self.assertEqual(recv.poll(), KVPoll.Transferring)
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    failed=[f"{ROOM}#0"], details={f"{ROOM}#0": _CANCELLED}
                )
                self.assertEqual(recv.poll(), KVPoll.Failed)
            with self.assertRaises(KVTransferError) as cm:
                recv.failure_exception()
            self.assertEqual(cm.exception.failure_reason, "Aborted by AbortReq.")

        with self.subTest("state_only_prefix_hit"):
            mgr = _decode_manager(hybrid=True, chunked=False, state=True)
            mgr._decode.local_blocks[ROOM] = []
            mgr._decode.local_swa_blocks[ROOM] = []
            mgr._on_blocks(_blocks(hybrid=True, state=True))
            recv = _receiver(mgr)
            with patch.object(RaidenKVReceiver, "_send_abort_notification"):
                self.assertEqual(recv.poll(), KVPoll.Transferring)
                mgr._engine.start_read.assert_not_called()
                mgr._swa_engine.start_read.assert_not_called()
                mgr._state_engine.start_read.assert_called_once()
                recv.abort()
                mgr._engine.unregister_active_plan.assert_not_called()
                mgr._swa_engine.unregister_active_plan.assert_not_called()
                mgr._state_engine.unregister_active_plan.assert_called_once_with(ROOM)
                mgr._state_engine.poll_stats_with_details.return_value = _polled(
                    failed=[f"state:{ROOM}"],
                    details={f"state:{ROOM}": _CANCELLED},
                )
                self.assertEqual(recv.poll(), KVPoll.Failed)
                self.assertEqual(len(mgr._pulling), 0)

    def test_room_failed_node_failure_and_sibling_fault_cancel_keeping_the_reason(
        self,
    ):
        """RoomFailed, prefill node death, and sibling native faults cancel
        unsettled receives best-effort without overwriting the primary reason."""
        with self.subTest("room_failed"):
            mgr = _decode_manager(hybrid=False, chunked=False)
            mgr._on_blocks(_blocks(hybrid=False))
            recv = _receiver(mgr)
            with patch.object(RaidenKVReceiver, "_send_abort_notification"):
                self.assertEqual(recv.poll(), KVPoll.Transferring)
                mgr._on_room_failed(
                    RoomFailed(room=ROOM, reason="prefill died").to_frames()
                )
                mgr._engine.unregister_active_plan.assert_called_once_with(ROOM)
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    failed=[str(ROOM)], details={str(ROOM): _CANCELLED}
                )
                self.assertEqual(recv.poll(), KVPoll.Failed)
            with self.assertRaises(KVTransferError) as cm:
                recv.failure_exception()
            self.assertEqual(cm.exception.failure_reason, "prefill died")

        with self.subTest("node_failure"):
            mgr = _decode_manager(hybrid=False, chunked=False)
            mgr.request_status[OTHER_ROOM] = KVPoll.WaitingForInput
            mgr._decode.local_blocks[OTHER_ROOM] = [31, 32, 33]
            mgr.addr_to_rooms_tracker = {
                "10.0.0.1:8998": {ROOM},
                "10.0.0.9:8998": {OTHER_ROOM},
            }
            for r, ep in ((ROOM, "10.0.0.1:31020"), (OTHER_ROOM, "10.0.0.9:31020")):
                mgr._on_blocks(_blocks(hybrid=False, room=r, endpoint=ep))
            recv_dead = _receiver(mgr, room=ROOM)
            recv_live = _receiver(mgr, room=OTHER_ROOM)
            with patch.object(RaidenKVReceiver, "_send_abort_notification"):
                self.assertEqual(recv_dead.poll(), KVPoll.Transferring)
                self.assertEqual(recv_live.poll(), KVPoll.Transferring)
                mgr._handle_node_failure("10.0.0.1:8998")
                mgr._engine.unregister_active_plan.assert_called_once_with(ROOM)
                self.assertIn(OTHER_ROOM, mgr._pull_sessions)

        with self.subTest("sibling_native_fault"):
            mgr = _decode_manager(hybrid=True, chunked=False)
            mgr._on_blocks(_blocks(hybrid=True))
            mgr._swa_engine.unregister_active_plan.side_effect = RuntimeError(
                "transport wedged"
            )
            recv = _receiver(mgr)
            with patch.object(RaidenKVReceiver, "_send_abort_notification"):
                self.assertEqual(recv.poll(), KVPoll.Transferring)
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    failed=[str(ROOM)],
                    details={str(ROOM): dict(code="INTERNAL", reason="H2D fault")},
                )
                with patch.object(raiden_logger, "warning") as log_warn:
                    self.assertEqual(recv.poll(), KVPoll.Transferring)
                    mgr._engine.poll_stats_with_details.return_value = _polled()
                    self.assertEqual(recv.poll(), KVPoll.Transferring)
                log_warn.assert_called_once()
                mgr._swa_engine.unregister_active_plan.assert_called_once_with(ROOM)
                mgr._engine.unregister_active_plan.assert_not_called()
                mgr._swa_engine.poll_stats_with_details.return_value = _polled(
                    failed=[f"swa:{ROOM}"],
                    details={
                        f"swa:{ROOM}": dict(
                            code="DEADLINE_EXCEEDED", reason="timed out"
                        )
                    },
                )
                self.assertEqual(recv.poll(), KVPoll.Failed)
                self.assertEqual(len(mgr._pulling), 0)
                self.assertEqual(mgr._cancelled_recvs, set())
                self.assertEqual(mgr._pending_unregisters, {})
            with self.assertRaises(KVTransferError) as cm:
                recv.failure_exception()
            self.assertIn("H2D fault", cm.exception.failure_reason)
            mgr.clear_room(ROOM)
            mgr._swa_engine.unregister_active_plan.assert_called_once_with(ROOM)

    def test_companion_start_read_error_cancels_started_and_respects_clear(self):
        """A companion start_read failure cancels already-started sibling reads,
        preserves an earlier abort reason, and skips recording if cleared."""
        for mode in ("live", "pre_aborted", "cleared"):
            with self.subTest(mode=mode):
                mgr = _decode_manager(hybrid=True, chunked=False, state=True)
                mgr._on_blocks(_blocks(hybrid=True, state=True))
                recv = _receiver(mgr)

                def swa_raises(*_args, _mgr=mgr, _recv=recv, _mode=mode):
                    if _mode == "pre_aborted":
                        _recv.abort()
                    elif _mode == "cleared":
                        _mgr.clear_room(ROOM)
                    raise RuntimeError("bad SWA peer")

                mgr._swa_engine.start_read.side_effect = swa_raises
                with patch.object(RaidenKVReceiver, "_send_abort_notification"):
                    if mode == "cleared":
                        self.assertFalse(mgr.maybe_start_pull(ROOM))
                        mgr._engine.unregister_active_plan.assert_called_once_with(ROOM)
                        mgr._state_engine.start_read.assert_not_called()
                        self.assertNotIn(str(ROOM), mgr._failed)
                        self.assertEqual(mgr.failure_records, {})
                    else:
                        self.assertEqual(recv.poll(), KVPoll.Transferring)
                        mgr._engine.unregister_active_plan.assert_called_once_with(ROOM)
                        mgr._state_engine.start_read.assert_not_called()
                        expected = (
                            "Aborted by AbortReq."
                            if mode == "pre_aborted"
                            else "bad SWA peer"
                        )
                        self.assertIn(expected, mgr.failure_records[ROOM])
                        mgr._engine.poll_stats_with_details.return_value = _polled(
                            failed=[str(ROOM)], details={str(ROOM): _CANCELLED}
                        )
                        self.assertEqual(recv.poll(), KVPoll.Failed)
                        self.assertEqual(len(mgr._pulling), 0)

    def test_start_read_unlock_races_and_fast_settle_room_reuse(self):
        """A room dying inside unlocked start_read cancels on return and retires
        cleanly; a read settling and clearing mid-call does not taint reuse."""
        cases = (
            ("abort", True, False),
            ("node_failure", False, False),
            ("node_failure_started", True, True),
            ("clear", False, False),
        )
        for death, hybrid, chunked in cases:
            with self.subTest(death=death):
                mgr = _decode_manager(hybrid=hybrid, chunked=chunked, state=hybrid)
                mgr.addr_to_rooms_tracker = {"10.0.0.1:8998": {ROOM}}
                recv = _receiver(mgr)
                if chunked:
                    recv._started = True
                    mgr._on_chunk(_chunk(0))
                    expected_uuid, settled_id = 900, f"{ROOM}#0"
                else:
                    mgr._on_blocks(_blocks(hybrid=hybrid, state=hybrid))
                    expected_uuid, settled_id = ROOM, str(ROOM)
                kill = {
                    "abort": recv.abort,
                    "node_failure": partial(mgr._handle_node_failure, "10.0.0.1:8998"),
                    "node_failure_started": partial(
                        mgr._handle_node_failure, "10.0.0.1:8998"
                    ),
                    "clear": partial(mgr.clear_room, ROOM),
                }[death]
                mgr._engine.start_read.side_effect = lambda *_a, _k=kill: _k()
                with patch.object(
                    RaidenKVReceiver, "_send_abort_notification"
                ) as notify_abort:
                    if death == "clear":
                        mgr.maybe_start_pull(ROOM)
                        self.assertEqual(mgr.failure_records, {})
                    else:
                        self.assertEqual(recv.poll(), KVPoll.Transferring)
                if death == "abort":
                    notify_abort.assert_called_once()
                else:
                    notify_abort.assert_not_called()
                mgr._engine.unregister_active_plan.assert_called_once_with(
                    expected_uuid
                )
                if hybrid and not chunked:
                    mgr._swa_engine.start_read.assert_not_called()
                    mgr._state_engine.start_read.assert_not_called()
                self.assertEqual(len(mgr._pulling), 1)
                detail = (
                    dict(code="DEADLINE_EXCEEDED", reason="timed out")
                    if death == "clear"
                    else _CANCELLED
                )
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    failed=[settled_id], details={settled_id: detail}
                )
                mgr._drain_engine()
                if death == "clear":
                    self.assertEqual(mgr.failure_records, {})
                self.assertEqual(len(mgr._pulling), 0)
                self.assertEqual(mgr._cancelled_recvs, set())

        with self.subTest("fast_settle_clear_and_room_reuse"):
            mgr = _decode_manager(hybrid=False, chunked=True)
            stale_uuid = 777001
            mgr._on_chunk(_chunk(0, uuid=stale_uuid))

            def settle_clear_and_reuse(*_args):
                mgr._engine.poll_stats_with_details.return_value = _polled(
                    received=[f"{ROOM}#0"]
                )
                mgr._drain_engine()
                mgr._engine.poll_stats_with_details.return_value = _polled()
                mgr.clear_room(ROOM)
                mgr.request_status[ROOM] = KVPoll.WaitingForInput
                mgr._decode.local_blocks[ROOM] = [41, 42, 43]

            mgr._engine.start_read.side_effect = settle_clear_and_reuse
            mgr.maybe_start_pull(ROOM)
            mgr._engine.start_read.assert_called_once()
            reused_recv = _receiver(mgr)
            with patch.object(RaidenKVReceiver, "_send_abort_notification"):
                reused_recv.abort()
            mgr._engine.unregister_active_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
