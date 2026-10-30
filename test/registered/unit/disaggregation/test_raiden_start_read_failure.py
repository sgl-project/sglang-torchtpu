"""A raiden start_read that raises fails its room, not the decode scheduler;
with SGLANG_RAIDEN_LISTENER_START_READ an announcement starts its read."""

import ast
import inspect
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import receiver as receiver_module
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.serde import Blocks, Chunk
from sglang_tpu.disaggregation.raiden.state import DecodeState

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 53
# SettledTransfer's failure fields, unset.
_UNREPORTED = dict(elapsed_ns=None, peer=None, code=None, reason=None, phase=None)


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


def _blocks(hybrid):
    """The prefill's closing reply for the room's three pages."""
    return Blocks(
        room=ROOM,
        blocks=[0, 1, 2],
        endpoint="127.0.0.1:31020",
        swa_blocks=[3] if hybrid else [],
        swa_endpoint="127.0.0.1:31022" if hybrid else "",
        aux_blob=b"",
    ).to_frames()


def _chunk(index):
    """The announcement of chunk ``index``, which holds page ``index``."""
    return Chunk(
        room=ROOM,
        req_id=f"{ROOM}#{index}",
        uuid=900 + index,
        start=index,
        total_pages=3,
        blocks=[index],
        endpoint="127.0.0.1:31020",
    ).to_frames()


def _started(engine):
    return [call.args[0] for call in engine.start_read.call_args_list]


def _decode_manager(*, hybrid, chunked, listener=False):
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
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._engine = Mock()
    mgr._swa_engine = Mock() if hybrid else None
    for engine in (mgr._engine, mgr._swa_engine):
        if engine is not None:
            engine.poll_stats_with_details.return_value = _polled([], [], [])
    mgr._dynamic_host_staging = True
    mgr._pulling = StagingAdmission(64, 8 if hybrid else None)
    mgr._pulling_cleared = set()
    mgr._full_engine_max_blocks = 16
    mgr._swa_engine_max_blocks = 4
    mgr._aborted = {}
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._superseded_sender_reads = set()
    mgr._send_reads = {}
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr._timeout_s = 120.0
    mgr._decode.local_blocks[ROOM] = [21, 22, 23]
    mgr._decode.local_swa_blocks[ROOM] = [24] if hybrid else []
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


if __name__ == "__main__":
    unittest.main()
