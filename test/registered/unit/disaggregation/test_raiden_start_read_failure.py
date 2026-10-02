"""A raiden start_read that raises fails its room, not the decode scheduler."""

import ast
import inspect
import threading
import unittest
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.raiden import receiver as receiver_module
from sglang.srt.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang.srt.disaggregation.raiden.host_pool import StagingAdmission
from sglang.srt.disaggregation.raiden.serde import Blocks
from sglang.srt.disaggregation.raiden.state import DecodeState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 53


def _decode_manager(*, hybrid, chunked):
    """A decode manager with its room tables and mock engines, no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr.attn_tp_rank = 0
    mgr.chunked_transfer = chunked
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._engine = Mock()
    mgr._swa_engine = Mock() if hybrid else None
    for engine in (mgr._engine, mgr._swa_engine):
        if engine is not None:
            engine.poll_stats.return_value = ([], [], [])
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
    mgr._on_blocks(
        Blocks(
            room=ROOM,
            blocks=[0, 1, 2],
            endpoint="127.0.0.1:31020",
            swa_blocks=[3] if hybrid else [],
            swa_endpoint="127.0.0.1:31022" if hybrid else "",
            aux_blob=b"",
        ).to_frames()
    )
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


class TestStartReadFailure(unittest.TestCase):
    def test_raising_read_fails_the_room_instead_of_the_poll(self):
        mgr = _decode_manager(hybrid=False, chunked=False)
        mgr._engine.start_read.side_effect = RuntimeError("injected start_read")
        recv = _receiver(mgr)

        with patch.object(RaidenKVReceiver, "_send_abort_notification") as abort:
            recv.poll()
            self.assertEqual(recv.poll(), KVPoll.Failed)

        abort.assert_called_once()
        self.assertEqual(len(mgr._pulling), 0)
        with self.assertRaisesRegex(Exception, "injected start_read"):
            recv.failure_exception()

    def test_window_read_failure_holds_the_room_while_the_full_read_runs(self):
        mgr = _decode_manager(hybrid=True, chunked=False)
        mgr._swa_engine.start_read.side_effect = RuntimeError("injected")
        recv = _receiver(mgr)

        with patch.object(RaidenKVReceiver, "_send_abort_notification"):
            recv.poll()
            # The full-attention read was issued and still writes the room's
            # pages, so the failure is withheld and the staging stays claimed.
            self.assertEqual(recv.poll(), KVPoll.Transferring)
            self.assertEqual(len(mgr._pulling), 1)

            mgr._engine.poll_stats.return_value = ([], [str(ROOM)], [])
            self.assertEqual(recv.poll(), KVPoll.Failed)

        self.assertEqual(len(mgr._pulling), 0)

    def test_raising_closing_chunk_read_never_issues_the_window_read(self):
        mgr = _decode_manager(hybrid=True, chunked=True)
        mgr._engine.start_read.side_effect = RuntimeError("injected")

        self.assertFalse(mgr.maybe_start_pull(ROOM))

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


if __name__ == "__main__":
    unittest.main()
