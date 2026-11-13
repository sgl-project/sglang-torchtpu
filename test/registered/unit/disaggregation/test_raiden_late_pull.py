"""A decode read that outwaited the transfer deadline fails instead of pulling."""

import threading
import time
import unittest
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.state import DecodeState

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 7
OTHER_ROOM = 8
TIMEOUT_S = 120.0


def _decode_manager(engine, chunked=False):
    """A decode-mode RaidenKVManager with one staging slot and a stub engine."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr.attn_tp_rank = 0
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._drain_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._aborted = {}
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._completed_recv = set()
    mgr._pulling_cleared = set()
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr._swa_engine = None
    mgr._decode.local_blocks = {ROOM: [21, 22]}
    mgr._decode.remote_blocks = {ROOM: [11, 12]}
    mgr._decode.remote_endpoint = {ROOM: "peer"}
    mgr._decode.remote_total = {ROOM: 2}
    mgr._timeout_s = TIMEOUT_S
    mgr.chunked_transfer = chunked
    mgr._full_engine_max_blocks = 4
    mgr._swa_engine_max_blocks = 0
    mgr._dynamic_host_staging = True
    mgr._pulling = StagingAdmission(4)
    mgr._ensure_engine = lambda: engine
    mgr._drain_engine = lambda: None
    mgr.is_recv_done = lambda room: False
    mgr.room_has_active_io = lambda room: False
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


class TestLatePullIsNotIssued(unittest.TestCase):
    def test_whole_read_past_deadline_fails_without_a_pull(self):
        engine = Mock()
        mgr = _decode_manager(engine)
        mgr._decode.announced_at[str(ROOM)] = time.monotonic() - TIMEOUT_S - 10

        self.assertFalse(mgr.maybe_start_pull(ROOM))

        engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_failed(ROOM))
        self.assertIn("transfer deadline", mgr.failure_records[ROOM])
        # The room stayed out of the slot: nothing was admitted.
        self.assertEqual(len(mgr._pulling), 0)

    def test_whole_read_within_deadline_is_pulled(self):
        engine = Mock()
        mgr = _decode_manager(engine)
        mgr._decode.announced_at[str(ROOM)] = time.monotonic()

        self.assertTrue(mgr.maybe_start_pull(ROOM))

        engine.start_read.assert_called_once()
        self.assertFalse(mgr.is_failed(ROOM))

    def test_chunk_past_deadline_fails_the_room(self):
        engine = Mock()
        mgr = _decode_manager(engine, chunked=True)
        mgr._decode.chunk_queue[ROOM] = [(f"{ROOM}#0", 900, [0, 1], [11, 12], "peer")]
        mgr._decode.announced_at[f"{ROOM}#0"] = time.monotonic() - TIMEOUT_S - 10

        self.assertFalse(mgr.maybe_start_pull(ROOM))

        engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_failed(ROOM))
        self.assertIn("transfer deadline", mgr.failure_records[ROOM])

    def test_expired_room_concludes_failed_and_tells_the_prefill(self):
        engine = Mock()
        mgr = _decode_manager(engine)
        mgr._decode.announced_at[str(ROOM)] = time.monotonic() - TIMEOUT_S - 10
        recv = _receiver(mgr)

        with patch.object(RaidenKVReceiver, "_send_abort_notification") as send_abort:
            first = recv.poll()
            second = recv.poll()

        self.assertEqual((first, second), (KVPoll.WaitingForInput, KVPoll.Failed))
        send_abort.assert_called_once()
        engine.start_read.assert_not_called()
        with self.assertRaisesRegex(Exception, "transfer deadline"):
            recv.failure_exception()

    def test_dropping_the_room_forgets_its_announcements(self):
        mgr = _decode_manager(Mock())
        mgr._decode.announced_at = {
            str(ROOM): 1.0,
            f"{ROOM}#0": 1.0,
            str(OTHER_ROOM): 2.0,
        }

        mgr._drop_room_state(ROOM)

        self.assertEqual(mgr._decode.announced_at, {str(OTHER_ROOM): 2.0})


if __name__ == "__main__":
    unittest.main()
