"""A raiden decode room's SGLANG_DISAGGREGATION_WAITING_TIMEOUT restarts at each
chunk or closing announcement."""

import threading
import time
import unittest
from unittest.mock import Mock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang.srt.disaggregation.raiden.serde import Blocks, Chunk
from sglang.srt.disaggregation.raiden.state import DecodeState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 61
OTHER_ROOM = 62
TIMEOUT_S = 300.0


def _decode_manager():
    """A chunked decode manager with ROOM registered and no pull ever issued;
    its monotonic clock is a Mock the test advances."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._pulling = None
    mgr._decode = DecodeState()
    mgr._decode.local_blocks[ROOM] = [21, 22, 23]
    mgr.chunked_transfer = True
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr.failure_lock = threading.Lock()
    mgr.connection_lock = threading.Lock()
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr._aborted = {}
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._send_done_at = {}
    mgr._send_reads = {}
    mgr.waiting_timeout = TIMEOUT_S
    mgr._timeout_s = 120.0
    mgr._monotonic = Mock(return_value=1000.0)
    mgr._drain_engine = lambda: None
    mgr.maybe_start_pull = lambda room: False
    return mgr


def _receiver(mgr):
    """ROOM's receiver, 3 s before its deadline from send_metadata."""
    recv = RaidenKVReceiver.__new__(RaidenKVReceiver)
    recv.kv_mgr = mgr
    recv.bootstrap_room = ROOM
    recv.conclude_state = None
    recv.abort_notified = False
    recv.bootstrap_infos = None  # no prefill to tell on failure
    recv._connection_pool_entries = {}
    recv.init_time = time.time() - (TIMEOUT_S - 3)
    return recv


def _advance(recv, seconds):
    """Let ``seconds`` pass on the manager's clock and on the wall clock the
    deadline from send_metadata reads."""
    recv.kv_mgr._monotonic.return_value += seconds
    recv.init_time -= seconds


def _chunk(index, room=ROOM):
    return Chunk(
        room=room,
        req_id=f"{room}#{index}",
        uuid=900 + index,
        start=index,
        total_pages=3,
        blocks=[index],
        endpoint="127.0.0.1:31020",
    ).to_frames()


def _blocks():
    return Blocks(
        room=ROOM,
        blocks=[0, 1, 2],
        endpoint="127.0.0.1:31020",
        swa_blocks=[],
        swa_endpoint="",
        aux_blob=b"",
    ).to_frames()


class TestWaitingTimeout(unittest.TestCase):
    def test_each_announcement_restarts_the_deadline(self):
        mgr = _decode_manager()
        recv = _receiver(mgr)

        mgr._on_chunk(_chunk(0))
        _advance(recv, 200)  # past the deadline from send_metadata
        self.assertNotEqual(recv.poll(), KVPoll.Failed)
        mgr._on_blocks(_blocks())
        _advance(recv, TIMEOUT_S - 1)  # past the deadline from the chunk
        self.assertNotEqual(recv.poll(), KVPoll.Failed)
        _advance(recv, 1)
        self.assertEqual(recv.poll(), KVPoll.Failed)
        self.assertIn(
            "timed out after 300.0s without a new KV announcement",
            mgr.failure_records[ROOM],
        )
        self.assertRegex(mgr.failure_records[ROOM], r"\(797\.\ds since send_metadata")

    def test_only_new_announcements_of_a_registered_room_are_recorded(self):
        mgr = _decode_manager()
        mgr._on_chunk(_chunk(0))
        mgr._monotonic.return_value += TIMEOUT_S
        # A duplicate is not progress; an unregistered room has no deadline.
        mgr._on_chunk(_chunk(0))
        mgr._on_chunk(_chunk(0, room=OTHER_ROOM))

        self.assertEqual(mgr.last_announcement_age(ROOM), TIMEOUT_S)
        self.assertIsNone(mgr.last_announcement_age(OTHER_ROOM))
        mgr.clear_room(ROOM)
        self.assertIsNone(mgr.last_announcement_age(ROOM))


if __name__ == "__main__":
    unittest.main()
