"""Raiden handshake steps log at DEBUG, and at INFO once a step is slow."""

import threading
import time
import unittest
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.common import (
    SLOW_HANDSHAKE_S,
    log_handshake_step,
)
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.serde import Blocks, PullReq
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 43
LOGGER = "sglang_tpu.disaggregation.raiden.conn"


def _decode_manager():
    """A decode manager with its room tables, no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr.attn_tp_rank = 0
    mgr.chunked_transfer = False
    mgr._state_lock = threading.Lock()
    mgr._aborted = {}
    mgr._completed_recv = set()
    mgr._swa_engine = None
    mgr._pulling = StagingAdmission(64, None)
    return mgr


def _prefill_manager():
    """A prefill manager with its room tables, no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = PrefillState(announce_pages=1)
    mgr._decode = None
    mgr._state_lock = threading.Lock()
    mgr._aborted = {}
    mgr._swa_engine = None
    return mgr


def _blocks():
    return Blocks(
        room=ROOM,
        blocks=[0, 1],
        endpoint="127.0.0.1:31020",
        swa_blocks=[],
        swa_endpoint="",
        aux_blob=b"",
    ).to_frames()


class TestLogHandshakeStep(unittest.TestCase):
    def test_fast_step_logs_at_debug(self):
        with self.assertLogs(LOGGER, level="DEBUG") as logs:
            log_handshake_step(ROOM, "step", time.time(), "start")
        self.assertEqual([r.levelname for r in logs.records], ["DEBUG"])

    def test_slow_step_logs_at_info_with_elapsed_time(self):
        with self.assertLogs(LOGGER, level="INFO") as logs:
            log_handshake_step(
                ROOM, "step", time.time() - SLOW_HANDSHAKE_S - 1, "start"
            )
        self.assertIn(f"Raiden room {ROOM}: step, ", logs.output[0])
        self.assertIn("s after start", logs.output[0])

    def test_step_without_start_time_logs_at_debug(self):
        with self.assertLogs(LOGGER, level="DEBUG") as logs:
            log_handshake_step(ROOM, "step", None, "")
        self.assertEqual([r.levelname for r in logs.records], ["DEBUG"])


class TestHandshakeLogPoints(unittest.TestCase):
    def test_decode_logs_slow_blocks_reply_at_info(self):
        mgr = _decode_manager()
        mgr._decode.local_blocks[ROOM] = [5, 6]
        mgr._decode.handshake_at[ROOM] = time.time() - SLOW_HANDSHAKE_S - 1

        with self.assertLogs(LOGGER, level="INFO") as logs:
            mgr._on_blocks(_blocks())

        self.assertIn("received BLOCKS from the prefill", logs.output[0])
        self.assertNotIn(ROOM, mgr._decode.handshake_at)

    def test_prefill_logs_slow_decode_ask_at_info(self):
        mgr = _prefill_manager()
        mgr._maybe_reply = Mock()
        mgr._prefill.handshake_at[ROOM] = time.time() - SLOW_HANDSHAKE_S - 1

        with self.assertLogs(LOGGER, level="INFO") as logs:
            mgr._on_pull_req(PullReq(ROOM, "10.0.0.2", 5555, 2).to_frames())

        self.assertIn("decode asked for its KV", logs.output[0])
        # The next step is timed from the ask.
        self.assertLess(time.time() - mgr._prefill.handshake_at[ROOM], 5)

    def test_cleared_room_drops_its_handshake_time(self):
        mgr = _prefill_manager()
        mgr._prefill.handshake_at[ROOM] = time.time()
        mgr._pulling = None
        mgr._completed_send = set()
        mgr._send_done_at = {}
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._native_failed = set()
        mgr._cleared_plans = {}

        mgr._drop_room_state(ROOM)

        self.assertNotIn(ROOM, mgr._prefill.handshake_at)


if __name__ == "__main__":
    unittest.main()
