"""Raiden's per-request transfer progress logs at DEBUG, not INFO."""

import threading
import unittest
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 47
LOGGER = "sglang_tpu.disaggregation.raiden.conn"


def _manager():
    """A manager with only what the logged steps touch, no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr.attn_tp_rank = 0
    mgr._state_lock = threading.Lock()
    mgr._aborted = {}
    mgr._send_reads = {}
    return mgr


class TestTransferProgressLogs(unittest.TestCase):
    def assert_debug_only(self, logs, *messages):
        self.assertEqual([r.levelname for r in logs.records], ["DEBUG"] * len(messages))
        for output, message in zip(logs.output, messages):
            self.assertIn(message, output)

    def test_decode_pull_started_logs_at_debug(self):
        with self.assertLogs(LOGGER, level="DEBUG") as logs:
            started = _manager()._start_read_or_fail(
                room=ROOM,
                engine=Mock(),
                req_id=str(ROOM),
                uuid=ROOM,
                endpoint="127.0.0.1:31020",
                remote=[0, 1],
                local=[5, 6],
            )
        self.assertTrue(started)
        self.assert_debug_only(logs, "decode pull started")

    def test_prefill_registered_read_logs_at_debug(self):
        engine = Mock()
        engine.register_read.return_value = True
        with self.assertLogs(LOGGER, level="DEBUG") as logs:
            _manager()._register_sender_read(ROOM, engine, str(ROOM), ROOM, [0, 1])
        self.assert_debug_only(logs, "prefill registered read")

    def test_settled_transfers_log_at_debug(self):
        with self.assertLogs(LOGGER, level="DEBUG") as logs:
            _manager()._log_settled_transfers(sent=[str(ROOM)], received=[str(ROOM)])
        self.assert_debug_only(logs, "prefill read served", "decode pull done")


if __name__ == "__main__":
    unittest.main()
