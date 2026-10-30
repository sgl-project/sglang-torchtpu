"""A decode that already holds every page of a request settles its raiden
transfer on the closing message instead of waiting for a read it cannot issue."""

import threading
import unittest
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.serde import Blocks
from sglang_tpu.disaggregation.raiden.state import DecodeState

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 41


def _decode_manager(hybrid):
    """A chunked-transfer decode manager with its room tables, no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr.attn_tp_rank = 0
    mgr.chunked_transfer = True
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.record_failure = Mock()
    mgr._engine = Mock()
    mgr._swa_engine = Mock() if hybrid else None
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
    mgr._timeout_s = 120.0
    return mgr


def _blocks(hybrid):
    return Blocks(
        room=ROOM,
        blocks=[0, 1, 2],
        endpoint="127.0.0.1:31020",
        swa_blocks=[3] if hybrid else [],
        swa_endpoint="127.0.0.1:31022" if hybrid else "",
        aux_blob=b"",
    ).to_frames()


class TestZeroPagePull(unittest.TestCase):
    def _settles_on_blocks(self, hybrid):
        mgr = _decode_manager(hybrid)
        # A page-aligned full prefix hit: the decode allocated no page, so the
        # prefill announced no chunk and registered no read.
        mgr._decode.local_blocks[ROOM] = []
        mgr._decode.local_swa_blocks[ROOM] = []

        mgr._on_blocks(_blocks(hybrid))

        # Nothing is pulled, and the room is received on every engine.
        self.assertFalse(mgr.maybe_start_pull(ROOM))
        self.assertTrue(mgr.is_recv_done(ROOM))
        mgr._engine.start_read.assert_not_called()
        if hybrid:
            mgr._swa_engine.start_read.assert_not_called()
        self.assertNotIn(str(ROOM), mgr._failed)

    def test_full_attention_model_settles_on_blocks(self):
        self._settles_on_blocks(hybrid=False)

    def test_hybrid_model_settles_both_engines_on_blocks(self):
        self._settles_on_blocks(hybrid=True)

    def test_decode_holding_a_page_still_pulls_it(self):
        mgr = _decode_manager(hybrid=False)
        mgr._decode.local_blocks[ROOM] = [100]
        mgr._decode.local_swa_blocks[ROOM] = []

        mgr._on_blocks(_blocks(hybrid=False))

        self.assertTrue(mgr.maybe_start_pull(ROOM))
        mgr._engine.start_read.assert_called_once_with(
            str(ROOM), ROOM, "127.0.0.1:31020", [2], [100]
        )
        self.assertFalse(mgr.is_recv_done(ROOM))

    def test_successful_transfer_logs_only_at_debug(self):
        # Successful transfers log several lines per request on every rank, so
        # they stay at DEBUG; at INFO they flood the server log of long P/D runs.
        mgr = _decode_manager(hybrid=False)
        mgr._decode.local_blocks[ROOM] = [100]
        mgr._decode.local_swa_blocks[ROOM] = []
        mgr._on_blocks(_blocks(hybrid=False))

        with self.assertLogs(
            "sglang_tpu.disaggregation.raiden.conn", level="DEBUG"
        ) as logs:
            self.assertTrue(mgr.maybe_start_pull(ROOM))
            mgr._log_settled_transfers(sent=[str(ROOM)], received=[str(ROOM)])

        self.assertEqual({record.levelname for record in logs.records}, {"DEBUG"})
        messages = "\n".join(logs.output)
        for event in ("decode pull started", "prefill read served", "decode pull done"):
            self.assertIn(event, messages)


if __name__ == "__main__":
    unittest.main()
