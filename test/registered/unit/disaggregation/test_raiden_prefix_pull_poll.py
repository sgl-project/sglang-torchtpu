"""CPU checks of how a Raiden prefill room reports while its stage pulls a prefix
from the decode, and of the bootstrap gate that waits for every decode rank."""

import threading
import unittest
from types import SimpleNamespace

from test_raiden_failure_handling import _prefill_manager

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import pp_reverse
from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer
from sglang_tpu.envs import envs

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 77


class _Reverse:
    def __init__(self, state, in_flight=False):
        self.pulls = {ROOM: None}
        self.state = state
        self._in_flight = in_flight

    def status(self, room):
        return self.state, "decode rank 2: busy"

    def in_flight(self, room):
        return self._in_flight


class TestPrefixPullPoll(unittest.TestCase):
    def _mgr(self, reverse):
        mgr = _prefill_manager()
        mgr._pipeline = SimpleNamespace(lock=threading.RLock(), reverse=reverse)
        return mgr

    def test_pending_holds_and_done_releases(self):
        mgr = self._mgr(_Reverse(pp_reverse.PENDING))
        wait = KVPoll.WaitingForInput
        self.assertEqual(mgr.prefix_pull_poll(ROOM, wait), KVPoll.Bootstrapping)
        self.assertIsNone(mgr.prefix_pull_poll(ROOM + 1, wait))
        self.assertIsNone(mgr.prefix_pull_poll(ROOM, KVPoll.Transferring))
        mgr._pipeline.reverse.state = pp_reverse.DONE
        self.assertIsNone(mgr.prefix_pull_poll(ROOM, wait))

    def test_failure_waits_for_an_in_flight_transfer(self):
        mgr = self._mgr(_Reverse(pp_reverse.FAILED, in_flight=True))
        self.assertEqual(
            mgr.prefix_pull_poll(ROOM, KVPoll.WaitingForInput), KVPoll.Bootstrapping
        )

    def test_strict_failure_fails_the_room_with_its_reason(self):
        mgr = self._mgr(_Reverse(pp_reverse.FAILED))
        with envs.SGLANG_TPU_PD_PREFIX_PULL_STRICT.override(True):
            poll = mgr.prefix_pull_poll(ROOM, KVPoll.WaitingForInput)
        self.assertEqual(poll, KVPoll.Failed)
        self.assertTrue(mgr.is_failed(ROOM))
        self.assertIn("Prefix pull from the decode failed", mgr.failure_records[ROOM])

    def test_recoverable_failure_leaves_the_room_alive(self):
        mgr = self._mgr(_Reverse(pp_reverse.FAILED))
        with envs.SGLANG_TPU_PD_PREFIX_PULL_STRICT.override(False):
            poll = mgr.prefix_pull_poll(ROOM, KVPoll.WaitingForInput)
        self.assertEqual(poll, KVPoll.Failed)
        self.assertFalse(mgr.is_failed(ROOM))


class TestDecodeRequestedGate(unittest.TestCase):
    def test_a_prefix_pull_waits_for_every_decode_rank(self):
        pipe = SimpleNamespace(
            replies={ROOM: {"tcp://d:1": {"attn_tp_size": 2}}}, reverse=object()
        )
        with envs.SGLANG_TPU_PD_PREFIX_PULL.override(False):
            self.assertTrue(PipelineTransfer.decode_requested(pipe, ROOM))
        with envs.SGLANG_TPU_PD_PREFIX_PULL.override(True):
            self.assertFalse(PipelineTransfer.decode_requested(pipe, ROOM))
            pipe.replies[ROOM]["tcp://d:2"] = {"attn_tp_size": 2}
            self.assertTrue(PipelineTransfer.decode_requested(pipe, ROOM))
            self.assertFalse(PipelineTransfer.decode_requested(pipe, ROOM + 1))


if __name__ == "__main__":
    unittest.main()
