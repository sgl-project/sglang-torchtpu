"""A pipeline-parallel TPU prefill starts a room only once every stage has the
decode's request: stage 0 polls the room, and the other stages, which are not
polled while it bootstraps, mark it ready when the request arrives."""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.pp_consensus import poll_with_pp_consensus
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import MessageTag, PullReq
from sglang_tpu.disaggregation.raiden.state import PrefillState
from sglang_tpu.disaggregation.tpu.conn import TpuKVSender

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 41


def _prefill_manager(pp_size):
    """A Raiden prefill manager with its room tables, no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr.pp_size = pp_size
    mgr.attn_tp_rank = 0
    mgr._prefill = PrefillState()
    mgr._pipeline = None
    mgr._state_lock = threading.Lock()
    mgr._aborted = {}
    mgr.request_status = {ROOM: KVPoll.Bootstrapping}
    mgr.update_status = lambda room, status: mgr.request_status.__setitem__(
        room, status
    )
    mgr._announce_chunk = Mock()
    mgr._maybe_reply = Mock()
    return mgr


def _pull_req():
    return PullReq(
        room=ROOM, decode_ip="127.0.0.1", decode_port=31020, decode_pages=2
    ).to_frames()


def _sender(cls, *, own, stages):
    sender = object.__new__(cls)
    sender.bootstrap_room = ROOM
    sender.poll = Mock(return_value=own)
    sender.kv_mgr = SimpleNamespace(check_status_pp_consensus=Mock(return_value=stages))
    return sender


class TestConsensusPoll(unittest.TestCase):
    def test_ready_only_when_every_stage_is_ready(self):
        cases = (
            (KVPoll.WaitingForInput, KVPoll.WaitingForInput, KVPoll.WaitingForInput),
            (KVPoll.WaitingForInput, KVPoll.Bootstrapping, KVPoll.Bootstrapping),
            # A stage that failed the room fails it on every stage.
            (KVPoll.WaitingForInput, KVPoll.Failed, KVPoll.Failed),
        )
        for own, stages, expected in cases:
            with self.subTest(own=own, stages=stages):
                sender = _sender(RaidenKVSender, own=own, stages=stages)
                self.assertEqual(poll_with_pp_consensus(sender), expected)

    def test_stage_zero_decides_alone_until_it_is_ready(self):
        for own in (KVPoll.Bootstrapping, KVPoll.Failed):
            with self.subTest(own=own):
                sender = _sender(RaidenKVSender, own=own, stages=KVPoll.WaitingForInput)
                self.assertEqual(poll_with_pp_consensus(sender), own)
                sender.kv_mgr.check_status_pp_consensus.assert_not_called()

    def test_both_tpu_senders_poll_with_consensus(self):
        for cls in (RaidenKVSender, TpuKVSender):
            with self.subTest(sender=cls.__name__):
                sender = _sender(
                    cls, own=KVPoll.WaitingForInput, stages=KVPoll.Bootstrapping
                )
                self.assertEqual(sender.poll_pp_consensus(), KVPoll.Bootstrapping)
                sender.kv_mgr.check_status_pp_consensus.assert_called_once_with(ROOM)


class TestRaidenStageReadiness(unittest.TestCase):
    def test_pull_request_starts_the_room_on_a_pipeline_stage(self):
        mgr = _prefill_manager(pp_size=2)
        mgr._on_pull_req(_pull_req())
        self.assertEqual(mgr.request_status[ROOM], KVPoll.WaitingForInput)

    def test_pull_request_leaves_a_single_stage_room_to_its_poll(self):
        mgr = _prefill_manager(pp_size=1)
        mgr._on_pull_req(_pull_req())
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)

    def test_pipeline_request_starts_the_room_on_a_pipeline_stage(self):
        mgr = _prefill_manager(pp_size=2)
        mgr._require_frames = Mock()
        pipeline = object.__new__(PipelineTransfer)
        pipeline.manager = mgr
        pipeline.producer = True
        pipeline.replies = {}
        pipeline.early = {}
        pipeline.stage = {}
        pipeline._reply = Mock()
        mgr._pipeline = pipeline
        pipeline._on_producer_message(
            ROOM,
            [
                MessageTag.PIPELINE_REQ.value,
                str(ROOM).encode(),
                b"127.0.0.1:31020",
                json.dumps({"attn_tp_size": 1}).encode(),
            ],
        )
        self.assertEqual(mgr.request_status[ROOM], KVPoll.WaitingForInput)

    def test_an_aborted_or_unrequested_room_keeps_bootstrapping(self):
        mgr = _prefill_manager(pp_size=2)
        with mgr._state_lock:
            mgr._finish_pp_bootstrap_locked(ROOM)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)
        mgr._prefill.decode_addr[ROOM] = ("127.0.0.1", 31020)
        mgr._aborted[ROOM] = 0.0
        with mgr._state_lock:
            mgr._finish_pp_bootstrap_locked(ROOM)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)


if __name__ == "__main__":
    unittest.main()
