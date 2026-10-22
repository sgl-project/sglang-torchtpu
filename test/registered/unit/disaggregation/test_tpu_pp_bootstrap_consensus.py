"""A pipeline-parallel TPU prefill starts a room only once every stage has the
decode's request: stage 0 polls the room and holds it to the bootstrap deadline,
and the other stages, which are not polled while it bootstraps, mark it ready
when the request arrives."""

import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import MessageTag, PullReq
from sglang_tpu.disaggregation.raiden.state import PrefillState
from sglang_tpu.disaggregation.tpu.conn import TpuKVSender

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 41
TIMEOUT = 1.0


def _prefill_manager(pp_size):
    """A Raiden prefill manager with its room tables, no sockets or engines."""
    mgr = object.__new__(RaidenKVManager)
    mgr.pp_size = pp_size
    mgr.attn_tp_rank = 0
    mgr.is_dummy_cp_rank = False
    mgr.bootstrap_timeout = TIMEOUT
    mgr._prefill = PrefillState(announce_pages=1)
    mgr._pipeline = None
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._aborted = {}
    mgr._failed = set()
    mgr.failure_lock = threading.Lock()
    mgr.failure_records = {}
    mgr.request_status = {ROOM: KVPoll.Bootstrapping}
    mgr.update_status = lambda room, status: mgr.request_status.__setitem__(
        room, status
    )
    mgr._announce_chunk = Mock()
    mgr._maybe_reply = Mock()
    # No transfer has started: nothing to drain, read or complete.
    mgr._drain_engine = Mock()
    mgr.is_failed = lambda room: str(room) in mgr._failed
    mgr.sender_has_active_io = Mock(return_value=False)
    mgr.is_send_done = Mock(return_value=False)
    return mgr


def _pull_req():
    return PullReq(
        room=ROOM, decode_ip="127.0.0.1", decode_port=31020, decode_pages=2
    ).to_frames()


def _raiden_sender(*, stages, waited):
    """Stage 0's sender for a room this stage's decode request already reached."""
    mgr = _prefill_manager(pp_size=2)
    mgr._prefill.decode_addr[ROOM] = ("127.0.0.1", 31020)
    mgr.check_status_pp_consensus = Mock(return_value=stages)
    sender = object.__new__(RaidenKVSender)
    sender.kv_mgr = mgr
    sender.bootstrap_room = ROOM
    sender.conclude_state = None
    sender.init_time = time.time() - waited
    return sender


def _tpu_sender(*, stages, waited):
    """Stage 0's TPU-transport sender for a room this stage has started."""
    sender = object.__new__(TpuKVSender)
    sender.kv_mgr = SimpleNamespace(
        check_status=Mock(return_value=KVPoll.WaitingForInput),
        check_status_pp_consensus=Mock(return_value=stages),
        bootstrap_timeout=TIMEOUT,
        record_failure=Mock(),
    )
    sender.bootstrap_room = ROOM
    sender.conclude_state = None
    sender._transfer_start_time = None
    sender.init_time = time.time() - waited
    return sender


class TestConsensusPoll(unittest.TestCase):
    def test_room_waits_for_the_least_advanced_stage(self):
        for make in (_raiden_sender, _tpu_sender):
            for stages, expected in (
                (KVPoll.WaitingForInput, KVPoll.WaitingForInput),
                (KVPoll.Bootstrapping, KVPoll.Bootstrapping),
                # A stage that failed the room fails it on every stage.
                (KVPoll.Failed, KVPoll.Failed),
            ):
                with self.subTest(sender=make.__name__, stages=stages):
                    sender = make(stages=stages, waited=0.0)
                    self.assertEqual(sender.poll_pp_consensus(), expected)

    def test_a_stage_without_the_decode_request_fails_at_the_deadline(self):
        # Stage 0 is ready; another stage never got the decode's request, and
        # only stage 0 polls the room while it bootstraps.
        raiden = _raiden_sender(stages=KVPoll.Bootstrapping, waited=2 * TIMEOUT)
        self.assertEqual(raiden.poll_pp_consensus(), KVPoll.Failed)
        mgr = raiden.kv_mgr
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn(str(ROOM), mgr._failed)
        self.assertIn(ROOM, mgr._aborted)
        self.assertIn("every pipeline stage", mgr.failure_records[ROOM])
        # The room stays failed on the next poll.
        self.assertEqual(raiden.poll_pp_consensus(), KVPoll.Failed)

        tpu = _tpu_sender(stages=KVPoll.Bootstrapping, waited=2 * TIMEOUT)
        self.assertEqual(tpu.poll_pp_consensus(), KVPoll.Failed)
        tpu.kv_mgr.record_failure.assert_called_once()
        self.assertEqual(tpu.poll_pp_consensus(), KVPoll.Failed)

    def test_the_local_poll_still_reads_only_this_stage(self):
        for make in (_raiden_sender, _tpu_sender):
            with self.subTest(sender=make.__name__):
                sender = make(stages=KVPoll.Bootstrapping, waited=2 * TIMEOUT)
                self.assertEqual(sender.poll(), KVPoll.WaitingForInput)


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
