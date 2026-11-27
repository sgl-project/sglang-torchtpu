"""A pipeline-parallel TPU prefill starts a P/D room only once every stage has
the decode's request. Only PP stage 0 polls a bootstrapping room, through its
sender's poll_pp_consensus, which reads every stage's status. The other stages
are not polled, so each marks the room ready when the decode's request lands,
and stage 0 holds the whole room to the bootstrap deadline."""

import threading
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVSender
from sglang_tpu.disaggregation.raiden.pool_transfer import RaidenPoolTransfer
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    PullReq,
    ReshardReq,
    RoomFailed,
)
from sglang_tpu.disaggregation.raiden.state import PrefillState
from sglang_tpu.disaggregation.tpu.conn import TpuKVManager, TpuKVSender
from sglang_tpu.disaggregation.tpu.serde import Transfer

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 41
TIMEOUT = 300.0
DECODE_ENDPOINT = "tcp://10.0.0.2:31000"


class _StageStatuses(dict):
    """PP stage 0's request_status: PPConsensusStore's ``collect`` over this
    stage's map and the later stages', read in place instead of replicated."""

    def __init__(self, later_stages):
        super().__init__()
        self.later_stages = list(later_stages)

    def collect(self, key):
        return [self.get(key), *(stage.get(key) for stage in self.later_stages)]


def _prefill_stages(make_stage, pp_size=2):
    """One manager per prefill stage; stage 0 reads every stage's status."""
    later = [make_stage(rank, pp_size, {}) for rank in range(1, pp_size)]
    first = make_stage(0, pp_size, _StageStatuses(m.request_status for m in later))
    return [first, *later]


def _sender(cls, mgr):
    with get_context().override_server_args(dp_size=1):
        return cls(mgr, "unused", ROOM)


def _age(sender, seconds):
    sender.init_time = time.time() - seconds


# --------------------------------------------------------------------------
# Raiden
# --------------------------------------------------------------------------


def _raiden_stage(pp_rank, pp_size, request_status):
    """A Raiden prefill manager for one stage, with its room tables and a
    pipeline transfer that records requests but never replies; no sockets or
    engines."""
    mgr = object.__new__(RaidenKVManager)
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr.pp_rank = pp_rank
    mgr.pp_size = pp_size
    mgr.attn_tp_rank = 0
    mgr.is_dummy_cp_rank = False
    mgr.bootstrap_timeout = TIMEOUT
    mgr.request_status = request_status
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._drain_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._prefill = PrefillState(announce_pages=1)
    mgr._decode = None
    mgr._engine = None
    mgr._swa_engine = None
    mgr._pulling = None
    mgr._pulling_cleared = set()
    mgr._completed_send = set()
    mgr._send_done_at = {}
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._aborted = {}
    mgr._reshard_slot_rooms = [None]
    mgr._reshard_slot_of = {}
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr.chunked_transfer = False
    mgr._send_reads = {}
    mgr._superseded_sender_reads = set()
    mgr._timeout_s = 120.0
    mgr._reshard_pushes_pending = {}
    mgr._staging_outstanding = {}
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr.defer_decode_allocation = False
    mgr.req_to_decode_prefix_len = {}
    mgr._send_multipart_locked = Mock()
    mgr._message_handlers = {
        PullReq.TAG: mgr._on_pull_req,
        ReshardReq.TAG: mgr._on_reshard_req,
    }
    pipeline = RaidenPoolTransfer.__new__(RaidenPoolTransfer)
    pipeline.manager = mgr
    pipeline.producer = True
    pipeline.lock = threading.RLock()
    pipeline.replies = {}
    pipeline.sent = {}
    pipeline.failed = set()
    pipeline.poll = lambda: (set(), set(), set())
    pipeline._init_overlap_state()
    mgr._pipeline = pipeline
    return mgr


def _pipeline_request(mgr):
    mgr._pipeline.on_message(
        [
            MessageTag.PIPELINE_REQ,
            str(ROOM).encode(),
            DECODE_ENDPOINT.encode(),
            b'{"attn_tp_size": 1, "attn_tp_rank": 0}',
        ]
    )


def _pull_request(mgr):
    mgr._dispatch(PullReq(ROOM, "10.0.0.2", 5555, 4).to_frames())


def _reshard_request(mgr):
    mgr._dispatch(
        ReshardReq(ROOM, "10.0.0.2", 5555, "push", [1, 2], 2, [], "").to_frames()
    )


class TestRaidenPrefillPpBootstrap(CustomTestCase):
    def test_room_starts_once_every_stage_has_the_decode_request(self):
        for request in (_pipeline_request, _pull_request, _reshard_request):
            with self.subTest(request=request.__name__):
                stages = _prefill_stages(_raiden_stage)
                first, later = [_sender(RaidenKVSender, mgr) for mgr in stages]
                request(stages[0])
                self.assertEqual(first.poll_pp_consensus(), KVPoll.Bootstrapping)
                # Stage 1 is never polled while the room bootstraps.
                request(stages[1])
                self.assertEqual(stages[1].request_status[ROOM], KVPoll.WaitingForInput)
                self.assertEqual(first.poll_pp_consensus(), KVPoll.WaitingForInput)

    def test_request_that_outran_a_stage_sender_starts_its_room(self):
        stages = _prefill_stages(_raiden_stage)
        first = _sender(RaidenKVSender, stages[0])
        _pipeline_request(stages[0])
        _pipeline_request(stages[1])
        _sender(RaidenKVSender, stages[1])
        self.assertEqual(stages[1].request_status[ROOM], KVPoll.WaitingForInput)
        self.assertEqual(first.poll_pp_consensus(), KVPoll.WaitingForInput)

    def test_a_stage_without_the_decode_request_fails_the_room_at_the_deadline(
        self,
    ):
        stages = _prefill_stages(_raiden_stage)
        first, _ = [_sender(RaidenKVSender, mgr) for mgr in stages]
        _pipeline_request(stages[0])
        self.assertEqual(first.poll_pp_consensus(), KVPoll.Bootstrapping)
        _age(first, 2 * TIMEOUT)
        self.assertEqual(first.poll_pp_consensus(), KVPoll.Failed)
        mgr = stages[0]
        # Tombstoned like any failed Raiden room: a late request is refused.
        self.assertTrue(mgr.is_aborted(ROOM))
        self.assertIn("every pipeline stage", mgr.failure_records[ROOM])
        # Stage 0's decode reserved the room's pages; it hears now, not at
        # its own waiting timeout.
        endpoint, frames = mgr._send_multipart_locked.call_args.args
        self.assertEqual(endpoint, DECODE_ENDPOINT)
        self.assertEqual(frames[0], RoomFailed.TAG)
        self.assertEqual(first.poll_pp_consensus(), KVPoll.Failed)

    def test_a_room_failed_on_another_stage_fails_on_stage_zero(self):
        stages = _prefill_stages(_raiden_stage)
        first, later = [_sender(RaidenKVSender, mgr) for mgr in stages]
        _pipeline_request(stages[0])
        later.abort()
        self.assertEqual(first.poll_pp_consensus(), KVPoll.Failed)

    def test_single_stage_prefill_leaves_the_room_to_its_poll(self):
        mgr = _raiden_stage(0, 1, {})
        mgr._pipeline = None
        sender = _sender(RaidenKVSender, mgr)
        _pull_request(mgr)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)


# --------------------------------------------------------------------------
# TPU transport
# --------------------------------------------------------------------------


def _tpu_stage(pp_rank, pp_size, request_status):
    """A TPU-transport prefill manager for one stage; no sockets or pools."""
    mgr = object.__new__(TpuKVManager)
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr.pp_rank = pp_rank
    mgr.pp_size = pp_size
    mgr.is_dummy_cp_rank = False
    mgr.bootstrap_timeout = TIMEOUT
    mgr.request_status = request_status
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._staging_outstanding = defaultdict(int)
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr.defer_decode_allocation = False
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    # What get_prefill_transfer_info reads: one layer of 64-token pages.
    mgr._kv_packing = 1
    mgr.kv_args = SimpleNamespace(engine_rank=0, total_kv_head_num=8)
    mgr.attn_tp_size = 1
    mgr.layer_buffers = [torch.zeros(1, 64, 8, 1)]
    return mgr


def _transfer_info(mgr):
    mgr._handle_transfer(
        Transfer(
            room=ROOM,
            decode_ip="10.0.0.2",
            decode_port=5555,
            num_pages=2,
            required_dst_info_num=1,
            decode_prefix_len=0,
            dst_tp_rank=0,
            decode_attn_tp_size=1,
        ).to_frames()
    )


class _DrainingTpuKVSender(TpuKVSender):
    """A TPU-transport backend that advances its data plane in _pre_poll."""

    pre_polls = 0

    def _pre_poll(self):
        self.pre_polls += 1


class TestTpuPrefillPpBootstrap(CustomTestCase):
    def test_room_starts_once_every_stage_has_the_transfer_info(self):
        stages = _prefill_stages(_tpu_stage)
        first, _ = [_sender(TpuKVSender, mgr) for mgr in stages]
        _transfer_info(stages[0])
        self.assertEqual(first.poll_pp_consensus(), KVPoll.Bootstrapping)
        _transfer_info(stages[1])
        self.assertEqual(first.poll_pp_consensus(), KVPoll.WaitingForInput)

    def test_a_stage_without_the_transfer_info_fails_the_room_at_the_deadline(
        self,
    ):
        stages = _prefill_stages(_tpu_stage)
        first, _ = [_sender(TpuKVSender, mgr) for mgr in stages]
        _transfer_info(stages[0])
        _age(first, 2 * TIMEOUT)
        self.assertEqual(first.poll_pp_consensus(), KVPoll.Failed)
        self.assertIn("Bootstrapping", stages[0].failure_records[ROOM])
        self.assertEqual(first.poll_pp_consensus(), KVPoll.Failed)

    def test_consensus_poll_runs_the_transport_pre_poll(self):
        stages = _prefill_stages(_tpu_stage)
        first, _ = [_sender(_DrainingTpuKVSender, mgr) for mgr in stages]
        first.poll_pp_consensus()
        self.assertEqual(first.pre_polls, 1)


if __name__ == "__main__":
    unittest.main()
