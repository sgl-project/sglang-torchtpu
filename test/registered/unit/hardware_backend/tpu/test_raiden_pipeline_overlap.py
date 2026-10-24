"""Early parts and stage groups of the Raiden pipeline transfer.

Two producer stages and one decode rank exchange their real control messages
through an in-memory bus; a fake controller "lands" a coordinated transfer by
reporting it on every engine. No TPU or tpu-sync is needed.
"""

import concurrent.futures
import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import pipeline as pl
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import MessageTag

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 123456789


def Tensor(*shape):
    return torch.zeros(shape, dtype=torch.bfloat16)


class Engine:
    def __init__(self):
        self.sent, self.received, self.failed = [], [], []

    def poll_stats_with_details(self):
        def settled(ids):
            return [SimpleNamespace(req_id=r, completed_ns=None) for r in ids]

        out = SimpleNamespace(
            sent=settled(self.sent),
            received=settled(self.received),
            failed=settled(self.failed),
        )
        self.sent, self.received, self.failed = [], [], []
        return out


class Controller:
    """Records registrations and lands every coordinated transfer at once."""

    def __init__(self):
        self.registered = {}
        self.transfers = []
        self.cancelled = []
        self.completed = []
        self.producers = []
        self.decodes = []

    def client(self, unit):
        controller = self

        class Client:
            def register_request_blocks(
                self, req_id, uuid, unit, block_ids, pool_spans
            ):
                controller.registered.setdefault(req_id, {})[unit.data_replica_idx] = (
                    uuid,
                    list(block_ids),
                    pool_spans,
                )

            def complete_request_blocks(self, req_id, uuid, unit):
                controller.completed.append(req_id)

            def cancel_request_blocks_if_unclaimed(self, req_id, uuid):
                claimed = any(t["req_id"] == req_id for t in controller.transfers)
                if not claimed:
                    controller.cancelled.append(req_id)
                return not claimed

            def coordinate_transfer(self, **kwargs):
                controller.transfers.append(kwargs)
                req_id = kwargs["req_id"]
                sources = {u.data_replica_idx for u in kwargs["src_units"]}
                assert set(controller.registered[req_id]) == sources, req_id
                if ".s" in req_id:
                    group = int(req_id.split(".s")[1])
                    for p in controller.producers:
                        if pl.stage_group(p.rank, p.stage_group_size) == group:
                            p.engine.sent.append(req_id)
                    for d in controller.decodes:
                        d.engine.received.append(req_id)
                    return True
                for p in controller.producers:
                    p.engine.sent.append(req_id)
                for d in controller.decodes:
                    d.engine.received.append(req_id)
                return True

        return Client()


class Manager:
    def __init__(self, mode, pp_rank, pp_size, bus, name):
        self.disaggregation_mode = mode
        self.pp_rank, self.pp_size = pp_rank, pp_size
        self.attn_tp_size, self.attn_tp_rank = 1, 0
        self._state_lock = threading.RLock()
        self._aborted = {}
        self._prefill = SimpleNamespace(send_blocks={}, send_swa_blocks={})
        self.kv_args = SimpleNamespace(
            total_kv_head_num=4, total_swa_kv_head_num=4, page_size=4
        )
        self._timeout_s = 60
        self.local_ip, self.rank_port = "10.0.0.1", 1000 + len(bus)
        self._completed_send, self._completed_recv = set(), set()
        self._failed = set()
        self._monotonic = time.monotonic
        self._engine_poll_failing_since = {}
        self.failure_lock = threading.Lock()
        self.failure_records = {}
        for method in (
            "_poll_engine_status",
            "_fail_room_locked",
            "_mark_aborted_locked",
            "is_aborted",
            "_tombstone_reason_locked",
            "_reply_room_failed",
        ):
            setattr(
                self, method, types.MethodType(getattr(RaidenKVManager, method), self)
            )
        self.status = {}
        self.bus = bus
        bus[f"tcp://{self.local_ip}:{self.rank_port}"] = name

    def update_status(self, room, status):
        self.status[room] = status

    @staticmethod
    def _norm_req_id(rid):
        return rid

    def _send_multipart_locked(self, endpoint, frames):
        self.bus["queue"].append((self.bus[endpoint], frames))

    @staticmethod
    def _require_frames(msg, tag, n):
        assert len(msg) == n


def make(mode, pp_rank, pp_size, controller, bus, name, heads, stage_parts=False):
    t = object.__new__(pl.PipelineTransfer)
    mgr = Manager(mode, pp_rank, pp_size, bus, name)
    groups = SimpleNamespace(
        full=[Tensor(32, 2 * heads, 8, 4)],
        swa=[Tensor(8, 2 * heads, 8, 4)],
        full_layer_ids=[2 * pp_rank + 1 if mode == DisaggregationMode.PREFILL else 1],
        swa_layer_ids=[2 * pp_rank if mode == DisaggregationMode.PREFILL else 0],
        num_draft_full=0,
        num_draft_swa=0,
    )
    if mode == DisaggregationMode.DECODE:  # the decode owns every layer
        groups.full = [Tensor(32, 2 * heads, 8, 4), Tensor(32, 2 * heads, 8, 4)]
        groups.swa = [Tensor(8, 2 * heads, 8, 4), Tensor(8, 2 * heads, 8, 4)]
        groups.full_layer_ids, groups.swa_layer_ids = [1, 3], [0, 2]
    t.manager = mgr
    t.producer = mode == DisaggregationMode.PREFILL
    t.lock = threading.RLock()
    t.pools = pl.pool_manifest(groups)
    t.tensors = list(groups.full) + list(groups.swa)
    t.rank = pp_rank
    t.workers = pp_size
    t.client_type = controller.client
    t._init_overlap_state()
    t.early_parts = True
    t.part_min_pages = 4
    t.stage_parts = stage_parts and t.producer
    t.executor = SyncExecutor()
    t.hbm_mem_type = 2
    t.sent, t.replies, t.release_requests, t.receives = {}, {}, {}, {}
    t.cleared, t.done_send, t.done_recv, t.failed, t.retired = (
        set(),
        set(),
        set(),
        set(),
        set(),
    )
    t.controller = "ctl:1"
    t.unit_fields = dict(
        job_name="sglang",
        job_replica_id=t.controller,
        data_name="prefill" if t.producer else "decode",
        data_replica_idx=t.rank,
    )
    t.unit = SimpleNamespace(**t.unit_fields)
    t.client = controller.client(t.unit)
    t.dst_units = [t.unit]
    t.max_blocks = 32
    t.admission = StagingAdmission(4 * 32)
    t.engine = Engine()
    (controller.producers if t.producer else controller.decodes).append(t)
    return t


def poll(t):
    """Poll one worker and record its rooms as RaidenKVManager._drain_engine does."""
    sent, received, failed = t.poll()
    t.manager._completed_send |= sent
    t.manager._completed_recv |= received
    t.manager._failed |= failed


class SyncExecutor:
    """Runs the call at once; like ThreadPoolExecutor, errors land in the future."""

    def submit(self, fn, *args, **kwargs):
        f = concurrent.futures.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as error:
            f.set_exception(error)
        return f


def build_world(stage_parts):
    bus = {"queue": []}
    controller = Controller()
    stages = [
        make(
            DisaggregationMode.PREFILL,
            r,
            2,
            controller,
            bus,
            f"p{r}",
            heads=4,
            stage_parts=stage_parts,
        )
        for r in range(2)
    ]
    decode = make(DisaggregationMode.DECODE, 0, 1, controller, bus, "d", heads=4)
    actors = {"p0": stages[0], "p1": stages[1], "d": decode}

    def pump():
        while bus["queue"]:
            name, frames = bus["queue"].pop(0)
            actors[name].on_message(frames)

    infos = [
        dict(
            pp_rank=s.manager.pp_rank,
            rank_ip=s.manager.local_ip,
            rank_port=s.manager.rank_port,
        )
        for s in stages
    ]
    return SimpleNamespace(
        stages=stages, decode=decode, controller=controller, pump=pump, infos=infos
    )


@pytest.fixture
def world():
    return build_world(stage_parts=False)


@pytest.fixture
def staged():
    return build_world(stage_parts=True)


class Fence(concurrent.futures.Future):
    """A make_fence() future; ``ready = True`` completes the probe copy."""

    @property
    def ready(self):
        return self.done()

    @ready.setter
    def ready(self, value):
        if value and not self.done():
            self.set_result(None)


def prefill_pages(stage, blocks, swa=None):
    with stage.manager._state_lock:
        acc = stage.manager._prefill.send_blocks.setdefault(ROOM, [])
        start = len(acc)
        acc.extend(blocks)
        if swa is not None:
            stage.manager._prefill.send_swa_blocks[ROOM] = list(swa)
    return start


def test_early_part_then_closing_transfer(world):
    dst_full, dst_swa = list(range(100, 112)), [200, 201]
    world.decode.register_decode(ROOM, dst_full, dst_swa, None, world.infos)
    world.pump()
    # A cached prefix of 8 pages becomes final at the first batch.
    for s in world.stages:
        start = prefill_pages(s, list(range(10, 18)))
        s.offer_part(ROOM, start, list(range(10, 18)))
    world.pump()
    assert world.decode.maybe_start_pull(ROOM) is False  # closing not announced
    part0 = [t for t in world.controller.transfers if t["req_id"] == f"{ROOM}.p0"]
    assert len(part0) == 1
    assert part0[0]["uuid"] == pl.part_uuid(ROOM, 0) != ROOM
    assert part0[0]["dst_device_block_ids"] == dst_full[:8] * 2  # two full pools
    assert all(not tag.startswith("swa.") for tag in part0[0]["transfer_pool_tags"])
    # Prefill ends: 3 more pages and the window.
    for s in world.stages:
        prefill_pages(s, [18, 19, 20], swa=[30, 31])
        assert s.finalize_send(ROOM, None)
    world.pump()
    assert world.decode.maybe_start_pull(ROOM) is True
    closing = [t for t in world.controller.transfers if t["req_id"] == str(ROOM)]
    assert len(closing) == 1 and closing[0]["uuid"] == ROOM
    tags = closing[0]["transfer_pool_tags"]
    assert sum(tag.startswith("swa.") for tag in tags) == 2
    assert closing[0]["dst_device_block_ids"] == dst_full[8:11] * 2 + dst_swa * 2
    for s in world.stages:
        assert world.controller.registered[str(ROOM)][s.rank][1] == [18, 19, 20]
    for actor in (*world.stages, world.decode):
        poll(actor)
    assert str(ROOM) in world.decode.manager._completed_recv
    for s in world.stages:
        assert str(ROOM) in s.manager._completed_send
        assert not s.active(ROOM)
    assert f"{ROOM}.p0" in world.controller.completed


def test_room_waits_for_its_early_parts(world):
    world.decode.register_decode(ROOM, list(range(100, 112)), [200], None, world.infos)
    world.pump()
    for s in world.stages:
        start = prefill_pages(s, list(range(10, 18)))
        s.offer_part(ROOM, start, list(range(10, 18)))
        prefill_pages(s, [18], swa=[30])
        s.finalize_send(ROOM, None)
    world.pump()
    assert world.decode.maybe_start_pull(ROOM)
    # Only the closing transfer has landed on the decode so far.
    world.decode.engine.received = [str(ROOM)]
    poll(world.decode)
    assert str(ROOM) not in world.decode.manager._completed_recv
    assert world.decode.active(ROOM)
    world.decode.engine.received = [f"{ROOM}.p0"]
    poll(world.decode)
    assert str(ROOM) in world.decode.manager._completed_recv


def test_closing_keeps_a_full_page_when_every_page_went_early(world):
    world.decode.register_decode(ROOM, list(range(100, 112)), [200], None, world.infos)
    world.pump()
    for s in world.stages:
        start = prefill_pages(s, list(range(10, 18)), swa=[30])
        s.offer_part(ROOM, start, list(range(10, 18)))
        s.finalize_send(ROOM, None)
        assert s.sent[ROOM]["start"] == 7 and s.sent[ROOM]["full"] == [17]


def test_abort_cancels_unclaimed_parts(world):
    world.decode.register_decode(ROOM, list(range(100, 112)), [200], None, world.infos)
    world.pump()
    s = world.stages[0]
    start = prefill_pages(s, list(range(10, 18)))
    s.offer_part(ROOM, start, list(range(10, 18)))
    assert s.active(ROOM)
    s.manager._aborted[ROOM] = time.time()
    poll(s)
    assert f"{ROOM}.p0" in world.controller.cancelled
    assert not s.active(ROOM) and ROOM in s.failed


def test_disabled_keeps_one_transfer(world):
    for s in world.stages:
        s.early_parts = False
    dst_full = list(range(100, 112))
    world.decode.register_decode(ROOM, dst_full, [200], None, world.infos)
    world.pump()
    for s in world.stages:
        start = prefill_pages(s, list(range(10, 18)))
        s.offer_part(ROOM, start, list(range(10, 18)))
        prefill_pages(s, [18], swa=[30])
        s.finalize_send(ROOM, None)
        assert s.sent[ROOM]["start"] == 0 and s.sent[ROOM]["parts"] == 0
    world.pump()
    assert world.decode.maybe_start_pull(ROOM)
    assert [t["req_id"] for t in world.controller.transfers] == [str(ROOM)]
    assert world.controller.transfers[0]["dst_device_block_ids"][:9] == dst_full[:9]


def test_part_ids_round_trip():
    assert pl.parse_req_id(pl.part_req_id(ROOM, 3)) == (ROOM, 3)
    assert pl.parse_req_id(str(ROOM)) == (ROOM, None)
    assert 0 < pl.part_uuid(ROOM, 3) < (1 << 63)


def offer_two_parts(world):
    """Each stage offers two 4-page early parts, then closes the room."""
    for s in world.stages:
        for blocks in (list(range(10, 14)), list(range(14, 18))):
            start = prefill_pages(s, blocks)
            s.offer_part(ROOM, start, blocks)
        prefill_pages(s, [18], swa=[30])
        s.finalize_send(ROOM, None)
    world.pump()


def test_each_early_part_holds_its_own_staging(world):
    """Two parts and the closing transfer need three staging slots; with room
    for one, each waits until the one before it lands."""
    d = world.decode
    d.admission = StagingAdmission(d.max_blocks)  # one transfer
    started = []

    def deferred(**kwargs):
        started.append(kwargs["req_id"])
        return True  # accepted; landing is reported later

    d.client_type = lambda address: SimpleNamespace(coordinate_transfer=deferred)
    d.register_decode(ROOM, list(range(100, 112)), [200], None, world.infos)
    world.pump()
    offer_two_parts(world)
    assert d.maybe_start_pull(ROOM) is False
    assert started == [f"{ROOM}.p0"]  # part 1 waits for staging
    for landed, expected in (
        (f"{ROOM}.p0", [f"{ROOM}.p0", f"{ROOM}.p1"]),
        (f"{ROOM}.p1", [f"{ROOM}.p0", f"{ROOM}.p1", str(ROOM)]),
    ):
        d.engine.received = [landed]
        poll(d)
        d.maybe_start_pull(ROOM)
        assert started == expected
    d.engine.received = [str(ROOM)]
    poll(d)
    assert str(ROOM) in d.manager._completed_recv
    assert d.admission.pages() == 0


def test_part_landing_before_this_rank_starts_it_is_kept(world):
    """On an attention-TP rank > 0, rank 0 starts the parts, so one can land
    while this rank still waits on staging for it. The landing must count and
    must not keep the room's place in line, which would block its own closing
    transfer."""
    d = world.decode
    d.manager.attn_tp_rank = 1  # never coordinates; only records landings
    d.admission = StagingAdmission(d.max_blocks)  # one transfer
    d.admission.admit("other", d.max_blocks)  # staging busy elsewhere
    d.register_decode(ROOM, list(range(100, 112)), [200], None, world.infos)
    world.pump()
    offer_two_parts(world)
    assert d.maybe_start_pull(ROOM) is False  # part 0 waits for staging
    assert d.admission.waiting == (ROOM, "p", 0)
    d.engine.received = [f"{ROOM}.p1", f"{ROOM}.p0"]  # rank 0 ran both
    poll(d)
    assert d.admission.waiting is None
    d.admission.release("other")
    assert d.maybe_start_pull(ROOM) is True
    d.engine.received = [str(ROOM)]
    poll(d)
    assert str(ROOM) in d.manager._completed_recv
    assert d.admission.pages() == 0
    assert d.admission.admit("next", d.max_blocks)


def test_rank_above_zero_fails_a_part_that_never_lands(world, monkeypatch):
    d = world.decode
    d.manager.attn_tp_rank = 1
    d.register_decode(ROOM, list(range(100, 112)), [200], None, world.infos)
    world.pump()
    for s in world.stages:
        start = prefill_pages(s, list(range(10, 18)))
        s.offer_part(ROOM, start, list(range(10, 18)))
    world.pump()
    d.maybe_start_pull(ROOM)
    assert d.receives[ROOM].parts[0]["accepted"]
    poll(d)
    assert str(ROOM) not in d.manager._failed
    now = time.monotonic() + 2 * d.manager._timeout_s + 1
    monkeypatch.setattr(pl.time, "monotonic", lambda: now)
    poll(d)
    assert str(ROOM) in d.manager._failed


def test_stage_parts_move_each_stage_when_its_forward_completes(staged):
    w = staged
    dst_full, dst_swa = list(range(100, 112)), [200, 201]
    w.decode.register_decode(ROOM, dst_full, dst_swa, None, w.infos)
    w.pump()
    fences = [Fence(), Fence()]
    for s, fence in zip(w.stages, fences):
        s.stage_ready(ROOM, list(range(10, 21)), [30, 31], fence)
        poll(s)
    w.pump()
    assert not w.controller.transfers  # no fence has completed
    fences[0].ready = True
    poll(w.stages[0])
    w.pump()
    assert w.decode.maybe_start_pull(ROOM) is False  # stage 1 still computing
    (t0,) = w.controller.transfers
    assert t0["req_id"] == f"{ROOM}.s0"
    assert [u.data_replica_idx for u in t0["src_units"]] == [0]
    assert set(t0["transfer_pool_tags"]) == set(
        w.stages[0].pools[i]["tag"] for i in range(2)
    )
    assert t0["dst_device_block_ids"] == dst_full[:11] + dst_swa
    fences[1].ready = True
    poll(w.stages[1])
    w.pump()
    assert w.decode.maybe_start_pull(ROOM) is True
    poll(w.decode)
    assert str(ROOM) not in w.decode.manager._completed_recv  # no metadata yet
    for s in w.stages:
        prefill_pages(s, list(range(10, 21)), swa=[30, 31])
        assert s.finalize_send(ROOM, None)
    w.pump()
    for actor in (*w.stages, w.decode):
        poll(actor)
    assert str(ROOM) in w.decode.manager._completed_recv
    for s in w.stages:
        assert str(ROOM) in s.manager._completed_send and not s.active(ROOM)


def test_stage_part_registers_at_finalize_when_fence_is_late(staged):
    w = staged
    w.decode.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    for s in w.stages:
        s.stage_ready(ROOM, list(range(10, 15)), [30], Fence())
        prefill_pages(s, list(range(10, 15)), swa=[30])
        s.finalize_send(ROOM, None)
    w.pump()
    assert w.decode.maybe_start_pull(ROOM)
    assert sorted(t["req_id"] for t in w.controller.transfers) == [
        f"{ROOM}.s0",
        f"{ROOM}.s1",
    ]
    for actor in (*w.stages, w.decode):
        poll(actor)
    assert str(ROOM) in w.decode.manager._completed_recv


def test_stage_parts_after_early_parts(staged):
    w = staged
    dst_full = list(range(100, 112))
    w.decode.register_decode(ROOM, dst_full, [200], None, w.infos)
    w.pump()
    for s in w.stages:
        start = prefill_pages(s, list(range(10, 18)))
        s.offer_part(ROOM, start, list(range(10, 18)))
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 21)), [30], fence)
        s.offer_part(ROOM, 8, [18, 19, 20, 21])  # late offer: ignored
        poll(s)
    w.pump()
    assert w.decode.maybe_start_pull(ROOM)
    ids = sorted(t["req_id"] for t in w.controller.transfers)
    assert ids == [f"{ROOM}.p0", f"{ROOM}.s0", f"{ROOM}.s1"]
    s0 = next(t for t in w.controller.transfers if t["req_id"] == f"{ROOM}.s0")
    assert s0["dst_device_block_ids"] == dst_full[8:11] + [200]
    for s in w.stages:
        prefill_pages(s, list(range(10, 21)), swa=[30])
        s.finalize_send(ROOM, None)
    w.pump()
    for actor in (*w.stages, w.decode):
        poll(actor)
    assert str(ROOM) in w.decode.manager._completed_recv


def test_abort_cancels_unclaimed_stage_part(staged):
    w = staged
    w.decode.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    s = w.stages[0]
    fence = Fence()
    fence.ready = True
    s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
    assert s.active(ROOM)
    s.manager._aborted[ROOM] = time.time()
    poll(s)
    assert f"{ROOM}.s0" in w.controller.cancelled
    assert not s.active(ROOM) and ROOM in s.failed


def test_rejected_stage_transfer_fails_the_room_not_the_scheduler(staged):
    w = staged
    w.decode.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()

    def reject(**kwargs):
        raise RuntimeError(
            "INVALID_ARGUMENT: Source transfer_rank values must be contiguous from zero"
        )

    w.decode.client_type = lambda address: SimpleNamespace(coordinate_transfer=reject)
    for s in w.stages:
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
        poll(s)
    w.pump()
    w.decode.maybe_start_pull(ROOM)
    poll(w.decode)  # must not raise
    assert str(ROOM) in w.decode.manager._failed
    assert "contiguous from zero" in w.decode.manager.failure_records[ROOM]
    # The controller arms receivers before senders, so the room keeps its
    # pages until the native deadline.
    assert w.decode.active(ROOM)


def test_failed_fence_waits_for_finalize(staged):
    w = staged
    w.decode.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    s = w.stages[0]
    fence = Fence()
    fence.set_exception(RuntimeError("probe copy failed"))
    s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
    poll(s)
    assert not s.stage[ROOM]["registered"]
    prefill_pages(s, list(range(10, 15)), swa=[30])
    assert s.finalize_send(ROOM, None)
    assert s.stage[ROOM]["registered"]


def test_stage_group_moves_its_stages_as_one_transfer(staged):
    w = staged
    for s in w.stages:
        s.stage_group_size = 2
    dst_full = list(range(100, 112))
    w.decode.register_decode(ROOM, dst_full, [200], None, w.infos)
    w.pump()
    fences = [Fence(), Fence()]
    for s, fence in zip(w.stages, fences):
        s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
    fences[0].ready = True
    w.stages[0].advance_stages()
    w.pump()
    assert w.decode.maybe_start_pull(ROOM) is False
    assert not w.controller.transfers  # the group waits for stage 1
    fences[1].ready = True
    w.stages[1].advance_stages()
    w.pump()
    assert w.decode.maybe_start_pull(ROOM) is True
    (t,) = w.controller.transfers
    assert t["req_id"] == f"{ROOM}.s0"
    assert sorted(u.data_replica_idx for u in t["src_units"]) == [0, 1]
    assert len(t["transfer_pool_tags"]) == 4  # both stages' layers
    for s in w.stages:
        prefill_pages(s, list(range(10, 15)), swa=[30])
        s.finalize_send(ROOM, None)
    w.pump()
    for actor in (*w.stages, w.decode):
        poll(actor)
    assert str(ROOM) in w.decode.manager._completed_recv
    for s in w.stages:
        assert str(ROOM) in s.manager._completed_send


def test_each_group_transfer_holds_its_own_staging(staged):
    """Two stage groups need two staging slots; with room for one, the second
    group waits until the first lands instead of over-committing staging."""
    w = staged
    w.decode.admission = StagingAdmission(w.decode.max_blocks)  # one transfer
    controller = w.controller
    held = []

    def deferred(**kwargs):
        held.append(kwargs["req_id"])
        controller.transfers.append(kwargs)
        return True  # accepted; landing is reported later

    w.decode.client_type = lambda address: SimpleNamespace(coordinate_transfer=deferred)
    w.decode.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    for s in w.stages:
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
        s.advance_stages()
    w.pump()
    w.decode.maybe_start_pull(ROOM)
    assert held == [f"{ROOM}.s0"]  # group 1 waits for staging
    poll(w.decode)
    assert held == [f"{ROOM}.s0"]
    w.decode.engine.received = [f"{ROOM}.s0"]
    poll(w.decode)  # group 0 landed: its slot frees, group 1 starts
    assert held == [f"{ROOM}.s0", f"{ROOM}.s1"]
    for s in w.stages:
        prefill_pages(s, list(range(10, 15)), swa=[30])
        s.finalize_send(ROOM, None)
    w.pump()
    w.decode.engine.received = [f"{ROOM}.s1"]
    poll(w.decode)
    assert str(ROOM) in w.decode.manager._completed_recv
    assert w.decode.admission.pages() == 0


def test_group_landing_before_this_rank_starts_it_is_kept(staged):
    """On an attention-TP rank > 0, rank 0 starts the groups: one can land
    while this rank still waits on staging for an earlier group. The landing
    must count, and the room's place in the staging line must not outlive it."""
    w = staged
    d = w.decode
    d.manager.attn_tp_rank = 1  # never coordinates; only records landings
    d.admission = StagingAdmission(d.max_blocks)  # one transfer
    d.admission.admit("other", d.max_blocks)  # staging busy elsewhere
    d.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    for s in w.stages:
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
        s.advance_stages()
    w.pump()
    assert d.maybe_start_pull(ROOM) is False  # group 0 waits for staging
    assert d.admission.waiting == (ROOM, "g", 0)
    d.engine.received = [f"{ROOM}.s1"]  # rank 0 ran group 1 already
    poll(d)
    d.admission.release("other")
    d.engine.received = [f"{ROOM}.s0"]
    poll(d)
    for s in w.stages:
        prefill_pages(s, list(range(10, 15)), swa=[30])
        s.finalize_send(ROOM, None)
    w.pump()
    poll(d)
    assert str(ROOM) in d.manager._completed_recv
    assert d.admission.pages() == 0
    assert d.admission.waiting is None


def test_settled_room_gives_up_its_place_in_the_staging_line(staged):
    w = staged
    d = w.decode
    d.manager.attn_tp_rank = 1
    d.admission = StagingAdmission(d.max_blocks)
    d.admission.admit("other", d.max_blocks)
    d.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    for s in w.stages:
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
        s.advance_stages()
    w.pump()
    d.maybe_start_pull(ROOM)
    assert d.admission.waiting == (ROOM, "g", 0)
    d.engine.received = [f"{ROOM}.s0", f"{ROOM}.s1"]
    for s in w.stages:
        prefill_pages(s, list(range(10, 15)), swa=[30])
        s.finalize_send(ROOM, None)
    w.pump()
    poll(d)
    assert str(ROOM) in d.manager._completed_recv
    assert d.admission.waiting is None  # the next room can be admitted
    d.admission.release("other")
    assert d.admission.admit("next", d.max_blocks)


def test_rank_above_zero_fails_a_group_that_never_lands(staged, monkeypatch):
    w = staged
    d = w.decode
    d.manager.attn_tp_rank = 1
    d.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    for s in w.stages:
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
        s.advance_stages()
    w.pump()
    assert d.maybe_start_pull(ROOM) is True
    poll(d)
    assert str(ROOM) not in d.manager._failed
    now = time.monotonic() + 2 * d.manager._timeout_s + 1
    monkeypatch.setattr(pl.time, "monotonic", lambda: now)
    poll(d)
    assert str(ROOM) in d.manager._failed


def give_aux_buffers(actor, rows=4):
    """First-token metadata rows like MetadataBuffers': token, then room."""
    actor.aux_rows = (
        torch.zeros((rows, 16), dtype=torch.int32),
        torch.zeros((rows, 8), dtype=torch.int64),
    )
    actor.manager.kv_args.aux_data_ptrs = [t.data_ptr() for t in actor.aux_rows]
    actor.manager.kv_args.aux_item_lens = [t[0].nbytes for t in actor.aux_rows]


def row_writer(actor, row, room):
    def write(token):
        actor.aux_rows[0][row][0] = token
        actor.aux_rows[1][row][0] = room

    return write


def first_token_aux(stage, token):
    """The last stage's metadata row once the PP loop copied its first token."""
    row_writer(stage, 0, ROOM)(token)
    mgr = stage.manager
    return pl.read_aux_blob(mgr.kv_args.aux_data_ptrs, mgr.kv_args.aux_item_lens, 0)


def early_close_world(world):
    """Early close: stage 0 hands its share over at launch; the last stage has
    yet to copy its sampled tokens."""
    for s in world.stages:
        s.early_close = True
        give_aux_buffers(s)
    give_aux_buffers(world.decode)
    world.decode.register_decode(ROOM, list(range(100, 112)), [200], 0, world.infos)
    world.pump()
    world.stages[0].close_early(ROOM, list(range(10, 15)), [30])
    world.pump()


def claim_without_landing(world):
    """The decode's coordinations are accepted but land only when reported."""
    world.decode.client_type = lambda address: SimpleNamespace(
        coordinate_transfer=lambda **kw: world.controller.transfers.append(kw) or True
    )


def test_early_close_gates_the_transfer_on_the_last_stages_tokens(world):
    """Stage 0 registers pages its forward may still be writing; the decode
    starts only once the last stage, whose forward needs stage 0's, has closed
    its share with the first token. The result point then adds nothing."""
    early_close_world(world)
    assert set(world.controller.registered[str(ROOM)]) == {0}
    assert world.decode.maybe_start_pull(ROOM) is False
    assert not world.controller.transfers
    last = world.stages[1]
    last.close_early(ROOM, list(range(10, 15)), [30], first_token_aux(last, 42))
    world.pump()
    assert world.decode.aux_rows[0][0][0] == 42
    assert world.decode.maybe_start_pull(ROOM) is True
    (closing,) = world.controller.transfers
    assert [u.data_replica_idx for u in closing["src_units"]] == [0, 1]
    for actor in (*world.stages, world.decode):
        poll(actor)
    assert str(ROOM) in world.decode.manager._completed_recv
    queued = len(world.stages[0].manager.bus["queue"])
    for s in world.stages:
        prefill_pages(s, list(range(10, 15)), swa=[30])
        assert s.finalize_send(ROOM, None)
    assert len(world.stages[0].manager.bus["queue"]) == queued


def test_early_close_sends_metadata_at_the_result_point_when_needed(world):
    """A last-stage share closed without the first token (logprobs, ...) gets
    the result point's metadata, alone; the decode holds the room until then."""
    early_close_world(world)
    world.stages[1].close_early(ROOM, list(range(10, 15)), [30])
    world.pump()
    assert world.decode.maybe_start_pull(ROOM) is True
    assert world.decode.aux_rows[1][0][0] == 0
    bus = world.stages[0].manager.bus["queue"]
    for s in world.stages:
        row_writer(s, 1, ROOM)(42)
        prefill_pages(s, list(range(10, 15)), swa=[30])
        assert s.finalize_send(ROOM, 1)
    assert [json.loads(frames[2]).get("aux_only") for _, frames in bus] == [True]
    world.pump()
    assert world.decode.aux_rows[0][0][0] == 42


def test_early_close_of_an_aborted_room_registers_nothing(world):
    early_close_world(world)
    last = world.stages[1]
    last.manager._aborted[ROOM] = time.time()
    last.close_early(ROOM, list(range(10, 15)), [30], first_token_aux(last, 42))
    assert 1 not in world.controller.registered[str(ROOM)]
    poll(last)
    assert not last.active(ROOM)


def test_early_close_after_early_parts_closes_only_the_rest(world):
    for s in world.stages:
        s.early_close = True
    dst_full = list(range(100, 112))
    world.decode.register_decode(ROOM, dst_full, [200], None, world.infos)
    world.pump()
    for s in world.stages:
        start = prefill_pages(s, list(range(10, 18)))
        s.offer_part(ROOM, start, list(range(10, 18)))  # one 8-page early part
        s.close_early(ROOM, list(range(10, 21)), [30])
        assert (s.sent[ROOM]["start"], s.sent[ROOM]["parts"]) == (8, 1)
    world.pump()
    assert world.decode.maybe_start_pull(ROOM)
    closing = next(t for t in world.controller.transfers if t["req_id"] == str(ROOM))
    assert closing["dst_device_block_ids"] == dst_full[8:11] * 2 + [200] * 2
    for actor in (*world.stages, world.decode):
        poll(actor)
    assert str(ROOM) in world.decode.manager._completed_recv


def test_early_close_page_mismatch_fails_the_room_and_keeps_claimed_pages(world):
    """Pages registered before the result point that differ from its own would
    let the decode read pages the request may no longer own: the room fails on
    both sides, and the claimed transfer keeps the pages until it settles."""
    early_close_world(world)
    claim_without_landing(world)
    world.stages[1].close_early(ROOM, list(range(10, 15)), [30])
    world.pump()
    assert world.decode.maybe_start_pull(ROOM)
    s = world.stages[0]
    bus = s.manager.bus["queue"]
    prefill_pages(s, [10, 11, 12, 13, 99], swa=[30])
    assert not s.finalize_send(ROOM, None)
    assert "differ" in s.manager.failure_records[ROOM]
    assert [frames[0] for _, frames in bus] == [MessageTag.ROOM_FAILED]
    poll(s)
    assert s.active(ROOM) and str(ROOM) in s.manager._failed
    s.engine.sent = [str(ROOM)]
    poll(s)
    assert not s.active(ROOM)


def test_early_close_refuses_stage_groups_and_a_loop_without_deferred_sends(
    monkeypatch,
):
    """Stage groups replace the closing transfer, and without the deferred
    output send the last stage would silently close at its result point."""
    t = object.__new__(pl.PipelineTransfer)
    t.producer = True

    def depth(value):
        monkeypatch.setattr(
            pl, "get_parallel", lambda: SimpleNamespace(pp_async_batch_depth=value)
        )

    depth(1)
    with envs.SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE.override(True):
        with envs.SGLANG_RAIDEN_PIPELINE_STAGE_PARTS.override(True):
            with pytest.raises(ValueError, match="stage groups"):
                t._init_overlap_state()
        depth(0)
        with pytest.raises(ValueError, match="pp-async-batch-depth"):
            t._init_overlap_state()
        depth(1)
        t._init_overlap_state()
    assert t.early_close and t.final_chunks is not None


def test_result_point_abort_keeps_pages_a_claimed_stage_group_transfer_reads(
    staged, monkeypatch
):
    """Stage groups register a stage's pages before the result point. An abort
    retired there freed them at once, under a transfer the decode may still be
    reading; and a stage freeing them on its own drain lets the stages' free
    pages, and so their next batches, differ. Each stage holds them until the
    in-flight queue's release consensus, which needs every stage drained."""
    from sglang.srt.disaggregation import prefill
    from sglang.srt.managers import scheduler

    w = staged
    claim_without_landing(w)
    w.decode.register_decode(ROOM, list(range(100, 112)), [200], None, w.infos)
    w.pump()
    for s in w.stages:
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 15)), [30], fence)
    w.pump()
    assert w.decode.maybe_start_pull(ROOM)
    s = w.stages[0]
    sender = RaidenKVSender.__new__(RaidenKVSender)
    sender.bootstrap_room = ROOM
    sender.kv_mgr = SimpleNamespace(_pipeline=s, sender_has_active_io=s.active)
    sender.abort = lambda: s.manager._aborted.update({ROOM: time.time()})
    sender.clear = lambda: None
    released = []
    for module in (prefill, scheduler):
        monkeypatch.setattr(
            module, "release_kv_cache", lambda req, *a, **k: released.append(req)
        )
    monkeypatch.setattr(scheduler, "get_parallel", lambda: SimpleNamespace(pp_size=2))
    sched = scheduler.Scheduler.__new__(scheduler.Scheduler)
    sched.transfer_backend = TransferBackend.RAIDEN
    sched.disagg_prefill_pending_chunk_rids = set()
    sched.req_to_metadata_buffer_idx_allocator = None
    sched.tree_cache = SimpleNamespace(finish=lambda *args: None)
    sched._deferred_chunked_sender_aborts = []
    sched.disagg_prefill_inflight_queue = []
    sched.scheduler_stage_metrics = None
    req = SimpleNamespace(
        rid="r",
        kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
        metadata_buffer_index=-1,
        disagg_kv_sender=sender,
        to_finish=None,
        pending_bootstrap=False,
        cache_request_handle=None,
    )
    assert sched._retire_aborted_prefill_result(req)
    assert released == [] and sched._deferred_chunked_sender_aborts == [req]
    poll(s)
    assert not sender.drained()  # the decode still reads group 0's pages
    s.engine.sent = [pl.stage_req_id(ROOM, 0)]
    poll(s)
    assert sender.drained()
    sched._drain_deferred_chunked_sender_aborts()
    sched.process_disagg_prefill_inflight_queue([])
    assert released == []
    sched.process_disagg_prefill_inflight_queue([req.rid])
    assert released == [req] and not sched._deferred_chunked_sender_aborts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
