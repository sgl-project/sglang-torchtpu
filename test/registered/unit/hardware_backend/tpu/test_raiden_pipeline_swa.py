"""Hybrid sliding-window KV on the Raiden pipeline transfer (pp_swa).

A hybrid pool's full-attention and sliding-window layers hold different block
counts, which one tpu-sync KVCacheManager rejects, so the window layers get a
second engine and move as twin transfers. Producer stages and decode ranks
exchange their real control messages through the in-memory bus of
test_raiden_pipeline_overlap.py; a fake controller per engine kind executes
each coordinated transfer byte for byte from the registered spans, so the
tests check which pages land on which attention-TP rank. No TPU or tpu-sync.
"""

import concurrent.futures
import importlib.util
import os
import pathlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import pipeline as pl
from sglang_tpu.disaggregation.raiden import pp_swa as ps
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import (
    StagingAdmission,
    slot_host_bytes,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "raiden_pipeline_overlap_harness",
    pathlib.Path(__file__).with_name("test_raiden_pipeline_overlap.py"),
)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

ROOM = 123456789
HEADS = 4  # model kv-heads; a producer stage holds every head
LANES, PAGE = 8, 4  # rank-4 RPA-seq page: [2 * heads, lanes, page] bytes
PRODUCER_BLOCKS = dict(full=32, swa=8)
DECODE_BLOCKS = dict(full=40, swa=6)
WINDOW_PAGES = 2
PREFILL = DisaggregationMode.PREFILL
DECODE = DisaggregationMode.DECODE
CANCELLED = "Request block registration was cancelled for req_id=x"


def kv(blocks, heads, generator=None):
    shape = (blocks, 2 * heads, LANES, PAGE)
    if generator is None:
        return torch.zeros(shape, dtype=torch.uint8)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)


def page_view(tensor):
    return tensor.view(torch.uint8).reshape(tensor.shape[0], -1)


def unit_key(unit):
    return (unit.job_replica_id, unit.data_name, unit.data_replica_idx)


class Domain:
    """A reshard store pair of one engine kind ("fa" or "swa"): registrations,
    claims and every coordinated transfer, executed byte for byte."""

    def __init__(self, kind):
        self.kind = kind
        self.producer_address = f"p{kind}"
        self.decode_address = f"d{kind}"
        self.registered = {}  # req_id -> unit key -> (uuid, block_ids, spans)
        self.transfers = []
        self.claimed = set()
        self.cancelled = []
        self.completed = []
        self.units = {}  # unit key -> (actor, {tag: tensor}, engine)
        self.auto_land = True
        self.reject = None  # Exception raised by the next coordinate_transfer
        self.fail_register = None

    def add_unit(self, actor, unit, tensors, engine):
        self.units[unit_key(unit)] = (actor, tensors, engine)

    def client(self, address):
        domain = self

        class Client:
            def register_request_blocks(
                self, req_id, uuid, unit, block_ids, pool_spans
            ):
                if domain.fail_register is not None:
                    raise domain.fail_register
                domain.registered.setdefault(req_id, {})[unit_key(unit)] = (
                    uuid,
                    list(block_ids),
                    pool_spans,
                )

            def complete_request_blocks(self, req_id, uuid, unit):
                domain.completed.append((req_id, unit.data_replica_idx))

            def cancel_request_blocks_if_unclaimed(self, req_id, uuid):
                if req_id in domain.claimed:
                    return False
                domain.cancelled.append(req_id)
                domain.registered.pop(req_id, None)
                return True

            def coordinate_transfer(self, **kwargs):
                assert kwargs["src_controller_address"] == address
                return domain.coordinate(**kwargs)

        return Client()

    def coordinate(self, **kw):
        assert kw["src_controller_address"] == self.producer_address
        assert kw["dst_controller_address"] == self.decode_address
        if self.reject is not None:
            error, self.reject = self.reject, None
            raise error
        req_id = kw["req_id"]
        tags = kw["transfer_pool_tags"]
        assert tags and all(t.startswith("swa.") == (self.kind == "swa") for t in tags)
        regs = self.registered.get(req_id, {})
        for unit in kw["src_units"]:
            reg = regs.get(unit_key(unit))
            if reg is None:
                raise RuntimeError(f"Missing producer block registration for {req_id}")
            assert reg[0] == kw["uuid"]
        self.claimed.add(req_id)
        self.transfers.append(kw)
        self._move(kw)
        if self.auto_land:
            self.land(req_id)
        return True

    def _move(self, kw):
        tags = kw["transfer_pool_tags"]
        counts, blocks = kw["dst_block_counts"], kw["dst_device_block_ids"]
        assert sum(counts) == len(blocks)
        tag_blocks, cursor = {}, 0
        for tag, count, skip in zip(tags, counts, kw["dst_skip_bytes"]):
            tag_blocks[tag] = (blocks[cursor : cursor + count], skip)
            cursor += count
        dst_units = kw["dst_units"]
        for unit in kw["src_units"]:
            _, block_ids, spans = self.registered[kw["req_id"]][unit_key(unit)]
            _, src_tensors, _ = self.units[unit_key(unit)]
            for entry in spans:
                if entry.tag not in tag_blocks:
                    continue
                dst_ids, skip = tag_blocks[entry.tag]
                src = page_view(src_tensors[entry.tag])
                for span in entry.spans:
                    assert span.count == 1
                    data = src[entry.block_ids[span.src_block_ordinal]][
                        span.src_offset_bytes : span.src_offset_bytes + span.size_bytes
                    ]
                    ordinal = getattr(span, "dst_unit_ordinal", None)
                    targets = dst_units if ordinal is None else [dst_units[ordinal]]
                    for dst in targets:
                        _, dst_tensors, _ = self.units[unit_key(dst)]
                        view = page_view(dst_tensors[entry.tag])
                        offset = span.dst_offset_bytes - skip
                        if offset < 0:
                            continue  # clipped destination prefix
                        page, within = divmod(offset, view.shape[1])
                        assert page < len(dst_ids), (kw["req_id"], entry.tag)
                        assert within + span.size_bytes <= view.shape[1]
                        view[dst_ids[page]][within : within + span.size_bytes] = data

    def land(self, req_id, *, failed=False):
        kw = next(t for t in self.transfers if t["req_id"] == req_id)
        for unit in kw["src_units"]:
            engine = self.units[unit_key(unit)][2]
            (engine.failed if failed else engine.sent).append(req_id)
        for unit in kw["dst_units"]:
            engine = self.units[unit_key(unit)][2]
            (engine.failed if failed else engine.received).append(req_id)


def make(mode, rank, world, name, *, pp_size, tp_size=1, stage_parts=False):
    """A PipelineTransfer with its window link, built like __init__ would."""
    t = object.__new__(pl.PipelineTransfer)
    producer = mode == PREFILL
    mgr = harness.Manager(mode, rank if producer else 0, pp_size, world.bus, name)
    mgr.attn_tp_size, mgr.attn_tp_rank = (1, 0) if producer else (tp_size, rank)
    mgr.kv_args.total_kv_head_num = HEADS
    mgr.kv_args.total_swa_kv_head_num = HEADS
    if producer:
        heads, full_ids, swa_ids = HEADS, [2 * rank + 1], [2 * rank]
        blocks = PRODUCER_BLOCKS
        gen = torch.Generator().manual_seed(1000 + rank)
    else:
        heads, full_ids, swa_ids = HEADS // tp_size, [1, 3], [0, 2]
        blocks, gen = DECODE_BLOCKS, None
    groups = SimpleNamespace(
        full=[kv(blocks["full"], heads, gen) for _ in full_ids],
        swa=[kv(blocks["swa"], heads, gen) for _ in swa_ids],
        full_layer_ids=full_ids,
        swa_layer_ids=swa_ids,
        num_draft_full=0,
        num_draft_swa=0,
    )
    side = "p" if producer else "d"
    t.manager = mgr
    t.producer = producer
    t.lock = threading.RLock()
    t.pools, swa_pools = ps.split_manifest(pl.pool_manifest(groups))
    t.tensors = list(groups.full)
    t.rank = mgr.pp_rank * mgr.attn_tp_size + mgr.attn_tp_rank
    t.workers = pp_size * mgr.attn_tp_size
    t.client_type = world.client_type
    t._init_overlap_state()
    t.early_parts = True
    t.part_min_pages = 4
    t.stage_parts = stage_parts and producer
    t.executor = harness.SyncExecutor()
    t.hbm_mem_type = 2
    t.sent, t.replies, t.release_requests, t.receives = {}, {}, {}, {}
    t.cleared, t.done_send, t.done_recv, t.failed, t.retired = (set() for _ in range(5))
    t.controller = f"{side}fa"
    t.unit_fields = dict(
        job_name="sglang",
        job_replica_id=t.controller,
        data_name="prefill" if producer else "decode",
        data_replica_idx=t.rank,
    )
    t.unit = SimpleNamespace(**t.unit_fields)
    t.client = world.client_type(t.controller)
    t.dst_units = [
        SimpleNamespace(**(t.unit_fields | {"data_replica_idx": r}))
        for r in range(mgr.attn_tp_size)
    ]
    t.max_blocks = PRODUCER_BLOCKS["full"]
    t.admission = StagingAdmission(4 * t.max_blocks, 4 * WINDOW_PAGES)
    t.engine = harness.Engine()
    link = ps.SwaLink(
        pools=swa_pools,
        tensors=list(groups.swa),
        controller=f"{side}swa",
        unit_fields=t.unit_fields | {"job_replica_id": f"{side}swa"},
        dst_units=[
            SimpleNamespace(
                **(
                    t.unit_fields
                    | {"job_replica_id": f"{side}swa", "data_replica_idx": r}
                )
            )
            for r in range(mgr.attn_tp_size)
        ],
        max_blocks=WINDOW_PAGES,
        slots=4,
        hold_undeclared=not producer,
    )
    link.client = world.client_type(link.controller)
    link.executor = None if producer else harness.SyncExecutor()
    link.engine = harness.Engine()
    t.swa = link
    world.fa.add_unit(
        t, t.unit, {p["tag"]: x for p, x in zip(t.pools, t.tensors)}, t.engine
    )
    world.swa.add_unit(
        t,
        link.unit,
        {p["tag"]: x for p, x in zip(link.pools, link.tensors)},
        link.engine,
    )
    return t


def build_world(*, tp_size=2, stage_parts=False):
    w = SimpleNamespace(bus={"queue": []}, fa=Domain("fa"), swa=Domain("swa"))
    by_address = {d.producer_address: d for d in (w.fa, w.swa)} | {
        d.decode_address: d for d in (w.fa, w.swa)
    }
    w.client_type = lambda address: by_address[address].client(address)
    w.stages = [
        make(PREFILL, r, w, f"p{r}", pp_size=2, stage_parts=stage_parts)
        for r in range(2)
    ]
    w.decodes = [
        make(DECODE, r, w, f"d{r}", pp_size=1, tp_size=tp_size) for r in range(tp_size)
    ]
    actors = {f"p{r}": s for r, s in enumerate(w.stages)} | {
        f"d{r}": d for r, d in enumerate(w.decodes)
    }

    def pump(only=None):
        keep = []
        while w.bus["queue"]:
            name, frames = w.bus["queue"].pop(0)
            if only is not None and name not in only:
                keep.append((name, frames))
                continue
            actors[name].on_message(frames)
        w.bus["queue"] = keep

    w.pump = pump
    w.infos = [
        dict(
            pp_rank=s.manager.pp_rank,
            rank_ip=s.manager.local_ip,
            rank_port=s.manager.rank_port,
        )
        for s in w.stages
    ]
    return w


@pytest.fixture
def world():
    return build_world()


@pytest.fixture
def staged():
    return build_world(stage_parts=True)


poll = harness.poll
Fence = harness.Fence

DST_FULL = list(range(20, 32))
DST_SWA = [3, 1]


def ask(w, full=DST_FULL, swa=DST_SWA, aux_index=None):
    for d in w.decodes:
        d.register_decode(ROOM, list(full), list(swa), aux_index, w.infos)
    w.pump()


def close(s, full, swa):
    harness.prefill_pages(s, list(full), swa=list(swa))
    assert s.finalize_send(ROOM, None)


def poll_all(w):
    for actor in (*w.stages, *w.decodes):
        poll(actor)


def tp_rows(rank, tp_size):
    per = HEADS // tp_size
    return slice(2 * per * rank, 2 * per * (rank + 1))


def assert_landed(w, pairs_by_kind):
    """Each decode rank holds exactly its kv-head slice of the listed source
    pages, and every other destination page is untouched."""
    tp_size = len(w.decodes)
    for rank, d in enumerate(w.decodes):
        tensors = {p["tag"]: x for p, x in zip(d.pools, d.tensors)} | {
            p["tag"]: x for p, x in zip(d.swa.pools, d.swa.tensors)
        }
        written = {tag: set() for tag in tensors}
        for kind, pairs in pairs_by_kind.items():
            for s in w.stages:
                src = {p["tag"]: x for p, x in zip(s.pools, s.tensors)} | {
                    p["tag"]: x for p, x in zip(s.swa.pools, s.swa.tensors)
                }
                for tag, x in src.items():
                    if tag.startswith("swa.") != (kind == "swa"):
                        continue
                    for src_block, dst_block in pairs:
                        expect = x[src_block][tp_rows(rank, tp_size)]
                        assert torch.equal(tensors[tag][dst_block], expect), (rank, tag)
                        written[tag].add(dst_block)
        for tag, x in tensors.items():
            for block in range(x.shape[0]):
                if block not in written[tag]:
                    assert not x[block].any(), (rank, tag, block)


def ids(domain):
    return [t["req_id"] for t in domain.transfers]


class Deferred:
    """An executor whose calls run only when the test runs them."""

    def __init__(self):
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        self.calls.append((future, lambda: fn(*args, **kwargs)))
        return future

    def run(self):
        for future, call in self.calls:
            future.set_result(call())
        self.calls = []


def stage_ready(w):
    """Every stage's forward of the final chunk completed."""
    for s in w.stages:
        fence = Fence()
        fence.ready = True
        s.stage_ready(ROOM, list(range(10, 13)), [6, 4], fence)
        s.advance_stages()


# ----- units -----------------------------------------------------------------


def test_split_manifest_rebases_storage_indices():
    groups = SimpleNamespace(
        full=[kv(32, 2), kv(32, 2)],
        swa=[kv(8, 2)],
        full_layer_ids=[1, 5],
        swa_layer_ids=[3],
        num_draft_full=0,
        num_draft_swa=0,
    )
    full, swa = ps.split_manifest(pl.pool_manifest(groups))
    assert [(p["tag"], p["storage_index"], p["num_blocks"]) for p in full] == [
        ("fa.l1", 0, 32),
        ("fa.l5", 1, 32),
    ]
    assert [(p["tag"], p["storage_index"], p["num_blocks"]) for p in swa] == [
        ("swa.l3", 0, 8)
    ]


def test_split_manifest_puts_draft_window_layers_on_the_window_engine():
    """A separate draft pool's layers follow their target kind: its window
    layers hold the window's block count, so they join the window engine."""
    groups = SimpleNamespace(
        full=[kv(32, 2), kv(32, 2)],
        swa=[kv(8, 2), kv(8, 2)],
        full_layer_ids=[1, 0],
        swa_layer_ids=[3, 1],
        num_draft_full=1,
        num_draft_swa=1,
    )
    full, swa = ps.split_manifest(pl.pool_manifest(groups))
    assert [(p["tag"], p["storage_index"], p["num_blocks"]) for p in full] == [
        ("fa.l1", 0, 32),
        ("draft.fa.l0", 1, 32),
    ]
    assert [(p["tag"], p["storage_index"], p["num_blocks"]) for p in swa] == [
        ("swa.l3", 0, 8),
        ("draft.swa.l1", 1, 8),
    ]


# ----- closing transfer ------------------------------------------------------


def test_closing_moves_full_and_window_pages_to_each_tp_rank(world):
    w = world
    ask(w)
    for s in w.stages:
        close(s, range(10, 15), [6, 4])
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    (fa,) = w.fa.transfers
    (swa,) = w.swa.transfers
    assert fa["req_id"] == str(ROOM) and fa["uuid"] == ROOM
    assert sorted(fa["transfer_pool_tags"]) == ["fa.l1", "fa.l3"]
    assert fa["dst_device_block_ids"] == DST_FULL[:5] * 2
    assert swa["req_id"] == f"swa:{ROOM}" and swa["uuid"] == ps.twin_uuid(str(ROOM))
    assert sorted(swa["transfer_pool_tags"]) == ["swa.l0", "swa.l2"]
    assert swa["dst_device_block_ids"] == DST_SWA * 2
    assert [unit_key(u) for u in swa["src_units"]] == [
        ("pswa", "prefill", 0),
        ("pswa", "prefill", 1),
    ]
    assert [unit_key(u) for u in swa["dst_units"]] == [
        ("dswa", "decode", 0),
        ("dswa", "decode", 1),
    ]
    poll_all(w)
    assert_landed(
        w,
        dict(
            fa=list(zip(range(10, 15), DST_FULL)),
            swa=list(zip([6, 4], DST_SWA)),
        ),
    )
    for d in w.decodes:
        assert str(ROOM) in d.manager._completed_recv and not d.active(ROOM)
        assert d.admission.pages() == 0 and d.admission.swa_pages() == 0
    for s in w.stages:
        assert str(ROOM) in s.manager._completed_send and not s.active(ROOM)
    assert sorted(w.fa.completed) == [(str(ROOM), 0), (str(ROOM), 1)]
    assert sorted(w.swa.completed) == [(f"swa:{ROOM}", 0), (f"swa:{ROOM}", 1)]


def test_early_close_registers_each_shares_window_twin(world):
    """SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE: each stage registers its window
    pages as the twin of its early share; the decode moves both halves once
    the last stage has closed, and the result point adds nothing."""
    w = world
    for s in w.stages:
        s.early_close = True
    ask(w)
    w.stages[0].close_early(ROOM, list(range(10, 15)), [6, 4])
    w.pump()
    assert set(w.fa.registered[str(ROOM)]) == {("pfa", "prefill", 0)}
    assert set(w.swa.registered[f"swa:{ROOM}"]) == {("pswa", "prefill", 0)}
    assert not any(d.maybe_start_pull(ROOM) for d in w.decodes)
    w.stages[1].close_early(ROOM, list(range(10, 15)), [6, 4])
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    assert ids(w.fa) == [str(ROOM)] and ids(w.swa) == [f"swa:{ROOM}"]
    poll_all(w)
    assert_landed(
        w,
        dict(
            fa=list(zip(range(10, 15), DST_FULL)),
            swa=list(zip([6, 4], DST_SWA)),
        ),
    )
    for d in w.decodes:
        assert str(ROOM) in d.manager._completed_recv and not d.active(ROOM)
    queued = len(w.bus["queue"])
    for s in w.stages:
        close(s, range(10, 15), [6, 4])
    assert len(w.bus["queue"]) == queued
    assert len(w.fa.transfers) == len(w.swa.transfers) == 1


@pytest.mark.parametrize("first", ["fa", "swa"])
def test_room_waits_for_both_halves(world, first):
    w = world
    w.fa.auto_land = w.swa.auto_land = False
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    domains = dict(fa=(w.fa, str(ROOM)), swa=(w.swa, f"swa:{ROOM}"))
    domain, req_id = domains[first]
    domain.land(req_id)
    poll_all(w)
    for actor in (*w.stages, *w.decodes):
        assert actor.active(ROOM)  # the other half may still move pages
    assert not any(str(ROOM) in d.manager._completed_recv for d in w.decodes)
    assert not any(str(ROOM) in s.manager._completed_send for s in w.stages)
    domain, req_id = domains["swa" if first == "fa" else "fa"]
    domain.land(req_id)
    poll_all(w)
    assert all(str(ROOM) in d.manager._completed_recv for d in w.decodes)
    assert all(not a.active(ROOM) for a in (*w.stages, *w.decodes))


def test_halves_landing_before_a_rank_reads_its_readies_are_held(world):
    """Rank 0 coordinates for every attention-TP rank, so both halves can land
    on rank 1 before rank 1 knows the room has a twin."""
    w = world
    d0, d1 = w.decodes
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump(only={"d0"})  # rank 1's READYs are still in flight
    assert d0.maybe_start_pull(ROOM)
    poll(d1)
    assert str(ROOM) not in d1.manager._completed_recv  # landed, but undeclared
    w.pump()
    assert d1.maybe_start_pull(ROOM)
    poll(d1)
    assert str(ROOM) in d1.manager._completed_recv and not d1.active(ROOM)
    assert d1.admission.pages() == 0 and d1.admission.swa_pages() == 0


def test_twin_landing_before_the_rank_registers_the_room_counts(world):
    """Rank 0 can also coordinate before rank 1 has registered the room at
    all; the window landing must still count once rank 1 accepts it."""
    w = world
    d0, d1 = w.decodes
    d0.register_decode(ROOM, DST_FULL, DST_SWA, None, w.infos)
    w.pump()
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    assert d0.maybe_start_pull(ROOM)
    poll(d1)  # both halves landed on rank 1, which has no record of the room
    d1.register_decode(ROOM, DST_FULL, DST_SWA, None, w.infos)
    w.pump()
    assert d1.maybe_start_pull(ROOM)
    poll(d1)
    assert str(ROOM) in d1.manager._completed_recv
    assert not d1.active(ROOM)
    assert d1.admission.pages() == 0 and d1.admission.swa_pages() == 0


@pytest.mark.parametrize("failing", ["swa", "fa"])
def test_failed_half_fails_the_room_once_the_other_half_settled(world, failing):
    """A failed half must not release the room's pages while the other half
    still moves bytes into (decode) or out of (stage) them."""
    w = world
    w.fa.auto_land = w.swa.auto_land = False
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    halves = dict(fa=(w.fa, str(ROOM)), swa=(w.swa, f"swa:{ROOM}"))
    domain, req_id = halves[failing]
    domain.land(req_id, failed=True)
    poll_all(w)
    for actor in (*w.stages, *w.decodes):
        assert str(ROOM) not in actor.manager._failed
        assert actor.active(ROOM)  # the other half still runs
    domain, req_id = halves["fa" if failing == "swa" else "swa"]
    domain.land(req_id)
    poll_all(w)
    for actor in (*w.stages, *w.decodes):
        assert str(ROOM) in actor.manager._failed and not actor.active(ROOM)
        assert str(ROOM) not in actor.manager._completed_send
        assert str(ROOM) not in actor.manager._completed_recv


@pytest.mark.parametrize("cancelled", ["swa", "fa"])
def test_cancelled_half_keeps_the_pages_its_twin_still_writes(cancelled, monkeypatch):
    """An abort makes the stages cancel whichever half nobody claimed yet. A
    claimed other half still lands in the room's pages, so the decode keeps
    them although the room failed."""
    w = build_world(tp_size=1)
    d = w.decodes[0]
    w.fa.auto_land = w.swa.auto_land = False
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    getattr(w, cancelled).reject = RuntimeError(CANCELLED)
    assert d.maybe_start_pull(ROOM)
    poll(d)
    assert ROOM in d.failed
    assert d.active(ROOM)
    claimed = w.swa if cancelled == "fa" else w.fa
    claimed.land(claimed.transfers[0]["req_id"])
    poll(d)
    if cancelled == "fa":
        assert not d.active(ROOM)
        return
    # The window half never settles, so the full-attention landing is not
    # reported; the room is held until that push's native deadline.
    assert d.active(ROOM)
    later = time.monotonic() + d.manager._timeout_s + 1
    monkeypatch.setattr(pl.time, "monotonic", lambda: later)
    assert not d.active(ROOM)


def test_transfer_accepted_after_the_room_failed_keeps_its_pages():
    """A coordination still running when the room fails can still be
    accepted, and its push then lands until the native deadline."""
    w = build_world(tp_size=1)
    d = w.decodes[0]
    w.swa.auto_land = False
    d.swa.executor = Deferred()
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    w.fa.reject = RuntimeError(CANCELLED)
    assert d.maybe_start_pull(ROOM)
    poll(d)
    assert ROOM in d.failed and d.active(ROOM)  # the twin is still coordinating
    d.swa.executor.run()  # the controller accepts the twin
    poll(d)
    assert d.active(ROOM)
    w.swa.land(f"swa:{ROOM}")
    poll(d)
    assert not d.active(ROOM)


def test_twin_that_cannot_register_cancels_its_transfer_and_fails_the_room(world):
    w = world
    ask(w)
    w.swa.fail_register = RuntimeError("window store unreachable")
    s = w.stages[0]
    harness.prefill_pages(s, list(range(10, 13)), swa=[6, 4])
    assert not s.finalize_send(ROOM, None)
    assert w.fa.cancelled == [str(ROOM)]
    assert "window store unreachable" in s.manager.failure_records[ROOM]
    assert not w.bus["queue"]  # nothing announced
    poll(s)
    assert not s.active(ROOM)


def test_release_cancels_the_transfer_and_its_twin(world):
    """A decode that already holds every page releases the room: the stage
    cancels both registrations, which nobody will claim."""
    w = world
    ask(w, full=[], swa=[])
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    w.pump()  # rank 0's RELEASE reaches both stages
    assert not w.fa.transfers and not w.swa.transfers
    assert sorted(w.fa.cancelled) == [str(ROOM)] * 2
    assert sorted(w.swa.cancelled) == [f"swa:{ROOM}"] * 2
    poll_all(w)
    for s in w.stages:
        assert str(ROOM) in s.manager._completed_send and not s.active(ROOM)
    assert all(str(ROOM) in d.manager._completed_recv for d in w.decodes)


def test_abort_cancels_an_unclaimed_transfer_and_twin(world):
    w = world
    ask(w)
    s = w.stages[0]
    close(s, range(10, 13), [6, 4])
    assert s.active(ROOM)
    s.manager._aborted[ROOM] = time.time()
    poll(s)
    assert w.fa.cancelled == [str(ROOM)] and w.swa.cancelled == [f"swa:{ROOM}"]
    assert ROOM in s.failed and not s.active(ROOM)


def test_abort_after_the_claim_keeps_pages_until_both_halves_settled(world):
    w = world
    w.fa.auto_land = w.swa.auto_land = False
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    assert w.decodes[0].maybe_start_pull(ROOM)
    s = w.stages[0]
    s.manager._aborted[ROOM] = time.time()
    poll(s)
    assert not w.fa.cancelled and not w.swa.cancelled and s.active(ROOM)
    w.fa.land(str(ROOM))
    poll(s)
    assert s.active(ROOM)
    w.swa.land(f"swa:{ROOM}")
    poll(s)
    assert not s.active(ROOM)


@pytest.mark.parametrize("stage_parts", [False, True])
def test_abort_cancels_an_unclaimed_half_beside_a_claimed_one(stage_parts):
    """The decode claimed the window half, but its full-attention coordination
    failed before claiming. The stage's abort still cancels the full-attention
    registration, and the transfer settles as failed once the window push
    lands, instead of holding the stage's pages until the claim expires."""
    w = build_world(tp_size=1, stage_parts=stage_parts)
    w.swa.auto_land = False
    w.fa.reject = RuntimeError("DEADLINE_EXCEEDED: coordinate_transfer")
    ask(w)
    if stage_parts:
        stage_ready(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    w.decodes[0].maybe_start_pull(ROOM)
    req_id = pl.stage_req_id(ROOM, 0) if stage_parts else str(ROOM)
    s = w.stages[0]
    s.manager._aborted[ROOM] = time.time()
    poll(s)
    assert req_id in w.fa.cancelled and f"swa:{req_id}" not in w.swa.cancelled
    assert s.active(ROOM)  # the window push still reads the stage's pages
    w.swa.land(f"swa:{req_id}")
    poll(s)
    assert ROOM in s.failed and not s.active(ROOM)
    assert (f"swa:{req_id}", 0) in w.swa.completed


def test_producer_and_decode_must_agree_on_the_window_engine():
    """A stage with a window engine announcing to a one-engine decode (or the
    reverse) fails the room instead of planning a transfer that cannot match."""
    w = build_world(tp_size=1)
    one_engine = w.decodes[0]
    one_engine.swa = None
    ask(w)
    close(w.stages[0], range(10, 13), [6, 4])
    with pytest.raises(ValueError, match="disagree on the sliding-window engine"):
        w.pump()


def test_unplaceable_window_fails_the_room_not_the_decode(world):
    w = world
    ask(w, swa=[])  # the decode holds no window page for a stage's window
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    d = w.decodes[0]
    assert d.maybe_start_pull(ROOM) is False
    assert "sliding window" in d.manager.failure_records[ROOM]
    assert ROOM in d.failed and not w.fa.transfers and not w.swa.transfers


def test_unplaceable_stage_group_fails_the_room_under_the_state_lock():
    """The decode poll starts stage groups too; failing one there takes the
    manager's state lock, as maybe_start_pull does."""
    w = build_world(tp_size=1, stage_parts=True)
    d = w.decodes[0]
    d.manager._state_lock = threading.Lock()
    held = []
    fail = d.manager._fail_room_locked

    def fail_room_locked(*args, **kwargs):
        held.append(d.manager._state_lock.locked())
        fail(*args, **kwargs)

    d.manager._fail_room_locked = fail_room_locked
    ask(w, swa=[])
    stage_ready(w)
    w.pump()
    poll(d)
    assert held == [True]
    assert "sliding window" in d.manager.failure_records[ROOM]


# ----- with the other pipeline features --------------------------------------


def test_early_parts_carry_no_window_and_the_closing_twin_does(world):
    w = world
    ask(w)
    for s in w.stages:
        start = harness.prefill_pages(s, list(range(10, 18)))
        s.offer_part(ROOM, start, list(range(10, 18)))
    w.pump()
    assert w.decodes[0].maybe_start_pull(ROOM) is False
    assert ids(w.fa) == [f"{ROOM}.p0"] and not w.swa.transfers
    for s in w.stages:
        close(s, [18, 19], [6, 4])
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    assert ids(w.fa) == [f"{ROOM}.p0", str(ROOM)]
    assert ids(w.swa) == [f"swa:{ROOM}"]
    poll_all(w)
    assert_landed(
        w,
        dict(
            fa=list(zip(range(10, 20), DST_FULL)),
            swa=list(zip([6, 4], DST_SWA)),
        ),
    )
    assert all(str(ROOM) in d.manager._completed_recv for d in w.decodes)


def test_stage_groups_move_each_groups_window_as_its_twin(staged):
    w = staged
    ask(w)
    fences = [Fence(), Fence()]
    for s, fence in zip(w.stages, fences):
        s.stage_ready(ROOM, list(range(10, 15)), [6, 4], fence)
    fences[0].ready = True
    w.stages[0].advance_stages()
    w.pump()
    assert w.decodes[0].maybe_start_pull(ROOM) is False  # stage 1 computing
    assert ids(w.fa) == [f"{ROOM}.s0"] and ids(w.swa) == [f"swa:{ROOM}.s0"]
    (twin,) = w.swa.transfers
    assert twin["transfer_pool_tags"] == ["swa.l0"]  # stage 0's window layer only
    assert [unit_key(u) for u in twin["src_units"]] == [("pswa", "prefill", 0)]
    assert twin["uuid"] == ps.twin_uuid(pl.stage_req_id(ROOM, 0))
    fences[1].ready = True
    w.stages[1].advance_stages()
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    assert ids(w.swa) == [f"swa:{ROOM}.s0", f"swa:{ROOM}.s1"]
    for s in w.stages:
        close(s, range(10, 15), [6, 4])
    w.pump()
    poll_all(w)
    assert_landed(
        w,
        dict(
            fa=list(zip(range(10, 15), DST_FULL)),
            swa=list(zip([6, 4], DST_SWA)),
        ),
    )
    for d in w.decodes:
        assert str(ROOM) in d.manager._completed_recv and not d.active(ROOM)
    for s in w.stages:
        assert str(ROOM) in s.manager._completed_send and not s.active(ROOM)
    assert sorted(w.swa.completed) == [(f"swa:{ROOM}.s0", 0), (f"swa:{ROOM}.s1", 1)]


def test_stage_group_waits_for_its_twin(staged):
    w = staged
    w.swa.auto_land = False
    ask(w)
    stage_ready(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    assert all(d.maybe_start_pull(ROOM) for d in w.decodes)
    poll_all(w)
    assert not any(str(ROOM) in d.manager._completed_recv for d in w.decodes)
    assert all(s.stage[ROOM]["done"] is False for s in w.stages)
    for group in range(2):
        w.swa.land(f"swa:{pl.stage_req_id(ROOM, group)}")
    poll_all(w)
    assert all(str(ROOM) in d.manager._completed_recv for d in w.decodes)
    assert all(str(ROOM) in s.manager._completed_send for s in w.stages)


def test_window_staging_is_admitted_with_the_full_attention_transfer():
    """With window staging for one twin, a second room waits in line without
    holding full-attention staging, and starts once the first room settles."""
    w = build_world(tp_size=1)
    d = w.decodes[0]
    d.admission = StagingAdmission(4 * d.max_blocks, WINDOW_PAGES)  # one twin
    w.fa.auto_land = w.swa.auto_land = False
    other = ROOM + 1
    for room in (ROOM, other):
        d.register_decode(room, DST_FULL[:3], DST_SWA, None, w.infos)
    w.pump()
    for room in (ROOM, other):
        for s in w.stages:
            with s.manager._state_lock:
                s.manager._prefill.send_blocks[room] = [10, 11, 12]
                s.manager._prefill.send_swa_blocks[room] = [6, 4]
            assert s.finalize_send(room, None)
    w.pump()
    assert d.maybe_start_pull(ROOM) is True
    assert d.maybe_start_pull(other) is False
    assert d.admission.waiting == other and other not in d.admission
    assert ids(w.fa) == [str(ROOM)] and ids(w.swa) == [f"swa:{ROOM}"]
    w.fa.land(str(ROOM))
    w.swa.land(f"swa:{ROOM}")
    poll(d)
    assert d.maybe_start_pull(other) is True
    assert ids(w.swa) == [f"swa:{ROOM}", f"swa:{other}"]


def test_twin_coordination_is_retried_while_its_registration_is_missing():
    w = build_world(tp_size=1)
    d = w.decodes[0]
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    w.swa.reject = RuntimeError("Missing producer block registration for swa")
    assert d.maybe_start_pull(ROOM)
    poll(d)
    assert ROOM not in d.failed and ROOM in d.admission  # kept for the retry
    twin = d.receives[ROOM].twins[()]
    assert twin["next_attempt"] > 0
    twin["next_attempt"] = 0.0
    assert d.maybe_start_pull(ROOM)
    poll(d)
    assert ids(w.swa) == [f"swa:{ROOM}"]
    assert str(ROOM) in d.manager._completed_recv and ROOM not in d.admission


def test_rejected_twin_fails_the_room_and_holds_its_pages():
    w = build_world(tp_size=1)
    d = w.decodes[0]
    w.fa.auto_land = False
    ask(w)
    for s in w.stages:
        close(s, range(10, 13), [6, 4])
    w.pump()
    w.swa.reject = RuntimeError("INVALID_ARGUMENT: Pools tagged swa.l0 differ")
    assert d.maybe_start_pull(ROOM)
    poll(d)  # must not raise
    assert "Pools tagged swa.l0" in d.manager.failure_records[ROOM]
    assert d.active(ROOM)  # the full-attention push still lands until its deadline


# ----- construction ----------------------------------------------------------


class Buffer:
    """A materialized fp8 rank-4 KV tensor stand-in."""

    dtype = "torch.float8_e4m3fn"

    def __init__(self, pages):
        self.shape = (pages, 2 * HEADS, 128, 256)
        self.materialized = True

    def element_size(self):
        return 1


def _initialize(groups, *, bounded_staging=True, real_budget=False):
    """PipelineTransfer(...) over fake tpu-sync natives; the result records
    the engines, stores, work units and slot plans it made. Slot counts come
    from RaidenKVManager's host budget with ``real_budget``, else fixed."""
    r = SimpleNamespace(pipe=None, error=None, engines=[], stores=[], units=[])
    r.slot_calls = []
    manager = SimpleNamespace(
        disaggregation_mode="decode",
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_dp_size=1,
        system_dp_size=1,
        attn_cp_size=1,
        pp_size=1,
        pp_rank=0,
        local_ip="10.0.0.1",
        _control_port=31000,
        _local_rank=0,
        _timeout_s=120,
        server_args=SimpleNamespace(nnodes=1),
        kv_args=SimpleNamespace(
            page_size=256, total_kv_head_num=HEADS, total_swa_kv_head_num=HEADS
        ),
        _max_request_blocks=lambda: 64,
        _swa_max_request_blocks=lambda: 3,
        _slot_ceiling=lambda: 64,
    )

    def num_slots(tensors, max_blocks, engines=1, **budget):
        r.slot_calls.append(
            (
                len(tensors),
                max_blocks,
                engines,
                budget.get("reserved_bytes", 0),
                budget.get("paired_slot_bytes", 0),
            )
        )
        if real_budget:
            return RaidenKVManager._num_slots(
                manager, tensors, max_blocks, engines, **budget
            )
        return 4 if max_blocks == 3 else 6

    manager._num_slots = num_slots

    class Engine:
        def __init__(self, **kwargs):
            # tpu-sync KVCacheManagerBase's block-count check.
            majors = [int(t.shape[0]) for t in kwargs["kv_caches"]]
            for layer, major in enumerate(majors):
                if major != majors[0]:
                    raise RuntimeError(
                        f"Layer {layer} has major dimension {major} but layer 0 has "
                        f"{majors[0]}; every layer must hold the same number of blocks "
                        "because they share one block-id space"
                    )
            r.engines.append(
                dict(kwargs, private=os.environ.get(ps._PRIVATE_WORKER_ENV))
            )
            self.transfer_address = f"10.0.0.1:{kwargs['local_control_port'] + 1}"
            self.listener_address = f"10.0.0.1:{kwargs['local_control_port'] + 100}"
            self.register_pools = Mock()

        if bounded_staging:

            @staticmethod
            def register_pools(
                pools, *, staging_leases=0, staging_blocks_per_pool=None
            ):
                pass  # signature only: the instance attribute records the calls

        else:

            @staticmethod
            def register_pools(pools):
                pass

    def store(raiden_id, store_server_ip):
        r.stores.append(
            (raiden_id["data_name"], os.environ.get(ps._PRIVATE_WORKER_ENV))
        )
        port = 32000 + 10 * len(r.stores)
        return SimpleNamespace(
            reshard_service_port=port, raiden_controller_address=f"10.0.0.1:{port + 1}"
        )

    def client(address):
        return SimpleNamespace(
            register_work_unit=lambda **kw: r.units.append((address, kw))
        )

    native = {
        "torch_tpu._internal.compile": SimpleNamespace(
            tpu_torch_compile=SimpleNamespace(
                get_device_layout_if_materialized=lambda t: (
                    (3, 2, 1, 0),
                    ((8, 128),),
                    8,
                )
            )
        ),
        "tpu_sync.api.torch.kv_cache_manager": SimpleNamespace(KVCacheManager=Engine),
        "tpu_sync.api.torch.reshard_client": SimpleNamespace(ReshardClient=client),
        "tpu_sync.api.torch.reshard_store": SimpleNamespace(
            RaidenId=lambda **kw: kw, ReshardStore=store
        ),
        "tpu_sync.rpc": SimpleNamespace(
            raiden_service_pb2=SimpleNamespace(
                MEMORY_TYPE_HBM=2,
                StartTransferRequest=SimpleNamespace(
                    DESCRIPTOR=SimpleNamespace(fields_by_name={"wire_pool_indices": 1})
                ),
            )
        ),
    }
    with patch.dict(sys.modules, native):
        try:
            r.pipe = pl.PipelineTransfer(manager, groups)
        except Exception as error:
            r.error = error
    for pipe in (r.pipe, getattr(r.pipe, "swa", None)):
        if pipe is not None and pipe.executor is not None:
            pipe.executor.shutdown(wait=False)
    return r


def hybrid_groups():
    return SimpleNamespace(
        full=[Buffer(5121)],
        swa=[Buffer(1297), Buffer(1297), Buffer(1297)],
        full_layer_ids=[3],
        swa_layer_ids=[0, 1, 2],
        num_draft_full=0,
        num_draft_swa=0,
    )


def test_hybrid_worker_builds_a_window_engine_beside_the_full_one():
    """One full layer of 5121 blocks and three window layers of 1297 blocks:
    two engines whose storages each share one block count, each registered
    with its own store and work unit; the window store and engine run on
    private native servers."""
    groups = hybrid_groups()
    r = _initialize(groups)
    assert r.error is None
    full, window = r.engines
    assert [b.shape[0] for b in full["kv_caches"]] == [5121]
    assert [b.shape[0] for b in window["kv_caches"]] == [1297] * 3
    assert full["local_control_port"] == 31000 and window["local_control_port"] == 31002
    assert full["private"] is None and window["private"] == "1"
    assert r.stores == [("pipeline", None), ("pipeline-swa", "1")]
    assert ps._PRIVATE_WORKER_ENV not in os.environ
    assert (full["max_blocks"], full["num_slots"]) == (64, 6)
    assert (window["max_blocks"], window["num_slots"]) == (3, 4)
    assert full["worker_id"] == window["worker_id"] == "worker_0"
    assert full["raiden_controller_address"] != window["raiden_controller_address"]
    pipe = r.pipe
    pipe.engine.register_pools.assert_called_once_with(
        pipe.pools, staging_leases=6, staging_blocks_per_pool=[64]
    )
    pipe.swa.engine.register_pools.assert_called_once_with(
        pipe.swa.pools, staging_leases=4, staging_blocks_per_pool=[3, 3, 3]
    )
    assert [p["tag"] for p in pipe.pools] == ["fa.l3"]
    assert [(p["tag"], p["storage_index"]) for p in pipe.swa.pools] == [
        ("swa.l0", 0),
        ("swa.l1", 1),
        ("swa.l2", 2),
    ]
    (fa_addr, fa_unit), (swa_addr, swa_unit) = r.units
    assert swa_addr == pipe.swa.controller != fa_addr
    assert swa_unit["unit"].job_replica_id == pipe.swa.controller
    assert swa_unit["transfer_rank"] == fa_unit["transfer_rank"] == 0
    # The window engine sizes its slots in pairs with full-attention ones and
    # takes its staging out of the rank's budget first.
    swa_call, fa_call = r.slot_calls
    assert swa_call[1:3] == (3, 1) and fa_call[1] == 64
    assert swa_call[4] == slot_host_bytes(groups.full, 64)
    assert fa_call[3:] == (4 * slot_host_bytes(groups.swa, 3), 0)
    # A twin's window pages are admitted with its full-attention transfer.
    assert (pipe.admission.capacity, pipe.admission.swa_capacity) == (6 * 64, 4 * 3)


@pytest.mark.parametrize("budget_mib, slots", [(128, (7, 7)), (143.75, (8, 7))])
def test_window_engine_never_has_more_slots_than_the_full_attention_one(
    budget_mib, slots, monkeypatch
):
    """A full-attention slot stages 16 MiB and a window slot 2.25 MiB. 128 MiB
    affords 7 pairs; 143.75 MiB leaves the full-attention engine one more
    slot, which early parts use without a twin."""
    monkeypatch.setenv("SGLANG_RAIDEN_HOST_POOL_GB", str(budget_mib / 1024))
    r = _initialize(hybrid_groups(), real_budget=True)
    assert r.error is None
    full, window = r.engines
    assert (full["num_slots"], window["num_slots"]) == slots


def test_window_engine_without_bounded_staging_is_refused_before_any_engine():
    r = _initialize(hybrid_groups(), bounded_staging=False)
    assert isinstance(r.error, RuntimeError)
    assert "--disable-hybrid-swa-memory" in str(r.error)
    assert r.engines == [] and r.stores == []


def test_full_attention_only_stage_is_unchanged():
    groups = SimpleNamespace(
        full=[Buffer(5121)],
        swa=[],
        full_layer_ids=[3],
        swa_layer_ids=[],
        num_draft_full=0,
        num_draft_swa=0,
    )
    r = _initialize(groups)
    (engine,) = r.engines
    assert engine["private"] is None and r.pipe.swa is None
    assert r.stores == [("pipeline", None)] and r.slot_calls == [(1, 64, 1, 0, 0)]
    assert r.pipe.admission.swa_capacity is None


@pytest.mark.parametrize("local_swa", [True, False])
def test_workers_must_agree_with_rank_zero_on_the_window_engine(local_swa):
    pipe = object.__new__(pl.PipelineTransfer)
    pipe.rank = 1
    pipe.manager = SimpleNamespace(
        pp_size=2, attn_tp_size=1, local_ip="10.0.0.2", _control_port=31004
    )
    rank0 = ("a:1", "b:2") if local_swa else ("a:1", "b:2", "c:3", "d:4")
    group = SimpleNamespace(broadcast_object=lambda obj, src: rank0)
    native = {
        "tpu_sync.api.torch.reshard_store": SimpleNamespace(
            RaidenId=Mock(), ReshardStore=Mock()
        )
    }
    with (
        patch.dict(sys.modules, native),
        patch.object(pl, "get_parallel", lambda: SimpleNamespace(pp_group=group)),
    ):
        with pytest.raises(ValueError, match="--disable-hybrid-swa-memory"):
            pipe._rendezvous(swa=local_swa)
        assert pipe._rendezvous(swa=not local_swa)[:2] == ("a:1", "b:2")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
