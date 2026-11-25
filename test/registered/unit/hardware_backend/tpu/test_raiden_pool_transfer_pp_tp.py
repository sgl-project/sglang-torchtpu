"""A pipeline-parallel prefill on the Raiden pool transfer (P: PP -> D: TP).

Each stage of a GLM-5.3-Flash prefill holds the split sparse-MLA pages of its
own DSA layers and the full-width state rows of its own KDA layers; the last
stage also holds the MTP draft's pool. Every attention-TP rank of the decode
holds every page whole, or with a split cache its share of each latent and
index-key page, and its heads' share of every state row, so each stage sends
each page or share to every decode rank and cuts each state row into the
ranks' shares. Of the indexer's raw rows only the last page moves, into the
request's last page or, on a decode with a raw ring, its ring page. The pools are the real ones, small; the workers exchange their real
control messages through the in-memory bus of test_raiden_pipeline_overlap.py,
and a fake reshard store per engine executes each coordinated transfer byte for
byte from the registered spans, into every destination rank. No TPU or tpu-sync.
"""

import importlib.util
import pathlib
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import kv_pool
from sglang_tpu.disaggregation.raiden import pool_side_engine as ps
from sglang_tpu.disaggregation.raiden import pool_transfer as pl
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool
from sglang_tpu.mamba_pool import TpuMambaPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).with_name(file)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load("raiden_pipeline_overlap_harness", "test_raiden_pipeline_overlap.py")
tp_test = _load("raiden_pool_transfer_tp_test", "test_raiden_pool_transfer_tp.py")
blocks_of, Store, HEADS = tp_test.blocks_of, tp_test.Store, tp_test.HEADS

ROOM = 987654321
TP = 4
# 64 tokens: four ranks' shares each hold whole indexer pools.
PAGE_TOKENS, PAGES = 64, 32
# Two stages of four layers, a DSA layer last in each, as in GLM-5.3-Flash.
STAGES = [range(0, 4), range(4, 8)]
DSA_LAYERS = [3, 7]
PREFILL, DECODE = DisaggregationMode.PREFILL, DisaggregationMode.DECODE
# A decode raw ring: request slots and ring pages per slot.
RING_SLOTS, RING_PAGES = 4, 2


def pools(layers, tp, *, draft, seed=None, shard_rank=None, ring=False):
    """What a worker registers for ``layers`` at attention TP ``tp``: their
    split pool, the MTP draft's with ``draft``, and their KDA state; with
    ``shard_rank``, the pools of that rank of a cache split ``tp`` ways, and
    with ``ring`` raw rows in a ring."""
    dsa = [layer for layer in layers if layer in DSA_LAYERS]
    kda = [layer for layer in layers if layer not in DSA_LAYERS]

    def split(count):
        with patch.object(
            kv_pool, "kpool_raw_ring_pages", return_value=RING_PAGES if ring else 0
        ):
            if shard_rank is None:
                made = unsplit(count)
            else:
                with (
                    patch.object(kv_pool, "mla_kv_shards", return_value=tp),
                    patch.object(kv_pool, "mla_kv_shard_role", return_value=shard_rank),
                    patch.object(kv_pool, "index_kv_shards", return_value=tp),
                ):
                    made = unsplit(count)
        made.attach_raw_ring(RING_SLOTS)
        return made

    def unsplit(count):
        return TpuSplitMLATokenToKVPool(
            size=PAGES * PAGE_TOKENS,
            page_size=PAGE_TOKENS,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=count,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            index_key_layers=[True] * count,
            index_kpool=4,
        )

    shape = KimiLinearStateShape.create(tp_world_size=tp, conv_kernel_size=4, **HEADS)
    pool = object.__new__(HybridLinearKVPool)
    pool.full_kv_pool = split(len(dsa))
    pool.mamba_pool = TpuMambaPool(
        size=4,
        spec_state_size=4,
        cache_params=KimiLinearCacheParams(shape=shape, layers=kda),
        mamba_layer_ids=kda,
        device="cpu",
    )
    pool.full_attention_layer_id_mapping = {layer: i for i, layer in enumerate(dsa)}
    pool.use_dsa = True
    groups = get_raiden_kv_caches(pool, split(1) if draft else None)
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        for tensor in (*groups.full, *groups.state):
            view = blocks_of(tensor)
            view.copy_(
                torch.randint(
                    0, 256, view.shape, dtype=torch.uint8, generator=generator
                )
            )
    return groups


class TpStore(Store):
    """Store whose transfers land in every destination unit: a span reaches
    the unit its dst_unit_ordinal names, or every unit without one."""

    def coordinate(self, **kw):
        req_id = kw["req_id"]
        registered = self.registered.get(req_id, {})
        if any(tp_test.unit_key(unit) not in registered for unit in kw["src_units"]):
            raise RuntimeError(f"Missing producer block registration for {req_id}")
        for ordinal, dst_unit in enumerate(kw["dst_units"]):
            dst_tensors, dst_engine = self.units[tp_test.unit_key(dst_unit)]
            cursor = 0
            for tag, count, skip in zip(
                kw["transfer_pool_tags"], kw["dst_block_counts"], kw["dst_skip_bytes"]
            ):
                assert skip == 0
                dst_ids = kw["dst_device_block_ids"][cursor : cursor + count]
                cursor += count
                dst = blocks_of(dst_tensors[tag])
                written = torch.zeros(count, dst.shape[1], dtype=torch.int32)
                for unit in kw["src_units"]:
                    src = blocks_of(self.units[tp_test.unit_key(unit)][0].get(tag, dst))
                    for entry in registered[tp_test.unit_key(unit)][1]:
                        if entry.tag != tag:
                            continue
                        assert entry.declared_bytes == sum(
                            span.size_bytes * span.count for span in entry.spans
                        )
                        for span in entry.spans:
                            target = getattr(span, "dst_unit_ordinal", None)
                            if target is not None and target != ordinal:
                                continue
                            for repeat in range(span.count):
                                at = (
                                    span.src_offset_bytes
                                    + repeat * span.src_stride_bytes
                                )
                                data = src[
                                    entry.block_ids[span.src_block_ordinal],
                                    at : at + span.size_bytes,
                                ]
                                to = (
                                    span.dst_offset_bytes
                                    + repeat * span.dst_stride_bytes
                                )
                                block = span.dst_block_index
                                if entry.dst_space_version == 1:
                                    block, to = divmod(to, dst.shape[1])
                                dst[dst_ids[block], to : to + span.size_bytes] = data
                                written[block, to : to + span.size_bytes] += 1
                                self.sources.setdefault(tag, {}).setdefault(
                                    block, set()
                                ).add(unit.data_replica_idx)
                assert (written == 1).all(), (req_id, tag, ordinal)
            assert cursor == len(kw["dst_device_block_ids"])
            dst_engine.received.append(req_id)
        for unit in kw["src_units"]:
            if any(entry.spans for entry in registered[tp_test.unit_key(unit)][1]):
                self.units[tp_test.unit_key(unit)][1].sent.append(req_id)
        return True


def make(w, mode, rank, name, *, split=False, ring=False):
    """A RaidenPoolTransfer with its state link: stage ``rank`` of the prefill,
    or attention-TP rank ``rank`` of the decode, with ``split`` on a split
    cache and ``ring`` with a raw ring."""
    producer = mode == PREFILL
    t = object.__new__(pl.RaidenPoolTransfer)
    if producer:
        mgr = harness.Manager(mode, rank, len(STAGES), w.bus, name)
        groups = pools(STAGES[rank], 1, draft=rank == len(STAGES) - 1, seed=100 + rank)
    else:
        mgr = harness.Manager(mode, 0, 1, w.bus, name)
        mgr.attn_tp_size, mgr.attn_tp_rank = TP, rank
        groups = pools(
            range(8), TP, draft=True, shard_rank=rank if split else None, ring=ring
        )
    mgr._prefill.send_state_rows = {}
    t.groups = groups
    t.manager, t.producer, t.lock = mgr, producer, threading.RLock()
    t.pools, state_pools = ps.split_manifest(
        pl.pool_manifest(t.groups, mgr.attn_tp_size)
    )
    t._check_pools(ps.family_of(state_pools))
    t.tensors = list(t.groups.full)
    t.rank = mgr.pp_rank * mgr.attn_tp_size + mgr.attn_tp_rank
    t.workers = mgr.pp_size * mgr.attn_tp_size
    t.client_type = w.client_type
    t._init_overlap_state()
    t.executor = harness.SyncExecutor()
    t.hbm_mem_type = 2
    t.sent, t.replies, t.release_requests, t.receives = {}, {}, {}, {}
    t.cleared, t.done_send, t.done_recv, t.failed, t.retired = (set() for _ in range(5))
    role = "p" if producer else "d"
    t.controller = f"{role}kv"

    def fields(replica, controller):
        return dict(
            job_name="sglang",
            job_replica_id=controller,
            data_name="prefill" if producer else "decode",
            data_replica_idx=replica,
        )

    t.unit_fields = fields(t.rank, t.controller)
    t.unit = SimpleNamespace(**t.unit_fields)
    t.client = w.client_type(t.controller)
    ranks = [t.rank] if producer else range(TP)
    t.dst_units = [SimpleNamespace(**fields(r, t.controller)) for r in ranks]
    t.max_blocks = PAGES
    t.admission = StagingAdmission(4 * PAGES, 4)
    t.engine = harness.Engine()
    link = ps.SideLink(
        family=ps.STATE,
        pools=state_pools,
        tensors=list(t.groups.state),
        controller=f"{role}state",
        unit_fields=fields(t.rank, f"{role}state"),
        dst_units=[SimpleNamespace(**fields(r, f"{role}state")) for r in ranks],
        max_blocks=1,
        slots=4,
        hold_undeclared=not producer,
    )
    link.client = w.client_type(link.controller)
    link.executor = None if producer else harness.SyncExecutor()
    link.engine = harness.Engine()
    t.side = link
    w.kv.add(t.unit, dict(zip((p["tag"] for p in t.pools), t.tensors)), t.engine)
    w.state.add(link.unit, dict(zip(link.tags, link.tensors)), link.engine)
    return t


@pytest.fixture(
    params=[(False, False), (True, False), (True, True)],
    ids=["whole-pages", "split-cache", "split-cache-raw-ring"],
)
def world(request):
    return make_world(*request.param)


def make_world(split, ring):
    w = SimpleNamespace(bus={"queue": []}, kv=TpStore(), state=TpStore())
    w.split, w.ring = split, ring
    w.client_type = lambda address: (
        w.state if address.endswith("state") else w.kv
    ).client()
    w.stages = [make(w, PREFILL, stage, f"p{stage}") for stage in range(len(STAGES))]
    w.decodes = [
        make(w, DECODE, rank, f"d{rank}", split=w.split, ring=w.ring)
        for rank in range(TP)
    ]
    actors = {f"p{s}": t for s, t in enumerate(w.stages)}
    actors |= {f"d{r}": t for r, t in enumerate(w.decodes)}

    def pump():
        while w.bus["queue"]:
            name, frames = w.bus["queue"].pop(0)
            actors[name].on_message(frames)

    w.pump = pump
    # As the bootstrap lists a pipeline prefill: one worker per stage.
    w.infos = [
        dict(pp_rank=s, rank_ip=t.manager.local_ip, rank_port=t.manager.rank_port)
        for s, t in enumerate(w.stages)
    ]
    return w


def shard_of(row, shard, rank):
    """Attention-TP rank ``rank``'s share of the full-width state ``row``."""
    axis = shard.axis
    if shard.groups is None:
        return row.chunk(TP, dim=axis)[rank]
    return torch.cat(
        [g.chunk(TP, dim=axis)[rank] for g in row.split(shard.groups, dim=axis)],
        dim=axis,
    )


def test_every_decode_rank_gets_each_stage_s_pages_and_its_state_shard(world):
    w = world
    src_pages, dst_pages = [5, 6, 7], [30, 21, 12]
    src_row, dst_row = 2, 3
    # Slot 1's ring page for the request's last token, its third page.
    ring_page = 1 * RING_PAGES + 2 % RING_PAGES
    for t in w.decodes:
        t.register_decode(
            ROOM,
            list(dst_pages),
            [dst_row],
            None,
            w.infos,
            ring_page=ring_page if w.ring else None,
        )
    w.pump()
    for t in w.stages:
        with t.manager._state_lock:
            t.manager._prefill.send_blocks[ROOM] = list(src_pages)
            t.manager._prefill.send_state_rows[ROOM] = [src_row]
        assert t.finalize_send(ROOM, None)
    w.pump()
    for t in w.decodes:
        assert t.maybe_start_pull(ROOM)
    for t in (*w.stages, *w.decodes):
        harness.poll(t)

    owner = {}
    for stage, t in enumerate(w.stages):
        for pool in t.pools:
            owner[pool["tag"]] = (stage, pool["tag"])
    for t in w.decodes:
        # Every page tag of the decode came from exactly one stage: the whole
        # page, or the rank's share of a split one, the slab of the page's
        # first axis the kernels read (mla_sparse/kv_shard.py).
        for pool, landed in zip(t.pools, t.tensors, strict=True):
            stage, tag = owner[pool["tag"]]
            src = w.kv.units[("pkv", stage)][0][tag]
            if pl.is_last_page_pool(tag):
                # Raw rows: only the last page, which holds the open pool, at
                # the request's last page or its ring page.
                at = ring_page if pool.get("ring_pages") else dst_pages[-1]
                expected = torch.zeros_like(landed)
                expected[at] = src[src_pages[-1]]
                assert torch.equal(blocks_of(landed), blocks_of(expected)), (
                    t.rank,
                    tag,
                )
                continue
            share = int(landed.shape[1])
            role = t.manager.attn_tp_rank % pool.get("shards", 1)
            expected = torch.zeros_like(landed)
            for s, d in zip(src_pages, dst_pages):
                expected[d] = src[s, role * share : (role + 1) * share]
            assert torch.equal(blocks_of(landed), blocks_of(expected)), (t.rank, tag)
            split = pl.pool_kind(tag).removeprefix("draft.") in ("mla.nope", "dsa.idx")
            assert (share < src.shape[1]) == (w.split and split), (t.rank, tag)
        rings = [p["tag"] for p in t.pools if p.get("ring_pages")]
        assert len(rings) == (len(DSA_LAYERS) + 1 if w.ring else 0), rings
        # Each state row is this rank's heads of the owning stage's row.
        for tag, landed, shard in zip(
            t.side.tags, t.groups.state, t.groups.state_shards, strict=True
        ):
            stage = next(s for s, p in enumerate(w.stages) if tag in p.side.tags)
            src = w.stages[stage].groups.state[w.stages[stage].side.tags.index(tag)]
            expected = torch.zeros_like(landed)
            expected[dst_row] = shard_of(src[src_row], shard, t.manager.attn_tp_rank)
            assert torch.equal(blocks_of(landed), blocks_of(expected)), (t.rank, tag)
    # Together the stages carry every pool of the decode, once.
    tags = [p["tag"] for t in w.stages for p in (*t.pools, *t.side.pools)]
    assert sorted(tags) == sorted(
        p["tag"] for p in (*w.decodes[0].pools, *w.decodes[0].side.pools)
    )
    for t in w.stages:
        assert str(ROOM) in t.manager._completed_send and not t.active(ROOM)
    for t in w.decodes:
        assert str(ROOM) in t.manager._completed_recv
    assert not any(t.manager._failed for t in (*w.stages, *w.decodes))


def test_a_long_request_moves_as_capped_transfers():
    """SGLANG_RAIDEN_PIPELINE_MAX_TRANSFER_PAGES 2: five pages move as two
    parts of two pages and a closing transfer of the last one, which also
    carries the raw rows and the state."""
    w = make_world(True, False)
    src_pages, dst_pages = [5, 6, 7, 8, 9], [30, 21, 12, 3, 17]
    for t in w.stages:
        t.max_transfer_pages = 2
    for t in w.decodes:
        t.register_decode(ROOM, list(dst_pages), [3], None, w.infos)
    w.pump()
    for t in w.stages:
        with t.manager._state_lock:
            t.manager._prefill.send_blocks[ROOM] = list(src_pages)
            t.manager._prefill.send_state_rows[ROOM] = [2]
        assert t.finalize_send(ROOM, None)
        assert [part["start"] for part in t.early[ROOM]["parts"]] == [0, 2]
        assert t.sent[ROOM]["full"] == src_pages[4:]
    w.pump()
    # A decode starts what was announced, then what its landing unblocks.
    for _ in range(3):
        for t in w.decodes:
            t.maybe_start_pull(ROOM)
        for t in (*w.stages, *w.decodes):
            harness.poll(t)
        w.pump()
    for stage, t in enumerate(w.stages):
        for tag in (pool["tag"] for pool in t.pools):
            src = w.kv.units[("pkv", stage)][0][tag]
            for d in w.decodes:
                pool = next(p for p in d.pools if p["tag"] == tag)
                landed = d.tensors[d.pools.index(pool)]
                share = int(landed.shape[1])
                role = d.manager.attn_tp_rank % pool.get("shards", 1)
                moved = (
                    [(src_pages[-1], dst_pages[-1])]
                    if pl.is_last_page_pool(tag)
                    else zip(src_pages, dst_pages)
                )
                expected = torch.zeros_like(landed)
                for s, dst in moved:
                    expected[dst] = src[s, role * share : (role + 1) * share]
                assert torch.equal(blocks_of(landed), blocks_of(expected)), (
                    tag,
                    d.rank,
                )
    for t in w.decodes:
        assert str(ROOM) in t.manager._completed_recv
    assert not any(t.manager._failed for t in (*w.stages, *w.decodes))


@pytest.mark.parametrize("decode_asks_first", [True, False])
def test_a_decode_prefix_stays_and_the_rest_moves(decode_asks_first):
    """A decode that holds the request's first two pages (its radix cache)
    registers the other three, and each stage sends those: here as a part of
    two pages and a closing transfer of the last one, with the raw rows and
    the state. A stage may close before the decode's request reaches it."""
    w = make_world(True, False)
    src_pages, dst_pages = [5, 6, 7, 8, 9], [12, 3, 17]
    prefix_len = 2 * w.stages[0].manager.kv_args.page_size
    for t in w.stages:
        t.max_transfer_pages = 2

    def ask():
        for t in w.decodes:
            t.register_decode(
                ROOM, list(dst_pages), [3], None, w.infos, prefix_len=prefix_len
            )
        w.pump()

    def close():
        for t in w.stages:
            with t.manager._state_lock:
                t.manager._prefill.send_blocks[ROOM] = list(src_pages)
                t.manager._prefill.send_state_rows[ROOM] = [2]
            assert t.finalize_send(ROOM, None)
        w.pump()

    for step in (ask, close) if decode_asks_first else (close, ask):
        step()
    for t in w.stages:
        assert [(part["start"], part["full"]) for part in t.early[ROOM]["parts"]] == [
            (0, src_pages[2:4])
        ]
        assert (t.sent[ROOM]["start"], t.sent[ROOM]["full"]) == (2, src_pages[4:])
    for _ in range(3):
        for t in w.decodes:
            t.maybe_start_pull(ROOM)
        for t in (*w.stages, *w.decodes):
            harness.poll(t)
        w.pump()
    for stage, t in enumerate(w.stages):
        for tag in (pool["tag"] for pool in t.pools):
            src = w.kv.units[("pkv", stage)][0][tag]
            for d in w.decodes:
                pool = next(p for p in d.pools if p["tag"] == tag)
                landed = d.tensors[d.pools.index(pool)]
                share = int(landed.shape[1])
                role = d.manager.attn_tp_rank % pool.get("shards", 1)
                moved = (
                    [(src_pages[-1], dst_pages[-1])]
                    if pl.is_last_page_pool(tag)
                    else zip(src_pages[2:], dst_pages)
                )
                expected = torch.zeros_like(landed)
                for s, dst in moved:
                    expected[dst] = src[s, role * share : (role + 1) * share]
                assert torch.equal(blocks_of(landed), blocks_of(expected)), (
                    tag,
                    d.rank,
                )
    for t in w.decodes:
        assert str(ROOM) in t.manager._completed_recv
    assert not any(t.manager._failed for t in (*w.stages, *w.decodes))


@pytest.mark.parametrize("case", ["early parts", "half a page", "whole request"])
def test_a_decode_prefix_the_stages_cannot_leave_out_fails_the_room(case):
    w = make_world(True, False)
    page = w.stages[0].manager.kv_args.page_size
    prefix_len = {
        "early parts": page,
        "half a page": page // 2,
        "whole request": 2 * page,
    }[case]
    for t in w.stages:
        t.early_parts = case == "early parts"
    for t in w.decodes:
        t.register_decode(ROOM, [12], [3], None, w.infos, prefix_len=prefix_len)
    if case != "whole request":
        # Refused as the decode's request arrives; the manager fails the room.
        with pytest.raises(ValueError, match="prefix"):
            w.pump()
        return
    w.pump()
    for t in w.stages:
        with t.manager._state_lock:
            t.manager._prefill.send_blocks[ROOM] = [5, 6]
            t.manager._prefill.send_state_rows[ROOM] = [2]
        assert not t.finalize_send(ROOM, None)
        assert t.manager._failed


def test_stage_and_decode_fingerprints_agree():
    layout = tp_test.natural_layout
    prints = set()
    for groups, tp in (
        (pools(STAGES[0], 1, draft=False), 1),
        (pools(STAGES[1], 1, draft=True), 1),
        (pools(range(8), TP, draft=True), TP),
        (pools(range(8), TP, draft=True, shard_rank=1), TP),
        (pools(range(8), TP, draft=True, shard_rank=1, ring=True), TP),
    ):
        kv, state = ps.split_manifest(pl.pool_manifest(groups, tp))
        prints.add(
            tuple(
                pl.layout_fingerprint(pools_, tensors, layout, ships_pd_hidden=False)
                for pools_, tensors in ((kv, groups.full), (state, groups.state))
            )
        )
    assert len(prints) == 1


def test_a_split_share_must_be_one_byte_range():
    groups = pools(range(8), TP, draft=True, shard_rank=1)
    kv, _ = ps.split_manifest(pl.pool_manifest(groups, TP))

    def tiled_tokens(tensor):  # a tile that reaches the page's token axis
        return tuple(reversed(range(tensor.dim()))), ((8, 4, 128),), None

    with pytest.raises(ValueError, match="one byte range"):
        pl.layout_fingerprint(kv, groups.full, tiled_tokens, ships_pd_hidden=False)


def test_a_decode_with_a_raw_ring_must_name_its_ring_page():
    w = make_world(True, True)
    with pytest.raises(ValueError, match="names no raw ring page"):
        w.decodes[0].register_decode(ROOM, [30], [3], None, w.infos)


def test_the_decode_names_the_ring_page_of_the_request_s_last_token():
    from sglang_tpu.disaggregation.raiden import raw_ring_hooks

    split = TpuSplitMLATokenToKVPool.__new__(TpuSplitMLATokenToKVPool)
    split.page_size, split.raw_ring_pages_cnt = PAGE_TOKENS, RING_PAGES
    queue = SimpleNamespace(token_to_kv_pool=split)
    decode_req = SimpleNamespace(
        kv_receiver=SimpleNamespace(raw_ring_page=None),
        req=SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3)),
    )
    calls = []
    raw_ring_hooks._send_kv_metadata(
        lambda *a, **k: calls.append((a, k)),
        queue,
        decode_req,
        [1, 2, 3],
        PAGE_TOKENS,
        receive_rows=(0, 2 * PAGE_TOKENS + 5),
    )
    # Token 2 * PAGE_TOKENS + 4 is on the request's third page: ring page 0 of slot 3.
    assert decode_req.kv_receiver.raw_ring_page == 3 * RING_PAGES + 0
    assert len(calls) == 1 and calls[0][0][1] is decode_req


def test_only_a_pipeline_transfer_decode_keeps_a_raw_ring():
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.tpu_platform import _validate_kpool_raw_ring

    def cfg(mode):
        return SimpleNamespace(
            disaggregation_mode=mode,
            disaggregation_transfer_backend="raiden",
            enable_prefill_cp=False,
            chunked_prefill_size=4096,
        )

    with tpu_envs.SGLANG_TPU_DSA_KPOOL_RAW_RING.override(True):
        with tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.override(True):
            _validate_kpool_raw_ring(cfg("decode"))
            with pytest.raises(ValueError, match="--disaggregation-mode 'prefill'"):
                _validate_kpool_raw_ring(cfg("prefill"))
        with pytest.raises(ValueError, match="--disaggregation-mode 'decode'"):
            _validate_kpool_raw_ring(cfg("decode"))


def test_a_split_prefill_is_refused():
    t = object.__new__(pl.RaidenPoolTransfer)
    t.manager = SimpleNamespace(attn_tp_size=TP)
    t.producer = True
    groups = pools(range(8), TP, draft=True, shard_rank=0)
    t.pools, _ = ps.split_manifest(pl.pool_manifest(groups, TP))
    t.pools = [p for p in t.pools if pl.is_replicated_pool(p["tag"])]
    with pytest.raises(NotImplementedError, match="split across ranks"):
        t._check_pools(None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
