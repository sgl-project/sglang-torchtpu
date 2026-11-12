"""A tensor-parallel prefill on the Raiden pool transfer (P: TP -> D: DP).

Every attention-TP rank of a GLM-5.3-Flash prefill holds each page of the split
sparse-MLA pool whole, and its heads' share of each linear-attention state row.
The ranks stripe the pages between them, and the attention-TP 1 decode gathers
their state shards into its full-width rows. The pools here are the real ones,
small. Producer ranks and the decode exchange their real control messages
through the in-memory bus of test_raiden_pipeline_overlap.py, and a fake
reshard store per engine executes each coordinated transfer byte for byte from
the registered spans. No TPU or tpu-sync.
"""

import importlib.util
import pathlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import pool_side_engine as ps
from sglang_tpu.disaggregation.raiden import pool_transfer as pl
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission
from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool
from sglang_tpu.mamba_pool import TpuMambaPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "raiden_pipeline_overlap_harness",
    pathlib.Path(__file__).with_name("test_raiden_pipeline_overlap.py"),
)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

ROOM = 123456789
TP = 8
PAGE_TOKENS, PAGES = 16, 32
DSA_LAYERS, KDA_LAYERS = [3, 7], [0, 1, 2, 4, 5, 6]
# A conv row is [value | key | key] over 16 value heads of 4 and 8 key heads of
# 2: sub-blocks of 64, 16 and 16 channels, each head-sharded on its own.
HEADS = dict(num_heads=16, head_dim=4, num_k_heads=8, head_k_dim=2)
PREFILL, DECODE = DisaggregationMode.PREFILL, DisaggregationMode.DECODE
KV_TAGS = [
    *(
        f"{kind}.l{layer}"
        for kind in ("mla.nope", "dsa.idx", "dsa.raw")
        for layer in DSA_LAYERS
    ),
    *(f"draft.{kind}.l0" for kind in ("mla.nope", "dsa.idx", "dsa.raw")),
]
STATE_TAGS = [
    f"gdn.{kind}.l{layer}" for kind in ("conv", "ssm") for layer in KDA_LAYERS
]


def blocks_of(tensor):
    """The tensor's blocks as rows of bytes, sharing its memory."""
    return tensor.view(torch.uint8).reshape(tensor.shape[0], -1)


def model_pools(tp, seed=None):
    """What a worker registers at attention TP ``tp``: the target's and the MTP
    draft's split pools and the KDA state, random with ``seed``."""

    def split(layers):
        return TpuSplitMLATokenToKVPool(
            size=PAGES * PAGE_TOKENS,
            page_size=PAGE_TOKENS,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=layers,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            index_key_layers=[True] * layers,
            index_kpool=4,
        )

    shape = KimiLinearStateShape.create(tp_world_size=tp, conv_kernel_size=4, **HEADS)
    # Only the attributes the P/D registration reads; the real constructor
    # needs a model config.
    pool = object.__new__(HybridLinearKVPool)
    pool.full_kv_pool = split(len(DSA_LAYERS))
    pool.mamba_pool = TpuMambaPool(
        size=4,
        spec_state_size=4,
        cache_params=KimiLinearCacheParams(shape=shape, layers=KDA_LAYERS),
        mamba_layer_ids=KDA_LAYERS,
        device="cpu",
    )
    pool.full_attention_layer_id_mapping = {
        layer: dense for dense, layer in enumerate(DSA_LAYERS)
    }
    pool.use_dsa = True
    groups = get_raiden_kv_caches(pool, split(1))
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


def full_width(shards, shard):
    """The attention-TP 1 state tensor whose heads the ranks' ``shards`` hold."""
    axis = 1 + shard.axis
    if shard.groups is None:
        return torch.cat(shards, dim=axis)
    local = [extent // len(shards) for extent in shard.groups]
    pieces = [tensor.split(local, dim=axis) for tensor in shards]
    return torch.cat(
        [torch.cat([rank[g] for rank in pieces], dim=axis) for g in range(len(local))],
        dim=axis,
    )


def unit_key(unit):
    return (unit.job_replica_id, unit.data_replica_idx)


class Store:
    """The reshard stores of one engine on both roles: the registrations, and
    every coordinated transfer executed byte for byte as its spans say. A
    transfer must write each byte of its destination blocks exactly once."""

    def __init__(self):
        self.registered = {}  # req_id -> unit key -> (block_ids, span entries)
        self.units = {}  # unit key -> ({tag: tensor}, engine)
        self.completed = []  # (req_id, rank)
        self.sources = {}  # tag -> destination block ordinal -> source ranks

    def add(self, unit, tensors, engine):
        self.units[unit_key(unit)] = (tensors, engine)

    def client(self):
        store = self

        class Client:
            def register_request_blocks(
                self, req_id, uuid, unit, block_ids, pool_spans
            ):
                store.registered.setdefault(req_id, {})[unit_key(unit)] = (
                    list(block_ids),
                    list(pool_spans),
                )

            def complete_request_blocks(self, req_id, uuid, unit):
                store.completed.append((req_id, unit.data_replica_idx))

            def coordinate_transfer(self, **kwargs):
                return store.coordinate(**kwargs)

        return Client()

    def coordinate(self, **kw):
        req_id = kw["req_id"]
        registered = self.registered.get(req_id, {})
        if any(unit_key(unit) not in registered for unit in kw["src_units"]):
            raise RuntimeError(f"Missing producer block registration for {req_id}")
        (dst_unit,) = kw["dst_units"]
        dst_tensors, dst_engine = self.units[unit_key(dst_unit)]
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
                src = blocks_of(self.units[unit_key(unit)][0][tag])
                for entry in registered[unit_key(unit)][1]:
                    if entry.tag != tag:
                        continue
                    assert entry.declared_bytes == sum(
                        span.size_bytes * span.count for span in entry.spans
                    )
                    for span in entry.spans:
                        for repeat in range(span.count):
                            at = span.src_offset_bytes + repeat * span.src_stride_bytes
                            data = src[
                                entry.block_ids[span.src_block_ordinal],
                                at : at + span.size_bytes,
                            ]
                            to = span.dst_offset_bytes + repeat * span.dst_stride_bytes
                            block = span.dst_block_index
                            if entry.dst_space_version == 1:
                                # Request-global bytes, split at destination pages.
                                block, to = divmod(to, dst.shape[1])
                            dst[dst_ids[block], to : to + span.size_bytes] = data
                            written[block, to : to + span.size_bytes] += 1
                            self.sources.setdefault(tag, {}).setdefault(
                                block, set()
                            ).add(unit.data_replica_idx)
            assert (written == 1).all(), (req_id, tag)
        assert cursor == len(kw["dst_device_block_ids"])
        # Only an engine with bytes to send runs a sender, and reports it.
        for unit in kw["src_units"]:
            if any(entry.spans for entry in registered[unit_key(unit)][1]):
                self.units[unit_key(unit)][1].sent.append(req_id)
        dst_engine.received.append(req_id)
        return True


def make(w, mode, rank, name):
    """A PoolTransfer with its state link, built like __init__ would."""
    producer = mode == PREFILL
    t = object.__new__(pl.PoolTransfer)
    mgr = harness.Manager(mode, 0, 1, w.bus, name)
    mgr.attn_tp_size, mgr.attn_tp_rank = (TP, rank) if producer else (1, 0)
    mgr._prefill.send_state_rows = {}
    t.groups = model_pools(mgr.attn_tp_size, seed=100 + rank if producer else None)
    t.manager, t.producer, t.lock = mgr, producer, threading.RLock()
    t.pools, state_pools = ps.split_manifest(
        pl.pool_manifest(t.groups, mgr.attn_tp_size)
    )
    t._check_pools(ps.family_of(state_pools))
    t.tensors = list(t.groups.full)
    t.rank, t.workers = mgr.attn_tp_rank, mgr.attn_tp_size
    if producer:
        t.stripe = (t.rank, t.workers)
    t.client_type = w.client_type
    t._init_overlap_state()
    t.executor = harness.SyncExecutor()
    t.hbm_mem_type = 2
    t.sent, t.replies, t.release_requests, t.receives = {}, {}, {}, {}
    t.cleared, t.done_send, t.done_recv, t.failed, t.retired = (set() for _ in range(5))
    role = "p" if producer else "d"
    t.controller = f"{role}kv"
    t.unit_fields = dict(
        job_name="sglang",
        job_replica_id=t.controller,
        data_name="prefill" if producer else "decode",
        data_replica_idx=t.rank,
    )
    t.unit = SimpleNamespace(**t.unit_fields)
    t.client = w.client_type(t.controller)
    t.dst_units = [t.unit]
    t.max_blocks = PAGES
    t.admission = StagingAdmission(4 * PAGES, 4)
    t.engine = harness.Engine()
    state_fields = t.unit_fields | {"job_replica_id": f"{role}state"}
    link = ps.SideLink(
        family=ps.STATE,
        pools=state_pools,
        tensors=list(t.groups.state),
        controller=f"{role}state",
        unit_fields=state_fields,
        dst_units=[SimpleNamespace(**state_fields)],
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


@pytest.fixture
def world():
    w = SimpleNamespace(bus={"queue": []}, kv=Store(), state=Store())
    w.client_type = lambda address: (
        w.state if address.endswith("state") else w.kv
    ).client()
    w.ranks = [make(w, PREFILL, rank, f"p{rank}") for rank in range(TP)]
    w.decode = make(w, DECODE, 0, "d")
    actors = {f"p{rank}": t for rank, t in enumerate(w.ranks)} | {"d": w.decode}

    def pump():
        while w.bus["queue"]:
            name, frames = w.bus["queue"].pop(0)
            actors[name].on_message(frames)

    w.pump = pump
    # As the bootstrap lists a tensor-parallel prefill: every rank, pp_rank 0.
    w.infos = [
        dict(pp_rank=0, rank_ip=t.manager.local_ip, rank_port=t.manager.rank_port)
        for t in w.ranks
    ]
    return w


def transfer(w, src_pages, dst_pages, src_row, dst_row):
    """One request end to end: the decode asks, every rank finishes its
    prefill, the decode pulls, and every worker polls its engines."""
    w.decode.register_decode(ROOM, list(dst_pages), [dst_row], None, w.infos)
    w.pump()
    for t in w.ranks:
        with t.manager._state_lock:
            t.manager._prefill.send_blocks[ROOM] = list(src_pages)
            t.manager._prefill.send_state_rows[ROOM] = [src_row]
        assert t.finalize_send(ROOM, None)
    w.pump()
    assert w.decode.maybe_start_pull(ROOM)
    for t in (*w.ranks, w.decode):
        harness.poll(t)


@pytest.mark.parametrize("num_pages", [19, 3])
def test_ranks_stripe_the_pages_and_the_decode_gathers_their_state(world, num_pages):
    """19 pages: rank r moves pages r, r + 8 and r + 16. 3 pages: ranks 3 to 7
    have no page, register nothing in the KV store and still move their state."""
    w = world
    src_pages = list(range(5, 5 + num_pages))
    dst_pages = list(range(30, 30 - num_pages, -1))
    src_row, dst_row = 2, 3
    transfer(w, src_pages, dst_pages, src_row, dst_row)

    # KV: destination page i is page i of rank i % TP, and nothing else moved.
    for tag, landed in zip(KV_TAGS, w.decode.tensors, strict=True):
        expected = torch.zeros_like(blocks_of(landed))
        for i, (src, dst) in enumerate(zip(src_pages, dst_pages)):
            expected[dst] = blocks_of(w.kv.units[("pkv", i % TP)][0][tag])[src]
            assert w.kv.sources[tag][i] == {i % TP}
        assert torch.equal(blocks_of(landed), expected), tag
    # State: the decode's row holds every rank's heads, sub-block by sub-block.
    state = w.decode.groups
    for index, (tag, landed) in enumerate(zip(STATE_TAGS, state.state, strict=True)):
        shards = [t.groups.state[index] for t in w.ranks]
        expected = torch.zeros_like(landed)
        expected[dst_row] = full_width(shards, state.state_shards[index])[src_row]
        # By bytes: random fp32 holds NaNs, which never compare equal.
        assert torch.equal(blocks_of(landed), blocks_of(expected)), tag
        assert w.state.sources[tag][0] == set(range(TP))

    owners = set(range(min(TP, num_pages)))
    kv_registered = w.kv.registered[str(ROOM)]
    for rank in range(TP):
        block_ids, entries = kv_registered[("pkv", rank)]
        if rank in owners:
            assert block_ids == src_pages and len(entries) == len(KV_TAGS)
        else:
            assert (block_ids, entries) == ([], [])
    # A rank completes what it registered, but never an empty registration.
    assert {rank for _, rank in w.kv.completed} == owners
    assert sorted(w.state.completed) == [(f"state:{ROOM}", r) for r in range(TP)]
    for t in w.ranks:
        assert str(ROOM) in t.manager._completed_send and not t.active(ROOM)
    assert str(ROOM) in w.decode.manager._completed_recv
    assert not any(t.manager._failed for t in (*w.ranks, w.decode))


def test_decode_refuses_a_source_named_twice(world):
    with pytest.raises(ValueError, match="distinct workers"):
        world.decode.register_decode(ROOM, [1], [1], None, [world.infos[0]] * 2)


def natural_layout(tensor):
    """A device layout that keeps the bytes of a row in row-major order."""
    return tuple(reversed(range(tensor.dim()))), ((1, 4),), None


def test_a_rank_and_the_decode_agree_on_pools_and_fingerprints():
    rank, decode = model_pools(TP), model_pools(1)
    fingerprints = []
    for groups, tp in ((rank, TP), (decode, 1)):
        kv, state = ps.split_manifest(pl.pool_manifest(groups, tp))
        assert [pool["tag"] for pool in kv] == KV_TAGS
        assert [pool["tag"] for pool in state] == STATE_TAGS
        assert ps.family_of(state) is ps.STATE
        fingerprints.append(
            [
                pl.layout_fingerprint(
                    pools, tensors, natural_layout, ships_pd_hidden=False
                )
                for pools, tensors in ((kv, groups.full), (state, groups.state))
            ]
        )
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0][0] != fingerprints[0][1]

    # Head shards are placed by byte offset, which a tiled row would scramble.
    _, state = ps.split_manifest(pl.pool_manifest(rank, TP))
    with pytest.raises(ValueError, match="row-major"):
        pl.layout_fingerprint(
            state,
            rank.state,
            lambda tensor: (tuple(reversed(range(tensor.dim()))), ((2, 2),), None),
            ships_pd_hidden=False,
        )


@pytest.mark.parametrize(
    "producer, per_head, error",
    [
        (True, True, "per-head KV"),
        (False, False, "attention-TP 1 decode"),
    ],
)
def test_pools_a_tensor_parallel_worker_cannot_move_are_refused(
    producer, per_head, error
):
    """A tensor-parallel prefill rank holds only its heads of per-head KV, and
    a tensor-parallel decode cannot take whole pages or state."""
    t = object.__new__(pl.PoolTransfer)
    t.producer = producer
    t.manager = SimpleNamespace(attn_tp_size=TP)
    t.pools = [dict(tag="fa.l0" if per_head else "mla.nope.l0")]
    with pytest.raises(NotImplementedError, match=error):
        t._check_pools(None if per_head else ps.STATE)


def test_a_prefill_rank_builds_a_state_engine_beside_the_page_engine():
    """Rank 5 of 8: both engines take their controllers from rank 0, the state
    engine on its reserved port with one row per transfer."""
    groups = model_pools(TP)
    engines, units = [], []
    manager = SimpleNamespace(
        disaggregation_mode=PREFILL,
        attn_tp_size=TP,
        attn_tp_rank=5,
        attn_cp_size=1,
        pp_size=1,
        pp_rank=0,
        local_ip="10.0.0.1",
        _control_port=31030,
        _local_rank=5,
        _timeout_s=120,
        kv_args=SimpleNamespace(page_size=PAGE_TOKENS),
        _max_request_blocks=lambda: PAGES,
        _state_max_blocks=1,
        _num_slots=lambda tensors, max_blocks, **budget: 6,
    )

    class Engine:
        def __init__(self, **kwargs):
            engines.append(kwargs)
            self.transfer_address = self.listener_address = "10.0.0.1:1"
            self.register_pools = Mock()

        @staticmethod
        def register_pools(pools, *, staging_leases=0, staging_blocks_per_pool=None):
            pass  # signature only: the instance attribute records the calls

    native = {
        "torch_tpu._internal.compile": SimpleNamespace(
            tpu_torch_compile=SimpleNamespace(
                get_device_layout_if_materialized=natural_layout
            )
        ),
        "tpu_sync.api.torch.kv_cache_manager": SimpleNamespace(KVCacheManager=Engine),
        "tpu_sync.api.torch.reshard_client": SimpleNamespace(
            ReshardClient=lambda address: SimpleNamespace(
                register_work_unit=lambda **kw: units.append((address, kw))
            )
        ),
        "tpu_sync.api.torch.reshard_store": SimpleNamespace(
            RaidenId=Mock(), ReshardStore=Mock()
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
    rank0 = ("kv:1", "kv:2", "state:1", "state:2")
    tp_group = SimpleNamespace(broadcast_object=lambda obj, src: rank0)
    with (
        patch.dict(sys.modules, native),
        patch.object(
            pl, "get_parallel", lambda: SimpleNamespace(attn_tp_group=tp_group)
        ),
    ):
        t = pl.PoolTransfer(manager, groups)
    t.executor.shutdown(wait=False)

    assert t.stripe == (5, TP) and t.moves_state
    pages, state = engines
    assert (pages["local_control_port"], state["local_control_port"]) == (31030, 31034)
    assert pages["raiden_controller_address"] == "kv:2"
    assert state["raiden_controller_address"] == "state:2"
    assert pages["worker_id"] == state["worker_id"] == "worker_5"
    assert (pages["max_blocks"], state["max_blocks"]) == (PAGES, 1)
    t.side.engine.register_pools.assert_called_once_with(
        t.side.pools, staging_leases=6, staging_blocks_per_pool=[1] * len(STATE_TAGS)
    )
    (kv_address, kv_unit), (state_address, state_unit) = units
    assert (kv_address, state_address) == ("kv:1", "state:1")
    assert kv_unit["transfer_rank"] == state_unit["transfer_rank"] == 5
    assert kv_unit["transfer_parallelism"] == state_unit["transfer_parallelism"] == TP


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
