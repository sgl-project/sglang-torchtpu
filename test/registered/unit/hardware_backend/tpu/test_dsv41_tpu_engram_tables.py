# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 TPU Engram tables held as uint8 bytes, the all-reduce DP lookup, and the
rank tensors that keep the traced lookups the same on every rank."""

import re
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41 import hooks
from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import ROW_START_ATTR

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ROWS, DIM, BLOCK = 96, 256, 32
RANK_TENSORS = ("tpu_dp_rank_mask", "tpu_dp_rank_index", "tpu_row_start")


class _Embedding(torch.nn.Module):
    """The EngramEmbedding attributes the hooks read; an nn.Module, as the real one,
    so tensor attributes go through nn.Module.__setattr__."""

    def __init__(self, row_start, rows, tp_size):
        super().__init__()
        self.host_table = None
        self.rows = rows
        self.row_start = row_start
        self._shared = False
        self.dim = DIM
        self.tp_size = tp_size
        self.weight = torch.nn.Parameter(
            torch.empty(rows, DIM, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.scale = torch.nn.Parameter(
            torch.empty(rows, DIM // BLOCK, dtype=torch.float8_e8m0fnu),
            requires_grad=False,
        )
        self.weight.weight_loader = "weight-loader"
        self.scale.weight_loader = "scale-loader"


def _bytes(seed):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (ROWS, DIM), dtype=torch.uint8, generator=g)
    weight[(weight & 0x7F) == 0x7F] = 0x38  # no e4m3fn NaN encodings
    scale = torch.randint(
        110, 140, (ROWS, DIM // BLOCK), dtype=torch.uint8, generator=g
    )
    return weight, scale


@pytest.fixture
def parallel(monkeypatch):
    """get_parallel() for the hooks; tests set attn_dp_size / attn_dp_rank."""
    from sglang.srt import runtime_context

    state = SimpleNamespace(attn_dp_size=1, attn_dp_rank=0)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: state)
    return state


def _fp8_embedding(row_start=0, rows=ROWS, tp_size=1):
    return _Embedding(row_start, rows, tp_size)


def _byte_embedding(weight, scale, rank, world):
    """Rank `rank` of `world`, owning its contiguous share of the full table, after
    tpu_dsv41_engram_bytes_init. get_parallel() must report `rank`."""
    rows = ROWS // world
    e = _fp8_embedding(row_start=rank * rows, rows=rows, tp_size=world)
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    e.weight.data.copy_(weight[rank * rows : (rank + 1) * rows])
    e.scale.data.copy_(scale[rank * rows : (rank + 1) * rows])
    # The AROUND hook on _owned_rows, as the hook registry installs it.
    e._owned_rows = lambda idx, e=e: hooks.tpu_dsv41_engram_owned_rows(None, e, idx)
    return e


def _core_load_rows(embedding, param, loaded_weight):
    rows = slice(embedding.row_start, embedding.row_start + embedding.rows)
    param.data.copy_(loaded_weight[rows])


def _reference(weight, scale, ids):
    """Full-table FP8 lookup with the hook's dequant, as bf16."""
    full = SimpleNamespace(
        host_table=None,
        rows=ROWS,
        row_start=0,
        tpu_row_start=torch.tensor(0, dtype=torch.int32),
        weight=weight.view(torch.float8_e4m3fn),
        scale=scale.view(torch.float8_e8m0fnu),
    )
    return hooks.tpu_dsv41_engram_owned_rows(None, full, ids)


def test_tables_become_uint8_and_keep_their_loaders(parallel):
    e = _fp8_embedding()
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    assert e.weight.dtype == e.scale.dtype == torch.uint8
    assert e.weight.shape == (ROWS, DIM) and e.scale.shape == (ROWS, DIM // BLOCK)
    assert e.weight.weight_loader == "weight-loader"
    assert e.scale.weight_loader == "scale-loader"


def test_rank_tensors_are_plain_attributes_with_this_ranks_position(parallel):
    parallel.attn_dp_size, parallel.attn_dp_rank = 4, 2
    e = _fp8_embedding(row_start=48, rows=24, tp_size=4)
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    assert e.tpu_dp_rank_mask.tolist() == [False, False, True, False]
    assert e.tpu_dp_rank_index.dtype == torch.int64
    assert e.tpu_dp_rank_index.tolist() == [2]
    assert e.tpu_row_start.dtype == torch.int32 and e.tpu_row_start.dim() == 0
    assert e.tpu_row_start.item() == 48
    # Not checkpoint weights: load_weights walks named_parameters(), and the
    # tensors stay out of the state dict and the buffers.
    assert dict(e.named_parameters()).keys() == {"weight", "scale"}
    assert not dict(e.named_buffers())
    assert e.state_dict().keys() == {"weight", "scale"}


def test_rank_tensors_follow_the_default_device(parallel):
    """The loader builds the model under `with target_device`; the rank tensors land
    there, as the tables do."""
    with torch.device("meta"):
        e = _fp8_embedding()
        hooks.tpu_dsv41_engram_bytes_init(None, e)
    assert e.weight.is_meta
    for name in RANK_TENSORS:
        assert getattr(e, name).is_meta, name


def test_host_tables_keep_fp8_but_get_rank_tensors(parallel):
    e = _fp8_embedding()
    e.host_table = object()
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    assert e.weight.dtype == torch.float8_e4m3fn
    # The DP lookup still runs for host tables.
    for name in RANK_TENSORS:
        assert isinstance(getattr(e, name), torch.Tensor), name


@pytest.mark.parametrize("tagged", [False, True])
def test_load_copies_bytes(parallel, tagged):
    weight, scale = _bytes(0)
    e = _fp8_embedding()
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    for param, raw, dtype in (
        (e.weight, weight, torch.float8_e4m3fn),
        (e.scale, scale, torch.float8_e8m0fnu),
    ):
        loaded = raw.view(dtype).clone()
        if tagged:  # an EP-local reader tensor holding exactly this rank's rows
            setattr(loaded, ROW_START_ATTR, 0)
        hooks.load_tpu_dsv41_engram_rows(_core_load_rows, e, param, loaded)
    assert torch.equal(e.weight.data, weight)
    assert torch.equal(e.scale.data, scale)


def test_uint8_lookup_matches_the_fp8_lookup_bit_for_bit(parallel):
    weight, scale = _bytes(1)
    e = _fp8_embedding()
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    e.weight.data.copy_(weight)
    e.scale.data.copy_(scale)
    ids = torch.randint(
        -4, ROWS + 4, (17, 6), generator=torch.Generator().manual_seed(2)
    )
    got = hooks.tpu_dsv41_engram_owned_rows(None, e, ids)
    want = _reference(weight, scale, ids)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_shard_lookup_offsets_by_tpu_row_start(parallel, dtype):
    """Each shard returns the full-table rows it owns and zero for the rest."""
    world = 4
    parallel.attn_dp_size = world
    weight, scale = _bytes(5)
    ids = torch.randint(
        -4, ROWS + 4, (17, 6), generator=torch.Generator().manual_seed(6)
    ).to(dtype)
    want = _reference(weight, scale, ids.long())
    rows = ROWS // world
    for rank in range(world):
        parallel.attn_dp_rank = rank
        e = _byte_embedding(weight, scale, rank, world)
        got = e._owned_rows(ids)
        owned = (ids >= rank * rows) & (ids < (rank + 1) * rows)
        assert got.shape == want.shape
        assert torch.equal(got[owned], want[owned])
        assert not got[~owned].any()


def test_dp_lookup_through_all_reduce_matches_a_full_table_lookup(monkeypatch):
    """Two DP ranks, each owning half of the table, look up their own tokens.

    The fake all-reduce sums the two ranks' tensors (threads meet at a
    barrier), as tensor_model_parallel_all_reduce does across the TP group.
    Equal up to the sign of zero: the owner's -0.0 plus another rank's +0.0
    is +0.0, as with the core reduce-scatter.
    """
    world, local_rows = 2, 5
    weight, scale = _bytes(3)
    ids = torch.randint(
        0, ROWS, (world, local_rows, 6), generator=torch.Generator().manual_seed(4)
    )
    tls = threading.local()
    barrier = threading.Barrier(world)
    slots = [None] * world
    results = [None] * world
    errors = []

    def fake_all_reduce(t):
        slots[tls.rank] = t
        barrier.wait()
        total = sum(slots)
        barrier.wait()
        return total

    from sglang.srt import distributed, runtime_context
    from sglang.srt.layers import dp_attention

    monkeypatch.setattr(
        distributed, "tensor_model_parallel_all_reduce", fake_all_reduce
    )
    monkeypatch.setattr(
        dp_attention, "get_global_dp_buffer_len", lambda: world * local_rows
    )
    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(attn_dp_size=world, attn_dp_rank=tls.rank),
    )

    def run(rank):
        try:
            tls.rank = rank
            e = _byte_embedding(weight, scale, rank, world)
            results[rank] = hooks.tpu_dsv41_engram_dp_lookup(None, e, ids[rank], None)
        except Exception as exc:  # surface thread failures in the test
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for rank in range(world):
        want = _reference(weight, scale, ids[rank])
        assert results[rank].shape == (local_rows, 6, DIM)
        assert torch.equal(results[rank].float(), want.float())


def _trace(fn, ids):
    """Dynamo graph of fn(ids) (one graph, no breaks), its example inputs, and the
    compiled result."""
    graphs = []

    def capture(gm, example_inputs):
        graphs.append((gm.code, list(example_inputs)))
        return gm.forward

    torch._dynamo.reset()
    try:
        out = torch.compile(fn, backend=capture, fullgraph=True)(ids)
    finally:
        torch._dynamo.reset()
    assert len(graphs) == 1
    return graphs[0][0], graphs[0][1], out


@pytest.mark.parametrize("hook", ["dp_lookup", "owned_rows"])
def test_traced_lookup_is_the_same_graph_on_every_rank(monkeypatch, parallel, hook):
    """Dynamo traces one program for all ranks: the rank and the table offset are
    graph inputs, not constants. The all-reduce is a traceable stand-in."""
    from sglang.srt import distributed
    from sglang.srt.layers import dp_attention

    world, local_rows = 4, 5
    rows = ROWS // world
    monkeypatch.setattr(
        distributed, "tensor_model_parallel_all_reduce", lambda t: t.clone()
    )
    monkeypatch.setattr(
        dp_attention, "get_global_dp_buffer_len", lambda: world * local_rows
    )
    parallel.attn_dp_size = world
    weight, scale = _bytes(7)
    ids = torch.randint(
        0, ROWS, (local_rows, 6), generator=torch.Generator().manual_seed(8)
    ).to(torch.int32)  # TPU hash ids are int32

    codes = []
    for rank in range(world):
        parallel.attn_dp_rank = rank
        e = _byte_embedding(weight, scale, rank, world)
        if hook == "dp_lookup":

            def fn(x, e=e):
                return hooks.tpu_dsv41_engram_dp_lookup(None, e, x, None)

            used = RANK_TENSORS
        else:

            def fn(x, e=e):
                return hooks.tpu_dsv41_engram_owned_rows(None, e, x)

            used = ("tpu_row_start",)
        code, inputs, out = _trace(fn, ids)
        assert torch.equal(out.view(torch.int16), fn(ids).view(torch.int16))
        for name in used:
            assert any(t is getattr(e, name) for t in inputs), (rank, name)
        if rank * rows not in (0, rows):  # literals that also occur as shapes
            assert not re.search(rf"\b{rank * rows}\b", code), (rank, code)
        codes.append(code)
    assert all(code == codes[0] for code in codes), codes


def test_trace_check_sees_a_python_int_rank(parallel):
    """Control for the test above: an offset read as a Python int is a graph
    constant, and the per-rank graphs differ."""
    codes = []
    for rank in range(2):
        e = _fp8_embedding(row_start=rank * 1000 + 7, rows=8, tp_size=2)

        def fn(x, e=e):
            return x - e.row_start

        codes.append(_trace(fn, torch.zeros(3, dtype=torch.int32))[0])
    assert codes[0] != codes[1]
    assert re.search(r"\b1007\b", codes[1])
