# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 TPU Engram tables held as uint8 bytes, and the all-reduce DP lookup."""

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41 import hooks
from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import ROW_START_ATTR

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROWS, DIM, BLOCK = 96, 256, 32


def _bytes(seed):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (ROWS, DIM), dtype=torch.uint8, generator=g)
    weight[(weight & 0x7F) == 0x7F] = 0x38  # no e4m3fn NaN encodings
    scale = torch.randint(
        110, 140, (ROWS, DIM // BLOCK), dtype=torch.uint8, generator=g
    )
    return weight, scale


def _fp8_embedding(row_start=0, rows=ROWS, tp_size=1):
    e = SimpleNamespace(
        host_table=None,
        rows=rows,
        row_start=row_start,
        _shared=False,
        dim=DIM,
        tp_size=tp_size,
    )
    e.weight = torch.nn.Parameter(
        torch.empty(rows, DIM, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    e.scale = torch.nn.Parameter(
        torch.empty(rows, DIM // BLOCK, dtype=torch.float8_e8m0fnu), requires_grad=False
    )
    e.weight.weight_loader = "weight-loader"
    e.scale.weight_loader = "scale-loader"
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
        weight=weight.view(torch.float8_e4m3fn),
        scale=scale.view(torch.float8_e8m0fnu),
    )
    return hooks.tpu_dsv41_engram_owned_rows(None, full, ids)


def test_tables_become_uint8_and_keep_their_loaders():
    e = _fp8_embedding()
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    assert e.weight.dtype == e.scale.dtype == torch.uint8
    assert e.weight.shape == (ROWS, DIM) and e.scale.shape == (ROWS, DIM // BLOCK)
    assert e.weight.weight_loader == "weight-loader"
    assert e.scale.weight_loader == "scale-loader"


def test_host_tables_are_left_alone():
    e = _fp8_embedding()
    e.host_table = object()
    hooks.tpu_dsv41_engram_bytes_init(None, e)
    assert e.weight.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("tagged", [False, True])
def test_load_copies_bytes(tagged):
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


def test_uint8_lookup_matches_the_fp8_lookup_bit_for_bit():
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

    import sglang.srt.distributed as distributed
    import sglang.srt.layers.dp_attention as dp_attention
    import sglang.srt.runtime_context as runtime_context

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
            half = ROWS // world
            e = _fp8_embedding(row_start=rank * half, rows=half, tp_size=world)
            hooks.tpu_dsv41_engram_bytes_init(None, e)
            e.weight.data.copy_(weight[rank * half : (rank + 1) * half])
            e.scale.data.copy_(scale[rank * half : (rank + 1) * half])
            e._owned_rows = lambda idx, e=e: hooks.tpu_dsv41_engram_owned_rows(
                None, e, idx
            )
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
