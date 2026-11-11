# SPDX-License-Identifier: Apache-2.0
"""EP-local reads for the DSV4.1 loader: the expert plan and Engram row ranges."""

import json
import struct
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import (
    ROW_START_ATTR,
    load_local_rows,
    plan_expert_reads,
    read_row_ranges,
    tp_row_range,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _weight_map(layers=2, experts=8):
    names = {"embed.weight": "s0", "layers.0.attn.wq_a.weight": "s0"}
    for layer in range(layers):
        for expert in range(experts):
            for w in ("w1", "w2", "w3"):
                for kind in ("weight", "scale"):
                    names[f"layers.{layer}.ffn.experts.{expert}.{w}.{kind}"] = (
                        f"e{expert}"
                    )
    names["mtp.0.ffn.experts.0.w1.weight"] = "mtp"
    return names


@pytest.mark.parametrize("ep_rank", range(4))
def test_keeps_local_experts_and_all_replicated_tensors(ep_rank):
    plan = plan_expert_reads(
        _weight_map(), num_layers=2, num_experts=8, ep_size=4, ep_rank=ep_rank
    )
    local = {2 * ep_rank, 2 * ep_rank + 1}
    assert plan.local_expert_tensors == 2 * 2 * 6
    assert "embed.weight" in plan.tensor_names
    assert "layers.0.attn.wq_a.weight" in plan.tensor_names
    assert not any(n.startswith("mtp.") for n in plan.tensor_names)
    assert plan.shard_names == {"s0"} | {f"e{e}" for e in local}


def test_fails_closed_on_missing_expert_tensors():
    names = _weight_map()
    del names["layers.1.ffn.experts.7.w3.scale"]
    with pytest.raises(ValueError, match="routed expert tensors"):
        plan_expert_reads(names, num_layers=2, num_experts=8, ep_size=4, ep_rank=0)


def test_rejects_indivisible_ep():
    with pytest.raises(ValueError):
        plan_expert_reads(
            _weight_map(), num_layers=2, num_experts=8, ep_size=3, ep_rank=0
        )


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).flatten().numpy().tobytes()
        dtype = {torch.bfloat16: "BF16", torch.float32: "F32"}[t.dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


@pytest.mark.parametrize("tp_size", [1, 3, 8])
def test_row_ranges_equal_table_slices(tmp_path, monkeypatch, tp_size):
    """Each rank's pread range must be exactly its EngramEmbedding rows, also
    when rows do not divide evenly and the range spans many read chunks."""
    from sglang_tpu.models.deepseek_v41 import ep_local_expert_reader as reader

    monkeypatch.setattr(reader, "_READ_CHUNK_BYTES", 96)
    tables = {
        "layers.1.engram.embed.weight": torch.randn(101, 16).bfloat16(),
        "layers.1.engram.embed.scale": torch.randn(101, 3),
        "other": torch.randn(5, 7),
    }
    path = tmp_path / "shard.safetensors"
    _write_safetensors(path, tables)
    names = {"layers.1.engram.embed.weight", "layers.1.engram.embed.scale"}
    for rank in range(tp_size):
        got = dict(read_row_ranges(str(path), names, rank, tp_size))
        lo, hi = tp_row_range(101, rank, tp_size)
        for name in names:
            assert torch.equal(got[name], tables[name][lo:hi])
            assert getattr(got[name], ROW_START_ATTR) == lo


def test_preread_rows_skip_the_original_slice_and_check_the_range():
    full = torch.arange(40.0).view(10, 4)
    calls = []
    embedding = SimpleNamespace(row_start=5, rows=5, _shared=True, host_table=None)
    param = torch.nn.Parameter(torch.zeros(10, 4), requires_grad=False)
    rows = full[5:].clone()
    setattr(rows, ROW_START_ATTR, 5)
    load_local_rows(lambda *a: calls.append(a), embedding, param, rows)
    assert torch.equal(param.data[5:], full[5:]) and not calls
    load_local_rows(lambda *a: calls.append(a), embedding, param, full)
    assert len(calls) == 1  # untagged tensors take the original path
    wrong = full[:5].clone()
    setattr(wrong, ROW_START_ATTR, 0)
    with pytest.raises(ValueError, match="do not match"):
        load_local_rows(lambda *a: None, embedding, param, wrong)
