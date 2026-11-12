# SPDX-License-Identifier: Apache-2.0
"""EP-local reads for the DSV4.1 loader: the expert plan and Engram row ranges."""

import json
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import (
    ROW_START_ATTR,
    load_local_rows,
    plan_expert_reads,
    read_row_ranges,
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


class _FakeFileStreamer:
    """RunAI FileStreamer stand-in: reads each requested range with plain file
    I/O and returns the ranges in reverse order, as native workers may."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream_files(self, requests, memory_limit=None):
        self.requests = requests

    def get_chunks(self):
        for request in self.requests:
            with open(request.path, "rb") as f:
                chunks = []
                for index, (offset, size) in enumerate(
                    zip(request.offsets, request.sizes)
                ):
                    f.seek(offset)
                    data = bytearray(f.read(size))
                    chunks.append(
                        (request.id, index, torch.frombuffer(data, dtype=torch.uint8))
                    )
            yield from reversed(chunks)


@pytest.fixture
def fake_runai(monkeypatch):
    module = ModuleType("runai_model_streamer.file_streamer")
    module.FileStreamer = _FakeFileStreamer
    module.FileChunks = lambda id, path, offsets, sizes: SimpleNamespace(
        id=id, path=path, offsets=offsets, sizes=sizes
    )
    monkeypatch.setitem(
        sys.modules, "runai_model_streamer", ModuleType("runai_model_streamer")
    )
    monkeypatch.setitem(sys.modules, "runai_model_streamer.file_streamer", module)


# Rows per rank for a 101-row table, written out so the oracle does not reuse
# tp_row_range: floor splits, uneven when 101 does not divide.
_ROWS_PER_RANK = {1: [101], 3: [33, 34, 34], 8: [12, 13, 12, 13, 13, 12, 13, 13]}


@pytest.mark.parametrize("tp_size", [1, 3, 8])
def test_rank_row_ranges_partition_each_table(
    tmp_path, monkeypatch, fake_runai, tp_size
):
    """Each rank reads only its rows: concatenated over ranks they rebuild every
    table exactly once, also when 101 rows do not divide and a range spans many
    out-of-order read chunks."""
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
    per_rank = [
        dict(read_row_ranges(str(path), names, r, tp_size)) for r in range(tp_size)
    ]
    for name in names:
        parts = [got[name] for got in per_rank]
        assert [p.shape[0] for p in parts] == _ROWS_PER_RANK[tp_size]
        starts = [getattr(p, ROW_START_ATTR) for p in parts]
        assert starts == [sum(_ROWS_PER_RANK[tp_size][:r]) for r in range(tp_size)]
        assert torch.equal(torch.cat(parts), tables[name])


def test_gcs_row_ranges_partition_each_table(tmp_path, monkeypatch):
    """gs:// URIs read only this TP rank's byte range via parallel range requests."""
    from sglang_tpu.models.deepseek_v41 import ep_local_expert_reader as reader

    monkeypatch.setattr(reader, "_READ_CHUNK_BYTES", 96)
    tables = {
        "layers.1.engram.embed.weight": torch.randn(101, 16).bfloat16(),
        "layers.1.engram.embed.scale": torch.randn(101, 3),
        "other": torch.randn(5, 7),
    }
    path = tmp_path / "shard.safetensors"
    _write_safetensors(path, tables)
    payload = path.read_bytes()

    class _FakeBlob:
        def download_as_bytes(self, start=None, end=None, raw_download=True):
            if start is None and end is None:
                return payload
            return payload[start : end + 1]

    monkeypatch.setattr(reader, "_open_gcs_blob", lambda uri: _FakeBlob())
    names = {"layers.1.engram.embed.weight", "layers.1.engram.embed.scale"}
    uri = "gs://fake-bucket/dsv41/shard.safetensors"
    per_rank = [dict(read_row_ranges(uri, names, r, 8)) for r in range(8)]
    for name in names:
        parts = [got[name] for got in per_rank]
        assert [p.shape[0] for p in parts] == _ROWS_PER_RANK[8]
        starts = [getattr(p, ROW_START_ATTR) for p in parts]
        assert starts == [sum(_ROWS_PER_RANK[8][:r]) for r in range(8)]
        assert torch.equal(torch.cat(parts), tables[name])


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
