# SPDX-License-Identifier: Apache-2.0
"""Unit tests for direct gs:// TP-sharded Engram row streaming."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41 import engram_row_reader as reader
from sglang_tpu.models.deepseek_v41.loader import DSV41RunaiModelStreamerLoader

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_DTYPE_NAMES = {
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.int32: "I32",
    torch.uint8: "U8",
}

_ROWS_PER_RANK = {8: [12, 13, 12, 13, 13, 12, 13, 13]}


def _write_safetensors(path, tensors: dict[str, torch.Tensor]) -> None:
    header = {}
    chunks = []
    offset = 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": _DTYPE_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        for chunk in chunks:
            f.write(chunk)


def test_gcs_row_ranges_partition_each_table(tmp_path, monkeypatch):
    """gs:// URIs stream only this TP rank's row range in bounded blocks via
    parallel HTTP range requests."""
    monkeypatch.setattr(reader, "_READ_CHUNK_BYTES", 96)
    monkeypatch.setattr(reader, "_ROW_BLOCK_BYTES", 160)
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
    for name in names:
        for r in range(8):
            blocks = [
                t
                for n, t in reader._read_row_ranges(uri, names, tp_rank=r, tp_size=8)
                if n == name
            ]
            if name.endswith(".weight"):
                assert len(blocks) > 1
            lo = sum(_ROWS_PER_RANK[8][:r])
            starts = [getattr(b, reader.ROW_START_ATTR) for b in blocks]
            assert starts[0] == lo
            assert starts == [
                lo + sum(b.shape[0] for b in blocks[:i]) for i in range(len(blocks))
            ]
            got = torch.cat(blocks)
            assert got.shape[0] == _ROWS_PER_RANK[8][r]
            assert torch.equal(got, tables[name][lo : lo + got.shape[0]])


def test_loader_split_kept_allows_gcs_and_rejects_other_obj_uris(monkeypatch):
    import sglang.srt.runtime_context as runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, tp_size=8),
    )
    loader = object.__new__(DSV41RunaiModelStreamerLoader)
    kept = frozenset(
        {
            "layers.1.engram.embed.weight",
            "layers.0.self_attn.q_proj.weight",
            "mtp.0.weight",
        }
    )
    gcs_shards = {
        "layers.1.engram.embed.weight": "gs://bucket/model/shard-0.safetensors",
        "layers.0.self_attn.q_proj.weight": "gs://bucket/model/shard-1.safetensors",
        "mtp.0.weight": "gs://bucket/model/shard-1.safetensors",
    }
    streamed, _rows = loader._split_kept(kept=kept, shard_of=gcs_shards)
    assert streamed == frozenset({"layers.0.self_attn.q_proj.weight"})

    s3_shards = dict(gcs_shards)
    s3_shards["layers.1.engram.embed.weight"] = "s3://bucket/model/shard-0.safetensors"
    with pytest.raises(ValueError, match="gs://"):
        loader._split_kept(kept=kept, shard_of=s3_shards)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
