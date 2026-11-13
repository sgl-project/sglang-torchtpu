# SPDX-License-Identifier: Apache-2.0
"""EP-local checkpoint reads for DeepSeek-V4.1 Flash.

Each EP rank reads only its routed experts plus the replicated tensors, and
only its own TP row range of every Engram hash table. Experts and replicated
tensors stream through RunAI; Engram tables (~100 GB single ranges, far above a
per-rank RunAI buffer) are read with concurrent ``pread`` of one contiguous byte
range per rank, like the GPU loader's background prefetch.

The read plan follows sgl-project/sglang-torchtpu#1744 (e0a43ce363).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Mapping

import torch

logger = logging.getLogger(__name__)

_EXPERT = re.compile(
    r"^layers\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w[123]\.(weight|scale)$"
)
_ENGRAM_TABLE = re.compile(r"^layers\.(0|[1-9]\d*)\.engram\.embed\.(weight|scale)$")

# Marks a tensor that holds only rows [value, value + shape[0]) of its table.
ROW_START_ATTR = "_sglang_dsv41_row_start"

_READ_CHUNK_BYTES = 64 * 1024 * 1024
# Rows yielded per Engram block; arbitrary, a few read chunks per block.
_ROW_BLOCK_BYTES = 1024 * 1024 * 1024

_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}


@dataclass(frozen=True)
class ExpertReadPlan:
    tensor_names: frozenset[str]
    shard_names: frozenset[str]
    total_tensors: int
    local_expert_tensors: int


def plan_expert_reads(
    weight_map: Mapping[str, str],
    *,
    num_layers: int,
    num_experts: int,
    ep_size: int,
    ep_rank: int,
) -> ExpertReadPlan:
    """Keep local routed experts and every replicated tensor; drop MTP.

    Unknown names are retained rather than silently dropped. Fails closed when
    the index does not list exactly ``num_layers * num_experts * 6`` expert
    tensors, so a naming mismatch cannot look like a fast load.
    """
    if num_layers <= 0 or num_experts <= 0:
        raise ValueError("Layer and expert counts must be positive")
    if ep_size <= 0 or num_experts % ep_size or not 0 <= ep_rank < ep_size:
        raise ValueError("Expert count must divide EP size and rank must be in range")
    local_count = num_experts // ep_size
    lo, hi = ep_rank * local_count, (ep_rank + 1) * local_count
    selected = set()
    expert_count = local_expert_count = 0
    for name in weight_map:
        if name.startswith("mtp."):
            continue
        match = _EXPERT.fullmatch(name)
        if match is not None:
            layer, expert = int(match[1]), int(match[2])
            if layer >= num_layers or expert >= num_experts:
                raise ValueError(f"Checkpoint expert outside configured model: {name}")
            expert_count += 1
            if not lo <= expert < hi:
                continue
            local_expert_count += 1
        selected.add(name)
    expected = num_layers * num_experts * 6
    if expert_count != expected:
        raise ValueError(
            f"Expected {expected} routed expert tensors, found {expert_count}"
        )
    return ExpertReadPlan(
        frozenset(selected),
        frozenset(weight_map[name] for name in selected),
        len(weight_map),
        local_expert_count,
    )


def tp_row_range(num_rows: int, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """The ``[start, end)`` rows ``EngramEmbedding`` assigns to ``tp_rank``."""
    return num_rows * tp_rank // tp_size, num_rows * (tp_rank + 1) // tp_size


def _read_range(path: str, offset: int, out: torch.Tensor) -> None:
    """Fill uint8 ``out`` from ``path[offset:]`` in 64 MB ranges with RunAI's
    FileStreamer (native workers, many ranges in flight); Python preads were
    measured at ~3.6 MB/s per rank on the GCS FUSE mount."""
    from runai_model_streamer.file_streamer import FileChunks, FileStreamer

    starts = list(range(0, out.numel(), _READ_CHUNK_BYTES))
    sizes = [min(_READ_CHUNK_BYTES, out.numel() - s) for s in starts]
    with FileStreamer() as streamer:
        streamer.stream_files(
            [FileChunks(0, path, [offset + s for s in starts], sizes)],
            memory_limit=8 * _READ_CHUNK_BYTES,
        )
        for _, index, chunk in streamer.get_chunks():
            start = starts[index]
            out[start : start + sizes[index]].copy_(chunk.view(-1))


def read_row_ranges(
    path: str, names: set[str], tp_rank: int, tp_size: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield this TP rank's rows of each 2-D tensor in ``names`` as blocks of at
    most ``_ROW_BLOCK_BYTES``, each tagged with its first row in ``ROW_START_ATTR``.
    The loader copies a block into the table before the next is read, so a rank
    holds one block instead of its whole 12 GB range."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    data_base = 8 + header_len
    for name in sorted(names):
        meta = header[name]
        dtype = _SAFETENSORS_DTYPES[meta["dtype"]]
        num_rows, *row_shape = meta["shape"]
        start, end = meta["data_offsets"]
        row_bytes = (end - start) // num_rows
        lo, hi = tp_row_range(num_rows, tp_rank, tp_size)
        block_rows = max(1, _ROW_BLOCK_BYTES // row_bytes)
        t0 = perf_counter()
        for first in range(lo, hi, block_rows):
            last = min(first + block_rows, hi)
            out = torch.empty((last - first) * row_bytes, dtype=torch.uint8)
            _read_range(path, data_base + start + first * row_bytes, out)
            tensor = out.view(dtype).view(last - first, *row_shape)
            setattr(tensor, ROW_START_ATTR, first)
            yield name, tensor
            del out, tensor
        logger.info(
            "V4.1 %s rows [%d, %d): %.2f GB in %.1f s",
            name,
            lo,
            hi,
            (hi - lo) * row_bytes / 1e9,
            perf_counter() - t0,
        )


class EpLocalExpertReader:
    """One rank's selected checkpoint tensors: RunAI for experts and replicated
    tensors, row-range preads for Engram tables."""

    def __init__(
        self,
        weight_map: Mapping[str, str],
        files: list[str],
        plan: ExpertReadPlan,
        *,
        tp_rank: int,
        tp_size: int,
    ):
        self.weight_map = weight_map
        self.plan = plan
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.by_name = {path.rsplit("/", 1)[-1]: path for path in files}
        if not plan.shard_names <= self.by_name.keys():
            raise ValueError("Shard list does not cover the selected checkpoint index")
        self.row_tables = {n for n in plan.tensor_names if _ENGRAM_TABLE.fullmatch(n)}
        self.streamed = plan.tensor_names - self.row_tables

    @property
    def stream_files(self) -> list[str]:
        shards = {self.weight_map[n] for n in self.streamed}
        return [self.by_name[s] for s in sorted(shards)]

    def iter_row_tables(self) -> Iterator[tuple[str, torch.Tensor]]:
        for shard in sorted({self.weight_map[n] for n in self.row_tables}):
            names = {n for n in self.row_tables if self.weight_map[n] == shard}
            yield from read_row_ranges(
                self.by_name[shard], names, self.tp_rank, self.tp_size
            )


def load_local_rows(original, embedding, param, loaded_weight):
    """AROUND ``EngramEmbedding._load_rows``: copy a block of this rank's rows
    (tagged by ``read_row_ranges``) into place; untagged tensors take the
    original full-table path."""
    row_start = getattr(loaded_weight, ROW_START_ATTR, None)
    if row_start is None:
        return original(embedding, param, loaded_weight)
    row_end = row_start + loaded_weight.shape[0]
    if (
        row_start < embedding.row_start
        or row_end > embedding.row_start + embedding.rows
    ):
        raise ValueError(
            f"Engram rows [{row_start}, {row_end}) do not match this rank's "
            f"[{embedding.row_start}, {embedding.row_start + embedding.rows})"
        )
    # The shared table holds every row; a per-rank shard starts at row_start.
    base = 0 if embedding._shared else embedding.row_start
    param.data[row_start - base : row_end - base].copy_(loaded_weight)
    if embedding.host_table is not None:
        embedding.host_table.dirty = True
