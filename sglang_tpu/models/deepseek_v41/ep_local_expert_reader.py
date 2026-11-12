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

import concurrent.futures
import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter, sleep
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
_GCS_WORKERS = 16
_GCS_RETRIES = 3

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


def is_gcs_uri(path: str) -> bool:
    return str(path).lower().startswith("gs://")


def _open_gcs_blob(uri: str):
    import requests.adapters
    from google.cloud import storage

    bucket_name, _, blob_name = uri[5:].partition("/")
    blob_name = blob_name.lstrip("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    client = storage.Client()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=_GCS_WORKERS * 2,
        pool_maxsize=_GCS_WORKERS * 2,
    )
    client._http.mount("https://", adapter)
    client._http.mount("http://", adapter)
    return client.bucket(bucket_name).blob(blob_name)


def _gcs_download_range(blob, start: int, length: int, name: str, uri: str) -> bytes:
    if length == 0:
        return b""
    last_err: Exception | None = None
    for attempt in range(_GCS_RETRIES):
        try:
            data = blob.download_as_bytes(
                start=start, end=start + length - 1, raw_download=True
            )
            if len(data) != length:
                raise EOFError(
                    f"Short GCS range read for {name}: expected {length}B, "
                    f"got {len(data)}B from {uri}"
                )
            return data
        except Exception as exc:
            last_err = exc
            if attempt + 1 < _GCS_RETRIES:
                sleep(0.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def read_gcs_text(uri: str) -> str:
    """Download a small GCS object (e.g. safetensors index JSON) as UTF-8 text."""
    return _open_gcs_blob(uri).download_as_bytes(raw_download=True).decode("utf-8")


def _read_gcs_range(
    blob, file_start: int, out: torch.Tensor, name: str, uri: str
) -> None:
    """Fill uint8 ``out`` from a GCS blob via parallel 64 MB HTTP range reads."""
    nbytes = out.numel()
    if nbytes == 0:
        return
    mv = memoryview(out.numpy())
    spans = [
        (pos, min(_READ_CHUNK_BYTES, nbytes - pos))
        for pos in range(0, nbytes, _READ_CHUNK_BYTES)
    ]

    def _read_span(span: tuple[int, int]) -> None:
        rel, length = span
        mv[rel : rel + length] = _gcs_download_range(
            blob, file_start + rel, length, name, uri
        )

    if len(spans) <= 1:
        for span in spans:
            _read_span(span)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_GCS_WORKERS, len(spans))
        ) as pool:
            for _ in pool.map(_read_span, spans):
                pass


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
    """Yield this TP rank's rows of each 2-D tensor in ``names``, tagged with
    ``ROW_START_ATTR``. Rows are contiguous in safetensors, so each table is one
    byte range."""
    use_gcs = is_gcs_uri(path)
    blob = _open_gcs_blob(path) if use_gcs else None
    if use_gcs:
        header_len = int.from_bytes(
            _gcs_download_range(blob, 0, 8, "__header_len__", path), "little"
        )
        header = json.loads(
            _gcs_download_range(blob, 8, header_len, "__header__", path)
        )
    else:
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
        out = torch.empty((hi - lo) * row_bytes, dtype=torch.uint8)
        t0 = perf_counter()
        if use_gcs:
            _read_gcs_range(blob, data_base + start + lo * row_bytes, out, name, path)
        else:
            _read_range(path, data_base + start + lo * row_bytes, out)
        logger.info(
            "V4.1 %s rows [%d, %d): %.2f GB in %.1f s",
            name,
            lo,
            hi,
            out.numel() / 1e9,
            perf_counter() - t0,
        )
        tensor = out.view(dtype).view(hi - lo, *row_shape)
        setattr(tensor, ROW_START_ATTR, lo)
        yield name, tensor


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
    """AROUND ``EngramEmbedding._load_rows``: accept a tensor already holding
    exactly this rank's rows; anything else takes the original full-table path."""
    row_start = getattr(loaded_weight, ROW_START_ATTR, None)
    if row_start is None:
        return original(embedding, param, loaded_weight)
    if row_start != embedding.row_start or loaded_weight.shape[0] != embedding.rows:
        raise ValueError(
            f"Engram rows [{row_start}, {row_start + loaded_weight.shape[0]}) do not "
            f"match this rank's [{embedding.row_start}, "
            f"{embedding.row_start + embedding.rows})"
        )
    if embedding._shared:
        rows = slice(embedding.row_start, embedding.row_start + embedding.rows)
        param.data[rows].copy_(loaded_weight)
    else:
        param.data.copy_(loaded_weight)
    if embedding.host_table is not None:
        embedding.host_table.dirty = True
