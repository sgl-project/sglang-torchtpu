# SPDX-License-Identifier: Apache-2.0
"""This TP rank's rows of each DeepSeek-V4.1 Engram table.

Each Engram hash table is one ~100 GB tensor, far above a per-rank Run:AI
buffer, and ``EngramEmbedding`` keeps only its TP row range. Each rank reads
that one contiguous byte range with Run:AI's FileStreamer, in blocks tagged
with their first row, and ``load_local_rows`` copies the blocks into place.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from time import perf_counter, sleep

import torch

logger = logging.getLogger(__name__)

_ENGRAM_TABLE = re.compile(r"^layers\.(0|[1-9]\d*)\.engram\.embed\.(weight|scale)$")

# Marks a tensor that holds only rows [value, value + shape[0]) of its table.
ROW_START_ATTR = "_sglang_dsv41_row_start"

_READ_CHUNK_BYTES = 64 * 1024 * 1024
# Rows yielded per Engram block; arbitrary, a few read chunks per block.
_ROW_BLOCK_BYTES = 1024 * 1024 * 1024
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


def is_engram_table(name: str) -> bool:
    return _ENGRAM_TABLE.fullmatch(name) is not None


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


def _read_row_ranges(
    path: str, names: set[str], *, tp_rank: int, tp_size: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield this TP rank's rows of each 2-D tensor in ``names`` in blocks of at
    most ``_ROW_BLOCK_BYTES``, each tagged with its first row in ``ROW_START_ATTR``."""
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
        block_rows = max(1, _ROW_BLOCK_BYTES // row_bytes)
        t0 = perf_counter()
        for first in range(lo, hi, block_rows):
            last = min(first + block_rows, hi)
            out = torch.empty((last - first) * row_bytes, dtype=torch.uint8)
            offset = data_base + start + first * row_bytes
            if use_gcs:
                _read_gcs_range(blob, offset, out, name, path)
            else:
                _read_range(path, offset, out)
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


def iter_row_tables(
    *, shard_of: Mapping[str, str], names: Iterable[str], tp_rank: int, tp_size: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """This TP rank's rows of the Engram tables ``names``, shard by shard;
    ``shard_of`` maps a tensor name to its local shard path."""
    by_shard: dict[str, set[str]] = {}
    for name in names:
        by_shard.setdefault(shard_of[name], set()).add(name)
    for path in sorted(by_shard):
        yield from _read_row_ranges(
            path, by_shard[path], tp_rank=tp_rank, tp_size=tp_size
        )


def load_local_rows(original, embedding, param, loaded_weight):
    """AROUND ``EngramEmbedding._load_rows``: copy a block of this rank's rows
    (tagged by ``iter_row_tables``) into place; untagged tensors take the
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
