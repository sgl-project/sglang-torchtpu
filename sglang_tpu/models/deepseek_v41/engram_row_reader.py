# SPDX-License-Identifier: Apache-2.0
"""This TP rank's rows of each DeepSeek-V4.1 Engram table.

Each Engram hash table is one ~100 GB tensor, far above a per-rank Run:AI
buffer, and ``EngramEmbedding`` keeps only its TP row range. Each rank reads
that one contiguous byte range with Run:AI's FileStreamer.

Shared host tables (placement cpu, replication per_node) are filled in place by
a child process (``engram_row_child.py``) started when the loader plans its
stream, so the rows download while the rest of the checkpoint streams. It runs
in its own process because Run:AI 0.17's GCS backend loses responses when two
streamers run at once in one process. It writes into the tables' memfds, so
there is no staging copy; the loader then yields one empty block per table,
tagged with its first row, so ``load_local_rows`` marks the table loaded.

Other layouts read each table after the stream, in blocks tagged with their
first row, and ``load_local_rows`` copies the blocks into place.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import weakref
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from time import perf_counter

import msgspec
import torch

logger = logging.getLogger(__name__)

_ENGRAM_TABLE = re.compile(r"^layers\.(0|[1-9]\d*)\.engram\.embed\.(weight|scale)$")

# Marks a tensor that holds only rows [value, value + shape[0]) of its table.
ROW_START_ATTR = "_sglang_dsv41_row_start"

_READ_CHUNK_BYTES = 64 * 1024 * 1024
# Ranges in flight (2 GiB): one retried range does not idle the stream; arbitrary.
_READ_INFLIGHT_CHUNKS = 32
# Rows yielded per Engram block; arbitrary, a few read chunks per block.
_ROW_BLOCK_BYTES = 1024 * 1024 * 1024
# The child's whole read; contended 2026-11 boots took up to 1500 s per table.
_CHILD_DEADLINE_S = 7200.0
# Extra wait past the child's own deadline before the parent kills it.
_CHILD_GRACE_S = 60.0
_CHILD_SCRIPT = Path(__file__).with_name("engram_row_child.py")

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


def engram_row_ranges(
    names: Iterable[str], hf_config, *, tp_rank: int, tp_size: int
) -> dict[str, tuple[int, int]]:
    """The ``[start, end)`` rows ``EngramEmbedding`` assigns to ``tp_rank``, per
    table: core's ``shard_range``, on the layer's hash-column boundaries when
    they divide over the TP group."""
    from sglang.srt.layers.engram import EngramLayout, shard_range

    layout = EngramLayout.from_config(hf_config)
    ranges = {}
    for name in names:
        index = layout.layer_ids.index(int(_ENGRAM_TABLE.fullmatch(name).group(1)))
        ranges[name] = shard_range(
            layout.num_embeddings[index],
            layout.column_bounds(index),
            tp_rank,
            tp_size,
        )
    return ranges


def _read_range(path: str, offset: int, out: torch.Tensor) -> None:
    """Fill uint8 ``out`` from ``path[offset:]`` in 64 MB ranges with RunAI's
    FileStreamer (native workers, ``_READ_INFLIGHT_CHUNKS`` ranges in flight);
    Python preads were measured at ~3.6 MB/s per rank on the GCS FUSE mount."""
    from runai_model_streamer.file_streamer import FileChunks, FileStreamer

    starts = list(range(0, out.numel(), _READ_CHUNK_BYTES))
    sizes = [min(_READ_CHUNK_BYTES, out.numel() - s) for s in starts]
    in_flight = max(sizes, default=1) * min(len(sizes), _READ_INFLIGHT_CHUNKS)
    with FileStreamer() as streamer:
        streamer.stream_files(
            [FileChunks(0, path, [offset + s for s in starts], sizes)],
            memory_limit=in_flight,
        )
        for _, index, chunk in streamer.get_chunks():
            start = starts[index]
            out[start : start + sizes[index]].copy_(chunk.view(-1))


def _read_safetensors_header(path: str) -> tuple[int, dict]:
    """(header length, header) of a local, mounted or object-storage shard."""
    from sglang.srt.utils.runai_utils import is_runai_obj_uri

    if not is_runai_obj_uri(path):
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            return header_len, json.loads(f.read(header_len))
    size = torch.empty(8, dtype=torch.uint8)
    _read_range(path, 0, size)
    header_len = int.from_bytes(size.numpy().tobytes(), "little")
    raw = torch.empty(header_len, dtype=torch.uint8)
    _read_range(path, 8, raw)
    return header_len, json.loads(raw.numpy().tobytes())


class SharedRows(msgspec.Struct, frozen=True):
    """Where this rank's rows of one table live in a shared host table: the
    memfd ``fd`` of this process, the byte offset of row ``lo`` and the row size."""

    fd: int
    dest_offset: int
    dest_row_bytes: int
    # An empty view of the table at row ``lo``: the block the loader yields.
    marker: torch.Tensor


def shared_table_rows(
    model, ranges: Mapping[str, tuple[int, int]]
) -> dict[str, SharedRows] | None:
    """``SharedRows`` per table when every table in ``ranges`` lives in a
    per_node host table and owns exactly those rows; None otherwise."""
    from sglang.srt.layers.engram import EngramEmbedding, EngramReplication

    found = {}
    for name, (lo, hi) in ranges.items():
        module, param_name = name.rsplit(".", 1)
        embed = model.get_submodule(f"model.{module}")
        if not isinstance(embed, EngramEmbedding):
            return None
        table = embed.host_table
        if (
            table is None
            or table.replication != EngramReplication.PER_NODE
            or table.fd is None
            or (lo, hi) != (embed.row_start, embed.row_start + embed.rows)
        ):
            return None
        rows = (embed.weight if param_name == "weight" else embed.scale).data[lo:hi]
        if not rows.is_contiguous():
            return None
        found[name] = SharedRows(
            fd=table.fd,
            dest_offset=rows.data_ptr() - table.bytes.data_ptr(),
            dest_row_bytes=rows.stride(0) * rows.element_size(),
            marker=rows[:0],
        )
    return found


def iter_row_tables(
    *,
    shard_of: Mapping[str, str],
    ranges: Mapping[str, tuple[int, int]],
    shared: Mapping[str, SharedRows] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Rows ``ranges[name]`` of each Engram table, as blocks tagged with their
    first row in ``ROW_START_ATTR``; ``shard_of`` maps a name to its shard.

    With ``shared`` (``shared_table_rows``) a child process starts writing the
    rows into the tables now; the iterator waits for it and yields one empty
    block per table. Without, the iterator reads the rows when consumed."""
    if not ranges:
        return iter(())
    names = sorted(ranges, key=lambda n: (shard_of[n], n))
    if shared is None:
        return _iter_serial(shard_of=shard_of, ranges=ranges, names=names)
    child = _RowChild(
        [
            {
                "name": n,
                "path": shard_of[n],
                "lo": ranges[n][0],
                "hi": ranges[n][1],
                "fd": shared[n].fd,
                "dest_offset": shared[n].dest_offset,
                "dest_row_bytes": shared[n].dest_row_bytes,
            }
            for n in names
        ]
    )
    return _iter_child(child=child, ranges=ranges, shared=shared, names=names)


class _RowChild:
    """The ``engram_row_child.py`` process and its status directory; killed and
    removed on every exit path (``close``, garbage collection, parent death)."""

    def __init__(self, tables: list[dict]):
        self.dir = tempfile.mkdtemp(
            prefix="dsv41-engram-",
            dir="/dev/shm" if os.path.isdir("/dev/shm") else None,
        )
        spec = os.path.join(self.dir, "spec.json")
        with open(spec, "w") as f:
            json.dump(
                {
                    "parent_pid": os.getpid(),
                    "tables": tables,
                    "chunk_bytes": _READ_CHUNK_BYTES,
                    "in_flight_chunks": _READ_INFLIGHT_CHUNKS,
                    "deadline_s": _CHILD_DEADLINE_S,
                },
                f,
            )
        self.started = time.monotonic()
        self._stderr = open(os.path.join(self.dir, "stderr"), "w+b")
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(_CHILD_SCRIPT), spec],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )
        self._finalizer = weakref.finalize(
            self, _kill_and_remove, self.proc, self.dir, self._stderr
        )

    def wait(self) -> list[dict]:
        """The child's ``done.json``; raises if it failed, exited without it, or
        missed the deadline (then it is killed)."""
        done = os.path.join(self.dir, "done.json")
        deadline = self.started + _CHILD_DEADLINE_S + _CHILD_GRACE_S
        while True:
            if os.path.exists(done):
                self.proc.wait(timeout=60)
                with open(done) as f:
                    return json.load(f)
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"Engram row reader exited with {code} before finishing: "
                    f"{self._error_text()}"
                )
            if time.monotonic() > deadline:
                self.proc.kill()
                raise TimeoutError(
                    f"Engram row reader did not finish in {_CHILD_DEADLINE_S:.0f} s"
                )
            time.sleep(0.5)

    def _error_text(self) -> str:
        error = os.path.join(self.dir, "error")
        if os.path.exists(error):
            return open(error).read()[-2000:]
        self._stderr.flush()
        self._stderr.seek(0)
        return self._stderr.read()[-2000:].decode(errors="replace")

    def close(self) -> None:
        self._finalizer()


def _kill_and_remove(proc: subprocess.Popen, path: str, stderr) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    stderr.close()
    shutil.rmtree(path, ignore_errors=True)


def _iter_child(
    *,
    child: _RowChild,
    ranges: Mapping[str, tuple[int, int]],
    shared: Mapping[str, SharedRows],
    names: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        t0 = perf_counter()
        done = {d["name"]: d for d in child.wait()}
        waited = perf_counter() - t0
        for name in names:
            d = done.get(name)
            if d is None or d["written"] != d["nbytes"]:
                raise RuntimeError(f"Engram row reader left {name} incomplete: {d}")
            lo, hi = ranges[name]
            logger.info(
                "V4.1 %s rows [%d, %d): %.2f GB in %.1f s (waited %.1f s)",
                name,
                lo,
                hi,
                d["nbytes"] / 1e9,
                d["seconds"],
                waited,
            )
        for name in names:
            marker = shared[name].marker
            setattr(marker, ROW_START_ATTR, ranges[name][0])
            yield name, marker
    finally:
        child.close()


def _iter_serial(
    *,
    shard_of: Mapping[str, str],
    ranges: Mapping[str, tuple[int, int]],
    names: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    headers: dict[str, tuple[int, dict]] = {}
    for name in names:
        path = shard_of[name]
        if path not in headers:
            headers[path] = _read_safetensors_header(path)
        header_len, header = headers[path]
        meta = header[name]
        dtype = _SAFETENSORS_DTYPES[meta["dtype"]]
        num_rows, *row_shape = meta["shape"]
        start, end = meta["data_offsets"]
        row_bytes = (end - start) // num_rows
        lo, hi = ranges[name]
        if hi > num_rows:
            raise ValueError(
                f"{name} has {num_rows} rows; this rank's range is [{lo}, {hi})"
            )
        block_rows = max(1, _ROW_BLOCK_BYTES // row_bytes)
        t0 = perf_counter()
        for first in range(lo, hi, block_rows):
            last = min(first + block_rows, hi)
            out = torch.empty((last - first) * row_bytes, dtype=torch.uint8)
            _read_range(path, 8 + header_len + start + first * row_bytes, out)
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
