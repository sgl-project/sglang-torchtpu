# SPDX-License-Identifier: Apache-2.0
"""RunAI transport with EP-local routed expert reads for V4.1 Flash."""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import logging
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Mapping

import torch

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader

logger = logging.getLogger(__name__)


_EXPERT = re.compile(
    r"^layers\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w[123]\.(weight|scale)$"
)
_ENGRAM_EMBED = re.compile(r"^layers\.(0|[1-9]\d*)\.engram\.embed\.(weight|scale)$")

_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I8": torch.int8,
}
_READ_CHUNK_BYTES = 64 << 20
_SEQ_STEP_BYTES = 16 << 20
_LOCAL_WORKERS = 8
_GCS_WORKERS = 16
_GCS_RETRIES = 3


class _ShardedRows:
    """View-like wrapper exposing only this TP rank's [row_start:row_end] slice."""

    def __init__(
        self, shard: torch.Tensor, full_rows: int, row_start: int, attr_name: str
    ):
        self._shard = shard
        self._row_start = row_start
        self._local_numel = int(shard.numel())
        self._itemsize = int(shard.element_size())
        self.shape = torch.Size((full_rows, *shard.shape[1:]))
        self.ndim = len(self.shape)
        self.dtype = shard.dtype
        self.device = shard.device
        setattr(self, attr_name, True)

    def numel(self) -> int:
        return self._local_numel

    def element_size(self) -> int:
        return self._itemsize

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def __getitem__(self, key):
        stop = self._row_start + self._shard.shape[0]
        if (
            not isinstance(key, slice)
            or key.step not in (None, 1)
            or key.start != self._row_start
            or key.stop != stop
        ):
            raise IndexError(
                f"TP-sharded Engram reader holds rows [{self._row_start}:{stop}], "
                f"got slice {key!r}"
            )
        return self._shard


def _is_gcs_uri(path: str | os.PathLike[str]) -> bool:
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


def _read_gcs_into(
    blob,
    file_start: int,
    raw: torch.Tensor,
    nbytes: int,
    name: str,
    uri: str,
) -> None:
    if nbytes == 0:
        return
    mv = memoryview(raw.numpy())

    def _read_span(span: tuple[int, int]) -> None:
        rel, length = span
        mv[rel : rel + length] = _gcs_download_range(
            blob, file_start + rel, length, name, uri
        )

    spans = [
        (pos, min(_READ_CHUNK_BYTES, nbytes - pos))
        for pos in range(0, nbytes, _READ_CHUNK_BYTES)
    ]
    if len(spans) <= 1:
        for span in spans:
            _read_span(span)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_GCS_WORKERS, len(spans))
        ) as pool:
            for _ in pool.map(_read_span, spans):
                pass


def _read_local_into(
    shard_path: str | os.PathLike[str],
    file_start: int,
    raw: torch.Tensor,
    nbytes: int,
    name: str,
) -> None:
    """Read [file_start, file_start + nbytes) with independent sequential handles.

    Sharing one descriptor across concurrent ``os.pread`` threads breaks FUSE/GCSFuse
    per-handle sequential-read tracking on every kernel page cluster. Giving each
    worker its own unbuffered ``open()`` handle over one contiguous slice preserves
    strictly sequential streaming on FUSE and avoids intermediate ``bytes`` copies.
    """
    if nbytes == 0:
        return
    mv = memoryview(raw.numpy())
    workers = min(
        _LOCAL_WORKERS, max(1, (nbytes + _READ_CHUNK_BYTES - 1) // _READ_CHUNK_BYTES)
    )
    spans = [
        (
            nbytes * i // workers,
            nbytes * (i + 1) // workers - nbytes * i // workers,
        )
        for i in range(workers)
        if nbytes * (i + 1) // workers > nbytes * i // workers
    ]

    def _read_contiguous(span: tuple[int, int]) -> None:
        rel, length = span
        with open(shard_path, "rb", buffering=0) as wf:
            wf.seek(file_start + rel)
            done = 0
            while done < length:
                step = min(_SEQ_STEP_BYTES, length - done)
                got = wf.readinto(mv[rel + done : rel + done + step])
                if not got:
                    raise EOFError(
                        f"Unexpected EOF reading {name} at {rel + done}/{nbytes} "
                        f"from {shard_path}"
                    )
                done += got

    if len(spans) <= 1:
        for span in spans:
            _read_contiguous(span)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(spans)) as pool:
            for _ in pool.map(_read_contiguous, spans):
                pass


def read_tp_sharded_safetensors_rows(
    shard_path: str | os.PathLike[str],
    tensor_names: list[str],
    *,
    tp_size: int,
    tp_rank: int,
    streamer_attr: str,
):
    """Read only ``tp_rank``'s owned row slice for multi-100-GB Engram tables.

    ``EngramEmbedding._load_rows`` slices ``loaded_weight[row_start:row_end]`` on
    every TP rank. Streaming the whole ~98 GB tensor through RunAI on all 8 ranks
    at once requires >= 8 x 98 GB of ring buffers (exceeding single-host RAM) and
    amplifies storage I/O 8x. Reading each rank's exact byte range keeps peak RAM
    to ~12 GB/rank and total I/O to 1x across both ``gs://`` and local paths.
    """
    use_gcs = _is_gcs_uri(shard_path)
    gcs_uri = str(shard_path)
    blob = _open_gcs_blob(gcs_uri) if use_gcs else None
    if use_gcs:
        hdr_raw = _gcs_download_range(blob, 0, 8, "__header_len__", gcs_uri)
        (hdr_len,) = struct.unpack("<Q", hdr_raw)
        header = json.loads(
            _gcs_download_range(blob, 8, hdr_len, "__header__", gcs_uri)
        )
    else:
        with open(shard_path, "rb") as f:
            hdr_raw = f.read(8)
            if len(hdr_raw) != 8:
                raise ValueError(f"Truncated safetensors header in {shard_path}")
            (hdr_len,) = struct.unpack("<Q", hdr_raw)
            header = json.loads(f.read(hdr_len))
    data_base = 8 + hdr_len

    for name in tensor_names:
        t0 = perf_counter()
        meta = header[name]
        dtype = _SAFETENSORS_DTYPES[meta["dtype"]]
        shape = [int(d) for d in meta["shape"]]
        off0, off1 = (int(x) for x in meta["data_offsets"])
        num_rows = shape[0]
        row_elems = math.prod(shape[1:])
        row_bytes = row_elems * dtype.itemsize
        if off1 - off0 != num_rows * row_bytes:
            raise ValueError(
                f"Unexpected byte span for {name}: {off1 - off0} vs "
                f"{num_rows * row_bytes}"
            )
        row_start = num_rows * tp_rank // tp_size
        row_end = num_rows * (tp_rank + 1) // tp_size
        local_rows = row_end - row_start
        nbytes = local_rows * row_bytes
        file_start = data_base + off0 + row_start * row_bytes

        raw = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
        if use_gcs:
            _read_gcs_into(blob, file_start, raw, nbytes, name, gcs_uri)
        else:
            _read_local_into(shard_path, file_start, raw, nbytes, name)
        if tp_rank == 0:
            logger.info(
                "V4.1 TP rank 0/%d: read %s slice [%d:%d] (%.2f GiB) in %.2f s",
                tp_size,
                name,
                row_start,
                row_end,
                nbytes / (1 << 30),
                perf_counter() - t0,
            )

        shard = raw.view(dtype).view(local_rows, *shape[1:])
        if tp_size == 1:
            setattr(shard, streamer_attr, True)
            yield name, shard, nbytes
        else:
            yield (
                name,
                _ShardedRows(shard, num_rows, row_start, streamer_attr),
                nbytes,
            )
        del shard, raw


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
    """Keep local routed experts and all replicated/host-table parameters.

    This is a payload filter, not a filter on the already-materialized iterator.
    Host Engram tensors must stay: the current shared table loader has every TP
    rank fill its own row range of each checkpoint tensor. MTP is not loaded by
    the target model. Unknown names are retained rather than silently dropped.
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
    # Fail closed on an incompatible naming scheme, partial index or wrong
    # config instead of reporting a speedup caused by silently missing weights.
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


class DSV41RunaiModelStreamerLoader(RunaiModelStreamerLoader):
    """Reuse construction/postprocessing; specialize only primary weight I/O."""

    def __init__(self, load_config, model_config):
        super().__init__(load_config)
        self._primary_model_config = model_config

    def _get_weights_iterator(self, source):
        config = getattr(source.model_config, "hf_config", None)
        if (
            source.model_config is not self._primary_model_config
            or getattr(config, "model_type", None) != "deepseek_v41"
            or getattr(source.model_config, "is_draft_model", False)
            or source.model_or_path
            != getattr(
                self._primary_model_config,
                "model_weights",
                self._primary_model_config.model_path,
            )
            or source.revision != self._primary_model_config.revision
            or source.prefix
            or self.load_config.draft_model_idx is not None
        ):
            yield from super()._get_weights_iterator(source)
            return

        from runai_model_streamer import SafetensorsStreamer

        from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

        if (
            "tensor_names"
            not in inspect.signature(SafetensorsStreamer.stream_files).parameters
        ):
            raise RuntimeError(
                "V4.1 selective loading requires RunAI tensor_names support (tested with 0.17.0)"
            )
        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        metadata_dir = (
            ObjectStorageModel.get_path(folder) if is_runai_obj_uri(folder) else folder
        )
        index = json.loads(
            (Path(metadata_dir) / "model.safetensors.index.json").read_text()
        )
        parallel = get_parallel()
        plan = plan_expert_reads(
            index["weight_map"],
            num_layers=config.num_hidden_layers,
            num_experts=config.n_routed_experts,
            ep_size=parallel.moe_ep_size,
            ep_rank=parallel.moe_ep_rank,
        )
        files = [path for path in files if path.rsplit("/", 1)[-1] in plan.shard_names]
        if len(files) != len(plan.shard_names):
            raise ValueError(
                "RunAI shard list does not cover the selected checkpoint index"
            )
        logger.info(
            "V4.1 EP rank %d/%d: reading %d/%d tensors (%d local expert tensors) from %d shards",
            parallel.moe_ep_rank,
            parallel.moe_ep_size,
            len(plan.tensor_names),
            plan.total_tensors,
            plan.local_expert_tensors,
            len(files),
        )
        # RunAI distributed streaming broadcasts one common request sequence.
        # Its contract requires identical tensor_names on every rank. Our EP
        # sets differ, so each rank reads its own ranges independently. All ranks
        # still enter stream_files: even False can perform group setup in 0.17.
        # Keep owned=True (the default): loaders may retain CPU expert tensors.
        start, size, count = perf_counter(), 0, 0
        files_by_shard = {path.rsplit("/", 1)[-1]: path for path in files}
        engram_embed_names = (
            sorted(name for name in plan.tensor_names if _ENGRAM_EMBED.fullmatch(name))
            if (_is_gcs_uri(folder) or not is_runai_obj_uri(folder))
            else []
        )
        stream_names = plan.tensor_names.difference(engram_embed_names)
        stream_shards = {index["weight_map"][name] for name in stream_names}
        stream_files = [
            path for path in files if path.rsplit("/", 1)[-1] in stream_shards
        ]
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                stream_files,
                device="cpu",
                is_distributed=False,
                tensor_names=stream_names,
            )
            for name, tensor in streamer.get_tensors():
                size += tensor.numel() * tensor.element_size()
                count += 1
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, tensor

        by_shard: dict[str, list[str]] = {}
        for name in engram_embed_names:
            by_shard.setdefault(index["weight_map"][name], []).append(name)
        for shard_name, names in by_shard.items():
            for name, tensor, nbytes in read_tp_sharded_safetensors_rows(
                files_by_shard[shard_name],
                names,
                tp_size=parallel.tp_size,
                tp_rank=parallel.tp_rank,
                streamer_attr=RUNAI_STREAMER_TENSOR_ATTR,
            ):
                size += nbytes
                count += 1
                yield name, tensor
        logger.info(
            "V4.1 EP rank %d: consumed %d selected tensors, %.3f GB in %.3f s (includes parameter loading)",
            parallel.moe_ep_rank,
            count,
            size / 1e9,
            perf_counter() - start,
        )
