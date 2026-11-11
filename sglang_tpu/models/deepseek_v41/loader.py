# SPDX-License-Identifier: Apache-2.0
"""RunAI transport with EP-local routed expert reads for V4.1 Flash.

Forked from sgl-project/sglang-torchtpu#1744 (e0a43ce363, sglang_tpu/dsv41/loader.py).
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import torch

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader

logger = logging.getLogger(__name__)


_EXPERT = re.compile(
    r"^layers\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w[123]\.(weight|scale)$"
)
# Engram hash tables are ~100 GB ranges, far above any per-rank RunAI buffer.
# Every rank copies only its own row range, so they are mapped, not streamed.
_ENGRAM_TABLE = re.compile(r"^layers\.(0|[1-9]\d*)\.engram\.embed\.(weight|scale)$")

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


def mapped_safetensors(path: str, names: set[str]):
    """Yield (name, tensor) views over an mmap of ``path``; no bytes are read
    until a slice of the view is touched."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    for name in sorted(names):
        entry = header[name]
        start, end = entry["data_offsets"]
        raw = torch.from_numpy(data[start:end])
        yield name, raw.view(_SAFETENSORS_DTYPES[entry["dtype"]]).view(entry["shape"])


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
        weight_map = index["weight_map"]
        mapped = {n for n in plan.tensor_names if _ENGRAM_TABLE.fullmatch(n)}
        streamed = plan.tensor_names - mapped
        if mapped and is_runai_obj_uri(folder):
            raise ValueError("Engram tables need a local or mounted checkpoint path")
        by_name = {path.rsplit("/", 1)[-1]: path for path in files}
        if not plan.shard_names <= by_name.keys():
            raise ValueError(
                "RunAI shard list does not cover the selected checkpoint index"
            )
        stream_shards = {weight_map[n] for n in streamed}
        files = [by_name[s] for s in sorted(stream_shards)]
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
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                files,
                device="cpu",
                is_distributed=False,
                tensor_names=streamed,
            )
            for name, tensor in streamer.get_tensors():
                size += tensor.numel() * tensor.element_size()
                count += 1
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, tensor
        for shard in sorted({weight_map[n] for n in mapped}):
            names = {n for n in mapped if weight_map[n] == shard}
            for name, tensor in mapped_safetensors(by_name[shard], names):
                count += 1
                yield name, tensor
        logger.info(
            "V4.1 EP rank %d: consumed %d selected tensors, %.3f GB in %.3f s (includes parameter loading)",
            parallel.moe_ep_rank,
            count,
            size / 1e9,
            perf_counter() - start,
        )
