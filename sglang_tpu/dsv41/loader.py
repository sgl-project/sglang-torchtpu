# SPDX-License-Identifier: Apache-2.0
"""RunAI transport with EP-local routed expert reads for V4.1 Flash."""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader

logger = logging.getLogger(__name__)


_EXPERT = re.compile(
    r"^layers\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w[123]\.(weight|scale)$"
)


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
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                files,
                device="cpu",
                is_distributed=False,
                tensor_names=plan.tensor_names,
            )
            for name, tensor in streamer.get_tensors():
                size += tensor.numel() * tensor.element_size()
                count += 1
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, tensor
        logger.info(
            "V4.1 EP rank %d: consumed %d selected tensors, %.3f GB in %.3f s (includes parameter loading)",
            parallel.moe_ep_rank,
            count,
            size / 1e9,
            perf_counter() - start,
        )
