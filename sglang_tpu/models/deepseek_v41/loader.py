# SPDX-License-Identifier: Apache-2.0
"""Run:AI loader for DeepSeek-V4.1 Flash: EP-local experts, TP-local Engram rows.

Routed experts follow the generic EP-local Run:AI read
(``sglang_tpu.ep_weight_shard``). Engram tables leave the stream and are read
as this rank's row range (``engram_row_reader``). The bundled DSpark draft
streams only its ``mtp.*`` tensors (``_draft_weights``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch

from sglang_tpu._logging import get_logger
from sglang_tpu.ep_local_expert_reader import local_expert_id
from sglang_tpu.ep_weight_shard import INDEX_NAME, EPLocalRunaiModelStreamerLoader
from sglang_tpu.models.deepseek_v41.engram_row_reader import (
    engram_row_ranges,
    is_engram_table,
    iter_row_tables,
    shared_table_rows,
)

logger = get_logger(__name__)

# DeepseekV4ForCausalLM.load_weights skips every MTP tensor of the target model.
_MTP_PREFIX = "mtp."
_DSPARK_DRAFT_ARCH = "DeepseekV4ForCausalLMDSpark"


def is_bundled_dspark_draft(model_config) -> bool:
    """The DSpark head bundled in a V4-family checkpoint (core remaps its arch)."""
    return (
        model_config.is_draft_model
        and _DSPARK_DRAFT_ARCH in model_config.hf_config.architectures
    )


class DSV41RunaiModelStreamerLoader(EPLocalRunaiModelStreamerLoader):
    """EP-local Run:AI stream without MTP and Engram tables, then Engram rows."""

    _plans_without_expert_range = True
    # The model being loaded; its Engram layout sets each table's row ranges.
    _hf_config = None
    # This EP rank while the target loads under SGLANG_TPU_DSV41_FUSED_EP_LAYOUT.
    _layout_rank = None
    # The model being loaded; its shared Engram host tables take rows in place.
    _model = None

    def _get_all_weights(self, model_config, model):
        from sglang_tpu.models.deepseek_v41 import expert_layout

        layout = expert_layout.active_layout()
        target = (
            self.load_config.draft_model_idx is None and not model_config.is_draft_model
        )
        self._model = model
        if layout is not None and target:
            from sglang.srt.runtime_context import get_parallel

            tagged = expert_layout.tag_target_layers(model, layout)
            self._layout_rank = int(get_parallel().moe_ep_rank)
            logger.info(
                "V4.1 expert layout %s: %d MoE layers placed (ep rank %d)",
                expert_layout.layout_path(),
                tagged,
                self._layout_rank,
            )
        yield from super()._get_all_weights(model_config, model)

    def _plan_expert_fetch(self, shard_of):
        if self._layout_rank is None:
            return super()._plan_expert_fetch(shard_of)
        from sglang_tpu.ep_weight_shard import ExpertFetchPlan
        from sglang_tpu.models.deepseek_v41 import expert_layout

        layout = expert_layout.active_layout()
        kept, skipped, local = set(), set(), set()
        for name in shard_of:
            keep = expert_layout.keep_expert(
                name, layout, self._layout_rank, self._expert_range
            )
            if keep is False:
                skipped.add(name)
                continue
            kept.add(name)
            if keep:
                local.add(name)
        return ExpertFetchPlan(
            kept=frozenset(kept),
            skipped=frozenset(skipped),
            local_experts=frozenset(local),
        )

    def _get_weights_iterator(self, source):
        self._hf_config = source.model_config.hf_config
        if (
            is_bundled_dspark_draft(source.model_config)
            and self.load_config.draft_model_idx is None
        ):
            yield from self._draft_weights(source)
            return
        yield from super()._get_weights_iterator(source)

    def _shard_of_names(self, streamer, files: list[str]) -> dict[str, str]:
        shard_of = super()._shard_of_names(streamer, files)
        if shard_of is None:
            # A whole-shard stream would hand every rank each ~100 GB table.
            raise RuntimeError(
                "V4.1 Engram row reads need Run:AI tensor_names support "
                "(>= 0.17.0) and readable shard headers"
            )
        return shard_of

    def _split_kept(
        self, *, kept: frozenset[str], shard_of: Mapping[str, str]
    ) -> tuple[frozenset[str], Iterator[tuple[str, torch.Tensor]]]:
        from sglang.srt.runtime_context import get_parallel

        tables = frozenset(name for name in kept if is_engram_table(name))
        streamed = frozenset(
            name
            for name in kept
            if name not in tables and not name.startswith(_MTP_PREFIX)
        )
        parallel = get_parallel()
        ranges = engram_row_ranges(
            tables,
            self._hf_config,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )
        shared = None if self._model is None else shared_table_rows(self._model, ranges)
        rows = iter_row_tables(shard_of=shard_of, ranges=ranges, shared=shared)
        return streamed, rows

    def _draft_weights(self, source):
        """The bundled DSpark draft reads only ``mtp.*`` tensors (its loader drops
        the rest); streaming the whole checkpoint needs a RunAI buffer as large
        as the ~100 GB Engram tables."""
        from runai_model_streamer import SafetensorsStreamer

        from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
        from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        # An object-storage checkpoint's index sits in the local metadata cache, as
        # the core Run:AI loader reads it.
        if is_runai_obj_uri(folder):
            folder = ObjectStorageModel.get_path(folder)
        weight_map = json.loads((Path(folder) / INDEX_NAME).read_text())["weight_map"]
        names = {n for n in weight_map if n.startswith(_MTP_PREFIX)}
        if not names:
            raise ValueError(
                "DSpark draft: the checkpoint index lists no mtp.* tensors"
            )
        names, local_experts = _draft_ep_local_names(
            names, config=source.model_config.hf_config
        )
        by_name = {path.rsplit("/", 1)[-1]: path for path in files}
        shards = sorted({by_name[weight_map[n]] for n in names})
        logger.info(
            "V4.1 DSpark draft: reading %d mtp tensors (%d local expert tensors) "
            "from %d shards",
            len(names),
            local_experts,
            len(shards),
        )
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                shards, device="cpu", is_distributed=False, tensor_names=names
            )
            for name, tensor in streamer.get_tensors():
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield source.prefix + name, tensor


def _draft_ep_local_names(names: set[str], *, config) -> tuple[set[str], int]:
    """Drop the draft's routed experts outside this EP rank's range.

    The draft MoE shards over the target's EP group (TpuDsparkDraftWorker),
    so a rank keeps experts [rank * E / ep, (rank + 1) * E / ep), as the target's
    EP-local read (sglang_tpu.ep_weight_shard) does; vllm-torchtpu filters its
    draft the same way.
    """
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    ep_size, ep_rank = parallel.moe_ep_size, parallel.moe_ep_rank
    num_experts = int(
        getattr(config, "dspark_n_routed_experts", 0) or config.n_routed_experts
    )
    if ep_size <= 1:
        expert_range = (0, num_experts)
    else:
        if num_experts % ep_size:
            raise ValueError(
                f"DSpark draft: {num_experts} routed experts do not divide EP {ep_size}"
            )
        local = num_experts // ep_size
        expert_range = (ep_rank * local, (ep_rank + 1) * local)
    kept, experts = set(), 0
    for name in names:
        expert = local_expert_id(name, expert_range)
        if expert == -1:
            continue
        experts += expert is not None
        kept.add(name)
    return kept, experts
