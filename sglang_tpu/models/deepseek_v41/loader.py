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
    is_engram_table,
    iter_row_tables,
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

    def _get_weights_iterator(self, source):
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
        rows = iter_row_tables(
            shard_of=shard_of,
            names=tables,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )
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
