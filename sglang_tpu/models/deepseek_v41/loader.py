# SPDX-License-Identifier: Apache-2.0
"""Run:AI loader for DeepSeek-V4.1 Flash: EP-local experts, TP-local Engram rows.

Routed experts follow the generic EP-local Run:AI read
(``sglang_tpu.ep_weight_shard``). Engram tables leave the stream and are read
as this rank's row range (``engram_row_reader``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch

from sglang_tpu.ep_weight_shard import EPLocalRunaiModelStreamerLoader
from sglang_tpu.models.deepseek_v41.engram_row_reader import (
    is_engram_table,
    is_gcs_uri,
    iter_row_tables,
)

# DeepseekV4ForCausalLM.load_weights skips every MTP tensor of the target model.
_MTP_PREFIX = "mtp."


class DSV41RunaiModelStreamerLoader(EPLocalRunaiModelStreamerLoader):
    """EP-local Run:AI stream without MTP and Engram tables, then Engram rows."""

    _plans_without_expert_range = True

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
        from sglang.srt.utils.runai_utils import is_runai_obj_uri

        tables = frozenset(name for name in kept if is_engram_table(name))
        if any(
            is_runai_obj_uri(shard_of[name]) and not is_gcs_uri(shard_of[name])
            for name in tables
        ):
            raise ValueError(
                "V4.1 Engram row reads need a local, mounted, or gs:// checkpoint path"
            )
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
