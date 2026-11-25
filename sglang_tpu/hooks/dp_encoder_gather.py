# SPDX-License-Identifier: Apache-2.0
"""DP-encoder outputs reach every attention-TP rank through per-rank broadcasts.

broadcast_rank_embeddings
  Why: with --mm-enable-dp-encoder, single-image, multi-image, and
       presharded-video forwards gather the per-rank ViT outputs across
       attention-TP ranks (gather_dp_encoder_rank_embeddings). On tpu7x TP8
       (GLM-5.3-Flash VL) the multi-image padded all_gather never returned on
       any rank on the first call, and raw single-image broadcasts leave
       view-backed or DEFER_AND_FUSE inputs unisolated.
  How: AROUND sglang.srt.multimodal.mm_utils.gather_dp_encoder_rank_embeddings,
       without calling the original: validate shapes and lengths on the host,
       skip empty/zero-width/TP=1 cases, materialize each sending rank's rows
       into a standalone contiguous base buffer via single-tensor torch.cat,
       drain pending TPU stream ops via tpu_sync(..., wait=False), and issue one
       attn_tp_group.broadcast(src=rank) per non-empty rank under
       EagerMode.DEFER_NEVER.
  Upstream PR or issue: not filed yet (TorchTPU eager collectives on an
       attention-TP group under OpSplitMode::kNone).
  Remove when: ProcessGroupTpu::_allgather_base splits and materializes constant,
       metadata-only view, and stream-deferred inputs before collective
       compilation.
"""

from __future__ import annotations

import math
from typing import Any, Callable, TypeVar

import torch

from sglang.srt.runtime_context import get_parallel

_T = TypeVar("_T")


def _normalize_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.resolve_conj().resolve_neg().contiguous()


def _materialize_standalone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    src = _normalize_contiguous(tensor)
    # OpName::kCatOut allocates a fresh non-constant, non-metadata base buffer
    # with dimensions() == src.sizes(); clone() shares the source storage base
    # on reinterpret views, and _is_view() is False inside inference_mode().
    return torch.cat([src], dim=0)


def _run_isolated_tpu_collective(
    tensor: torch.Tensor,
    fn: Callable[[], _T],
    *,
    sync_operand: bool = True,
) -> _T:
    if tensor.device.type != "tpu":
        return fn()

    from torch_tpu._internal import execution_mode
    from torch_tpu._internal.sync.sync import synchronize as tpu_sync

    # Start materializing the collective operand and drain pending stream ops so
    # OpSplitMode::kNone cannot fuse rank-local ops into the collective HLO.
    if sync_operand:
        tpu_sync(tensor, wait=False)
    tpu_sync(None, wait=False)
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):
        return fn()


def _validate_dp_encoder_gather_inputs(
    image_embeds_local: torch.Tensor,
    grouped_output_lengths: list[int],
    *,
    tp_size: int,
    local_rank: int,
) -> None:
    if len(grouped_output_lengths) != tp_size:
        raise ValueError(
            f"Expected len(grouped_output_lengths) == attn_tp_size ({tp_size}), "
            f"got {len(grouped_output_lengths)}"
        )
    if not (0 <= local_rank < tp_size):
        raise ValueError(
            f"attn_tp_rank ({local_rank}) out of range for attn_tp_size ({tp_size})"
        )
    expected_local_len = grouped_output_lengths[local_rank]
    actual_local_len = image_embeds_local.shape[0]
    if actual_local_len != expected_local_len:
        raise ValueError(
            f"Rank {local_rank} local embedding row count ({actual_local_len}) "
            f"!= grouped_output_lengths[{local_rank}] ({expected_local_len})"
        )


def _isolated_tpu_broadcast_one(
    group: Any,
    *,
    local_rank: int,
    src: int,
    image_embeds_local: torch.Tensor,
    output_length: int,
) -> torch.Tensor:
    tail_shape = image_embeds_local.shape[1:]
    if output_length == 0 or math.prod(tail_shape) == 0:
        return image_embeds_local.new_empty((output_length, *tail_shape))
    if local_rank == src:
        rank_embed = _materialize_standalone_tensor(image_embeds_local[:output_length])
    else:
        rank_embed = image_embeds_local.new_empty((output_length, *tail_shape))
    _run_isolated_tpu_collective(
        rank_embed,
        lambda: group.broadcast(rank_embed, src=src),
        sync_operand=(local_rank == src),
    )
    return rank_embed


def _broadcast_rank_embeddings(
    *,
    group: Any,
    local_rank: int,
    image_embeds_local: torch.Tensor,
    grouped_output_lengths: list[int],
) -> list[torch.Tensor]:
    return [
        _isolated_tpu_broadcast_one(
            group,
            local_rank=local_rank,
            src=rank,
            image_embeds_local=image_embeds_local,
            output_length=output_length,
        )
        for rank, output_length in enumerate(grouped_output_lengths)
    ]


def broadcast_rank_embeddings(
    original: Callable,
    image_embeds_local: torch.Tensor,
    grouped_output_lengths: list[int],
) -> list[torch.Tensor]:
    del original
    if image_embeds_local.ndim < 1:
        raise ValueError(
            "Expected image_embeds_local to have at least 1 dimension, "
            f"got shape {tuple(image_embeds_local.shape)}"
        )
    if any(length < 0 for length in grouped_output_lengths):
        raise ValueError(
            f"grouped_output_lengths must be non-negative, got {grouped_output_lengths}"
        )
    if not grouped_output_lengths:
        return []

    parallel = get_parallel()
    tp_size = parallel.attn_tp_size
    local_rank = parallel.attn_tp_rank
    _validate_dp_encoder_gather_inputs(
        image_embeds_local,
        grouped_output_lengths,
        tp_size=tp_size,
        local_rank=local_rank,
    )

    tail_shape = image_embeds_local.shape[1:]
    if max(grouped_output_lengths) == 0 or math.prod(tail_shape) == 0:
        return [
            image_embeds_local.new_empty((output_length, *tail_shape))
            for output_length in grouped_output_lengths
        ]
    if tp_size == 1:
        return [image_embeds_local]

    return _broadcast_rank_embeddings(
        group=parallel.attn_tp_group,
        local_rank=local_rank,
        image_embeds_local=image_embeds_local,
        grouped_output_lengths=grouped_output_lengths,
    )
