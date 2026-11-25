# SPDX-License-Identifier: Apache-2.0
"""DP-encoder outputs reach every attention-TP rank via an isolated padded all-gather.

gather_rank_embeddings_tpu
  Why: SGLang-TorchTPU runs with TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS
       set to false (OpSplitMode::kNone in ProcessGroupTpu), and
       ProcessGroupTpu::_allgather_base also inlines IsConstantOp inputs
       (OpName::kEmpty / torch.empty -> BuildFillUninitialized) even when
       collective splitting is enabled. In
       _pad_mrope_vision_embeddings_for_tp_gather, full ranks pass a materialized
       buffer (1-arg all_gather), partially filled ranks pass torch.empty plus
       slice copy_ (OpName::kEmpty + metadata-only OpName::kAsStridedInverse,
       fusing constant(nan) + dynamic_update_slice + all_gather), and empty ranks
       pass torch.empty (0-arg all_gather(broadcast(constant(nan)))). XLA's
       AlgebraicSimplifierVisitor::HandleAllGather (MatchUniformCollectiveOperand)
       eliminates all_gather on a constant broadcast so empty ranks return
       immediately without entering the ICI collective while non-empty ranks hang
       in tt_jit_distributed.all_gather_into_tensor; slice/reshape views
       (including views created inside torch.inference_mode(), aten._unsafe_view,
       and squeeze_(), where _is_view() is False) or pending stream ops
       (DEFER_AND_FUSE) likewise fuse rank-local ops into _allgather_base and
       diverge XLA collective executables across ranks.
  How: AROUND sglang.srt.multimodal.mm_utils.gather_dp_encoder_rank_embeddings:
       validate shapes and lengths on the host; broadcast directly from the sole
       owner when only one rank produced rows; otherwise pad empty ranks with
       new_zeros (OpName::kNewZeros, non-constant, non-metadata), partially
       filled ranks with F.pad (OpName::kConstantPadNd, non-constant,
       non-metadata), and full ranks unconditionally via single-tensor torch.cat
       (OpName::kCatOut) so every rank holds a fresh standalone contiguous base
       buffer with matching dimensions; start materializing the operand and
       drain pending stream ops via tpu_sync(..., wait=False); and execute
       attn_tp_group.all_gather under EagerMode.DEFER_NEVER. Setting
       SGLANG_TPU_DP_ENCODER_GATHER_MODE=broadcast selects isolated per-rank
       broadcasts over non-empty ranks as an opt-in fallback.
  Upstream PR or issue: sgl-project/sglang-torchtpu#2225 (TorchTPU eager
       _allgather_base collective input splitting and constant materialization).
  Remove when: ProcessGroupTpu::_allgather_base splits and materializes constant,
       metadata-only view, and stream-deferred inputs into standalone buffers
       before collective compilation under
       TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false.
"""

from __future__ import annotations

import math
from typing import Any, Callable, TypeVar

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_parallel
from sglang_tpu.envs import envs as tpu_envs

_T = TypeVar("_T")

_MODE_ALL_GATHER = "all_gather"
_MODE_BROADCAST = "broadcast"
_VALID_MODES = (_MODE_ALL_GATHER, _MODE_BROADCAST)


def _resolve_gather_mode() -> str:
    raw = tpu_envs.SGLANG_TPU_DP_ENCODER_GATHER_MODE.get()
    mode = (raw or "").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unsupported SGLANG_TPU_DP_ENCODER_GATHER_MODE={raw!r}; "
            f"expected one of {_VALID_MODES}"
        )
    return mode


def _normalize_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.resolve_conj().resolve_neg().contiguous()


def _materialize_standalone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    src = _normalize_contiguous(tensor)
    # OpName::kCatOut allocates a fresh non-constant, non-metadata base buffer
    # with dimensions() == src.sizes(); clone() shares the source storage base
    # on reinterpret views, and _is_view() is False for views created inside
    # torch.inference_mode(), aten._unsafe_view, or squeeze_().
    return torch.cat([src], dim=0)


def _pad_dp_encoder_embeddings_tpu(
    image_embeds_local: torch.Tensor, max_len_per_rank: int
) -> torch.Tensor:
    if image_embeds_local.ndim < 1:
        raise ValueError(
            "Expected image_embeds_local to have at least 1 dimension, "
            f"got shape {tuple(image_embeds_local.shape)}"
        )

    tail_shape = image_embeds_local.shape[1:]
    if max_len_per_rank <= 0 or math.prod(tail_shape) == 0:
        return image_embeds_local.new_zeros((max(0, max_len_per_rank), *tail_shape))

    current_len = image_embeds_local.shape[0]
    if current_len == 0:
        # OpName::kNewZeros is non-constant and non-metadata; torch.empty inlines
        # BuildFillUninitialized into _allgather_base as a 0-parameter HLO.
        return image_embeds_local.new_zeros((max_len_per_rank, *tail_shape))

    if current_len < max_len_per_rank:
        src = _normalize_contiguous(image_embeds_local)
        # OpName::kConstantPadNd is non-constant and non-metadata; slice copy_
        # into torch.empty/zeros leaves kAsStridedInverse deferred.
        pad = (0, 0) * (src.ndim - 1) + (0, max_len_per_rank - current_len)
        return F.pad(src, pad)

    full = (
        image_embeds_local[:max_len_per_rank]
        if current_len > max_len_per_rank
        else image_embeds_local
    )
    return _materialize_standalone_tensor(full)


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


def _isolated_tpu_all_gather(group: Any, padded: torch.Tensor) -> torch.Tensor:
    return _run_isolated_tpu_collective(padded, lambda: group.all_gather(padded, dim=0))


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


def broadcast_dp_encoder_embedding_tpu(
    image_embeds_local: torch.Tensor,
    *,
    output_length: int,
    src: int,
) -> torch.Tensor:
    if image_embeds_local.ndim < 1:
        raise ValueError(
            "Expected image_embeds_local to have at least 1 dimension, "
            f"got shape {tuple(image_embeds_local.shape)}"
        )
    if output_length < 0:
        raise ValueError(f"output_length must be non-negative, got {output_length}")

    parallel = get_parallel()
    tp_size = parallel.attn_tp_size
    local_rank = parallel.attn_tp_rank
    if not (0 <= src < tp_size):
        raise ValueError(f"src ({src}) out of range for attn_tp_size ({tp_size})")
    if not (0 <= local_rank < tp_size):
        raise ValueError(
            f"attn_tp_rank ({local_rank}) out of range for attn_tp_size ({tp_size})"
        )
    if local_rank == src and image_embeds_local.shape[0] != output_length:
        actual_local_len = image_embeds_local.shape[0]
        raise ValueError(
            f"Rank {local_rank} local embedding row count ({actual_local_len}) "
            f"does not match output_length ({output_length})"
        )

    tail_shape = image_embeds_local.shape[1:]
    if output_length == 0 or math.prod(tail_shape) == 0:
        return image_embeds_local.new_empty((output_length, *tail_shape))
    if tp_size == 1:
        return image_embeds_local

    return _isolated_tpu_broadcast_one(
        parallel.attn_tp_group,
        local_rank=local_rank,
        src=src,
        image_embeds_local=image_embeds_local,
        output_length=output_length,
    )


def gather_dp_encoder_rank_embeddings_tpu(
    image_embeds_local: torch.Tensor,
    grouped_output_lengths: list[int],
) -> list[torch.Tensor]:
    mode = _resolve_gather_mode()
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
    max_len_per_rank = max(grouped_output_lengths)
    if max_len_per_rank == 0 or math.prod(tail_shape) == 0:
        return [
            image_embeds_local.new_empty((output_length, *tail_shape))
            for output_length in grouped_output_lengths
        ]
    if tp_size == 1:
        return [image_embeds_local]

    group = parallel.attn_tp_group
    non_empty_ranks = [
        rank
        for rank, output_length in enumerate(grouped_output_lengths)
        if output_length > 0
    ]
    if len(non_empty_ranks) == 1:
        owner_rank = non_empty_ranks[0]
        owner_embed = _isolated_tpu_broadcast_one(
            group,
            local_rank=local_rank,
            src=owner_rank,
            image_embeds_local=image_embeds_local,
            output_length=grouped_output_lengths[owner_rank],
        )
        empty = (0, *tail_shape)
        return [
            owner_embed if rank == owner_rank else image_embeds_local.new_empty(empty)
            for rank in range(tp_size)
        ]

    if mode == _MODE_BROADCAST:
        return _broadcast_rank_embeddings(
            group=group,
            local_rank=local_rank,
            image_embeds_local=image_embeds_local,
            grouped_output_lengths=grouped_output_lengths,
        )

    padded = _pad_dp_encoder_embeddings_tpu(image_embeds_local, max_len_per_rank)
    gathered = _isolated_tpu_all_gather(group, padded)

    rank_embeddings = list[torch.Tensor]()
    for rank, output_length in enumerate(grouped_output_lengths):
        if output_length == 0:
            rank_embeddings.append(image_embeds_local.new_empty((0, *tail_shape)))
        else:
            start_idx = rank * max_len_per_rank
            rank_embeddings.append(gathered[start_idx : start_idx + output_length])
    return rank_embeddings


def gather_rank_embeddings_tpu(
    original: Callable,
    image_embeds_local: torch.Tensor,
    grouped_output_lengths: list[int],
) -> list[torch.Tensor]:
    del original
    return gather_dp_encoder_rank_embeddings_tpu(
        image_embeds_local, grouped_output_lengths
    )
