"""The MoE expert-parallel rank as a rank-lifted trace input, and the expert
mask that uses it."""

from typing import Optional

import torch

from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel


def traced_moe_ep_rank() -> Optional[torch.Tensor]:
    """This rank's MoE EP index as a traced tensor, or None if it is not lifted.

    Rank lifting requires the TP group to span the world, so the traced input
    is the TP rank. Within a TP group, ranks are laid out as
    ``(moe_dp_rank * moe_ep_size + moe_ep_rank) * moe_tp_size + moe_tp_rank``.
    """
    # Probe before get_parallel(): untraced callers may not have published it.
    # Every rank is rank 0 of a size-1 group, so the probe never fails the rank
    # check and is None exactly when no rank input is lifted.
    if current_platform.get_traced_rank(0, group_size=1) is None:
        return None
    parallel = get_parallel()
    if parallel.moe_ep_size == 1:
        return None
    tp_rank = current_platform.get_traced_rank(
        parallel.tp_rank, group_size=parallel.tp_size
    )
    return tp_rank // parallel.moe_tp_size % parallel.moe_ep_size


def mask_tpu_moe_non_local_experts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    moe_ep_rank: int,
    *,
    traced_ep_rank: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate global expert IDs to this EP rank's local IDs and mask the rest.

    The dispatcher hands the TPU runner global expert IDs
    (skip_local_expert_mapping). The TPU Pallas fused-MoE kernel takes local
    expert IDs here, so shift IDs into this rank's local range and mark the
    rest -1 (dropped by the kernel) with zero routing weight. Inside a
    rank-lifted trace, pass ``traced_ep_rank`` (from ``traced_moe_ep_rank``)
    so the shift follows the rank input instead of the tracing rank's int.
    """
    rank = moe_ep_rank if traced_ep_rank is None else traced_ep_rank
    local_ids = topk_ids - rank * num_local_experts
    valid = (local_ids >= 0) & (local_ids < num_local_experts)
    return (
        topk_weights * valid.to(topk_weights.dtype),
        torch.where(valid, local_ids, -1),
    )
