"""TPU-specific fused MoE helpers (mirrors rocm_moe_utils.py)."""

from __future__ import annotations

import torch


def mask_tpu_moe_non_local_experts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    moe_ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate global expert IDs to this EP rank's local IDs and mask the rest.

    The dispatcher hands the TPU runner global expert IDs
    (skip_local_expert_mapping). The TPU Pallas fused-MoE kernel expects
    non-negative local expert IDs, so shift IDs into this rank's local range,
    route non-local slots to expert 0, and zero their routing weights.
    """
    local_ids = topk_ids - moe_ep_rank * num_local_experts
    valid = (local_ids >= 0) & (local_ids < num_local_experts)
    return (
        topk_weights * valid.to(topk_weights.dtype),
        torch.where(valid, local_ids, 0),
    )
