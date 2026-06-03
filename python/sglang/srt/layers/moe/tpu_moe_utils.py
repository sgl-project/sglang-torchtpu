"""TPU-specific fused MoE helpers (mirrors rocm_moe_utils.py)."""

from __future__ import annotations

import torch


def mask_tpu_moe_non_local_experts(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask EP non-local expert slots before the TPU Pallas MoE kernel.

    StandardDispatcher remaps experts not owned by the current EP rank to -1.
    The TPU Pallas fused-MoE kernel expects non-negative local expert IDs, so
    route those slots to expert 0 and zero their routing weights.
    """
    valid = topk_ids >= 0
    return (
        topk_weights * valid.to(topk_weights.dtype),
        torch.where(valid, topk_ids, torch.zeros_like(topk_ids)),
    )
