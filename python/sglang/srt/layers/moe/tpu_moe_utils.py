"""TPU-specific fused MoE helpers (mirrors rocm_moe_utils.py)."""

from __future__ import annotations

from typing import Optional

import torch

# SGLANG_TPU_INT4_MOE_MODE values; None = default W4A16 group products.
INT4_MOE_MODE_VMEM_BF16 = "vmem-bf16"
INT4_MOE_MODE_VMEM_FP8 = "vmem-fp8"


def resolve_int4_moe_mode() -> Optional[str]:
    """The INT4 MoE mode from the environment, validated.

    Returns:
        ``None`` for the default W4A16 path, else one of
        ``INT4_MOE_MODE_VMEM_BF16`` / ``INT4_MOE_MODE_VMEM_FP8``.

    Raises:
        ValueError: The environment names a mode that does not exist.
    """
    from sglang.srt.environ import envs

    mode = envs.SGLANG_TPU_INT4_MOE_MODE.get()
    if mode in (None, ""):
        return None
    if mode not in (INT4_MOE_MODE_VMEM_BF16, INT4_MOE_MODE_VMEM_FP8):
        raise ValueError(f"Unsupported TPU INT4 MoE mode: {mode!r}")
    return mode


def mask_tpu_moe_non_local_experts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    moe_ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate global expert IDs to this EP rank's local IDs and mask the rest.

    The dispatcher hands the TPU runner global expert IDs
    (skip_local_expert_mapping). The TPU Pallas fused-MoE kernel takes local
    expert IDs here, so shift IDs into this rank's local range and mark the
    rest -1 (dropped by the kernel) with zero routing weight.
    """
    local_ids = topk_ids - moe_ep_rank * num_local_experts
    valid = (local_ids >= 0) & (local_ids < num_local_experts)
    return (
        topk_weights * valid.to(topk_weights.dtype),
        torch.where(valid, local_ids, -1),
    )
