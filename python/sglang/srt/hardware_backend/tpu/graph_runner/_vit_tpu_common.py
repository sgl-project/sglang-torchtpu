"""Model-agnostic helpers shared by the TPU ViT compile runners.

Extracted from ``vit_tpu_compile_runner`` so the Qwen3-VL runner and the MoonViT
(Kimi-K2.5) runner share one copy of the patch-bucket / padding / segment-id
logic. Nothing here is model-specific.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Unlike the upstream CUDA and NPU runners, TPU precompiles the vision encoder
# for fixed patch counts. Real images are often non-square, so runtime pads them
# to the nearest bucket. MMMU reached 3400 patches; 4900 = 70^2 leaves headroom
# without warming a huge grid.
_DEFAULT_MAX_PATCHES = 4900


def _compute_patch_counts(
    merge_size: int, min_patches: int = 196, max_patches: int = _DEFAULT_MAX_PATCHES
) -> list[int]:
    """Compute square-image patch-count buckets aligned to merger windows."""
    counts = []
    min_grid = max(
        merge_size, math.ceil(math.sqrt(min_patches) / merge_size) * merge_size
    )
    grid_dim = min_grid
    while True:
        patch_count = grid_dim * grid_dim
        if patch_count > max_patches:
            break
        counts.append(patch_count)
        grid_dim += merge_size
    return counts


def _validate_patch_buckets(buckets: list[int], merge_size: int) -> list[int]:
    merge_sq = merge_size**2
    for bucket in buckets:
        grid = math.isqrt(bucket)
        if bucket <= 0 or bucket % merge_sq != 0 or grid * grid != bucket:
            raise ValueError(
                "SGLANG_TPU_VIT_PRECOMPILE_SIZES entries must be positive "
                f"perfect squares divisible by {merge_sq}; got {bucket}."
            )
    return buckets


def _cu_seqlens_to_segment_ids(
    cu_seqlens: torch.Tensor,
    total_patches: int,
    device: torch.device,
) -> torch.Tensor:
    """Build segment IDs on the host, then transfer them to the patch device."""
    if cu_seqlens.device.type != "cpu":
        raise ValueError("TPU vision cu_seqlens must remain on CPU")

    cu = cu_seqlens.to(dtype=torch.int64)
    if cu.ndim != 1 or cu.numel() == 0 or int(cu[0]) != 0:
        raise ValueError("cu_seqlens must be a non-empty 1D tensor starting at zero")
    if int(cu[-1]) != total_patches:
        raise ValueError(
            f"cu_seqlens[-1]={int(cu[-1])} != total_patches={total_patches}"
        )

    lengths = cu[1:] - cu[:-1]
    if bool(torch.any(lengths < 0)):
        raise ValueError("cu_seqlens must be nondecreasing")
    segment_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype=torch.int32), lengths
    ).unsqueeze(0)
    return segment_ids.to(device=device, non_blocking=True)


@dataclass(frozen=True)
class VisionPaddingContext:
    real_seq_len: int
    bucket_len: int | None
    padded: bool
    merge_sq: int
