"""ViT precompile bucket ladder, importable before torch_tpu is.

The compile policy warms these patch counts; the server-args defaults read the
same ladder to cap image size, so the two can never disagree.
"""

from __future__ import annotations

import math

from sglang.srt.environ import envs

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


def env_patch_buckets(merge_size: int) -> list[int]:
    """The ladder the env vars select for an encoder with this merge size."""
    # Set but empty disables precompile: EnvTuple drops empty entries, so the
    # override is (), which validates trivially and warms nothing.
    if envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.is_set():
        buckets = sorted({int(x) for x in envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()})
        return _validate_patch_buckets(buckets, merge_size)
    return _compute_patch_counts(
        merge_size,
        max_patches=envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get(),
    )
