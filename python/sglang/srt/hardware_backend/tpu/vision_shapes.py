# SPDX-License-Identifier: Apache-2.0
"""Patch-count bucketing for vision towers that run eager on TPU.

A compiled tower pads to the precompiled ladder in
``graph_runner/vit_tpu_compile_runner.py``. An eager one has no ladder, and
eager torch_tpu still traces per input shape, so a tower whose patch count
follows the image aspect ratio pays a compile per distinct image. These helpers
give it the same bounded set of shapes without a compiled boundary.
"""

from __future__ import annotations

import torch


def patch_count_buckets(
    max_patches: int, *, align: int, num_buckets: int = 4
) -> list[int]:
    """Ascending ladder of at most ``num_buckets`` counts ending at ``max_patches``.

    ``align`` is the tower's pooling window, so every bucket stays a whole
    number of pooled tokens. Worst-case padding waste is one ladder step.
    """
    if max_patches <= 0 or align <= 0 or num_buckets <= 0:
        raise ValueError(
            f"max_patches, align and num_buckets must be positive; got "
            f"{max_patches}, {align}, {num_buckets}"
        )
    if max_patches % align:
        raise ValueError(f"max_patches {max_patches} is not a multiple of {align}")
    step = max(align, -(-max_patches // num_buckets // align) * align)
    buckets = list(range(step, max_patches + 1, step))
    if buckets[-1] != max_patches:
        buckets.append(max_patches)
    return buckets


def pad_patches_to_bucket(
    pixel_values: torch.Tensor,
    position_ids: torch.Tensor,
    buckets: list[int],
    *,
    pad_position: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the patch axis of ``[batch, patches, ...]`` up to the smallest fitting bucket.

    Padded positions get ``pad_position``, the sentinel the tower already uses to
    mask padding, so the encoder output is unchanged. A count above the ladder is
    returned untouched and traces at its own shape.
    """
    patches = pixel_values.shape[-2]
    bucket = next((b for b in buckets if b >= patches), None)
    if bucket is None or bucket == patches:
        return pixel_values, position_ids
    pad = bucket - patches
    pixel_values = torch.nn.functional.pad(pixel_values, (0, 0, 0, pad))
    position_ids = torch.nn.functional.pad(
        position_ids, (0, 0, 0, pad), value=pad_position
    )
    return pixel_values, position_ids
