# SPDX-License-Identifier: Apache-2.0
"""NVFP4 expert requantization to wider scale blocks (pure torch, any device)."""

from __future__ import annotations

import torch

# Served requant blocks: the decode megakernel's and fused-RS's W4A16 block-scaled arms.
QBLOCK_SIZES = (256,)
NVFP4_SERVED_GROUP = 16
_E2M1_MAX = 6.0
# e2m1 magnitude midpoints; True where a tie rounds up (to the even code).
_E2M1_MIDPOINTS = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)


def _e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    exp = ((codes >> 1) & 3).float()
    mant = (codes & 1).float()
    mag = torch.where(exp == 0, 0.5 * mant, (1.0 + 0.5 * mant) * torch.exp2(exp - 1.0))
    return torch.where(codes >= 8, -mag, mag)


def requant_nvfp4_blocks(
    packed: torch.Tensor, scale: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused-RS NVFP4 [E, K, N/2] uint8 (low nibble column 2m) + [E, K/16, N] scales -> codes
    re-rounded to one bf16 scale (amax / 6) per block K rows, and those scales [E, K/block, N].
    """
    e, k, half_n = packed.shape
    n = 2 * half_n
    if k % block or block % NVFP4_SERVED_GROUP:
        raise ValueError(f"cannot requantize K={k} to blocks of {block}")
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(e, k, n)
    w = _e2m1_values(codes) * scale.float().repeat_interleave(
        k // scale.shape[1], dim=1
    )
    wb = w.reshape(e, k // block, block, n)
    sb = (wb.abs().amax(dim=2) / _E2M1_MAX).to(torch.bfloat16)
    # |w| against midpoint x scale: both products are exact in f32 (few significant bits), so
    # ties resolve exactly where a TPU division would round them away.
    sbf = torch.where(sb > 0, sb, 1.0).float()[:, :, None]
    a = wb.abs()
    idx = torch.zeros_like(a, dtype=torch.uint8)
    for mid, tie_up in _E2M1_MIDPOINTS:
        idx += (a >= mid * sbf if tie_up else a > mid * sbf).to(torch.uint8)
    out = (idx | (wb < 0).to(torch.uint8) << 3).reshape(e, k, half_n, 2)
    return out[..., 0] | (out[..., 1] << 4), sb
