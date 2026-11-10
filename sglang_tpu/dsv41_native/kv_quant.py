# SPDX-License-Identifier: Apache-2.0
"""KV-cache quantizers that store codes and scales instead of a dequantized copy.

Each `quantize_*` follows the matching in-place quantizer of kernel_ref.py step by step (same amax
floor, same scale rounding, same clamp and rounding of the codes) but returns the codes in an fp8
container and the scale per block. `dequant` multiplies them back; the product is the value the
in-place form writes, so a cache read gives the same bf16 tensor as before, bit for bit.

- fp8 window KV: e4m3 codes, block 32, UE8M0 scale (act_quant with scale_fmt="ue8m0").
- fp4 index K: e2m1 codes held in e4m3 (every e2m1 value is exact in e4m3), block 32, E8M0 scale.
- fp4 compressed KV: e2m1 codes held in e4m3, block 16, E4M3 scale.
E8M0 scales are raw exponent bytes (uint8), decoded by kernel_ref.scale_f32.
"""

from __future__ import annotations

import torch

from sglang_tpu.dsv41_native.kernel_ref import (
    FP4_MAX,
    FP8_MAX,
    _blocks,
    _fp4_round,
    _pow2_ceil,
    scale_f32,
)

CODE_DTYPE = torch.float8_e4m3fn


def _e8m0_byte(s: torch.Tensor) -> torch.Tensor:
    """Power-of-two fp32 scale -> its biased exponent byte (the E8M0 encoding)."""
    return ((s.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def quantize_fp8_ue8m0(
    x: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """act_quant(x, block_size, "ue8m0", inplace=True), as (e4m3 codes [..., N], uint8 [..., N/bs])."""
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / FP8_MAX)
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(CODE_DTYPE)
    return q.flatten(-2), _e8m0_byte(s.squeeze(-1))


def quantize_fp4_e8m0(
    x: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp4_act_quant(x, block_size, True) with E8M0 scales, as (codes in e4m3, uint8 scales)."""
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * 2**-126)
    s = _pow2_ceil(amax / FP4_MAX)
    q = _fp4_round((xb / s).clamp(-FP4_MAX, FP4_MAX))
    return q.to(CODE_DTYPE).flatten(-2), _e8m0_byte(s.squeeze(-1))


def quantize_fp4_e4m3(
    x: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp4_act_quant(x, block_size, True, scale_dtype=e4m3), as (codes in e4m3, e4m3 scales)."""
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * 2**-9)
    s = (amax / FP4_MAX).to(torch.float8_e4m3fn)
    q = _fp4_round((xb / s.float()).clamp(-FP4_MAX, FP4_MAX))
    return q.to(CODE_DTYPE).flatten(-2), s.squeeze(-1)


def dequant(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """codes [..., N] and scales [..., N / block] -> dtype, the value the in-place quantizer writes.

    Computed in `dtype` directly: an e4m3 or e2m1 code times a power-of-two or e4m3 scale has at
    most 5 significant bits and an fp32 exponent, so the product is exact in bf16 and no fp32 copy
    of the gathered rows is made (decode reads [B, context / ratio, d] rows per indexer)."""
    block_size = q.size(-1) // s.size(-1)
    v = q.to(dtype).unflatten(-1, (-1, block_size)) * scale_f32(s).to(dtype).unsqueeze(
        -1
    )
    return v.flatten(-2)
