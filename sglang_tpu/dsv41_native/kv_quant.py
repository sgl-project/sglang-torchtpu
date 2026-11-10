# SPDX-License-Identifier: Apache-2.0
"""KV-cache quantizers that store fp8 codes and scales instead of a dequantized copy.

- Window KV: e4m3 codes, block 32, UE8M0 scale. Same steps as the reference's
  act_quant(scale_fmt="ue8m0") on the window KV, so a read gives the reference's bf16 bit for bit.
- Index K: e4m3 codes, one UE8M0 scale per 128-dim vector, as the production TPU index cache
  (kernels/deepseek_v41/index_selection/fp8/cache.py quantize_index_rows).
- Compressed (CSA) KV: the production TPU layout (dsv4/compress_and_store
  quantize_fp8_lane_periodic). The 448 nope dims are e4m3 codes with one UE8M0 scale per lane of
  a 64-lane period (group j holds dims j, j + 64, ..., j + 384). The 64 RoPE dims are kept in bf16.
The reference quantizes index K and compressed KV to fp4. These two are fp8 (the deployment's KV
dtype), so they do not match the reference bit for bit.
E8M0 scales are raw exponent bytes (uint8), decoded by kernel_ref.scale_f32.
"""

from __future__ import annotations

import torch

from sglang_tpu.dsv41_native.kernel_ref import FP8_MAX, _blocks, _pow2_ceil, scale_f32

CODE_DTYPE = torch.float8_e4m3fn
INDEX_K_BLOCK = 128  # one scale per index-key vector
CSA_SCALE_PERIOD = 64  # CSA nope scale lanes
CSA_ROPE_DIM = 64  # CSA dims kept in bf16


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


def dequant(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """codes [..., N] and contiguous-block scales [..., N / block] -> dtype.

    Computed in `dtype` directly: an e4m3 code times a power-of-two scale is exact in bf16, so no
    fp32 copy of the gathered rows is made (decode reads [B, context / ratio, d] rows per indexer)."""
    block_size = q.size(-1) // s.size(-1)
    v = q.to(dtype).unflatten(-1, (-1, block_size)) * scale_f32(s).to(dtype).unsqueeze(
        -1
    )
    return v.flatten(-2)


def quantize_fp8_lane_periodic(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CSA latent [..., 512] -> (e4m3 nope codes [..., 448], uint8 scales [..., 64], bf16 rope
    [..., 64]). Scale j covers nope dims d with d % 64 == j; amax floor 1e-4 and UE8M0 rounding as
    the production quantizer."""
    nope = x.shape[-1] - CSA_ROPE_DIM
    xn = x[..., :nope].float()
    lanes = xn.unflatten(-1, (nope // CSA_SCALE_PERIOD, CSA_SCALE_PERIOD))
    amax = lanes.abs().amax(-2).clamp_min(1e-4)  # [..., 64]
    s = _pow2_ceil(amax / FP8_MAX)
    q = (lanes / s.unsqueeze(-2)).clamp(-FP8_MAX, FP8_MAX).to(CODE_DTYPE)
    return q.flatten(-2), _e8m0_byte(s), x[..., nope:].to(torch.bfloat16)


def dequant_lane_periodic(
    q: torch.Tensor, s: torch.Tensor, rope: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of quantize_fp8_lane_periodic: [..., 448] codes, [..., 64] scales, [..., 64] rope
    -> [..., 512] in dtype (exact for bf16, as `dequant`)."""
    lanes = q.to(dtype).unflatten(-1, (-1, CSA_SCALE_PERIOD))
    nope = (lanes * scale_f32(s).to(dtype).unsqueeze(-2)).flatten(-2)
    return torch.cat([nope, rope.to(dtype)], dim=-1)
