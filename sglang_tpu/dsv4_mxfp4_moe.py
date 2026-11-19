# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 block-32 MXFP4 routed experts -> TPU block-512 packed-FP4 (W4A8).

DeepSeek-V4 checkpoints (V4 Flash/Pro, V4.1) store the routed experts of an
otherwise FP8 checkpoint as block-32 MXFP4; the TPU Pallas gmm_v2 path reads
block-512 K-major packed FP4. prepare_tpu_fp8_moe_weights calls
prepare_tpu_dsv4_mxfp4_moe_weights for such a layer.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.utils import copy_or_rebind_param
from sglang_tpu.fp8_moe import (
    _clear_replaced_param,
    _free_replaced_storage,
    _own_storage,
)

_MXFP4_BLOCK_SIZE = 32
_TPU_FP4_BLOCK_SIZE = 512
_FP4_PACKING = 2
_FP4_MAX = 6.0
_MXFP4_EXPERTS_PER_CHUNK = 8
_FP4_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_FP4_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _dequantize_dsv4_mxfp4_block32(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize block-32 MXFP4 packed weights ``[E, N, K/2]`` to float32 ``[E, N, K]``."""
    w_u8 = weight_packed.view(torch.uint8)
    low = (w_u8 & 0x0F).to(torch.long)
    high = ((w_u8 >> 4) & 0x0F).to(torch.long)
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=w_u8.device)
    # A 1-D index gathers ~13x faster on TPU than an n-d one.
    w_fp32 = torch.stack(
        (lut[low.flatten()].view(low.shape), lut[high.flatten()].view(high.shape)),
        dim=-1,
    ).flatten(-2)

    if scale.dtype == torch.uint8:
        scale_fp32 = torch.exp2(scale.to(torch.int32).to(torch.float32) - 127.0)
    else:
        scale_fp32 = scale.to(torch.float32)

    return (
        w_fp32.unflatten(-1, (-1, _MXFP4_BLOCK_SIZE)) * scale_fp32.unsqueeze(-1)
    ).flatten(-2)


def _quantize_tpu_fp4_block512_kmajor(
    weight_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize float32 ``[E, N, K]`` to gmm_v2 packed-FP4 ``(E, K/2, N)`` and scale ``(E, K/512, 1, N)``."""
    num_experts, out_dim, in_dim = weight_fp32.shape
    if in_dim % _TPU_FP4_BLOCK_SIZE != 0:
        raise ValueError(
            "TPU packed-FP4 K dimension must be divisible by "
            f"{_TPU_FP4_BLOCK_SIZE}, got {in_dim}."
        )
    num_blocks_k = in_dim // _TPU_FP4_BLOCK_SIZE
    w_blocked = weight_fp32.reshape(
        num_experts, out_dim, num_blocks_k, _TPU_FP4_BLOCK_SIZE
    )
    abs_max = w_blocked.abs().amax(dim=-1, keepdim=True)
    block_scale = abs_max / _FP4_MAX
    scale_inv = torch.where(
        (block_scale == 0) | ~torch.isfinite(block_scale),
        torch.zeros_like(block_scale),
        1.0 / block_scale,
    )
    w_scaled = (w_blocked * scale_inv).clamp(-_FP4_MAX, _FP4_MAX)
    abs_scaled = w_scaled.abs()
    boundaries = torch.tensor(
        _FP4_E2M1_THRESHOLDS, dtype=abs_scaled.dtype, device=abs_scaled.device
    )
    mag_idx = torch.bucketize(abs_scaled, boundaries, out_int32=True, right=False).to(
        torch.uint8
    )
    sign_bit = ((w_scaled < 0) & (mag_idx > 0)).to(torch.uint8) << 3
    indices = (mag_idx | sign_bit).reshape(num_experts, out_dim, in_dim)
    packed_nk = indices[..., 0::2] | (indices[..., 1::2] << 4)
    packed_kn = packed_nk.permute(0, 2, 1).contiguous()
    scale_gmm = block_scale.squeeze(-1).permute(0, 2, 1).unsqueeze(2).contiguous()
    return packed_kn, _own_storage(scale_gmm)


def _convert_mxfp4_experts(
    w13_packed, w13_scale, w2_packed, w2_scale, pad_gate_up, pad_last
):
    """Block-32 MXFP4 -> TPU block-512 K-major FP4 for experts on the tensors'
    own device, in chunks of _MXFP4_EXPERTS_PER_CHUNK experts to bound the fp32
    intermediates. Returns (w13_q, w13_s, w2_q, w2_s)."""
    num_experts, two_inter, _ = w13_packed.shape
    if two_inter % 2 != 0:
        raise ValueError(
            f"Expected even w13 output dimension (2 * intermediate), got {two_inter}."
        )
    inter = two_inter // 2
    pad_inter = (
        _TPU_FP4_BLOCK_SIZE - inter % _TPU_FP4_BLOCK_SIZE
    ) % _TPU_FP4_BLOCK_SIZE
    # Each output is allocated once and filled chunk by chunk, so a layer's
    # converted experts never exist twice (chunk list + concatenation).
    out: list = [None, None, None, None]

    def put(slot: int, c0: int, chunk: torch.Tensor) -> None:
        if out[slot] is None:
            out[slot] = chunk.new_empty((num_experts, *chunk.shape[1:]))
        out[slot][c0 : c0 + chunk.shape[0]].copy_(chunk)

    for c0 in range(0, num_experts, _MXFP4_EXPERTS_PER_CHUNK):
        c1 = min(c0 + _MXFP4_EXPERTS_PER_CHUNK, num_experts)
        w13 = pad_gate_up(
            _dequantize_dsv4_mxfp4_block32(w13_packed[c0:c1], w13_scale[c0:c1]),
            inter,
            pad_inter,
        )
        q, sc = _quantize_tpu_fp4_block512_kmajor(w13)
        del w13
        put(0, c0, q)
        put(1, c0, sc)
        del q, sc
        w2 = pad_last(
            _dequantize_dsv4_mxfp4_block32(w2_packed[c0:c1], w2_scale[c0:c1]),
            pad_inter,
        )
        q, sc = _quantize_tpu_fp4_block512_kmajor(w2)
        del w2
        put(2, c0, q)
        put(3, c0, sc)
        del q, sc
    return tuple(out)


def prepare_tpu_dsv4_mxfp4_moe_weights(layer: torch.nn.Module) -> None:
    """Convert DeepSeek-V4 block-32 MXFP4 MoE weights into TPU block-512 packed-FP4
    (W4A8) layout on the device that holds them (no host round trip); the
    result is bit-identical to a host conversion."""
    from sglang.srt.layers.quantization.mxfp4_tpu_util import (
        _pad_gate_up_halves_axis1,
        _pad_last_axis,
    )

    w13_param, w2_param = layer.w13_weight, layer.w2_weight
    num_experts, _, hidden_packed = w13_param.shape
    hidden = hidden_packed * _FP4_PACKING
    if hidden % _TPU_FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"TPU packed-FP4 MoE requires hidden_size divisible by {_TPU_FP4_BLOCK_SIZE}, "
            f"got {hidden}."
        )
    inter = w13_param.shape[1] // 2
    expected_w2_shape = (num_experts, hidden, inter // _FP4_PACKING)
    if tuple(w2_param.shape) != expected_w2_shape:
        raise ValueError(
            f"Unexpected w2_weight shape {tuple(w2_param.shape)}, "
            f"expected {expected_w2_shape}."
        )
    w13_w, w13_s, w2_w, w2_s = _convert_mxfp4_experts(
        w13_param.data,
        layer.w13_weight_scale_inv.data,
        w2_param.data,
        layer.w2_weight_scale_inv.data,
        _pad_gate_up_halves_axis1,
        _pad_last_axis,
    )
    for param in (w13_param, w2_param):
        old = param.data
        param.data = torch.empty(0, dtype=old.dtype, device=old.device)
        _free_replaced_storage(old, param.data)
    _clear_replaced_param(layer, "w13_weight_scale_inv")
    _clear_replaced_param(layer, "w2_weight_scale_inv")
    copy_or_rebind_param(layer, "w13_weight", w13_w)
    copy_or_rebind_param(layer, "w2_weight", w2_w)
    copy_or_rebind_param(layer, "w13_weight_scale", _own_storage(w13_s))
    copy_or_rebind_param(layer, "w2_weight_scale", _own_storage(w2_s))
    layer.weight_block_size = None
