# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/google-pytorch/torchtpu-vllm/blob/main/src/vllm_torchtpu/layers/common/quantization.py
# Copyright 2025 Google LLC
"""TPU FP8 dense linear helpers."""

import logging

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.runtime_context import get_parallel
from sglang_tpu.fp8_moe import _clear_replaced_param, _get_tpu_fp8_runtime

logger = logging.getLogger(__name__)


def compute_tpu_fp8_linear_scale_shape(
    layer: Module,
    *,
    input_size: int,
    input_size_per_partition: int,
    output_size: int,
    output_partition_sizes: list[int],
    block_n: int,
    block_k: int,
    output_size_per_partition: int | None = None,
) -> tuple[int, int]:
    """Record TPU TP partition offsets on ``layer`` and compute block-FP8 scale shape."""
    if output_size_per_partition is None:
        output_size_per_partition = sum(output_partition_sizes)
    if (
        output_size != output_size_per_partition
        or input_size != input_size_per_partition
    ):
        tp_rank = getattr(layer, "tp_rank", None)
        if tp_rank is None:
            tp_rank = get_parallel().tp_rank
    else:
        tp_rank = 0

    output_tp_rank = tp_rank if output_size != output_size_per_partition else 0
    input_offset = (
        tp_rank * input_size_per_partition
        if input_size != input_size_per_partition
        else 0
    )
    layer.tpu_fp8_output_tp_rank = output_tp_rank
    layer.tpu_fp8_input_offset = input_offset

    scale_rows = sum(
        (output_tp_rank * partition_size + partition_size + block_n - 1) // block_n
        - (output_tp_rank * partition_size) // block_n
        for partition_size in output_partition_sizes
    )
    if input_size != input_size_per_partition:
        scale_cols = (
            input_offset + input_size_per_partition + block_k - 1
        ) // block_k - input_offset // block_k
    else:
        scale_cols = (input_size_per_partition + block_k - 1) // block_k
    return scale_rows, scale_cols


def _requantize_tpu_block_fp8_to_channelwise(
    layer: Module,
    weight_block_size: list[int] | tuple[int, int],
) -> None:
    """Convert block-FP8 dense linear weights (including sub-block TP shards) to channelwise FP8."""
    weight = layer.weight.data
    weight_scale_inv = layer.weight_scale_inv.data
    block_n, block_k = weight_block_size
    out_size, in_size = weight.shape
    logical_widths = getattr(layer, "logical_widths", None) or [out_size]
    if sum(logical_widths) != out_size:
        raise ValueError(
            "TPU FP8 block linear logical widths must cover the local output "
            f"dimension, got {logical_widths=} and {out_size=}."
        )
    tp_rank = getattr(layer, "tpu_fp8_output_tp_rank", None)
    if tp_rank is None:
        if getattr(layer, "output_size", out_size) != out_size:
            raw_tp_rank = getattr(layer, "tp_rank", None)
            if raw_tp_rank is None:
                raw_tp_rank = get_parallel().tp_rank
            tp_rank = raw_tp_rank
        else:
            tp_rank = 0
    input_offset = getattr(layer, "tpu_fp8_input_offset", None)
    if input_offset is None:
        if getattr(layer, "input_size", in_size) != in_size:
            raw_tp_rank = getattr(layer, "tp_rank", None)
            if raw_tp_rank is None:
                raw_tp_rank = get_parallel().tp_rank
            input_offset = raw_tp_rank * in_size
        else:
            input_offset = 0

    first_k_block = input_offset // block_k
    last_k_block = (input_offset + in_size + block_k - 1) // block_k
    k_blocks = last_k_block - first_k_block
    k_offset = input_offset - first_k_block * block_k
    k_block_ids = (torch.arange(in_size, device=weight.device) + k_offset) // block_k

    dequantized_shards = []
    weight_offset = 0
    scale_offset = 0
    for logical_width in logical_widths:
        global_start = tp_rank * logical_width
        first_block = global_start // block_n
        last_block = (global_start + logical_width + block_n - 1) // block_n
        num_blocks = last_block - first_block
        row_offset = global_start - first_block * block_n

        shard_weight = weight[weight_offset : weight_offset + logical_width]
        shard_scale = weight_scale_inv[scale_offset : scale_offset + num_blocks]
        if shard_scale.shape != (num_blocks, k_blocks):
            raise ValueError(
                "TPU FP8 block linear scale shard shape mismatch: "
                f"got {tuple(shard_scale.shape)}, expected {(num_blocks, k_blocks)}."
            )

        n_block_ids = (
            torch.arange(logical_width, device=weight.device) + row_offset
        ) // block_n
        expanded_scale = shard_scale[n_block_ids][:, k_block_ids].to(torch.float32)
        dequantized_shards.append(shard_weight.to(torch.float32) * expanded_scale)
        weight_offset += logical_width
        scale_offset += num_blocks

    if scale_offset != weight_scale_inv.shape[0]:
        raise ValueError(
            "TPU FP8 block linear scale rows were not fully consumed: "
            f"used {scale_offset}, got {weight_scale_inv.shape[0]}."
        )
    weight_dequant = torch.cat(dequantized_shards, dim=0)

    dtype_max = torch.finfo(weight.dtype).max
    weight_scale = torch.clamp(
        weight_dequant.abs().amax(dim=1, keepdim=True) / dtype_max,
        min=1e-12,
    )
    weight = torch.clamp(
        weight_dequant / weight_scale,
        min=-dtype_max,
        max=dtype_max,
    ).to(weight.dtype)

    layer.weight = Parameter(weight.t().contiguous(), requires_grad=False)
    layer.weight_scale = Parameter(weight_scale, requires_grad=False)
    _clear_replaced_param(layer, "weight_scale_inv")
    layer.weight_block_size = None
    layer.input_scale = None


def prepare_tpu_fp8_linear_weights(
    layer: Module,
    *,
    weight_block_size: list[int] | tuple[int, int],
) -> None:
    """Post-load preparation for TPU block-FP8 dense linear layers."""
    block_n, block_k = weight_block_size
    out_size, in_size = layer.weight.shape
    logical_widths = getattr(layer, "logical_widths", None) or [out_size]
    has_subblock_partition = (
        in_size % block_k != 0
        or out_size % block_n != 0
        or any(w % block_n != 0 for w in logical_widths)
    )
    if has_subblock_partition:
        logger.warning_once(
            "TPU dense FP8 linear is converting sub-block-partitioned "
            "checkpoint weights to channelwise runtime scales because "
            "local partition shapes are not divisible by weight_block_size."
        )
        _requantize_tpu_block_fp8_to_channelwise(layer, weight_block_size)
    elif _get_tpu_fp8_runtime() == "channelwise":
        logger.warning_once(
            "TPU dense FP8 linear is converting block-FP8 checkpoint "
            "weights to channelwise runtime scales by default for "
            "performance. Use --tpu-fp8-runtime=channelwise to make "
            "this explicit, or --tpu-fp8-runtime=blockwise to preserve "
            "blockwise runtime scales."
        )
        _requantize_tpu_block_fp8_to_channelwise(layer, weight_block_size)
    else:
        layer.weight_block_size = list(weight_block_size)


def register_tpu_block_fp8_linear() -> None:
    """Register the Pallas block-FP8 matmul custom op before torch.compile tracing."""
    from sglang_tpu.quantized_matmul import register_tpu_block_fp8_matmul_op

    register_tpu_block_fp8_matmul_op()


def _pad_rows_to_scaled_mm_alignment(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    pad = (16 - (x.shape[0] % 16)) % 16
    if pad == 0:
        return x, 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad)), pad


def _quantize_activation_per_row(
    x: torch.Tensor,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype_max = torch.finfo(target_dtype).max
    x_scale = torch.clamp(
        x.abs().amax(dim=1, keepdim=True).float() / dtype_max, min=1e-12
    )
    # No clamp on the quotient: x_scale is this row's amax / dtype_max, so
    # |x / x_scale| <= dtype_max by construction, and the 1e-12 floor binds only
    # where the numerator is small enough to stay in range. Keep the scale
    # per-row -- the fp8 cast does not saturate, so an over-range quotient
    # becomes NaN (pinned by test_fp8_activation_quantize.py). Saves one [M,K]
    # elementwise kernel per dense linear, ~120 per decode step.
    x_q = (x.float() / x_scale).to(target_dtype)
    return x_q, x_scale


def apply_tpu_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply dense FP8 linear on TPU.

    ``weight`` follows SGLang's post-load dense-linear layout ``[K, N]``.
    ``weight_scale`` may be scalar or per-output-channel ``[N]``.
    """

    if weight_scale.ndim == 2 and weight_scale.shape[1] == 1:
        weight_scale = weight_scale.reshape(-1)
    if weight_scale.ndim > 1:
        raise NotImplementedError(
            "TPU dense FP8 linear only supports per-tensor or per-output-channel "
            f"weight scales; got {tuple(weight_scale.shape)}."
        )
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"Input hidden dim {x.shape[-1]} must match weight hidden dim "
            f"{weight.shape[0]}."
        )
    if weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
        raise ValueError(
            "torch_tpu torch._scaled_mm requires K and N to be multiples of 16; "
            f"got weight shape {tuple(weight.shape)}."
        )

    out_shape = (*x.shape[:-1], weight.shape[1])
    x_2d = x.reshape(-1, x.shape[-1])
    num_tokens = x_2d.shape[0]
    x_2d, pad = _pad_rows_to_scaled_mm_alignment(x_2d)
    x_q, x_scale = _quantize_activation_per_row(x_2d, weight.dtype)

    # TODO(torch_tpu#1820): pass rowwise x_scale and channelwise weight_scale
    # directly to torch._scaled_mm once torch_tpu supports those scale shapes.
    # TODO(torch_tpu#1823): remove the M-dimension pad/narrow once torch_tpu
    # supports ragged-M scaled_mm for decode. For now, use upstream SGLang's
    # unfused-DQ fallback: compute the raw FP8 dot in f32, then apply
    # row/column scales outside the matmul.
    one = torch.ones(1, dtype=torch.float32, device=x.device)
    out = torch._scaled_mm(
        x_q,
        weight,
        scale_a=one,
        scale_b=one,
        out_dtype=torch.float32,
    )
    out = out * x_scale * weight_scale.reshape(1, -1).float()
    if pad:
        out = torch.narrow(out, 0, 0, num_tokens)
    out = out.to(x.dtype).reshape(out_shape)
    if bias is not None:
        out = out + bias
    return out


def tpu_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """TPU backend for SGLang's block-FP8 dense linear dispatch."""
    if input_scale is not None:
        raise NotImplementedError("TPU FP8 block linear only supports dynamic inputs.")
    if len(block_size) != 2:
        raise ValueError(f"Expected 2D block size, got {block_size}.")

    block_n, block_k = block_size
    n, k = weight.shape
    if input.shape[-1] != k:
        raise ValueError(
            f"Input hidden dim {input.shape[-1]} must match weight hidden dim {k}."
        )
    if k % block_k != 0 or n % block_n != 0:
        raise ValueError(
            "TPU block FP8 linear currently requires local weight dimensions to "
            f"be divisible by block_size={block_size}; got weight shape {tuple(weight.shape)}."
        )
    expected_scale_shape = (n // block_n, k // block_k)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            "TPU block FP8 weight_scale shape mismatch: "
            f"got {tuple(weight_scale.shape)}, expected {expected_scale_shape}."
        )
    if block_k % 16 != 0 or n % 16 != 0:
        raise ValueError(
            "TPU block FP8 matmul requires K-block and N to be multiples "
            f"of 16; got block_k={block_k}, N={n}."
        )

    out_shape = (*input.shape[:-1], n)
    # TODO(torch_tpu#1831): try torch._scaled_mm_v2 once torch_tpu registers a
    # native block-scaled matmul op. Until then, keep true blockwise FP8 behind
    # one Pallas custom op, matching torchtpu-vllm's serving path.
    from sglang_tpu.quantized_matmul import (
        blockwise_quantized_matmul,
    )

    kernel_scale = (
        weight_scale.t()
        .contiguous()
        .repeat_interleave(block_n, dim=1)
        .unsqueeze(1)
        .to(torch.float32)
    )
    out = blockwise_quantized_matmul(input, weight, kernel_scale).reshape(out_shape)
    if bias is not None:
        out = out + bias
    return out
