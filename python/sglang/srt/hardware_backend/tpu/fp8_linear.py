"""TPU FP8 dense linear helpers."""

import torch


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
    x_q = torch.clamp(x.float() / x_scale, min=-dtype_max, max=dtype_max).to(
        target_dtype
    )
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
    from sglang.srt.hardware_backend.tpu.quantized_matmul import (
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
