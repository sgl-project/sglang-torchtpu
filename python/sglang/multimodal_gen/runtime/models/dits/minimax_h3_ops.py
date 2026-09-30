# SPDX-License-Identifier: Apache-2.0
"""H3 tensor operations with device-local optional kernel imports.

Keep the native PyTorch paths importable without the CUDA/Triton toolchain.
The fused paths retain the model's existing eligibility and rounding rules.
"""

import torch
import torch.nn as nn

_BF16_DTYPE = torch.bfloat16


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply indexed affine modulation, reusing disposable CUDA BF16 input."""
    if (
        x.is_cuda
        and x.dtype == _BF16_DTYPE
        and dtype == _BF16_DTYPE
        and shift.dtype == _BF16_DTYPE
        and scale.dtype == _BF16_DTYPE
        and x.is_contiguous()
    ):
        from sglang.kernels.ops.diffusion import indexed_scale_shift_bf16_

        return indexed_scale_shift_bf16_(x, shift, scale, indices)
    return (
        x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(dtype)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
    allow_inplace: bool = True,
) -> torch.Tensor:
    """Apply an indexed gated residual, optionally reusing the input buffer."""
    if (
        x.is_cuda
        and x.dtype == _BF16_DTYPE
        and dtype == _BF16_DTYPE
        and gate.dtype == _BF16_DTYPE
        and other.dtype == _BF16_DTYPE
        and x.is_contiguous()
        and other.is_contiguous()
    ):
        if allow_inplace:
            from sglang.kernels.ops.diffusion import indexed_gate_bf16_

            return indexed_gate_bf16_(x, gate, other, indices)
        from sglang.kernels.ops.diffusion import indexed_gate_bf16

        return indexed_gate_bf16(x, gate, other, indices)
    return (x + gate.index_select(0, indices) * other).to(dtype)


def _silu_mul(hidden: torch.Tensor, *, reuse_input: bool) -> torch.Tensor:
    if (
        reuse_input
        and hidden.is_cuda
        and hidden.dtype == _BF16_DTYPE
        and hidden.is_contiguous()
        and hidden.shape[-1] % 16 == 0
    ):
        from sglang.kernels.ops.activation.activation import (
            silu_and_mul_with_activation_rounding_,
        )

        return silu_and_mul_with_activation_rounding_(hidden)
    gate, up = hidden.chunk(2, dim=-1)
    return nn.functional.silu(gate) * up


def _apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    head_dim: int,
    *,
    cuda_platform: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        q.is_cuda
        and q.dtype == _BF16_DTYPE
        and q.dtype == k.dtype == q_norm.weight.dtype == k_norm.weight.dtype
        and q.stride(-1) == k.stride(-1) == 1
        and q.stride(-2) == k.stride(-2) == head_dim
        and q_norm.eps == k_norm.eps
        and not torch.compiler.is_compiling()
        and cuda_platform
    ):
        from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

        fused_inplace_qknorm(
            q,
            k,
            q_norm.weight,
            k_norm.weight,
            eps=q_norm.eps,
            head_dim=head_dim,
        )
        return q, k
    return q_norm(q), k_norm(k)
