# SPDX-License-Identifier: Apache-2.0
"""Unconnected, mesh-free ratio-1 compressor CPU-export building block."""

import math
from threading import Lock

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing.compress_rope import (
    default_config,
    fused_compress_rope,
)
from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing.rope_gather import (
    gather_rows,
)

_OPS: dict = {}
_OPS_LOCK = Lock()


def _compress_rope(
    x: jax.Array,
    weight: jax.Array,
    gamma: jax.Array,
    cos_sin: jax.Array,
    positions: jax.Array,
    valid: jax.Array,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if x.shape[0] == 0:
        return (
            jnp.empty((0, 512), jnp.bfloat16),
            jnp.empty((0, 128), jnp.int32),
            jnp.empty((0, 1), jnp.float32),
        )
    rows = gather_rows(cos_sin, positions)
    latent, words, scales, _ = fused_compress_rope(
        x,
        weight,
        gamma,
        rows,
        jnp.ones(valid.shape, jnp.int32),
        valid,
        eps=eps,
        ratio=1,
        precision="core_finish_bf16",
        # Shape-only configuration avoids runtime TPU discovery at export.
        config=default_config(-(-x.shape[0] // 16) * 16),
    )
    return latent, words, scales


def _fake(x, weight, gamma, cos_sin, positions, valid, eps=1e-6):
    return (
        x.new_empty((x.shape[0], 512)),
        x.new_empty((x.shape[0], 128), dtype=torch.int32),
        x.new_empty((x.shape[0], 1), dtype=torch.float32),
    )


def _op():
    op = _OPS.get("compress_rope")
    if op is not None:
        return op
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OPS_LOCK:
        op = _OPS.get("compress_rope")
        if op is None:
            op = cpu_export_jax_op(
                "sglang_tpu_pallas::dsv41_compress_rope", _compress_rope
            )
            op.register_fake(_fake)
            _OPS["compress_rope"] = op
    return op


def register_ops() -> None:
    """Call outside torch.compile, before tracing any future consumer."""
    _op()


def compress_rope(x, weight, gamma, cos_sin, positions, valid, eps=1e-6):
    """Return pre-RoPE BF16 [T,512], INT32 [T,128], FP32 scales [T,1].

    No state or donation. Invalid rows emit zero latent/words and the encoder's
    ordinary zero-row scale (1e-4/448). Positions clamp inside JAX, including
    invalid rows. Gamma is the original frozen norm tensor, never downcast.
    This building block does not enable model dispatch or claim TPU validation.
    """
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if x.ndim != 2 or x.shape[1] != 5120 or x.dtype != torch.bfloat16:
        raise ValueError("x must be BF16 [T,5120]")
    if (
        weight.shape != (5120, 512)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous BF16 canonical [5120,512]")
    if gamma.shape != (512,) or gamma.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("gamma must be original BF16 or FP32 [512]")
    if (
        cos_sin.ndim != 2
        or cos_sin.shape[0] == 0
        or cos_sin.shape[1] != 64
        or cos_sin.dtype != torch.float32
    ):
        raise ValueError("cos_sin must be nonempty FP32 [max_pos,64]")
    if positions.shape != (x.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("positions must be integer [T]")
    if valid.shape != (x.shape[0], 1) or valid.dtype != torch.int32:
        raise ValueError("valid must be INT32 [T,1]")
    if any(t.device != x.device for t in (weight, gamma, cos_sin, positions, valid)):
        raise ValueError("all compressor inputs must share a device")
    if not x.shape[0]:
        return _fake(x, weight, gamma, cos_sin, positions, valid, eps)
    return _op()(
        x, weight, gamma, cos_sin, positions.to(torch.int32), valid, float(eps)
    )
