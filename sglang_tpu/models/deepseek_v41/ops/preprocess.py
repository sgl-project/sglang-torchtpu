# SPDX-License-Identifier: Apache-2.0
"""Mesh-free TorchTPU CPU-export bridge for the fused Q-B + RoPE kernel.

The backend supplies normalized BF16 rows, the postquant canonical FP8 KN
weight and a layer-local full FP32 cos/sin table. Row gathering stays inside
JAX; no Torch-side gather, activation quantization or weight conversion occurs.
"""

# Keep real jax.Array annotations: cpu_export_jax_op inspects them.
from threading import Lock

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing.qb_rope import (
    fused_qb_rope,
)
from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing.rope_gather import (
    gather_rows,
)

HEAD_DIM = 512
ROPE_DIM = 64
_OPS: dict = {}
_OPS_LOCK = Lock()


def _qb_rope(
    qr: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    cos_sin: jax.Array,
    positions: jax.Array,
) -> jax.Array:
    if qr.shape[0] == 0:
        return jnp.empty((0, weight.shape[1] // HEAD_DIM, HEAD_DIM), jnp.bfloat16)
    rows = gather_rows(cos_sin, positions)
    return fused_qb_rope(qr, weight, scale, rows, head_dim=HEAD_DIM, exact_rows=True)


def _fake_qb_rope(qr, weight, scale, cos_sin, positions):
    return qr.new_empty((qr.shape[0], weight.shape[1] // HEAD_DIM, HEAD_DIM))


def _op():
    op = _OPS.get("qb_rope")
    if op is not None:
        return op
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OPS_LOCK:
        op = _OPS.get("qb_rope")
        if op is None:
            op = cpu_export_jax_op("sglang_tpu_pallas::dsv41_qb_rope", _qb_rope)
            op.register_fake(_fake_qb_rope)
            _OPS["qb_rope"] = op
    return op


def register_ops() -> None:
    """Register the CPU-export op before torch.compile traces the model."""
    _op()


def qb_rope(
    qr: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Project + rotate to exact BF16 ``[T, H, 512]`` (including ``T == 0``).

    Inputs: BF16 ``qr[T, rank]``, FP8 E4M3FN ``weight[rank, H*512]``,
    FP32 per-column ``scale[H*512]``, FP32 ``cos_sin[max_pos, 64]`` packed
    ``[cos | sin]``, and integer ``positions[T]``. Positions are clamped by
    the existing JAX gather (including padded positions). The caller owns
    masking of padded output rows. No argument is donated or mutated.
    """
    if qr.ndim != 2 or qr.dtype != torch.bfloat16:
        raise ValueError("qr must be BF16 [T, rank]")
    if (
        weight.ndim != 2
        or weight.dtype != torch.float8_e4m3fn
        or weight.shape[0] != qr.shape[1]
        or weight.shape[1] % HEAD_DIM
    ):
        raise ValueError("weight must be canonical FP8 [rank, H*512]")
    if scale.dtype != torch.float32 or scale.shape != (weight.shape[1],):
        raise ValueError("scale must be FP32 [H*512]")
    if (
        cos_sin.ndim != 2
        or cos_sin.shape[0] == 0
        or cos_sin.shape[1] != ROPE_DIM
        or cos_sin.dtype != torch.float32
    ):
        raise ValueError("cos_sin must be nonempty FP32 [max_pos, 64]")
    if positions.shape != (qr.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("positions must be integer [T]")
    if not qr.shape[0]:
        return _fake_qb_rope(qr, weight, scale, cos_sin, positions)
    return _op()(qr, weight, scale, cos_sin, positions.to(torch.int32))
