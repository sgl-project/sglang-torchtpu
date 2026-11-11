# SPDX-License-Identifier: Apache-2.0
"""Mesh-free CPU-export bridge for merged Q-A/KV + norms + KV RoPE.

This is an independent, currently unconnected early-QKV building block. The
caller supplies postquant canonical weights and bind-time FP32 gamma upcasts;
row gathering and activation quantization stay inside JAX.
"""

# Keep real jax.Array annotations: cpu_export_jax_op inspects them.
import math
from threading import Lock

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing.qa_rope import (
    fused_qa_kv_norm_rope,
)
from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing.rope_gather import (
    gather_rows,
)

HIDDEN_SIZE = 5120
QR_RANK = 1280
HEAD_DIM = 512
ROPE_DIM = 64
EPS = 1e-6
_OPS: dict = {}
_OPS_LOCK = Lock()


def _qa_rope(
    x: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    gamma_qr: jax.Array,
    gamma_kv: jax.Array,
    cos_sin: jax.Array,
    positions: jax.Array,
    eps: float = EPS,
) -> tuple[jax.Array, jax.Array]:
    if x.shape[0] == 0:
        return (
            jnp.empty((0, QR_RANK), jnp.bfloat16),
            jnp.empty((0, HEAD_DIM), jnp.bfloat16),
        )
    rows = gather_rows(cos_sin, positions)
    return fused_qa_kv_norm_rope(x, weight, scale, gamma_qr, gamma_kv, rows, eps=eps)


def _fake_qa_rope(x, weight, scale, gamma_qr, gamma_kv, cos_sin, positions, eps=EPS):
    return x.new_empty((x.shape[0], QR_RANK)), x.new_empty((x.shape[0], HEAD_DIM))


def _op():
    op = _OPS.get("qa_rope")
    if op is not None:
        return op
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OPS_LOCK:
        op = _OPS.get("qa_rope")
        if op is None:
            op = cpu_export_jax_op("sglang_tpu_pallas::dsv41_qa_rope", _qa_rope)
            op.register_fake(_fake_qa_rope)
            _OPS["qa_rope"] = op
    return op


def register_ops() -> None:
    """Register before torch.compile traces the future early-QKV caller."""
    _op()


def qa_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    gamma_qr: torch.Tensor,
    gamma_kv: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 normalized QR ``[T,1280]`` and norm+rotated KV ``[T,512]``.

    Inputs are BF16 ``x[T,5120]``, canonical FP8 E4M3FN KN weight
    ``[5120,1792]``, FP32 scale ``[1792]``, exact FP32 upcasts of the core's
    BF16 norm weights ``[1280]``/``[512]``, FP32 packed ``[cos|sin]`` table
    ``[max_pos,64]``, and integer positions ``[T]``. Both norms must share positive finite ``eps``.
    Positions are clamped by the JAX gather; callers own padded-row masking.
    No argument is donated or mutated; empty T never enters Pallas/export.

    FP32 gamma preserves core RMSNorm's FP32 normalized-value multiply, rather
    than the kernel's default BF16-gamma intermediate rounding. The projection
    has a BF16 boundary before either norm. This is not a bitwise-equivalence
    promise: FP8 dot/reduction/RoPE implementations can round differently, and
    the kernel uses amax/448 (zero scale -> 1), whereas the generic TPU row
    quantizer floors scale at 1e-12. Tiny nonzero rows below amax=448e-12 can
    therefore differ. No quantization policy or checkpoint is changed here.
    """
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite shared norm epsilon")
    if x.ndim != 2 or x.shape[1] != HIDDEN_SIZE or x.dtype != torch.bfloat16:
        raise ValueError("x must be BF16 [T,5120]")
    if (
        weight.shape != (HIDDEN_SIZE, QR_RANK + HEAD_DIM)
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous canonical FP8 [5120,1792]")
    if scale.shape != (QR_RANK + HEAD_DIM,) or scale.dtype != torch.float32:
        raise ValueError("scale must be FP32 [1792]")
    if gamma_qr.shape != (QR_RANK,) or gamma_qr.dtype != torch.float32:
        raise ValueError("gamma_qr must be an exact FP32 upcast of BF16 [1280]")
    if gamma_kv.shape != (HEAD_DIM,) or gamma_kv.dtype != torch.float32:
        raise ValueError("gamma_kv must be an exact FP32 upcast of BF16 [512]")
    if (
        cos_sin.ndim != 2
        or cos_sin.shape[0] == 0
        or cos_sin.shape[1] != ROPE_DIM
        or cos_sin.dtype != torch.float32
    ):
        raise ValueError("cos_sin must be nonempty FP32 [max_pos,64]")
    if positions.shape != (x.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("positions must be integer [T]")
    if any(
        t.device != x.device
        for t in (weight, scale, gamma_qr, gamma_kv, cos_sin, positions)
    ):
        raise ValueError("all QA inputs must be on the same device")
    if not x.shape[0]:
        return _fake_qa_rope(
            x, weight, scale, gamma_qr, gamma_kv, cos_sin, positions, eps
        )
    return _op()(
        x,
        weight,
        scale,
        gamma_qr,
        gamma_kv,
        cos_sin,
        positions.to(torch.int32),
        float(eps),
    )
