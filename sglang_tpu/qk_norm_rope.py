"""Fused QK RMSNorm + RoPE as one TPU Pallas op.

Consumes the raw bf16 QKV projection output and returns normed and rotated
q and k plus the unchanged v.
"""

from threading import Lock
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

if TYPE_CHECKING:
    import jax

# Max token rows per Pallas program.
# 256 bounds per-program VMEM for large compiled token buckets. This is still
# not tuned. It would be worthwhile to tune
# if this kernel ever accounts for a large enough share of step time to matter.
_MAX_BLOCK_TOKENS = 256


def _rms(x: "jax.Array", weight: "jax.Array", eps: float) -> "jax.Array":
    import jax
    import jax.numpy as jnp

    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * weight


def _rope_neox(x: "jax.Array", cos: "jax.Array", sin: "jax.Array") -> "jax.Array":
    import jax.numpy as jnp

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _qk_norm_rope_kernel(
    qkv_ref,
    cs_ref,
    qw_ref,
    kw_ref,
    q_ref,
    k_ref,
    *,
    eps: float,
    num_q: int,
    num_kv: int,
    hd: int,
):
    import jax.numpy as jnp

    # The input block covers only the q and k columns, so the v columns are
    # never fetched into VMEM and the caller returns v as a view of the
    # original QKV tensor.
    raw = qkv_ref[...]
    bm = raw.shape[0]
    q = raw[:, : num_q * hd].astype(jnp.float32).reshape(bm, num_q, hd)
    k = (
        raw[:, num_q * hd : (num_q + num_kv) * hd]
        .astype(jnp.float32)
        .reshape(bm, num_kv, hd)
    )

    cs = cs_ref[...].astype(jnp.float32)
    half = hd // 2
    cos = cs[:, None, :half]
    sin = cs[:, None, half:]

    q = _rope_neox(_rms(q, qw_ref[...].astype(jnp.float32), eps), cos, sin)
    k = _rope_neox(_rms(k, kw_ref[...].astype(jnp.float32), eps), cos, sin)

    q_ref[...] = q.reshape(bm, num_q * hd).astype(q_ref.dtype)
    k_ref[...] = k.reshape(bm, num_kv * hd).astype(k_ref.dtype)


def _qk_norm_rope_jax(
    qkv: "jax.Array",  # [tokens, (num_q_heads + 2 * num_kv_heads) * head_dim], bf16
    cos_sin: "jax.Array",  # [tokens, head_dim]
    q_weight: "jax.Array",  # [head_dim]
    k_weight: "jax.Array",  # [head_dim]
    eps: float,
    num_q: int,
    num_kv: int,
    hd: int,
) -> "tuple[jax.Array, jax.Array]":
    import jax
    from jax.experimental import pallas as pl

    m = qkv.shape[0]
    q_size, kv_size = num_q * hd, num_kv * hd

    bm = min(m, _MAX_BLOCK_TOKENS)
    grid = (pl.cdiv(m, bm),)
    kernel = lambda *refs: _qk_norm_rope_kernel(
        *refs, eps=eps, num_q=num_q, num_kv=num_kv, hd=hd
    )
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            # Only the q and k columns, so v is never DMAd into VMEM.
            pl.BlockSpec((bm, q_size + kv_size), lambda i: (i, 0)),
            pl.BlockSpec((bm, hd), lambda i: (i, 0)),
            pl.BlockSpec((hd,), lambda i: (0,)),
            pl.BlockSpec((hd,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((bm, q_size), lambda i: (i, 0)),
            pl.BlockSpec((bm, kv_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((m, q_size), qkv.dtype),
            jax.ShapeDtypeStruct((m, kv_size), qkv.dtype),
        ],
        name="fused_qk_norm_rope",
    )(qkv, cos_sin, q_weight, k_weight)


_QK_NORM_ROPE_OP: Callable | None = None
_QK_NORM_ROPE_LOCK = Lock()


def _get_qk_norm_rope_op() -> Callable:
    global _QK_NORM_ROPE_OP
    if _QK_NORM_ROPE_OP is not None:
        return _QK_NORM_ROPE_OP

    with _QK_NORM_ROPE_LOCK:
        if _QK_NORM_ROPE_OP is not None:
            return _QK_NORM_ROPE_OP

        import jax  # noqa: F401  (force loud failure if JAX is missing)

        op_name = "sglang_tpu_pallas::fused_qk_norm_rope"

        def _wrapped(
            qkv: jax.Array,
            cos_sin: jax.Array,
            q_weight: jax.Array,
            k_weight: jax.Array,
            eps: float,
            num_q: int,
            num_kv: int,
            hd: int,
        ) -> tuple[jax.Array, jax.Array]:
            return _qk_norm_rope_jax(
                qkv, cos_sin, q_weight, k_weight, eps, num_q, num_kv, hd
            )

        # Rank-local kernel: no mesh or partition specs needed.
        op = cpu_export_jax_op(op_name, _wrapped)

        def _fake(qkv, cos_sin, q_weight, k_weight, eps, num_q, num_kv, hd):
            del cos_sin, q_weight, k_weight, eps
            m = qkv.shape[0]
            return (
                qkv.new_empty((m, num_q * hd)),
                qkv.new_empty((m, num_kv * hd)),
            )

        op.register_fake(_fake)
        _QK_NORM_ROPE_OP = op
        return op


def register_tpu_qk_norm_rope_op() -> None:
    """Register the op before torch.compile traces a model."""

    _get_qk_norm_rope_op()


# The kernel is written and validated only for 128-wide heads (it rotates the
# whole head and casts a [tokens, 128] slice per head). Widen this set only
# once other sizes are tested.
_SUPPORTED_HEAD_DIM = 128


def can_use_tpu_fused_qk_norm_rope(
    *,
    rotary_emb: object,
    dtype: torch.dtype,
    head_dim: int,
) -> bool:
    """Whether a rotary embedding is eligible for the fused TPU op."""
    if dtype != torch.bfloat16:
        return False
    if head_dim != _SUPPORTED_HEAD_DIM:
        return False
    if getattr(rotary_emb, "is_neox_style", None) is not True:
        return False
    # The op rotates the whole head, so it is only correct when rotary_dim
    # covers the head (no partial rotary).
    if getattr(rotary_emb, "rotary_dim", None) != head_dim:
        return False
    cache = getattr(rotary_emb, "cos_sin_cache", None)
    if cache is None or cache.dim() != 2 or cache.shape[-1] != head_dim:
        return False
    return True


def tpu_fused_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split QKV and apply per-head RMSNorm + neox RoPE to q and k.

    Returns ``(q, k, v)`` where q and k are freshly computed and v is a view
    into ``qkv`` (the op never touches v).
    """
    if qkv.dtype != torch.bfloat16:
        raise ValueError(f"Only bfloat16 activations are supported, got {qkv.dtype}.")
    if qkv.dim() != 2:
        raise ValueError(f"qkv must be 2-D [tokens, width], got {tuple(qkv.shape)}.")
    if qkv.shape[0] == 0:
        raise ValueError("qkv must have at least one token row, got 0.")
    if positions.numel() != qkv.shape[0]:
        raise ValueError(
            f"positions has {positions.numel()} entries but qkv has "
            f"{qkv.shape[0]} rows."
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            f"num_q_heads and num_kv_heads must be positive, got "
            f"{num_q_heads} and {num_kv_heads}."
        )
    if head_dim != _SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"The Pallas kernel is only validated for head_dim "
            f"{_SUPPORTED_HEAD_DIM}, got {head_dim}."
        )
    expected_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    if qkv.shape[-1] != expected_width:
        raise ValueError(
            f"qkv width {qkv.shape[-1]} does not match "
            f"(num_q + 2 * num_kv) * head_dim = {expected_width}."
        )
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(
            f"q/k norm weights must both be [{head_dim}], got "
            f"{tuple(q_weight.shape)} and {tuple(k_weight.shape)}."
        )
    if cos_sin_cache.dim() != 2:
        # The kernel gathers one cache row per position, so the cache must be a
        # 2-D [cache_len, head_dim] table. Anything else is a rotary variant
        # this op does not implement, for example a per-head Fourier cache. A
        # caller can reach here by caching the eligibility decision and then
        # swapping rotary_emb, so name the flag that turns the fused path off.
        raise ValueError(
            f"cos_sin_cache must be 2-D [cache_len, {head_dim}], got "
            f"{tuple(cos_sin_cache.shape)}. This rotary embedding is not "
            "supported by the fused TPU path, so serve it with "
            "--enable-fused-qk-norm-rope off."
        )
    if cos_sin_cache.shape[-1] != head_dim:
        # The kernel rotates the whole head, so a narrower row means the model
        # uses partial rotary, which this op does not implement.
        raise ValueError(
            f"cos_sin_cache width {cos_sin_cache.shape[-1]} must equal head_dim "
            f"{head_dim}. Partial rotary is unsupported."
        )

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    # v is a pass-through slice of the input QKV. Return it as a view so the op
    # never loads, allocates, or copies it.
    v = qkv[:, q_size + kv_size :]

    cos_sin = cos_sin_cache[positions.reshape(-1).long()].to(qkv.dtype)
    q, k = _get_qk_norm_rope_op()(
        qkv,
        cos_sin,
        q_weight,
        k_weight,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    return q, k, v
