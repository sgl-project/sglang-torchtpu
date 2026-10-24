"""Fused QK RMSNorm + RoPE as one TPU Pallas op.

Consumes the raw bf16 QKV projection output and returns normed and rotated
q and k plus the unchanged v.

Two kernels live here:

* ``tpu_fused_qk_norm_rope`` (Qwen3-MoE): norm and rotation in fp32 with a
  single output rounding, cos/sin cast to the activation dtype.
* ``tpu_fused_qk_norm_position`` (Beam): the norm output is rounded to the
  activation dtype before the position step, matching
  ``round_norm_before_rope=True`` on GPU. The position step is either NeoX RoPE
  with fp32 cos/sin (sliding layers) or a per-token fp32 q scale with k
  unscaled (NoPE + SSMax full layers)::

      from sglang_tpu.qk_norm_rope import (
          compute_ssmax_scale,          # positions -> fp32 [T], once per forward
          tpu_fused_qk_norm_position,
      )
      q, k, v = tpu_fused_qk_norm_position(
          qkv,                           # [T, >= (Hq + 2 Hkv) * D] bf16
          q_weight, k_weight,            # [D]
          eps, num_q_heads, num_kv_heads, head_dim,
          positions=positions,           # with cos_sin_cache=... (RoPE mode)
          cos_sin_cache=rotary_emb.cos_sin_cache,
          # or: q_scale=ssmax_scale      # fp32 [T] (SSMax mode)
      )
"""

from threading import Lock
from typing import TYPE_CHECKING, Callable

import torch

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

        # Imported here so the module (and its torch references) load without
        # torch_tpu, e.g. in CPU parity tests.
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

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


# ---------------------------------------------------------------------------
# Rounded-norm variant with a selectable position step (Beam).
#
# Differences from the op above, all for parity with the GPU reference:
#   * the RMSNorm output is rounded to the activation dtype before the
#     position step (round_norm_before_rope=True);
#   * cos/sin stay fp32 (the GPU cache is fp32);
#   * the position step is either NeoX RoPE over the whole head or a per-token
#     fp32 scale on q only (SSMax: q = bf16(bf16(norm(q)) * s)), k unscaled;
#   * the input may carry trailing columns after v (for example a packed
#     attention gate); only the q and k columns are DMAd.
# ---------------------------------------------------------------------------

_MODE_ROPE = "rope"
_MODE_Q_SCALE = "q_scale"


def compute_ssmax_scale(positions: torch.Tensor) -> torch.Tensor:
    """Beam SSMax per-token q scale ``(1 + (pos + 1) / 2048) ** 0.1`` in fp32.

    Same expression as the reference (``beam.py:84``); compute it once per
    forward and share it across the full-attention layers.
    """
    return torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))


def qk_norm_position_ref(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference: the GPU reference's per-head norm then RoPE or SSMax.

    Norm: ``bf16(x_f32 * rsqrt(mean(x_f32^2) + eps) * w)`` per (token, head).
    RoPE: NeoX rotate-half in fp32 on the rounded norm with fp32 cos/sin,
    rounded once. SSMax: ``bf16(q_norm_f32 * s[:, None])``; k only normed.
    """
    _check_position_mode(cos_sin_cache, q_scale)
    dtype = qkv.dtype
    t = qkv.shape[0]
    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    q = qkv[:, :q_size].reshape(t, num_q_heads, head_dim)
    k = qkv[:, q_size : q_size + kv_size].reshape(t, num_kv_heads, head_dim)
    v = qkv[:, q_size + kv_size : q_size + 2 * kv_size]

    def norm(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * w).to(dtype)

    q, k = norm(q, q_weight), norm(k, k_weight)
    if cos_sin_cache is not None:
        cos_sin = cos_sin_cache[positions.reshape(-1).long()].float()
        half = head_dim // 2
        cos, sin = cos_sin[:, None, :half], cos_sin[:, None, half:]

        def rope(x):
            xf = x.float()
            x1, x2 = xf[..., :half], xf[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(dtype)

        q, k = rope(q), rope(k)
    else:
        q = (q.float() * q_scale.reshape(t, 1, 1).float()).to(dtype)
    return q.reshape(t, q_size), k.reshape(t, kv_size), v


def _qk_norm_position_kernel(
    qkv_ref,
    aux_ref,
    qw_ref,
    kw_ref,
    q_ref,
    k_ref,
    *,
    eps: float,
    num_q: int,
    num_kv: int,
    hd: int,
    mode: str,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    act = q_ref.dtype
    qw = qw_ref[...].astype(jnp.float32)
    kw = kw_ref[...].astype(jnp.float32)
    half = hd // 2

    if mode == _MODE_ROPE:
        # aux is [cos | sin] per row. Rotate-half as a lane roll, which is a
        # native lane rotate on TPU instead of a 64-lane slice and concat:
        #   out = x * [cos | cos] + roll(x, half) * [-sin | sin]
        # which equals [x1 cos - x2 sin | x2 cos + x1 sin] exactly in IEEE fp32.
        cs = aux_ref[...].astype(jnp.float32)
        lane = jax.lax.broadcasted_iota(jnp.int32, cs.shape, 1)
        first = lane < half
        cs_rot = pltpu.roll(cs, half, 1)  # [sin | cos]
        cos = jnp.where(first, cs, cs_rot)
        sin = jnp.where(first, -cs_rot, cs)
    else:
        scale = aux_ref[...].astype(jnp.float32)  # [bm, 1]

    # One lane-aligned 128-wide slice per head: no lane reshapes in Mosaic.
    for h in range(num_q + num_kv):
        is_q = h < num_q
        x = qkv_ref[:, h * hd : (h + 1) * hd].astype(jnp.float32)
        rstd = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
        y = (x * rstd * (qw if is_q else kw)).astype(act).astype(jnp.float32)
        if mode == _MODE_ROPE:
            y = y * cos + pltpu.roll(y, half, 1) * sin
        elif is_q:
            y = y * scale
        out_ref, col = (q_ref, h) if is_q else (k_ref, h - num_q)
        out_ref[:, col * hd : (col + 1) * hd] = y.astype(out_ref.dtype)


def _qk_norm_position_jax(
    qkv: "jax.Array",  # [tokens, width >= (num_q + 2 * num_kv) * hd]
    aux: "jax.Array",  # rope: fp32 [tokens, hd] cos|sin; q_scale: fp32 [tokens, 1]
    q_weight: "jax.Array",  # [hd]
    k_weight: "jax.Array",  # [hd]
    eps: float,
    num_q: int,
    num_kv: int,
    hd: int,
    mode: str,
    *,
    interpret: bool = False,
) -> "tuple[jax.Array, jax.Array]":
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m = qkv.shape[0]
    q_size, kv_size = num_q * hd, num_kv * hd
    bm = min(m, _MAX_BLOCK_TOKENS)
    kernel = lambda *refs: _qk_norm_position_kernel(
        *refs, eps=eps, num_q=num_q, num_kv=num_kv, hd=hd, mode=mode
    )
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(m, bm),),
        in_specs=[
            # Only the q and k columns; v and anything after it stay in HBM.
            pl.BlockSpec((bm, q_size + kv_size), lambda i: (i, 0)),
            pl.BlockSpec((bm, aux.shape[1]), lambda i: (i, 0)),
            pl.BlockSpec((1, hd), lambda i: (0, 0)),
            pl.BlockSpec((1, hd), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bm, q_size), lambda i: (i, 0)),
            pl.BlockSpec((bm, kv_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((m, q_size), qkv.dtype),
            jax.ShapeDtypeStruct((m, kv_size), qkv.dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=f"fused_qk_norm_{mode}",
    )(qkv, aux, q_weight.reshape(1, hd), k_weight.reshape(1, hd))


_QK_NORM_POSITION_OPS: dict[str, Callable] = {}


def _get_qk_norm_position_op(mode: str) -> Callable:
    op = _QK_NORM_POSITION_OPS.get(mode)
    if op is not None:
        return op
    with _QK_NORM_ROPE_LOCK:
        if mode in _QK_NORM_POSITION_OPS:
            return _QK_NORM_POSITION_OPS[mode]

        import jax  # noqa: F401  (force loud failure if JAX is missing)

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _wrapped(
            qkv: jax.Array,
            aux: jax.Array,
            q_weight: jax.Array,
            k_weight: jax.Array,
            eps: float,
            num_q: int,
            num_kv: int,
            hd: int,
        ) -> tuple[jax.Array, jax.Array]:
            return _qk_norm_position_jax(
                qkv, aux, q_weight, k_weight, eps, num_q, num_kv, hd, mode
            )

        # Rank-local kernel: no mesh or partition specs needed.
        op = cpu_export_jax_op(f"sglang_tpu_pallas::fused_qk_norm_{mode}", _wrapped)

        def _fake(qkv, aux, q_weight, k_weight, eps, num_q, num_kv, hd):
            del aux, q_weight, k_weight, eps
            m = qkv.shape[0]
            return qkv.new_empty((m, num_q * hd)), qkv.new_empty((m, num_kv * hd))

        op.register_fake(_fake)
        _QK_NORM_POSITION_OPS[mode] = op
        return op


def register_tpu_qk_norm_position_ops() -> None:
    """Register both position modes before torch.compile traces a model."""
    _get_qk_norm_position_op(_MODE_ROPE)
    _get_qk_norm_position_op(_MODE_Q_SCALE)


def _check_position_mode(cos_sin_cache, q_scale) -> None:
    if (cos_sin_cache is None) == (q_scale is None):
        raise ValueError("Pass exactly one of cos_sin_cache (RoPE) or q_scale (SSMax).")


def tpu_fused_qk_norm_position(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-head q/k RMSNorm (rounded) then RoPE or a q-only scale.

    Returns ``(q [T, Hq*D], k [T, Hkv*D], v)``; v is a view of ``qkv``.
    RoPE mode: pass ``positions`` [T] and ``cos_sin_cache`` [L, D] (fp32 for
    GPU parity; a bf16 cache is upcast, not rounded further). SSMax mode: pass
    ``q_scale`` fp32 [T] from ``compute_ssmax_scale``. Off TPU this runs
    ``qk_norm_position_ref``.
    """
    _check_position_mode(cos_sin_cache, q_scale)
    if qkv.dtype != torch.bfloat16:
        raise ValueError(f"Only bfloat16 activations are supported, got {qkv.dtype}.")
    if qkv.dim() != 2:
        raise ValueError(f"qkv must be 2-D [tokens, width], got {tuple(qkv.shape)}.")
    num_tokens = qkv.shape[0]
    if num_tokens == 0:
        raise ValueError("qkv must have at least one token row, got 0.")
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
    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    if qkv.shape[1] < q_size + 2 * kv_size:
        raise ValueError(
            f"qkv width {qkv.shape[1]} is narrower than "
            f"(num_q + 2 * num_kv) * head_dim = {q_size + 2 * kv_size}."
        )
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(
            f"q/k norm weights must both be [{head_dim}], got "
            f"{tuple(q_weight.shape)} and {tuple(k_weight.shape)}."
        )
    if cos_sin_cache is not None:
        if positions is None or positions.numel() != num_tokens:
            raise ValueError("RoPE mode needs one position per qkv row.")
        if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] != head_dim:
            raise ValueError(
                f"cos_sin_cache must be [cache_len, {head_dim}] (full rotary), "
                f"got {tuple(cos_sin_cache.shape)}."
            )
    elif q_scale.numel() != num_tokens:
        raise ValueError(
            f"q_scale has {q_scale.numel()} entries but qkv has {num_tokens} rows."
        )

    if qkv.device.type != "tpu":
        return qk_norm_position_ref(
            qkv,
            q_weight,
            k_weight,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            q_scale=q_scale,
        )
    if cos_sin_cache is not None:
        mode = _MODE_ROPE
        aux = cos_sin_cache[positions.reshape(-1).long()].to(torch.float32)
    else:
        mode = _MODE_Q_SCALE
        aux = q_scale.reshape(num_tokens, 1).to(torch.float32)

    v = qkv[:, q_size + kv_size : q_size + 2 * kv_size]
    q, k = _get_qk_norm_position_op(mode)(
        qkv, aux, q_weight, k_weight, eps, num_q_heads, num_kv_heads, head_dim
    )
    return q, k, v
