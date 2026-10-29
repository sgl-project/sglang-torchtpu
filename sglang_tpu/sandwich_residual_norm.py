# SPDX-License-Identifier: Apache-2.0
"""Fused sandwich post-norm, depth-scaled residual and next pre-norm (Pallas).

Public entry points (import path ``sglang_tpu.sandwich_residual_norm``)::

    tpu_sandwich_residual_rmsnorm(
        branch: Tensor,                     # [T, H] bf16, sub-block output f
        residual: Tensor,                   # [T, H] residual stream r
        sandwich_weight: Tensor | None,     # [H] post-norm weight (None: no post-norm)
        out_weight: Tensor | None,          # [H] next pre-norm / final norm weight
        eps: float,
        scales: tuple[float, float],        # (branch_scale, residual_scale)
    ) -> tuple[Tensor | None, Tensor]       # (normed, combined)

    tpu_rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor   # stage entry

    beam_residual_scales(layer_id: int) -> tuple[attn_scales, mlp_scales]

``tpu_sandwich_residual_rmsnorm`` has the signature of the reference
``BeamSandwichResidualRMSNormOp.forward_native``: ``combined`` is the new
residual stream and ``normed`` is ``out_weight``-RMSNorm of it, or None when
``out_weight`` is None (last layer of a non-final pipeline stage). On the last
layer of the last stage pass the final ``model.norm`` weight as ``out_weight``.
The first layer of a stage has no preceding transition, so its
``input_layernorm`` is the standalone ``tpu_rmsnorm``.

Arithmetic, matching the reference's rounding points (plain ``x * w`` RMSNorm,
fp32 reduction):

    f'       = bf16(f * rsqrt(mean(f^2) + eps) * w_post)
    combined = bf16(bf16(f' * c_branch) + bf16(bf16(r) * c_residual))
    normed   = bf16(combined * rsqrt(mean(combined^2) + eps) * w_next)

with ``c_branch = 1/sqrt(a)``, ``c_residual = sqrt(a-1)/sqrt(a)``, ``a = 2(l+1)``
after attention and ``a = 2(l+1)+1`` after the MLP (``beam_residual_scales``).

One HBM read of ``branch`` and ``residual`` and one write each of ``normed`` and
``combined``. The grid tiles token rows; the hidden dimension (5120 for Beam)
stays whole in a block because both norms reduce over it. Off TPU the wrapper
runs the torch reference, so CPU callers and tests share one entry point.

No ``from __future__ import annotations``: torch_tpu resolves the ``_wrapped``
signatures with ``eval_str``, and ``jax`` is only a local name there, so the
annotations must be real objects.
"""

import math
from threading import Lock
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    import jax

# Token rows per Pallas program. At H=5120 each bf16 row is 10 KiB, so a block
# moves 4 x 64 x 10 KiB = 2.5 MiB of I/O (5 MiB double-buffered) plus a few
# 64 x 20 KiB fp32 temporaries: ~10 MiB of VMEM, under every chip's scoped
# default once the limit below is applied. 64 rows keep the per-step DMA well
# above the ~0.35 us pipeline overhead. Not tuned on hardware.
_MAX_BLOCK_TOKENS = 64
_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def beam_residual_scales(
    layer_id: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(attn_scales, mlp_scales) for 0-indexed ``layer_id``, as (branch, residual)."""
    attn_depth = 2 * (layer_id + 1)
    mlp_depth = attn_depth + 1
    attn_den = math.sqrt(attn_depth)
    mlp_den = math.sqrt(mlp_depth)
    return (
        (1.0 / attn_den, math.sqrt(attn_depth - 1) / attn_den),
        (1.0 / mlp_den, math.sqrt(mlp_depth - 1) / mlp_den),
    )


# --------------------------------------------------------------------------
# Torch reference (exact reference math; also the off-TPU fallback)
# --------------------------------------------------------------------------


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``RMSNorm.forward_native`` with a plain ``x * w`` weight, one rounding."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return (xf * weight).to(orig_dtype)


def sandwich_residual_rmsnorm_ref(
    branch: torch.Tensor,
    residual: torch.Tensor,
    sandwich_weight: Optional[torch.Tensor],
    out_weight: Optional[torch.Tensor],
    eps: float,
    scales: tuple[float, float],
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """``BeamSandwichResidualRMSNormOp.forward_native`` verbatim.

    ``branch * s`` and ``residual * s`` are bf16 tensor-by-Python-scalar
    products, which torch computes in fp32 and rounds once; the bf16 add
    likewise rounds once.
    """
    if sandwich_weight is not None:
        branch = rmsnorm_ref(branch, sandwich_weight, eps)
    residual = residual.to(branch.dtype)
    combined = branch * scales[0] + residual * scales[1]
    out = rmsnorm_ref(combined, out_weight, eps) if out_weight is not None else None
    return out, combined


# --------------------------------------------------------------------------
# Pallas kernel
# --------------------------------------------------------------------------


def _round(x: "jax.Array", dtype) -> "jax.Array":
    """Round fp32 to the activation dtype and back, as a bf16 store would."""
    import jax.numpy as jnp

    return x.astype(dtype).astype(jnp.float32)


def _rms_scale(x: "jax.Array", eps: float) -> "jax.Array":
    import jax
    import jax.numpy as jnp

    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return jax.lax.rsqrt(var + eps)


def _sandwich_kernel(
    *refs,
    eps: float,
    branch_scale: float,
    residual_scale: float,
    has_branch: bool,
    has_post: bool,
    has_next: bool,
):
    import jax.numpy as jnp

    refs = list(refs)
    branch_ref = refs.pop(0) if has_branch else None
    residual_ref = refs.pop(0)
    post_w_ref = refs.pop(0) if has_post else None
    next_w_ref = refs.pop(0) if has_next else None
    normed_ref = refs.pop(0) if has_next else None
    combined_ref = refs.pop(0) if has_branch else None
    assert not refs

    act = (normed_ref if normed_ref is not None else combined_ref).dtype
    if has_branch:
        x = branch_ref[...].astype(jnp.float32)
        if has_post:
            w = post_w_ref[...].astype(jnp.float32)
            # (x * rstd) * w, left to right like the reference, one rounding.
            x = _round(x * _rms_scale(x, eps) * w, act)
        # depth_scaled_residual casts the residual to the branch dtype first.
        r = _round(residual_ref[...].astype(jnp.float32), act)
        xs = _round(x * branch_scale, act)
        rs = _round(r * residual_scale, act)
        combined = (xs + rs).astype(act)
        combined_ref[...] = combined.astype(combined_ref.dtype)
        c = combined.astype(jnp.float32)
    else:
        # Stage entry: normed = input_layernorm(residual); residual unchanged.
        c = residual_ref[...].astype(jnp.float32)

    if has_next:
        w2 = next_w_ref[...].astype(jnp.float32)
        normed_ref[...] = (c * _rms_scale(c, eps) * w2).astype(normed_ref.dtype)


def sandwich_residual_rmsnorm_jax(
    branch: "jax.Array | None",
    residual: "jax.Array",
    sandwich_weight: "jax.Array | None",
    out_weight: "jax.Array | None",
    *,
    eps: float,
    branch_scale: float = 1.0,
    residual_scale: float = 1.0,
    interpret: bool = False,
) -> "tuple[jax.Array | None, jax.Array | None]":
    """JAX-level fused transition; returns ``(normed, combined)``.

    ``branch=None`` is the stage-entry mode: only ``normed = rmsnorm(residual)``
    is computed (``combined`` is None and the caller keeps ``residual``).
    ``out_weight=None`` skips ``normed``. Variants are selected at trace time.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    has_branch = branch is not None
    has_post = has_branch and sandwich_weight is not None
    has_next = out_weight is not None
    if not has_branch and not has_next:
        raise ValueError("nothing to compute: no branch and no out_weight")

    m, h = residual.shape
    act_dtype = branch.dtype if has_branch else residual.dtype
    # A block equal to the full row count is always legal; otherwise 64 rows
    # are a multiple of the bf16 (16, 128) tile. A ragged final block computes
    # padded rows whose writes are masked off.
    bm = min(m, _MAX_BLOCK_TOKENS)
    grid = (pl.cdiv(m, bm),)
    row_spec = pl.BlockSpec((bm, h), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, h), lambda i: (0, 0))

    inputs, in_specs = [], []
    if has_branch:
        inputs.append(branch)
        in_specs.append(row_spec)
    inputs.append(residual)
    in_specs.append(row_spec)
    if has_post:
        inputs.append(sandwich_weight.reshape(1, h))
        in_specs.append(w_spec)
    if has_next:
        inputs.append(out_weight.reshape(1, h))
        in_specs.append(w_spec)

    out_shape, out_specs = [], []
    if has_next:
        out_shape.append(jax.ShapeDtypeStruct((m, h), act_dtype))
        out_specs.append(row_spec)
    if has_branch:
        out_shape.append(jax.ShapeDtypeStruct((m, h), act_dtype))
        out_specs.append(row_spec)

    kernel = lambda *refs: _sandwich_kernel(
        *refs,
        eps=eps,
        branch_scale=branch_scale,
        residual_scale=residual_scale,
        has_branch=has_branch,
        has_post=has_post,
        has_next=has_next,
    )
    outs = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="sandwich_residual_rmsnorm",
    )(*inputs)
    outs = list(outs)
    normed = outs.pop(0) if has_next else None
    combined = outs.pop(0) if has_branch else None
    return normed, combined


# --------------------------------------------------------------------------
# torch custom-op registration (one op per variant; no Optional tensor args)
# --------------------------------------------------------------------------

_OPS: dict[str, Callable] = {}
_OPS_LOCK = Lock()


def _build_op(variant: str) -> Callable:
    import jax  # noqa: F401  (force loud failure if JAX is missing)

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    if variant == "transition":

        def _wrapped(
            branch: jax.Array,
            residual: jax.Array,
            sandwich_weight: jax.Array,
            out_weight: jax.Array,
            eps: float,
            branch_scale: float,
            residual_scale: float,
        ) -> tuple[jax.Array, jax.Array]:
            return sandwich_residual_rmsnorm_jax(
                branch,
                residual,
                sandwich_weight,
                out_weight,
                eps=eps,
                branch_scale=branch_scale,
                residual_scale=residual_scale,
            )

        def _fake(branch, residual, sandwich_weight, out_weight, *_):
            return branch.new_empty(branch.shape), branch.new_empty(branch.shape)

    elif variant == "residual_only":

        def _wrapped(
            branch: jax.Array,
            residual: jax.Array,
            sandwich_weight: jax.Array,
            eps: float,
            branch_scale: float,
            residual_scale: float,
        ) -> jax.Array:
            return sandwich_residual_rmsnorm_jax(
                branch,
                residual,
                sandwich_weight,
                None,
                eps=eps,
                branch_scale=branch_scale,
                residual_scale=residual_scale,
            )[1]

        def _fake(branch, residual, sandwich_weight, *_):
            return branch.new_empty(branch.shape)

    elif variant == "rmsnorm":

        def _wrapped(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
            return sandwich_residual_rmsnorm_jax(None, x, None, weight, eps=eps)[0]

        def _fake(x, weight, eps):
            return x.new_empty(x.shape)

    else:  # pragma: no cover
        raise ValueError(variant)

    # Rank-local kernel: no mesh or partition specs needed.
    op = cpu_export_jax_op(f"sglang_tpu_pallas::sandwich_{variant}", _wrapped)
    op.register_fake(_fake)
    return op


def _get_op(variant: str) -> Callable:
    op = _OPS.get(variant)
    if op is not None:
        return op
    with _OPS_LOCK:
        if variant not in _OPS:
            _OPS[variant] = _build_op(variant)
        return _OPS[variant]


def register_tpu_sandwich_residual_norm_ops() -> None:
    """Register every variant before torch.compile traces a model."""
    for variant in ("transition", "residual_only", "rmsnorm"):
        _get_op(variant)


# --------------------------------------------------------------------------
# Dispatch wrappers (the model calls these)
# --------------------------------------------------------------------------


def _check_rows(name: str, t: torch.Tensor, hidden: int | None = None) -> None:
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {t.dtype}.")
    if t.dim() != 2:
        raise ValueError(f"{name} must be 2-D [tokens, hidden], got {tuple(t.shape)}.")
    if t.shape[0] == 0:
        raise ValueError(f"{name} must have at least one token row, got 0.")
    if hidden is not None and t.shape[1] != hidden:
        raise ValueError(f"{name} hidden {t.shape[1]} != {hidden}.")


def _check_weight(name: str, w: torch.Tensor | None, hidden: int) -> None:
    if w is not None and tuple(w.shape) != (hidden,):
        raise ValueError(f"{name} must be [{hidden}], got {tuple(w.shape)}.")


def tpu_sandwich_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    sandwich_weight: Optional[torch.Tensor],
    out_weight: Optional[torch.Tensor],
    eps: float,
    scales: tuple[float, float],
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Fused post-norm + depth-scaled residual + next norm. See module doc."""
    _check_rows("branch", branch)
    hidden = branch.shape[1]
    if tuple(residual.shape) != tuple(branch.shape):
        raise ValueError(
            f"residual {tuple(residual.shape)} must match branch {tuple(branch.shape)}."
        )
    _check_weight("sandwich_weight", sandwich_weight, hidden)
    _check_weight("out_weight", out_weight, hidden)
    if branch.device.type != "tpu" or sandwich_weight is None:
        # Off TPU, run the reference. Beam always has the post-norm; the
        # no-post-norm variant would need a fourth op for no caller.
        return sandwich_residual_rmsnorm_ref(
            branch, residual, sandwich_weight, out_weight, eps, scales
        )
    branch_scale, residual_scale = (float(s) for s in scales)
    if out_weight is None:
        combined = _get_op("residual_only")(
            branch, residual, sandwich_weight, eps, branch_scale, residual_scale
        )
        return None, combined
    normed, combined = _get_op("transition")(
        branch,
        residual,
        sandwich_weight,
        out_weight,
        eps,
        branch_scale,
        residual_scale,
    )
    return normed, combined


def tpu_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain ``x * w`` RMSNorm (stage entry: the first ``input_layernorm``)."""
    _check_rows("x", x)
    _check_weight("weight", weight, x.shape[1])
    if x.device.type != "tpu":
        return rmsnorm_ref(x, weight, eps)
    return _get_op("rmsnorm")(x, weight, eps)
