# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant RMSNorm: a row is one reduction, whatever the call's row count."""

# NOTE: do *not* add ``from __future__ import annotations`` here: the torch_tpu
# Pallas bridge reads the wrapped functions' annotations (``jax.Array``).

import functools

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.rl.true_on_policy.common import register, row_block

# Elements per norm block (rows x row width); measured up to a 5120-wide row.
_NORM_BLOCK_ELEMS = 1 << 20


def _norm_kernel(*refs, eps, has_residual):
    if has_residual:
        x_ref, r_ref, w_ref, o_ref, ro_ref = refs
        s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
    else:
        x_ref, w_ref, o_ref = refs
        s = x_ref[...].astype(jnp.float32)
    var = jnp.mean(s * s, axis=1, keepdims=True)
    o_ref[...] = (s * jax.lax.rsqrt(var + eps) * w_ref[...].astype(jnp.float32)).astype(
        o_ref.dtype
    )


def _rms_norm_call(args, *, eps, has_residual):
    """RMSNorm as RMSNorm.forward_native computes it: normalize the fp32 sum, round once to x's dtype."""
    x = args[0]
    m, h = x.shape
    # The largest power-of-two row count within the block budget.
    cap = 1 << ((_NORM_BLOCK_ELEMS // h).bit_length() - 1)
    rows = row_block(m, cap, 16)
    blk = pl.BlockSpec((rows, h), lambda i: (i, 0))
    n_in = 2 if has_residual else 1
    out_shape = jax.ShapeDtypeStruct((m, h), x.dtype)
    return pl.pallas_call(
        functools.partial(_norm_kernel, eps=eps, has_residual=has_residual),
        grid=(pl.cdiv(m, rows),),
        in_specs=[blk] * n_in + [pl.BlockSpec((1, h), lambda i: (0, 0))],
        out_specs=[blk, blk] if has_residual else blk,
        out_shape=[out_shape, out_shape] if has_residual else out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(*args[:n_in], args[n_in].reshape(1, h))


# eps is a static (non-tensor) op argument: one program per eps value.
def _jax_rms_norm(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
    return _rms_norm_call((x, w), eps=eps, has_residual=False)


def _jax_rms_norm_residual(
    x: jax.Array, r: jax.Array, w: jax.Array, eps: float
) -> tuple[jax.Array, jax.Array]:
    return tuple(_rms_norm_call((x, r, w), eps=eps, has_residual=True))


_rms_norm = register("rms_norm", _jax_rms_norm)
_rms_norm_residual = register("rms_norm_residual", _jax_rms_norm_residual)


def rms_norm(x: torch.Tensor, residual, weight: torch.Tensor, eps: float):
    """[M, H] x (and residual) -> out, or (out, new residual) when a residual is given."""
    if residual is None:
        return _rms_norm(x, weight, eps)
    return _rms_norm_residual(x, residual, weight, eps)


def _norm_input_grad(s, w, grad_out, eps):
    """d/ds of s * rsqrt(mean(s^2) + eps) * w, all fp32; returns (grad_s, grad_w)."""
    g = grad_out.float()
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    s_hat = s * rstd
    gw = g * w.float()
    grad_s = rstd * (gw - s_hat * (gw * s_hat).mean(-1, keepdim=True))
    return grad_s, (g * s_hat).sum(0)


def _norm_setup(ctx, inputs, output):
    # eps, the last input, is a float: kept on ctx, and its gradient slot below is None.
    ctx.save_for_backward(*inputs[:-1])
    ctx.eps = inputs[-1]


def _norm_grad(ctx, grad):
    x, w = ctx.saved_tensors
    grad_s, grad_w = _norm_input_grad(x.float(), w, grad, ctx.eps)
    return grad_s.to(x.dtype), grad_w.to(w.dtype), None


def _norm_residual_grad(ctx, grad_out, grad_residual):
    x, r, w = ctx.saved_tensors
    grad_s, grad_w = _norm_input_grad(x.float() + r.float(), w, grad_out, ctx.eps)
    # The new residual is bf16(s); its gradient passes straight through the rounding.
    grad_s = grad_s + grad_residual.float()
    return grad_s.to(x.dtype), grad_s.to(r.dtype), grad_w.to(w.dtype), None


_rms_norm.register_autograd(_norm_grad, setup_context=_norm_setup)
_rms_norm_residual.register_autograd(_norm_residual_grad, setup_context=_norm_setup)
