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


def _norm_kernel(*refs, eps, has_residual, gemma):
    if has_residual:
        x_ref, r_ref, w_ref, o_ref, ro_ref = refs
        s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
        if gemma:
            # GemmaRMSNorm adds in the activation dtype, so it normalizes the rounded sum.
            s = s.astype(ro_ref.dtype).astype(jnp.float32)
    else:
        x_ref, w_ref, o_ref = refs
        s = x_ref[...].astype(jnp.float32)
    w = w_ref[...].astype(jnp.float32)
    if gemma:
        w = 1.0 + w
    var = jnp.mean(s * s, axis=1, keepdims=True)
    o_ref[...] = (s * jax.lax.rsqrt(var + eps) * w).astype(o_ref.dtype)


def _gated_norm_kernel(x_ref, z_ref, w_ref, o_ref, *, eps):
    s = x_ref[...].astype(jnp.float32)
    var = jnp.mean(s * s, axis=1, keepdims=True)
    normed = s * jax.lax.rsqrt(var + eps) * w_ref[...].astype(jnp.float32)
    gate = jax.nn.silu(z_ref[...].astype(jnp.float32))
    o_ref[...] = (normed * gate).astype(o_ref.dtype)


def _norm_call(args, kernel, n_out=1):
    """One norm kernel over row blocks: args are the [M, H] inputs, then the [H] weight."""
    *row_inputs, w = args
    x = row_inputs[0]
    m, h = x.shape
    # The largest power-of-two row count within the block budget.
    cap = 1 << ((_NORM_BLOCK_ELEMS // h).bit_length() - 1)
    rows = row_block(m, cap, 16)
    blk = pl.BlockSpec((rows, h), lambda i: (i, 0))
    out_shape = jax.ShapeDtypeStruct((m, h), x.dtype)
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(m, rows),),
        in_specs=[blk] * len(row_inputs) + [pl.BlockSpec((1, h), lambda i: (0, 0))],
        out_specs=blk if n_out == 1 else [blk] * n_out,
        out_shape=out_shape if n_out == 1 else [out_shape] * n_out,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(*row_inputs, w.reshape(1, h))


# eps and gemma are static (non-tensor) op arguments: one program per value.
def _jax_rms_norm(x: jax.Array, w: jax.Array, eps: float, gemma: bool) -> jax.Array:
    kernel = functools.partial(_norm_kernel, eps=eps, has_residual=False, gemma=gemma)
    return _norm_call((x, w), kernel)


def _jax_rms_norm_residual(
    x: jax.Array, r: jax.Array, w: jax.Array, eps: float, gemma: bool
) -> tuple[jax.Array, jax.Array]:
    kernel = functools.partial(_norm_kernel, eps=eps, has_residual=True, gemma=gemma)
    return tuple(_norm_call((x, r, w), kernel, n_out=2))


def _jax_gated_rms_norm(
    x: jax.Array, z: jax.Array, w: jax.Array, eps: float
) -> jax.Array:
    return _norm_call((x, z, w), functools.partial(_gated_norm_kernel, eps=eps))


_rms_norm = register("rms_norm", _jax_rms_norm)
_rms_norm_residual = register("rms_norm_residual", _jax_rms_norm_residual)
_gated_rms_norm = register("gated_rms_norm", _jax_gated_rms_norm)


def rms_norm(
    x: torch.Tensor, residual, weight: torch.Tensor, eps: float, *, gemma=False
):
    """[M, H] x (and residual) -> out, or (out, new residual) when a residual is given.

    As RMSNorm.forward_native rounds: the fp32 sum is normalized and rounded once. With gemma, as
    GemmaRMSNorm.forward_native: the weight is 1 + weight, and the sum is rounded before it is normalized.
    """
    if residual is None:
        return _rms_norm(x, weight, eps, gemma)
    return _rms_norm_residual(x, residual, weight, eps, gemma)


def gated_rms_norm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float):
    """[M, H] x and gate input z -> rms_norm(x) * silu(z), as the pure-torch RMSNormGated computes it."""
    return _gated_rms_norm(x, z, weight, eps)


def _norm_input_grad(s, w, grad_out, eps):
    """d/ds of s * rsqrt(mean(s^2) + eps) * w, all fp32; returns (grad_s, grad_w)."""
    g = grad_out.float()
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    s_hat = s * rstd
    gw = g * w.float()
    grad_s = rstd * (gw - s_hat * (gw * s_hat).mean(-1, keepdim=True))
    return grad_s, (g * s_hat).sum(0)


def _norm_setup(ctx, inputs, output):
    # The trailing static arguments (eps, gemma) are kept on ctx; their gradient slots below are None.
    tensors = [arg for arg in inputs if isinstance(arg, torch.Tensor)]
    ctx.save_for_backward(*tensors)
    ctx.static = inputs[len(tensors) :]


def _norm_weight(w, gemma):
    return 1.0 + w.float() if gemma else w.float()


def _norm_grad(ctx, grad):
    x, w = ctx.saved_tensors
    eps, gemma = ctx.static
    grad_s, grad_w = _norm_input_grad(x.float(), _norm_weight(w, gemma), grad, eps)
    return grad_s.to(x.dtype), grad_w.to(w.dtype), None, None


def _norm_residual_grad(ctx, grad_out, grad_residual):
    x, r, w = ctx.saved_tensors
    eps, gemma = ctx.static
    s = x.float() + r.float()
    if gemma:
        s = s.to(x.dtype).float()
    grad_s, grad_w = _norm_input_grad(s, _norm_weight(w, gemma), grad_out, eps)
    # The new residual is bf16(s); its gradient passes straight through the rounding.
    grad_s = grad_s + grad_residual.float()
    return grad_s.to(x.dtype), grad_s.to(r.dtype), grad_w.to(w.dtype), None, None


def _gated_norm_grad(ctx, grad):
    x, z, w = ctx.saved_tensors
    (eps,) = ctx.static
    xf, zf, g = x.float(), z.float(), grad.float()
    sig = torch.sigmoid(zf)
    grad_x, grad_w = _norm_input_grad(xf, w.float(), g * zf * sig, eps)
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    # d silu(z) / dz = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    grad_z = g * normed * sig * (1.0 + zf * (1.0 - sig))
    return grad_x.to(x.dtype), grad_z.to(z.dtype), grad_w.to(w.dtype), None


_rms_norm.register_autograd(_norm_grad, setup_context=_norm_setup)
_rms_norm_residual.register_autograd(_norm_residual_grad, setup_context=_norm_setup)
_gated_rms_norm.register_autograd(_gated_norm_grad, setup_context=_norm_setup)
