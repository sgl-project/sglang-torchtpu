# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant log-prob of a target token: log-softmax over ``_LP_COLS``-column tiles, in order."""

# NOTE: do *not* add ``from __future__ import annotations`` here: the torch_tpu
# Pallas bridge reads the wrapped functions' annotations (``jax.Array``).

import functools

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.rl.true_on_policy.common import register, row_block, save_inputs

_LP_ROWS, _LP_COLS = 1024, 2048


def _lp_kernel(
    logits_ref, temp_ref, target_ref, o_ref, m_ref, s_ref, pick_ref, *, vocab, nc
):
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        s_ref[...] = jnp.zeros_like(s_ref)
        pick_ref[...] = jnp.zeros_like(pick_ref)

    col = j * _LP_COLS + jax.lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1)
    x = logits_ref[...].astype(jnp.float32) / temp_ref[...]
    x = jnp.where(col < vocab, x, -jnp.inf)
    m_new = jnp.maximum(m_ref[...], jnp.max(x, axis=1, keepdims=True))
    # While every logit so far is -inf the max is -inf: rescale against 0 there, not -inf - -inf = NaN.
    m_safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
    s_ref[...] = s_ref[...] * jnp.exp(m_ref[...] - m_safe) + jnp.sum(
        jnp.exp(x - m_safe), axis=1, keepdims=True
    )
    m_ref[...] = m_new
    pick_ref[...] += jnp.sum(
        jnp.where(col == target_ref[...], x, 0.0), axis=1, keepdims=True
    )

    @pl.when(j == nc - 1)
    def _():
        o_ref[...] = pick_ref[...] - (m_ref[...] + jnp.log(s_ref[...]))


def _jax_logprob(
    logits: jax.Array, temperature: jax.Array, target: jax.Array
) -> jax.Array:
    """log_softmax(logits.float() / temperature)[target] per row: logits [M, V], temperature [M] f32, target [M]."""
    m, v = logits.shape
    nc = pl.cdiv(v, _LP_COLS)
    rows = row_block(m, _LP_ROWS, 16)
    row = lambda i, j: (i, 0)  # noqa: E731
    out = pl.pallas_call(
        functools.partial(_lp_kernel, vocab=v, nc=nc),
        grid=(pl.cdiv(m, rows), nc),
        in_specs=[
            pl.BlockSpec((rows, _LP_COLS), lambda i, j: (i, j)),
            pl.BlockSpec((rows, 1), row),
            pl.BlockSpec((rows, 1), row),
        ],
        out_specs=pl.BlockSpec((rows, 1), row),
        out_shape=jax.ShapeDtypeStruct((m, 1), jnp.float32),
        scratch_shapes=[pltpu.VMEM((rows, 1), jnp.float32)] * 3,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
    )(
        logits,
        temperature.reshape(m, 1).astype(jnp.float32),
        target.reshape(m, 1).astype(jnp.int32),
    )
    return out[:, 0]


logprob = register("logprob", _jax_logprob)


def _logprob_grad(ctx, grad):
    logits, temperature, target = ctx.saved_tensors
    t = temperature.float()[:, None]
    # (onehot(target) - softmax) * grad / T, built in the softmax's own [M, V] fp32 buffer.
    g = torch.softmax(logits.float() / t, dim=-1).neg_()
    g.scatter_add_(1, target.long()[:, None], torch.ones_like(t))
    return g.mul_(grad.float()[:, None] / t).to(logits.dtype), None, None


logprob.register_autograd(_logprob_grad, setup_context=save_inputs)
