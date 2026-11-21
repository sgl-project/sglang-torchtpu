# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant matmul: fp32 accumulation over K in ``_MM_BK`` steps, in order."""

# No ``from __future__ import annotations``: the Pallas bridge reads the annotations.

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.rl.true_on_policy.common import register, row_block, save_inputs

_MM_BM, _MM_BN, _MM_BK = 1024, 2048, 512
_MM_VMEM_BYTES = 48 << 20


# nk is passed in: pl.num_programs returned a stale extent here.
def _mm_kernel(x_ref, w_ref, o_ref, acc_ref, *, nk):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    @pl.when(k == nk - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _jax_matmul(x: jax.Array, w: jax.Array) -> jax.Array:
    """x [M, K] @ w [N, K]^T -> [M, N] in x's dtype; fp32 accumulation over K in ``_MM_BK`` steps, in order."""
    m, k = x.shape
    n = w.shape[0]
    if k % _MM_BK:
        raise ValueError(f"True on-policy matmul needs K % {_MM_BK} == 0, got K={k}")
    bm = row_block(m, _MM_BM, 128)
    return pl.pallas_call(
        functools.partial(_mm_kernel, nk=k // _MM_BK),
        grid=(pl.cdiv(m, bm), pl.cdiv(n, _MM_BN), k // _MM_BK),
        in_specs=[
            pl.BlockSpec((bm, _MM_BK), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((_MM_BN, _MM_BK), lambda i, j, kk: (j, kk)),
        ],
        out_specs=pl.BlockSpec((bm, _MM_BN), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=[pltpu.VMEM((bm, _MM_BN), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=_MM_VMEM_BYTES,
        ),
    )(x, w)


matmul = register("matmul", _jax_matmul)


def _matmul_grad(ctx, grad):
    x, w = ctx.saved_tensors
    return grad.mm(w.to(grad.dtype)).to(x.dtype), grad.t().mm(x.to(grad.dtype)).to(
        w.dtype
    )


matmul.register_autograd(_matmul_grad, setup_context=save_inputs)
