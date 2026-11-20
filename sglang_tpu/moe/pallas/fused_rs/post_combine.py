# SPDX-License-Identifier: Apache-2.0

"""Two-level fused-RS post-kernel combine for models with a post-expert norm.

Sums the receive blocks, applies the post-expert RMS norm and each row's top-k weight in
one Pallas pass. Every bf16 cast of the JAX reference is kept (torch_tpu compiles without
XLA excess precision), so results differ only in f32 summation order.
"""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Receive-block bytes per grid step (scoped VMEM holds 1.5x this).
_BLOCK_BYTES = 8 << 20


def _twolevel_combine_kernel(x_ref, gain_ref, w_ref, o_ref, *, eps):
    blocks = x_ref.shape[0]
    hidden = x_ref.shape[2] * x_ref.shape[3]
    acc = x_ref[0].astype(jnp.float32)
    for e in range(1, blocks):
        acc = acc + x_ref[e].astype(jnp.float32)
    y = acc.astype(x_ref.dtype).astype(jnp.float32) * gain_ref[...][None]
    sum_sq = jnp.sum(jnp.sum(y * y, axis=1), axis=-1, keepdims=True)
    z = (y * jax.lax.rsqrt(sum_sq / hidden + eps)[:, :, None]).astype(x_ref.dtype)
    w = w_ref[...][:, :1]
    o_ref[...] = (z.astype(jnp.float32) * w[:, :, None]).astype(o_ref.dtype)


def twolevel_post_combine(
    recv: jax.Array,  # [blocks * rows, H1, H2] two-level receive blocks, rank order
    post_norm_weight: jax.Array,  # [H1 * H2], zero-centered gain
    rank_weight: jax.Array,  # [rows] f32: the top-k weight of each rank's (token, slot)
    *,
    rows: int,
    eps: float,
) -> jax.Array:
    """[rows, H1, H2] bf16: block sum, post-expert RMS norm and top-k weight, per rank row."""
    h1, h2 = recv.shape[1:]
    blocks = recv.shape[0] // rows
    # Up to 256 rows per step, fewer when the receive blocks would not fit VMEM; a
    # partial last step writes only the rows that exist.
    rb = 256
    while (
        rb > 8 and blocks * rb * h1 * h2 * recv.dtype.itemsize > _BLOCK_BYTES * 3 // 2
    ):
        rb //= 2
    rb = min(rb, rows)
    x = recv.reshape(blocks, rows, h1, h2)
    gain = (post_norm_weight.astype(jnp.float32) + 1.0).reshape(h1, h2)
    w = jnp.broadcast_to(rank_weight.astype(jnp.float32)[:, None], (rows, 128))
    return pl.pallas_call(
        functools.partial(_twolevel_combine_kernel, eps=eps),
        grid=(pl.cdiv(rows, rb),),
        in_specs=[
            pl.BlockSpec((blocks, rb, h1, h2), lambda i: (0, i, 0, 0)),
            pl.BlockSpec((h1, h2), lambda i: (0, 0)),
            pl.BlockSpec((rb, 128), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((rb, h1, h2), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, h1, h2), recv.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, gain, w)
