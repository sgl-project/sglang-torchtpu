# SPDX-License-Identifier: Apache-2.0

"""One-pass expert-TP fused-RS post-kernel combine for models with a post-expert norm.

Replaces three XLA fusions (ep-shard sum, per-slot RMS variance, normalize +
top-k
weighted sum) with one Pallas pass over the kernel's partial rows. Every bf16
cast of the JAX reference is
kept (torch_tpu compiles without XLA excess precision), so results differ only
in
f32 summation order.
"""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Tokens per grid step: 32 overflowed scoped VMEM at a 4096-wide hidden.
_TOKENS_PER_BLOCK = 16


def _combine_kernel(x_ref, gain_ref, w_ref, o_ref, *, tokens, topk, ep, eps):
    rows = tokens * topk
    hidden = x_ref.shape[1] * x_ref.shape[2]
    x = x_ref[...].reshape(rows, ep, *x_ref.shape[1:])
    acc = x[:, 0].astype(jnp.float32)
    for e in range(1, ep):
        acc = acc + x[:, e].astype(jnp.float32)
    y = acc.astype(x_ref.dtype).astype(jnp.float32) * gain_ref[...][None]
    sum_sq = jnp.sum(jnp.sum(y * y, axis=1), axis=-1, keepdims=True)
    z = (y * jax.lax.rsqrt(sum_sq / hidden + eps)[:, :, None]).astype(x_ref.dtype)
    z = z.astype(jnp.float32).reshape(tokens, topk, *x_ref.shape[1:])
    w = w_ref[...].astype(jnp.float32)
    out = None
    for k in range(topk):
        prod = (z[:, k] * w[:, k : k + 1, None]).astype(o_ref.dtype)
        out = (
            prod.astype(jnp.float32) if out is None else out + prod.astype(jnp.float32)
        )
    o_ref[...] = out.astype(o_ref.dtype)


def supports_fused_post_combine(out_buf, chunk_size: int, norm_size: int) -> bool:
    """Tiled expert-TP output, unpadded hidden, and a token count the blocks tile."""
    if out_buf.ndim != 3 or out_buf.shape[1] * out_buf.shape[2] != norm_size:
        return False
    return chunk_size % 8 == 0 or chunk_size < 8


def fused_post_combine(
    out_buf: jax.Array,  # [chunk * topk * ep, H1, H2] partial rows, token-major
    post_norm_weight: jax.Array,  # [H1 * H2], zero-centered gain
    topk_weights: jax.Array,  # [chunk, topk]
    *,
    chunk_size: int,
    topk: int,
    ep_size: int,
    eps: float,
) -> jax.Array:
    """[chunk, H1 * H2]: per-slot ep sum, post-expert RMS norm, top-k weighted sum."""
    h1, h2 = out_buf.shape[1:]
    tokens = min(_TOKENS_PER_BLOCK, chunk_size)
    while chunk_size % tokens:
        tokens //= 2
    if tokens % 8 and tokens != chunk_size:
        tokens = chunk_size
    gain = (post_norm_weight.astype(jnp.float32) + 1.0).reshape(h1, h2)
    out_dtype = jnp.result_type(out_buf.dtype, topk_weights.dtype)
    out = pl.pallas_call(
        functools.partial(
            _combine_kernel, tokens=tokens, topk=topk, ep=ep_size, eps=eps
        ),
        grid=(chunk_size // tokens,),
        in_specs=[
            pl.BlockSpec((tokens * topk * ep_size, h1, h2), lambda i: (i, 0, 0)),
            pl.BlockSpec((h1, h2), lambda i: (0, 0)),
            pl.BlockSpec((tokens, topk), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tokens, h1, h2), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((chunk_size, h1, h2), out_dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(out_buf, gain, topk_weights)
    return out.reshape(chunk_size, h1 * h2)


def _pair_combine_kernel(x_ref, gain_ref, w_ref, o_ref, *, eps):
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


def pair_post_combine(
    recv: jax.Array,  # [blocks * rows, H1, H2] pair-exchange receive blocks, rank order
    post_norm_weight: jax.Array,  # [H1 * H2], zero-centered gain
    rank_weight: jax.Array,  # [rows] f32: the top-k weight of each rank's (token, slot)
    *,
    rows: int,
    eps: float,
    rows_per_block: int = 256,
) -> jax.Array:
    """[rows, H1, H2] bf16: block sum, post-expert RMS norm and top-k weight, per rank row."""
    h1, h2 = recv.shape[1:]
    blocks = recv.shape[0] // rows
    rb = min(rows_per_block, rows)
    x = recv.reshape(blocks, rows, h1, h2)
    gain = (post_norm_weight.astype(jnp.float32) + 1.0).reshape(h1, h2)
    w = jnp.broadcast_to(rank_weight.astype(jnp.float32)[:, None], (rows, 128))
    return pl.pallas_call(
        functools.partial(_pair_combine_kernel, eps=eps),
        grid=(rows // rb,),
        in_specs=[
            pl.BlockSpec((blocks, rb, h1, h2), lambda i: (0, i, 0, 0)),
            pl.BlockSpec((h1, h2), lambda i: (0, 0)),
            pl.BlockSpec((rb, 128), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((rb, h1, h2), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, h1, h2), recv.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, gain, w)


def _pair_combine_scatter_kernel(
    tok_ref,
    x_hbm_ref,
    gain_ref,
    w_ref,
    o_hbm_ref,
    x_buf_ref,
    acc_ref,
    row_buf_ref,
    in_sem,
    out_sem,
    *,
    num_steps: int,
    rb: int,
    out_b: int,
    chunk_size: int,
    blocks: int,
    eps: float,
):
    """Pallas kernel: block sum + RMSNorm + top-k weight + VMEM scatter-add."""
    h1, h2 = x_hbm_ref.shape[2], x_hbm_ref.shape[3]
    hidden = h1 * h2

    def _start_in(slot, step):
        pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[:, pl.ds(step * rb, rb), :, :],
            dst_ref=x_buf_ref.at[slot],
            sem=in_sem.at[slot],
        ).start()

    def _wait_in(slot):
        pltpu.make_async_copy(
            src_ref=x_buf_ref.at[slot],
            dst_ref=x_buf_ref.at[slot],
            sem=in_sem.at[slot],
        ).wait()

    _start_in(0, 0)
    if num_steps > 1:
        _start_in(1, 1)

    zero_tile = jnp.zeros((out_b, h1, h2), jnp.float32)

    def _zero_acc(c, carry):
        acc_ref[pl.ds(c * out_b, out_b)] = zero_tile
        return carry

    jax.lax.fori_loop(0, chunk_size // out_b, _zero_acc, 0)

    def _step(step, carry):
        slot = step % 2
        _wait_in(slot)
        acc = x_buf_ref[slot, 0].astype(jnp.float32)
        for e in range(1, blocks):
            acc = acc + x_buf_ref[slot, e].astype(jnp.float32)

        @pl.when(step + 2 < num_steps)
        def _():
            _start_in(slot, step + 2)

        y = acc.astype(x_buf_ref.dtype).astype(jnp.float32) * gain_ref[...][None]
        sum_sq = jnp.sum(jnp.sum(y * y, axis=1), axis=-1, keepdims=True)
        z = (y * jax.lax.rsqrt(sum_sq / hidden + eps)[:, :, None]).astype(
            x_buf_ref.dtype
        )
        w = w_ref[pl.ds(step * rb, rb), :1]
        row_buf_ref[...] = (z.astype(jnp.float32) * w[:, :, None]).astype(
            row_buf_ref.dtype
        )

        base = step * rb

        def _scatter8(k, c):
            for u in range(8):
                r = k * 8 + u
                t = tok_ref[base + r]
                acc_ref[t] = acc_ref[t] + row_buf_ref[r].astype(jnp.float32)
            return c

        jax.lax.fori_loop(0, rb // 8, _scatter8, 0)
        return carry

    jax.lax.fori_loop(0, num_steps, _step, 0)

    def _store_acc(c, carry):
        row_buf_ref[pl.ds(0, out_b)] = acc_ref[pl.ds(c * out_b, out_b)].astype(
            row_buf_ref.dtype
        )
        pltpu.make_async_copy(
            src_ref=row_buf_ref.at[pl.ds(0, out_b)],
            dst_ref=o_hbm_ref.at[pl.ds(c * out_b, out_b)],
            sem=out_sem.at[0],
        ).start()
        pltpu.make_async_copy(
            src_ref=row_buf_ref.at[pl.ds(0, out_b)],
            dst_ref=row_buf_ref.at[pl.ds(0, out_b)],
            sem=out_sem.at[0],
        ).wait()
        return carry

    jax.lax.fori_loop(0, chunk_size // out_b, _store_acc, 0)


def pair_post_combine_scatter(
    recv: jax.Array,  # [blocks * rows, H1, H2] pair-exchange receive blocks
    post_norm_weight: jax.Array,  # [H1 * H2], zero-centered gain
    rank_weight: jax.Array,  # [rows] f32: top-k weight per (token, slot)
    rank_token: jax.Array,  # [rows] i32: local token index in [0, chunk_size)
    *,
    rows: int,
    chunk_size: int,
    eps: float,
    out_dtype: jnp.dtype,
    rows_per_block: int = 128,
) -> jax.Array:
    """[chunk_size, H1 * H2]: block sum, RMS norm, top-k weight, scatter-add."""
    h1, h2 = recv.shape[1:]
    blocks = recv.shape[0] // rows
    rb = min(rows_per_block, rows)
    while rows % rb or rb % 8:
        rb //= 2
    out_b = min(rb, chunk_size)
    while chunk_size % out_b or out_b % 8:
        out_b //= 2
    num_steps = rows // rb
    vmem_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.95)
    x = recv.reshape(blocks, rows, h1, h2)
    gain = (post_norm_weight.astype(jnp.float32) + 1.0).reshape(h1, h2)
    w = jnp.broadcast_to(rank_weight.astype(jnp.float32)[:, None], (rows, 128))
    tok = rank_token.astype(jnp.int32)
    out = pl.pallas_call(
        functools.partial(
            _pair_combine_scatter_kernel,
            num_steps=num_steps,
            rb=rb,
            out_b=out_b,
            chunk_size=chunk_size,
            blocks=blocks,
            eps=eps,
        ),
        out_shape=jax.ShapeDtypeStruct((chunk_size, h1, h2), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, blocks, rb, h1, h2), recv.dtype),
                pltpu.VMEM((chunk_size, h1, h2), jnp.float32),
                pltpu.VMEM((rb, h1, h2), out_dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        name=f"pair_post_combine_scatter-C_{chunk_size}-R_{rows}",
    )(tok, x, gain, w)
    return out.reshape(chunk_size, h1 * h2)
