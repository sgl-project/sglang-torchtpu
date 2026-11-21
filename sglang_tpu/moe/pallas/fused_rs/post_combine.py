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
# Share of VMEM the scatter combine may take; the rest is Mosaic's internal scratch.
_SCATTER_VMEM_FRACTION = 0.75


def _normed_weighted_rows(block, blocks, gain, w, *, eps, dtype):
    """[rb, H1, H2] in dtype: sum of block(0..blocks-1), post-expert RMS norm, weight w."""
    acc = block(0).astype(jnp.float32)
    for e in range(1, blocks):
        acc = acc + block(e).astype(jnp.float32)
    hidden = acc.shape[1] * acc.shape[2]
    y = acc.astype(dtype).astype(jnp.float32) * gain[None]
    sum_sq = jnp.sum(jnp.sum(y * y, axis=1), axis=-1, keepdims=True)
    z = (y * jax.lax.rsqrt(sum_sq / hidden + eps)[:, :, None]).astype(dtype)
    return (z.astype(jnp.float32) * w[:, :, None]).astype(dtype)


def _twolevel_combine_kernel(x_ref, gain_ref, w_ref, o_ref, *, eps):
    o_ref[...] = _normed_weighted_rows(
        lambda e: x_ref[e],
        x_ref.shape[0],
        gain_ref[...],
        w_ref[...][:, :1],
        eps=eps,
        dtype=x_ref.dtype,
    ).astype(o_ref.dtype)


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


def _pow2_divisor(n: int, cap: int) -> int:
    """Largest power of two <= cap that divides n and is a multiple of 8 (0 if none)."""
    b = cap
    while b >= 8 and n % b:
        b //= 2
    return b if b >= 8 else 0


def _scatter_tiles(
    *, rows: int, chunk: int, blocks: int, h: int, dtype, out_dtype
) -> tuple[int, int, int] | None:
    """(rows per step, rows per output store, VMEM bytes) of the scatter combine, or None
    when its f32 [chunk, H] accumulator and buffers do not fit VMEM."""
    out_b = _pow2_divisor(chunk, 256)
    if not out_b:
        return None
    budget = int(pltpu.get_tpu_info().vmem_capacity_bytes * _SCATTER_VMEM_FRACTION)
    fixed = (
        chunk * h * 4  # f32 accumulator
        + rows * 128 * 4  # lane-broadcast weights
        + h * 4  # gain
        + out_b * h * jnp.dtype(out_dtype).itemsize
    )
    rb = _pow2_divisor(rows, 128)
    while rb >= 8:
        need = fixed + (2 * blocks + 1) * rb * h * jnp.dtype(dtype).itemsize
        if need <= budget:
            return rb, out_b, need
        rb //= 2
    return None


def supports_twolevel_post_combine_scatter(
    recv: jax.Array, *, rows: int, chunk: int, out_dtype
) -> bool:
    """Whether twolevel_post_combine_scatter fits this chunk."""
    if recv.ndim != 3 or rows % 8:
        return False
    h = recv.shape[1] * recv.shape[2]
    tiles = _scatter_tiles(
        rows=rows,
        chunk=chunk,
        blocks=recv.shape[0] // rows,
        h=h,
        dtype=recv.dtype,
        out_dtype=out_dtype,
    )
    return tiles is not None


def _twolevel_combine_scatter_kernel(
    token_ref,  # SMEM [rows] i32: this rank's token of each rank row
    x_hbm_ref,  # HBM [blocks, rows, H1, H2]
    gain_ref,  # VMEM [H1, H2] f32
    w_ref,  # VMEM [rows, 128] f32
    o_hbm_ref,  # HBM [chunk, H1, H2]
    x_buf_ref,  # VMEM [2, blocks, rb, H1, H2]
    acc_ref,  # VMEM [chunk, H1, H2] f32
    row_buf_ref,  # VMEM [rb, H1, H2] recv dtype
    out_buf_ref,  # VMEM [out_b, H1, H2] out dtype
    in_sem,
    out_sem,
    *,
    eps: float,
):
    blocks, rows = x_hbm_ref.shape[:2]
    rb = row_buf_ref.shape[0]
    out_b = out_buf_ref.shape[0]
    chunk = acc_ref.shape[0]
    num_steps = rows // rb

    def _in_copy(slot, step):
        return pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[:, pl.ds(step * rb, rb), :, :],
            dst_ref=x_buf_ref.at[slot],
            sem=in_sem.at[slot],
        )

    _in_copy(0, 0).start()
    if num_steps > 1:
        _in_copy(1, 1).start()

    def _zero(c, carry):
        acc_ref[pl.ds(c * out_b, out_b)] = jnp.zeros(
            (out_b, *acc_ref.shape[1:]), jnp.float32
        )
        return carry

    jax.lax.fori_loop(0, chunk // out_b, _zero, 0)

    def _step(step, carry):
        slot = step % 2
        _in_copy(slot, step).wait()
        row_buf_ref[...] = _normed_weighted_rows(
            lambda e: x_buf_ref[slot, e],
            blocks,
            gain_ref[...],
            w_ref[pl.ds(step * rb, rb), :1],
            eps=eps,
            dtype=row_buf_ref.dtype,
        )

        @pl.when(step + 2 < num_steps)
        def _():
            _in_copy(slot, step + 2).start()

        base = step * rb

        def _add8(k, c):
            for u in range(8):
                r = k * 8 + u
                t = token_ref[base + r]
                acc_ref[t] = acc_ref[t] + row_buf_ref[r].astype(jnp.float32)
            return c

        jax.lax.fori_loop(0, rb // 8, _add8, 0)
        return carry

    jax.lax.fori_loop(0, num_steps, _step, 0)

    def _store(c, carry):
        out_buf_ref[...] = acc_ref[pl.ds(c * out_b, out_b)].astype(out_buf_ref.dtype)
        copy = pltpu.make_async_copy(
            src_ref=out_buf_ref,
            dst_ref=o_hbm_ref.at[pl.ds(c * out_b, out_b)],
            sem=out_sem.at[0],
        )
        copy.start()
        copy.wait()
        return carry

    jax.lax.fori_loop(0, chunk // out_b, _store, 0)


def twolevel_post_combine_scatter(
    recv: jax.Array,  # [blocks * rows, H1, H2] two-level receive blocks, rank order
    post_norm_weight: jax.Array,  # [H1 * H2], zero-centered gain
    rank_weight: jax.Array,  # [rows] f32: the top-k weight of each rank's (token, slot)
    rank_token: jax.Array,  # [rows] i32: each rank's token, in [0, chunk)
    *,
    rows: int,
    chunk: int,
    eps: float,
    out_dtype,
) -> jax.Array:
    """[chunk, H1, H2]: twolevel_post_combine's rows, summed per token in an f32 VMEM
    accumulator (one pass, no [chunk, rows] 0/1 matmul). Call only when
    supports_twolevel_post_combine_scatter holds."""
    h1, h2 = recv.shape[1:]
    blocks = recv.shape[0] // rows
    rb, out_b, need = _scatter_tiles(
        rows=rows,
        chunk=chunk,
        blocks=blocks,
        h=h1 * h2,
        dtype=recv.dtype,
        out_dtype=out_dtype,
    )
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    gain = (post_norm_weight.astype(jnp.float32) + 1.0).reshape(h1, h2)
    w = jnp.broadcast_to(rank_weight.astype(jnp.float32)[:, None], (rows, 128))
    return pl.pallas_call(
        functools.partial(_twolevel_combine_scatter_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct((chunk, h1, h2), out_dtype),
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
                pltpu.VMEM((chunk, h1, h2), jnp.float32),
                pltpu.VMEM((rb, h1, h2), recv.dtype),
                pltpu.VMEM((out_b, h1, h2), out_dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(min(capacity * 0.95, need + (16 << 20))),
        ),
        name=f"twolevel_post_combine_scatter-C_{chunk}-R_{rows}",
    )(rank_token.astype(jnp.int32), recv.reshape(blocks, rows, h1, h2), gain, w)
