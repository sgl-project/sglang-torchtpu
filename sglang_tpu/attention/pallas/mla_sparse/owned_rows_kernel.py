# SPDX-License-Identifier: Apache-2.0
"""``kv_shard.owned_rows`` as one Pallas pass on pool planes.

A row's ``group`` consecutive entries share a pool and so an owner. The rows go
in as ``group`` planes of one entry per pool ([group, tokens, pools], an XLA
transpose); the kernel decides ownership and runs the shift network once per
pool, moving the planes together, then the planes transpose back to rows."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Tokens per grid step.
_BLOCK_TOKENS = 64
_LANES = 128


def supported(rows_shape: tuple[int, int], *, page_size: int, shards: int, group: int) -> bool:
    num_tokens, width = rows_shape
    return (
        group & (group - 1) == 0
        and width % group == 0
        and page_size % shards == 0
        and num_tokens % 8 == 0
        and num_tokens > 0
    )


def _left(x, bit, width):
    # x[:, j + bit] at lane j; lanes past the row read the wrapped head (masked by callers).
    return pltpu.roll(x, width - bit, 1)


def _kernel(planes_ref, upper_ref, out_ref, *, page_size, sub, role):
    group, _, pools = planes_ref.shape
    head = planes_ref[0]
    lane = lax.broadcasted_iota(jnp.int32, head.shape, 1)
    keep = jnp.logical_and(head >= 0, (head % page_size) // sub == role)
    # Dropped pools up to here (inclusive prefix count, exact: 0/1 terms in f32).
    dropped = jnp.where(keep, 0.0, 1.0).astype(jnp.bfloat16)
    shift = jnp.dot(dropped, upper_ref[...], preferred_element_type=jnp.float32)
    # A kept pool carries its shift (dropped pools before it), others -1.
    shift = jnp.where(keep, shift.astype(jnp.int32), -1)
    planes = [jnp.where(keep, planes_ref[r], -1) for r in range(group)]
    bit = 1
    while bit < pools:
        inside = lane < pools - bit
        ls = _left(shift, bit, pools)
        arrive = jnp.logical_and(inside, jnp.logical_and(ls >= 0, (ls & bit) != 0))
        stay = jnp.logical_and(shift >= 0, (shift & bit) == 0)
        planes = [
            jnp.where(arrive, _left(p, bit, pools), jnp.where(stay, p, -1))
            for p in planes
        ]
        shift = jnp.where(arrive, ls, jnp.where(stay, shift, -1))
        bit *= 2
    out_pools = out_ref.shape[2]
    for r in range(group):
        out_ref[r, :, :pools] = planes[r]
        if out_pools > pools:
            out_ref[r, :, pools:] = jnp.full(
                (head.shape[0], out_pools - pools), -1, jnp.int32
            )


@functools.partial(
    jax.jit, static_argnames=("page_size", "shards", "role", "group", "out_width")
)
def owned_rows_kernel(
    rows: jax.Array,
    *,
    page_size: int,
    shards: int,
    role: int,
    group: int,
    out_width: int,
) -> jax.Array:
    """Bit-identical to ``kv_shard.owned_rows`` (see ``supported``)."""
    num_tokens, width = rows.shape
    pools = width // group
    # Pools padded to whole lanes with -1 pools (dropped, so the output is unchanged).
    lane_pools = -(-pools // _LANES) * _LANES
    out_pools = max(-(-(out_width // group) // _LANES) * _LANES, lane_pools)
    planes = jnp.pad(
        rows.astype(jnp.int32).reshape(num_tokens, pools, group),
        ((0, 0), (0, lane_pools - pools), (0, 0)),
        constant_values=-1,
    ).transpose(2, 0, 1)
    tb = min(_BLOCK_TOKENS, num_tokens)
    while num_tokens % tb:
        tb //= 2
    # upper[i, j] = 1 for i <= j: dropped @ upper is the inclusive prefix count.
    upper = jnp.triu(jnp.ones((lane_pools, lane_pools), jnp.bfloat16))
    out = pl.pallas_call(
        functools.partial(
            _kernel, page_size=page_size, sub=page_size // shards, role=role
        ),
        out_shape=jax.ShapeDtypeStruct((group, num_tokens, out_pools), jnp.int32),
        grid=(num_tokens // tb,),
        in_specs=[
            pl.BlockSpec((group, tb, lane_pools), lambda i: (0, i, 0)),
            pl.BlockSpec(
                (lane_pools, lane_pools), lambda i: (0, 0), pipeline_mode=pl.Buffered(1)
            ),
        ],
        out_specs=pl.BlockSpec((group, tb, out_pools), lambda i: (0, i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="dsa_owned_rows",
    )(planes, upper)
    return out.transpose(1, 2, 0).reshape(num_tokens, out_pools * group)[:, :out_width]
