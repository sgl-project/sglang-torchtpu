# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/publication.py
# Revision: 6fc30f15d40c2b5089b6605454d1045dce6b5b86 (PR #2430)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseCore block maxima of representative score rows for candidate publication.

Unlike the source, each score tile's maxima are written back as soon as they are
reduced, so subcore VMEM is bounded by the tile and not by the context length.
"""

from functools import reduce

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from .tuned_block_sizes import PACKED_FP8_BLOCK_MAX_CONFIG


def representative_block_max(
    scores: jax.Array, query_rows: jax.Array | None = None, *, block: int
) -> jax.Array:
    """Return FP32 [len(query_rows), width // block] maxima of aligned blocks.

    ``scores`` is FP32 [rows, width]; ``query_rows`` holds int32 row IDs into it.
    Omit ``query_rows`` to reduce all rows in order, without a row-ID operand.
    This avoids lifting a constant HBM row-ID buffer into enclosing XLA loops.
    The width must be a multiple of ``block`` times the SparseCore lane count,
    which every packed-FP8 score width (a multiple of 256 keys) satisfies for
    the model's 8-key candidate blocks.
    """
    if scores.dtype != jnp.float32 or scores.ndim != 2:
        raise ValueError("Block maxima require FP32 [rows, width] scores")
    if block < 1 or (
        query_rows is not None
        and (query_rows.ndim != 1 or query_rows.dtype != jnp.int32)
    ):
        raise ValueError("Block maxima require int32 row IDs and a positive block")
    identity_rows = query_rows is None
    groups = scores.shape[0] if identity_rows else query_rows.size
    width = scores.shape[1]
    if groups == 0 or width == 0 or scores.shape[0] == 0:
        return jnp.full((groups, pl.cdiv(width, block)), -jnp.inf, jnp.float32)
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    if block < 1 or width % (block * lanes):
        raise ValueError(
            f"Score width {width} must be a multiple of block {block} x {lanes} lanes"
        )
    tile_tokens, unroll = PACKED_FP8_BLOCK_MAX_CONFIG.get((width, block), (4096, 2))
    tile_tokens = min(tile_tokens, width)
    tile_tokens -= tile_tokens % (block * lanes)
    tiles = pl.cdiv(width, tile_tokens)
    tile_columns = tile_tokens // block
    columns = width // block
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(groups, info.num_subcores)),
    )

    @pl.kernel(
        mesh=mesh,
        out_type=jax.ShapeDtypeStruct((groups, tiles * tile_columns), jnp.float32),
        scratch_types=(
            pltpu.VMEM((2 * tile_tokens,), jnp.float32),
            pltpu.VMEM((2, tile_columns), jnp.float32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA((5,)),
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="mqa_representative_block_max",
    )
    def kernel(src, *refs):
        if identity_rows:
            dst, buffers, output, row_ids, sem = refs
        else:
            rows, dst, buffers, output, row_ids, sem = refs
        first = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(first, groups, step=mesh.num_cores * mesh.num_subcores)
        def row(group):
            if identity_rows:
                source_row = group
            else:
                pltpu.async_copy(
                    rows.at[pl.ds(group // lanes * lanes, lanes)], row_ids, sem.at[4]
                ).wait()
                source_row = plsc.load_gather(
                    row_ids, [jnp.full((lanes,), group % lanes)]
                )[0]
            valid_row = (source_row >= 0) & (source_row < scores.shape[0])
            source_row = jnp.clip(source_row, 0, scores.shape[0] - 1)

            def transfer(tile, *, wait):
                slot = tile % 2

                def move(size):
                    copy = pltpu.make_async_copy(
                        src.at[source_row, pl.ds(tile * tile_tokens, size)],
                        buffers.at[pl.ds(slot * tile_tokens, size)],
                        sem.at[slot],
                    )
                    copy.wait() if wait else copy.start()

                if width % tile_tokens:
                    jax.lax.cond(
                        tile == tiles - 1,
                        lambda: move(width % tile_tokens),
                        lambda: move(tile_tokens),
                    )
                else:
                    move(tile_tokens)

            def publish(tile, *, wait):
                copy = pltpu.make_async_copy(
                    output.at[tile % 2],
                    dst.at[group, pl.ds(tile * tile_columns, tile_columns)],
                    sem.at[2 + tile % 2],
                )
                copy.wait() if wait else copy.start()

            transfer(jnp.int32(0), wait=False)

            @pl.loop(0, tiles)
            def tile(t):
                pl.when(t + 1 < tiles)(lambda: transfer(t + 1, wait=False))
                transfer(t, wait=True)
                # The output slot was last written two tiles ago.
                pl.when(t >= 2)(lambda: publish(t - 2, wait=True))
                count = jnp.minimum(tile_tokens, width - t * tile_tokens) // block

                @plsc.parallel_loop(0, tile_columns, lanes, unroll=unroll)
                def maxima(i):
                    # Tail lanes must not read the uninitialized remainder of
                    # a partial DMA, even though their final outputs are masked.
                    indices = (
                        t % 2 * tile_tokens
                        + jnp.minimum(i + jnp.arange(lanes), count - 1) * block
                    )
                    values = reduce(
                        jnp.maximum,
                        (
                            plsc.load_gather(buffers, [indices + j])
                            for j in range(block)
                        ),
                    )
                    # A partial final tile publishes -inf beyond its columns.
                    output[t % 2, pl.ds(i, lanes)] = jnp.where(
                        valid_row & (i + jnp.arange(lanes) < count), values, -jnp.inf
                    )

                publish(t, wait=False)

            # tiles is static: drain the (at most two) writes still in flight.
            for back in range(min(tiles, 2), 0, -1):
                publish(jnp.int32(tiles - back), wait=True)

    inputs = () if identity_rows else (jnp.pad(query_rows, (0, -groups % 128)),)
    return kernel(scores, *inputs)[:, :columns]
