# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/publication.py
# Revision: 42c35ec6166a19b3f0db90f31807cc3df44602ec (epic/dsv41)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-block candidate scores for candidate publication.

A candidate-source layer publishes, per query, the IDs of its best token
blocks for later layers to rescore. This module computes the block scores:
each block's maximum visible token score, with the newest visible block forced
to +inf so it is always published. Selecting the top blocks happens in
``selection._publish_candidate_blocks``. The source's SparseCore
``publish_candidates`` kernel serves only its BF16 decode path and is not
ported.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.deepseek_v41.index_selection import tuned_block_sizes


@functools.partial(jax.jit, static_argnames=("block_size",))
def candidate_block_scores(
    scores: jax.Array, lengths: jax.Array, *, block_size: int
) -> jax.Array:
    """Reduce visible token scores by block and force the newest block to win."""
    rows, width = scores.shape
    blocks = pl.cdiv(width, block_size)
    lengths = jnp.clip(lengths, 0, width)
    config = None
    if (
        scores.dtype == jnp.float32
        and lengths.dtype == jnp.int32
        and any(
            key[1:] == (rows, width, block_size)
            for key in tuned_block_sizes.PUBLISH_BLOCK_MAX_CONFIG
        )
        and jax.default_backend() == "tpu"
    ):
        config = tuned_block_sizes.PUBLISH_BLOCK_MAX_CONFIG.get(
            (pltpu.get_tpu_info().generation, rows, width, block_size)
        )
    if config is not None:
        return _candidate_block_scores(scores, lengths, block_size, config)
    masked = jnp.where(jnp.arange(width)[None, :] < lengths[:, None], scores, -jnp.inf)
    reduced = (
        jnp.pad(
            masked,
            ((0, 0), (0, blocks * block_size - width)),
            constant_values=-jnp.inf,
        )
        .reshape(rows, blocks, block_size)
        .max(-1)
    )
    last = (lengths - 1) // block_size
    return jnp.where(jnp.arange(blocks)[None, :] == last[:, None], jnp.inf, reduced)


def _candidate_block_scores(scores, lengths, block_size, config):
    rows, width = scores.shape
    row_tile, block_tile = config
    lanes = pltpu.get_tpu_info().num_lanes
    blocks = pl.cdiv(width, block_size)
    padded_blocks = pl.cdiv(blocks, block_tile) * block_tile
    token_tile = block_tile * block_size

    def kernel(length_ref, score_ref, out_ref):
        start = pl.program_id(1) * block_tile
        limits = jnp.stack(
            [length_ref[pl.program_id(0) * row_tile + i] for i in range(row_tile)]
        )
        columns = jnp.arange(token_tile) + start * block_size
        values = jnp.where(columns[None, :] < limits[:, None], score_ref[...], -jnp.inf)
        vectors = values.reshape(row_tile, token_tile // lanes, lanes)
        # Each lane ending a block receives its maximum; gather those lanes
        # without expanding the block axis into padded vectors.
        for bit in range(block_size.bit_length() - 1):
            vectors = jnp.maximum(vectors, pltpu.roll(vectors, 1 << bit, axis=2))
        indices = jnp.arange(lanes // block_size) * block_size + block_size - 1
        indices = jnp.broadcast_to(indices, (*vectors.shape[:-1], indices.size))
        reduced = jnp.take_along_axis(vectors, indices, axis=2).reshape(
            row_tile, block_tile
        )
        last = (limits - 1) // block_size
        out_ref[...] = jnp.where(
            jnp.arange(block_tile)[None, :] + start == last[:, None], jnp.inf, reduced
        )

    # The final input window can be partial. Its invisible columns are masked
    # in the kernel, avoiding a full score-buffer copy to pad the DMA tile.
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, padded_blocks), scores.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(rows // row_tile, padded_blocks // block_tile),
            in_specs=(pl.BlockSpec((row_tile, token_tile), lambda i, j, _: (i, j)),),
            out_specs=pl.BlockSpec((row_tile, block_tile), lambda i, j, _: (i, j)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="indexer_candidate_block_max",
    )(lengths, scores)
    return result[:, :blocks]
