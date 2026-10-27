# SPDX-License-Identifier: Apache-2.0
"""K4/D128 pooling directly from raw pages into the native pooled cache.

Short queries batch independent aligned raw tiles across requests. Long queries
stream larger contiguous tiles through two buffers. Dispatch uses a static
per-request query bound, regardless of whether the caller is decoding or prefilling.
Any page size that is a multiple of 16 works: every raw tile divides a page,
so neither a raw tile nor its destination quads cross a page boundary.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kpool.io import LongQueryIO, ShortQueryIO
from .kpool.pipeline import LongQueryPipeline, ShortQueryPipeline

# Raw rows per short-schedule tile: four pools, one destination quad row.
_SHORT_TILE_ROWS = 16


def supports_page_size(page_size):
    return page_size % _SHORT_TILE_ROWS == 0


def _tile_slots(max_q, tile_rows):
    """Bound aligned tiles touched by newly closed pools in one request."""
    return (max_q + tile_rows - 5) // tile_rows + 1


def _long_tile_rows(max_q, page_size):
    """The streaming tile height: the largest candidate up to the measured
    target (32 below 128 queries, else 128) that divides the page."""
    target = 32 if max_q < 128 else 128
    return max(
        rows for rows in (16, 32, 64, 128) if rows <= target and page_size % rows == 0
    )


def _descriptors(
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    num_tokens,
    tile_rows,
    max_q,
    page_size,
):
    """Compact request tile slots in request order for the streaming schedule."""
    batch = kv_lens.shape[0]
    tiles = _request_descriptors(
        kv_lens, page_indices, cu_q_lens, distribution, tile_rows, max_q, page_size
    )
    # Preserve request-major order. The short schedule instead groups the
    # first tiles of independent requests to expose concurrent transfers.
    tiles = tiles.reshape(7, -1, batch).transpose(0, 2, 1).reshape(7, -1)
    live = tiles[6] != 0
    capacity = min(num_tokens, num_tokens // tile_rows + 2 * batch)
    chosen = jnp.nonzero(live, size=capacity, fill_value=0)[0]
    # Streaming IO uses: raw page, pooled page, pooled quad, first/last row,
    # and raw tile offset. Only the compacted live prefix is consumed.
    desc = tiles[jnp.array([0, 2, 3, 4, 5, 1]), :][:, chosen]
    return jnp.sum(live, dtype=jnp.int32)[None], desc


def _request_descriptors(
    kv_lens, page_indices, cu_q_lens, distribution, tile_rows, max_q, page_size
):
    """Fixed tile slots cover every newly completed pool, including verify Q>1."""
    batch = kv_lens.shape[0]
    # Closed groups end at 3 mod 4. A request's first end can be at the last
    # group in a tile; account for that alignment when bounding tile slots.
    slots = _tile_slots(max_q, tile_rows)
    req = jnp.arange(batch, dtype=jnp.int32)[None, :]
    q_len = cu_q_lens[1:] - cu_q_lens[:-1]
    start = kv_lens - q_len
    first_end = (start // 4) * 4 + 3
    tile_start = (first_end // tile_rows) * tile_rows
    tile_start = tile_start[None, :] + jnp.arange(slots)[:, None] * tile_rows
    first_in_tile = jnp.maximum(first_end[None, :], tile_start + 3)
    live = (req < distribution[2]) & (q_len > 0) & (first_in_tile < kv_lens)
    # Inactive slots still need safe metadata gathers, but issue no DMA.
    safe_start = jnp.where(live, tile_start, 0)
    pages_per_seq = page_indices.shape[0] // batch
    # Pool p sits at slot p of the pooled cache, so one pooled page holds the
    # pools of 4 * page_size raw tokens, four pools per destination quad row.
    pooled_span = 4 * page_size
    # Tile-major order keeps the first tiles of different requests adjacent.
    # Request-major order would insert unused slots between Q=1 requests.
    return (
        jnp.stack(
            (
                page_indices[req * pages_per_seq + safe_start // page_size],
                (safe_start % page_size) // tile_rows,
                page_indices[req * pages_per_seq + safe_start // pooled_span],
                (safe_start % pooled_span) // 16,
                start[None, :] - tile_start,
                kv_lens[None, :] - tile_start,
                live.astype(jnp.int32),
            )
        )
        .reshape(7, slots * batch)
        .astype(jnp.int32)
    )


def _short_kernel(
    desc_ref,
    raw_ref,
    old_ref,
    ape_ref,
    mask_ref,
    out_ref,
    raw_buf,
    out_buf,
    read_sem,
    old_sem,
    write_sem,
    *,
    group_size,
):
    del old_ref  # out_ref aliases the pooled input, including untouched bytes.
    io = ShortQueryIO(
        desc_ref,
        raw_ref,
        out_ref.bitcast(jnp.uint32),
        raw_buf,
        out_buf,
        read_sem,
        old_sem,
        write_sem,
    )
    ShortQueryPipeline(group_size).emit(io, ape_ref, mask_ref)


def _long_kernel(
    count_ref,
    desc_ref,
    raw_ref,
    old_ref,
    ape_ref,
    out_ref,
    raw_buf,
    out_buf,
    pooled_buf,
    read_sem,
    write_sem,
    old_sem,
    *,
    tile_rows,
):
    del old_ref
    io = LongQueryIO(
        desc_ref,
        raw_ref,
        out_ref.bitcast(jnp.uint32),
        raw_buf,
        out_buf,
        read_sem,
        write_sem,
        old_sem,
        tile_rows,
    )
    LongQueryPipeline(tile_rows).emit(io, ape_ref, pooled_buf, count_ref[0])


def compress_and_store(
    raw_cache,
    pooled_cache,
    ape,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    num_tokens,
    kpool=4,
    *,
    max_query_length=None,
):
    """Compress newly closed K4 groups into the aliased native pooled cache.

    Raw writes must already be complete. Destination records must be unique;
    distinct records in a shared physical block are supported.
    max_query_length is a static upper bound on each request's newly supplied
    query tokens, including every verification row, not historical KV length.
    Omit it to use num_tokens as a conservative bound for ragged requests.
    Short prefill, decode and verification use the same batched implementation.
    """
    page_size = raw_cache.shape[1]
    if kpool != 4 or raw_cache.shape[2] != 256 or not supports_page_size(page_size):
        raise ValueError(
            "fused pooling requires K=4, D=128 and a page size that is a multiple "
            f"of {_SHORT_TILE_ROWS}"
        )
    if pooled_cache.shape != (raw_cache.shape[0], page_size // 4, 4, 256):
        raise ValueError("unexpected pooled-cache layout")
    if raw_cache.dtype != jnp.bfloat16 or pooled_cache.dtype != jnp.uint8:
        raise ValueError("expected BF16 raw cache and uint8 pooled cache")
    if not num_tokens:
        return pooled_cache
    max_q = num_tokens if max_query_length is None else max_query_length
    if not isinstance(max_q, int) or max_q < 1:
        raise ValueError("max_query_length must be a positive static integer")
    # Small query spans amortize DMA across requests. A single group of
    # tiles also avoids the streaming descriptor compaction overhead.
    short_tiles = _tile_slots(max_q, _SHORT_TILE_ROWS) * kv_lens.shape[0]
    if max_q <= 64 or short_tiles <= 16:
        desc = _request_descriptors(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            _SHORT_TILE_ROWS,
            max_q,
            page_size,
        )
        group_size = min(16, desc.shape[1])
        padding = (-desc.shape[1]) % group_size
        desc = jnp.pad(desc, ((0, 0), (0, padding)))
        count = desc.shape[1]
        groups = desc.reshape((7, count // group_size, group_size))
        same_quad = (groups[2, :, :, None] == groups[2, :, None, :]) & (
            groups[3, :, :, None] == groups[3, :, None, :]
        )
        live_pair = (groups[6, :, :, None] != 0) & (groups[6, :, None, :] != 0)
        index = jnp.arange(group_size)
        conflict = jnp.any(
            same_quad & live_pair & (index[:, None] < index[None, :]), axis=(1, 2)
        )
        desc = jnp.concatenate(
            (desc, jnp.repeat(conflict, group_size)[None, :].astype(jnp.int32)),
            axis=0,
        )
        ends = jnp.arange(4) * 4 + 3
        live = (ends[None, :] >= desc[4, :, None]) & (ends[None, :] < desc[5, :, None])
        mask = jnp.sum(
            jnp.where(live, jnp.uint32(255), jnp.uint32(0))
            << (jnp.arange(4, dtype=jnp.uint32) * 8)[None, :],
            axis=1,
        )[:, None]
        return pl.pallas_call(
            partial(_short_kernel, group_size=group_size),
            out_shape=jax.ShapeDtypeStruct(pooled_cache.shape, pooled_cache.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1,),
                in_specs=(
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(),
                    pl.BlockSpec(),
                ),
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=(
                    pltpu.VMEM((group_size, 16, 256), jnp.bfloat16),
                    pltpu.VMEM((group_size, 1, 256), jnp.uint32),
                    pltpu.SemaphoreType.DMA((group_size,)),
                    pltpu.SemaphoreType.DMA((group_size,)),
                    pltpu.SemaphoreType.DMA((group_size,)),
                ),
            ),
            input_output_aliases={2: 0},
            name="kpool_short_query_store",
        )(desc, raw_cache, pooled_cache, ape.astype(jnp.float32), mask)
    tile_rows = _long_tile_rows(max_q, page_size)
    count, desc = _descriptors(
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_tokens,
        tile_rows,
        max_q,
        page_size,
    )
    return pl.pallas_call(
        partial(_long_kernel, tile_rows=tile_rows),
        out_shape=jax.ShapeDtypeStruct(pooled_cache.shape, pooled_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.VMEM((2, tile_rows, 256), jnp.bfloat16),
                pltpu.VMEM((2, tile_rows // 16, 256), jnp.uint32),
                pltpu.VMEM((tile_rows // 4, 128), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={3: 0},
        name="kpool_long_query_store",
    )(count, desc, raw_cache, pooled_cache, ape.astype(jnp.float32))
