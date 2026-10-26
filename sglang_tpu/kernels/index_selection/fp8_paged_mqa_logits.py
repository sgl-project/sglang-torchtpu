# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP8 MQA logits for TPU v7."""

from dataclasses import replace
from functools import partial, reduce
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .tuned_block_sizes import (
    PACKED_FP8_MAX_QUERY_TILE,
    PACKED_FP8_MQA_CONFIG,
    paged_mqa_key_tile,
)


class QueryTile(NamedTuple):
    start: jax.Array
    count: jax.Array
    page_row: jax.Array


def _next_query_tile(query, offsets, tile_size, end):
    start = query.start + query.count
    request = lax.while_loop(
        lambda r: (r < offsets.shape[0] - 2) & (start >= offsets[r + 1]),
        lambda r: r + 1,
        query.page_row,
    )
    return QueryTile(
        start,
        jnp.minimum(tile_size, jnp.minimum(offsets[request + 1], end) - start),
        request,
    )


def _encode_page_table(page_ids, layout, first_bank_rows, tile_pages):
    valid = (page_ids >= 0) & (page_ids < layout.pages)
    # Spread masked reads: repeating page zero serializes a partially filled tile.
    safe = jnp.where(valid, page_ids, jnp.arange(page_ids.shape[-1]) % layout.pages)
    global_rows = layout.page_offset + safe * layout.page_stride
    bank = (global_rows >= first_bank_rows).astype(jnp.int32)
    local_rows = global_rows - bank * first_bank_rows
    full = jnp.all(valid.reshape(-1, tile_pages), axis=1).astype(jnp.int32)
    full = jnp.repeat(full, tile_pages).reshape(page_ids.shape)
    return (local_rows << 3) | (full << 2) | (valid.astype(jnp.int32) << 1) | bank


def _head_groups(queries, heads, sublanes):
    start = 0
    while start < queries:
        count = 1 << ((queries - start).bit_length() - 1)
        copies = sublanes // count
        yield start, count, heads // (2 * copies), copies
        start += count


def _head_order(x, queries, sublanes):
    heads, width = x.shape[0] // queries, x.shape[-1]
    x = x.reshape(queries, heads, width)
    parts = []
    for start, count, groups, copies in _head_groups(queries, heads, sublanes):
        part = x[start : start + count]
        if x.dtype == jnp.float32:
            low = part[:, : heads // 2].transpose(1, 0, 2)
            high = part[:, heads // 2 :].transpose(1, 0, 2)
            part = pltpu.pack_elementwise((low, high), packed_dtype=jnp.bfloat16)
        else:
            part = part.reshape(count, 2, groups, copies, width)
            part = part.transpose(2, 3, 0, 1, 4)
        parts.append(part.reshape(-1, width))
    ordered = jnp.concatenate(parts)
    return pltpu.bitcast(ordered, jnp.bfloat16) if x.dtype == jnp.float32 else ordered


def _reduce_heads(products, selector, *, sublanes, use_mxu):
    unpack = partial(
        pltpu.unpack_elementwise, packed_dtype=jnp.bfloat16, unpacked_dtype=jnp.float32
    )
    if use_mxu:
        values = lax.dot(selector, products, preferred_element_type=jnp.float32)
    else:
        queries, width = selector.shape[0], products.shape[-1]
        heads = products.shape[0] // queries
        values = None
        for start, count, groups, _ in _head_groups(queries, heads, sublanes):
            packed = pltpu.bitcast(
                products[start * heads : (start + count) * heads], jnp.uint32
            ).reshape(groups, sublanes, width)
            summed = reduce(
                jnp.add,
                (unpack(packed[g], index=h) for g in range(groups) for h in range(2)),
            )
            offset = sublanes // 2
            while offset >= count:
                summed = summed + pltpu.roll(summed, offset, axis=0)
                offset //= 2
            values = (
                summed
                if values is None
                else jnp.where(jnp.arange(sublanes)[:, None] < start, values, summed)
            )
        values = cast(jax.Array, values)[:queries]
    return values.astype(jnp.bfloat16).astype(jnp.float32)


def _dma_rows(src, dst, sem, count, wait=False, src_start=0, dst_start=0):
    for local in range(min(src.shape[0], dst.shape[0])):

        @pl.when(local < count)
        def copy(local=local):
            source = src.at[pl.ds(src_start + local, 1)]
            target = dst.at[pl.ds(dst_start + local, 1)]
            completion = sem.at[local] if sem.shape else sem
            dma = pltpu.make_async_copy(source, target, completion)
            (dma.wait if wait else dma.start)()


def _kernel(
    offsets, lengths, query_start, inputs, out, scratch, *, config, info, layout
):
    mxu_size, sublanes = info.mxu_column_size, info.num_sublanes
    q, weights, pool, table = inputs
    buffers, semaphores = scratch
    queries, keys, scores, counts, pages = buffers
    read_sem, write_sem, q_sem, page_sem = semaphores
    # JAX 0.10.2 has no pltpu.annotate(no_store=...). Keep the same refs:
    # weights are read-only and keys are explicitly managed by the DMA ring.
    tokens, heads, dim = q.shape
    begin = query_start[()]
    end = begin + out.shape[0]
    q_tile, key_tile = config.query_tile, config.key_tile
    page_size, page_rows = layout.page_size, layout.page_rows
    page_count, tile_pages = table.shape[1], key_tile // page_size
    tiles, fragments = out.shape[1] // key_tile, key_tile // mxu_size

    def load_table(query, slot, wait=False):
        dst = pages.at[pl.ds(slot * page_count, page_count)]
        dma = pltpu.make_async_copy(table.at[query.page_row], dst, page_sem.at[slot])
        (dma.wait if wait else dma.start)()

    def page_record(tile, i, slot):
        # Keep SMEM flat: 2D indexing adds scalar address work in masked tiles.
        return pages[slot * page_count + tile * tile_pages + i]

    def copy_queries(query, slot, wait=False, zero=False):
        dst = queries.at[pl.ds(slot * q_tile, q_tile)]
        if zero:
            dst[...] = jnp.zeros(dst.shape, dst.dtype)
        _dma_rows(q, dst, q_sem.at[slot], query.count, wait, src_start=query.start)

    def release_outputs(pair):
        count = counts[pair]
        for slot in (2 * pair, 2 * pair + 1)[:tiles]:
            src = scores.at[slot]
            _dma_rows(src, src, write_sem.at[slot], count, wait=True)

    def load_page(tile, i, page_slot, slot):
        record = page_record(tile, i, page_slot)
        address = lax.shift_right_logical(record, jnp.int32(3))

        def load(bank):
            pltpu.make_async_copy(
                pool[bank].at[pl.ds(address, page_rows)],
                keys.at[slot, pl.ds(i * page_rows, page_rows)],
                read_sem.at[slot],
            ).start()

        if len(pool) == 1:
            load(0)
        else:
            lax.cond((record & 1) == 0, lambda: load(0), lambda: load(1))

    next_query = partial(_next_query_tile, offsets=offsets, tile_size=q_tile, end=end)

    def row_ids(query):
        return [jnp.minimum(query.start + i, tokens - 1) for i in range(q_tile)]

    def row_lengths(query):
        rows = enumerate(row_ids(query))
        return [jnp.where(i < query.count, lengths[r], 0) for i, r in rows]

    def live_tiles(query):
        limit = reduce(jnp.maximum, row_lengths(query), jnp.int32(0))
        return pl.cdiv(jnp.minimum(limit, out.shape[1]), key_tile)

    owners = _head_order(
        jnp.repeat(jnp.arange(q_tile), heads)[:, None], q_tile, sublanes
    )[:, 0]
    selector = (jnp.arange(q_tile)[:, None] == owners[None, :]).astype(jnp.bfloat16)

    def query_block(state):
        query, following, lookahead, tile_counts, step, key_base, ready = state
        tile_count, next_tiles, ahead_tiles = tile_counts
        q_slot, page_slot = step & 1, step & 3
        farthest = next_query(lookahead)

        def key_slot(tile):
            return (key_base + tile) & (config.key_buffers - 1)

        def output_slot(tile):
            return 2 * q_slot + (tile & 1)

        pl.when(farthest.start < end)(lambda: load_table(farthest, (page_slot + 3) & 3))
        copy_queries(following, 1 - q_slot, zero=True)
        lengths_vector = jnp.stack(row_lengths(query))
        minimum_length = reduce(jnp.minimum, row_lengths(query))

        def publish(tile, wait=False, slot=None):
            slot = output_slot(tile) if slot is None else slot
            src, sem = scores.at[slot], write_sem.at[slot]
            dst = out.at[:, pl.ds(tile * key_tile, key_tile)]
            _dma_rows(src, dst, sem, query.count, wait, dst_start=query.start - begin)

        def tile_source(future):
            first, after = future < tile_count, future - tile_count
            third = after >= next_tiles
            return (
                jnp.where(first, future, after - jnp.where(third, next_tiles, 0)),
                (page_slot + jnp.where(first, 0, jnp.where(third, 2, 1))) & 3,
            )

        pl.when(lookahead.start < end)(
            lambda: load_table(lookahead, (page_slot + 2) & 3, wait=True)
        )
        release_outputs(q_slot)
        for i in range(2):

            @pl.when((i >= ready) & (i < tile_count + next_tiles + ahead_tiles))
            def prefill(i=i):
                target, owner = tile_source(i)
                for page in range(tile_pages):
                    load_page(target, page, owner, key_slot(i))

        copy_queries(query, q_slot, wait=True)

        def emit_tile(tile, *, prefetch, full):
            weight = jnp.concatenate([weights[r][:, None] for r in row_ids(query)])
            head_weights = _head_order(weight, q_tile, sublanes)
            query_slice = pl.ds(q_slot * q_tile, q_tile)
            lhs = _head_order(
                queries[query_slice].reshape(q_tile * heads, dim), q_tile, sublanes
            )
            target, metadata = tile_source(tile + 2)
            following_slot = key_slot(tile + 2)

            def key_rows(start, count):
                return keys[key_slot(tile), pl.ds(start, count)]

            base = tile * key_tile + jnp.arange(PAGE_RECORDS)
            remaining = jnp.maximum(lengths_vector, 0)[:, None] - base[None, :]

            def qk(fragment):
                parts = []
                if page_size != PAGE_RECORDS:
                    for start in range(
                        fragment * mxu_size,
                        (fragment + 1) * mxu_size,
                        min(page_size, mxu_size),
                    ):
                        page, column = divmod(start, page_size)
                        raw = key_rows(
                            page * page_rows,
                            (PAGE_ROWS - 1) * page_size // PAGE_RECORDS,
                        )
                        if page_size < PAGE_RECORDS:
                            raw = jnp.repeat(raw, PAGE_RECORDS // page_size, axis=0)
                            columns = (
                                jnp.arange(PAGE_ROWS - 1)[:, None] * page_size
                                + jnp.arange(PAGE_RECORDS)[None, :] % page_size
                            ) % PAGE_RECORDS
                            words = jnp.take_along_axis(raw, columns, axis=1)[
                                :, :page_size
                            ]
                        else:
                            words = raw.reshape(PAGE_ROWS - 1, page_size)
                        parts.append(
                            words[:, column : column + min(page_size, mxu_size)]
                        )
                    rhs = pltpu.bitcast(jnp.concatenate(parts, axis=1), q.dtype)
                    logits = lax.dot(lhs, rhs, preferred_element_type=jnp.float32)
                    return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))
                for page in range(
                    fragment * mxu_size // PAGE_RECORDS,
                    (fragment + 1) * mxu_size // PAGE_RECORDS,
                ):
                    words = key_rows(page * PAGE_ROWS, PAGE_ROWS - 1)
                    parts.append(pltpu.bitcast(words, q.dtype))
                logits = lax.dot(
                    lhs,
                    jnp.concatenate(parts, axis=1),
                    preferred_element_type=jnp.float32,
                )
                return jnp.maximum(logits.astype(jnp.bfloat16), jnp.bfloat16(0))

            def store(fragment, values):
                if page_size != PAGE_RECORDS:
                    scales, masks = [], []
                    for start in range(
                        fragment * mxu_size,
                        (fragment + 1) * mxu_size,
                        min(page_size, mxu_size),
                    ):
                        page, column = divmod(start, page_size)
                        scale = key_rows(page * page_rows, page_rows).reshape(-1)[
                            (PAGE_ROWS - 1) * page_size : PAGE_ROWS * page_size
                        ]
                        scales.append(
                            lax.bitcast_convert_type(
                                scale[column : column + min(page_size, mxu_size)],
                                jnp.float32,
                            )
                        )
                        valid = full or (page_record(tile, page, page_slot) & 2) != 0
                        masks.append(
                            jnp.full((min(page_size, mxu_size),), valid, jnp.int32)
                        )
                    scaled = values * jnp.concatenate(scales)[None, :]
                    if not full:
                        positions = (
                            tile * key_tile + fragment * mxu_size + jnp.arange(mxu_size)
                        )
                        scaled = jnp.where(
                            (jnp.concatenate(masks)[None, :] != 0)
                            & (lengths_vector[:, None] > positions[None, :]),
                            scaled,
                            -jnp.inf,
                        )
                    scores[
                        output_slot(tile), :, pl.ds(fragment * mxu_size, mxu_size)
                    ] = scaled
                    return
                for offset in range(0, mxu_size, PAGE_RECORDS):
                    local_page = (fragment * mxu_size + offset) // PAGE_RECORDS
                    valid = full or (page_record(tile, local_page, page_slot) & 2) != 0
                    scale_row = (local_page + 1) * PAGE_ROWS - 1
                    scale = lax.bitcast_convert_type(
                        key_rows(scale_row, 1), jnp.float32
                    )
                    scaled = values[:, offset : offset + PAGE_RECORDS] * scale
                    column = local_page * PAGE_RECORDS
                    if not full:
                        bound = jnp.where(valid, column, jnp.iinfo(jnp.int32).max)
                        scaled = jnp.where(remaining > bound, scaled, -jnp.inf)
                    scores[output_slot(tile), :, pl.ds(column, PAGE_RECORDS)] = scaled

            pending = {}
            for stage in range(fragments + config.issue_depth):
                first = pl.cdiv(stage * mxu_size, page_size)
                last = pl.cdiv((stage + 1) * mxu_size, page_size)
                middle = (first + last + 1) // 2
                if stage < fragments:
                    if prefetch:
                        for page in range(first, middle):
                            load_page(target, page, metadata, following_slot)
                    pending[stage] = qk(stage)
                if stage >= config.issue_depth:
                    fragment = stage - config.issue_depth
                    values = _reduce_heads(
                        pending.pop(fragment) * head_weights,
                        selector,
                        sublanes=sublanes,
                        use_mxu=stage < fragments and bool(fragment % 2),
                    )
                    store(fragment, values)
                if stage < fragments and prefetch:
                    for page in range(middle, last):
                        load_page(target, page, metadata, following_slot)

        def step_tile(tile, _):
            src = keys.at[key_slot(tile)]
            pltpu.make_async_copy(src, src, read_sem.at[key_slot(tile)]).wait()
            pl.when(tile >= 2)(lambda: publish(tile - 2, wait=True))
            pending = tile + 2 < tile_count + next_tiles + ahead_tiles
            encoded = page_record(tile, 0, page_slot)
            full = ((tile + 1) * key_tile <= minimum_length) & ((encoded & 4) != 0)

            def emit(full):
                lax.cond(
                    pending,
                    lambda: emit_tile(tile, prefetch=True, full=full),
                    lambda: emit_tile(tile, prefetch=False, full=full),
                )

            lax.cond(full, lambda: emit(True), lambda: emit(False))
            publish(tile)

        if tiles == 1:
            pl.when(tile_count > 0)(lambda: step_tile(jnp.int32(0), None))
        else:
            lax.fori_loop(0, tile_count, step_tile, None)

        counts[q_slot] = jnp.where(tile_count == tiles, query.count, 0)

        @pl.when(tile_count < tiles)
        def fill_padding():
            @pl.when(tile_count > 0)
            def drain():
                publish(tile_count - 1, wait=True)
                pl.when(tile_count > 1)(lambda: publish(tile_count - 2, wait=True))

            scores[output_slot(0)] = jnp.full(scores.shape[1:], -jnp.inf, jnp.float32)
            for wait in (False, True):

                def body(tile, _, wait=wait):
                    publish(tile, wait=wait, slot=output_slot(0))

                lax.fori_loop(tile_count, tiles, body, None)

        return (
            following,
            lookahead,
            farthest,
            (next_tiles, ahead_tiles, live_tiles(farthest)),
            step + 1,
            key_slot(tile_count),
            jnp.minimum(2, next_tiles + ahead_tiles),
        )

    zero = jnp.int32(0)
    first = next_query(QueryTile(begin, zero, zero))
    second = next_query(first)
    third = next_query(second)
    initial = (first, second, third)
    for pair in range(2):
        counts[pair] = jnp.int32(0)
    load_table(first, 0)
    pl.when(second.start < end)(lambda: load_table(second, 1))
    pl.when(third.start < end)(lambda: load_table(third, 2))
    copy_queries(first, 0, zero=True)
    load_table(first, 0, wait=True)
    pl.when(second.start < end)(lambda: load_table(second, 1, wait=True))
    lax.while_loop(
        lambda state: state[0].start < end,
        query_block,
        (*initial, tuple(live_tiles(q) for q in initial), zero, zero, zero),
    )
    release_outputs(0)
    release_outputs(1)


@jax.jit(static_argnames=("cache_layout", "max_model_len", "query_size", "query_count"))
def fp8_paged_mqa_logits(
    q: jax.Array,
    kv_cache: jax.Array | tuple[jax.Array, jax.Array],
    weights: jax.Array,
    context_lens: jax.Array,
    block_table: jax.Array,
    *,
    query_start_loc: jax.Array,
    max_model_len: int,
    q_scale: jax.Array,
    query_size: int,
    cache_layout: PackedFP8Layout | None = None,
    query_start: jax.Array | int = 0,
    query_count: int | None = None,
) -> jax.Array:
    """Return FP32 scores for the query window, with invalid positions set to -inf.

    Args:
        q: FP8 E4M3FN [T,32,128], in request order.
        kv_cache: One or two int32 [rows,128] banks in packed_fp8_cache format.
        weights: FP32 [T,32]; weighted BF16 products must be finite.
        context_lens: Int32 [T] visible index-key counts, zero for padding.
        block_table: Int32 [G,P] page IDs covering the requested output width.
        query_start_loc: Int32 [G+1], monotonic from 0 to T; repeats mean empty.
        max_model_len: Nonnegative output width in index keys.
        q_scale: FP32 [T,32] query scales; K scales reside inside each page.
        query_size: Static query-tile bound, normally speculative tokens+1.
        cache_layout: Shared cache placement; None means tightly packed pages.
        query_start: First query to read from the full input arrays and offsets.
        query_count: Output rows; defaults to the full query batch.
    """
    if q.ndim != 3 or q.shape[1:] != (32, 128) or q.dtype != jnp.float8_e4m3fn:
        raise ValueError("Paged MQA requires FP8 E4M3 queries [T,32,128]")
    if query_size < 1 or max_model_len < 0:
        raise ValueError("Query tile must be positive and score width nonnegative")
    if weights.shape != q.shape[:2] or q_scale.shape != weights.shape:
        raise ValueError("Weights and Q scales must have shape [T,32]")
    if context_lens.shape != q.shape[:1] or context_lens.dtype != jnp.int32:
        raise ValueError("Context lengths must be int32 [T]")
    if block_table.ndim != 2 or query_start_loc.shape != (block_table.shape[0] + 1,):
        raise ValueError("Page table and query offsets must cover the same requests")
    tokens, heads, dim = q.shape
    tokens = tokens if query_count is None else query_count
    banks = kv_cache if isinstance(kv_cache, tuple) else (kv_cache,)
    layout = cache_layout or PackedFP8Layout(
        sum(bank.shape[0] for bank in banks) // PAGE_ROWS
    )
    layout.validate(tuple(bank.shape[0] for bank in banks))
    tile = min(query_size, PACKED_FP8_MAX_QUERY_TILE)
    config = PACKED_FP8_MQA_CONFIG
    config = replace(
        config,
        query_tile=tile,
        key_tile=paged_mqa_key_tile(max_model_len, layout.page_size, config.key_tile),
    )
    width = pl.cdiv(max_model_len, config.key_tile) * config.key_tile
    if tokens == 0 or width == 0:
        return jnp.empty((tokens, max_model_len), jnp.float32)
    info = cast(pltpu.TpuInfo, pltpu.get_tpu_info())
    page_count = width // layout.page_size
    tile_pages = config.key_tile // layout.page_size
    page_ids = block_table[:, :page_count]
    page_ids = jnp.pad(
        page_ids, ((0, 0), (0, page_count - page_ids.shape[1])), constant_values=-1
    )
    aligned_pages = pl.cdiv(page_count, info.num_lanes) * info.num_lanes
    scalar_bytes = (
        4 * aligned_pages + context_lens.size + query_start_loc.size + 3
    ) * 4
    vector_bytes = (
        2 * tile * heads * dim
        + config.key_buffers * tile_pages * layout.page_rows * dim * 4
        + 4 * pl.cdiv(tile, info.num_sublanes) * info.num_sublanes * config.key_tile * 4
        + pl.cdiv(weights.shape[0], info.num_sublanes)
        * info.num_sublanes
        * info.num_lanes
        * 4
    )
    table_bytes = (
        pl.cdiv(page_ids.shape[0], info.num_sublanes)
        * info.num_sublanes
        * aligned_pages
        * 4
    )
    # Reserve half of remaining on-chip memory for compiler temporaries.
    if width > config.key_tile and (
        scalar_bytes > info.smem_capacity_bytes // 2
        or table_bytes > (info.vmem_capacity_bytes - vector_bytes) // 2
    ):

        def step(index, output):
            start = index * config.key_tile
            values = fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                jnp.maximum(context_lens - start, 0),
                lax.dynamic_slice_in_dim(
                    page_ids, index * tile_pages, tile_pages, axis=1
                ),
                query_start_loc=query_start_loc,
                max_model_len=config.key_tile,
                q_scale=q_scale,
                query_size=query_size,
                cache_layout=layout,
                query_start=query_start,
                query_count=query_count,
            )
            return lax.dynamic_update_slice_in_dim(output, values, start, axis=1)

        return lax.fori_loop(
            0, width // config.key_tile, step, jnp.empty((tokens, width), jnp.float32)
        )[:, :max_model_len]
    table = _encode_page_table(page_ids, layout, banks[0].shape[0], tile_pages)
    page_count = pl.cdiv(page_count, info.num_lanes) * info.num_lanes
    table = jnp.pad(table, ((0, 0), (0, page_count - table.shape[1])))
    scratch = (
        (
            pltpu.VMEM((2 * tile, heads, dim), q.dtype),
            pltpu.VMEM(
                (config.key_buffers, tile_pages * layout.page_rows, dim), jnp.int32
            ),
            pltpu.VMEM((4, tile, config.key_tile), jnp.float32),
            pltpu.SMEM((2,), jnp.int32),
            pltpu.SMEM((4 * page_count,), jnp.int32),
        ),
        tuple(
            pltpu.SemaphoreType.DMA(shape)
            for shape in ((config.key_buffers,), (4, tile), (2,), (4,))
        ),
    )
    hbm, vmem = (pl.BlockSpec(memory_space=s) for s in (pltpu.HBM, pltpu.VMEM))
    inputs = (q, weights * q_scale, banks, table)
    return pl.pallas_call(
        partial(_kernel, config=config, info=info, layout=layout),
        out_shape=jax.ShapeDtypeStruct((tokens, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=((hbm, vmem, tuple(hbm for _ in banks), vmem),),
            out_specs=hbm,
            scratch_shapes=(scratch,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=info.vmem_capacity_bytes,
            disable_bounds_checks=True,
        ),
        name="fp8_paged_mqa_logits",
    )(query_start_loc, context_lens, jnp.asarray(query_start, jnp.int32), inputs)[
        :, :max_model_len
    ]
