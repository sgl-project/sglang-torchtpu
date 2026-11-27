# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 index selection: dense scoring with top-k.

Layers score the paged cache with ``fp8_paged_mqa_logits``; prefill bounds the
score buffer by query chunks pipelined against streaming top-k. ``TileBlocks``
also returns each row's highest-scoring key blocks, which model-specific
candidate selection builds on. Torch op registration stays with the model layer.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels import scheduling_group
from sglang_tpu.kernels.index_selection import (
    block_topk,
    streaming_topk,
)
from sglang_tpu.kernels.index_selection.fp8.mqa_logits.dense import (
    fp8_paged_mqa_logits,
)
from sglang_tpu.kernels.index_selection.fp8.tuned import (
    PACKED_FP8_MQA_CONFIG,
    paged_mqa_key_tile,
)


def pipeline_score_topk(
    num_tiles: int | jax.Array,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[jax.Array, Any, Any], Any],
    output: Any,
    *,
    bound_lifetimes: bool,
) -> Any:
    """Score query tiles and select each tile's top-k.

    Callbacks accept an optional scheduling group after their data arguments.
    Without ``bound_lifetimes``, independent tiles stay available to the
    scheduler; use for tuned small-query batches. With it, each score buffer
    waits on the previous selection; up to 16 tiles, each next-tile scoring and
    current-tile selection pair also gets a program-unique group so the compiler
    keeps TensorCore scoring between SparseCore selection start and completion.
    A traced ``num_tiles``, at most the tiles ``output`` holds, always runs the
    ungrouped loop with bounded lifetimes.
    """
    if isinstance(num_tiles, jax.Array):
        # The loop scores tile 0 before checking the count.
        count = jnp.maximum(num_tiles.astype(jnp.int32), 1)
        return _pipeline_score_topk_looped(count, prepare, score, select, write, output)
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    if not bound_lifetimes:
        for tile in range(num_tiles):
            index = jnp.int32(tile)
            scores = score(prepare(index), None)
            selected = select(index, scores, None)
            output = write(index, selected, output)
        return output
    if num_tiles > 16:
        # Tagging the loop-body pairs fails TPU compilation (libtpu 0.0.48 and
        # 0.0.50: "annotation groups with gaps"), so looped pairs stay ungrouped.
        return _pipeline_score_topk_looped(
            num_tiles, prepare, score, select, write, output
        )

    group_base = scheduling_group.reserve(num_tiles - 1) if num_tiles > 1 else 0
    # Gate current scores and next preparation on the previous selection
    # so unrolling does not extend every score buffer's lifetime.
    with jax.named_scope("ScoreTopKBoundedPipeline"):
        current = score(prepare(jnp.int32(0)), None)
        previous = None
        for tile in range(num_tiles - 1):
            prepared = prepare(jnp.int32(tile + 1))
            if previous is not None:
                previous, current, prepared = lax.optimization_barrier(
                    (previous, current, prepared)
                )
                output = write(jnp.int32(tile - 1), previous, output)
            group = group_base + tile
            following = score(prepared, group)
            previous = select(jnp.int32(tile), current, group)
            current = following
        if previous is not None:
            previous, current = lax.optimization_barrier((previous, current))
            output = write(jnp.int32(num_tiles - 2), previous, output)
        last = jnp.int32(num_tiles - 1)
        return write(last, select(last, current, None), output)


def _pipeline_score_topk_looped(num_tiles, prepare, score, select, write, output):
    """Order TensorCore scoring and SparseCore selection across two query tiles.

    The barrier gates score-buffer reuse on selection completion, before
    output scatter. Physical execution overlap depends on the compiler.
    """
    first = score(prepare(jnp.int32(0)), None)

    def advance(tile, state):
        current, result = state
        following = score(prepare(tile + 1), None)
        return following, write(tile, select(tile, current, None), result)

    def body(pair, state):
        tile = 2 * pair
        current, result = state
        following = score(prepare(tile + 1), None)
        selected = select(tile, current, None)
        prepared, selected = lax.optimization_barrier((prepare(tile + 2), selected))
        replacement = score(prepared, None)
        following_selected = select(tile + 1, following, None)
        result = write(tile, selected, result)
        result = write(tile + 1, following_selected, result)
        return replacement, result

    pairs = (num_tiles - 1) // 2
    with jax.named_scope("ScoreTopKPipeline"):
        state = lax.fori_loop(0, pairs, body, (first, output))
        if isinstance(num_tiles, jax.Array):
            # An even count leaves one unpaired tile; a 0/1-trip loop avoids a cond.
            state = lax.fori_loop(2 * pairs, num_tiles - 1, advance, state)
        elif 2 * pairs < num_tiles - 1:
            state = advance(jnp.int32(2 * pairs), state)
        last, result = state
        return write(
            jnp.int32(num_tiles - 1),
            select(jnp.int32(num_tiles - 1), last, None),
            result,
        )


_INDEX_UNTILED_SCORE_ELEMENTS = 1 << 27
# Keys per block maximum that block top-k reduces on SparseCore when the caller
# asks for no top blocks: one maximum per half 16-lane vector.
_BLOCK_TOPK_BLOCK = 8


@dataclass(frozen=True)
class TileBlocks:
    """Also return each query row's highest-scoring key blocks with its top-k.

    ``rows(scores, lengths, scheduling_group_id=None)`` returns one score tile's
    int32 [rows, count] block IDs. Decode instead takes them from
    ``block_topk``'s single SparseCore pass over the same block maxima, where
    ``force_last_block`` ranks each row's last visible block first; ``rows``
    must apply the same rule.
    """

    count: int
    block_size: int
    force_last_block: bool
    rows: Callable[..., jax.Array]


def _topk_from_scores(
    scores: jax.Array,
    lengths: jax.Array,
    topk: int,
    blocks: TileBlocks | None = None,
):
    """Ascending top-k visible key IDs per row, ``-1`` padded at the tail; with
    ``blocks``, also each row's top blocks."""
    count, width = scores.shape
    if topk < 1 or width < 1:
        raise ValueError("Positive top-k and score width required")
    lengths = jnp.clip(lengths.reshape(count), 0, width)
    scores = jnp.where(jnp.arange(width) < lengths[:, None], scores, -jnp.inf)
    values, ids = jax.lax.top_k(scores, min(topk, width))
    # Ascending keeps logical order, hence the attention accumulation order.
    limit = jnp.iinfo(jnp.int32).max
    ids = jnp.sort(jnp.where(values > -jnp.inf, ids, limit), axis=-1)
    selected = jnp.where(ids == limit, -1, ids).astype(jnp.int32)
    if blocks is None:
        return selected
    return selected, blocks.rows(scores, lengths)


def indexer_query_rows(tokens: int, query_tile: int) -> int:
    """Derive query chunks from query tiles and the SparseCore wave size."""
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        return tokens
    wave = math.lcm(query_tile, sc.num_cores * sc.num_subcores)
    chunk = math.gcd(tokens, wave)
    return chunk if chunk % query_tile == 0 else tokens


def _query_start_loc(requests: jax.Array, num_requests: int) -> jax.Array:
    """[num_requests + 1] row boundaries; padding (-1) rows join the last request."""
    last = num_requests - 1
    owners = jnp.where(requests >= 0, jnp.minimum(requests, last), last)
    bounds = jnp.arange(num_requests + 1, dtype=owners.dtype)
    return jnp.sum(owners[:, None] < bounds, axis=0, dtype=jnp.int32)


def _select_topk_indices(
    q: jax.Array,
    cache: jax.Array,
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    topk: int,
    unordered: bool,
    score_width: int,
    blocks: TileBlocks | None = None,
    *,
    query_start_loc: jax.Array | None = None,
    prefill_chunk_rows: int | None = None,
):
    page_size = cache.shape[-1]
    tokens, width = q.shape[0], score_width
    if tokens == 0:
        selected = jnp.empty((0, min(topk, width)), jnp.int32)
        if blocks is None:
            return selected
        return selected, jnp.empty((0, blocks.count), jnp.int32)
    if q.shape[1:] != (32, 128) or width < 1:
        raise ValueError(
            "Packed FP8 model scorer requires H32/D128 and nonempty context"
        )
    key_tile = paged_mqa_key_tile(width, page_size, PACKED_FP8_MQA_CONFIG.key_tile)
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    capacity = (width + key_tile - 1) // key_tile * key_tile
    pages = page_table[:, : (width + page_size - 1) // page_size]
    pages = jnp.pad(
        pages,
        ((0, 0), (0, capacity // page_size - pages.shape[1])),
        constant_values=-1,
    )

    rows = prefill_chunk_rows or min(
        max(tokens, query_tile),
        max(query_tile, _INDEX_UNTILED_SCORE_ELEMENTS // capacity),
    )
    padded = (tokens + rows - 1) // rows * rows
    valid_requests = (requests >= 0) & (requests < page_table.shape[0])
    lengths = jnp.where(valid_requests, jnp.clip(lengths, 0, width), 0)
    inputs = tuple(
        jnp.pad(x, ((0, padded - tokens),) + ((0, 0),) * (x.ndim - 1))
        for x in (q, weights, lengths)
    )
    if query_start_loc is None:
        query_start_loc = _query_start_loc(requests, page_table.shape[0])
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("query_start_loc needs one entry per page-table row, plus one")
    # Trailing padding rows, tile padding included, belong to the last request.
    query_offsets = jnp.minimum(query_start_loc.astype(jnp.int32), tokens)
    query_offsets = query_offsets.at[-1].set(padded)
    stream_select = unordered and streaming_topk.is_supported(
        width,
        topk,
        reconstruct_threshold=streaming_topk.topk_reconstruction_threshold(topk),
    )
    with_blocks = blocks is not None
    select_topk = functools.partial(
        streaming_topk.streaming_topk,
        k=topk,
        exclude_neg_inf=True,
        reconstruct_threshold=streaming_topk.topk_reconstruction_threshold(topk),
        # Attention re-sorts the records; skip ID tie-breaks and output sorting.
        unordered=unordered,
    )
    chunk_rows = prefill_chunk_rows or indexer_query_rows(tokens, query_tile)
    # Decode top-k scans only the blocks that can hold a winner. Callers asking
    # for top blocks rank the same SparseCore block maxima for them.
    select_block = blocks.block_size if with_blocks else _BLOCK_TOPK_BLOCK
    block_select = (
        stream_select
        and prefill_chunk_rows is None
        and block_topk.is_supported(chunk_rows, capacity, topk, select_block)
    )

    def score(start, rows, inputs, *, pipelined, group=None):
        """Scores of `rows` queries from `start`; decode tiles share the inputs."""
        shared = pipelined and prefill_chunk_rows is None
        queries, row_weights, row_lengths = (
            inputs
            if shared
            else (jax.lax.dynamic_slice_in_dim(x, start, rows) for x in inputs)
        )
        # Pipelined tiles inline inside the outer model jit so scheduling
        # metadata reaches the Pallas call instead of the inner jit boundary.
        scorer = fp8_paged_mqa_logits.__wrapped__ if pipelined else fp8_paged_mqa_logits
        # Block top-k and block publication callbacks may read the full score row.
        full_row_consumer = with_blocks or (pipelined and block_select)
        scores = scorer(
            queries,
            cache,
            row_weights.astype(jnp.float32),
            row_lengths,
            pages,
            query_start_loc=(
                query_offsets if shared else jnp.clip(query_offsets - start, 0, rows)
            ),
            max_model_len=capacity if stream_select else width,
            q_scale=jnp.ones_like(row_weights, dtype=jnp.float32),
            query_size=query_tile,
            query_start=start if shared else 0,
            query_count=rows,
            reuse_keys=prefill_chunk_rows is not None and capacity == key_tile,
            skip_tail_padding=stream_select and not full_row_consumer,
        )
        return scheduling_group.tag(scores, group)

    def select_serial():
        output = jnp.full((padded, min(topk, width)), -1, jnp.int32)
        if with_blocks:
            output = (output, jnp.full((padded, blocks.count), -1, jnp.int32))
        # Unroll SparseCore calls to avoid XLA loop lowering failures;
        # barriers bound score-buffer lifetimes.
        for index in range(padded // rows):
            start, output = jax.lax.optimization_barrier(
                (jnp.int32(index * rows), output)
            )
            scores = score(start, rows, inputs, pipelined=False)
            row_lengths = jax.lax.dynamic_slice_in_dim(inputs[2], start, rows)
            if not stream_select:
                selected = _topk_from_scores(
                    scores,
                    row_lengths,
                    topk,
                    blocks=blocks,
                )
            else:
                selected = select_topk(scores, valid_lengths=row_lengths)
                if with_blocks:
                    selected = (selected, blocks.rows(scores, row_lengths))
            output = jax.tree.map(
                lambda dst, src: jax.lax.dynamic_update_slice_in_dim(
                    dst, src, start, axis=0
                ),
                output,
                selected,
            )
        return jax.tree.map(lambda x: x[:tokens], output)

    if not stream_select or chunk_rows > rows or padded != tokens:
        return select_serial()

    def select(tile, scores, group):
        row_lengths = jax.lax.dynamic_slice_in_dim(
            inputs[2], tile * chunk_rows, chunk_rows
        )
        if block_select:
            # One SparseCore pass ranks the block maxima for both outputs.
            return block_topk.block_topk(
                scores,
                row_lengths,
                k=topk,
                block_size=select_block,
                top_blocks=blocks.count if with_blocks else 0,
                force_last_block=with_blocks and blocks.force_last_block,
                scheduling_group_id=group,
            )
        selected = select_topk(
            scores, valid_lengths=row_lengths, scheduling_group_id=group
        )
        if with_blocks:
            return selected, blocks.rows(scores, row_lengths, scheduling_group_id=group)
        return selected

    def write(tile, selected, output):
        update = functools.partial(
            jax.lax.dynamic_update_slice_in_dim, start_index=tile * chunk_rows, axis=0
        )
        return jax.tree.map(update, output, selected)

    def select_pipelined():
        # Score tiles may split a request; the kernel resumes it mid-group.
        output = jnp.full((tokens, topk), -1, jnp.int32)
        if with_blocks:
            output = (output, jnp.full((tokens, blocks.count), -1, jnp.int32))
        selected = pipeline_score_topk(
            padded // chunk_rows,
            lambda tile: tile * chunk_rows,
            lambda start, group: score(
                start, chunk_rows, inputs, pipelined=True, group=group
            ),
            select,
            write,
            output,
            bound_lifetimes=prefill_chunk_rows is not None or with_blocks,
        )
        if with_blocks:
            selected, top_blocks = selected
            top_blocks = jnp.where(valid_requests[:, None], top_blocks, -1)
        # Batch-wide short-context bypass: every visible key, as in the reference.
        # A select rather than a cond keeps the scoring path free of branches.
        short = jnp.minimum(batch_max.max(), width) <= topk
        ids = jnp.arange(topk, dtype=jnp.int32)
        visible = jnp.where(ids < lengths[:tokens, None], ids, -1)
        selected = jnp.where(short, visible, selected)
        return (selected, top_blocks) if with_blocks else selected

    return select_pipelined()


def select_topk_indices(
    q: jax.Array,
    cache: jax.Array,
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    topk: int,
    unordered: bool,
    score_width: int,
    blocks: TileBlocks | None = None,
    *,
    query_start_loc: jax.Array | None = None,
):
    """Select exact FP8 scores with a bounded, overlapped prefill working set.

    Rows must group each request's queries contiguously in ascending request
    order, with padding rows (``requests == -1``) trailing.

    Returns int32 [tokens, min(topk, score_width)] key IDs, -1 padded; with
    ``blocks``, also each row's int32 [tokens, blocks.count] top block IDs.
    """
    arguments = (
        q,
        cache,
        page_table,
        weights,
        lengths,
        requests,
        batch_max,
        topk,
        unordered,
    )
    page_size = cache.shape[-1]
    width = paged_mqa_key_tile(score_width, page_size, PACKED_FP8_MQA_CONFIG.key_tile)
    capacity = (score_width + width - 1) // width * width
    chunk = max(
        PACKED_FP8_MQA_CONFIG.query_tile,
        _INDEX_UNTILED_SCORE_ELEMENTS // max(capacity, 1),
    )
    if (
        width >= score_width
        or chunk >= q.shape[0]
        or q.shape[0] % chunk
        or blocks is not None
        or not unordered
        or topk > width
    ):
        return _select_topk_indices(
            *arguments, score_width, blocks, query_start_loc=query_start_loc
        )

    # Preserve score-buffer row boundaries for both widths. Only contexts
    # covered by one K tile can narrow the scores and retain their keys.
    live = (requests >= 0) & (requests < page_table.shape[0])
    longest = jnp.where(live, lengths, 0).max()
    return jax.lax.cond(
        longest <= width,
        lambda: _select_topk_indices(
            *arguments,
            width,
            query_start_loc=query_start_loc,
            prefill_chunk_rows=chunk,
        ),
        lambda: _select_topk_indices(
            *arguments,
            score_width,
            query_start_loc=query_start_loc,
            prefill_chunk_rows=chunk,
        ),
    )
