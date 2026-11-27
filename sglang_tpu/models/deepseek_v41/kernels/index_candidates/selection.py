# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 candidate selection over the packed FP8 index keys.

Candidate sources score the paged cache densely and also publish each query's
best token blocks. Candidate consumers gather their request unions once with
``prepare_candidate_keys`` and score only those blocks with
``fp8_gathered_mqa_logits``. Dense scoring and top-k are the shared
``sglang_tpu.kernels.index_selection.selection``.
"""

import functools

import jax
import jax.numpy as jnp

from sglang_tpu.kernels import scheduling_group
from sglang_tpu.kernels.index_selection import streaming_topk
from sglang_tpu.kernels.index_selection.fp8.tuned import PACKED_FP8_MQA_CONFIG
from sglang_tpu.kernels.index_selection.selection import (
    TileBlocks,
    indexer_query_rows,
    pipeline_score_topk,
    select_topk_indices,
)
from sglang_tpu.models.deepseek_v41.kernels.index_candidates.candidates import (
    PreparedCandidates,
    prepare_candidates,
)
from sglang_tpu.models.deepseek_v41.kernels.index_candidates.publication import (
    publish_candidate_blocks,
)
from sglang_tpu.models.deepseek_v41.kernels.index_candidates.sparse import (
    candidate_windows,
    fp8_gathered_mqa_logits,
)


def _select_candidate_tokens(
    scores: jax.Array,
    candidates: jax.Array,
    *,
    k: int,
    block_size: int,
    unordered: bool,
    valid_lengths: jax.Array | None = None,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """Select compact scores and map winners to logical token IDs."""
    if k == 0:
        return jnp.empty((scores.shape[0], 0), jnp.int32)
    if unordered and streaming_topk.is_supported(
        scores.shape[1],
        k,
        reconstruct_threshold=streaming_topk.topk_reconstruction_threshold(k),
        logical_block_count=candidates.shape[1],
    ):
        return streaming_topk.streaming_topk(
            scores,
            valid_lengths=(
                None if valid_lengths is None else valid_lengths.reshape(-1, 1)
            ),
            logical_blocks=candidates,
            k=k,
            exclude_neg_inf=True,
            unordered=True,
            reconstruct_threshold=streaming_topk.topk_reconstruction_threshold(k),
            scheduling_group_id=scheduling_group_id,
        )
    if unordered:
        values, slots = jax.lax.top_k(scores, k)
        logical = (
            jnp.take_along_axis(candidates, slots // block_size, axis=1) * block_size
            + slots % block_size
        )
        return jnp.where(jnp.isfinite(values), logical, -1)
    slots = jnp.broadcast_to(jnp.arange(scores.shape[1]), scores.shape)
    logical = (
        jnp.take_along_axis(candidates, slots // block_size, axis=1) * block_size
        + slots % block_size
    )
    _, logical, slots = jax.lax.sort(
        (~streaming_topk.ordered_bits(scores), logical, slots), dimension=1, num_keys=2
    )
    logical, slots = logical[:, :k], slots[:, :k]
    values = jnp.take_along_axis(scores, slots, axis=1)
    valid = values > -jnp.inf
    sentinel = jnp.iinfo(jnp.int32).max
    logical = jnp.sort(jnp.where(valid, logical, sentinel))
    return jnp.where(logical < sentinel, logical, -1)


def _score_and_select_candidates(
    q,
    weights,
    lengths,
    gathered_candidates,
    topk,
    candidate_block_size,
    unordered,
    context_width,
):
    (
        candidate_keys,
        candidate_block_ids,
        query_group_offsets,
        valid_queries,
        block_spans,
    ) = gathered_candidates
    k = min(topk, context_width, candidate_block_ids.shape[1] * candidate_block_size)
    lengths = jnp.where(valid_queries, jnp.clip(lengths, 0, context_width), 0)
    windows = candidate_windows(
        candidate_block_ids, block_spans, lengths, candidate_block_size
    )
    candidate_token_counts = jnp.maximum(windows[:, 2], windows[:, 3]) - windows[:, 0]
    num_queries = q.shape[0]
    candidate_token_capacity = candidate_block_ids.shape[1] * candidate_block_size
    query_chunk_size = indexer_query_rows(num_queries, PACKED_FP8_MQA_CONFIG.query_tile)
    query_chunk_size = (
        query_chunk_size
        if unordered and num_queries % query_chunk_size == 0
        else num_queries
    )

    def score(start, group):
        scorer = (
            fp8_gathered_mqa_logits.__wrapped__
            if query_chunk_size < num_queries
            else fp8_gathered_mqa_logits
        )
        scores = scorer(
            q,
            candidate_keys,
            weights.astype(jnp.float32),
            windows,
            query_start_loc=query_group_offsets,
            output_width=candidate_token_capacity,
            query_start=start,
            query_count=query_chunk_size,
        )
        return scheduling_group.tag(scores, group)

    if unordered:

        def select(tile, scores, group):
            return _select_candidate_tokens(
                scores,
                jax.lax.dynamic_slice_in_dim(
                    candidate_block_ids, tile * query_chunk_size, query_chunk_size
                ),
                k=k,
                block_size=candidate_block_size,
                unordered=True,
                valid_lengths=jax.lax.dynamic_slice_in_dim(
                    candidate_token_counts, tile * query_chunk_size, query_chunk_size
                ),
                scheduling_group_id=group,
            )

        selected = pipeline_score_topk(
            num_queries // query_chunk_size,
            lambda tile: tile * query_chunk_size,
            score,
            select,
            lambda tile, src, dst: jax.lax.dynamic_update_slice_in_dim(
                dst, src, tile * query_chunk_size, axis=0
            ),
            jnp.full((num_queries, k), -1, jnp.int32),
            bound_lifetimes=False,
        )
    else:
        scores = score(jnp.int32(0), None)
        selected = _select_candidate_tokens(
            scores,
            candidate_block_ids,
            k=k,
            block_size=candidate_block_size,
            unordered=False,
        )
    return jnp.pad(
        selected,
        ((0, 0), (0, min(topk, context_width) - k)),
        constant_values=-1,
    )


def select_indices_and_blocks(
    q: jax.Array,
    cache: jax.Array,
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    unordered: bool,
    score_width: int,
    *,
    query_start_loc: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Score the paged cache densely and optionally publish candidate blocks."""
    dense = functools.partial(
        select_topk_indices,
        q,
        cache,
        page_table,
        weights,
        lengths,
        requests,
        batch_max,
        topk,
        unordered,
        score_width,
        query_start_loc=query_start_loc,
    )
    if not candidate_write or candidates.shape[1] * block_size >= score_width:
        return dense(), candidates
    blocks = TileBlocks(
        candidates.shape[1],
        block_size,
        True,
        functools.partial(
            publish_candidate_blocks,
            count=candidates.shape[1],
            block_size=block_size,
            unordered=unordered,
        ),
    )
    return dense(blocks)


def prepare_candidate_keys(
    cache: jax.Array,
    page_table: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    block: int,
    width: int,
) -> tuple[jax.Array, ...]:
    """Gather published candidate rows, whose IDs ascend before -1 padding."""
    return tuple(
        prepare_candidates(
            cache, page_table, candidates, requests, block, width, ascending=True
        )
    )


def select_from_prepared_candidates(
    q: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    gathered_candidates: tuple[jax.Array, ...],
    topk: int,
    candidate_block_size: int,
    unordered: bool,
    context_width: int,
) -> jax.Array:
    gathered_candidates = PreparedCandidates(*gathered_candidates)
    lengths = jnp.where(gathered_candidates.valid_queries, lengths, 0)

    def short():
        ids = jnp.broadcast_to(
            jnp.arange(min(topk, context_width), dtype=jnp.int32),
            (q.shape[0], min(topk, context_width)),
        )
        return jnp.where(ids < lengths[:, None], ids, -1)

    return jax.lax.cond(
        jnp.max(lengths) <= topk,
        short,
        lambda: _score_and_select_candidates(
            q,
            weights,
            lengths,
            gathered_candidates,
            topk,
            candidate_block_size,
            unordered,
            context_width,
        ),
    )
