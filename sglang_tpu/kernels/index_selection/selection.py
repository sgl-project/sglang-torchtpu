# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm-torchtpu PR #2430, revision 6fc30f15d40c.
"""Shape-generic FP8 index selection with exact per-query candidate semantics.

Q is E4M3 [T,32,128], weights are FP32 with Q scales already folded in.
All score paths use fp8_paged_mqa_logits and all selections use streaming_topk.
Candidate publications are [T,C], NOT a representative shared by a request.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .block_max import representative_block_max
from .fp8_candidate_cache import canonical_candidate_blocks, gather_candidate_keys
from .fp8_paged_mqa_logits import fp8_paged_mqa_logits
from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .streaming_topk import streaming_topk
from .tuned_block_sizes import (
    PACKED_FP8_DENSE_TOPK_THRESHOLD,
    PACKED_FP8_MQA_CONFIG,
    PACKED_FP8_QUERY_ROWS,
    PACKED_FP8_SCORE_ELEMENTS,
    paged_mqa_key_tile,
)


class RequestCandidates(NamedTuple):
    """Stage-local packed union; retain the exact publication for each query.

    Reuse only for the same K source/version AND candidate publication/version.
    keys is a compact FP8 owner, block_ids [G,U] maps its slots to logical blocks,
    pages [G,P] addresses keys, requests [T] owns queries, candidates [T,C] is
    the original publication, valid_queries [T] masks invalid request IDs, and
    bypass is the original batch-wide short-context exemption.
    """

    keys: jax.Array
    block_ids: jax.Array
    pages: jax.Array
    requests: jax.Array
    candidates: jax.Array
    valid_queries: jax.Array
    bypass: jax.Array


def _ceil(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _geometry(width, block, topk):
    if width < 1 or block not in (8, 16, 32, 64, 128) or topk < 1:
        raise ValueError("Require positive width/topk and candidate blocks 8..128")
    # Also aligns block maxima to SparseCore lanes, and full packed pages.
    return _ceil(max(width, topk), math.lcm(256, block * 128))


def _validate(q, weights, requests, lengths):
    if q.ndim != 3 or q.shape[1:] != (32, 128) or q.dtype != jnp.float8_e4m3fn:
        raise ValueError("Indexer queries must be E4M3 [T,32,128]")
    if weights.shape != q.shape[:2] or weights.dtype != jnp.float32:
        raise ValueError("Indexer weights must be FP32 [T,32] with Q scales folded in")
    if requests.shape != q.shape[:1] or lengths.shape != q.shape[:1]:
        raise ValueError("Require one request ID and causal length per query")
    if requests.dtype != jnp.int32 or lengths.dtype != jnp.int32:
        raise ValueError("Request IDs and causal lengths must be int32")


def _ordered(requests, lengths, groups, width):
    valid = (requests >= 0) & (requests < groups)
    owners = jnp.clip(requests, 0, groups - 1)
    order = jnp.argsort(owners, stable=True)
    lengths = jnp.where(valid, jnp.clip(lengths, 0, width), 0)
    counts = jnp.bincount(owners, length=groups)
    offsets = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(counts)))
    return owners, order, lengths, offsets


def _tile_rows(tokens, width, page_size):
    key_tile = paged_mqa_key_tile(width, page_size, PACKED_FP8_MQA_CONFIG.key_tile)
    capacity = _ceil(width, key_tile)
    return min(
        tokens,
        max(1, min(PACKED_FP8_QUERY_ROWS, PACKED_FP8_SCORE_ELEMENTS // capacity)),
    )


def _score(q, weights, lengths, banks, pages, offsets, layout, width):
    return fp8_paged_mqa_logits(
        q,
        banks,
        weights,
        lengths,
        pages,
        query_start_loc=offsets,
        max_model_len=width,
        q_scale=jnp.ones_like(weights),
        query_size=PACKED_FP8_MQA_CONFIG.query_tile,
        cache_layout=layout,
    )


def _topk(scores, lengths, k, logical_blocks=None):
    selected = streaming_topk(
        scores,
        valid_lengths=lengths,
        k=k,
        exclude_neg_inf=True,
        reconstruct_threshold=PACKED_FP8_DENSE_TOPK_THRESHOLD,
    )
    if logical_blocks is None:
        return selected
    # Keep context-sized mapping in HBM, not SparseCore VMEM. Sorted unique
    # block IDs preserve compact-slot order and ties: map only the K winners.
    records_per_block = scores.shape[1] // logical_blocks.shape[1]
    blocks = jnp.take_along_axis(
        logical_blocks, jnp.maximum(selected, 0) // records_per_block, axis=1
    )
    return jnp.where(
        selected >= 0, blocks * records_per_block + selected % records_per_block, -1
    )


def select_dense(
    q,
    weights,
    banks,
    pages,
    requests,
    lengths,
    candidates,
    *,
    layout: PackedFP8Layout,
    score_width: int,
    topk: int = 512,
    block_size: int = 8,
    publish: bool = False,
):
    """Return selected [T,topk] and per-query candidate blocks [T,C].

    Pages contain physical page IDs, not record offsets. Invalid owners and
    zero-length padding select only -1. Unsorted/ragged requests are supported.
    The newest visible block is retained in every nonempty publication.
    """
    _validate(q, weights, requests, lengths)
    tokens = q.shape[0]
    width = _geometry(score_width, block_size, topk)
    if not tokens:
        return jnp.full((0, topk), -1, jnp.int32), candidates
    if not pages.shape[0]:
        return jnp.full((tokens, topk), -1, jnp.int32), jnp.full_like(candidates, -1)
    _owners, order, lengths, offsets = _ordered(
        requests, lengths, pages.shape[0], score_width
    )
    q, weights, lengths = q[order], weights[order], lengths[order]
    rows = _tile_rows(tokens, width, layout.page_size)
    padded = _ceil(tokens, rows)
    padding = padded - tokens
    q = jnp.pad(q, ((0, padding), (0, 0), (0, 0)))
    weights = jnp.pad(weights, ((0, padding), (0, 0)))
    lengths = jnp.pad(lengths, (0, padding))
    published = jnp.pad(candidates[order], ((0, padding), (0, 0)), constant_values=-1)
    # Zero-length padding belongs to the final request, keeping offsets sorted.
    offsets = offsets.at[-1].set(padded)
    selected = jnp.full((padded, topk), -1, jnp.int32)

    def tile(index, state):
        selected, published = state
        start = index * rows
        queries = jax.lax.dynamic_slice_in_dim(q, start, rows)
        visible = jax.lax.dynamic_slice_in_dim(lengths, start, rows)
        scores = _score(
            queries,
            jax.lax.dynamic_slice_in_dim(weights, start, rows),
            visible,
            banks,
            pages,
            jnp.clip(offsets - start, 0, rows),
            layout,
            width,
        )
        selected = jax.lax.dynamic_update_slice_in_dim(
            selected, _topk(scores, visible, topk), start, axis=0
        )
        if publish and candidates.shape[1]:
            maxima = representative_block_max(scores, block=block_size)
            last = (visible - 1) // block_size
            maxima = jnp.where(
                (jnp.arange(maxima.shape[1])[None, :] == last[:, None])
                & (visible[:, None] > 0),
                jnp.inf,
                maxima,
            )
            # Fixed C even when width contains fewer than C blocks. Padding
            # rather than a different selector keeps the same top-k contract.
            maxima = jnp.pad(
                maxima,
                ((0, 0), (0, max(0, candidates.shape[1] - maxima.shape[1]))),
                constant_values=-jnp.inf,
            )
            chosen = _topk(
                maxima, (visible + block_size - 1) // block_size, candidates.shape[1]
            )
            published = jax.lax.dynamic_update_slice_in_dim(
                published, chosen, start, axis=0
            )
        return selected, published

    selected, published = jax.lax.fori_loop(
        0, padded // rows, tile, (selected, published)
    )
    inverse = jnp.argsort(order)
    return selected[:tokens][inverse], published[:tokens][inverse]


def prepare_selection(
    banks,
    pages,
    requests,
    lengths,
    candidates,
    *,
    layout: PackedFP8Layout,
    score_width: int,
    block_size: int = 8,
    topk: int = 512,
    max_queries_per_request: int | None = None,
):
    """Gather the exact union of per-query candidates once per request.

    Union capacity is min(context blocks, M*C + ceil(topk/block)), rounded to
    packed pages; M defaults to T. A request union may cover the full context;
    no representative truncation is permitted. The [G,context_blocks] bitmap
    is temporary, not [T,context_keys].

    ``max_queries_per_request`` is a STATIC caller-proven bound on the count
    of query rows for every valid request ID. Pass 1 for decode and the host
    metadata's largest extend length for prefill (including any valid-owner
    padding rows), or omit it for the safe T bound. The caller must validate
    this invariant when building metadata: an undersized bound can truncate
    a union. We deliberately do not synchronize device IDs to the host to
    validate it here. This bound changes allocation, never scorer routing.
    """
    _geometry(score_width, block_size, topk)
    groups, tokens = pages.shape[0], requests.size
    if max_queries_per_request is None:
        max_queries_per_request = tokens
    elif type(max_queries_per_request) is not int or max_queries_per_request < 1:
        raise ValueError("max_queries_per_request must be a positive static integer")
    max_queries_per_request = min(tokens, max_queries_per_request)
    blocks = (score_width + block_size - 1) // block_size
    prefix_blocks = (min(topk, score_width) + block_size - 1) // block_size
    capacity = min(
        blocks, max_queries_per_request * candidates.shape[1] + prefix_blocks
    )
    capacity = _ceil(max(1, capacity), PAGE_RECORDS // block_size)
    valid = (requests >= 0) & (requests < groups)
    owners = jnp.clip(requests, 0, max(0, groups - 1))
    live_lengths = jnp.where(valid, jnp.clip(lengths, 0, score_width), 0)
    bypass = jnp.max(live_lengths, initial=0) <= topk
    if not tokens or not groups:
        return RequestCandidates(
            jnp.zeros((PAGE_ROWS, PAGE_RECORDS), jnp.int32),
            jnp.full((groups, capacity), -1, jnp.int32),
            jnp.full((groups, capacity * block_size // PAGE_RECORDS), -1, jnp.int32),
            requests,
            candidates,
            valid,
            bypass,
        )
    active = valid[:, None] & (candidates >= 0) & (candidates < blocks)
    membership = jnp.zeros((groups, blocks + 1), jnp.bool_)
    # Invalid entries target a discarded sentinel column. Scatter max avoids
    # false entries overwriting a true membership in duplicate writes.
    membership = membership.at[
        owners[:, None], jnp.where(active, candidates, blocks)
    ].max(active)
    present = jnp.bincount(owners, weights=valid.astype(jnp.int32), length=groups) > 0
    membership = membership.at[:, :prefix_blocks].max(bypass & present[:, None])
    union = jnp.sort(
        jnp.where(membership[:, :blocks], jnp.arange(blocks), blocks), axis=1
    )
    union = jnp.pad(
        union, ((0, 0), (0, max(0, capacity - blocks))), constant_values=blocks
    )[:, :capacity]
    union = jnp.where(union < blocks, union, -1).astype(jnp.int32)
    counts = jnp.bincount(owners, weights=valid.astype(jnp.int32), length=groups)
    offsets = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(counts)))
    canonical, physical = canonical_candidate_blocks(
        union,
        pages,
        offsets,
        block=block_size,
        width=score_width,
        cache_pages=layout.pages,
        page_size=layout.page_size,
    )
    fragment = min(block_size, layout.page_size)
    keys = gather_candidate_keys(
        banks, physical, offsets, block=fragment, layout=layout
    )
    compact_pages = jnp.arange(
        groups * capacity * block_size // PAGE_RECORDS, dtype=jnp.int32
    ).reshape(groups, -1)
    return RequestCandidates(
        keys, canonical, compact_pages, requests, candidates, valid, bypass
    )


def select_prepared(
    q,
    weights,
    lengths,
    prepared: RequestCandidates,
    *,
    score_width: int,
    topk: int = 512,
    block_size: int = 8,
):
    """Score a packed request union, applying exact per-query candidate masks."""
    keys, block_ids, pages, requests, candidates, valid, bypass = prepared
    _validate(q, weights, requests, lengths)
    tokens = q.shape[0]
    if not tokens or not pages.shape[0]:
        return jnp.full((tokens, topk), -1, jnp.int32)
    owners, order, lengths, offsets = _ordered(
        requests, lengths, pages.shape[0], score_width
    )
    lengths = jnp.where(valid, lengths, 0)
    owners, q, weights, lengths, candidates = (
        x[order] for x in (owners, q, weights, lengths, candidates)
    )
    capacity = block_ids.shape[1] * block_size
    width = _geometry(capacity, block_size, topk)
    mapping = jnp.pad(
        block_ids,
        ((0, 0), (0, width // block_size - block_ids.shape[1])),
        constant_values=-1,
    )
    layout = PackedFP8Layout(keys.shape[0] // PAGE_ROWS)
    rows = _tile_rows(tokens, width, layout.page_size)
    padded = _ceil(tokens, rows)
    padding = padded - tokens
    q = jnp.pad(q, ((0, padding), (0, 0), (0, 0)))
    weights = jnp.pad(weights, ((0, padding), (0, 0)))
    lengths = jnp.pad(lengths, (0, padding))
    candidates = jnp.pad(candidates, ((0, padding), (0, 0)), constant_values=-1)
    owners = jnp.pad(owners, (0, padding), constant_values=pages.shape[0] - 1)
    offsets = offsets.at[-1].set(padded)
    selected = jnp.full((padded, topk), -1, jnp.int32)
    logical_blocks = (score_width + block_size - 1) // block_size

    def tile(index, selected):
        start = index * rows
        queries = jax.lax.dynamic_slice_in_dim(q, start, rows)
        logical = mapping[jax.lax.dynamic_slice_in_dim(owners, start, rows)]
        visible = jax.lax.dynamic_slice_in_dim(lengths, start, rows)
        compact_lengths = jnp.where(
            logical >= 0,
            jnp.clip(visible[:, None] - logical * block_size, 0, block_size),
            0,
        ).sum(axis=1, dtype=jnp.int32)
        scores = _score(
            queries,
            jax.lax.dynamic_slice_in_dim(weights, start, rows),
            compact_lengths,
            (keys,),
            pages,
            jnp.clip(offsets - start, 0, rows),
            layout,
            width,
        )
        row_candidates = jax.lax.dynamic_slice_in_dim(candidates, start, rows)
        allowed = jnp.zeros((rows, logical_blocks + 1), jnp.bool_)
        active = (row_candidates >= 0) & (row_candidates < logical_blocks)
        allowed = allowed.at[
            jnp.arange(rows)[:, None],
            jnp.where(active, row_candidates, logical_blocks),
        ].max(active)
        eligible = (
            jnp.take_along_axis(allowed, jnp.clip(logical, 0, logical_blocks), axis=1)
            | bypass
        )
        eligible &= logical >= 0
        scores = jnp.where(jnp.repeat(eligible, block_size, axis=1), scores, -jnp.inf)
        return jax.lax.dynamic_update_slice_in_dim(
            selected, _topk(scores, compact_lengths, topk, logical), start, axis=0
        )

    selected = jax.lax.fori_loop(0, padded // rows, tile, selected)
    return selected[:tokens][jnp.argsort(order)]
