# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's work on the device, as one JAX function.

For each layer that has an indexer, a forward pass does three things:

1. Quantize this step's queries and keys to FP8.
2. Store the new keys in the layer's index-key cache.
3. Score every cached key against each query and keep the top-k positions.

All three run inside one custom op, so an indexer layer costs one kernel
launch per forward pass.

The index-key cache is paged like the KV cache and uses the same page table.
Each token has one record in it: the key's FP8 bytes, then one byte holding
the key's scale, then zero padding.

Step 2 is an XLA scatter of whole cache rows (see ``_scatter_whole_rows``),
or for decode-sized batches the Pallas writer in ``dsa_key_store``.
"""

import jax
import jax.numpy as jnp

from sglang_tpu.attention.pallas.dsa_key_store import store_keys_pallas
from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils
from sglang_tpu.attention.pallas.util import align_to, quantize_fp8_ue8m0
from sglang_tpu.dsv4.sparsecore_topk import BITS_NEG_INF, sparsecore_topk
from sglang_tpu.dsv4.streamindex_topk import (
    KVLayout,
    cp_local_to_global,
    streamindex_topk,
)
from sglang_tpu.dsv4.streamindex_topk.metadata import cp_global_to_local
from sglang_tpu.dsv4.streamindex_topk.streamindex_topk import (
    _select_owned_winners,
    get_kv_cache_shape,
)

# Block sizes for the top-k kernel's (decode batch, decode tail, mixed)
# passes. A decode query scores a 128-token page in a few cycles, so at 2
# pages per block the per-block overhead dominates: 16 pages cut a bs1 8K
# indexer decode step from 73 to 63 us on TPU7x. The mixed pass keeps the
# kernel authors' value; it is not tuned for GLM or Kimi.
_NUM_KV_PAGES_PER_BLOCK = (16, 16, 2)
_NUM_QUERIES_PER_BLOCK = (1, 128, 128)


def index_key_cache_shape(
    num_pages: int, page_size: int, index_head_dim: int
) -> tuple[int, ...]:
    """Shape of one layer's index-key cache, as the top-k kernel reads it.

    ``uint8[num_pages, page_size // 4, 4, record_bytes]``.

    ``record_bytes`` is the size of one token's record. The kernel computes it
    as: the key, padded to whole 128-byte lanes, plus one scale byte per lane,
    rounded up to whole lanes again. For a 128-wide key that is 128 + 1 = 129
    bytes, rounded up to 256.

    ``KVLayout.HEAD_ALONG_SUBLANE`` is not a size. It names how the kernel
    arranges the records inside a page.
    """
    return get_kv_cache_shape(
        num_pages, page_size, index_head_dim, KVLayout.HEAD_ALONG_SUBLANE
    )


def _store_keys(
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    key: jax.Array,  # fp8[max_num_tokens, index_head_dim]
    key_scale_byte: jax.Array,  # u8[max_num_tokens, 1]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> jax.Array:
    """Write this step's keys into the index-key cache and return the cache."""
    records = _key_records(cache_k, key, key_scale_byte)
    page_size = cache_k.shape[1] * cache_k.shape[2]
    # A batch is padded to a fixed number of rows. A padded row still carries
    # one query token, but it is not a real request, so its key must not be
    # stored. distribution[2] is the number of real requests; clipping
    # cu_q_lens there makes every token after them count as "not in the batch".
    active_cu_q_lens = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])
    # Find each token's page and slot the same way the KV writer does, so a
    # token's key and its KV entry always sit at the same position.
    page, slot = kv_cache_utils.get_page_and_slot(
        key.shape[0], kv_lens, page_indices, active_cu_q_lens, page_size
    )
    return _scatter_whole_rows(cache_k, records, page, slot)


def _key_records(
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    key: jax.Array,  # fp8[max_num_tokens, index_head_dim]
    key_scale_byte: jax.Array,  # u8[max_num_tokens, 1]
) -> jax.Array:  # u8[max_num_tokens, record_bytes]
    """One cache record per token: key bytes padded to whole lanes, the scale
    byte, then zeros up to the record size."""
    index_head_dim = key.shape[1]
    num_pages, rows_per_page, tokens_per_row, record_bytes = cache_k.shape
    page_size = rows_per_page * tokens_per_row
    # The pool allocates this buffer without importing the kernel, so check
    # here that its shape is the one the kernel reads.
    expected_shape = index_key_cache_shape(num_pages, page_size, index_head_dim)
    if cache_k.shape != expected_shape:
        raise ValueError(
            f"index-key cache has shape {cache_k.shape}, but the top-k kernel "
            f"reads {expected_shape}."
        )

    # Build one record per token: key bytes, padded to whole lanes, then the
    # scale byte, then zeros up to the record size.
    key_bytes = jax.lax.bitcast_convert_type(key, jnp.uint8)
    key_lanes = align_to(index_head_dim, kv_cache_utils.TILE_LANE_BYTES)
    key_bytes = jnp.pad(key_bytes, ((0, 0), (0, key_lanes - index_head_dim)))
    records = jnp.concatenate([key_bytes, key_scale_byte], axis=-1)
    return jnp.pad(records, ((0, 0), (0, record_bytes - records.shape[-1])))


def _scatter_whole_rows(
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    records: jax.Array,  # u8[num_tokens, record_bytes]
    page: jax.Array,  # i32[num_tokens]; out of range for tokens not stored
    slot: jax.Array,  # i32[num_tokens]
) -> jax.Array:
    """Write each token's record at (page, slot), one whole cache row at a time.

    The tokens of a cache row share 32-bit words, so XLA lowers a per-token
    scatter by converting the whole cache to one row per token and back: two
    full-pool copies per call. Writing whole rows keeps the native layout.

    A token's row is rebuilt from the cache plus every token of this batch that
    lands in it. Those are its neighbours on the token axis, because a
    request's tokens are contiguous there and occupy consecutive slots. Only
    the first of them writes the row, so the scatter indices are unique.
    """
    num_tokens = records.shape[0]
    num_pages, _, tokens_per_row, _ = cache_k.shape
    row, col = slot // tokens_per_row, slot % tokens_per_row
    rows = cache_k[jnp.minimum(page, num_pages - 1), row]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)

    def shifted(x, d):
        """x[t + d] for every t, edge-clamped; d is a static offset."""
        d = max(-num_tokens, min(d, num_tokens))
        if d > 0:
            return jnp.concatenate([x[d:], jnp.repeat(x[-1:], d, axis=0)])
        if d < 0:
            return jnp.concatenate([jnp.repeat(x[:1], -d, axis=0), x[: num_tokens + d]])
        return x

    # Slot j of token t's row holds token t - col[t] + j when that token is in
    # this batch and lands there. The offset d = j - col[t] takes only
    # tokens_per_row values per j, so read the neighbours through static
    # shifts instead of a gather per slot (XLA gathers of 1K rows are slow).
    new_rows = []
    for j in range(tokens_per_row):
        slot_j = rows[:, j]
        for d in range(j - tokens_per_row + 1, j + 1):
            lands_here = (
                (col == j - d)
                & (tok + d >= 0)
                & (tok + d < num_tokens)
                & (shifted(page, d) == page)
                & (shifted(row, d) == row)
                & (shifted(col, d) == j)
            )
            slot_j = jnp.where(lands_here[:, None], shifted(records, d), slot_j)
        new_rows.append(slot_j)
    rows = jnp.stack(new_rows, axis=1)
    first = (tok == 0) | (shifted(page, -1) != page) | (shifted(row, -1) != row)
    # Tokens that are not stored, and all but the first token of a row, get an
    # out-of-range page; mode="drop" skips them.
    page = jnp.where(first, page, kv_cache_utils.OOB_PAGE)
    return cache_k.at[page, row].set(rows, mode="drop", unique_indices=True)


def dsa_indexer_step(
    query: jax.Array,  # float[max_num_tokens, index_n_heads, index_head_dim]
    key: jax.Array,  # float[max_num_tokens, index_head_dim]
    head_weights: jax.Array,  # float[max_num_tokens, index_n_heads]
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    row_start: jax.Array | None = None,  # i32[1]; first scored row (token split)
    *,
    weight_scale: float,
    topk: int,
    decode_query_size: int = 1,
    segments: str = "all",
    key_store: str = "xla",
    row_count: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Store this step's index keys, then pick the top-k keys for each query.

    A key's score for a query token is
    ``sum over heads of relu(query_head . key) * head_weight``. Only keys at or
    before the query's own position are scored.

    ``weight_scale`` is a constant factor on the head weights: the indexer's
    softmax scale times ``index_n_heads ** -0.5``.

    ``decode_query_size`` is the number of query rows of each decode-range
    sequence (fixed-width speculative verify); 1 for ordinary decode.

    ``segments`` is the forward's ``PallasForwardMetadata.segments``: the
    kernel passes ``distribution`` can populate. The others are not built.

    ``key_store`` picks the index-key writer: ``"xla"`` (the whole-row
    scatter, any batch) or ``"pallas"`` (one Pallas kernel, decode-sized
    batches; same bytes).

    ``row_count`` > 0 (with ``row_start``) is the DSA indexer token split
    (attention_pallas_mla_sparse.py): the keys of every token are still
    stored, but only query rows ``[row_start, row_start + row_count)`` are
    scored and selected, and ``topk_indices`` is ``i32[row_count, topk]``.
    Each row equals the same row of an unsplit call (rows past the batch are
    ``-1``); the caller assembles the slices of the TP ranks.

    Returns ``(topk_indices, cache_k)``:

    * ``topk_indices``: ``i32[max_num_tokens, topk]``. Each row lists the
      chosen positions, counted from the start of the token's own sequence.
      The chosen positions come first, in no particular order, and the rest of
      the row is ``-1``. A token with at most ``topk`` keys gets all of them.
    * ``cache_k``: the cache with this step's keys stored.
    """
    q_fp8, q_scale, _ = quantize_fp8_ue8m0(query)
    if key_store == "xla":
        k_fp8, _, k_scale_byte = quantize_fp8_ue8m0(key)
        cache_k = _store_keys(
            cache_k, k_fp8, k_scale_byte, kv_lens, page_indices, cu_q_lens, distribution
        )
    else:
        cache_k = store_keys_pallas(
            cache_k, key, kv_lens, page_indices, cu_q_lens, distribution
        )

    # The kernel reads each key's scale from the cache. It has no input for the
    # query's scale, so multiply that into the head weight instead.
    weights = head_weights.astype(jnp.float32) * q_scale[..., 0] * weight_scale

    # Keep enable_early_exit off. It is a shortcut for batches where every
    # sequence is shorter than top-k. With it on, the server returned wrong
    # logits for the sampled token (seen with GLM-5.2 at TP8). The cause is
    # not known yet.
    # TODO(sglang-torchtpu#1072): find the cause, then turn it on.
    topk_indices = streamindex_topk(
        q_fp8,
        weights,
        cache_k,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=topk,
        compression_ratio=1,
        num_kv_pages_per_block=_NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
        decode_query_size=decode_query_size,
        segments=segments,
        enable_early_exit=False,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        **(
            dict(row_start=row_start.reshape(()), row_count=row_count)
            if row_count
            else {}
        ),
    )
    return topk_indices, cache_k


def store_index_keys_dcp(
    key: jax.Array,  # float[max_num_tokens, index_head_dim]
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]; this rank's shard
    seq_lens: jax.Array,  # i32[max_num_seqs]; global, counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]; widened pages
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    dcp_rank: jax.Array,  # i32[1]
    *,
    dcp_size: int,
    interleave_size: int,
) -> jax.Array:
    """Store the keys of this step this rank owns in its index-key shard, at
    the (page, slot) the MLA latent writer uses for them."""
    k_fp8, _, k_scale_byte = quantize_fp8_ue8m0(key)
    records = _key_records(cache_k, k_fp8, k_scale_byte)
    active_cu_q_lens = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])
    page, slot = kv_cache_utils.get_dcp_page_and_slot(
        key.shape[0],
        seq_lens,
        page_indices,
        active_cu_q_lens,
        dcp_rank.reshape(()),
        page_size=cache_k.shape[1] * cache_k.shape[2],
        num_pages=cache_k.shape[0],
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    return _scatter_whole_rows(cache_k, records, page, slot)


def dsa_indexer_step_dcp(
    query: jax.Array,  # float[max_num_tokens, index_n_heads, index_head_dim]
    key: jax.Array,  # float[max_num_tokens, index_head_dim]
    head_weights: jax.Array,  # float[max_num_tokens, index_n_heads]
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]; this rank's shard
    seq_lens: jax.Array,  # i32[max_num_seqs]; global, counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]; widened pages
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    dcp_rank: jax.Array,  # i32[1]
    *,
    weight_scale: float,
    topk: int,
    dcp_size: int,
    interleave_size: int,
    segments: str = "all",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``dsa_indexer_step`` over a DCP position-sharded index-key cache.

    Each rank stores only the keys it owns (``get_dcp_page_and_slot``, the
    rule the MLA latent writer uses) and scores only its shard, so its top-k
    is the best ``topk`` of the keys it owns. Every key of the global top-k is
    in the top-k of its owner, so ``merge_dcp_topk`` over all ranks'
    candidates recovers the global top-k.

    Returns ``(global_indices, scores, cache_k)``: ``i32[max_num_tokens,
    topk]`` positions counted from the start of the token's sequence (-1
    padded), their scores as ``i32`` f32 bits, and the updated shard.
    """
    rank = dcp_rank.reshape(())
    q_fp8, q_scale, _ = quantize_fp8_ue8m0(query)
    cache_k = store_index_keys_dcp(
        key,
        cache_k,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        dcp_rank,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )

    weights = head_weights.astype(jnp.float32) * q_scale[..., 0] * weight_scale
    local_indices, scores = streamindex_topk(
        q_fp8,
        weights,
        cache_k,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=topk,
        compression_ratio=1,
        num_kv_pages_per_block=_NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
        segments=segments,
        enable_early_exit=False,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        cp_size=dcp_size,
        cp_rank=rank,
        interleave_size=interleave_size,
        return_scores=True,
    )
    # streamindex_topk returns the shard's rank-local positions under cp_size > 1.
    global_indices = jnp.where(
        local_indices >= 0,
        cp_local_to_global(local_indices, rank, dcp_size, interleave_size),
        -1,
    )
    return global_indices, scores, cache_k


def merge_dcp_topk(
    cand_indices: jax.Array,  # i32[dcp_size, max_num_tokens, topk]; global, -1 padded
    cand_scores: jax.Array,  # i32[dcp_size, max_num_tokens, topk]; f32 bits
    dcp_rank: jax.Array,  # i32[1]
    *,
    topk: int,
    dcp_size: int,
    interleave_size: int,
) -> jax.Array:  # i32[max_num_tokens, topk]
    """The global top-k of every rank's candidates, as this rank's rows.

    Returns the winners this rank owns, left-packed as rank-local positions
    (the coordinates its shard and the sparse kernel use), -1 after them.
    Same algorithm as ``streamindex_topk_dcp``, with the candidates gathered
    by the caller.
    """
    num_tokens = cand_indices.shape[1]
    indices = jnp.moveaxis(cand_indices, 0, 1).reshape(num_tokens, dcp_size * topk)
    scores = jnp.moveaxis(cand_scores, 0, 1).reshape(num_tokens, dcp_size * topk)
    scores = jnp.where(indices >= 0, scores, jnp.int32(BITS_NEG_INF))
    slots = sparsecore_topk(
        scores,
        topk,
        row_lengths=jnp.full((num_tokens,), dcp_size * topk, dtype=jnp.int32),
        write_empty_rows=True,
    )
    merged = jnp.take_along_axis(indices, jnp.maximum(slots, 0), axis=1)
    merged = jnp.where(slots >= 0, merged, -1)
    return _select_owned_winners(
        merged, dcp_size, interleave_size, dcp_rank.reshape(())
    )


def _sortable_u32(score_bits: jax.Array) -> jax.Array:
    """f32 bits (as i32) to u32 keys whose unsigned order is the float order."""
    flipped = jnp.where(score_bits < 0, score_bits ^ jnp.int32(0x7FFFFFFF), score_bits)
    return jax.lax.bitcast_convert_type(flipped, jnp.uint32) ^ jnp.uint32(0x80000000)


def merge_dcp_topk_threshold(
    all_scores: jax.Array,  # i32[dcp_size, max_num_tokens, topk]; f32 bits, every rank
    own_indices: jax.Array,  # i32[max_num_tokens, topk]; this rank's, global, -1 padded
    own_scores: jax.Array,  # i32[max_num_tokens, topk]; this rank's f32 bits
    dcp_rank: jax.Array,  # i32[1]
    *,
    topk: int,
    dcp_size: int,
    interleave_size: int,
) -> jax.Array:  # i32[max_num_tokens, topk]
    """``merge_dcp_topk`` without gathering the candidates' indices.

    The global top-k is every candidate above the k-th largest score ``t``
    plus as many ties at ``t`` as fill k, ties taken in (rank, slot) order.
    ``t`` comes from a 32-step bitwise search over counts, so no rank needs
    another rank's indices: each keeps its own winners and left-packs them
    as rank-local positions. Every rank must gather its scores with -inf at
    its invalid (-1) candidates, since the merge never sees another rank's
    indices.
    """
    rank = dcp_rank.reshape(())
    neg_inf = jnp.int32(BITS_NEG_INF)
    own_valid = own_indices >= 0
    own_keys = _sortable_u32(jnp.where(own_valid, own_scores, neg_inf))
    keys = _sortable_u32(all_scores)  # [dcp, T, k]
    keys = jnp.moveaxis(keys, 0, 1).reshape(keys.shape[1], dcp_size * topk)

    def bit_step(bit, threshold):
        candidate = threshold | (jnp.uint32(1) << (31 - bit).astype(jnp.uint32))
        count = jnp.sum(keys >= candidate[:, None], axis=1, dtype=jnp.int32)
        return jnp.where(count >= topk, candidate, threshold)

    threshold = jax.lax.fori_loop(
        0, 32, bit_step, jnp.zeros((keys.shape[0],), jnp.uint32)
    )[:, None]
    above = jnp.sum(keys > threshold, axis=1, dtype=jnp.int32)[:, None]
    ties_needed = topk - above
    per_rank_ties = jnp.sum(
        (keys == threshold).reshape(keys.shape[0], dcp_size, topk),
        axis=2,
        dtype=jnp.int32,
    )  # [T, dcp]
    lower = jnp.arange(dcp_size, dtype=jnp.int32)[None, :] < rank
    ties_before = jnp.sum(jnp.where(lower, per_rank_ties, 0), axis=1, keepdims=True)
    own_tie = own_valid & (own_keys == threshold)
    tie_slot = jnp.cumsum(own_tie, axis=1, dtype=jnp.int32) - own_tie
    mine = own_valid & (
        (own_keys > threshold) | (own_tie & (ties_before + tie_slot < ties_needed))
    )
    local = cp_global_to_local(jnp.maximum(own_indices, 0), dcp_size, interleave_size)
    # Left-pack with a stable two-operand sort: an XLA scatter or gather of
    # these [T, topk] rows costs ~0.5-1 ms per layer on TPU at T = 64.
    _, packed = jax.lax.sort(
        (jnp.where(mine, 0, 1).astype(jnp.int32), jnp.where(mine, local, -1)),
        dimension=1,
        is_stable=True,
        num_keys=1,
    )
    return packed
