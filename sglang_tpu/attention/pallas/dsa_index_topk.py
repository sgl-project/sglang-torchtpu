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
from sglang_tpu.dsv4.streamindex_topk import (
    KVLayout,
    streamindex_topk,
)
from sglang_tpu.dsv4.streamindex_topk.streamindex_topk import (
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
    num_tokens, index_head_dim = key.shape
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
    records = jnp.pad(records, ((0, 0), (0, record_bytes - records.shape[-1])))

    # A batch is padded to a fixed number of rows. A padded row still carries
    # one query token, but it is not a real request, so its key must not be
    # stored. distribution[2] is the number of real requests; clipping
    # cu_q_lens there makes every token after them count as "not in the batch".
    active_cu_q_lens = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])
    # Find each token's page and slot the same way the KV writer does, so a
    # token's key and its KV entry always sit at the same position.
    page, slot = kv_cache_utils.get_page_and_slot(
        num_tokens, kv_lens, page_indices, active_cu_q_lens, page_size
    )
    return _scatter_whole_rows(cache_k, records, page, slot)


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

    ``row_count`` > 0 (with ``row_start``) is the token split
    (SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT): the keys of every token are still
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
