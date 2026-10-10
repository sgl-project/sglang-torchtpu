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

Step 2 is a plain XLA scatter.
TODO(sglang-torchtpu#1072): use the Pallas key writer once it is ported.
"""

import jax
import jax.numpy as jnp

from sglang.srt.hardware_backend.tpu.dsv4.streamindex_topk import (
    KVLayout,
    streamindex_topk,
)
from sglang.srt.hardware_backend.tpu.dsv4.streamindex_topk.streamindex_topk import (
    get_kv_cache_shape,
)
from sglang.srt.layers.attention.pallas.mla_sparse import kv_cache_utils
from sglang.srt.layers.attention.pallas.util import align_to, quantize_fp8_ue8m0

# Block sizes for the top-k kernel's (decode, prefill, mixed) passes. These
# are the values the kernel's authors use; they are not tuned for GLM.
_NUM_KV_PAGES_PER_BLOCK = (2, 2, 2)
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
    # Tokens that are not in the batch get an out-of-range page index.
    # mode="drop" makes the scatter skip them.
    return cache_k.at[page, slot // tokens_per_row, slot % tokens_per_row].set(
        records, mode="drop"
    )


def dsa_indexer_step(
    query: jax.Array,  # float[max_num_tokens, index_n_heads, index_head_dim]
    key: jax.Array,  # float[max_num_tokens, index_head_dim]
    head_weights: jax.Array,  # float[max_num_tokens, index_n_heads]
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    *,
    weight_scale: float,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Store this step's index keys, then pick the top-k keys for each query.

    A key's score for a query token is
    ``sum over heads of relu(query_head . key) * head_weight``. Only keys at or
    before the query's own position are scored.

    ``weight_scale`` is a constant factor on the head weights: the indexer's
    softmax scale times ``index_n_heads ** -0.5``.

    Returns ``(topk_indices, cache_k)``:

    * ``topk_indices``: ``i32[max_num_tokens, topk]``. Each row lists the
      chosen positions, counted from the start of the token's own sequence.
      The chosen positions come first, in no particular order, and the rest of
      the row is ``-1``. A token with at most ``topk`` keys gets all of them.
    * ``cache_k``: the cache with this step's keys stored.
    """
    q_fp8, q_scale, _ = quantize_fp8_ue8m0(query)
    k_fp8, _, k_scale_byte = quantize_fp8_ue8m0(key)
    cache_k = _store_keys(
        cache_k, k_fp8, k_scale_byte, kv_lens, page_indices, cu_q_lens, distribution
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
        enable_early_exit=False,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    )
    return topk_indices, cache_k
