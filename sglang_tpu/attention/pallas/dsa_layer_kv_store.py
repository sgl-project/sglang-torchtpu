# SPDX-License-Identifier: Apache-2.0
"""Store one step of a DSA layer's tokens without attending or scoring.

The sparse MLA step (``mla_sparse_fused``) stores the step's latents and then
attends; the indexer step (``dsa_index_topk``, or ``dsa_kpool_topk`` for the
pooled indexer) stores the step's index keys and then scores them. These
functions run only the stores, through the same writers, so the caches end up
with the bytes the full steps leave behind.
"""

import jax

from sglang_tpu.attention.pallas.dsa_index_topk import store_index_keys
from sglang_tpu.attention.pallas.dsa_kpool_topk import store_pooled_keys
from sglang_tpu.attention.pallas.mla_sparse_fused import (
    NOPE_DIM,
    ROPE_DIM,
    _store_step,
)


def _store_latents(
    cache_nope, cache_rope, new_kv_c, new_k_pe, metadata, *, write_rope
) -> tuple[jax.Array, jax.Array]:
    if new_kv_c.shape[-1] != NOPE_DIM or new_k_pe.shape[-1] != ROPE_DIM:
        raise ValueError(
            f"sparse MLA needs a {NOPE_DIM}-wide latent and a {ROPE_DIM}-wide rope, "
            f"got {new_kv_c.shape[-1]} and {new_k_pe.shape[-1]}."
        )
    return _store_step(
        cache_nope, cache_rope, new_kv_c, new_k_pe, *metadata, write_rope=write_rope
    )


def store_dsa_layer_kv(
    cache_nope: jax.Array,  # u8[num_pages, page_size, 4, 128]
    cache_rope: jax.Array,  # u8[num_pages, page_size // 4, 4, 128]
    cache_k: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    new_kv_c: jax.Array,  # fp8[max_num_tokens, 512]
    new_k_pe: jax.Array,  # fp8[max_num_tokens, 64]
    index_keys: jax.Array,  # float[max_num_tokens, index_head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    *,
    key_store: str = "xla",
    write_rope: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Plain DSA: store the step's latents and index keys; return the caches.

    ``key_store`` is the indexer step's writer choice and ``write_rope`` is
    False exactly where the attention op is built with ``zero_rope``. Padded
    requests (past ``distribution[2]``) store nothing, as in the full steps.
    """
    metadata = (kv_lens, page_indices, cu_q_lens, distribution)
    cache_nope, cache_rope = _store_latents(
        cache_nope, cache_rope, new_kv_c, new_k_pe, metadata, write_rope=write_rope
    )
    cache_k = store_index_keys(cache_k, index_keys, *metadata, key_store=key_store)
    return cache_nope, cache_rope, cache_k


def store_dsa_layer_kv_pooled(
    cache_nope: jax.Array,  # u8[num_pages, page_size, 4, 128]
    cache_rope: jax.Array,  # u8[num_pages, page_size // 4, 4, 128] or a placeholder
    raw_cache: jax.Array,  # bf16[num_raw_pages, page_size, 2 * d]: key | gate
    pooled_cache: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    new_kv_c: jax.Array,  # fp8[max_num_tokens, 512]
    new_k_pe: jax.Array,  # fp8[max_num_tokens, 64]
    keys: jax.Array,  # bf16[max_num_tokens, d]
    gates: jax.Array,  # bf16[max_num_tokens, d]
    ape: jax.Array,  # f32[kpool, d]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    raw_page_base: jax.Array,  # i32[max_num_seqs]; read only with a raw ring
    *,
    kpool: int,
    max_query_length: int,
    raw_ring_pages_cnt: int,
    write_rope: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pooled indexer: store the step's latents and raw key | gate records,
    and compress the pools the step closes; return the four caches.

    ``kpool``, ``max_query_length`` (0 for none) and ``raw_ring_pages_cnt`` are
    what the pooled indexer op passes ``kpool_indexer_step`` for the same step;
    ``write_rope`` is False exactly where the attention op is built with
    ``zero_rope``.
    """
    metadata = (kv_lens, page_indices, cu_q_lens, distribution)
    cache_nope, cache_rope = _store_latents(
        cache_nope, cache_rope, new_kv_c, new_k_pe, metadata, write_rope=write_rope
    )
    raw_cache, pooled_cache = store_pooled_keys(
        keys,
        gates,
        ape,
        raw_cache,
        pooled_cache,
        *metadata,
        kpool=kpool,
        max_query_length=max_query_length or None,
        raw_page_base=raw_page_base,
        raw_ring_pages_cnt=raw_ring_pages_cnt,
    )
    return cache_nope, cache_rope, raw_cache, pooled_cache
