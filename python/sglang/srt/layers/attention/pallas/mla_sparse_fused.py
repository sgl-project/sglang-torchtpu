# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention over the fused FP8 latent cache.

The sparse MLA kernels (``mla_sparse``) read a split cache, one NoPE buffer
and one RoPE buffer per layer. ``TpuFusedMLATokenToKVPool`` allocates one fused
buffer per layer instead. Its FP8 token record already holds the same bytes,
512 of NoPE then 64 of RoPE then padding, so this module derives the split
buffers from it inside the op, runs the write and the attention, and merges
them back.

The derivation is a slice and the merge is a concatenation: each copies the
whole pool, per layer, per step. That is acceptable for the pool sizes a 2k
context needs and nothing more.
TODO(sglang-torchtpu#1072): drop this module once a pool allocates the split
buffers natively.

The top-k rows come from the DSA indexer.
"""

import jax
import jax.numpy as jnp

from sglang.srt.layers.attention.pallas.mla_sparse import kv_cache_utils
from sglang.srt.layers.attention.pallas.mla_sparse.kernel import (
    TENSORCORE_CACHE_LAYOUT,
    sparse_ragged_paged_attention,
)
from sglang.srt.layers.attention.pallas.mla_sparse.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

# The only geometry the sparse kernels accept.
NOPE_DIM = 512
ROPE_DIM = 64
# A RoPE token is stored zero-padded to one 128-byte lane row.
_ROPE_ROW_BYTES = kv_cache_utils.TILE_LANE_BYTES
_FUSED_TOKEN_BYTES = NOPE_DIM + _ROPE_ROW_BYTES


def _split_specs(
    kv_cache_fused: jax.Array,
) -> tuple[SparseMLAKVCacheSpec, SparseMLAKVCacheSpec]:
    num_pages, rows_per_page, kv_packing, token_bytes = kv_cache_fused.shape
    if kv_cache_fused.dtype != jnp.float8_e4m3fn or token_bytes != _FUSED_TOKEN_BYTES:
        raise ValueError(
            "sparse MLA needs the fused latent cache as float8_e4m3fn with "
            f"{_FUSED_TOKEN_BYTES}-byte tokens, got {kv_cache_fused.dtype} "
            f"with {token_bytes}."
        )
    page_size = rows_per_page * kv_packing
    # TENSORCORE keeps a token's bytes contiguous, which is how the fused
    # record already stores them; SPARSECORE would need a byte transpose.
    return (
        SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, KVCacheLayout.TENSORCORE, num_pages, page_size, NOPE_DIM
        ),
        SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, KVCacheLayout.TENSORCORE, num_pages, page_size, ROPE_DIM
        ),
    )


def sparse_mla_attention_fused(
    ql_nope: jax.Array,  # bf16[max_num_tokens, num_q_heads, 512]
    q_pe: jax.Array,  # bf16[max_num_tokens, num_q_heads, 64]
    new_kv_c: jax.Array,  # fp8[max_num_tokens, 512]
    new_k_pe: jax.Array,  # fp8[max_num_tokens, 64]
    kv_cache_fused: jax.Array,  # fp8[num_pages, page_size // 4, 4, 640]
    kv_lens: jax.Array,  # i32[max_num_seqs], includes this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the active sequence count
    topk_indices: jax.Array,  # i32[max_num_tokens, index_topk], from the indexer
    *,
    sm_scale: float,
    k_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """One step on the fused cache: write the new latents, then attend.

    A row of ``topk_indices`` holds positions within the token's own sequence,
    valid ones first, then ``-1``. Returns the latent output
    ``[max_num_tokens, num_q_heads, 512]`` and the updated fused cache.
    """
    if new_kv_c.shape[-1] != NOPE_DIM or new_k_pe.shape[-1] != ROPE_DIM:
        raise ValueError(
            f"sparse MLA needs a {NOPE_DIM}-wide latent and a {ROPE_DIM}-wide rope, "
            f"got {new_kv_c.shape[-1]} and {new_k_pe.shape[-1]}."
        )
    nope_spec, rope_spec = _split_specs(kv_cache_fused)

    token_rows = jax.lax.bitcast_convert_type(kv_cache_fused, jnp.uint8).reshape(
        -1, _FUSED_TOKEN_BYTES
    )
    cache_nope = token_rows[:, :NOPE_DIM].reshape(nope_spec.shape)
    cache_rope = token_rows[:, NOPE_DIM:].reshape(rope_spec.shape)

    # Padded batch rows carry a query token as well; the writer would store it.
    active_cu_q_lens = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])
    cache_nope, cache_rope = kv_cache_utils.update_sparse_mla_kv_cache(
        cache_nope,
        cache_rope,
        new_kv_c,
        new_k_pe,
        kv_lens,
        page_indices,
        active_cu_q_lens,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
    )

    out = sparse_ragged_paged_attention(
        jnp.concatenate([ql_nope, q_pe], axis=-1),
        cache_nope,
        cache_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        cache_layout=TENSORCORE_CACHE_LAYOUT,
    )

    token_rows = jnp.concatenate(
        [
            cache_nope.reshape(-1, NOPE_DIM),
            cache_rope.reshape(-1, _ROPE_ROW_BYTES),
        ],
        axis=-1,
    )
    kv_cache_fused = jax.lax.bitcast_convert_type(
        token_rows.reshape(kv_cache_fused.shape), kv_cache_fused.dtype
    )
    # The kernel attends over nope + rope keys; only the nope part is the value.
    return out[..., :NOPE_DIM], kv_cache_fused
