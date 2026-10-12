# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention on the fused FP8 KV cache.

The sparse MLA kernels (``mla_sparse``) expect two cache buffers per layer:
one for the NoPE part of each token and one for the RoPE part.
``TpuFusedMLATokenToKVPool`` has a single buffer per layer instead. In it,
each token is one 640-byte record: 512 NoPE bytes, 64 RoPE bytes, 64 bytes of
padding.

The bytes are the same, only the grouping differs. So this module splits the
fused buffer into the two buffers the kernels want, runs the cache write and
the attention, and joins the two buffers back into one.

The split and the join each copy the whole pool, for every layer, on every
step. The cost grows with the pool.
TODO(sglang-torchtpu#1072): remove this module once a pool allocates the two
buffers directly.
"""

import jax
import jax.numpy as jnp

from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils
from sglang_tpu.attention.pallas.mla_sparse.kernel import (
    TENSORCORE_CACHE_LAYOUT,
    sparse_ragged_paged_attention,
)
from sglang_tpu.attention.pallas.mla_sparse.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

# The sparse kernels only support these two sizes.
NOPE_DIM = 512
ROPE_DIM = 64
# In the cache, a token's 64 RoPE bytes are zero-padded to one 128-byte lane.
_ROPE_ROW_BYTES = kv_cache_utils.TILE_LANE_BYTES
_FUSED_TOKEN_BYTES = NOPE_DIM + _ROPE_ROW_BYTES


def _split_specs(
    kv_cache_fused: jax.Array,  # fp8[num_pages, page_size // 4, 4, 640]
) -> tuple[SparseMLAKVCacheSpec, SparseMLAKVCacheSpec]:
    """Describe the NoPE and RoPE buffers that hold the fused cache's tokens.

    A spec gives a buffer's shape and dtype. The kernels' cache writer needs
    both specs, and the caller reshapes the fused bytes to the specs' shapes.

    Raises ``ValueError`` if the fused cache is not FP8 with 640-byte tokens.
    """
    num_pages, rows_per_page, tokens_per_row, token_bytes = kv_cache_fused.shape
    if kv_cache_fused.dtype != jnp.float8_e4m3fn or token_bytes != _FUSED_TOKEN_BYTES:
        raise ValueError(
            "sparse MLA needs the fused latent cache as float8_e4m3fn with "
            f"{_FUSED_TOKEN_BYTES}-byte tokens, got {kv_cache_fused.dtype} "
            f"with {token_bytes}."
        )
    page_size = rows_per_page * tokens_per_row
    # The kernels support two cache layouts. TENSORCORE keeps each token's
    # bytes together, as the fused record does, so a reshape is enough.
    # SPARSECORE would need the bytes reordered.
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
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    topk_indices: jax.Array,  # i32[max_num_tokens, index_topk], from the indexer
    *,
    sm_scale: float,
    k_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Run one step: store the new tokens in the cache, then attend.

    Each row of ``topk_indices`` lists the key positions one query token may
    attend to, counted from the start of its own sequence. The valid positions
    come first and the rest of the row is ``-1``.

    Returns ``(output, kv_cache_fused)``: the attention output
    ``[max_num_tokens, num_q_heads, 512]`` and the updated cache.
    """
    if new_kv_c.shape[-1] != NOPE_DIM or new_k_pe.shape[-1] != ROPE_DIM:
        raise ValueError(
            f"sparse MLA needs a {NOPE_DIM}-wide latent and a {ROPE_DIM}-wide rope, "
            f"got {new_kv_c.shape[-1]} and {new_k_pe.shape[-1]}."
        )
    nope_spec, rope_spec = _split_specs(kv_cache_fused)

    # Split the fused cache into the two buffers the kernels read.
    # token_rows: u8[num_pages * page_size, 640], one row per token slot.
    token_rows = jax.lax.bitcast_convert_type(kv_cache_fused, jnp.uint8).reshape(
        -1, _FUSED_TOKEN_BYTES
    )
    # cache_nope: u8[num_pages, page_size, 4, 128], bytes 0..511 of each token.
    cache_nope = token_rows[:, :NOPE_DIM].reshape(nope_spec.shape)
    # cache_rope: u8[num_pages, page_size // 4, 4, 128], bytes 512..639.
    cache_rope = token_rows[:, NOPE_DIM:].reshape(rope_spec.shape)

    # A batch is padded to a fixed number of rows. A padded row still carries
    # one query token, but it is not a real request, so its token must not be
    # stored. distribution[2] is the number of real requests; clipping
    # cu_q_lens there makes the writer skip every token after them.
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

    # out: bf16[max_num_tokens, num_q_heads, 576].
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

    # Join the two buffers back into the fused layout.
    # token_rows: u8[num_pages * page_size, 640].
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
    # The kernel returns 512 NoPE + 64 RoPE values per head. Only the NoPE part
    # is the attention output.
    return out[..., :NOPE_DIM], kv_cache_fused
