# SPDX-License-Identifier: Apache-2.0
"""One step of sparse MLA attention, as one JAX function.

On each forward pass, an attention layer does two things:

1. Store this step's new tokens in the layer's KV cache.
2. Attend, for each query token, over the keys the DSA indexer picked for it.

The sparse MLA kernels (``mla_sparse``) keep a layer's cache in two buffers
and have a separate call for each of the two steps. This module puts them
together for the attention op.

The two buffers hold FP8 values as raw bytes:

* NoPE, ``uint8[num_pages, page_size, 4, 128]``: 512 bytes per token.
* RoPE, ``uint8[num_pages, page_size // 4, 4, 128]``: 64 bytes per token,
  padded to 128.

``TpuSplitMLATokenToKVPool`` allocates them. The kernels call this the
TensorCore layout; it keeps all of a token's bytes together.
TODO(sglang-torchtpu#1072): measure the kernels' other layout (SparseCore),
which is meant to gather keys faster.
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


def split_cache_specs(
    num_pages: int, page_size: int
) -> tuple[SparseMLAKVCacheSpec, SparseMLAKVCacheSpec]:
    """Describe one layer's NoPE and RoPE buffers.

    A spec gives a buffer's shape and dtype. The kernels' cache writer checks
    the buffers it is given against these specs.
    """
    return (
        SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, KVCacheLayout.TENSORCORE, num_pages, page_size, NOPE_DIM
        ),
        SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, KVCacheLayout.TENSORCORE, num_pages, page_size, ROPE_DIM
        ),
    )


def sparse_mla_attention(
    ql_nope: jax.Array,  # bf16[max_num_tokens, num_q_heads, 512]
    q_pe: jax.Array,  # bf16[max_num_tokens, num_q_heads, 64]
    new_kv_c: jax.Array,  # fp8[max_num_tokens, 512]
    new_k_pe: jax.Array,  # fp8[max_num_tokens, 64]
    cache_nope: jax.Array,  # u8[num_pages, page_size, 4, 128]
    cache_rope: jax.Array,  # u8[num_pages, page_size // 4, 4, 128]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    topk_indices: jax.Array,  # i32[max_num_tokens, index_topk], from the indexer
    *,
    sm_scale: float,
    k_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run one step: store the new tokens in the cache, then attend.

    Each row of ``topk_indices`` lists the key positions one query token may
    attend to, counted from the start of its own sequence. The valid positions
    come first and the rest of the row is ``-1``.

    Returns ``(output, cache_nope, cache_rope)``: the attention output
    ``[max_num_tokens, num_q_heads, 512]`` and the two updated buffers.
    """
    if new_kv_c.shape[-1] != NOPE_DIM or new_k_pe.shape[-1] != ROPE_DIM:
        raise ValueError(
            f"sparse MLA needs a {NOPE_DIM}-wide latent and a {ROPE_DIM}-wide rope, "
            f"got {new_kv_c.shape[-1]} and {new_k_pe.shape[-1]}."
        )
    # The pool allocates the buffers without importing the kernels. The writer
    # below checks them against these specs, so a buffer with the wrong shape
    # or dtype fails here instead of being written with a wrong layout.
    num_pages, page_size = cache_nope.shape[:2]
    nope_spec, rope_spec = split_cache_specs(num_pages, page_size)

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
    # The kernel returns 512 NoPE + 64 RoPE values per head. Only the NoPE part
    # is the attention output.
    return out[..., :NOPE_DIM], cache_nope, cache_rope
