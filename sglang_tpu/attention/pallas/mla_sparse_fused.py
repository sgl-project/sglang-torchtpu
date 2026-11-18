# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention with the cache write fused in.

On each forward pass, an attention layer does two things:

1. Store this step's new tokens in the layer's KV cache.
2. Attend, for each query token, over the keys the DSA indexer picked for it.

The sparse MLA kernels (``mla_sparse``) have a separate call for each of the
two. This module fuses them into one JAX function, which the attention op
runs.

The kernels keep a layer's cache in two buffers of raw FP8 bytes:

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
    DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
    TENSORCORE_CACHE_LAYOUT,
    sparse_ragged_paged_attention,
)
from sglang_tpu.attention.pallas.mla_sparse.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from sglang_tpu.attention.pallas.mla_sparse_qblock import (
    MIN_QUERY_BLOCK,
    QUERY_BLOCK,
    sparse_mla_prefill_qblock,
    vmem_limit_bytes,
    vmem_required_by_sparse_mla,
)
from sglang_tpu.attention.pallas.mla_sparse_qblock_tokens import (
    sparse_mla_prefill_qblock_tokens,
)

# Query tokens per sparse-attention gather chunk under the q-block opt-in. The
# kernel unrolls its chunk loop at trace time, one SparseCore gather program per
# chunk; its default (64) builds 128 per layer at an 8192-token prefill, 256
# builds 32. Each chunk's gather intermediate grows with it (4x at 256), so
# models that do not opt in keep the default.
QBLOCK_GATHER_CHUNK_TOKENS = 256
# With qblock_kpool set, batches of at least this many rows take the
# query-blocked kernel. Only prefill ops set it: the caller keeps decode,
# verify and draft extend on an op without it.
QBLOCK_MIN_TOKENS = 512
# ... and only up to this many pages per request: the kernel builds its page
# membership over every (block, page, query, selected pool), so that setup
# grows with the context. Measured at 80 pages (10k context); longer contexts
# keep the per-query kernel until measured.
QBLOCK_MAX_PAGES_PER_SEQ = 128
# Query blocks the query-blocked kernel may take, largest first: its default
# and its smallest. Its VMEM grows with num_heads * query_block, so the op
# takes the largest whose VMEM estimate fits
# (mla_sparse_qblock.vmem_required_by_sparse_mla).
QBLOCK_QUERY_BLOCKS = (QUERY_BLOCK, MIN_QUERY_BLOCK)
# Plain DSA (token top-k rows, sparse_mla_prefill_fused_tokens): the bitmap
# setup grows only with the bitmap width, so the bound is the attention itself.
# A block of 256 queries with 2048 picks each touches nearly every page of the
# context: its work grows with the context while the per-query kernel's does
# not. 256 pages (32k tokens) is a conservative cut; 73 pages (9k) measured.
QBLOCK_TOKENS_MAX_PAGES_PER_SEQ = 256

# The sparse kernels only support these two sizes.
NOPE_DIM = 512
ROPE_DIM = 64


def qblock_query_block(
    num_heads: int,
    *,
    q_width: int,
    page_size: int,
    pages_per_seq: int,
    dtype: jnp.dtype,
) -> int | None:
    """Queries per block of the query-blocked kernel for ``num_heads`` heads
    per chip: the largest choice that fits in VMEM, or None if none fits."""
    limit = vmem_limit_bytes()
    return next(
        (
            qb
            for qb in QBLOCK_QUERY_BLOCKS
            if vmem_required_by_sparse_mla(
                num_heads,
                qb,
                q_width=q_width,
                page_size=page_size,
                pages_per_seq=pages_per_seq,
                dtype=dtype,
            )
            <= limit
        ),
        None,
    )


def _split_specs(
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


def _store_step(
    cache_nope: jax.Array,
    cache_rope: jax.Array,
    new_kv_c: jax.Array,  # fp8[max_num_tokens, 512]
    new_k_pe: jax.Array,  # fp8[max_num_tokens, 64]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    write_rope: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Store this step's tokens in the layer's two cache buffers.

    ``write_rope=False`` leaves the rope buffer untouched (see
    ``update_sparse_mla_kv_cache``).
    """
    # The pool allocates the buffers without importing the kernels. The writer
    # below checks them against these specs, so a buffer with the wrong shape
    # or dtype fails here instead of being written with a wrong layout.
    num_pages, page_size = cache_nope.shape[:2]
    nope_spec, rope_spec = _split_specs(num_pages, page_size)
    # A batch is padded to a fixed number of rows. A padded row still carries
    # one query token, but it is not a real request, so its token must not be
    # stored. distribution[2] is the number of real requests; clipping
    # cu_q_lens there makes the writer skip every token after them.
    active_cu_q_lens = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])
    return kv_cache_utils.update_sparse_mla_kv_cache(
        cache_nope,
        cache_rope,
        new_kv_c,
        new_k_pe,
        kv_lens,
        page_indices,
        active_cu_q_lens,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        write_rope=write_rope,
    )


def sparse_mla_attention_fused(
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
    qblock_kpool: int = 0,
    gather_chunk_tokens: int = DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
    zero_rope: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run one step: store the new tokens in the cache, then attend.

    Each row of ``topk_indices`` lists the key positions one query token may
    attend to, counted from the start of its own sequence. The valid positions
    come first and the rest of the row is ``-1``.

    ``zero_rope`` is for a model without rope, whose rope lane (``q_pe`` and
    ``new_k_pe``) is identically zero. The rope cache is allocated zeroed, so
    the step neither writes nor reads it: the rope score term is 0 either way,
    and the output does not change.

    Returns ``(output, cache_nope, cache_rope)``: the attention output
    ``[max_num_tokens, num_q_heads, 512]`` and the two updated buffers.
    """
    if new_kv_c.shape[-1] != NOPE_DIM or new_k_pe.shape[-1] != ROPE_DIM:
        raise ValueError(
            f"sparse MLA needs a {NOPE_DIM}-wide latent and a {ROPE_DIM}-wide rope, "
            f"got {new_kv_c.shape[-1]} and {new_k_pe.shape[-1]}."
        )
    cache_nope, cache_rope = _store_step(
        cache_nope,
        cache_rope,
        new_kv_c,
        new_k_pe,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        write_rope=not zero_rope,
    )

    pages_per_seq = page_indices.shape[0] // (cu_q_lens.shape[0] - 1)
    query_block = qblock_query_block(
        ql_nope.shape[1],
        q_width=ql_nope.shape[-1] + q_pe.shape[-1],
        page_size=cache_nope.shape[1],
        pages_per_seq=pages_per_seq,
        dtype=ql_nope.dtype,
    )
    if (
        qblock_kpool
        and query_block is not None
        and ql_nope.shape[0] >= QBLOCK_MIN_TOKENS
        and pages_per_seq <= QBLOCK_MAX_PAGES_PER_SEQ
        # One 32-bit selection word per page: page_size // kpool pools.
        and cache_nope.shape[1] // qblock_kpool == 32
        and topk_indices.shape[1] % qblock_kpool == 0
    ):
        # Rows from the pooled indexer (pool size qblock_kpool): one program per
        # block of queries reads the union of their pages once.
        out = sparse_mla_prefill_qblock(
            ql_nope,
            q_pe,
            cache_nope,
            cache_rope,
            topk_indices,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            kpool=qblock_kpool,
            query_block=query_block,
            read_rope=not zero_rope,
        )
        return out, cache_nope, cache_rope

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
        gather_and_attention_chunk_size=min(
            gather_chunk_tokens, max(64, ql_nope.shape[0])
        ),
        gather_rope=not zero_rope,
    )
    # The kernel returns 512 NoPE + 64 RoPE values per head. Only the NoPE part
    # is the attention output.
    return out[..., :NOPE_DIM], cache_nope, cache_rope


def qblock_tokens_supported(
    num_tokens: int, pages_per_seq: int, page_size: int
) -> bool:
    """Whether sparse_mla_prefill_fused_tokens takes the query-blocked kernel."""
    return (
        num_tokens >= QBLOCK_MIN_TOKENS
        and num_tokens % 128 == 0
        and pages_per_seq <= QBLOCK_TOKENS_MAX_PAGES_PER_SEQ
        and page_size % 128 == 0
        and page_size & (page_size - 1) == 0
    )


def sparse_mla_prefill_window(
    ql_nope: jax.Array,  # bf16[window, all heads, 512]
    q_pe: jax.Array,  # bf16[window, all heads, 64]
    new_kv_c: jax.Array,  # fp8[max_num_tokens, 512]
    new_k_pe: jax.Array,  # fp8[max_num_tokens, 64]
    cache_nope: jax.Array,  # u8[num_pages, page_size, 4, 128]
    cache_rope: jax.Array,  # u8[num_pages, page_size // 4, 4, 128]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1], of the whole batch
    window_cu_q_lens: jax.Array,  # i32[max_num_seqs + 1], of the window
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    topk_indices: jax.Array,  # i32[window, index_topk], the window's rows
    *,
    sm_scale: float,
    k_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Store every token of the step, then attend one window of it.

    SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT: each attention-TP rank passes
    its window's queries for every head and the indexer rows it selected for
    them. ``window_cu_q_lens`` is ``clip(cu_q_lens - window_start, 0,
    window)``: each request keeps its slot, with its tokens inside the window.
    Each (token, head) then goes through the per-query path of
    ``sparse_mla_attention_fused`` as it does there: the same gathered cache
    rows, mask and softmax.

    Returns ``(output, cache_nope, cache_rope)``: ``[window, all heads, 512]``
    and the two updated buffers.
    """
    cache_nope, cache_rope = _store_step(
        cache_nope,
        cache_rope,
        new_kv_c,
        new_k_pe,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )
    out = sparse_ragged_paged_attention(
        jnp.concatenate([ql_nope, q_pe], axis=-1),
        cache_nope,
        cache_rope,
        topk_indices,
        page_indices,
        window_cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        cache_layout=TENSORCORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=min(
            DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE, max(64, ql_nope.shape[0])
        ),
    )
    return out[..., :NOPE_DIM], cache_nope, cache_rope


def sparse_mla_prefill_fused_tokens(
    ql_nope: jax.Array,  # bf16[num_q_heads, max_num_tokens, 512], head-major
    q_pe: jax.Array,  # bf16[num_q_heads, max_num_tokens, 64], head-major
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
    """``sparse_mla_attention_fused`` for plain-DSA prefill, head-major queries.

    Rows are exact token top-k lists (``index_kpool == 1``). Batches of
    ``QBLOCK_MIN_TOKENS``+ tokens, a multiple of 128, with contexts up
    to ``QBLOCK_TOKENS_MAX_PAGES_PER_SEQ`` pages, take the query-blocked kernel
    (``mla_sparse_qblock_tokens``); the others the per-query kernel.

    Returns ``(output, cache_nope, cache_rope)`` with the output head-major,
    ``[num_q_heads, max_num_tokens, 512]``.
    """
    num_tokens = ql_nope.shape[1]
    page_size = cache_nope.shape[1]
    pages_per_seq = page_indices.shape[0] // (cu_q_lens.shape[0] - 1)
    if qblock_tokens_supported(num_tokens, pages_per_seq, page_size):
        cache_nope, cache_rope = _store_step(
            cache_nope,
            cache_rope,
            new_kv_c,
            new_k_pe,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
        out = sparse_mla_prefill_qblock_tokens(
            ql_nope,
            q_pe,
            cache_nope,
            cache_rope,
            topk_indices,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
        )
        return out, cache_nope, cache_rope
    out, cache_nope, cache_rope = sparse_mla_attention_fused(
        ql_nope.transpose(1, 0, 2),
        q_pe.transpose(1, 0, 2),
        new_kv_c,
        new_k_pe,
        cache_nope,
        cache_rope,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        topk_indices,
        sm_scale=sm_scale,
        k_scale=k_scale,
    )
    return out.transpose(1, 0, 2), cache_nope, cache_rope
