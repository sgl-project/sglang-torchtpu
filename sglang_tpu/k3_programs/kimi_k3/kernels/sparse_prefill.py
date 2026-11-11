# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1's DSA attention for the K3 prefill program's MLA layers.

The prefill's dense MLA (``prefill_megakernel.make_mla_attention_layer_sp_ring``)
attends each window token over its whole prefix and the window rows before it.
Kimi-K3.1 attends only the ``index_topk`` (2048) tokens its DSA indexer scores
highest, as sglang's native path does with ``dsa_indexer_step`` and
``sparse_mla_prefill_fused_tokens``. This module runs those kernels on the
prefill's per-rank view (inside its ``k3_sp`` shard_map):

1. A scratch latent cache and index-key cache hold, per sequence, its gathered
   prefix (the arena pages, in arena order: the prefix's key order does not
   change its attention; exactly tied index scores may break differently)
   followed by its window rows, written by upstream's cache writers.
2. The indexer is token-split: rank ``r`` scores window rows ``[r T/32, (r +
   1) T/32)`` with all 32 index heads (each rank projects its own head for
   every row; an all-to-all hands each rank its rows' heads), and the ranks'
   top-k rows are all-gathered.
3. The sparse MLA kernel attends this rank's three heads (absorbed queries)
   over each row's top-k.

The window's sequences start on page boundaries (the prefill's descriptor
contract) and their prefixes cover whole pages.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_decode
from sglang_tpu.k3_programs.kimi_k3.layout import INDEX_SUBPAGE_ROWS, INDEX_SUBPAGE_TOKENS

RANKS = 32  # the prefill's k3_sp mesh
HEADS = 3  # MLA heads a rank attends
LATENT, ROPE = 512, 64
INDEX_HEADS, INDEX_DIM = 32, 128
SM_SCALE = 192**-0.5  # K3's MLA softmax scale (NoPE 128 + rope lanes 64)


def index_records(rows):
    """Index-pool rows ``[pages, R, 4, 128]`` uint8 -> per-token ``(key fp8
    [pages * T, 128], scale byte uint8 [pages * T, 1])`` (``dsa_dcp``'s layout)."""
    pages, row_count = rows.shape[:2]
    subpages = row_count // INDEX_SUBPAGE_ROWS
    split = rows.reshape(pages, subpages, INDEX_SUBPAGE_ROWS, 4, INDEX_DIM)
    key = split[:, :, :32].reshape(pages * subpages * INDEX_SUBPAGE_TOKENS, INDEX_DIM)
    scale = split[:, :, 32, 0].reshape(pages * subpages * INDEX_SUBPAGE_TOKENS, 1)
    return jax.lax.bitcast_convert_type(key, jnp.float8_e4m3fn), scale


def gathered_index_pages(index_pool, ids, axis):
    """Every rank's prefix index rows in ``gathered_prefix_pages``'s slot layout
    (rank ``r``'s local page ``q`` at slot ``r * slots + 1 + q``; slot 0 of
    each rank is a placeholder)."""
    slots = ids.shape[0]
    local = index_pool[jnp.where(jnp.arange(slots) == 0, 0, ids)]  # [slots, R, 4, 128]
    gathered = jax.lax.all_gather(local, axis)  # [ranks, slots, R, 4, 128]
    return gathered.reshape(-1, *index_pool.shape[1:])


def _sequence_layout(segments, *, count, tokens, capacity, page_tokens):
    """``(kv_lens, page_indices, cu_q_lens)`` of the window's sequences over the
    scratch caches: arena page ``i`` is scratch page ``i``, window page ``w``
    scratch page ``capacity / page + w``."""
    offsets, lengths = segments[:count], segments[count:2 * count]
    bases, prefixes = segments[3 * count:4 * count], segments[4 * count:5 * count]
    arena_pages = capacity // page_tokens
    pages_per_seq = arena_pages + tokens // page_tokens
    # A sequence's query rows run to the next sequence's offset (padding rows
    # after its tokens come last, so no real row sees them).
    ends = jnp.concatenate([offsets[1:], jnp.asarray([tokens], jnp.int32)])
    ends = jnp.where(jnp.arange(count) < count - 1, ends, jnp.maximum(offsets + lengths, ends))
    q_lens = jnp.where(lengths > 0, ends - offsets, 0)
    cu_q_lens = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(q_lens)]).astype(jnp.int32)
    kv_lens = (prefixes + q_lens).astype(jnp.int32)
    index = jnp.arange(pages_per_seq)[None, :]
    prefix_pages = (prefixes // page_tokens)[:, None]
    page = jnp.where(index < prefix_pages, (bases // page_tokens)[:, None] + index,
                     arena_pages + (offsets // page_tokens)[:, None] + index - prefix_pages)
    return kv_lens, page.reshape(-1).astype(jnp.int32), cu_q_lens


def sparse_mla(query, latent, index_key, q_latent, hidden, weights, arena_tokens, index_rows,
               segments, *, count, capacity, page_tokens, axis, topk=2048):
    """This rank's three heads' DSA attention over a packed window.

    ``query`` ``[3, T, 256]`` bf16 (nope 128 | rope 64 | 0, unscaled),
    ``latent`` ``[T, 576]`` and ``index_key`` ``[T, 128]`` the window's latents
    and normed index keys, ``q_latent`` ``[T, 1536]``, ``hidden`` ``[T,
    7168]``; ``weights`` holds ``key_value_up_projection`` ``[512, 768]`` and the
    indexer's ``index_query_projection`` (this rank's head, ``[1536, 128]``) and
    ``index_gate_projection`` (``[7168, 128]``); ``arena_tokens`` ``[capacity,
    640]`` FP8 the gathered prefix latents in arena order and ``index_rows``
    their index rows (``[capacity / page, R, 4, 128]``); ``segments`` the
    window's ``[5 * count]`` descriptor. Returns ``[3, T, 128]`` bf16."""
    from sglang_tpu.attention.pallas import dsa_index_topk as dit
    from sglang_tpu.attention.pallas import mla_sparse_fused as msf
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils

    tokens = query.shape[1]
    ranks = RANKS
    rank = jax.lax.axis_index(axis)
    rows = tokens // ranks
    arena_pages = capacity // page_tokens
    num_pages = arena_pages + tokens // page_tokens
    distribution = jnp.asarray([0, 0, count], jnp.int32)
    kv_lens, page_indices, cu_q_lens = _sequence_layout(
        segments, count=count, tokens=tokens, capacity=capacity, page_tokens=page_tokens)

    # 1. Scratch caches with every sequence's prefix (one pseudo-sequence over
    # the arena pages); the kernels store the window rows themselves.
    whole = jnp.asarray([capacity], jnp.int32)
    identity = jnp.arange(arena_pages, dtype=jnp.int32)
    span = jnp.asarray([0, capacity], jnp.int32)
    nope_spec, rope_spec = msf._split_specs(num_pages, page_tokens)
    cache_nope = jnp.zeros(nope_spec.shape, nope_spec.jax_dtype)
    cache_rope = jnp.zeros(rope_spec.shape, rope_spec.jax_dtype)
    cache_nope, cache_rope = kv_cache_utils.update_sparse_mla_kv_cache(
        cache_nope, cache_rope, arena_tokens[:, :LATENT], arena_tokens[:, LATENT:LATENT + ROPE],
        whole, identity, span, nope_spec=nope_spec, rope_spec=rope_spec)
    cache_k = jnp.zeros(dit.index_key_cache_shape(num_pages, page_tokens, INDEX_DIM), jnp.uint8)
    prefix_key, prefix_scale = index_records(index_rows)
    cache_k = dit._store_keys(cache_k, prefix_key, prefix_scale, whole, identity, span,
                              jnp.asarray([0, 0, 1], jnp.int32))

    # 2. Token-split indexer: this rank's rows with all 32 heads.
    own = jnp.dot(q_latent.astype(jnp.bfloat16), weights["index_query_projection"],
                  preferred_element_type=jnp.float32).astype(jnp.bfloat16)  # [T, 128]: head ``rank``
    heads = jax.lax.all_to_all(own.reshape(ranks, rows, INDEX_DIM), axis, 0, 0)  # [32 heads, rows, 128]
    first = rank * rows
    gate = jnp.dot(jax.lax.dynamic_slice_in_dim(hidden, first, rows).astype(jnp.bfloat16),
                   weights["index_gate_projection"], preferred_element_type=jnp.float32)
    gate = dsa_decode._bf16(dsa_decode._bf16(gate[:, :INDEX_HEADS]) * dsa_decode.HEAD_GATE_SCALE)
    query_rows = jnp.zeros((tokens, INDEX_HEADS, INDEX_DIM), jnp.float32)
    query_rows = jax.lax.dynamic_update_slice_in_dim(
        query_rows, jnp.transpose(heads, (1, 0, 2)).astype(jnp.float32), first, 0)
    gate_rows = jax.lax.dynamic_update_slice_in_dim(
        jnp.zeros((tokens, INDEX_HEADS), jnp.float32), gate, first, 0)
    selected, _ = dit.dsa_indexer_step(
        query_rows, index_key.astype(jnp.float32), gate_rows, cache_k, kv_lens, page_indices, cu_q_lens,
        distribution, jnp.reshape(first, (1,)).astype(jnp.int32), weight_scale=dsa_decode.WEIGHT_SCALE,
        topk=topk, segments="all", key_store="xla", row_count=rows)  # [rows, topk]
    selected = jax.lax.all_gather(selected, axis, tiled=True)  # [T, topk]

    # 3. Sparse MLA over each row's top-k (absorbed queries, latent outputs).
    up = weights["key_value_up_projection"]  # [512, 3 x (128 key | 128 value)]
    w_uk = jnp.stack([up[:, h * 256:h * 256 + 128] for h in range(HEADS)])  # [3, 512, 128]
    w_uv = jnp.stack([up[:, h * 256 + 128:h * 256 + 256] for h in range(HEADS)])  # [3, 512, 128]
    ql_nope = jnp.einsum("htn,hln->htl", query[:, :, :128].astype(jnp.float32),
                         w_uk.astype(jnp.float32)).astype(jnp.bfloat16)
    q_pe = query[:, :, 128:128 + ROPE].astype(jnp.bfloat16)
    out, _, _ = msf.sparse_mla_prefill_fused_tokens(
        ql_nope, q_pe, latent[:, :LATENT].astype(jnp.float8_e4m3fn),
        latent[:, LATENT:LATENT + ROPE].astype(jnp.float8_e4m3fn), cache_nope, cache_rope, kv_lens,
        page_indices, cu_q_lens, distribution, selected, sm_scale=SM_SCALE, k_scale=1.0)
    return jnp.einsum("htl,hlv->htv", out.astype(jnp.float32), w_uv.astype(jnp.float32)).astype(jnp.bfloat16)
