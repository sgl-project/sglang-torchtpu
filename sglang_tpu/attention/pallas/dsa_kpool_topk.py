# SPDX-License-Identifier: Apache-2.0
"""The pooled ("kpool") DSA indexer of GLM-5.3-Flash on the device, as one JAX function.

GLM-5.3-Flash scores pools of ``kpool`` consecutive tokens instead of single
tokens (upstream ``IndexerKPool``, CUDA only). Per indexer layer and forward:

1. Store each new token's raw index key (k_norm(wk(x)), bf16) and compression
   gate score (compress_gate(x), bf16) in the layer's raw cache.
2. Every token that closes a pool (``pos % kpool == kpool - 1``) compresses it:
   per channel, a softmax over the pool's slots of ``gate + ape`` weights the
   raw keys; the sum is rounded to bf16, Hadamard-rotated in f32 and quantized
   to FP8 with a power-of-two scale. Pool ``p`` is stored at position
   ``p`` of the request in the pooled cache (the GLM-5.2 index-key layout).
3. Each query scores the closed pools before it,
   ``sum over heads of w_h * relu(q_h . k_pool) * scale``, keeps the top
   ``topk // kpool`` pools (all of them when there are no more), expands each
   to its ``kpool`` tokens and appends the open pool's tokens (the tail).

Pools closed by rejected speculative tokens are rewritten when the pool closes
again, and a query only sees pools closed at or before its own position.

Step 3 runs on the StreamIndex top-k kernel (DeepSeek-V4's compressed indexer)
with a closed-pools-only causal mask.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas.kpool_compress import (
    compress_and_store,
    supports_page_size,
)
from sglang_tpu.dsv4.streamindex_topk import KVLayout, streamindex_topk

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0
# The index-key record layout (dsa_index_topk.index_key_cache_shape): FP8 key
# bytes in whole lanes, one scale byte per lane, rounded up to whole lanes.
_LANE_BYTES = 128
_TOKENS_PER_ROW = 4
# Upstream act_quant / the kpool writer floor the absolute max before scaling.
_AMAX_FLOOR = 1e-4
# StreamIndex block sizes for its (decode, prefill, mixed) passes, as in
# dsa_index_topk; not tuned for GLM.
_NUM_KV_PAGES_PER_BLOCK = (2, 2, 2)
_NUM_QUERIES_PER_BLOCK = (1, 128, 128)
# Query rows per scoring chunk: the kernel's score buffer is rows x positions.
_CHUNK_TOKENS = 2048
# Row-chunk only when one pass's f32 score buffer would exceed this; an
# unchunked 8k prefill at 10k context is ~84 MiB.
_MAX_UNCHUNKED_SCORE_BYTES = 512 << 20


def _hadamard_matrix(n: int) -> np.ndarray:
    """Sylvester-ordered +-1 Hadamard matrix, the fast_hadamard_transform order."""
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def hadamard_rotate(x: jax.Array) -> jax.Array:
    """x @ H_d * d**-0.5 in f32, d = x.shape[-1] (a power of two)."""
    d = x.shape[-1]
    h = jnp.asarray(_hadamard_matrix(d))
    return jnp.einsum(
        "...i,ij->...j",
        x.astype(jnp.float32),
        h,
        precision=jax.lax.Precision.HIGHEST,
    ) * (d**-0.5)


def quantize_ue8m0(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-row FP8 e4m3 with a power-of-two scale over the last axis.

    Returns (fp8 values, f32 scale [..., 1], u8 scale exponent byte [..., 1]),
    matching upstream's ue8m0 rule: amax floored at 1e-4,
    scale = 2 ** ceil(log2(amax / 448)).
    """
    xf = x.astype(jnp.float32)
    amax = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True), _AMAX_FLOOR)
    exponent = jnp.ceil(jnp.log2(amax / _FP8_MAX))
    scale = jnp.exp2(exponent)
    q = jnp.clip(xf / scale, -_FP8_MAX, _FP8_MAX).astype(jnp.float8_e4m3fn)
    scale_byte = (exponent + 127).astype(jnp.uint8)
    return q, scale, scale_byte


def _token_rows(kv_lens, cu_q_lens, distribution, num_tokens):
    """Per token: request index, position in its sequence, and whether it is real."""
    num_reqs = distribution[2]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    req = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32)
    req = jnp.minimum(req, kv_lens.shape[0] - 1)
    q_len = cu_q_lens[req + 1] - cu_q_lens[req]
    pos = kv_lens[req] - q_len + (tok - cu_q_lens[req])
    valid = tok < cu_q_lens[num_reqs]
    return req, pos, valid


def _page_slot(req, pos, page_indices, num_seqs, page_size, num_pages, valid):
    """Physical (page, slot) of sequence position ``pos`` of request ``req``;
    invalid rows get an out-of-range page so scatters drop them."""
    pages_per_seq = page_indices.shape[0] // num_seqs
    page = page_indices[req * pages_per_seq + pos // page_size]
    page = jnp.where(valid, page, num_pages)
    return page, pos % page_size


def _align(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def pooled_record_bytes(head_dim: int) -> int:
    key_bytes = _align(head_dim, _LANE_BYTES)
    return _align(key_bytes + key_bytes // _LANE_BYTES, _LANE_BYTES)


def _pool_records(compressed: jax.Array, record_bytes: int) -> jax.Array:
    """GLM-5.2 index-key records: FP8 key bytes in whole lanes, scale byte, zeros."""
    k_fp8, _, k_scale_byte = quantize_ue8m0(compressed)
    key_bytes = jax.lax.bitcast_convert_type(k_fp8, jnp.uint8)
    head_dim = key_bytes.shape[-1]
    key_lanes = _align(head_dim, _LANE_BYTES)
    key_bytes = jnp.pad(key_bytes, ((0, 0), (0, key_lanes - head_dim)))
    records = jnp.concatenate([key_bytes, k_scale_byte], axis=-1)
    return jnp.pad(records, ((0, 0), (0, record_bytes - records.shape[-1])))


def compress_pools(keys: jax.Array, gates: jax.Array, ape: jax.Array) -> jax.Array:
    """keys / gates [..., kpool, d] (bf16), ape [kpool, d] f32 -> rotated pooled
    keys [..., d] f32. The pooled sum is rounded to bf16; the rotation is not,
    unlike upstream's writer (see kpool/compute.py pack_pools)."""
    score = gates.astype(jnp.float32) + ape.astype(jnp.float32)
    weight = jax.nn.softmax(score, axis=-2)
    pooled = _round_bf16(jnp.sum(weight * keys.astype(jnp.float32), axis=-2))
    return hadamard_rotate(pooled)


def _round_bf16(x: jax.Array) -> jax.Array:
    # XLA's excess-precision pass may drop an f32->bf16->f32 convert pair;
    # reduce_precision is always honored.
    return jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)


def kpool_indexer_step(
    query: jax.Array,  # bf16[num_tokens, n_heads, d]; wq_b(q_lora), not rotated
    key: jax.Array,  # bf16[num_tokens, d]; k_norm(wk(x)), not rotated
    gate: jax.Array,  # bf16[num_tokens, d]; compress_gate(x)
    head_weights: jax.Array,  # f32[num_tokens, n_heads]; weights_proj(x)
    ape: jax.Array,  # f32[kpool, d]
    raw_cache: jax.Array,  # bf16[num_pages, page_size, 2 * d]: key | gate
    pooled_cache: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    *,
    weight_scale: float,
    topk: int,
    kpool: int,
    out_width: int = 0,
    decode_query_size: int = 1,
    max_query_length: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (topk_indices i32[num_tokens, max(out_width, topk + kpool - 1)],
    raw_cache, pooled_cache). Rows list sequence-relative token positions: the
    selected pools' tokens, then the tail, then -1."""
    num_tokens, n_heads, d = query.shape
    num_pages, page_size, _ = raw_cache.shape
    _, tokens_per_row, record_bytes = pooled_cache.shape[1:]
    num_seqs = kv_lens.shape[0]
    expected = (
        num_pages,
        page_size // _TOKENS_PER_ROW,
        _TOKENS_PER_ROW,
        pooled_record_bytes(d),
    )
    if pooled_cache.shape != expected:
        raise ValueError(
            f"pooled cache has shape {pooled_cache.shape}; the index-key layout "
            f"is {expected}"
        )
    if topk % kpool != 0 or page_size % kpool != 0:
        raise ValueError("topk and page_size must be multiples of kpool")

    req, pos, valid = _token_rows(kv_lens, cu_q_lens, distribution, num_tokens)
    loc = dict(
        page_indices=page_indices,
        num_seqs=num_seqs,
        page_size=page_size,
        num_pages=num_pages,
    )

    # 1. Raw key | gate records of this step's tokens.
    page, slot = _page_slot(req, pos, valid=valid, **loc)
    raw = jnp.concatenate([key, gate], axis=-1).astype(raw_cache.dtype)
    raw_cache = raw_cache.at[page, slot].set(raw, mode="drop")

    # 2. Compress the pools this step closes, reading the cache just written.
    fused = (
        kpool == 4
        and d == 128
        and supports_page_size(page_size)
        and raw_cache.dtype == jnp.bfloat16
    )
    if fused:
        # Keep the native cache layout through the update. A generic scatter
        # can introduce full-pool layout conversions even for a few new pools.
        pooled_cache = compress_and_store(
            raw_cache,
            pooled_cache,
            ape,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_tokens,
            max_query_length=max_query_length,
        )
    else:
        # Logged at trace time, once per compiled shape.
        logger.warning(
            "kpool indexer: kpool=%d, head_dim=%d, page_size=%d, raw dtype=%s has "
            "no fused pooling kernel; using the generic scatter, which can copy "
            "the whole pooled cache per step.",
            kpool,
            d,
            page_size,
            raw_cache.dtype,
        )
        closing = valid & (pos % kpool == kpool - 1)
        first = pos - (kpool - 1)
        member = first[:, None] + jnp.arange(kpool, dtype=jnp.int32)[None, :]
        m_page, m_slot = _page_slot(
            jnp.broadcast_to(req[:, None], member.shape),
            jnp.maximum(member, 0),
            valid=jnp.broadcast_to(closing[:, None], member.shape),
            **loc,
        )
        rows = raw_cache[jnp.minimum(m_page, num_pages - 1), m_slot]  # [T, kpool, 2d]
        pooled = compress_pools(rows[..., :d], rows[..., d:], ape)
        records = _pool_records(pooled, record_bytes)
        pool_id = pos // kpool
        p_page, p_slot = _page_slot(req, pool_id, valid=closing, **loc)
        pooled_cache = pooled_cache.at[
            p_page, p_slot // tokens_per_row, p_slot % tokens_per_row
        ].set(records, mode="drop")

    # 3. Score each query against its request's closed pools and keep the best
    # topk // kpool. Pool p sits at position p of its request, so the kernel
    # needs only the first 1/kpool of each request's page table.
    # Round to whole kernel blocks so no block reads past a request's row.
    pages_per_seq = page_indices.shape[0] // num_seqs
    block_pages = max(_NUM_KV_PAGES_PER_BLOCK)
    pool_pages = min(_align(-(-pages_per_seq // kpool), block_pages), pages_per_seq)
    pool_page_indices = page_indices.reshape(num_seqs, pages_per_seq)[
        :, :pool_pages
    ].reshape(-1)
    # Upstream rotates the bf16 query (rotate_activation) before quantizing.
    query = hadamard_rotate(query).astype(jnp.bfloat16)
    q_fp8, q_scale, _ = quantize_ue8m0(query)
    w = head_weights.astype(jnp.float32) * q_scale[..., 0] * weight_scale  # [T, H]
    n_pick = topk // kpool
    chunked = (
        num_tokens > _CHUNK_TOKENS
        and num_tokens % _CHUNK_TOKENS == 0
        and num_tokens * pool_pages * page_size * 4 > _MAX_UNCHUNKED_SCORE_BYTES
    )
    # causal_offset = kpool - 1: a query at pos sees pools < (pos + 1) // kpool,
    # the ones closed at or before it.
    picked = streamindex_topk(
        q_fp8,
        w,
        pooled_cache,
        kv_lens,
        pool_page_indices,
        cu_q_lens,
        distribution,
        k=n_pick,
        compression_ratio=kpool,
        causal_offset=kpool - 1,
        num_kv_pages_per_block=_NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
        decode_query_size=decode_query_size,
        enable_early_exit=False,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        chunk_tokens=_CHUNK_TOKENS if chunked else None,
        overlap_chunks=False,
    )  # i32[T, n_pick], chosen pools first, then -1

    pool_len = (pos + 1) // kpool
    picked_ok = picked >= 0
    tokens = picked[:, :, None] * kpool + jnp.arange(kpool, dtype=jnp.int32)
    tokens = jnp.where(picked_ok[:, :, None], tokens, -1).reshape(num_tokens, -1)

    # Tail: the open pool's tokens, right after the selected history.
    hist_len = jnp.minimum(pool_len, n_pick) * kpool
    tail_count = (pos + 1) % kpool
    width = max(out_width, topk + kpool - 1)
    col = jnp.arange(width, dtype=jnp.int32)[None, :]
    tail_off = col - hist_len[:, None]
    tail = jnp.where(
        (tail_off >= 0) & (tail_off < tail_count[:, None]),
        pool_len[:, None] * kpool + tail_off,
        -1,
    )
    hist = jnp.pad(tokens, ((0, 0), (0, width - topk)), constant_values=-1)
    out = jnp.where(col < hist_len[:, None], hist, tail)
    out = jnp.where(valid[:, None], out, -1).astype(jnp.int32)
    return out, raw_cache, pooled_cache
