# SPDX-License-Identifier: Apache-2.0
"""The pooled ("kpool") DSA indexer of GLM-5.3-Flash on the device, as one JAX function.

GLM-5.3-Flash scores pools of ``kpool`` consecutive tokens instead of single
tokens (upstream ``IndexerKPool``, CUDA only). Per indexer layer and forward:

1. Store each new token's raw index key (k_norm(wk(x)), bf16) and compression
   gate score (compress_gate(x), bf16) in the layer's raw cache.
2. Every token that closes a pool (``pos % kpool == kpool - 1``) compresses it:
   per channel, a softmax over the pool's slots of ``gate + ape`` weights the
   raw keys; the sum is rounded to bf16, Hadamard-rotated in f32 and quantized
   to FP8 with a power-of-two scale. Pool ``p`` is stored in the page of its
   own tokens, record ``p % (page_size // kpool)``, so a request never writes
   a page it shares with another request through the radix tree.
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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas.kpool_compress import (
    compress_and_store,
    supports_page_size,
)
from sglang_tpu.attention.pallas.mla_sparse.kv_shard import inclusive_count
from sglang_tpu.dsv4.sparsecore_topk import sparsecore_topk, sparsecore_topk_dcp_merge
from sglang_tpu.dsv4.streamindex_topk import KVLayout, streamindex_topk
from sglang_tpu.dsv4.streamindex_topk.metadata import cp_local_to_global
from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0
# The index-key record layout (dsa_index_topk.index_key_cache_shape): FP8 key
# bytes in whole lanes, one scale byte per lane, rounded up to whole lanes.
_LANE_BYTES = 128
_TOKENS_PER_ROW = 4
# Upstream act_quant / the kpool writer floor the absolute max before scaling.
_AMAX_FLOOR = 1e-4
# Pools per StreamIndex KV block for its (decode, prefill, mixed) passes. The kernel's
# per-step schedule is a scalar-prefetch (SMEM) operand of seqs x query blocks x KV
# blocks entries: at a 1M context with 8-page blocks it is 513 KB for 16 sequences and
# the mixed pass fails to compile; 32-page blocks keep it under 128 KB.
_NUM_POOLS_PER_BLOCK = 1024
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


def _token_rows(kv_lens, cu_q_lens, distribution, num_tokens, row_start=0):
    """Per token ``row_start + i``: request index, position in its sequence,
    and whether it is real."""
    num_reqs = distribution[2]
    tok = row_start + jnp.arange(num_tokens, dtype=jnp.int32)
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


def _raw_page_slot(
    req: jax.Array,
    pos: jax.Array,
    *,
    raw_page_base: jax.Array | None,
    raw_ring_pages_cnt: int,
    valid: jax.Array,
    loc: dict[str, Any],
    num_raw: int,
) -> tuple[jax.Array, jax.Array]:
    """Physical (page, slot) of a raw key|gate row. With a raw ring, request
    ``req`` keeps its rows at ``raw_page_base[req] + page % raw_ring_pages_cnt``;
    otherwise they share the request's page ids."""
    if not raw_ring_pages_cnt:
        return _page_slot(req, pos, valid=valid, **loc)
    page = raw_page_base[req] + (pos // loc["page_size"]) % raw_ring_pages_cnt
    page = jnp.where(valid, page, num_raw)
    return page, pos % loc["page_size"]


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


def store_pooled_keys(
    key: jax.Array,  # bf16[num_tokens, d]; k_norm(wk(x)), not rotated
    gate: jax.Array,  # bf16[num_tokens, d]; compress_gate(x)
    ape: jax.Array,  # f32[kpool, d]
    raw_cache: jax.Array,  # bf16[num_pages, page_size, 2 * d]: key | gate
    pooled_cache: jax.Array,  # u8[num_pages, page_size // 4, 4, record_bytes]
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    *,
    kpool: int,
    max_query_length: int | None = None,
    raw_page_base: jax.Array | None = None,  # i32[max_num_seqs]; raw ring only
    raw_ring_pages_cnt: int = 0,
    shards: int = 1,
    role: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Store this step's raw key | gate records and compress the pools it
    closes; return ``(raw_cache, pooled_cache)``.

    The store half of ``kpool_indexer_step``: any other writer of these caches
    must call it to leave the indexer step's bytes. ``shards`` > 1: the pooled
    cache is rank ``role``'s share of each page (``kpool_indexer_local``).
    """
    num_tokens, d = key.shape
    num_raw, page_size, _ = raw_cache.shape
    num_pages = pooled_cache.shape[0]
    _, tokens_per_row, record_bytes = pooled_cache.shape[1:]
    num_seqs = kv_lens.shape[0]
    expected = (
        num_pages,
        page_size // kpool // _TOKENS_PER_ROW // shards,
        _TOKENS_PER_ROW,
        pooled_record_bytes(d),
    )
    if pooled_cache.shape != expected:
        raise ValueError(
            f"pooled cache has shape {pooled_cache.shape}; the index-key layout "
            f"is {expected}"
        )
    if page_size % (kpool * _TOKENS_PER_ROW) != 0:
        raise ValueError(f"page_size must be a multiple of {kpool * _TOKENS_PER_ROW}")

    req, pos, valid = _token_rows(kv_lens, cu_q_lens, distribution, num_tokens)
    loc = dict(
        page_indices=page_indices,
        num_seqs=num_seqs,
        page_size=page_size,
        num_pages=num_pages,
    )

    raw_loc = dict(
        raw_page_base=raw_page_base,
        raw_ring_pages_cnt=raw_ring_pages_cnt,
        loc=loc,
        num_raw=num_raw,
    )
    # 1. Raw key | gate records of this step's tokens.
    page, slot = _raw_page_slot(req, pos, valid=valid, **raw_loc)
    raw = jnp.concatenate([key, gate], axis=-1).astype(raw_cache.dtype)
    raw_cache = raw_cache.at[page, slot].set(raw, mode="drop")

    # 2. Compress the pools this step closes, reading the cache just written.
    fused = (
        kpool == 4
        and d == 128
        and supports_page_size(page_size)
        and raw_cache.dtype == jnp.bfloat16
    )
    if shards > 1 and not fused:
        raise NotImplementedError("a split pooled cache needs the fused pooling kernel")
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
            raw_page_base=raw_page_base,
            raw_ring_pages_cnt=raw_ring_pages_cnt,
            shards=shards,
            role=role,
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
        m_page, m_slot = _raw_page_slot(
            jnp.broadcast_to(req[:, None], member.shape),
            jnp.maximum(member, 0),
            valid=jnp.broadcast_to(closing[:, None], member.shape),
            **raw_loc,
        )
        rows = raw_cache[jnp.minimum(m_page, num_raw - 1), m_slot]  # [T, kpool, 2d]
        pooled = compress_pools(rows[..., :d], rows[..., d:], ape)
        records = _pool_records(pooled, record_bytes)
        p_page, p_slot = _page_slot(req, jnp.maximum(first, 0), valid=closing, **loc)
        record = p_slot // kpool
        pooled_cache = pooled_cache.at[
            p_page, record // tokens_per_row, record % tokens_per_row
        ].set(records, mode="drop")

    return raw_cache, pooled_cache


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
    row_start: jax.Array | None = None,  # i32[1]; first scored row (token split)
    row_count: int = 0,
    raw_page_base: jax.Array | None = None,  # i32[max_num_seqs]; raw ring only
    raw_ring_pages_cnt: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (topk_indices i32[num_tokens, max(out_width, topk + kpool - 1)],
    raw_cache, pooled_cache). Rows list sequence-relative token positions: the
    selected pools' tokens, then the tail, then -1.

    ``row_count`` > 0 (with ``row_start``) is the DSA indexer token split
    (attention_pallas_mla_sparse.py): every token's raw key is stored and
    every closing pool compressed, as without it, but only query rows
    ``[row_start, row_start + row_count)`` are scored, and the first output is
    those rows' picked pools, ``i32[row_count, topk // kpool]`` (chosen pools
    first, then -1; rows past the batch are all -1), not token rows. The caller
    puts the ranks' slices together and expands them with
    ``kpool_expand_rows``; each row then equals the same row of an unsplit call.
    """
    num_tokens = query.shape[0]
    if topk % kpool != 0:
        raise ValueError(f"topk ({topk}) must be a multiple of kpool ({kpool})")
    raw_cache, pooled_cache = store_pooled_keys(
        key,
        gate,
        ape,
        raw_cache,
        pooled_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        kpool=kpool,
        max_query_length=max_query_length,
        raw_page_base=raw_page_base,
        raw_ring_pages_cnt=raw_ring_pages_cnt,
    )
    _, pos, valid = _token_rows(kv_lens, cu_q_lens, distribution, num_tokens)
    picked = _score_pools(
        query,
        head_weights,
        pooled_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        weight_scale=weight_scale,
        topk=topk,
        kpool=kpool,
        decode_query_size=decode_query_size,
        row_start=row_start,
        row_count=row_count,
    )  # i32[T or row_count, n_pick], chosen pools first, then -1
    if row_count:
        return picked, raw_cache, pooled_cache
    out = expand_picked(picked, pos, valid, topk=topk, kpool=kpool, out_width=out_width)
    return out, raw_cache, pooled_cache


def _score_pools(
    query: jax.Array,  # bf16[num_tokens, n_heads, d]; not rotated
    head_weights: jax.Array,  # f32[num_tokens, n_heads]
    pooled_cache: jax.Array,  # u8[num_pages, rows, 4, record_bytes]
    kv_lens: jax.Array,
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    weight_scale: float,
    topk: int,
    kpool: int,
    decode_query_size: int,
    row_start: jax.Array | None = None,
    row_count: int = 0,
    **topk_kwargs,
):
    """Step 3 of the module docstring on StreamIndex: each query's best
    ``topk // kpool`` pools of ``pooled_cache`` (``topk_kwargs`` go to
    streamindex_topk)."""
    if row_count:
        topk_kwargs.update(row_start=row_start.reshape(()), row_count=row_count)
    num_tokens = query.shape[0]
    num_seqs = kv_lens.shape[0]
    # The pooled cache is paged like the raw cache (pooled_cache.shape[1] rows of
    # _TOKENS_PER_ROW pools per page), so the kernel reads the request's own page
    # table. Pad each row to whole kernel blocks so no block reads past it;
    # page 0 is never a real page and kv_lens masks the padding.
    pages_per_seq = page_indices.shape[0] // num_seqs
    pool_page_size = pooled_cache.shape[1] * _TOKENS_PER_ROW
    block_pages = max(1, _NUM_POOLS_PER_BLOCK // pool_page_size)
    pool_pages = _align(pages_per_seq, block_pages)
    pool_page_indices = jnp.pad(
        page_indices.reshape(num_seqs, pages_per_seq),
        ((0, 0), (0, pool_pages - pages_per_seq)),
    ).reshape(-1)
    # Upstream rotates the bf16 query (rotate_activation) before quantizing.
    query = hadamard_rotate(query).astype(jnp.bfloat16)
    q_fp8, q_scale, _ = quantize_ue8m0(query)
    w = head_weights.astype(jnp.float32) * q_scale[..., 0] * weight_scale  # [T, H]
    # A token split window scores few rows; the kernel takes no row chunks
    # with a window.
    chunked = (
        not row_count
        and num_tokens > _CHUNK_TOKENS
        and num_tokens % _CHUNK_TOKENS == 0
        and num_tokens * pool_pages * pool_page_size * 4 > _MAX_UNCHUNKED_SCORE_BYTES
    )
    # causal_offset = kpool - 1: a query at pos sees pools < (pos + 1) // kpool,
    # the ones closed at or before it.
    return streamindex_topk(
        q_fp8,
        w,
        pooled_cache,
        kv_lens,
        pool_page_indices,
        cu_q_lens,
        distribution,
        k=topk // kpool,
        compression_ratio=kpool,
        causal_offset=kpool - 1,
        num_kv_pages_per_block=(block_pages,) * 3,
        num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
        decode_query_size=decode_query_size,
        enable_early_exit=False,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        # GLM-5.3-Flash only; the DeepSeek-V4.1 indexer keeps the prefetched table.
        page_table_in_hbm=tpu_envs.SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM.get(),
        chunk_tokens=_CHUNK_TOKENS if chunked else None,
        overlap_chunks=False,
        **topk_kwargs,
    )


def kpool_indexer_local(
    query: jax.Array,  # bf16[num_tokens, n_heads, d]; wq_b(q_lora), not rotated
    key: jax.Array,  # bf16[num_tokens, d]
    gate: jax.Array,  # bf16[num_tokens, d]
    head_weights: jax.Array,  # f32[num_tokens, n_heads]
    ape: jax.Array,  # f32[kpool, d]
    raw_cache: jax.Array,  # bf16[raw_pages, page_size, 2 * d]
    pooled_cache: jax.Array,  # u8[num_pages, page_size // 16 // shards, 4, record_bytes]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    weight_scale: float,
    topk: int,
    kpool: int,
    shards: int,
    role: int,
    decode_query_size: int = 1,
    max_query_length: int | None = None,
    raw_page_base: jax.Array | None = None,
    raw_ring_pages_cnt: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``kpool_indexer_step`` on a pooled cache split across ``shards`` ranks.

    Rank ``role`` holds the pools at its own offsets of each page (the latent
    cache's split, attention/pallas/mla_sparse/kv_shard.py): it stores those of
    this step's closing pools, then scores every query against its own closed
    pools. Returns ``(picked i32[T, topk // kpool], scores i32[T, topk // kpool],
    raw_cache, pooled_cache)``: its local top pools (rank-local indices, then -1)
    and their f32 score bits, for ``kpool_merge_rows`` after an all-gather.
    """
    raw_cache, pooled_cache = store_pooled_keys(
        key,
        gate,
        ape,
        raw_cache,
        pooled_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        kpool=kpool,
        max_query_length=max_query_length,
        raw_page_base=raw_page_base,
        raw_ring_pages_cnt=raw_ring_pages_cnt,
        shards=shards,
        role=role,
    )
    own_pools = pooled_cache.shape[1] * _TOKENS_PER_ROW  # per page
    picked, scores = _score_pools(
        query,
        head_weights,
        pooled_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        weight_scale=weight_scale,
        topk=topk,
        kpool=kpool,
        decode_query_size=decode_query_size,
        cp_size=shards,
        cp_rank=role,
        interleave_size=own_pools * kpool,
        return_scores=True,
    )
    return picked, scores, raw_cache, pooled_cache


def kpool_merge_rows(
    all_scores: jax.Array,  # i32[shards * T, topk // kpool]; ranks' scores, rank-major
    picked: jax.Array,  # i32[T, topk // kpool]; this rank's local pools
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    shards: int,
    role: int,
    own_pools: int,  # this rank's pools per page
    topk: int,
    kpool: int,
    out_width: int = 0,
) -> jax.Array:
    """This rank's token rows from ``kpool_indexer_local``: the exact global top
    ``topk // kpool`` pools over every rank's local ones (ties go to the lower
    rank), kept where this rank holds them, expanded like ``expand_picked``.
    The latent cache is split the same way, so these are the candidates this
    rank attends."""
    num_tokens, n_pick = picked.shape
    scores = (
        all_scores.reshape(shards, num_tokens, n_pick)
        .transpose(1, 0, 2)
        .reshape(num_tokens, shards * n_pick)
    )
    owned = sparsecore_topk_dcp_merge(
        scores, picked, n_pick, dcp_rank=jnp.int32(role), dcp_size=shards
    )  # i32[T, n_pick]; rank-local pools, then -1
    pools = jnp.where(
        owned >= 0, cp_local_to_global(owned, role, shards, own_pools), -1
    )
    _, pos, valid = _token_rows(kv_lens, cu_q_lens, distribution, num_tokens)
    return expand_picked(pools, pos, valid, topk=topk, kpool=kpool, out_width=out_width)


# Threshold key of a row with no candidate: above every score's key.
_NO_CUT_KEY = np.iinfo(np.int32).max


def _score_key(bits: jax.Array) -> jax.Array:
    """f32 score bits -> i32 keys in the SparseCore top-k's order (its
    ``monotone_key``): -0.0 sorts below +0.0, which f32 compares equal."""
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def kpool_slice_cuts(
    slice_scores: jax.Array,  # i32[shards, rows, n_pick]; rank i's local top scores
) -> jax.Array:
    """The exact global top ``n_pick`` of each row of this rank's slice over
    every rank's local candidates (``kpool_indexer_local`` scores of the same
    rows), as a cut per destination rank: ``i32[shards, rows, 2]`` of (threshold
    key, last tied slot). Rank ``s`` keeps its candidates above the threshold
    and those equal to it at local slots <= its last tied slot.

    Ties at the threshold go to the lower rank, then the lower local slot: the
    SparseCore top-k's column order on rank-major rows, as
    ``sparsecore_topk_dcp_merge`` breaks them, so each rank keeps exactly the
    pools ``kpool_merge_rows`` keeps. The unsplit top-k may break such a tie the
    other way."""
    shards, rows, n_pick = slice_scores.shape
    scores = slice_scores.transpose(1, 0, 2).reshape(rows, shards * n_pick)
    won, won_bits = sparsecore_topk(scores, n_pick, return_scores=True)
    won_key = jnp.where(won >= 0, _score_key(won_bits), _NO_CUT_KEY)
    cut = jnp.min(won_key, axis=1)  # [rows]; _NO_CUT_KEY without a winner
    n_tied = jnp.sum((won >= 0) & (won_key == cut[:, None]), axis=1)
    tied = _score_key(slice_scores) == cut[None, :, None]  # [shards, rows, n_pick]
    per_rank = jnp.sum(tied, axis=2, dtype=jnp.int32)
    before = jnp.cumsum(per_rank, axis=0) - per_rank  # lower ranks' tied entries
    quota = n_tied[None, :] - before  # tied entries rank s keeps, if positive
    rank_in = inclusive_count(tied.reshape(shards * rows, n_pick))
    kept = tied & (rank_in.reshape(shards, rows, n_pick) <= quota[..., None])
    slot = jnp.arange(n_pick, dtype=jnp.int32)
    last = jnp.max(jnp.where(kept, slot, -1), axis=2)  # [shards, rows]
    return jnp.stack([jnp.broadcast_to(cut, last.shape), last], axis=-1)


def kpool_cut_pools(
    picked: jax.Array,  # i32[T, n_pick]; this rank's local pools, then -1
    scores: jax.Array,  # i32[T, n_pick]; their f32 score bits
    cuts: jax.Array,  # i32[>= T, 2]; each row's cut for this rank (kpool_slice_cuts)
    *,
    shards: int,
    role: int,
    own_pools: int,  # this rank's pools per page
) -> jax.Array:
    """This rank's global top pools of each row, ``i32[T, n_pick]`` global pool
    ids in local order, -1 elsewhere: the set ``kpool_merge_rows`` keeps."""
    num_tokens, n_pick = picked.shape
    cut, last = cuts[:num_tokens, 0:1], cuts[:num_tokens, 1:2]
    key = _score_key(scores)
    slot = jnp.arange(n_pick, dtype=jnp.int32)[None, :]
    keep = (picked >= 0) & ((key > cut) | ((key == cut) & (slot <= last)))
    return jnp.where(keep, cp_local_to_global(picked, role, shards, own_pools), -1)


def kpool_cut_rows(
    picked: jax.Array,
    scores: jax.Array,
    cuts: jax.Array,
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    shards: int,
    role: int,
    own_pools: int,
    topk: int,
    kpool: int,
    out_width: int = 0,
) -> jax.Array:
    """``kpool_merge_rows`` from the row slices' cuts: this rank's token rows,
    expanded like ``expand_picked``."""
    pools = kpool_cut_pools(
        picked, scores, cuts, shards=shards, role=role, own_pools=own_pools
    )
    _, pos, valid = _token_rows(kv_lens, cu_q_lens, distribution, picked.shape[0])
    return expand_picked(pools, pos, valid, topk=topk, kpool=kpool, out_width=out_width)


def expand_picked(
    picked: jax.Array,  # i32[num_tokens, topk // kpool]; chosen pools, then -1
    pos: jax.Array,  # i32[num_tokens]; each token's position in its sequence
    valid: jax.Array,  # bool[num_tokens]; real tokens
    *,
    topk: int,
    kpool: int,
    out_width: int = 0,
) -> jax.Array:
    """Token rows from picked pools: each pool's ``kpool`` tokens, then the
    open pool's tokens (the tail), then -1. ``i32[num_tokens, width]``."""
    num_tokens, n_pick = picked.shape
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
    return jnp.where(valid[:, None], out, -1).astype(jnp.int32)


def kpool_expand_rows(
    picked: jax.Array,  # i32[rows, topk // kpool]; batch rows from row_start
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    row_start: jax.Array | int = 0,  # i32[1] or 0; batch row of picked[0]
    *,
    topk: int,
    kpool: int,
    out_width: int = 0,
) -> jax.Array:
    """The token rows of ``kpool_indexer_step`` from the pools its token split
    picked: every rank's slices put together (``row_start`` 0), or one rank's
    window (its ``row_start``); rows past the batch are all -1."""
    _, pos, valid = _token_rows(
        kv_lens,
        cu_q_lens,
        distribution,
        picked.shape[0],
        row_start=jnp.asarray(row_start, jnp.int32).reshape(()),
    )
    return expand_picked(
        picked, pos, valid, topk=topk, kpool=kpool, out_width=out_width
    )
