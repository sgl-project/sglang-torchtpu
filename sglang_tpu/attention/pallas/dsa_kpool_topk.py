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
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas.kpool_compress import (
    compress_and_store,
    supports_page_size,
)
from sglang_tpu.dsv4.streamindex_topk import KVLayout, streamindex_topk
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
# TPU7x DMA latency / bandwidth constants and XLA scatter lowering cutoffs
# used by _should_use_page_write to compare expected cycles at trace time
# (validated on Ghostfish TPU7x in cl/1025124431).
_DMA_HBM_TO_VMEM = 692.0
_DMA_VMEM_TO_HBM = 594.0
_DMA_HBM_TO_SMEM = 722.0
_BW_HBM = 1688.0  # B/cycle HBM <-> VMEM
_BW_SMEM = 32.0  # B/cycle HBM -> SMEM
# ~31.2 MiB scatter window (32 MiB VMEM leaving 1/40 headroom).
_VMEM_WINDOW_BYTES = 32 * 1024 * 1024 * 39 // 40
_VMEM_STAGING_BYTES = 32704 * 1024  # ~31.9 MiB before staging spills to HBM
_FLAT_MIN_TOKENS = 1024  # below this XLA keeps 3-D scatter on per-row path


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


def _write_raw_pages(raw_cache, raw, req, page, slot, valid, max_runs):
    """``raw_cache.at[page, slot].set(raw, mode="drop")`` a page at a time.

    A request fills consecutive slots of pages no other request writes, so a
    step is one run of rows per (request, page), each on its own page. The
    runs' pages are read into a small buffer, the rows scattered there (far
    cheaper on a small table than on the cache), and written back by one scatter.
    """
    num_raw = raw_cache.shape[0]
    tok = jnp.arange(raw.shape[0], dtype=jnp.int32)
    first = valid & ((tok == 0) | (req != jnp.roll(req, 1)) | (slot == 0))
    run = jnp.where(valid, jnp.cumsum(first) - 1, max_runs)
    run_page = jnp.full((max_runs,), num_raw, page.dtype)
    run_page = run_page.at[run].set(page, mode="drop")
    pages = raw_cache[jnp.minimum(run_page, num_raw - 1)]
    pages = pages.at[run, slot].set(raw, mode="drop")
    return raw_cache.at[run_page].set(pages, mode="drop")


def _vmem_window_pages(num_tokens: int, page_size: int, raw_dim: int) -> int:
    """Max pages fitting in one f32-unpacked VMEM scatter window (241 for 128x256)."""
    upd_i = min(512, num_tokens)
    avail = _VMEM_WINDOW_BYTES - 2 * upd_i * raw_dim * 4
    return max(1, avail // (page_size * raw_dim * 4))


def _xla_scatter_branch(
    num_tokens: int, num_pages: int, page_size: int, raw_dim: int
) -> str:
    """XLA TPU scatter lowering branch selected from static shapes."""
    if num_tokens < _FLAT_MIN_TOKENS:
        return "per-row"
    if num_pages <= _vmem_window_pages(num_tokens, page_size, raw_dim):
        return "in-vmem"
    log2_t = max(1, math.ceil(math.log2(max(2, num_tokens))))
    if (
        11 * num_tokens
        + page_size * num_pages
        + int(0.01 * num_tokens * log2_t * log2_t)
        < 80 * num_tokens
    ):
        return "sort+stream"
    return "per-row"


def _row_scatter_cycles(
    num_tokens: int, num_raw: int, page_size: int, raw_dim: int
) -> float:
    """Estimated TPU7x cycles for ``raw_cache.at[page, slot].set(raw)``."""
    row_bytes = float(raw_dim * 2)
    page_bytes = float(page_size * raw_dim * 2)
    # Default T(8,128)(2,1) layout copies an 8-row tile per single-row RMW.
    tile_bytes = 8.0 * row_bytes
    n_win = _vmem_window_pages(num_tokens, page_size, raw_dim)
    branch = _xla_scatter_branch(num_tokens, num_raw, page_size, raw_dim)
    if branch == "in-vmem":
        return 10.4 * num_tokens + 8.0 * num_raw
    if branch == "sort+stream":
        num_windows = math.ceil(num_raw / n_win)
        dma = (
            num_windows * (_DMA_HBM_TO_VMEM + _DMA_VMEM_TO_HBM)
            + 2.0 * (num_raw * page_bytes) / _BW_HBM
        )
        return 9000.0 + dma + 55.0 * num_raw + 10.4 * num_tokens
    # Per-row HBM read-modify-write, 16 rows per non-pipelined iteration.
    smem = math.ceil(num_tokens / 256) * (_DMA_HBM_TO_SMEM + 1024.0 / _BW_SMEM)
    iters = math.ceil(num_tokens / 16.0)
    uniq = (
        min(num_tokens, math.ceil(num_tokens / 8.0) + 1)
        if num_tokens < 256
        else num_tokens
    )
    dma = (
        iters * (_DMA_HBM_TO_VMEM + _DMA_VMEM_TO_HBM)
        + (num_tokens * (tile_bytes + row_bytes) + uniq * tile_bytes) / _BW_HBM
        + ((num_tokens + uniq) * 38.0 if uniq > 16 else uniq * 15.0)
    )
    tc = iters * (1860.0 if uniq > 16 else 650.0)
    startup = 8000.0 if num_tokens >= 256 else 0.0
    return max(6500.0, smem + dma + tc - startup)


def _page_write_cycles(
    num_tokens: int,
    num_seqs: int,
    num_raw: int,
    page_size: int,
    raw_dim: int,
) -> float:
    """Estimated worst-case TPU7x cycles for ``_write_raw_pages`` (R_act = R)."""
    max_runs = num_tokens // page_size + 2 * num_seqs
    row_bytes = float(raw_dim * 2)
    page_bytes = float(page_size * raw_dim * 2)
    tile_bytes = 8.0 * row_bytes
    n_win = _vmem_window_pages(num_tokens, page_size, raw_dim)
    branch = _xla_scatter_branch(num_tokens, max_runs, page_size, raw_dim)
    fits_vmem = max_runs * page_bytes + num_tokens * row_bytes <= _VMEM_STAGING_BYTES

    # Step 1: run-id cumsum and run_page scatter.
    s1 = (
        _DMA_HBM_TO_SMEM
        + (4.0 * num_tokens) / _BW_SMEM
        + 1.5 * num_tokens
        + 6.0 * max_runs
    )

    # Step 2: gather max_runs whole pages HBM -> VMEM (spills if > VMEM budget).
    s2_dma = (
        math.ceil(max_runs / 16.0) * _DMA_HBM_TO_VMEM
        + (max_runs * page_bytes) / _BW_HBM
        + max_runs * (38.0 if num_tokens >= _FLAT_MIN_TOKENS else 12.0)
    )
    if not fits_vmem:
        s2_dma += (
            math.ceil(max_runs / 16.0) * _DMA_VMEM_TO_HBM
            + (max_runs * page_bytes) / _BW_HBM
        )
    s2 = _DMA_HBM_TO_SMEM + (4.0 * max_runs) / _BW_SMEM + max(s2_dma, 24.0 * max_runs)

    # Step 3: row scatter into the small [max_runs, page_size, raw_dim] buffer.
    if branch == "in-vmem":
        setup = 0.0 if (num_raw <= n_win and max_runs <= 16) else 5000.0
        s3 = (
            setup
            + _DMA_HBM_TO_VMEM
            + (num_tokens * row_bytes) / _BW_HBM
            + 125.0 * max_runs
            + 9.5 * num_tokens
        )
    elif branch == "sort+stream":
        num_windows = math.ceil(max_runs / n_win)
        if fits_vmem:
            s3 = (
                15000.0
                + num_windows * 2500.0
                + _DMA_HBM_TO_VMEM
                + (num_tokens * row_bytes) / _BW_HBM
                + 155.0 * max_runs
                + 9.5 * num_tokens
            )
        else:
            s3 = (
                15000.0
                + num_windows * (_DMA_HBM_TO_VMEM + _DMA_VMEM_TO_HBM)
                + 2.0 * (max_runs * page_bytes) / _BW_HBM
                + 65.0 * max_runs
                + 9.5 * num_tokens
            )
    else:
        iters = math.ceil(num_tokens / 16.0)
        if fits_vmem:
            s3 = _DMA_HBM_TO_SMEM + iters * 1560.0
        else:
            s3 = _DMA_HBM_TO_SMEM + iters * (
                _DMA_HBM_TO_VMEM
                + _DMA_VMEM_TO_HBM
                + (32.0 * tile_bytes + 16.0 * row_bytes) / _BW_HBM
                + 32.0 * 38.0
                + 1860.0
            )

    # Step 4: whole-page scatter VMEM -> HBM (worst case: all max_runs active).
    scan = 95.0 if (num_tokens >= _FLAT_MIN_TOKENS or not fits_vmem) else 45.0
    s4 = (
        _DMA_HBM_TO_SMEM
        + (4.0 * max_runs) / _BW_SMEM
        + scan * max_runs
        + math.ceil(max_runs / 16.0) * _DMA_VMEM_TO_HBM
        + (max_runs * page_bytes) / _BW_HBM
        + 85.0 * max_runs
    )
    return s1 + s2 + s3 + s4


def _should_use_page_write(
    num_tokens: int,
    num_seqs: int,
    num_raw: int,
    page_size: int,
    raw_dim: int,
) -> bool:
    """Picks ``_write_raw_pages`` when its worst-case cycle cost beats row scatter."""
    # A single <=16-row scatter iteration (or empty input) never pays off 4-step
    # page staging.
    if num_tokens <= 16 or num_seqs <= 0 or num_raw <= 0:
        return False
    return _page_write_cycles(
        num_tokens, num_seqs, num_raw, page_size, raw_dim
    ) < _row_scatter_cycles(num_tokens, num_raw, page_size, raw_dim)


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
    num_tokens, n_heads, d = query.shape
    num_raw, page_size, _ = raw_cache.shape
    num_pages = pooled_cache.shape[0]
    _, tokens_per_row, record_bytes = pooled_cache.shape[1:]
    num_seqs = kv_lens.shape[0]
    expected = (
        num_pages,
        page_size // kpool // _TOKENS_PER_ROW,
        _TOKENS_PER_ROW,
        pooled_record_bytes(d),
    )
    if pooled_cache.shape != expected:
        raise ValueError(
            f"pooled cache has shape {pooled_cache.shape}; the index-key layout "
            f"is {expected}"
        )
    if topk % kpool != 0 or page_size % (kpool * _TOKENS_PER_ROW) != 0:
        raise ValueError(
            f"topk must be a multiple of kpool and page_size of {kpool * _TOKENS_PER_ROW}"
        )

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
    # Runs: a page per page_size tokens and two partial pages per request.
    max_runs = num_tokens // page_size + 2 * num_seqs
    if _should_use_page_write(
        num_tokens, num_seqs, num_raw, page_size, raw_cache.shape[2]
    ):
        raw_cache = _write_raw_pages(raw_cache, raw, req, page, slot, valid, max_runs)
    else:
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
            raw_page_base=raw_page_base,
            raw_ring_pages_cnt=raw_ring_pages_cnt,
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

    # 3. Score each query against its request's closed pools and keep the best
    # topk // kpool. The pooled cache is paged like the raw cache with
    # page_size // kpool pools per page, so the kernel reads the request's own
    # page table. Pad each row to whole kernel blocks so no block reads past it;
    # page 0 is never a real page and kv_lens masks the padding.
    pages_per_seq = page_indices.shape[0] // num_seqs
    pool_page_size = page_size // kpool
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
    n_pick = topk // kpool
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
        num_kv_pages_per_block=(block_pages,) * 3,
        num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
        decode_query_size=decode_query_size,
        enable_early_exit=False,
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        # GLM-5.3-Flash only; the DeepSeek-V4.1 indexer keeps the prefetched table.
        page_table_in_hbm=tpu_envs.SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM.get(),
        chunk_tokens=_CHUNK_TOKENS if chunked else None,
        overlap_chunks=False,
        **(
            dict(row_start=row_start.reshape(()), row_count=row_count)
            if row_count
            else {}
        ),
    )  # i32[T or row_count, n_pick], chosen pools first, then -1
    if row_count:
        return picked, raw_cache, pooled_cache
    out = expand_picked(picked, pos, valid, topk=topk, kpool=kpool, out_width=out_width)
    return out, raw_cache, pooled_cache


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
    picked: jax.Array,  # i32[num_tokens, topk // kpool], all ranks' slices
    kv_lens: jax.Array,  # i32[max_num_seqs]; already counts this step's tokens
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    *,
    topk: int,
    kpool: int,
    out_width: int = 0,
) -> jax.Array:
    """The token rows of ``kpool_indexer_step`` from the pools its token split
    picked, put together over the ranks (``row_count``)."""
    _, pos, valid = _token_rows(kv_lens, cu_q_lens, distribution, picked.shape[0])
    return expand_picked(
        picked, pos, valid, topk=topk, kpool=kpool, out_width=out_width
    )
