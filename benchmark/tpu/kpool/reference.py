# SPDX-License-Identifier: Apache-2.0
"""Frozen step-2 oracle from dsa_kpool_topk.py at ab8861d5bac88f2e7010c8e808e61088dc8191a1.

Kept independent of the candidate and serving imports for module validation.
The pooled sum is rounded with reduce_precision, which XLA cannot drop; the
rotation is quantized from F32 without a second rounding.
"""

import jax
import jax.numpy as jnp
import numpy as np

_FP8_MAX = 448.0
# The index-key record layout (dsa_index_topk.index_key_cache_shape): FP8 key
# bytes in whole lanes, one scale byte per lane, rounded up to whole lanes.
_LANE_BYTES = 128
_TOKENS_PER_ROW = 4
# Upstream act_quant / the kpool writer floor the absolute max before scaling.
_AMAX_FLOOR = 1e-4


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
    keys [..., d] f32; the pooled sum is rounded to bf16, the rotation is not."""
    score = gates.astype(jnp.float32) + ape.astype(jnp.float32)
    weight = jax.nn.softmax(score, axis=-2)
    pooled = jnp.sum(weight * keys.astype(jnp.float32), axis=-2)
    # reduce_precision, not an astype round trip: XLA's excess-precision pass
    # can drop f32->bf16->f32 pairs.
    pooled = jax.lax.reduce_precision(pooled, exponent_bits=8, mantissa_bits=7)
    return hadamard_rotate(pooled)


def computation(
    raw_cache,
    pooled_cache,
    ape,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    num_tokens,
    kpool,
):
    """Unoptimized step 2; raw_cache already includes this forward's raw writes."""
    num_pages, page_size, doubled_d = raw_cache.shape
    d = doubled_d // 2
    num_seqs = kv_lens.shape[0]
    _, tokens_per_row, record_bytes = pooled_cache.shape[1:]
    req, pos, valid = _token_rows(kv_lens, cu_q_lens, distribution, num_tokens)
    loc = {
        "page_indices": page_indices,
        "num_seqs": num_seqs,
        "page_size": page_size,
        "num_pages": num_pages,
    }
    # 2. Compress the pools this step closes, reading the cache just written.
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

    return pooled_cache
