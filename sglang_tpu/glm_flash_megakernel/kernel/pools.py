"""Serving's KV pools as the megakernel reads them: byte layouts and page-table DMAs.

Nope pages are `u8 [pages, page, 4, 128]`, one token per 512 fp8 bytes; the pool is
read as `u32 [pages, page, 1, 128]` so a token is one 128-lane word row, and unpacked
in VMEM as the sparse MLA kernel does. Pooled index-key records are
`u8 [pages, page/kpool/4, 4, 256]`: 128 fp8 bytes of the Hadamard-rotated pooled key,
then its ue8m0 scale byte (`dsa_kpool_topk._pool_records`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
BF16 = jnp.bfloat16
U8 = jnp.uint8
U32 = jnp.uint32
F8 = jnp.float8_e4m3fn

# A token's nope bytes in the pool, and the u32 words they fold into.
NOPE_BYTES = 512
WORD_BYTES = 4


def dispatch(idx, n, fn):
    """`fn(j)` for the static or traced layer ordinal `idx`; per-layer pools are separate arrays."""
    if isinstance(idx, int):
        fn(idx)
        return
    for j in range(n):
        pl.when(idx == j)(lambda j=j: fn(j))


def wait_dma(dst, sem):
    """Wait for every transfer issued on `sem` into `dst` (byte-counted, so one wait covers a batch)."""
    pltpu.make_async_copy(dst, dst, sem).wait()


def page_of(tab_ref, b, pg, pages_per_seq):
    """Page id of request `b`'s page `pg`; past the table it is page 0, which is never real."""
    return tab_ref[b * pages_per_seq + jnp.minimum(pg, pages_per_seq - 1)]


def nope_page(pool, page, page_size):
    """HBM ref of one page's tokens as `u32 [page_size, 128]` word rows."""
    return pool.bitcast(U32).at[page].reshape(page_size, NOPE_BYTES // WORD_BYTES)


def record_page(pool, page, records_per_page, record_bytes):
    """HBM ref of one page's pooled records as `u32 [records/4, record_bytes]`."""
    return (
        pool.bitcast(U32).at[page].reshape(records_per_page // WORD_BYTES, record_bytes)
    )


def unpack_nope(words, k_scale_dtype=BF16):
    """`u32 [T, 128]` word rows -> `[T, 512]` in fp8 units (times the layer's k_scale)."""
    t = words.shape[0]
    bytes_ = pltpu.bitcast(words, U8).reshape(t, NOPE_BYTES)
    return pltpu.bitcast(bytes_, F8).astype(k_scale_dtype)


def unpack_records(words, head_dim):
    """`u32 [P/4, record_bytes]` -> dequantized rotated pooled keys `f32 [P, head_dim]`."""
    bytes_ = pltpu.bitcast(words, U8)
    keys = pltpu.bitcast(bytes_[:, :head_dim], F8).astype(F32)
    e = bytes_[:, head_dim : head_dim + 1].astype(F32)
    return keys * jnp.exp2(e - 127.0)


def hadamard(d):
    """The +-1 Walsh-Hadamard matrix `[d, d]` (bf16, exact); the caller applies `d**-0.5`."""
    i = jax.lax.broadcasted_iota(jnp.int32, (d, d), 0)
    j = jax.lax.broadcasted_iota(jnp.int32, (d, d), 1)
    x = i & j
    for s in (1, 2, 4, 8):
        x = x ^ jax.lax.shift_right_logical(x, s)
    return jnp.where((x & 1) == 1, -1.0, 1.0).astype(BF16)


def rotate(x, h):
    """`hadamard_rotate` of bf16-rounded rows: exact +-1 products, f32 sums, then d**-0.5."""
    d = x.shape[-1]
    return jnp.dot(x.astype(BF16), h, preferred_element_type=F32) * (d**-0.5)


def record_bytes(head_dim):
    """Bytes per pooled record: key in whole 128-byte lanes plus one scale byte per lane, lane-rounded."""
    lane = 128
    key = -(-head_dim // lane) * lane
    return -(-(key + key // lane) // lane) * lane


def closing_pools(pos, Q, kp):
    """(first pool, count) of the pools whose last token is one of pos .. pos+Q-1."""
    first = (pos + kp) // kp - 1
    first = jnp.where((first + 1) * kp - 1 >= pos, first, first + 1)
    last = (pos + Q) // kp - 1
    return first, jnp.maximum(last - first + 1, 0)


def pool_records(keys, record_bytes):
    """Serving's records of rotated pooled keys `[P, d] f32`, as `_pool_records` writes them."""
    from sglang_tpu.attention.pallas.dsa_kpool_topk import _pool_records

    return _pool_records(keys, record_bytes)
