# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1's DSA indexer on the K3 programs' decode context parallelism.

Index keys live in one uint8 pool per MLA layer beside the unified pools
(``kimi_k3/layout.index_block_shape``): pool row ``r`` holds the index keys of
the tokens whose latents are in block ``r`` (``layout.tokens_per_block`` of
them, the rank's DCP chunk), in 128-token sub-pages of 33 rows x 4 x 128
bytes: rows 0..31 hold four tokens' 128 fp8 key bytes each (token ``t`` of the
sub-page: row ``t // 4``, sublane ``t % 4``), row 32 sublane 0 lane ``t``
holds its UE8M0 scale byte (``exponent + 127``). The selection itself is in
``kimi_k3/dsa.py`` (reference) and the decode kernel.

This module holds the layout helpers and the XLA-level commit of the new
tokens' index keys (after a decode step, as ``paged.commit_latent_rows`` for
the latents).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sglang_tpu.k3_programs.kimi_k3.layout import INDEX_SUBPAGE_ROWS, INDEX_SUBPAGE_TOKENS

HEAD_DIM = 128
RECORD = HEAD_DIM + 1  # key bytes, then the scale byte


def record_place(token):
    """``(row, sublane)`` of token ``token``'s key bytes in a pool row, and the
    ``(row, lane)`` of its scale byte (sublane 0); ``token`` is the offset in
    the row (traced or not)."""
    subpage, t = token // INDEX_SUBPAGE_TOKENS, token % INDEX_SUBPAGE_TOKENS
    base = subpage * INDEX_SUBPAGE_ROWS
    return (base + t // 4, t % 4), (base + INDEX_SUBPAGE_ROWS - 1, t)


def key_records(keys: jax.Array) -> jax.Array:
    """Normed index keys f32/bf16 ``[..., 128]`` -> uint8 records ``[..., 129]``
    (fp8 E4M3 bytes at the UE8M0 scale, then ``exponent + 127``), as
    ``kimi_k3/dsa.quantize_fp8``."""
    x = keys.astype(jnp.float32)
    bits = jax.lax.bitcast_convert_type(jnp.max(jnp.abs(x), axis=-1, keepdims=True), jnp.int32)
    biased = bits >> 23
    exp = biased - (127 + 8) + ((bits & 0x7FFFFF) > 0x600000).astype(jnp.int32)
    exp = jnp.where(biased == 0, jnp.where(bits == 0, 0, -126), exp)
    exp = jnp.clip(exp, -126, 127)
    inv = jax.lax.bitcast_convert_type((127 - exp) << 23, jnp.float32)
    values = jnp.clip(x * inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    key_bytes = jax.lax.bitcast_convert_type(values, jnp.uint8)
    return jnp.concatenate([key_bytes, (exp + 127).astype(jnp.uint8)], axis=-1)


def commit_index_rows(index_pools, records, block_tables, positions, *, group=32,
                      owner_table=None, page_tokens=None, global_pages=False):
    """Write each row's new index-key records into the DSA index pools on the
    rank that owns its position (XLA level, inside ``shard_map`` over ``tp``):
    MLA layer ``m``'s record ``records[b, m]`` (uint8 ``[129]``) of the token
    at ``positions[b]`` lands in ``index_pools[m]`` in the block the row's
    latent goes to (``paged.commit_latent_rows``'s rule: global page ``p //
    page_tokens`` is owned by member ``p % group``, local tables hold one entry
    per ``group`` pages). Rows this rank does not own are dropped."""
    from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp

    member = mla_dcp.owner_index(owner_table) % group
    pools = list(index_pools)
    if page_tokens is None:
        subpages = pools[0].shape[1] // INDEX_SUBPAGE_ROWS
        page_tokens = subpages * INDEX_SUBPAGE_TOKENS
    rows = records.shape[0]
    pos = positions[:rows]
    pages = pos // page_tokens
    owned = pages % group == member
    blocks = block_tables[jnp.arange(rows), pages if global_pages else pages // group]
    (key_row, sublane), (scale_row, lane) = record_place(pos % page_tokens)
    for m, pool in enumerate(pools):
        target = jnp.where(owned, blocks, pool.shape[0])
        pool = pool.at[target, key_row, sublane, :].set(records[:, m, :HEAD_DIM], mode="drop")
        pools[m] = pool.at[target, scale_row, 0, lane].set(records[:, m, HEAD_DIM], mode="drop")
    return tuple(pools)


def read_records(pool_row: jax.Array, tokens: int) -> tuple[jax.Array, jax.Array]:
    """One index pool row -> (fp8-valued key f32 ``[tokens, 128]``, exponent
    i32 ``[tokens, 1]``): the inverse of the layout (host checks and tests)."""
    subpages = tokens // INDEX_SUBPAGE_TOKENS
    rows = pool_row.reshape(subpages, INDEX_SUBPAGE_ROWS, 4, HEAD_DIM)
    keys = rows[:, :32].reshape(subpages * INDEX_SUBPAGE_TOKENS, HEAD_DIM)
    exps = rows[:, 32, 0].reshape(subpages * INDEX_SUBPAGE_TOKENS, 1).astype(jnp.int32) - 127
    values = jax.lax.bitcast_convert_type(keys, jnp.float8_e4m3fn).astype(jnp.float32)
    return values, exps
