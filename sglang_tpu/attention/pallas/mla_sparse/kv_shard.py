# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA KV sharded by page across a group of attention ranks.

Page id ``g`` of the MLA latent cache lives only on the rank with role
``g % shards``, at physical page ``g // shards``; the DSA indexer caches stay
whole on every rank. A rank attends the candidates on its own pages for every
head of its group and returns a partial output with its log-sum-exp; the
group merges the partials in FP32. The page ids are the allocator's, so the
scheduler and the radix tree see no change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mla_page_indices(
    page_indices: jax.Array, *, shards: int, role: int | jax.Array, oob_page: int
) -> jax.Array:
    """This rank's physical MLA page for each page id; ``oob_page`` where another
    rank owns the page, so writers skip it."""
    return jnp.where(page_indices % shards == role, page_indices // shards, oob_page)


def mla_shard_pages(num_pages: int, shards: int) -> int:
    """Physical MLA pages one rank allocates for ``num_pages`` page ids."""
    return -(-num_pages // shards)


def compact_left(values: jax.Array, keep: jax.Array) -> jax.Array:
    """Stable compaction along axis 1: kept entries first, in order, then -1.

    Each kept entry moves left by the number of dropped entries before it.
    Applying the shift one bit at a time from the lowest never lands two kept
    entries on one slot (the shifts never decrease along the row), so every
    stage is a static slice and a select instead of a sort or scatter.
    """
    shift = jnp.cumsum(~keep, axis=1, dtype=jnp.int32)  # dropped entries up to here
    groups = keep.shape[1]
    bit = 1
    while bit < groups:
        moves = keep & ((shift & bit) != 0)

        def left(x, fill):
            pad = jnp.full(x.shape[:1] + (bit,) + x.shape[2:], fill, x.dtype)
            return jnp.concatenate([x[:, bit:], pad], axis=1)

        arrive = left(moves, False)  # an entry `bit` slots to the right moves here
        stay = keep & ~moves
        values = jnp.where(
            arrive[..., None],
            left(values, -1),
            jnp.where(stay[..., None], values, -1),
        )
        shift = jnp.where(arrive, left(shift, 0), shift)
        keep = arrive | stay
        bit *= 2
    return values


def merge_partials(out: jax.Array, lse: jax.Array) -> jax.Array:
    """Merge ``shards`` partial attentions: ``out`` f32[shards, ..., d] normalized
    per shard, ``lse`` f32[shards, ...] natural-log sums (-inf for an empty shard)."""
    m = jnp.max(lse, axis=0)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.exp(lse - m)  # 0 for an empty shard
    out = jnp.where(jnp.isfinite(lse)[..., None], out, 0.0)
    num = jnp.sum(w[..., None] * out, axis=0)
    return num / jnp.maximum(jnp.sum(w, axis=0), 1e-30)[..., None]
