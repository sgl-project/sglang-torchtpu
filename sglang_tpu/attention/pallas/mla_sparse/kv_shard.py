# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA latent cache split inside every page across a group of attention ranks.

With ``shards`` ranks and ``page_size`` tokens per page, rank ``role`` holds token
offsets ``[role * sub, (role + 1) * sub)`` of every page, ``sub = page_size //
shards``, at row ``offset - role * sub`` of that page; the DSA indexer caches
stay whole on every rank. Every buffer is indexed by the allocator's page id, so
the scheduler, the radix tree and page copies (host tier) see no change. A rank
attends the candidates at its own offsets for every head of its group and
returns a partial output with its log-sum-exp; the group merges the partials in
FP32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def own_page_slots(
    page: jax.Array,
    slot: jax.Array,
    *,
    sub_page: int,
    role: int | jax.Array,
    oob_page: int,
) -> tuple[jax.Array, jax.Array]:
    """A token's (page, offset in the page) -> (page, row in this rank's share);
    ``oob_page`` where another rank owns the offset, so writers skip it."""
    return jnp.where(slot // sub_page == role, page, oob_page), slot % sub_page


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
