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
    shift = inclusive_count(~keep)  # dropped entries up to here
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


# Widths up to this take the matmul prefix count (a [G, G] bf16 triangle).
_MATMUL_COUNT_MAX_WIDTH = 2048


def inclusive_count(flags: jax.Array) -> jax.Array:
    """``jnp.cumsum(flags, axis=1)`` for bool ``[N, G]`` as i32. On TPU XLA lowers the
    cumsum to a reduce-window; a 0/1 bf16 matmul with an upper triangle accumulates
    in f32 and is exact for counts below 2**24."""
    assert flags.ndim == 2, flags.shape
    width = flags.shape[1]
    if width > _MATMUL_COUNT_MAX_WIDTH:
        return jnp.cumsum(flags, axis=1, dtype=jnp.int32)
    row = jax.lax.broadcasted_iota(jnp.int32, (width, width), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (width, width), 1)
    upper = (row <= col).astype(jnp.bfloat16)
    count = jnp.dot(
        flags.astype(jnp.bfloat16), upper, preferred_element_type=jnp.float32
    )
    return count.astype(jnp.int32)


def owned_rows(
    rows: jax.Array,  # i32[tokens, width]; sequence positions, -1 padded
    *,
    page_size: int,
    shards: int,
    role: int | jax.Array,
    group: int,
    out_width: int,
) -> jax.Array:
    """This rank's entries of each top-k row, in order, then -1:
    ``i32[tokens, out_width]``, ``out_width`` >= the row width rounded up to
    ``group``. ``group`` consecutive entries share a pool, so an owner."""
    num_tokens, width = rows.shape
    sub = page_size // shards
    padded = -(-width // group) * group
    assert out_width >= padded, (out_width, padded)
    rows = jnp.pad(rows, ((0, 0), (0, padded - width)), constant_values=-1)
    rows = rows.reshape(num_tokens, padded // group, group)
    first = rows[..., 0]
    keep = (first >= 0) & ((first % page_size) // sub == role)
    rows = jnp.where(keep[..., None], rows, -1)
    rows = compact_left(rows, keep).reshape(num_tokens, padded)
    return jnp.pad(rows, ((0, 0), (0, out_width - padded)), constant_values=-1)


def merge_two(
    out_a: jax.Array, lse_a: jax.Array, out_b: jax.Array, lse_b: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Merge two partial attentions over disjoint candidates of the same rows:
    ``out`` [..., d] normalized per part, ``lse`` [..., 1] f32 natural-log sums
    (a part with no candidate has a very negative lse). Returns (out, lse), out
    in ``out_a``'s dtype."""
    m = jnp.maximum(lse_a, lse_b)
    w_a, w_b = jnp.exp(lse_a - m), jnp.exp(lse_b - m)
    total = w_a + w_b
    out = jnp.where(w_a > 0, w_a * out_a.astype(jnp.float32), 0.0) + jnp.where(
        w_b > 0, w_b * out_b.astype(jnp.float32), 0.0
    )
    return (out / total).astype(out_a.dtype), m + jnp.log(total)


def merge_partials(out: jax.Array, lse: jax.Array) -> jax.Array:
    """Merge ``shards`` partial attentions: ``out`` f32[shards, ..., d] normalized
    per shard, ``lse`` f32[shards, ...] natural-log sums (-inf for an empty shard)."""
    m = jnp.max(lse, axis=0)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.exp(lse - m)  # 0 for an empty shard
    out = jnp.where(jnp.isfinite(lse)[..., None], out, 0.0)
    num = jnp.sum(w[..., None] * out, axis=0)
    return num / jnp.maximum(jnp.sum(w, axis=0), 1e-30)[..., None]
