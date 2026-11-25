# SPDX-License-Identifier: Apache-2.0
"""Per-rank dispatch dedup for the push layer (K32Z recipes at 512-1024 rows
per rank).

A token that selects several experts on one destination rank sends its row
there once: to its OWNER slot, the lowest local expert id among its slots on
that rank. Every other slot
on that rank is a DUPLICATE: the row is not sent, and the destination copies
it into the duplicate's tile row from the owner's slab row, which landed at
an earlier visit. Only data movement changes; every tile row holds the same
token row and activation scale as without the dedup, so the output is the
same bit for bit.

The pieces, all without a sort beyond the existing counting sort, a scalar
loop over pairs, or an HBM to HBM fan-out:

* Sender (`sender_tables`, XLA on [t_local, topk]): owner flags, the sort key
  2 * id + dup (each (destination, expert) run is owned rows first, token
  order, then the duplicates, token order), each pair's position `idx` in its
  run, and the count words n | n_own << 16. The exchange payload keeps its
  width: a pair word is id | idx << 16.
* Table kernel (plan_kernel._tables with dedup on): which runs are deduplicated
  (TensorCore visit positions, no overflow), the rows each run lands
  (`landed`: n_own there, n elsewhere), the dispatch counts and receive
  counts from them, and the duplicate base of each run in the SRC table.
* Receiver (`receiver_tables`, XLA on the exchanged pairs): the slab row of
  every pair (run base + idx, which replaces the window-rank machinery) and
  SRC, the owner's slab row for each duplicate row, by one scatter of the
  scale-slab scatter's shape.
* Kernel: a TensorCore expert with duplicates loads a tile as the landed
  stretches between duplicate rows plus one row copy per duplicate from SRC;
  every copy signals the tile's semaphore, so its byte-count wait is exact.

`x_tables` alone (no dedup) carries `idx` = the token-order rank, the same
tables as the default path from the new lookups: it tests the table path.
"""

import jax.numpy as jnp
from jax import lax

from sglang_tpu.moe.pallas.fused_moe_v2.host import (
    _lookup_per_pair,
    _rank_within_block,
    off_shard_slab_row,
    pair_block_hist,
)

# Rows per rank the dedup applies at (both ends included); elsewhere the layer
# builds its default program. It is scoped to the 512 and 1024 prefill
# buckets: decode pays the tables without the bytes to save, and the 2048
# bucket all-gathers.
X_MIN_TOKENS = 256
X_MAX_TOKENS = 1024
# A pair word is id | idx << X_IDX_SHIFT; a count word n | n_own << shift.
X_IDX_SHIFT = 16
X_LOW_MASK = (1 << X_IDX_SHIFT) - 1


def x_applies(t_local):
    return X_MIN_TOKENS <= t_local <= X_MAX_TOKENS


def default_src_cap(ep, t_local):
    """SRC words (duplicate rows a destination gathers), one per arriving
    token row. Uniform top-6 of 384 over 8 ranks gives ~0.17 of that at
    512 rows per rank; a destination over the cap takes full runs instead
    (plan_kernel._tables), so this bounds SMEM, not correctness."""
    return ep * t_local


def owner_flags(ids, me, *, g_local, e_total):
    """[t, K] bool, True where slot k's row is sent to its destination.

    Slot k with destination d is the token's OWNER on d when its expert id is
    the least over the token's slots on d. Own-rank slots are always owned;
    slots that route nowhere count as owned (they sort past every expert and
    are never sent)."""
    valid = ids < e_total
    dest = jnp.where(valid, ids // g_local, -1)
    same = (dest[:, :, None] == dest[:, None, :]) & valid[:, None, :]
    least = jnp.min(jnp.where(same, ids[:, None, :], jnp.int32(2**30)), axis=2)
    owned = (ids == least) | (dest == me)
    return owned | jnp.logical_not(valid)


def sender_tables(
    ids, me, *, g_local, e_total, block, block_hist=None, dedup=True
):
    """This rank's half of the tables, from its own topk ids [t, K] (mesh
    order, e_total = routes nowhere).

    Returns (idx [t, K], count words [e_total], sort key [t * K], sort
    counts): idx is each pair's position in its (destination, expert) run,
    the count words go into the exchange payload, and the key and its
    counts drive the local counting sort (host.local_sort_tokens).

    dedup=False (x_tables alone): key = id, idx = the token-order rank, the
    count words the plain counts; `block_hist` (the layer's pair_block_hist
    over the same blocks) is reused. dedup=True: key = 2 * id + dup over
    2 * e_total bins, idx = owned rows first then duplicates, and the count
    words n | n_own << 16."""
    t, K = ids.shape
    n_blocks = t * K // block
    if not dedup:
        blocks = ids.reshape(n_blocks, block)
        hist = pair_block_hist(blocks, e_total) if block_hist is None else block_hist
        off = jnp.cumsum(hist, axis=0) - hist
        bins = jnp.arange(e_total, dtype=jnp.int32)
        idx = _lookup_per_pair(off, blocks, bins) + _rank_within_block(blocks, block)
        counts = hist.sum(axis=0).astype(jnp.int32)
        return idx.reshape(t, K).astype(jnp.int32), counts, ids.reshape(-1), counts
    owned = owner_flags(ids, me, g_local=g_local, e_total=e_total)
    valid = ids < e_total
    dup = jnp.logical_and(valid, jnp.logical_not(owned)).astype(jnp.int32)
    key = jnp.where(valid, 2 * ids + dup, jnp.int32(2 * e_total))
    kb = key.reshape(n_blocks, block)
    bins = jnp.arange(2 * e_total, dtype=jnp.int32)
    hist2 = pair_block_hist(kb, 2 * e_total)  # [n_blocks, 2 E]
    cnt2 = hist2.sum(axis=0).astype(jnp.int32)
    own = cnt2[0::2]
    n = own + cnt2[1::2]
    # A duplicate's run position starts past the run's owned rows.
    lift = jnp.where((bins & 1) == 1, jnp.repeat(own, 2), 0)
    off2 = jnp.cumsum(hist2, axis=0) - hist2 + lift[None, :]
    idx = _lookup_per_pair(off2, kb, bins) + _rank_within_block(kb, block)
    words = n | jnp.left_shift(own, X_IDX_SHIFT)
    return idx.reshape(t, K).astype(jnp.int32), words, key.reshape(-1), cnt2


def pack_pairs(ids, idx):
    """The payload's pair words: id | idx << 16 (the receiver masks)."""
    return (ids | jnp.left_shift(idx, X_IDX_SHIFT)).astype(jnp.int32)


def unpack_counts(words):
    """(n, n_own) from count words (both < 2**15)."""
    return words & X_LOW_MASK, jnp.right_shift(words, X_IDX_SHIFT)


def pair_positions(ids, idx, recv_base_me, mirror_base):
    """This rank's combine tables: arrival row and mirror position of each of
    its pairs, the run's base plus idx (the default path's block offsets and
    within-block rank, as one lookup and an add). Pairs routed nowhere read
    idx alone, as the default path's do."""
    n = recv_base_me.shape[0]
    hit = ids.reshape(-1)[:, None] == jnp.arange(n, dtype=jnp.int32)[None, :]

    def take(table):
        out = jnp.sum(jnp.where(hit, table.astype(jnp.int32)[None, :], 0), axis=1)
        return out.astype(jnp.int32).reshape(ids.shape)

    return take(recv_base_me) + idx, take(mirror_base) + idx


def _group_fold(v, k, op):
    """`op` over each run of k consecutive entries of v's last axis, broadcast
    back onto every entry of its run: the reduction of v.reshape(..., n // k,
    k) over its last axis, without that [..., k]-minor layout (k = topk lanes
    padded to 128, a relayout copy each way). Each entry folds in its run
    neighbours at distance 1..k-1 from the shifted copies of v, and keeps
    its value where the neighbour lies in another run (no identity needed).
    Exact for min and for i32 add (any order, wrap-around)."""
    n = v.shape[-1]
    pos = lax.broadcasted_iota(jnp.int32, v.shape, v.ndim - 1) % k
    out = v
    for d in range(1, k):
        # The d entries rotated in at either end are never taken: they sit
        # where pos says the neighbour is out of the run.
        ahead = jnp.concatenate([v[..., d:], v[..., :d]], axis=-1)  # v[i + d]
        behind = jnp.concatenate([v[..., n - d :], v[..., : n - d]], axis=-1)
        out = jnp.where(pos < k - d, op(out, ahead), out)
        out = jnp.where(pos >= d, op(out, behind), out)
    return out


def receiver_tables(
    pairs, me, run_base, *, g_local, t_local, topk, shard_stride, ep, x_g=None,
    cap=0,
):  # fmt: skip
    """The destination's tables from the exchanged pair words [ep, t*K].

    slab_row [ep * t * K]: run_base[s, e] + idx for pairs on this rank, the
    sharded plan's off-shard sentinel (rebased as local_slab_rows rebases
    it) elsewhere -- the default path's values where idx is the token-order
    rank. With `x_g` [ep, g] (the table kernel's landed | dup base << 16 per
    (source, local expert) run): also SRC [cap], the owner's slab row for
    every duplicate row, at dup base + idx - landed. The owner is the least
    id among the token's slots here, as the sender chose it.

    The owner's least id and slab row per token are folded by _group_fold on
    the flat [ep, t * K] tables instead of [ep, t, K] reshapes, which would
    relayout topk-minor (four relayout copies ahead of the SparseCore
    scatters and pre-push at 1024 rows per rank)."""
    words = pairs.astype(jnp.int32).reshape(ep, t_local * topk)
    ids = words & X_LOW_MASK
    idx = jnp.right_shift(words, X_IDX_SHIFT)
    el = ids - me * g_local
    on_me = (el >= 0) & (el < g_local)
    hit = el[:, :, None] == jnp.arange(g_local, dtype=jnp.int32)[None, None, :]

    def take(table):
        """table [ep, g] at each pair's (source, local expert); 0 off me."""
        return jnp.sum(jnp.where(hit, table.astype(jnp.int32)[:, None, :], 0), axis=2)

    new_row = take(run_base) + idx
    off_shard = off_shard_slab_row(shard_stride, ep) - me * shard_stride
    slab_row = jnp.where(on_me, new_row, off_shard).reshape(-1)
    if x_g is None:
        return slab_row, None
    xw = take(x_g)
    landed = xw & X_LOW_MASK
    dbase = jnp.right_shift(xw, X_IDX_SHIFT)
    least = _group_fold(jnp.where(on_me, ids, jnp.int32(2**30)), topk, jnp.minimum)
    is_owner = on_me & (ids == least)
    src_val = _group_fold(jnp.where(is_owner, new_row, 0), topk, jnp.add)
    # Owned rows lead each run, so a pair past the rows its run landed is a
    # duplicate: its row was not sent (the run is TensorCore-pushed and its
    # destination under the SRC cap; otherwise landed = n).
    dup = on_me & (idx >= landed)
    src_pos = jnp.where(dup, dbase + idx - landed, jnp.int32(cap))
    src = (
        jnp.zeros((cap,), jnp.int32)
        .at[src_pos.reshape(-1)]
        .add(src_val.reshape(-1), mode="drop")
    )
    return slab_row, src
