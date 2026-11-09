# SPDX-License-Identifier: Apache-2.0
"""Dense tile plan, fold groups, expert units and compact row buckets."""

import functools

import jax.numpy as jnp

from sglang_tpu.muse_megakernel.mk.consts import LANE


@functools.lru_cache(maxsize=None)
def _choose_tile(K, N, tk_max, tn_max):
    tns = [d for d in range(LANE, min(N, tn_max) + 1, LANE) if N % d == 0] or [N]
    tks = [d for d in range(16, min(K, tk_max) + 1, 16) if K % d == 0] or [K]
    return max(tks), max(tns)


# ----------------------------------------------------------------------------- weight stream
_PACK = {jnp.dtype(jnp.bfloat16): 2, jnp.dtype(jnp.float8_e4m3fn): 4}


def dense_plan(c):
    """(name, K, N, dtype) of every dense per-layer weight, in consumption order."""
    shapes = (
        ("wqkvg", c.hidden, c.qkvg_dim),
        ("wo", c.q_dim, c.hidden),
        ("wpool", c.hidden, c.expert_hidden_local),
        ("wrouter", c.hidden, c.n_experts_padded),
        ("wunpool", c.expert_hidden, c.hidden_local),
    )
    # qkvg / o / pool / unpool stream as fp8 e4m3 with per-output-channel scales
    fp8 = ("wqkvg", "wo", "wpool", "wunpool")
    return tuple(
        (n, K, N, jnp.float8_e4m3fn if n in fp8 else jnp.bfloat16) for n, K, N in shapes
    )


def ring_slot_shape(c):
    """u32 words per ring slot: one [tile_k, tile_n] bf16 tile."""
    return (c.tile_k // 2, c.tile_n)


def fold_groups(c):
    """Weights whose matvecs consume an all-gather's column blocks as they arrive."""
    return (("wqkvg",),) * c.fold_f + (("wpool", "wrouter"),) * c.fold_o


def _fold_g(c, grp):
    """Column blocks per fold group: the widest K tile of the group, in blocks."""
    rk, rn = ring_slot_shape(c)
    return (
        max(
            _choose_tile(K, N, rk * _PACK[jnp.dtype(dt)], rn)[0]
            for name, K, N, dt in dense_plan(c)
            if name in grp
        )
        // c.hidden_local
    )


def owner_window_rows(c):
    """Returns the rows of the 8-row-aligned window around an owner block.

    Owner blocks of rows_per_dev rows start at 8-row boundaries only when
    rows_per_dev is a multiple of 8; otherwise the block is read from the aligned
    window around it.

    Args:
        c: the MKConfig.
    """
    rc = c.rows_per_dev
    return rc if rc % 8 == 0 else -(-(rc + 7) // 8) * 8


def fold_fits(c):
    """Returns whether every fold group's tiles fit in the weight stream.

    A fold group waits for all its tiles before its first dot. A 3-slot ring
    (stream depth 2) is too short for the folds.

    Args:
        c: the MKConfig, with its ring and stream depth chosen.
    """
    hl = c.hidden_local
    for grp in fold_groups(c):
        tiles = [t for t in dense_tiles(c) if t[0] in grp]
        g = _fold_g(c, grp)
        if sum(1 for t in tiles if t[1] < g // (t[3] // hl)) > c.stream_depth:
            return False
    return True


def dense_tiles(c):
    """(name, ki, ni, tk, tn, dtype) of every ring tile of one layer, in consumption order.

    A fold group's tiles run K group by K group (all its weights per group); ki is
    then logical:
    group j holds the physical K blocks a fold consumes j-th
    (_WeightStream.phys_k).
    """
    rk, rn = ring_slot_shape(c)
    geo = {}
    for name, K, N, dt in dense_plan(c):
        geo[name] = (K, N, dt) + _choose_tile(K, N, rk * _PACK[jnp.dtype(dt)], rn)
    folded = {n: grp for grp in fold_groups(c) for n in grp}
    tiles, done = [], set()
    for name, K, N, dt in dense_plan(c):
        if name in done:
            continue
        if name not in folded:
            _, _, _, tk, tn = geo[name]
            tiles += [
                (name, ki, ni, tk, tn, dt)
                for ni in range(N // tn)
                for ki in range(K // tk)
            ]
            continue
        grp = folded[name]
        G = _fold_g(c, grp)
        for j in range(K // (G * c.hidden_local)):
            for w in grp:
                _, Nw, dtw, tk, tn = geo[w]
                s_per = G * c.hidden_local // tk
                tiles += [
                    (w, j * s_per + s, ni, tk, tn, dtw)
                    for s in range(s_per)
                    for ni in range(Nw // tn)
                ]
        done |= set(grp)
    return tuple(tiles)


def fit_ring_slots(c, cap):
    """Largest ring size <= cap that divides the dense tiles per layer (the slot schedule repeats per layer)."""
    T = len(dense_tiles(c))
    return max(d for d in range(2, cap + 1) if T % d == 0)


def expert_units(c):
    """Expert operand units in order: ("layer", l) with 4 refs, or ("group", lo, hi) with 4

    stacked refs.
    """
    units, l = [], 0
    starts = {lo: hi for lo, hi in c.expert_groups}
    while l < c.n_layers:
        if l in starts:
            units.append(("group", l, starts[l]))
            l = starts[l]
        else:
            units.append(("layer", l))
            l += 1
    return units


def n_expert_refs(c):
    return 4 * len(expert_units(c))


def route_blocks(c):
    """Lane blocks per vec_route field: (row, slot) pairs and distinct experts both fit."""
    return -(-c.rows * c.top_k // LANE)


def route_rows(c):
    """vec_route table rows: uniq, count, offset, perm fields, then n_u, then blk_end (and blk_mid)."""
    return -(-(4 * route_blocks(c) + 2 + bool(c.split_rows)) // 8) * 8


def compact_buckets(c):
    """Row buckets of the compact expert variants, ascending: an expert with n live pairs runs

    the smallest bucket >= n (busier experts take the full path).
    """
    arms = c.compact_arms
    if (
        not arms
        and c.ex_opt
        and c.compact_rows == 8
        and c.rows <= 64
        and len(expert_units(c)) <= 8
    ):
        arms = (1, 2, 4, 6, 8)
    if arms:
        # buckets past the first that covers every row are unreachable (one pair per row)
        cover = [b for b in arms if b >= c.rows][:1]
        return tuple(b for b in arms if b < c.rows) + tuple(cover)
    return tuple(b for b in (1, 2, 4, 8) if b <= c.compact_rows)


def few_rows(c):
    """Rows a step can have, if few enough for the gather-only collectives (else 1, unused)."""
    n = c.n_req * c.rows_per_req
    return n if n <= c.few_rows_max else 1
