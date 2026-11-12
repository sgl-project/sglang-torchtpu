# SPDX-License-Identifier: Apache-2.0
"""Row norms, rope and router top-k shared by the megakernel programs."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.consts import LANE


# ----------------------------------------------------------------------------- small helpers
def _rms(x, eps, w=None):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def _heads_rms(x, n_heads, d, eps, gain=1.0):
    parts = [_rms(x[:, h * d : (h + 1) * d], eps) * gain for h in range(n_heads)]
    return parts[0] if n_heads == 1 else jnp.concatenate(parts, axis=1)


def _rope_roll(x, n_heads, d, cos_x, sin_x):
    """Interleaved rotary with lane-expanded tables: the pair partner is one lane roll away."""
    w = n_heads * d
    xv = x.astype(jnp.float32)
    c = jnp.concatenate([cos_x] * n_heads, axis=1) if n_heads > 1 else cos_x
    s = jnp.concatenate([sin_x] * n_heads, axis=1) if n_heads > 1 else sin_x
    lane = jax.lax.broadcasted_iota(jnp.int32, xv.shape, 1)
    even = (lane % 2) == 0
    partner = jnp.where(
        even, pltpu.roll(xv, shift=w - 1, axis=1), pltpu.roll(xv, shift=1, axis=1)
    )
    return xv * c + jnp.where(even, -1.0, 1.0) * partner * s


def _topk_by_rank(scores, k, n_live, rank_ref):
    """Top-k by rank: ``rank[r, e]`` = number of experts beating ``e`` (higher score, ties to

    the lower id), so pick ``j`` is the expert of rank ``j`` -- the same picks as
    ``k`` argmax
    rounds, from one ``[E, E]`` comparison per live row instead of a dependent
    chain. Rows past
    ``n_live`` keep stale ranks.
    """
    R, E = scores.shape
    s_t = scores.T  # [E, R]: e' on sublanes
    lower = jax.lax.broadcasted_iota(jnp.int32, (E, E), 0) < jax.lax.broadcasted_iota(
        jnp.int32, (E, E), 1
    )
    for r in range(R):

        @pl.when(r < n_live)
        def _(r=r):
            col, row = s_t[:, r : r + 1], scores[r : r + 1, :]
            beats = (col > row) | ((col == row) & lower)
            rank_ref[r : r + 1, :] = jnp.sum(
                beats.astype(jnp.int32), axis=0, keepdims=True
            )

    rank = rank_ref[...]
    idx = jax.lax.broadcasted_iota(jnp.int32, (R, E), 1)
    picks = [rank == j for j in range(k)]
    return [jnp.sum(jnp.where(p, idx, 0), axis=1, keepdims=True) for p in picks], picks


def _topk_iter(scores, k):
    """Top-k as k rounds of a row max and its lowest index: the picks of _topk_by_rank (higher

    score first, ties to the lower id), a cost flat in the rows where the ranks
    pay one [E, E]
    comparison per row (faster from 16 rows, slower at 8).
    """
    R, E = scores.shape
    idx = jax.lax.broadcasted_iota(jnp.int32, (R, E), 1)
    ids, picks = [], []
    for _ in range(k):
        m = jnp.max(scores, axis=1, keepdims=True)
        sel = jnp.min(jnp.where(scores == m, idx, E), axis=1, keepdims=True)
        pick = idx == sel
        ids.append(sel)
        picks.append(pick)
        scores = jnp.where(pick, -jnp.inf, scores)
    return ids, picks


def _topk_xpose(sc, bias, k, router_eps, route_scale):
    """Transposed batched top-k for up to LANE rows (zero cross-lane XLU reductions).

    ``sc`` is ``[br, E]`` f32 (``1 <= br <= LANE``, ``E % LANE == 0``), ``bias`` is
    ``[1, E]`` f32. Transposes to ``[E, LANE]`` once so all ``k`` max / tie-break /
    gather rounds reduce vertically along sublanes, then transposes ``[2*k, LANE]``
    back in a single 128x128 XLU op.
    """
    br, E = sc.shape
    assert 1 <= br <= LANE and E % LANE == 0
    sc_pad = (
        sc
        if br == LANE
        else jnp.concatenate([sc, jnp.zeros((LANE - br, E), jnp.float32)], axis=0)
    )
    s_t = jnp.transpose(sc_pad + bias)  # [E, LANE]
    idx_t = jax.lax.broadcasted_iota(jnp.int32, (E, LANE), 0)
    ids_rows = []
    for _ in range(k):
        m = jnp.max(s_t, axis=0, keepdims=True)
        sel = jnp.min(jnp.where(s_t == m, idx_t, E), axis=0, keepdims=True)
        ids_rows.append(sel)
        s_t = jnp.where(idx_t == sel, -jnp.inf, s_t)
    sc_t = jnp.transpose(sc_pad)  # [E, LANE]
    wsel_rows = [
        jnp.sum(jnp.where(idx_t == sel, sc_t, 0.0), axis=0, keepdims=True)
        for sel in ids_rows
    ]
    wsel_t = jnp.concatenate(wsel_rows, axis=0)  # [k, LANE]
    wsel_t = (
        wsel_t / (jnp.sum(wsel_t, axis=0, keepdims=True) + router_eps) * route_scale
    )
    ids_full = jnp.transpose(
        jnp.concatenate(
            [
                jnp.concatenate(ids_rows, axis=0),
                jnp.zeros((LANE - k, LANE), jnp.int32),
            ],
            axis=0,
        )
    )  # [LANE, LANE] int32
    wsel_full = jnp.transpose(
        jnp.concatenate([wsel_t, jnp.zeros((LANE - k, LANE), jnp.float32)], axis=0)
    )  # [LANE, LANE] f32
    return ids_full[:br], wsel_full[:br]
