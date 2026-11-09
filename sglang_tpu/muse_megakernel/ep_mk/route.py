# SPDX-License-Identifier: Apache-2.0
"""This rank's slots from the replicated top-k: per local expert, its live (row, slot) pairs in
row order (vector ops, one DMA to SMEM), then work items of at most item_rows slots."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.config import EPPlan
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE


def local_slots(c: MKConfig, p: EPPlan, S: dict, me, n_live) -> None:
    """S["ep_route_s"][e, j] = row * K + slot of local expert e's j-th live pair (rows
    ascending; each row picks an expert at most once), S["ep_route_s"][EL, e] its count.

    A pair's place among its expert's pairs is an inclusive row prefix sum (on the MXU: 0/1
    and counts <= R are exact in bf16, summed in f32); the j-th pair is the row whose prefix
    reaches j + 1.
    """
    R, K, EL = c.rows, c.top_k, p.experts
    f32 = jnp.float32
    ids = S["ids_v"][:, pl.ds(0, K)] - me * EL  # [R, K] local expert ids
    r_col = jax.lax.broadcasted_iota(jnp.int32, (R, 1), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (R, LANE), 1)
    live = r_col < n_live
    hit = jnp.zeros((R, LANE), f32)
    slot = jnp.zeros((R, LANE), jnp.int32)
    for k in range(K):
        h = (ids[:, k : k + 1] == lane) & live & (lane < EL)
        hit = hit + jnp.where(h, 1.0, 0.0)
        slot = slot + jnp.where(h, k, 0)
    tri = jnp.where(
        jax.lax.broadcasted_iota(jnp.int32, (R, R), 1) <= r_col, 1.0, 0.0
    ).astype(jnp.bfloat16)
    cum = jnp.dot(
        tri, hit.astype(jnp.bfloat16), preferred_element_type=f32
    )  # [R, LANE]
    count = cum[R - 1 : R, :].astype(jnp.int32)  # [1, LANE]
    enc = jnp.where(hit > 0, r_col * K + slot, 0)
    tagged = jnp.where(hit > 0, cum, 0.0)  # a hit row's 1-based place in its expert

    rows_v = S["ep_route_v"].shape[0]
    crow = rows_v - 8
    pad = p.slots_pad - LANE
    count_row = (
        jnp.concatenate([count, jnp.zeros((1, pad), jnp.int32)], axis=1)
        if pad
        else count
    )
    S["ep_route_v"][pl.ds(crow, 8), :] = jnp.concatenate(
        [count_row, jnp.zeros((7, p.slots_pad), jnp.int32)], axis=0
    )
    sem = S["ep_route_sem"].at[0]
    cnt_cp = pltpu.make_async_copy(
        S["ep_route_v"].at[pl.ds(crow, 8)], S["ep_route_s"].at[pl.ds(crow, 8)], sem
    )
    cnt_cp.start()
    cnt_cp.wait()
    max_cnt = jax.lax.fori_loop(
        0, EL, lambda e, m: jnp.maximum(m, S["ep_route_s"][crow, e]), jnp.int32(0)
    )

    def block(jb, carry):
        for jj in range(8):
            j = (jb * 8 + jj + 1).astype(f32)
            S["ep_tab"][pl.ds(jb * 8 + jj, 1), :] = jnp.sum(
                jnp.where(tagged == j, enc, 0), axis=0, keepdims=True
            )
        return carry

    jax.lax.fori_loop(0, (max_cnt + 7) // 8, block, 0)
    tab_t = jnp.transpose(
        S["ep_tab"][...]
    )  # [LANE, slots_pad]; slots past a count: stale
    S["ep_route_v"][pl.ds(0, crow), :] = tab_t[:crow]
    tab_cp = pltpu.make_async_copy(
        S["ep_route_v"].at[pl.ds(0, crow)], S["ep_route_s"].at[pl.ds(0, crow)], sem
    )
    tab_cp.start()
    tab_cp.wait()


def count_row(p: EPPlan, S: dict) -> int:
    """The SMEM row of the per-expert slot counts."""
    return S["ep_route_s"].shape[0] - 8


def build_items(p: EPPlan, S: dict) -> jax.Array:
    """S["ep_items"] = [n_items, (expert, first slot, slots) per item...]; returns n_items."""
    crow = count_row(p, S)
    tab = S["ep_items"]

    def expert(e, n_items):
        cnt = S["ep_route_s"][crow, e]

        def item(b, n):
            tab[1 + 3 * n] = e
            tab[2 + 3 * n] = b * p.item_rows
            tab[3 + 3 * n] = jnp.minimum(cnt - b * p.item_rows, p.item_rows)
            return n + 1

        return jax.lax.fori_loop(
            0, (cnt + p.item_rows - 1) // p.item_rows, item, n_items
        )

    n_items = jax.lax.fori_loop(0, p.experts, expert, jnp.int32(0))
    tab[0] = n_items
    return n_items
