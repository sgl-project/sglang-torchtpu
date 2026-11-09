# SPDX-License-Identifier: Apache-2.0
"""Routing of the EP FFN section: the top-k of this rank's own rows (once per row, on its
owner), each own slot's place in the send buffer, and on the receiving side the received
slots grouped by local expert into work items."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.config import SEG, EPPlan
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE, SMALL
from sglang_tpu.muse_megakernel.mk.numerics import _topk_iter


def _ceil_seg(x):
    return (x + SEG - 1) // SEG * SEG


def own_topk(c: MKConfig, p: EPPlan, S: dict, smr, win) -> None:
    """ep_ids / ep_wsel lanes [0, K): the top-k of the window rows (own rows among them) from
    every row's router logits in S["acc"] (TP's top-k math)."""
    K, E = c.top_k, c.n_experts
    w0, wn, _ = win
    rows = pl.ds(w0, wn)
    bias = smr[SMALL["expert_bias"] : SMALL["expert_bias"] + 1, :E]
    sc = jax.nn.sigmoid(S["acc"][rows, pl.ds(0, E)] * c.router_mult)
    ids_list, picks = _topk_iter(sc + bias, K)
    wsel = jnp.concatenate(
        [jnp.sum(jnp.where(q, sc, 0.0), axis=1, keepdims=True) for q in picks], axis=1
    )
    wsel = wsel / (jnp.sum(wsel, axis=1, keepdims=True) + c.router_eps) * c.route_scale
    S["ep_ids"][:, pl.ds(0, K)] = jnp.concatenate(ids_list, axis=1)
    S["ep_wsel"][:, pl.ds(0, K)] = wsel


def _to_dst(E: int, el: int):
    """[E, LANE] f32: expert -> its rank's lane."""
    return (
        jax.lax.broadcasted_iota(jnp.int32, (E, LANE), 0) // el
        == jax.lax.broadcasted_iota(jnp.int32, (E, LANE), 1)
    ).astype(jnp.float32)


def _dot(a, b):
    return jnp.dot(
        a, b, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    )


def send_places(c: MKConfig, p: EPPlan, S: dict, n_live, win) -> None:
    """ep_csend row 0: this rank's live slots per expert; ep_pos_v / ep_pos_s lanes [0, K)
    (window rows): each own slot's row in ep_xs, where slots run by expert, each destination
    rank's experts in a segment padded to SEG rows, rows ascending within an expert.

    Small exact integers through f32 matmuls of one-hots against prefix masks.
    """
    K, E, rc, el = c.top_k, c.n_experts, p.rows_own, p.experts
    w0, wn, w_off = win
    f32 = jnp.float32
    ids = S["ep_ids"][:, pl.ds(0, K)]
    r_col = jax.lax.broadcasted_iota(jnp.int32, (wn, 1), 0)
    live = (r_col >= w_off) & (r_col < w_off + rc) & (w0 + r_col < n_live)
    lane = jax.lax.broadcasted_iota(jnp.int32, (wn, E), 1)
    onehots = [((ids[:, k : k + 1] == lane) & live).astype(f32) for k in range(K)]
    hit = sum(onehots)  # [window, E]: a row picks an expert at most once
    count = jnp.sum(hit, axis=0, keepdims=True)  # [1, E]
    e_r = jax.lax.broadcasted_iota(jnp.int32, (E, E), 0)
    e_c = jax.lax.broadcasted_iota(jnp.int32, (E, E), 1)
    pre = _dot(count, ((e_r // el == e_c // el) & (e_r < e_c)).astype(f32))  # [1, E]
    to_dst = _to_dst(E, el)
    seg = _ceil_seg(_dot(count, to_dst).astype(jnp.int32)).astype(f32)  # [1, LANE]
    d_r = jax.lax.broadcasted_iota(jnp.int32, (LANE, LANE), 0)
    d_c = jax.lax.broadcasted_iota(jnp.int32, (LANE, LANE), 1)
    base = _dot(_dot(seg, (d_r < d_c).astype(f32)), jnp.transpose(to_dst))  # [1, E]
    tri = (jax.lax.broadcasted_iota(jnp.int32, (wn, wn), 1) < r_col).astype(f32)
    place = base + pre + _dot(tri, hit)  # [window, E]
    pos = jnp.concatenate(
        [jnp.sum(oh * place, axis=1, keepdims=True) for oh in onehots], axis=1
    )
    S["ep_pos_v"][:, pl.ds(0, K)] = pos.astype(jnp.int32)
    S["ep_csend"][...] = jnp.concatenate(
        [count.astype(jnp.int32), jnp.zeros((7, E), jnp.int32)], axis=0
    )
    cp = pltpu.make_async_copy(S["ep_pos_v"], S["ep_pos_s"], S["ep_misc_sems"].at[2])
    cp.start()
    cp.wait()


def slot_matrix(c: MKConfig, S: dict) -> None:
    """ep_nmat_s[s, d]: slots rank s sends rank d; ep_cnt_s: every rank's counts row."""
    E, el, tp = c.n_experts, c.n_experts // c.tp, c.tp
    n = _dot(S["ep_cnt_v"][...].astype(jnp.float32), _to_dst(E, el))  # [tp, LANE]
    S["ep_nmat_v"][pl.ds(0, tp), :] = n.astype(jnp.int32)
    cps = [
        pltpu.make_async_copy(S["ep_nmat_v"], S["ep_nmat_s"], S["ep_misc_sems"].at[2]),
        pltpu.make_async_copy(S["ep_cnt_v"], S["ep_cnt_s"], S["ep_misc_sems"].at[3]),
    ]
    for cp in cps:
        cp.start()
    for cp in cps:
        cp.wait()


def seg_start(S: dict, tp: int, s, d, *, by_source: bool):
    """Start row of the (s -> d) segment: in rank s's send buffer (by_source=False: its
    destinations before d) or in rank d's receive buffer (by_source=True: the sources before
    s); every segment padded to SEG rows."""
    n = S["ep_nmat_s"]
    total = jnp.int32(0)
    for x in range(tp):
        cnt = n[x, d] if by_source else n[s, x]
        before = x < s if by_source else x < d
        total = total + jnp.where(before, _ceil_seg(cnt), 0)
    return total


def seg_rows(S: dict, s, d):
    """Slots of the (s -> d) segment, and its 8-row DMAs."""
    n = S["ep_nmat_s"][s, d]
    return n, (n + SEG - 1) // SEG


def build_items(c: MKConfig, p: EPPlan, S: dict, me) -> jax.Array:
    """ep_rows: received slot rows grouped by local expert (each source's slots for an expert
    are contiguous in its segment); ep_items = [n_items, (expert, first, rows)...] over
    ep_rows, at most item_rows each. Returns n_items."""
    tp, el, ri = c.tp, p.experts, p.item_rows
    cnt, rows, items = S["ep_cnt_s"], S["ep_rows"], S["ep_items"]
    starts = [seg_start(S, tp, s, me, by_source=True) for s in range(tp)]

    def expert(e, carry):
        n_rows, n_items = carry
        first = n_rows
        for s in range(tp):

            def before(e2, acc, s=s):
                return acc + cnt[s, me * el + e2]

            base = starts[s] + jax.lax.fori_loop(0, e, before, jnp.int32(0))

            def one(t, n, base=base):
                rows[n] = base + t
                return n + 1

            n_rows = jax.lax.fori_loop(0, cnt[s, me * el + e], one, n_rows)
        total = n_rows - first

        def item(b, n):
            items[1 + 3 * n] = e
            items[2 + 3 * n] = first + b * ri
            items[3 + 3 * n] = jnp.minimum(total - b * ri, ri)
            return n + 1

        n_items = jax.lax.fori_loop(0, (total + ri - 1) // ri, item, n_items)
        return n_rows, n_items

    _, n_items = jax.lax.fori_loop(0, el, expert, (jnp.int32(0), jnp.int32(0)))
    items[0] = n_items
    return n_items
