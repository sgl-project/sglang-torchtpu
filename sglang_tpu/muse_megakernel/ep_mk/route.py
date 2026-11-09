# SPDX-License-Identifier: Apache-2.0
"""Routing of the EP FFN section: the top-k of this rank's own rows (once per row, on its
owner), each own slot's place in the send buffer, and on the receiving side the received
slots grouped by local expert into work items."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.config import SEG, TAB_ROWS, EPPlan
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
    """ep_csend row 0: this rank's live slots per expert; ep_pos_v / ep_pos_s [k, window row]:
    each own slot's row in ep_xs, where slots run by expert, each destination rank's experts
    in a segment padded to SEG rows, rows ascending within an expert.

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
    pos_t = jnp.transpose(  # [LANE, window]: slot k's places on row k
        jnp.concatenate(
            [pos.astype(jnp.int32), jnp.zeros((wn, LANE - K), jnp.int32)], axis=1
        )
    )
    S["ep_pos_v"][...] = (
        jnp.concatenate([pos_t[:8], jnp.zeros((8, LANE - wn), jnp.int32)], axis=1)
        if wn < LANE
        else pos_t[:8]
    )
    S["ep_csend"][...] = jnp.concatenate(
        [count.astype(jnp.int32), jnp.zeros((7, E), jnp.int32)], axis=0
    )
    cp = pltpu.make_async_copy(S["ep_pos_v"], S["ep_pos_s"], S["ep_misc_sems"].at[2])
    cp.start()
    cp.wait()


# rows of S["ep_tab_s"] (route.slot_tables): per (source s / destination d) lane
T_N = 0  # [s, d]: slots s sends d (8 rows)
T_MINE = 8  # [s, e]: slots s sends this rank's local expert e (8 rows)
T_PRE = 16  # [s, e]: the slots before e's within s's segment here (8 rows)
T_RECV = 24  # [s]: s's segment start in this rank's receive buffer
T_SEND = 25  # [d]: this rank's segment for d in its send buffer
T_DST = 26  # [d]: this rank's segment start in d's receive buffer
T_BACK = 27  # [s]: s's segment for this rank in s's send buffer (where y goes back)
T_DMAS = 28  # [0]: SEG-row DMAs into this rank's receive buffer, [1]: into its return


def slot_tables(c: MKConfig, p: EPPlan, S: dict, me) -> None:
    """S["ep_tab_s"] from every rank's per-expert counts (ep_cnt_v), all on the VPU / MXU:
    the slot matrix, segment starts (SEG-padded) on both sides, and this rank's experts'
    counts and offsets per source."""
    E, el, tp = c.n_experts, p.experts, c.tp
    f32, i32 = jnp.float32, jnp.int32
    cnt = S["ep_cnt_v"][...].astype(f32)  # [tp, E]
    n = _dot(cnt, _to_dst(E, el))  # [tp, LANE]
    sel = (
        jax.lax.broadcasted_iota(i32, (E, LANE), 0)
        == me * el + jax.lax.broadcasted_iota(i32, (E, LANE), 1)
    ) & (jax.lax.broadcasted_iota(i32, (E, LANE), 1) < el)
    mine = _dot(cnt, sel.astype(f32))  # [tp, LANE]
    l_r = jax.lax.broadcasted_iota(i32, (LANE, LANE), 0)
    l_c = jax.lax.broadcasted_iota(i32, (LANE, LANE), 1)
    upper = (l_r < l_c).astype(f32)  # x @ upper: exclusive prefix along lanes
    pre = _dot(mine, upper)
    seg = _ceil_seg(n.astype(i32)).astype(f32)  # [tp, LANE]
    s_col = jax.lax.broadcasted_iota(i32, (tp, 1), 0)
    lane = jax.lax.broadcasted_iota(i32, (1, LANE), 1)
    at_me = (lane == me).astype(f32)  # [1, LANE]
    col_me = jnp.sum(seg * at_me, axis=1, keepdims=True)  # [tp, 1]: seg[s, me]
    row_me = jnp.sum(
        jnp.where(s_col == me, seg, 0.0), axis=0, keepdims=True
    )  # seg[me, :]

    def as_row(col):  # [tp, 1] -> [1, LANE], lane s = col[s]
        return jnp.sum(jnp.where(s_col == lane, col, 0.0), axis=0, keepdims=True)

    recv = as_row(
        jnp.sum(jnp.where(s_col > s_col.T, col_me.T, 0.0), axis=1, keepdims=True)
    )
    send = _dot(row_me, upper)
    dst = jnp.sum(jnp.where(s_col < me, seg, 0.0), axis=0, keepdims=True)
    back = as_row(jnp.sum(seg * (lane < me).astype(f32), axis=1, keepdims=True))
    dmas = jnp.where(
        lane == 0,
        jnp.sum(col_me) / SEG,
        jnp.where(lane == 1, jnp.sum(row_me) / SEG, 0.0),
    )
    rows = [n, mine, pre, recv, send, dst, back, dmas]
    rows.append(jnp.zeros((TAB_ROWS - T_DMAS - 1, LANE), f32))
    S["ep_tab_v"][...] = jnp.concatenate(rows, axis=0).astype(i32)
    cp = pltpu.make_async_copy(S["ep_tab_v"], S["ep_tab_s"], S["ep_misc_sems"].at[2])
    cp.start()
    cp.wait()


def seg_dmas(S: dict, s, d):
    """SEG-row DMAs of the (s -> d) segment."""
    return (S["ep_tab_s"][T_N + s, d] + SEG - 1) // SEG


def build_items(c: MKConfig, p: EPPlan, S: dict) -> jax.Array:
    """ep_rows: received slot rows grouped by local expert (each source's slots for an expert
    are contiguous in its segment); ep_items = [n_items, (expert, first, rows)...] over
    ep_rows, at most item_rows each. Returns n_items."""
    tp, el, ri = c.tp, p.experts, p.item_rows
    tab, rows, items = S["ep_tab_s"], S["ep_rows"], S["ep_items"]

    def expert(e, carry):
        n_rows, n_items = carry
        first = n_rows
        for s in range(tp):
            base = tab[T_RECV, s] + tab[T_PRE + s, e]

            def one(t, n, base=base):
                rows[n] = base + t
                return n + 1

            n_rows = jax.lax.fori_loop(0, tab[T_MINE + s, e], one, n_rows)
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
