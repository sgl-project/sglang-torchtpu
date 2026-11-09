# SPDX-License-Identifier: Apache-2.0
"""Slot rows in and out of an expert item: each slot's x row from the replicated xfull (local
DMA), and each slot's y row to its row owner's ep_pad[row, slot] (remote DMA, or local when
this rank owns the row)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.config import EPBuffers, EPPlan
from sglang_tpu.muse_megakernel.ep_mk.route import slot_pair
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE

_PEND = 0  # ep_cnt[0]: outstanding y row DMAs out of ep_y
_SELF = 1  # ep_cnt[1]: this rank's own (row, slot) pairs it computed itself


def slot_of(c: MKConfig, p: EPPlan, S: dict, e, j):
    """(row, slot) of local expert e's j-th pair."""
    q = slot_pair(p, S, e, j)
    return q // c.top_k, q % c.top_k


def x_row_copy(S: dict, buf: EPBuffers, b, t, r):
    return pltpu.make_async_copy(
        S["xfull"].at[pl.ds(r, 1)], buf.x(b).at[pl.ds(t, 1)], S["ep_x_sems"].at[b]
    )


def gather_x_start(
    c: MKConfig, p: EPPlan, S: dict, buf: EPBuffers, b, e, j0, n
) -> None:
    """Starts the item's n x-row copies into buf.x(b)."""

    def one(t, carry):
        r, _ = slot_of(c, p, S, e, j0 + t)
        x_row_copy(S, buf, b, t, r).start()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


def gather_x_wait(S: dict, buf: EPBuffers, b, n) -> None:
    def one(t, carry):
        x_row_copy(S, buf, b, 0, 0).wait()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


def _y_row_template(S: dict, buf: EPBuffers):
    return pltpu.make_async_copy(
        buf.y.at[pl.ds(0, 1)], S["ep_pad"].at[0, pl.ds(0, 1)], S["ep_ys_sems"].at[0]
    )


def drain_y(S: dict, buf: EPBuffers) -> None:
    """Waits for every row DMA out of buf.y (local or remote: same bytes)."""

    def one(t, carry):
        _y_row_template(S, buf).wait()
        return carry

    jax.lax.fori_loop(0, S["ep_cnt"][_PEND], one, 0)
    S["ep_cnt"][_PEND] = 0


def send_y(c: MKConfig, p: EPPlan, S: dict, buf: EPBuffers, me, e, j0, n) -> None:
    """Each of the item's n rows in buf.y to its owner's ep_pad[row % Rc, slot]."""
    Rc = c.rows_per_dev

    def one(t, carry):
        r, k = slot_of(c, p, S, e, j0 + t)
        owner, rl = r // Rc, r % Rc
        src = buf.y.at[pl.ds(t, 1)]
        dst = S["ep_pad"].at[rl, pl.ds(k, 1)]

        @pl.when(owner == me)
        def _():
            pltpu.make_async_copy(src, dst, S["ep_ys_sems"].at[0]).start()
            S["ep_cnt"][_SELF] = S["ep_cnt"][_SELF] + 1

        @pl.when(owner != me)
        def _():
            pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=S["ep_ys_sems"].at[0],
                recv_sem=S["ep_yr_sem"].at[0],
                device_id=(owner,),
            ).start()

        return carry

    jax.lax.fori_loop(0, n, one, 0)
    S["ep_cnt"][_PEND] = S["ep_cnt"][_PEND] + n


def reset_counts(S: dict) -> None:
    S["ep_cnt"][_PEND] = 0
    S["ep_cnt"][_SELF] = 0


def send_counts(c: MKConfig, p: EPPlan, S: dict, me, n_items) -> None:
    """Tells every rank how many y rows this rank will send it (ep_cin[me] there, lane 0):
    the owner then waits for what is coming, whatever its own routing says."""
    Rc, tp, items = c.rows_per_dev, c.tp, S["ep_items"]
    for o in range(tp):
        S["ep_cnt_s"][o] = 0

    def item(i, carry):
        e, j0, n = items[1 + 3 * i], items[2 + 3 * i], items[3 + 3 * i]

        def one(t, carry2):
            r, _ = slot_of(c, p, S, e, j0 + t)
            S["ep_cnt_s"][r // Rc] = S["ep_cnt_s"][r // Rc] + 1
            return carry2

        return jax.lax.fori_loop(0, n, one, carry)

    jax.lax.fori_loop(0, n_items, item, 0)
    for o in range(tp):
        S["ep_cout"][pl.ds(o, 1), :] = jnp.full((1, LANE), S["ep_cnt_s"][o], jnp.int32)
    for o in range(tp):
        src, dst = S["ep_cout"].at[pl.ds(o, 1)], S["ep_cin"].at[pl.ds(me, 1)]

        @pl.when(me == o)
        def _(src=src, dst=dst):
            pltpu.make_async_copy(src, dst, S["ep_cin_sem"].at[0]).start()

        @pl.when(me != o)
        def _(src=src, dst=dst, o=o):
            pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=S["ep_cout_sem"].at[0],
                recv_sem=S["ep_cin_sem"].at[0],
                device_id=(o,),
            ).start()


def wait_my_rows(c: MKConfig, S: dict, buf: EPBuffers, me) -> None:
    """Waits for every rank's count message, then for the remote y rows they announced (the
    ones this rank computed itself landed through drain_y)."""
    tp = c.tp
    tmpl = pltpu.make_async_copy(
        S["ep_cout"].at[pl.ds(0, 1)], S["ep_cin"].at[pl.ds(0, 1)], S["ep_cin_sem"].at[0]
    )
    for _ in range(tp):
        tmpl.wait()
    sent = pltpu.make_async_copy(
        S["ep_cout"].at[pl.ds(0, 1)],
        S["ep_cin"].at[pl.ds(0, 1)],
        S["ep_cout_sem"].at[0],
    )
    for _ in range(tp - 1):
        sent.wait()
    cp = pltpu.make_async_copy(S["ep_cin"], S["ep_cin_s"], S["ep_route_sem"].at[0])
    cp.start()
    cp.wait()
    coming = sum(S["ep_cin_s"][o, 0] for o in range(tp))
    msg = pltpu.make_async_remote_copy(
        src_ref=buf.y.at[pl.ds(0, 1)],
        dst_ref=S["ep_pad"].at[0, pl.ds(0, 1)],
        send_sem=S["ep_ys_sems"].at[0],
        recv_sem=S["ep_yr_sem"].at[0],
        device_id=(me,),
    )

    def one(t, carry):
        msg.wait_recv()
        return carry

    jax.lax.fori_loop(0, coming - S["ep_cnt"][_SELF], one, 0)
