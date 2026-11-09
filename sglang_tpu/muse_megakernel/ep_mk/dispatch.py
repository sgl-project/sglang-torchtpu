# SPDX-License-Identifier: Apache-2.0
"""Slot rows in and out of an expert item: each slot's x row from the replicated xfull (local
DMA), and each slot's y row to its row owner's pad_y[0][row, slot] (remote DMA, or local when
this rank owns the row)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.config import MKConfig

_PEND = 0  # ep_cnt[b]: outstanding y row DMAs out of staging buffer b (slots 0, 1)
_SELF = 2  # ep_cnt[2]: this rank's own (row, slot) pairs it computed itself


def slot_of(c: MKConfig, S: dict, e, j):
    """(row, slot) of local expert e's j-th pair."""
    q = S["ep_route_s"][e, j]
    return q // c.top_k, q % c.top_k


def x_row_copy(S: dict, b, t, r):
    return pltpu.make_async_copy(
        S["xfull"].at[pl.ds(r, 1)], S["ep_x"].at[b, pl.ds(t, 1)], S["ep_x_sems"].at[b]
    )


def gather_x_start(c: MKConfig, S: dict, b, e, j0, n) -> None:
    """Starts the item's n x-row copies into ep_x[b]."""

    def one(t, carry):
        r, _ = slot_of(c, S, e, j0 + t)
        x_row_copy(S, b, t, r).start()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


def gather_x_wait(S: dict, b, n) -> None:
    def one(t, carry):
        x_row_copy(S, b, 0, 0).wait()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


def _y_row_template(S: dict, b):
    return pltpu.make_async_copy(
        S["ep_y"].at[b, pl.ds(0, 1)],
        S["pad_y"].at[0, 0, pl.ds(0, 1)],
        S["ep_ys_sems"].at[b],
    )


def drain_y(S: dict, b) -> None:
    """Waits for every row DMA out of ep_y[b] (local or remote: same bytes)."""

    def one(t, carry):
        _y_row_template(S, b).wait()
        return carry

    jax.lax.fori_loop(0, S["ep_cnt"][_PEND + b], one, 0)
    S["ep_cnt"][_PEND + b] = 0


def send_y(c: MKConfig, S: dict, me, b, e, j0, n) -> None:
    """Each of the item's n rows in ep_y[b] to its owner's pad_y[0][row % Rc, slot]."""
    Rc = c.rows_per_dev

    def one(t, carry):
        r, k = slot_of(c, S, e, j0 + t)
        owner, rl = r // Rc, r % Rc
        src = S["ep_y"].at[b, pl.ds(t, 1)]
        dst = S["pad_y"].at[0, rl, pl.ds(k, 1)]

        @pl.when(owner == me)
        def _():
            pltpu.make_async_copy(src, dst, S["ep_ys_sems"].at[b]).start()
            S["ep_cnt"][_SELF] = S["ep_cnt"][_SELF] + 1

        @pl.when(owner != me)
        def _():
            pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=S["ep_ys_sems"].at[b],
                recv_sem=S["ep_yr_sem"].at[0],
                device_id=(owner,),
            ).start()

        return carry

    jax.lax.fori_loop(0, n, one, 0)
    S["ep_cnt"][_PEND + b] = S["ep_cnt"][_PEND + b] + n


def reset_counts(S: dict) -> None:
    for i in (_PEND, _PEND + 1, _SELF):
        S["ep_cnt"][i] = 0


def wait_my_rows(c: MKConfig, S: dict, me, n_live) -> None:
    """Waits for the remote y rows of this rank's live rows (every row has top_k slots; the
    ones this rank computed itself landed through drain_y)."""
    Rc = c.rows_per_dev
    mine = jnp.clip(n_live - me * Rc, 0, Rc) * c.top_k
    msg = pltpu.make_async_remote_copy(
        src_ref=S["ep_y"].at[0, pl.ds(0, 1)],
        dst_ref=S["pad_y"].at[0, 0, pl.ds(0, 1)],
        send_sem=S["ep_ys_sems"].at[0],
        recv_sem=S["ep_yr_sem"].at[0],
        device_id=(me,),
    )

    def one(t, carry):
        msg.wait_recv()
        return carry

    jax.lax.fori_loop(0, mine - S["ep_cnt"][_SELF], one, 0)
