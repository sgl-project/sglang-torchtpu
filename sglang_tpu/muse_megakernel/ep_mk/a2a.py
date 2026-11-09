# SPDX-License-Identifier: Apache-2.0
"""The EP FFN section's ragged all-to-alls by remote DMA: every rank's slot counts first (one
row each), then per (source, destination) one contiguous segment of slot rows, moved in
SEG-row DMAs (segments start on SEG-row boundaries) and waited for in SEG-row units."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.config import SEG
from sglang_tpu.muse_megakernel.ep_mk.route import seg_rows, seg_start
from sglang_tpu.muse_megakernel.mk.config import MKConfig


def exchange_counts(c: MKConfig, S: dict, me) -> None:
    """ep_cnt_v[s] = rank s's slots per expert, on every rank."""
    src = S["ep_csend"].at[pl.ds(0, 1)]
    dst = S["ep_cnt_v"].at[pl.ds(me, 1)]
    recv, sent = S["ep_cnt_sems"].at[0], S["ep_cnt_sems"].at[1]
    pltpu.make_async_copy(src, dst, recv).start()
    for d in range(1, c.tp):
        peer = (me + d) % c.tp
        pltpu.make_async_remote_copy(
            src_ref=src, dst_ref=dst, send_sem=sent, recv_sem=recv, device_id=(peer,)
        ).start()
    tmpl = pltpu.make_async_copy(src, dst, recv)
    for _ in range(c.tp):
        tmpl.wait()
    sent_tmpl = pltpu.make_async_copy(src, dst, sent)
    for _ in range(c.tp - 1):
        sent_tmpl.wait()


def _seg_copy(src_buf, dst_buf, s0, d0, i, peer, me, sems):
    src = src_buf.at[pl.ds(pl.multiple_of(s0 + i * SEG, SEG), SEG)]
    dst = dst_buf.at[pl.ds(pl.multiple_of(d0 + i * SEG, SEG), SEG)]
    recv, sent = sems

    @pl.when(peer == me)
    def _():
        pltpu.make_async_copy(src, dst, recv).start()

    @pl.when(peer != me)
    def _():
        pltpu.make_async_remote_copy(
            src_ref=src, dst_ref=dst, send_sem=sent, recv_sem=recv, device_id=(peer,)
        ).start()


def dispatch(c: MKConfig, S: dict, me) -> None:
    """This rank's send segment for each destination d (ep_xs) into d's receive buffer
    (ep_xr) after the sources before this rank."""
    sems = (S["ep_a2a_sems"].at[0], S["ep_a2a_sems"].at[1])
    for d in range(c.tp):
        _, n_dma = seg_rows(S, me, d)
        s0 = seg_start(S, c.tp, me, d, by_source=False)
        d0 = seg_start(S, c.tp, me, d, by_source=True)

        def one(i, carry, s0=s0, d0=d0, d=d):
            _seg_copy(S["ep_xs"], S["ep_xr"], s0, d0, i, d, me, sems)
            return carry

        jax.lax.fori_loop(0, n_dma, one, 0)


def give_back(c: MKConfig, S: dict, me) -> None:
    """Each source s's received segment's y (ep_yr) back to s's return buffer (ep_yb), where
    s sent it from."""
    sems = (S["ep_a2a_sems"].at[2], S["ep_a2a_sems"].at[3])
    for s in range(c.tp):
        _, n_dma = seg_rows(S, s, me)
        r0 = seg_start(S, c.tp, s, me, by_source=True)
        b0 = seg_start(S, c.tp, s, me, by_source=False)

        def one(i, carry, r0=r0, b0=b0, s=s):
            _seg_copy(S["ep_yr"], S["ep_yb"], r0, b0, i, s, me, sems)
            return carry

        jax.lax.fori_loop(0, n_dma, one, 0)


def _wait(S: dict, buf: str, sem, n) -> None:
    tmpl = pltpu.make_async_copy(
        S[buf].at[pl.ds(0, SEG)], S[buf].at[pl.ds(SEG, SEG)], sem
    )

    def one(i, carry):
        tmpl.wait()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


def _incoming(c: MKConfig, S: dict, me, *, as_source: bool):
    total = jnp.int32(0)
    for x in range(c.tp):
        total = (
            total + seg_rows(S, me, x)[1]
            if as_source
            else total + seg_rows(S, x, me)[1]
        )
    return total


def wait_dispatch(c: MKConfig, S: dict, me) -> None:
    """Waits for every source's segment into ep_xr and this rank's sends out of ep_xs."""
    _wait(S, "ep_xr", S["ep_a2a_sems"].at[0], _incoming(c, S, me, as_source=False))
    sent = jnp.int32(0)
    for d in range(c.tp):
        sent = sent + jnp.where(d == me, 0, seg_rows(S, me, d)[1])
    _wait(S, "ep_xs", S["ep_a2a_sems"].at[1], sent)


def wait_give_back(c: MKConfig, S: dict, me) -> None:
    """Waits for every destination's y of this rank's slots into ep_yb, and this rank's
    sends out of ep_yr."""
    _wait(S, "ep_yb", S["ep_a2a_sems"].at[2], _incoming(c, S, me, as_source=True))
    sent = jnp.int32(0)
    for s in range(c.tp):
        sent = sent + jnp.where(s == me, 0, seg_rows(S, s, me)[1])
    _wait(S, "ep_yr", S["ep_a2a_sems"].at[3], sent)
