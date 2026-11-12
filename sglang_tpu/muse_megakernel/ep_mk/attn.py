# SPDX-License-Identifier: Apache-2.0
"""The EP kernel's o-projection: every rank's partial of every row, reduce-scattered by ROW
(each wo column tile's rows go to their owner as it finishes), then the post-attention
residual on own rows only; no all-gather (the FFN section runs on own rows)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.config import EPPlan, window_rows
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import SMALL
from sglang_tpu.muse_megakernel.mk.hooks import EPOProjStep
from sglang_tpu.muse_megakernel.mk.numerics import _rms
from sglang_tpu.muse_megakernel.mk.packing import _pack_bf16x2, _unpack_bf16x2

f32, bf16 = jnp.float32, jnp.bfloat16


def _row(smr, name, n):
    return smr[SMALL[name] : SMALL[name] + 1, :n]


def _aligned8(x):
    """x, hinted 8-row aligned when traced."""
    return x if isinstance(x, int) else pl.multiple_of(x, 8)


def tile_value(row, m: int, blk):
    """A row tile's [m, H] value: from its block ref, or row by row."""
    if blk is not None:
        return blk[...]
    return jnp.concatenate([row(i)[...] for i in range(m)], axis=0)


def tile_store(row, m: int, blk, v) -> None:
    if blk is not None:
        blk[...] = v
        return
    for i in range(m):
        row(i)[...] = v[i : i + 1]


def window(c: MKConfig, me):
    """(w0, win, w_off): the 16-row aligned window holding own rows at [w_off, w_off + Rc)."""
    rc, win = c.rows_per_dev, window_rows(c)
    if win == rc:
        return pl.multiple_of(me * rc, 16), win, 0
    w0 = pl.multiple_of(jnp.minimum((me * rc) // 16 * 16, c.rows - win), 16)
    return w0, win, me * rc - w0


def o_proj(step: EPOProjStep, p: EPPlan, stream_rows) -> None:
    """Own rows: o = sum over ranks of the o-proj partials (bf16, as TP's o_send), then
    h_stream = attn_alpha * stream + attn_beta * norm(o) and the window's S["lhs"] =
    pre-FFN-normed stream (own rows real; the window's other rows only feed unused logits).

    stream_rows: layer._stream_rows on S["h_stream"].
    """
    c, S, smr, me = step.c, step.S, step.smr, step.me
    tp, H, Hl, rc = c.tp, c.hidden, c.hidden_local, p.rows_own
    send, recv = S["ep_osend"], S["ep_orecv"]
    rsem = S["ep_o_sems"].at[0]

    def sent(b):
        # one count per send buffer: a shared one lets tile ni - 1's finished sends stand in
        # for tile ni - 2's, whose buffer is then overwritten mid-flight
        return S["ep_o_sems"].at[1 + b]

    def wait_sent(b):
        for _ in range(tp - 1):
            pltpu.make_async_copy(send.at[0, 0], recv.at[0], sent(b)).wait()

    def ship(ni, col):
        b = ni % 2
        if ni >= 2:  # tile ni - 2's sends out of this buffer have left
            wait_sent(b)
        for o in range(tp):
            blk = _pack_bf16x2(S["acc"][pl.ds(o * rc, rc), pl.ds(0, Hl)])

            @pl.when(o == me)
            def _(blk=blk):
                recv[me * tp + col, pl.ds(0, rc), :] = blk

            @pl.when(o != me)
            def _(blk=blk, o=o):
                send[b, o, pl.ds(0, rc), :] = blk
                pltpu.make_async_remote_copy(
                    src_ref=send.at[b, o],
                    dst_ref=recv.at[me * tp + col],
                    send_sem=sent(b),
                    recv_sem=rsem,
                    device_id=(o,),
                ).start()

    step.mv(ship)
    n_tiles = H // Hl
    for ni in range(max(0, n_tiles - 2), n_tiles):
        wait_sent(ni % 2)
    for _ in range(n_tiles * (tp - 1)):  # every other rank's block of every column tile
        pltpu.make_async_copy(send.at[0, 0], recv.at[0], rsem).wait()

    _, rows = stream_rows
    a_alpha, a_beta = _row(smr, "attn_alpha", H), _row(smr, "attn_beta", H)
    w0, wn, w_off = window(c, me)
    s_win = S["ep_f"]  # [window, H] f32: own rows' post-attention stream, others zero
    s_win[...] = jnp.zeros((wn, H), f32)

    def residual(row, t0, m, blk):
        # the tile's m rows at once (one-row arrays cost a permute per bf16 unpack)
        a = (
            jnp.concatenate(
                [
                    sum(
                        _unpack_bf16x2(recv[src * tp + col, pl.ds(t0, m), :])
                        for src in range(tp)
                    )
                    for col in range(H // Hl)
                ],
                axis=1,
            )
            .astype(bf16)
            .astype(f32)
        )  # [m, H]
        s_a = jax.lax.rsqrt(
            jnp.mean(jnp.square(a), axis=-1, keepdims=True) + c.post_eps
        )
        h = tile_value(row, m, blk)
        s = a_alpha * h + a_beta * (a * s_a).astype(bf16).astype(f32)
        tile_store(row, m, blk, s)
        if rc % 8 == 0:  # the window offset is 8-row aligned
            s_win[pl.ds(_aligned8(w_off + t0), m), :] = s
        else:  # an unaligned window offset: one-row stores
            for i in range(m):
                s_win[pl.ds(w_off + t0 + i, 1), :] = s[i : i + 1]

    rows(w0 + w_off, rc, residual)
    S["lhs"][pl.ds(w0, wn), :] = _rms(
        s_win[...], c.rms_eps, _row(smr, "pre_ffn_w", H)
    ).astype(bf16)
