# SPDX-License-Identifier: Apache-2.0
"""Row-tiled layer and step driver of the decode megakernel (MKConfig.row_tile)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.common.consts import (
    LANE,
    SMALL,
)
from sglang_tpu.muse_megakernel.common.control import for_layer_experts
from sglang_tpu.muse_megakernel.common.hooks import (
    EPLayerStep,
)
from sglang_tpu.muse_megakernel.common.numerics import (
    _rms,
)
from sglang_tpu.muse_megakernel.common.scratch import (
    _TS_AUX,
    _TS_OUT,
    _TS_ROPE,
)
from sglang_tpu.muse_megakernel.common.trace import (
    _scope,
)
from sglang_tpu.muse_megakernel.decode.tp.ctx import KernelCtx


def layer_tiled(k: KernelCtx, l, kind, next_layer, prefetched, pend=False):
    H = k.H
    L = k.L
    NT = k.NT
    S = k.S
    W = k.W
    ag_tile_ready = k.ag_tile_ready
    ag_tile_sent = k.ag_tile_sent
    attention_tiled = k.attention_tiled
    bf16 = k.bf16
    c = k.c
    capture_tile = k.capture_tile
    ep = k.ep
    f32 = k.f32
    flags = k.flags
    gathered_tile = k.gathered_tile
    group_lo = k.group_lo
    hag_forward = k.hag_forward
    hag_ready = k.hag_ready
    hag_sent = k.hag_sent
    kind_layers = k.kind_layers
    layer_ex = k.layer_ex
    me = k.me
    moe_tiled = k.moe_tiled
    mv_tiled = k.mv_tiled
    n_live = k.n_live
    pad_view = k.pad_view
    prefetch_smalls = k.prefetch_smalls
    smalls_copy = k.smalls_copy
    stream_view = k.stream_view
    tile_pass = k.tile_pass
    vpads_tiles = k.vpads_tiles
    ws = k.ws
    nxt = l + 1 if next_layer else None
    slot = l % 2
    if not prefetched:
        smalls_copy(l, slot).start()
        ws.prologue(l)
    with _scope("smalls_wait"):
        # prefetched under EP: the previous layer's next_smalls waited
        if ep is None or not prefetched:
            smalls_copy(l, slot).wait()
    smr = S["smalls"].at[slot]
    use_rope, is_global = flags(l)

    def row(name, n=H):
        return smr[SMALL[name] : SMALL[name] + 1, :n]

    if ep is not None and prefetched:  # gathered by the previous layer's hook
        ep.pre_attn(c, S)
    elif not pend:
        if c.tiled_vmem_pads:

            def _tile(rs, t):
                S["lhs"][rs] = _rms(
                    S["h_stream"][rs], c.rms_eps, row("pre_attn_w")
                ).astype(bf16)

            vpads_tiles(_tile)
        else:

            def pre_attn_tile(rows, t, v):
                S["lhs"][rows] = _rms(v[0][...], c.rms_eps, row("pre_attn_w")).astype(
                    bf16
                )
                return []

            tile_pass([(stream_view, S["sin"])], pre_attn_tile)
    with _scope("attention"):
        attention_tiled(l, use_rope, is_global, nxt, smr, pend=pend)
    prefetch_smalls(l, nxt)

    if ep is None and not c.fold_o:  # the EP hook's o_proj did it on own rows
        if c.tiled_vmem_pads:
            if c.pipe_ag:
                o_ag_args = ("o_ag", S["o_send"], S["pad_o_ag"], S["oag_tsems"])
                hag_forward(*o_ag_args, 0, S["oag_ptsems"])

            def _tile(rs, t):
                if c.pipe_ag:
                    pl.when(t + 1 < NT)(
                        lambda: hag_forward(*o_ag_args, t + 1, S["oag_ptsems"])
                    )
                    hag_ready("o_ag", S["pad_o_ag"], t, S["oag_ptsems"])
                s = row("attn_alpha") * S["h_stream"][rs] + row("attn_beta") * _rms(
                    gathered_tile(S["pad_o_ag"].at[:, rs]), c.post_eps
                ).astype(bf16).astype(f32)
                S["h_stream"][rs] = s
                S["lhs"][rs] = _rms(s, c.rms_eps, row("pre_ffn_w")).astype(bf16)

            vpads_tiles(_tile)
            if c.pipe_ag:  # o_send (and what borrows it) is reused after this
                hag_sent(*o_ag_args, S["oag_ptsems"])
        else:
            oag_rdy = ag_tile_ready(S["o_send"], S["pad_o_ag"], S["oag_tsems"])

            def post_attn_tile(rows, t, v):
                s = row("attn_alpha") * v[1][...] + row("attn_beta") * _rms(
                    gathered_tile(v[0]), c.post_eps
                ).astype(bf16).astype(f32)
                S["lhs"][rows] = _rms(s, c.rms_eps, row("pre_ffn_w")).astype(bf16)
                return [s]

            tile_pass(
                [
                    (pad_view(S["pad_o_ag"]), S["ptile"]),
                    (stream_view, S["sin"]),
                ],
                post_attn_tile,
                [(stream_view, S["sout"])],
                ready=[oag_rdy, None],
            )
            ag_tile_sent("o_ag", S["o_send"], S["pad_o_ag"])
    if ep is not None:
        ep_for_layer = functools.partial(
            for_layer_experts, l, kind, layer_ex, kind_layers, group_lo
        )

        def next_smalls():
            """The next layer's smalls slab, waited for (its layer_tiled does not)."""
            nslot = (l + 1) % 2
            if isinstance(l, int):
                smalls_copy(l + 1, nslot).wait()
            else:
                pl.when(l + 1 < L)(smalls_copy(l + 1, nslot).wait)
            return S["smalls"].at[nslot]

        with _scope("moe"):
            ep.layer_moe(
                EPLayerStep(
                    c=c,
                    S=S,
                    W=W,
                    smr=smr,
                    layer=l,
                    kind=kind,
                    for_layer=ep_for_layer,
                    wait_refs=layer_ex[
                        l if isinstance(l, int) else kind_layers[kind][0]
                    ],
                    me=me,
                    n_live=n_live,
                    router=lambda: mv_tiled(l, "wrouter", nxt, smr),
                    capture=lambda rows, s: capture_tile(l, rows, s),
                    next_smalls=None if nxt is None else next_smalls,
                )
            )
    else:
        with _scope("moe"):
            moe_tiled(l, smr, kind, nxt)

        if not (c.fold_f and nxt is not None):
            if c.tiled_vmem_pads:
                if c.pipe_ag:
                    f_ag_args = ("f", S["f_send"], S["pad_f"], S["fag_tsems"])
                    hag_forward(*f_ag_args, 0, S["fag_ptsems"])

                def _tile(rs, t):
                    if c.pipe_ag:
                        pl.when(t + 1 < NT)(
                            lambda: hag_forward(*f_ag_args, t + 1, S["fag_ptsems"])
                        )
                        hag_ready("f", S["pad_f"], t, S["fag_ptsems"])
                    s = row("ffn_alpha") * S["h_stream"][rs] + row("ffn_beta") * _rms(
                        gathered_tile(S["pad_f"].at[:, rs]) * row("post_ffn_w"),
                        c.post_eps,
                    ).astype(bf16).astype(f32)
                    capture_tile(l, rs, s)
                    S["h_stream"][rs] = s

                vpads_tiles(_tile)
                if c.pipe_ag:
                    hag_sent(*f_ag_args, S["fag_ptsems"])
            else:
                fag_rdy = ag_tile_ready(S["f_send"], S["pad_f"], S["fag_tsems"])

                def ffn_tile(rows, t, v):
                    s = row("ffn_alpha") * v[1][...] + row("ffn_beta") * _rms(
                        gathered_tile(v[0]) * row("post_ffn_w"), c.post_eps
                    ).astype(bf16).astype(f32)
                    capture_tile(l, rows, s)
                    return [s]

                tile_pass(
                    [
                        (pad_view(S["pad_f"]), S["ptile"]),
                        (stream_view, S["sin"]),
                    ],
                    ffn_tile,
                    [(stream_view, S["sout"])],
                    ready=[fag_rdy, None],
                )
                ag_tile_sent("f", S["f_send"], S["pad_f"])


def tiled_main(
    k: KernelCtx,
):
    # rotary rows for every row's position, 8 rows per round of DMAs
    H = k.H
    L = k.L
    R = k.R
    RT = k.RT
    S = k.S
    W = k.W
    aux_out = k.aux_out
    bf16 = k.bf16
    c = k.c
    dma = k.dma
    emb_ref = k.emb_ref
    f32 = k.f32
    hd = k.hd
    hid_out = k.hid_out
    layer_tiled = k.layer_tiled
    ncap = k.ncap
    pos_ref = k.pos_ref
    roll_hi = k.roll_hi
    roll_lo = k.roll_lo
    stream_view = k.stream_view
    tile_live = k.tile_live
    tile_pass = k.tile_pass
    tsem = k.tsem
    vpads_tiles = k.vpads_tiles

    def rope_round(b, carry):
        r0 = pl.multiple_of(b * 8, 8)
        cps, ps_ = [], []
        for i in range(8):
            p = jnp.clip(pos_ref[r0 + i], 0, c.max_position - 1)
            cp = pltpu.make_async_copy(
                W["rope"].at[pl.ds(pl.multiple_of((p // 8) * 8, 8), 8)],
                S["rope_blk"].at[i],
                tsem(_TS_ROPE + i),
            )
            cp.start()
            cps.append(cp)
            ps_.append(p)
        rr = []
        for i in range(8):
            cps[i].wait()
            sel = jax.lax.broadcasted_iota(jnp.int32, (8, 2 * hd), 0) == (
                ps_[i] - (ps_[i] // 8) * 8
            )
            rr.append(
                jnp.sum(jnp.where(sel, S["rope_blk"][i], 0.0), axis=0, keepdims=True)
            )
        S["rtab"][pl.ds(r0, 8), :] = jnp.concatenate(rr, axis=0)
        return carry

    jax.lax.fori_loop(0, R // 8, rope_round, 0)

    # embedding: weightless RMS norm of the bf16 rows, kept in bf16 precision
    if c.tiled_vmem_pads:
        dma(emb_ref, S["lhs"], _TS_AUX)

        def _tile(rs, t):
            S["h_stream"][rs] = (
                _rms(S["lhs"][rs].astype(f32), c.rms_eps).astype(bf16).astype(f32)
            )

        vpads_tiles(_tile)
    else:

        def emb_tile(rows, t, v):
            return [_rms(v[0][...].astype(f32), c.rms_eps).astype(bf16).astype(f32)]

        tile_pass(
            [(lambda rows: emb_ref.at[rows], S["bin"])],
            emb_tile,
            [(stream_view, S["sout"])],
        )
    pref = False
    pend = False
    for l in range(L):
        if roll_lo is not None and l == roll_lo and l < roll_hi:
            lo, hi = l, roll_hi
            if not pref:
                layer_tiled(lo, "fp4", lo + 1 < L, False, pend=pend)
                pref = lo + 1 < L
                pend = c.fold_f and (lo + 1 < L)
                lo += 1
            last = hi if hi < L else hi - 1

            def rolled(i, v):
                assert pend == c.fold_f
                layer_tiled(i, "fp4", True, True, pend=c.fold_f)
                return v

            if last > lo:
                jax.lax.fori_loop(lo, last, rolled, 0)
                pref = True
                pend = c.fold_f
            if hi == L and hi - 1 >= lo:
                layer_tiled(hi - 1, "fp4", False, pref, pend=pend)
                pref = False
                pend = False
        elif roll_lo is None or not (roll_lo <= l < roll_hi):
            layer_tiled(l, c.expert_kinds[l], l + 1 < L, pref, pend=pend)
            pref = l + 1 < L
            pend = c.fold_f and (l + 1 < L)
    dma(W["final_norm"], S["final_norm"], _TS_OUT)

    # final norm on the bf16-cast stream, as the native path
    if c.tiled_vmem_pads:
        if ncap == 0:
            S["btile"][...] = jnp.zeros((RT, H), bf16)

        def _tile(rs, t):
            if ncap == 0:
                dma(S["btile"].at[:, pl.ds(0, LANE)], aux_out.at[rs], _TS_AUX)
            hidden = _rms(
                S["h_stream"][rs].astype(bf16).astype(f32),
                c.rms_eps,
                S["final_norm"][0:1, :],
            )
            S["lhs"][rs] = jnp.where(tile_live(t), hidden, 0.0).astype(bf16)

        vpads_tiles(_tile)
        dma(S["lhs"], hid_out, _TS_OUT)
    else:

        def final_tile(rows, t, v):
            if ncap == 0:
                S["btile"][...] = jnp.zeros((RT, H), bf16)
                dma(S["btile"].at[:, pl.ds(0, LANE)], aux_out.at[rows], _TS_AUX)
            hidden = _rms(
                v[0][...].astype(bf16).astype(f32),
                c.rms_eps,
                S["final_norm"][0:1, :],
            )
            return [jnp.where(tile_live(t), hidden, 0.0).astype(bf16)]

        tile_pass(
            [(stream_view, S["sin"])],
            final_tile,
            [(lambda rows: hid_out.at[rows], S["bin"])],
        )
