# SPDX-License-Identifier: Apache-2.0
"""One layer's EP expert section (the row-tiled megakernel's hook): route this rank's slots,
stream each picked expert's weights once in I-chunks through the TP expert ring, compute its
items, send every slot's y row to the row owner, then combine this rank's rows into u."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk import dispatch
from sglang_tpu.muse_megakernel.ep_mk.config import EPPlan, ep_plan, ep_scratch
from sglang_tpu.muse_megakernel.ep_mk.experts import (
    ChunkRefs,
    chunk_compute,
    item_sizes,
    sized,
)
from sglang_tpu.muse_megakernel.ep_mk.route import build_items, local_slots
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import FP4_GROUP, FP4_PACK, LANE, SMALL
from sglang_tpu.muse_megakernel.mk.hooks import EPMoEStep
from sglang_tpu.muse_megakernel.mk.packing import _pack_bf16x2, _unpack_bf16x2
from sglang_tpu.muse_megakernel.mk.plan import owner_window_rows


class EPExperts:
    """EPHook: whole experts per rank (ep_plan) for the row-tiled megakernel."""

    def __init__(self, cfg: MKConfig, item_rows: int = 64):
        self.plan = ep_plan(cfg, item_rows)

    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        return plan + ep_scratch(cfg, self.plan)

    def moe(self, step: EPMoEStep) -> None:
        c, S, p = step.c, step.S, self.plan
        local_slots(c, p, S, step.me, step.n_live)
        n_items = build_items(p, S)
        dispatch.reset_counts(S)
        _expert_loop(step, p, n_items)
        step.flush_dense()
        dispatch.drain_y(S, 0)
        dispatch.drain_y(S, 1)
        dispatch.wait_my_rows(c, S, step.me, step.n_live)
        _combine(step)


def _chunk_copies(step: EPMoEStep, p: EPPlan, refs, off, e, j, slot) -> list:
    """The six DMAs of expert e's I-chunk j into ring slot `slot`."""
    c, S, kind = step.c, step.S, step.kind
    He = c.expert_hidden
    CI = p.chunk_fp4 if kind == "fp4" else p.chunk_fp8
    pack = FP4_PACK if kind == "fp4" else 4
    W13, S13 = _ring(step, p)
    sems = S["ep_w_sems"]
    src = [r if off is None else r.at[off] for r in refs]
    w13, s13, w2, s2 = (
        src[0].bitcast(jnp.uint32),
        src[1],
        src[2].bitcast(jnp.uint32),
        src[3],
    )
    g0 = pl.multiple_of(j * CI, CI)
    u0 = pl.multiple_of(p.inter + j * CI, CI)
    k2 = pl.multiple_of(j * (CI // pack), CI // pack)
    if kind == "fp4":
        srows, s2_src = (
            He // FP4_GROUP,
            s2.at[e, pl.ds(j * (CI // FP4_GROUP), CI // FP4_GROUP)],
        )
        s2_dst = S["w2s"].at[slot, pl.ds(0, CI // FP4_GROUP)]
    else:
        srows, s2_src = 2, s2.at[e]
        s2_dst = S["w2s"].at[slot, pl.ds(0, 2)]
    return [
        pltpu.make_async_copy(
            w13.at[e, :, pl.ds(g0, CI)], W13.at[slot, :, pl.ds(0, CI)], sems.at[slot, 0]
        ),
        pltpu.make_async_copy(
            w13.at[e, :, pl.ds(u0, CI)],
            W13.at[slot, :, pl.ds(CI, CI)],
            sems.at[slot, 1],
        ),
        pltpu.make_async_copy(
            s13.at[e, :, pl.ds(g0, CI)],
            S13.at[slot, pl.ds(0, srows), pl.ds(0, CI)],
            sems.at[slot, 2],
        ),
        pltpu.make_async_copy(
            s13.at[e, :, pl.ds(u0, CI)],
            S13.at[slot, pl.ds(0, srows), pl.ds(CI, CI)],
            sems.at[slot, 3],
        ),
        pltpu.make_async_copy(
            w2.at[e, pl.ds(k2, CI // pack)], S["w2"].at[slot], sems.at[slot, 4]
        ),
        pltpu.make_async_copy(s2_src, s2_dst, sems.at[slot, 5]),
    ]


def _ring(step: EPMoEStep, p: EPPlan):
    """The TP expert ring as I-chunk slots: (w13, w13 scales) views of this layer's kind."""
    S, He = step.S, step.c.expert_hidden
    if step.kind == "fp4":
        return S["w13"], S["w13s"]
    return S["w13"].reshape(p.nslot, He // 4, 2 * p.chunk_fp8), S["w13s"]


def _chunk_refs(step: EPMoEStep, p: EPPlan, slot) -> ChunkRefs:
    S, kind = step.S, step.kind
    W13, S13 = _ring(step, p)
    CI = p.chunk_fp4 if kind == "fp4" else p.chunk_fp8
    srows = step.c.expert_hidden // FP4_GROUP if kind == "fp4" else 2
    s2rows = CI // FP4_GROUP if kind == "fp4" else 2
    return ChunkRefs(
        wg=W13.at[slot, :, pl.ds(0, CI)],
        wu=W13.at[slot, :, pl.ds(CI, CI)],
        sg=S13.at[slot, pl.ds(0, srows), pl.ds(0, CI)],
        su=S13.at[slot, pl.ds(0, srows), pl.ds(CI, CI)],
        w2=S["w2"].at[slot],
        s2=S["w2s"].at[slot, pl.ds(0, s2rows)],
    )


def _expert_loop(step: EPMoEStep, p: EPPlan, n_items) -> None:
    c, S = step.c, step.S
    He, NC, D, RI = c.expert_hidden, p.n_chunks(step.kind), p.nslot, p.item_rows
    CI = p.chunk_fp4 if step.kind == "fp4" else p.chunk_fp8
    sizes = item_sizes(RI)
    items = S["ep_items"]
    total = n_items * NC

    def item(k):
        i = k // NC
        return i, items[1 + 3 * i], items[2 + 3 * i], items[3 + 3 * i], k % NC

    def start(k):
        @pl.when(k < total)
        def _():
            _, e, _, _, j = item(k)
            step.for_layer(
                lambda refs, off: [
                    cp.start() for cp in _chunk_copies(step, p, refs, off, e, j, k % D)
                ]
            )

    def wait(slot):
        refs, off = step.wait_refs
        for cp in _chunk_copies(step, p, refs, off, 0, 0, slot):
            cp.wait()

    for q in range(D - 1):
        start(jnp.int32(q))

    @pl.when(n_items > 0)
    def _():
        dispatch.gather_x_start(c, S, 0, items[1], items[2], items[3])

    def body(k, carry):
        i, e, j0, n, j = item(k)
        b = i % 2  # x, accumulator and y staging alternate between items
        slot = k % D

        @pl.when(j == 0)
        def _():
            dispatch.gather_x_wait(S, b, n)

            @pl.when(i + 1 < n_items)
            def _():
                dispatch.gather_x_start(
                    c, S, 1 - b, items[4 + 3 * i], items[5 + 3 * i], items[6 + 3 * i]
                )

            def zero(z):
                S["ep_acc"][b, pl.ds(0, z), :] = jnp.zeros((z, He), jnp.float32)

            sized(n, sizes, zero)

        start(k + D - 1)
        wait(slot)
        chunk_compute(
            step.kind,
            n,
            lambda R: _unpack_bf16x2(S["ep_x"][b, pl.ds(0, R), :]).astype(jnp.bfloat16),
            S["ep_acc"].at[b],
            _chunk_refs(step, p, slot),
            He,
            CI,
            RI,
        )

        @pl.when(j == NC - 1)
        def _():
            dispatch.drain_y(S, b)

            def pack(z):
                S["ep_y"][b, pl.ds(0, z), :] = _pack_bf16x2(
                    S["ep_acc"][b, pl.ds(0, z), :]
                )

            sized(n, sizes, pack)
            dispatch.send_y(c, S, step.me, b, e, j0, n)

        return carry

    jax.lax.fori_loop(0, total, body, 0)


def _combine(step: EPMoEStep) -> None:
    """u = sum over a row's slots of weight * norm(post_expert_w * y), this rank's rows (the
    TP combine's math on ep_pad, where the y rows landed)."""
    c, S, smr, me, n_live = step.c, step.S, step.smr, step.me, step.n_live
    K, He, Rc, R = c.top_k, c.expert_hidden, c.rows_per_dev, c.rows
    f32, bf16 = jnp.float32, jnp.bfloat16
    pw = smr[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
    win = owner_window_rows(c)
    if win == Rc:
        w0, w_off = pl.multiple_of(me * Rc, 8), 0
    else:
        w0 = pl.multiple_of(jnp.minimum((me * Rc) // 8 * 8, R - win), 8)
        w_off = me * Rc - w0
    wmine = S["wsel_v"][pl.ds(w0, win), pl.ds(0, K)]  # [win, K]
    wk = jnp.concatenate([wmine[:, k : k + 1][:, :, None] for k in range(K)], axis=1)
    S["wk_s"][...] = jnp.broadcast_to(wk, (win, K, LANE))
    pad = S["ep_pad"]

    def row(r, carry):
        ysum = _unpack_bf16x2(pad.at[r][...])  # [K, He]
        ysum = jnp.where(me * Rc + r < n_live, ysum, 0.0).astype(bf16).astype(f32)
        ysc = ysum * pw
        nrm = jax.lax.rsqrt(
            jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
        )
        ynorm = (ysc * nrm).astype(bf16).astype(f32)
        u = jnp.sum(ynorm * S["wk_s"][w_off + r, :, 0:1], axis=0, keepdims=True)
        S["u_send"][pl.ds(r, 1), :] = _pack_bf16x2(u)
        return carry

    jax.lax.fori_loop(0, Rc, row, 0)
