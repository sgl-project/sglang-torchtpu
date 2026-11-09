# SPDX-License-Identifier: Apache-2.0
"""One layer's EP FFN section (the row-tiled megakernel's hook), on this rank's own rows: the
router's top-k, the replicated pool, the dispatch all-to-all of every slot's x to its
expert's rank, 32 whole experts streamed in I-chunks through the TP expert ring, the
all-to-all of y back, the weighted sum, the replicated unpool and the FFN residual, then one
all-gather of the rows for the next layer's attention.

The row window: own rows [me * Rc, (me + 1) * Rc) are read and computed through the 16-row
aligned window around them (window_rows); only own rows are written or sent.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk import a2a, route
from sglang_tpu.muse_megakernel.ep_mk.config import (
    EPPlan,
    ep_buffers,
    ep_plan,
    ep_scratch,
    window_rows,
)
from sglang_tpu.muse_megakernel.ep_mk.dense import stream_matmul
from sglang_tpu.muse_megakernel.ep_mk.experts import (
    ChunkRefs,
    chunk_compute,
    item_sizes,
    sized,
)
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import FP4_GROUP, FP4_PACK, SMALL
from sglang_tpu.muse_megakernel.mk.hooks import EPLayerStep
from sglang_tpu.muse_megakernel.mk.numerics import _rms
from sglang_tpu.muse_megakernel.mk.packing import _pack_bf16x2, _unpack_bf16x2

f32, bf16 = jnp.float32, jnp.bfloat16


class EPExperts:
    """EPHook: sequence-parallel FFN with 32 whole experts per rank (ep_plan)."""

    def __init__(self, cfg: MKConfig, item_rows: int = 32):
        self.plan = ep_plan(cfg, item_rows)

    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        return plan + ep_scratch(cfg, self.plan)

    def layer_moe(self, step: EPLayerStep) -> None:
        c, S, p, me = step.c, step.S, self.plan, step.me
        win = _window(c, me)
        step.router()
        route.own_topk(c, p, S, step.smr, win)
        route.send_places(c, p, S, step.n_live, win)
        _pool(step, win)
        _fill_send(step, p, win)
        a2a.exchange_counts(c, S, me)
        route.slot_matrix(c, S)
        a2a.dispatch(c, S, me)
        n_items = route.build_items(c, p, S, me)
        a2a.wait_dispatch(c, S, me)
        _expert_loop(step, p, n_items)
        a2a.give_back(c, S, me)
        a2a.wait_give_back(c, S, me)
        _combine(step, win)
        _unpool_residual(step, win)
        _all_gather_rows(step, p)


def _window(c: MKConfig, me):
    """(w0, win, w_off): the 16-row aligned window holding own rows at [w_off, w_off + Rc)."""
    rc, win = c.rows_per_dev, window_rows(c)
    if win == rc:
        return pl.multiple_of(me * rc, 16), win, 0
    w0 = pl.multiple_of(jnp.minimum((me * rc) // 16 * 16, c.rows - win), 16)
    return w0, win, me * rc - w0


def _row(smr, name, n):
    return smr[SMALL[name] : SMALL[name] + 1, :n]


def _pool(step: EPLayerStep, win) -> None:
    """ep_xo: the window rows' pooled, pre-expert-normed bf16 expert input (pairs)."""
    c, S, smr = step.c, step.S, step.smr
    H, He = c.hidden, c.expert_hidden
    w0, wn, _ = win

    def put(n0, y):
        S["ep_u"][:, pl.ds(n0, y.shape[1])] = y.astype(bf16).astype(f32)

    stream_matmul(
        c,
        S,
        step.W["wpool"],
        step.layer,
        lambda k0, n: S["lhs"][pl.ds(w0, wn), pl.ds(k0, n)],
        _row(smr, "wpool_s", He),
        H,
        He,
        put,
    )
    x = _rms(S["ep_u"][...], c.rms_eps, _row(smr, "pre_expert_w", He)).astype(bf16)
    S["ep_xo"][...] = _pack_bf16x2(x.astype(f32))


def _fill_send(step: EPLayerStep, p: EPPlan, win) -> None:
    """ep_xs: every own live slot's x row at its place (route.send_places)."""
    c, S = step.c, step.S
    K, rc = c.top_k, p.rows_own
    w0, _, w_off = win
    sem = S["ep_misc_sems"].at[0]

    def row(r, n):
        wr = w_off + r
        live = w0 + wr < step.n_live

        @pl.when(live)
        def _():
            for k in range(K):
                pltpu.make_async_copy(
                    S["ep_xo"].at[pl.ds(wr, 1)],
                    S["ep_xs"].at[pl.ds(S["ep_pos_s"][wr, k], 1)],
                    sem,
                ).start()

        return n + jnp.where(live, K, 0)

    n = jax.lax.fori_loop(0, rc, row, jnp.int32(0))
    tmpl = pltpu.make_async_copy(
        S["ep_xo"].at[pl.ds(0, 1)], S["ep_xs"].at[pl.ds(0, 1)], sem
    )

    def wait(i, carry):
        tmpl.wait()
        return carry

    jax.lax.fori_loop(0, n, wait, 0)


def _chunk_copies(step: EPLayerStep, p: EPPlan, refs, off, e, j, slot) -> list:
    """The six DMAs of expert e's I-chunk j into ring slot `slot`."""
    c, S, kind = step.c, step.S, step.kind
    He = c.expert_hidden
    CI = p.chunk(kind)
    pack = FP4_PACK if kind == "fp4" else 4
    W13, S13, W2 = _ring(step, p)
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
        srows = He // FP4_GROUP
        s2_src = s2.at[e, pl.ds(j * (CI // FP4_GROUP), CI // FP4_GROUP)]
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
            w2.at[e, pl.ds(k2, CI // pack)], W2.at[slot], sems.at[slot, 4]
        ),
        pltpu.make_async_copy(s2_src, s2_dst, sems.at[slot, 5]),
    ]


def _ring(step: EPLayerStep, p: EPPlan):
    """The TP expert ring as I-chunk slots: (w13, w13 scales, w2) of this layer's kind; an fp8
    chunk spans two fp4 slots, as TP's fp8 slabs do."""
    S, He, CI = step.S, step.c.expert_hidden, p.chunk(step.kind)
    if step.kind == "fp4":
        return S["w13"], S["w13s"], S["w2"]
    n = p.slots("fp8")
    return (
        S["w13"].reshape(n, He // 4, 2 * CI),
        S["w13s"],
        S["w2"].reshape(n, CI // 4, He),
    )


def _chunk_refs(step: EPLayerStep, p: EPPlan, slot) -> ChunkRefs:
    S, kind = step.S, step.kind
    W13, S13, W2 = _ring(step, p)
    CI = p.chunk(kind)
    if kind == "fp8":  # per-column scales: the arm reads row 0 of the whole slabs
        sg = su = S13.at[slot]
        s2 = S["w2s"].at[slot]
    else:
        sg, su = S13.at[slot, :, pl.ds(0, CI)], S13.at[slot, :, pl.ds(CI, CI)]
        s2 = S["w2s"].at[slot, pl.ds(0, CI // FP4_GROUP)]
    return ChunkRefs(
        wg=W13.at[slot, :, pl.ds(0, CI)],
        wu=W13.at[slot, :, pl.ds(CI, CI)],
        sg=sg,
        su=su,
        w2=W2.at[slot],
        s2=s2,
    )


def _item_rows(S: dict, buf, b, first, n, start: bool) -> None:
    """Starts (or waits for) the item's n x-row copies from ep_xr into buf.x(b)."""
    sem = S["ep_x_sems"].at[b]

    def one(t, carry):
        src = S["ep_xr"].at[pl.ds(S["ep_rows"][first + t], 1)]
        cp = pltpu.make_async_copy(src, buf.x(b).at[pl.ds(t, 1)], sem)
        if start:
            cp.start()
        else:
            cp.wait()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


def _expert_loop(step: EPLayerStep, p: EPPlan, n_items) -> None:
    """Every received slot's expert output into ep_yr at its received row."""
    c, S = step.c, step.S
    He, NC, RI = c.expert_hidden, p.n_chunks(step.kind), p.item_rows
    D, CI = p.slots(step.kind), p.chunk(step.kind)
    sizes = item_sizes(RI)
    items = S["ep_items"]
    buf = ep_buffers(c, p, S)
    total = n_items * NC
    ysem = S["ep_misc_sems"].at[1]

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
        _item_rows(S, buf, 0, items[2], items[3], start=True)

    def body(k, carry):
        i, _, first, n, j = item(k)
        b = i % 2  # x rows alternate between items
        slot = k % D

        @pl.when(j == 0)
        def _():
            _item_rows(S, buf, b, first, n, start=False)

            @pl.when(i + 1 < n_items)
            def _():
                _item_rows(
                    S, buf, 1 - b, items[5 + 3 * i], items[6 + 3 * i], start=True
                )

            def zero(z):
                buf.acc[pl.ds(0, z), :] = jnp.zeros((z, He), f32)

            sized(n, sizes, zero)

        start(k + D - 1)
        wait(slot)
        chunk_compute(
            step.kind,
            n,
            lambda R: _unpack_bf16x2(buf.x(b)[pl.ds(0, R), :]),
            buf.acc,
            _chunk_refs(step, p, slot),
            He,
            CI,
            RI,
        )

        @pl.when(j == NC - 1)
        def _():
            # the previous item's y rows have landed: buf.y is free
            tmpl = pltpu.make_async_copy(
                buf.y.at[pl.ds(0, 1)], S["ep_yr"].at[pl.ds(0, 1)], ysem
            )

            @pl.when(i > 0)
            def _():
                def w(t, carry2):
                    tmpl.wait()
                    return carry2

                jax.lax.fori_loop(0, items[3 * i], w, 0)

            def pack(z):
                buf.y[pl.ds(0, z), :] = _pack_bf16x2(buf.acc[pl.ds(0, z), :])

            sized(n, sizes, pack)

            def put(t, carry2):
                pltpu.make_async_copy(
                    buf.y.at[pl.ds(t, 1)],
                    S["ep_yr"].at[pl.ds(S["ep_rows"][first + t], 1)],
                    ysem,
                ).start()
                return carry2

            jax.lax.fori_loop(0, n, put, 0)

        return carry

    jax.lax.fori_loop(0, total, body, 0)

    @pl.when(n_items > 0)
    def _():
        tmpl = pltpu.make_async_copy(
            buf.y.at[pl.ds(0, 1)], S["ep_yr"].at[pl.ds(0, 1)], ysem
        )

        def w(t, carry):
            tmpl.wait()
            return carry

        jax.lax.fori_loop(0, items[3 * n_items], w, 0)


def _combine(step: EPLayerStep, win) -> None:
    """ep_u (own rows of the window): sum over a row's slots of weight * norm(post_expert_w *
    y), TP's combine math on the y rows back in ep_yb."""
    c, S, smr = step.c, step.S, step.smr
    K, He, rc = c.top_k, c.expert_hidden, c.rows_per_dev
    w0, _, w_off = win
    pw = _row(smr, "post_expert_w", He)
    sems = S["ep_yrow_sems"]

    def copies(r, b):
        wr = w_off + r
        return [
            pltpu.make_async_copy(
                S["ep_yb"].at[pl.ds(S["ep_pos_s"][wr, k], 1)],
                S["ep_yrow"].at[b, pl.ds(k, 1)],
                sems.at[b],
            )
            for k in range(K)
        ]

    def live(r):
        return w0 + w_off + r < step.n_live

    @pl.when(live(0))
    def _():
        for cp in copies(0, 0):
            cp.start()

    def row(r, carry):
        b = r % 2
        wr = w_off + r

        @pl.when((r + 1 < rc) & live(r + 1))
        def _():
            for cp in copies(r + 1, 1 - b):
                cp.start()

        @pl.when(live(r))
        def _():
            for cp in copies(r, b):
                cp.wait()

        ysum = _unpack_bf16x2(S["ep_yrow"][b])  # [K, He]
        ysum = jnp.where(live(r), ysum, 0.0).astype(bf16).astype(f32)
        ysc = ysum * pw
        nrm = jax.lax.rsqrt(
            jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
        )
        ynorm = (ysc * nrm).astype(bf16).astype(f32)
        wk = jnp.transpose(S["ep_wsel"][pl.ds(wr, 1), :])[:K]  # [K, 1]
        u = jnp.sum(ynorm * wk, axis=0, keepdims=True)
        S["ep_u"][pl.ds(wr, 1), :] = jnp.where(live(r), u, 0.0).astype(bf16).astype(f32)
        return carry

    jax.lax.fori_loop(0, rc, row, 0)


def _stream_rows(step: EPLayerStep):
    """(load(row0, n), store(row0, rows)) on S["h_stream"]: VMEM with tiled_vmem_pads, else
    HBM through the row-tile staging S["sin"][0] (n <= row_tile)."""
    S, hs = step.S, step.S["h_stream"]
    if step.c.tiled_vmem_pads:

        def load_v(r0, n):
            return hs[pl.ds(r0, n), :]

        def store_v(r0, v):
            hs[pl.ds(r0, v.shape[0]), :] = v

        return load_v, store_v
    stage, sem = S["sin"].at[0], S["ep_misc_sems"].at[3]

    def load(r0, n):
        cp = pltpu.make_async_copy(hs.at[pl.ds(r0, n)], stage.at[pl.ds(0, n)], sem)
        cp.start()
        cp.wait()
        return stage[pl.ds(0, n), :]

    def store(r0, v):
        n = v.shape[0]
        stage[pl.ds(0, n), :] = v
        cp = pltpu.make_async_copy(stage.at[pl.ds(0, n)], hs.at[pl.ds(r0, n)], sem)
        cp.start()
        cp.wait()

    return load, store


def _unpool_residual(step: EPLayerStep, win) -> None:
    """Own rows of S["h_stream"]: ffn_alpha * stream + ffn_beta * norm(unpool(u) * post_ffn_w)."""
    c, S, smr = step.c, step.S, step.smr
    H, He, rc = c.hidden, c.expert_hidden, c.rows_per_dev
    w0, wn, w_off = win

    def put(n0, y):
        S["ep_f"][pl.ds(0, wn), pl.ds(n0, y.shape[1])] = y.astype(bf16).astype(f32)

    stream_matmul(
        c,
        S,
        step.W["wunpool"],
        step.layer,
        lambda k0, n: S["ep_u"][:, pl.ds(k0, n)].astype(bf16),
        _row(smr, "wunpool_s", H),
        He,
        H,
        put,
    )
    fw = S["ep_f"][pl.ds(0, wn), :] * _row(smr, "post_ffn_w", H)
    S["ep_f"][pl.ds(0, wn), :] = _rms(fw, c.post_eps).astype(bf16).astype(f32)
    load, store = _stream_rows(step)
    alpha, beta = _row(smr, "ffn_alpha", H), _row(smr, "ffn_beta", H)
    for r in range(rc):  # own rows only: the all-gather fills the others
        wr = w_off + r
        branch = S["ep_f"][pl.ds(wr, 1), :]
        store(w0 + wr, alpha * load(w0 + wr, 1) + beta * branch)


def _all_gather_rows(step: EPLayerStep, p: EPPlan) -> None:
    """Own rows of S["h_stream"] to every rank (the next layer's attention reads them all),
    then the capture of every row."""
    c, S, me = step.c, step.S, step.me
    rc, RT, R = p.rows_own, c.row_tile, c.rows
    hs = S["h_stream"]
    recv, sent = S["ep_ag_sems"].at[0], S["ep_ag_sems"].at[1]
    parts = (
        [(pl.ds(pl.multiple_of(me * rc, 8), rc),)]
        if rc % 8 == 0
        else [(pl.ds(me * rc + r, 1),) for r in range(rc)]
    )
    for d in range(1, c.tp):
        peer = (me + d) % c.tp
        for (rows,) in parts:
            pltpu.make_async_remote_copy(
                src_ref=hs.at[rows],
                dst_ref=hs.at[rows],
                send_sem=sent,
                recv_sem=recv,
                device_id=(peer,),
            ).start()
    n = rc if rc % 8 == 0 else 1
    tmpl_r = pltpu.make_async_copy(hs.at[pl.ds(0, n)], hs.at[pl.ds(0, n)], recv)
    tmpl_s = pltpu.make_async_copy(hs.at[pl.ds(0, n)], hs.at[pl.ds(0, n)], sent)
    for _ in range((c.tp - 1) * len(parts)):
        tmpl_r.wait()
    for _ in range((c.tp - 1) * len(parts)):
        tmpl_s.wait()
    load, _ = _stream_rows(step)
    is_cap = None
    for cl in c.capture_layers:
        hit = step.layer == cl
        is_cap = hit if is_cap is None else (is_cap | hit)
    if is_cap is None:
        return

    @pl.when(is_cap)
    def _():
        for t in range(R // RT):
            step.capture(pl.ds(t * RT, RT), load(t * RT, RT))
