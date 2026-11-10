# SPDX-License-Identifier: Apache-2.0
"""One layer's EP FFN section (the row-tiled megakernel's hook), on this rank's own rows: the
router's top-k, the replicated pool, the dispatch all-to-all of every slot's x to its
expert's rank, 32 whole experts streamed in I-chunks through the TP expert ring, the
all-to-all of y back, the weighted sum, the replicated unpool and the FFN residual, then one
all-gather of the rows for the next layer's attention.

The row window: own rows [me * Rc, (me + 1) * Rc) are read and computed through the 16-row
aligned window around them (window_rows); only own rows are written or sent.
"""

import contextlib
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk import a2a, attn, route
from sglang_tpu.muse_megakernel.ep_mk.config import (
    EPPlan,
    ep_buffers,
    ep_plan,
    ep_scratch,
)
from sglang_tpu.muse_megakernel.ep_mk.dense import stream_matmul
from sglang_tpu.muse_megakernel.ep_mk.experts import (
    ChunkRefs,
    chunk_compute,
    item_sizes,
    qblock,
    qblock_s2_rows,
    scale_group,
    sized,
)
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import FP4_GROUP, FP4_PACK, SMALL
from sglang_tpu.muse_megakernel.mk.hooks import EPLayerStep, EPOProjStep
from sglang_tpu.muse_megakernel.mk.numerics import _rms
from sglang_tpu.muse_megakernel.mk.packing import _pack_bf16x2, _unpack_bf16x2

f32, bf16 = jnp.float32, jnp.bfloat16


def _phase(name: str, level: int = 1):
    """A named in-kernel region per phase with SGLANG_TPU_EP_TRACE >= level at trace time
    (xprof shows them with libtpu's --xla_enable_custom_call_region_trace=true); level 2:
    per expert chunk too."""
    if int(os.environ.get("SGLANG_TPU_EP_TRACE", "0")) >= level:
        return jax.named_scope(f"ep_{name}")
    return contextlib.nullcontext()


def _phase_fn(name: str, level: int = 1):
    """fn run inside _phase(name, level)."""

    def deco(fn):
        def run():
            with _phase(name, level):
                fn()

        return run

    return deco


def _dev_skip(phase: str) -> bool:
    """Timing only (wrong results): SGLANG_TPU_EP_DEV_SKIP lists phases every rank skips:
    residual (own rows' FFN residual), capture, allgather, experts (the math; slots move),
    lhs_f32 (the next layer's input as the f32 stream all-gather, as before),
    wwait (the in-loop waits for the expert weight streams: the math races them),
    wdma (the expert weight streams after the first ring fill: the math reruns those slots),
    pool, unpool (their weight streams), a2a (counts, both all-to-alls: no slot moves).
    Read at trace time: a warm compile process imports this module before a sweep's env."""
    return phase in os.environ.get("SGLANG_TPU_EP_DEV_SKIP", "").split(",")


class EPExperts:
    """EPHook: sequence-parallel FFN with 32 whole experts per rank (ep_plan)."""

    def __init__(self, cfg: MKConfig, item_rows: int = 32):
        self.plan = ep_plan(cfg, item_rows)

    def o_proj(self, step: EPOProjStep) -> None:
        attn.o_proj(step, self.plan, _stream_rows(step.c, step.S))

    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        # TP's expert section's unused buffers leave the plan (SMEM especially: past its
        # capacity the compiler spills scalars into HBM)
        unused = {
            "route_v",
            "route_s",
            "ids_s",
            "ystage",
            "ystage_sems",
            "yrow",
            "yrow_sems",
        }
        return [e for e in plan if e[0] not in unused] + ep_scratch(cfg, self.plan)

    def layer_moe(self, step: EPLayerStep) -> None:
        c, S, p, me = step.c, step.S, self.plan, step.me
        win = _window(c, me)
        with _phase("router"):
            step.router()
            route.own_topk(c, p, S, step.smr, win)
            route.send_places(c, p, S, step.n_live, win)
        with _phase("pool"):
            _pool(step, win)
        with _phase("fill"):
            _fill_send(step, p, win)
        moving = not _dev_skip("a2a")
        with _phase("counts"):
            if moving:
                a2a.exchange_counts(c, S, me)
            else:
                S["ep_cnt_v"][...] = jnp.zeros(S["ep_cnt_v"].shape, jnp.int32)
            route.slot_tables(c, p, S, me)
        with _phase("dispatch"):
            if moving:
                a2a.dispatch(c, S, me)
        with _phase("items"):
            n_items = route.build_items(c, p, S)

        def dispatched():
            with _phase("dispatch_wait"):
                if moving:
                    a2a.wait_dispatch(c, S, me)

        with _phase("experts"):
            _expert_loop(step, p, n_items, dispatched)
        with _phase("combine_a2a"):
            if moving:
                a2a.wait_give_back(c, S, me)
        with _phase("combine"):
            _combine(step, win)
        with _phase("unpool"):
            _unpool_residual(step, win)
        with _phase("allgather"):
            if step.next_smalls is not None and not _dev_skip("lhs_f32"):
                _gather_lhs(step, p, step.next_smalls())
                _all_gather_rows(step, p, only_captured=True)
            else:
                _all_gather_rows(step, p)

    def pre_attn(self, c: MKConfig, S: dict) -> None:
        _pre_attn_lhs(c, S)


_window = attn.window


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
        skip=_dev_skip("pool"),
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
                    S["ep_xs"].at[pl.ds(S["ep_pos_s"][k, wr], 1)],
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


def _divmod(k: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """k // n, k % n of a k >= 0; a shift and a mask for a power of two (floor division of a
    signed index costs sign fix-ups in the scalar code)."""
    if n & (n - 1):
        return k // n, k % n
    return k >> (n.bit_length() - 1), k & (n - 1)


def _chunk_copies(step: EPLayerStep, p: EPPlan, refs, off, e, j, slot) -> list:
    """The DMAs of expert e's I-chunk j into ring slot `slot` (w13 and its scales in
    chunk_cols order: a chunk's gate | up columns are one slab); three under
    SGLANG_TPU_EP_QBLOCK, whose w13 scale slab carries the w2 scales."""
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
    c0 = pl.multiple_of(j * 2 * CI, 2 * CI)
    k2 = pl.multiple_of(j * (CI // pack), CI // pack)
    q = kind == "fp4" and qblock()
    if kind == "fp4":
        g = scale_group(kind)
        srows = He // g + (qblock_s2_rows(He, g) if q else 0)
        s2_src = s2.at[e, pl.ds(j * (CI // g), CI // g)]
        s2_dst = S["w2s"].at[slot, pl.ds(0, CI // g)]
    else:
        srows, s2_src = 2, s2.at[e]
        s2_dst = S["w2s"].at[slot, pl.ds(0, 2)]
    copies = [
        pltpu.make_async_copy(
            w13.at[e, :, pl.ds(c0, 2 * CI)],
            W13.at[slot, :, pl.ds(0, 2 * CI)],
            sems.at[slot, 0],
        ),
        pltpu.make_async_copy(
            s13.at[e, pl.ds(0, srows), pl.ds(c0, 2 * CI)],
            S13.at[slot, pl.ds(0, srows), pl.ds(0, 2 * CI)],
            sems.at[slot, 1],
        ),
        pltpu.make_async_copy(
            w2.at[e, pl.ds(k2, CI // pack)], W2.at[slot], sems.at[slot, 2]
        ),
    ]
    if q:  # the w2 scales came with the w13 scale slab
        return copies
    return copies + [pltpu.make_async_copy(s2_src, s2_dst, sems.at[slot, 3])]


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
        # the q arms read the w2 scales from the whole w13 scale slab (qblock_scale_slab)
        s2 = S13.at[slot] if qblock() else S["w2s"].at[slot, pl.ds(0, CI // FP4_GROUP)]
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
        src = S["ep_xr"].at[pl.ds(S["ep_rows"][first + t] & (2**route.ROW_BITS - 1), 1)]
        cp = pltpu.make_async_copy(src, buf.x(b).at[pl.ds(t, 1)], sem)
        if start:
            cp.start()
        else:
            cp.wait()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


def _expert_loop(step: EPLayerStep, p: EPPlan, n_items, dispatched) -> None:
    """Every received slot's expert output back to its source's ep_yb row, item by item;
    dispatched() waits for the slots (the first weight chunks are in flight by then). The
    loop runs inside the layer's for_layer leaf: a chunk's copies take no branch-tree walk."""
    step.for_layer(
        lambda refs, off: _expert_loop_on(step, p, n_items, dispatched, refs, off)
    )


def _expert_loop_on(
    step: EPLayerStep, p: EPPlan, n_items, dispatched, refs, off
) -> None:
    c, S, me = step.c, step.S, step.me
    He, NC, RI = c.expert_hidden, p.n_chunks(step.kind), p.item_rows
    D, CI = p.slots(step.kind), p.chunk(step.kind)
    sizes = item_sizes(RI)
    items = S["ep_items"]
    buf = ep_buffers(c, p, S)
    total = n_items * NC
    ysem = S["ep_misc_sems"].at[1]

    def item(k):
        i, j = _divmod(k, NC)
        return i, items[1 + 3 * i], items[2 + 3 * i], items[3 + 3 * i], j

    def start(k):
        @pl.when(k < (jnp.minimum(total, D) if _dev_skip("wdma") else total))
        def _():
            _, e, _, _, j = item(k)
            for cp in _chunk_copies(step, p, refs, off, e, j, _divmod(k, D)[1]):
                cp.start()

    def wait(slot, k):
        wrefs, woff = step.wait_refs

        def w():
            for cp in _chunk_copies(step, p, wrefs, woff, 0, 0, slot):
                cp.wait()

        if _dev_skip("wdma") or _dev_skip("wwait"):
            pl.when(k < D)(w)
        else:
            w()

    for q in range(D - 1):
        start(jnp.int32(q))
    dispatched()

    @pl.when(n_items > 0)
    def _():
        _item_rows(S, buf, 0, items[2], items[3], start=True)

    def body(k, carry):
        i, _, first, n, j = item(k)
        b = i % 2  # x rows alternate between items
        slot = _divmod(k, D)[1]

        @pl.when(j == 0)
        @_phase_fn("item_in", 2)
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

        with _phase("issue", 2):
            start(k + D - 1)
            wait(slot, k)
        if not _dev_skip("experts"):
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
        @_phase_fn("item_out", 2)
        def _():
            # the previous item's y rows have landed: buf.y is free
            tmpl = pltpu.make_async_copy(
                buf.y.at[pl.ds(0, 1)], S["ep_yb"].at[pl.ds(0, 1)], ysem
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
                # straight back to the slot's source: its return buffer row
                v = S["ep_rows"][first + t]
                src = (v >> route.ROW_BITS) & (2**route.SRC_BITS - 1)
                dst = S["ep_yb"].at[pl.ds(v >> (route.ROW_BITS + route.SRC_BITS), 1)]

                @pl.when(src == me)
                def _():
                    pltpu.make_async_copy(buf.y.at[pl.ds(t, 1)], dst, ysem).start()

                @pl.when(src != me)
                def _():
                    pltpu.make_async_remote_copy(
                        src_ref=buf.y.at[pl.ds(t, 1)],
                        dst_ref=dst,
                        send_sem=ysem,
                        recv_sem=S["ep_a2a_sems"].at[2],
                        device_id=(src,),
                    ).start()

                return carry2

            jax.lax.fori_loop(0, n, put, 0)

        return carry

    jax.lax.fori_loop(0, total, body, 0)
    if _dev_skip("wwait"):  # the skipped waits, after the loop

        def late(k, carry):
            wait(k % D, jnp.int32(0))
            return carry

        jax.lax.fori_loop(D, jnp.maximum(total, D), late, 0)

    @pl.when(n_items > 0)
    def _():
        tmpl = pltpu.make_async_copy(
            buf.y.at[pl.ds(0, 1)], S["ep_yb"].at[pl.ds(0, 1)], ysem
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
                S["ep_yb"].at[pl.ds(S["ep_pos_s"][k, wr], 1)],
                S["ep_yrow"].at[b, pl.ds(k, 1)],
                sems.at[b],
            )
            for k in range(K)
        ]

    def live(r):
        return w0 + w_off + r < step.n_live

    Q = S["ep_yrow"].shape[0]  # rows in flight: the gathers' latency, not their bytes

    for q in range(min(Q - 1, rc)):

        @pl.when(live(q))
        def _(q=q):
            for cp in copies(q, q):
                cp.start()

    def row(r, carry):
        b = r % Q
        wr = w_off + r

        @pl.when((r + Q - 1 < rc) & live(r + Q - 1))
        def _():
            for cp in copies(r + Q - 1, (r + Q - 1) % Q):
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


def _stream_rows(c: MKConfig, S: dict):
    """(load(row0, n), rows(row0, n, fn)) on S["h_stream"]: VMEM with tiled_vmem_pads, else
    HBM through the row-tile staging S["sin"][0]. load: n <= row_tile rows as a value;
    rows: fn(row, t0, m) on each tile of rows [row0 + t0, row0 + t0 + m), row(i) the ref of
    its row i, written back unless write_back=False. HBM: one copy each way per tile when row0
    is 8-row aligned (n % 8 == 0: row0 = rank * n), else one per row, all in flight (an
    aligned tile's write-back of another rank's row would race with its all-gather)."""
    hs, RT = S["h_stream"], c.row_tile
    if c.tiled_vmem_pads:

        def load_v(r0, n):
            return hs[pl.ds(r0, n), :]

        def rows_v(r0, n, fn, write_back=True):
            for t0 in range(0, n, RT):
                fn(lambda i, t0=t0: hs.at[pl.ds(r0 + t0 + i, 1)], t0, min(RT, n - t0))

        return load_v, rows_v
    stage, sem = S["sin"].at[0], S["ep_misc_sems"].at[3]

    def copy(src, dst):
        cp = pltpu.make_async_copy(src, dst, sem)
        cp.start()
        cp.wait()

    def load(r0, n):
        copy(hs.at[pl.ds(r0, n)], stage.at[pl.ds(0, n)])
        return stage[pl.ds(0, n), :]

    def rows(r0, n, fn, write_back=True):
        aligned = n % 8 == 0
        if aligned:
            r0 = pl.multiple_of(r0, 8)
        for t0 in range(0, n, RT):
            m = min(RT, n - t0)
            # one copy per tile, or per row in flight together
            pairs = (
                [(pl.ds(r0 + t0, m), pl.ds(0, m))]
                if aligned
                else [(pl.ds(r0 + t0 + i, 1), pl.ds(i, 1)) for i in range(m)]
            )
            cps = [pltpu.make_async_copy(hs.at[a], stage.at[b], sem) for a, b in pairs]
            for cp in cps:
                cp.start()
            for cp in cps:
                cp.wait()
            fn(lambda i: stage.at[pl.ds(i, 1)], t0, m)
            if write_back:
                cps = [
                    pltpu.make_async_copy(stage.at[b], hs.at[a], sem) for a, b in pairs
                ]
                for cp in cps:
                    cp.start()
                for cp in cps:
                    cp.wait()

    return load, rows


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
        skip=_dev_skip("unpool"),
    )
    fw = S["ep_f"][pl.ds(0, wn), :] * _row(smr, "post_ffn_w", H)
    S["ep_f"][pl.ds(0, wn), :] = _rms(fw, c.post_eps).astype(bf16).astype(f32)
    _, rows = _stream_rows(c, S)
    alpha, beta = _row(smr, "ffn_alpha", H), _row(smr, "ffn_beta", H)

    def residual(row, t0, m):
        for i in range(m):
            branch = S["ep_f"][pl.ds(w_off + t0 + i, 1), :]
            row(i)[...] = alpha * row(i)[...] + beta * branch

    if not _dev_skip("residual"):  # own rows only: the all-gather fills the others
        rows(w0 + w_off, rc, residual)


def _gather_lhs(step: EPLayerStep, p: EPPlan, nsmr) -> None:
    """Own rows' next-layer pre-attention input (rms with its pre_attn_w, bf16 pairs) into
    every rank's ep_lhs_g: half the bytes of the f32 stream rows, which other ranks no longer
    read (o_proj and the FFN section run on own rows)."""
    c, S, me = step.c, step.S, step.me
    rc, H = p.rows_own, c.hidden
    # own rows staged in ep_f (free after the residual)
    own = S["ep_f"].bitcast(jnp.uint32).at[:, pl.ds(0, H // 2)]
    g, sems, rsems = S["ep_lhs_g"], S["ep_lhs_sems"], S["ep_lhs_rsems"]
    w = nsmr[SMALL["pre_attn_w"] : SMALL["pre_attn_w"] + 1, :H]
    _, rows = _stream_rows(c, S)

    def norm(row, t0, m):
        for i in range(m):
            own[pl.ds(t0 + i, 1), :] = _pack_bf16x2(
                _rms(row(i)[...], c.rms_eps, w).astype(bf16).astype(f32)
            )

    rows(me * rc, rc, norm, write_back=False)
    n = rc if rc % 8 == 0 else 1  # rows per copy: a block, or a row of an unaligned one

    def block(k):
        """Per copy: (rows in the [rc]-row own buffer, owner k's rows in ep_lhs_g)."""
        if rc % 8 == 0:
            return [(pl.ds(0, rc), pl.ds(pl.multiple_of(k * rc, 8), rc))]
        return [(pl.ds(r, 1), pl.ds(k * rc + r, 1)) for r in range(rc)]

    def send(src_buf, k, peer):  # owner k's rows from src_buf to peer's ep_lhs_g
        for src, dst in block(k):
            sr = src_buf.at[src] if src_buf is own else src_buf.at[dst]
            pltpu.make_async_remote_copy(
                src_ref=sr,
                dst_ref=g.at[dst],
                send_sem=sems.at[1],
                recv_sem=rsems.at[k],
                device_id=(peer,),
            ).start()

    def wait(sem, copies):
        tmpl = pltpu.make_async_copy(own.at[pl.ds(0, n)], g.at[pl.ds(0, n)], sem)
        for _ in range(copies):
            tmpl.wait()

    for src, dst in block(me):
        pltpu.make_async_copy(own.at[src], g.at[dst], sems.at[2]).start()
    if c.tp != 8:  # every peer directly
        for d in range(1, c.tp):
            send(own, me, (me + d) % c.tp)
        for d in range(1, c.tp):
            wait(rsems.at[(me + d) % c.tp], len(block(0)))
        wait(sems.at[1], (c.tp - 1) * len(block(0)))
    else:
        # 4 chips x 2 cores: once to the same core of each other chip and to the partner
        # core, which relays the other chips' blocks on-chip (half the ICI bytes)
        core, chip, partner = me % 2, me // 2, jnp.bitwise_xor(me, 1)
        send(own, me, partner)
        for d in (1, 2, 3):
            send(own, me, 2 * jnp.bitwise_xor(chip, d) + core)
        for d in (1, 2, 3):
            k = 2 * jnp.bitwise_xor(chip, d) + core
            wait(rsems.at[k], len(block(0)))
            send(g, k, partner)
        wait(rsems.at[partner], len(block(0)))
        for d in (1, 2, 3):
            wait(rsems.at[2 * jnp.bitwise_xor(chip, d) + 1 - core], len(block(0)))
        wait(sems.at[1], 7 * len(block(0)))
    wait(sems.at[2], len(block(0)))


def _pre_attn_lhs(c: MKConfig, S: dict) -> None:
    """S["lhs"] = every row of ep_lhs_g as bf16, row tile by row tile (double-buffered)."""
    RT, R, Hh = c.row_tile, c.rows, c.hidden // 2
    g, sems = S["ep_lhs_g"], S["ep_lhs_sems"]
    st = S["ep_lhs_stage"] if c.tiled_vmem_pads else S["sin"].at[0].bitcast(jnp.uint32)

    def stage(b):  # two slots of [RT, H / 2] words side by side
        return st.at[:, pl.ds(b * Hh, Hh)]

    def cp(t):
        return pltpu.make_async_copy(
            g.at[pl.ds(t * RT, RT)], stage(t % 2), sems.at[3 + t % 2]
        )

    cp(0).start()
    for t in range(R // RT):
        if t + 1 < R // RT:
            cp(t + 1).start()
        cp(t).wait()
        S["lhs"][pl.ds(t * RT, RT), :] = _unpack_bf16x2(stage(t % 2)[...]).astype(bf16)


def _all_gather_rows(step: EPLayerStep, p: EPPlan, only_captured: bool = False) -> None:
    """Own rows of S["h_stream"] to every rank, then the capture of every row: on every layer,
    or only_captured on the capture layers (others read only the gathered next-layer input)."""
    c, S, me = step.c, step.S, step.me
    rc, RT, R = p.rows_own, c.row_tile, c.rows
    hs = S["h_stream"]
    recv, sent = S["ep_ag_sems"].at[0], S["ep_ag_sems"].at[1]
    parts = (
        [(pl.ds(pl.multiple_of(me * rc, 8), rc),)]
        if rc % 8 == 0
        else [(pl.ds(me * rc + r, 1),) for r in range(rc)]
    )
    if _dev_skip("allgather"):
        parts = []

    def gather():
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

    load, _ = _stream_rows(c, S)

    def capture():
        for t in range(R // RT):
            step.capture(pl.ds(t * RT, RT), load(t * RT, RT))

    if isinstance(step.layer, int):
        is_cap = step.layer in c.capture_layers
        if is_cap or not only_captured:
            gather()
        if is_cap and not _dev_skip("capture"):
            capture()
        return
    is_cap = None
    for cl in c.capture_layers:
        hit = step.layer == cl
        is_cap = hit if is_cap is None else (is_cap | hit)
    if not only_captured:
        gather()
    if is_cap is None:
        return

    @pl.when(is_cap)
    def _():
        if only_captured:
            gather()
        if not _dev_skip("capture"):
            capture()
