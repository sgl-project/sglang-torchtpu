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
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel.common import a2a
from sglang_tpu.muse_megakernel.common.config import MKConfig
from sglang_tpu.muse_megakernel.common.consts import (
    FP4_GROUP,
    FP4_PACK,
    SMALL,
    small_row,
)
from sglang_tpu.muse_megakernel.common.ep_layout import qblock
from sglang_tpu.muse_megakernel.common.hooks import EPLayerStep, EPOProjStep
from sglang_tpu.muse_megakernel.common.numerics import _rms
from sglang_tpu.muse_megakernel.common.packing import _pack_bf16x2, _unpack_bf16x2
from sglang_tpu.muse_megakernel.decode.spep import attn, route
from sglang_tpu.muse_megakernel.decode.spep.config import (
    DN_TN,
    TP_SCRATCH_UNUSED,
    EPPlan,
    col_pool,
    ep_buffers,
    ep_plan,
    ep_scratch,
)
from sglang_tpu.muse_megakernel.decode.spep.dense import stream_matmul, stream_prefetch
from sglang_tpu.muse_megakernel.decode.spep.experts import (
    ChunkRefs,
    arm_index,
    chunk_arms,
    item_sizes,
    scale_group,
    sized,
)

f32, bf16 = jnp.float32, jnp.bfloat16


def _phase(name: str, level: int = 1):
    """A named in-kernel region per phase with SGLANG_TPU_MK_TRACE_REGIONS >= level at trace
    time (xprof shows them with libtpu's --xla_enable_custom_call_region_trace=true); level 2:
    per expert chunk too."""
    if tpu_envs.SGLANG_TPU_MK_TRACE_REGIONS.get() >= level:
        return jax.named_scope(f"ep_{name}")
    return contextlib.nullcontext()


class EPExperts:
    """EPHook: sequence-parallel FFN with 32 whole experts per rank (ep_plan)."""

    def __init__(self, cfg: MKConfig, item_rows: int = 32):
        self.plan = ep_plan(cfg, item_rows)

    def o_proj(self, step: EPOProjStep) -> None:
        attn.o_proj(step, self.plan, _stream_rows(step.c, step.S))

    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        return [e for e in plan if e[0] not in TP_SCRATCH_UNUSED] + ep_scratch(
            cfg, self.plan
        )

    def layer_moe(self, step: EPLayerStep) -> None:
        c, S, p, me = step.c, step.S, self.plan, step.me
        win = attn.window(c, me)
        with _phase("router"):
            step.router()
            route.own_topk(c, S, step.smr, win)
            route.send_places(c, p, S, step.n_live, win)
        with _phase("pool"):
            if col_pool(c):
                _pool_cols(step, p, win)
            else:
                _pool(step, win)
        with _phase("fill"):
            _fill_send(step, p, win)
        with _phase("counts"):
            a2a.mk_exchange_counts(c, S, me)
            route.slot_tables(c, p, S, me)
        with _phase("dispatch"):
            a2a.mk_dispatch(c, S, me)
        with _phase("items"):
            n_items = route.build_items(c, p, S)

        def dispatched():
            with _phase("dispatch_wait"):
                a2a.mk_wait_dispatch(c, S, me)

        with _phase("experts"):
            expert_loop(step, p, n_items, dispatched)
        # its first weight tiles stream under the y return
        if col_pool(c):
            hc = c.hidden // c.tp
            stream_prefetch(
                S, step.W["wunpool"], step.layer, c.expert_hidden, hc, me * hc, hc
            )
        else:
            stream_prefetch(
                S,
                step.W["wunpool"],
                step.layer,
                c.expert_hidden,
                c.hidden,
                0,
                DN_TN,
            )
        with _phase("combine_a2a"):
            a2a.mk_wait_give_back(c, S, me)
        with _phase("combine"):
            _combine(step, win)
        with _phase("unpool"):
            if col_pool(c):
                _unpool_cols(step, p, win)
            else:
                _unpool_rows(step, win)
            _ffn_residual(step, win)
        with _phase("allgather"):
            if step.next_smalls is not None:
                _gather_lhs(step, p, step.next_smalls())
                _all_gather_rows(step, p, only_captured=True)
            else:
                _all_gather_rows(step, p)

    def pre_attn(self, c: MKConfig, S: dict) -> None:
        with _phase("pre_attn"):
            _hbm_rows_to_lhs(c, S, S["ep_lhs_g"])  # every row of ep_lhs_g as bf16


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
        small_row(smr, "wpool_s", He),
        H,
        He,
        put,
    )
    x = _rms(S["ep_u"][...], c.rms_eps, small_row(smr, "pre_expert_w", He)).astype(bf16)
    S["ep_xo"][...] = _pack_bf16x2(x.astype(f32))


def _scale_slice(S: dict, smr, name: str, me, n: int) -> jax.Array:
    """[1, n] f32: columns [me * n, (me + 1) * n) of smalls row `name`."""
    sc = S["ep_csc"]
    cp = pltpu.make_async_copy(
        smr.at[pl.ds(SMALL[name], 1), pl.ds(pl.multiple_of(me * n, n), n)],
        sc.at[pl.ds(0, 1), pl.ds(0, n)],
        S["ep_cx_sems"].at[2],
    )
    cp.start()
    cp.wait()
    return sc[pl.ds(0, 1), pl.ds(0, n)]


def _stage_by_owner(stage, rc: int, y: jax.Array) -> None:
    """stage[k, :rc] = owner k's rows of y ([R, W] f32) as bf16 pairs."""
    v = _pack_bf16x2(y)
    for k in range(stage.shape[0]):
        stage[k, pl.ds(0, rc), :] = v[k * rc : (k + 1) * rc]


def _blocks_to_owners(c: MKConfig, me, stage, rcv, sems) -> None:
    """stage[k] (owner k's rows of this rank's column block, by _stage_by_owner) into rcv[me]
    on rank k, one tile-aligned slot per DMA (a row offset off the (8, 128) tile is not a
    valid DMA); returns once every rank's block of own rows is here. sems: [0] received,
    [1] sent."""
    pltpu.make_async_copy(stage.at[me], rcv.at[me], sems.at[0]).start()
    for d in range(1, c.tp):
        peer = (me + d) % c.tp
        pltpu.make_async_remote_copy(
            src_ref=stage.at[peer],
            dst_ref=rcv.at[me],
            send_sem=sems.at[1],
            recv_sem=sems.at[0],
            device_id=(peer,),
        ).start()
    for sem, k in ((sems.at[0], c.tp), (sems.at[1], c.tp - 1)):
        tmpl = pltpu.make_async_copy(stage.at[0], rcv.at[0], sem)
        for _ in range(k):
            tmpl.wait()


def _own_rows_of_blocks(c: MKConfig, rc: int, rcv) -> jax.Array:
    """[rc, tp * W] f32: own rows of every rank's column block, side by side."""
    return jnp.concatenate(
        [_unpack_bf16x2(rcv[d, pl.ds(0, rc)]) for d in range(c.tp)], axis=1
    )


def _store_own(ref, w_off, rc: int, v: jax.Array) -> None:
    """ref rows [w_off, w_off + rc) = v (an 8-row aligned block, else row by row)."""
    if rc % 8 == 0:
        ref[pl.ds(attn.aligned8(w_off), rc), :] = v
        return
    for r in range(rc):
        ref[pl.ds(w_off + r, 1), :] = v[r : r + 1]


def _pool_cols(step: EPLayerStep, p: EPPlan, win) -> None:
    """ep_xo own rows (bf16 pairs): pool on this rank's He / tp column block of every row
    (own pre-FFN rows all-gathered in), every rank's block of own rows back to here."""
    c, S, smr, me = step.c, step.S, step.smr, step.me
    H, He, rc = c.hidden, c.expert_hidden, p.rows_own
    W = He // c.tp
    w0, wn, w_off = win
    own = S["ep_f"].bitcast(jnp.uint32).at[:, pl.ds(0, H // 2)]  # free until the unpool
    own[pl.ds(0, wn), :] = _pack_bf16x2(S["lhs"][pl.ds(w0, wn), :].astype(f32))
    with _phase("pool_ag", 2):
        _bcast_own_rows(
            c, me, rc, own, w_off, S["ep_lhs_g"], S["ep_lhs_sems"], S["ep_lhs_rsems"]
        )
        _hbm_rows_to_lhs(c, S, S["ep_lhs_g"])
    stage = S["ep_pstage"]

    def put(n0, y):
        _stage_by_owner(stage, rc, y)

    stream_matmul(
        c,
        S,
        step.W["wpool"],
        step.layer,
        lambda k0, n: S["lhs"][:, pl.ds(k0, n)],
        _scale_slice(S, smr, "wpool_s", me, W),
        H,
        W,
        put,
        n_off=me * W,
        tn=W,
    )
    pre = small_row(smr, "pre_expert_w", He)
    with _phase("pool_x", 2):
        _blocks_to_owners(c, me, stage, S["ep_prcv"], S["ep_cx_sems"])
    x = _rms(_own_rows_of_blocks(c, rc, S["ep_prcv"]), c.rms_eps, pre).astype(bf16)
    _store_own(S["ep_xo"], w_off, rc, _pack_bf16x2(x.astype(f32)))


def _unpool_cols(step: EPLayerStep, p: EPPlan, win) -> None:
    """ep_f own rows: unpool on this rank's H / tp column block of every row (own rows of u
    all-gathered in), every rank's block of own rows back to here."""
    c, S, smr, me = step.c, step.S, step.smr, step.me
    H, He, rc = c.hidden, c.expert_hidden, p.rows_own
    Hc = H // c.tp
    _, wn, w_off = win
    own = S["ep_f"].bitcast(jnp.uint32).at[:, pl.ds(0, He // 2)]
    own[pl.ds(0, wn), :] = _pack_bf16x2(S["ep_u"][...])
    with _phase("unpool_ag", 2):
        _bcast_own_rows(
            c, me, rc, own, w_off, S["ep_ug"], S["ep_lhs_sems"], S["ep_lhs_rsems"]
        )
        _hbm_rows_to_lhs(c, S, S["ep_ug"])  # lhs[:, :He]: every row's u
    stage = S["ep_ustage"]

    def put(n0, y):
        _stage_by_owner(stage, rc, y)

    stream_matmul(
        c,
        S,
        step.W["wunpool"],
        step.layer,
        lambda k0, n: S["lhs"][:, pl.ds(k0, n)],
        _scale_slice(S, smr, "wunpool_s", me, Hc),
        He,
        Hc,
        put,
        n_off=me * Hc,
        tn=Hc,
        prefetched=True,
    )
    with _phase("unpool_x", 2):
        _blocks_to_owners(c, me, stage, S["ep_urcv"], S["ep_cx_sems"])
    f = _own_rows_of_blocks(c, rc, S["ep_urcv"])
    _store_own(S["ep_f"], w_off, rc, f.astype(bf16).astype(f32))


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
    """The four DMAs of expert e's I-chunk j into ring slot `slot` (w13 and its scales in
    chunk_cols order: a chunk's gate | up columns are one slab)."""
    c, S, kind = step.c, step.S, step.kind
    He = c.expert_hidden
    CI = p.chunk
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
    if kind == "fp4":
        g = scale_group(kind)
        srows = He // g
        s2_src = s2.at[e, pl.ds(j * (CI // g), CI // g)]
        s2_dst = S["w2s"].at[slot, pl.ds(0, CI // g)]
    else:
        srows, s2_src = 2, s2.at[e]
        s2_dst = S["w2s"].at[slot, pl.ds(0, 2)]
    return [
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
        pltpu.make_async_copy(s2_src, s2_dst, sems.at[slot, 3]),
    ]


def _ring(step: EPLayerStep, p: EPPlan):
    """The TP expert ring as I-chunk slots: (w13, w13 scales, w2) of this layer's kind; an fp8
    chunk spans two fp4 slots, as TP's fp8 slabs do."""
    S, He, CI = step.S, step.c.expert_hidden, p.chunk
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
    CI = p.chunk
    if kind == "fp8":  # per-column scales: the arm reads row 0 of the whole slabs
        sg = su = S13.at[slot]
        s2 = S["w2s"].at[slot]
    else:
        sg, su = S13.at[slot, :, pl.ds(0, CI)], S13.at[slot, :, pl.ds(CI, CI)]
        # the q arms read their rows of the whole slab (a short row slice won't lay out)
        s2 = (
            S["w2s"].at[slot]
            if qblock()
            else S["w2s"].at[slot, pl.ds(0, CI // FP4_GROUP)]
        )
    return ChunkRefs(
        wg=W13.at[slot, :, pl.ds(0, CI)],
        wu=W13.at[slot, :, pl.ds(CI, CI)],
        sg=sg,
        su=su,
        w2=W2.at[slot],
        s2=s2,
    )


def _item_rows(S: dict, buf, b, first, n, start: bool, src_buf: str) -> None:
    """Starts (or waits for) the item's n x-row copies from src_buf into buf.x(b)."""
    sem = S["ep_x_sems"].at[b]

    def one(t, carry):
        row = S["ep_rows"][first + t] & (2**route.ROW_BITS - 1)
        src = S[src_buf].at[pl.ds(row, 1)]
        cp = pltpu.make_async_copy(src, buf.x(b).at[pl.ds(t, 1)], sem)
        if start:
            cp.start()
        else:
            cp.wait()
        return carry

    jax.lax.fori_loop(0, n, one, 0)


# emit(i, first, n): an item's finished acc rows (its last chunk), in place of the y return
ItemOut = Callable[[Any, Any, Any], None]


def expert_loop(
    step: EPLayerStep,
    p: EPPlan,
    n_items,
    dispatched: Callable[[], None],
    x_src: str = "ep_xr",
    emit: ItemOut | None = None,
) -> None:
    """Every received slot's expert output back to its source's ep_yb row (or to emit), item
    by item; x rows come from x_src; dispatched() waits for the slots (the first weight
    chunks are in flight by then). The loop runs inside the layer's for_layer leaf: a chunk's
    copies take no branch-tree walk."""
    step.for_layer(
        lambda refs, off: _expert_loop_on(
            step, p, n_items, dispatched, refs, off, x_src, emit
        )
    )


def _expert_loop_on(
    step: EPLayerStep,
    p: EPPlan,
    n_items,
    dispatched: Callable[[], None],
    refs,
    off,
    x_src: str,
    emit: ItemOut | None,
) -> None:
    c, S, me = step.c, step.S, step.me
    He, NC, RI = c.expert_hidden, p.n_chunks, p.item_rows
    D, CI = p.slots(step.kind), p.chunk
    sizes = item_sizes(RI)
    items = S["ep_items"]
    buf = ep_buffers(c, p, S)
    total = n_items * NC
    ysem = S["ep_misc_sems"].at[1]

    def item(k):
        i, j = _divmod(k, NC)
        return i, items[1 + 3 * i], items[2 + 3 * i], items[3 + 3 * i], j

    def start(k):
        @pl.when(k < total)
        def _():
            _, e, _, _, j = item(k)
            for cp in _chunk_copies(step, p, refs, off, e, j, _divmod(k, D)[1]):
                cp.start()

    def wait(slot):
        wrefs, woff = step.wait_refs
        for cp in _chunk_copies(step, p, wrefs, woff, 0, 0, slot):
            cp.wait()

    for q in range(D - 1):
        start(jnp.int32(q))
    dispatched()

    @pl.when(n_items > 0)
    def _():
        _item_rows(S, buf, 0, items[2], items[3], start=True, src_buf=x_src)

    def x_rows(b):
        return lambda R: _unpack_bf16x2(buf.x(b)[pl.ds(0, R), :])

    def item_in(i, b, first, n):
        _item_rows(S, buf, b, first, n, start=False, src_buf=x_src)

        @pl.when(i + 1 < n_items)
        def _():
            _item_rows(
                S,
                buf,
                1 - b,
                items[5 + 3 * i],
                items[6 + 3 * i],
                start=True,
                src_buf=x_src,
            )

        def zero(z):
            buf.acc[pl.ds(0, z), :] = jnp.zeros((z, He), f32)

        sized(n, sizes, zero)

    def item_out(i, first, n):
        if emit is not None:
            emit(i, first, n)
            return
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

    def item_body(i, carry):
        first, n = items[2 + 3 * i], items[3 + 3 * i]
        b = i % 2  # x rows alternate between items
        with _phase("item_in", 2):
            item_in(i, b, first, n)
        bounds, arms = chunk_arms(step.kind, x_rows(b), buf.acc, He, CI, RI)

        def chunks(arm):
            def chunk(j, carry2):
                k = i * NC + j
                slot = _divmod(k, D)[1]
                with _phase("issue", 2):
                    start(k + D - 1)
                    wait(slot)
                arm(_chunk_refs(step, p, slot))
                return carry2

            jax.lax.fori_loop(0, NC, chunk, 0)

        jax.lax.switch(
            arm_index(n, bounds), [functools.partial(chunks, a) for a in arms]
        )
        with _phase("item_out", 2):
            item_out(i, first, n)
        return carry

    jax.lax.fori_loop(0, n_items, item_body, 0)
    if emit is not None:
        return

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
    pw = small_row(smr, "post_expert_w", He)
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
    rows: fn(row, t0, m, blk) on each tile of rows [row0 + t0, row0 + t0 + m), row(i) the ref
    of its row i, blk the tile's [m, H] ref (None where VMEM rows are not 8-row aligned),
    written back unless write_back=False. HBM: one copy each way per tile when row0
    is 8-row aligned (n % 8 == 0: row0 = rank * n), else one per row, all in flight (an
    aligned tile's write-back of another rank's row would race with its all-gather)."""
    hs, RT = S["h_stream"], c.row_tile
    if c.tiled_vmem_pads:

        def load_v(r0, n):
            return hs[pl.ds(r0, n), :]

        def rows_v(r0, n, fn, write_back=True):
            for t0 in range(0, n, RT):
                m = min(RT, n - t0)
                blk = hs.at[pl.ds(attn.aligned8(r0 + t0), m)] if n % 8 == 0 else None
                fn(lambda i, t0=t0: hs.at[pl.ds(r0 + t0 + i, 1)], t0, m, blk)

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
            fn(lambda i: stage.at[pl.ds(i, 1)], t0, m, stage.at[pl.ds(0, m)])
            if write_back:
                cps = [
                    pltpu.make_async_copy(stage.at[b], hs.at[a], sem) for a, b in pairs
                ]
                for cp in cps:
                    cp.start()
                for cp in cps:
                    cp.wait()

    return load, rows


def _unpool_rows(step: EPLayerStep, win) -> None:
    """ep_f (window rows) = unpool(u) on every column, own rows of the window."""
    c, S, smr = step.c, step.S, step.smr
    H, He = c.hidden, c.expert_hidden
    _, wn, _ = win

    def put(n0, y):
        S["ep_f"][pl.ds(0, wn), pl.ds(n0, y.shape[1])] = y.astype(bf16).astype(f32)

    stream_matmul(
        c,
        S,
        step.W["wunpool"],
        step.layer,
        lambda k0, n: S["ep_u"][:, pl.ds(k0, n)].astype(bf16),
        small_row(smr, "wunpool_s", H),
        He,
        H,
        put,
        prefetched=True,
    )


def _ffn_residual(step: EPLayerStep, win) -> None:
    """Own rows of S["h_stream"] += the FFN branch: ffn_beta * norm(ep_f * post_ffn_w)."""
    c, S, smr = step.c, step.S, step.smr
    H, rc = c.hidden, c.rows_per_dev
    w0, wn, w_off = win
    fw = S["ep_f"][pl.ds(0, wn), :] * small_row(smr, "post_ffn_w", H)
    S["ep_f"][pl.ds(0, wn), :] = _rms(fw, c.post_eps).astype(bf16).astype(f32)
    _, rows = _stream_rows(c, S)
    alpha, beta = small_row(smr, "ffn_alpha", H), small_row(smr, "ffn_beta", H)

    def residual(row, t0, m, blk):
        h = attn.tile_value(row, m, blk)
        if rc % 8 == 0:  # the window offset is 8-row aligned
            branch = S["ep_f"][pl.ds(attn.aligned8(w_off + t0), m), :]
        else:  # an unaligned window offset: one-row loads
            branch = jnp.concatenate(
                [S["ep_f"][pl.ds(w_off + t0 + i, 1), :] for i in range(m)], axis=0
            )
        attn.tile_store(row, m, blk, alpha * h + beta * branch)

    rows(w0 + w_off, rc, residual)  # own rows only: the all-gather fills the others


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

    def norm(row, t0, m, blk):
        h = attn.tile_value(row, m, blk)
        own[pl.ds(t0, m), :] = _pack_bf16x2(
            _rms(h, c.rms_eps, w).astype(bf16).astype(f32)
        )

    with _phase("lhs_norm"):
        rows(me * rc, rc, norm, write_back=False)
    _bcast_own_rows(c, me, rc, own, 0, g, sems, rsems)


def _bcast_own_rows(c: MKConfig, me, rc: int, own, off, g, sems, rsems) -> None:
    """Rows [off, off + rc) of the VMEM u32 own buffer into rows [me * rc, (me + 1) * rc) of
    g ([R, W] u32) on every rank; returns once every owner's rows have landed here.
    sems: [1] sent, [2] own copy; rsems[k]: owner k's rows received."""
    # rows per copy: a block, or a row of an unaligned one
    blocked = rc % 8 == 0 and isinstance(off, int) and off % 8 == 0
    n = rc if blocked else 1

    def block(k):
        """Per copy: (rows in own, owner k's rows in g)."""
        if blocked:
            return [(pl.ds(off, rc), pl.ds(pl.multiple_of(k * rc, 8), rc))]
        return [(pl.ds(off + r, 1), pl.ds(k * rc + r, 1)) for r in range(rc)]

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
        with _phase("lhs_relay"):
            for d in (1, 2, 3):
                k = 2 * jnp.bitwise_xor(chip, d) + core
                wait(rsems.at[k], len(block(0)))
                send(g, k, partner)
        with _phase("lhs_wait"):
            wait(rsems.at[partner], len(block(0)))
            for d in (1, 2, 3):
                wait(rsems.at[2 * jnp.bitwise_xor(chip, d) + 1 - core], len(block(0)))
            wait(sems.at[1], 7 * len(block(0)))
    wait(sems.at[2], len(block(0)))


def _hbm_rows_to_lhs(c: MKConfig, S: dict, g) -> None:
    """S["lhs"][:, :2 W] = every row of g ([R, W] bf16 pairs in HBM, W <= H / 2) as bf16, row
    tile by row tile (double-buffered through two [RT, H / 2] word slots)."""
    RT, R, Hh, W = c.row_tile, c.rows, c.hidden // 2, g.shape[1]
    sems = S["ep_lhs_sems"]
    st = S["ep_lhs_stage"] if c.tiled_vmem_pads else S["sin"].at[0].bitcast(jnp.uint32)

    def stage(b):
        return st.at[:, pl.ds(b * Hh, W)]

    def cp(t):
        return pltpu.make_async_copy(
            g.at[pl.ds(t * RT, RT)], stage(t % 2), sems.at[3 + t % 2]
        )

    cp(0).start()
    for t in range(R // RT):
        if t + 1 < R // RT:
            cp(t + 1).start()
        cp(t).wait()
        S["lhs"][pl.ds(t * RT, RT), pl.ds(0, 2 * W)] = _unpack_bf16x2(
            stage(t % 2)[...]
        ).astype(bf16)


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
        if is_cap:
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
        capture()
