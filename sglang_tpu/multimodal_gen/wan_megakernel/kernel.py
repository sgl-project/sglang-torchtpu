# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-A14B's DiT block stack, every layer in one Pallas kernel.

``stack`` runs ``reference.block`` for every layer in one ``pallas_call``:
the weights stay in HBM and stream through VMEM one ``[H, cols]`` tile at a
time, and the activations, norms, modulation, RoPE, softmax, residuals and
the Ulysses exchange stay inside the kernel between layers. Under a
``shard_map`` over ``axis`` each of ``P`` devices holds ``R = S / P``
consecutive rows and every weight; it owns heads ``[me * H/P, (me+1) * H/P)``
for self-attention. Each layer runs three phases:

- A, per row tile: LayerNorm and modulation, then the QKV projection in
  streamed column tiles (q tiles, k tiles, v tiles). Each v tile goes out at
  once, one DMA per head to the head's owner; q and k, normalized over all
  heads, rotated and rounded, go out after their last tile: q's heads during
  the k tiles (their vector work under the tiles' matmuls), k's during the v
  tiles.
- B, per (own head, query tile): flash attention over every row, the keys
  past the real rows masked; the output tile goes by DMA to the rows' owner.
- C, per row tile: output projection and gated residual, the affine
  LayerNorm, the cross-attention query (normalized over all heads), softmax
  over the text keys of every head (the prompt's distinct keys, each weighted
  by the text rows it stands for: see ``stack``), its output projection and
  residual, then LayerNorm and modulation and the GELU MLP as streamed (up,
  down) tile pairs, and the gated residual.

Exchanges land in HBM buffers double buffered by layer parity: a device
writes a peer's parity-``p`` buffer of layer ``l + 2`` only after that peer
sent it layer ``l + 1`` data, which the peer produces after it is done
reading layer ``l``'s buffer.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.multimodal_gen.wan_megakernel import reference as ref
from sglang_tpu.multimodal_gen.wan_megakernel.tiling import Tiling

F32, BF16 = jnp.float32, jnp.bfloat16
LAST = (((1,), (1,)), ((), ()))  # a @ b.T
FIRST = (((0,), (0,)), ((), ()))  # a.T @ b
_COLLECTIVE_ID = 9
_MASK = -1e30  # a masked score: exp2 of it minus any real max is 0
LOG2E = math.log2(math.e)
# Softmax without a running max (exact, by bounds). Scores s (log2 units) of a
# query tile are bounded by |q| max|k| / sqrt(D) (Cauchy-Schwarz on the bf16
# values the MXU multiplies). V is stored times 2^-17 (phase A; a power of two,
# so exact) and the output divides by l 2^-17. "fixed": every |s| <= 109 ->
# p = exp2(s) in [2^-109, 2^109] (no offset to subtract), a sum over 2^15 keys
# below 2^124 and times |v 2^-17| < 2^3 below 2^127. "rowfix": p = exp2(s -
# (bound_row - 80)) <= 2^80. Either is exact unless a row's largest weights are
# lost to underflow (fixed: a real row whose best score is under about -80, its
# products p v 2^-17 under 2^-126; rowfix: a bound loose by over ~160), which a
# row sum under 2^-80 flags: then the tile reruns with a running max.
FIXED_BOUND = 109.0
V_SCALE = 2.0**-17
ROW_HEADROOM, ROW_FLOOR = 80.0, 2.0**-80
# Squared norms summed on the MXU from bf16-rounded squares: relative error
# under 2^-8 (2^-9 a term, summed); this factor keeps them upper bounds.
NORM_SLACK = 1.0 + 2.0**-6

# One f32 row of each layer's vectors (``pack_kernel_weights``): name -> length.
_VEC_ORDER = (
    "bqkv",
    "qn",
    "kn",
    "bo",
    "n2w",
    "n2b",
    "bq2",
    "qn2",
    "bo2",
    "b2",
    "b1",
)


def _vec_layout(cfg: ref.Config) -> dict[str, tuple[int, int]]:
    sizes = {"bqkv": 3 * cfg.hidden, "b1": cfg.ffn}
    out, at = {}, 0
    for name in _VEC_ORDER:
        n = sizes.get(name, cfg.hidden)
        out[name] = (at, n)
        at += n
    return out


def pack_kernel_weights(
    stacked: dict[str, Any], cfg: ref.Config, tiling: Tiling
) -> dict[str, Any]:
    """``reference.pack_layer`` arrays stacked over layers -> the kernel's operands.

    Column-tiled projections become ``[L, N/cols, H, cols]`` (each tile contiguous),
    the MLP down projection ``[L, F/cols, cols, H]`` (or ``[L, H/down_cols, F,
    down_cols]`` for ``mlp_mode="twopass"``), all bf16, and the per-layer vectors
    one ``[L, 1, V]`` f32 row each (``_vec_layout``).
    """
    t = tiling.cols
    L = stacked["wqkv"].shape[0]

    def cols(w):
        w = jnp.asarray(w, BF16)
        k, n = w.shape[1], w.shape[2]
        return w.reshape(L, k, n // t, t).transpose(0, 2, 1, 3)

    w2 = jnp.asarray(stacked["w2"], BF16)
    if tiling.mlp_mode == "twopass":  # [L, H/TD, F, TD] output-column tiles
        td = tiling.down_cols
        w2_tiles = w2.reshape(L, cfg.ffn, cfg.hidden // td, td).transpose(0, 2, 1, 3)
    else:  # [L, F/cols, cols, H] row tiles
        w2_tiles = w2.reshape(L, cfg.ffn // t, t, cfg.hidden)
    out = {
        "wqkv": cols(stacked["wqkv"]),
        "wo": cols(stacked["wo"]),
        "wq2": cols(stacked["wq2"]),
        "wo2": cols(stacked["wo2"]),
        "w1": cols(stacked["w1"]),
        "w2": w2_tiles,
        "vecs": jnp.concatenate(
            [jnp.asarray(stacked[n], F32).reshape(L, 1, -1) for n in _VEC_ORDER],
            axis=2,
        ),
    }
    return out


KERNEL_WEIGHTS = ("wqkv", "wo", "wq2", "wo2", "w1", "w2", "vecs")


def kernel_weight_shapes(
    cfg: ref.Config, tiling: Tiling, layers: int
) -> dict[str, tuple[tuple[int, ...], Any]]:
    """``pack_kernel_weights``' shapes and dtypes, without the weights."""
    t, H, F, L = tiling.cols, cfg.hidden, cfg.ffn, layers
    td = tiling.down_cols
    nv = sum(n for _, n in _vec_layout(cfg).values())
    return {
        "wqkv": ((L, 3 * H // t, H, t), BF16),
        "wo": ((L, H // t, H, t), BF16),
        "wq2": ((L, H // t, H, t), BF16),
        "wo2": ((L, H // t, H, t), BF16),
        "w1": ((L, F // t, H, t), BF16),
        "w2": (
            (L, H // td, F, td) if tiling.mlp_mode == "twopass" else (L, F // t, t, H),
            BF16,
        ),
        "vecs": ((L, 1, nv), F32),
    }


def cross_counts(tiling: Tiling, text_len: int) -> tuple[int, ...]:
    """The cross-attention key counts with their own code path: those of
    ``tiling.cross_keys`` below the text length (any other count attends to every
    text row)."""
    return tuple(nk for nk in tiling.cross_keys if nk < text_len)


def _stream(tiles: int, fetch, body) -> None:
    """``body(i, slot)`` over ``tiles`` while tile ``i + 1`` streams into the
    other slot of a two-slot ring. ``fetch(i, slot)`` lists tile ``i``'s copies;
    tile ``i`` uses slot ``i % 2``, and the caller has already started tile 0 into
    slot 0 (each stream's first tile is issued during the work before it)."""

    def step(i, carry):
        slot = i % 2
        for c in fetch(i, slot):
            c.wait()

        @pl.when(i + 1 < tiles)
        def _next():
            for c in fetch(i + 1, 1 - slot):
                c.start()

        body(i, slot)
        return carry

    lax.fori_loop(0, tiles, step, None)


def _stream_pipelined(tiles: int, fetch, produce, consume) -> None:
    """``_stream`` with each tile's epilogue in two parts: ``produce(i, slot)``
    (tile ``i``'s matmul and what takes its result out of the MXU) in step ``i``,
    ``consume(i)`` (the rest) in step ``i + 1``'s basic block ahead of that step's
    ``produce``, so it runs under that matmul (the last one after the loop). A
    buffer ``produce`` writes and ``consume`` reads is read before it is
    rewritten. The caller has started tile 0 into slot 0."""
    for c in fetch(0, 0):
        c.wait()
    if tiles > 1:
        for c in fetch(1, 1):
            c.start()
    produce(0, 0)

    def step(i, carry):
        slot = i % 2
        for c in fetch(i, slot):
            c.wait()

        @pl.when(i + 1 < tiles)
        def _next():
            for c in fetch(i + 1, 1 - slot):
                c.start()

        consume(i - 1)
        produce(i, slot)
        return carry

    lax.fori_loop(1, tiles, step, None)
    consume(tiles - 1)


def _kernel(
    meta_ref,  # SMEM [2] int32: real rows of the whole sequence, cross-attention keys
    x_in,
    e_ref,
    k2_ref,
    v2_ref,
    kw_ref,
    cos_ref,
    sin_ref,
    wqkv_ref,
    wo_ref,
    wq2_ref,
    wo2_ref,
    w1_ref,
    w2_ref,
    vecs_ref,
    x_ref,
    kv_buf,
    attn_buf,
    kv_mine,
    n3_hbm,
    kmax_in,  # outputs (x aliased to x_in)
    vec_v,
    e_v,
    kw_v,
    sems,
    bulk_send,
    kv_recv,
    attn_send,
    attn_recv,
    loc_sem,
    kmax_recv,
    floor_flag,  # SMEM [1] int32: an attention item's row-sum floor check (phase B)
    cfg: ref.Config,
    tiling: Tiling,
    axis: str | None,
    devices: int,
    softmax: str = "auto",
    skip: frozenset = frozenset(),
    trace: int = 0,
):
    del x_in
    H, D, heads, F = cfg.hidden, cfg.head_dim, cfg.heads, cfg.ffn
    P = devices
    LH = heads // P  # heads I own
    # Bulk exchange: phase A sends each owner only its first head; the rest wait
    # in my HBM (kv_mine) and go out in phase B, head h + 1 while I attend head h.
    bulk = P > 1 and LH > 1 and "nobulk" not in skip and "noremote" not in skip
    R = x_ref.shape[0]  # rows I hold
    S = R * P
    L = e_ref.shape[0]
    TN = tiling.cols
    chunk = tiling.chunk_rows
    eps = cfg.eps
    me = lax.axis_index(axis) if P > 1 else 0
    valid = meta_ref[0]
    # Cross-attention attends to the first ``keys`` of the T text rows of K2/V2:
    # one of ``counts`` (weighted keys, see ``stack``), else all T.
    T = k2_ref.shape[2]
    keys = meta_ref[1]
    counts = cross_counts(tiling, T)
    vl = _vec_layout(cfg)
    copy = pltpu.make_async_copy
    scale_l2 = LOG2E / math.sqrt(D)

    def vec(name, start=0, size=None):
        off, n = vl[name]
        size = n if size is None else size
        if not isinstance(start, int):
            start = pl.multiple_of(start, 128)
        return vec_v[:, pl.ds(off + start, size)]

    def erow(i, cols=None):
        if cols is None:
            return e_v[i : i + 1, :]
        return e_v[i : i + 1, cols]

    def device(index):
        return {axis: index}

    def scope(name, level=1):
        """An xprof trace region (``trace >= level``), else nothing."""
        import contextlib

        return jax.named_scope(name) if trace >= level else contextlib.nullcontext()

    NS = tiling.weight_splits

    def wcopies(src, ring_ref, slot, sem):
        """A weight tile ``src`` -> ``ring_ref[slot]`` as ``NS`` row-chunk DMAs, each
        on its own semaphore ``sem[slot, c]`` (several DMAs in flight per tile)."""
        n = src.shape[0] // NS
        return [
            copy(
                src.at[pl.ds(c * n, n)],
                ring_ref.at[slot, pl.ds(c * n, n)],
                sem.at[slot, c],
            )
            for c in range(NS)
        ]

    def start(copies):
        for c in copies:
            c.start()

    def wait(copies):
        for c in copies:
            c.wait()

    def send(src, dst, owner, send_sem, recv_sem, local_sem):
        """Starts ``src`` -> ``dst`` on ``owner`` (a local copy if me); returns
        the wait after which ``src`` may be rewritten. A remote copy lands on
        ``recv_sem`` of the owner (waited in bulk there); a local one is done
        once waited here."""
        local = copy(src, dst, local_sem)
        if P == 1 or "noremote" in skip:
            local.start()
            return local.wait
        remote = pltpu.make_async_remote_copy(
            src, dst, send_sem, recv_sem, device_id=device(owner)
        )
        pl.when(owner == me)(local.start)
        pl.when(owner != me)(remote.start)

        def wait():
            pl.when(owner == me)(local.wait)
            pl.when(owner != me)(remote.wait_send)

        return wait

    def await_bytes(view, sem):
        """Waits until ``sem`` received as many bytes as ``view`` holds."""
        if "noremote" in skip:
            return
        pltpu.make_async_copy(view, view, sem).wait()

    def layer_norm_rows(src, rs):
        """Rows ``rs`` of ``src`` (bf16) layer-normalized, in f32. The mean and the
        mean square come from one sweep, so no second reduction runs over the rows
        minus their mean; at [chunk_rows, H] that halves the loop's spills."""
        if "ln" in skip:
            return src[rs, :].astype(F32)
        xs = src[rs, :].astype(F32)
        mean = jnp.mean(xs, axis=1, keepdims=True)
        # E[x^2] - mean^2 loses precision only when |mean| >> std; the max keeps
        # rounding from taking it below 0.
        var = jnp.maximum(jnp.mean(xs * xs, axis=1, keepdims=True) - mean * mean, 0.0)
        return (src[rs, :].astype(F32) - mean) * lax.rsqrt(var + eps)

    def for_chunks(rows, fn):
        def body(i, carry):
            fn(pl.ds(pl.multiple_of(i * chunk, chunk), chunk))
            return carry

        lax.fori_loop(0, rows // chunk, body, None)

    def rope(v, cos, sin):
        return v * cos + pltpu.roll(v, D // 2, 1) * sin

    # ------------------------------------------------------------ phase A
    def phase_a(layer, parity):
        TA = tiling.qkv_rows
        tiles = R // TA
        nq = H // TN  # weight tiles each of q, k, v
        hpt = TN // D  # heads a tile
        VS = 4  # v staging slots
        QS = tiling.qk_slots  # row tiles of q and k in flight
        total = 3 * nq

        def head_send(which, src, h, r0, sem2):
            """Starts head ``h``'s rows of ``src [TA, D]`` toward its owner. ``sem2``:
            the staging slot's own ``[2]`` DMA semaphores (remote send, local copy),
            so a wait for the slot's sends counts only them. Under ``bulk`` a peer's
            heads after its first are parked in ``kv_mine`` (a local copy)."""
            owner, lh = h // LH, h % LH
            dst = kv_buf.at[
                parity, which, lh, pl.ds(pl.multiple_of(me * R + r0, TA), TA)
            ]
            if not bulk:
                return send(src, dst, owner, sem2.at[0], kv_recv.at[0], sem2.at[1])
            local = copy(src, dst, sem2.at[1])
            park = copy(
                src,
                kv_mine.at[which, owner, jnp.maximum(lh - 1, 0), pl.ds(r0, TA)],
                sem2.at[1],
            )
            remote = pltpu.make_async_remote_copy(
                src, dst, sem2.at[0], kv_recv.at[0], device_id=device(owner)
            )
            pl.when(owner == me)(local.start)
            pl.when((owner != me) & (lh != 0))(park.start)
            pl.when((owner != me) & (lh == 0))(remote.start)

        def scoped(
            nbuf,
            qbuf,
            kbuf,
            vstage,
            sumsq,
            cosv,
            sinv,
            rsem,
            qk_sems,
            v_sems,
            ring,
            wsem,
            kpeak,
            kpk_sem,
            fac,
        ):
            class _Tile:
                def __init__(self, g, slot):
                    self.copies = wcopies(wqkv_ref.at[layer, g], ring, slot, wsem)

                def start(self):
                    for c in self.copies:
                        c.start()

                def wait(self):
                    for c in self.copies:
                        c.wait()

            wqkv_copy = _Tile

            def wait_sends(sem2, heads_list):
                """Waits for one staging slot's sends: ``heads_list`` lists each send's
                head (static or traced); every send moved ``[TA, D]`` bf16."""
                src = vstage.at[0, :, pl.ds(0, D)]
                dst = kv_buf.at[parity, 0, 0, pl.ds(0, TA)]
                local = copy(src, dst, sem2.at[1])
                if P == 1 or "noremote" in skip:
                    for _ in heads_list:
                        local.wait()
                    return
                remote = pltpu.make_async_remote_copy(
                    src, dst, sem2.at[0], kv_recv.at[0], device_id=device(0)
                )
                if bulk:  # remote only for a peer's first head
                    n_local = sum(
                        jnp.where((h // LH == me) | (h % LH != 0), 1, 0)
                        for h in heads_list
                    )
                else:
                    n_local = sum(jnp.where(h // LH == me, 1, 0) for h in heads_list)

                def body_l(i, c):
                    local.wait()
                    return c

                def body_r(i, c):
                    remote.wait_send()
                    return c

                lax.fori_loop(0, n_local, body_l, None)
                lax.fori_loop(0, len(heads_list) - n_local, body_r, None)

            def qk_wait(ts, which):
                """Waits for row-tile slot ``ts``'s q (``which`` 0) or k (1) sends."""
                wait_sends(qk_sems.at[ts, which], list(range(heads)))

            def wait_v_slot(vs, gv_old):
                j = gv_old % nq
                wait_sends(v_sems.at[vs], [j * hpt + k for k in range(hpt)])

            def set_fac(which):
                """q's (``which`` 0) or k's (1) RMSNorm factor over all heads into
                ``fac``, lane-broadcast: one rsqrt a row once the projection's last
                tile is stored, so a head only multiplies by it (and reads no
                ``sumsq``, which the next projection's tiles are writing)."""
                if "finish" in skip:
                    return
                factor = lax.rsqrt(sumsq[:, which : which + 1] / H + eps)
                fac[...] = jnp.broadcast_to(factor, fac.shape)

            def finish_compute(buf, ts, which, norm_name, h, r0):
                """Head ``h``'s RMSNorm and RoPE in place (and k's peak norm). Vector
                work only, no DMA: emitted before the next projection tile's matmul,
                it runs under that matmul's MXU stream instead of after it;
                ``finish_send`` starts the head's send once the matmul is out."""
                if "finish" in skip:
                    return
                cols = pl.ds(pl.multiple_of(h * D, D), D)
                v = buf[ts, :, cols].astype(F32) * fac[...]
                v = (v * vec(norm_name, h * D, D)).astype(BF16).astype(F32)
                rot = rope(v, cosv[...], sinv[...]).astype(BF16)
                buf[ts, :, cols] = rot
                if which == 1:  # this head's largest squared key norm over my real rows
                    kf = rot.astype(F32)
                    n2 = jnp.sum(kf * kf, axis=1, keepdims=True)
                    row = lax.broadcasted_iota(jnp.int32, n2.shape, 0) + (me * R + r0)
                    m = jnp.max(jnp.where(row < valid, n2, 0.0))
                    lane = lax.broadcasted_iota(jnp.int32, kpeak.shape, 1)
                    kpeak[...] = jnp.where(
                        lane == h, jnp.maximum(kpeak[...], m), kpeak[...]
                    )

            def finish_send(buf, ts, which, h, r0):
                """Starts head ``h``'s finished rows toward its owner."""
                cols = pl.ds(pl.multiple_of(h * D, D), D)
                head_send(which, buf.at[ts, :, cols], h, r0, qk_sems.at[ts, which])

            def matmul(g, slot, scale=None):
                """QKV tile ``g`` (bf16); ``scale`` (a power of two) applied before the
                rounding, so it is exact."""
                if "qkv" in skip:
                    acc = jnp.zeros((TA, TN), F32)
                else:
                    acc = jnp.dot(nbuf[...], ring[slot], preferred_element_type=F32)
                acc = acc + vec("bqkv", g * TN, TN)
                return (acc if scale is None else acc * scale).astype(BF16)

            def qk_store(i, vals, buf, ts, which):
                buf[ts, :, pl.ds(pl.multiple_of(i * TN, TN), TN)] = vals
                vf = vals.astype(F32)
                sumsq[:, which : which + 1] += jnp.sum(vf * vf, axis=1, keepdims=True)

            wqkv_copy(0, 0).start()
            kpeak[...] = jnp.zeros(kpeak.shape, F32)

            def row_tile(t, carry):
                ts = t % QS
                r0 = pl.multiple_of(t * TA, TA)

                cx = copy(x_ref.at[pl.ds(r0, TA)], nbuf, rsem.at[0])
                cc = copy(cos_ref.at[pl.ds(r0, TA)], cosv, rsem.at[1])
                cs = copy(sin_ref.at[pl.ds(r0, TA)], sinv, rsem.at[2])
                for c in (cx, cc, cs):
                    c.start()
                cx.wait()

                def norm1(rs):
                    n = layer_norm_rows(nbuf, rs)
                    nbuf[rs, :] = (
                        n * (1.0 + erow(ref.SCALE)) + erow(ref.SHIFT)
                    ).astype(BF16)

                with scope("A.ln1", 2):
                    for_chunks(TA, norm1)
                sumsq[...] = jnp.zeros(sumsq.shape, F32)
                cc.wait()
                cs.wait()

                def advance(g):
                    """Waits for weight tile ``g``; starts the next (this row tile's or the
                    next's first) into the other ring slot. The slot parity runs on across
                    row tiles, so the next tile's first never lands in the slot in use
                    (whatever the parity of ``total``). Returns tile ``g``'s slot."""
                    cur = (t * total + g) % 2
                    wqkv_copy(g, cur).wait()

                    @pl.when(g + 1 < total)
                    def _():
                        wqkv_copy(g + 1, 1 - cur).start()

                    @pl.when((g + 1 == total) & (t + 1 < tiles))
                    def _():
                        wqkv_copy(0, 1 - cur).start()

                    return cur

                def seg_q(i, c):
                    qk_store(i, matmul(i, advance(i)), qbuf, ts, 0)
                    return c

                with scope("A.qkwait", 2):

                    @pl.when(t >= QS)
                    def _free_q():  # slot ts held row tile t - QS's q until its sends finished
                        qk_wait(ts, 0)

                with scope("A.q", 2):
                    lax.fori_loop(0, nq, seg_q, None)
                    set_fac(0)

                with scope("A.qkwait", 2):

                    @pl.when(t >= QS)
                    def _free_k():
                        qk_wait(ts, 1)

                # A k (v) tile's step finishes q's (k's) heads of the same index:
                # their vector work ahead of the tile's matmul, their sends after it.
                def seg_k(i, c):
                    g = nq + i
                    slot = advance(g)
                    for k in range(hpt):
                        finish_compute(qbuf, ts, 0, "qn", i * hpt + k, r0)
                    qk_store(i, matmul(g, slot), kbuf, ts, 1)
                    for k in range(hpt):
                        finish_send(qbuf, ts, 0, i * hpt + k, r0)
                    return c

                with scope("A.k", 2):
                    lax.fori_loop(0, nq, seg_k, None)
                    set_fac(1)

                def seg_v(i, c):
                    g = 2 * nq + i
                    slot = advance(g)
                    gv = t * nq + i
                    vs = gv % VS

                    # The staging slot's old sends are waited before the matmul: a wait
                    # between the matmul and its store would end the scheduling block
                    # and send the tile through a spill.
                    @pl.when(gv >= VS)
                    def _():
                        wait_v_slot(vs, gv - VS)

                    for k in range(hpt):
                        finish_compute(kbuf, ts, 1, "kn", i * hpt + k, r0)
                    # V is stored times 2^-17 (see FIXED_BOUND).
                    vstage[vs] = matmul(g, slot, V_SCALE)
                    for k in range(hpt):
                        head_send(
                            2,
                            vstage.at[vs, :, pl.ds(k * D, D)],
                            i * hpt + k,
                            r0,
                            v_sems.at[vs],
                        )
                    for k in range(hpt):
                        finish_send(kbuf, ts, 1, i * hpt + k, r0)
                    return c

                with scope("A.v", 2):
                    lax.fori_loop(0, nq, seg_v, None)
                return carry

            lax.fori_loop(0, tiles, row_tile, None)
            # Everything still in flight from this phase.
            for back in range(min(QS, tiles), 0, -1):
                qk_wait((tiles - back) % QS, 0)
                qk_wait((tiles - back) % QS, 1)
            gv_end = tiles * nq
            for back in range(min(VS, gv_end), 0, -1):
                gv = gv_end - back
                wait_v_slot(gv % VS, gv)
            # My key peaks to every device (phase B bounds its scores with them).
            for o in range(P):
                dst = kmax_in.at[parity, me]
                if P == 1 or "noremote" in skip:
                    if o == 0:
                        c = copy(
                            kpeak,
                            kmax_in.at[parity, 0 if P == 1 else me],
                            kpk_sem.at[1],
                        )
                        c.start()
                        c.wait()
                    continue
                local = copy(kpeak, dst, kpk_sem.at[1])
                remote = pltpu.make_async_remote_copy(
                    kpeak, dst, kpk_sem.at[0], kmax_recv.at[0], device_id=device(o)
                )
                pl.when(o == me)(local.start)
                pl.when(o != me)(remote.start)
                pl.when(o == me)(local.wait)
                pl.when(o != me)(remote.wait_send)

        pl.run_scoped(
            scoped,
            pltpu.VMEM((TA, H), BF16),  # x tile, then n1
            pltpu.VMEM((QS, TA, H), BF16),  # q, by row-tile slot
            pltpu.VMEM((QS, TA, H), BF16),  # k
            pltpu.VMEM((VS, TA, TN), BF16),  # v tiles in flight
            pltpu.VMEM((TA, 2), F32),  # sums of squares of q, k rows
            pltpu.VMEM((TA, D), F32),
            pltpu.VMEM((TA, D), F32),
            pltpu.SemaphoreType.DMA((3,)),
            pltpu.SemaphoreType.DMA(
                (QS, 2, 2)
            ),  # q, k sends per row-tile slot: remote, local
            pltpu.SemaphoreType.DMA((VS, 2)),  # v sends per staging slot
            pltpu.VMEM((2, H, TN), BF16),  # weight ring
            pltpu.SemaphoreType.DMA((2, NS)),
            pltpu.VMEM(
                (8, 128), F32
            ),  # per-head largest squared key norm (lane = head)
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((TA, D), F32),  # q's or k's RMSNorm factor, lane-broadcast
        )

    # ------------------------------------------------------------ phase B
    def phase_b(layer, parity):
        BQ, BKV, SUB = tiling.block_q, tiling.block_kv, tiling.sub_kv
        q_tiles = S // BQ
        full_tiles = valid // BKV  # key tiles of real rows only
        has_edge = valid % BKV != 0
        n_all = full_tiles + has_edge.astype(jnp.int32)
        if (
            P > 1 and not bulk
        ):  # every peer's q, k, v of my heads (my own rows' copies were waited)
            await_bytes(kv_buf.at[parity, :, :, pl.ds(0, (P - 1) * R)], kv_recv.at[0])

        def bulk_sends(lh_next, wait_only):
            """Head ``lh_next``'s q, k, v of my rows from ``kv_mine`` to every peer."""
            for o in range(P):
                for which in range(3):
                    src = kv_mine.at[which, o, lh_next - 1]
                    dst = kv_buf.at[
                        parity, which, lh_next, pl.ds(pl.multiple_of(me * R, R), R)
                    ]
                    remote = pltpu.make_async_remote_copy(
                        src,
                        dst,
                        bulk_send.at[0],
                        kv_recv.at[lh_next],
                        device_id=device(o),
                    )
                    pl.when(o != me)(remote.wait_send if wait_only else remote.start)

        def scoped(qd, k, v, m_run, l_run, o_run, mrow, kall, stage, bsem, osem):
            ones8 = jnp.ones((8, D), BF16)

            def kv_fetch(lh, j, slot):
                rs = pl.ds(pl.multiple_of(j * BKV, BKV), BKV)
                return [
                    copy(kv_buf.at[parity, 1, lh, rs], k.at[slot], bsem.at[slot]),
                    copy(kv_buf.at[parity, 2, lh, rs], v.at[slot], bsem.at[2 + slot]),
                ]

            def sq_rows(x):
                """``[8, n]`` f32 upper bounds of each row's squared norm (``x [n, D]``)."""
                xf = x.astype(F32)
                return (
                    lax.dot_general(
                        ones8, (xf * xf).astype(BF16), LAST, preferred_element_type=F32
                    )
                    * NORM_SLACK
                )

            # Every device's per-head key peaks (phase A): the largest over devices.
            if P > 1 and "noremote" not in skip:
                await_bytes(kmax_in.at[parity, pl.ds(0, P - 1)], kmax_recv.at[0])
            ck = copy(kmax_in.at[parity], kall, bsem.at[4])
            ck.start()
            ck.wait()
            kpeak_heads = (
                jnp.max(kall[:, 0:1, :], axis=0) * NORM_SLACK
            )  # [1, 128]: lane = head
            lane = lax.broadcasted_iota(jnp.int32, kpeak_heads.shape, 1)

            def q_copy(it, slot):
                lh, qt = it // q_tiles, it % q_tiles
                return copy(
                    kv_buf.at[parity, 0, lh, pl.ds(pl.multiple_of(qt * BQ, BQ), BQ)],
                    qd.at[slot],
                    bsem.at[5 + slot],
                )

            def update(slot, start, variant, masked, q):
                s = (
                    lax.dot_general(
                        k[slot, pl.ds(start, SUB), :],
                        q[...],
                        LAST,
                        preferred_element_type=F32,
                    )
                    * scale_l2
                )
                if masked is not None:
                    kidx = lax.broadcasted_iota(jnp.int32, s.shape, 0) + masked
                    s = jnp.where(kidx < valid, s, _MASK)
                vs = v[slot, pl.ds(start, SUB), :]
                if variant == "online":
                    m_prev = m_run[0:1, :]
                    m_new = jnp.maximum(m_prev, jnp.max(s, axis=0, keepdims=True))
                    alpha = jnp.exp2(m_prev - m_new)
                    p_ = jnp.exp2(s - m_new)
                    l_run[0:1, :] = alpha * l_run[0:1, :] + jnp.sum(
                        p_, axis=0, keepdims=True
                    )
                    o_run[...] = alpha * o_run[...] + lax.dot_general(
                        vs, p_.astype(BF16), FIRST, preferred_element_type=F32
                    )
                    m_run[0:1, :] = m_new
                    return
                p_ = jnp.exp2(s if variant == "fixed" else s - mrow[0:1, :])
                l_run[0:1, :] += jnp.sum(p_, axis=0, keepdims=True)
                o_run[...] += lax.dot_general(
                    vs, p_.astype(BF16), FIRST, preferred_element_type=F32
                )

            def attend(lh, variant, q, m0):
                """The key tiles of real rows only against the query tile in ``q``, one
                softmax variant, from a fresh state whose running max is ``m0`` (the
                fixed or per-row offset: the state is then exactly an online one).
                Starts the edge tile's copy; ``edge`` finishes it."""
                m_run[0:1, :] = m0
                l_run[...] = jnp.zeros(l_run.shape, F32)
                o_run[...] = jnp.zeros(o_run.shape, F32)
                for c in kv_fetch(lh, 0, 0):
                    c.start()

                def step(j, carry):
                    slot = j % 2
                    for c in kv_fetch(lh, j, slot):
                        c.wait()

                    @pl.when(j + 1 < n_all)
                    def _next():
                        for c in kv_fetch(lh, j + 1, 1 - slot):
                            c.start()

                    if variant == "online" and BKV > 2 * SUB:
                        # rolled: the online body (a few tiles in a thousand) stays small
                        def sub(i, c):
                            update(slot, pl.multiple_of(i * SUB, SUB), variant, None, q)
                            return c

                        lax.fori_loop(0, BKV // SUB, sub, None)
                    else:
                        for st in range(0, BKV, SUB):
                            update(slot, st, variant, None, q)
                    return carry

                lax.fori_loop(0, full_tiles, step, None)

            def edge_wait(lh):
                for c in kv_fetch(lh, full_tiles, full_tiles % 2):
                    c.wait()

            def edge(lh, q, variant):
                """The tile holding the last real row, masked keys weighing 0, in the
                variant of the state before it."""

                @pl.when(has_edge)
                def _():
                    edge_wait(lh)
                    base = full_tiles * BKV

                    def sub(
                        i, c
                    ):  # rolled: one sub-block of code a variant (the edge is 1 tile of many)
                        st = pl.multiple_of(i * SUB, SUB)
                        update(full_tiles % 2, st, variant, base + st, q)
                        return c

                    lax.fori_loop(0, BKV // SUB, sub, None)

            def item(it, carry):
                lh, qt = it // q_tiles, it % q_tiles

                qs = it % 2
                q = qd.at[qs]

                @pl.when(qt == 0)
                def _new_head():
                    if (
                        bulk
                    ):  # this head's rows from every peer, then the next head's out
                        with scope("B.headwait", 2):
                            await_bytes(
                                kv_buf.at[parity, :, lh, pl.ds(0, (P - 1) * R)],
                                kv_recv.at[lh],
                            )

                        @pl.when(lh + 1 < LH)
                        def _():
                            bulk_sends(lh + 1, wait_only=False)

                    q_copy(
                        it, qs
                    ).start()  # a head's first query tile: only once its data is in

                q_copy(it, qs).wait()

                @pl.when((it + 1 < LH * q_tiles) & ((it + 1) % q_tiles != 0))
                def _prefetch_q():  # the next query tile of this head, during this item
                    q_copy(it + 1, 1 - qs).start()

                # Per query row, an upper bound of its scores (log2 units): |q| max|k| scale.
                qsq = sq_rows(q[...])
                kmax_s = jnp.max(
                    jnp.where(lane == me * LH + lh, kpeak_heads, 0.0),
                    axis=1,
                    keepdims=True,
                )
                bound_row = jnp.sqrt(qsq[0:1, :] * kmax_s) * scale_l2
                bound = jnp.max(bound_row)
                if "attn" in skip:
                    l_run[...] = jnp.ones(l_run.shape, F32)
                    o_run[...] = jnp.zeros(o_run.shape, F32)
                elif softmax == "auto":
                    # A row whose sum is under ROW_FLOOR may have lost its largest
                    # weights to underflow: the item reruns with a running max.
                    # Fixed: a real row whose best score is under about -80 sums
                    # under the floor (its products p v 2^-17 can flush to zero);
                    # padding rows do not count. Rowfix: every row, as its own guard
                    # (a padding row's sum can underflow to 0, and 0/0 would reach
                    # the next layer's masked keys). Laid out so the fixed path
                    # (~98% of items) runs one span of code: its check ends the
                    # fixed branch, and rowfix and the rerun sit in one branch it
                    # skips. floor_flag carries the check out of the branch that
                    # made it.
                    fixed_ok = bound <= FIXED_BOUND

                    @pl.when(fixed_ok)
                    def _fixed():
                        attend(lh, "fixed", q, jnp.zeros(bound_row.shape, F32))
                        edge(lh, q, "fixed")
                        # Real rows only: padding rows' fixed sums are >= 2^-109 a
                        # key, their outputs finite.
                        qrow = lax.broadcasted_iota(jnp.int32, (1, BQ), 1) + qt * BQ
                        lmin = jnp.min(
                            jnp.where(qrow < valid, l_run[0:1, :], ROW_FLOOR)
                        )
                        floor_flag[0] = jnp.logical_not(lmin >= ROW_FLOOR).astype(
                            jnp.int32
                        )

                    # Without the fixed path the flag is stale, but the branch runs.
                    @pl.when(jnp.logical_not(fixed_ok) | (floor_flag[0] != 0))
                    def _cold():
                        @pl.when(jnp.logical_not(fixed_ok))  # also a NaN bound
                        def _rowfix():
                            mrow[0:1, :] = bound_row - ROW_HEADROOM
                            attend(lh, "rowfix", q, mrow[0:1, :])
                            edge(lh, q, "rowfix")
                            lmin = jnp.min(l_run[0:1, :])  # every row
                            floor_flag[0] = jnp.logical_not(lmin >= ROW_FLOOR).astype(
                                jnp.int32
                            )

                        @pl.when(floor_flag[0] != 0)
                        def _online():
                            attend(
                                lh, "online", q, jnp.full(bound_row.shape, _MASK, F32)
                            )
                            edge(lh, q, "online")

                else:  # forced (tests): the caller vouches for the bounds
                    mrow[0:1, :] = bound_row - ROW_HEADROOM
                    m0 = {
                        "fixed": jnp.zeros(bound_row.shape, F32),
                        "rowfix": mrow[0:1, :],
                        "online": jnp.full(bound_row.shape, _MASK, F32),
                    }[softmax]
                    attend(lh, softmax, q, m0)
                    edge(lh, q, softmax)

                slot = it % 2

                @pl.when(it >= 2)
                def _free():  # the slot's send from two items ago
                    out_send(it - 2, slot, wait_only=True)

                stage[slot] = (o_run[...] / (l_run[0:1, :] * V_SCALE)).T.astype(BF16)
                out_send(it, slot, wait_only=False)
                return carry

            def out_send(it, slot, wait_only):
                lh, qt = it // q_tiles, it % q_tiles
                owner = (qt * BQ) // R
                row = qt * BQ - owner * R
                dst = attn_buf.at[
                    parity, me * LH + lh, pl.ds(pl.multiple_of(row, BQ), BQ)
                ]
                src = stage.at[slot]
                local = copy(src, dst, osem.at[slot, 1])
                if P == 1 or "noremote" in skip:
                    (local.wait if wait_only else local.start)()
                    return
                remote = pltpu.make_async_remote_copy(
                    src, dst, osem.at[slot, 0], attn_recv.at[0], device_id=device(owner)
                )
                if wait_only:
                    pl.when(owner == me)(local.wait)
                    pl.when(owner != me)(remote.wait_send)
                else:
                    pl.when(owner == me)(local.start)
                    pl.when(owner != me)(remote.start)

            items = LH * q_tiles
            lax.fori_loop(0, items, item, None)
            for back in range(min(2, items), 0, -1):
                out_send(items - back, (items - back) % 2, wait_only=True)
            if bulk:

                def wait_bulk(lh_next, c):
                    bulk_sends(lh_next, wait_only=True)
                    return c

                lax.fori_loop(1, LH, wait_bulk, None)

        pl.run_scoped(
            scoped,
            pltpu.VMEM((2, BQ, D), BF16),  # query tiles (this item's, the next one's)
            pltpu.VMEM((2, BKV, D), BF16),
            pltpu.VMEM((2, BKV, D), BF16),
            pltpu.VMEM((8, BQ), F32),  # running max (online)
            pltpu.VMEM((8, BQ), F32),  # running sum
            pltpu.VMEM((D, BQ), F32),  # running output, transposed
            pltpu.VMEM((8, BQ), F32),  # per-query offsets (rowfix)
            pltpu.VMEM((P, 8, 128), F32),  # every device's per-head key peaks
            pltpu.VMEM((2, BQ, D), BF16),
            pltpu.SemaphoreType.DMA((7,)),
            pltpu.SemaphoreType.DMA(
                (2, 2)
            ),  # output sends per stage slot: remote, local
        )

    # ------------------------------------------------------------ phase C
    def phase_c(layer, parity):
        TC = tiling.mlp_rows
        tiles = R // TC
        # Cross-attention heads a K2/V2 group (one DMA each), and a step of its loop.
        CH = next(c for c in (8, 4, 2, 1) if heads % c == 0)
        CS = min(tiling.cross_step, CH)
        groups = heads // CH
        nh = H // TN
        nf = F // TN
        twopass = tiling.mlp_mode == "twopass"
        cross = "cross" not in skip
        mlp_acc = not twopass and "mlp" not in skip
        pipe = tiling.pipelined_epilogues

        def scoped(
            abuf,
            nbuf,
            xbuf,
            acc,
            hbuf,
            k2t,
            v2t,
            sumsq,
            csem,
            ring,
            ring_rows,
            wsem,
            wsem_rows,
            *stage,  # pipelined_epilogues: o and o2 between their steps
        ):
            ostage = stage[0] if pipe else None

            def rows(t):
                return pl.ds(pl.multiple_of(t * TC, TC), TC)

            # Each DMA group has its own semaphore: several are in flight at once.
            def attn_rows(t):
                """Row tile ``t``'s attention output into abuf, a DMA a head."""
                return [
                    copy(
                        attn_buf.at[parity, h, rows(t)],
                        abuf.at[:, pl.ds(h * D, D)],
                        csem.at[0],
                    )
                    for h in range(heads)
                ]

            # Row tile t's x, in its slot of xbuf: in, and back out once updated.
            def x_in(t):
                return copy(x_ref.at[rows(t)], xbuf.at[t % 2], csem.at[1])

            def x_out(t):
                return copy(xbuf.at[t % 2], x_ref.at[rows(t)], csem.at[2])

            def n3_out(t):
                return copy(nbuf, n3_hbm.at[rows(t)], csem.at[3])

            def kv2(nk, g, slot):
                """Cross-attention head group ``g``'s first ``nk`` K2/V2 rows (one DMA
                each)."""
                hs = pl.ds(pl.multiple_of(g * CH, CH), CH)
                ks = pl.ds(0, nk)
                return [
                    copy(
                        k2_ref.at[layer, hs, ks], k2t.at[slot, :, ks], csem.at[4 + slot]
                    ),
                    copy(
                        v2_ref.at[layer, hs, ks], v2t.at[slot, :, ks], csem.at[6 + slot]
                    ),
                ]

            def by_count(fn):
                """``fn(nk)`` for this device's key count: one of ``counts``, else
                ``T`` (every text row)."""
                if not cross or not counts:
                    fn(T)
                    return
                rest = True
                for nk in counts:
                    pl.when(keys == nk)(functools.partial(fn, nk))
                    rest = jnp.logical_and(rest, keys != nk)
                pl.when(rest)(functools.partial(fn, T))

            def weights(w_ref):
                def fetch(i, slot):
                    return wcopies(w_ref.at[layer, i], ring, slot, wsem)

                return fetch

            def fetch_f(f, slot):
                return wcopies(w1_ref.at[layer, f], ring, slot, wsem) + wcopies(
                    w2_ref.at[layer, f], ring_rows, slot, wsem_rows
                )

            # Every stream's tile 0 goes into slot 0, started once the previous
            # stream's loop is done (both slots free): a LayerNorm, the cross-
            # attention or a row tile's end lies between, longer than the copy.
            def proj(w_ref, src, epilogue, split=None):
                """``epilogue(i, cols, src @ W tile i [TC, TN] f32)`` over H/TN tiles,
                tile 0 already started. pipelined_epilogues: ``split = (produce,
                consume)``, the epilogue as ``produce(i, cols, y)`` and ``consume(i,
                cols)`` (``_stream_pipelined``)."""

                def matmul(i, slot):
                    if "cproj" in skip:
                        return jnp.zeros((TC, TN), F32)
                    return jnp.dot(src[...], ring[slot], preferred_element_type=F32)

                def body(i, slot):
                    y = matmul(i, slot)
                    epilogue(i, pl.ds(pl.multiple_of(i * TN, TN), TN), y)

                if not (pipe and split):
                    _stream(nh, weights(w_ref), body)
                    return

                def cols(i):
                    return pl.ds(pl.multiple_of(i * TN, TN), TN)

                produce, consume = split
                _stream_pipelined(
                    nh,
                    weights(w_ref),
                    lambda i, slot: produce(i, cols(i), matmul(i, slot)),
                    lambda i: consume(i, cols(i)),
                )

            # pipelined_epilogues: a tile's rounded projection waits in ostage for
            # its residual add, made in the next tile's step.
            def round_out(bias):
                def produce(i, cols, y):
                    ostage[...] = (y + vec(bias, i * TN, TN)).astype(BF16)

                return produce

            def row_tile(t, carry):
                xv = xbuf.at[t % 2]
                more = t + 1 < tiles

                # Prefetched during the previous row tile (or before the first).
                wait(attn_rows(t) + [x_in(t)])

                # o = bf16(a Wo + bo); x = bf16(x + o * gate)
                def ep_o(i, cols, y):
                    o = (y + vec("bo", i * TN, TN)).astype(BF16).astype(F32)
                    xv[:, cols] = (
                        xv[:, cols].astype(F32) + o * erow(ref.GATE, cols)
                    ).astype(BF16)

                def o_add(i, cols):  # ep_o's residual, from ostage
                    o = ostage[...].astype(F32)
                    xv[:, cols] = (
                        xv[:, cols].astype(F32) + o * erow(ref.GATE, cols)
                    ).astype(BF16)

                with scope("C.wo", 2):
                    proj(wo_ref, abuf, ep_o, (round_out("bo"), o_add))
                start(weights(wq2_ref)(0, 0))

                # The previous row tile's write-backs, waited only now: its x slot
                # takes the next row tile's x, and nbuf this one's n2.
                @pl.when(t > 0)
                def _():
                    x_out(t - 1).wait()
                    if twopass:
                        n3_out(t - 1).wait()

                pl.when(more)(lambda: x_in(t + 1).start())

                def norm2(rs):
                    n = layer_norm_rows(xv, rs)
                    nbuf[rs, :] = (n * vec("n2w") + vec("n2b")).astype(BF16)

                with scope("C.ln2", 2):
                    for_chunks(TC, norm2)
                sumsq[...] = jnp.zeros(sumsq.shape, F32)

                # q2 = bf16(n2 Wq2 + bq2) into abuf; its rms over all heads
                def ep_q2(i, cols, y):
                    q2 = (y + vec("bq2", i * TN, TN)).astype(BF16)
                    abuf[:, cols] = q2
                    qf = q2.astype(F32)
                    sumsq[...] += jnp.sum(qf * qf, axis=1, keepdims=True)

                def q2_store(i, cols, y):  # ep_q2 split: q2 into abuf, then its sum
                    abuf[:, cols] = (y + vec("bq2", i * TN, TN)).astype(BF16)

                def q2_sumsq(i, cols):
                    qf = abuf[:, cols].astype(F32)
                    sumsq[...] += jnp.sum(qf * qf, axis=1, keepdims=True)

                with scope("C.wq2", 2):
                    proj(wq2_ref, nbuf, ep_q2, (q2_store, q2_sumsq))
                start(weights(wo2_ref)(0, 0))
                factor = lax.rsqrt(sumsq[...] / H + eps)

                # Cross-attention into nbuf over the first nk text keys (one loop per
                # key count; this device's count picks one): K2/V2 by groups of CH
                # heads, a group in steps of CS heads (independent chains the scheduler
                # interleaves; rolled steps keep this row-tile loop, every count's loop
                # included, within an instruction overlay). With fewer than T keys,
                # each key's softmax term is multiplied by its weight (the text rows it
                # stands for); the copies of a repeated key score alike, so the max is
                # the one over every row.
                def cross_attention(nk):
                    ks = pl.ds(0, nk)

                    def head(g, slot, j):
                        h = g * CH + j
                        cols = pl.ds(pl.multiple_of(h * D, D), D)
                        qh = (
                            abuf[:, cols].astype(F32) * factor * vec("qn2", h * D, D)
                        ).astype(BF16)
                        s = (
                            lax.dot_general(
                                k2t[slot, j, ks], qh, LAST, preferred_element_type=F32
                            )
                            * scale_l2
                        )
                        m = jnp.max(s, axis=0, keepdims=True)
                        p = jnp.exp2(s - m)
                        if nk < T:
                            p = p * kw_v[ks, :]
                        l = jnp.sum(p, axis=0, keepdims=True)
                        o = lax.dot_general(
                            v2t[slot, j, ks],
                            p.astype(BF16),
                            FIRST,
                            preferred_element_type=F32,
                        )
                        nbuf[:, cols] = (o / l).T.astype(BF16)

                    def head_group(g, slot):
                        def step(i, c):
                            for j in range(CS):
                                head(g, slot, i * CS + j)
                            return c

                        if CS == CH:
                            step(0, None)
                        else:
                            lax.fori_loop(0, CH // CS, step, None)

                    _stream(groups, functools.partial(kv2, nk), head_group)

                if cross:
                    with scope("C.cross", 2):
                        by_count(cross_attention)

                # The next row tile's first head group, and its attention rows (abuf
                # is free now).
                @pl.when(more)
                def _():
                    if cross:
                        by_count(lambda nk: start(kv2(nk, 0, 0)))
                    start(attn_rows(t + 1))

                # o2 = bf16(a2 Wo2 + bo2); x = x + o2 (a bf16 add)
                def ep_o2(i, cols, y):
                    o2 = (y + vec("bo2", i * TN, TN)).astype(BF16).astype(F32)
                    xv[:, cols] = (xv[:, cols].astype(F32) + o2).astype(BF16)

                def o2_add(i, cols):  # ep_o2's residual, from ostage
                    o2 = ostage[...].astype(F32)
                    xv[:, cols] = (xv[:, cols].astype(F32) + o2).astype(BF16)

                with scope("C.wo2", 2):
                    proj(wo2_ref, nbuf, ep_o2, (round_out("bo2"), o2_add))
                if mlp_acc:
                    start(fetch_f(0, 0))
                else:
                    pl.when(more)(lambda: start(weights(wo_ref)(0, 0)))

                def norm3(rs):
                    n = layer_norm_rows(xv, rs)
                    nbuf[rs, :] = (
                        n * (1.0 + erow(ref.C_SCALE)) + erow(ref.C_SHIFT)
                    ).astype(BF16)

                # twopass: x2 and n3 to HBM, waited a row tile later; the MLP runs in
                # phase C2.
                if twopass:
                    x_out(t).start()  # LN3 only reads x
                    with scope("C.ln3", 2):
                        for_chunks(TC, norm3)
                    n3_out(t).start()
                    return carry

                with scope("C.ln3", 2):
                    for_chunks(TC, norm3)

                # MLP: acc = sum_f bf16(gelu(bf16(n3 W1_f + b1_f))) W2_f   (bf16 weights only)
                acc[...] = jnp.zeros(acc.shape, F32)

                def mlp(f, slot):
                    hpre = jnp.dot(nbuf[...], ring[slot], preferred_element_type=F32)
                    hpre = (hpre + vec("b1", f * TN, TN)).astype(BF16).astype(F32)
                    hbuf[...] = ref.gelu_tanh(hpre).astype(BF16)
                    acc[...] += jnp.dot(
                        hbuf[...], ring_rows[slot], preferred_element_type=F32
                    )

                if mlp_acc:
                    with scope("C.mlp", 2):
                        _stream(nf, fetch_f, mlp)
                    pl.when(more)(lambda: start(weights(wo_ref)(0, 0)))

                def out(rs):
                    f_ = (acc[rs, :] + vec("b2")).astype(BF16).astype(F32)
                    xv[rs, :] = (xv[rs, :].astype(F32) + f_ * erow(ref.C_GATE)).astype(
                        BF16
                    )

                with scope("C.out", 2):
                    for_chunks(TC, out)
                x_out(t).start()
                return carry

            # The first row tile's weight tile, K2/V2 group and x stream in while
            # the attention rows arrive.
            start(weights(wo_ref)(0, 0))
            if cross:
                by_count(lambda nk: start(kv2(nk, 0, 0)))
            x_in(0).start()
            if P > 1:  # every peer's heads' output for my rows (mine were waited)
                with scope("C.wait", 2):
                    await_bytes(
                        attn_buf.at[parity, pl.ds(0, (P - 1) * LH)], attn_recv.at[0]
                    )
            start(attn_rows(0))
            lax.fori_loop(0, tiles, row_tile, None)
            x_out(tiles - 1).wait()
            if twopass:
                n3_out(tiles - 1).wait()

        small = (8, 128)
        pl.run_scoped(
            scoped,
            pltpu.VMEM((TC, H), BF16),  # a, then q2
            pltpu.VMEM((TC, H), BF16),  # n2, then a2, then n3
            pltpu.VMEM((2, TC, H), BF16),  # x, by row-tile parity
            pltpu.VMEM(small if twopass else (TC, H), F32),  # MLP accumulator
            pltpu.VMEM(small if twopass else (TC, TN), BF16),  # gelu tile
            pltpu.VMEM((2, CH, cfg.text_len, D), BF16),  # cross K, V: CH heads a slot
            pltpu.VMEM((2, CH, cfg.text_len, D), BF16),
            pltpu.VMEM((TC, 1), F32),
            # attention rows, x in, x out, n3 out, K2 slots, V2 slots
            pltpu.SemaphoreType.DMA((8,)),
            pltpu.VMEM((2, H, TN), BF16),  # weight ring
            pltpu.VMEM((1, 8, 128) if twopass else (2, TN, H), BF16),  # MLP down ring
            pltpu.SemaphoreType.DMA((2, NS)),
            pltpu.SemaphoreType.DMA((2, NS)),
            *([pltpu.VMEM((TC, TN), BF16)] if pipe else []),
        )
        if twopass:
            phase_c2(layer)

    # ------------------------------------------------------------ phase C2
    def phase_c2(layer):
        """The MLP in two passes a row tile (``mlp_mode="twopass"``): pass 1 the up
        projection and GELU of every column into ``h [TC, F]``, pass 2 the down
        projection in ``down_cols``-wide output tiles (the whole ``F`` contraction in
        one dot, no accumulator), each with its gated residual. A row tile's n3 is
        read into its own buffer during the previous one's pass 2."""
        if "mlp" in skip:
            return
        TC = tiling.mlp_rows
        TD = tiling.down_cols
        tiles = R // TC
        nf, nd = F // TN, H // TD
        # pipelined_epilogues: a tile's GELU in the next tile's step, its
        # pre-activation waiting in hs[0] (_stream_pipelined).
        pipe = tiling.pipelined_epilogues
        US = 2  # up-projection weight ring slots

        def outer(hfull, n3v, c2sem):
            def n3_in(t):
                return copy(
                    n3_hbm.at[pl.ds(pl.multiple_of(t * TC, TC), TC)],
                    n3v,
                    c2sem.at[0],
                )

            def row_tile(t, carry):
                r0 = pl.multiple_of(t * TC, TC)

                def pass1(ring1, wsem1, hs):
                    def w1_tile(f, slot):
                        return wcopies(w1_ref.at[layer, f], ring1, slot, wsem1)

                    def up(f, slot):
                        """bf16(n3 W1_f + b1_f)."""
                        hp = jnp.dot(n3v[...], ring1[slot], preferred_element_type=F32)
                        return (hp + vec("b1", f * TN, TN)).astype(BF16)

                    def gelu(f, hp):
                        hfull[:, pl.ds(pl.multiple_of(f * TN, TN), TN)] = ref.gelu_tanh(
                            hp.astype(F32)
                        ).astype(BF16)

                    # The first weight tile loads during n3's wait.
                    start(w1_tile(0, 0))
                    n3_in(t).wait()
                    if pipe:

                        def up_out(f, slot):
                            hs[0] = up(f, slot)

                        with scope("C2.up", 2):
                            _stream_pipelined(
                                nf, w1_tile, up_out, lambda f: gelu(f, hs[0])
                            )
                        return
                    with scope("C2.up", 2):
                        _stream(
                            nf,
                            w1_tile,
                            lambda f, slot: gelu(f, up(f, slot)),
                        )

                def pass2(ring2, wsem2, xt, xsem):
                    def xcopy(j, out):
                        """Output tile ``j``'s x columns between HBM and x slot ``j % 3``
                        (in, or back out once updated)."""
                        cols = pl.ds(pl.multiple_of(j * TD, TD), TD)
                        xs = j % 3
                        src, dst = (
                            (xt.at[xs], x_ref.at[pl.ds(r0, TC), cols])
                            if out
                            else (x_ref.at[pl.ds(r0, TC), cols], xt.at[xs])
                        )
                        return copy(src, dst, xsem.at[xs, 1 if out else 0])

                    def fetch(j):
                        return wcopies(w2_ref.at[layer, j], ring2, j % 2, wsem2) + [
                            xcopy(j, False)
                        ]

                    start(fetch(0))
                    # n3's buffer is free: the next row tile's n3, waited in its
                    # pass 1.
                    pl.when(t + 1 < tiles)(n3_in(t + 1).start)

                    def down(j, carry):
                        for c in fetch(j):
                            c.wait()

                        @pl.when(j + 1 < nd)
                        def _next():
                            @pl.when(j >= 2)
                            def _():  # tile j + 1's x slot is tile j - 2's: its write-back first
                                xcopy(j - 2, True).wait()

                            for c in fetch(j + 1):
                                c.start()

                        y = jnp.dot(
                            hfull[...], ring2[j % 2], preferred_element_type=F32
                        )
                        cols = pl.ds(pl.multiple_of(j * TD, TD), TD)
                        f_ = (y + vec("b2", j * TD, TD)).astype(BF16).astype(F32)
                        xs = j % 3
                        xt[xs] = (
                            xt[xs].astype(F32) + f_ * erow(ref.C_GATE, cols)
                        ).astype(BF16)
                        xcopy(j, True).start()
                        return carry

                    with scope("C2.down", 2):
                        lax.fori_loop(0, nd, down, None)
                    for j in range(
                        max(0, nd - 3), nd
                    ):  # write-backs not waited in the loop
                        xcopy(j, True).wait()

                pl.run_scoped(
                    pass1,
                    pltpu.VMEM((US, H, TN), BF16),
                    pltpu.SemaphoreType.DMA((US, NS)),
                    # pipe: the pre-activation waiting for its GELU
                    pltpu.VMEM((1, TC, TN) if pipe else (1, 16, 128), BF16),
                )
                pl.run_scoped(
                    pass2,
                    pltpu.VMEM((2, F, TD), BF16),
                    pltpu.SemaphoreType.DMA((2, NS)),
                    pltpu.VMEM((3, TC, TD), BF16),
                    pltpu.SemaphoreType.DMA((3, 2)),
                )
                return carry

            n3_in(0).start()
            lax.fori_loop(0, tiles, row_tile, None)

        pl.run_scoped(
            outer,
            pltpu.VMEM((TC, F), BF16),
            pltpu.VMEM((TC, H), BF16),  # n3
            pltpu.SemaphoreType.DMA((1,)),
        )

    # ------------------------------------------------------------ layers
    # The cross keys' weights (the same every layer), in while the peers arrive.
    if counts:
        cw = copy(kw_ref.at[pl.ds(0, kw_v.shape[0])], kw_v, sems.at[2])
        cw.start()
    if P > 1:
        # Every peer must be inside the kernel before anyone writes into it.
        barrier = pltpu.get_barrier_semaphore()
        for off in range(1, P):
            pl.semaphore_signal(barrier, 1, device_id=device((me + off) % P))
        pl.semaphore_wait(barrier, P - 1)
    if counts:
        cw.wait()

    def layer_body(layer, carry):
        parity = layer % 2
        cv = copy(vecs_ref.at[layer], vec_v, sems.at[0])
        ce = copy(e_ref.at[layer], e_v, sems.at[1])
        cv.start()
        ce.start()
        cv.wait()
        ce.wait()
        with scope("A"):
            phase_a(layer, parity)
        with scope("B"):
            phase_b(layer, parity)
        with scope("C"):
            phase_c(layer, parity)
        return carry

    lax.fori_loop(0, L, layer_body, None)


def stack(
    x: jax.Array,
    e: jax.Array,
    k2: jax.Array,
    v2: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    weights: dict[str, jax.Array],
    valid: int | jax.Array,
    keys: int | jax.Array | None = None,
    key_weights: jax.Array | None = None,
    *,
    cfg: ref.Config,
    tiling: Tiling,
    axis: str | None = None,
    devices: int = 1,
    interpret: Any = False,
    softmax: str = "auto",
    skip: frozenset = frozenset(),
    trace: int = 0,
) -> jax.Array:
    """Every block on this device's rows (call inside ``shard_map`` when
    ``devices > 1``).

    Args:
      x: ``[R, H]`` bf16, this device's consecutive rows of the padded sequence.
      e: ``[L, 8, H]`` f32 modulation rows (``scale_shift_table + temb``; rows
        6-7 unused).
      k2, v2: ``[L, heads, T_ctx, D]`` bf16, ``reference.cross_kv`` per layer,
        head-major.
      cos, sin: ``[R, D]`` f32, this device's rows of ``reference.rope_tables``.
      weights: ``pack_kernel_weights``.
      valid: real rows of the whole sequence; later rows are padding (masked
        as keys, their outputs undefined).
      keys, key_weights: the cross-attention keys (``pack.cross_keys``), both or
        neither. With ``keys`` one of ``tiling.cross_keys`` below ``T_ctx``, it
        attends to the first ``keys`` text rows only, each key's softmax term
        times its ``key_weights [T_ctx, 1]`` f32 row: a key whose K2/V2 row
        repeats in every later text row stands for all of them (the prompt's
        padding rows), so the result is the all-rows softmax up to f32 rounding
        of its sums. Any other ``keys``, or neither (the default): all ``T_ctx``
        rows, unweighted. A device's own count picks its code path at run time,
        so CFG branches with different prompts share one compiled kernel.
    """
    R, H = x.shape
    P = devices
    tiling.check(R, P, H, cfg.ffn, cfg.heads, cfg.text_len)
    LH = cfg.heads // P
    S = R * P
    T = k2.shape[2]
    if (keys is None) != (key_weights is None):
        raise ValueError("stack: pass keys and key_weights together (pack.cross_keys)")
    if keys is None:
        keys, key_weights = T, jnp.ones((T, 1), F32)
    if jnp.size(keys) != 1 or not jnp.issubdtype(jnp.asarray(keys).dtype, jnp.integer):
        raise ValueError(f"stack: keys must be one integer, got {keys!r}")
    if key_weights.shape != (T, 1) or key_weights.dtype != F32:
        raise ValueError(
            f"stack: key_weights must be [{T}, 1] float32, got "
            f"{list(key_weights.shape)} {key_weights.dtype}"
        )
    meta = jnp.concatenate(
        [
            jnp.asarray(valid, jnp.int32).reshape(1),
            jnp.asarray(keys, jnp.int32).reshape(1),
        ]
    )
    kernel = functools.partial(
        _kernel,
        cfg=cfg,
        tiling=tiling,
        axis=axis,
        devices=P,
        softmax=softmax,
        skip=frozenset(skip),
        trace=trace,
    )
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    operands = (
        x,
        e,
        k2,
        v2,
        key_weights,
        cos,
        sin,
        *(weights[n] for n in KERNEL_WEIGHTS),
    )
    nv = weights["vecs"].shape[-1]
    out = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((R, H), BF16),
            jax.ShapeDtypeStruct((2, 3, LH, S, cfg.head_dim), BF16),
            jax.ShapeDtypeStruct((2, cfg.heads, R, cfg.head_dim), BF16),
            jax.ShapeDtypeStruct(
                (3, P, max(LH - 1, 1), R, cfg.head_dim), BF16
            ),  # kv_mine
            jax.ShapeDtypeStruct(
                (R, H) if tiling.mlp_mode == "twopass" else (8, 128), BF16
            ),  # n3 (C1 -> C2)
            jax.ShapeDtypeStruct(
                (2, P, 8, 128), F32
            ),  # every device's per-head key peaks
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * len(operands),
            out_specs=[hbm] * 6,
            scratch_shapes=[
                pltpu.VMEM((1, nv), F32),
                pltpu.VMEM((8, H), F32),
                # the cross keys' weights (the most a key count reads)
                pltpu.VMEM((max(cross_counts(tiling, T), default=8), 1), F32),
                pltpu.SemaphoreType.DMA((3,)),  # vecs, e, key weights
                pltpu.SemaphoreType.DMA((1,)),  # bulk sends (phase B)
                pltpu.SemaphoreType.DMA((LH,)),  # kv receive, one per own head
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((1,)),  # key peaks receive
                pltpu.SMEM((1,), jnp.int32),  # phase B's row-sum floor flag
            ],
        ),
        input_output_aliases={1: 0},
        compiler_params=pltpu.CompilerParams(
            collective_id=_COLLECTIVE_ID if P > 1 else None,
            vmem_limit_bytes=tiling.vmem_limit_bytes,
            **({} if tiling.dma_bounds_checks else {"disable_bounds_checks": True}),
        ),
        interpret=interpret,
        name="wan_stack",
    )(meta, *operands)
    return out[0]
