# SPDX-License-Identifier: Apache-2.0
"""SP + TP x_norm all-gather and qkvg card as one Pallas kernel.

The e4m3 own rows go to every rank by in-kernel remote DMA (common.coll.GridColl's grid route:
x, y, then the partner core) one row tile at a time, and each tile's qkvg runs as soon as it has
landed, so the gather of the next tiles hides under the matmul (XLA's AG plus a chunked Pallas
qkvg overlapped only partly). Tile j holds rows [j * seg, (j + 1) * seg) of every rank's block.
The layer's wqkvg stays in VMEM for the whole kernel (one HBM read per layer, not per chunk).
"""

import contextlib

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel_v2.common.coll import GridColl, ag_kinds, run_pipeline
from sglang_tpu.muse_megakernel_v2.common.config import PFConfig
from sglang_tpu.muse_megakernel_v2.common.consts import (
    SPTP_XAG_QKVG_COLLECTIVE_ID,
    small_row,
)
from sglang_tpu.muse_megakernel_v2.common.numerics import _rope_roll
from sglang_tpu.muse_megakernel_v2.common.qkv import qkv_from_proj
from sglang_tpu.muse_megakernel_v2.common.trace import _scope
from sglang_tpu.muse_megakernel_v2.prefill.sptp.qkvg import _q_rpa, qkv_epilogue

# Barrier id of this kernel; distinct from common.consts' (0, 6, 7, 9-12).
_FP8_MAX = 448.0


def _split_dot(a, b, dims=(((1,), (0,)), ((), ()))):
    """f32 a @ 0/1 matrix b as two bf16 MXU passes (a's bf16 head plus its bf16 residual):
    relative error ~2^-17, a third of Precision.HIGHEST's passes."""
    hi = a.astype(jnp.bfloat16)
    lo = (a - hi.astype(jnp.float32)).astype(jnp.bfloat16)
    bb = b.astype(jnp.bfloat16)
    f32 = jnp.float32
    return jax.lax.dot_general(
        hi, bb, dims, preferred_element_type=f32
    ) + jax.lax.dot_general(lo, bb, dims, preferred_element_type=f32)


def _heads_rms_split(x, n_heads: int, d: int, eps: float, gain=1.0):
    """qkvg._heads_rms_mxu with the two head-matrix dots as _split_dot."""
    w = n_heads * d
    blk = (
        jax.lax.broadcasted_iota(jnp.int32, (w, 128), 0) // d
        == jax.lax.broadcasted_iota(jnp.int32, (w, 128), 1)
    ).astype(jnp.float32)
    inv = jax.lax.rsqrt(_split_dot(x * x, blk) * (1.0 / d) + eps)
    inv_w = _split_dot(inv, blk, (((1,), (1,)), ((), ())))
    return x * inv_w * gain


def _rope_tiled(x, n_heads: int, d: int, cos_x, sin_x):
    """numerics._rope_roll with the tables widened once to a 128-lane vreg (two heads) and the
    pair sign folded into the sin table, so the per-head copies are whole vregs."""
    w = n_heads * d
    if 128 % d or w % 128:
        return _rope_roll(x, n_heads, d, cos_x, sin_x)
    r = 128 // d
    lane = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 128), 1)
    even = (lane % 2) == 0
    c1 = jnp.concatenate([cos_x] * r, axis=1) if r > 1 else cos_x
    s1 = jnp.concatenate([sin_x] * r, axis=1) if r > 1 else sin_x
    s1 = jnp.where(even, -s1, s1)
    c = jnp.concatenate([c1] * (w // 128), axis=1)
    sg = jnp.concatenate([s1] * (w // 128), axis=1)
    lw = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    partner = jnp.where(
        (lw % 2) == 0,
        pltpu.roll(x, shift=w - 1, axis=1),
        pltpu.roll(x, shift=1, axis=1),
    )
    return x * c + partner * sg


def qkv_epilogue_split(c: PFConfig, qkvg, rope):
    """qkvg.qkv_epilogue with _heads_rms_split and _rope_tiled."""
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    q = _heads_rms_split(qkvg[:, :qd], c.q_heads, hd, c.rms_eps, gain=c.scale_query_by)
    k = _heads_rms_split(qkvg[:, qd : qd + kvd], c.kv_heads, hd, c.rms_eps)
    if rope is not None:
        q = _rope_tiled(q, c.q_heads, hd, rope[:, :hd], rope[:, hd:])
        k = _rope_tiled(k, c.kv_heads, hd, rope[:, :hd], rope[:, hd:])
    kv = jnp.concatenate([k, qkvg[:, qd + kvd : qd + 2 * kvd]], axis=1)
    return (
        q.astype(jnp.bfloat16),
        kv.astype(jnp.bfloat16),
        qkvg[:, qd + 2 * kvd :].astype(jnp.bfloat16),
    )


def _e4m3_rne(v):
    """f32 v (|v| <= 448) rounded to the nearest e4m3fn value, ties to even, still f32: the
    cast after it is exact whatever rounding the kernel's f32 -> e4m3 conversion uses."""
    b = pltpu.bitcast(v, jnp.int32)
    # normals: keep 3 of the 23 mantissa bits
    lsb = jax.lax.shift_right_logical(b, 20) & 1
    normal = pltpu.bitcast((b + (0x7FFFF + lsb)) & -(1 << 20), jnp.float32)
    # subnormals (|v| < 2^-6): multiples of 2^-9; every step below is exact
    t = jnp.abs(v) * 512.0
    n = t.astype(jnp.int32)
    fr = t - n.astype(jnp.float32)
    up = (fr > 0.5) | ((fr == 0.5) & ((n & 1) == 1))
    sub = (n + up.astype(jnp.int32)).astype(jnp.float32) * (1.0 / 512.0)
    sub = jnp.where(v < 0, -sub, sub)
    return jnp.where(jnp.abs(v) < 2.0**-6, sub, normal)


def _pipeline(
    stages: tuple,
    order: tuple,
    nt: int,
    gap: int,
    off: int = 0,
    pre=None,
    n_pre: int = 0,
) -> None:
    """common.coll.run_pipeline with a chosen stage order per iteration: stage k runs on tile
    i - k * (1 + gap). Running the DMA relays before the compute gives every relay's sends a
    whole compute to land (run_pipeline's later-stages-first order issues the partner share
    right after the compute that waits for it)."""
    lag = 1 + gap
    n_it = max(nt + (len(stages) - 1) * lag, n_pre)
    # pre(i), i < n_pre: extra compute in the iterations before the last stage's first tile
    if pre is not None and n_pre > (len(stages) - 1) * lag:
        raise ValueError(f"{n_pre} pre tiles do not fit the fill iterations")

    def body(i, carry):
        for k in order:
            j = i - k * lag

            @pl.when(jnp.logical_and(j >= 0, j < nt))
            def _(st=stages[k], j=j):
                with _scope(f"pf_{st.__name__}"):
                    st(j + off)

        if pre is not None:
            pl.when(i < n_pre)(lambda: pre(i))
        return carry

    jax.lax.fori_loop(0, n_it, body, 0)


class _Multi:
    """Local copies started and waited together (one tile's loads)."""

    def __init__(self, cps):
        self.cps = cps

    def start(self):
        for cp in self.cps:
            cp.start()

    def wait(self):
        for cp in self.cps:
            cp.wait()


class _Copies:
    """Several remote copies on one semaphore pair, started and waited together."""

    def __init__(self, cps):
        self.cps = cps

    def start(self):
        for cp in self.cps:
            cp.start()

    def wait_recv(self):
        for cp in self.cps:
            cp.wait_recv()

    def wait_send(self):
        for cp in self.cps:
            cp.wait_send()


class _SplitColl(GridColl):
    """GridColl whose row-block copies go as n column slices: one DMA stream per block moved
    a 256 KB block at about half the link rate."""

    def __init__(self, tp, me, sems, n: int, width: int, route: int = 0):
        super().__init__(tp, me, sems)
        self.n, self.width, self.route = n, width, route

    def masks(self):
        """route 0: GridColl's (core parity picks x or y first); 1 / 2: every core x / y first;
        3: parity swapped (diagnostics of the x and y link rates)."""
        if self.route == 0:
            return super().masks()
        if self.route == 3:
            m1 = jnp.where(self.core == 0, 1, 2).astype(jnp.int32)
        else:
            m1 = jnp.full((), 2 if self.route == 1 else 1, jnp.int32)
        return m1, 3 - m1

    # blocks seen by the receiver: with own_first a rank's block sits at a position relative
    # to the viewing rank, so a remote copy's destination is the peer's position, not ours
    @staticmethod
    def _blk(pad, slot, r, viewer):
        if isinstance(pad, _RowBlocks):
            return pad.view(slot, r, viewer)
        return pad.at[slot, r]

    def ag2_first(self, kind, slot, pad):
        m1, _ = self.masks()
        nb = self._nbr(m1)
        b = self._blk
        return [
            self._cp(
                kind + "_x",
                slot,
                b(pad, slot, self.me, self.me),
                b(pad, slot, self.me, nb),
                nb,
            )
        ]

    def ag2_second(self, kind, slot, pad):
        m1, m2 = self.masks()
        nb = self._nbr(m2)
        b = self._blk
        return [
            self._cp(
                kind + "_y", slot, b(pad, slot, r, self.me), b(pad, slot, r, nb), nb
            )
            for r in (self.me, self._nbr(m1))
        ]

    def ag_pair(self, kind, slot, pad):
        b = self._blk
        out = []
        for ch in range(self.nchip):
            r = self.rank(ch, self.core)
            out.append(
                self._cp(
                    kind + "_pair",
                    slot,
                    b(pad, slot, r, self.me),
                    b(pad, slot, r, self.partner),
                    self.partner,
                )
            )
        return out

    def _cp(self, kind, slot, src, dst, peer):
        if self.n == 1 or src.shape[-1] != self.width:
            return super()._cp(kind, slot, src, dst, peer)
        w = self.width // self.n
        return _Copies(
            [
                super(_SplitColl, self)._cp(
                    kind,
                    slot,
                    src.at[:, pl.ds(i * w, w)],
                    dst.at[:, pl.ds(i * w, w)],
                    peer,
                )
                for i in range(self.n)
            ]
        )


class _RowBlocks:
    """pad [S, tp * seg, W] seen as GridColl's [S, tp] blocks: block (s, r) = rows of rank r."""

    def __init__(self, ref, seg: int, pos=None, tp: int = 8):
        self.ref, self.seg, self.pos, self.tp = ref, seg, pos, tp

    def view(self, s, r, viewer):
        """Block of rank r as rank viewer lays out its pad."""
        if self.pos is None:
            return self[s, r]
        p = jax.lax.rem(r - viewer + self.tp, self.tp)
        return self.ref.at[s, pl.ds(pl.multiple_of(p * self.seg, self.seg), self.seg)]

    @property
    def at(self):
        return self

    def __getitem__(self, idx):
        s, r = idx
        if self.pos is not None:
            r = self.pos(r)
        r0 = (
            r * self.seg
            if isinstance(r, int)
            else pl.multiple_of(r * self.seg, self.seg)
        )
        return self.ref.at[s, pl.ds(r0, self.seg)]


def xag_qkvg_call(
    c: PFConfig,
    seg: int,
    rope: bool,
    *,
    mxu_norm: bool = True,
    kc: int = 512,
    q_rpa: bool = True,
    gap: int = 0,
    fuse_norm: bool = False,
    axis: str = "model",
    interpret=None,
    dbg_xs: bool = False,
    dma_first: bool = True,
    split_epi: bool = False,
    exact_cast: bool = False,
    gate4: bool = False,
    w_start: int = 0,
    m_split: int = 1,
    dma_split: int = 1,
    n_tail: bool = False,
    dbg_ag_only: bool = False,
    route: int = 0,
    own_first: bool = False,
    q_strided: bool = False,
    n_pre: int = 0,
    wk: int = 1,
    defer_res: bool = False,
    kv_rpa: bool = False,
):
    """(layer [1] i32, x8 [Ts, H] e4m3 own rows, xs [T, 1] f32 every row's scale, rope [T, 2 hd]
    f32, stacked wqkvg [L, H, nq] e4m3 and smalls [L, 16, H] f32) -> (q, kv [T, 2 kvd], gate
    [T, qd]) bf16, q [Hkv, T, G, 128] (q_rpa) or [T, qd]; rope False: a NoPE layer. gate4: gate
    in q's [Hkv, T, G, 128] layout too (head lanes zero-padded), the o gate's head layout.
    fuse_norm: (layer, stream [Ts, H] f32 own rows, rope, wqkvg, smalls) instead; the pre-attention
    rms and e4m3 row quant (program._rms_fp8_rows) run per tile here and the row scales travel
    with the rows (one [8, 128] f32 tile per rank block, scales in row 0's first seg lanes)."""
    T, H, tp, nq = c.rows, c.hidden, c.tp, c.qkvg_dim
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    kvh, gq = c.kv_heads, c.q_heads // c.kv_heads
    Ts = T // tp
    tm = tp * seg
    nt = Ts // seg
    # fp8 sublane tile is 32 rows: a rank block of seg rows must be whole tiles
    if Ts % seg or seg % 32 or H % kc:
        raise ValueError(f"{Ts} rows per rank / hidden {H} do not tile by {seg} / {kc}")
    # a peer runs at most one stage spacing ahead per stage (post_moe's / attn_out's sizing)
    n_slots = 4 * (1 + gap) + 1
    if fuse_norm and seg > 128:
        raise ValueError(f"{seg} row scales do not fit one lane row")
    kinds = ag_kinds("x") + (ag_kinds("xsc") if fuse_norm else ())
    # own_first: own rows sit at tile position 0 and their qkvg (all tiles, two halves of Ts / 2
    # rows) runs in the fill iterations, when the TensorCore only waits for the first tiles'
    # gather chain; each tile then computes only the 7 gathered blocks
    half = Ts // 2
    # q_strided: q as [Hkv, T * G, 128] (the same bytes as [Hkv, T, G, 128] in HBM): head g of
    # every row goes to rows g, g + G, ... by a sublane-strided f32 store, then one cast, instead
    # of _q_rpa's 0/1 dot and [rows, G, 128] relayout
    # n_pre: tiles 0..n_pre-1 arrive gathered (post_ffn.py); call (lyr, stream, xg, xsc, rope,
    # wqkvg, smalls); they are computed in the gather's fill iterations
    # defer_res (with n_pre): the previous layer's FFN residual of tiles >= n_pre runs here, per
    # tile before its norm (post_ffn.py defer_res hands over f and the inverse rms); call
    # (lyr, stream_old, stream_new (pre tiles written; aliased to the 4th output), f, inv, xg,
    # xsc, rope, wqkvg, smalls) -> (q, kv, gate, stream_new)
    if defer_res and not n_pre:
        raise ValueError("defer_res needs n_pre")
    if wk > 1 and (not n_pre or H % (wk * kc) or n_tail):
        raise ValueError(f"wk {wk}: needs n_pre and H % (wk * kc) == 0")
    if n_pre and (not fuse_norm or own_first or dbg_xs or not dma_first or n_pre >= nt):
        raise ValueError("n_pre needs fuse_norm, dma_first and not own_first / dbg_xs")
    # kv_rpa: kv as batched RPA's new-KV operand, e4m3 [2 Hkv (k0, v0, k1, v1), D / 4, 4, T]
    # (sptp/rpa.py new_kv of the same values): per tile the rows are transposed, cast and
    # placed in a VMEM image of the whole operand, written once at the end
    if kv_rpa and (own_first or seg % 32 or Ts % 128 or 128 % seg or hd % 4):
        raise ValueError("kv_rpa needs no own_first, seg | 128 and Ts % 128 == 0")
    if q_strided and (not q_rpa or own_first or hd != 64):
        raise ValueError("q_strided needs q_rpa, head_dim 64 and not own_first")
    if own_first and (not fuse_norm or dbg_xs or nt < 3 or half % 128 or seg != 32):
        raise ValueError(
            "own_first needs fuse_norm, seg 32, >= 3 tiles and Ts / 2 % 128 == 0"
        )

    pre_refs = {}

    def kernel(lyr, *refs):
        if n_pre and defer_res:
            (
                x_hbm,
                _stn_in,
                pre_refs["f"],
                pre_refs["inv"],
                pre_refs["xg"],
                pre_refs["xsc"],
                rope_hbm,
                w_hbm,
                sm_hbm,
                q_o,
                kv_o,
                g_o,
                pre_refs["stn"],
                *rest,
            ) = refs
            rest = [q_o, kv_o, g_o] + rest
            xs_hbm = None
        elif n_pre:
            x_hbm, pre_refs["xg"], pre_refs["xsc"], rope_hbm, w_hbm, sm_hbm, *rest = (
                refs
            )
            xs_hbm = None
        elif fuse_norm and not dbg_xs:
            x_hbm, rope_hbm, w_hbm, sm_hbm, *rest = refs
            xs_hbm = None
        else:
            x_hbm, xs_hbm, rope_hbm, w_hbm, sm_hbm, *rest = refs
        body(lyr, x_hbm, xs_hbm, rope_hbm, w_hbm, sm_hbm, *rest)

    def body(
        lyr,
        x_hbm,
        xs_hbm,
        rope_hbm,
        w_hbm,
        sm_hbm,
        q_hbm,
        kv_hbm,
        g_hbm,
        w_v,
        sm_v,
        pad,
        xs_v,
        rope_v,
        q_v,
        kv_v,
        g_v,
        w_sem,
        own_sem,
        in_sem,
        out_sem,
        *rest,
    ):
        if fuse_norm:
            xin_v, sc_pad, *rest = rest
        else:
            xin_v, sc_pad = None, None
        if own_first:
            (
                own8,
                own_sc,
                orope_v,
                oq_v,
                okv_v,
                og_v,
                orope_sem,
                oout_sem,
                *coll_sems,
            ) = rest
        else:
            coll_sems = rest
        qs_v = None
        if defer_res:
            *coll_sems, fin_v, iv_v, xo_v, smp_v, fsem, st_sem, smp_sem = coll_sems
        if wk > 1:
            *coll_sems, wk_sem = coll_sems
        if n_pre:
            *coll_sems, pre_v, presc_v, pre_sem = coll_sems
        if q_strided:
            *coll_sems, qs_v = coll_sems
        if kv_rpa:
            *coll_sems, kvn_v, kvn_sem = coll_sems
        me = jax.lax.axis_index(axis).astype(jnp.int32)

        def slot_of(j):
            return (j - n_pre) % n_slots

        def st_write(j):
            return pltpu.make_async_copy(
                xo_v.at[j % 2],
                pre_refs["stn"].at[pl.ds(pl.multiple_of(j * seg, seg), seg)],
                st_sem.at[j % 2],
            )

        def pre_loads(i):
            b = i % 2
            return [
                pltpu.make_async_copy(
                    pre_refs["xg"].at[i], pre_v.at[b], pre_sem.at[0, b]
                ),
                pltpu.make_async_copy(
                    pre_refs["xsc"].at[i], presc_v.at[b], pre_sem.at[1, b]
                ),
            ]

        def rank_of(p: int):
            return jax.lax.rem(me + p, tp) if own_first else p

        def pos_of(r):
            return jax.lax.rem(r - me + tp, tp)

        col = _SplitColl(
            tp,
            me,
            {k: coll_sems[2 * i : 2 * i + 2] for i, k in enumerate(kinds)},
            dma_split,
            H,
            route,
        )
        blocks = _RowBlocks(pad, seg, pos_of if own_first else None, tp)
        kp = (("x", blocks),) + ((("xsc", sc_pad),) if fuse_norm else ())
        l = lyr[0]
        # smalls first: the fused norm of tile 0 needs them, the dot not before iteration 3
        w_cps = [
            pltpu.make_async_copy(sm_hbm.at[l], sm_v, w_sem.at[1]),
            pltpu.make_async_copy(w_hbm.at[l], w_v, w_sem.at[0]),
        ]

        def own_load(j):
            rows = x_hbm.at[pl.ds(pl.multiple_of(j * seg, seg), seg)]
            if defer_res:
                rr = pl.ds(pl.multiple_of(j * seg, seg), seg)
                return _Multi(
                    [
                        pltpu.make_async_copy(rows, xin_v.at[j % 2], own_sem.at[j % 2]),
                        pltpu.make_async_copy(
                            pre_refs["f"].at[rr], fin_v.at[j % 2], fsem.at[0, j % 2]
                        ),
                        pltpu.make_async_copy(
                            pre_refs["inv"].at[rr], iv_v.at[j % 2], fsem.at[1, j % 2]
                        ),
                    ]
                )
            if fuse_norm:
                return pltpu.make_async_copy(rows, xin_v.at[j % 2], own_sem.at[j % 2])
            return pltpu.make_async_copy(
                rows, blocks.at[slot_of(j), me], own_sem.at[j % 2]
            )

        def norm_rows(x32):
            """program._rms_fp8_rows -> (e4m3 rows, [8, 128] tile of the scales in row 0)."""
            w = small_row(sm_v[...], "pre_attn_w")
            r = jax.lax.rsqrt(
                jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + c.rms_eps
            )
            m = jnp.max(jnp.abs(x32 * w), axis=-1, keepdims=True)
            sc = jnp.maximum(m * r, 1e-30) / _FP8_MAX
            v = x32 * (r / sc) * w
            if exact_cast:
                v = _e4m3_rne(v)
            diag = jax.lax.broadcasted_iota(
                jnp.int32, (seg, 128), 0
            ) == jax.lax.broadcasted_iota(jnp.int32, (seg, 128), 1)
            row = jnp.sum(jnp.where(diag, sc, 0.0), axis=0, keepdims=True)
            return v.astype(jnp.float8_e4m3fn), jnp.broadcast_to(row, (8, 128))

        def norm_own_tiles(t0: int, t1: int):
            """own_first: own tiles [t0, t1) normed into own8 / own_sc (loads double-buffered)."""
            ld = lambda t: pltpu.make_async_copy(
                x_hbm.at[pl.ds(t * seg, seg)], xin_v.at[t % 2], own_sem.at[t % 2]
            )
            ld(t0).start()
            for t in range(t0, t1):
                ld(t).wait()
                if t + 1 < t1:
                    ld(t + 1).start()
                own8[t * seg : (t + 1) * seg], own_sc[t] = norm_rows(xin_v[t % 2])

        def tile_scales(tiles):
            """[8, 128] scale tiles -> their rows' [n * seg, 1] column."""
            diag = jax.lax.broadcasted_iota(
                jnp.int32, (seg, 128), 0
            ) == jax.lax.broadcasted_iota(jnp.int32, (seg, 128), 1)
            return jnp.concatenate(
                [
                    jnp.sum(jnp.where(diag, t[0:1, :], 0.0), axis=1, keepdims=True)
                    for t in tiles
                ],
                axis=0,
            )

        def own_norm(j):
            """program._rms_fp8_rows of own tile j into its pad block and scale tile."""
            s = slot_of(j)
            x32 = xin_v[j % 2]
            if defer_res:
                # previous layer's FFN residual (spep.residual), then this layer's norm
                smp = smp_v[...]
                d = fin_v[j % 2].astype(jnp.float32) * small_row(smp, "post_ffn_w")
                x32 = small_row(smp, "ffn_alpha") * x32 + small_row(smp, "ffn_beta") * (
                    (d * iv_v[j % 2]).astype(jnp.bfloat16).astype(jnp.float32)
                )

                @pl.when(j >= n_pre + 2)
                def _():
                    st_write(j - 2).wait()

                xo_v[j % 2] = x32
                st_write(j).start()
            w = small_row(sm_v[...], "pre_attn_w")
            r = jax.lax.rsqrt(
                jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + c.rms_eps
            )
            m = jnp.max(jnp.abs(x32 * w), axis=-1, keepdims=True)
            sc = jnp.maximum(m * r, 1e-30) / _FP8_MAX
            v = x32 * (r / sc) * w
            if exact_cast:
                v = _e4m3_rne(v)
            blocks.at[s, me][...] = v.astype(jnp.float8_e4m3fn)
            # [seg, 1] -> lanes of one row (exact: one nonzero term per lane)
            diag = jax.lax.broadcasted_iota(
                jnp.int32, (seg, 128), 0
            ) == jax.lax.broadcasted_iota(jnp.int32, (seg, 128), 1)
            row = jnp.sum(jnp.where(diag, sc, 0.0), axis=0, keepdims=True)
            sc_pad[s, me] = jnp.broadcast_to(row, (8, 128))

        def gathered_scales(s):
            """Every rank block's scales of slot s as the tile's [tm, 1] column."""
            diag = jax.lax.broadcasted_iota(
                jnp.int32, (seg, 128), 0
            ) == jax.lax.broadcasted_iota(jnp.int32, (seg, 128), 1)
            cols = [
                jnp.sum(
                    jnp.where(diag, sc_pad[s, rank_of(p)][0:1, :], 0.0),
                    axis=1,
                    keepdims=True,
                )
                for p in range(tp)
            ]
            return jnp.concatenate(cols, axis=0)

        def src_rows(j, r):
            return pl.ds(pl.multiple_of(r * Ts + j * seg, seg), seg)

        # tile positions that this rank computes per tile (own_first: not its own block)
        positions = range(1, tp) if own_first else range(tp)

        def tile_rows(r):
            return pl.ds(r * seg, seg)

        def in_loads(j):
            b = j % 2
            srcs = [] if fuse_norm and not dbg_xs else [(xs_hbm, xs_v)]
            srcs += [(rope_hbm, rope_v)] if rope else []
            return [
                pltpu.make_async_copy(
                    h.at[src_rows(j, rank_of(p))],
                    v.at[b, tile_rows(p)],
                    in_sem.at[i, b],
                )
                for i, (h, v) in enumerate(srcs)
                for p in positions
            ]

        def out_writes(j):
            b = j % 2
            cps = []
            for p in positions:
                dst = src_rows(j, rank_of(p))
                if q_rpa:
                    if q_strided:
                        cps.append(
                            (
                                q_v.at[b, :, pl.ds(p * seg * gq, seg * gq)],
                                q_hbm.at[
                                    :,
                                    pl.ds(
                                        pl.multiple_of(
                                            (rank_of(p) * Ts + j * seg) * gq, seg * gq
                                        ),
                                        seg * gq,
                                    ),
                                ],
                            )
                        )
                    else:
                        cps.append((q_v.at[b, :, tile_rows(p)], q_hbm.at[:, dst]))
                else:
                    cps.append((q_v.at[b, tile_rows(p)], q_hbm.at[dst]))
                if not kv_rpa:
                    cps.append((kv_v.at[b, tile_rows(p)], kv_hbm.at[dst]))
                if gate4:
                    cps.append((g_v.at[b, :, tile_rows(p)], g_hbm.at[:, dst]))
                else:
                    cps.append((g_v.at[b, tile_rows(p)], g_hbm.at[dst]))
            return [
                pltpu.make_async_copy(a, d, out_sem.at[i % 3, b])
                for i, (a, d) in enumerate(cps)
            ]

        def own_rows(h):
            return pl.ds(pl.multiple_of(me * Ts + h * half, half), half)

        def own_rope_cp():
            return pltpu.make_async_copy(
                rope_hbm.at[pl.ds(pl.multiple_of(me * Ts, Ts), Ts)],
                orope_v,
                orope_sem.at[0],
            )

        def own_writes(h):
            cps = []
            if q_rpa:
                cps.append((oq_v.at[h], q_hbm.at[:, own_rows(h)]))
            else:
                cps.append((oq_v.at[h], q_hbm.at[own_rows(h)]))
            cps.append((okv_v.at[h], kv_hbm.at[own_rows(h)]))
            if gate4:
                cps.append((og_v.at[h], g_hbm.at[:, own_rows(h)]))
            else:
                cps.append((og_v.at[h], g_hbm.at[own_rows(h)]))
            return [
                pltpu.make_async_copy(a, d, oout_sem.at[i, h])
                for i, (a, d) in enumerate(cps)
            ]

        # tile 0's own rows ahead of the 18.9 MB weight copy, which it would queue behind
        w_cps[0].start()

        def smp_cp():
            return pltpu.make_async_copy(sm_hbm.at[l - 1], smp_v, smp_sem.at[0])

        if defer_res:
            smp_cp().start()
        if not own_first:
            own_load(n_pre).start()
        if n_pre:
            for cp in pre_loads(0) + in_loads(0):
                cp.start()
        # every peer is in the kernel before anyone writes into its pads
        col.barrier()
        wkr = H // wk

        def wk_cp(i):
            return pltpu.make_async_copy(
                w_hbm.at[l, pl.ds(i * wkr, wkr)],
                w_v.at[pl.ds(i * wkr, wkr)],
                wk_sem.at[i],
            )

        if wk > 1:
            for i in range(wk):
                wk_cp(i).start()
        elif w_start == 0:
            w_cps[1].start()

        def start_w_at(stage: int, j):
            if w_start == stage:
                pl.when(j == 0)(w_cps[1].start)

        def s1_own_first(j):
            s = j % n_slots

            @pl.when(j == 0)
            def _():
                w_cps[0].wait()
                with _scope("xq_norm"):
                    norm_own_tiles(0, 1)

            blocks.at[s, me][...] = own8[pl.ds(pl.multiple_of(j * seg, seg), seg)]
            sc_pad[s, me] = own_sc[j]
            col.ag_start(kp, s)

            @pl.when(j == 0)
            def _():
                if rope:
                    own_rope_cp().start()
                with _scope("xq_norm"):
                    norm_own_tiles(1, nt)

            if dbg_ag_only:
                return
            for h in range(2):

                @pl.when(j == 1 + h)
                def _(h=h):
                    if h == 0:
                        w_cps[1].wait()
                        if rope:
                            own_rope_cp().wait()
                    with _scope("qk_own"):
                        sm = sm_v[...]
                        xs = tile_scales(
                            [own_sc[t] for t in range(h * nt // 2, (h + 1) * nt // 2)]
                        )
                        q, kv, g = compute_rows(
                            lambda r0, n: own8[h * half + r0 : h * half + r0 + n],
                            [(0, half // 2), (half // 2, half // 2)],
                            xs,
                            (lambda r0, n: orope_v[h * half + r0 : h * half + r0 + n])
                            if rope
                            else None,
                            sm,
                        )
                        oq_v[h] = _q_rpa(q, kvh, gq, hd) if q_rpa else q
                        okv_v[h] = kv
                        og_v[h] = _q_rpa(g, kvh, gq, hd) if gate4 else g
                    for cp in own_writes(h):
                        cp.start()

        def compute_rows(x_of, chunks, xs, rope_of, sm):
            """qkvg + epilogue of row chunks [(r0, n)] in one block; xs / rope_of indexed alike."""
            epi = qkv_epilogue if mxu_norm else qkv_from_proj
            epi = qkv_epilogue_split if split_epi else epi
            base = chunks[0][0]
            outs = []
            for r0, n in chunks:
                acc = kdot(x_of(r0, n), w_v[...])
                qkvg = (
                    acc
                    * xs[r0 - base : r0 - base + n]
                    * small_row(sm, "wqkvg_s")[:, :nq]
                )
                outs.append(
                    epi(c, qkvg, rope_of(r0, n) if rope_of is not None else None)
                )
            return tuple(
                jnp.concatenate([o[i] for o in outs], axis=0) for i in range(3)
            )

        def s1(j):
            if own_first:
                return s1_own_first(j)
            own_load(j).wait()

            @pl.when(j + 1 < nt)
            def _():
                own_load(j + 1).start()

            if fuse_norm:

                @pl.when(j == n_pre)
                def _():
                    w_cps[0].wait()
                    if defer_res:
                        smp_cp().wait()

                with _scope("xq_norm"):
                    own_norm(j)
            col.ag_start(kp, slot_of(j))

        def s2a(j):
            col.ag_relay(kp, slot_of(j))
            start_w_at(1, j)

        def s2(j):
            col.ag_to_partner(kp, slot_of(j))
            start_w_at(2, j)
            for cp in in_loads(j):
                cp.start()

        def kdot(x, w, kwait=None):
            acc = None
            for k0 in range(0, H, kc):
                if kwait is not None and k0 % wkr == 0:
                    kwait(k0 // wkr)
                part = jnp.dot(
                    x[:, k0 : k0 + kc],
                    w[k0 : k0 + kc, :],
                    preferred_element_type=jnp.float32,
                )
                acc = part if acc is None else acc + part
            return acc

        def qk_dot(x_of, r0, hm, kwait=None):
            x = x_of(r0, hm)
            if not n_tail:
                return kdot(x, w_v, kwait)
            # the last 256-column block as two row halves: 9 blocks split 5:4 over the 2 MXUs
            nm = nq - 256
            head = kdot(x, w_v[:, :nm])
            h2 = hm // 2
            tail = jnp.concatenate(
                [kdot(x[:h2], w_v[:, nm:]), kdot(x[h2:], w_v[:, nm:])], axis=0
            )
            return jnp.concatenate([head, tail], axis=1)

        def s3(j):
            s, b = slot_of(j), j % 2

            if not own_first and not n_pre:

                @pl.when(j == 0)
                def _():
                    for cp in w_cps[0 if not fuse_norm else 1 :]:
                        cp.wait()

            with _scope("xq_wait"):
                col.ag_wait(kp, s)
                for cp in in_loads(j):
                    cp.wait()
            if dbg_ag_only:
                return
            sm = sm_v[...]
            xs_all = gathered_scales(s) if fuse_norm and not dbg_xs else xs_v[b]
            if own_first:
                with _scope("qk_dotepi"):
                    q, kv, g = compute_rows(
                        lambda r0, n: pad[s, r0 : r0 + n],
                        [(seg, 3 * seg), (4 * seg, tm - 4 * seg)],
                        xs_all[seg:],
                        (lambda r0, n: rope_v[b, r0 : r0 + n]) if rope else None,
                        sm,
                    )

                @pl.when(j >= 2)
                def _():
                    for cp in out_writes(j - 2):
                        cp.wait()

                if q_rpa:
                    q_v[b, :, seg:] = _q_rpa(q, kvh, gq, hd)
                else:
                    q_v[b, seg:] = q
                kv_v[b, seg:] = kv
                if gate4:
                    g_v[b, :, seg:] = _q_rpa(g, kvh, gq, hd)
                else:
                    g_v[b, seg:] = g
                for cp in out_writes(j):
                    cp.start()
                return
            tile_compute(j, b, lambda r0, n: pad[s, r0 : r0 + n, :], xs_all, sm)

        def tile_compute(j, b, x_of, xs_all, sm, kwait=None):
            """qkvg, epilogue and the output writes of tile j from rows x_of(r0, n)."""
            epi = qkv_epilogue if mxu_norm else qkv_from_proj
            epi = qkv_epilogue_split if split_epi else epi
            hm = tm // m_split
            outs = []
            # m_split > 1: row halves in one block, so a half's epilogue (VPU) can overlap the
            # next half's dot (MXU)
            # (trace regions are scheduling barriers: one region around the split halves)
            sc = (lambda n: contextlib.nullcontext()) if m_split > 1 else _scope
            with _scope("qk_dotepi") if m_split > 1 else contextlib.nullcontext():
                for h in range(m_split):
                    r0 = h * hm
                    with sc("qk_dot"):
                        acc = qk_dot(x_of, r0, hm, kwait if h == 0 else None)
                        qkvg = (
                            acc
                            * xs_all[r0 : r0 + hm]
                            * small_row(sm, "wqkvg_s")[:, :nq]
                        )
                    with sc("qk_epilogue"):
                        outs.append(
                            epi(c, qkvg, rope_v[b, r0 : r0 + hm] if rope else None)
                        )
            if m_split == 1:
                q, kv, g = outs[0]
            else:
                q, kv, g = (
                    jnp.concatenate([o[i] for o in outs], axis=0) for i in range(3)
                )

            # out buffers are double-buffered: tile j - 2's writes must have drained
            @pl.when(j >= 2)
            def _():
                for cp in out_writes(j - 2):
                    cp.wait()

            if q_strided:
                qf = q.astype(jnp.float32)
                lane = jax.lax.broadcasted_iota(jnp.int32, (qf.shape[0], 128), 1)
                for k in range(kvh):
                    for i in range(gq // 2):
                        v = qf[:, (k * gq // 2 + i) * 128 : (k * gq // 2 + i + 1) * 128]
                        lo = jnp.where(lane < 64, v, 0.0)
                        hi = jnp.where(lane < 64, pltpu.roll(v, 64, 1), 0.0)
                        qs_v[k, pl.ds(2 * i, tm, stride=gq), :] = lo
                        qs_v[k, pl.ds(2 * i + 1, tm, stride=gq), :] = hi
                    q_v[b, k] = qs_v[k].astype(jnp.bfloat16)
            else:
                q_v[b] = _q_rpa(q, kvh, gq, hd) if q_rpa else q
            if kv_rpa:
                kv_place(j, kv)
            else:
                kv_v[b] = kv
            g_v[b] = _q_rpa(g, kvh, gq, hd) if gate4 else g
            for cp in out_writes(j):
                cp.start()

        def kv_place(j, kv):
            """Tile j's kv rows [tm, 2 kvd] into kvn_v: u32 words of 4 e4m3 head lanes,
            lanes = tokens; position p's seg tokens go to lanes p * Ts + j * seg."""
            # e4m3 first: the transpose moves 8-bit tiles (XLU) instead of bf16 ones
            kvt = kv.astype(jnp.float8_e4m3fn).T  # [2 kvd, tm]
            rows = []
            for h in range(kvh):
                rows += [
                    kvt[h * hd : (h + 1) * hd],
                    kvt[kvd + h * hd : kvd + (h + 1) * hd],
                ]
            w = pltpu.bitcast(
                jnp.concatenate(rows, axis=0), jnp.uint32
            )  # [2 kvd / 4, tm]
            nw = w.shape[0]
            kvn_u32 = kvn_v.bitcast(jnp.uint32).reshape(nw, T)
            per = 128 // seg
            sub = jax.lax.rem(j, per) * seg
            lane = jax.lax.broadcasted_iota(jnp.int32, (nw, 128), 1)
            m = jnp.logical_and(lane >= sub, lane < sub + seg)
            for p in range(tp):
                off = (p * seg) % 128
                src = w[:, p * seg - off : p * seg - off + 128]
                rolled = pltpu.roll(src, jax.lax.rem(sub - off + 128, 128), 1)
                col = pl.ds(pl.multiple_of(p * Ts + (j // per) * 128, 128), 128)
                kvn_u32[:, col] = jnp.where(m, rolled, kvn_u32[:, col])

        def pre(i):
            """Tile i < n_pre, gathered by the previous layer's post_ffn kernel."""
            b = i % 2

            if wk == 1:

                @pl.when(i == 0)
                def _():
                    w_cps[1].wait()

            @pl.when(i + 1 < n_pre)
            def _():
                for cp in pre_loads(i + 1) + in_loads(i + 1):
                    cp.start()

            with _scope("xq_wait"):
                for cp in pre_loads(i) + in_loads(i):
                    cp.wait()
            sm = sm_v[...]
            xs_all = tile_scales([presc_v[b, r] for r in range(tp)])
            x_of = lambda r0, n: pre_v[b, r0 : r0 + n, :]
            with _scope("qk_pre"):
                if wk == 1:
                    tile_compute(i, b, x_of, xs_all, sm)
                else:
                    # tile 0 waits for each weight K chunk as its dot reaches it
                    @pl.when(i == 0)
                    def _():
                        tile_compute(i, b, x_of, xs_all, sm, lambda k: wk_cp(k).wait())

                    pl.when(i > 0)(lambda: tile_compute(i, b, x_of, xs_all, sm))

        stages = (s1, s2a, s2, s3)
        if n_pre:
            _pipeline(
                stages, (1, 2, 0, 3), nt - n_pre, gap, off=n_pre, pre=pre, n_pre=n_pre
            )
        elif dma_first:
            _pipeline(stages, (1, 2, 0, 3), nt, gap)
        else:
            run_pipeline(stages, nt, gap)
        for j in range(max(0, nt - 2), nt) if not dbg_ag_only else ():
            for cp in out_writes(j):
                cp.wait()
        if kv_rpa:
            cp = pltpu.make_async_copy(kvn_v, kv_hbm, kvn_sem.at[0])
            cp.start()
            cp.wait()
        if defer_res:
            for j in range(max(n_pre, nt - 2), nt):
                st_write(j).wait()
        if own_first and not dbg_ag_only:
            for h in range(2):
                for cp in own_writes(h):
                    cp.wait()

    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    q_blk = (2, kvh, tm, gq, 128) if q_rpa else (2, tm, qd)
    q_blk = (2, kvh, tm * gq, 128) if q_strided else q_blk
    g_blk = (2, kvh, tm, gq, 128) if gate4 else (2, tm, qd)
    scratch = [
        vmem((H, nq), jnp.float8_e4m3fn),
        vmem((16, H), jnp.float32),
        vmem((n_slots, tm, H), jnp.float8_e4m3fn),
        vmem((2, tm, 1), jnp.float32),
        vmem((2, tm, 2 * hd), jnp.float32),
        vmem(q_blk, jnp.bfloat16),
        vmem((2, tm, 2 * kvd), jnp.bfloat16),
        vmem(g_blk, jnp.bfloat16),
        sem((2,)),
        sem((2,)),
        sem((2, 2)),
        sem((3, 2)),
    ]
    if fuse_norm:
        scratch += [
            vmem((2, seg, H), jnp.float32),
            vmem((n_slots, tp, 8, 128), jnp.float32),
        ]
    if own_first:
        oq = (2, kvh, half, gq, 128) if q_rpa else (2, half, qd)
        og = (2, kvh, half, gq, 128) if gate4 else (2, half, qd)
        scratch += [
            vmem((Ts, H), jnp.float8_e4m3fn),
            vmem((nt, 8, 128), jnp.float32),
            vmem((Ts, 2 * hd), jnp.float32),
            vmem(oq, jnp.bfloat16),
            vmem((2, half, 2 * kvd), jnp.bfloat16),
            vmem(og, jnp.bfloat16),
            sem((1,)),
            sem((3, 2)),
        ]
    scratch += [sem((n_slots,)) for _ in range(2 * len(kinds))]
    kvn_shape = (2 * kvh, hd // 4, 4, T)
    if kv_rpa:
        scratch += [vmem(kvn_shape, jnp.float8_e4m3fn), sem((1,))]
    if q_strided:
        scratch += [vmem((kvh, tm * gq, 128), jnp.float32)]
    if n_pre:
        scratch += [
            vmem((2, tm, H), jnp.float8_e4m3fn),
            vmem((2, tp, 8, 128), jnp.float32),
            sem((2, 2)),
        ]
    if wk > 1:
        scratch += [sem((wk,))]
    if defer_res:
        scratch += [
            vmem((2, seg, H), jnp.bfloat16),
            vmem((2, seg, 1), jnp.float32),
            vmem((2, seg, H), jnp.float32),
            vmem((16, H), jnp.float32),
            sem((2, 2)),
            sem((2,)),
            sem((1,)),
        ]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    q_shape = (kvh, T, gq, 128) if q_rpa else (T, qd)
    q_shape = (kvh, T * gq, 128) if q_strided else q_shape
    name = (
        f"muse_sptp_xag_qkvg_t{T}_s{seg}_k{kc}_g{gap}_r{int(rope)}{'_df' if dma_first else ''}{'_se' if split_epi else ''}{'_g4' if gate4 else ''}_w{w_start}{f'_ms{m_split}' if m_split > 1 else ''}{f'_ds{dma_split}' if dma_split > 1 else ''}{'_nt' if n_tail else ''}{'_dbgag' if dbg_ag_only else ''}{f'_rt{route}' if route else ''}{'_of' if own_first else ''}{'_qs' if q_strided else ''}{f'_pre{n_pre}' if n_pre else ''}{f'_wk{wk}' if wk > 1 else ''}{'_dr' if defer_res else ''}{'_kvr' if kv_rpa else ''}"
        f"{'_mxun' if mxu_norm else ''}{'_qrpa' if q_rpa else ''}{'_norm' if fuse_norm else ''}"
    )

    def call(lyr, x, *args):
        """(lyr, x8, xs, rope_rows, wqkvg, smalls); fuse_norm: (lyr, stream, rope_rows, ...)."""
        n_args = (
            (8 if defer_res else 5) if n_pre else (3 if fuse_norm and not dbg_xs else 4)
        )
        if x.shape != (Ts, H) or len(args) != n_args:
            raise ValueError(f"own rows {x.shape} / {len(args) + 2} operands")
        xg_i = 3 if defer_res else 0
        if n_pre and args[xg_i].shape != (n_pre, tm, H):
            raise ValueError(f"xg {args[xg_i].shape}: expected ({n_pre}, {tm}, {H})")
        if (not fuse_norm or dbg_xs) and args[0].shape != (T, 1):
            raise ValueError(f"xs {args[0].shape}: expected ({T}, 1)")
        outs = [
            jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(kvn_shape, jnp.float8_e4m3fn)
            if kv_rpa
            else jax.ShapeDtypeStruct((T, 2 * kvd), jnp.bfloat16),
            jax.ShapeDtypeStruct((kvh, T, gq, 128) if gate4 else (T, qd), jnp.bfloat16),
        ]
        if defer_res:
            outs.append(jax.ShapeDtypeStruct((Ts, H), jnp.float32))
        return pl.pallas_call(
            kernel,
            out_shape=tuple(outs),
            input_output_aliases={2: 3} if defer_res else {},
            in_specs=[smem] + [hbm] * (len(args) + 1),
            out_specs=(hbm,) * len(outs),
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=SPTP_XAG_QKVG_COLLECTIVE_ID,
                vmem_limit_bytes=60 * 2**20,
            ),
            name=name,
            **({"interpret": interpret} if interpret is not None else {}),
        )(lyr, x, *args)

    return call
