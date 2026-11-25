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


def _pipeline(stages: tuple, order: tuple, nt: int, gap: int) -> None:
    """common.coll.run_pipeline with a chosen stage order per iteration: stage k runs on tile
    i - k * (1 + gap). Running the DMA relays before the compute gives every relay's sends a
    whole compute to land (run_pipeline's later-stages-first order issues the partner share
    right after the compute that waits for it)."""
    lag = 1 + gap
    n_it = nt + (len(stages) - 1) * lag

    def body(i, carry):
        for k in order:
            j = i - k * lag

            @pl.when(jnp.logical_and(j >= 0, j < nt))
            def _(st=stages[k], j=j):
                with _scope(f"pf_{st.__name__}"):
                    st(j)

        return carry

    jax.lax.fori_loop(0, n_it, body, 0)


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

    def __init__(self, tp, me, sems, n: int, width: int):
        super().__init__(tp, me, sems)
        self.n, self.width = n, width

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

    def __init__(self, ref, seg: int):
        self.ref, self.seg = ref, seg

    @property
    def at(self):
        return self

    def __getitem__(self, idx):
        s, r = idx
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

    def kernel(lyr, *refs):
        if fuse_norm and not dbg_xs:
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
            xin_v, sc_pad, *coll_sems = rest
        else:
            xin_v, sc_pad, coll_sems = None, None, rest
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        col = _SplitColl(
            tp,
            me,
            {k: coll_sems[2 * i : 2 * i + 2] for i, k in enumerate(kinds)},
            dma_split,
            H,
        )
        blocks = _RowBlocks(pad, seg)
        kp = (("x", blocks),) + ((("xsc", sc_pad),) if fuse_norm else ())
        l = lyr[0]
        # smalls first: the fused norm of tile 0 needs them, the dot not before iteration 3
        w_cps = [
            pltpu.make_async_copy(sm_hbm.at[l], sm_v, w_sem.at[1]),
            pltpu.make_async_copy(w_hbm.at[l], w_v, w_sem.at[0]),
        ]

        def own_load(j):
            rows = x_hbm.at[pl.ds(pl.multiple_of(j * seg, seg), seg)]
            if fuse_norm:
                return pltpu.make_async_copy(rows, xin_v.at[j % 2], own_sem.at[j % 2])
            return pltpu.make_async_copy(
                rows, blocks.at[j % n_slots, me], own_sem.at[j % 2]
            )

        def own_norm(j):
            """program._rms_fp8_rows of own tile j into its pad block and scale tile."""
            s = j % n_slots
            x32 = xin_v[j % 2]
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
                    jnp.where(diag, sc_pad[s, r][0:1, :], 0.0), axis=1, keepdims=True
                )
                for r in range(tp)
            ]
            return jnp.concatenate(cols, axis=0)

        def src_rows(j, r):
            return pl.ds(pl.multiple_of(r * Ts + j * seg, seg), seg)

        def tile_rows(r):
            return pl.ds(r * seg, seg)

        def in_loads(j):
            b = j % 2
            srcs = [] if fuse_norm and not dbg_xs else [(xs_hbm, xs_v)]
            srcs += [(rope_hbm, rope_v)] if rope else []
            return [
                pltpu.make_async_copy(
                    h.at[src_rows(j, r)], v.at[b, tile_rows(r)], in_sem.at[i, b]
                )
                for i, (h, v) in enumerate(srcs)
                for r in range(tp)
            ]

        def out_writes(j):
            b = j % 2
            cps = []
            for r in range(tp):
                if q_rpa:
                    cps.append(
                        (q_v.at[b, :, tile_rows(r)], q_hbm.at[:, src_rows(j, r)])
                    )
                else:
                    cps.append((q_v.at[b, tile_rows(r)], q_hbm.at[src_rows(j, r)]))
                cps.append((kv_v.at[b, tile_rows(r)], kv_hbm.at[src_rows(j, r)]))
                if gate4:
                    cps.append(
                        (g_v.at[b, :, tile_rows(r)], g_hbm.at[:, src_rows(j, r)])
                    )
                else:
                    cps.append((g_v.at[b, tile_rows(r)], g_hbm.at[src_rows(j, r)]))
            return [
                pltpu.make_async_copy(a, d, out_sem.at[i % 3, b])
                for i, (a, d) in enumerate(cps)
            ]

        # tile 0's own rows ahead of the 18.9 MB weight copy, which it would queue behind
        w_cps[0].start()
        own_load(0).start()
        # every peer is in the kernel before anyone writes into its pads
        col.barrier()
        if w_start == 0:
            w_cps[1].start()

        def start_w_at(stage: int, j):
            if w_start == stage:
                pl.when(j == 0)(w_cps[1].start)

        def s1(j):
            own_load(j).wait()

            @pl.when(j + 1 < nt)
            def _():
                own_load(j + 1).start()

            if fuse_norm:

                @pl.when(j == 0)
                def _():
                    w_cps[0].wait()

                with _scope("xq_norm"):
                    own_norm(j)
            col.ag_start(kp, j % n_slots)

        def s2a(j):
            col.ag_relay(kp, j % n_slots)
            start_w_at(1, j)

        def s2(j):
            col.ag_to_partner(kp, j % n_slots)
            start_w_at(2, j)
            for cp in in_loads(j):
                cp.start()

        def kdot(x, w):
            acc = None
            for k0 in range(0, H, kc):
                part = jnp.dot(
                    x[:, k0 : k0 + kc],
                    w[k0 : k0 + kc, :],
                    preferred_element_type=jnp.float32,
                )
                acc = part if acc is None else acc + part
            return acc

        def qk_dot(pad, s, r0, hm):
            x = pad[s, r0 : r0 + hm, :]
            if not n_tail:
                return kdot(x, w_v[...])
            # the last 256-column block as two row halves: 9 blocks split 5:4 over the 2 MXUs
            nm = nq - 256
            head = kdot(x, w_v[:, :nm])
            h2 = hm // 2
            tail = jnp.concatenate(
                [kdot(x[:h2], w_v[:, nm:]), kdot(x[h2:], w_v[:, nm:])], axis=0
            )
            return jnp.concatenate([head, tail], axis=1)

        def s3(j):
            s, b = j % n_slots, j % 2

            @pl.when(j == 0)
            def _():
                for cp in w_cps[0 if not fuse_norm else 1 :]:
                    cp.wait()

            with _scope("xq_wait"):
                col.ag_wait(kp, s)
                for cp in in_loads(j):
                    cp.wait()
            sm = sm_v[...]
            xs_all = gathered_scales(s) if fuse_norm and not dbg_xs else xs_v[b]
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
                        acc = qk_dot(pad, s, r0, hm)
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

            q_v[b] = _q_rpa(q, kvh, gq, hd) if q_rpa else q
            kv_v[b] = kv
            g_v[b] = _q_rpa(g, kvh, gq, hd) if gate4 else g
            for cp in out_writes(j):
                cp.start()

        stages = (s1, s2a, s2, s3)
        if dma_first:
            _pipeline(stages, (1, 2, 0, 3), nt, gap)
        else:
            run_pipeline(stages, nt, gap)
        for j in range(max(0, nt - 2), nt):
            for cp in out_writes(j):
                cp.wait()

    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    q_blk = (2, kvh, tm, gq, 128) if q_rpa else (2, tm, qd)
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
    scratch += [sem((n_slots,)) for _ in range(2 * len(kinds))]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    q_shape = (kvh, T, gq, 128) if q_rpa else (T, qd)
    name = (
        f"muse_sptp_xag_qkvg_t{T}_s{seg}_k{kc}_g{gap}_r{int(rope)}{'_df' if dma_first else ''}{'_se' if split_epi else ''}{'_g4' if gate4 else ''}_w{w_start}{f'_ms{m_split}' if m_split > 1 else ''}{f'_ds{dma_split}' if dma_split > 1 else ''}{'_nt' if n_tail else ''}"
        f"{'_mxun' if mxu_norm else ''}{'_qrpa' if q_rpa else ''}{'_norm' if fuse_norm else ''}"
    )

    def call(lyr, x, *args):
        """(lyr, x8, xs, rope_rows, wqkvg, smalls); fuse_norm: (lyr, stream, rope_rows, ...)."""
        if x.shape != (Ts, H) or len(args) != (3 if fuse_norm and not dbg_xs else 4):
            raise ValueError(f"own rows {x.shape} / {len(args) + 2} operands")
        if (not fuse_norm or dbg_xs) and args[0].shape != (T, 1):
            raise ValueError(f"xs {args[0].shape}: expected ({T}, 1)")
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
                jax.ShapeDtypeStruct((T, 2 * kvd), jnp.bfloat16),
                jax.ShapeDtypeStruct(
                    (kvh, T, gq, 128) if gate4 else (T, qd), jnp.bfloat16
                ),
            ),
            in_specs=[smem] + [hbm] * (len(args) + 1),
            out_specs=(hbm,) * 3,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=SPTP_XAG_QKVG_COLLECTIVE_ID,
                vmem_limit_bytes=60 * 2**20,
            ),
            name=name,
            **({"interpret": interpret} if interpret is not None else {}),
        )(lyr, x, *args)

    return call
