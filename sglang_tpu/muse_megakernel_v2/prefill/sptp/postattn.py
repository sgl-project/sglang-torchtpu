# SPDX-License-Identifier: Apache-2.0
"""SP + TP own-rows post-attention card as two Pallas kernels.

A: the a2a'd gated attention rows -> row requant -> o proj (replicated wo, streamed by column
block) -> attention residual -> pool input rms + e4m3, router and top-k.
B: pool (replicated wpool, streamed) -> pre-expert rms + e4m3.

All Ts own rows stay in VMEM so every weight byte is read once per layer. Two kernels so the
ids / weights all-gather runs under B. Weights stay stacked [L, ...] and the layer comes in as a
prefetched scalar: a per-layer slice passed to a custom call is materialized (64 MB for wo)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel_v2.common.config import PFConfig
from sglang_tpu.muse_megakernel_v2.common.consts import SMALL, SMALL_ROWS
from sglang_tpu.muse_megakernel_v2.common.trace import _scope

_F32, _BF16, _FP8 = jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn
_FP8_MAX = 448.0
# router dot rows per pass: the N = E dot is bound by MXU weight loads, amortized over the rows
_ROUTER_ROWS = 256


def _row(sm_ref, name: str, lo, n: int):
    i = SMALL[name]
    return sm_ref[i : i + 1, pl.ds(lo, n)]


# fp8 contraction depth per MXU pass (common/dots._fp8_dot)
_KC = 512


def _dims(c: PFConfig):
    """(Ts, tp, qd): own rows, ranks, one rank's head width (one K block of wo / wpool)."""
    return c.rows // c.tp, c.tp, c.q_dim


def oproj_call(c: PFConfig, n_router: int, tn: int = 512, rc: int = 64):
    """(lyr [1] i32, g8 [tp, Ts, qd] e4m3 (the token-major gated output a2a'd by row: block src
    is rank src's heads of own rows), gs [tp, Ts, 1] f32 (their row scales), stream [Ts, H] f32,
    smalls [L, 16, H], wo [L, H, H] e4m3, wrouter [L, H, E] bf16) -> (stream [Ts, H] f32,
    ids [k, Ts] i32, tw [k, Ts] f32, f8 [tp, Ts, qd] e4m3 (the pool input by K block),
    fs [Ts, 1] f32)."""
    Ts, tp, qd = _dims(c)
    H, k_top, E = c.hidden, c.top_k, n_router
    nN = H // tn
    rr = min(_ROUTER_ROWS, Ts)
    if H % tn or Ts % rc or Ts % rr or H != tp * qd or qd % _KC or nN < 2:
        raise ValueError(f"o proj tiles {tn} / {rc} do not fit H {H} Ts {Ts}")
    n_rc = Ts // rc

    def kernel(
        lyr, g8_hbm, gs_ref, st_hbm, sm_ref, wo_ref, wr_hbm,
        so_hbm, ids_ref, tw_ref, f8_hbm, fs_ref,
        gq, big, wr, sbuf, obuf, gsc, sso, sem, sin, sout,
    ):  # fmt: skip
        s = pl.program_id(0)
        wr_cp = pltpu.make_async_copy(wr_hbm.at[lyr[0]], wr, sem.at[1])

        @pl.when(s == 0)
        def _():
            g_cp = pltpu.make_async_copy(g8_hbm, gq, sem.at[0])
            g_cp.start()
            wr_cp.start()
            g_cp.wait()
            # src rows were quantized per (row, src); one scale per own row for the o proj
            with _scope("pa_requant"):
                m = jnp.zeros((Ts, 1), _F32)
                for src in range(tp):
                    a = gq[src].astype(_F32) * gs_ref[src]
                    m = jnp.maximum(m, jnp.max(jnp.abs(a), axis=1, keepdims=True))
                sc = jnp.maximum(m, 1e-30) / _FP8_MAX
                gsc[...] = sc
                inv = 1.0 / sc
                for src in range(tp):
                    gq[src] = (gq[src].astype(_F32) * (gs_ref[src] * inv)).astype(_FP8)

        with _scope("pa_oproj"):
            acc = None
            for k0 in range(0, H, _KC):
                src, c0 = divmod(k0, qd)
                part = jnp.dot(
                    gq[src, :, c0 : c0 + _KC],
                    wo_ref[k0 : k0 + _KC, :],
                    preferred_element_type=_F32,
                )
                acc = part if acc is None else acc + part
            lo = pl.multiple_of(s * tn, tn)
            o = acc * gsc[...] * _row(sm_ref, "wo_s", lo, tn)
            big[:, pl.ds(lo, tn)] = o
            # the residual's rms of o, accumulated under the MXU-bound steps
            ss = jnp.sum(o * o, axis=1, keepdims=True)

            @pl.when(s == 0)
            def _():
                sso[...] = ss

            @pl.when(s > 0)
            def _():
                sso[...] += ss

        ep = _Epilogue(c, Ts, rc, sm_ref, st_hbm, so_hbm, big, sbuf, obuf, sin, sout)

        @pl.when(s == nN - 2)
        def _():
            ep.prefetch()

        @pl.when(s == nN - 1)
        def _():
            ep.residual(
                gq, fs_ref, lambda rows: big[rows, :], lambda rows: sso[rows, :]
            )
            ep.route(gq, f8_hbm, sem.at[0], wr, wr_cp, E, ids_ref, tw_ref)

    anyspec = pl.BlockSpec(memory_space=pl.ANY)

    def whole(*shape):
        return pl.BlockSpec(shape, lambda s, lyr: (0,) * len(shape))

    in_specs = [
        anyspec,
        whole(tp, Ts, 1),
        anyspec,
        pl.BlockSpec((None, SMALL_ROWS, H), lambda s, lyr: (lyr[0], 0, 0)),
        pl.BlockSpec((None, H, tn), lambda s, lyr: (lyr[0], 0, s)),
        anyspec,
    ]
    out_specs = (anyspec, whole(k_top, Ts), whole(k_top, Ts), anyspec, whole(Ts, 1))
    out_shape = (
        jax.ShapeDtypeStruct((Ts, H), _F32),
        jax.ShapeDtypeStruct((k_top, Ts), jnp.int32),
        jax.ShapeDtypeStruct((k_top, Ts), _F32),
        jax.ShapeDtypeStruct((tp, Ts, qd), _FP8),
        jax.ShapeDtypeStruct((Ts, 1), _F32),
    )
    vmem = pltpu.VMEM
    scratch = [
        vmem((tp, Ts, qd), _FP8),
        vmem((Ts, H), _F32),
        vmem((H, E), _BF16),
        vmem((2, rc, H), _F32),
        vmem((2, rc, H), _F32),
        vmem((Ts, 1), _F32),
        vmem((Ts, 1), _F32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(nN,),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch,
    )
    params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",), vmem_limit_bytes=60 * 2**20
    )

    def call(layer: int, g8, gs, stream, smalls, wo, wrouter):
        return pl.pallas_call(
            kernel,
            grid_spec=grid_spec,
            out_shape=out_shape,
            compiler_params=params,
            name=f"muse_sptp_pa_oproj_t{c.rows}_n{tn}_r{rc}",
        )(jnp.full((1,), layer, jnp.int32), g8, gs, stream, smalls, wo, wrouter)

    return call


def oproj_a2a_call(
    c: PFConfig,
    n_router: int,
    collective_id: int,
    tn: int = 512,
    rc: int = 64,
    la: int = 7,
    pool_tn: int = 0,
    ns: int = 2,
    order: tuple = (),
    head: bool = False,
    tmaj: bool = False,
):
    """oproj_call with the gated-output a2a inside: (lyr_me [2] i32 (layer, rank), g8 [T, qd]
    e4m3 (this rank's heads of every row, token major), gs [1, T] f32 (their row scales),
    stream, smalls, wo, wrouter) -> oproj_call's outputs.

    Every rank DMAs rows [d * Ts, (d + 1) * Ts) to rank d's VMEM at the start; the o proj
    walks wo by K block (one source rank's heads) from its own block on (nearest
    sources first), so each block's
    matmul runs while later blocks land. Each source block keeps its own row scale (applied to
    its partial product): no requant, one e4m3 rounding of the gated output instead of two.
    la: sends in flight ahead of the receivers' use (7: all at the start).
    pool_tn > 0: pool_call's steps (wpool [L, H, He] after wrouter, column blocks of pool_tn)
    follow in this kernel (the pool input stays in VMEM, wpool's first block loads under the
    epilogue); outputs (stream, ids, tw, xe8 [Ts, He] e4m3, xs [1, Ts] f32).
    ns: stream row chunks in flight + 1; the first ns - 1 load under the last o proj steps.
    order: step r takes source m ^ order[r] (default r: partner core, then chip ^ 1, ^ 2, ^ 3).
    head: g8 in the o gate's head layout [kvh, T, w] (w = qd / kvh, wo's row order within a
    rank) instead of token major; each send gathers rank d's rows of every head group.
    tmaj: g8 [qd, T] (rows in lanes: the o gate fusion's own layout, no relayout before the
    kernel); blocks land as [qd, Ts] and are transposed once per o proj step; the pool input gets its own [tp, Ts, qd] buffer."""
    Ts, tp, qd = _dims(c)
    H, E = c.hidden, n_router
    if (
        H % tn
        or Ts % rc
        or Ts % min(_ROUTER_ROWS, Ts)
        or H != tp * qd
        or qd % _KC
        or tp & (tp - 1)
    ):
        raise ValueError(f"o proj tiles {tn} / {rc} do not fit H {H} Ts {Ts}")

    if head and tmaj:
        raise ValueError("head and tmaj layouts are exclusive")
    act = (tp, c.kv_heads, Ts, qd // c.kv_heads) if head else (tp, Ts, qd)
    if head and act[-1] % _KC:
        raise ValueError(f"head group width {act[-1]} is not a multiple of {_KC}")
    f8_shape = act
    if tmaj:
        act, f8_shape = (tp, qd, Ts), (Ts, H)
    order = tuple(order) or tuple(range(tp))
    if sorted(order) != list(range(tp)) or order[0] != 0:
        raise ValueError(
            f"source order {order} is not a permutation of 0..{tp - 1} from 0"
        )

    def peer_xor(r):
        # order[r] for a traced step r
        x = jnp.int32(0)
        for k, v in enumerate(order):
            x = jnp.where(r == k, v, x)
        return x

    fuse = pool_tn > 0
    He = c.expert_hidden
    nP = He // pool_tn if fuse else 0
    if fuse and (He % pool_tn or He > H):
        raise ValueError(f"pool tile {pool_tn} does not fit He {He}")

    def kernel(lm, *refs):
        if fuse:
            (
                g8_hbm, gs_hbm, st_hbm, sm_ref, wo_hbm, wr_hbm, wp_ref,
                so_hbm, ids_ref, tw_ref, xe_ref, xs_ref,
                gq, gsr, big, wr, sbuf, obuf, sso, sem, rg, rs, sg, ss_, sin, sout,
                wbuf, wsem, fs_ref, ssq, amx, xsc,
            ) = refs  # fmt: skip
            f8_hbm = None
        else:
            (
                g8_hbm, gs_hbm, st_hbm, sm_ref, wo_hbm, wr_hbm,
                so_hbm, ids_ref, tw_ref, f8_hbm, fs_ref,
                gq, gsr, big, wr, sbuf, obuf, sso, sem, rg, rs, sg, ss_, sin, sout,
                wbuf, wsem,
            ) = refs  # fmt: skip
        # the pool input's buffer once the o proj is done: gq, or with tmaj ([qd, Ts] blocks)
        # wo's first buffer as [Ts, H]
        f8b = wbuf.at[0, pl.ds(0, Ts)] if tmaj else gq
        r = pl.program_id(0)
        me = lm[1]
        # step r takes source m ^ r: own block, the partner core, then the other chips' cores
        # nearest first (2 x 2 chips, 2 cores each: chip ^ 3 is the diagonal, two hops); rank m
        # sends to m ^ 1, m ^ 2, ... in the same order
        src = jnp.bitwise_xor(me, peer_xor(r))
        wr_cp = pltpu.make_async_copy(wr_hbm.at[lm[0]], wr, sem.at[1])

        def g8_rows(rows):
            # token-major [T, qd] or the head layout [kvh, T, w]: rank d's rows of every group
            return g8_hbm.at[:, rows] if head or tmaj else g8_hbm.at[rows]

        def sends(d):
            rows = pl.ds(pl.multiple_of(d * Ts, Ts), Ts)
            return (
                pltpu.make_async_remote_copy(
                    g8_rows(rows), gq.at[me], sg.at[d], rg.at[me],
                    device_id=(d,), device_id_type=pl.DeviceIdType.MESH,
                ),
                pltpu.make_async_remote_copy(
                    gs_hbm.at[:, rows], gsr.at[me], ss_.at[d], rs.at[me],
                    device_id=(d,), device_id_type=pl.DeviceIdType.MESH,
                ),
            )  # fmt: skip

        def locals_(d):
            rows = pl.ds(pl.multiple_of(d * Ts, Ts), Ts)
            return (
                pltpu.make_async_copy(g8_rows(rows), gq.at[d], rg.at[d]),
                pltpu.make_async_copy(gs_hbm.at[:, rows], gsr.at[d], rs.at[d]),
            )

        def wo_cp(step, slot):
            # wo's K block of step's source, by hand: the first one loads under the barrier
            k0 = pl.multiple_of(jnp.bitwise_xor(me, peer_xor(step)) * qd, qd)
            return pltpu.make_async_copy(
                wo_hbm.at[lm[0], pl.ds(k0, qd)], wbuf.at[slot], wsem.at[slot]
            )

        @pl.when(r == 0)
        def _():
            wo_cp(0, 0).start()
            for cp in locals_(me):
                cp.start()
            # every peer is in this kernel (its gq free) before anyone writes into it
            with _scope("pa_barrier"):
                bar = pltpu.get_barrier_semaphore()
                for p in range(tp):
                    pl.semaphore_signal(
                        bar, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
                    )
                pl.semaphore_wait(bar, tp)
            for k in range(1, min(la, tp - 1) + 1):
                for cp in sends(jnp.bitwise_xor(me, order[k])):
                    cp.start()
            wo_cp(1, 1).start()
            wr_cp.start()

        @pl.when(jnp.logical_and(r >= 1, r + 1 < tp))
        def _():
            wo_cp(r + 1, (r + 1) % 2).start()

        if la < tp - 1:
            # one more send per step, la ahead of its use: a block lands every step instead of
            # every block late on links all the sends share
            @pl.when(jnp.logical_and(r >= 1, r + la <= tp - 1))
            def _():
                for cp in sends(jnp.bitwise_xor(me, peer_xor(r + la))):
                    cp.start()

        @pl.when(r < tp)
        def _():
            with _scope("pa_a2a_wait"):
                for cp in locals_(src):
                    cp.wait()
                wo_cp(r, r % 2).wait()
            wo_ref = wbuf.at[r % 2]
            with _scope("pa_oproj"):
                # src's row scales as a column
                g_col = jnp.transpose(jnp.broadcast_to(gsr[src], (8, Ts)))[:, 0:1]

                # tmaj: the landed [qd, Ts] block transposed once per step (a transposed-lhs
                # dot per N tile re-transposed it every time: 47 vs 42 us)
                x_t = jnp.transpose(gq[src]) if tmaj else None

                def block(n0, first=False):
                    acc = None
                    for k0 in range(0, qd, _KC):
                        w_k = wo_ref[k0 : k0 + _KC, n0 : n0 + tn]
                        if tmaj:
                            part = jnp.dot(
                                x_t[:, k0 : k0 + _KC], w_k, preferred_element_type=_F32
                            )
                        else:
                            part = jnp.dot(
                                _kblk(gq, k0, src), w_k, preferred_element_type=_F32
                            )
                        acc = part if acc is None else acc + part
                    acc = acc * g_col
                    return acc if first else big[:, n0 : n0 + tn] + acc

                @pl.when(r == 0)
                def _():
                    for n0 in range(0, H, tn):
                        big[:, n0 : n0 + tn] = block(n0, first=True)

                @pl.when(jnp.logical_and(r > 0, r < tp - 1))
                def _():
                    for n0 in range(0, H, tn):
                        big[:, n0 : n0 + tn] = block(n0)

                # the last block: wo's column scales and the residual's row sums of squares under
                # its matmuls
                @pl.when(r == tp - 1)
                def _():
                    ss = jnp.zeros((Ts, 1), _F32)
                    for n0 in range(0, H, tn):
                        o = block(n0) * _row(sm_ref, "wo_s", n0, tn)
                        big[:, n0 : n0 + tn] = o
                        ss = ss + jnp.sum(o * o, axis=1, keepdims=True)
                    sso[...] = ss

        ep = _Epilogue(c, Ts, rc, sm_ref, st_hbm, so_hbm, big, sbuf, obuf, sin, sout)

        @pl.when(r == tp - 2)
        def _():
            ep.prefetch()

        @pl.when(r == tp - 1)
        def _():
            ep.residual(
                f8b, fs_ref, lambda rows: big[rows, :], lambda rows: sso[rows, :]
            )
            ep.route(f8b, f8_hbm, sem.at[0], wr, wr_cp, E, ids_ref, tw_ref)
            for k in range(1, tp):
                d = jnp.bitwise_xor(me, order[k])
                for cp in sends(d):
                    cp.wait_send()

        if fuse:
            # the pool input stays in gq; pooled over big's dead router input
            @pl.when(r >= tp)
            def _():
                _pool_step(
                    c, r - tp, nP, pool_tn, rc, f8b, fs_ref, sm_ref, wp_ref, big, ssq,
                    amx, xsc, xe_ref, xs_ref,
                )  # fmt: skip

    anyspec = pl.BlockSpec(memory_space=pl.ANY)

    def whole(*shape):
        return pl.BlockSpec(shape, lambda r, lm: (0,) * len(shape))

    in_specs = [
        anyspec,
        anyspec,
        anyspec,
        pl.BlockSpec((None, SMALL_ROWS, H), lambda r, lm: (lm[0], 0, 0)),
        anyspec,
        anyspec,
    ]
    k_top = c.top_k
    out_specs = [anyspec, whole(k_top, Ts), whole(k_top, Ts)]
    out_shape = [
        jax.ShapeDtypeStruct((Ts, H), _F32),
        jax.ShapeDtypeStruct((k_top, Ts), jnp.int32),
        jax.ShapeDtypeStruct((k_top, Ts), _F32),
    ]
    if fuse:
        in_specs.append(
            pl.BlockSpec(
                (None, H, pool_tn), lambda r, lm: (lm[0], 0, jnp.maximum(r - tp, 0))
            )
        )
        out_specs += [whole(Ts, He), whole(1, Ts)]
        out_shape += [
            jax.ShapeDtypeStruct((Ts, He), _FP8),
            jax.ShapeDtypeStruct((1, Ts), _F32),
        ]
    else:
        out_specs += [anyspec, whole(Ts, 1)]
        out_shape += [
            jax.ShapeDtypeStruct(f8_shape, _FP8),
            jax.ShapeDtypeStruct((Ts, 1), _F32),
        ]
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA
    scratch = [
        vmem(act, _FP8),
        vmem((tp, 1, Ts), _F32),
        vmem((Ts, H), _F32),
        vmem((H, E), _BF16),
        vmem((ns, rc, H), _F32),
        vmem((2, rc, H), _F32),
        vmem((Ts, 1), _F32),
        dma((2,)),
        dma((tp,)),
        dma((tp,)),
        dma((tp,)),
        dma((tp,)),
        dma((ns,)),
        dma((2,)),
    ]
    scratch += [vmem((2, qd, H), _FP8), dma((2,))]
    if fuse:
        scratch += [vmem((Ts, 1), _F32) for _ in range(4)]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(tp + nP,),
        in_specs=in_specs,
        out_specs=tuple(out_specs),
        scratch_shapes=scratch,
    )
    params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=62 * 2**20,
        collective_id=collective_id,
    )

    def call(layer: int, me, g8, gs, stream, smalls, wo, wrouter, *wpool):
        lm = jnp.stack([jnp.asarray(layer, jnp.int32), me.astype(jnp.int32)])
        return pl.pallas_call(
            kernel,
            grid_spec=grid_spec,
            out_shape=tuple(out_shape),
            compiler_params=params,
            name=f"muse_sptp_pa_oproj_a2a_t{c.rows}_n{tn}_r{rc}_la{la}_p{pool_tn}_s{ns}_o{''.join(map(str, order))}{'_h' if head else ''}{'_t' if tmaj else ''}",
        )(lm, g8, gs, stream, smalls, wo, wrouter, *wpool)

    return call


def _kblk(ref, k0, src=None):
    """K columns [k0, k0 + _KC) of an activation buffer by K block: [tp, Ts, qd] or the head
    layout [tp, kvh, Ts, w] (block src, head group k); src: a traced block, k0 within it."""
    if ref.ndim == 2:
        return ref[:, k0 : k0 + _KC]
    if ref.ndim == 3:
        b, c0 = divmod(k0, ref.shape[2])
        return ref[b if src is None else src, :, c0 : c0 + _KC]
    kvh, w = ref.shape[1], ref.shape[3]
    b, rem = divmod(k0, kvh * w)
    k, c0 = divmod(rem, w)
    return ref[b if src is None else src, k, :, c0 : c0 + _KC]


def _kstore(ref, rows, x):
    """x [rc, H] into an activation buffer's rows, by K block (_kblk's layouts)."""
    if ref.ndim == 2:
        ref[rows, :] = x
        return
    if ref.ndim == 3:
        qd = ref.shape[2]
        for b in range(ref.shape[0]):
            ref[b, rows, :] = x[:, b * qd : (b + 1) * qd]
        return
    kvh, w = ref.shape[1], ref.shape[3]
    for b in range(ref.shape[0]):
        for k in range(kvh):
            c0 = (b * kvh + k) * w
            ref[b, k, rows, :] = x[:, c0 : c0 + w]


class _Epilogue:
    """Kernel A's tail on the o proj rows in big: attention residual (stream in / out by row
    chunk), pool input rms + e4m3 into gq, router input over big, router and top-k."""

    def __init__(self, c, Ts, rc, sm_ref, st_hbm, so_hbm, big, sbuf, obuf, sin, sout):
        self.c, self.Ts, self.rc, self.n_rc = c, Ts, rc, Ts // rc
        self.sm, self.st, self.so, self.big = sm_ref, st_hbm, so_hbm, big
        self.sbuf, self.obuf, self.sin, self.sout = sbuf, obuf, sin, sout

    def cp_in(self, i, slot):
        rows = pl.ds(i * self.rc, self.rc)
        return pltpu.make_async_copy(
            self.st.at[rows], self.sbuf.at[slot], self.sin.at[slot]
        )

    def cp_out(self, i, slot):
        rows = pl.ds(i * self.rc, self.rc)
        return pltpu.make_async_copy(
            self.obuf.at[slot], self.so.at[rows], self.sout.at[slot]
        )

    def prefetch(self):
        """The first sbuf slots - 1 stream chunks (the loop keeps that many in flight)."""
        for i in range(min(self.sbuf.shape[0] - 1, self.n_rc)):
            self.cp_in(i, i).start()

    def residual(self, gq, fs_ref, o_rows, sso_rows):
        """o_rows(rows) -> o [rc, H] f32; sso_rows(rows) -> its row sums of squares [rc, 1]."""
        c, rc, n_rc, sm, big = self.c, self.rc, self.n_rc, self.sm, self.big
        H = c.hidden
        alpha = _row(sm, "attn_alpha", 0, H)
        beta = _row(sm, "attn_beta", 0, H)
        wf = _row(sm, "pre_ffn_w", 0, H)

        ns = self.sbuf.shape[0]

        def body(i, carry):
            slot = i % 2
            rows = pl.ds(pl.multiple_of(i * rc, rc), rc)

            @pl.when(i + ns - 1 < n_rc)
            def _():
                self.cp_in(i + ns - 1, (i + ns - 1) % ns).start()

            self.cp_in(i, i % ns).wait()

            @pl.when(i >= 2)
            def _():
                self.cp_out(i - 2, slot).wait()

            o = o_rows(rows)
            ro = jax.lax.rsqrt(sso_rows(rows) * (1.0 / H) + c.post_eps)
            new = alpha * self.sbuf[i % ns] + beta * (o * ro).astype(_BF16).astype(_F32)
            self.obuf[slot] = new
            self.cp_out(i, slot).start()
            r = jax.lax.rsqrt(jnp.mean(new * new, axis=1, keepdims=True) + c.rms_eps)
            # the router input (rms * pre_ffn_w) over o's dead rows
            big[rows, :] = new * r * wf
            # pool input: rms + e4m3 with the amax from the same pass (program._rms_fp8_rows)
            mf = jnp.max(jnp.abs(new * wf), axis=1, keepdims=True)
            fs = jnp.maximum(mf * r, 1e-30) / _FP8_MAX
            fs_ref[rows, :] = fs
            f8 = (new * (r / fs) * wf).astype(_FP8)
            # gq is free after the o proj: the pool input by K block
            _kstore(gq, rows, f8)
            return carry

        with _scope("pa_residual"):
            jax.lax.fori_loop(0, n_rc, body, 0)

    def route(self, gq, f8_hbm, f8_sem, wr, wr_cp, E, ids_ref, tw_ref):
        c, Ts, big = self.c, self.Ts, self.big
        f8_cp = None if f8_hbm is None else pltpu.make_async_copy(gq, f8_hbm, f8_sem)
        if f8_cp is not None:
            f8_cp.start()
        wr_cp.wait()
        bias = _row(self.sm, "expert_bias", 0, E)
        rr = min(_ROUTER_ROWS, Ts)
        with _scope("pa_router"):
            for r0 in range(0, Ts, rr):
                x = big[r0 : r0 + rr, :].astype(_BF16)
                sc = jax.nn.sigmoid(
                    jnp.dot(x, wr[...], preferred_element_type=_F32) * c.router_mult
                )
                ids, tw = _top_k(sc, bias, c.top_k, c.router_eps, c.route_scale)
                ids_ref[:, r0 : r0 + rr] = ids
                tw_ref[:, r0 : r0 + rr] = tw
        if f8_cp is not None:
            f8_cp.wait()
        for i in range(max(self.n_rc - 2, 0), self.n_rc):
            self.cp_out(i, i % 2).wait()


def _top_k(sc, bias, k: int, eps: float, scale: float):
    """spep's route on sigmoid scores sc [rows, E]: k rounds of the biased row max (ties to the
    lower id) -> (ids i32, normalized weights f32) [k, rows]. Rows go to lanes (one transpose):
    the rounds reduce over sublanes and the outputs are lane dense ([rows, k] outputs pad k to
    128 lanes, and their all-gathers move the padding)."""
    rows, E = sc.shape
    st = jnp.transpose(sc)
    b = jnp.transpose(sc + bias)
    idx = jax.lax.broadcasted_iota(jnp.int32, (E, rows), 0).astype(_F32)
    sub = jax.lax.broadcasted_iota(jnp.int32, (k, rows), 0)
    ids = jnp.zeros((k, rows), _F32)
    wts = jnp.zeros((k, rows), _F32)
    for t in range(k):
        m = jnp.max(b, axis=0, keepdims=True)
        sel = jnp.min(jnp.where(b == m, idx, float(E)), axis=0, keepdims=True)
        pick = idx == sel
        wsel = jnp.sum(jnp.where(pick, st, 0.0), axis=0, keepdims=True)
        ids = jnp.where(sub == t, sel, ids)
        wts = jnp.where(sub == t, wsel, wts)
        b = jnp.where(pick, -jnp.inf, b)
    tw = wts / (jnp.sum(wts, axis=0, keepdims=True) + eps) * scale
    return ids.astype(jnp.int32), tw


def pool_call(
    c: PFConfig, tpn: int = 512, rc: int = 64, head: bool = False, flat: bool = False
):
    """(lyr [1] i32, f8 [tp, Ts, qd] e4m3, fs [Ts, 1] f32, smalls [L, 16, H],
    wpool [L, H, He] e4m3) -> (xe8 [Ts, He] e4m3, xs [1, Ts] f32): pool, pre-expert rms + e4m3."""
    Ts, tp, qd = _dims(c)
    H, He = c.hidden, c.expert_hidden
    nP = He // tpn
    if He % tpn or Ts % rc:
        raise ValueError(f"pool tiles {tpn} / {rc} do not fit He {He} Ts {Ts}")
    n_rc = Ts // rc

    def kernel(
        lyr, f8_ref, fs_ref, sm_ref, wp_ref, xe_ref, xs_ref, pooled, ssq, amx, xsc
    ):
        _pool_step(
            c, pl.program_id(0), nP, tpn, rc, f8_ref, fs_ref, sm_ref, wp_ref, pooled,
            ssq, amx, xsc, xe_ref, xs_ref,
        )  # fmt: skip

    def whole(*shape):
        return pl.BlockSpec(shape, lambda p, lyr: (0,) * len(shape))

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(nP,),
        in_specs=[
            whole(
                *(
                    (Ts, H)
                    if flat
                    else (tp, c.kv_heads, Ts, qd // c.kv_heads)
                    if head
                    else (tp, Ts, qd)
                )
            ),
            whole(Ts, 1),
            pl.BlockSpec((None, SMALL_ROWS, H), lambda p, lyr: (lyr[0], 0, 0)),
            pl.BlockSpec((None, H, tpn), lambda p, lyr: (lyr[0], 0, p)),
        ],
        out_specs=(whole(Ts, He), whole(1, Ts)),
        scratch_shapes=[
            pltpu.VMEM((Ts, He), _F32),
            pltpu.VMEM((Ts, 1), _F32),
            pltpu.VMEM((Ts, 1), _F32),
            pltpu.VMEM((Ts, 1), _F32),
        ],
    )
    params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",), vmem_limit_bytes=60 * 2**20
    )

    def call(layer: int, f8, fs, smalls, wpool):
        return pl.pallas_call(
            kernel,
            grid_spec=grid_spec,
            out_shape=(
                jax.ShapeDtypeStruct((Ts, He), _FP8),
                jax.ShapeDtypeStruct((1, Ts), _F32),
            ),
            compiler_params=params,
            name=f"muse_sptp_pa_pool_t{c.rows}_p{tpn}{'_h' if head else ''}{'_f' if flat else ''}",
        )(jnp.full((1,), layer, jnp.int32), f8, fs, smalls, wpool)

    return call


def _pool_step(c, p, nP, tpn, rc, f8_ref, fs_ref, sm_ref, wp_ref, pooled, ssq, amx, xsc, xe_ref, xs_ref):  # fmt: skip
    """Pool column block p of nP (f8 [tp, Ts, qd] by K block) into pooled[:, p * tpn ...], the
    pre-expert rms sums and amax; the last block quantizes the rows into xe_ref / xs_ref."""
    Ts, H, He = pooled.shape[0], c.hidden, c.expert_hidden
    n_rc = Ts // rc
    with _scope("pa_pool"):
        acc = None
        for k0 in range(0, H, _KC):
            part = jnp.dot(
                _kblk(f8_ref, k0),
                wp_ref[k0 : k0 + _KC, :],
                preferred_element_type=_F32,
            )
            acc = part if acc is None else acc + part
        lo = pl.multiple_of(p * tpn, tpn)
        pb = acc * fs_ref[...] * _row(sm_ref, "wpool_s", lo, tpn)
        pooled[:, pl.ds(lo, tpn)] = pb
        ss = jnp.sum(pb * pb, axis=1, keepdims=True)
        pe = _row(sm_ref, "pre_expert_w", lo, tpn)
        mx = jnp.max(jnp.abs(pb * pe), axis=1, keepdims=True)

        @pl.when(p == 0)
        def _():
            ssq[...] = ss
            amx[...] = mx

        @pl.when(p > 0)
        def _():
            ssq[...] += ss
            amx[...] = jnp.maximum(amx[...], mx)

    @pl.when(p == nP - 1)
    def _():
        with _scope("pa_xe_quant"):
            r = jax.lax.rsqrt(ssq[...] * (1.0 / He) + c.rms_eps)
            xs = jnp.maximum(amx[...] * r, 1e-30) / _FP8_MAX
            xsc[...] = xs
            # lane dense [1, Ts] (a [Ts, 1] output pads every row to 128 lanes)
            xs_ref[...] = jnp.transpose(jnp.broadcast_to(xs, (Ts, 128)))[0:1]
            pe = _row(sm_ref, "pre_expert_w", 0, He)

            def body(i, carry):
                rows = pl.ds(pl.multiple_of(i * rc, rc), rc)
                f = jax.lax.rsqrt(ssq[rows, :] * (1.0 / He) + c.rms_eps)
                f = f / xsc[rows, :]
                xe_ref[rows, :] = (pooled[rows, :He] * f * pe).astype(_FP8)
                return carry

            jax.lax.fori_loop(0, n_rc, body, 0)
