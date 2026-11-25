# SPDX-License-Identifier: Apache-2.0
"""SP + TP post-FFN card as one Pallas kernel, which also starts the next layer's x_norm gather.

unpool (own MoE rows, row e4m3 @ replicated e4m3 wunpool streamed in column chunks) -> post-FFN
norm and the FFN residual gate (spep.residual) -> the next layer's pre-attention rms + e4m3 row
quant of the first n_pre row tiles, gathered by remote DMA (common.coll.GridColl, as
sptp/xag_qkvg.py) while the remaining rows' residual runs. The gathered tiles go to HBM; the next
layer's xag_qkvg (n_pre) computes them in the iterations it would otherwise spend filling its
gather pipeline. DMAs cannot outlive a kernel, so the n_pre tiles complete here.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel_v2.common.coll import GridColl, ag_kinds
from sglang_tpu.muse_megakernel_v2.common.config import PFConfig
from sglang_tpu.muse_megakernel_v2.common.consts import small_row
from sglang_tpu.muse_megakernel_v2.common.trace import _scope
from sglang_tpu.muse_megakernel_v2.prefill.sptp.xag_qkvg import _RowBlocks

# Barrier id; distinct from common.consts' (0, 6, 7, 9-12) and sptp/xag_qkvg (15).
SPTP_POST_FFN_COLLECTIVE_ID = 16
_FP8_MAX = 448.0
_KINDS = ag_kinds("x") + ag_kinds("xsc")


def post_ffn_call(
    c: PFConfig,
    seg: int,
    n_pre: int,
    ncw: int = 512,
    axis: str = "model",
    interpret=None,
    two_pass: bool = False,
    defer_res: bool = False,
):
    """(layer [1] i32, y [Ts, He] own MoE rows, stream [Ts, H] f32, stacked wunpool [L, He, H]
    e4m3 and smalls [L, 16, H] f32) -> (stream' [Ts, H] f32, xg [n_pre, tp * seg, H] e4m3 the
    gathered next-layer rows of tiles 0..n_pre-1 (rank-major within a tile), xsc [n_pre, tp, 8,
    128] f32 their row scales in xag_qkvg's packed form). Layer l + 1 must exist."""
    T, H, tp, He = c.rows, c.hidden, c.tp, c.expert_hidden
    Ts = T // tp
    tm = tp * seg
    nt = Ts // seg
    nc = H // ncw
    if two_pass and (H // ncw) % 2:
        raise ValueError(
            "two_pass chains the weight chunks by parity: even chunk count"
        )
    if Ts % seg or seg % 32 or H % ncw or ncw % 128 or not 1 <= n_pre <= nt:
        raise ValueError(f"rows {Ts} / seg {seg} / chunk {ncw} / n_pre {n_pre}")
    # defer_res: only tiles < n_pre get their residual here; f and the inverse rms go out as
    # (f [Ts, H] bf16, inv [Ts, 1] f32) for the next xag_qkvg (defer_res) to finish the rest
    n_res = n_pre if defer_res else nt
    # phase-1 sends of the n_pre tiles overlap the residual of tiles [n_pre, mid), the
    # phase-2 sends tiles [mid, nt)
    mid = n_pre + (nt - n_pre) // 2

    def kernel(
        lyr,
        y_hbm,
        st_hbm,
        w_hbm,
        sm_hbm,
        sto_hbm,
        xg_hbm,
        xsc_hbm,
        *rest,
    ):
        if defer_res:
            f_hbm, inv_hbm, *rest = rest
        body(
            lyr,
            y_hbm,
            st_hbm,
            w_hbm,
            sm_hbm,
            sto_hbm,
            xg_hbm,
            xsc_hbm,
            *rest,
            f_hbm=f_hbm if defer_res else None,
            inv_hbm=inv_hbm if defer_res else None,
        )

    def body(
        lyr,
        y_hbm,
        st_hbm,
        w_hbm,
        sm_hbm,
        sto_hbm,
        xg_hbm,
        xsc_hbm,
        y_v,
        y8_v,
        ys_v,
        w_v,
        f_v,
        sm_v,
        xin_v,
        xout_v,
        pad,
        sc_pad,
        inv_v,
        ld_sem,
        w_sem,
        in_sem,
        out_sem,
        fin_sem,
        *coll_sems,
        f_hbm=None,
        inv_hbm=None,
    ):
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        col = GridColl(
            tp, me, {k: coll_sems[2 * i : 2 * i + 2] for i, k in enumerate(_KINDS)}
        )
        blocks = _RowBlocks(pad, seg)
        kp = (("x", blocks), ("xsc", sc_pad))
        l = lyr[0]
        sm_cps = [
            pltpu.make_async_copy(sm_hbm.at[l + i], sm_v.at[i], ld_sem.at[i])
            for i in range(2)
        ]
        y_cp = pltpu.make_async_copy(y_hbm, y_v, ld_sem.at[2])

        def w_cp(cc):
            return pltpu.make_async_copy(
                w_hbm.at[l, :, pl.ds(pl.multiple_of(cc * ncw, ncw), ncw)],
                w_v.at[cc % 2],
                w_sem.at[cc % 2],
            )

        def st_in(t):
            return pltpu.make_async_copy(
                st_hbm.at[pl.ds(pl.multiple_of(t * seg, seg), seg)],
                xin_v.at[t % 2],
                in_sem.at[t % 2],
            )

        def st_out(t):
            return pltpu.make_async_copy(
                xout_v.at[t % 2],
                sto_hbm.at[pl.ds(pl.multiple_of(t * seg, seg), seg)],
                out_sem.at[t % 2],
            )

        for cp in sm_cps:
            cp.start()
        y_cp.start()
        w_cp(0).start()
        if nc > 1:
            w_cp(1).start()
        st_in(0).start()
        col.barrier()
        for cp in sm_cps:
            cp.wait()
        y_cp.wait()

        # spep's _fp8_rows of the MoE rows, 128 rows at a time
        with _scope("pf_yquant"):
            for r0 in range(0, Ts, 128):
                yr = y_v[r0 : r0 + 128].astype(jnp.float32)
                s = (
                    jnp.maximum(jnp.max(jnp.abs(yr), axis=1, keepdims=True), 1e-30)
                    / _FP8_MAX
                )
                y8_v[r0 : r0 + 128] = (yr / s).astype(jnp.float8_e4m3fn)
                ys_v[r0 : r0 + 128] = s

        def unpool_pass(r0: int, n: int, again: bool, mid_hook=None):
            """f and the post-FFN inverse rms of own rows [r0, r0 + n), wunpool streamed once;
            again: the last two chunk slots start the next pass's first two chunks."""

            def chunk(cc, ss):
                w_cp(cc).wait()
                cols = pl.ds(pl.multiple_of(cc * ncw, ncw), ncw)
                acc = jnp.dot(
                    y8_v[r0 : r0 + n], w_v[cc % 2], preferred_element_type=jnp.float32
                )

                @pl.when(cc + 2 < nc)
                def _():
                    w_cp(cc + 2).start()

                if again:

                    @pl.when(cc + 2 >= nc)
                    def _():
                        w_cp(cc + 2 - nc).start()

                if mid_hook is not None:
                    pl.when(cc == nc // 2)(mid_hook)

                ws = sm_v[0, 13:14, cols]  # wunpool_s
                pw = sm_v[0, 4:5, cols]  # post_ffn_w
                f = (acc * ys_v[r0 : r0 + n] * ws).astype(jnp.bfloat16)
                f_v[r0 : r0 + n, cols] = f
                d = f.astype(jnp.float32) * pw
                return ss + jnp.sum(d * d, axis=1, keepdims=True)

            with _scope("pf_unpool"):
                ss = jax.lax.fori_loop(0, nc, chunk, jnp.zeros((n, 1), jnp.float32))
            inv_v[r0 : r0 + n] = jax.lax.rsqrt(ss * (1.0 / H) + c.post_eps)

        ra = n_pre * seg
        if two_pass:
            unpool_pass(0, ra, True)
        else:
            unpool_pass(0, Ts, False)

        diag = jax.lax.broadcasted_iota(
            jnp.int32, (seg, 128), 0
        ) == jax.lax.broadcasted_iota(jnp.int32, (seg, 128), 1)

        def res_tile(t, publish: bool):
            """FFN residual of own tile t; publish: also the next layer's norm into the pad."""
            st_in(t).wait()

            @pl.when(t + 1 < n_res)
            def _():
                st_in(t + 1).start()

            @pl.when(t >= 2)
            def _():
                st_out(t - 2).wait()

            rows = pl.ds(pl.multiple_of(t * seg, seg), seg)
            sm = sm_v[0]
            d = f_v[rows].astype(jnp.float32) * small_row(sm, "post_ffn_w")
            r = inv_v[rows]
            x = small_row(sm, "ffn_alpha") * xin_v[t % 2] + small_row(
                sm, "ffn_beta"
            ) * ((d * r).astype(jnp.bfloat16).astype(jnp.float32))
            xout_v[t % 2] = x
            st_out(t).start()
            if publish:
                w = small_row(sm_v[1], "pre_attn_w")
                rr = jax.lax.rsqrt(
                    jnp.mean(jnp.square(x), axis=-1, keepdims=True) + c.rms_eps
                )
                m = jnp.max(jnp.abs(x * w), axis=-1, keepdims=True)
                sc = jnp.maximum(m * rr, 1e-30) / _FP8_MAX
                blocks.at[t, me][...] = (x * (rr / sc) * w).astype(jnp.float8_e4m3fn)
                row = jnp.sum(jnp.where(diag, sc, 0.0), axis=0, keepdims=True)
                sc_pad[t, me] = jnp.broadcast_to(row, (8, 128))
                col.ag_start(kp, t)

        def loop(lo, hi, publish=False):
            def body(t, carry):
                res_tile(t, publish)
                return carry

            if hi > lo:
                jax.lax.fori_loop(lo, hi, body, 0)

        if defer_res:
            with _scope("pf_res"):
                for t in range(n_pre):
                    res_tile(t, True)
            if two_pass:
                # phase-1 sends land during the first half of the second pass, phase 2 during
                # the rest
                def relay():
                    for t in range(n_pre):
                        col.ag_relay(kp, t)

                unpool_pass(ra, Ts - ra, False, relay)
            dfin = [
                pltpu.make_async_copy(f_v, f_hbm, fin_sem.at[2]),
                pltpu.make_async_copy(inv_v, inv_hbm, fin_sem.at[3]),
            ]
            for cp in dfin:
                cp.start()
            if not two_pass:
                with _scope("pf_relay"):
                    for t in range(n_pre):
                        col.ag_relay(kp, t)
        elif two_pass:
            # the n_pre tiles' gather runs under the second unpool pass
            with _scope("pf_res"):
                for t in range(n_pre):
                    res_tile(t, True)
            unpool_pass(ra, Ts - ra, False)
            with _scope("pf_relay"):
                for t in range(n_pre):
                    col.ag_relay(kp, t)
            with _scope("pf_res"):
                loop(n_pre, nt)
        else:
            with _scope("pf_res"):
                for t in range(n_pre):
                    res_tile(t, True)
                loop(n_pre, mid)
            with _scope("pf_relay"):
                for t in range(n_pre):
                    col.ag_relay(kp, t)
            with _scope("pf_res"):
                loop(mid, nt)
        with _scope("pf_partner"):
            for t in range(n_pre):
                col.ag_to_partner(kp, t)
        with _scope("pf_agwait"):
            for t in range(n_pre):
                col.ag_wait(kp, t)
        fin = [
            pltpu.make_async_copy(pad, xg_hbm, fin_sem.at[0]),
            pltpu.make_async_copy(sc_pad, xsc_hbm, fin_sem.at[1]),
        ]
        for cp in fin:
            cp.start()
        for t in range(max(0, n_res - 2), n_res):
            st_out(t).wait()
        for cp in fin + (dfin if defer_res else []):
            cp.wait()

    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA

    def call(lyr, y, stream, wunpool, smalls):
        if y.shape != (Ts, He) or stream.shape != (Ts, H):
            raise ValueError(f"y {y.shape} / stream {stream.shape}")
        scratch = [
            vmem((Ts, He), y.dtype),
            vmem((Ts, He), jnp.float8_e4m3fn),
            vmem((Ts, 1), jnp.float32),
            vmem((2, He, ncw), jnp.float8_e4m3fn),
            vmem((Ts, H), jnp.bfloat16),
            vmem((2, 16, H), jnp.float32),
            vmem((2, seg, H), jnp.float32),
            vmem((2, seg, H), jnp.float32),
            vmem((n_pre, tm, H), jnp.float8_e4m3fn),
            vmem((n_pre, tp, 8, 128), jnp.float32),
            vmem((Ts, 1), jnp.float32),
            sem((3,)),
            sem((2,)),
            sem((2,)),
            sem((2,)),
            sem((4,)),
        ] + [sem((n_pre,)) for _ in range(2 * len(_KINDS))]
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((Ts, H), jnp.float32),
                jax.ShapeDtypeStruct((n_pre, tm, H), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((n_pre, tp, 8, 128), jnp.float32),
            )
            + (
                (
                    jax.ShapeDtypeStruct((Ts, H), jnp.bfloat16),
                    jax.ShapeDtypeStruct((Ts, 1), jnp.float32),
                )
                if defer_res
                else ()
            ),
            in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] + [hbm] * 4,
            out_specs=(hbm,) * (5 if defer_res else 3),
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=SPTP_POST_FFN_COLLECTIVE_ID, vmem_limit_bytes=60 * 2**20
            ),
            name=f"muse_sptp_post_ffn_t{T}_s{seg}_p{n_pre}_c{ncw}{'_2p' if two_pass else ''}{'_dr' if defer_res else ''}",
            **({"interpret": interpret} if interpret is not None else {}),
        )(lyr, y, stream, wunpool, smalls)

    return call
