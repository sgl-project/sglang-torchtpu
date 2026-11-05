"""Scatter a step's new rows and KDA checkpoints into serving's pools in place.

A separate local kernel: XLA copies every operand that a kernel with remote DMA
aliases, so the megakernel emits its rows as outputs and this one writes them.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pools as PL


def row_write_call(spec, cfg):
    """`f(meta, pos, tab, ckrow, nkv, nraw, nrec, recur, conv, *pools) -> pools`.

    Row r = b*Q + qi is token pos[b]+qi of request b. The Q checkpoint rows of a
    request are contiguous (`TpuMambaPool.checkpoint_rows`), one DMA per layer.
    """
    B, Q, R = cfg.n_req, cfg.n_query, cfg.rows
    ps, kp, pps = cfg.page_size, spec.index_kpool, cfg.pages_per_seq
    n_dsa, n_kda = len(spec.dsa_layers), len(spec.kda_layers)
    n_pool = n_dsa * 3 + n_kda * 2
    if Q > 8:
        raise ValueError(f"n_query={Q}: a step closes at most 8 pools per request")

    def kernel(meta_ref, pos_ref, tab_ref, ckrow_ref, nkv, nraw, nrec, recur, conv, *rest):
        out = rest[n_pool : 2 * n_pool]
        sems = rest[2 * n_pool]
        nope, raw, rec = out[:n_dsa], out[n_dsa : 2 * n_dsa], out[2 * n_dsa : 3 * n_dsa]
        tmp, cnv = out[3 * n_dsa : 3 * n_dsa + n_kda], out[3 * n_dsa + n_kda :]
        n_live = meta_ref[0]
        issued = []

        def dma(cond, src, dst, sem):
            cp = pltpu.make_async_copy(src, dst, sem)
            pl.when(cond)(cp.start)
            issued.append((cond, cp))

        for b in range(B):
            live = b < n_live
            p = pos_ref[b]
            fp0, fresh = PL.closing_pools(p, Q, kp)
            for qi in range(Q):
                r = b * Q + qi
                tok = p + qi
                page = PL.page_of(tab_ref, b, tok // ps, pps)
                off = tok % ps
                pool_tok = (fp0 + qi) * kp
                rpage = PL.page_of(tab_ref, b, pool_tok // ps, pps)
                ri = (pool_tok % ps) // kp
                for d in range(n_dsa):
                    dma(live, nkv.at[d, r], nope[d].at[page, off], sems.at[0, d])
                    dma(
                        live,
                        nraw.at[d, pl.ds(r, 1)],
                        raw[d].at[page, pl.ds(off, 1)],
                        sems.at[1, d],
                    )
                    dma(
                        live & (qi < fresh),
                        nrec.at[d, pl.ds(r, 1)],
                        rec[d].at[rpage, ri // 4, pl.ds(ri % 4, 1)],
                        sems.at[2, d],
                    )
            row0 = ckrow_ref[b * Q]
            for k in range(n_kda):
                dma(
                    live,
                    recur.at[k, pl.ds(b * Q, Q)],
                    tmp[k].at[pl.ds(row0, Q)],
                    sems.at[3, k],
                )
                dma(
                    live,
                    conv.at[k, pl.ds(b * Q, Q)],
                    cnv[k].at[pl.ds(row0, Q)],
                    sems.at[4, k],
                )
        for cond, cp in issued:
            pl.when(cond)(cp.wait)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    n_in = 9

    def call(meta, pos, tab, ckrow, nkv, nraw, nrec, recur, conv, *pools):
        assert len(pools) == n_pool, (len(pools), n_pool)
        return pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(c.shape, c.dtype) for c in pools),
            in_specs=[smem] * 4 + [anyspace] * (5 + n_pool),
            out_specs=(anyspace,) * n_pool,
            scratch_shapes=[pltpu.SemaphoreType.DMA((5, max(n_dsa, n_kda)))],
            input_output_aliases={n_in + i: i for i in range(n_pool)},
            name=f"glm_flash_mk_write_r{R}",
        )(meta, pos, tab, ckrow, nkv, nraw, nrec, recur, conv, *pools)

    return call
