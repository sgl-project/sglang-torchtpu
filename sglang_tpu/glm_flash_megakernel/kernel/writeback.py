"""Scatter a step's new nope rows and KDA checkpoints into serving's pools in place.

A separate local kernel: XLA copies every operand that a kernel with remote DMA
aliases, so the megakernel emits its rows as outputs and this one writes them.
Raw index rows and pooled records are sub-tile slices of their pools, which a DMA
cannot address; `runner` scatters those with XLA, as serving's kpool step does.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pools as PL


def row_write_call(spec, cfg):
    """`f(meta, pos, tab, ckrow, nkv, recur, conv, *nope, *temporal, *conv) -> pools`.

    Row r = b*Q + qi is token pos[b]+qi of request b. The Q checkpoint rows of a
    request are contiguous (`TpuMambaPool.checkpoint_rows`), one DMA per layer.
    """
    B, Q, R = cfg.n_req, cfg.n_query, cfg.rows
    ps, pps = cfg.page_size, cfg.pages_per_seq
    n_dsa, n_kda = len(spec.dsa_layers), len(spec.kda_layers)
    n_pool = n_dsa + n_kda * 2

    def kernel(meta_ref, pos_ref, tab_ref, ckrow_ref, nkv, recur, conv, *rest):
        out = rest[n_pool : 2 * n_pool]
        sems = rest[2 * n_pool]
        nope = out[:n_dsa]
        tmp, cnv = out[n_dsa : n_dsa + n_kda], out[n_dsa + n_kda :]
        n_live = meta_ref[0]
        issued = []

        def dma(cond, src, dst, sem):
            cp = pltpu.make_async_copy(src, dst, sem)
            pl.when(cond)(cp.start)
            issued.append((cond, cp))

        for b in range(B):
            live = b < n_live
            p = pos_ref[b]
            for qi in range(Q):
                r = b * Q + qi
                tok = p + qi
                page = PL.page_of(tab_ref, b, tok // ps, pps)
                off = tok % ps
                for d in range(n_dsa):
                    dma(live, nkv.at[d, r], nope[d].at[page, off], sems.at[0, d])
            row0 = ckrow_ref[b * Q]
            for k in range(n_kda):
                dma(
                    live,
                    recur.at[k, pl.ds(b * Q, Q)],
                    tmp[k].at[pl.ds(row0, Q)],
                    sems.at[1, k],
                )
                dma(
                    live,
                    conv.at[k, pl.ds(b * Q, Q)],
                    cnv[k].at[pl.ds(row0, Q)],
                    sems.at[2, k],
                )
        for cond, cp in issued:
            pl.when(cond)(cp.wait)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    n_in = 7

    def call(meta, pos, tab, ckrow, nkv, recur, conv, *pools):
        assert len(pools) == n_pool, (len(pools), n_pool)
        return pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(c.shape, c.dtype) for c in pools),
            in_specs=[smem] * 4 + [anyspace] * (3 + n_pool),
            out_specs=(anyspace,) * n_pool,
            scratch_shapes=[pltpu.SemaphoreType.DMA((3, max(n_dsa, n_kda)))],
            input_output_aliases={n_in + i: i for i in range(n_pool)},
            name=f"glm_flash_mk_write_r{R}",
        )(meta, pos, tab, ckrow, nkv, recur, conv, *pools)

    return call
