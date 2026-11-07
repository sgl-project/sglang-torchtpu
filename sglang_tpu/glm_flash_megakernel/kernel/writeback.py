"""Scatter a step's new nope rows into serving's pools in place.

A separate local kernel: XLA copies every operand that a kernel with remote DMA
aliases, so the megakernel emits its rows as outputs and this one writes them.
Raw index rows and pooled records are sub-tile slices of their pools, which a DMA
cannot address, and the KDA checkpoints are too big for an HBM->HBM DMA (~33 GB/s
on v7x); `runner` scatters those with XLA, as serving's kpool step does.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pools as PL


def row_write_call(spec, cfg):
    """`f(meta, pos, tab, nkv, *nope) -> nope`. Row r = b*Q + qi is token pos[b]+qi of request b."""
    B, Q, R = cfg.n_req, cfg.n_query, cfg.rows
    ps, pps = cfg.page_size, cfg.pages_per_seq
    n_dsa = len(spec.dsa_layers)

    def kernel(meta_ref, pos_ref, tab_ref, nkv, *rest):
        nope = rest[n_dsa : 2 * n_dsa]
        sems = rest[2 * n_dsa]
        n_live = meta_ref[0]
        issued = []
        for b in range(B):
            live = b < n_live
            p = pos_ref[b]
            for qi in range(Q):
                r = b * Q + qi
                tok = p + qi
                page = PL.page_of(tab_ref, b, tok // ps, pps)
                off = tok % ps
                for d in range(n_dsa):
                    cp = pltpu.make_async_copy(
                        nkv.at[d, r], nope[d].at[page, off], sems.at[d]
                    )
                    pl.when(live)(cp.start)
                    issued.append((live, cp))
        for live, cp in issued:
            pl.when(live)(cp.wait)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    n_in = 4

    def call(meta, pos, tab, nkv, *nope):
        assert len(nope) == n_dsa, (len(nope), n_dsa)
        return pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(c.shape, c.dtype) for c in nope),
            in_specs=[smem] * 3 + [anyspace] * (1 + n_dsa),
            out_specs=(anyspace,) * n_dsa,
            scratch_shapes=[pltpu.SemaphoreType.DMA((n_dsa,))],
            input_output_aliases={n_in + i: i for i in range(n_dsa)},
            name=f"glm_flash_mk_write_r{R}",
        )(meta, pos, tab, nkv, *nope)

    return call
