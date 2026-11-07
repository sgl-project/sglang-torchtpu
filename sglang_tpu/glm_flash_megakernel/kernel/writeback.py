"""Scatter a step's new nope rows and KDA conv checkpoints into serving's pools in place.

A separate local kernel: XLA copies every operand that a kernel with remote DMA
aliases, so the megakernel emits its rows as outputs and this one writes them.
Raw index rows and pooled records are sub-tile slices of their pools, which a DMA
cannot address, and the KDA temporal checkpoints are too big for an HBM->HBM DMA
(~33 GB/s on v7x); `runner` scatters those with XLA, as serving's kpool step does.
The conv checkpoints (36 KB per row) go here: an XLA scatter on the [slots, K-1, 1,
dim] pool squeezes the unit dim and relayouts the whole pool three times per layer.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pools as PL


def row_write_call(spec, cfg):
    """`f(meta, pos, tab, ckrow, nkv, conv, *nope, *conv_pools) -> (*nope, *conv_pools)`.
    Row r = b*Q + qi is token pos[b]+qi of request b; its conv checkpoint goes to
    pool row ckrow[r]."""
    B, Q, R = cfg.n_req, cfg.n_query, cfg.rows
    ps, pps = cfg.page_size, cfg.pages_per_seq
    n_dsa, n_kda = len(spec.dsa_layers), len(spec.kda_layers)

    def kernel(meta_ref, pos_ref, tab_ref, ckrow_ref, nkv, conv, *rest):
        nope = rest[n_dsa + n_kda : 2 * n_dsa + n_kda]
        cpool = rest[2 * n_dsa + n_kda : 2 * (n_dsa + n_kda)]
        sems, csems = rest[2 * (n_dsa + n_kda)], rest[2 * (n_dsa + n_kda) + 1]
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
                row = ckrow_ref[r]
                for k in range(n_kda):
                    cp = pltpu.make_async_copy(
                        conv.at[k, r], cpool[k].at[row], csems.at[k]
                    )
                    pl.when(live)(cp.start)
                    issued.append((live, cp))
        for live, cp in issued:
            pl.when(live)(cp.wait)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    n_in = 6

    def call(meta, pos, tab, ckrow, nkv, conv, *pools):
        assert len(pools) == n_dsa + n_kda, (len(pools), n_dsa, n_kda)
        return pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(c.shape, c.dtype) for c in pools),
            in_specs=[smem] * 4 + [anyspace] * (2 + len(pools)),
            out_specs=(anyspace,) * len(pools),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((n_dsa,)),
                pltpu.SemaphoreType.DMA((n_kda,)),
            ],
            input_output_aliases={n_in + i: i for i in range(len(pools))},
            name=f"glm_flash_mk_write_r{R}",
        )(meta, pos, tab, ckrow, nkv, conv, *pools)

    return call
