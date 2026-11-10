# SPDX-License-Identifier: Apache-2.0
"""The replicated pool / unpool of the EP FFN section on this rank's rows: [K, N] fp8 tiles
of DN_TK x DN_TN (one expert-ring slot each) stream from HBM through the TP expert ring,
outside the dense ring, which leaves them out (dense_skip)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.config import DN_TK, DN_TN
from sglang_tpu.muse_megakernel.mk.config import MKConfig


def stream_matmul(
    c: MKConfig,
    S: dict,
    w_hbm,
    layer,
    x_cols: Callable[[int, int], jax.Array],
    scale: jax.Array,
    K: int,
    N: int,
    on_cols: Callable[[int, jax.Array], None],
    skip: bool = False,
    n_off: jax.Array | int = 0,
    tn: int = DN_TN,
) -> None:
    """x [rows, K] bf16 @ w_hbm[layer] columns [n_off, n_off + N) ([L, K, *] fp8, per-column
    scale [1, N] f32 of those columns); x_cols(k0, n) loads x's columns [k0, k0 + n),
    on_cols(n0, y) gets each finished [rows, tn] f32 column tile (n0 from 0). skip: timing
    only, no stream and no output."""
    ring, sems = S["w13"], S["ep_dn_sems"]
    if skip:
        return
    nslot = ring.shape[0]
    nk, nn = K // DN_TK, N // tn
    tiles = [(ni, ki) for ni in range(nn) for ki in range(nk)]
    w32 = w_hbm.bitcast(jnp.uint32)  # [L, K / 4, N]

    def copy(t):
        ni, ki = tiles[t]
        c0 = n_off + ni * tn
        return pltpu.make_async_copy(
            w32.at[
                layer,
                pl.ds(ki * DN_TK // 4, DN_TK // 4),
                pl.ds(c0 if isinstance(c0, int) else pl.multiple_of(c0, tn), tn),
            ],
            ring.at[t % nslot, pl.ds(0, DN_TK // 4), pl.ds(0, tn)],
            sems.at[t % nslot],
        )

    depth = nslot - 1
    for t in range(min(depth, len(tiles))):
        copy(t).start()
    acc = None
    for t, (ni, ki) in enumerate(tiles):
        if t + depth < len(tiles):
            copy(t + depth).start()
        copy(t).wait()
        w = ring.at[t % nslot].bitcast(jnp.float8_e4m3fn)[pl.ds(0, DN_TK), pl.ds(0, tn)]
        part = jnp.dot(x_cols(ki * DN_TK, DN_TK), w, preferred_element_type=jnp.float32)
        acc = part if ki == 0 else acc + part
        if ki == nk - 1:
            on_cols(ni * tn, acc * scale[:, ni * tn : (ni + 1) * tn])
