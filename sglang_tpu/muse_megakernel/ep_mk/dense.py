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
) -> None:
    """x [rows, K] bf16 @ w_hbm[layer] ([L, K, N] fp8, per-column scale [1, N] f32);
    x_cols(k0, n) loads x's columns [k0, k0 + n), on_cols(n0, y) gets each finished
    [rows, DN_TN] f32 column tile."""
    ring, sems = S["w13"], S["ep_dn_sems"]
    nslot = ring.shape[0]
    nk, nn = K // DN_TK, N // DN_TN
    tiles = [(ni, ki) for ni in range(nn) for ki in range(nk)]
    w32 = w_hbm.bitcast(jnp.uint32)  # [L, K / 4, N]

    def copy(t):
        ni, ki = tiles[t]
        return pltpu.make_async_copy(
            w32.at[layer, pl.ds(ki * DN_TK // 4, DN_TK // 4), pl.ds(ni * DN_TN, DN_TN)],
            ring.at[t % nslot, pl.ds(0, DN_TK // 4), pl.ds(0, DN_TN)],
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
        w = ring.at[t % nslot].bitcast(jnp.float8_e4m3fn)[
            pl.ds(0, DN_TK), pl.ds(0, DN_TN)
        ]
        part = jnp.dot(x_cols(ki * DN_TK, DN_TK), w, preferred_element_type=jnp.float32)
        acc = part if ki == 0 else acc + part
        if ki == nk - 1:
            on_cols(ni * DN_TN, acc * scale[:, ni * DN_TN : (ni + 1) * DN_TN])
