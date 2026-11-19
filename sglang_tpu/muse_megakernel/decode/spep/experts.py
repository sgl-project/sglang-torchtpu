# SPDX-License-Identifier: Apache-2.0
"""Whole-expert compute on one streamed I-chunk: a work item of at most 4 slots goes through
the block-diagonal fp8 dot (exact; VPU work grows with the slots), a busier one dequantizes
each weight block to bf16 once for all its rows. NVFP4 weights and x come in kperm order so a
dequantized vreg takes its block scales without a broadcast."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel.common.consts import (
    FP4_GROUP,
    FP4_PACK,
)
from sglang_tpu.muse_megakernel.common.dots import (
    QBLOCK_TN,
    _fp4_blockdiag_dot,
    _fp8_dot,
    qblock_dot,
)
from sglang_tpu.muse_megakernel.common.ep_layout import KPERM_BLOCK, qblock

BD_KC = KPERM_BLOCK
BD_N_TILE = 256  # [128, 256] f32 accumulators: 32 vregs
BD_ARMS = (
    2,
    4,
)  # block-diagonal arms: an item of at most 4 slots (1 slot: Mosaic lays out a
# one-row block-diagonal LHS's mask in the megakernel only as replicated rows, and fails)
DENSE_ARMS = (8, 16, 32)  # bf16-dequant arms: rows of the item's plain dot
W2_N_TILE = 512  # down-projection columns per dot below 4 slots ([32, 512] f32 at 2)
DQ_KC, DQ_TN = 256, 256  # one bf16-dequantized weight block: 32 vregs
FP8_CHUNK_WORDS = 128  # K words per fp8 dot (512 rows)


class ChunkRefs(NamedTuple):
    """One I-chunk of an expert in VMEM: gate / up columns, their block scales, w2 rows."""

    wg: Any
    wu: Any
    sg: Any
    su: Any
    w2: Any
    s2: Any


def dequant_dot(x, wref, sref, K, N):
    """x [R, K] bf16 (kperm order) @ an NVFP4 slab dequantized to bf16 per [DQ_KC, DQ_TN]
    block -> f32 [R, N]; one dequant serves every row (code x scale rounded to bf16)."""
    cols = []
    for n0 in range(0, N, DQ_TN):
        acc = None
        for k0 in range(0, K, DQ_KC):
            w = pltpu.bitcast(
                wref[pl.ds(k0 // FP4_PACK, DQ_KC // FP4_PACK), pl.ds(n0, DQ_TN)],
                jnp.float4_e2m1fn,
            ).astype(jnp.bfloat16)
            s = sref[pl.ds(k0 // FP4_GROUP, DQ_KC // FP4_GROUP), pl.ds(n0, DQ_TN)]
            w = (w.reshape(FP4_GROUP, DQ_KC // FP4_GROUP, DQ_TN) * s[None]).reshape(
                DQ_KC, DQ_TN
            )
            part = jnp.dot(x[:, k0 : k0 + DQ_KC], w, preferred_element_type=jnp.float32)
            acc = part if acc is None else acc + part
        cols.append(acc)
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)


def scale_group(kind: str) -> int:
    """K rows per scale row of a chunk's fp4 scale slabs."""
    assert kind == "fp4", kind
    return qblock() or FP4_GROUP


def _q_arm(
    R: int, B: int, x_rows: Callable, acc, w: ChunkRefs, He: int, CI: int
) -> None:
    """Up to R slots on B-row block-scaled weights: plain R-row dots on the codes. h comes in
    B-column blocks and w2 takes each block for every column tile before the next, so the MXU
    streams w2 while the last block's SiLU is in flight."""
    xt = x_rows(R).astype(jnp.bfloat16)
    hs = []
    for c0 in range(0, CI, B):
        g = qblock_dot(xt, w.wg, w.sg, 0, He, c0, B, B)
        u = qblock_dot(xt, w.wu, w.su, 0, He, c0, B, B)
        hs.append((g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16))
    tiles = range(0, He, QBLOCK_TN)
    parts = [
        [qblock_dot(h, w.w2, w.s2, kb * B, B, n0, QBLOCK_TN, B) for n0 in tiles]
        for kb, h in enumerate(hs)
    ]
    for t, n0 in enumerate(tiles):
        y = parts[0][t]
        for p in parts[1:]:
            y = y + p[t]
        acc[pl.ds(0, R), pl.ds(n0, QBLOCK_TN)] += y


def _bd_arm(A: int, x_rows: Callable, acc, w: ChunkRefs, He: int, CI: int) -> None:
    """At most 4 slots: an A-row block-diagonal dot (A * 16 LHS rows)."""
    rows = pl.ds(0, A)
    xt = x_rows(
        A
    )  # f32: the dot casts each K chunk (a bf16 row broadcast fails to lay out)
    g = _fp4_blockdiag_dot(xt, w.wg, w.sg, He, CI, BD_KC, kperm=True, n_tile=BD_N_TILE)
    u = _fp4_blockdiag_dot(xt, w.wu, w.su, He, CI, BD_KC, kperm=True, n_tile=BD_N_TILE)
    h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16)
    if A >= 4:  # _fp4_blockdiag_dot tiles N itself from 4 rows up
        acc[rows, :] += _fp4_blockdiag_dot(
            h, w.w2, w.s2, CI, He, BD_KC, kperm=True, n_tile=BD_N_TILE
        )
        return
    for n0 in range(0, He, W2_N_TILE):
        cols = pl.ds(n0, W2_N_TILE)
        acc[rows, cols] += _fp4_blockdiag_dot(
            h, w.w2.at[:, cols], w.s2.at[:, cols], CI, W2_N_TILE, BD_KC, kperm=True
        )


def _dense_arm(R: int, x_rows: Callable, acc, w: ChunkRefs, He: int, CI: int) -> None:
    """Up to R slots: dequantize each weight block once for all R rows."""
    xt = x_rows(R).astype(jnp.bfloat16)
    g = dequant_dot(xt, w.wg, w.sg, He, CI)
    u = dequant_dot(xt, w.wu, w.su, He, CI)
    h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16)
    for n0 in range(0, He, DQ_TN):
        cols = pl.ds(n0, DQ_TN)
        acc[pl.ds(0, R), cols] += dequant_dot(
            h, w.w2.at[:, cols], w.s2.at[:, cols], CI, DQ_TN
        )


def _fp8_arm(R: int, x_rows: Callable, acc, w: ChunkRefs, He: int, CI: int) -> None:
    """fp8 experts (per-column scales in row 0 of sg = su = the gate|up scale slab, and of s2):
    a mixed bf16 x fp8 dot, exact, any rows."""
    xt = x_rows(R).astype(jnp.bfloat16)
    s13 = w.sg[0:1, :].astype(jnp.float32)  # [1, 2 CI]: gate | up
    s2 = w.s2[0:1, :].astype(jnp.float32)
    g = _fp8_dot(xt, w.wg, s13[:, :CI], He, CI, FP8_CHUNK_WORDS)
    u = _fp8_dot(xt, w.wu, s13[:, CI : 2 * CI], He, CI, FP8_CHUNK_WORDS)
    h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16)
    for n0 in range(0, He, W2_N_TILE):
        cols = pl.ds(n0, W2_N_TILE)
        acc[pl.ds(0, R), cols] += _fp8_dot(
            h, w.w2.at[:, cols], s2[:, n0 : n0 + W2_N_TILE], CI, W2_N_TILE, CI // 4
        )


def item_sizes(item_rows: int) -> tuple[int, ...]:
    """Rows an item loads, computes and stores: the dense arm sizes up to item_rows."""
    sizes = tuple(z for z in DENSE_ARMS if z <= item_rows)
    assert sizes[-1] == item_rows, (item_rows, DENSE_ARMS)
    return sizes


def chunk_arms(
    kind: str,
    x_rows: Callable[[int], jax.Array],
    acc,
    He: int,
    CI: int,
    item_rows: int,
) -> tuple[tuple[int, ...], list[Callable[[ChunkRefs], None]]]:
    """(bounds, arms): arms[arm_index(n, bounds)](w) adds n slots' output over one I-chunk.

    Args:
      x_rows: x_rows(R) -> f32 [R, He] (bf16 values), the item's first R rows (kperm order
        for fp4).
      acc: f32 [item_rows, He] ref.
    """
    sizes = item_sizes(item_rows)
    if kind == "fp4" and qblock():
        arms = [
            lambda w, R=R: _q_arm(R, qblock(), x_rows, acc, w, He, CI) for R in sizes
        ]
        bounds = sizes[:-1]
        names = [f"q{R}" for R in sizes]
    elif kind == "fp4":
        dense = tuple(R for R in sizes if R > BD_ARMS[-1])
        arms = [lambda w, A=A: _bd_arm(A, x_rows, acc, w, He, CI) for A in BD_ARMS]
        arms += [lambda w, R=R: _dense_arm(R, x_rows, acc, w, He, CI) for R in dense]
        bounds = BD_ARMS + dense[:-1]
        names = [f"bd{A}" for A in BD_ARMS] + [f"dq{R}" for R in dense]
    else:
        arms = [lambda w, R=R: _fp8_arm(R, x_rows, acc, w, He, CI) for R in sizes]
        bounds = sizes[:-1]
        names = [f"f8_{R}" for R in sizes]
    if tpu_envs.SGLANG_TPU_MK_TRACE_REGIONS.get() >= 2:  # each arm a region (xprof)
        arms = [
            lambda w, f=f, m=m: _named(f"ep_arm_{m}", lambda: f(w))
            for f, m in zip(arms, names)
        ]
    return bounds, arms


def arm_index(n: jax.Array, bounds: tuple[int, ...]):
    """The arm of chunk_arms that serves n slots."""
    return sum((n > b).astype(jnp.int32) for b in bounds)


def _named(name: str, fn: Callable[[], None]) -> None:
    with jax.named_scope(name):
        fn()


def sized(n: jax.Array, sizes: tuple[int, ...], fn: Callable[[int], None]) -> None:
    """fn(S) at the smallest of sizes holding n rows (DMA sizes are static)."""
    which = sum((n > z).astype(jnp.int32) for z in sizes[:-1])
    jax.lax.switch(which, [lambda z=z: fn(z) for z in sizes])
