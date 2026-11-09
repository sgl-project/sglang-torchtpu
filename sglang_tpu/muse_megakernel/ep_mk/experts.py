# SPDX-License-Identifier: Apache-2.0
"""Whole-expert compute on one streamed I-chunk: a work item of at most 4 slots goes through
the block-diagonal fp8 dot (exact; VPU work grows with the slots), a busier one dequantizes
each weight block to bf16 once for all its rows. NVFP4 weights and x come in kperm order so a
dequantized vreg takes its block scales without a broadcast."""

import os
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.consts import FP4_GROUP, FP4_PACK
from sglang_tpu.muse_megakernel.mk.dots import _fp4_blockdiag_dot, _fp8_dot

BD_KC = 256
BD_N_TILE = 256  # [128, 256] f32 accumulators: 32 vregs
BD_ARMS = (
    2,
    4,
)  # block-diagonal arms: an item of at most 4 slots (1 slot: Mosaic lays out a
# one-row block-diagonal LHS's mask in the megakernel only as replicated rows, and fails)
DENSE_ARMS = (8, 16, 32, 64)  # bf16-dequant arms: rows of the item's plain dot
W2_N_TILE = 512  # down-projection columns per dot below 4 slots ([32, 512] f32 at 2)
DQ_KC, DQ_TN = 256, 256  # one bf16-dequantized weight block: 32 vregs
FP8_CHUNK_WORDS = 128  # K words per fp8 dot (512 rows)
QB_SIZES = (64, 128, 256)  # SGLANG_TPU_EP_QBLOCK: K rows per scale
QB_TN = 512  # q arm output columns per accumulator
E2M1_MAX = 6.0  # e2m1's largest magnitude
E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class ChunkRefs(NamedTuple):
    """One I-chunk of an expert in VMEM: gate / up columns, their block scales, w2 rows."""

    wg: Any
    wu: Any
    sg: Any
    su: Any
    w2: Any
    s2: Any


def kperm(n: int) -> np.ndarray:
    """The kernels' order of n indices: in each 256-block, position r * 16 + g holds index
    g * 16 + r, so a 16-row bf16 vreg of dequantized weights spans 16 scale groups and takes
    the block's [16, N] scales as they are (an involution)."""
    p = np.arange(n)
    return p - p % BD_KC + (p % 16) * 16 + (p % BD_KC) // 16


def chunk_cols(two_i: int, chunk: int) -> np.ndarray:
    """w13's (and its scales') column order in the EP kernel: chunk j's gate columns then its
    up columns, so one copy moves a chunk's [., 2 chunk] slab."""
    i = two_i // 2
    j = np.arange(i // chunk)[:, None, None]
    half = np.arange(2)[None, :, None] * i
    return (half + j * chunk + np.arange(chunk)[None, None, :]).reshape(-1)


def permute_fp4_rows(words: jax.Array) -> jax.Array:
    """u32 words [..., K/8, N] (nibble j of word w is row 8w + j) with rows in kperm order."""
    *lead, kw, n = words.shape
    shifts = jnp.arange(0, 32, 4, dtype=jnp.uint32)[:, None]
    codes = ((words[..., :, None, :] >> shifts) & 0xF).reshape(*lead, kw * FP4_PACK, n)
    codes = codes[..., kperm(kw * FP4_PACK), :].reshape(*lead, kw, FP4_PACK, n)
    return jnp.sum(codes << shifts, axis=-2, dtype=jnp.uint32)


def permute_fp8_rows(words: jax.Array) -> jax.Array:
    """u32 words [..., K/4, N] (byte j of word w is row 4w + j) with rows in kperm order."""
    *lead, kw, n = words.shape
    shifts = jnp.arange(0, 32, 8, dtype=jnp.uint32)[:, None]
    b = ((words[..., :, None, :] >> shifts) & 0xFF).reshape(*lead, kw * 4, n)
    b = b[..., kperm(kw * 4), :].reshape(*lead, kw, 4, n)
    return jnp.sum(b << shifts, axis=-2, dtype=jnp.uint32)


def fp4_weights_kperm(
    w13: jax.Array, s13: jax.Array, w2: jax.Array, s2: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The chunk arms' NVFP4 layout from the natural one: hidden rows of w13 and intermediate
    rows of w2 in kperm order, and the gate / up columns in kperm order too, so h comes out in
    the order w2's rows are in. Block scales keep their group order."""
    cols = kperm(w13.shape[-1])
    return permute_fp4_rows(w13)[..., cols], s13[..., cols], permute_fp4_rows(w2), s2


def x_kperm(x: jax.Array) -> jax.Array:
    """x's columns in kperm order."""
    return x[..., kperm(x.shape[-1])]


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


def qblock() -> int:
    """SGLANG_TPU_EP_QBLOCK in QB_SIZES: fp4 experts requantized to one scale per that many K
    rows of the kernel's (kperm) order, scale slabs [K / block, N]; each dot takes raw codes
    and one scale row."""
    b = int(os.environ.get("SGLANG_TPU_EP_QBLOCK", "0"))
    if b not in (0, *QB_SIZES):
        raise ValueError(f"SGLANG_TPU_EP_QBLOCK={b}: 0 or one of {QB_SIZES}")
    return b


def requant_fp4_kperm(
    words: jax.Array, scale: jax.Array, block: int
) -> tuple[jax.Array, jax.Array]:
    """Natural NVFP4 u32 words [..., K/8, N] + block-16 scales [..., K/16, N] -> words with rows
    in kperm order, requantized to one scale per block rows of that order (amax / 6, bf16,
    e2m1 codes re-rounded), and the scales [..., K / block, N]. Lossy."""
    *lead, kw, n = words.shape
    k = kw * FP4_PACK
    shifts = jnp.arange(0, 32, 4, dtype=jnp.uint32)[:, None]
    codes = ((words[..., :, None, :] >> shifts) & 0xF).reshape(*lead, k, n)
    mags = jnp.asarray(E2M1_MAGS)
    v = jnp.where(codes >= 8, -1.0, 1.0) * mags[codes & 7]
    w = (v * jnp.repeat(scale.astype(jnp.float32), FP4_GROUP, axis=-2))[
        ..., kperm(k), :
    ]
    wb = w.reshape(*lead, k // block, block, n)
    sb = (jnp.max(jnp.abs(wb), axis=-2) / E2M1_MAX).astype(jnp.bfloat16)
    sbf = sb.astype(jnp.float32)
    q = jnp.clip(wb / jnp.where(sbf > 0, sbf, 1.0)[..., None, :], -E2M1_MAX, E2M1_MAX)
    qa = jnp.abs(
        q.astype(jnp.float4_e2m1fn).astype(jnp.float32)
    )  # round to nearest even
    idx = jnp.argmax(qa[..., None] == mags, axis=-1).astype(jnp.uint32)
    qc = (idx | jnp.where(q < 0, 8, 0).astype(jnp.uint32)).reshape(
        *lead, kw, FP4_PACK, n
    )
    out = jnp.sum(qc << shifts, axis=-2, dtype=jnp.uint32)
    return out, sb


def scale_group(kind: str) -> int:
    """K rows per scale row of a chunk's fp4 scale slabs."""
    assert kind == "fp4", kind
    return qblock() or FP4_GROUP


def qblock_dot(x, wref, sref, K, N, block):
    """x [R, K] bf16 @ the codes of a block-scaled NVFP4 slab -> f32 [R, N]: each dot pushes
    raw codes (as e4m3) of one block and scales its [R, tn] partial by the block's row."""
    cols = []
    for n0 in range(0, N, QB_TN):
        tn = min(QB_TN, N - n0)
        acc = None
        for k0 in range(0, K, block):
            w = pltpu.bitcast(
                wref[pl.ds(k0 // FP4_PACK, block // FP4_PACK), pl.ds(n0, tn)],
                jnp.float4_e2m1fn,
            ).astype(jnp.float8_e4m3fn)
            part = jnp.dot(x[:, k0 : k0 + block], w, preferred_element_type=jnp.float32)
            s = sref[pl.ds(k0 // block, 1), pl.ds(n0, tn)].astype(jnp.float32)
            acc = part * s if acc is None else acc + part * s
        cols.append(acc)
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)


def _q_arm(
    R: int, B: int, x_rows: Callable, acc, w: ChunkRefs, He: int, CI: int
) -> None:
    """Up to R slots on B-row block-scaled weights: plain R-row dots on the codes."""
    xt = x_rows(R).astype(jnp.bfloat16)
    g = qblock_dot(xt, w.wg, w.sg, He, CI, B)
    u = qblock_dot(xt, w.wu, w.su, He, CI, B)
    h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16)
    for n0 in range(0, He, QB_TN):
        cols = pl.ds(n0, QB_TN)
        acc[pl.ds(0, R), cols] += qblock_dot(
            h, w.w2.at[:, cols], w.s2.at[:, cols], CI, QB_TN, B
        )


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


def bd_arms() -> tuple[int, ...]:
    """The block-diagonal arms' rows: BD_ARMS, or up to SGLANG_TPU_EP_BD_MAX (A/B)."""
    top = int(os.environ.get("SGLANG_TPU_EP_BD_MAX", str(BD_ARMS[-1])))
    return tuple(A for A in (2, 4, 8, 16) if A <= top)


def item_sizes(item_rows: int) -> tuple[int, ...]:
    """Rows an item loads, computes and stores: the dense arm sizes up to item_rows."""
    sizes = tuple(z for z in DENSE_ARMS if z <= item_rows)
    assert sizes[-1] == item_rows, (item_rows, DENSE_ARMS)
    return sizes


def chunk_compute(
    kind: str,
    n: jax.Array,
    x_rows: Callable[[int], jax.Array],
    acc,
    w: ChunkRefs,
    He: int,
    CI: int,
    item_rows: int,
    prologue: Callable[[], None] | None = None,
) -> None:
    """acc[:n] += the n slots' expert output over this I-chunk.

    Args:
      x_rows: x_rows(R) -> f32 [R, He] (bf16 values), the item's first R rows (kperm order
        for fp4).
      acc: f32 [item_rows, He] ref.
      prologue: run first inside each arm, so its scalar work (the ring's DMA issue and
        waits) shares the arm's bundles.
    """
    sizes = item_sizes(item_rows)
    if kind == "fp4" and qblock():
        arms = [lambda R=R: _q_arm(R, qblock(), x_rows, acc, w, He, CI) for R in sizes]
        bounds = sizes[:-1]
        names = [f"q{R}" for R in sizes]
    elif kind == "fp4":
        bd = bd_arms()
        dense = tuple(R for R in sizes if R > bd[-1])
        arms = [lambda A=A: _bd_arm(A, x_rows, acc, w, He, CI) for A in bd]
        arms += [lambda R=R: _dense_arm(R, x_rows, acc, w, He, CI) for R in dense]
        bounds = bd + dense[:-1]
        names = [f"bd{A}" for A in bd] + [f"dq{R}" for R in dense]
    else:
        arms = [lambda R=R: _fp8_arm(R, x_rows, acc, w, He, CI) for R in sizes]
        bounds = sizes[:-1]
        names = [f"f8_{R}" for R in sizes]
    if prologue is not None:
        arms = [lambda f=f: (prologue(), f()) for f in arms]
    if (
        int(os.environ.get("SGLANG_TPU_EP_TRACE", "0")) >= 2
    ):  # each arm a region (xprof)
        arms = [lambda f=f, m=m: _named(f"ep_arm_{m}", f) for f, m in zip(arms, names)]
    jax.lax.switch(sum((n > b).astype(jnp.int32) for b in bounds), arms)


def _named(name: str, fn: Callable[[], None]) -> None:
    with jax.named_scope(name):
        fn()


def sized(n: jax.Array, sizes: tuple[int, ...], fn: Callable[[int], None]) -> None:
    """fn(S) at the smallest of sizes holding n rows (DMA sizes are static)."""
    which = sum((n > z).astype(jnp.int32) for z in sizes[:-1])
    jax.lax.switch(which, [lambda z=z: fn(z) for z in sizes])
