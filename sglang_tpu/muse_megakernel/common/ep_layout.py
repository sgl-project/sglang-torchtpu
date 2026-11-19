# SPDX-License-Identifier: Apache-2.0
"""The EP weight layout the Muse EP programs share: hidden / intermediate rows in kperm order, w13
columns in chunk_cols order, fp4 optionally requantized to SGLANG_TPU_MUSE_MK_QBLOCK blocks of
the kperm order."""

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel.common.consts import (
    E2M1_MAX,
    FP4_GROUP,
    FP4_PACK,
    QBLOCK_SIZES,
)

# kperm permutes within blocks of this many indices (the block-diagonal dot's K chunk)
KPERM_BLOCK = 256
E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def kperm(n: int) -> np.ndarray:
    """The kernels' order of n indices: in each 256-block, position r * 16 + g holds index
    g * 16 + r, so a 16-row bf16 vreg of dequantized weights spans 16 scale groups and takes
    the block's [16, N] scales as they are (an involution)."""
    p = np.arange(n)
    return p - p % KPERM_BLOCK + (p % 16) * 16 + (p % KPERM_BLOCK) // 16


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


def qblock() -> int:
    """SGLANG_TPU_MUSE_MK_QBLOCK: fp4 experts requantized to one scale per that many K rows of
    the kernel's (kperm) order, scale slabs [K / block, N]; each dot takes raw codes and one
    scale row."""
    b = tpu_envs.SGLANG_TPU_MUSE_MK_QBLOCK.get()
    if b not in (0, *QBLOCK_SIZES):
        raise ValueError(f"SGLANG_TPU_MUSE_MK_QBLOCK={b}: 0 or one of {QBLOCK_SIZES}")
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


def ep_expert_weights(
    kind: str,
    w13: jax.Array,
    s13: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    chunk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One rank's whole experts in the EP kernel's layout, from the natural one ([E, H_e/pack,
    2I] w13, [E, I/pack, H_e] w2, int32 words): hidden rows in kperm order; for fp4
    the gate / up columns and w2's rows in kperm order too (the dense arm's scale layout);
    then w13's columns in chunk_cols order (chunk: the MKConfig's inter_local). Under
    SGLANG_TPU_MUSE_MK_QBLOCK, fp4 rows are requantized to its blocks of the kperm order."""
    cols = chunk_cols(w13.shape[-1], chunk)
    if kind == "fp4" and qblock():
        u32 = jax.numpy.uint32
        w13k, s13k = requant_fp4_kperm(w13.view(u32), s13, qblock())
        w2k, s2k = requant_fp4_kperm(w2.view(u32), s2, qblock())
        gu = kperm(w13.shape[-1])
        return (
            w13k.view(jax.numpy.int32)[..., gu][..., cols],
            s13k[..., gu][..., cols],
            w2k.view(jax.numpy.int32),
            s2k,
        )
    if kind == "fp4":
        w13k, s13k, w2k, s2k = fp4_weights_kperm(
            w13.view(jax.numpy.uint32), s13, w2.view(jax.numpy.uint32), s2
        )
        return (
            w13k.view(jax.numpy.int32)[..., cols],
            s13k[..., cols],
            w2k.view(jax.numpy.int32),
            s2k,
        )
    return (
        permute_fp8_rows(w13.view(jax.numpy.uint32)).view(jax.numpy.int32)[..., cols],
        s13[..., cols],
        w2,
        s2,
    )
