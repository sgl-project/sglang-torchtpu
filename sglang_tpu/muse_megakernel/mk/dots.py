# SPDX-License-Identifier: Apache-2.0
"""NVFP4 / fp8 weight dequantization and the matvec dots on them."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.consts import FP4_GROUP, FP4_PACK
from sglang_tpu.muse_megakernel.mk.packing import _quant_fp8_rows


def _dequant_fp4(words, scale, via_fp8=False):
    """``[cw, N]`` u32 + ``[cw*8/16, N]`` scales -> ``[cw*8, N]`` bf16."""
    cw, n = words.shape
    f4 = pltpu.bitcast(words, jnp.float4_e2m1fn)
    v = (
        f4.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
        if via_fp8
        else f4.astype(jnp.bfloat16)
    )
    gq = cw * FP4_PACK // FP4_GROUP
    s = scale if scale.dtype == jnp.bfloat16 else scale.astype(jnp.bfloat16)
    s = jnp.broadcast_to(s[:, None, :], (gq, FP4_GROUP, n)).reshape(cw * FP4_PACK, n)
    return v * s


def _dequant_fp4_kperm(words, scale, via_fp8=False, out_fp8=False, vec=False):
    """Zero-shuffle NVFP4 dequant for K permuted as (B, r=16, g=16).

    ``out_fp8``: the dequantized weights go to the MXU as e4m3 (ex_kperm >= 3: the
    block scales are normalized so every weight is at most WFP8_TARGET_MAX).
    """
    cw, n = words.shape
    if vec and out_fp8 and cw == 64:
        # Dequantize two 32-word (256-row) halves sequentially into fp8 so peak bf16
        # live registers stay [256, n] (32 vregs at n=256) instead of [512, n] (64 vregs).
        w0 = _dequant_fp4_kperm(
            words[:32], scale[:16], via_fp8=via_fp8, out_fp8=True, vec=True
        )
        w1 = _dequant_fp4_kperm(
            words[32:], scale[16:], via_fp8=via_fp8, out_fp8=True, vec=True
        )
        return jnp.concatenate([w0, w1], axis=0)
    nb = cw // 32
    f4 = pltpu.bitcast(words, jnp.float4_e2m1fn)
    v = (
        f4.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
        if via_fp8
        else f4.astype(jnp.bfloat16)
    )
    s = scale if scale.dtype == jnp.bfloat16 else scale.astype(jnp.bfloat16)
    if vec:
        res = (v.reshape(nb, 16, 16, n) * s.reshape(nb, 1, 16, n)).reshape(
            cw * FP4_PACK, n
        )
    else:
        parts = []
        for b in range(nb):
            sb = s[b * 16 : (b + 1) * 16]
            base = b * 256
            for r in range(16):
                parts.append(v[base + r * 16 : base + (r + 1) * 16] * sb)
        res = jnp.concatenate(parts, axis=0)
    return res.astype(jnp.float8_e4m3fn) if out_fp8 else res


def _scale_rows(sslot, g0, gn, N, blk, n0=0):
    """Scale rows [g0, g0 + gn) of a slab as f32.

    e4m3 slabs tile 32 rows: a narrower slice reads its whole tile once (cached in
    blk) and slices it.
    """
    if sslot.dtype != jnp.float8_e4m3fn or gn % 32 == 0:
        return sslot[pl.ds(g0, gn), pl.ds(n0, N)].astype(jnp.float32), blk
    if g0 % 32 == 0 or blk is None:
        blk = sslot[pl.ds(g0 - g0 % 32, 32), pl.ds(n0, N)].astype(jnp.float32)
    return blk[g0 % 32 : g0 % 32 + gn], blk


QBLOCK_TN = 512  # output columns per accumulator (up to 32 rows)


def _fp4_qblock_dot(
    x: jax.Array | None,
    wslot,
    sslot,
    K: int,
    N: int,
    block: int,
    lhs_chunk_fn: Callable[[int, int, bool], jax.Array] | None = None,
    out_stage_ref=None,
    post_scale: jax.Array | None = None,
    nrows: int | None = None,
) -> jax.Array | None:
    """x [R, K] @ an NVFP4 slab requantized to one scale per block K rows (scale rows
    [0, K / block) of sslot) -> f32 [R, N]: raw codes go to the MXU as e4m3 (exact) and each
    block's partial takes its scale row after the dot, as ep_mk.experts.qblock_dot."""
    m = nrows if nrows is not None else x.shape[0]
    tn_step = min(N, QBLOCK_TN if m <= 32 else QBLOCK_TN // 2)
    xb = None if x is None else x.astype(jnp.bfloat16)
    cols = []
    for n0 in range(0, N, tn_step):
        tn = min(tn_step, N - n0)
        acc = None
        for k0 in range(0, K, block):
            w = pltpu.bitcast(
                wslot[pl.ds(k0 // FP4_PACK, block // FP4_PACK), pl.ds(n0, tn)],
                jnp.float4_e2m1fn,
            ).astype(jnp.float8_e4m3fn)
            if lhs_chunk_fn is not None:
                lhs = lhs_chunk_fn(k0 // FP4_PACK, block // FP4_PACK, False).astype(
                    jnp.bfloat16
                )
            else:
                lhs = xb[:, k0 : k0 + block]
            part = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
            s = sslot[pl.ds(k0 // block, 1), pl.ds(n0, tn)].astype(jnp.float32)
            acc = part * s if acc is None else acc + part * s
        if post_scale is not None:
            acc = acc * post_scale
        if out_stage_ref is not None:
            out_stage_ref[:, pl.ds(n0, tn)] = acc.astype(out_stage_ref.dtype)
        else:
            cols.append(acc)
    if out_stage_ref is not None:
        return None
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)


def _fp4_qblock_ffn(
    x: jax.Array, w13, s13, w2, s2, He: int, Il: int, block: int
) -> jax.Array:
    """silu(x @ gate) * (x @ up) @ w2 on block-scaled slabs -> f32 [R, He], all in values
    (no VMEM staging, so the LLO sees per-vreg dependencies from w13 through w2)."""
    gu = _fp4_qblock_dot(x, w13, s13, He, 2 * Il, block)
    gate, up = gu[:, :Il], gu[:, Il:]
    h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
    return _fp4_qblock_dot(h, w2, s2, Il, He, block)


def _fp4_dot(
    x,
    wslot,
    sslot,
    K,
    N,
    chunk_words,
    fast_bf16=False,
    kperm=0,
    via_fp8=False,
    vec_dequant=False,
    fp8_lhs=False,
    n_tile=0,
    lhs_chunk_fn=None,
    prequant_sc=None,
    out_stage_ref=None,
    post_scale=None,
    force_bf16_w=False,
    nrows=None,
    qblock=0,
):
    if qblock:
        if prequant_sc is not None:
            raise ValueError("qblock fp4 dots take a bf16 LHS (no prequantized fp8 x)")
        return _fp4_qblock_dot(
            x,
            wslot,
            sslot,
            K,
            N,
            qblock,
            lhs_chunk_fn,
            out_stage_ref,
            post_scale,
            nrows,
        )
    kw = K // FP4_PACK
    m_rows = (
        nrows
        if nrows is not None
        else (
            x.shape[0]
            if x is not None
            else (out_stage_ref.shape[0] if out_stage_ref is not None else 0)
        )
    )
    can_fp8_m = m_rows >= 32 and m_rows % 32 == 0
    use_fp8_lhs = (fp8_lhs or prequant_sc is not None) and kperm >= 3 and can_fp8_m
    if kperm > 0:
        # Native fp8 x fp8 vmatmul.e4m3 contracts K = 512 (64 words) per MXU pass.
        chunk_words = min(kw, 64 if use_fp8_lhs else (32 if vec_dequant else 64))
    elif fast_bf16:
        chunk_words = min(kw, 32)
    use_bf16 = (fast_bf16 or kperm > 0) and not use_fp8_lhs
    xf = x_sc = xb = None
    if lhs_chunk_fn is None:
        if use_fp8_lhs and prequant_sc is None:
            xf, x_sc = _quant_fp8_rows(x)
        elif use_fp8_lhs:
            xf, x_sc = x, prequant_sc
        else:
            xb = (
                (x if x.dtype == jnp.bfloat16 else x.astype(jnp.bfloat16))
                if use_bf16
                else None
            )
    else:
        x_sc = prequant_sc
    out_fp8_w = (kperm >= 3) and not (force_bf16_w and not use_fp8_lhs)
    direct_bf16 = (use_bf16 or use_fp8_lhs) and sslot.dtype == jnp.bfloat16
    tn_step = max(256, n_tile // 4) if (n_tile > 0 and N > 512) else N
    cols = [] if out_stage_ref is None else None
    for n0 in range(0, N, tn_step):
        tn = min(tn_step, N - n0)
        acc = blk = None
        for c in range(0, kw, chunk_words):
            cw = min(chunk_words, kw - c)
            g0 = c * FP4_PACK // FP4_GROUP
            gq = cw * FP4_PACK // FP4_GROUP
            if direct_bf16:
                sc = sslot[pl.ds(g0, gq), pl.ds(n0, tn)]
            else:
                sc, blk = _scale_rows(sslot, g0, gq, tn, blk, n0=n0)
            if kperm > 0:
                w = _dequant_fp4_kperm(
                    wslot[pl.ds(c, cw), pl.ds(n0, tn)],
                    sc,
                    via_fp8=(via_fp8 or kperm >= 2),
                    out_fp8=out_fp8_w,
                    vec=vec_dequant,
                )
            else:
                w = _dequant_fp4(
                    wslot[pl.ds(c, cw), pl.ds(n0, tn)], sc, via_fp8=via_fp8
                )
            if lhs_chunk_fn is not None:
                lhs = lhs_chunk_fn(c, cw, use_fp8_lhs)
            elif use_fp8_lhs:
                lhs = xf[:, c * FP4_PACK : (c + cw) * FP4_PACK]
            elif xb is not None:
                lhs = xb[:, c * FP4_PACK : (c + cw) * FP4_PACK]
            else:
                lhs = x[:, c * FP4_PACK : (c + cw) * FP4_PACK].astype(jnp.bfloat16)
            part = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
            acc = part if acc is None else acc + part
        if use_fp8_lhs and x_sc is not None:
            acc = acc * (x_sc * post_scale if post_scale is not None else x_sc)
        elif post_scale is not None:
            acc = acc * post_scale
        if out_stage_ref is not None:
            out_stage_ref[:, pl.ds(n0, tn)] = (
                acc if out_stage_ref.dtype == jnp.float32 else acc.astype(jnp.bfloat16)
            )
        else:
            cols.append(acc)
    if out_stage_ref is not None:
        return None
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)


def _fp4_blockdiag_dot(
    xc,
    wslot,
    sslot,
    K,
    N,
    kc,
    kperm=False,
    fp8_lhs=False,
    n_tile=0,
    lhs_chunk_fn=None,
    prequant_sc=None,
    nrows=None,
    post_scale=None,
    qblock=0,
):
    """``xc [A, K]`` bf16 @ an NVFP4 slab (``wslot`` u32 words ``[K/8, N]``, block-16 scales

    ``sslot [K/16, N]``) -> f32 ``[A, N]``, exact in f32 (or native fp8xfp8 when
    ``fp8_lhs``). qblock: the slab is block-scaled instead and A rows run a plain dot.
    """
    if qblock:
        if prequant_sc is not None or lhs_chunk_fn is not None:
            raise ValueError("qblock fp4 dots take a bf16 LHS (no prequantized fp8 x)")
        return _fp4_qblock_dot(
            xc, wslot, sslot, K, N, qblock, post_scale=post_scale, nrows=nrows
        )
    A = (
        nrows
        if nrows is not None
        else (xc.shape[0] if xc is not None else prequant_sc.shape[0])
    )
    if kperm:
        assert kc == 256, f"kperm blockdiag requires kc=256, got {kc}"
    kc = min(K, kc)
    gpc = kc // FP4_GROUP
    rows = A * gpc
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 1)
    diag = jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 0) % gpc == (
        col_iota % FP4_GROUP if kperm else col_iota // FP4_GROUP
    )
    can_fp8_m = A >= 32 and A % 32 == 0
    use_fp8_lhs = (fp8_lhs or prequant_sc is not None) and can_fp8_m
    if lhs_chunk_fn is None:
        if use_fp8_lhs and prequant_sc is None:
            xf, x_sc = _quant_fp8_rows(xc)
        elif use_fp8_lhs:
            xf, x_sc = xc, prequant_sc
        else:
            xf = x_sc = None
    else:
        xf = None
        x_sc = prequant_sc
    z_lhs = jnp.zeros((rows, kc), jnp.float8_e4m3fn if use_fp8_lhs else jnp.bfloat16)

    def make_lhs(c0):
        if lhs_chunk_fn is not None:
            xs = lhs_chunk_fn(c0 // FP4_PACK, kc // FP4_PACK, use_fp8_lhs)
        elif use_fp8_lhs:
            xs = xf[:, c0 : c0 + kc]
        else:
            xs = xc[:, c0 : c0 + kc].astype(jnp.bfloat16)
        return jnp.where(
            diag,
            jnp.broadcast_to(xs[:, None, :], (A, gpc, kc)).reshape(rows, kc),
            z_lhs,
        )

    # At A >= 4 (rows >= 64), cap column tile at 512 ([64, 512] f32 = 32 vregs) and
    # compute make_lhs(c0) on the fly so %v never spills on the 64-vreg v7x file.
    bd_tn = min(n_tile, 512) if (n_tile > 0 and A >= 4) else n_tile
    if bd_tn > 0 and N > bd_tn and A >= 4:
        cols = []
        for n0 in range(0, N, bd_tn):
            tn = min(bd_tn, N - n0)
            acc = blk = None
            for c0 in range(0, K, kc):
                lhs = make_lhs(c0)
                w = pltpu.bitcast(
                    wslot[pl.ds(c0 // FP4_PACK, kc // FP4_PACK), pl.ds(n0, tn)],
                    jnp.float4_e2m1fn,
                ).astype(jnp.float8_e4m3fn)
                res = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
                s, blk = _scale_rows(sslot, c0 // FP4_GROUP, gpc, tn, blk, n0=n0)
                part = res * (jnp.concatenate([s] * A, axis=0) if A > 1 else s)
                acc = part if acc is None else acc + part
            cols.append(
                jnp.concatenate(
                    [
                        jnp.sum(acc[a * gpc : (a + 1) * gpc], axis=0, keepdims=True)
                        for a in range(A)
                    ],
                    axis=0,
                )
            )
        out = jnp.concatenate(cols, axis=1)
        sc_mul = x_sc if (use_fp8_lhs and x_sc is not None) else None
        if post_scale is not None:
            sc_mul = post_scale if sc_mul is None else sc_mul * post_scale
        return out * sc_mul if sc_mul is not None else out

    acc = blk = None
    for c0 in range(0, K, kc):
        lhs = make_lhs(c0)
        w = pltpu.bitcast(
            wslot[pl.ds(c0 // FP4_PACK, kc // FP4_PACK), pl.ds(0, N)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        res = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
        s, blk = _scale_rows(sslot, c0 // FP4_GROUP, gpc, N, blk)
        part = res * (jnp.concatenate([s] * A, axis=0) if A > 1 else s)
        acc = part if acc is None else acc + part
    out = jnp.concatenate(
        [
            jnp.sum(acc[a * gpc : (a + 1) * gpc], axis=0, keepdims=True)
            for a in range(A)
        ],
        axis=0,
    )
    sc_mul = x_sc if (use_fp8_lhs and x_sc is not None) else None
    if post_scale is not None:
        sc_mul = post_scale if sc_mul is None else sc_mul * post_scale
    return out * sc_mul if sc_mul is not None else out


def _fp8_dot(
    x,
    wslot,
    scale_row,
    K,
    N,
    chunk_words,
    fp8_lhs=False,
    lhs_chunk_fn=None,
    prequant_sc=None,
    out_stage_ref=None,
    post_scale=None,
    nrows=None,
):
    """fp8 weights held as ``[K/4, N]`` u32 words; channelwise ``scale_row [1, N]``."""
    kw = K // 4
    m_rows = (
        nrows
        if nrows is not None
        else (
            x.shape[0]
            if x is not None
            else (out_stage_ref.shape[0] if out_stage_ref is not None else 0)
        )
    )
    can_fp8_m = m_rows >= 32 and m_rows % 32 == 0
    use_fp8_lhs = (fp8_lhs or prequant_sc is not None) and can_fp8_m
    if use_fp8_lhs:
        chunk_words = min(kw, 128)  # 128 u32 words = 512 fp8 contraction depth
    xf = x_sc = None
    if lhs_chunk_fn is None:
        if use_fp8_lhs and prequant_sc is None:
            xf, x_sc = _quant_fp8_rows(x)
        elif use_fp8_lhs:
            xf, x_sc = x, prequant_sc
    else:
        x_sc = prequant_sc
    tn_step = 256 if (out_stage_ref is not None and N > 256) else N
    cols = [] if out_stage_ref is None else None
    for n0 in range(0, N, tn_step):
        tn = min(tn_step, N - n0)
        acc = None
        for c in range(0, kw, chunk_words):
            cw = min(chunk_words, kw - c)
            w = pltpu.bitcast(wslot[pl.ds(c, cw), pl.ds(n0, tn)], jnp.float8_e4m3fn)
            if lhs_chunk_fn is not None:
                lhs = lhs_chunk_fn(c // 2, cw // 2, use_fp8_lhs)
            elif use_fp8_lhs:
                lhs = xf[:, c * 4 : (c + cw) * 4]
            else:
                lhs = x[:, c * 4 : (c + cw) * 4].astype(jnp.bfloat16)
            part = jnp.dot(
                lhs,
                w,
                preferred_element_type=jnp.float32,
            )
            acc = part if acc is None else acc + part
        sc_sub = scale_row[:, n0 : n0 + tn]
        row_mul = x_sc if (use_fp8_lhs and x_sc is not None) else None
        if post_scale is not None:
            row_mul = post_scale if row_mul is None else row_mul * post_scale
        acc = acc * (row_mul * sc_sub) if row_mul is not None else acc * sc_sub
        if out_stage_ref is not None:
            out_stage_ref[:, pl.ds(n0, tn)] = (
                acc if out_stage_ref.dtype == jnp.float32 else acc.astype(jnp.bfloat16)
            )
        else:
            cols.append(acc)
    if out_stage_ref is not None:
        return None
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)
