# SPDX-License-Identifier: Apache-2.0
"""Pallas o-gate card: raw batched-RPA output + gate -> sigmoid(gate) * head_rms(o) as token-major
e4m3 rows and row scales for the row a2a. Port of junyanxu/mk-v2-sptp-proto f9d38ba14f's
o_gate_call, plus gate4: gate in the qkvg kernel's [Hkv, T, G, 128] head layout (SPTP_XAG_GATE4),
unpadded by the same 0/1 dot as o."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel_v2.common.config import PFConfig
from sglang_tpu.muse_megakernel_v2.prefill.sptp.qkvg import _heads_rms_mxu


def o_gate_call(c: PFConfig, tm: int = 256, gate4: bool = False):
    """(o [Hkv, T, G / 2, 2, 128] bf16 raw batched-RPA output, gate [T, qd] bf16, or
    [Hkv, T, G, 128] with gate4) -> (e4m3 [T, qd], f32 [T, 1] row scale)."""
    T, qd, hd = c.rows, c.q_dim, c.head_dim
    hkv, gq = c.kv_heads, c.q_heads // c.kv_heads
    w = gq * hd
    if T % tm:
        raise ValueError(f"{T} rows do not tile by {tm}")

    def kernel(o_ref, g_ref, x_ref, s_ref):
        # 0/1 dot: each head's first hd of 128 lanes -> dense [tm, G * hd] (exact for bf16)
        unpad = (
            jax.lax.broadcasted_iota(jnp.int32, (gq * 128, w), 0)
            == jax.lax.broadcasted_iota(jnp.int32, (gq * 128, w), 1) // hd * 128
            + jax.lax.broadcasted_iota(jnp.int32, (gq * 128, w), 1) % hd
        ).astype(jnp.bfloat16)

        def dense(ref):
            parts = []
            for k in range(hkv):
                rk = ref[k].reshape(tm, gq * 128)
                parts.append(jnp.dot(rk, unpad, preferred_element_type=jnp.float32))
            return jnp.concatenate(parts, axis=1)

        o = dense(o_ref)
        g = dense(g_ref) if gate4 else g_ref[...].astype(jnp.float32)
        a = jax.nn.sigmoid(g) * _heads_rms_mxu(o, c.q_heads, hd, c.rms_eps)
        s = jnp.maximum(jnp.max(jnp.abs(a), axis=1, keepdims=True), 1e-30) / 448.0
        x_ref[...] = (a / s).astype(jnp.float8_e4m3fn)
        s_ref[...] = s

    head = lambda: pl.BlockSpec((hkv, tm, gq, 128), lambda i: (0, i, 0, 0))

    def call(o, gate):
        o4 = o.reshape(hkv, T, gq, 128)
        return pl.pallas_call(
            kernel,
            grid=(T // tm,),
            in_specs=[
                head(),
                head() if gate4 else pl.BlockSpec((tm, qd), lambda i: (i, 0)),
            ],
            out_specs=(
                pl.BlockSpec((tm, qd), lambda i: (i, 0)),
                pl.BlockSpec((tm, 1), lambda i: (i, 0)),
            ),
            out_shape=(
                jax.ShapeDtypeStruct((T, qd), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((T, 1), jnp.float32),
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            name=f"muse_sptp_o_gate_t{T}_m{tm}{'_g4' if gate4 else ''}",
        )(o4, gate)

    return call


def o_gate_pk_call(c: PFConfig, tm: int = 256, v: int = 1):
    """(o [Hkv, T, G / 2, 2, 128] bf16 raw batched-RPA output, gate [T, qd] bf16) -> (e4m3
    [T, qd], f32 [T, 1] row scale), the XLA o gate's math (SPTP_PA_OGATE_TOK) in one pass.

    The RPA writes each head pair as one (2, 128) bf16 tile, i.e. one u32 row whose halves are
    heads 2p and 2p + 1: a stride-G/2 u32 load gives a pair of every token, the halves widen to f32
    exactly, and a 64-lane roll packs the pair into gate's / the output's 128 lanes. No relayout."""
    T, qd, hd = c.rows, c.q_dim, c.head_dim
    hkv, gq = c.kv_heads, c.q_heads // c.kv_heads
    npair = gq // 2
    if T % tm or hd != 64 or gq % 2:
        raise ValueError(f"o gate pk: {T} rows / tile {tm}, head_dim {hd}, group {gq}")

    def kernel_v12(o_ref, g_ref, x_ref, s_ref, a_ref):
        # v12 / v13: 32-row chunks (one e4m3 vreg per 128 lanes), statically unrolled so the
        # stride-G/2 loads have static starts; head sums of squares as two bf16 dots (hi + lo),
        # not a 6-pass f32 dot. v12 quantizes from registers; v13 writes [qd, T] (SPTP_PA_TMAJ's
        # layout) from a VMEM copy, one 128 x 128 transpose per block.
        R = 32
        ou = o_ref.bitcast(jnp.uint32)
        lane = jax.lax.broadcasted_iota(jnp.int32, (R, 128), 1)
        lo_half = lane < hd
        blk = (
            jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0) // hd
            == jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1) // hd
        ).astype(jnp.bfloat16)
        for ci in range(tm // R):
            r0 = ci * R
            m = jnp.zeros((R, 128), jnp.float32)
            acts = []
            for k in range(hkv):
                for p in range(npair):
                    w = ou[k, pl.ds(r0 * npair + p, R, stride=npair), :]
                    ev = pltpu.bitcast(w << 16, jnp.float32)
                    od = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), jnp.float32)
                    x2 = jnp.where(lo_half, ev, pltpu.roll(od, hd, 1))
                    q = x2 * x2
                    qh = q.astype(jnp.bfloat16)
                    ql = (q - qh.astype(jnp.float32)).astype(jnp.bfloat16)
                    ss = jnp.dot(qh, blk, preferred_element_type=jnp.float32) + jnp.dot(
                        ql, blk, preferred_element_type=jnp.float32
                    )
                    col = k * npair * 128 + p * 128
                    gf = g_ref[pl.ds(r0, R), col : col + 128].astype(jnp.float32)
                    a = jax.nn.sigmoid(gf) * x2 * jax.lax.rsqrt(ss / hd + c.rms_eps)
                    if v == 13:
                        a_ref[pl.ds(r0, R), col : col + 128] = a
                    else:
                        acts.append(a)
                    m = jnp.maximum(m, jnp.abs(a))
            sc = jnp.maximum(jnp.max(m, axis=1, keepdims=True), 1e-30) / 448.0
            s_ref[pl.ds(r0, R), :] = sc
            if v == 13:
                continue
            inv = 1.0 / sc
            for j, a in enumerate(acts):
                x_ref[pl.ds(r0, R), j * 128 : (j + 1) * 128] = jnp.clip(
                    a * inv, -448.0, 448.0
                ).astype(jnp.float8_e4m3fn)
        if v != 13:
            return
        for r0 in range(0, tm, 128):
            inv = 1.0 / s_ref[pl.ds(r0, 128), :]
            for j in range(qd // 128):
                a = jnp.clip(
                    a_ref[pl.ds(r0, 128), j * 128 : (j + 1) * 128] * inv, -448.0, 448.0
                )
                x_ref[j * 128 : (j + 1) * 128, pl.ds(r0, 128)] = a.T.astype(
                    jnp.float8_e4m3fn
                )

    def kernel_v2(o_ref, g_ref, x_ref, s_ref, a_ref):
        # v2: each lane's head sum of squares by one f32 dot with the 64-lane block-ones matrix
        # (no XLU lane reductions, already broadcast), one lane max per row at the end
        ou = o_ref.bitcast(jnp.uint32)
        lane = jax.lax.broadcasted_iota(jnp.int32, (tm, 128), 1)
        lo_half = lane < hd
        blk = (
            jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0) // hd
            == jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1) // hd
        ).astype(jnp.float32)
        mabs = jnp.zeros((tm, 128), jnp.float32)
        for k in range(hkv):
            for p in range(npair):
                w = ou[k, pl.ds(p, tm, stride=npair), :]
                ev = pltpu.bitcast(w << 16, jnp.float32)
                od = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), jnp.float32)
                x2 = jnp.where(lo_half, ev, pltpu.roll(od, hd, 1))
                ss = jnp.dot(
                    x2 * x2,
                    blk,
                    preferred_element_type=jnp.float32,
                    precision=jax.lax.Precision.HIGHEST,
                )
                col = k * npair * 128 + p * 128
                gf = g_ref[:, col : col + 128].astype(jnp.float32)
                a = jax.nn.sigmoid(gf) * x2 * jax.lax.rsqrt(ss / hd + c.rms_eps)
                a_ref[:, col : col + 128] = a
                mabs = jnp.maximum(mabs, jnp.abs(a))
        s = jnp.maximum(jnp.max(mabs, axis=1, keepdims=True), 1e-30) / 448.0
        x_ref[...] = jnp.clip(a_ref[...] / s, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        s_ref[...] = s

    def kernel(o_ref, g_ref, x_ref, s_ref, a_ref):
        ou = o_ref.bitcast(jnp.uint32)  # [Hkv, tm * G / 2, 128]
        lane = jax.lax.broadcasted_iota(jnp.int32, (tm, 128), 1)
        lo_half = lane < hd
        m = jnp.zeros((tm, 1), jnp.float32)
        for k in range(hkv):
            for p in range(npair):
                w = ou[k, pl.ds(p, tm, stride=npair), :]
                ev = pltpu.bitcast(w << 16, jnp.float32)
                od = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), jnp.float32)
                sse = jnp.sum(jnp.where(lo_half, ev * ev, 0.0), axis=1, keepdims=True)
                sso = jnp.sum(jnp.where(lo_half, od * od, 0.0), axis=1, keepdims=True)
                ev = ev * jax.lax.rsqrt(sse / hd + c.rms_eps)
                od = od * jax.lax.rsqrt(sso / hd + c.rms_eps)
                o2 = jnp.where(lo_half, ev, pltpu.roll(od, hd, 1))
                col = k * npair * 128 + p * 128
                a = jax.nn.sigmoid(g_ref[:, col : col + 128].astype(jnp.float32)) * o2
                a_ref[:, col : col + 128] = a
                m = jnp.maximum(m, jnp.max(jnp.abs(a), axis=1, keepdims=True))
        s = jnp.maximum(m, 1e-30) / 448.0
        # Mosaic's divide can land a row max just above 448, which e4m3fn has no code for
        x_ref[...] = jnp.clip(a_ref[...] / s, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        s_ref[...] = s

    def call(o, gate):
        o3 = o.reshape(hkv, T * gq, 128)
        return pl.pallas_call(
            {12: kernel_v12, 13: kernel_v12}.get(v, kernel_v2 if v >= 2 else kernel),
            grid=(T // tm,),
            in_specs=[
                pl.BlockSpec((hkv, tm * gq, 128), lambda i: (0, i, 0)),
                pl.BlockSpec((tm, qd), lambda i: (i, 0)),
            ],
            out_specs=(
                pl.BlockSpec((qd, tm), lambda i: (0, i))
                if v == 13
                else pl.BlockSpec((tm, qd), lambda i: (i, 0)),
                pl.BlockSpec((tm, 1), lambda i: (i, 0)),
            ),
            out_shape=(
                jax.ShapeDtypeStruct(
                    (qd, T) if v == 13 else (T, qd), jnp.float8_e4m3fn
                ),
                jax.ShapeDtypeStruct((T, 1), jnp.float32),
            ),
            scratch_shapes=[pltpu.VMEM((tm, qd), jnp.float32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            name=f"muse_sptp_o_gate_pk_t{T}_m{tm}_v{v}",
        )(o3, gate.reshape(T, qd))

    return call
