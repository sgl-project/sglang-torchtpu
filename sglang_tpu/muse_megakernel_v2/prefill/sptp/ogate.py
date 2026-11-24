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
