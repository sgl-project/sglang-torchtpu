# SPDX-License-Identifier: Apache-2.0
"""SP + TP qkvg card as one Pallas kernel: gathered e4m3 rows (row scales) @ this rank's e4m3 wqkvg,
then the qk head norms, rope and the q / kv / gate split per row tile in VMEM (XLA spends as long
on relayouts for the 3-D head norm as on the fp8 dot)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel_v2.common.config import PFConfig
from sglang_tpu.muse_megakernel_v2.common.consts import small_row
from sglang_tpu.muse_megakernel_v2.common.numerics import _rope_roll
from sglang_tpu.muse_megakernel_v2.common.qkv import qkv_from_proj
from sglang_tpu.muse_megakernel_v2.common.trace import _scope

# fp8 contraction depth per MXU pass (common/dots._fp8_dot)
_KC = 512


def _heads_rms_mxu(x, n_heads: int, d: int, eps: float, gain=1.0):
    """common.numerics._heads_rms without per-head lane slices: per-head sums of squares and
    their broadcast back as two f32 dots with a 0/1 head matrix."""
    w = n_heads * d
    blk = (
        jax.lax.broadcasted_iota(jnp.int32, (w, 128), 0) // d
        == jax.lax.broadcasted_iota(jnp.int32, (w, 128), 1)
    ).astype(jnp.float32)
    hi = jax.lax.Precision.HIGHEST
    ss = jnp.dot(x * x, blk, precision=hi, preferred_element_type=jnp.float32)
    inv = jax.lax.rsqrt(ss * (1.0 / d) + eps)
    inv_w = jax.lax.dot_general(
        inv,
        blk,
        (((1,), (1,)), ((), ())),
        precision=hi,
        preferred_element_type=jnp.float32,
    )
    return x * inv_w * gain


def qkv_epilogue(c: PFConfig, qkvg, rope):
    """common.qkv.qkv_from_proj with the head norms on the MXU."""
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    q = _heads_rms_mxu(qkvg[:, :qd], c.q_heads, hd, c.rms_eps, gain=c.scale_query_by)
    k = _heads_rms_mxu(qkvg[:, qd : qd + kvd], c.kv_heads, hd, c.rms_eps)
    if rope is not None:
        q = _rope_roll(q, c.q_heads, hd, rope[:, :hd], rope[:, hd:])
        k = _rope_roll(k, c.kv_heads, hd, rope[:, :hd], rope[:, hd:])
    kv = jnp.concatenate([k, qkvg[:, qd + kvd : qd + 2 * kvd]], axis=1)
    return (
        q.astype(jnp.bfloat16),
        kv.astype(jnp.bfloat16),
        qkvg[:, qd + 2 * kvd :].astype(jnp.bfloat16),
    )


def _q_rpa(q, kvh: int, g: int, d: int):
    """[tm, kvh * g * d] heads -> [kvh, tm, g, 128] with each head's d lanes zero-padded to 128 (the
    batched RPA kernel's query layout), the padding as a 0/1 dot (exact for bf16)."""
    w = g * d
    pad = (
        jax.lax.broadcasted_iota(jnp.int32, (w, g * 128), 0) // d * 128
        + jax.lax.broadcasted_iota(jnp.int32, (w, g * 128), 0) % d
        == jax.lax.broadcasted_iota(jnp.int32, (w, g * 128), 1)
    ).astype(jnp.bfloat16)
    parts = []
    for k in range(kvh):
        qp = jnp.dot(q[:, k * w : (k + 1) * w], pad, preferred_element_type=jnp.float32)
        parts.append(qp.astype(jnp.bfloat16).reshape(q.shape[0], g, 128))
    return jnp.stack(parts)


def qkvg_call(
    c: PFConfig,
    tm: int,
    rope: bool,
    mxu_norm: bool = True,
    kc: int = _KC,
    q_rpa: bool = False,
):
    """(x8 [T, H] e4m3, xs [T, 1] f32, w [H, nq] e4m3, sm [16, H] f32, rope [T, 2 hd] f32) ->
    (q [T, qd], kv [T, 2 kvd], gate [T, qd]) bf16; rope False: a NoPE layer (rope unread)."""
    T, H, nq = c.rows, c.hidden, c.qkvg_dim
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    if T % tm or H % kc:
        raise ValueError(f"{T} rows / hidden {H} do not tile by {tm} / {kc}")

    def kernel(x_ref, xs_ref, w_ref, sm_ref, rope_ref, q_ref, kv_ref, g_ref):
        with _scope("qk_dot"):
            acc = None
            for k0 in range(0, H, kc):
                part = jnp.dot(
                    x_ref[:, k0 : k0 + kc],
                    w_ref[k0 : k0 + kc, :],
                    preferred_element_type=jnp.float32,
                )
                acc = part if acc is None else acc + part
            qkvg = acc * xs_ref[...] * small_row(sm_ref[...], "wqkvg_s")[:, :nq]
        with _scope("qk_epilogue"):
            epi = qkv_epilogue if mxu_norm else qkv_from_proj
            q, kv, g = epi(c, qkvg, rope_ref[...] if rope else None)
        if q_rpa:
            q_ref[...] = _q_rpa(q, c.kv_heads, c.q_heads // c.kv_heads, hd)
        else:
            q_ref[...] = q
        kv_ref[...] = kv
        g_ref[...] = g

    rows = lambda w: pl.BlockSpec((tm, w), lambda i: (i, 0))
    whole = lambda a, b: pl.BlockSpec((a, b), lambda i: (0, 0))
    gq = c.q_heads // c.kv_heads
    q_shape = (c.kv_heads, T, gq, 128) if q_rpa else (T, qd)
    outs = (
        jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct((T, 2 * kvd), jnp.bfloat16),
        jax.ShapeDtypeStruct((T, qd), jnp.bfloat16),
    )
    params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",), vmem_limit_bytes=60 * 2**20
    )
    name = (
        f"muse_sptp_qkvg_t{T}_m{tm}_k{kc}_r{int(rope)}{'_mxun' if mxu_norm else ''}"
        f"{'_qrpa' if q_rpa else ''}"
    )

    def q_spec(blk):
        if q_rpa:
            return pl.BlockSpec((c.kv_heads, tm, gq, 128), lambda i: (0, blk(i), 0, 0))
        return pl.BlockSpec((tm, qd), lambda i: (blk(i), 0))

    def call(x8, xs, w, sm, rope_rows):
        return pl.pallas_call(
            kernel,
            grid=(T // tm,),
            in_specs=[
                rows(H),
                rows(1),
                whole(H, nq),
                whole(sm.shape[0], H),
                rows(2 * hd),
            ],
            out_specs=(q_spec(lambda i: i), rows(2 * kvd), rows(qd)),
            out_shape=outs,
            compiler_params=params,
            name=name,
        )(x8, xs, w, sm, rope_rows)

    def chunk_call(chunk, n_chunks, x8, xs, w, sm, rope_rows, prev):
        """Rows of one all-gathered chunk (rank r's rows [chunk * Ts / n, (chunk + 1) * Ts / n),
        rank-major) into their places in the [T, .] outputs, which alias prev (the previous
        chunk's outputs; None: the first chunk). xs: every row's scale [T, 1] (final order)."""
        ts = T // c.tp
        sub = ts // n_chunks
        if sub % tm:
            raise ValueError(f"{sub} rows per rank and chunk do not tile by {tm}")
        bpr = sub // tm

        def dst(j):
            return ((j // bpr) * ts + chunk * sub) // tm + j % bpr

        drows = lambda wd: pl.BlockSpec((tm, wd), lambda j: (dst(j), 0))
        args = [x8, xs, w, sm, rope_rows]
        specs = [rows(H), drows(1), whole(H, nq), whole(sm.shape[0], H), drows(2 * hd)]
        aliases = {}
        if prev is not None:
            args += list(prev)
            specs += [pl.BlockSpec(memory_space=pl.ANY)] * 3
            aliases = {5: 0, 6: 1, 7: 2}

        def body(*refs):
            kernel(*refs[:5], *refs[-3:])

        return pl.pallas_call(
            body,
            grid=(c.tp * bpr,),
            in_specs=specs,
            out_specs=(q_spec(dst), drows(2 * kvd), drows(qd)),
            out_shape=outs,
            input_output_aliases=aliases,
            compiler_params=params,
            name=f"{name}_c{chunk}of{n_chunks}",
        )(*args)

    call.chunk = chunk_call
    return call


def dequant_rows_call(rows: int, cols: int, tm: int = 512):
    """(x8 [rows, cols] e4m3, s [rows, 1] f32) -> bf16 x8 * s, written row-major: the fused-RS
    MoE reads its input in that layout (XLA's dequant fusion picked a transposed one and a
    layout-conversion copy)."""
    if rows % tm:
        raise ValueError(f"{rows} rows do not tile by {tm}")

    def kernel(x_ref, s_ref, o_ref):
        o_ref[...] = (x_ref[...].astype(jnp.float32) * s_ref[...]).astype(jnp.bfloat16)

    def call(x8, s):
        return pl.pallas_call(
            kernel,
            grid=(rows // tm,),
            in_specs=[
                pl.BlockSpec((tm, cols), lambda i: (i, 0)),
                pl.BlockSpec((tm, 1), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((tm, cols), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct((rows, cols), jnp.bfloat16),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            name=f"muse_sptp_dequant_t{rows}_c{cols}",
        )(x8, s)

    return call


def o_gate_call(c: PFConfig, tm: int = 256):
    """(o [Hkv, T, G / 2, 2, 128] bf16 raw batched-RPA output, gate [T, qd] bf16) ->
    (e4m3 [T, qd], f32 [T, 1] row scale) of sigmoid(gate) * head_rms(o), token-major for the
    row a2a; XLA spent ~2.9 ms / chunk of layout copies on this between the RPA and the a2a."""
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
        parts = []
        for k in range(hkv):
            ok = o_ref[k].reshape(tm, gq * 128)
            parts.append(jnp.dot(ok, unpad, preferred_element_type=jnp.float32))
        o = jnp.concatenate(parts, axis=1)
        a = jax.nn.sigmoid(g_ref[...].astype(jnp.float32)) * _heads_rms_mxu(
            o, c.q_heads, hd, c.rms_eps
        )
        s = jnp.maximum(jnp.max(jnp.abs(a), axis=1, keepdims=True), 1e-30) / 448.0
        x_ref[...] = (a / s).astype(jnp.float8_e4m3fn)
        s_ref[...] = s

    def call(o, gate):
        o4 = o.reshape(hkv, T, gq, 128)
        return pl.pallas_call(
            kernel,
            grid=(T // tm,),
            in_specs=[
                pl.BlockSpec((hkv, tm, gq, 128), lambda i: (0, i, 0, 0)),
                pl.BlockSpec((tm, qd), lambda i: (i, 0)),
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
            name=f"muse_sptp_o_gate_t{T}_m{tm}",
        )(o4, gate)

    return call
