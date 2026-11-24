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


def qkvg_call(c: PFConfig, tm: int, rope: bool, mxu_norm: bool = True):
    """(x8 [T, H] e4m3, xs [T, 1] f32, w [H, nq] e4m3, sm [16, H] f32, rope [T, 2 hd] f32) ->
    (q [T, qd], kv [T, 2 kvd], gate [T, qd]) bf16; rope False: a NoPE layer (rope unread)."""
    T, H, nq = c.rows, c.hidden, c.qkvg_dim
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    if T % tm or H % _KC:
        raise ValueError(f"{T} rows / hidden {H} do not tile by {tm} / {_KC}")

    def kernel(x_ref, xs_ref, w_ref, sm_ref, rope_ref, q_ref, kv_ref, g_ref):
        with _scope("qk_dot"):
            acc = None
            for k0 in range(0, H, _KC):
                part = jnp.dot(
                    x_ref[:, k0 : k0 + _KC],
                    w_ref[k0 : k0 + _KC, :],
                    preferred_element_type=jnp.float32,
                )
                acc = part if acc is None else acc + part
            qkvg = acc * xs_ref[...] * small_row(sm_ref[...], "wqkvg_s")[:, :nq]
        with _scope("qk_epilogue"):
            epi = qkv_epilogue if mxu_norm else qkv_from_proj
            q, kv, g = epi(c, qkvg, rope_ref[...] if rope else None)
        q_ref[...] = q
        kv_ref[...] = kv
        g_ref[...] = g

    rows = lambda w: pl.BlockSpec((tm, w), lambda i: (i, 0))
    whole = lambda a, b: pl.BlockSpec((a, b), lambda i: (0, 0))

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
            out_specs=(rows(qd), rows(2 * kvd), rows(qd)),
            out_shape=(
                jax.ShapeDtypeStruct((T, qd), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, 2 * kvd), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, qd), jnp.bfloat16),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",), vmem_limit_bytes=60 * 2**20
            ),
            name=f"muse_sptp_qkvg_t{T}_m{tm}_r{int(rope)}{'_mxun' if mxu_norm else ''}",
        )(x8, xs, w, sm, rope_rows)

    return call
