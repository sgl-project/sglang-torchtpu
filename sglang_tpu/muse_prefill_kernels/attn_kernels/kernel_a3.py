"""Kernel A3 (`pallas_gate_oproj_rs`) and Option 3D (`a3_option3d_a2a_sp`) for Muse Spark on TPU v7x.

Implements:
  1. `a3_unfused_ar_bf16`: Unfused XLA baseline (sigmoid(gate) * attn_out + BF16 o_proj + full-T psum All-Reduce).
  2. `a3_unfused_rs_fp8w`: Unfused XLA baseline with FP8 weight + Reduce-Scatter (`[T, 8192] -> [T // TP, 8192]`).
  3. `pallas_gate_oproj`: Core fused Pallas kernel for Prologue (`sigmoid(gate.float()).to(bf16) * attn_out`)
     + `o_proj` (`[T, K_local] @ [K_local, 8192]`) with resident `wo_fp8` (`8 MiB` at TP8, `4 MiB` at TP16).
     Supports both `act_dtype="bf16"` (BF16×FP8) and `act_dtype="fp8"` (in-VMEM FP8 row quant + FP8×FP8),
     plus `sub_n` column sub-tiling (`256`, `512`, or `None`) to eliminate VREG accumulator spills on Ghostfish.
  4. `a3_v1_pallas_rs` & `a3_v3_2half_overlap_rs`: Fused Pallas `gate_oproj` + `psum_scatter` (single and 2-half overlapped).
  5. `a3_v4_option3d_a2a_sp` (Fable's Option 3D Sequence-Parallel `o_proj`):
     - Sender computes `gated = sigmoid(gate.float()).to(bf16) * normed_attn_out` (`bf16[T, 8192 // TP]`).
     - `lax.all_to_all` redistributes `[T, 8192 // TP] -> [T // TP, 8192]` on token owner (`14 MiB/rank` at T=8K TP8,
       8x smaller ICI payload than Reduce-Scatter!).
     - Token owner runs LHS-resident, N-streamed Pallas kernel `pallas_option3d_wo_sp` on `[T // TP, 8192] @ Wo_fp8[8192, 8192]`
       producing `o_shard: bf16[T // TP, 8192]` directly with ZERO Reduce-Scatter and single FP32 MXU accumulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernel_a1 import _dual_head_rmsnorm_128
from .reference import FP8_MAX


def _store_head_rmsnorm(src_ref, dst_ref, eps: float) -> None:
    """dst = bf16 per-64-dim-head weightless RMSNorm of src (the model's `_per_head_normalize`)."""
    tm, k = src_ref.shape
    lane = lax.broadcasted_iota(jnp.int32, (tm, 128), 1)
    is_lo = lane < 64
    mask_lo = is_lo.astype(jnp.float32)
    for c in range(0, k, 128):
        dst_ref[:, c : c + 128] = _dual_head_rmsnorm_128(
            src_ref[:, c : c + 128].astype(jnp.float32),
            is_lo,
            mask_lo,
            1.0 - mask_lo,
            1.0,
            eps=eps,
        ).astype(jnp.bfloat16)


def _dot_strip_f32(
    lhs: jax.Array,
    w_ref,
    k_dim: int,
    n0: int,
    tn: int,
    bk: int,
) -> jax.Array:
    """Computes `lhs[:, :k_dim] @ w_ref[:k_dim, n0:n0+tn]` with static K-strip unrolling."""
    if bk == k_dim and n0 == 0 and tn == w_ref.shape[1]:
        return jnp.dot(lhs, w_ref[...], preferred_element_type=jnp.float32)
    acc = jnp.dot(
        lhs[:, 0:bk],
        w_ref[0:bk, n0 : n0 + tn],
        preferred_element_type=jnp.float32,
    )
    for k0 in range(bk, k_dim, bk):
        acc = acc + jnp.dot(
            lhs[:, k0 : k0 + bk],
            w_ref[k0 : k0 + bk, n0 : n0 + tn],
            preferred_element_type=jnp.float32,
        )
    return acc


# -----------------------------------------------------------------------------
# 1. Core Fused Pallas Kernel A3 (`pallas_gate_oproj`)
# -----------------------------------------------------------------------------
def pallas_gate_oproj(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8: jax.Array,
    wo_scale_2d: jax.Array,
    act_dtype: str = "bf16",
    tm: int = 256,
    bk: int | None = None,
    sub_n: int | None = None,
    vmem_limit_mb: int = 60,
    head_norm_eps: float | None = None,
    row_half: tuple[int, int] | None = None,
) -> jax.Array:
    """Fuses `sigmoid(gate.float()).to(bf16) * normed_attn_out` + `o_proj` in Pallas VMEM.

    `head_norm_eps`: the input is the raw attention output; RMS-normalize each head first.
    `row_half=(half, tp_deg)`: read only half `half` of every rank's `T / tp_deg` rows of the
    full `[T, k]` inputs, in place of a sliced copy (rows ordered rank-major, as `a3_v3`).
    """
    t_in, k_local = normed_attn_out.shape
    _, n_dim = wo_fp8.shape
    bk_eff = bk or k_local
    tn_step = sub_n or n_dim
    if row_half is None:
        t_len = t_in
        in_map = lambda i: (i, 0)
    else:
        half, tp_deg = row_half
        t_len = t_in // 2
        blocks_per_half = t_in // tp_deg // 2 // tm
        assert blocks_per_half * tm * 2 * tp_deg == t_in, (t_in, tm)
        in_map = lambda i: (
            (i // blocks_per_half * 2 + half) * blocks_per_half + i % blocks_per_half,
            0,
        )

    def _kernel(attn_ref, gate_ref, w_ref, w_sc_ref, out_ref, *scratch):
        g_sig = jax.nn.sigmoid(gate_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
        if head_norm_eps is not None:
            _store_head_rmsnorm(attn_ref, scratch[0], head_norm_eps)
            attn_ref = scratch[0]
        gated_bf16 = g_sig * attn_ref[...]

        if act_dtype == "fp8":
            gf = gated_bf16.astype(jnp.float32)
            amax = jnp.maximum(jnp.max(jnp.abs(gf), axis=1, keepdims=True), 1e-12)
            row_sc = amax * (1.0 / FP8_MAX)
            lhs = jnp.clip(gf / row_sc, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
        else:
            lhs = gated_bf16
            row_sc = None

        if tn_step == n_dim:
            acc = _dot_strip_f32(lhs, w_ref, k_local, 0, n_dim, bk_eff)
            acc = (
                acc * (row_sc * w_sc_ref[...])
                if row_sc is not None
                else acc * w_sc_ref[...]
            )
            out_ref[...] = acc.astype(jnp.bfloat16)
        else:
            for n0 in range(0, n_dim, tn_step):
                tn = min(tn_step, n_dim - n0)
                acc = _dot_strip_f32(lhs, w_ref, k_local, n0, tn, bk_eff)
                sc_sub = w_sc_ref[:, n0 : n0 + tn]
                acc = acc * (row_sc * sc_sub) if row_sc is not None else acc * sc_sub
                out_ref[:, n0 : n0 + tn] = acc.astype(jnp.bfloat16)

    return pl.pallas_call(
        _kernel,
        grid=(t_len // tm,),
        in_specs=[
            pl.BlockSpec((tm, k_local), in_map),
            pl.BlockSpec((tm, k_local), in_map),
            pl.BlockSpec((k_local, n_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tm, n_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((t_len, n_dim), jnp.bfloat16),
        scratch_shapes=(
            [pltpu.VMEM((tm, k_local), jnp.bfloat16)]
            if head_norm_eps is not None
            else []
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(normed_attn_out, gate, wo_fp8, wo_scale_2d)


# -----------------------------------------------------------------------------
# 2. Option 3D LHS-Resident, N-Streamed Sequence-Parallel `Wo` Pallas Kernel
# -----------------------------------------------------------------------------
def pallas_option3d_wo_sp(
    gated_shard: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    gated_scale_128: jax.Array | None = None,
    tm: int = 512,
    bn: int = 1024,
    bk: int = 2048,
    sub_n: int | None = None,
    vmem_limit_mb: int = 60,
) -> jax.Array:
    """LHS-resident, N-streamed Pallas GEMM for Option 3D Sequence-Parallel `o_proj`.

    Computes `o_shard = (gated_shard @ wo_fp8_full) * wo_scale_2d` on `[T // TP, 8192]`.
    """
    t_shard, k_dim = gated_shard.shape
    _, n_dim = wo_fp8_full.shape
    tm_eff = min(tm, t_shard)
    tn_step = sub_n or bn
    use_fp8_act = gated_shard.dtype == jnp.float8_e4m3fn

    if use_fp8_act:
        assert gated_scale_128 is not None

        def _kernel_3d_fp8(x_ref, x_sc_ref, w_ref, w_sc_ref, out_ref):
            row_sc = x_sc_ref[:, :1]
            if tn_step == bn:
                acc = _dot_strip_f32(x_ref[...], w_ref, k_dim, 0, bn, bk)
                out_ref[...] = (acc * (row_sc * w_sc_ref[...])).astype(jnp.bfloat16)
            else:
                for n0 in range(0, bn, tn_step):
                    tn = min(tn_step, bn - n0)
                    acc = _dot_strip_f32(x_ref[...], w_ref, k_dim, n0, tn, bk)
                    sc_sub = w_sc_ref[:, n0 : n0 + tn]
                    out_ref[:, n0 : n0 + tn] = (acc * (row_sc * sc_sub)).astype(
                        jnp.bfloat16
                    )

        return pl.pallas_call(
            _kernel_3d_fp8,
            grid=(t_shard // tm_eff, n_dim // bn),
            in_specs=[
                pl.BlockSpec((tm_eff, k_dim), lambda i, j: (i, 0)),
                pl.BlockSpec((tm_eff, 128), lambda i, j: (i, 0)),
                pl.BlockSpec((k_dim, bn), lambda i, j: (0, j)),
                pl.BlockSpec((1, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((tm_eff, bn), lambda i, j: (i, j)),
            out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
        )(gated_shard, gated_scale_128, wo_fp8_full, wo_scale_2d)

    else:

        def _kernel_3d_bf16(x_ref, w_ref, w_sc_ref, out_ref):
            if tn_step == bn:
                acc = _dot_strip_f32(x_ref[...], w_ref, k_dim, 0, bn, bk)
                out_ref[...] = (acc * w_sc_ref[...]).astype(jnp.bfloat16)
            else:
                for n0 in range(0, bn, tn_step):
                    tn = min(tn_step, bn - n0)
                    acc = _dot_strip_f32(x_ref[...], w_ref, k_dim, n0, tn, bk)
                    sc_sub = w_sc_ref[:, n0 : n0 + tn]
                    out_ref[:, n0 : n0 + tn] = (acc * sc_sub).astype(jnp.bfloat16)

        return pl.pallas_call(
            _kernel_3d_bf16,
            grid=(t_shard // tm_eff, n_dim // bn),
            in_specs=[
                pl.BlockSpec((tm_eff, k_dim), lambda i, j: (i, 0)),
                pl.BlockSpec((k_dim, bn), lambda i, j: (0, j)),
                pl.BlockSpec((1, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((tm_eff, bn), lambda i, j: (i, j)),
            out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
        )(gated_shard, wo_fp8_full, wo_scale_2d)


# -----------------------------------------------------------------------------
# 3. End-to-End TP8 / TP16 Wrappers for Kernel A3 & Option 3D
# -----------------------------------------------------------------------------
def a3_unfused_ar_bf16(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_bf16: jax.Array,
    tp_axis: str = "tp",
) -> jax.Array:
    """Unfused baseline: sigmoid(gate) * normed_attn_out + BF16 o_proj + full-T psum All-Reduce."""
    gated = (
        jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16) * normed_attn_out
    )
    o_partial = jnp.dot(gated, wo_bf16, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    return lax.psum(o_partial, axis_name=tp_axis)


def a3_unfused_rs_fp8w(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
) -> jax.Array:
    """Unfused XLA FP8-weight o_proj + Reduce-Scatter ([T, 8192] -> [T // TP, 8192])."""
    gated = (
        jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16) * normed_attn_out
    )
    o_partial = (
        jnp.dot(gated, wo_fp8, preferred_element_type=jnp.float32) * wo_scale_2d
    ).astype(jnp.bfloat16)
    return lax.psum_scatter(
        o_partial, axis_name=tp_axis, scatter_dimension=0, tiled=True
    )


def a3_v1_pallas_rs(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    act_dtype: str = "bf16",
    tm: int = 256,
    bk: int | None = None,
    sub_n: int | None = None,
) -> jax.Array:
    """Variant A3-v1/v2: Fused Pallas `pallas_gate_oproj` + Reduce-Scatter."""
    o_partial = pallas_gate_oproj(
        normed_attn_out,
        gate,
        wo_fp8,
        wo_scale_2d,
        act_dtype=act_dtype,
        tm=tm,
        bk=bk,
        sub_n=sub_n,
    )
    return lax.psum_scatter(
        o_partial, axis_name=tp_axis, scatter_dimension=0, tiled=True
    )


def a3_v3_2half_overlap_rs(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    act_dtype: str = "bf16",
    tm: int = 256,
    bk: int | None = None,
    sub_n: int | None = None,
    head_norm_eps: float | None = None,
) -> jax.Array:
    """Variant A3-v3: 2-Half Overlapped `pallas_gate_oproj` + Reduce-Scatter.

    `head_norm_eps` set: the input is the raw attention output, normalized per head in the
    kernel prologue, and each half is read in place through the BlockSpec index map.
    """
    if head_norm_eps is not None:
        tp_deg = 8192 // normed_attn_out.shape[1]
        halves = [
            lax.psum_scatter(
                pallas_gate_oproj(
                    normed_attn_out,
                    gate,
                    wo_fp8,
                    wo_scale_2d,
                    act_dtype=act_dtype,
                    tm=tm,
                    bk=bk,
                    sub_n=sub_n,
                    head_norm_eps=head_norm_eps,
                    row_half=(h, tp_deg),
                ),
                axis_name=tp_axis,
                scatter_dimension=0,
                tiled=True,
            )
            for h in range(2)
        ]
        return jnp.concatenate(halves, axis=0)
    t_total, k_local = normed_attn_out.shape
    t_half = t_total // 2
    tp_deg = 8192 // k_local
    h_sub = (t_total // tp_deg) // 2

    attn_r = normed_attn_out.reshape(tp_deg, 2, h_sub, k_local)
    gate_r = gate.reshape(tp_deg, 2, h_sub, k_local)
    attn0 = attn_r[:, 0, :, :].reshape(t_half, k_local)
    gate0 = gate_r[:, 0, :, :].reshape(t_half, k_local)
    attn1 = attn_r[:, 1, :, :].reshape(t_half, k_local)
    gate1 = gate_r[:, 1, :, :].reshape(t_half, k_local)

    o0_partial = pallas_gate_oproj(
        attn0,
        gate0,
        wo_fp8,
        wo_scale_2d,
        act_dtype=act_dtype,
        tm=tm,
        bk=bk,
        sub_n=sub_n,
    )
    rs0 = lax.psum_scatter(
        o0_partial, axis_name=tp_axis, scatter_dimension=0, tiled=True
    )
    o1_partial = pallas_gate_oproj(
        attn1,
        gate1,
        wo_fp8,
        wo_scale_2d,
        act_dtype=act_dtype,
        tm=tm,
        bk=bk,
        sub_n=sub_n,
    )
    rs1 = lax.psum_scatter(
        o1_partial, axis_name=tp_axis, scatter_dimension=0, tiled=True
    )
    return jnp.concatenate([rs0, rs1], axis=0)


def a3_v4_option3d_a2a_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    use_fp8_act: bool = False,
    tm: int = 512,
    bn: int = 1024,
    bk: int = 2048,
    sub_n: int | None = None,
) -> jax.Array:
    """Variant A3-v4/v5 (Fable's Option 3D): 14 MiB All-to-All on gated attn + local SP Wo Pallas kernel."""
    gated_local = (
        jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16) * normed_attn_out
    )
    gated_token_shard = lax.all_to_all(
        gated_local,
        axis_name=tp_axis,
        split_axis=0,
        concat_axis=1,
        tiled=True,
    )
    if use_fp8_act:
        gf = gated_token_shard.astype(jnp.float32)
        amax = jnp.maximum(jnp.max(jnp.abs(gf), axis=1, keepdims=True), 1e-12)
        row_sc = amax * (1.0 / FP8_MAX)
        gated_fp8 = jnp.clip(gf / row_sc, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
        sc_128 = jnp.broadcast_to(row_sc, (gated_token_shard.shape[0], 128))
        return pallas_option3d_wo_sp(
            gated_fp8,
            wo_fp8_full,
            wo_scale_2d,
            gated_scale_128=sc_128,
            tm=tm,
            bn=bn,
            bk=bk,
            sub_n=sub_n,
        )
    else:
        return pallas_option3d_wo_sp(
            gated_token_shard,
            wo_fp8_full,
            wo_scale_2d,
            tm=tm,
            bn=bn,
            bk=bk,
            sub_n=sub_n,
        )


def a3_v6_option3d_2half_a2a_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    use_fp8_act: bool = False,
    bn: int = 1024,
    bk: int = 2048,
    sub_n: int | None = 256,
) -> jax.Array:
    """Variant A3-v6: 2-Half Overlapped Option 3D (7 MiB x 2 All-to-All overlapped with local Wo_fp8)."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    h_sub = t_shard // 2
    t_half = t_total // 2

    gated_local = (
        jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16) * normed_attn_out
    )
    gated_r = gated_local.reshape(tp_deg, 2, h_sub, k_local)
    g0_in = gated_r[:, 0, :, :].reshape(t_half, k_local)
    g1_in = gated_r[:, 1, :, :].reshape(t_half, k_local)

    s0 = lax.all_to_all(
        g0_in, axis_name=tp_axis, split_axis=0, concat_axis=1, tiled=True
    )
    s1 = lax.all_to_all(
        g1_in, axis_name=tp_axis, split_axis=0, concat_axis=1, tiled=True
    )

    if use_fp8_act:
        gf0 = s0.astype(jnp.float32)
        sc0 = jnp.maximum(jnp.max(jnp.abs(gf0), axis=1, keepdims=True), 1e-12) * (
            1.0 / FP8_MAX
        )
        s0_fp8 = jnp.clip(gf0 / sc0, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
        sc0_128 = jnp.broadcast_to(sc0, (h_sub, 128))
        o0 = pallas_option3d_wo_sp(
            s0_fp8,
            wo_fp8_full,
            wo_scale_2d,
            gated_scale_128=sc0_128,
            tm=h_sub,
            bn=bn,
            bk=bk,
            sub_n=sub_n,
        )

        gf1 = s1.astype(jnp.float32)
        sc1 = jnp.maximum(jnp.max(jnp.abs(gf1), axis=1, keepdims=True), 1e-12) * (
            1.0 / FP8_MAX
        )
        s1_fp8 = jnp.clip(gf1 / sc1, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
        sc1_128 = jnp.broadcast_to(sc1, (h_sub, 128))
        o1 = pallas_option3d_wo_sp(
            s1_fp8,
            wo_fp8_full,
            wo_scale_2d,
            gated_scale_128=sc1_128,
            tm=h_sub,
            bn=bn,
            bk=bk,
            sub_n=sub_n,
        )
    else:
        o0 = pallas_option3d_wo_sp(
            s0, wo_fp8_full, wo_scale_2d, tm=h_sub, bn=bn, bk=bk, sub_n=sub_n
        )
        o1 = pallas_option3d_wo_sp(
            s1, wo_fp8_full, wo_scale_2d, tm=h_sub, bn=bn, bk=bk, sub_n=sub_n
        )
    return jnp.concatenate([o0, o1], axis=0)


# -----------------------------------------------------------------------------
# 4. Option 3D v7/v8: Fused Prologue FP8 Quant + Zero-Transpose 3D All-to-All (7 MiB)
#    + Zero-Copy 3D Block-Scaled FP8xFP8 Pallas Wo Kernel
# -----------------------------------------------------------------------------
def pallas_gate_fp8_quant(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    tm: int = 256,
    vmem_limit_mb: int = 32,
) -> tuple[jax.Array, jax.Array]:
    """Fuses `sigmoid(gate.float()).to(bf16) * normed_attn_out` + per-row FP8 quant in VMEM before All-to-All."""
    t_len, k_local = normed_attn_out.shape
    tm_eff = min(tm, t_len)

    def _quant_kernel(attn_ref, gate_ref, q_out_ref, sc_out_ref):
        g_sig = jax.nn.sigmoid(gate_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
        gf = (g_sig * attn_ref[...]).astype(jnp.float32)
        amax = jnp.maximum(jnp.max(jnp.abs(gf), axis=1, keepdims=True), 1e-12)
        row_sc = amax * (1.0 / FP8_MAX)
        q_out_ref[...] = jnp.clip(gf / row_sc, -FP8_MAX, FP8_MAX).astype(
            jnp.float8_e4m3fn
        )
        sc_out_ref[...] = jnp.broadcast_to(row_sc, (tm_eff, 128))

    return pl.pallas_call(
        _quant_kernel,
        grid=(t_len // tm_eff,),
        in_specs=[
            pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
            pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
            pl.BlockSpec((tm_eff, 128), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((t_len, k_local), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t_len, 128), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(normed_attn_out, gate)


def pallas_option3d_wo_3d_fp8(
    q_3d: jax.Array,
    sc_3d: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tm: int = 512,
    bn: int = 1024,
    sub_n: int | None = 256,
    vmem_limit_mb: int = 60,
) -> jax.Array:
    """Zero-Copy 3D-Indexed Block-Scaled FP8xFP8 Pallas Kernel for Option 3D.

    Consumes `q_3d: fp8_e4m3fn[tp_deg, t_shard, k_local]` and `sc_3d: fp32[tp_deg, t_shard, 128]`
    directly from `lax.all_to_all(..., split_axis=0, concat_axis=0, tiled=True)` without any
    `concat_axis=1` transpose or post-collective HBM pass.
    """
    tp_deg, t_shard, k_local = q_3d.shape
    _, n_dim = wo_fp8_full.shape
    tm_eff = min(tm, t_shard)
    tn_step = sub_n or bn
    w_3d = wo_fp8_full.reshape(tp_deg, k_local, n_dim)

    def _kernel_3d_zc(q_ref, sc_ref, w_ref, w_sc_ref, out_ref):
        if tn_step == bn:
            acc = (
                jnp.dot(
                    q_ref[0, :, :],
                    w_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )
                * sc_ref[0, :, :1]
            )
            for r in range(1, tp_deg):
                acc = acc + (
                    jnp.dot(
                        q_ref[r, :, :],
                        w_ref[r, :, :],
                        preferred_element_type=jnp.float32,
                    )
                    * sc_ref[r, :, :1]
                )
            out_ref[...] = (acc * w_sc_ref[...]).astype(jnp.bfloat16)
        else:
            for n0 in range(0, bn, tn_step):
                tn = min(tn_step, bn - n0)
                acc = (
                    jnp.dot(
                        q_ref[0, :, :],
                        w_ref[0, :, n0 : n0 + tn],
                        preferred_element_type=jnp.float32,
                    )
                    * sc_ref[0, :, :1]
                )
                for r in range(1, tp_deg):
                    acc = acc + (
                        jnp.dot(
                            q_ref[r, :, :],
                            w_ref[r, :, n0 : n0 + tn],
                            preferred_element_type=jnp.float32,
                        )
                        * sc_ref[r, :, :1]
                    )
                out_ref[:, n0 : n0 + tn] = (acc * w_sc_ref[:, n0 : n0 + tn]).astype(
                    jnp.bfloat16
                )

    return pl.pallas_call(
        _kernel_3d_zc,
        grid=(n_dim // bn, t_shard // tm_eff),
        in_specs=[
            pl.BlockSpec((tp_deg, tm_eff, k_local), lambda j, i: (0, i, 0)),
            pl.BlockSpec((tp_deg, tm_eff, 128), lambda j, i: (0, i, 0)),
            pl.BlockSpec((tp_deg, k_local, bn), lambda j, i: (0, 0, j)),
            pl.BlockSpec((1, bn), lambda j, i: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm_eff, bn), lambda j, i: (i, j)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "parallel"),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(q_3d, sc_3d, w_3d, wo_scale_2d)


def a3_v7_option3d_prequant_fp8_a2a_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    tm: int = 512,
    bn: int = 1024,
    sub_n: int | None = 256,
) -> jax.Array:
    """Variant A3-v7: Fused Prologue FP8 Quant + 7 MiB Zero-Transpose 3D All-to-All + 3D Block-Scaled Pallas Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg

    q_local, sc_local = pallas_gate_fp8_quant(normed_attn_out, gate, tm=256)
    q_3d_in = q_local.reshape(tp_deg, t_shard, k_local)
    sc_3d_in = sc_local.reshape(tp_deg, t_shard, 128)

    q_3d = lax.all_to_all(
        q_3d_in, axis_name=tp_axis, split_axis=0, concat_axis=0, tiled=True
    )
    sc_3d = lax.all_to_all(
        sc_3d_in, axis_name=tp_axis, split_axis=0, concat_axis=0, tiled=True
    )
    return pallas_option3d_wo_3d_fp8(
        q_3d, sc_3d, wo_fp8_full, wo_scale_2d, tm=tm, bn=bn, sub_n=sub_n
    )


def a3_v8_option3d_2half_prequant_fp8_a2a_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    sub_n: int | None = 256,
) -> jax.Array:
    """Variant A3-v8: 2-Half Overlapped 3.5 MiB x 2 Zero-Transpose FP8 All-to-All + 3D Block-Scaled Pallas Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    h_sub = t_shard // 2
    t_half = t_total // 2

    attn_r = normed_attn_out.reshape(tp_deg, 2, h_sub, k_local)
    gate_r = gate.reshape(tp_deg, 2, h_sub, k_local)
    a0 = attn_r[:, 0, :, :].reshape(t_half, k_local)
    g0 = gate_r[:, 0, :, :].reshape(t_half, k_local)
    a1 = attn_r[:, 1, :, :].reshape(t_half, k_local)
    g1 = gate_r[:, 1, :, :].reshape(t_half, k_local)

    q0, sc0 = pallas_gate_fp8_quant(a0, g0, tm=min(256, t_half))
    q0_3d = lax.all_to_all(
        q0.reshape(tp_deg, h_sub, k_local),
        axis_name=tp_axis,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )
    sc0_3d = lax.all_to_all(
        sc0.reshape(tp_deg, h_sub, 128),
        axis_name=tp_axis,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )

    a1_b, g1_b, q0_3d, sc0_3d = lax.optimization_barrier((a1, g1, q0_3d, sc0_3d))
    q1, sc1 = pallas_gate_fp8_quant(a1_b, g1_b, tm=min(256, t_half))
    q1_3d = lax.all_to_all(
        q1.reshape(tp_deg, h_sub, k_local),
        axis_name=tp_axis,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )
    sc1_3d = lax.all_to_all(
        sc1.reshape(tp_deg, h_sub, 128),
        axis_name=tp_axis,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )

    o0 = pallas_option3d_wo_3d_fp8(
        q0_3d, sc0_3d, wo_fp8_full, wo_scale_2d, tm=h_sub, bn=bn, sub_n=sub_n
    )
    q1_3d, sc1_3d, o0 = lax.optimization_barrier((q1_3d, sc1_3d, o0))
    o1 = pallas_option3d_wo_3d_fp8(
        q1_3d, sc1_3d, wo_fp8_full, wo_scale_2d, tm=h_sub, bn=bn, sub_n=sub_n
    )
    return jnp.concatenate([o0, o1], axis=0)


# -----------------------------------------------------------------------------
# 5. Option 3D v9/v10: Single Packed FP8 All-to-All (`q + scale` in 1 collective!)
#    + N-Outer Weight-Resident 3D Pallas Wo Kernel
# -----------------------------------------------------------------------------
def pallas_gate_fp8_quant_packed(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    tm: int = 256,
    vmem_limit_mb: int = 32,
) -> jax.Array:
    """Fuses `sigmoid(gate) * attn` + per-row FP8 quant + Hi+Lo E4M3 scale packing into `fp8_e4m3fn[T, k_local + 128]`.

    Decomposes the per-row scale `s * 512.0` into a high E4M3 byte `s_hi` (lanes 0..63) and a
    residual E4M3 byte `s_lo = ((s * 512.0 - s_hi) * 16.0)` (lanes 64..127), achieving 8-bit
    mantissa precision (`0.185%` max rel error, 2x more accurate than BF16!) while still packing
    activations and scales into a single `fp8_e4m3fn` tensor for 1-collective All-to-All.
    """
    t_len, k_local = normed_attn_out.shape
    tm_eff = min(tm, t_len)

    def _quant_packed_kernel(attn_ref, gate_ref, packed_out_ref):
        g_sig = jax.nn.sigmoid(gate_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
        gf = (g_sig * attn_ref[...]).astype(jnp.float32)
        amax = jnp.maximum(jnp.max(jnp.abs(gf), axis=1, keepdims=True), 1e-12)
        s_scaled = jnp.clip(amax * (512.0 / 416.0), 0.03125, FP8_MAX)
        s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
        s_hi_f32 = s_hi_f8.astype(jnp.float32)
        s_lo_f8 = jnp.clip((s_scaled - s_hi_f32) * 16.0, -FP8_MAX, FP8_MAX).astype(
            jnp.float8_e4m3fn
        )
        sc_exact = (s_hi_f32 + s_lo_f8.astype(jnp.float32) * (1.0 / 16.0)) * (
            1.0 / 512.0
        )
        q_f8 = jnp.clip(gf / sc_exact, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
        packed_out_ref[:, :k_local] = q_f8
        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        s_pack = jnp.where(
            lane < 64,
            jnp.broadcast_to(s_hi_f8, (tm_eff, 128)),
            jnp.broadcast_to(s_lo_f8, (tm_eff, 128)),
        )
        packed_out_ref[:, k_local : k_local + 128] = s_pack

    return pl.pallas_call(
        _quant_packed_kernel,
        grid=(t_len // tm_eff,),
        in_specs=[
            pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
            pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tm_eff, k_local + 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((t_len, k_local + 128), jnp.float8_e4m3fn),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(normed_attn_out, gate)


def pallas_option3d_wo_3d_packed_fp8(
    packed_3d: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tm: int = 1024,
    bn: int = 1024,
    sub_n: int | None = None,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """N-Outer Weight-Resident 3D Pallas Wo Kernel consuming single-A2A `packed_3d: fp8[tp_deg, t_shard, k_local + 128]`."""
    tp_deg, t_shard, k_packed = packed_3d.shape
    k_local = k_packed - 128
    _, n_dim = wo_fp8_full.shape
    tm_eff = min(tm, t_shard)
    tn_step = sub_n or bn
    w_3d = wo_fp8_full.reshape(tp_deg, k_local, n_dim)

    def _reconstruct_sc(packed_ref, r: int) -> jax.Array:
        s_hi = packed_ref[r, :, k_local : k_local + 1].astype(jnp.float32)
        s_lo = packed_ref[r, :, k_local + 64 : k_local + 65].astype(jnp.float32)
        return s_hi + s_lo * (1.0 / 16.0)

    def _kernel_3d_packed(packed_ref, w_ref, w_sc_ref, out_ref):
        if tn_step == bn:
            sc0 = _reconstruct_sc(packed_ref, 0)
            acc = (
                jnp.dot(
                    packed_ref[0, :, :k_local],
                    w_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )
                * sc0
            )
            for r in range(1, tp_deg):
                scr = _reconstruct_sc(packed_ref, r)
                acc = acc + (
                    jnp.dot(
                        packed_ref[r, :, :k_local],
                        w_ref[r, :, :],
                        preferred_element_type=jnp.float32,
                    )
                    * scr
                )
            out_ref[...] = (acc * (w_sc_ref[...] * (1.0 / 512.0))).astype(jnp.bfloat16)
        else:
            for n0 in range(0, bn, tn_step):
                tn = min(tn_step, bn - n0)
                sc0 = _reconstruct_sc(packed_ref, 0)
                acc = (
                    jnp.dot(
                        packed_ref[0, :, :k_local],
                        w_ref[0, :, n0 : n0 + tn],
                        preferred_element_type=jnp.float32,
                    )
                    * sc0
                )
                for r in range(1, tp_deg):
                    scr = _reconstruct_sc(packed_ref, r)
                    acc = acc + (
                        jnp.dot(
                            packed_ref[r, :, :k_local],
                            w_ref[r, :, n0 : n0 + tn],
                            preferred_element_type=jnp.float32,
                        )
                        * scr
                    )
                out_ref[:, n0 : n0 + tn] = (
                    acc * (w_sc_ref[:, n0 : n0 + tn] * (1.0 / 512.0))
                ).astype(jnp.bfloat16)

    return pl.pallas_call(
        _kernel_3d_packed,
        grid=(n_dim // bn, t_shard // tm_eff),
        in_specs=[
            pl.BlockSpec((tp_deg, tm_eff, k_packed), lambda j, i: (0, i, 0)),
            pl.BlockSpec((tp_deg, k_local, bn), lambda j, i: (0, 0, j)),
            pl.BlockSpec((1, bn), lambda j, i: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm_eff, bn), lambda j, i: (i, j)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "parallel"),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(packed_3d, w_3d, wo_scale_2d)


def a3_v9_option3d_single_a2a_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    tm: int = 1024,
    bn: int = 1024,
    sub_n: int | None = None,
) -> jax.Array:
    """Variant A3-v9: Single Packed FP8 All-to-All (`q + HiLo scale` in 1 collective) + N-Outer 3D Pallas Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg

    packed_local = pallas_gate_fp8_quant_packed(normed_attn_out, gate, tm=256)
    packed_3d = lax.all_to_all(
        packed_local.reshape(tp_deg, t_shard, k_local + 128),
        axis_name=tp_axis,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )
    return pallas_option3d_wo_3d_packed_fp8(
        packed_3d, wo_fp8_full, wo_scale_2d, tm=tm, bn=bn, sub_n=sub_n
    )


# -----------------------------------------------------------------------------
# 6. Wave 5: Pallas Async Remote-Copy All-to-All (`make_async_remote_copy`),
#    Static-Scale FP8 Option 3D, and Fused In-Kernel Async A2A + Local-First Wo
# -----------------------------------------------------------------------------
STATIC_A3_ACT_MULT = 32.0
STATIC_A3_ACT_SCALE = 1.0 / STATIC_A3_ACT_MULT


def pallas_async_all_to_all_3d(
    src_3d: jax.Array,
    tp_axis: str = "tp",
) -> jax.Array:
    """Standalone Pallas `pltpu.make_async_remote_copy` All-to-All on `[tp_deg, t_shard, cols]`.

    Launches all `tp_deg - 1` remote peer DMAs simultaneously plus 1 local copy without XLA
    `all_to_all` serialization overhead.
    """
    tp_deg, t_shard, cols = src_3d.shape

    def _a2a_kernel(src_ref, dst_ref, sems):
        me = lax.axis_index(tp_axis)
        if tp_deg > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, tp_deg):
                pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
            pl.semaphore_wait(barrier, tp_deg - 1)
        recv, sent = sems.at[0], sems.at[1]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(me, 1)],
            dst_ref.at[pl.ds(me, 1)],
            recv,
        ).start()
        for d in range(1, tp_deg):
            peer = lax.rem(me + d, tp_deg)
            pltpu.make_async_remote_copy(
                src_ref=src_ref.at[pl.ds(peer, 1)],
                dst_ref=dst_ref.at[pl.ds(me, 1)],
                send_sem=sent,
                recv_sem=recv,
                device_id=(peer,),
            ).start()
        tmpl_r = pltpu.make_async_copy(
            dst_ref.at[pl.ds(0, 1)], dst_ref.at[pl.ds(0, 1)], recv
        )
        for _ in range(tp_deg):
            tmpl_r.wait()
        tmpl_s = pltpu.make_async_copy(
            src_ref.at[pl.ds(0, 1)], src_ref.at[pl.ds(0, 1)], sent
        )
        for _ in range(tp_deg - 1):
            tmpl_s.wait()

    return pl.pallas_call(
        _a2a_kernel,
        grid=(1,),
        in_specs=[pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct((tp_deg, t_shard, cols), src_3d.dtype),
        scratch_shapes=[pltpu.SemaphoreType.DMA((2,))],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=1 if tp_deg > 1 else None,
        ),
    )(src_3d)


def a3_v10_option3d_async_a2a_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    tm: int = 1024,
    bn: int = 1024,
    sub_n: int | None = None,
) -> jax.Array:
    """Variant A3-v10: Packed FP8 Prologue + Pallas `make_async_remote_copy` All-to-All + 3D Pallas Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg

    packed_local = pallas_gate_fp8_quant_packed(normed_attn_out, gate, tm=256)
    packed_3d = pallas_async_all_to_all_3d(
        packed_local.reshape(tp_deg, t_shard, k_local + 128),
        tp_axis=tp_axis,
    )
    return pallas_option3d_wo_3d_packed_fp8(
        packed_3d, wo_fp8_full, wo_scale_2d, tm=tm, bn=bn, sub_n=sub_n
    )


def pallas_gate_static_fp8_quant(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    tm: int = 512,
    vmem_limit_mb: int = 32,
    head_norm_eps: float | None = None,
    rpa_packed_out: bool = False,
    valid_tokens_1d: jax.Array | None = None,
) -> jax.Array:
    """One-pass static-scale FP8 prologue (`sigmoid(gate) * normed_attn_out * 32.0 -> fp8_e4m3fn[T, k_local]`).

    `head_norm_eps` set: the input is the raw attention output; RMS-normalize each head first.
    `rpa_packed_out`: `normed_attn_out` is RPA v3's KV-packed output (`bf16[1, T, Hq/2, 2, 128]`
    or `bf16[T, Hq, 128]`). Each head pair `(2s, 2s+1)` is diagonally selected from its valid
    64-lane KV-group half (`lanes 0:64` for `s < Hq/4`, `lanes 64:128` for `s >= Hq/4`) in VREGs
    with zero intermediate HBM relayout or VMEM scratch round-trip.

    Because `normed_attn_out` is per-head RMS-normalized (`|x| <= 8`) and `sigmoid(gate) in (0, 1)`,
    multiplying by 32.0 centers values in `fp8_e4m3fn`'s normal range `[2^-6, 448]` without any
    per-row `amax` reduction or scale columns (cutting A2A payload by 12.5% and prologue time by >2x!).
    """
    t_len, k_local = gate.shape
    tm_eff = min(tm, t_len)
    has_skip = valid_tokens_1d is not None
    from sglang_tpu.envs import envs as _tpu_envs

    zs_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A3_ZERO_SPILL_PROLOGUE", None)
    use_zero_spill = bool(zs_env is not None and zs_env.get()) or (t_len % tm_eff != 0)
    if use_zero_spill:
        tm_eff = min(tm_eff, 128)
    if t_len % tm_eff != 0:
        for cand_tm in (512, 256, 128, 64):
            if cand_tm <= t_len and t_len % cand_tm == 0:
                tm_eff = cand_tm
                break

    if rpa_packed_out and normed_attn_out.size != t_len * k_local:
        num_q_slices = k_local // 128
        attn_5d = normed_attn_out.reshape(1, t_len, num_q_slices, 2, 128)

        def _rpa_packed_quant_tile(attn_ref, gate_ref, q_out_ref):
            lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo
            attn_words = (
                attn_ref.bitcast(jnp.uint32).at[0].reshape(tm_eff * num_q_slices, 128)
            )
            for s in range(num_q_slices):
                c0, c1 = s * 128, (s + 1) * 128
                w = attn_words[pl.ds(s, tm_eff, stride=num_q_slices)]
                even_raw = pltpu.bitcast(lax.shift_left(w, jnp.uint32(16)), jnp.float32)
                odd_raw = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), jnp.float32)
                if s < num_q_slices // 2:
                    pair_f32 = jnp.where(
                        is_lo, even_raw, pltpu.roll(odd_raw, shift=64, axis=1)
                    )
                else:
                    pair_f32 = jnp.where(
                        is_lo, pltpu.roll(even_raw, shift=64, axis=1), odd_raw
                    )
                pair_f32 = pair_f32 + 0.0
                if head_norm_eps is not None:
                    pair_bf16 = _dual_head_rmsnorm_128(
                        pair_f32, is_lo, mask_lo, mask_hi, 1.0, eps=head_norm_eps
                    ).astype(jnp.bfloat16)
                else:
                    pair_bf16 = pair_f32.astype(jnp.bfloat16)
                g_sig = jax.nn.sigmoid(gate_ref[:, c0:c1].astype(jnp.float32)).astype(
                    jnp.bfloat16
                )
                gf = (g_sig * pair_bf16).astype(jnp.float32) * STATIC_A3_ACT_MULT
                q_out_ref[:, c0:c1] = jnp.clip(gf, -FP8_MAX, FP8_MAX).astype(
                    jnp.float8_e4m3fn
                )

        if not has_skip:
            return pl.pallas_call(
                _rpa_packed_quant_tile,
                grid=(t_len // tm_eff,),
                in_specs=[
                    pl.BlockSpec(
                        (1, tm_eff, num_q_slices, 2, 128), lambda i: (0, i, 0, 0, 0)
                    ),
                    pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
                out_shape=jax.ShapeDtypeStruct((t_len, k_local), jnp.float8_e4m3fn),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel",),
                    vmem_limit_bytes=vmem_limit_mb * (1 << 20),
                ),
                name="pallas_gate_static_fp8_quant_rpa_out",
            )(attn_5d, gate)

        def _rpa_packed_quant_skip(vtok_ref, attn_ref, gate_ref, q_out_ref):
            row0 = pl.program_id(0) * tm_eff

            @pl.when(row0 < vtok_ref[0])
            def _():
                _rpa_packed_quant_tile(attn_ref, gate_ref, q_out_ref)

            @pl.when(row0 >= vtok_ref[0])
            def _():
                q_out_ref[:, :] = jnp.zeros((tm_eff, k_local), jnp.float8_e4m3fn)

        return pl.pallas_call(
            _rpa_packed_quant_skip,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(t_len // tm_eff,),
                in_specs=[
                    pl.BlockSpec(
                        (1, tm_eff, num_q_slices, 2, 128),
                        lambda i, _vt: (0, i, 0, 0, 0),
                    ),
                    pl.BlockSpec((tm_eff, k_local), lambda i, _vt: (i, 0)),
                ],
                out_specs=pl.BlockSpec((tm_eff, k_local), lambda i, _vt: (i, 0)),
            ),
            out_shape=jax.ShapeDtypeStruct((t_len, k_local), jnp.float8_e4m3fn),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
            name="pallas_gate_static_fp8_quant_rpa_out_skip",
        )(valid_tokens_1d.reshape(1).astype(jnp.int32), attn_5d, gate)

    def _static_quant_tile_sub128(attn_ref, gate_ref, q_out_ref):
        sub_r = min(128, tm_eff)
        lane = lax.broadcasted_iota(jnp.int32, (sub_r, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        for r0 in range(0, tm_eff, sub_r):
            r1 = r0 + sub_r
            for c0 in range(0, k_local, 128):
                c1 = c0 + 128
                a_f32 = attn_ref[r0:r1, c0:c1].astype(jnp.float32)
                if head_norm_eps is not None:
                    a_bf16 = _dual_head_rmsnorm_128(
                        a_f32, is_lo, mask_lo, mask_hi, 1.0, eps=head_norm_eps
                    ).astype(jnp.bfloat16)
                else:
                    a_bf16 = a_f32.astype(jnp.bfloat16)
                g_sig = jax.nn.sigmoid(
                    gate_ref[r0:r1, c0:c1].astype(jnp.float32)
                ).astype(jnp.bfloat16)
                gf = (g_sig * a_bf16).astype(jnp.float32) * STATIC_A3_ACT_MULT
                q_out_ref[r0:r1, c0:c1] = jnp.clip(gf, -FP8_MAX, FP8_MAX).astype(
                    jnp.float8_e4m3fn
                )

    def _static_quant_tile(attn_ref, gate_ref, q_out_ref, *scratch):
        if use_zero_spill:
            _static_quant_tile_sub128(attn_ref, gate_ref, q_out_ref)
            return
        g_sig = jax.nn.sigmoid(gate_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
        if head_norm_eps is not None:
            _store_head_rmsnorm(attn_ref, scratch[0], head_norm_eps)
            attn_ref = scratch[0]
        gf = (g_sig * attn_ref[...]).astype(jnp.float32) * STATIC_A3_ACT_MULT
        q_out_ref[...] = jnp.clip(gf, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)

    scratches = (
        [pltpu.VMEM((tm_eff, k_local), jnp.bfloat16)]
        if (head_norm_eps is not None and not use_zero_spill)
        else []
    )
    if not has_skip:
        return pl.pallas_call(
            _static_quant_tile,
            grid=(t_len // tm_eff,),
            in_specs=[
                pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
                pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((tm_eff, k_local), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct((t_len, k_local), jnp.float8_e4m3fn),
            scratch_shapes=scratches,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
        )(normed_attn_out, gate)

    def _static_quant_skip(vtok_ref, attn_ref, gate_ref, q_out_ref, *scratch):
        row0 = pl.program_id(0) * tm_eff

        @pl.when(row0 < vtok_ref[0])
        def _():
            _static_quant_tile(attn_ref, gate_ref, q_out_ref, *scratch)

        @pl.when(row0 >= vtok_ref[0])
        def _():
            q_out_ref[:, :] = jnp.zeros((tm_eff, k_local), jnp.float8_e4m3fn)

    return pl.pallas_call(
        _static_quant_skip,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(t_len // tm_eff,),
            in_specs=[
                pl.BlockSpec((tm_eff, k_local), lambda i, _vt: (i, 0)),
                pl.BlockSpec((tm_eff, k_local), lambda i, _vt: (i, 0)),
            ],
            out_specs=pl.BlockSpec((tm_eff, k_local), lambda i, _vt: (i, 0)),
            scratch_shapes=scratches,
        ),
        out_shape=jax.ShapeDtypeStruct((t_len, k_local), jnp.float8_e4m3fn),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name="pallas_gate_static_fp8_quant_skip",
    )(valid_tokens_1d.reshape(1).astype(jnp.int32), normed_attn_out, gate)


def pallas_option3d_wo_3d_static_fp8(
    q_3d: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tm: int = 1024,
    bn: int = 1024,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """Pure-MXU-Accumulation 3D Pallas Wo Kernel for Static-Scale FP8 Option 3D.

    Eliminates the 8 per-source-rank VPU rescales (`* scr`) inside the K-loop: all `tp_deg`
    K-blocks accumulate directly in FP32 on the MXU and apply `w_sc_ref[...] * (1.0 / 32.0)` once!
    """
    tp_deg, t_shard, k_local = q_3d.shape
    _, n_dim = wo_fp8_full.shape
    tm_eff = min(tm, t_shard)
    w_3d = wo_fp8_full.reshape(tp_deg, k_local, n_dim)

    def _kernel_3d_static(q_ref, w_ref, w_sc_ref, out_ref):
        acc = jnp.dot(
            q_ref[0, :, :],
            w_ref[0, :, :],
            preferred_element_type=jnp.float32,
        )
        for r in range(1, tp_deg):
            acc = acc + jnp.dot(
                q_ref[r, :, :],
                w_ref[r, :, :],
                preferred_element_type=jnp.float32,
            )
        out_ref[...] = (acc * (w_sc_ref[...] * STATIC_A3_ACT_SCALE)).astype(
            jnp.bfloat16
        )

    return pl.pallas_call(
        _kernel_3d_static,
        grid=(n_dim // bn, t_shard // tm_eff),
        in_specs=[
            pl.BlockSpec((tp_deg, tm_eff, k_local), lambda j, i: (0, i, 0)),
            pl.BlockSpec((tp_deg, k_local, bn), lambda j, i: (0, 0, j)),
            pl.BlockSpec((1, bn), lambda j, i: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm_eff, bn), lambda j, i: (i, j)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "parallel"),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(q_3d, w_3d, wo_scale_2d)


def a3_v11_option3d_static_fp8_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    use_pallas_a2a: bool = True,
    tm: int = 1024,
    bn: int = 1024,
) -> jax.Array:
    """Variant A3-v11: Static-Scale FP8 Prologue (7 MiB remote, no amax/scale cols) + A2A + Pure-MXU 3D Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg

    q_local_3d = pallas_gate_static_fp8_quant(
        normed_attn_out, gate, tm=min(512, t_total)
    ).reshape(tp_deg, t_shard, k_local)

    if use_pallas_a2a:
        q_3d = pallas_async_all_to_all_3d(q_local_3d, tp_axis=tp_axis)
    else:
        q_3d = lax.all_to_all(
            q_local_3d, axis_name=tp_axis, split_axis=0, concat_axis=0, tiled=True
        )
    return pallas_option3d_wo_3d_static_fp8(
        q_3d, wo_fp8_full, wo_scale_2d, tm=tm, bn=bn
    )


def pallas_option3d_fused_async_a2a_wo_3d_static_fp8(
    q_local_3d: jax.Array,
    q_own_2d: jax.Array,
    w_own_2d: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """Fuses `make_async_remote_copy` All-to-All directly into `pallas_option3d_wo_3d_static_fp8`.

    At `j == 0`:
      1. Starts `tp_deg - 1` remote ICI DMAs from `q_local_3d[peer]` into `recv_vmem[me]` on `peer`
         (writing directly into receiver VMEM, zero HBM round-trip!).
      2. While the `tp_deg - 1` remote K-blocks are in flight over ICI, immediately computes
         rank `me`'s own K-block (`q_own_2d @ w_own_2d[:, :bn]`) on the MXU!
      3. Waits on `rsem_peers` (`tp_deg - 1` remote blocks), accumulates the remote K-blocks for
         `j == 0`, and populates `recv_vmem[me] = q_own_2d` for subsequent N-tiles.
    At `j = 1 .. (n_dim // bn) - 1`:
      - Reuses `recv_vmem` (`[tp_deg, t_shard, k_local]`) directly from VMEM with zero HBM reads
        and zero collective waits!
    At `j == (n_dim // bn) - 1`:
      - Waits on `ssem` (`tp_deg - 1` outgoing sends).
    """
    tp_deg, t_shard, k_local = q_local_3d.shape
    _, n_dim = wo_fp8_full.shape
    num_n = n_dim // bn
    w_3d = wo_fp8_full.reshape(tp_deg, k_local, n_dim)
    tm_step = min(1024, t_shard)

    def _fused_a2a_wo_kernel(
        src_hbm_ref,
        q_own_ref,
        w_own_ref,
        w_ref,
        w_sc_ref,
        out_ref,
        recv_vmem,
        zero_vmem,
        acc_vmem,
        sems,
    ):
        j = pl.program_id(0)
        me = lax.axis_index(tp_axis)
        rsem_peers, ssem, local_sem = sems.at[0], sems.at[1], sems.at[2]

        @pl.when(j == 0)
        def _():
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
                pl.semaphore_wait(barrier, tp_deg - 1)
            # Zero out recv_vmem[me] before remote waits so static r-loop doesn't double-count me
            zero_vmem[...] = jnp.zeros((1, t_shard, k_local), dtype=jnp.float8_e4m3fn)
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(me, 1)],
                local_sem,
            ).start()
            for d in range(1, tp_deg):
                peer = lax.rem(me + d, tp_deg)
                pltpu.make_async_remote_copy(
                    src_ref=src_hbm_ref.at[pl.ds(peer, 1)],
                    dst_ref=recv_vmem.at[pl.ds(me, 1)],
                    send_sem=ssem,
                    recv_sem=rsem_peers,
                    device_id=(peer,),
                ).start()
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(0, 1)],
                local_sem,
            ).wait()

            # Compute rank `me`'s own K-block on the MXU and store to `acc_vmem` while
            # the (tp_deg - 1) remote DMAs are in flight over ICI!
            for m0 in range(0, t_shard, tm_step):
                acc_vmem[m0 : m0 + tm_step, :] = jnp.dot(
                    q_own_ref[m0 : m0 + tm_step, :],
                    w_own_ref[:, :],
                    preferred_element_type=jnp.float32,
                )

            # Wait for the (tp_deg - 1) remote K-blocks to land in recv_vmem
            tmpl_r = pltpu.make_async_copy(
                recv_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(0, 1)],
                rsem_peers,
            )
            for _ in range(tp_deg - 1):
                tmpl_r.wait()

            # Accumulate the remote K-blocks onto `acc_vmem` (recv_vmem[me] is 0!)
            for m0 in range(0, t_shard, tm_step):
                acc = acc_vmem[m0 : m0 + tm_step, :]
                for r in range(tp_deg):
                    acc = acc + jnp.dot(
                        recv_vmem[r, m0 : m0 + tm_step, :],
                        w_ref[r, :, :],
                        preferred_element_type=jnp.float32,
                    )
                out_ref[m0 : m0 + tm_step, :] = (
                    acc * (w_sc_ref[...] * STATIC_A3_ACT_SCALE)
                ).astype(jnp.bfloat16)

            # Populate recv_vmem[me] = q_own_ref so j = 1 .. num_n - 1 can run a single uniform loop
            zero_vmem[0, :, :] = q_own_ref[...]
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(me, 1)],
                local_sem,
            ).start()
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(0, 1)],
                local_sem,
            ).wait()

        @pl.when(j > 0)
        def _():
            for m0 in range(0, t_shard, tm_step):
                acc = jnp.dot(
                    recv_vmem[0, m0 : m0 + tm_step, :],
                    w_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )
                for r in range(1, tp_deg):
                    acc = acc + jnp.dot(
                        recv_vmem[r, m0 : m0 + tm_step, :],
                        w_ref[r, :, :],
                        preferred_element_type=jnp.float32,
                    )
                out_ref[m0 : m0 + tm_step, :] = (
                    acc * (w_sc_ref[...] * STATIC_A3_ACT_SCALE)
                ).astype(jnp.bfloat16)

        @pl.when(j == num_n - 1)
        def _():
            tmpl_s = pltpu.make_async_copy(
                src_hbm_ref.at[pl.ds(0, 1)],
                src_hbm_ref.at[pl.ds(0, 1)],
                ssem,
            )
            for _ in range(tp_deg - 1):
                tmpl_s.wait()

    w_own_first = w_own_2d[:, :bn]
    return pl.pallas_call(
        _fused_a2a_wo_kernel,
        grid=(num_n,),
        in_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec((t_shard, k_local), lambda j: (0, 0)),
            pl.BlockSpec((k_local, bn), lambda j: (0, 0)),
            pl.BlockSpec((tp_deg, k_local, bn), lambda j: (0, 0, j)),
            pl.BlockSpec((1, bn), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((t_shard, bn), lambda j: (0, j)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((tp_deg, t_shard, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((1, t_shard, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((t_shard, bn), jnp.float32),
            pltpu.SemaphoreType.DMA((3,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=2 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(q_local_3d, q_own_2d, w_own_first, w_3d, wo_scale_2d)


def a3_v12_option3d_fused_async_a2a_wo_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_own: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
) -> jax.Array:
    """Variant A3-v12: Static-Scale FP8 Prologue + Fused In-Kernel `make_async_remote_copy` A2A + Local-First 3D Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    me = lax.axis_index(tp_axis)
    bn_eff = min(bn, 512) if t_shard >= 2048 else bn

    q_local_3d = pallas_gate_static_fp8_quant(
        normed_attn_out, gate, tm=min(512, t_total)
    ).reshape(tp_deg, t_shard, k_local)
    q_own_2d = lax.dynamic_index_in_dim(q_local_3d, me, axis=0, keepdims=False)

    return pallas_option3d_fused_async_a2a_wo_3d_static_fp8(
        q_local_3d,
        q_own_2d,
        wo_fp8_own,
        wo_fp8_full,
        wo_scale_2d,
        tp_axis=tp_axis,
        bn=bn_eff,
    )


def pallas_option3d_fused_async_a2a_wo_3d_hilo_fp8(
    packed_local_3d: jax.Array,
    packed_own_2d: jax.Array,
    w_own_2d: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """Fuses `make_async_remote_copy` All-to-All directly into `Hi+Lo`-Packed Dynamic-Scale 3D Wo.

    Combines the 10-bit `Hi+Lo` dynamic row scale accuracy (`cos_sim >= 0.99992`) of `a3_v9`
    with in-kernel VMEM-direct `make_async_remote_copy` and immediate local K-block overlap.
    """
    tp_deg, t_shard, k_packed = packed_local_3d.shape
    k_local = k_packed - 128
    _, n_dim = wo_fp8_full.shape
    num_n = n_dim // bn
    w_3d = wo_fp8_full.reshape(tp_deg, k_local, n_dim)
    tm_step = min(1024, t_shard)

    def _fused_a2a_wo_hilo_kernel(
        src_hbm_ref,
        q_own_ref,
        w_own_ref,
        w_ref,
        w_sc_ref,
        out_ref,
        recv_vmem,
        zero_vmem,
        acc_vmem,
        sems,
    ):
        j = pl.program_id(0)
        me = lax.axis_index(tp_axis)
        rsem_peers, ssem, local_sem = sems.at[0], sems.at[1], sems.at[2]

        @pl.when(j == 0)
        def _():
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
                pl.semaphore_wait(barrier, tp_deg - 1)
            zero_vmem[...] = jnp.zeros((1, t_shard, k_packed), dtype=jnp.float8_e4m3fn)
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(me, 1)],
                local_sem,
            ).start()
            for d in range(1, tp_deg):
                peer = lax.rem(me + d, tp_deg)
                pltpu.make_async_remote_copy(
                    src_ref=src_hbm_ref.at[pl.ds(peer, 1)],
                    dst_ref=recv_vmem.at[pl.ds(me, 1)],
                    send_sem=ssem,
                    recv_sem=rsem_peers,
                    device_id=(peer,),
                ).start()
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(0, 1)],
                local_sem,
            ).wait()

            # Compute rank `me`'s own K-block while remote DMAs fly over ICI
            for m0 in range(0, t_shard, tm_step):
                s_hi_own = q_own_ref[m0 : m0 + tm_step, k_local : k_local + 1].astype(
                    jnp.float32
                ) * (1.0 / 512.0)
                s_lo_own = q_own_ref[
                    m0 : m0 + tm_step, k_local + 64 : k_local + 65
                ].astype(jnp.float32) * (1.0 / 32768.0)
                sc_own = s_hi_own + s_lo_own
                acc_vmem[m0 : m0 + tm_step, :] = (
                    jnp.dot(
                        q_own_ref[m0 : m0 + tm_step, :k_local],
                        w_own_ref[:, :],
                        preferred_element_type=jnp.float32,
                    )
                    * sc_own
                )

            tmpl_r = pltpu.make_async_copy(
                recv_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(0, 1)],
                rsem_peers,
            )
            for _ in range(tp_deg - 1):
                tmpl_r.wait()

            for m0 in range(0, t_shard, tm_step):
                acc = acc_vmem[m0 : m0 + tm_step, :]
                for r in range(tp_deg):
                    s_hi_r = recv_vmem[
                        r, m0 : m0 + tm_step, k_local : k_local + 1
                    ].astype(jnp.float32) * (1.0 / 512.0)
                    s_lo_r = recv_vmem[
                        r, m0 : m0 + tm_step, k_local + 64 : k_local + 65
                    ].astype(jnp.float32) * (1.0 / 32768.0)
                    scr = s_hi_r + s_lo_r
                    acc = acc + (
                        jnp.dot(
                            recv_vmem[r, m0 : m0 + tm_step, :k_local],
                            w_ref[r, :, :],
                            preferred_element_type=jnp.float32,
                        )
                        * scr
                    )
                out_ref[m0 : m0 + tm_step, :] = (acc * w_sc_ref[...]).astype(
                    jnp.bfloat16
                )

            zero_vmem[0, :, :] = q_own_ref[...]
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(me, 1)],
                local_sem,
            ).start()
            pltpu.make_async_copy(
                zero_vmem.at[pl.ds(0, 1)],
                recv_vmem.at[pl.ds(0, 1)],
                local_sem,
            ).wait()

        @pl.when(j > 0)
        def _():
            for m0 in range(0, t_shard, tm_step):
                s_hi_0 = recv_vmem[0, m0 : m0 + tm_step, k_local : k_local + 1].astype(
                    jnp.float32
                ) * (1.0 / 512.0)
                s_lo_0 = recv_vmem[
                    0, m0 : m0 + tm_step, k_local + 64 : k_local + 65
                ].astype(jnp.float32) * (1.0 / 32768.0)
                sc0 = s_hi_0 + s_lo_0
                acc = (
                    jnp.dot(
                        recv_vmem[0, m0 : m0 + tm_step, :k_local],
                        w_ref[0, :, :],
                        preferred_element_type=jnp.float32,
                    )
                    * sc0
                )
                for r in range(1, tp_deg):
                    s_hi_r = recv_vmem[
                        r, m0 : m0 + tm_step, k_local : k_local + 1
                    ].astype(jnp.float32) * (1.0 / 512.0)
                    s_lo_r = recv_vmem[
                        r, m0 : m0 + tm_step, k_local + 64 : k_local + 65
                    ].astype(jnp.float32) * (1.0 / 32768.0)
                    scr = s_hi_r + s_lo_r
                    acc = acc + (
                        jnp.dot(
                            recv_vmem[r, m0 : m0 + tm_step, :k_local],
                            w_ref[r, :, :],
                            preferred_element_type=jnp.float32,
                        )
                        * scr
                    )
                out_ref[m0 : m0 + tm_step, :] = (acc * w_sc_ref[...]).astype(
                    jnp.bfloat16
                )

        @pl.when(j == num_n - 1)
        def _():
            tmpl_s = pltpu.make_async_copy(
                src_hbm_ref.at[pl.ds(0, 1)],
                src_hbm_ref.at[pl.ds(0, 1)],
                ssem,
            )
            for _ in range(tp_deg - 1):
                tmpl_s.wait()

    w_own_first = w_own_2d[:, :bn]
    return pl.pallas_call(
        _fused_a2a_wo_hilo_kernel,
        grid=(num_n,),
        in_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec((t_shard, k_packed), lambda j: (0, 0)),
            pl.BlockSpec((k_local, bn), lambda j: (0, 0)),
            pl.BlockSpec((tp_deg, k_local, bn), lambda j: (0, 0, j)),
            pl.BlockSpec((1, bn), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((t_shard, bn), lambda j: (0, j)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((tp_deg, t_shard, k_packed), jnp.float8_e4m3fn),
            pltpu.VMEM((1, t_shard, k_packed), jnp.float8_e4m3fn),
            pltpu.VMEM((t_shard, bn), jnp.float32),
            pltpu.SemaphoreType.DMA((3,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=3 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(packed_local_3d, packed_own_2d, w_own_first, w_3d, wo_scale_2d)


def a3_v13_option3d_fused_async_a2a_wo_3d_hilo_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_own: jax.Array,
    wo_fp8_full: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
) -> jax.Array:
    """Variant A3-v13: `Hi+Lo`-Packed FP8 Prologue + Fused In-Kernel `make_async_remote_copy` A2A + Local-First 3D Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    me = lax.axis_index(tp_axis)
    bn_eff = min(bn, 512) if t_shard >= 2048 else bn

    packed_local_3d = pallas_gate_fp8_quant_packed(
        normed_attn_out, gate, tm=min(512, t_total)
    ).reshape(tp_deg, t_shard, k_local + 128)
    packed_own_2d = lax.dynamic_index_in_dim(
        packed_local_3d, me, axis=0, keepdims=False
    )

    return pallas_option3d_fused_async_a2a_wo_3d_hilo_fp8(
        packed_local_3d,
        packed_own_2d,
        wo_fp8_own,
        wo_fp8_full,
        wo_scale_2d,
        tp_axis=tp_axis,
        bn=bn_eff,
    )


def pallas_barrier_only_bench(
    dummy_buf: jax.Array,
    tp_axis: str = "tp",
) -> jax.Array:
    """Zero-byte barrier-only baseline (#102 Check 3) to isolate barrier + launch overhead from P2P route time."""
    tp_deg = lax.psum(1, tp_axis)

    def _barrier_kernel(in_ref, out_ref):
        del in_ref, out_ref
        me = lax.axis_index(tp_axis)
        if tp_deg > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, tp_deg):
                pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
            pl.semaphore_wait(barrier, tp_deg - 1)

    return pl.pallas_call(
        _barrier_kernel,
        grid=(1,),
        in_specs=[pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct(dummy_buf.shape, dummy_buf.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=9 if tp_deg > 1 else None,
        ),
    )(dummy_buf)


def pallas_p2p_route_bw_bench(
    src_buf: jax.Array,
    target_rank: int,
    num_streams: int = 1,
    src_rank: int = 0,
    tp_axis: str = "tp",
) -> jax.Array:
    """Point-to-point route bandwidth microbenchmark (#87 Experiment 1, #102 Check 3 & #105 Item 3).

    Supports `target_rank == src_rank` (`src_rank -> src_rank` local `make_async_copy` baseline) and
    `target_rank != src_rank` (`pltpu.make_async_remote_copy` from `src_rank` to `target_rank`)
    split across `num_streams` (`1, 2, 4`) concurrent DMA descriptors.
    """
    tp_deg = lax.psum(1, tp_axis)
    s_cnt, chunk_rows, cols = src_buf.shape

    def _route_kernel(src_ref, dst_ref, sems):
        me = lax.axis_index(tp_axis)
        if tp_deg > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, tp_deg):
                pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
            pl.semaphore_wait(barrier, tp_deg - 1)

        if target_rank == src_rank:

            @pl.when(me == src_rank)
            def _():
                for s in range(num_streams):
                    pltpu.make_async_copy(
                        src_ref.at[pl.ds(s, 1)],
                        dst_ref.at[pl.ds(s, 1)],
                        sems.at[s],
                    ).start()
                for s in range(num_streams):
                    pltpu.make_async_copy(
                        dst_ref.at[pl.ds(s, 1)],
                        dst_ref.at[pl.ds(s, 1)],
                        sems.at[s],
                    ).wait()
        else:

            @pl.when(me == src_rank)
            def _():
                for s in range(num_streams):
                    pltpu.make_async_remote_copy(
                        src_ref=src_ref.at[pl.ds(s, 1)],
                        dst_ref=dst_ref.at[pl.ds(s, 1)],
                        send_sem=sems.at[s],
                        recv_sem=sems.at[num_streams + s],
                        device_id=(target_rank,),
                    ).start()
                for s in range(num_streams):
                    pltpu.make_async_copy(
                        src_ref.at[pl.ds(s, 1)],
                        src_ref.at[pl.ds(s, 1)],
                        sems.at[s],
                    ).wait()

            @pl.when(me == target_rank)
            def _():
                for s in range(num_streams):
                    pltpu.make_async_copy(
                        dst_ref.at[pl.ds(s, 1)],
                        dst_ref.at[pl.ds(s, 1)],
                        sems.at[num_streams + s],
                    ).wait()

    return pl.pallas_call(
        _route_kernel,
        grid=(1,),
        in_specs=[pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct((s_cnt, chunk_rows, cols), src_buf.dtype),
        scratch_shapes=[pltpu.SemaphoreType.DMA((2 * num_streams,))],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=10 + src_rank * 16 + target_rank if tp_deg > 1 else None,
        ),
    )(src_buf)


def pallas_option3d_persrc_async_a2a_wo_3d_static_fp8(
    q_local_3d: jax.Array,
    q_own_2d: jax.Array,
    wo_fp8_xor_3d: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    stage_sends: bool = False,
    d_order: tuple[int, ...] | None = None,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """Step-Outer (`grid=(2, tp_deg, 4)`) Per-Source-Semaphore Fused A2A + 3D Wo (#87/#89/#90/#102/#103/#106).

    Uses `grid = (2, tp_deg, 4)` (`n_half = 0..1` outer, `step = 0..tp_deg-1` middle, `j = 0..3` inner):
      1. **1.0× Weight Streaming (`64 MiB`) + Zero Dynamic `m_idx` Slices + Ample VMEM Headroom at `16K` (`#103`, `#106`)**:
         - `acc_vmem` has shape `(4, t_shard, 1024)` (`16 MiB` at `T=8K`, `32 MiB` at `T=16K`).
         - `out_vmem` has shape `(1, t_shard, 1024)` (`2 MiB` at `T=8K`, `4 MiB` at `T=16K`), keeping total VMEM
           at `T=16K` (`t_shard=2048`) to `58 MiB` (`6 MiB` margin below `64 MiB`, `#106` Item 2).
         - `wo_fp8_xor_3d` is streamed once (`64 MiB` total) via `(step, 0, n_half * 4 + j)`.
      2. **Per-Source Receive Semaphores (`rsems.at[step]`) + Route-Ordered XOR Schedule (`peer = me ^ d_order[step]`)**:
         - During `n_half == 0`, `step == 0` computes rank `me`'s own K-block across `j = 0..3` (`4,096` cols)
           before any remote wait; `step = 1 .. tp_deg - 1` waits only on `rsems.at[step]` at `j == 0`
           (starting with the empirically fastest route `d_order[1]`, e.g. on-chip partner `me ^ 1`)
           and computes `4,096` cols per step while remaining ICI peers arrive.
         - During `n_half == 1`, all `tp_deg - 1` remote blocks are already in `recv_vmem`, so `n_half == 1`
           computes the second `4,096` columns with zero collective waits!
    """
    tp_deg, t_shard, k_local = q_local_3d.shape
    _, _, n_dim = wo_fp8_xor_3d.shape
    num_n_total = n_dim // bn
    num_n_half = 2
    num_j = num_n_total // num_n_half  # 4 for n_dim=8192, bn=1024
    tm_step = min(1024, t_shard)
    step_xors = tuple(range(tp_deg)) if d_order is None else tuple(d_order)

    def _step_outer_a2a_wo_kernel(
        src_hbm_ref,
        q_own_ref,
        w_xor_ref,
        w_sc_ref,
        out_hbm_ref,
        recv_vmem,
        acc_vmem,
        out_vmem,
        rsems,
        ssems,
        out_sems,
    ):
        n_half = pl.program_id(0)
        step = pl.program_id(1)
        j = pl.program_id(2)
        me = lax.axis_index(tp_axis)

        @pl.when((n_half == 0) & (step == 0) & (j == 0))
        def _():
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
                pl.semaphore_wait(barrier, tp_deg - 1)
            if stage_sends:
                for d in range(1, min(3, tp_deg)):
                    peer = lax.bitwise_xor(me, step_xors[d])
                    pltpu.make_async_remote_copy(
                        src_ref=src_hbm_ref.at[pl.ds(peer, 1)],
                        dst_ref=recv_vmem.at[pl.ds(d, 1)],
                        send_sem=ssems.at[d],
                        recv_sem=rsems.at[d],
                        device_id=(peer,),
                    ).start()
            else:
                for d in range(1, tp_deg):
                    peer = lax.bitwise_xor(me, step_xors[d])
                    pltpu.make_async_remote_copy(
                        src_ref=src_hbm_ref.at[pl.ds(peer, 1)],
                        dst_ref=recv_vmem.at[pl.ds(d, 1)],
                        send_sem=ssems.at[d],
                        recv_sem=rsems.at[d],
                        device_id=(peer,),
                    ).start()

        @pl.when(step == 0)
        def _():
            for m0 in range(0, t_shard, tm_step):
                acc_vmem[j, m0 : m0 + tm_step, :] = jnp.dot(
                    q_own_ref[m0 : m0 + tm_step, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )

        @pl.when((n_half == 0) & (step > 0) & (j == 0))
        def _():
            if stage_sends:
                for d in range(1, tp_deg):

                    @pl.when(step == d)
                    def _():
                        next_d = d + 2
                        if next_d < tp_deg:
                            next_peer = lax.bitwise_xor(me, step_xors[next_d])
                            pltpu.make_async_remote_copy(
                                src_ref=src_hbm_ref.at[pl.ds(next_peer, 1)],
                                dst_ref=recv_vmem.at[pl.ds(next_d, 1)],
                                send_sem=ssems.at[next_d],
                                recv_sem=rsems.at[next_d],
                                device_id=(next_peer,),
                            ).start()
                        pltpu.make_async_copy(
                            recv_vmem.at[pl.ds(d, 1)],
                            recv_vmem.at[pl.ds(d, 1)],
                            rsems.at[d],
                        ).wait()
            else:
                for d in range(1, tp_deg):

                    @pl.when(step == d)
                    def _():
                        pltpu.make_async_copy(
                            recv_vmem.at[pl.ds(d, 1)],
                            recv_vmem.at[pl.ds(d, 1)],
                            rsems.at[d],
                        ).wait()

        @pl.when((step > 0) & (step < tp_deg - 1))
        def _():
            for m0 in range(0, t_shard, tm_step):
                acc_vmem[j, m0 : m0 + tm_step, :] = acc_vmem[
                    j, m0 : m0 + tm_step, :
                ] + jnp.dot(
                    recv_vmem[step, m0 : m0 + tm_step, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )

        @pl.when(step == tp_deg - 1)
        def _():
            @pl.when((n_half > 0) | (j >= 1))
            def _():
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(0, 1), :, :].reshape(t_shard, bn),
                    out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                    out_sems.at[0],
                ).wait()

            sc_vec = w_sc_ref[...] * STATIC_A3_ACT_SCALE
            for m0 in range(0, t_shard, tm_step):
                acc = acc_vmem[j, m0 : m0 + tm_step, :] + jnp.dot(
                    recv_vmem[step, m0 : m0 + tm_step, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )
                out_vmem[0, m0 : m0 + tm_step, :] = (acc * sc_vec).astype(jnp.bfloat16)
            col_idx = n_half * num_j + j
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(0, 1), :, :].reshape(t_shard, bn),
                out_hbm_ref.at[
                    pl.ds(0, t_shard),
                    pl.ds(col_idx * bn, bn),
                ],
                out_sems.at[0],
            ).start()

        @pl.when((n_half == num_n_half - 1) & (step == tp_deg - 1) & (j == num_j - 1))
        def _():
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(0, 1), :, :].reshape(t_shard, bn),
                out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                out_sems.at[0],
            ).wait()
            for d in range(1, tp_deg):
                pltpu.make_async_copy(
                    src_hbm_ref.at[pl.ds(0, 1)],
                    src_hbm_ref.at[pl.ds(0, 1)],
                    ssems.at[d],
                ).wait()

    return pl.pallas_call(
        _step_outer_a2a_wo_kernel,
        grid=(num_n_half, tp_deg, num_j),
        in_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec((t_shard, k_local), lambda n_half, step, j: (0, 0)),
            pl.BlockSpec(
                (1, k_local, bn),
                lambda n_half, step, j: (step, 0, n_half * num_j + j),
            ),
            pl.BlockSpec((1, bn), lambda n_half, step, j: (0, n_half * num_j + j)),
        ],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((tp_deg, t_shard, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((num_j, t_shard, bn), jnp.float32),
            pltpu.VMEM((1, t_shard, bn), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            collective_id=(4 if stage_sends else 5) if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(q_local_3d, q_own_2d, wo_fp8_xor_3d, wo_scale_2d)


def a3_v14_option3d_persrc_sem_async_a2a_wo_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_xor_3d: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    stage_sends: bool = False,
    d_order: tuple[int, ...] | None = None,
) -> jax.Array:
    """Variant A3-v14 / A3-v15: Step-Outer Per-Source-Semaphore (+ Staged Sends) Fused A2A + 3D Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    me = lax.axis_index(tp_axis)

    q_local_3d = pallas_gate_static_fp8_quant(
        normed_attn_out, gate, tm=min(512, t_total)
    ).reshape(tp_deg, t_shard, k_local)
    q_own_2d = lax.dynamic_index_in_dim(q_local_3d, me, axis=0, keepdims=False)
    return pallas_option3d_persrc_async_a2a_wo_3d_static_fp8(
        q_local_3d,
        q_own_2d,
        wo_fp8_xor_3d,
        wo_scale_2d,
        tp_axis=tp_axis,
        bn=bn,
        stage_sends=stage_sends,
        d_order=d_order,
    )


def pallas_incast_outcast_bidir_bench(
    src_3d: jax.Array,
    mode: str = "incast_6peers",
    tp_axis: str = "tp",
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """Diagnostic P2P Contention & Duplex Microbenchmark (#115 & #117).

    `src_3d` has shape `(tp_deg, 2048, 1024)` (`2.0 MiB` per slice in `fp8_e4m3fn`).
    Every launched remote copy awaits BOTH `send_sem` on the sender rank AND `recv_sem`
    on the receiver rank before exiting so Mosaic's semaphore-zero termination check passes.
    Modes:
      - `"incast_2peers"` / `"incast_6peers"`: Off-chip ranks (`r = 2..3` or `r = 2..7`) simultaneously
        send `2.0 MiB` to rank `0`. Both sender `ssems.at[r]` and receiver `rsems.at[r]` are awaited.
      - `"outcast_2peers"` / `"outcast_6peers"`: Rank `0` simultaneously sends `2.0 MiB` to off-chip
        ranks (`r = 2..3` or `r = 2..7`). Both sender `ssems.at[r]` and receiver `rsems.at[r]` are awaited.
      - `"bidir_r0_r1"`: On-chip TensorCore partners `0 <-> 1` simultaneously send `2.0 MiB` to each other.
      - `"bidir_r0_r2"`: Off-chip ICI partners `0 <-> 2` simultaneously send `2.0 MiB` to each other.
    """
    tp_deg, rows, cols = src_3d.shape

    def _bench_kernel(src_ref, dst_ref, rsems, ssems):
        me = lax.axis_index(tp_axis)
        if tp_deg > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, tp_deg):
                pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
            pl.semaphore_wait(barrier, tp_deg - 1)

        if mode in ("incast_2peers", "incast_6peers"):
            end_r = 4 if mode == "incast_2peers" else tp_deg
            for r in range(2, end_r):

                @pl.when(me == r)
                def _(_r=r):
                    pltpu.make_async_remote_copy(
                        src_ref=src_ref.at[pl.ds(_r, 1)],
                        dst_ref=dst_ref.at[pl.ds(_r, 1)],
                        send_sem=ssems.at[_r],
                        recv_sem=rsems.at[_r],
                        device_id=(0,),
                    ).start()
                    pltpu.make_async_copy(
                        src_ref.at[pl.ds(_r, 1)],
                        src_ref.at[pl.ds(_r, 1)],
                        ssems.at[_r],
                    ).wait()

            @pl.when(me == 0)
            def _():
                for r in range(2, end_r):
                    pltpu.make_async_copy(
                        dst_ref.at[pl.ds(r, 1)],
                        dst_ref.at[pl.ds(r, 1)],
                        rsems.at[r],
                    ).wait()

        elif mode in ("outcast_2peers", "outcast_6peers"):
            end_r = 4 if mode == "outcast_2peers" else tp_deg

            @pl.when(me == 0)
            def _():
                for r in range(2, end_r):
                    pltpu.make_async_remote_copy(
                        src_ref=src_ref.at[pl.ds(r, 1)],
                        dst_ref=dst_ref.at[pl.ds(r, 1)],
                        send_sem=ssems.at[r],
                        recv_sem=rsems.at[r],
                        device_id=(r,),
                    ).start()
                for r in range(2, end_r):
                    pltpu.make_async_copy(
                        src_ref.at[pl.ds(r, 1)],
                        src_ref.at[pl.ds(r, 1)],
                        ssems.at[r],
                    ).wait()

            for r in range(2, end_r):

                @pl.when(me == r)
                def _(_r=r):
                    pltpu.make_async_copy(
                        dst_ref.at[pl.ds(_r, 1)],
                        dst_ref.at[pl.ds(_r, 1)],
                        rsems.at[_r],
                    ).wait()

        elif mode in ("bidir_r0_r1", "bidir_r0_r2"):
            peer_b = 1 if mode == "bidir_r0_r1" else 2

            @pl.when(me == 0)
            def _():
                pltpu.make_async_remote_copy(
                    src_ref=src_ref.at[pl.ds(0, 1)],
                    dst_ref=dst_ref.at[pl.ds(0, 1)],
                    send_sem=ssems.at[0],
                    recv_sem=rsems.at[0],
                    device_id=(peer_b,),
                ).start()
                pltpu.make_async_copy(
                    dst_ref.at[pl.ds(peer_b, 1)],
                    dst_ref.at[pl.ds(peer_b, 1)],
                    rsems.at[peer_b],
                ).wait()
                pltpu.make_async_copy(
                    src_ref.at[pl.ds(0, 1)],
                    src_ref.at[pl.ds(0, 1)],
                    ssems.at[0],
                ).wait()

            @pl.when(me == peer_b)
            def _():
                pltpu.make_async_remote_copy(
                    src_ref=src_ref.at[pl.ds(peer_b, 1)],
                    dst_ref=dst_ref.at[pl.ds(peer_b, 1)],
                    send_sem=ssems.at[peer_b],
                    recv_sem=rsems.at[peer_b],
                    device_id=(0,),
                ).start()
                pltpu.make_async_copy(
                    dst_ref.at[pl.ds(0, 1)],
                    dst_ref.at[pl.ds(0, 1)],
                    rsems.at[0],
                ).wait()
                pltpu.make_async_copy(
                    src_ref.at[pl.ds(peer_b, 1)],
                    src_ref.at[pl.ds(peer_b, 1)],
                    ssems.at[peer_b],
                ).wait()

    return pl.pallas_call(
        _bench_kernel,
        grid=(1,),
        in_specs=[pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct((tp_deg, rows, cols), src_3d.dtype),
        scratch_shapes=[
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((tp_deg,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=15 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(src_3d)


def a3_v16_step_peer(
    me: int,
    d: int,
    tp_deg: int = 8,
    *,
    d_order: tuple[int, ...] | None = None,
    parity_stagger: bool = False,
) -> int:
    """Symmetric involution peer for A3 v16 step ``d`` on rank ``me``.

    When ``parity_stagger=True`` and ``tp_deg == 8`` (2x2x1 TPU7x mesh, 2 TCs/chip), uses an
    exact 1-factorization of K_8 where ``d=1`` is intra-chip D2D (``me ^ 1``), ``d=2..5`` assign
    the two TCs of every chip to orthogonal horizontal + vertical torus links simultaneously
    (1 flow per link, 0 same-link collisions), and ``d=6..7`` cover the two diagonal matchings.
    Because every step ``d`` is a symmetric involution (``peer(peer(me, d), d) == me``), sender
    ``me`` always writes into receiver ``peer``'s slot ``d`` (``d_recv == d_send == d``).
    """
    if d_order is not None or not parity_stagger or tp_deg != 8:
        xor = d if d_order is None else d_order[d]
        return int(me) ^ int(xor)
    r = int(me)
    if d == 0:
        xor = 0
    elif d == 1:
        xor = 1
    elif d == 2:
        xor = 2 if (r & 1) == 0 else 4
    elif d == 3:
        xor = 4 if (r & 1) == 0 else 2
    elif d == 4:
        pop_odd = (((r ^ (r >> 1) ^ (r >> 2)) & 1) != 0)
        xor = 5 if pop_odd else 3
    elif d == 5:
        pop_odd = (((r ^ (r >> 1) ^ (r >> 2)) & 1) != 0)
        xor = 3 if pop_odd else 5
    else:
        xor = d
    return r ^ xor


def _a3_v16_step_peer_jax(
    me: jax.Array,
    d: int,
    tp_deg: int,
    *,
    step_xors: tuple[int, ...],
    parity_stagger: bool,
) -> jax.Array:
    """Scalar JAX counterpart of ``a3_v16_step_peer`` inside Pallas kernels."""
    if not parity_stagger or tp_deg != 8:
        return lax.bitwise_xor(me, jnp.int32(step_xors[d]))
    if d == 0:
        return me
    if d == 1:
        return lax.bitwise_xor(me, jnp.int32(1))
    if d in (2, 3):
        is_even = lax.bitwise_and(me, jnp.int32(1)) == jnp.int32(0)
        xor = jnp.where(
            is_even,
            jnp.int32(2 if d == 2 else 4),
            jnp.int32(4 if d == 2 else 2),
        )
        return lax.bitwise_xor(me, xor)
    if d in (4, 5):
        pop1 = lax.bitwise_xor(me, lax.shift_right_logical(me, jnp.int32(1)))
        pop2 = lax.bitwise_xor(pop1, lax.shift_right_logical(me, jnp.int32(2)))
        pop_odd = lax.bitwise_and(pop2, jnp.int32(1)) != jnp.int32(0)
        xor = jnp.where(
            pop_odd,
            jnp.int32(5 if d == 4 else 3),
            jnp.int32(3 if d == 4 else 5),
        )
        return lax.bitwise_xor(me, xor)
    return lax.bitwise_xor(me, jnp.int32(d))


def pallas_option3d_zero_copy9_persrc_async_a2a_wo_3d_static_fp8(
    q_local_3d: jax.Array,
    wo_fp8_xor_3d: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    stage_sends: bool = False,
    d_order: tuple[int, ...] | None = None,
    opt_wo: bool = False,
    valid_tokens_1d: jax.Array | None = None,
    vmem_limit_mb: int = 64,
    stage_sends_mode: int | None = None,
) -> jax.Array:
    """Wave 7 (`a3_v16`, `#114`, `#115`, `#116`): Token-Half Outer + Full-N Pass (`grid = (num_t_half, tp_deg, 8)`) + Fixed `d_order = (0, 1, ...)` + Zero `%copy.8`.

    Applies all 4 advisor-verified fixes from `#114`, `#115`, and `#116`:
      1. Fixed `d_order = (0, 1, 2, 3, 4, 5, 6, 7)` so `step = 1` always consumes `me ^ 1` (`4.15 µs` on-chip partner!).
      2. Full-N pass (`num_j = 8, bn = 1024`) with Token-Half outer loop (`num_t_half = t_shard // 1024`, i.e.,
         `1` at `T=8K` and `2` at `T=16K`, `tm_half = min(1024, t_shard)`):
           - At `8K` (`num_t_half = 1`): 100% of `Wo` (`84 µs`, `10.5 µs`/step) runs inside the `~85 µs` A2A window!
           - At `16K` (`num_t_half = 2`): splits along `1,024`-row token halves (`1.0 MiB` per remote transfer),
             keeping `acc_vmem (8, 1024, 1024) fp32 = 32 MiB` (`54 MiB` total VMEM < `64 MiB`) and overlapping
             `t_half = 0` and `t_half = 1` across all `8,192` columns!
      3. Zero `q_own_2d` operand (`-13.5 / -21.5 µs` `%copy.8`): copies `src_hbm_ref.at[pl.ds(me, 1)]` into
         `recv_vmem` inside `(0, 0, 0)` via local `make_async_copy`.
      4. Supports both `stage_sends=False` (all sends launched at `(0, 0, 0)`), `stage_sends_mode=1/2/3`
         inside `_opt_wo_kernel` (deferring `t_half=1` sends until `t_half=0` lands + parity-staggered
         1-factorization `d_order`), and `stage_sends=True` (`#116`'s `1-in-flight` schedule).
    """
    from sglang_tpu.envs import envs as _tpu_envs

    tp_deg, t_shard, k_local = q_local_3d.shape
    _, _, n_dim = wo_fp8_xor_3d.shape
    ss_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS", None)
    eff_stage_mode = (
        int(stage_sends_mode)
        if stage_sends_mode is not None
        else (int(ss_env.get()) if ss_env is not None else 0)
    )
    use_parity_stagger = (
        d_order is None and tp_deg == 8 and eff_stage_mode in (1, 2)
    )
    bn2048_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A3_BN2048", None)
    allow_bn2048 = bool(
        bn2048_env is not None
        and bn2048_env.get()
        and t_shard <= 4096
        and (t_shard <= 2048 or t_shard % 512 == 0)
    )
    bn_eff = (
        2048
        if (
            opt_wo
            and not stage_sends
            and (t_shard <= 1024 or allow_bn2048)
            and n_dim % 2048 == 0
        )
        else bn
    )
    max_tm_half = (
        1024
        if (t_shard <= 2048 or (t_shard == 3072 and bn_eff == 1024))
        else 512
    )
    tm_half = next(
        (
            t
            for t in (1024, 512, 256, 128)
            if t <= max_tm_half and t <= t_shard and t_shard % t == 0
        ),
        t_shard,
    )
    num_t_half = t_shard // tm_half  # 1 at 8K, 2 at 16K, 3-8 at 24K/32K
    num_j = n_dim // bn_eff  # 8 for bn=1024, 4 for bn=2048
    step_xors = tuple(range(tp_deg)) if d_order is None else tuple(d_order)
    src_4d = q_local_3d.reshape(tp_deg, num_t_half, tm_half, k_local)
    has_skip = valid_tokens_1d is not None

    if opt_wo and not stage_sends:

        def _opt_wo_kernel(*args):
            if has_skip:
                vtok_ref = args[0]
                (
                    src_hbm_ref,
                    w_xor_ref,
                    w_sc_ref,
                    out_hbm_ref,
                    recv_vmem,
                    acc_vmem,
                    out_vmem,
                    rsems,
                    ssems,
                    out_sems,
                ) = args[1:]
            else:
                vtok_ref = None
                (
                    src_hbm_ref,
                    w_xor_ref,
                    w_sc_ref,
                    out_hbm_ref,
                    recv_vmem,
                    acc_vmem,
                    out_vmem,
                    rsems,
                    ssems,
                    out_sems,
                ) = args
            t_half = pl.program_id(0)
            step = pl.program_id(1)
            j = pl.program_id(2)
            me = lax.axis_index(tp_axis)
            half_active = (
                (me * t_shard + t_half * tm_half) < vtok_ref[0] if has_skip else True
            )

            def _send_remote(th_idx: int, d_idx: int):
                peer = _a3_v16_step_peer_jax(
                    me,
                    d_idx,
                    tp_deg,
                    step_xors=step_xors,
                    parity_stagger=use_parity_stagger,
                )
                pltpu.make_async_remote_copy(
                    src_ref=src_hbm_ref.at[
                        pl.ds(peer, 1), pl.ds(th_idx, 1), :, :
                    ],
                    dst_ref=recv_vmem.at[
                        pl.ds(th_idx, 1), pl.ds(d_idx, 1), :, :
                    ],
                    send_sem=ssems.at[th_idx, d_idx],
                    recv_sem=rsems.at[th_idx, d_idx],
                    device_id={tp_axis: peer},
                ).start()

            @pl.when((t_half == 0) & (step == 0) & (j == 0))
            def _():
                for th in range(num_t_half):
                    pltpu.make_async_copy(
                        src_hbm_ref.at[pl.ds(me, 1), pl.ds(th, 1), :, :],
                        recv_vmem.at[pl.ds(th, 1), pl.ds(0, 1), :, :],
                        rsems.at[th, 0],
                    ).start()
                if tp_deg > 1:
                    barrier = pltpu.get_barrier_semaphore()
                    for d in range(1, tp_deg):
                        pl.semaphore_signal(
                            barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)}
                        )
                    pl.semaphore_wait(barrier, tp_deg - 1)
                    if eff_stage_mode == 2:
                        # 1-in-flight staged: start d=1 (intra-chip) + d=2 (1st off-chip) for th=0.
                        for d in range(1, min(3, tp_deg)):
                            _send_remote(0, d)
                    elif eff_stage_mode in (1, 3):
                        # Half-deferred: launch only t_half=0 remote sends now so t_half=1 never
                        # interleaves with t_half=0 on ICI links!
                        for d in range(1, tp_deg):
                            _send_remote(0, d)
                    else:
                        for th in range(num_t_half):
                            for d in range(1, tp_deg):
                                _send_remote(th, d)
                for th in range(num_t_half):
                    pltpu.make_async_copy(
                        recv_vmem.at[pl.ds(th, 1), pl.ds(0, 1), :, :],
                        recv_vmem.at[pl.ds(th, 1), pl.ds(0, 1), :, :],
                        rsems.at[th, 0],
                    ).wait()

            @pl.when((step == 0) & half_active if has_skip else (step == 0))
            def _():
                acc_vmem[j, :, :] = jnp.dot(
                    recv_vmem[t_half, 0, :, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )

            @pl.when((step > 0) & (j == 0))
            def _():
                if eff_stage_mode == 2:
                    for _th in range(num_t_half):

                        @pl.when(t_half == _th)
                        def _(_th=_th):
                            for _d in range(1, tp_deg):

                                @pl.when(step == _d)
                                def _(_d=_d):
                                    pltpu.make_async_copy(
                                        recv_vmem.at[pl.ds(_th, 1), pl.ds(_d, 1), :, :],
                                        recv_vmem.at[pl.ds(_th, 1), pl.ds(_d, 1), :, :],
                                        rsems.at[_th, _d],
                                    ).wait()
                                    if tp_deg > 1:
                                        next_d = _d + 1
                                        if _d >= 2 and next_d < tp_deg:
                                            _send_remote(_th, next_d)
                                        elif (
                                            _d == tp_deg - 1
                                            and _th + 1 < num_t_half
                                        ):
                                            for d_init in range(1, min(3, tp_deg)):
                                                _send_remote(_th + 1, d_init)
                else:
                    pltpu.make_async_copy(
                        recv_vmem.at[pl.ds(t_half, 1), pl.ds(step, 1), :, :],
                        recv_vmem.at[pl.ds(t_half, 1), pl.ds(step, 1), :, :],
                        rsems.at[t_half, step],
                    ).wait()
                    if tp_deg > 1 and eff_stage_mode in (1, 3) and num_t_half > 1:
                        for _th in range(num_t_half - 1):

                            @pl.when((t_half == _th) & (step == tp_deg - 1))
                            def _(_next_th=_th + 1):
                                for d in range(1, tp_deg):
                                    _send_remote(_next_th, d)

            @pl.when(
                (step > 0) & (step < tp_deg - 1) & half_active
                if has_skip
                else ((step > 0) & (step < tp_deg - 1))
            )
            def _():
                acc_vmem[j, :, :] = acc_vmem[j, :, :] + jnp.dot(
                    recv_vmem[t_half, step, :, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )

            @pl.when(step == tp_deg - 1)
            def _():
                out_slot = lax.rem(j, 2)

                @pl.when((t_half > 0) | (j >= 2))
                def _():
                    pltpu.make_async_copy(
                        out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(tm_half, bn_eff),
                        out_hbm_ref.at[pl.ds(0, tm_half), pl.ds(0, bn_eff)],
                        out_sems.at[out_slot],
                    ).wait()

                def _epilogue_dot_and_scale():
                    acc_vmem[j, :, :] = acc_vmem[j, :, :] + jnp.dot(
                        recv_vmem[t_half, step, :, :],
                        w_xor_ref[0, :, :],
                        preferred_element_type=jnp.float32,
                    )
                    sub_r = min(128, tm_half)
                    sub_c = min(256, bn_eff)
                    for rb in range(tm_half // sub_r):
                        r0, r1 = rb * sub_r, (rb + 1) * sub_r
                        for cb in range(bn_eff // sub_c):
                            c0, c1 = cb * sub_c, (cb + 1) * sub_c
                            sc_sub = w_sc_ref[:, c0:c1] * STATIC_A3_ACT_SCALE
                            out_vmem[out_slot, r0:r1, c0:c1] = (
                                acc_vmem[j, r0:r1, c0:c1] * sc_sub
                            ).astype(jnp.bfloat16)

                if has_skip:

                    @pl.when(half_active)
                    def _():
                        _epilogue_dot_and_scale()

                    @pl.when(~half_active)
                    def _():
                        out_vmem[out_slot, :, :] = jnp.zeros(
                            (tm_half, bn_eff), jnp.bfloat16
                        )
                else:
                    _epilogue_dot_and_scale()

                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(tm_half, bn_eff),
                    out_hbm_ref.at[
                        pl.ds(t_half * tm_half, tm_half),
                        pl.ds(j * bn_eff, bn_eff),
                    ],
                    out_sems.at[out_slot],
                ).start()

            @pl.when(
                (t_half == num_t_half - 1) & (step == tp_deg - 1) & (j == num_j - 1)
            )
            def _():
                for s in range(min(2, num_j)):
                    pltpu.make_async_copy(
                        out_vmem.at[pl.ds(s, 1), :, :].reshape(tm_half, bn_eff),
                        out_hbm_ref.at[pl.ds(0, tm_half), pl.ds(0, bn_eff)],
                        out_sems.at[s],
                    ).wait()
                if tp_deg > 1:
                    for th in range(num_t_half):
                        for d in range(1, tp_deg):
                            pltpu.make_async_copy(
                                src_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), :, :],
                                src_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), :, :],
                                ssems.at[th, d],
                            ).wait()

        any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
        scratches = [
            pltpu.VMEM((num_t_half, tp_deg, tm_half, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((num_j, tm_half, bn_eff), jnp.float32),
            pltpu.VMEM((2, tm_half, bn_eff), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((num_t_half, tp_deg)),
            pltpu.SemaphoreType.DMA((num_t_half, tp_deg)),
            pltpu.SemaphoreType.DMA((2,)),
        ]
        cp = pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            collective_id=6 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        )
        ss_tag = f"_ss{eff_stage_mode}" if eff_stage_mode > 0 else ""
        if not has_skip:
            return pl.pallas_call(
                _opt_wo_kernel,
                grid=(num_t_half, tp_deg, num_j),
                in_specs=[
                    any_spec,
                    pl.BlockSpec(
                        (1, k_local, bn_eff), lambda t_half, step, j: (step, 0, j)
                    ),
                    pl.BlockSpec((1, bn_eff), lambda t_half, step, j: (0, j)),
                ],
                out_specs=any_spec,
                out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
                scratch_shapes=scratches,
                compiler_params=cp,
                name=f"pallas_a3_wo_v16_opt_bn{bn_eff}{ss_tag}",
            )(src_4d, wo_fp8_xor_3d, wo_scale_2d)

        return pl.pallas_call(
            _opt_wo_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(num_t_half, tp_deg, num_j),
                in_specs=[
                    any_spec,
                    pl.BlockSpec(
                        (1, k_local, bn_eff),
                        lambda t_half, step, j, _vt: (step, 0, j),
                    ),
                    pl.BlockSpec((1, bn_eff), lambda t_half, step, j, _vt: (0, j)),
                ],
                out_specs=any_spec,
                scratch_shapes=scratches,
            ),
            out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
            compiler_params=cp,
            name=f"pallas_a3_wo_v16_opt_bn{bn_eff}{ss_tag}_skip",
        )(
            valid_tokens_1d.reshape(1).astype(jnp.int32),
            src_4d,
            wo_fp8_xor_3d,
            wo_scale_2d,
        )

    def _zero_copy9_a2a_wo_kernel(
        src_hbm_ref,
        w_xor_ref,
        w_sc_ref,
        out_hbm_ref,
        recv_vmem,
        acc_vmem,
        out_vmem,
        rsems,
        ssems,
        out_sems,
    ):
        t_half = pl.program_id(0)
        step = pl.program_id(1)
        j = pl.program_id(2)
        me = lax.axis_index(tp_axis)

        @pl.when((t_half == 0) & (step == 0) & (j == 0))
        def _():
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(
                        barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)}
                    )
                pl.semaphore_wait(barrier, tp_deg - 1)
            if not stage_sends:
                # Unstaged: launch all 7 remote 1.0 MiB copies for th=0 first, then th=1 (#114 Fix 2)
                for th in range(num_t_half):
                    for d in range(1, tp_deg):
                        peer = lax.bitwise_xor(me, step_xors[d])
                        pltpu.make_async_remote_copy(
                            src_ref=src_hbm_ref.at[pl.ds(peer, 1), pl.ds(th, 1), :, :],
                            dst_ref=recv_vmem.at[pl.ds(th, 1), pl.ds(d, 1), :, :],
                            send_sem=ssems.at[th, d],
                            recv_sem=rsems.at[th, d],
                            device_id={tp_axis: peer},
                        ).start()
            else:
                # 1-in-flight staged (#116): launch d=1 (on-chip 2.1 us) and d=2 (1st off-chip 12.5 us) for t_half=0
                for d in range(1, min(3, tp_deg)):
                    peer = lax.bitwise_xor(me, step_xors[d])
                    pltpu.make_async_remote_copy(
                        src_ref=src_hbm_ref.at[pl.ds(peer, 1), pl.ds(0, 1), :, :],
                        dst_ref=recv_vmem.at[pl.ds(0, 1), pl.ds(d, 1), :, :],
                        send_sem=ssems.at[0, d],
                        recv_sem=rsems.at[0, d],
                        device_id={tp_axis: peer},
                    ).start()

            # Local copy of own block (d = 0) for all token halves (~0.2-0.4 us)
            for th in range(num_t_half):
                pltpu.make_async_copy(
                    src_hbm_ref.at[pl.ds(me, 1), pl.ds(th, 1), :, :],
                    recv_vmem.at[pl.ds(th, 1), pl.ds(0, 1), :, :],
                    rsems.at[th, 0],
                ).start()
            for th in range(num_t_half):
                pltpu.make_async_copy(
                    recv_vmem.at[pl.ds(th, 1), pl.ds(0, 1), :, :],
                    recv_vmem.at[pl.ds(th, 1), pl.ds(0, 1), :, :],
                    rsems.at[th, 0],
                ).wait()

        @pl.when(step == 0)
        def _():
            acc_vmem[j, :, :] = jnp.dot(
                recv_vmem[t_half, 0, :, :],
                w_xor_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )

        @pl.when((step > 0) & (j == 0))
        def _():
            for th in range(num_t_half):

                @pl.when(t_half == th)
                def _(_th=th):
                    for d in range(1, tp_deg):

                        @pl.when(step == d)
                        def _(_d=d):
                            pltpu.make_async_copy(
                                recv_vmem.at[pl.ds(_th, 1), pl.ds(_d, 1), :, :],
                                recv_vmem.at[pl.ds(_th, 1), pl.ds(_d, 1), :, :],
                                rsems.at[_th, _d],
                            ).wait()
                            if stage_sends:
                                # 1-in-flight (#116): as soon as d >= 2 arrives, start next_d = d + 1!
                                next_d = _d + 1
                                if _d >= 2 and next_d < tp_deg:
                                    peer_next = lax.bitwise_xor(me, step_xors[next_d])
                                    pltpu.make_async_remote_copy(
                                        src_ref=src_hbm_ref.at[
                                            pl.ds(peer_next, 1), pl.ds(_th, 1), :, :
                                        ],
                                        dst_ref=recv_vmem.at[
                                            pl.ds(_th, 1), pl.ds(next_d, 1), :, :
                                        ],
                                        send_sem=ssems.at[_th, next_d],
                                        recv_sem=rsems.at[_th, next_d],
                                        device_id={tp_axis: peer_next},
                                    ).start()
                                elif _d == tp_deg - 1 and _th + 1 < num_t_half:
                                    # When t_half=0's last block arrives, start d=1 and d=2 for t_half=1!
                                    for d_init in range(1, min(3, tp_deg)):
                                        peer_init = lax.bitwise_xor(
                                            me, step_xors[d_init]
                                        )
                                        pltpu.make_async_remote_copy(
                                            src_ref=src_hbm_ref.at[
                                                pl.ds(peer_init, 1),
                                                pl.ds(_th + 1, 1),
                                                :,
                                                :,
                                            ],
                                            dst_ref=recv_vmem.at[
                                                pl.ds(_th + 1, 1),
                                                pl.ds(d_init, 1),
                                                :,
                                                :,
                                            ],
                                            send_sem=ssems.at[_th + 1, d_init],
                                            recv_sem=rsems.at[_th + 1, d_init],
                                            device_id={tp_axis: peer_init},
                                        ).start()

        @pl.when((step > 0) & (step < tp_deg - 1))
        def _():
            acc_vmem[j, :, :] = acc_vmem[j, :, :] + jnp.dot(
                recv_vmem[t_half, step, :, :],
                w_xor_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )

        @pl.when(step == tp_deg - 1)
        def _():
            out_slot = lax.rem(j, 2)

            @pl.when((t_half > 0) | (j >= 2))
            def _():
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(tm_half, bn),
                    out_hbm_ref.at[pl.ds(0, tm_half), pl.ds(0, bn)],
                    out_sems.at[out_slot],
                ).wait()

            sc_vec = w_sc_ref[...] * STATIC_A3_ACT_SCALE
            acc = acc_vmem[j, :, :] + jnp.dot(
                recv_vmem[t_half, step, :, :],
                w_xor_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )
            out_vmem[out_slot, :, :] = (acc * sc_vec).astype(jnp.bfloat16)
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(tm_half, bn),
                out_hbm_ref.at[
                    pl.ds(t_half * tm_half, tm_half),
                    pl.ds(j * bn, bn),
                ],
                out_sems.at[out_slot],
            ).start()

        @pl.when((t_half == num_t_half - 1) & (step == tp_deg - 1) & (j == num_j - 1))
        def _():
            for s in range(min(2, num_j)):
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(s, 1), :, :].reshape(tm_half, bn),
                    out_hbm_ref.at[pl.ds(0, tm_half), pl.ds(0, bn)],
                    out_sems.at[s],
                ).wait()
            for th in range(num_t_half):
                for d in range(1, tp_deg):
                    pltpu.make_async_copy(
                        src_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), :, :],
                        src_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), :, :],
                        ssems.at[th, d],
                    ).wait()

    return pl.pallas_call(
        _zero_copy9_a2a_wo_kernel,
        grid=(num_t_half, tp_deg, num_j),
        in_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(
                (1, k_local, bn),
                lambda t_half, step, j: (step, 0, j),
            ),
            pl.BlockSpec((1, bn), lambda t_half, step, j: (0, j)),
        ],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((num_t_half, tp_deg, tm_half, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((num_j, tm_half, bn), jnp.float32),
            pltpu.VMEM((2, tm_half, bn), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((num_t_half, tp_deg)),
            pltpu.SemaphoreType.DMA((num_t_half, tp_deg)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            collective_id=6 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(src_4d, wo_fp8_xor_3d, wo_scale_2d)



def pallas_option3d_2chunk_async_a2a_wo_3d_static_fp8(
    q_part0_3d: jax.Array,
    q_part1_3d: jax.Array,
    wo_fp8_xor_3d: jax.Array,
    wo_scale_2d: jax.Array,
    *,
    tp_axis: str = "model",
    bn: int = 1024,
    vmem_limit_mb: int = 62,
    d_order: tuple[int, ...] | None = None,
    stage_sends_mode: int | None = None,
) -> jax.Array:
    """Zero-concat 2-chunk CCS variant of ``_opt_wo_kernel`` reading ``(q_part0_3d, q_part1_3d)`` directly.

    Eliminates the external ``torch.cat(q_parts, dim=1)`` (``%pad_maximum_fusion``: ``+5.14 ms/fwd``
    and ``+2.94 GB/fwd`` at ``c24k``) by issuing 64-row-aligned local/remote DMAs directly from
    ``q_part0_3d [tp, c0, k_local]`` and ``q_part1_3d [tp, c1, k_local]`` into ``recv_vmem``.
    """
    from sglang_tpu.envs import envs as _tpu_envs

    tp_deg, c0, k_local = q_part0_3d.shape
    tp_deg_1, c1, k_local_1 = q_part1_3d.shape
    assert tp_deg == tp_deg_1 and k_local == k_local_1
    t_shard = c0 + c1
    if c0 < 64 or c1 < 64 or c0 % 64 != 0 or c1 % 64 != 0:
        return pallas_option3d_zero_copy9_persrc_async_a2a_wo_3d_static_fp8(
            jnp.concatenate([q_part0_3d, q_part1_3d], axis=1),
            wo_fp8_xor_3d,
            wo_scale_2d,
            tp_axis=tp_axis,
            bn=bn,
            vmem_limit_mb=vmem_limit_mb,
            opt_wo=True,
            d_order=d_order,
            stage_sends_mode=stage_sends_mode,
        )

    _, _, n_dim = wo_fp8_xor_3d.shape
    ss_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS", None)
    eff_stage_mode = (
        int(stage_sends_mode)
        if stage_sends_mode is not None
        else (int(ss_env.get()) if ss_env is not None else 0)
    )
    use_parity_stagger = (
        d_order is None and tp_deg == 8 and eff_stage_mode in (1, 2)
    )
    bn2048_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A3_BN2048", None)
    allow_bn2048 = bool(
        bn2048_env is not None
        and bn2048_env.get()
        and t_shard <= 4096
        and (t_shard <= 2048 or t_shard % 512 == 0)
    )
    bn_eff = (
        2048
        if ((t_shard <= 1024 or allow_bn2048) and n_dim % 2048 == 0)
        else bn
    )
    max_tm_half = (
        1024
        if (t_shard <= 2048 or (t_shard == 3072 and bn_eff == 1024))
        else 512
    )
    tm_half = next(
        (
            t
            for t in (1024, 512, 256, 128)
            if t <= max_tm_half and t <= t_shard and t_shard % t == 0
        ),
        t_shard,
    )
    num_t_half = t_shard // tm_half
    num_j = n_dim // bn_eff
    step_xors = tuple(range(tp_deg)) if d_order is None else tuple(d_order)

    # Static DMA segment table: each entry is (seg_id, th_i, part_idx, src_r0, dst_r0, nrows).
    segs: list[tuple[int, int, int, int, int, int]] = []
    th_segs: list[list[tuple[int, int, int, int, int, int]]] = [
        [] for _ in range(num_t_half)
    ]
    for th_i in range(num_t_half):
        r0 = th_i * tm_half
        r1 = r0 + tm_half
        if r1 <= c0:
            item = (len(segs), th_i, 0, r0, 0, tm_half)
            segs.append(item)
            th_segs[th_i].append(item)
        elif r0 >= c0:
            item = (len(segs), th_i, 1, r0 - c0, 0, tm_half)
            segs.append(item)
            th_segs[th_i].append(item)
        else:
            len0 = c0 - r0
            len1 = r1 - c0
            item0 = (len(segs), th_i, 0, r0, 0, len0)
            segs.append(item0)
            th_segs[th_i].append(item0)
            item1 = (len(segs), th_i, 1, 0, len0, len1)
            segs.append(item1)
            th_segs[th_i].append(item1)
    num_segs = len(segs)

    def _opt_wo_2chunk_kernel(
        src0_hbm_ref,
        src1_hbm_ref,
        w_xor_ref,
        w_sc_ref,
        out_hbm_ref,
        recv_vmem,
        acc_vmem,
        out_vmem,
        rsems,
        ssems,
        out_sems,
    ):
        t_half = pl.program_id(0)
        step = pl.program_id(1)
        j = pl.program_id(2)
        me = lax.axis_index(tp_axis)

        def _src_slice(p_idx: int, r_idx, s_r0: int, nr: int):
            ref = src0_hbm_ref if p_idx == 0 else src1_hbm_ref
            return ref.at[pl.ds(r_idx, 1), pl.ds(s_r0, nr), :]

        def _dst_slice(th_i: int, d_idx, d_r0: int, nr: int):
            return recv_vmem.at[pl.ds(th_i * tp_deg + d_idx, 1), pl.ds(d_r0, nr), :]

        def _send_remote(th_idx: int, d_idx: int):
            peer = _a3_v16_step_peer_jax(
                me,
                d_idx,
                tp_deg,
                step_xors=step_xors,
                parity_stagger=use_parity_stagger,
            )
            for seg_id, _, p_idx, s_r0, d_r0, nr in th_segs[th_idx]:
                pltpu.make_async_remote_copy(
                    src_ref=_src_slice(p_idx, peer, s_r0, nr),
                    dst_ref=_dst_slice(th_idx, d_idx, d_r0, nr),
                    send_sem=ssems.at[seg_id, d_idx],
                    recv_sem=rsems.at[seg_id, d_idx],
                    device_id={tp_axis: peer},
                ).start()

        @pl.when((t_half == 0) & (step == 0) & (j == 0))
        def _():
            for seg_id, th_i, p_idx, s_r0, d_r0, nr in segs:
                pltpu.make_async_copy(
                    _src_slice(p_idx, me, s_r0, nr),
                    _dst_slice(th_i, 0, d_r0, nr),
                    rsems.at[seg_id, 0],
                ).start()
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(
                        barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)}
                    )
                pl.semaphore_wait(barrier, tp_deg - 1)
                if eff_stage_mode == 2:
                    for d in range(1, min(3, tp_deg)):
                        _send_remote(0, d)
                elif eff_stage_mode in (1, 3):
                    for d in range(1, tp_deg):
                        _send_remote(0, d)
                else:
                    for th_i in range(num_t_half):
                        for d in range(1, tp_deg):
                            _send_remote(th_i, d)
            for seg_id, th_i, _, _, d_r0, nr in segs:
                pltpu.make_async_copy(
                    _dst_slice(th_i, 0, d_r0, nr),
                    _dst_slice(th_i, 0, d_r0, nr),
                    rsems.at[seg_id, 0],
                ).wait()

        @pl.when(step == 0)
        def _():
            acc_vmem[j, :, :] = jnp.dot(
                recv_vmem[t_half * tp_deg, :, :],
                w_xor_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )

        @pl.when((step > 0) & (j == 0))
        def _():
            if eff_stage_mode == 2:
                for _th in range(num_t_half):

                    @pl.when(t_half == _th)
                    def _(_th=_th):
                        for _d in range(1, tp_deg):

                            @pl.when(step == _d)
                            def _(_d=_d):
                                for seg_id, _, _, _, d_r0, nr in th_segs[_th]:
                                    pltpu.make_async_copy(
                                        _dst_slice(_th, _d, d_r0, nr),
                                        _dst_slice(_th, _d, d_r0, nr),
                                        rsems.at[seg_id, _d],
                                    ).wait()
                                if tp_deg > 1:
                                    next_d = _d + 1
                                    if _d >= 2 and next_d < tp_deg:
                                        _send_remote(_th, next_d)
                                    elif _d == tp_deg - 1 and _th + 1 < num_t_half:
                                        for d_init in range(1, min(3, tp_deg)):
                                            _send_remote(_th + 1, d_init)
            else:
                for _th in range(num_t_half):

                    @pl.when(t_half == _th)
                    def _(_th=_th):
                        for seg_id, _, _, _, d_r0, nr in th_segs[_th]:
                            pltpu.make_async_copy(
                                _dst_slice(_th, step, d_r0, nr),
                                _dst_slice(_th, step, d_r0, nr),
                                rsems.at[seg_id, step],
                            ).wait()
                        if (
                            tp_deg > 1
                            and eff_stage_mode in (1, 3)
                            and _th + 1 < num_t_half
                        ):

                            @pl.when(step == tp_deg - 1)
                            def _(_next_th=_th + 1):
                                for d in range(1, tp_deg):
                                    _send_remote(_next_th, d)

        @pl.when((step > 0) & (step < tp_deg - 1))
        def _():
            acc_vmem[j, :, :] = acc_vmem[j, :, :] + jnp.dot(
                recv_vmem[t_half * tp_deg + step, :, :],
                w_xor_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )

        @pl.when(step == tp_deg - 1)
        def _():
            out_slot = lax.rem(j, 2)
            slot_row0 = pl.multiple_of(out_slot * tm_half, 8)

            @pl.when((t_half > 0) | (j >= 2))
            def _():
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(slot_row0, tm_half), :],
                    out_hbm_ref.at[pl.ds(0, tm_half), pl.ds(0, bn_eff)],
                    out_sems.at[out_slot],
                ).wait()

            acc_vmem[j, :, :] = acc_vmem[j, :, :] + jnp.dot(
                recv_vmem[t_half * tp_deg + step, :, :],
                w_xor_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )
            sub_r = min(128, tm_half)
            sub_c = min(256, bn_eff)
            for rb in range(tm_half // sub_r):
                r0_b, r1_b = rb * sub_r, (rb + 1) * sub_r
                dst_r0 = pl.multiple_of(out_slot * tm_half + r0_b, 8)
                for cb in range(bn_eff // sub_c):
                    c0_b, c1_b = cb * sub_c, (cb + 1) * sub_c
                    sc_sub = w_sc_ref[:, c0_b:c1_b] * STATIC_A3_ACT_SCALE
                    out_vmem[pl.ds(dst_r0, sub_r), c0_b:c1_b] = (
                        acc_vmem[j, r0_b:r1_b, c0_b:c1_b] * sc_sub
                    ).astype(jnp.bfloat16)

            pltpu.make_async_copy(
                out_vmem.at[pl.ds(slot_row0, tm_half), :],
                out_hbm_ref.at[
                    pl.ds(t_half * tm_half, tm_half),
                    pl.ds(j * bn_eff, bn_eff),
                ],
                out_sems.at[out_slot],
            ).start()

        @pl.when(
            (t_half == num_t_half - 1) & (step == tp_deg - 1) & (j == num_j - 1)
        )
        def _():
            for s in range(min(2, num_j)):
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(s * tm_half, tm_half), :],
                    out_hbm_ref.at[pl.ds(0, tm_half), pl.ds(0, bn_eff)],
                    out_sems.at[s],
                ).wait()
            if tp_deg > 1:
                for seg_id, _, p_idx, s_r0, _, nr in segs:
                    for d in range(1, tp_deg):
                        pltpu.make_async_copy(
                            _src_slice(p_idx, 0, s_r0, nr),
                            _src_slice(p_idx, 0, s_r0, nr),
                            ssems.at[seg_id, d],
                        ).wait()

    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    scratches = [
        pltpu.VMEM((num_t_half * tp_deg, tm_half, k_local), jnp.float8_e4m3fn),
        pltpu.VMEM((num_j, tm_half, bn_eff), jnp.float32),
        pltpu.VMEM((2 * tm_half, bn_eff), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((num_segs, tp_deg)),
        pltpu.SemaphoreType.DMA((num_segs, tp_deg)),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
        collective_id=6 if tp_deg > 1 else None,
        vmem_limit_bytes=vmem_limit_mb * (1 << 20),
    )
    ss_tag = f"_ss{eff_stage_mode}" if eff_stage_mode > 0 else ""
    return pl.pallas_call(
        _opt_wo_2chunk_kernel,
        grid=(num_t_half, tp_deg, num_j),
        in_specs=[
            any_spec,
            any_spec,
            pl.BlockSpec(
                (1, k_local, bn_eff), lambda t_half, step, j: (step, 0, j)
            ),
            pl.BlockSpec((1, bn_eff), lambda t_half, step, j: (0, j)),
        ],
        out_specs=any_spec,
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=scratches,
        compiler_params=cp,
        name=f"pallas_a3_wo_2chunk_c{c0}_{c1}_bn{bn_eff}{ss_tag}",
    )(q_part0_3d, q_part1_3d, wo_fp8_xor_3d, wo_scale_2d)

def a3_v16_option3d_zero_copy9_persrc_sem_async_a2a_wo_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_xor_3d: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    stage_sends: bool = False,
    d_order: tuple[int, ...] | None = None,
    head_norm_eps: float | None = None,
    rpa_packed_out: bool = False,
    prologue_tm: int = 512,
    opt_wo: bool = False,
    valid_tokens_1d: jax.Array | None = None,
) -> jax.Array:
    """Variant A3-v16 (`#114`, `#115`, `#116`): Token-Half Outer + Full-N Pass + Fixed `d_order` + Zero `%copy.8`."""
    t_total, k_local = gate.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    eff_ptm = 512 if valid_tokens_1d is not None else (1024 if opt_wo else prologue_tm)

    q_local_3d = pallas_gate_static_fp8_quant(
        normed_attn_out,
        gate,
        tm=min(eff_ptm, t_total),
        head_norm_eps=head_norm_eps,
        rpa_packed_out=rpa_packed_out,
        valid_tokens_1d=valid_tokens_1d,
    ).reshape(tp_deg, t_shard, k_local)
    return pallas_option3d_zero_copy9_persrc_async_a2a_wo_3d_static_fp8(
        q_local_3d,
        wo_fp8_xor_3d,
        wo_scale_2d,
        tp_axis=tp_axis,
        bn=bn,
        stage_sends=stage_sends,
        d_order=d_order,
        opt_wo=opt_wo,
        valid_tokens_1d=valid_tokens_1d,
    )


def pallas_option3d_single_kernel_fused_gate_a2a_wo_3d_static_fp8(
    normed_attn_3d: jax.Array,
    gate_3d: jax.Array,
    wo_fp8_xor_3d: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    d_order: tuple[int, ...] | None = None,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """Wave 7 (`a3_v17`, `#112` B.1): Single-Kernel Fused Gate + Static-FP8 Quant + Staged Push A2A + 3D Wo!

    Passes `normed_attn_3d` and `gate_3d` (`bf16[tp_deg, t_shard, k_local]` in HBM) directly into
    the `Wo` kernel. At `(n_half == 0) & (step == 0) & (j == 0)`:
      1. Loops `d = 1 .. tp_deg - 1` (`peer = me ^ step_xors[d]`): async-DMAs `attn_3d[peer]` and
         `gate_3d[peer]` into a `4 MiB` BF16 VMEM staging tile, computes `clip(sigmoid(gate)*attn*32.0).astype(fp8)`
         into `send_vmem[d]`, and **immediately** starts `make_async_remote_copy` to `peer`'s `recv_vmem[d]`
         after each `~2.3 µs` slice!
      2. Then quantizes `d = 0` (`me`) directly into `recv_vmem[0]`.
    Eliminates the standalone `18.5 / 36.1 µs` prologue kernel launch, eliminates `16 / 32 MiB` HBM
    round-trip of `q_local_3d`, eliminates `%copy.9` (`13.3 / 21.4 µs`), and starts ICI transfers
    after the very first `2.3 µs` slice!
    """
    tp_deg, t_shard, k_local = normed_attn_3d.shape
    _, _, n_dim = wo_fp8_xor_3d.shape
    num_n_total = n_dim // bn
    # At T=16K (t_shard=2048), use num_n_half=4 (num_j=2, acc_vmem=16 MiB) so send_vmem (16 MiB) +
    # recv_vmem (16 MiB) + acc_vmem (16 MiB) + ag_tile_vmem (4 MiB) + out_vmem (4 MiB) = 56 MiB < 64 MiB!
    num_n_half = 2 if t_shard <= 1024 else 4
    num_j = num_n_total // num_n_half
    tm_step = min(1024, t_shard)
    step_xors = tuple(range(tp_deg)) if d_order is None else tuple(d_order)

    def _single_kernel_fused_a3(
        attn_hbm_ref,
        gate_hbm_ref,
        w_xor_ref,
        w_sc_ref,
        out_hbm_ref,
        recv_vmem,
        send_vmem,
        ag_tile_vmem,
        acc_vmem,
        out_vmem,
        rsems,
        ssems,
        local_sems,
        out_sems,
    ):
        n_half = pl.program_id(0)
        step = pl.program_id(1)
        j = pl.program_id(2)
        me = lax.axis_index(tp_axis)

        @pl.when((n_half == 0) & (step == 0) & (j == 0))
        def _():
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
                pl.semaphore_wait(barrier, tp_deg - 1)
            # Quantize and push remote slices d = 1 .. tp_deg - 1 first so ICI starts at t ~ 2.3 us!
            for d in range(1, tp_deg):
                peer = lax.bitwise_xor(me, step_xors[d])
                for m0 in range(0, t_shard, tm_step):
                    pltpu.make_async_copy(
                        attn_hbm_ref.at[pl.ds(peer, 1), pl.ds(m0, tm_step), :],
                        ag_tile_vmem.at[pl.ds(0, 1), :, :],
                        local_sems.at[0],
                    ).start()
                    pltpu.make_async_copy(
                        gate_hbm_ref.at[pl.ds(peer, 1), pl.ds(m0, tm_step), :],
                        ag_tile_vmem.at[pl.ds(1, 1), :, :],
                        local_sems.at[1],
                    ).start()
                    pltpu.make_async_copy(
                        ag_tile_vmem.at[pl.ds(0, 1), :, :],
                        ag_tile_vmem.at[pl.ds(0, 1), :, :],
                        local_sems.at[0],
                    ).wait()
                    pltpu.make_async_copy(
                        ag_tile_vmem.at[pl.ds(1, 1), :, :],
                        ag_tile_vmem.at[pl.ds(1, 1), :, :],
                        local_sems.at[1],
                    ).wait()
                    g_sig = jax.nn.sigmoid(
                        ag_tile_vmem[1, :, :].astype(jnp.float32)
                    ).astype(jnp.bfloat16)
                    gf = (g_sig * ag_tile_vmem[0, :, :]).astype(
                        jnp.float32
                    ) * STATIC_A3_ACT_MULT
                    send_vmem[d, m0 : m0 + tm_step, :] = jnp.clip(
                        gf, -FP8_MAX, FP8_MAX
                    ).astype(jnp.float8_e4m3fn)
                pltpu.make_async_remote_copy(
                    src_ref=send_vmem.at[pl.ds(d, 1)],
                    dst_ref=recv_vmem.at[pl.ds(d, 1)],
                    send_sem=ssems.at[d],
                    recv_sem=rsems.at[d],
                    device_id=(peer,),
                ).start()
            # Now quantize own slice (d = 0, peer = me) directly into recv_vmem[0] while ICI streams!
            for m0 in range(0, t_shard, tm_step):
                pltpu.make_async_copy(
                    attn_hbm_ref.at[pl.ds(me, 1), pl.ds(m0, tm_step), :],
                    ag_tile_vmem.at[pl.ds(0, 1), :, :],
                    local_sems.at[0],
                ).start()
                pltpu.make_async_copy(
                    gate_hbm_ref.at[pl.ds(me, 1), pl.ds(m0, tm_step), :],
                    ag_tile_vmem.at[pl.ds(1, 1), :, :],
                    local_sems.at[1],
                ).start()
                pltpu.make_async_copy(
                    ag_tile_vmem.at[pl.ds(0, 1), :, :],
                    ag_tile_vmem.at[pl.ds(0, 1), :, :],
                    local_sems.at[0],
                ).wait()
                pltpu.make_async_copy(
                    ag_tile_vmem.at[pl.ds(1, 1), :, :],
                    ag_tile_vmem.at[pl.ds(1, 1), :, :],
                    local_sems.at[1],
                ).wait()
                g_sig = jax.nn.sigmoid(
                    ag_tile_vmem[1, :, :].astype(jnp.float32)
                ).astype(jnp.bfloat16)
                gf = (g_sig * ag_tile_vmem[0, :, :]).astype(
                    jnp.float32
                ) * STATIC_A3_ACT_MULT
                recv_vmem[0, m0 : m0 + tm_step, :] = jnp.clip(
                    gf, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

        @pl.when(step == 0)
        def _():
            for m0 in range(0, t_shard, tm_step):
                acc_vmem[j, m0 : m0 + tm_step, :] = jnp.dot(
                    recv_vmem[0, m0 : m0 + tm_step, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )

        @pl.when((n_half == 0) & (step > 0) & (j == 0))
        def _():
            for d in range(1, tp_deg):

                @pl.when(step == d)
                def _():
                    pltpu.make_async_copy(
                        recv_vmem.at[pl.ds(d, 1)],
                        recv_vmem.at[pl.ds(d, 1)],
                        rsems.at[d],
                    ).wait()

        @pl.when((step > 0) & (step < tp_deg - 1))
        def _():
            for m0 in range(0, t_shard, tm_step):
                acc_vmem[j, m0 : m0 + tm_step, :] = acc_vmem[
                    j, m0 : m0 + tm_step, :
                ] + jnp.dot(
                    recv_vmem[step, m0 : m0 + tm_step, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )

        @pl.when(step == tp_deg - 1)
        def _():
            @pl.when((n_half > 0) | (j >= 1))
            def _():
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(0, 1), :, :].reshape(t_shard, bn),
                    out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                    out_sems.at[0],
                ).wait()

            sc_vec = w_sc_ref[...] * STATIC_A3_ACT_SCALE
            for m0 in range(0, t_shard, tm_step):
                acc = acc_vmem[j, m0 : m0 + tm_step, :] + jnp.dot(
                    recv_vmem[step, m0 : m0 + tm_step, :],
                    w_xor_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                )
                out_vmem[0, m0 : m0 + tm_step, :] = (acc * sc_vec).astype(jnp.bfloat16)
            col_idx = n_half * num_j + j
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(0, 1), :, :].reshape(t_shard, bn),
                out_hbm_ref.at[
                    pl.ds(0, t_shard),
                    pl.ds(col_idx * bn, bn),
                ],
                out_sems.at[0],
            ).start()

        @pl.when((n_half == num_n_half - 1) & (step == tp_deg - 1) & (j == num_j - 1))
        def _():
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(0, 1), :, :].reshape(t_shard, bn),
                out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                out_sems.at[0],
            ).wait()
            for d in range(1, tp_deg):
                pltpu.make_async_copy(
                    send_vmem.at[pl.ds(0, 1)],
                    send_vmem.at[pl.ds(0, 1)],
                    ssems.at[d],
                ).wait()

    return pl.pallas_call(
        _single_kernel_fused_a3,
        grid=(num_n_half, tp_deg, num_j),
        in_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(
                (1, k_local, bn),
                lambda n_half, step, j: (step, 0, n_half * num_j + j),
            ),
            pl.BlockSpec((1, bn), lambda n_half, step, j: (0, n_half * num_j + j)),
        ],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((tp_deg, t_shard, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((tp_deg, t_shard, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((2, tm_step, k_local), jnp.bfloat16),
            pltpu.VMEM((num_j, t_shard, bn), jnp.float32),
            pltpu.VMEM((1, t_shard, bn), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            collective_id=7 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(normed_attn_3d, gate_3d, wo_fp8_xor_3d, wo_scale_2d)


def a3_v17_option3d_single_kernel_fused_gate_a2a_wo_3d_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8_xor_3d: jax.Array,
    wo_scale_2d: jax.Array,
    tp_axis: str = "tp",
    bn: int = 1024,
    d_order: tuple[int, ...] | None = None,
) -> jax.Array:
    """Variant A3-v17 (`#112` B.1): Single-Kernel Fused Gate+Quant + Staged Push A2A + 3D Wo."""
    t_total, k_local = normed_attn_out.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    return pallas_option3d_single_kernel_fused_gate_a2a_wo_3d_static_fp8(
        normed_attn_out.reshape(tp_deg, t_shard, k_local),
        gate.reshape(tp_deg, t_shard, k_local),
        wo_fp8_xor_3d,
        wo_scale_2d,
        tp_axis=tp_axis,
        bn=bn,
        d_order=d_order,
    )
