"""Kernel A1 (`pallas_ag_qkvg_qk_rope_kvwrite`) and Bare-Dot Rate Probes for Muse Spark on TPU v7x.

Implements:
  0. `pallas_bare_dot`: Pallas MXU bare-dot rate probe ([M, 8192] @ [8192, N]) across
     bf16×bf16, bf16×fp8, and fp8×fp8 at K-strips {512, 1024, 2048, 8192} and N-subtiles {128, 256, 512, N}.
  1. `pallas_write_kv_pages`: In-place Pallas DMA page writer (`input_output_aliases`) into
     `SEQ_ALONG_LANE` KV cache `(num_pages, H_kv_local * 2, 16, 4, 256)` with zero XLA RMW copies.
  2. `pallas_qkvg_qk_rope_kv`: Core fused Pallas kernel on TPU v7x:
     - Weight-resident `wqkvg_fp8` ([8192, 2304] at TP8 / [8192, 1152] at TP16) via constant BlockSpec
     - Supports `act_dtype="bf16"` (BF16×FP8) and `act_dtype="fp8"` (native FP8×FP8 with row scale)
     - Supports column-strip sub-tiling (`sub_n in {128, 256, None}`) so only `[tm, sub_n]` f32
       accumulators (32 VREGs at tm=256, sub_n=128) are live in registers at once (zero VMEM spills!)
     - Static Python-unrolled K-strips (`bk in {512, 1024, 2048, 8192}`)
     - In-VMEM 128-lane dual-head FP32 RMSNorm (zero <128-lane reshapes) with optional folded Q scale
       (`5.4730064863875 * (64**-0.5) * k_scale * log2(e)`)
     - In-VMEM 128-lane `pltpu.roll` interleaved RoPE (and half-split RoPE comparison)
     - In-VMEM FP8 KV quantization + Pallas DMA page write into `SEQ_ALONG_LANE` KV cache.
  3. Multi-device TP8 / TP16 collective wrappers:
     - `a1_unfused_replicated_bf16`: Unfused XLA baseline
     - `a1_v1_bf16_ag_pallas`: SP input_layernorm -> BF16 All-Gather -> `pallas_qkvg_qk_rope_kv(bf16)`
     - `a1_v2_fp8_ag_pallas`: Algebraic-cancel SP input_layernorm + FP8 row quant -> FP8 All-Gather
       (2x smaller ICI payload) -> `pallas_qkvg_qk_rope_kv(fp8)`
     - `a1_v3_2half_overlap`: 2-Half (`T // 2` x 2) overlapped All-Gather + Pallas Kernel A1 (with
       zero-copy 4D `(TP, 2, half_tok, cols)` output buffer to preserve exact `[0..T-1]` token order!)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .reference import (
    FP8_MAX,
    HEAD_DIM,
    LOG2E,
    QUERY_SCALE,
    SM_SCALE,
    input_layernorm_ref,
)


def pallas_axis_index_1d(tp_axis: str = "model") -> jax.Array:
    """Returns ``int32[1]`` holding ``lax.axis_index(tp_axis)`` inside a Pallas custom call.

    Avoids emitting outer ``stablehlo.partition_id``, which TorchTPU GSPMD rejects with
    ``NotImplementedError: partitionId instruction is not supported for SPMD partitioning``.
    """

    def _axis_idx_kernel(out_ref):
        out_ref[...] = jnp.broadcast_to(
            lax.axis_index(tp_axis).astype(jnp.int32), (8, 128)
        )

    tile = pl.pallas_call(
        _axis_idx_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=(),
            out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"pallas_axis_index_{tp_axis}",
    )()
    return tile[0, 0:1]


# -----------------------------------------------------------------------------
# 0. Bare-Dot Rate Probe (Wave 0/1 MXU Ceiling Calibration)
# -----------------------------------------------------------------------------
def _subtiled_dot_f32(
    x_ref,
    w_ref,
    k_dim: int,
    n0: int,
    tn: int,
    bk: int,
) -> jax.Array:
    """Computes `x_ref[:, :k_dim] @ w_ref[:k_dim, n0:n0+tn]` with static K-strip unrolling."""
    if bk == k_dim and n0 == 0 and tn == w_ref.shape[1] and x_ref.shape[1] == k_dim:
        return jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    acc = jnp.dot(
        x_ref[:, 0:bk],
        w_ref[0:bk, n0 : n0 + tn],
        preferred_element_type=jnp.float32,
    )
    for k0 in range(bk, k_dim, bk):
        acc = acc + jnp.dot(
            x_ref[:, k0 : k0 + bk],
            w_ref[k0 : k0 + bk, n0 : n0 + tn],
            preferred_element_type=jnp.float32,
        )
    return acc


def pallas_bare_dot(
    x: jax.Array,
    w: jax.Array,
    tm: int = 256,
    bk: int = 2048,
    sub_n: int | None = None,
    vmem_limit_mb: int = 60,
) -> jax.Array:
    """Measures bare Pallas MXU throughput for [M, 8192] @ [8192, N] with resident weight."""
    m_dim, k_dim = x.shape
    _, n_dim = w.shape
    tn_step = sub_n or n_dim

    def _dot_kernel(x_ref, w_ref, out_ref):
        if tn_step == n_dim:
            acc = _subtiled_dot_f32(x_ref, w_ref, k_dim, 0, n_dim, bk)
            out_ref[...] = acc.astype(jnp.bfloat16)
        else:
            for n0 in range(0, n_dim, tn_step):
                tn = min(tn_step, n_dim - n0)
                acc = _subtiled_dot_f32(x_ref, w_ref, k_dim, n0, tn, bk)
                out_ref[:, n0 : n0 + tn] = acc.astype(jnp.bfloat16)

    return pl.pallas_call(
        _dot_kernel,
        grid=(m_dim // tm,),
        in_specs=[
            pl.BlockSpec((tm, k_dim), lambda i: (i, 0)),
            pl.BlockSpec((k_dim, n_dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tm, n_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m_dim, n_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(x, w)


# -----------------------------------------------------------------------------
# 1. Zero-RMW Pallas DMA Page Writer for `SEQ_ALONG_LANE` KV Cache
# -----------------------------------------------------------------------------
def pallas_write_kv_pages(
    page_indices: jax.Array,
    pages_sal: jax.Array,
    kv_cache: jax.Array,
) -> jax.Array:
    """In-place Pallas DMA page writer (`kv_cache[page_indices[i]] = pages_sal[i]`).

    Avoids XLA scatter RMW copies on `SEQ_ALONG_LANE` `(num_pages, H_kv_local*2, 16, 4, 256)`.
    """
    num_pages_write = pages_sal.shape[0]
    block = (None, *kv_cache.shape[1:])
    zeros = (0,) * (kv_cache.ndim - 1)

    def _write_page_kernel(ids_ref, pages_ref, kv_in_ref, kv_out_ref):
        del ids_ref, kv_in_ref
        kv_out_ref[...] = pages_ref[...]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(num_pages_write,),
        in_specs=[
            pl.BlockSpec(block, lambda i, ids: (i, *zeros)),
            pl.BlockSpec(block, lambda i, ids: (ids[i], *zeros)),
        ],
        out_specs=pl.BlockSpec(block, lambda i, ids: (ids[i], *zeros)),
    )
    return pl.pallas_call(
        _write_page_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        input_output_aliases={2: 0},
    )(page_indices[:num_pages_write], pages_sal, kv_cache)


def pack_kv_packed_to_seq_along_lane(
    kv_fp8_packed: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_kv_local: int,
    page_size: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Splits kv_fp8_packed [T, 2 * kv_size], packs to SEQ_ALONG_LANE pages, and DMAs via Pallas."""
    t_len = kv_fp8_packed.shape[0]
    kv_size = h_kv_local * HEAD_DIM
    num_pages_used = t_len // page_size

    k_fp8 = kv_fp8_packed[:, :kv_size]
    v_fp8 = kv_fp8_packed[:, kv_size : 2 * kv_size]

    k_pages = k_fp8.reshape(num_pages_used, page_size, h_kv_local, 16, 4)
    v_pages = v_fp8.reshape(num_pages_used, page_size, h_kv_local, 16, 4)
    kv_interleaved = jnp.stack([k_pages, v_pages], axis=3).reshape(
        num_pages_used, page_size, h_kv_local * 2, 16, 4
    )
    pages_sal = jnp.transpose(kv_interleaved, (0, 2, 3, 4, 1))
    kv_cache_out = pallas_write_kv_pages(page_indices, pages_sal, kv_cache)
    return k_fp8, v_fp8, kv_cache_out


# -----------------------------------------------------------------------------
# 2. Algebraic-Cancel Input RMSNorm + FP8 Row Quantization
# -----------------------------------------------------------------------------
def sp_input_layernorm_bf16(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Token-sharded input_layernorm returning bf16[T // TP, 8192]."""
    return input_layernorm_ref(stream_shard, norm_weight, eps=eps)


def sp_input_layernorm_algebraic_fp8(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Algebraic-cancel fused input_layernorm + dynamic per-row FP8 quantization.

    Since `y = x * inv_rms * gain` with scalar `inv_rms > 0` per row:
      `y / (amax(|y|) / 448) == (x * gain) / (amax(|x * gain|) / 448)`
    and the effective row dequant scale is `x_scale = (amax(|x * gain|) / 448) * inv_rms`.

    Returns:
      x_fp8_shard: float8_e4m3fn[T // TP, 8192]
      x_scale_128_shard: float32[T // TP, 128] (row scale broadcast across 128 lanes)
    """
    xf = stream_shard.astype(jnp.float32)
    gain = norm_weight.astype(jnp.float32) + 1.0
    xg = xf * gain[None, :]
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    inv_rms = lax.rsqrt(var + eps)
    amax_xg = jnp.maximum(jnp.max(jnp.abs(xg), axis=-1, keepdims=True), 1e-12)
    sc_xg = amax_xg * (1.0 / FP8_MAX)
    x_fp8 = jnp.clip(xg / sc_xg, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    eff_row_scale = sc_xg * inv_rms
    x_scale_128 = jnp.broadcast_to(eff_row_scale, (stream_shard.shape[0], 128))
    return x_fp8, x_scale_128


# -----------------------------------------------------------------------------
# 3. Core Fused Pallas Kernel A1 (`pallas_qkvg_qk_rope_kv`)
# -----------------------------------------------------------------------------
def _dual_head_rmsnorm_128(
    pair_f32: jax.Array,
    is_lo: jax.Array,
    mask_lo: jax.Array,
    mask_hi: jax.Array,
    scale_factor: float,
    eps: float = 1e-5,
    *,
    fast_6roll: bool = False,
) -> jax.Array:
    """Per-head FP32 RMSNorm on a [tm, 128] slice holding two 64-dim heads without <128 reshape."""
    sq = pair_f32 * pair_f32
    if fast_6roll:
        # 6-step intra-half prefix butterfly: lane 63 holds exact sum(sq[:, 0:64]) and
        # lane 127 holds exact sum(sq[:, 64:128]), halving cross-lane sums and rsqrt ops.
        s = sq
        for sh in (32, 16, 8, 4, 2, 1):
            s = s + pltpu.roll(s, shift=sh, axis=1)
        inv_both = lax.rsqrt(s * (1.0 / 64.0) + eps)
        tm = pair_f32.shape[0]
        inv_rms = jnp.where(
            is_lo,
            jnp.broadcast_to(inv_both[:, 63:64], (tm, 128)),
            jnp.broadcast_to(inv_both[:, 127:128], (tm, 128)),
        )
        return (
            (pair_f32 * (inv_rms * scale_factor))
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    mean_lo = jnp.sum(sq * mask_lo, axis=1, keepdims=True) * (1.0 / 64.0)
    mean_hi = jnp.sum(sq * mask_hi, axis=1, keepdims=True) * (1.0 / 64.0)
    inv_rms = jnp.where(is_lo, lax.rsqrt(mean_lo + eps), lax.rsqrt(mean_hi + eps))
    return (
        (pair_f32 * (inv_rms * scale_factor)).astype(jnp.bfloat16).astype(jnp.float32)
    )


def _rope_128_vreg(
    x_f32: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    lane: jax.Array,
    rope_mode: str = "interleaved_roll",
) -> jax.Array:
    """Applies RoPE in VREGs on [tm, 128] (two 64-dim heads) using pltpu.roll."""
    if rope_mode == "interleaved_roll":
        even = (lane & 1) == 0
        partner = jnp.where(
            even,
            pltpu.roll(x_f32, shift=127, axis=1),
            pltpu.roll(x_f32, shift=1, axis=1),
        )
        return (x_f32 * cos_128 + partner * sin_signed_128).astype(jnp.bfloat16)
    elif rope_mode == "half_split":
        first_half = (lane & 32) == 0
        partner = jnp.where(
            first_half,
            pltpu.roll(x_f32, shift=96, axis=1),
            pltpu.roll(x_f32, shift=32, axis=1),
        )
        return (x_f32 * cos_128 + partner * sin_signed_128).astype(jnp.bfloat16)
    else:
        raise ValueError(f"Unsupported rope_mode={rope_mode}")


def _rpa_q_word(q_rot: jax.Array, is_lo: jax.Array, mask: jax.Array) -> jax.Array:
    """[tm, 128] bf16 pair (heads 2s, 2s+1) -> u32 words of RPA v3's packed q rows.

    RPA's bf16 q ``[1, T, Hq/2, 2, 128]`` (tile (2,128)(2,1)) holds expanded heads 2s and
    2s+1 in the low and high halves of one 32-bit word per lane. KV-packed heads keep
    their 64 lanes in their kv head's half (mask) and are zero elsewhere, bit for bit
    as attention_pallas_rpa_base._pack_qkv_heads' one-hot multiply (q_h * 0 keeps its sign).
    """
    pair = q_rot.astype(jnp.float32)
    rolled = pltpu.roll(pair, shift=64, axis=1)
    even = jnp.where(is_lo, pair, rolled)  # head 2s in both halves
    odd = jnp.where(is_lo, rolled, pair)
    lo = pltpu.bitcast(even * mask, jnp.uint32)
    hi = pltpu.bitcast(odd * mask, jnp.uint32)
    return lax.shift_right_logical(lo, jnp.uint32(16)) | (hi & jnp.uint32(0xFFFF0000))


def _f8_code_u32(x_f8: jax.Array) -> jax.Array:
    """e4m3fn values -> their 8-bit codes as u32, with 32-bit ops only (finite inputs)."""
    x = x_f8.astype(jnp.float32)
    bits = pltpu.bitcast(x, jnp.uint32)
    mag = bits & jnp.uint32(0x7FFFFFFF)
    sign = lax.shift_right_logical(bits, jnp.uint32(31)) << 7
    # Normal: f32 exponent rebias 127 -> 7 and the top 3 mantissa bits.
    normal = lax.shift_right_logical(mag, jnp.uint32(20)) - jnp.uint32(120 << 3)
    # Subnormal (|x| < 2**-6): the code is |x| / 2**-9.
    sub = (jnp.abs(x) * 512.0).astype(jnp.int32).astype(jnp.uint32)
    return sign | jnp.where(mag >= jnp.uint32(0x3C800000), normal, sub)


def pallas_qkvg_qk_rope_kv(
    x_full: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    x_scale_128: jax.Array | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 2048,
    sub_n: int | None = None,
    vmem_limit_mb: int = 60,
    kernel_name: str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused Pallas kernel for wqkvg GEMM + Q/K RMSNorm + Interleaved RoPE + FP8 KV quant.

    When `sub_n` is set (e.g. `128` or `256`), dots and epilogues execute strip-by-strip
    so only `[tm, sub_n]` f32 accumulators (`32 VREGs` at `tm=256, sub_n=128`) are live at once.
    """
    t_len, k_dim = x_full.shape
    _, n_dim = wqkvg_fp8.shape
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM
    num_q_slices = q_size // 128
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    use_fp8_act = x_full.dtype == jnp.float8_e4m3fn

    def _run_body(
        x_ref,
        x_sc_ref,
        w_ref,
        w_sc_ref,
        cos_ref,
        sin_ref,
        q_out_ref,
        gate_out_ref,
        kv_out_ref,
    ):
        row_sc = x_sc_ref[:, :1] if x_sc_ref is not None else None
        lane = lax.broadcasted_iota(jnp.int32, (tm, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[...]
        sin_tile = sin_ref[...]

        if sub_n is not None and sub_n in (128, 256):
            # Strip-fused mode: keep live f32 accumulator <= 32-64 VREGs!
            # 1. Q strips (0 .. q_size)
            for c0 in range(0, q_size, sub_n):
                acc_strip = _subtiled_dot_f32(x_ref, w_ref, k_dim, c0, sub_n, bk)
                sc_strip = w_sc_ref[:, c0 : c0 + sub_n]
                acc_strip = (
                    acc_strip * (row_sc * sc_strip)
                    if row_sc is not None
                    else acc_strip * sc_strip
                )
                for s_off in range(0, sub_n, 128):
                    q_pair = acc_strip[:, s_off : s_off + 128]
                    q_norm = _dual_head_rmsnorm_128(
                        q_pair, is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                    )
                    q_rot = _rope_128_vreg(
                        q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                    )
                    q_out_ref[:, c0 + s_off : c0 + s_off + 128] = q_rot

            # 2. Gate strips (q_size .. 2 * q_size)
            for c0 in range(0, q_size, sub_n):
                w_c0 = q_size + c0
                acc_strip = _subtiled_dot_f32(x_ref, w_ref, k_dim, w_c0, sub_n, bk)
                sc_strip = w_sc_ref[:, w_c0 : w_c0 + sub_n]
                acc_strip = (
                    acc_strip * (row_sc * sc_strip)
                    if row_sc is not None
                    else acc_strip * sc_strip
                )
                gate_out_ref[:, c0 : c0 + sub_n] = acc_strip.astype(jnp.bfloat16)

            # 3. K and V strips (2 * q_size .. 2 * q_size + 2 * kv_size)
            kv_base = 2 * q_size
            if h_kv_local == 2:
                k_pair = _subtiled_dot_f32(x_ref, w_ref, k_dim, kv_base, 128, bk)
                sc_k = w_sc_ref[:, kv_base : kv_base + 128]
                k_pair = (
                    k_pair * (row_sc * sc_k) if row_sc is not None else k_pair * sc_k
                )
                k_norm = _dual_head_rmsnorm_128(
                    k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                kv_out_ref[:, 0:128] = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

                v_pair = _subtiled_dot_f32(x_ref, w_ref, k_dim, kv_base + 128, 128, bk)
                sc_v = w_sc_ref[:, kv_base + 128 : kv_base + 256]
                v_pair = (
                    v_pair * (row_sc * sc_v) if row_sc is not None else v_pair * sc_v
                ).astype(jnp.bfloat16)
                kv_out_ref[:, 128:256] = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
            else:
                kv_pair = _subtiled_dot_f32(x_ref, w_ref, k_dim, kv_base, 128, bk)
                sc_kv = w_sc_ref[:, kv_base : kv_base + 128]
                kv_pair = (
                    kv_pair * (row_sc * sc_kv)
                    if row_sc is not None
                    else kv_pair * sc_kv
                )
                sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                cos_kv = jnp.where(is_lo, cos_tile, 1.0)
                sin_kv = jnp.where(is_lo, sin_tile, 0.0)
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                kv_out_ref[:, 0:128] = jnp.clip(
                    kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
        else:
            # Full-N dot mode
            acc = _subtiled_dot_f32(x_ref, w_ref, k_dim, 0, n_dim, bk)
            acc = (
                acc * (row_sc * w_sc_ref[...])
                if row_sc is not None
                else acc * w_sc_ref[...]
            )
            gate_out_ref[...] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)

            for s in range(num_q_slices):
                c0 = s * 128
                c1 = c0 + 128
                q_pair = acc[:, c0:c1]
                q_norm = _dual_head_rmsnorm_128(
                    q_pair, is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                )
                q_rot = _rope_128_vreg(
                    q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                q_out_ref[:, c0:c1] = q_rot

            if h_kv_local == 2:
                k_pair = acc[:, 2 * q_size : 2 * q_size + 128]
                v_pair = acc[:, 2 * q_size + 128 : 2 * q_size + 256].astype(
                    jnp.bfloat16
                )
                k_norm = _dual_head_rmsnorm_128(
                    k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                kv_out_ref[:, 0:128] = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                kv_out_ref[:, 128:256] = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
            else:
                kv_pair = acc[:, 2 * q_size : 2 * q_size + 128]
                sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                cos_kv = jnp.where(is_lo, cos_tile, 1.0)
                sin_kv = jnp.where(is_lo, sin_tile, 0.0)
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                kv_out_ref[:, 0:128] = jnp.clip(
                    kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

    if use_fp8_act:
        assert x_scale_128 is not None
        return pl.pallas_call(
            _run_body,
            name=kernel_name,
            grid=(t_len // tm,),
            in_specs=[
                pl.BlockSpec((tm, k_dim), lambda i: (i, 0)),
                pl.BlockSpec((tm, 128), lambda i: (i, 0)),
                pl.BlockSpec((k_dim, n_dim), lambda i: (0, 0)),
                pl.BlockSpec((1, n_dim), lambda i: (0, 0)),
                pl.BlockSpec((tm, 128), lambda i: (i, 0)),
                pl.BlockSpec((tm, 128), lambda i: (i, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tm, q_size), lambda i: (i, 0)),
                pl.BlockSpec((tm, q_size), lambda i: (i, 0)),
                pl.BlockSpec((tm, 2 * kv_size), lambda i: (i, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((t_len, q_size), jnp.bfloat16),
                jax.ShapeDtypeStruct((t_len, q_size), jnp.bfloat16),
                jax.ShapeDtypeStruct((t_len, 2 * kv_size), jnp.float8_e4m3fn),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
        )(x_full, x_scale_128, wqkvg_fp8, wqkvg_scale_2d, cos_128, sin_signed_128)
    else:

        def _run_bf16(
            x_ref,
            w_ref,
            w_sc_ref,
            cos_ref,
            sin_ref,
            q_out_ref,
            gate_out_ref,
            kv_out_ref,
        ):
            _run_body(
                x_ref,
                None,
                w_ref,
                w_sc_ref,
                cos_ref,
                sin_ref,
                q_out_ref,
                gate_out_ref,
                kv_out_ref,
            )

        return pl.pallas_call(
            _run_bf16,
            name=kernel_name,
            grid=(t_len // tm,),
            in_specs=[
                pl.BlockSpec((tm, k_dim), lambda i: (i, 0)),
                pl.BlockSpec((k_dim, n_dim), lambda i: (0, 0)),
                pl.BlockSpec((1, n_dim), lambda i: (0, 0)),
                pl.BlockSpec((tm, 128), lambda i: (i, 0)),
                pl.BlockSpec((tm, 128), lambda i: (i, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tm, q_size), lambda i: (i, 0)),
                pl.BlockSpec((tm, q_size), lambda i: (i, 0)),
                pl.BlockSpec((tm, 2 * kv_size), lambda i: (i, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((t_len, q_size), jnp.bfloat16),
                jax.ShapeDtypeStruct((t_len, q_size), jnp.bfloat16),
                jax.ShapeDtypeStruct((t_len, 2 * kv_size), jnp.float8_e4m3fn),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
        )(x_full, wqkvg_fp8, wqkvg_scale_2d, cos_128, sin_signed_128)


# -----------------------------------------------------------------------------
# 4. End-to-End Single-Rank & Multi-Rank TP Wrappers for Kernel A1
# -----------------------------------------------------------------------------
def a1_v1_bf16_ag_pallas(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str | None = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 2048,
    sub_n: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v1: Token-sharded input_layernorm -> BF16 All-Gather -> Pallas BF16×FP8 A1."""
    x_shard_bf16 = sp_input_layernorm_bf16(stream_shard, norm_weight)
    if tp_axis is not None:
        x_full = lax.all_gather(x_shard_bf16, axis_name=tp_axis, axis=0, tiled=True)
    else:
        x_full = x_shard_bf16

    q_out, gate_out, kv_fp8_packed = pallas_qkvg_qk_rope_kv(
        x_full,
        wqkvg_fp8,
        wqkvg_scale_2d,
        cos_128,
        sin_signed_128,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
        bk=bk,
        sub_n=sub_n,
    )
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_fp8_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


def a1_v2_fp8_ag_pallas(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str | None = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 2048,
    sub_n: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v2: Algebraic-cancel FP8 quant on [T//TP, 8192] -> FP8 All-Gather -> Pallas FP8×FP8 A1."""
    x_fp8_shard, x_sc_shard = sp_input_layernorm_algebraic_fp8(
        stream_shard, norm_weight
    )
    if tp_axis is not None:
        x_fp8_full = lax.all_gather(x_fp8_shard, axis_name=tp_axis, axis=0, tiled=True)
        x_sc_full = lax.all_gather(x_sc_shard, axis_name=tp_axis, axis=0, tiled=True)
    else:
        x_fp8_full = x_fp8_shard
        x_sc_full = x_sc_shard

    q_out, gate_out, kv_fp8_packed = pallas_qkvg_qk_rope_kv(
        x_fp8_full,
        wqkvg_fp8,
        wqkvg_scale_2d,
        cos_128,
        sin_signed_128,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        x_scale_128=x_sc_full,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
        bk=bk,
        sub_n=sub_n,
    )
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_fp8_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


def a1_v3_2half_overlap(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "tp",
    use_fp8_act: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 2048,
    sub_n: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v3/v4: 2-Half Overlapped All-Gather + Pallas Kernel A1."""
    t_shard = stream_shard.shape[0]
    h_sub = t_shard // 2
    t_total = cos_128.shape[0]
    t_half = t_total // 2
    tp_deg = t_total // t_shard

    cos_r = cos_128.reshape(tp_deg, 2, h_sub, 128)
    sin_r = sin_signed_128.reshape(tp_deg, 2, h_sub, 128)
    cos0 = cos_r[:, 0, :, :].reshape(t_half, 128)
    sin0 = sin_r[:, 0, :, :].reshape(t_half, 128)
    cos1 = cos_r[:, 1, :, :].reshape(t_half, 128)
    sin1 = sin_r[:, 1, :, :].reshape(t_half, 128)

    if not use_fp8_act:
        x_shard = sp_input_layernorm_bf16(stream_shard, norm_weight)
        ag0 = lax.all_gather(x_shard[:h_sub], axis_name=tp_axis, axis=0, tiled=True)
        x_h1_b, ag0 = lax.optimization_barrier((x_shard[h_sub:], ag0))
        ag1 = lax.all_gather(x_h1_b, axis_name=tp_axis, axis=0, tiled=True)
        q0, g0, kv0 = pallas_qkvg_qk_rope_kv(
            ag0,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos0,
            sin0,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )
        ag1, q0, g0, kv0 = lax.optimization_barrier((ag1, q0, g0, kv0))
        q1, g1, kv1 = pallas_qkvg_qk_rope_kv(
            ag1,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos1,
            sin1,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )
    else:
        x_fp8_s, x_sc_s = sp_input_layernorm_algebraic_fp8(stream_shard, norm_weight)
        ag_x0 = lax.all_gather(x_fp8_s[:h_sub], axis_name=tp_axis, axis=0, tiled=True)
        ag_s0 = lax.all_gather(x_sc_s[:h_sub], axis_name=tp_axis, axis=0, tiled=True)
        x_h1_b, s_h1_b, ag_x0, ag_s0 = lax.optimization_barrier(
            (x_fp8_s[h_sub:], x_sc_s[h_sub:], ag_x0, ag_s0)
        )
        ag_x1 = lax.all_gather(x_h1_b, axis_name=tp_axis, axis=0, tiled=True)
        ag_s1 = lax.all_gather(s_h1_b, axis_name=tp_axis, axis=0, tiled=True)
        q0, g0, kv0 = pallas_qkvg_qk_rope_kv(
            ag_x0,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos0,
            sin0,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            x_scale_128=ag_s0,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )
        ag_x1, ag_s1, q0, g0, kv0 = lax.optimization_barrier(
            (ag_x1, ag_s1, q0, g0, kv0)
        )
        q1, g1, kv1 = pallas_qkvg_qk_rope_kv(
            ag_x1,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos1,
            sin1,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            x_scale_128=ag_s1,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )

    q_out = jnp.stack(
        [q0.reshape(tp_deg, h_sub, -1), q1.reshape(tp_deg, h_sub, -1)], axis=1
    ).reshape(t_total, -1)
    gate_out = jnp.stack(
        [g0.reshape(tp_deg, h_sub, -1), g1.reshape(tp_deg, h_sub, -1)], axis=1
    ).reshape(t_total, -1)
    kv_packed = jnp.stack(
        [kv0.reshape(tp_deg, h_sub, -1), kv1.reshape(tp_deg, h_sub, -1)], axis=1
    ).reshape(t_total, -1)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


def pallas_qkvg_qk_rope_kv_into_half(
    x_half_3d: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_4d: jax.Array,
    sin_4d: jax.Array,
    q_4d: jax.Array,
    gate_4d: jax.Array,
    kv_4d: jax.Array,
    half_idx: int,
    h_q_local: int,
    h_kv_local: int,
    x_scale_3d: jax.Array | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 8192,
    sub_n: int | None = None,
    vmem_limit_mb: int = 60,
    rpa_layout: bool = False,
    row_block_off: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Writes Half `half_idx` (0 or 1) directly into aliased 4D `(TP, 2, h_sub, cols)` outputs.

    Eliminates the post-A1 `jnp.stack` copy: `q_4d.reshape(T, q_size)` is already in exact
    contiguous `[0 .. T-1]` token order!

    `rpa_layout`: q_4d is bf16 `(TP, 2, h_sub, Hq/2, 2, 128)` and kv_4d fp8 `(TP, 2, h_sub, 4, 128)`,
    RPA v3's packed-KV-head q and merged kv rows (SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT).

    `row_block_off` (bf16 x): the part's first row in the outputs and cos / sin, in `tm` blocks
    past `half_idx` (uneven parts write into one `(TP, 1, T / TP, cols)` buffer).
    """
    tp_deg, h_sub, x_cols = x_half_3d.shape
    k_dim, n_dim = wqkvg_fp8.shape
    has_packed_hilo = x_cols == k_dim + 128
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM
    num_q_slices = q_size // 128
    tm_eff = min(tm, h_sub)
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale

    def _body_4d(
        x_ref,
        x_sc_ref,
        w_ref,
        w_sc_ref,
        cos_ref,
        sin_ref,
        q_in_ref,
        gate_in_ref,
        kv_in_ref,
        q_out_ref,
        gate_out_ref,
        kv_out_ref,
    ):
        del q_in_ref, gate_in_ref, kv_in_ref
        if x_sc_ref is not None:
            row_sc = x_sc_ref[:, :1]
        elif has_packed_hilo:
            row_sc = x_ref[:, k_dim : k_dim + 64].astype(jnp.float32)[:, :1] * (
                1.0 / 8192.0
            ) + x_ref[:, k_dim + 64 : k_dim + 128].astype(jnp.float32)[:, :1] * (
                1.0 / 524288.0
            )
        else:
            row_sc = None
        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[...]
        sin_tile = sin_ref[...]

        if sub_n is not None:
            for c0 in range(0, q_size, sub_n):
                acc_strip = _subtiled_dot_f32(x_ref, w_ref, k_dim, c0, sub_n, bk)
                sc_strip = w_sc_ref[:, c0 : c0 + sub_n]
                acc_strip = (
                    acc_strip * (row_sc * sc_strip)
                    if row_sc is not None
                    else acc_strip * sc_strip
                )
                for s_off in range(0, sub_n, 128):
                    q_pair = acc_strip[:, s_off : s_off + 128]
                    q_norm = _dual_head_rmsnorm_128(
                        q_pair, is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                    )
                    q_rot = _rope_128_vreg(
                        q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                    )
                    q_out_ref[:, c0 + s_off : c0 + s_off + 128] = q_rot

            for c0 in range(0, q_size, sub_n):
                w_c0 = q_size + c0
                acc_strip = _subtiled_dot_f32(x_ref, w_ref, k_dim, w_c0, sub_n, bk)
                sc_strip = w_sc_ref[:, w_c0 : w_c0 + sub_n]
                acc_strip = (
                    acc_strip * (row_sc * sc_strip)
                    if row_sc is not None
                    else acc_strip * sc_strip
                )
                gate_out_ref[:, c0 : c0 + sub_n] = acc_strip.astype(jnp.bfloat16)

            kv_base = 2 * q_size
            if h_kv_local == 2:
                k_pair = _subtiled_dot_f32(x_ref, w_ref, k_dim, kv_base, 128, bk)
                sc_k = w_sc_ref[:, kv_base : kv_base + 128]
                k_pair = (
                    k_pair * (row_sc * sc_k) if row_sc is not None else k_pair * sc_k
                )
                k_norm = _dual_head_rmsnorm_128(
                    k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                kv_out_ref[:, 0:128] = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

                v_pair = _subtiled_dot_f32(x_ref, w_ref, k_dim, kv_base + 128, 128, bk)
                sc_v = w_sc_ref[:, kv_base + 128 : kv_base + 256]
                v_pair = (
                    v_pair * (row_sc * sc_v) if row_sc is not None else v_pair * sc_v
                ).astype(jnp.bfloat16)
                kv_out_ref[:, 128:256] = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
            else:
                kv_pair = _subtiled_dot_f32(x_ref, w_ref, k_dim, kv_base, 128, bk)
                sc_kv = w_sc_ref[:, kv_base : kv_base + 128]
                kv_pair = (
                    kv_pair * (row_sc * sc_kv)
                    if row_sc is not None
                    else kv_pair * sc_kv
                )
                sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                cos_kv = jnp.where(is_lo, cos_tile, 1.0)
                sin_kv = jnp.where(is_lo, sin_tile, 0.0)
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                kv_out_ref[:, 0:128] = jnp.clip(
                    kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
        else:
            acc = _subtiled_dot_f32(x_ref, w_ref, k_dim, 0, n_dim, bk)
            acc = (
                acc * (row_sc * w_sc_ref[...])
                if row_sc is not None
                else acc * w_sc_ref[...]
            )
            gate_out_ref[...] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)
            if rpa_layout:
                q_words = (
                    q_out_ref.bitcast(jnp.uint32)
                    .at[0, 0]
                    .reshape(tm_eff * num_q_slices, 128)
                )

            for s in range(num_q_slices):
                c0 = s * 128
                c1 = c0 + 128
                q_pair = acc[:, c0:c1]
                q_norm = _dual_head_rmsnorm_128(
                    q_pair, is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                )
                q_rot = _rope_128_vreg(
                    q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                if rpa_layout:
                    kv_head0 = s < num_q_slices // 2
                    q_words[pl.ds(s, tm_eff, stride=num_q_slices)] = _rpa_q_word(
                        q_rot, is_lo, mask_lo if kv_head0 else mask_hi
                    )
                else:
                    q_out_ref[:, c0:c1] = q_rot

            if h_kv_local == 2:
                k_pair = acc[:, 2 * q_size : 2 * q_size + 128]
                v_pair = acc[:, 2 * q_size + 128 : 2 * q_size + 256].astype(
                    jnp.bfloat16
                )
                k_norm = _dual_head_rmsnorm_128(
                    k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                k8 = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                v8 = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                if rpa_layout:
                    # One u32 per lane holds kv rows (k, v, 0, 0) of the (4,128)(4,1) fp8 tile.
                    kv_out_ref.bitcast(jnp.uint32).at[0, 0].reshape(tm_eff, 128)[
                        ...
                    ] = _f8_code_u32(k8) | (_f8_code_u32(v8) << 8)
                else:
                    kv_out_ref[:, 0:128] = k8
                    kv_out_ref[:, 128:256] = v8
            else:
                kv_pair = acc[:, 2 * q_size : 2 * q_size + 128]
                sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                cos_kv = jnp.where(is_lo, cos_tile, 1.0)
                sin_kv = jnp.where(is_lo, sin_tile, 0.0)
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                kv_out_ref[:, 0:128] = jnp.clip(
                    kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

    if x_scale_3d is not None:
        assert not rpa_layout and row_block_off == 0
        return pl.pallas_call(
            _body_4d,
            grid=(tp_deg, h_sub // tm_eff),
            in_specs=[
                pl.BlockSpec((None, tm_eff, k_dim), lambda r, m: (r, m, 0)),
                pl.BlockSpec((None, tm_eff, 128), lambda r, m: (r, m, 0)),
                pl.BlockSpec((k_dim, n_dim), lambda r, m: (0, 0)),
                pl.BlockSpec((1, n_dim), lambda r, m: (0, 0)),
                pl.BlockSpec(
                    (None, None, tm_eff, 128), lambda r, m: (r, half_idx, m, 0)
                ),
                pl.BlockSpec(
                    (None, None, tm_eff, 128), lambda r, m: (r, half_idx, m, 0)
                ),
                pl.BlockSpec(
                    (None, None, tm_eff, q_size), lambda r, m: (r, half_idx, m, 0)
                ),
                pl.BlockSpec(
                    (None, None, tm_eff, q_size), lambda r, m: (r, half_idx, m, 0)
                ),
                pl.BlockSpec(
                    (None, None, tm_eff, 2 * kv_size),
                    lambda r, m: (r, half_idx, m, 0),
                ),
            ],
            out_specs=[
                pl.BlockSpec(
                    (None, None, tm_eff, q_size), lambda r, m: (r, half_idx, m, 0)
                ),
                pl.BlockSpec(
                    (None, None, tm_eff, q_size), lambda r, m: (r, half_idx, m, 0)
                ),
                pl.BlockSpec(
                    (None, None, tm_eff, 2 * kv_size),
                    lambda r, m: (r, half_idx, m, 0),
                ),
            ],
            out_shape=[
                jax.ShapeDtypeStruct(q_4d.shape, jnp.bfloat16),
                jax.ShapeDtypeStruct(gate_4d.shape, jnp.bfloat16),
                jax.ShapeDtypeStruct(kv_4d.shape, jnp.float8_e4m3fn),
            ],
            input_output_aliases={6: 0, 7: 1, 8: 2},
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
        )(
            x_half_3d,
            x_scale_3d,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos_4d,
            sin_4d,
            q_4d,
            gate_4d,
            kv_4d,
        )
    else:

        def _body_4d_bf16(
            x_ref,
            w_ref,
            w_sc_ref,
            cos_ref,
            sin_ref,
            q_in_ref,
            gate_in_ref,
            kv_in_ref,
            q_out_ref,
            gate_out_ref,
            kv_out_ref,
        ):
            _body_4d(
                x_ref,
                None,
                w_ref,
                w_sc_ref,
                cos_ref,
                sin_ref,
                q_in_ref,
                gate_in_ref,
                kv_in_ref,
                q_out_ref,
                gate_out_ref,
                kv_out_ref,
            )

        def _row_block(m):
            # Unshifted when 0, so the default index maps trace unchanged.
            return m + row_block_off if row_block_off else m

        if rpa_layout:
            assert sub_n is None and h_kv_local == 2 and h_q_local % 4 == 0
            # Unsqueezed blocks: Mosaic's memref bitcast rejects squeezed leading dims.
            q_spec = pl.BlockSpec(
                (1, 1, tm_eff, *q_4d.shape[3:]),
                lambda r, m: (r, half_idx, _row_block(m), 0, 0, 0),
            )
            kv_spec = pl.BlockSpec(
                (1, 1, tm_eff, *kv_4d.shape[3:]),
                lambda r, m: (r, half_idx, _row_block(m), 0, 0),
            )
        else:
            q_spec = pl.BlockSpec(
                (None, None, tm_eff, q_size),
                lambda r, m: (r, half_idx, _row_block(m), 0),
            )
            kv_spec = pl.BlockSpec(
                (None, None, tm_eff, 2 * kv_size),
                lambda r, m: (r, half_idx, _row_block(m), 0),
            )

        return pl.pallas_call(
            _body_4d_bf16,
            grid=(tp_deg, h_sub // tm_eff),
            in_specs=[
                pl.BlockSpec((None, tm_eff, x_cols), lambda r, m: (r, m, 0)),
                pl.BlockSpec((k_dim, n_dim), lambda r, m: (0, 0)),
                pl.BlockSpec((1, n_dim), lambda r, m: (0, 0)),
                pl.BlockSpec(
                    (None, None, tm_eff, 128),
                    lambda r, m: (r, half_idx, _row_block(m), 0),
                ),
                pl.BlockSpec(
                    (None, None, tm_eff, 128),
                    lambda r, m: (r, half_idx, _row_block(m), 0),
                ),
                q_spec,
                pl.BlockSpec(
                    (None, None, tm_eff, q_size),
                    lambda r, m: (r, half_idx, _row_block(m), 0),
                ),
                kv_spec,
            ],
            out_specs=[
                q_spec,
                pl.BlockSpec(
                    (None, None, tm_eff, q_size),
                    lambda r, m: (r, half_idx, _row_block(m), 0),
                ),
                kv_spec,
            ],
            out_shape=[
                jax.ShapeDtypeStruct(q_4d.shape, jnp.bfloat16),
                jax.ShapeDtypeStruct(gate_4d.shape, jnp.bfloat16),
                jax.ShapeDtypeStruct(kv_4d.shape, jnp.float8_e4m3fn),
            ],
            input_output_aliases={5: 0, 6: 1, 7: 2},
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
        )(
            x_half_3d,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos_4d,
            sin_4d,
            q_4d,
            gate_4d,
            kv_4d,
        )


def a1_v5_2half_zerocopy_4d(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "tp",
    use_fp8_act: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 8192,
    sub_n: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v5: Zero-Copy 4D `(TP, 2, h_sub, cols)` 2-Half Overlapped AG + Pallas A1."""
    t_shard = stream_shard.shape[0]
    h_sub = t_shard // 2
    t_total = cos_128.shape[0]
    tp_deg = t_total // t_shard
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    cos_4d = cos_128.reshape(tp_deg, 2, h_sub, 128)
    sin_4d = sin_signed_128.reshape(tp_deg, 2, h_sub, 128)
    q_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    gate_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    kv_4d = jnp.empty((tp_deg, 2, h_sub, 2 * kv_size), dtype=jnp.float8_e4m3fn)

    if not use_fp8_act:
        x_shard = sp_input_layernorm_bf16(stream_shard, norm_weight)
        ag0 = lax.all_gather(x_shard[:h_sub], axis_name=tp_axis, axis=0, tiled=False)
        x_h1_b, ag0 = lax.optimization_barrier((x_shard[h_sub:], ag0))
        ag1 = lax.all_gather(x_h1_b, axis_name=tp_axis, axis=0, tiled=False)
        q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
            ag0,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos_4d,
            sin_4d,
            q_4d,
            gate_4d,
            kv_4d,
            half_idx=0,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )
        ag1, q_4d, gate_4d, kv_4d = lax.optimization_barrier(
            (ag1, q_4d, gate_4d, kv_4d)
        )
        q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
            ag1,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos_4d,
            sin_4d,
            q_4d,
            gate_4d,
            kv_4d,
            half_idx=1,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )
    else:
        x_fp8_s, x_sc_s = sp_input_layernorm_algebraic_fp8(stream_shard, norm_weight)
        ag_x0 = lax.all_gather(x_fp8_s[:h_sub], axis_name=tp_axis, axis=0, tiled=False)
        ag_s0 = lax.all_gather(x_sc_s[:h_sub], axis_name=tp_axis, axis=0, tiled=False)
        x_h1_b, s_h1_b, ag_x0, ag_s0 = lax.optimization_barrier(
            (x_fp8_s[h_sub:], x_sc_s[h_sub:], ag_x0, ag_s0)
        )
        ag_x1 = lax.all_gather(x_h1_b, axis_name=tp_axis, axis=0, tiled=False)
        ag_s1 = lax.all_gather(s_h1_b, axis_name=tp_axis, axis=0, tiled=False)
        q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
            ag_x0,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos_4d,
            sin_4d,
            q_4d,
            gate_4d,
            kv_4d,
            half_idx=0,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            x_scale_3d=ag_s0,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )
        ag_x1, ag_s1, q_4d, gate_4d, kv_4d = lax.optimization_barrier(
            (ag_x1, ag_s1, q_4d, gate_4d, kv_4d)
        )
        q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
            ag_x1,
            wqkvg_fp8,
            wqkvg_scale_2d,
            cos_4d,
            sin_4d,
            q_4d,
            gate_4d,
            kv_4d,
            half_idx=1,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            x_scale_3d=ag_s1,
            k_scale=k_scale,
            v_scale=v_scale,
            fold_q_scale=fold_q_scale,
            rope_mode=rope_mode,
            tm=tm,
            bk=bk,
            sub_n=sub_n,
        )

    q_out = q_4d.reshape(t_total, q_size)
    gate_out = gate_4d.reshape(t_total, q_size)
    kv_packed = kv_4d.reshape(t_total, 2 * kv_size)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


# -----------------------------------------------------------------------------
# 5. Wave 5: Static-Scale Single-AG 2-Half A1 (`a1_v6`) and Sequence-Parallel
#    Hybrid `FP8 Q/G + BF16 K/V` Local `wqkvg` + Pallas Async All-to-All (`a1_v7_sp`)
# -----------------------------------------------------------------------------
STATIC_A1_ACT_MULT = 32.0
STATIC_A1_ACT_SCALE = 1.0 / STATIC_A1_ACT_MULT


def sp_input_layernorm_static_fp8(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """One-pass static-scale FP8 input_layernorm (`fp8_e4m3fn[T // TP, 8192]`, zero scale tensor!)."""
    xf = stream_shard.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    inv_rms = lax.rsqrt(var + eps)
    gain = (norm_weight.astype(jnp.float32) + 1.0) * STATIC_A1_ACT_MULT
    y_scaled = xf * inv_rms * gain[None, :]
    return jnp.clip(y_scaled, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)


def sp_input_layernorm_packed_hilo_fp8(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Algebraic-cancel FP8 input_layernorm with `Hi+Lo` E4M3 row scale packed into columns `[8192:8320]`."""
    xf = stream_shard.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    inv_rms = lax.rsqrt(var + eps)
    gain = norm_weight.astype(jnp.float32) + 1.0
    scaled_unnorm = xf * gain[None, :]
    amax = jnp.maximum(jnp.max(jnp.abs(scaled_unnorm), axis=-1, keepdims=True), 1e-12)
    inv_amax = FP8_MAX / amax
    x_fp8 = jnp.clip(scaled_unnorm * inv_amax, -FP8_MAX, FP8_MAX).astype(
        jnp.float8_e4m3fn
    )
    eff_row_scale = (inv_rms * amax) * (1.0 / FP8_MAX)
    s_scaled = jnp.minimum(eff_row_scale * 8192.0, FP8_MAX)
    s_hi = s_scaled.astype(jnp.float8_e4m3fn)
    s_rem = jnp.clip((s_scaled - s_hi.astype(jnp.float32)) * 64.0, -FP8_MAX, FP8_MAX)
    s_lo = s_rem.astype(jnp.float8_e4m3fn)
    t_shard = stream_shard.shape[0]
    return jnp.concatenate(
        [
            x_fp8,
            jnp.broadcast_to(s_hi, (t_shard, 64)),
            jnp.broadcast_to(s_lo, (t_shard, 64)),
        ],
        axis=1,
    )


def a1_v5b_2half_packed_hilo_fp8_single_ag(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 8192,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v5b (`@advisor-argon-high` §4.1): Packed `Hi+Lo` FP8 Scale (`[T//TP, 8320]`) + Single AG per Half.

    Eliminates `ag_s0` and `ag_s1` (`8-16 MiB` float32 scale All-Gathers) while preserving dynamic
    per-row algebraic-cancel FP8 scaling (`0.18%` scale precision, 10 bits of mantissa).
    """
    t_shard = stream_shard.shape[0]
    h_sub = t_shard // 2
    t_total = cos_128.shape[0]
    tp_deg = t_total // t_shard
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    cos_4d = cos_128.reshape(tp_deg, 2, h_sub, 128)
    sin_4d = sin_signed_128.reshape(tp_deg, 2, h_sub, 128)
    q_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    gate_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    kv_4d = jnp.empty((tp_deg, 2, h_sub, 2 * kv_size), dtype=jnp.float8_e4m3fn)

    x_packed = sp_input_layernorm_packed_hilo_fp8(stream_shard, norm_weight)
    ag_p0 = lax.all_gather(x_packed[:h_sub], axis_name=tp_axis, axis=0, tiled=False)
    x_h1_b, ag_p0 = lax.optimization_barrier((x_packed[h_sub:], ag_p0))
    ag_p1 = lax.all_gather(x_h1_b, axis_name=tp_axis, axis=0, tiled=False)

    q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
        ag_p0,
        wqkvg_fp8,
        wqkvg_scale_2d,
        cos_4d,
        sin_4d,
        q_4d,
        gate_4d,
        kv_4d,
        half_idx=0,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        x_scale_3d=None,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
        bk=bk,
        sub_n=None,
    )
    ag_p1, q_4d, gate_4d, kv_4d = lax.optimization_barrier(
        (ag_p1, q_4d, gate_4d, kv_4d)
    )
    q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
        ag_p1,
        wqkvg_fp8,
        wqkvg_scale_2d,
        cos_4d,
        sin_4d,
        q_4d,
        gate_4d,
        kv_4d,
        half_idx=1,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        x_scale_3d=None,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
        bk=bk,
        sub_n=None,
    )

    q_out = q_4d.reshape(t_total, q_size)
    gate_out = gate_4d.reshape(t_total, q_size)
    kv_packed = kv_4d.reshape(t_total, 2 * kv_size)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


def a1_v6_2half_static_fp8_single_ag(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    bk: int = 8192,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v6: Static-Scale FP8 input_layernorm + Single All-Gather per Half + Zero-Copy 4D Pallas A1.

    Eliminates `x_sc_s` and its `all_gather` (1 collective per half instead of 2) and folds
    `STATIC_A1_ACT_SCALE` (`1 / 32`) directly into `wqkvg_scale_2d`!
    """
    t_shard = stream_shard.shape[0]
    h_sub = t_shard // 2
    t_total = cos_128.shape[0]
    tp_deg = t_total // t_shard
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    cos_4d = cos_128.reshape(tp_deg, 2, h_sub, 128)
    sin_4d = sin_signed_128.reshape(tp_deg, 2, h_sub, 128)
    q_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    gate_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    kv_4d = jnp.empty((tp_deg, 2, h_sub, 2 * kv_size), dtype=jnp.float8_e4m3fn)

    x_fp8_s = sp_input_layernorm_static_fp8(stream_shard, norm_weight)
    w_sc_folded = wqkvg_scale_2d * STATIC_A1_ACT_SCALE

    ag0 = lax.all_gather(x_fp8_s[:h_sub], axis_name=tp_axis, axis=0, tiled=False)
    x_h1_b, ag0 = lax.optimization_barrier((x_fp8_s[h_sub:], ag0))
    ag1 = lax.all_gather(x_h1_b, axis_name=tp_axis, axis=0, tiled=False)

    q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
        ag0,
        wqkvg_fp8,
        w_sc_folded,
        cos_4d,
        sin_4d,
        q_4d,
        gate_4d,
        kv_4d,
        half_idx=0,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        x_scale_3d=None,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
        bk=bk,
        sub_n=None,
    )
    ag1, q_4d, gate_4d, kv_4d = lax.optimization_barrier((ag1, q_4d, gate_4d, kv_4d))
    q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half(
        ag1,
        wqkvg_fp8,
        w_sc_folded,
        cos_4d,
        sin_4d,
        q_4d,
        gate_4d,
        kv_4d,
        half_idx=1,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        x_scale_3d=None,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
        bk=bk,
        sub_n=None,
    )

    q_out = q_4d.reshape(t_total, q_size)
    gate_out = gate_4d.reshape(t_total, q_size)
    kv_packed = kv_4d.reshape(t_total, 2 * kv_size)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


def pallas_sp_qg_rope_fp8(
    x_fp8_shard: jax.Array,
    x_sc_shard: jax.Array,
    w_qg_3d: jax.Array,
    w_qg_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    h_q_local: int,
    k_scale: float = 1.0,
    fold_q_scale: bool = False,
    keep_gate_local: bool = True,
    rope_mode: str = "interleaved_roll",
    tm: int = 512,
    vmem_limit_mb: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Sequence-Parallel `Q + Gate` Pallas kernel on token owner (`[T_shard, 8192] @ [tp_deg, 8192, 2*q_size]`).

    Runs at full FP8×FP8 MXU rate, fuses dual-head `Q` RMSNorm and Interleaved RoPE on `T_shard`
    tokens, and outputs `q_3d_local: bf16[tp_deg, T_shard, q_size]` (ready for zero-transpose A2A!)
    and `gate_3d_local: bf16[T_shard, tp_deg, q_size]` (when `keep_gate_local=True`, ready for zero-copy
    `.reshape(T_shard, tp_deg * q_size)`!).
    """
    t_shard, k_dim = x_fp8_shard.shape
    tp_deg, _, two_q_size = w_qg_3d.shape
    q_size = two_q_size // 2
    num_q_slices = q_size // 128
    tm_eff = min(tm, t_shard)
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )

    def _sp_qg_kernel(
        x_ref, x_sc_ref, w_ref, w_sc_ref, cos_ref, sin_ref, q_out_ref, g_out_ref
    ):
        row_sc = x_sc_ref[:, :1]
        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[...]
        sin_tile = sin_ref[...]

        acc = jnp.dot(
            x_ref[...],
            w_ref[0, :, :],
            preferred_element_type=jnp.float32,
        ) * (row_sc * w_sc_ref[0, :, :])
        if keep_gate_local:
            g_out_ref[:, :] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)
        else:
            g_out_ref[0, :, :] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)

        for s in range(num_q_slices):
            c0 = s * 128
            c1 = c0 + 128
            q_pair = acc[:, c0:c1]
            q_norm = _dual_head_rmsnorm_128(
                q_pair, is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
            )
            q_rot = _rope_128_vreg(
                q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
            )
            q_out_ref[0, :, c0:c1] = q_rot

    g_out_spec = (
        pl.BlockSpec((tm_eff, q_size), lambda r, i: (i, r))
        if keep_gate_local
        else pl.BlockSpec((1, tm_eff, q_size), lambda r, i: (r, i, 0))
    )
    g_out_shape = (
        jax.ShapeDtypeStruct((t_shard, tp_deg * q_size), jnp.bfloat16)
        if keep_gate_local
        else jax.ShapeDtypeStruct((tp_deg, t_shard, q_size), jnp.bfloat16)
    )
    return pl.pallas_call(
        _sp_qg_kernel,
        grid=(tp_deg, t_shard // tm_eff),
        in_specs=[
            pl.BlockSpec((tm_eff, k_dim), lambda r, i: (i, 0)),
            pl.BlockSpec((tm_eff, 128), lambda r, i: (i, 0)),
            pl.BlockSpec((1, k_dim, two_q_size), lambda r, i: (r, 0, 0)),
            pl.BlockSpec((1, 1, two_q_size), lambda r, i: (r, 0, 0)),
            pl.BlockSpec((tm_eff, 128), lambda r, i: (i, 0)),
            pl.BlockSpec((tm_eff, 128), lambda r, i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, tm_eff, q_size), lambda r, i: (r, i, 0)),
            g_out_spec,
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tp_deg, t_shard, q_size), jnp.bfloat16),
            g_out_shape,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "parallel"),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(x_fp8_shard, x_sc_shard, w_qg_3d, w_qg_sc_3d, cos_shard, sin_shard)


def pallas_sp_kv_rope_bf16(
    x_bf16_shard: jax.Array,
    w_kv_3d: jax.Array,
    w_kv_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    h_kv_local: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rope_mode: str = "interleaved_roll",
    tm: int = 512,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """Sequence-Parallel `BF16×FP8` `K + V` Pallas kernel on token owner (`[T_shard, 8192] @ [tp_deg, 8192, 2*kv_size]`).

    Runs `BF16×FP8` on the `2,048` `K+V` columns (`11.1%` of `wqkvg`) while `Q_3d` All-to-All is in
    flight over ICI, restoring persisted FP8 KV cache cosine similarity from `0.99916` to `0.99993+`!
    """
    t_shard, k_dim = x_bf16_shard.shape
    tp_deg, _, two_kv_size = w_kv_3d.shape
    tm_eff = min(tm, t_shard)
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale

    def _sp_kv_kernel(x_ref, w_ref, w_sc_ref, cos_ref, sin_ref, kv_out_ref):
        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[...]
        sin_tile = sin_ref[...]

        for r in range(tp_deg):
            acc_kv = (
                jnp.dot(
                    x_ref[...],
                    w_ref[r, :, :],
                    preferred_element_type=jnp.float32,
                )
                * w_sc_ref[r, :, :]
            )
            if h_kv_local == 2:
                k_pair = acc_kv[:, 0:128]
                v_pair = acc_kv[:, 128:256].astype(jnp.bfloat16)
                k_norm = _dual_head_rmsnorm_128(
                    k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                kv_out_ref[r, :, 0:128] = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                kv_out_ref[r, :, 128:256] = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
            else:
                kv_pair = acc_kv[:, 0:128]
                sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                cos_kv = jnp.where(is_lo, cos_tile, 1.0)
                sin_kv = jnp.where(is_lo, sin_tile, 0.0)
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                kv_out_ref[r, :, 0:128] = jnp.clip(
                    kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

    return pl.pallas_call(
        _sp_kv_kernel,
        grid=(t_shard // tm_eff,),
        in_specs=[
            pl.BlockSpec((tm_eff, k_dim), lambda i: (i, 0)),
            pl.BlockSpec((tp_deg, k_dim, two_kv_size), lambda i: (0, 0, 0)),
            pl.BlockSpec((tp_deg, 1, two_kv_size), lambda i: (0, 0, 0)),
            pl.BlockSpec((tm_eff, 128), lambda i: (i, 0)),
            pl.BlockSpec((tm_eff, 128), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tp_deg, tm_eff, two_kv_size), lambda i: (0, i, 0)),
        out_shape=jax.ShapeDtypeStruct(
            (tp_deg, t_shard, two_kv_size), jnp.float8_e4m3fn
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(x_bf16_shard, w_kv_3d, w_kv_sc_3d, cos_shard, sin_shard)


def a1_v7_sp_local_qkvg_hybrid_a2a(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    w_qg_3d: jax.Array,
    w_qg_sc_3d: jax.Array,
    w_kv_3d: jax.Array,
    w_kv_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    a2a_fn,
    tp_axis: str = "tp",
    keep_gate_local: bool = True,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    tm: int = 512,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v7 (`a1_sp`): Sequence-Parallel Hybrid `FP8 Q/Gate + BF16 K/V` + Overlapped All-to-All!

    1. Computes `x_bf16_shard` and `(x_fp8_shard, x_sc_shard)` locally on `T_shard` tokens.
    2. Runs `pallas_sp_qg_rope_fp8` (`FP8×FP8` on `16,384` `Q+Gate` cols).
    3. Starts `Q_3d` All-to-All (`14 MiB` remote at T=8K TP8) overlapped with `pallas_sp_kv_rope_bf16`
       (`BF16×FP8` on `2,048` `K+V` cols for `0.99993+` KV cache cosine!).
    4. Runs `KV_3d` All-to-All (`1.75 MiB` remote at T=8K TP8) + `pack_kv_packed_to_seq_along_lane`.
    """
    t_shard = stream_shard.shape[0]
    tp_deg = w_qg_3d.shape[0]
    t_total = t_shard * tp_deg
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    x_bf16_shard = sp_input_layernorm_bf16(stream_shard, norm_weight)
    x_fp8_shard, x_sc_shard = sp_input_layernorm_algebraic_fp8(
        stream_shard, norm_weight
    )

    q_3d_local, g_3d_local = pallas_sp_qg_rope_fp8(
        x_fp8_shard,
        x_sc_shard,
        w_qg_3d,
        w_qg_sc_3d,
        cos_shard,
        sin_shard,
        h_q_local=h_q_local,
        k_scale=k_scale,
        fold_q_scale=fold_q_scale,
        keep_gate_local=keep_gate_local,
        tm=min(tm, t_shard),
    )

    # Launch Q All-to-All overlapped with BF16×FP8 K/V GEMM+RoPE!
    q_3d_recv = a2a_fn(q_3d_local, tp_axis=tp_axis)
    x_bf16_b, q_3d_recv = lax.optimization_barrier((x_bf16_shard, q_3d_recv))
    kv_3d_local = pallas_sp_kv_rope_bf16(
        x_bf16_b,
        w_kv_3d,
        w_kv_sc_3d,
        cos_shard,
        sin_shard,
        h_kv_local=h_kv_local,
        k_scale=k_scale,
        v_scale=v_scale,
        tm=min(tm, t_shard),
    )
    kv_3d_recv = a2a_fn(kv_3d_local, tp_axis=tp_axis)

    q_out = q_3d_recv.reshape(t_total, q_size)
    if keep_gate_local:
        gate_out = g_3d_local.reshape(t_shard, tp_deg * q_size)
    else:
        gate_out = a2a_fn(g_3d_local, tp_axis=tp_axis).reshape(t_total, q_size)

    kv_packed = kv_3d_recv.reshape(t_total, 2 * kv_size)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


def pallas_sp_staged_q_a2a_then_gate_fp8(
    x_fp8_shard: jax.Array,
    x_sc_shard: jax.Array,
    w_q_then_g_xor_3d: jax.Array,
    w_q_then_g_xor_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    h_q_local: int,
    tp_axis: str = "tp",
    k_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    vmem_limit_mb: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Interleaved `(Q_d, G_d)` Staged Per-Destination `Q` Async All-to-All + `Gate` Pallas Kernel (#87/#90/#102/#103/#105).

    Uses 1D `grid=(2 * tp_deg,)` with interleaved weight slices `w_q_then_g_xor_3d` where:
      - Even `step = 2 * d`: `w_q_3d[me ^ d]` -> computes `Q` for `dst = me ^ d`, applies dual-head `Q`
        RMSNorm + Interleaved RoPE, writes `q_send_vmem[q_slot]`, and immediately starts
        `make_async_copy` (`d == 0`) or `make_async_remote_copy` (`d > 0`, `device_id=(me ^ d,)`).
      - Odd `step = 2 * d + 1`: `w_g_3d[d]` -> computes `Gate` slice `d` (`10 µs` at 8K / `20 µs` at 16K)
        locally into `g_out_hbm_ref[:, d*q_size:(d+1)*q_size]`, filling the ICI transfer window between
        consecutive `Q` sends so the MXU never idles waiting for a send slot (#105 Item 1)!
      - At `T=8K` (`t_shard <= 1024`), uses `num_q_slots = tp_deg` (`16 MiB` VMEM, zero ring-slot waits);
        at `T=16K` (`t_shard = 2048`), uses `num_q_slots = 4` (`16 MiB` VMEM, reused after `4 Q + 4 G = 8`
        MXU steps = `~160 µs`) (#105 Item 2)!
    """
    t_shard, k_dim = x_fp8_shard.shape
    two_tp_deg, _, q_size = w_q_then_g_xor_3d.shape
    tp_deg = two_tp_deg // 2
    num_q_slices = q_size // 128
    tm_eff = min(1024, t_shard)
    num_q_slots = tp_deg if t_shard <= 1024 else min(4, tp_deg)
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )

    def _staged_qg_a2a_kernel(
        x_ref,
        x_sc_ref,
        w_ref,
        w_sc_ref,
        cos_ref,
        sin_ref,
        q_recv_hbm_ref,
        g_out_hbm_ref,
        q_send_vmem,
        g_vmem,
        rsem_buf,
        local_sem_buf,
        ssems,
        g_sems,
    ):
        step = pl.program_id(0)
        me = lax.axis_index(tp_axis)
        rsem = rsem_buf.at[0]
        local_sem = local_sem_buf.at[0]
        is_q = lax.rem(step, 2) == 0
        d_idx = lax.div(step, 2)

        @pl.when(step == 0)
        def _():
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
                pl.semaphore_wait(barrier, tp_deg - 1)

        @pl.when(is_q)
        def _():
            q_slot = lax.rem(d_idx, num_q_slots)
            if num_q_slots < tp_deg:
                # At d_idx == num_q_slots (slot 0's first reuse), wait on `local_sem` from d_idx == 0's local copy (#106 Item 1)!
                @pl.when(d_idx == num_q_slots)
                def _():
                    pltpu.make_async_copy(
                        q_send_vmem.at[pl.ds(0, 1)],
                        q_recv_hbm_ref.at[pl.ds(0, 1)],
                        local_sem,
                    ).wait()

                @pl.when(d_idx > num_q_slots)
                def _():
                    pltpu.make_async_copy(
                        q_send_vmem.at[pl.ds(q_slot, 1)],
                        q_send_vmem.at[pl.ds(q_slot, 1)],
                        ssems.at[q_slot],
                    ).wait()

            lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo
            w_sc_vec = w_sc_ref[0, :, :]

            for m0 in range(0, t_shard, tm_eff):
                row_sc = x_sc_ref[m0 : m0 + tm_eff, :1]
                cos_tile = cos_ref[m0 : m0 + tm_eff, :]
                sin_tile = sin_ref[m0 : m0 + tm_eff, :]
                acc = jnp.dot(
                    x_ref[m0 : m0 + tm_eff, :],
                    w_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                ) * (row_sc * w_sc_vec)
                for s in range(num_q_slices):
                    c0 = s * 128
                    c1 = c0 + 128
                    q_pair = acc[:, c0:c1]
                    q_norm = _dual_head_rmsnorm_128(
                        q_pair, is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                    )
                    q_rot = _rope_128_vreg(
                        q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                    )
                    q_send_vmem[q_slot, m0 : m0 + tm_eff, c0:c1] = q_rot

            @pl.when(d_idx == 0)
            def _():
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(0, 1)],
                    q_recv_hbm_ref.at[pl.ds(me, 1)],
                    local_sem,
                ).start()

            for d in range(1, tp_deg):

                @pl.when(d_idx == d)
                def _():
                    slot_d = d % num_q_slots
                    peer = lax.bitwise_xor(me, d)
                    pltpu.make_async_remote_copy(
                        src_ref=q_send_vmem.at[pl.ds(slot_d, 1)],
                        dst_ref=q_recv_hbm_ref.at[pl.ds(me, 1)],
                        send_sem=ssems.at[slot_d],
                        recv_sem=rsem,
                        device_id=(peer,),
                    ).start()

        @pl.when(~is_q)
        def _():
            g_rank = d_idx
            slot = lax.rem(g_rank, 2)

            @pl.when(g_rank >= 2)
            def _():
                pltpu.make_async_copy(
                    g_vmem.at[pl.ds(slot, 1), :, :].reshape(t_shard, q_size),
                    g_out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, q_size)],
                    g_sems.at[slot],
                ).wait()

            w_sc_vec = w_sc_ref[0, :, :]
            for m0 in range(0, t_shard, tm_eff):
                row_sc = x_sc_ref[m0 : m0 + tm_eff, :1]
                acc = jnp.dot(
                    x_ref[m0 : m0 + tm_eff, :],
                    w_ref[0, :, :],
                    preferred_element_type=jnp.float32,
                ) * (row_sc * w_sc_vec)
                g_vmem[slot, m0 : m0 + tm_eff, :] = acc.astype(jnp.bfloat16)
            pltpu.make_async_copy(
                g_vmem.at[pl.ds(slot, 1), :, :].reshape(t_shard, q_size),
                g_out_hbm_ref.at[
                    pl.ds(0, t_shard),
                    pl.ds(g_rank * q_size, q_size),
                ],
                g_sems.at[slot],
            ).start()

        @pl.when(step == two_tp_deg - 1)
        def _():
            for s in range(2):
                pltpu.make_async_copy(
                    g_vmem.at[pl.ds(s, 1), :, :].reshape(t_shard, q_size),
                    g_out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, q_size)],
                    g_sems.at[s],
                ).wait()
            if num_q_slots >= tp_deg:
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(0, 1)],
                    q_recv_hbm_ref.at[pl.ds(0, 1)],
                    local_sem,
                ).wait()
            tmpl_r = pltpu.make_async_copy(
                q_recv_hbm_ref.at[pl.ds(0, 1)],
                q_recv_hbm_ref.at[pl.ds(0, 1)],
                rsem,
            )
            for _ in range(tp_deg - 1):
                tmpl_r.wait()
            start_s = 1 if num_q_slots == tp_deg else 0
            for s in range(start_s, num_q_slots):
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(s, 1)],
                    q_send_vmem.at[pl.ds(s, 1)],
                    ssems.at[s],
                ).wait()

    return pl.pallas_call(
        _staged_qg_a2a_kernel,
        grid=(two_tp_deg,),
        in_specs=[
            pl.BlockSpec((t_shard, k_dim), lambda step: (0, 0)),
            pl.BlockSpec((t_shard, 128), lambda step: (0, 0)),
            pl.BlockSpec((1, k_dim, q_size), lambda step: (step, 0, 0)),
            pl.BlockSpec((1, 1, q_size), lambda step: (step, 0, 0)),
            pl.BlockSpec((t_shard, 128), lambda step: (0, 0)),
            pl.BlockSpec((t_shard, 128), lambda step: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tp_deg, t_shard, q_size), jnp.bfloat16),
            jax.ShapeDtypeStruct((t_shard, tp_deg * q_size), jnp.bfloat16),
        ],
        scratch_shapes=[
            pltpu.VMEM((num_q_slots, t_shard, q_size), jnp.bfloat16),
            pltpu.VMEM((2, t_shard, q_size), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((num_q_slots,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=7 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(
        x_fp8_shard,
        x_sc_shard,
        w_q_then_g_xor_3d,
        w_q_then_g_xor_sc_3d,
        cos_shard,
        sin_shard,
    )


def a1_v8_sp_staged_qg_async_a2a(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    w_q_then_g_3d: jax.Array,
    w_q_then_g_sc_3d: jax.Array,
    w_kv_3d: jax.Array,
    w_kv_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    a2a_fn,
    tp_axis: str = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    tm: int = 512,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v8 (`a1_sp` Staged): Per-Destination `Q` Async All-to-All Overlapped with `Gate` + Hybrid `BF16 K/V`."""
    t_shard = stream_shard.shape[0]
    tp_deg = w_kv_3d.shape[0]
    t_total = t_shard * tp_deg
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    x_bf16_shard = sp_input_layernorm_bf16(stream_shard, norm_weight)
    x_fp8_shard, x_sc_shard = sp_input_layernorm_algebraic_fp8(
        stream_shard, norm_weight
    )

    q_3d_recv, gate_out = pallas_sp_staged_q_a2a_then_gate_fp8(
        x_fp8_shard,
        x_sc_shard,
        w_q_then_g_3d,
        w_q_then_g_sc_3d,
        cos_shard,
        sin_shard,
        h_q_local=h_q_local,
        tp_axis=tp_axis,
        k_scale=k_scale,
        fold_q_scale=fold_q_scale,
    )
    kv_3d_local = pallas_sp_kv_rope_bf16(
        x_bf16_shard,
        w_kv_3d,
        w_kv_sc_3d,
        cos_shard,
        sin_shard,
        h_kv_local=h_kv_local,
        k_scale=k_scale,
        v_scale=v_scale,
        tm=min(tm, t_shard),
    )
    kv_3d_recv = a2a_fn(kv_3d_local, tp_axis=tp_axis)

    q_out = q_3d_recv.reshape(t_total, q_size)
    kv_packed = kv_3d_recv.reshape(t_total, 2 * kv_size)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


# -----------------------------------------------------------------------------
# 6. Wave 7: 1-Way HBM Write 2-Half A1 (`a1_v6_1way`, #112 B.2) and Single-Kernel
#    Fused `Q + Gate + Hi/Lo KV` Staged Async All-to-All (`a1_v9_sp`, #112 B.3)
# -----------------------------------------------------------------------------
def pallas_qkvg_qk_rope_kv_into_half_1way(
    x_half_3d: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_4d: jax.Array,
    sin_4d: jax.Array,
    q_4d: jax.Array,
    gate_4d: jax.Array,
    kv_4d: jax.Array,
    half_idx: int,
    h_q_local: int,
    h_kv_local: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    vmem_limit_mb: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """1-Way HBM Write Pallas A1 into half `half_idx` of `q_4d`, `gate_4d`, `kv_4d` (#112 B.2).

    Eliminates the wasted 34 MiB (8K) / 68 MiB (16K) HBM->VMEM read of uninitialized
    `q_4d, gate_4d, kv_4d` by keeping `q_4d, gate_4d, kv_4d` in `pltpu.HBM` (`input_output_aliases`)
    and writing from double-buffered VMEM scratch via 1-way `pltpu.make_async_copy`.
    """
    tp_deg, h_sub, x_cols = x_half_3d.shape
    k_dim, n_dim = wqkvg_fp8.shape
    has_packed_f8_sc = x_cols == k_dim + 128
    if not has_packed_f8_sc and x_cols != k_dim:
        raise ValueError(
            f"Expected x_half_3d cols in ({k_dim}, {k_dim + 128}), got {x_cols}"
        )
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM
    num_q_slices = q_size // 128
    tm_eff = min(tm, h_sub)
    num_m = h_sub // tm_eff
    total_steps = tp_deg * num_m
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale

    def _body_1way(
        x_ref,
        w_ref,
        w_sc_ref,
        cos_ref,
        sin_ref,
        q_in_hbm_ref,
        gate_in_hbm_ref,
        kv_in_hbm_ref,
        q_out_hbm_ref,
        gate_out_hbm_ref,
        kv_out_hbm_ref,
        q_vmem,
        g_vmem,
        kv_vmem,
        q_sems,
        g_sems,
        kv_sems,
    ):
        del q_in_hbm_ref, gate_in_hbm_ref, kv_in_hbm_ref
        step = pl.program_id(0)
        r = lax.div(step, num_m)
        m = lax.rem(step, num_m)
        slot = lax.rem(step, 2)

        @pl.when(step >= 2)
        def _():
            pltpu.make_async_copy(
                q_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                q_out_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, tm_eff), :],
                q_sems.at[slot],
            ).wait()
            pltpu.make_async_copy(
                g_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                gate_out_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, tm_eff), :],
                g_sems.at[slot],
            ).wait()
            pltpu.make_async_copy(
                kv_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(
                    1, 1, tm_eff, 2 * kv_size
                ),
                kv_out_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, tm_eff), :],
                kv_sems.at[slot],
            ).wait()

        from sglang_tpu.envs import envs as _tpu_envs

        sub_n_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_SUB_N", None)
        eff_sub_n = int(sub_n_env.get() or 0) if sub_n_env is not None else 0
        if eff_sub_n not in (128, 256, 512) or q_size % eff_sub_n != 0:
            eff_sub_n = 0
        fn_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM", None)
        use_fast_norm = bool(fn_env is not None and fn_env.get())

        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[0, 0, :, :]
        sin_tile = sin_ref[0, 0, :, :]

        if eff_sub_n > 0:
            if has_packed_f8_sc:
                s_hi = x_ref[0, :, k_dim : k_dim + 1].astype(jnp.float32)
                s_lo = x_ref[0, :, k_dim + 64 : k_dim + 65].astype(jnp.float32)
                row_sc = (
                    s_hi + s_lo * jnp.float32(1.0 / 16.0)
                ) * jnp.float32(A1_DYN_FP8_SCALE_UNPACK)
                x_lhs = x_ref[0, :, :k_dim]
            else:
                row_sc = None
                x_lhs = x_ref[0, :, :]

            def _strip_dot_1w(w_c0: int, w_c1: int) -> jax.Array:
                raw = jnp.dot(
                    x_lhs,
                    w_ref[:, w_c0:w_c1],
                    preferred_element_type=jnp.float32,
                )
                w_s = w_sc_ref[:, w_c0:w_c1]
                return raw * (row_sc * w_s) if row_sc is not None else raw * w_s

            for c0 in range(0, q_size, eff_sub_n):
                c1 = c0 + eff_sub_n
                g_vmem[slot, 0, 0, :, c0:c1] = _strip_dot_1w(
                    q_size + c0, q_size + c1
                ).astype(jnp.bfloat16)
            pltpu.make_async_copy(
                g_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                gate_out_hbm_ref.at[
                    pl.ds(r, 1),
                    pl.ds(half_idx, 1),
                    pl.ds(m * tm_eff, tm_eff),
                    :,
                ],
                g_sems.at[slot],
            ).start()

            for c0 in range(0, q_size, eff_sub_n):
                c1 = c0 + eff_sub_n
                acc_q = _strip_dot_1w(c0, c1)
                for s_loc in range(eff_sub_n // 128):
                    q_pair = acc_q[:, s_loc * 128 : (s_loc + 1) * 128]
                    q_norm = _dual_head_rmsnorm_128(
                        q_pair,
                        is_lo,
                        mask_lo,
                        mask_hi,
                        q_eff_scale,
                        eps=1e-5,
                        fast_6roll=use_fast_norm,
                    )
                    q_rot = _rope_128_vreg(
                        q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                    )
                    gc0 = c0 + s_loc * 128
                    q_vmem[slot, 0, 0, :, gc0 : gc0 + 128] = q_rot
            pltpu.make_async_copy(
                q_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                q_out_hbm_ref.at[
                    pl.ds(r, 1),
                    pl.ds(half_idx, 1),
                    pl.ds(m * tm_eff, tm_eff),
                    :,
                ],
                q_sems.at[slot],
            ).start()

            acc_kv = _strip_dot_1w(2 * q_size, 2 * q_size + 2 * kv_size)
            k_pair_src = acc_kv[:, :128]
            v_pair_src = acc_kv[:, 128:256] if h_kv_local == 2 else None
        else:
            if has_packed_f8_sc:
                s_hi = x_ref[0, :, k_dim : k_dim + 1].astype(jnp.float32)
                s_lo = x_ref[0, :, k_dim + 64 : k_dim + 65].astype(jnp.float32)
                row_sc = (
                    s_hi + s_lo * jnp.float32(1.0 / 16.0)
                ) * jnp.float32(A1_DYN_FP8_SCALE_UNPACK)
                acc = jnp.dot(
                    x_ref[0, :, :k_dim],
                    w_ref[:, :],
                    preferred_element_type=jnp.float32,
                ) * (row_sc * w_sc_ref[:, :])
            else:
                acc = (
                    jnp.dot(
                        x_ref[0, :, :],
                        w_ref[:, :],
                        preferred_element_type=jnp.float32,
                    )
                    * w_sc_ref[:, :]
                )

            g_vmem[slot, 0, 0, :, :] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)
            pltpu.make_async_copy(
                g_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                gate_out_hbm_ref.at[
                    pl.ds(r, 1),
                    pl.ds(half_idx, 1),
                    pl.ds(m * tm_eff, tm_eff),
                    :,
                ],
                g_sems.at[slot],
            ).start()

            for s in range(num_q_slices):
                c0 = s * 128
                c1 = c0 + 128
                q_pair = acc[:, c0:c1]
                q_norm = _dual_head_rmsnorm_128(
                    q_pair,
                    is_lo,
                    mask_lo,
                    mask_hi,
                    q_eff_scale,
                    eps=1e-5,
                    fast_6roll=use_fast_norm,
                )
                q_rot = _rope_128_vreg(
                    q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                q_vmem[slot, 0, 0, :, c0:c1] = q_rot
            pltpu.make_async_copy(
                q_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                q_out_hbm_ref.at[
                    pl.ds(r, 1),
                    pl.ds(half_idx, 1),
                    pl.ds(m * tm_eff, tm_eff),
                    :,
                ],
                q_sems.at[slot],
            ).start()
            k_pair_src = acc[:, 2 * q_size : 2 * q_size + 128]
            v_pair_src = (
                acc[:, 2 * q_size + 128 : 2 * q_size + 256]
                if h_kv_local == 2
                else None
            )

        if h_kv_local == 2:
            k_pair = k_pair_src
            v_pair = v_pair_src.astype(jnp.bfloat16)
            k_norm = _dual_head_rmsnorm_128(
                k_pair,
                is_lo,
                mask_lo,
                mask_hi,
                1.0,
                eps=1e-5,
                fast_6roll=use_fast_norm,
            )
            k_rot = _rope_128_vreg(
                k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
            )
            kv_vmem[slot, 0, 0, :, 0:128] = jnp.clip(
                k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
            kv_vmem[slot, 0, 0, :, 128:256] = jnp.clip(
                v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
        else:
            kv_pair = k_pair_src
            sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                1.0 / 64.0
            )
            inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
            kv_norm = (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
            cos_kv = jnp.where(is_lo, cos_tile, 1.0)
            sin_kv = jnp.where(is_lo, sin_tile, 0.0)
            kv_rot = _rope_128_vreg(kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode)
            kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
            kv_vmem[slot, 0, 0, :, 0:128] = jnp.clip(
                kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
        pltpu.make_async_copy(
            kv_vmem.at[pl.ds(slot, 1), :, :, :, :].reshape(1, 1, tm_eff, 2 * kv_size),
            kv_out_hbm_ref.at[
                pl.ds(r, 1),
                pl.ds(half_idx, 1),
                pl.ds(m * tm_eff, tm_eff),
                :,
            ],
            kv_sems.at[slot],
        ).start()

        @pl.when(step == total_steps - 1)
        def _():
            for s in range(min(2, total_steps)):
                pltpu.make_async_copy(
                    q_vmem.at[pl.ds(s, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                    q_out_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, tm_eff), :],
                    q_sems.at[s],
                ).wait()
                pltpu.make_async_copy(
                    g_vmem.at[pl.ds(s, 1), :, :, :, :].reshape(1, 1, tm_eff, q_size),
                    gate_out_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, tm_eff), :],
                    g_sems.at[s],
                ).wait()
                pltpu.make_async_copy(
                    kv_vmem.at[pl.ds(s, 1), :, :, :, :].reshape(
                        1, 1, tm_eff, 2 * kv_size
                    ),
                    kv_out_hbm_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, tm_eff), :],
                    kv_sems.at[s],
                ).wait()

    hbm_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    return pl.pallas_call(
        _body_1way,
        grid=(total_steps,),
        in_specs=[
            pl.BlockSpec(
                (1, tm_eff, x_cols),
                lambda step: (lax.div(step, num_m), lax.rem(step, num_m), 0),
            ),
            pl.BlockSpec((k_dim, n_dim), lambda step: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step: (0, 0)),
            pl.BlockSpec(
                (1, 1, tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m),
                    half_idx,
                    lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (1, 1, tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m),
                    half_idx,
                    lax.rem(step, num_m),
                    0,
                ),
            ),
            hbm_spec,
            hbm_spec,
            hbm_spec,
        ],
        out_specs=[hbm_spec, hbm_spec, hbm_spec],
        out_shape=[
            jax.ShapeDtypeStruct(q_4d.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(gate_4d.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(kv_4d.shape, jnp.float8_e4m3fn),
        ],
        scratch_shapes=[
            pltpu.VMEM((2, 1, 1, tm_eff, q_size), jnp.bfloat16),
            pltpu.VMEM((2, 1, 1, tm_eff, q_size), jnp.bfloat16),
            pltpu.VMEM((2, 1, 1, tm_eff, 2 * kv_size), jnp.float8_e4m3fn),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        input_output_aliases={5: 0, 6: 1, 7: 2},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(
        x_half_3d,
        wqkvg_fp8,
        wqkvg_scale_2d,
        cos_4d,
        sin_4d,
        q_4d,
        gate_4d,
        kv_4d,
    )


A1_DYN_FP8_SCALE_PACK = 8.0
A1_DYN_FP8_SCALE_UNPACK = 1.0 / A1_DYN_FP8_SCALE_PACK


def quant_a1_dyn_fp8_packed(x_bf16: jax.Array) -> jax.Array:
    """Dynamic per-row FP8 quant -> ``fp8_e4m3fn[rows, 8320]`` with hi+lo f8 scale in ``cols 8192:8320``."""
    rows = x_bf16.shape[0]
    xf = x_bf16.astype(jnp.float32)
    row_amax = jnp.max(jnp.abs(xf), axis=1, keepdims=True)
    row_sc = jnp.where(row_amax == 0.0, 0.0, row_amax * jnp.float32(1.0 / FP8_MAX))
    inv_sc = jnp.where(
        row_amax == 0.0, 0.0, jnp.float32(FP8_MAX) / jnp.maximum(row_amax, 1e-12)
    )
    x_f8 = jnp.clip(xf * inv_sc, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    s_scaled = jnp.clip(row_sc * jnp.float32(A1_DYN_FP8_SCALE_PACK), 0.0, FP8_MAX)
    s_hi = s_scaled.astype(jnp.float8_e4m3fn)
    s_lo = jnp.clip(
        (s_scaled - s_hi.astype(jnp.float32)) * jnp.float32(16.0),
        -FP8_MAX,
        FP8_MAX,
    ).astype(jnp.float8_e4m3fn)
    lane = lax.broadcasted_iota(jnp.int32, (rows, 128), 1)
    s_pack = jnp.where(
        lane < 64,
        jnp.broadcast_to(s_hi, (rows, 128)),
        jnp.broadcast_to(s_lo, (rows, 128)),
    )
    return jnp.concatenate([x_f8, s_pack], axis=1)


def pallas_a1_chunk_1way(
    x_chunk_3d: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_3d: jax.Array,
    sin_3d: jax.Array,
    outs: tuple[jax.Array, jax.Array, jax.Array] | None,
    *,
    t_shard: int,
    row_off: int,
    h_q_local: int,
    h_kv_local: int,
    rpa_layout: bool = False,
    mode: str = "all",
    tp_axis: str = "model",
    q_gate_only: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    vmem_limit_mb: int = 64,
    next_ag_in: jax.Array | None = None,
    collective_id: int = 13,
) -> tuple[jax.Array, ...]:
    """1-Way Async-HBM-Write A1 Chunk Kernel supporting uneven chunks, `rpa_layout`, `local`/`remote` modes, and `q_gate_only`.

    - When `outs is None`, allocates fresh 3D+ `(tp_deg, t_shard, *cols)` outputs in `pltpu.HBM`
      (eliminating `jnp.empty` input copies `copy.{q,gate,kv}_in_ref`).
    - Writes each `(dst_r, row_off + m * tm_eff)` block via 1-way `pltpu.make_async_copy` from
      double-buffered VMEM scratch (eliminating the `34 MiB` HBM->VMEM garbage read).
    - `mode`:
        - `"all"`: processes all `tp_deg` ranks (`x_chunk_3d` has shape `[tp_deg, h_sub, k_dim]`).
        - `"local"`: processes only rank `me`'s local shard (`x_chunk_3d` has shape `[1, h_sub, k_dim]`).
        - `"remote"`: processes only the `tp_deg - 1` remote peers (`r != me`) of `x_chunk_3d`.
    """
    tp_deg = cos_3d.shape[0]
    _, h_sub, x_cols = x_chunk_3d.shape
    k_dim, n_dim = wqkvg_fp8.shape
    has_packed_f8_sc = x_cols == k_dim + 128
    if not has_packed_f8_sc and x_cols != k_dim:
        raise ValueError(
            f"Expected x_chunk_3d cols in ({k_dim}, {k_dim + 128}), got {x_cols}"
        )
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM
    num_q_slices = q_size // 128
    cos_is_chunk = cos_3d.shape[1] == h_sub
    tm_eff = next(
        t
        for t in (256, 128, 64, 32)
        if t <= tm and h_sub % t == 0 and (cos_is_chunk or row_off % t == 0)
    )
    num_m = h_sub // tm_eff
    n_ranks = 1 if mode == "local" else (tp_deg - 1 if mode == "remote" else tp_deg)
    total_steps = n_ranks * num_m
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale

    q_cols = (h_q_local // 2, 2, 128) if rpa_layout else (q_size,)
    kv_cols = (4, 128) if rpa_layout else (2 * kv_size,)

    ag_spec = None
    if next_ag_in is not None:
        if mode != "all":
            raise ValueError("next_ag_in requires mode='all'")
        if next_ag_in.ndim == 2:
            next_ag_in = next_ag_in[None]
        _, next_rows, next_cols = next_ag_in.shape
        from sglang_tpu.kernels.xor1_bridge_all_gather import (
            xor1_ag_min_row_align,
            xor1_ag_schedule_spec,
        )

        item_bytes = jax.dtypes.itemsize_bits(next_ag_in.dtype) // 8
        min_row_align = xor1_ag_min_row_align(item_bytes)
        if (
            tp_deg < 4
            or tp_deg % 2 != 0
            or next_rows % min_row_align != 0
            or next_cols % 128 != 0
        ):
            raise ValueError(
                f"Incompatible next_ag_in shape {next_ag_in.shape} for tp_deg={tp_deg}"
            )

        ag_spec = xor1_ag_schedule_spec(tp_deg, int(next_rows), item_bytes, xy_torus=True)

    def _run_chunk(*args):
        if mode == "all":
            peers_ref = None
            rest = args
        else:
            peers_ref = args[0]
            rest = args[1:]

        x_ref, w_ref, w_sc_ref, cos_ref, sin_ref = rest[:5]
        n_out = 2 if q_gate_only else 3
        n_in_extra = (n_out if outs is not None else 0) + (1 if ag_spec is not None else 0)
        next_ag_hbm = rest[5 + (n_out if outs is not None else 0)] if ag_spec is not None else None
        out_refs_all = rest[5 + n_in_extra : 5 + n_in_extra + n_out + (1 if ag_spec is not None else 0)]
        out_refs = out_refs_all[:n_out]
        ag_out_hbm = out_refs_all[n_out] if ag_spec is not None else None
        scratch_all = rest[5 + n_in_extra + len(out_refs_all) :]
        scratch_refs = scratch_all[: 2 * n_out]
        ag_sems = scratch_all[2 * n_out :]

        if q_gate_only:
            q_out_hbm, g_out_hbm = out_refs
            q_vmem, g_vmem, q_sems, g_sems = scratch_refs
            kv_out_hbm = kv_vmem = kv_sems = None
        else:
            q_out_hbm, g_out_hbm, kv_out_hbm = out_refs
            q_vmem, g_vmem, kv_vmem, q_sems, g_sems, kv_sems = scratch_refs

        step = pl.program_id(0)
        r_idx = lax.div(step, num_m)
        m = lax.rem(step, num_m)
        slot = lax.rem(step, 2)
        dst_r = r_idx if mode == "all" else peers_ref[r_idx]

        if ag_spec is not None:
            from sglang_tpu.kernels.xor1_bridge_all_gather import emit_xor1_ag_phases

            ag_kind, ag_part_rows, _ = ag_spec
            ag_phases = emit_xor1_ag_phases(
                next_ag_hbm,
                ag_out_hbm,
                *ag_sems,
                axis_name=tp_axis,
                tp_size=tp_deg,
                kind=ag_kind,
                part_rows=ag_part_rows,
            )
            n_prep = len(ag_phases) - 1
            if total_steps > n_prep:
                for p_idx, ph_fn in enumerate(ag_phases[:-1]):
                    t_step = (p_idx * (total_steps - 1)) // n_prep

                    @pl.when(step == t_step)
                    def _(_fn=ph_fn):
                        _fn()
            else:

                @pl.when(step == 0)
                def _():
                    for ph_fn in ag_phases:
                        ph_fn()

        @pl.when(step >= 2)
        def _():
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                q_sems.at[slot],
            ).wait()
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                g_sems.at[slot],
            ).wait()
            if not q_gate_only:
                pltpu.make_async_copy(
                    kv_vmem.at[slot],
                    kv_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                    kv_sems.at[slot],
                ).wait()

        from sglang_tpu.envs import envs as _tpu_envs

        sub_n_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_SUB_N", None)
        eff_sub_n = int(sub_n_env.get() or 0) if sub_n_env is not None else 0
        if eff_sub_n not in (128, 256, 512) or q_size % eff_sub_n != 0:
            eff_sub_n = 0
        fn_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM", None)
        use_fast_norm = bool(fn_env is not None and fn_env.get())

        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[0, :, :]
        sin_tile = sin_ref[0, :, :]

        if eff_sub_n > 0:
            if has_packed_f8_sc:
                s_hi = x_ref[0, :, k_dim : k_dim + 1].astype(jnp.float32)
                s_lo = x_ref[0, :, k_dim + 64 : k_dim + 65].astype(jnp.float32)
                row_sc = (
                    s_hi + s_lo * jnp.float32(1.0 / 16.0)
                ) * jnp.float32(A1_DYN_FP8_SCALE_UNPACK)
                x_lhs = x_ref[0, :, :k_dim]
            else:
                row_sc = None
                x_lhs = x_ref[0, :, :]

            def _strip_dot_c(w_c0: int, w_c1: int) -> jax.Array:
                raw = jnp.dot(
                    x_lhs,
                    w_ref[:, w_c0:w_c1],
                    preferred_element_type=jnp.float32,
                )
                w_s = w_sc_ref[:, w_c0:w_c1]
                return raw * (row_sc * w_s) if row_sc is not None else raw * w_s

            for c0 in range(0, q_size, eff_sub_n):
                c1 = c0 + eff_sub_n
                g_vmem[slot, 0, :, c0:c1] = _strip_dot_c(
                    q_size + c0, q_size + c1
                ).astype(jnp.bfloat16)
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(dst_r, 1), pl.ds(row_off + m * tm_eff, tm_eff)],
                g_sems.at[slot],
            ).start()

            if rpa_layout:
                q_words = (
                    q_vmem.bitcast(jnp.uint32)
                    .at[slot, 0]
                    .reshape(tm_eff * num_q_slices, 128)
                )
            for c0 in range(0, q_size, eff_sub_n):
                c1 = c0 + eff_sub_n
                acc_q = _strip_dot_c(c0, c1)
                s_base = c0 // 128
                for s_loc in range(eff_sub_n // 128):
                    s = s_base + s_loc
                    q_pair = acc_q[:, s_loc * 128 : (s_loc + 1) * 128]
                    q_norm = _dual_head_rmsnorm_128(
                        q_pair,
                        is_lo,
                        mask_lo,
                        mask_hi,
                        q_eff_scale,
                        eps=1e-5,
                        fast_6roll=use_fast_norm,
                    )
                    q_rot = _rope_128_vreg(
                        q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                    )
                    if rpa_layout:
                        kv_head0 = (h_kv_local == 1) or (s < num_q_slices // 2)
                        q_words[pl.ds(s, tm_eff, stride=num_q_slices)] = _rpa_q_word(
                            q_rot, is_lo, mask_lo if kv_head0 else mask_hi
                        )
                    else:
                        q_vmem[slot, 0, :, s * 128 : (s + 1) * 128] = q_rot
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(dst_r, 1), pl.ds(row_off + m * tm_eff, tm_eff)],
                q_sems.at[slot],
            ).start()

            if not q_gate_only:
                if h_kv_local == 2:
                    acc_kv = _strip_dot_c(2 * q_size, 2 * q_size + 256)
                    k_pair = acc_kv[:, 0:128]
                    v_pair = acc_kv[:, 128:256].astype(jnp.bfloat16)
                else:
                    assert h_kv_local == 1
                    kv_pair = _strip_dot_c(2 * q_size, 2 * q_size + 128)
        else:
            if has_packed_f8_sc:
                s_hi = x_ref[0, :, k_dim : k_dim + 1].astype(jnp.float32)
                s_lo = x_ref[0, :, k_dim + 64 : k_dim + 65].astype(jnp.float32)
                row_sc = (
                    s_hi + s_lo * jnp.float32(1.0 / 16.0)
                ) * jnp.float32(A1_DYN_FP8_SCALE_UNPACK)
                acc = jnp.dot(
                    x_ref[0, :, :k_dim],
                    w_ref[:, :],
                    preferred_element_type=jnp.float32,
                ) * (row_sc * w_sc_ref[:, :])
            else:
                acc = (
                    jnp.dot(
                        x_ref[0, :, :],
                        w_ref[:, :],
                        preferred_element_type=jnp.float32,
                    )
                    * w_sc_ref[:, :]
                )

            g_vmem[slot, 0, :, :] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(dst_r, 1), pl.ds(row_off + m * tm_eff, tm_eff)],
                g_sems.at[slot],
            ).start()

            if rpa_layout:
                q_words = (
                    q_vmem.bitcast(jnp.uint32)
                    .at[slot, 0]
                    .reshape(tm_eff * num_q_slices, 128)
                )
            for s in range(num_q_slices):
                c0, c1 = s * 128, (s + 1) * 128
                q_pair = acc[:, c0:c1]
                q_norm = _dual_head_rmsnorm_128(
                    q_pair,
                    is_lo,
                    mask_lo,
                    mask_hi,
                    q_eff_scale,
                    eps=1e-5,
                    fast_6roll=use_fast_norm,
                )
                q_rot = _rope_128_vreg(
                    q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                if rpa_layout:
                    kv_head0 = (h_kv_local == 1) or (s < num_q_slices // 2)
                    q_words[pl.ds(s, tm_eff, stride=num_q_slices)] = _rpa_q_word(
                        q_rot, is_lo, mask_lo if kv_head0 else mask_hi
                    )
                else:
                    q_vmem[slot, 0, :, c0:c1] = q_rot
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(dst_r, 1), pl.ds(row_off + m * tm_eff, tm_eff)],
                q_sems.at[slot],
            ).start()
            if not q_gate_only:
                if h_kv_local == 2:
                    k_pair = acc[:, 2 * q_size : 2 * q_size + 128]
                    v_pair = acc[:, 2 * q_size + 128 : 2 * q_size + 256].astype(
                        jnp.bfloat16
                    )
                else:
                    assert h_kv_local == 1
                    kv_pair = acc[:, 2 * q_size : 2 * q_size + 128]

        if not q_gate_only:
            if h_kv_local == 2:
                k_norm = _dual_head_rmsnorm_128(
                    k_pair,
                    is_lo,
                    mask_lo,
                    mask_hi,
                    1.0,
                    eps=1e-5,
                    fast_6roll=use_fast_norm,
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                k8 = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                v8 = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                if rpa_layout:
                    kv_vmem.bitcast(jnp.uint32).at[slot, 0].reshape(tm_eff, 128)[...] = (
                        _f8_code_u32(k8) | (_f8_code_u32(v8) << 8)
                    )
                else:
                    kv_vmem[slot, 0, :, 0:128] = k8
                    kv_vmem[slot, 0, :, 128:256] = v8
            else:
                sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                cos_kv = jnp.where(is_lo, cos_tile, 1.0)
                sin_kv = jnp.where(is_lo, sin_tile, 0.0)
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_f32 = kv_rot.astype(jnp.float32)
                if rpa_layout:
                    k8 = jnp.clip(
                        jnp.where(is_lo, kv_f32 * inv_k_scale, 0.0),
                        -FP8_MAX,
                        FP8_MAX,
                    ).astype(jnp.float8_e4m3fn)
                    v8 = jnp.clip(
                        jnp.where(is_lo, pltpu.roll(kv_f32, shift=64, axis=1) * inv_v_scale, 0.0),
                        -FP8_MAX,
                        FP8_MAX,
                    ).astype(jnp.float8_e4m3fn)
                    kv_vmem.bitcast(jnp.uint32).at[slot, 0].reshape(tm_eff, 128)[...] = (
                        _f8_code_u32(k8) | (_f8_code_u32(v8) << 8)
                    )
                else:
                    kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                    kv_vmem[slot, 0, :, 0:128] = jnp.clip(
                        kv_f32 * kv_sc, -FP8_MAX, FP8_MAX
                    ).astype(jnp.float8_e4m3fn)
            pltpu.make_async_copy(
                kv_vmem.at[slot],
                kv_out_hbm.at[pl.ds(dst_r, 1), pl.ds(row_off + m * tm_eff, tm_eff)],
                kv_sems.at[slot],
            ).start()

        @pl.when(step == total_steps - 1)
        def _():
            for s in range(min(2, total_steps)):
                pltpu.make_async_copy(
                    q_vmem.at[s],
                    q_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                    q_sems.at[s],
                ).wait()
                pltpu.make_async_copy(
                    g_vmem.at[s],
                    g_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                    g_sems.at[s],
                ).wait()
                if not q_gate_only:
                    pltpu.make_async_copy(
                        kv_vmem.at[s],
                        kv_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                        kv_sems.at[s],
                    ).wait()
            if ag_spec is not None and total_steps > len(ag_phases) - 1:
                ag_phases[-1]()

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    row_blk_base = 0 if cos_is_chunk else row_off // tm_eff
    if mode == "all":
        in_specs = [
            pl.BlockSpec(
                (1, tm_eff, x_cols),
                lambda step: (lax.div(step, num_m), lax.rem(step, num_m), 0),
            ),
            pl.BlockSpec((k_dim, n_dim), lambda step: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step: (0, 0)),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m),
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m),
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]
    elif mode == "local":
        in_specs = [
            pl.BlockSpec((1, tm_eff, x_cols), lambda step, me_r: (0, step, 0)),
            pl.BlockSpec((k_dim, n_dim), lambda step, me_r: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step, me_r: (0, 0)),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step, me_r: (me_r[0], row_blk_base + step, 0),
            ),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step, me_r: (me_r[0], row_blk_base + step, 0),
            ),
        ]
    else:
        in_specs = [
            pl.BlockSpec(
                (1, tm_eff, x_cols),
                lambda step, peers: (
                    peers[lax.div(step, num_m)],
                    lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec((k_dim, n_dim), lambda step, peers: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step, peers: (0, 0)),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step, peers: (
                    peers[lax.div(step, num_m)],
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step, peers: (
                    peers[lax.div(step, num_m)],
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]

    n_out = 2 if q_gate_only else 3
    call_args: tuple[jax.Array, ...] = (
        x_chunk_3d,
        wqkvg_fp8,
        wqkvg_scale_2d,
        cos_3d,
        sin_3d,
    )
    aliases: dict[int, int] = {}
    if outs is not None:
        in_specs = in_specs + [hbm_spec] * n_out
        call_args = call_args + tuple(outs[:n_out])
        base_idx = (0 if mode == "all" else 1) + 5
        aliases = {base_idx + k: k for k in range(n_out)}

    out_shapes = [
        pltpu.HBM((tp_deg, t_shard, *q_cols), jnp.bfloat16),
        pltpu.HBM((tp_deg, t_shard, q_size), jnp.bfloat16),
    ]
    scratches = [
        pltpu.VMEM((2, 1, tm_eff, *q_cols), jnp.bfloat16),
        pltpu.VMEM((2, 1, tm_eff, q_size), jnp.bfloat16),
    ]
    sems = [pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,))]
    if not q_gate_only:
        out_shapes.append(pltpu.HBM((tp_deg, t_shard, *kv_cols), jnp.float8_e4m3fn))
        scratches.append(pltpu.VMEM((2, 1, tm_eff, *kv_cols), jnp.float8_e4m3fn))
        sems.append(pltpu.SemaphoreType.DMA((2,)))
    out_specs = [hbm_spec] * n_out
    if ag_spec is not None:
        in_specs = in_specs + [hbm_spec]
        call_args = call_args + (next_ag_in,)
        out_specs = out_specs + [hbm_spec]
        out_shapes.append(
            pltpu.HBM((tp_deg, int(next_ag_in.shape[1]), int(next_ag_in.shape[2])), next_ag_in.dtype)
        )
        sems.extend(pltpu.SemaphoreType.DMA((n,)) for n in ag_spec[2])

    from sglang_tpu.envs import envs as _a1_name_envs

    _sn_env = getattr(_a1_name_envs, "SGLANG_TPU_MUSE_KERNEL_A1_SUB_N", None)
    _sn_val = int(_sn_env.get() or 0) if _sn_env is not None else 0
    if _sn_val not in (128, 256, 512) or q_size % _sn_val != 0:
        _sn_val = 0
    _fn_env = getattr(_a1_name_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM", None)
    _fn_on = bool(_fn_env is not None and _fn_env.get())
    name = (
        f"pallas_a1_1way_{mode}_off{row_off}_h{h_sub}_tm{tm_eff}"
        + (f"_sn{_sn_val}" if _sn_val > 0 else "")
        + ("_fn" if _fn_on else "")
        + ("_rpal" if rpa_layout else "")
        + (f"_inag_{ag_spec[0]}" if ag_spec is not None else "")
    )
    if mode == "all":
        c_params = (
            pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
                collective_id=collective_id,
            )
            if ag_spec is not None
            else pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            )
        )
        return pl.pallas_call(
            _run_chunk,
            grid=(total_steps,),
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shapes,
            scratch_shapes=scratches + sems,
            input_output_aliases=aliases,
            compiler_params=c_params,
            name=name,
        )(*call_args)

    me_1d = pallas_axis_index_1d(tp_axis)
    me = me_1d[0]
    if mode == "local":
        pref = me_1d
    else:
        idx = jnp.arange(tp_deg - 1, dtype=jnp.int32)
        pref = jnp.where(idx < me, idx, idx + 1)
    return pl.pallas_call(
        _run_chunk,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(total_steps,),
            in_specs=in_specs,
            out_specs=[hbm_spec] * n_out,
            scratch_shapes=scratches + sems,
        ),
        out_shape=out_shapes,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=name,
    )(pref, *call_args)


def pallas_sp_kv_local_bf16_rpal(
    x_bf16_shard: jax.Array,
    w_kv_3d: jax.Array,
    w_kv_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    *,
    h_kv_local: int = 2,
    rpa_layout: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    vmem_limit_mb: int = 60,
) -> jax.Array:
    """Local `BF16 × FP8` `K + V` kernel on token owner (`[t_shard, 8192] @ [tp_deg, 8192, 256]`).

    Produces exact `v5x4`-grade persisted `K` and `V` (`fp8[tp_deg, t_shard, 4, 128]` under
    `rpa_layout` or `fp8[tp_deg, t_shard, 256]`), running on TensorCore completely hidden
    underneath Chunk 0's SparseCore `FP8` All-Gather in `v6_bf16kv`.
    """
    assert h_kv_local == 2
    t_shard, k_dim = x_bf16_shard.shape
    tp_deg, _, two_kv_size = w_kv_3d.shape
    tm_eff = min(tm, t_shard)
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    kv_cols = (4, 128) if rpa_layout else (two_kv_size,)

    def _sp_kv_rpal_kernel(x_ref, w_ref, w_sc_ref, cos_ref, sin_ref, kv_out_ref):
        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[...]
        sin_tile = sin_ref[...]
        if rpa_layout:
            kv_u32 = kv_out_ref.bitcast(jnp.uint32)

        for r in range(tp_deg):
            acc_kv = (
                jnp.dot(
                    x_ref[...],
                    w_ref[r, :, :],
                    preferred_element_type=jnp.float32,
                )
                * w_sc_ref[r, :, :]
            )
            k_pair = acc_kv[:, 0:128]
            v_pair = acc_kv[:, 128:256].astype(jnp.bfloat16)
            k_norm = _dual_head_rmsnorm_128(
                k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
            )
            k_rot = _rope_128_vreg(
                k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
            )
            k8 = jnp.clip(
                k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
            v8 = jnp.clip(
                v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
            if rpa_layout:
                kv_u32.at[r].reshape(tm_eff, 128)[...] = _f8_code_u32(k8) | (
                    _f8_code_u32(v8) << 8
                )
            else:
                kv_out_ref[r, :, 0:128] = k8
                kv_out_ref[r, :, 128:256] = v8

    out_spec = (
        pl.BlockSpec((tp_deg, tm_eff, 4, 128), lambda i: (0, i, 0, 0))
        if rpa_layout
        else pl.BlockSpec((tp_deg, tm_eff, two_kv_size), lambda i: (0, i, 0))
    )
    return pl.pallas_call(
        _sp_kv_rpal_kernel,
        grid=(t_shard // tm_eff,),
        in_specs=[
            pl.BlockSpec((tm_eff, k_dim), lambda i: (i, 0)),
            pl.BlockSpec((tp_deg, k_dim, two_kv_size), lambda i: (0, 0, 0)),
            pl.BlockSpec((tp_deg, 1, two_kv_size), lambda i: (0, 0, 0)),
            pl.BlockSpec((tm_eff, 128), lambda i: (i, 0)),
            pl.BlockSpec((tm_eff, 128), lambda i: (i, 0)),
        ],
        out_specs=out_spec,
        out_shape=jax.ShapeDtypeStruct((tp_deg, t_shard, *kv_cols), jnp.float8_e4m3fn),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name="pallas_sp_kv_local_bf16" + ("_rpal" if rpa_layout else ""),
    )(x_bf16_shard, w_kv_3d, w_kv_sc_3d, cos_shard, sin_shard)


# Pack g_sc = g_amax / 127.0 into E4M3 (s_hi, s_lo) with * 8.0 so g_amax saturates at
# 448.0 * 127.0 / 8.0 = 7112.0 instead of 111.125 (avoiding long-8k outlier clipping).
S8_SCALE_PACK_MUL = 8.0
S8_SCALE_UNPACK_MUL = 1.0 / S8_SCALE_PACK_MUL


def quant_chunk_s8_packed(
    x_chunk_bf16: jax.Array,
    inv_gain_2d: jax.Array,
    *,
    per_group_128: bool = False,
) -> jax.Array:
    """Quantizes `[c_sz, 8192]` bf16 -> `int8[c_sz, 8320]` with `w_norm` channel-gain equalization and `hi+lo` f8 scales in `cols 8192:8320`."""
    rows = x_chunk_bf16.shape[0]
    x_eq = (x_chunk_bf16 * inv_gain_2d).astype(jnp.float32)
    if per_group_128:
        xg = x_eq.reshape(rows, 64, 128)
        g_amax = jnp.max(jnp.abs(xg), axis=-1)
        g_sc = jnp.where(g_amax == 0.0, 0.0, g_amax * jnp.float32(1.0 / 127.0))
        inv_sc = jnp.where(
            g_amax == 0.0, 0.0, jnp.float32(127.0) / jnp.maximum(g_amax, 1e-12)
        )[..., None]
        x_i8 = jnp.rint(xg * inv_sc).astype(jnp.int8).reshape(rows, 8192)
        s_scaled = jnp.clip(g_sc * jnp.float32(S8_SCALE_PACK_MUL), 0.0, FP8_MAX)
        s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
        s_lo_f8 = jnp.clip(
            (s_scaled - s_hi_f8.astype(jnp.float32)) * jnp.float32(16.0),
            -FP8_MAX,
            FP8_MAX,
        ).astype(jnp.float8_e4m3fn)
        s_pack_i8 = lax.bitcast_convert_type(
            jnp.concatenate([s_hi_f8, s_lo_f8], axis=1), jnp.int8
        )
        return jnp.concatenate([x_i8, s_pack_i8], axis=1)

    row_amax = jnp.max(jnp.abs(x_eq), axis=1, keepdims=True)
    row_sc = jnp.where(row_amax == 0.0, 0.0, row_amax * jnp.float32(1.0 / 127.0))
    inv_sc = jnp.where(
        row_amax == 0.0, 0.0, jnp.float32(127.0) / jnp.maximum(row_amax, 1e-12)
    )
    x_i8 = jnp.rint(x_eq * inv_sc).astype(jnp.int8)
    s_scaled = jnp.clip(row_sc * jnp.float32(S8_SCALE_PACK_MUL), 0.0, FP8_MAX)
    s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
    s_lo_f8 = jnp.clip(
        (s_scaled - s_hi_f8.astype(jnp.float32)) * jnp.float32(16.0),
        -FP8_MAX,
        FP8_MAX,
    ).astype(jnp.float8_e4m3fn)
    lane = lax.broadcasted_iota(jnp.int32, (rows, 128), 1)
    s_pack_f8 = jnp.where(
        lane < 64,
        jnp.broadcast_to(s_hi_f8, (rows, 128)),
        jnp.broadcast_to(s_lo_f8, (rows, 128)),
    )
    s_pack_i8 = lax.bitcast_convert_type(s_pack_f8, jnp.int8)
    return jnp.concatenate([x_i8, s_pack_i8], axis=1)


def pallas_a1_qg_1way_fkv(
    x_3d: jax.Array,
    gain_2d: jax.Array | None,
    wqg_fp8: jax.Array,
    wqg_scale_2d: jax.Array,
    cos_3d: jax.Array,
    sin_3d: jax.Array,
    outs: tuple[jax.Array, jax.Array] | None,
    *,
    kv_local: jax.Array | None = None,
    s8_packed: bool = False,
    per_group_128: bool = False,
    t_shard: int,
    row_off: int,
    h_q_local: int,
    rpa_layout: bool = False,
    mode: str = "all",
    tp_axis: str = "model",
    fold_q_scale: bool = False,
    k_scale: float = 1.0,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    vmem_limit_mb: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array | None]:
    """1-Way `Q + Gate` Chunk Kernel (`fp8@fp8` or `s8 -> bf16@fp8`) with optional zero-VMEM `kv_local` P2P All-to-All overlapped under Chunk 0's MXU compute."""
    tp_deg = cos_3d.shape[0]
    _, h_sub, in_cols = x_3d.shape
    k_dim = in_cols - 128 if s8_packed else in_cols
    _, n_dim = wqg_fp8.shape
    q_size = h_q_local * HEAD_DIM
    num_q_slices = q_size // 128
    cos_is_chunk = cos_3d.shape[1] == h_sub
    tm_eff = next(
        t
        for t in (256, 128, 64, 32)
        if t <= tm and h_sub % t == 0 and (cos_is_chunk or row_off % t == 0)
    )
    num_m = h_sub // tm_eff
    n_ranks = tp_deg - 1 if mode == "remote" else tp_deg
    total_steps = n_ranks * num_m
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    q_cols = (h_q_local // 2, 2, 128) if rpa_layout else (q_size,)
    fuse_kv = kv_local is not None
    kv_3d = kv_local.reshape(tp_deg, kv_local.shape[1], -1) if fuse_kv else None
    has_gain = gain_2d is not None
    strip_k = 128 if per_group_128 else (256 if tm_eff == 256 else 512)

    def _run(*args):
        idx = 0
        if mode != "all":
            peers_ref = args[idx]
            idx += 1
        else:
            peers_ref = None

        x_ref = args[idx]
        idx += 1
        if has_gain:
            gain_ref = args[idx]
            idx += 1
        else:
            gain_ref = None
        w_ref, w_sc_ref, cos_ref, sin_ref = args[idx : idx + 4]
        idx += 4

        if outs is not None:
            idx += 2
        if fuse_kv:
            kv_src_ref = args[idx]
            idx += 1
        else:
            kv_src_ref = None

        q_out_hbm, g_out_hbm = args[idx : idx + 2]
        idx += 2
        if fuse_kv:
            kv_dst_hbm = args[idx]
            idx += 1
        else:
            kv_dst_hbm = None

        if s8_packed:
            x_bf16_vmem = args[idx]
            idx += 1
        else:
            x_bf16_vmem = None
        q_vmem, g_vmem, q_sems, g_sems = args[idx : idx + 4]
        idx += 4
        kv_sems = args[idx] if fuse_kv else None

        step = pl.program_id(0)
        r_idx = lax.div(step, num_m)
        m = lax.rem(step, num_m)
        slot = lax.rem(step, 2)
        dst_r = r_idx if mode == "all" else peers_ref[r_idx]
        me_id = lax.axis_index(tp_axis)

        if fuse_kv:

            @pl.when(step == 0)
            def _start_kv_a2a():
                if tp_deg > 1:
                    barrier = pltpu.get_barrier_semaphore()
                    for d in range(1, tp_deg):
                        pl.semaphore_signal(
                            barrier, device_id={tp_axis: lax.rem(me_id + d, tp_deg)}
                        )
                    pl.semaphore_wait(barrier, tp_deg - 1)
                pltpu.make_async_copy(
                    kv_src_ref.at[pl.ds(me_id, 1)],
                    kv_dst_hbm.at[pl.ds(me_id, 1)],
                    kv_sems.at[0],
                ).start()
                for d in range(1, tp_deg):
                    peer_d = lax.rem(me_id + d, tp_deg)
                    pltpu.make_async_remote_copy(
                        src_ref=kv_src_ref.at[pl.ds(peer_d, 1)],
                        dst_ref=kv_dst_hbm.at[pl.ds(me_id, 1)],
                        send_sem=kv_sems.at[1],
                        recv_sem=kv_sems.at[0],
                        device_id={tp_axis: peer_d},
                    ).start()

        @pl.when(step >= 2)
        def _():
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                q_sems.at[slot],
            ).wait()
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                g_sems.at[slot],
            ).wait()

        if s8_packed:
            s_pack_f8 = lax.bitcast_convert_type(
                x_ref[0, :, k_dim : k_dim + 128], jnp.float8_e4m3fn
            )
            if per_group_128:
                sc_64 = (
                    (
                        s_pack_f8[:, :64].astype(jnp.float32)
                        + s_pack_f8[:, 64:].astype(jnp.float32)
                        * jnp.float32(1.0 / 16.0)
                    )
                    * jnp.float32(S8_SCALE_UNPACK_MUL)
                ).astype(jnp.bfloat16)
                for g_i in range(64):
                    c0, c1 = g_i * 128, (g_i + 1) * 128
                    deq = (
                        x_ref[0, :, c0:c1].astype(jnp.bfloat16)
                        * sc_64[:, g_i : g_i + 1]
                    )
                    x_bf16_vmem[:, c0:c1] = (
                        deq * gain_ref[:, c0:c1] if has_gain else deq
                    )
            else:
                s_hi = s_pack_f8[:, 0:1].astype(jnp.float32)
                s_lo = s_pack_f8[:, 64:65].astype(jnp.float32)
                sc_row = (
                    (s_hi + s_lo * jnp.float32(1.0 / 16.0))
                    * jnp.float32(S8_SCALE_UNPACK_MUL)
                ).astype(jnp.bfloat16)
                for c_blk in range(k_dim // strip_k):
                    c0, c1 = c_blk * strip_k, (c_blk + 1) * strip_k
                    deq = x_ref[0, :, c0:c1].astype(jnp.bfloat16) * sc_row
                    x_bf16_vmem[:, c0:c1] = (
                        deq * gain_ref[:, c0:c1] if has_gain else deq
                    )
            lhs_dot = x_bf16_vmem[...]
        else:
            lhs_dot = x_ref[0, :, :]

        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile = cos_ref[0, :, :]
        sin_tile = sin_ref[0, :, :]

        acc = (
            jnp.dot(lhs_dot, w_ref[:, :], preferred_element_type=jnp.float32)
            * w_sc_ref[:, :]
        )

        g_vmem[slot, 0, :, :] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)
        pltpu.make_async_copy(
            g_vmem.at[slot],
            g_out_hbm.at[pl.ds(dst_r, 1), pl.ds(row_off + m * tm_eff, tm_eff)],
            g_sems.at[slot],
        ).start()

        if rpa_layout:
            q_words = (
                q_vmem.bitcast(jnp.uint32)
                .at[slot, 0]
                .reshape(tm_eff * num_q_slices, 128)
            )
        for s in range(num_q_slices):
            c0, c1 = s * 128, (s + 1) * 128
            q_norm = _dual_head_rmsnorm_128(
                acc[:, c0:c1], is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
            )
            q_rot = _rope_128_vreg(
                q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
            )
            if rpa_layout:
                kv_head0 = s < num_q_slices // 2
                q_words[pl.ds(s, tm_eff, stride=num_q_slices)] = _rpa_q_word(
                    q_rot, is_lo, mask_lo if kv_head0 else mask_hi
                )
            else:
                q_vmem[slot, 0, :, c0:c1] = q_rot
        pltpu.make_async_copy(
            q_vmem.at[slot],
            q_out_hbm.at[pl.ds(dst_r, 1), pl.ds(row_off + m * tm_eff, tm_eff)],
            q_sems.at[slot],
        ).start()

        @pl.when(step == total_steps - 1)
        def _():
            for s_i in range(min(2, total_steps)):
                pltpu.make_async_copy(
                    q_vmem.at[s_i],
                    q_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                    q_sems.at[s_i],
                ).wait()
                pltpu.make_async_copy(
                    g_vmem.at[s_i],
                    g_out_hbm.at[pl.ds(0, 1), pl.ds(0, tm_eff)],
                    g_sems.at[s_i],
                ).wait()
            if fuse_kv:
                tmpl_r = pltpu.make_async_copy(
                    kv_dst_hbm.at[pl.ds(0, 1)],
                    kv_dst_hbm.at[pl.ds(0, 1)],
                    kv_sems.at[0],
                )
                for _ in range(tp_deg):
                    tmpl_r.wait()
                tmpl_s = pltpu.make_async_copy(
                    kv_src_ref.at[pl.ds(0, 1)],
                    kv_src_ref.at[pl.ds(0, 1)],
                    kv_sems.at[1],
                )
                for _ in range(tp_deg - 1):
                    tmpl_s.wait()

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    row_blk_base = 0 if cos_is_chunk else row_off // tm_eff
    if mode == "all":
        in_specs = [
            pl.BlockSpec(
                (1, tm_eff, in_cols),
                lambda step: (lax.div(step, num_m), lax.rem(step, num_m), 0),
            )
        ]
        if has_gain:
            in_specs.append(pl.BlockSpec((1, k_dim), lambda step: (0, 0)))
        in_specs += [
            pl.BlockSpec((k_dim, n_dim), lambda step: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step: (0, 0)),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m),
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m),
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]
    else:
        in_specs = [
            pl.BlockSpec(
                (1, tm_eff, in_cols),
                lambda step, peers: (
                    peers[lax.div(step, num_m)],
                    lax.rem(step, num_m),
                    0,
                ),
            )
        ]
        if has_gain:
            in_specs.append(pl.BlockSpec((1, k_dim), lambda step, peers: (0, 0)))
        in_specs += [
            pl.BlockSpec((k_dim, n_dim), lambda step, peers: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step, peers: (0, 0)),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step, peers: (
                    peers[lax.div(step, num_m)],
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (1, tm_eff, 128),
                lambda step, peers: (
                    peers[lax.div(step, num_m)],
                    row_blk_base + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]

    call_args_list: list[jax.Array] = [x_3d]
    if has_gain:
        call_args_list.append(gain_2d)
    call_args_list += [wqg_fp8, wqg_scale_2d, cos_3d, sin_3d]

    aliases: dict[int, int] = {}
    if outs is not None:
        base_idx = (0 if mode == "all" else 1) + len(call_args_list)
        aliases = {base_idx: 0, base_idx + 1: 1}
        in_specs = in_specs + [hbm_spec, hbm_spec]
        call_args_list += [outs[0], outs[1]]

    if fuse_kv:
        in_specs.append(any_spec)
        call_args_list.append(kv_3d)

    out_specs = [hbm_spec, hbm_spec] + ([any_spec] if fuse_kv else [])
    out_shapes = [
        pltpu.HBM((tp_deg, t_shard, *q_cols), jnp.bfloat16),
        pltpu.HBM((tp_deg, t_shard, q_size), jnp.bfloat16),
    ]
    if fuse_kv:
        out_shapes.append(jax.ShapeDtypeStruct(kv_3d.shape, kv_3d.dtype))

    scratches = []
    if s8_packed:
        scratches.append(pltpu.VMEM((tm_eff, k_dim), jnp.bfloat16))
    scratches += [
        pltpu.VMEM((2, 1, tm_eff, *q_cols), jnp.bfloat16),
        pltpu.VMEM((2, 1, tm_eff, q_size), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    if fuse_kv:
        scratches.append(pltpu.SemaphoreType.DMA((2,)))

    tag = ("s8grp128" if per_group_128 else "s8row") if s8_packed else "fp8"
    name = (
        f"pallas_a1_{tag}_{mode}_off{row_off}_h{h_sub}_tm{tm_eff}"
        + ("_fkv" if fuse_kv else "")
        + ("_rpal" if rpa_layout else "")
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        collective_id=19 if (fuse_kv and tp_deg > 1) else None,
        vmem_limit_bytes=vmem_limit_mb * (1 << 20),
    )
    if mode == "all":
        res = pl.pallas_call(
            _run,
            grid=(total_steps,),
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shapes,
            scratch_shapes=scratches,
            input_output_aliases=aliases,
            compiler_params=cp,
            name=name,
        )(*call_args_list)
    else:
        me = lax.axis_index(tp_axis).astype(jnp.int32)
        idx_arr = jnp.arange(tp_deg - 1, dtype=jnp.int32)
        pref = jnp.where(idx_arr < me, idx_arr, idx_arr + 1)
        res = pl.pallas_call(
            _run,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(total_steps,),
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratches,
            ),
            out_shape=out_shapes,
            input_output_aliases=aliases,
            compiler_params=cp,
            name=name,
        )(pref, *call_args_list)

    if fuse_kv:
        return (res[0], res[1]), res[2].reshape(kv_local.shape)
    return (res[0], res[1]), None


def a1_v6_1way_static_fp8_single_ag(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    prenormed_fp8_shard: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v6-1way (#112 B.2): Static-FP8 Single-AG 2-Half A1 with 1-Way HBM Writes (0 wasted read)."""
    t_shard = stream_shard.shape[0]
    h_sub = t_shard // 2
    t_total = cos_128.shape[0]
    tp_deg = t_total // t_shard
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    cos_4d = cos_128.reshape(tp_deg, 2, h_sub, 128)
    sin_4d = sin_signed_128.reshape(tp_deg, 2, h_sub, 128)
    q_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    gate_4d = jnp.empty((tp_deg, 2, h_sub, q_size), dtype=jnp.bfloat16)
    kv_4d = jnp.empty((tp_deg, 2, h_sub, 2 * kv_size), dtype=jnp.float8_e4m3fn)

    x_fp8_s = (
        prenormed_fp8_shard
        if prenormed_fp8_shard is not None
        else sp_input_layernorm_static_fp8(stream_shard, norm_weight)
    )
    w_sc_folded = wqkvg_scale_2d * STATIC_A1_ACT_SCALE

    ag0 = lax.all_gather(x_fp8_s[:h_sub], axis_name=tp_axis, axis=0, tiled=False)
    x_h1_b, ag0 = lax.optimization_barrier((x_fp8_s[h_sub:], ag0))
    ag1 = lax.all_gather(x_h1_b, axis_name=tp_axis, axis=0, tiled=False)

    q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half_1way(
        ag0,
        wqkvg_fp8,
        w_sc_folded,
        cos_4d,
        sin_4d,
        q_4d,
        gate_4d,
        kv_4d,
        half_idx=0,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
    )
    ag1, q_4d, gate_4d, kv_4d = lax.optimization_barrier((ag1, q_4d, gate_4d, kv_4d))
    q_4d, gate_4d, kv_4d = pallas_qkvg_qk_rope_kv_into_half_1way(
        ag1,
        wqkvg_fp8,
        w_sc_folded,
        cos_4d,
        sin_4d,
        q_4d,
        gate_4d,
        kv_4d,
        half_idx=1,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
        rope_mode=rope_mode,
        tm=tm,
    )

    q_out = q_4d.reshape(t_total, q_size)
    gate_out = gate_4d.reshape(t_total, q_size)
    kv_packed = kv_4d.reshape(t_total, 2 * kv_size)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


def sp_input_layernorm_hilo_fp8(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-pass algebraic-cancel Hi+Lo FP8 input_layernorm returning `(x_hi_fp8, x_lo_fp8, x_sc_shard)`.

    - `x_hi_fp8`: `fp8_e4m3fn[T_shard, 8192]` (used at 1x FP8 MXU rate for `Q` and `Gate`).
    - `x_lo_fp8`: `fp8_e4m3fn[T_shard, 8192]` (64x scaled residual; `x_hi + (1/64)*x_lo` gives 7-bit mantissa
      == BF16 accuracy for the 256 `K/V` columns without materializing `x_bf16_shard`!).
    - `x_sc_shard`: `fp32[T_shard, 128]` broadcast row scale.
    """
    xf = stream_shard.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    inv_rms = lax.rsqrt(var + eps)
    gain = norm_weight.astype(jnp.float32) + 1.0
    scaled_unnorm = xf * gain[None, :]
    amax = jnp.maximum(jnp.max(jnp.abs(scaled_unnorm), axis=-1, keepdims=True), 1e-12)
    inv_amax = FP8_MAX / amax
    y_scaled = scaled_unnorm * inv_amax
    x_hi_fp8 = jnp.clip(y_scaled, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    y_rem = jnp.clip(
        (y_scaled - x_hi_fp8.astype(jnp.float32)) * 16.0, -FP8_MAX, FP8_MAX
    )
    x_lo_fp8 = y_rem.astype(jnp.float8_e4m3fn)
    eff_row_scale = (inv_rms * amax) * (1.0 / FP8_MAX)
    x_sc_shard = jnp.broadcast_to(eff_row_scale, (stream_shard.shape[0], 128))
    return x_hi_fp8, x_lo_fp8, x_sc_shard


def pallas_sp_single_kernel_staged_qkvg_a2a(
    x_hi_fp8_shard: jax.Array,
    x_lo_fp8_shard: jax.Array,
    x_sc_shard: jax.Array,
    w_q_then_g_xor_3d: jax.Array,
    w_q_then_g_xor_sc_3d: jax.Array,
    w_kv_xor_3d: jax.Array,
    w_kv_xor_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    vmem_limit_mb: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-Kernel Fused Staged `(Q_d, G_d + KV_d)` Pallas Kernel with In-Kernel `Q` & `KV` Async A2A (#112 B.3).

    Fuses `pallas_sp_staged_q_a2a_then_gate_fp8`, `pallas_sp_kv_rope_bf16`, and `kv_a2a` into ONE Pallas
    kernel under ONE `(t_half == 0) & (step == 0)` cross-device barrier, tiled over `num_t_half = t_shard // 1024`
    (`1` at `T=8K`, `2` at `T=16K`) with single-buffered `x_hi_vmem`/`x_lo_vmem` (`51.5 MiB` total VMEM at both `8K` & `16K`!):
      - Even `step = 2 * d`: computes `Q` for `peer = me ^ d` (`FP8×FP8`), applies dual-head `Q` RMSNorm + RoPE,
        and immediately pushes `q_send_vmem[q_slot] -> q_recv_hbm_ref[me, t_half]` on `peer`.
      - Odd `step = 2 * d + 1`: computes `Gate` slice `d` (`FP8×FP8`) into `g_out_hbm_ref`, AND computes
        `KV` for `peer = me ^ d` using `Hi+Lo` FP8 (`x_hi @ w_kv + (1/16) * (x_lo @ w_kv)`, unclipped 8-bit mantissa > BF16
        for `0.999918+` K and `1.0000` V cache cosine!), applies `K` RMSNorm + RoPE + FP8 quant into `kv_send_vmem[d]`,
        and immediately pushes `kv_send_vmem[d] -> kv_recv_hbm_ref[me, t_half]` on `peer`!
    """
    t_shard, k_dim = x_hi_fp8_shard.shape
    two_tp_deg, _, q_size = w_q_then_g_xor_3d.shape
    tp_deg = two_tp_deg // 2
    _, _, two_kv_size = w_kv_xor_3d.shape
    num_q_slices = q_size // 128
    tm_eff = min(1024, t_shard)
    num_t_half = t_shard // tm_eff
    num_q_slots = min(4, tp_deg)
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale

    def _single_qkvg_a2a_kernel(
        x_hi_hbm_ref,
        x_lo_hbm_ref,
        x_sc_hbm_ref,
        w_qg_ref,
        w_qg_sc_ref,
        w_kv_ref,
        w_kv_sc_ref,
        cos_hbm_ref,
        sin_hbm_ref,
        q_recv_hbm_ref,
        g_out_hbm_ref,
        kv_recv_hbm_ref,
        x_hi_vmem,
        x_lo_vmem,
        x_sc_vmem,
        cos_vmem,
        sin_vmem,
        q_send_vmem,
        g_vmem,
        kv_send_vmem,
        q_rsem_buf,
        q_local_sem_buf,
        q_ssems,
        g_sems,
        kv_rsem_buf,
        kv_local_sem_buf,
        kv_ssems,
    ):
        t_half = pl.program_id(0)
        step = pl.program_id(1)
        me = lax.axis_index(tp_axis)
        q_rsem = q_rsem_buf.at[0]
        q_local_sem = q_local_sem_buf.at[t_half]
        kv_rsem = kv_rsem_buf.at[0]
        kv_local_sem = kv_local_sem_buf.at[t_half]
        is_q = lax.rem(step, 2) == 0
        d_idx = lax.div(step, 2)

        @pl.when((t_half == 0) & (step == 0))
        def _():
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(barrier, device_id=(lax.rem(me + d, tp_deg),))
                pl.semaphore_wait(barrier, tp_deg - 1)

        @pl.when(step == 0)
        def _():
            pltpu.sync_copy(
                x_hi_hbm_ref.at[pl.ds(t_half * tm_eff, tm_eff), :],
                x_hi_vmem,
            )
            pltpu.sync_copy(
                x_lo_hbm_ref.at[pl.ds(t_half * tm_eff, tm_eff), :],
                x_lo_vmem,
            )
            pltpu.sync_copy(
                x_sc_hbm_ref.at[pl.ds(t_half * tm_eff, tm_eff), :],
                x_sc_vmem,
            )
            pltpu.sync_copy(
                cos_hbm_ref.at[pl.ds(t_half * tm_eff, tm_eff), :],
                cos_vmem,
            )
            pltpu.sync_copy(
                sin_hbm_ref.at[pl.ds(t_half * tm_eff, tm_eff), :],
                sin_vmem,
            )

        @pl.when(is_q)
        def _():
            q_slot = lax.rem(d_idx, num_q_slots)
            if num_q_slots < tp_deg:

                @pl.when(d_idx == num_q_slots)
                def _():
                    pltpu.make_async_copy(
                        q_send_vmem.at[pl.ds(0, 1)],
                        q_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                        q_local_sem,
                    ).wait()

                @pl.when(d_idx > num_q_slots)
                def _():
                    pltpu.make_async_copy(
                        q_send_vmem.at[pl.ds(q_slot, 1)],
                        q_send_vmem.at[pl.ds(q_slot, 1)],
                        q_ssems.at[t_half, q_slot],
                    ).wait()
            elif num_t_half > 1:

                @pl.when((t_half > 0) & (d_idx == 0))
                def _():
                    pltpu.make_async_copy(
                        q_send_vmem.at[pl.ds(0, 1)],
                        q_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                        q_local_sem_buf.at[t_half - 1],
                    ).wait()

                @pl.when((t_half > 0) & (d_idx > 0) & (d_idx < num_q_slots))
                def _():
                    pltpu.make_async_copy(
                        q_send_vmem.at[pl.ds(q_slot, 1)],
                        q_send_vmem.at[pl.ds(q_slot, 1)],
                        q_ssems.at[t_half - 1, q_slot],
                    ).wait()

            lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo
            w_sc_vec = w_qg_sc_ref[0, :, :]

            row_sc = x_sc_vmem[:, :1]
            cos_tile = cos_vmem[...]
            sin_tile = sin_vmem[...]
            acc = jnp.dot(
                x_hi_vmem[...],
                w_qg_ref[0, :, :],
                preferred_element_type=jnp.float32,
            ) * (row_sc * w_sc_vec)
            for s in range(num_q_slices):
                c0 = s * 128
                c1 = c0 + 128
                q_pair = acc[:, c0:c1]
                q_norm = _dual_head_rmsnorm_128(
                    q_pair, is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                )
                q_rot = _rope_128_vreg(
                    q_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                q_send_vmem[q_slot, :, c0:c1] = q_rot

            @pl.when(d_idx == 0)
            def _():
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(0, 1)],
                    q_recv_hbm_ref.at[pl.ds(me, 1), pl.ds(t_half * tm_eff, tm_eff), :],
                    q_local_sem,
                ).start()

            for d in range(1, tp_deg):

                @pl.when(d_idx == d)
                def _():
                    slot_d = d % num_q_slots
                    peer = lax.bitwise_xor(me, d)
                    pltpu.make_async_remote_copy(
                        src_ref=q_send_vmem.at[pl.ds(slot_d, 1)],
                        dst_ref=q_recv_hbm_ref.at[
                            pl.ds(me, 1), pl.ds(t_half * tm_eff, tm_eff), :
                        ],
                        send_sem=q_ssems.at[t_half, slot_d],
                        recv_sem=q_rsem,
                        device_id=(peer,),
                    ).start()

        @pl.when(~is_q)
        def _():
            g_rank = d_idx
            slot = lax.rem(g_rank, 2)

            @pl.when(g_rank >= 2)
            def _():
                pltpu.make_async_copy(
                    g_vmem.at[pl.ds(slot, 1), :, :].reshape(tm_eff, q_size),
                    g_out_hbm_ref.at[pl.ds(0, tm_eff), pl.ds(0, q_size)],
                    g_sems.at[t_half, slot],
                ).wait()

            if num_t_half > 1:

                @pl.when((t_half > 0) & (g_rank < 2))
                def _():
                    pltpu.make_async_copy(
                        g_vmem.at[pl.ds(slot, 1), :, :].reshape(tm_eff, q_size),
                        g_out_hbm_ref.at[pl.ds(0, tm_eff), pl.ds(0, q_size)],
                        g_sems.at[t_half - 1, slot],
                    ).wait()

                @pl.when((t_half > 0) & (g_rank == 0))
                def _():
                    pltpu.make_async_copy(
                        kv_send_vmem.at[pl.ds(0, 1)],
                        kv_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                        kv_local_sem_buf.at[t_half - 1],
                    ).wait()

                @pl.when((t_half > 0) & (g_rank > 0))
                def _():
                    pltpu.make_async_copy(
                        kv_send_vmem.at[pl.ds(g_rank, 1)],
                        kv_send_vmem.at[pl.ds(g_rank, 1)],
                        kv_ssems.at[t_half - 1, g_rank],
                    ).wait()

            w_sc_vec = w_qg_sc_ref[0, :, :]
            w_kv_sc_vec = w_kv_sc_ref[0, :, :]
            lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo

            row_sc = x_sc_vmem[:, :1]
            cos_tile = cos_vmem[...]
            sin_tile = sin_vmem[...]
            x_hi_tile = x_hi_vmem[...]
            x_lo_tile = x_lo_vmem[...]
            acc_g = jnp.dot(
                x_hi_tile,
                w_qg_ref[0, :, :],
                preferred_element_type=jnp.float32,
            ) * (row_sc * w_sc_vec)
            g_vmem[slot, :, :] = acc_g.astype(jnp.bfloat16)

            # Hi+Lo FP8×FP8 KV GEMM (unclipped * 16.0 => 8-bit mantissa > BF16 accuracy!)
            acc_kv_hi = jnp.dot(
                x_hi_tile,
                w_kv_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )
            acc_kv_lo = jnp.dot(
                x_lo_tile,
                w_kv_ref[0, :, :],
                preferred_element_type=jnp.float32,
            )
            acc_kv = (acc_kv_hi + (1.0 / 16.0) * acc_kv_lo) * (row_sc * w_kv_sc_vec)
            if h_kv_local == 2:
                k_pair = acc_kv[:, 0:128]
                v_pair = acc_kv[:, 128:256].astype(jnp.bfloat16)
                k_norm = _dual_head_rmsnorm_128(
                    k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
                )
                kv_send_vmem[g_rank, :, 0:128] = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                kv_send_vmem[g_rank, :, 128:256] = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
            else:
                kv_pair = acc_kv[:, 0:128]
                sq_lo = jnp.sum(kv_pair * kv_pair * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (kv_pair * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                cos_kv = jnp.where(is_lo, cos_tile, 1.0)
                sin_kv = jnp.where(is_lo, sin_tile, 0.0)
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                kv_send_vmem[g_rank, :, 0:128] = jnp.clip(
                    kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

            pltpu.make_async_copy(
                g_vmem.at[pl.ds(slot, 1), :, :].reshape(tm_eff, q_size),
                g_out_hbm_ref.at[
                    pl.ds(t_half * tm_eff, tm_eff),
                    pl.ds(g_rank * q_size, q_size),
                ],
                g_sems.at[t_half, slot],
            ).start()

            @pl.when(g_rank == 0)
            def _():
                pltpu.make_async_copy(
                    kv_send_vmem.at[pl.ds(0, 1)],
                    kv_recv_hbm_ref.at[pl.ds(me, 1), pl.ds(t_half * tm_eff, tm_eff), :],
                    kv_local_sem,
                ).start()

            for d in range(1, tp_deg):

                @pl.when(g_rank == d)
                def _():
                    peer = lax.bitwise_xor(me, d)
                    pltpu.make_async_remote_copy(
                        src_ref=kv_send_vmem.at[pl.ds(d, 1)],
                        dst_ref=kv_recv_hbm_ref.at[
                            pl.ds(me, 1), pl.ds(t_half * tm_eff, tm_eff), :
                        ],
                        send_sem=kv_ssems.at[t_half, d],
                        recv_sem=kv_rsem,
                        device_id=(peer,),
                    ).start()

        @pl.when((t_half == num_t_half - 1) & (step == two_tp_deg - 1))
        def _():
            for s in range(2):
                pltpu.make_async_copy(
                    g_vmem.at[pl.ds(s, 1), :, :].reshape(tm_eff, q_size),
                    g_out_hbm_ref.at[pl.ds(0, tm_eff), pl.ds(0, q_size)],
                    g_sems.at[num_t_half - 1, s],
                ).wait()
            if num_q_slots >= tp_deg:
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(0, 1)],
                    q_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                    q_local_sem_buf.at[num_t_half - 1],
                ).wait()
            tmpl_qr = pltpu.make_async_copy(
                q_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                q_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                q_rsem,
            )
            for _ in range((tp_deg - 1) * num_t_half):
                tmpl_qr.wait()
            start_s = 1 if num_q_slots == tp_deg else 0
            for s in range(start_s, num_q_slots):
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(s, 1)],
                    q_send_vmem.at[pl.ds(s, 1)],
                    q_ssems.at[num_t_half - 1, s],
                ).wait()

            pltpu.make_async_copy(
                kv_send_vmem.at[pl.ds(0, 1)],
                kv_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                kv_local_sem_buf.at[num_t_half - 1],
            ).wait()
            tmpl_kvr = pltpu.make_async_copy(
                kv_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                kv_recv_hbm_ref.at[pl.ds(0, 1), pl.ds(0, tm_eff), :],
                kv_rsem,
            )
            for _ in range((tp_deg - 1) * num_t_half):
                tmpl_kvr.wait()
            for d in range(1, tp_deg):
                pltpu.make_async_copy(
                    kv_send_vmem.at[pl.ds(d, 1)],
                    kv_send_vmem.at[pl.ds(d, 1)],
                    kv_ssems.at[num_t_half - 1, d],
                ).wait()

    return pl.pallas_call(
        _single_qkvg_a2a_kernel,
        grid=(num_t_half, two_tp_deg),
        in_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec((1, k_dim, q_size), lambda t_half, step: (step, 0, 0)),
            pl.BlockSpec((1, 1, q_size), lambda t_half, step: (step, 0, 0)),
            pl.BlockSpec(
                (1, k_dim, two_kv_size), lambda t_half, step: (lax.div(step, 2), 0, 0)
            ),
            pl.BlockSpec(
                (1, 1, two_kv_size), lambda t_half, step: (lax.div(step, 2), 0, 0)
            ),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
            pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tp_deg, t_shard, q_size), jnp.bfloat16),
            jax.ShapeDtypeStruct((t_shard, tp_deg * q_size), jnp.bfloat16),
            jax.ShapeDtypeStruct((tp_deg, t_shard, two_kv_size), jnp.float8_e4m3fn),
        ],
        scratch_shapes=[
            pltpu.VMEM((tm_eff, k_dim), jnp.float8_e4m3fn),
            pltpu.VMEM((tm_eff, k_dim), jnp.float8_e4m3fn),
            pltpu.VMEM((tm_eff, 128), jnp.float32),
            pltpu.VMEM((tm_eff, 128), jnp.float32),
            pltpu.VMEM((tm_eff, 128), jnp.float32),
            pltpu.VMEM((num_q_slots, tm_eff, q_size), jnp.bfloat16),
            pltpu.VMEM((2, tm_eff, q_size), jnp.bfloat16),
            pltpu.VMEM((tp_deg, tm_eff, two_kv_size), jnp.float8_e4m3fn),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((num_t_half,)),
            pltpu.SemaphoreType.DMA((num_t_half, num_q_slots)),
            pltpu.SemaphoreType.DMA((num_t_half, 2)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((num_t_half,)),
            pltpu.SemaphoreType.DMA((num_t_half, tp_deg)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            collective_id=14 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(
        x_hi_fp8_shard,
        x_lo_fp8_shard,
        x_sc_shard,
        w_q_then_g_xor_3d,
        w_q_then_g_xor_sc_3d,
        w_kv_xor_3d,
        w_kv_xor_sc_3d,
        cos_shard,
        sin_shard,
    )


def a1_v9_sp_single_kernel_staged_qkvg_a2a(
    stream_shard: jax.Array,
    norm_weight: jax.Array,
    w_q_then_g_xor_3d: jax.Array,
    w_q_then_g_xor_sc_3d: jax.Array,
    w_kv_xor_3d: jax.Array,
    w_kv_xor_sc_3d: jax.Array,
    cos_shard: jax.Array,
    sin_shard: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "tp",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    prenormed_hilo_tuple: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Variant A1-v9 (`a1_sp` Single-Kernel Staged Q+G+KV, #112 B.3): Fuses Q, Gate, and Hi+Lo KV + Async A2As into 1 kernel!"""
    t_shard = stream_shard.shape[0]
    tp_deg = w_kv_xor_3d.shape[0]
    t_total = t_shard * tp_deg
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    if prenormed_hilo_tuple is not None:
        x_hi_fp8, x_lo_fp8, x_sc_shard = prenormed_hilo_tuple
    else:
        x_hi_fp8, x_lo_fp8, x_sc_shard = sp_input_layernorm_hilo_fp8(
            stream_shard, norm_weight
        )

    q_3d_recv, gate_out, kv_3d_recv = pallas_sp_single_kernel_staged_qkvg_a2a(
        x_hi_fp8,
        x_lo_fp8,
        x_sc_shard,
        w_q_then_g_xor_3d,
        w_q_then_g_xor_sc_3d,
        w_kv_xor_3d,
        w_kv_xor_sc_3d,
        cos_shard,
        sin_shard,
        h_q_local=h_q_local,
        h_kv_local=h_kv_local,
        tp_axis=tp_axis,
        k_scale=k_scale,
        v_scale=v_scale,
        fold_q_scale=fold_q_scale,
    )

    q_out = q_3d_recv.reshape(t_total, q_size)
    kv_packed = kv_3d_recv.reshape(t_total, 2 * kv_size)
    k_fp8, v_fp8, kv_cache_out = pack_kv_packed_to_seq_along_lane(
        kv_packed, kv_cache, page_indices, h_kv_local=h_kv_local
    )
    return q_out, gate_out, k_fp8, v_fp8, kv_cache_out


# -----------------------------------------------------------------------------
# Wave 6: Zero-Copy 2D Tiled A1 + Fused `kv_local` Strided-128 `s8` Pre-Quant
#         + Zero-Recompile Active-Token Tile Skip (`valid_tokens_1d`)
# -----------------------------------------------------------------------------
def pallas_quant_chunk0_s8_unsliced(
    x_full_bf16: jax.Array,
    inv_gain_2d: jax.Array,
    *,
    c0_rows: int,
    strided_128: bool = True,
    vmem_limit_mb: int = 16,
) -> jax.Array:
    """Quantizes `x_full_bf16[:c0_rows]` to `int8[c0_rows, k_dim + 128]` directly without slicing `x_full_bf16`."""
    _, k_dim = x_full_bf16.shape
    tm_q = next(t for t in (256, 128, 64, 32) if c0_rows % t == 0)

    def _q_kernel(x_ref, ig_ref, s8_out_ref):
        lane = lax.broadcasted_iota(jnp.int32, (tm_q, 128), 1)
        is_lo = lane < 64
        amax_128 = jnp.zeros((tm_q, 128), dtype=jnp.float32)
        for s in range(k_dim // 128):
            c0, c1 = s * 128, (s + 1) * 128
            xs = (x_ref[:, c0:c1] * ig_ref[:, c0:c1]).astype(jnp.float32)
            amax_128 = jnp.maximum(amax_128, jnp.abs(xs))
        if strided_128:
            g_amax = jnp.maximum(amax_128[:, :64], amax_128[:, 64:])
            g_sc = jnp.where(g_amax == 0.0, 0.0, g_amax * jnp.float32(1.0 / 127.0))
            inv_sc_64 = jnp.where(
                g_amax == 0.0, 0.0, jnp.float32(127.0) / jnp.maximum(g_amax, 1e-12)
            )
            inv_sc_128 = jnp.concatenate([inv_sc_64, inv_sc_64], axis=1)
            s_scaled = jnp.clip(g_sc * jnp.float32(S8_SCALE_PACK_MUL), 0.0, FP8_MAX)
            s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
            s_lo_f8 = jnp.clip(
                (s_scaled - s_hi_f8.astype(jnp.float32)) * jnp.float32(16.0),
                -FP8_MAX,
                FP8_MAX,
            ).astype(jnp.float8_e4m3fn)
            s_pack_f8 = jnp.concatenate([s_hi_f8, s_lo_f8], axis=1)
        else:
            row_amax = jnp.max(amax_128, axis=1, keepdims=True)
            row_sc = jnp.where(
                row_amax == 0.0, 0.0, row_amax * jnp.float32(1.0 / 127.0)
            )
            inv_sc_128 = jnp.broadcast_to(
                jnp.where(
                    row_amax == 0.0,
                    0.0,
                    jnp.float32(127.0) / jnp.maximum(row_amax, 1e-12),
                ),
                (tm_q, 128),
            )
            s_scaled = jnp.clip(row_sc * jnp.float32(S8_SCALE_PACK_MUL), 0.0, FP8_MAX)
            s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
            s_lo_f8 = jnp.clip(
                (s_scaled - s_hi_f8.astype(jnp.float32)) * jnp.float32(16.0),
                -FP8_MAX,
                FP8_MAX,
            ).astype(jnp.float8_e4m3fn)
            s_pack_f8 = jnp.where(
                is_lo,
                jnp.broadcast_to(s_hi_f8, (tm_q, 128)),
                jnp.broadcast_to(s_lo_f8, (tm_q, 128)),
            )
        for s in range(k_dim // 128):
            c0, c1 = s * 128, (s + 1) * 128
            xs = (x_ref[:, c0:c1] * ig_ref[:, c0:c1]).astype(jnp.float32)
            s8_out_ref[:, c0:c1] = jnp.rint(xs * inv_sc_128).astype(jnp.int8)
        s8_out_ref[:, k_dim : k_dim + 128] = lax.bitcast_convert_type(
            s_pack_f8, jnp.int8
        )

    return pl.pallas_call(
        _q_kernel,
        grid=(c0_rows // tm_q,),
        in_specs=[
            pl.BlockSpec((tm_q, k_dim), lambda i: (i, 0)),
            pl.BlockSpec((1, k_dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tm_q, k_dim + 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((c0_rows, k_dim + 128), jnp.int8),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=f"pallas_quant_c0_{'s8str128' if strided_128 else 's8row'}_h{c0_rows}_tm{tm_q}",
    )(x_full_bf16, inv_gain_2d)


def pallas_sp_kv_local_bf16_fkvq(
    x_bf16_shard: jax.Array,
    w_kv_3d: jax.Array,
    w_kv_sc_3d: jax.Array,
    cos_2d: jax.Array,
    sin_2d: jax.Array,
    inv_gain_2d: jax.Array | None = None,
    *,
    quant_s8: bool = True,
    strided_128: bool = True,
    h_kv_local: int = 2,
    rpa_layout: bool = True,
    tp_axis: str = "model",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rope_mode: str = "interleaved_roll",
    tm: int = 128,
    vmem_limit_mb: int = 60,
) -> tuple[jax.Array, jax.Array | None]:
    """Single-pass local `BF16 × FP8` `KV` (`(tp_deg, t_shard, 1, 4, 128)`) + optional `s8` pre-quantization of `x_bf16_shard`."""
    assert h_kv_local == 2
    t_shard, k_dim = x_bf16_shard.shape
    tp_deg, _, two_kv_size = w_kv_3d.shape
    tm_eff = next(t for t in (256, 128, 64, 32) if t <= tm and t_shard % t == 0)
    num_m_shard = t_shard // tm_eff
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    kv_cols = (1, 4, 128) if rpa_layout else (two_kv_size,)
    me_1d = pallas_axis_index_1d(tp_axis)

    def _kernel(*args):
        me_ref, x_ref, w_ref, w_sc_ref, cos_ref, sin_ref = args[:6]
        del me_ref
        if quant_s8:
            ig_ref, kv_out_ref, s8_out_ref = args[6], args[7], args[8]
        else:
            ig_ref, kv_out_ref, s8_out_ref = None, args[6], None

        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo
        cos_tile, sin_tile = cos_ref[...], sin_ref[...]
        if rpa_layout:
            kv_u32 = kv_out_ref.bitcast(jnp.uint32)
        for r in range(tp_deg):
            acc_kv = (
                jnp.dot(x_ref[...], w_ref[r, :, :], preferred_element_type=jnp.float32)
                * w_sc_ref[r, :, :]
            )
            k_norm = _dual_head_rmsnorm_128(
                acc_kv[:, :128], is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
            )
            k_rot = _rope_128_vreg(
                k_norm, cos_tile, sin_tile, lane, rope_mode=rope_mode
            )
            k8 = jnp.clip(
                k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
            v8 = jnp.clip(
                acc_kv[:, 128:256].astype(jnp.bfloat16).astype(jnp.float32)
                * inv_v_scale,
                -FP8_MAX,
                FP8_MAX,
            ).astype(jnp.float8_e4m3fn)
            if rpa_layout:
                kv_u32.at[r].reshape(tm_eff, 128)[...] = _f8_code_u32(k8) | (
                    _f8_code_u32(v8) << 8
                )
            else:
                kv_out_ref[r, :, 0:128] = k8
                kv_out_ref[r, :, 128:256] = v8

        if quant_s8:
            amax_128 = jnp.zeros((tm_eff, 128), dtype=jnp.float32)
            for s in range(k_dim // 128):
                c0, c1 = s * 128, (s + 1) * 128
                xs = (x_ref[:, c0:c1] * ig_ref[:, c0:c1]).astype(jnp.float32)
                amax_128 = jnp.maximum(amax_128, jnp.abs(xs))
            if strided_128:
                g_amax = jnp.maximum(amax_128[:, :64], amax_128[:, 64:])
                g_sc = jnp.where(g_amax == 0.0, 0.0, g_amax * jnp.float32(1.0 / 127.0))
                inv_sc_64 = jnp.where(
                    g_amax == 0.0, 0.0, jnp.float32(127.0) / jnp.maximum(g_amax, 1e-12)
                )
                inv_sc_128 = jnp.concatenate([inv_sc_64, inv_sc_64], axis=1)
                s_scaled = jnp.clip(g_sc * jnp.float32(S8_SCALE_PACK_MUL), 0.0, FP8_MAX)
                s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
                s_lo_f8 = jnp.clip(
                    (s_scaled - s_hi_f8.astype(jnp.float32)) * jnp.float32(16.0),
                    -FP8_MAX,
                    FP8_MAX,
                ).astype(jnp.float8_e4m3fn)
                s_pack_f8 = jnp.concatenate([s_hi_f8, s_lo_f8], axis=1)
            else:
                row_amax = jnp.max(amax_128, axis=1, keepdims=True)
                row_sc = jnp.where(
                    row_amax == 0.0, 0.0, row_amax * jnp.float32(1.0 / 127.0)
                )
                inv_sc_128 = jnp.broadcast_to(
                    jnp.where(
                        row_amax == 0.0,
                        0.0,
                        jnp.float32(127.0) / jnp.maximum(row_amax, 1e-12),
                    ),
                    (tm_eff, 128),
                )
                s_scaled = jnp.clip(
                    row_sc * jnp.float32(S8_SCALE_PACK_MUL), 0.0, FP8_MAX
                )
                s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
                s_lo_f8 = jnp.clip(
                    (s_scaled - s_hi_f8.astype(jnp.float32)) * jnp.float32(16.0),
                    -FP8_MAX,
                    FP8_MAX,
                ).astype(jnp.float8_e4m3fn)
                s_pack_f8 = jnp.where(
                    is_lo,
                    jnp.broadcast_to(s_hi_f8, (tm_eff, 128)),
                    jnp.broadcast_to(s_lo_f8, (tm_eff, 128)),
                )
            for s in range(k_dim // 128):
                c0, c1 = s * 128, (s + 1) * 128
                xs = (x_ref[:, c0:c1] * ig_ref[:, c0:c1]).astype(jnp.float32)
                s8_out_ref[:, c0:c1] = jnp.rint(xs * inv_sc_128).astype(jnp.int8)
            s8_out_ref[:, k_dim : k_dim + 128] = lax.bitcast_convert_type(
                s_pack_f8, jnp.int8
            )

    kv_spec = (
        pl.BlockSpec((tp_deg, tm_eff, 1, 4, 128), lambda i, me_r: (0, i, 0, 0, 0))
        if rpa_layout
        else pl.BlockSpec((tp_deg, tm_eff, two_kv_size), lambda i, me_r: (0, i, 0))
    )
    in_specs = [
        pl.BlockSpec((tm_eff, k_dim), lambda i, me_r: (i, 0)),
        pl.BlockSpec((tp_deg, k_dim, two_kv_size), lambda i, me_r: (0, 0, 0)),
        pl.BlockSpec((tp_deg, 1, two_kv_size), lambda i, me_r: (0, 0, 0)),
        pl.BlockSpec((tm_eff, 128), lambda i, me_r: (me_r[0] * num_m_shard + i, 0)),
        pl.BlockSpec((tm_eff, 128), lambda i, me_r: (me_r[0] * num_m_shard + i, 0)),
    ]
    call_args = [x_bf16_shard, w_kv_3d, w_kv_sc_3d, cos_2d, sin_2d]
    if quant_s8:
        assert inv_gain_2d is not None
        in_specs.append(pl.BlockSpec((1, k_dim), lambda i, me_r: (0, 0)))
        call_args.append(inv_gain_2d)
        out_specs = (
            kv_spec,
            pl.BlockSpec((tm_eff, k_dim + 128), lambda i, me_r: (i, 0)),
        )
        out_shape = (
            jax.ShapeDtypeStruct((tp_deg, t_shard, *kv_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t_shard, k_dim + 128), jnp.int8),
        )
        name = (
            f"pallas_sp_kv_local_bf16_fkvq_{'s8str128' if strided_128 else 's8row'}"
            + ("_rpal" if rpa_layout else "")
        )
    else:
        out_specs = kv_spec
        out_shape = jax.ShapeDtypeStruct((tp_deg, t_shard, *kv_cols), jnp.float8_e4m3fn)
        name = "pallas_sp_kv_local_bf16_zc" + ("_rpal" if rpa_layout else "")

    res = pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_m_shard,),
            in_specs=in_specs,
            out_specs=out_specs,
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=name,
    )(me_1d, *call_args)
    return res if quant_s8 else (res, None)


def pallas_a1_qg_1way_fkv_zc(
    ag_2d: jax.Array,
    gain_2d: jax.Array | None,
    wqg_fp8: jax.Array,
    wqg_scale_2d: jax.Array,
    cos_2d: jax.Array,
    sin_2d: jax.Array,
    outs: tuple[jax.Array, jax.Array] | None,
    *,
    kv_local: jax.Array | None = None,
    s8_packed: bool = True,
    strided_128: bool = True,
    valid_tokens_1d: jax.Array | None = None,
    tp_deg: int = 8,
    t_shard: int,
    row_off: int,
    h_q_local: int,
    rpa_layout: bool = True,
    mode: str = "remote",
    tp_axis: str = "model",
    fold_q_scale: bool = False,
    k_scale: float = 1.0,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    vmem_limit_mb: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array | None]:
    """Zero-copy 1-Way `Q + Gate` kernel: 2D `ag_2d` (`tiled=True`), unsliced 2D `cos_2d`/`sin_2d`, unreshaped `kv_local`, and SMEM `valid_tokens_1d` skip."""
    h_sub = ag_2d.shape[0] // (1 if mode == "local" else tp_deg)
    in_cols = ag_2d.shape[1]
    k_dim = in_cols - 128 if s8_packed else in_cols
    _, n_dim = wqg_fp8.shape
    q_size = h_q_local * HEAD_DIM
    num_q_slices = q_size // 128
    tm_eff = next(
        t for t in (256, 128, 64, 32) if t <= tm and h_sub % t == 0 and row_off % t == 0
    )
    num_m = h_sub // tm_eff
    num_m_shard = t_shard // tm_eff
    off_blk = row_off // tm_eff
    n_ranks = 1 if mode == "local" else (tp_deg - 1 if mode == "remote" else tp_deg)
    total_steps = n_ranks * num_m
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    q_cols = (h_q_local // 2, 2, 128) if rpa_layout else (q_size,)
    fuse_kv = kv_local is not None
    has_gain = gain_2d is not None
    has_skip = valid_tokens_1d is not None
    use_pref = (mode != "all") or has_skip
    strip_k = 256 if tm_eff == 256 else 512

    def _run(*args):
        idx = 0
        pref_ref = args[idx] if use_pref else None
        idx += 1 if use_pref else 0
        x_ref = args[idx]
        idx += 1
        gain_ref = args[idx] if has_gain else None
        idx += 1 if has_gain else 0
        w_ref, w_sc_ref, cos_ref, sin_ref = args[idx : idx + 4]
        idx += 4
        idx += 2 if outs is not None else 0
        kv_src_ref = args[idx] if fuse_kv else None
        idx += 1 if fuse_kv else 0
        q_out_hbm, g_out_hbm = args[idx : idx + 2]
        idx += 2
        kv_dst_hbm = args[idx] if fuse_kv else None
        idx += 1 if fuse_kv else 0
        x_bf16_vmem = args[idx] if s8_packed else None
        idx += 1 if s8_packed else 0
        q_vmem, g_vmem, q_sems, g_sems = args[idx : idx + 4]
        idx += 4
        kv_sems = args[idx] if fuse_kv else None

        step = pl.program_id(0)
        r_idx = lax.div(step, num_m)
        m = lax.rem(step, num_m)
        slot = lax.rem(step, 2)
        dst_r = (
            r_idx
            if mode == "all"
            else (pref_ref[0] if mode == "local" else pref_ref[r_idx])
        )
        me_id = lax.axis_index(tp_axis)

        if fuse_kv:

            @pl.when(step == 0)
            def _start_kv_a2a():
                if tp_deg > 1:
                    barrier = pltpu.get_barrier_semaphore()
                    for d in range(1, tp_deg):
                        pl.semaphore_signal(
                            barrier, device_id={tp_axis: lax.rem(me_id + d, tp_deg)}
                        )
                    pl.semaphore_wait(barrier, tp_deg - 1)
                pltpu.make_async_copy(
                    kv_src_ref.at[pl.ds(me_id, 1)],
                    kv_dst_hbm.at[pl.ds(me_id, 1)],
                    kv_sems.at[0],
                ).start()
                for d in range(1, tp_deg):
                    peer_d = lax.rem(me_id + d, tp_deg)
                    pltpu.make_async_remote_copy(
                        src_ref=kv_src_ref.at[pl.ds(peer_d, 1)],
                        dst_ref=kv_dst_hbm.at[pl.ds(me_id, 1)],
                        send_sem=kv_sems.at[1],
                        recv_sem=kv_sems.at[0],
                        device_id={tp_axis: peer_d},
                    ).start()

        @pl.when(step >= 2)
        def _():
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(0, tm_eff)],
                q_sems.at[slot],
            ).wait()
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(0, tm_eff)],
                g_sems.at[slot],
            ).wait()

        def _compute_and_dma():
            if s8_packed:
                s_pack_f8 = lax.bitcast_convert_type(
                    x_ref[:, k_dim : k_dim + 128], jnp.float8_e4m3fn
                )
                if strided_128:
                    sc_64 = (
                        (
                            s_pack_f8[:, :64].astype(jnp.float32)
                            + s_pack_f8[:, 64:].astype(jnp.float32)
                            * jnp.float32(1.0 / 16.0)
                        )
                        * jnp.float32(S8_SCALE_UNPACK_MUL)
                    ).astype(jnp.bfloat16)
                    sc_strip = jnp.tile(
                        jnp.concatenate([sc_64, sc_64], axis=1), (1, strip_k // 128)
                    )
                else:
                    sc_strip = (
                        (
                            s_pack_f8[:, 0:1].astype(jnp.float32)
                            + s_pack_f8[:, 64:65].astype(jnp.float32)
                            * jnp.float32(1.0 / 16.0)
                        )
                        * jnp.float32(S8_SCALE_UNPACK_MUL)
                    ).astype(jnp.bfloat16)
                for c_blk in range(k_dim // strip_k):
                    c0, c1 = c_blk * strip_k, (c_blk + 1) * strip_k
                    deq = x_ref[:, c0:c1].astype(jnp.bfloat16) * sc_strip
                    x_bf16_vmem[:, c0:c1] = (
                        deq * gain_ref[:, c0:c1] if has_gain else deq
                    )
                lhs_dot = x_bf16_vmem[...]
            else:
                lhs_dot = x_ref[:, :]

            lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo
            acc = (
                jnp.dot(lhs_dot, w_ref[:, :], preferred_element_type=jnp.float32)
                * w_sc_ref[:, :]
            )
            dst_row = dst_r * t_shard + row_off + m * tm_eff
            g_vmem[slot, :, :] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(dst_row, tm_eff)],
                g_sems.at[slot],
            ).start()

            if rpa_layout:
                q_words = (
                    q_vmem.bitcast(jnp.uint32)
                    .at[slot]
                    .reshape(tm_eff * num_q_slices, 128)
                )
            for s in range(num_q_slices):
                c0, c1 = s * 128, (s + 1) * 128
                q_norm = _dual_head_rmsnorm_128(
                    acc[:, c0:c1], is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                )
                q_rot = _rope_128_vreg(
                    q_norm, cos_ref[:, :], sin_ref[:, :], lane, rope_mode=rope_mode
                )
                if rpa_layout:
                    q_words[pl.ds(s, tm_eff, stride=num_q_slices)] = _rpa_q_word(
                        q_rot, is_lo, mask_lo if s < num_q_slices // 2 else mask_hi
                    )
                else:
                    q_vmem[slot, :, c0:c1] = q_rot
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(dst_row, tm_eff)],
                q_sems.at[slot],
            ).start()

        if has_skip:
            vtok_idx = 0 if mode == "all" else (1 if mode == "local" else tp_deg - 1)
            valid_tok = pref_ref[vtok_idx]
            dst_row = dst_r * t_shard + row_off + m * tm_eff

            @pl.when(dst_row < valid_tok)
            def _():
                _compute_and_dma()

            @pl.when(dst_row >= valid_tok)
            def _():
                g_vmem[slot, :, :] = jnp.zeros((tm_eff, q_size), jnp.bfloat16)
                q_vmem[slot] = jnp.zeros_like(q_vmem[slot])
                pltpu.make_async_copy(
                    g_vmem.at[slot],
                    g_out_hbm.at[pl.ds(dst_row, tm_eff)],
                    g_sems.at[slot],
                ).start()
                pltpu.make_async_copy(
                    q_vmem.at[slot],
                    q_out_hbm.at[pl.ds(dst_row, tm_eff)],
                    q_sems.at[slot],
                ).start()
        else:
            _compute_and_dma()

        @pl.when(step == total_steps - 1)
        def _():
            for s_i in range(min(2, total_steps)):
                pltpu.make_async_copy(
                    q_vmem.at[s_i],
                    q_out_hbm.at[pl.ds(0, tm_eff)],
                    q_sems.at[s_i],
                ).wait()
                pltpu.make_async_copy(
                    g_vmem.at[s_i],
                    g_out_hbm.at[pl.ds(0, tm_eff)],
                    g_sems.at[s_i],
                ).wait()
            if fuse_kv:
                tmpl_r = pltpu.make_async_copy(
                    kv_dst_hbm.at[pl.ds(0, 1)],
                    kv_dst_hbm.at[pl.ds(0, 1)],
                    kv_sems.at[0],
                )
                for _ in range(tp_deg):
                    tmpl_r.wait()
                tmpl_s = pltpu.make_async_copy(
                    kv_src_ref.at[pl.ds(0, 1)],
                    kv_src_ref.at[pl.ds(0, 1)],
                    kv_sems.at[1],
                )
                for _ in range(tp_deg - 1):
                    tmpl_s.wait()

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    if not use_pref:
        in_specs = [pl.BlockSpec((tm_eff, in_cols), lambda step: (step, 0))]
        if has_gain:
            in_specs.append(pl.BlockSpec((1, k_dim), lambda step: (0, 0)))
        in_specs += [
            pl.BlockSpec((k_dim, n_dim), lambda step: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step: (0, 0)),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]
    else:
        r_fn = (
            (lambda step, pref: lax.div(step, num_m))
            if mode == "all"
            else (
                (lambda step, pref: pref[0])
                if mode == "local"
                else (lambda step, pref: pref[lax.div(step, num_m)])
            )
        )
        x_blk_fn = (
            (lambda step, pref: step)
            if mode in ("all", "local")
            else (
                lambda step, pref: (
                    pref[lax.div(step, num_m)] * num_m + lax.rem(step, num_m)
                )
            )
        )
        in_specs = [
            pl.BlockSpec(
                (tm_eff, in_cols), lambda step, pref: (x_blk_fn(step, pref), 0)
            )
        ]
        if has_gain:
            in_specs.append(pl.BlockSpec((1, k_dim), lambda step, pref: (0, 0)))
        in_specs += [
            pl.BlockSpec((k_dim, n_dim), lambda step, pref: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step, pref: (0, 0)),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step, pref: (
                    r_fn(step, pref) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step, pref: (
                    r_fn(step, pref) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]

    call_args = (
        [ag_2d]
        + ([gain_2d] if has_gain else [])
        + [wqg_fp8, wqg_scale_2d, cos_2d, sin_2d]
    )
    aliases: dict[int, int] = {}
    if outs is not None:
        base_idx = (1 if use_pref else 0) + len(call_args)
        aliases = {base_idx: 0, base_idx + 1: 1}
        in_specs += [hbm_spec, hbm_spec]
        call_args += [outs[0], outs[1]]
    if fuse_kv:
        in_specs.append(any_spec)
        call_args.append(kv_local)

    out_specs = [hbm_spec, hbm_spec] + ([any_spec] if fuse_kv else [])
    out_shapes = [
        pltpu.HBM((tp_deg * t_shard, *q_cols), jnp.bfloat16),
        pltpu.HBM((tp_deg * t_shard, q_size), jnp.bfloat16),
    ] + ([jax.ShapeDtypeStruct(kv_local.shape, kv_local.dtype)] if fuse_kv else [])
    scratches = (
        ([pltpu.VMEM((tm_eff, k_dim), jnp.bfloat16)] if s8_packed else [])
        + [
            pltpu.VMEM((2, tm_eff, *q_cols), jnp.bfloat16),
            pltpu.VMEM((2, tm_eff, q_size), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ]
        + ([pltpu.SemaphoreType.DMA((2,))] if fuse_kv else [])
    )
    tag = (
        ("s8str128" if strided_128 else "s8row")
        if s8_packed
        else ("fp8" if ag_2d.dtype == jnp.float8_e4m3fn else "bf16")
    )
    name = (
        f"pallas_a1_zc_{tag}_{mode}_off{row_off}_h{h_sub}_tm{tm_eff}"
        + ("_fkv" if fuse_kv else "")
        + ("_skip" if has_skip else "")
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        collective_id=19 if (fuse_kv and tp_deg > 1) else None,
        vmem_limit_bytes=vmem_limit_mb * (1 << 20),
    )

    if not use_pref:
        res = pl.pallas_call(
            _run,
            grid=(total_steps,),
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shapes,
            scratch_shapes=scratches,
            input_output_aliases=aliases,
            compiler_params=cp,
            name=name,
        )(*call_args)
    else:
        me_1d = pallas_axis_index_1d(tp_axis)
        me = me_1d[0]
        if mode == "all":
            pref = valid_tokens_1d.reshape(1).astype(jnp.int32)
        elif mode == "local":
            pref = (
                jnp.concatenate(
                    [me_1d, valid_tokens_1d.reshape(1).astype(jnp.int32)]
                )
                if has_skip
                else me_1d
            )
        else:
            idx_arr = jnp.arange(tp_deg - 1, dtype=jnp.int32)
            peers = jnp.where(idx_arr < me, idx_arr, idx_arr + 1)
            pref = (
                jnp.concatenate([peers, valid_tokens_1d.reshape(1).astype(jnp.int32)])
                if has_skip
                else peers
            )
        res = pl.pallas_call(
            _run,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(total_steps,),
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratches,
            ),
            out_shape=out_shapes,
            input_output_aliases=aliases,
            compiler_params=cp,
            name=name,
        )(pref, *call_args)
    return ((res[0], res[1]), res[2]) if fuse_kv else ((res[0], res[1]), None)


def pallas_a1_chunk_1way_zc(
    ag_2d: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale_2d: jax.Array,
    cos_2d: jax.Array,
    sin_2d: jax.Array,
    outs: tuple[jax.Array, jax.Array, jax.Array] | None,
    *,
    valid_tokens_1d: jax.Array | None = None,
    tp_deg: int = 8,
    t_shard: int,
    row_off: int,
    h_q_local: int,
    h_kv_local: int,
    rpa_layout: bool = True,
    fold_q_scale: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    vmem_limit_mb: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Tier 1 Lossless (`v5x4`) Zero-Copy 1-Way Chunk Kernel: 2D `ag_2d` (`tiled=True`), unsliced 2D `cos_2d`/`sin_2d`, direct 4D `KV`, and SMEM `valid_tokens_1d` skip."""
    assert h_kv_local == 2
    h_sub, k_dim = ag_2d.shape[0] // tp_deg, ag_2d.shape[1]
    _, n_dim = wqkvg_fp8.shape
    q_size, kv_size = h_q_local * HEAD_DIM, h_kv_local * HEAD_DIM
    num_q_slices = q_size // 128
    tm_eff = next(
        t for t in (256, 128, 64, 32) if t <= tm and h_sub % t == 0 and row_off % t == 0
    )
    num_m = h_sub // tm_eff
    num_m_shard = t_shard // tm_eff
    off_blk = row_off // tm_eff
    total_steps = tp_deg * num_m
    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale
    q_cols = (h_q_local // 2, 2, 128) if rpa_layout else (q_size,)
    kv_cols = (1, 4, 128) if rpa_layout else (2 * kv_size,)
    has_skip = valid_tokens_1d is not None

    def _run(*args):
        idx = 0
        pref_ref = args[idx] if has_skip else None
        idx += 1 if has_skip else 0
        x_ref, w_ref, w_sc_ref, cos_ref, sin_ref = args[idx : idx + 5]
        idx += 5
        idx += 3 if outs is not None else 0
        q_out_hbm, g_out_hbm, kv_out_hbm = args[idx : idx + 3]
        idx += 3
        q_vmem, g_vmem, kv_vmem, q_sems, g_sems, kv_sems = args[idx : idx + 6]

        step = pl.program_id(0)
        r_idx = lax.div(step, num_m)
        m = lax.rem(step, num_m)
        slot = lax.rem(step, 2)
        dst_row = r_idx * t_shard + row_off + m * tm_eff

        @pl.when(step >= 2)
        def _():
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(0, tm_eff)],
                q_sems.at[slot],
            ).wait()
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(0, tm_eff)],
                g_sems.at[slot],
            ).wait()
            pltpu.make_async_copy(
                kv_vmem.at[slot],
                kv_out_hbm.at[pl.ds(0, tm_eff)],
                kv_sems.at[slot],
            ).wait()

        def _compute_and_dma():
            lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo
            acc = (
                jnp.dot(x_ref[:, :], w_ref[:, :], preferred_element_type=jnp.float32)
                * w_sc_ref[:, :]
            )
            g_vmem[slot, :, :] = acc[:, q_size : 2 * q_size].astype(jnp.bfloat16)
            pltpu.make_async_copy(
                g_vmem.at[slot],
                g_out_hbm.at[pl.ds(dst_row, tm_eff)],
                g_sems.at[slot],
            ).start()

            if rpa_layout:
                q_words = (
                    q_vmem.bitcast(jnp.uint32)
                    .at[slot]
                    .reshape(tm_eff * num_q_slices, 128)
                )
            for s in range(num_q_slices):
                c0, c1 = s * 128, (s + 1) * 128
                q_norm = _dual_head_rmsnorm_128(
                    acc[:, c0:c1], is_lo, mask_lo, mask_hi, q_eff_scale, eps=1e-5
                )
                q_rot = _rope_128_vreg(
                    q_norm, cos_ref[:, :], sin_ref[:, :], lane, rope_mode=rope_mode
                )
                if rpa_layout:
                    q_words[pl.ds(s, tm_eff, stride=num_q_slices)] = _rpa_q_word(
                        q_rot, is_lo, mask_lo if s < num_q_slices // 2 else mask_hi
                    )
                else:
                    q_vmem[slot, :, c0:c1] = q_rot
            pltpu.make_async_copy(
                q_vmem.at[slot],
                q_out_hbm.at[pl.ds(dst_row, tm_eff)],
                q_sems.at[slot],
            ).start()

            k_pair = acc[:, 2 * q_size : 2 * q_size + kv_size]
            v_pair = acc[:, 2 * q_size + kv_size : 2 * q_size + 2 * kv_size].astype(
                jnp.bfloat16
            )
            k_norm = _dual_head_rmsnorm_128(
                k_pair, is_lo, mask_lo, mask_hi, 1.0, eps=1e-5
            )
            k_rot = _rope_128_vreg(
                k_norm, cos_ref[:, :], sin_ref[:, :], lane, rope_mode=rope_mode
            )
            k8 = jnp.clip(
                k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
            v8 = jnp.clip(
                v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
            ).astype(jnp.float8_e4m3fn)
            if rpa_layout:
                kv_vmem.bitcast(jnp.uint32).at[slot].reshape(tm_eff, 128)[...] = (
                    _f8_code_u32(k8) | (_f8_code_u32(v8) << 8)
                )
            else:
                kv_vmem[slot, :, :128] = k8
                kv_vmem[slot, :, 128:256] = v8
            pltpu.make_async_copy(
                kv_vmem.at[slot],
                kv_out_hbm.at[pl.ds(dst_row, tm_eff)],
                kv_sems.at[slot],
            ).start()

        if has_skip:
            valid_tok = pref_ref[0]

            @pl.when(dst_row < valid_tok)
            def _():
                _compute_and_dma()

            @pl.when(dst_row >= valid_tok)
            def _():
                g_vmem[slot, :, :] = jnp.zeros((tm_eff, q_size), jnp.bfloat16)
                q_vmem[slot] = jnp.zeros_like(q_vmem[slot])
                kv_vmem[slot] = jnp.zeros_like(kv_vmem[slot])
                pltpu.make_async_copy(
                    g_vmem.at[slot],
                    g_out_hbm.at[pl.ds(dst_row, tm_eff)],
                    g_sems.at[slot],
                ).start()
                pltpu.make_async_copy(
                    q_vmem.at[slot],
                    q_out_hbm.at[pl.ds(dst_row, tm_eff)],
                    q_sems.at[slot],
                ).start()
                pltpu.make_async_copy(
                    kv_vmem.at[slot],
                    kv_out_hbm.at[pl.ds(dst_row, tm_eff)],
                    kv_sems.at[slot],
                ).start()
        else:
            _compute_and_dma()

        @pl.when(step == total_steps - 1)
        def _():
            for s_i in range(min(2, total_steps)):
                pltpu.make_async_copy(
                    q_vmem.at[s_i],
                    q_out_hbm.at[pl.ds(0, tm_eff)],
                    q_sems.at[s_i],
                ).wait()
                pltpu.make_async_copy(
                    g_vmem.at[s_i],
                    g_out_hbm.at[pl.ds(0, tm_eff)],
                    g_sems.at[s_i],
                ).wait()
                pltpu.make_async_copy(
                    kv_vmem.at[s_i],
                    kv_out_hbm.at[pl.ds(0, tm_eff)],
                    kv_sems.at[s_i],
                ).wait()

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if not has_skip:
        in_specs = [
            pl.BlockSpec((tm_eff, k_dim), lambda step: (step, 0)),
            pl.BlockSpec((k_dim, n_dim), lambda step: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step: (0, 0)),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step: (
                    lax.div(step, num_m) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]
    else:
        in_specs = [
            pl.BlockSpec((tm_eff, k_dim), lambda step, pref: (step, 0)),
            pl.BlockSpec((k_dim, n_dim), lambda step, pref: (0, 0)),
            pl.BlockSpec((1, n_dim), lambda step, pref: (0, 0)),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step, pref: (
                    lax.div(step, num_m) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
            pl.BlockSpec(
                (tm_eff, 128),
                lambda step, pref: (
                    lax.div(step, num_m) * num_m_shard + off_blk + lax.rem(step, num_m),
                    0,
                ),
            ),
        ]
    call_args = [ag_2d, wqkvg_fp8, wqkvg_scale_2d, cos_2d, sin_2d]
    aliases: dict[int, int] = {}
    if outs is not None:
        base_idx = (1 if has_skip else 0) + 5
        aliases = {base_idx: 0, base_idx + 1: 1, base_idx + 2: 2}
        in_specs += [hbm_spec, hbm_spec, hbm_spec]
        call_args += [outs[0], outs[1], outs[2]]
    out_shapes = [
        pltpu.HBM((tp_deg * t_shard, *q_cols), jnp.bfloat16),
        pltpu.HBM((tp_deg * t_shard, q_size), jnp.bfloat16),
        pltpu.HBM((tp_deg * t_shard, *kv_cols), jnp.float8_e4m3fn),
    ]
    scratches = [
        pltpu.VMEM((2, tm_eff, *q_cols), jnp.bfloat16),
        pltpu.VMEM((2, tm_eff, q_size), jnp.bfloat16),
        pltpu.VMEM((2, tm_eff, *kv_cols), jnp.float8_e4m3fn),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    name = f"pallas_a1_1way_zc_off{row_off}_h{h_sub}_tm{tm_eff}" + (
        "_skip" if has_skip else ""
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=vmem_limit_mb * (1 << 20),
    )
    if not has_skip:
        return pl.pallas_call(
            _run,
            grid=(total_steps,),
            in_specs=in_specs,
            out_specs=[hbm_spec] * 3,
            out_shape=out_shapes,
            scratch_shapes=scratches,
            input_output_aliases=aliases,
            compiler_params=cp,
            name=name,
        )(*call_args)
    return pl.pallas_call(
        _run,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(total_steps,),
            in_specs=in_specs,
            out_specs=[hbm_spec] * 3,
            scratch_shapes=scratches,
        ),
        out_shape=out_shapes,
        input_output_aliases=aliases,
        compiler_params=cp,
        name=name,
    )(valid_tokens_1d.reshape(1).astype(jnp.int32), *call_args)


# -----------------------------------------------------------------------------
# Wave 9: Static-FP8 Ulysses Sequence-Parallel Attention (`v10_sp`, TP=8 / TP=16)
# -----------------------------------------------------------------------------
def pallas_a1_ulysses_qkv_local(
    x_fp8_shard: jax.Array,
    w_qkv_3d: jax.Array,
    w_qkv_sc_3d: jax.Array,
    cos_own: jax.Array,
    sin_own: jax.Array,
    *,
    h_q_local: int = 8,
    h_kv_local: int = 1,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tm: int = 256,
    sub_n: int | None = None,
    vmem_limit_mb: int = 48,
) -> tuple[jax.Array, jax.Array]:
    """Local Ulysses A1 QKV+Norm+RoPE on token shard (`c_k` rows x all `tp_deg` head groups).

    Multiplies `x_fp8_shard [c_k, H]` once per `t_blk <= 1024` block against per-head-rank
    weights `w_qkv_3d [tp_deg, H, Q + 2*KV]` (`[Q | K | V]`, `w_qkv_sc_3d` pre-multiplied by
    `STATIC_A1_ACT_SCALE`), applies dual-head RMSNorm + RoPE using rank-local `cos_own/sin_own`,
    and returns unpacked `(q_3d: bf16[tp_deg, c_k, Q], kv_3d: fp8[tp_deg, c_k, 2*KV])` ready for
    direct 2-op `lax.all_to_all` exchange before RPA packing.
    """
    assert h_kv_local in (1, 2) and h_q_local % 2 == 0
    c_k, k_dim = x_fp8_shard.shape
    tp_deg, w_k, qkv_cols = w_qkv_3d.shape
    assert w_k == k_dim
    q_size = h_q_local * HEAD_DIM
    kv_cols = 2 * h_kv_local * HEAD_DIM
    assert qkv_cols == q_size + kv_cols
    num_q_slices = q_size // 128
    eff_vmem_mb = max(vmem_limit_mb, 60) if qkv_cols > 768 else vmem_limit_mb

    if c_k <= 1024:
        t_blk = c_k
    else:
        t_blk = next(
            (t for t in range(1024, 0, -64) if c_k % t == 0),
            next(t for t in (1024, 512, 256, 128, 64, 32, 16, 8) if c_k % t == 0),
        )
    tm_eff = next(
        t for t in (256, 128, 64, 32, 16, 8) if t <= tm and t_blk % t == 0
    )
    num_blk = c_k // t_blk
    num_m_blk = t_blk // tm_eff

    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale

    from sglang_tpu.envs import envs as _tpu_envs

    if sub_n is None:
        sub_n_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_SUB_N", None)
        eff_sub_n = int(sub_n_env.get() or 0) if sub_n_env is not None else 0
    else:
        eff_sub_n = int(sub_n)
    if eff_sub_n not in (128, 256, 512) or q_size % eff_sub_n != 0:
        eff_sub_n = 0
    fn_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM", None)
    use_fast_norm = bool(fn_env is not None and fn_env.get())

    def _ulysses_qkv_kernel(x_ref, w_ref, w_sc_ref, cos_ref, sin_ref, q_out_ref, kv_out_ref):
        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo

        for m in range(num_m_blk):
            m0 = m * tm_eff
            m1 = m0 + tm_eff
            x_m = x_ref[m0:m1, :]
            cos_m = cos_ref[m0:m1, :]
            sin_m = sin_ref[m0:m1, :]

            if eff_sub_n > 0:
                for c0 in range(0, q_size, eff_sub_n):
                    c1 = c0 + eff_sub_n
                    acc_q = (
                        jnp.dot(
                            x_m,
                            w_ref[0, :, c0:c1],
                            preferred_element_type=jnp.float32,
                        )
                        * w_sc_ref[0, :, c0:c1]
                    )
                    for s_loc in range(eff_sub_n // 128):
                        d0 = s_loc * 128
                        d1 = d0 + 128
                        q_norm = _dual_head_rmsnorm_128(
                            acc_q[:, d0:d1],
                            is_lo,
                            mask_lo,
                            mask_hi,
                            q_eff_scale,
                            eps=1e-5,
                            fast_6roll=use_fast_norm,
                        )
                        q_out_ref[0, m0:m1, c0 + d0 : c0 + d1] = _rope_128_vreg(
                            q_norm, cos_m, sin_m, lane, rope_mode=rope_mode
                        )
                acc_kv = (
                    jnp.dot(
                        x_m,
                        w_ref[0, :, q_size : q_size + kv_cols],
                        preferred_element_type=jnp.float32,
                    )
                    * w_sc_ref[0, :, q_size : q_size + kv_cols]
                )
            else:
                acc = (
                    jnp.dot(x_m, w_ref[0, :, :], preferred_element_type=jnp.float32)
                    * w_sc_ref[0, :, :]
                )
                for s in range(num_q_slices):
                    c0, c1 = s * 128, (s + 1) * 128
                    q_norm = _dual_head_rmsnorm_128(
                        acc[:, c0:c1],
                        is_lo,
                        mask_lo,
                        mask_hi,
                        q_eff_scale,
                        eps=1e-5,
                        fast_6roll=use_fast_norm,
                    )
                    q_out_ref[0, m0:m1, c0:c1] = _rope_128_vreg(
                        q_norm, cos_m, sin_m, lane, rope_mode=rope_mode
                    )
                acc_kv = acc[:, q_size : q_size + kv_cols]

            if h_kv_local == 1:
                cos_kv = jnp.where(is_lo, cos_m, 1.0)
                sin_kv = jnp.where(is_lo, sin_m, 0.0)
                sq_lo = jnp.sum(acc_kv * acc_kv * mask_lo, axis=1, keepdims=True) * (
                    1.0 / 64.0
                )
                inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                kv_norm = (
                    (acc_kv * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                )
                kv_rot = _rope_128_vreg(
                    kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                )
                kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                kv_out_ref[0, m0:m1, :] = jnp.clip(
                    kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
            else:
                k_pair = acc_kv[:, 0:128]
                v_pair = acc_kv[:, 128:256].astype(jnp.bfloat16)
                k_norm = _dual_head_rmsnorm_128(
                    k_pair,
                    is_lo,
                    mask_lo,
                    mask_hi,
                    1.0,
                    eps=1e-5,
                    fast_6roll=use_fast_norm,
                )
                k_rot = _rope_128_vreg(
                    k_norm, cos_m, sin_m, lane, rope_mode=rope_mode
                )
                kv_out_ref[0, m0:m1, 0:128] = jnp.clip(
                    k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)
                kv_out_ref[0, m0:m1, 128:256] = jnp.clip(
                    v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                ).astype(jnp.float8_e4m3fn)

    return pl.pallas_call(
        _ulysses_qkv_kernel,
        grid=(num_blk, tp_deg),
        in_specs=[
            pl.BlockSpec((t_blk, k_dim), lambda blk, r: (blk, 0)),
            pl.BlockSpec((1, k_dim, qkv_cols), lambda blk, r: (r, 0, 0)),
            pl.BlockSpec((1, 1, qkv_cols), lambda blk, r: (r, 0, 0)),
            pl.BlockSpec((t_blk, 128), lambda blk, r: (blk, 0)),
            pl.BlockSpec((t_blk, 128), lambda blk, r: (blk, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, t_blk, q_size), lambda blk, r: (r, blk, 0)),
            pl.BlockSpec((1, t_blk, kv_cols), lambda blk, r: (r, blk, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tp_deg, c_k, q_size), jnp.bfloat16),
            jax.ShapeDtypeStruct((tp_deg, c_k, kv_cols), jnp.float8_e4m3fn),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=eff_vmem_mb * (1 << 20),
        ),
        name=f"pallas_a1_ulysses_qkv_c{c_k}_q{q_size}_kv{kv_cols}_tb{t_blk}_tm{tm_eff}"
        + (f"_sn{eff_sub_n}" if eff_sub_n > 0 else "")
        + ("_fn" if use_fast_norm else ""),
    )(x_fp8_shard, w_qkv_3d, w_qkv_sc_3d, cos_own, sin_own)


def pallas_pack_ulysses_qkv_for_rpa(
    q_2d: jax.Array,
    kv_2d: jax.Array,
    *,
    h_q_local: int = 8,
    h_kv_local: int = 1,
    tm: int = 512,
    vmem_limit_mb: int = 32,
) -> tuple[jax.Array, jax.Array]:
    """Packs post-A2A `q_2d [c_tok, Q]` & `kv_2d [c_tok, 2*KV]` into RPA v3 `rpa_layout`."""
    assert h_kv_local in (1, 2) and h_q_local % 2 == 0
    c_tok, q_size = q_2d.shape
    kv_cols = 2 * h_kv_local * HEAD_DIM
    assert q_size == h_q_local * HEAD_DIM and kv_2d.shape == (c_tok, kv_cols)
    num_q_slices = q_size // 128
    tm_eff = next(
        t for t in (1024, 512, 256, 128, 64, 32, 16, 8) if t <= tm and c_tok % t == 0
    )

    def _pack_kernel(q_in_ref, kv_in_ref, q_out_ref, kv_out_ref):
        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo

        q_words = (
            q_out_ref.bitcast(jnp.uint32)
            .at[0]
            .reshape(tm_eff * num_q_slices, 128)
        )
        for s in range(num_q_slices):
            kv_head0 = (h_kv_local == 1) or (s < num_q_slices // 2)
            q_words[pl.ds(s, tm_eff, stride=num_q_slices)] = _rpa_q_word(
                q_in_ref[:, s * 128 : (s + 1) * 128],
                is_lo,
                mask_lo if kv_head0 else mask_hi,
            )

        if h_kv_local == 1:
            kv_u32 = _f8_code_u32(kv_in_ref[...])
            zero_u32 = jnp.uint32(0)
            k_code = jnp.where(is_lo, kv_u32, zero_u32)
            v_code = jnp.where(is_lo, pltpu.roll(kv_u32, shift=64, axis=1), zero_u32)
            kv_word = k_code | (v_code << 8)
        else:
            k8 = kv_in_ref[:, 0:128]
            v8 = kv_in_ref[:, 128:256]
            kv_word = _f8_code_u32(k8) | (_f8_code_u32(v8) << 8)
        kv_out_ref.bitcast(jnp.uint32).at[0].reshape(tm_eff, 128)[...] = kv_word

    q_5d, kv_buf = pl.pallas_call(
        _pack_kernel,
        grid=(c_tok // tm_eff,),
        in_specs=[
            pl.BlockSpec((tm_eff, q_size), lambda i: (i, 0)),
            pl.BlockSpec((tm_eff, kv_cols), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, tm_eff, num_q_slices, 2, 128), lambda i: (0, i, 0, 0, 0)),
            pl.BlockSpec((1, tm_eff, 4, 128), lambda i: (0, i, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((1, c_tok, num_q_slices, 2, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, c_tok, 4, 128), jnp.float8_e4m3fn),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=f"pallas_pack_ulysses_rpa_c{c_tok}_q{q_size}_kv{kv_cols}_tm{tm_eff}",
    )(q_2d, kv_2d)
    return q_5d, kv_buf.reshape(c_tok, 1, 4, 128)


def pallas_ulysses_local_gate(
    x_fp8_shard: jax.Array,
    w_gate_fp8: jax.Array,
    w_gate_sc_2d: jax.Array,
    *,
    tm: int = 256,
    bn: int = 1024,
    vmem_limit_mb: int = 48,
) -> jax.Array:
    """Token-local Gate projection (`[c_k, H] fp8 @ [H, H] fp8 -> bf16[c_k, H]`).

    Keeps each `t_blk <= 1024` input block resident in VMEM across output column tiles `bn`
    so `x_fp8_shard` and `w_gate_fp8` are each read once from HBM.
    """
    c_k, k_dim = x_fp8_shard.shape
    w_k, n_dim = w_gate_fp8.shape
    assert w_k == k_dim
    eff_bn = next(b for b in (bn, 1024, 512, 256, 128) if b <= n_dim and n_dim % b == 0)

    if c_k <= 1024:
        t_blk = c_k
    else:
        t_blk = next(
            (t for t in range(1024, 0, -64) if c_k % t == 0),
            next(t for t in (1024, 512, 256, 128, 64, 32, 16, 8) if c_k % t == 0),
        )
    tm_eff = next(
        t for t in (256, 128, 64, 32, 16, 8) if t <= tm and t_blk % t == 0
    )
    num_blk = c_k // t_blk
    num_j = n_dim // eff_bn

    def _gate_kernel(x_ref, w_ref, w_sc_ref, out_ref):
        sc = w_sc_ref[...]
        for m0 in range(0, t_blk, tm_eff):
            m1 = m0 + tm_eff
            acc = jnp.dot(x_ref[m0:m1, :], w_ref[...], preferred_element_type=jnp.float32)
            out_ref[m0:m1, :] = (acc * sc).astype(jnp.bfloat16)

    return pl.pallas_call(
        _gate_kernel,
        grid=(num_blk, num_j),
        in_specs=[
            pl.BlockSpec((t_blk, k_dim), lambda blk, j: (blk, 0)),
            pl.BlockSpec((k_dim, eff_bn), lambda blk, j: (0, j)),
            pl.BlockSpec((1, eff_bn), lambda blk, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((t_blk, eff_bn), lambda blk, j: (blk, j)),
        out_shape=jax.ShapeDtypeStruct((c_k, n_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=f"pallas_ulysses_gate_c{c_k}_n{n_dim}_tb{t_blk}_bn{eff_bn}",
    )(x_fp8_shard, w_gate_fp8, w_gate_sc_2d)


def pallas_a1_ulysses_qkv_gate_async_a2a(
    x_fp8_shard: jax.Array,
    w_qkv_3d: jax.Array,
    w_qkv_sc_3d: jax.Array,
    w_gate_fp8: jax.Array,
    w_gate_sc_2d: jax.Array,
    cos_own: jax.Array,
    sin_own: jax.Array,
    *,
    h_q_local: int = 8,
    h_kv_local: int = 1,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
    rope_mode: str = "interleaved_roll",
    tp_axis: str = "model",
    max_q_slots: int = 16,
    vmem_limit_mb: int = 62,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused A1 Ulysses QKV+Norm+RoPE + async P2P A2A overlapped with local Gate.

    Reads ``x_fp8_shard`` once per ``t_blk <= 1024`` block and interleaves per-rank Gate
    slices ``d`` with XOR peer QKV+Norm+RoPE + ``make_async_remote_copy`` A2A pushes
    using up to ``max_q_slots`` (16) VMEM send slots so ICI transfers overlap MXU compute
    at 0-ULP bit-exact parity with ``pallas_a1_ulysses_qkv_local`` + ``lax.all_to_all`` +
    ``pallas_ulysses_local_gate``.
    """
    assert h_kv_local in (1, 2) and h_q_local % 2 == 0
    c_k, k_dim = x_fp8_shard.shape
    tp_deg, w_k, qkv_cols = w_qkv_3d.shape
    wg_k, n_dim = w_gate_fp8.shape
    assert w_k == k_dim and wg_k == k_dim
    q_size = h_q_local * HEAD_DIM
    kv_cols = 2 * h_kv_local * HEAD_DIM
    assert qkv_cols == q_size + kv_cols and n_dim == tp_deg * q_size
    assert q_size <= 512 and c_k >= 128 and c_k % 128 == 0

    num_q_slices = q_size // 128
    if c_k >= 1536 and c_k % 768 == 0:
        t_blk = 768
    else:
        t_blk = next(
            t for t in range(min(1024, c_k), 0, -64) if c_k % t == 0 and t % 128 == 0
        )
    num_blk = c_k // t_blk
    tm_eff = next(t for t in (256, 128) if t_blk % t == 0)
    num_q_slots = min(max_q_slots, tp_deg)

    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    inv_k_scale = 1.0 / k_scale
    inv_v_scale = 1.0 / v_scale

    from sglang_tpu.envs import envs as _tpu_envs

    fn_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM", None)
    use_fast_norm = bool(fn_env is not None and fn_env.get())

    def _kernel(
        x_hbm,
        wqkv_hbm,
        wqkv_sc_ref,
        wg_hbm,
        wg_sc_ref,
        cos_hbm,
        sin_hbm,
        q_out_hbm,
        kv_out_hbm,
        g_out_hbm,
        x_vmem,
        cos_vmem,
        sin_vmem,
        wqkv_vmem,
        wg_vmem,
        q_send_vmem,
        kv_send_vmem,
        g_vmem,
        x_sem,
        wq_sems,
        wg_sems,
        q_rsems,
        kv_rsems,
        q_local_sem,
        kv_local_sem,
        q_ssems,
        kv_ssems,
        g_sems,
    ):
        blk = pl.program_id(0)
        me = lax.axis_index(tp_axis)
        row0 = blk * t_blk

        pltpu.make_async_copy(x_hbm.at[pl.ds(row0, t_blk), :], x_vmem, x_sem.at[0]).start()
        pltpu.make_async_copy(cos_hbm.at[pl.ds(row0, t_blk), :], cos_vmem, x_sem.at[1]).start()
        pltpu.make_async_copy(sin_hbm.at[pl.ds(row0, t_blk), :], sin_vmem, x_sem.at[2]).start()
        pltpu.make_async_copy(
            wqkv_hbm.at[pl.ds(me, 1), :, :],
            wqkv_vmem.at[pl.ds(0, 1), :, :],
            wq_sems.at[0],
        ).start()
        pltpu.make_async_copy(
            wg_hbm.at[:, pl.ds(0, q_size)],
            wg_vmem.at[pl.ds(0, k_dim), :],
            wg_sems.at[0],
        ).start()

        if tp_deg > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, tp_deg):
                pl.semaphore_signal(barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)})
            pl.semaphore_wait(barrier, tp_deg - 1)

        pltpu.make_async_copy(x_vmem, x_hbm.at[pl.ds(0, t_blk), :], x_sem.at[0]).wait()
        pltpu.make_async_copy(cos_vmem, cos_hbm.at[pl.ds(0, t_blk), :], x_sem.at[1]).wait()
        pltpu.make_async_copy(sin_vmem, sin_hbm.at[pl.ds(0, t_blk), :], x_sem.at[2]).wait()

        lane = lax.broadcasted_iota(jnp.int32, (tm_eff, 128), 1)
        is_lo = lane < 64
        mask_lo = is_lo.astype(jnp.float32)
        mask_hi = 1.0 - mask_lo

        for d in range(tp_deg):
            wslot = d % 2
            qslot = d % num_q_slots
            r_cur = me if d == 0 else lax.bitwise_xor(me, d)
            if d + 1 < tp_deg:
                nslot = (d + 1) % 2
                r_nxt = lax.bitwise_xor(me, d + 1)
                pltpu.make_async_copy(
                    wqkv_hbm.at[pl.ds(r_nxt, 1), :, :],
                    wqkv_vmem.at[pl.ds(nslot, 1), :, :],
                    wq_sems.at[nslot],
                ).start()
                pltpu.make_async_copy(
                    wg_hbm.at[:, pl.ds((d + 1) * q_size, q_size)],
                    wg_vmem.at[pl.ds(nslot * k_dim, k_dim), :],
                    wg_sems.at[nslot],
                ).start()

            if d == num_q_slots and num_q_slots < tp_deg:
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(0, 1), :, :],
                    q_out_hbm.at[pl.ds(0, 1), pl.ds(0, t_blk), :],
                    q_local_sem.at[0],
                ).wait()
                pltpu.make_async_copy(
                    kv_send_vmem.at[pl.ds(0, 1), :, :],
                    kv_out_hbm.at[pl.ds(0, 1), pl.ds(0, t_blk), :],
                    kv_local_sem.at[0],
                ).wait()
            elif d > num_q_slots and num_q_slots < tp_deg:
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(qslot, 1), :, :],
                    q_send_vmem.at[pl.ds(qslot, 1), :, :],
                    q_ssems.at[qslot],
                ).wait()
                pltpu.make_async_copy(
                    kv_send_vmem.at[pl.ds(qslot, 1), :, :],
                    kv_send_vmem.at[pl.ds(qslot, 1), :, :],
                    kv_ssems.at[qslot],
                ).wait()

            pltpu.make_async_copy(
                wqkv_vmem.at[pl.ds(wslot, 1), :, :],
                wqkv_hbm.at[pl.ds(0, 1), :, :],
                wq_sems.at[wslot],
            ).wait()

            w_qkv_tile = wqkv_vmem[wslot, :, :]
            w_sc_vec = wqkv_sc_ref[r_cur, :, :]
            for m0 in range(0, t_blk, tm_eff):
                m1 = m0 + tm_eff
                cos_m = cos_vmem[m0:m1, :]
                sin_m = sin_vmem[m0:m1, :]
                acc = (
                    jnp.dot(
                        x_vmem[m0:m1, :],
                        w_qkv_tile,
                        preferred_element_type=jnp.float32,
                    )
                    * w_sc_vec
                )
                for s in range(num_q_slices):
                    c0, c1 = s * 128, (s + 1) * 128
                    q_norm = _dual_head_rmsnorm_128(
                        acc[:, c0:c1],
                        is_lo,
                        mask_lo,
                        mask_hi,
                        q_eff_scale,
                        eps=1e-5,
                        fast_6roll=use_fast_norm,
                    )
                    q_send_vmem[qslot, m0:m1, c0:c1] = _rope_128_vreg(
                        q_norm, cos_m, sin_m, lane, rope_mode=rope_mode
                    )

                acc_kv = acc[:, q_size : q_size + kv_cols]
                if h_kv_local == 1:
                    cos_kv = jnp.where(is_lo, cos_m, 1.0)
                    sin_kv = jnp.where(is_lo, sin_m, 0.0)
                    sq_lo = jnp.sum(acc_kv * acc_kv * mask_lo, axis=1, keepdims=True) * (
                        1.0 / 64.0
                    )
                    inv_rms_kv = jnp.where(is_lo, lax.rsqrt(sq_lo + 1e-5), 1.0)
                    kv_norm = (
                        (acc_kv * inv_rms_kv).astype(jnp.bfloat16).astype(jnp.float32)
                    )
                    kv_rot = _rope_128_vreg(
                        kv_norm, cos_kv, sin_kv, lane, rope_mode=rope_mode
                    )
                    kv_sc = jnp.where(is_lo, inv_k_scale, inv_v_scale)
                    kv_send_vmem[qslot, m0:m1, :] = jnp.clip(
                        kv_rot.astype(jnp.float32) * kv_sc, -FP8_MAX, FP8_MAX
                    ).astype(jnp.float8_e4m3fn)
                else:
                    k_pair = acc_kv[:, 0:128]
                    v_pair = acc_kv[:, 128:256].astype(jnp.bfloat16)
                    k_norm = _dual_head_rmsnorm_128(
                        k_pair,
                        is_lo,
                        mask_lo,
                        mask_hi,
                        1.0,
                        eps=1e-5,
                        fast_6roll=use_fast_norm,
                    )
                    k_rot = _rope_128_vreg(
                        k_norm, cos_m, sin_m, lane, rope_mode=rope_mode
                    )
                    kv_send_vmem[qslot, m0:m1, 0:128] = jnp.clip(
                        k_rot.astype(jnp.float32) * inv_k_scale, -FP8_MAX, FP8_MAX
                    ).astype(jnp.float8_e4m3fn)
                    kv_send_vmem[qslot, m0:m1, 128:256] = jnp.clip(
                        v_pair.astype(jnp.float32) * inv_v_scale, -FP8_MAX, FP8_MAX
                    ).astype(jnp.float8_e4m3fn)

            if d == 0:
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(0, 1), :, :],
                    q_out_hbm.at[pl.ds(me, 1), pl.ds(row0, t_blk), :],
                    q_local_sem.at[0],
                ).start()
                pltpu.make_async_copy(
                    kv_send_vmem.at[pl.ds(0, 1), :, :],
                    kv_out_hbm.at[pl.ds(me, 1), pl.ds(row0, t_blk), :],
                    kv_local_sem.at[0],
                ).start()
            else:
                peer = lax.bitwise_xor(me, d)
                pltpu.make_async_remote_copy(
                    src_ref=q_send_vmem.at[pl.ds(qslot, 1), :, :],
                    dst_ref=q_out_hbm.at[pl.ds(me, 1), pl.ds(row0, t_blk), :],
                    send_sem=q_ssems.at[qslot],
                    recv_sem=q_rsems.at[me],
                    device_id={tp_axis: peer},
                ).start()
                pltpu.make_async_remote_copy(
                    src_ref=kv_send_vmem.at[pl.ds(qslot, 1), :, :],
                    dst_ref=kv_out_hbm.at[pl.ds(me, 1), pl.ds(row0, t_blk), :],
                    send_sem=kv_ssems.at[qslot],
                    recv_sem=kv_rsems.at[me],
                    device_id={tp_axis: peer},
                ).start()

            pltpu.make_async_copy(
                wg_vmem.at[pl.ds(wslot * k_dim, k_dim), :],
                wg_hbm.at[:, pl.ds(0, q_size)],
                wg_sems.at[wslot],
            ).wait()
            if d >= 2:
                pltpu.make_async_copy(
                    g_vmem.at[pl.ds(wslot * t_blk, t_blk), :],
                    g_out_hbm.at[pl.ds(0, t_blk), pl.ds(0, q_size)],
                    g_sems.at[wslot],
                ).wait()
            w_g_tile = wg_vmem[pl.ds(wslot * k_dim, k_dim), :]
            wg_sc_vec = wg_sc_ref[:, d * q_size : (d + 1) * q_size]
            for m0 in range(0, t_blk, tm_eff):
                m1 = m0 + tm_eff
                g_acc = (
                    jnp.dot(
                        x_vmem[m0:m1, :],
                        w_g_tile,
                        preferred_element_type=jnp.float32,
                    )
                    * wg_sc_vec
                )
                g_vmem[pl.ds(wslot * t_blk + m0, tm_eff), :] = g_acc.astype(
                    jnp.bfloat16
                )
            pltpu.make_async_copy(
                g_vmem.at[pl.ds(wslot * t_blk, t_blk), :],
                g_out_hbm.at[pl.ds(row0, t_blk), pl.ds(d * q_size, q_size)],
                g_sems.at[wslot],
            ).start()

        for s in range(min(2, tp_deg)):
            pltpu.make_async_copy(
                g_vmem.at[pl.ds(s * t_blk, t_blk), :],
                g_out_hbm.at[pl.ds(0, t_blk), pl.ds(0, q_size)],
                g_sems.at[s],
            ).wait()
        if num_q_slots >= tp_deg:
            pltpu.make_async_copy(
                q_send_vmem.at[pl.ds(0, 1), :, :],
                q_out_hbm.at[pl.ds(0, 1), pl.ds(0, t_blk), :],
                q_local_sem.at[0],
            ).wait()
            pltpu.make_async_copy(
                kv_send_vmem.at[pl.ds(0, 1), :, :],
                kv_out_hbm.at[pl.ds(0, 1), pl.ds(0, t_blk), :],
                kv_local_sem.at[0],
            ).wait()
        for s in range(num_q_slots):
            if s != 0 or tp_deg > num_q_slots:
                pltpu.make_async_copy(
                    q_send_vmem.at[pl.ds(s, 1), :, :],
                    q_send_vmem.at[pl.ds(s, 1), :, :],
                    q_ssems.at[s],
                ).wait()
                pltpu.make_async_copy(
                    kv_send_vmem.at[pl.ds(s, 1), :, :],
                    kv_send_vmem.at[pl.ds(s, 1), :, :],
                    kv_ssems.at[s],
                ).wait()
        for r in range(tp_deg):

            @pl.when(r != me)
            def _(r=r):
                pltpu.make_async_copy(
                    q_out_hbm.at[pl.ds(r, 1), pl.ds(row0, t_blk), :],
                    q_out_hbm.at[pl.ds(r, 1), pl.ds(row0, t_blk), :],
                    q_rsems.at[r],
                ).wait()
                pltpu.make_async_copy(
                    kv_out_hbm.at[pl.ds(r, 1), pl.ds(row0, t_blk), :],
                    kv_out_hbm.at[pl.ds(r, 1), pl.ds(row0, t_blk), :],
                    kv_rsems.at[r],
                ).wait()

    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    scratches = [
        pltpu.VMEM((t_blk, k_dim), jnp.float8_e4m3fn),
        pltpu.VMEM((t_blk, 128), jnp.float32),
        pltpu.VMEM((t_blk, 128), jnp.float32),
        pltpu.VMEM((2, k_dim, qkv_cols), jnp.float8_e4m3fn),
        pltpu.VMEM((2 * k_dim, q_size), jnp.float8_e4m3fn),
        pltpu.VMEM((num_q_slots, t_blk, q_size), jnp.bfloat16),
        pltpu.VMEM((num_q_slots, t_blk, kv_cols), jnp.float8_e4m3fn),
        pltpu.VMEM((2 * t_blk, q_size), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((3,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((tp_deg,)),
        pltpu.SemaphoreType.DMA((tp_deg,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((num_q_slots,)),
        pltpu.SemaphoreType.DMA((num_q_slots,)),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    return pl.pallas_call(
        _kernel,
        grid=(num_blk,),
        in_specs=[
            any_spec,
            any_spec,
            pl.BlockSpec((tp_deg, 1, qkv_cols), lambda _: (0, 0, 0)),
            any_spec,
            pl.BlockSpec((1, n_dim), lambda _: (0, 0)),
            any_spec,
            any_spec,
        ],
        out_specs=[any_spec, any_spec, any_spec],
        out_shape=[
            jax.ShapeDtypeStruct((tp_deg, c_k, q_size), jnp.bfloat16),
            jax.ShapeDtypeStruct((tp_deg, c_k, kv_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((c_k, n_dim), jnp.bfloat16),
        ],
        scratch_shapes=scratches,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=7 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=f"pallas_a1_ulysses_qkvg_async_a2a_c{c_k}_q{q_size}_kv{kv_cols}_tb{t_blk}_ns{num_q_slots}"
        + ("_fn" if use_fast_norm else ""),
    )(
        x_fp8_shard,
        w_qkv_3d,
        w_qkv_sc_3d,
        w_gate_fp8,
        w_gate_sc_2d,
        cos_own,
        sin_own,
    )
