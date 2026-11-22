# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas decode QKV epilogue for Muse Spark (HANDOFF D4).

Fuses the ~12 per-layer XLA ops between ``qkv_proj`` and ``RPAd`` on decode
(``B <= 256``) into a single pure-VMEM Pallas kernel with zero 64-wide head
relayouts:

  1. ``qkv.split([1024, 128, 128])`` (``16`` Q heads, ``2`` K heads, ``2`` V heads, ``d=64``)
  2. Per-head ``fp32`` RMSNorm on Q (``* query_scale -> bf16``) and K (``-> bf16``)
  3. Optional interleaved ``fp32`` RoPE on Q and K (``-> bf16``)
  4. ``KV_PACK_HEADS=1`` diagonal head-group lane placement into ``bf16[B, 16, 128]``
  5. Per-tensor ``fp8_e4m3fn`` quantization of K and V + ``merge_kv`` packing into
     ``fp8_e4m3fn[B_pad32, 1, 4, 128]``
"""

import functools

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.util import align_to, get_tpu_version

_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
_FP8_MIN = float(jnp.finfo(jnp.float8_e4m3fn).min)


def build_decode_rope_step_tables(
    positions: jax.Array,  # i32[B]
    cos_sin_pairs: jax.Array,  # f32[ceil(max_pos / 2), 128]
) -> jax.Array:
    """Builds ``fp32[B_pad32, 2, 128]`` ``[cos_128, sin_signed_128]`` once per step."""
    max_pos_2 = cos_sin_pairs.shape[0]
    pos_clamped = jnp.clip(positions, 0, max_pos_2 * 2 - 1)
    rows = cos_sin_pairs[pos_clamped // 2]  # [B, 128]
    cos_sin = jnp.where((pos_clamped % 2 == 0)[:, None], rows[:, :64], rows[:, 64:])
    cos_32 = cos_sin[:, :32]
    sin_32 = cos_sin[:, 32:]
    cos_64 = jnp.repeat(cos_32, 2, axis=-1)
    sin_64 = jnp.repeat(sin_32, 2, axis=-1)
    sign_64 = jnp.where((jnp.arange(64) & 1) == 0, -1.0, 1.0).astype(jnp.float32)[
        None, :
    ]
    sin_s64 = sin_64 * sign_64
    cos_128 = jnp.concatenate([cos_64, cos_64], axis=-1)
    sin_s128 = jnp.concatenate([sin_s64, sin_s64], axis=-1)
    stacked = jnp.stack([cos_128, sin_s128], axis=1)  # [B, 2, 128]
    b_tok = stacked.shape[0]
    b_pad = align_to(max(b_tok, 1), 32)
    if b_tok < b_pad:
        stacked = jnp.pad(stacked, ((0, b_pad - b_tok), (0, 0), (0, 0)))
    return stacked


def _broadcast_minor_128(x: jax.Array) -> jax.Array:
    if x.shape[-1] == 128:
        return x
    return jnp.broadcast_to(x, (*x.shape[:-1], 128))


def _roll_lanes(x: jax.Array, shift: int, *, interpret_mode: bool) -> jax.Array:
    if interpret_mode:
        return jnp.roll(x, shift=shift, axis=1)
    return pltpu.roll(x, shift=shift, axis=1)


def _decode_qkv_epilogue_kernel(
    qkv_vmem_ref,  # bf16[B_pad32, 10, 128]
    cos_sin_vmem_ref,  # f32[B_pad32, 2, 128] or None
    q_out_vmem_ref,  # bf16[B_pad32, 16, 128]
    kv_out_vmem_ref,  # fp8[B_pad32, 1, 4, 128]
    *,
    norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
):
    b_pad = qkv_vmem_ref.shape[0]
    tile_r = 64 if (b_pad % 64 == 0) else 32
    n_tiles = b_pad // tile_r
    lane_iota = lax.broadcasted_iota(jnp.int32, (tile_r, 128), 1)
    mask_lo = lane_iota < 64
    even_lane = (lane_iota & 1) == 0
    inv_k_scale = 1.0 / float(k_scale)
    inv_v_scale = 1.0 / float(v_scale)

    def norm_pair(x_bf16: jax.Array, scale: float) -> jax.Array:
        x = x_bf16.astype(jnp.float32)
        x2 = x * x
        ms_lo = _broadcast_minor_128(
            jnp.sum(jnp.where(mask_lo, x2, 0.0), axis=1, keepdims=True)
        ) * (1.0 / 64.0)
        ms_hi = _broadcast_minor_128(
            jnp.sum(jnp.where(mask_lo, 0.0, x2), axis=1, keepdims=True)
        ) * (1.0 / 64.0)
        inv_rms = jnp.where(
            mask_lo, lax.rsqrt(ms_lo + norm_eps), lax.rsqrt(ms_hi + norm_eps)
        )
        y = x * inv_rms
        if scale != 1.0:
            y = y * scale
        return y.astype(jnp.bfloat16)

    def rope_pair(
        x_bf16: jax.Array, cos_128: jax.Array, sin_s128: jax.Array
    ) -> jax.Array:
        xv = x_bf16.astype(jnp.float32)
        partner = jnp.where(
            even_lane,
            _roll_lanes(xv, 127, interpret_mode=interpret_mode),
            _roll_lanes(xv, 1, interpret_mode=interpret_mode),
        )
        return (xv * cos_128 + partner * sin_s128).astype(jnp.bfloat16)

    def quant_fp8_u8(x_bf16: jax.Array, inv_scale: float) -> jax.Array:
        xf = x_bf16.astype(jnp.float32)
        if inv_scale != 1.0:
            xf = xf * inv_scale
        f8 = jnp.clip(xf, _FP8_MIN, _FP8_MAX).astype(jnp.float8_e4m3fn)
        if interpret_mode:
            return lax.bitcast_convert_type(f8, jnp.uint8)
        return pltpu.bitcast(f8, jnp.uint8)

    for t_i in range(n_tiles):
        b0 = t_i * tile_r
        if has_rope:
            cos_128 = cos_sin_vmem_ref[pl.ds(b0, tile_r), 0, :]
            sin_s128 = cos_sin_vmem_ref[pl.ds(b0, tile_r), 1, :]
        else:
            cos_128 = None
            sin_s128 = None

        z_bf16 = jnp.zeros((tile_r, 128), jnp.bfloat16)
        # Process 8 Q pairs -> 16 packed Q heads in [B_pad32, 16, 128]
        for p_idx in range(8):
            qp = qkv_vmem_ref[pl.ds(b0, tile_r), p_idx, :]
            qp = norm_pair(qp, query_scale)
            if has_rope:
                qp = rope_pair(qp, cos_128, sin_s128)
            qp_swapped = _roll_lanes(qp, 64, interpret_mode=interpret_mode)
            if p_idx < 4:
                # Heads 0..7 -> lanes 0..63, zeros in 64..127
                h_even = jnp.where(mask_lo, qp, z_bf16)
                h_odd = jnp.where(mask_lo, qp_swapped, z_bf16)
            else:
                # Heads 8..15 -> lanes 64..127, zeros in 0..63
                h_even = jnp.where(mask_lo, z_bf16, qp_swapped)
                h_odd = jnp.where(mask_lo, z_bf16, qp)
            q_out_vmem_ref[pl.ds(b0, tile_r), 2 * p_idx, :] = h_even
            q_out_vmem_ref[pl.ds(b0, tile_r), 2 * p_idx + 1, :] = h_odd

        # K (pair 8) + V (pair 9) -> fp8 quant + merge_kv [tile_r, 1, 4, 128]
        kp = qkv_vmem_ref[pl.ds(b0, tile_r), 8, :]
        vp = qkv_vmem_ref[pl.ds(b0, tile_r), 9, :]
        kp = norm_pair(kp, 1.0)
        if has_rope:
            kp = rope_pair(kp, cos_128, sin_s128)

        k_u8 = quant_fp8_u8(kp, inv_k_scale)
        v_u8 = quant_fp8_u8(vp, inv_v_scale)
        kv_u32 = k_u8.astype(jnp.uint32) | (v_u8.astype(jnp.uint32) << 8)
        if interpret_mode:
            z_u8 = jnp.zeros((tile_r, 128), jnp.uint8)
            packed_u8 = jnp.stack([k_u8, v_u8, z_u8, z_u8], axis=1)[:, None, :, :]
            kv_out_vmem_ref[pl.ds(b0, tile_r)] = lax.bitcast_convert_type(
                packed_u8, jnp.float8_e4m3fn
            )
        else:
            kv_out_u32_ref = kv_out_vmem_ref.bitcast(jnp.uint32).reshape(b_pad, 128)
            kv_out_u32_ref[pl.ds(b0, tile_r)] = kv_u32


def run_decode_qkv_epilogue(
    qkv: jax.Array,  # bf16[B, 1280]
    cos_sin_step: jax.Array | None = None,  # f32[B_pad32, 2, 128] or None
    *,
    norm_eps: float = 1e-6,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Runs fused decode QKV epilogue -> ``(q_packed [B, 16, 128], kv_merged [B_pad32, 1, 4, 128])``."""
    b_tok = qkv.shape[0]
    assert qkv.shape[1] == 1280 and qkv.dtype == jnp.bfloat16
    b_pad = align_to(max(b_tok, 1), 32)
    qkv_3d = qkv.reshape(b_tok, 10, 128)
    if b_tok < b_pad:
        qkv_3d = jnp.pad(qkv_3d, ((0, b_pad - b_tok), (0, 0), (0, 0)))

    has_rope = cos_sin_step is not None
    in_specs = [
        pl.BlockSpec(qkv_3d.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
        (
            pl.BlockSpec(
                cos_sin_step.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM
            )
            if has_rope
            else None
        ),
    ]
    out_specs = [
        pl.BlockSpec((b_pad, 16, 128), lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
        pl.BlockSpec(
            (b_pad, 1, 4, 128), lambda *_: (0, 0, 0, 0), memory_space=pltpu.VMEM
        ),
    ]
    out_shape = [
        jax.ShapeDtypeStruct(shape=(b_pad, 16, 128), dtype=jnp.bfloat16),
        jax.ShapeDtypeStruct(shape=(b_pad, 1, 4, 128), dtype=jnp.float8_e4m3fn),
    ]

    scope = f"muse_decode_qkv_epi_b{b_pad}_{'rope' if has_rope else 'nope'}"
    pallas_kw = {
        "grid_spec": pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(1,),
        ),
        "out_shape": out_shape,
        "name": scope,
    }
    if interpret:
        pallas_kw["interpret"] = pltpu.InterpretParams(
            detect_races=True, uninitialized_memory="nan"
        )
    else:
        pallas_kw["compiler_params"] = pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        )

    q_pad, kv_merged = pl.pallas_call(
        functools.partial(
            _decode_qkv_epilogue_kernel,
            norm_eps=float(norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            has_rope=has_rope,
            interpret_mode=interpret,
        ),
        **pallas_kw,
    )(qkv_3d, cos_sin_step)

    return q_pad[:b_tok], (kv_merged[:b_tok] if b_tok < b_pad else kv_merged)
