"""Exact pure-JAX Muse Spark reference implementations for Attention-Side Prefill Kernels.

Matches `sglang_tpu/models/muse_spark.py` numerics line-for-line:
  - input_layernorm: eps=1e-5, gain = weight.float() + 1.0, cast to bf16
  - q_norm & k_norm: weightless per-head RMSNorm (head_dim=64, eps=1e-5);
    Q scaled by QUERY_SCALE = 5.4730064863875 before .astype(bf16) and before RoPE
  - Interleaved RoPE: MuseSparkInterleavedRope (even/odd adjacent pairs in fp32 -> bf16)
  - KV quantization: static/channel scale -> clip([-448, 448]) -> float8_e4m3fn
  -KV cache layout: SEQ_ALONG_LANE (num_pages, H_kv_local * 2, 16, 4, 256)
  - Kernel A3: gated = sigmoid(gate.float()).to(bf16) * normed_attn_out -> o_proj -> TP sum/RS
  - Entry (Kernel 0): embed lookup -> TP sum/RS -> rms_norm(eps=1e-5).to(bf16).float()
  - Exit: last_index gather -> model.norm(gain_center=0.0, eps=1e-5) -> lm_head * 0.17677669529663687 -> 20 * tanh(x / 20)
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

HIDDEN_SIZE = 8192
NUM_Q_HEADS_GLOBAL = 128
NUM_KV_HEADS_GLOBAL = 16
HEAD_DIM = 64
QUERY_SCALE = 5.4730064863875
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)  # 0.125
LOG2E = 1.4426950408889634
LM_HEAD_SCALE = 0.17677669529663687  # 1 / sqrt(32)
LOGIT_CAP = 20.0
FP8_MAX = 448.0


def rms_normalize_fp32(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Pure FP32 RMS normalization matching `_rms_normalize` in `muse_spark.py`."""
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    return xf * lax.rsqrt(var + eps)


def input_layernorm_ref(
    stream: jax.Array, weight: jax.Array, eps: float = 1e-5
) -> jax.Array:
    """MuseSparkRMSNorm(eps=1e-5, gain_center=1.0, scale_input=False)."""
    normed = rms_normalize_fp32(stream, eps=eps)
    gain = weight.astype(jnp.float32) + 1.0
    return (normed * gain).astype(jnp.bfloat16)


def per_head_normalize_ref(
    x: jax.Array, head_dim: int = HEAD_DIM, eps: float = 1e-5
) -> jax.Array:
    """Weightless per-head RMSNorm in FP32 matching `_per_head_normalize`."""
    orig_shape = x.shape
    xh = x.reshape(*orig_shape[:-1], -1, head_dim)
    return rms_normalize_fp32(xh, eps=eps).reshape(orig_shape)


def build_rope_tables_ref(
    max_pos: int = 32768,
    head_dim: int = HEAD_DIM,
    rope_theta: float = 500000.0,
) -> tuple[jax.Array, jax.Array]:
    """Builds (cos_table, sin_table) of shape [max_pos, head_dim // 2] in FP32."""
    inv_freq = 1.0 / (
        rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim))
    )
    t = jnp.arange(max_pos, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def build_batch_cos_sin_128(
    positions: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Builds 128-lane VREG-ready RoPE tables once per step for [T] positions.

    Returns:
      batch_cos_sin: fp32[T, 128] where [:64] is cos_interleaved and [64:128] is sin_interleaved
        (compatible with `mk/numerics.py:_rope_roll`).
      cos_128: fp32[T, 128] with 64-wide interleaved cos duplicated across both 64-lane halves.
      sin_signed_128: fp32[T, 128] with 64-wide signed interleaved sin ([-sin0, +sin0, ...])
        duplicated across both 64-lane halves, so `x * cos_128 + partner * sin_signed_128`
        rotates two 64-dim heads in a single 128-lane VREG op without a separate sign multiply.
    """
    cos = cos_table[positions]  # [T, 32]
    sin = sin_table[positions]  # [T, 32]
    cos_64 = jnp.repeat(cos, 2, axis=-1)  # [T, 64]
    sin_64 = jnp.repeat(sin, 2, axis=-1)  # [T, 64]
    batch_cos_sin = jnp.concatenate([cos_64, sin_64], axis=-1)  # [T, 128]

    sign_64 = jnp.where((jnp.arange(64) & 1) == 0, -1.0, 1.0).astype(jnp.float32)[
        None, :
    ]
    sin_signed_64 = sin_64 * sign_64
    cos_128 = jnp.concatenate([cos_64, cos_64], axis=-1)
    sin_signed_128 = jnp.concatenate([sin_signed_64, sin_signed_64], axis=-1)
    return batch_cos_sin, cos_128, sin_signed_128


def interleaved_rope_ref(
    q: jax.Array,
    k: jax.Array,
    positions: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    head_dim: int = HEAD_DIM,
) -> tuple[jax.Array, jax.Array]:
    """Exact MuseSparkInterleavedRope forward pass in FP32 -> BF16."""
    cos = cos_table[positions][:, None, :]  # [T, 1, 32]
    sin = sin_table[positions][:, None, :]  # [T, 1, 32]

    def _rotate(x: jax.Array) -> jax.Array:
        t_len = x.shape[0]
        pairs = x.astype(jnp.float32).reshape(t_len, -1, head_dim // 2, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated = jnp.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1)
        return rotated.reshape(x.shape).astype(jnp.bfloat16)

    return _rotate(q), _rotate(k)


def quantize_fp8_static_ref(x: jax.Array, inv_scale: float = 1.0) -> jax.Array:
    """Quantizes BF16/FP32 tensor to float8_e4m3fn with static scale."""
    scaled = jnp.clip(x.astype(jnp.float32) * inv_scale, -FP8_MAX, FP8_MAX)
    return scaled.astype(jnp.float8_e4m3fn)


def pack_kv_seq_along_lane_ref(
    k_fp8: jax.Array,
    v_fp8: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    page_size: int = 256,
) -> jax.Array:
    """Packs token-major K/V [T, H_kv_local * 64] into SEQ_ALONG_LANE kv_cache.

    kv_cache shape: (num_pages, H_kv_local * 2, 16, 4, page_size)
      where minor axis is page_size (256) and sublanes hold (16, 4) = 64 head_dim.
    """
    t_len = k_fp8.shape[0]
    num_pages_used = t_len // page_size
    h_kv_local = k_fp8.shape[1] // HEAD_DIM

    k_pages = k_fp8.reshape(num_pages_used, page_size, h_kv_local, 16, 4)
    v_pages = v_fp8.reshape(num_pages_used, page_size, h_kv_local, 16, 4)
    # Interleave K and V along the head axis: [num_pages_used, page_size, h_kv_local * 2, 16, 4]
    kv_interleaved = jnp.stack([k_pages, v_pages], axis=3).reshape(
        num_pages_used, page_size, h_kv_local * 2, 16, 4
    )
    # Transpose to SEQ_ALONG_LANE: (num_pages_used, h_kv_local * 2, 16, 4, page_size)
    kv_pages_sal = jnp.transpose(kv_interleaved, (0, 2, 3, 4, 1))
    return kv_cache.at[page_indices[:num_pages_used]].set(kv_pages_sal)


def unpack_kv_seq_along_lane_ref(
    kv_cache: jax.Array,
    page_indices: jax.Array,
    num_tokens: int,
    h_kv_local: int,
    page_size: int = 256,
) -> tuple[jax.Array, jax.Array]:
    """Unpacks SEQ_ALONG_LANE kv_cache pages back to token-major K, V [T, H_kv_local * 64]."""
    num_pages_used = num_tokens // page_size
    pages = kv_cache[page_indices[:num_pages_used]]  # [P, H_kv*2, 16, 4, 256]
    # Transpose to [P, 256, H_kv*2, 16, 4] -> [T, H_kv, 2, 64]
    unpacked = jnp.transpose(pages, (0, 4, 1, 2, 3)).reshape(
        num_tokens, h_kv_local, 2, HEAD_DIM
    )
    k_out = unpacked[:, :, 0, :].reshape(num_tokens, h_kv_local * HEAD_DIM)
    v_out = unpacked[:, :, 1, :].reshape(num_tokens, h_kv_local * HEAD_DIM)
    return k_out, v_out


def kernel_a1_reference(
    stream_full: jax.Array,
    norm_weight: jax.Array,
    wqkvg_fp8: jax.Array,
    wqkvg_scale: jax.Array,
    positions: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    kv_cache: jax.Array,
    page_indices: jax.Array,
    h_q_local: int,
    h_kv_local: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fold_q_scale: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Exact pure-JAX reference for Kernel A1.

    wqkvg_fp8 column layout: [Q (h_q_local*64) | Gate (h_q_local*64) | K (h_kv_local*64) | V (h_kv_local*64)].
    Returns: (q_rot, gate, k_fp8, v_fp8, updated_kv_cache).
    """
    q_size = h_q_local * HEAD_DIM
    kv_size = h_kv_local * HEAD_DIM

    attn_in = input_layernorm_ref(stream_full, norm_weight, eps=1e-5)
    qkvg = (
        jnp.dot(
            attn_in,
            wqkvg_fp8,
            preferred_element_type=jnp.float32,
        )
        * wqkvg_scale[None, :]
    ).astype(jnp.bfloat16)

    q_raw = qkvg[:, :q_size]
    gate = qkvg[:, q_size : 2 * q_size]
    k_raw = qkvg[:, 2 * q_size : 2 * q_size + kv_size]
    v_raw = qkvg[:, 2 * q_size + kv_size : 2 * q_size + 2 * kv_size]

    q_eff_scale = (
        QUERY_SCALE * SM_SCALE * k_scale * LOG2E if fold_q_scale else QUERY_SCALE
    )
    q_normed = (per_head_normalize_ref(q_raw, eps=1e-5) * q_eff_scale).astype(
        jnp.bfloat16
    )
    k_normed = per_head_normalize_ref(k_raw, eps=1e-5).astype(jnp.bfloat16)

    q_rot, k_rot = interleaved_rope_ref(
        q_normed, k_normed, positions, cos_table, sin_table
    )
    k_fp8 = quantize_fp8_static_ref(k_rot, inv_scale=1.0 / k_scale)
    v_fp8 = quantize_fp8_static_ref(v_raw, inv_scale=1.0 / v_scale)
    kv_cache_out = pack_kv_seq_along_lane_ref(k_fp8, v_fp8, kv_cache, page_indices)
    return q_rot, gate, k_fp8, v_fp8, kv_cache_out


def kernel_a3_local_reference(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_fp8: jax.Array,
    wo_scale: jax.Array,
) -> jax.Array:
    """Single-rank reference for Kernel A3 prologue + o_proj before collective."""
    gated = (
        jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16) * normed_attn_out
    )
    o_partial = (
        jnp.dot(gated, wo_fp8, preferred_element_type=jnp.float32) * wo_scale[None, :]
    ).astype(jnp.bfloat16)
    return o_partial
