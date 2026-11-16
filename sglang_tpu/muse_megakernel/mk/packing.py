# SPDX-License-Identifier: Apache-2.0
"""bf16 / fp8 / int8 row packing into u32 words and per-row fp8 scaling."""

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE


def _pack_xfull_256(x):
    """Pack f32 ``[M, He]`` into u32 ``[M, He // 2]`` in 256-col (128-word) blocks."""
    nb = x.shape[-1] // (2 * LANE)
    lo = jnp.concatenate(
        [x[..., b * 2 * LANE : (2 * b + 1) * LANE] for b in range(nb)], axis=-1
    )
    hi = jnp.concatenate(
        [x[..., (2 * b + 1) * LANE : (2 * b + 2) * LANE] for b in range(nb)],
        axis=-1,
    )
    return _bf16_bits(lo) | (_bf16_bits(hi) << 16)


def _unpack_xfull_256_chunk(w, out_fp8: bool = False):
    """Unpack ``n_w`` (multiple of LANE) u32 cols packed by ``_pack_xfull_256``."""
    nb = w.shape[-1] // LANE
    parts = [_unpack_bf16x2(w[..., b * LANE : (b + 1) * LANE]) for b in range(nb)]
    res = parts[0] if nb == 1 else jnp.concatenate(parts, axis=-1)
    return res.astype(jnp.float8_e4m3fn) if out_fp8 else res.astype(jnp.bfloat16)


def _bf16_bits(x):
    """f32 -> its round-to-nearest-even bf16 bit pattern in the low 16 bits of a u32."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint32)
    return (b + jnp.uint32(0x7FFF) + ((b >> 16) & jnp.uint32(1))) >> 16


def _pack_bf16x2(x):
    """f32 ``[..., 2n]`` -> u32 ``[..., n]``: bf16 of the first half low, of the second half high.

    Halves a collective payload without a 16-row bf16 tile on 8-row buffers.
    """
    n = x.shape[-1] // 2
    return _bf16_bits(x[..., :n]) | (_bf16_bits(x[..., n:]) << 16)


def _unpack_bf16x2(w):
    lo = jax.lax.bitcast_convert_type(w << 16, jnp.float32)
    hi = jax.lax.bitcast_convert_type(w & jnp.uint32(0xFFFF0000), jnp.float32)
    return jnp.concatenate([lo, hi], axis=-1)


def y_words(c: MKConfig) -> int:
    """u32 words per (row, slot) expert partial in ysend / pad_y."""
    return (c.expert_hidden // 4 + LANE) if c.y_quant else (c.expert_hidden // 2)


def _row_amax(ax):
    """Row-wise max of non-negative 2D tensor [M, K] using 128-lane VPU tree before 1 XLU max."""
    if ax.ndim == 2 and ax.shape[1] > LANE and ax.shape[1] % LANE == 0:
        n_tiles = ax.shape[1] // LANE
        tiles = [ax[:, k * LANE : (k + 1) * LANE] for k in range(n_tiles)]
        while len(tiles) > 1:
            tiles = [
                jnp.maximum(tiles[2 * i], tiles[2 * i + 1])
                if 2 * i + 1 < len(tiles)
                else tiles[2 * i]
                for i in range((len(tiles) + 1) // 2)
            ]
        return jnp.max(tiles[0], axis=1, keepdims=True)
    return jnp.max(ax, axis=1, keepdims=True)


def _scale_fp8_range(x, nonneg: bool = False):
    """Per-row scale of x [M, K] into [-448, 448] f32 without fp8 cast -> (x_scaled, x_sc [M, 1])."""
    x32 = x if x.dtype == jnp.float32 else x.astype(jnp.float32)
    amax = _row_amax(x32 if nonneg else jnp.abs(x32))
    sc = amax * jnp.float32(1.0 / 448.0)
    # Route sc through select so libtpu does not rematerialize amax / 448 at every consumer.
    sc = jnp.where(amax == 0, jnp.zeros_like(sc), sc)
    inv = jnp.where(
        amax == 0,
        jnp.zeros_like(amax),
        jnp.float32(448.0) / jnp.maximum(amax, jnp.float32(1e-30)),
    )
    return x32 * inv, sc


def _quant_fp8_rows(x, nonneg: bool = False):
    """Per-row dynamic float8_e4m3fn quantization of x [M, K] -> (x_fp8, x_sc [M, 1] f32)."""
    x_scaled, sc = _scale_fp8_range(x, nonneg=nonneg)
    return x_scaled.astype(jnp.float8_e4m3fn), sc


def _pack_int8_scaled(x):
    """f32/bf16 [..., 4n] -> u32 [..., n + LANE]: 4 signed int8s per u32 + per-lane (block-32) f32 scales."""
    n = x.shape[-1] // 4
    n_lanes = (4 * n) // LANE
    amax = jnp.abs(x[..., :LANE]).astype(jnp.float32)
    for k in range(1, n_lanes):
        amax = jnp.maximum(
            amax, jnp.abs(x[..., k * LANE : (k + 1) * LANE]).astype(jnp.float32)
        )
    sc = amax * jnp.float32(1.0 / 127.0)
    sc = jnp.where(amax == 0, jnp.zeros_like(sc), sc)
    inv = jnp.where(
        amax == 0,
        jnp.zeros_like(amax),
        jnp.float32(127.0) / jnp.maximum(amax, jnp.float32(1e-30)),
    )
    inv_n = inv if n == LANE else jnp.concatenate([inv] * (n // LANE), axis=-1)
    words = None
    for b in range(4):
        xb = x[..., b * n : (b + 1) * n]
        xb32 = xb if xb.dtype == jnp.float32 else xb.astype(jnp.float32)
        qb = jnp.clip(jnp.rint(xb32 * inv_n), -127.0, 127.0).astype(jnp.int32)
        ub = jax.lax.bitcast_convert_type(qb, jnp.uint32) & jnp.uint32(0xFF)
        words = ub if b == 0 else (words | (ub << (8 * b)))
    sc_u32 = jax.lax.bitcast_convert_type(sc, jnp.uint32)
    return jnp.concatenate([words, sc_u32], axis=-1)


def _unpack_int8_scaled(w):
    """u32 [..., n + LANE] -> f32 [..., 4n] using lane-replicated scale (zero XLU ops)."""
    n = w.shape[-1] - LANE
    words = jax.lax.bitcast_convert_type(w[..., :n], jnp.int32)
    sc_tile = jax.lax.bitcast_convert_type(w[..., n : n + LANE], jnp.float32)
    sc_n = sc_tile if n == LANE else jnp.concatenate([sc_tile] * (n // LANE), axis=-1)
    q0 = ((words << 24) >> 24).astype(jnp.float32) * sc_n
    q1 = ((words << 16) >> 24).astype(jnp.float32) * sc_n
    q2 = ((words << 8) >> 24).astype(jnp.float32) * sc_n
    q3 = (words >> 24).astype(jnp.float32) * sc_n
    return jnp.concatenate([q0, q1, q2, q3], axis=-1)


def _pack_y(x, c: MKConfig):
    if c.y_quant:
        return _pack_int8_scaled(x)
    return _pack_bf16x2(x if x.dtype == jnp.float32 else x.astype(jnp.float32))


def _unpack_y(w, c: MKConfig):
    return _unpack_int8_scaled(w) if c.y_quant else _unpack_bf16x2(w)
