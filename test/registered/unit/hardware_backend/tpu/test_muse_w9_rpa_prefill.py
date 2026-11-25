"""Wave-9 Prefill RPA (`w9-k-pre-rpa`) unit, numerical parity, and TPU7x compile/export tests.

Covers:
1. Default-OFF flag preservation (`SGLANG_TPU_RPA_V3_PREFILL_W9`, `SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q`)
   and AST guard forbidding bare `EnvField` truthiness checks across touched files.
2. `_invoke_v3_kernel` & `TpuPallasRpaV3Backend` routing:
   - `w9` / `SGLANG_TPU_RPA_V3_PREFILL_W9=1` enables `mk128`, `fp8_qk`, `prefix_fast`, `sw_short_fuse`,
     `sw2048_fuse`, and `>32K` `bq4096/bkv8192/staged_qo`, even when `LC_OPTS`/`W7_OPTS` carry legacy W8 flags.
   - `LC_OPTS` (`vreg_acc1,mrb1`) stays scoped to Full-Attn `>32K` and never bleeds onto SWA layers.
   - `TP=16` (`8` Q heads / `1` KV head, `d=64`) prepared `(1, T, 4, 2, 128)` -> `2D [T, 512]` `a3_out_layout`
     and unprepared `(T, 8, 64)` routing, while preserving legacy `TP16_NATIVE_PREFILL` when `tp16_sq` is off.
3. Exact JAX numerical parity vs FP32 reference across `TP=8` (`16:2` GQA) and `TP=16` (`8:1` GQA):
   - Zero-concat `dense2=True` and `dense2=False` `M_k=128` KV slice unpacking (`0x7F` NaN-poisoned tail).
   - Pipelined `_eval_mk128_head` (`none` `fori_loop`, `sw9_fast` `ut0/ut1 + 14*none + lt0/lt1`, `sw_short_fuse`,
     and dynamic-mask fallback) + `[T, 1024]` / `[T, 512]` / packed-u32 epilogues.
   - Extreme outlier Q/K stress test (`#2468`, `exp2 > bf16 max` where stock `bf16` `l/acc` overflows to `NaN`).
4. TPU7x (`2x2x1` TP=8 and `2x2x2` TP=16) Mosaic `jax.export` and offline `libtpu` `.compile()` verification.
"""

from __future__ import annotations

import ast
import functools
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
from jax import export as jax_export
from jax import lax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[5]
_TOUCHED_FILES = (
    "sglang_tpu/envs.py",
    "sglang_tpu/attention_pallas_rpa_v3.py",
    "sglang_tpu/attention/pallas/ragged_paged_attention_v3.py",
)


def _u8_to_fp8(arr_u8: np.ndarray) -> jax.Array:
    return lax.bitcast_convert_type(jnp.asarray(arr_u8, dtype=jnp.uint8), jnp.float8_e4m3fn)


def _fp8_to_u8(arr_f8: jax.Array) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(arr_f8, jnp.uint8))


def _rand_fp8_u8(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.4) -> np.ndarray:
    vals = np.clip(rng.standard_normal(shape) * scale, -448.0, 448.0).astype(np.float32)
    return _fp8_to_u8(jnp.asarray(vals, dtype=jnp.float8_e4m3fn))


def _pack_q_tp8_5d(q_16_64: np.ndarray) -> jax.Array:
    """Pack `[T, 16, 64]` into Muse TP=8 prepared `(1, T, 8, 2, 128)` bf16."""
    t = q_16_64.shape[0]
    q_16_128 = np.zeros((t, 16, 128), dtype=np.float32)
    q_16_128[:, :8, :64] = q_16_64[:, :8, :]
    q_16_128[:, 8:, 64:] = q_16_64[:, 8:, :]
    return jnp.asarray(q_16_128, dtype=jnp.bfloat16).reshape(1, t, 8, 2, 128)


def _pack_q_tp16_5d(q_8_64: np.ndarray) -> jax.Array:
    """Pack `[T, 8, 64]` into Muse TP=16 `A1_RPA_LAYOUT=1` prepared `(1, T, 4, 2, 128)` bf16."""
    t = q_8_64.shape[0]
    q_8_128 = np.zeros((t, 8, 128), dtype=np.float32)
    q_8_128[:, :, :64] = q_8_64
    return jnp.asarray(q_8_128, dtype=jnp.bfloat16).reshape(1, t, 4, 2, 128)


def _pack_stock_kv_to_dense2_u8(stock_u8: np.ndarray) -> np.ndarray:
    """Pack `(T, 1, 4, 128)` u8 (`0=K, 1=V`) into `(T//2, 1, 4, 128)` dense2 u8."""
    assert stock_u8.shape[0] % 2 == 0
    d2 = np.full((stock_u8.shape[0] // 2, 1, 4, 128), 0x7F, dtype=np.uint8)
    d2[:, 0, 0, :] = stock_u8[0::2, 0, 0, :]
    d2[:, 0, 1, :] = stock_u8[0::2, 0, 1, :]
    d2[:, 0, 2, :] = stock_u8[1::2, 0, 0, :]
    d2[:, 0, 3, :] = stock_u8[1::2, 0, 1, :]
    return d2


def _ref_attention_gqa(
    q_f32: np.ndarray,
    k_f32: np.ndarray,
    v_f32: np.ndarray,
    *,
    q_first: int,
    kv_len: int,
    sm_scale: float,
    sliding_window: int | None,
) -> np.ndarray:
    """Exact FP64 reference returning `[T_q, num_q_heads * 64]` f32."""
    tq, n_qh, d = q_f32.shape
    n_kvh = k_f32.shape[1]
    gqa = n_qh // n_kvh
    q_pos = q_first + np.arange(tq, dtype=np.int32)[:, None]
    k_pos = np.arange(k_f32.shape[0], dtype=np.int32)[None, :]
    vis = (k_pos < kv_len) & (k_pos <= q_pos)
    if sliding_window is not None:
        vis = vis & ((q_pos - k_pos) < sliding_window)

    out_heads = np.zeros((tq, n_qh, d), dtype=np.float32)
    for h in range(n_qh):
        kvh = h // gqa
        qf = np.asarray(jnp.asarray(q_f32[:, h, :], dtype=jnp.bfloat16).astype(jnp.float32), dtype=np.float64)
        kf = k_f32[:, kvh, :].astype(np.float64)
        vf = v_f32[:, kvh, :].astype(np.float64)
        logits = (qf @ kf.T) * float(sm_scale)
        logits = np.where(vis, logits, -np.inf)
        m = np.max(logits, axis=-1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        p = np.where(vis, np.exp(logits - m), 0.0)
        denom = np.sum(p, axis=-1, keepdims=True)
        o_h = np.where(denom > 0.0, (p @ vf) / np.maximum(denom, 1e-30), 0.0)
        out_heads[:, h, :] = o_h.astype(np.float32)
    return out_heads.reshape(tq, n_qh * d)


def _sim_mk128_strip(
    q_5d_bf16: jax.Array,
    kv_buf_u8: np.ndarray,
    *,
    sq_kv_heads: int,
    use_d2: bool,
    use_fp8_qk: bool,
    q_first: int,
    kv_len: int,
    sm_scale: float,
    sliding_window: int | None,
    mode: str,
) -> np.ndarray:
    """Execute the exact `mk128` bit-unpack, Q-decode, `_eval_mk128_head`, and A3 epilogue in JAX."""
    tq_sz = q_5d_bf16.shape[1]
    assert tq_sz == 256
    sq_pairs = 4 * sq_kv_heads
    sq_q_heads = 8 * sq_kv_heads
    exp_scale = float(sm_scale) * 1.4426950408889634
    mask_val = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

    # 1. Exact dense2 / non-dense2 KV unpack into k_sub `(sq_kv_heads, n_h, 128, 128)` & vt_sub `(sq_kv_heads, n_h, 64, 128)`.
    total_k = kv_buf_u8.shape[0] * (2 if use_d2 else 1)
    assert total_k % 256 == 0
    num_s = total_k // 256
    n_h_tot = 2 * num_s
    k_sub = np.zeros((sq_kv_heads, n_h_tot, 128, 128), dtype=np.float32)
    vt_sub = np.zeros((sq_kv_heads, n_h_tot, 64, 128), dtype=np.float32)

    kv_u = kv_buf_u8[:, 0, :, :].astype(np.uint32)
    raw_u32 = jnp.asarray(
        kv_u[:, 0, :] | (kv_u[:, 1, :] << 8) | (kv_u[:, 2, :] << 16) | (kv_u[:, 3, :] << 24),
        dtype=jnp.uint32,
    )

    def _unpack_128_u32(kv_u32_128: jax.Array, valid_n: int) -> tuple[jax.Array, jax.Array]:
        r_io = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
        kv_u32_m = jnp.where(r_io < valid_n, kv_u32_128, jnp.uint32(0))
        k0_f8 = lax.bitcast_convert_type((kv_u32_m & jnp.uint32(0xFF)).astype(jnp.uint8), jnp.float8_e4m3fn)
        v_bf16 = lax.bitcast_convert_type(((kv_u32_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(jnp.uint8), jnp.float8_e4m3fn).astype(jnp.bfloat16)
        vt_128 = jnp.swapaxes(v_bf16, 0, 1)
        ks = [k0_f8 if use_fp8_qk else k0_f8.astype(jnp.bfloat16)]
        vts = [vt_128[:64, :]]
        if sq_kv_heads > 1:
            k1_u32 = jnp.roll(kv_u32_m, shift=64, axis=1)
            k1_f8 = lax.bitcast_convert_type((k1_u32 & jnp.uint32(0xFF)).astype(jnp.uint8), jnp.float8_e4m3fn)
            ks.append(k1_f8 if use_fp8_qk else k1_f8.astype(jnp.bfloat16))
            vts.append(vt_128[64:, :])
        return jnp.stack(ks, axis=0), jnp.stack(vts, axis=0)

    k_slices = []
    vt_slices = []
    if use_d2:
        for s_loc in range(num_s):
            pair_u32 = raw_u32[s_loc * 128 : (s_loc + 1) * 128]
            r_128 = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
            tok_even = s_loc * 256 + 2 * r_128
            lo_u32 = jnp.where(tok_even < kv_len, pair_u32 & jnp.uint32(0xFFFF), jnp.uint32(0))
            hi_u32 = jnp.where((tok_even + 1) < kv_len, pair_u32 >> jnp.uint32(16), jnp.uint32(0))
            k_lo, vt_lo = _unpack_128_u32(lo_u32, 128)
            k_hi, vt_hi = _unpack_128_u32(hi_u32, 128)
            k_slices.extend([k_lo, k_hi])
            vt_slices.extend([vt_lo, vt_hi])
    else:
        for h_loc in range(n_h_tot):
            u_h = raw_u32[h_loc * 128 : (h_loc + 1) * 128]
            valid_n = int(np.clip(kv_len - h_loc * 128, 0, 128))
            k_h, vt_h = _unpack_128_u32(u_h, valid_n)
            k_slices.append(k_h)
            vt_slices.append(vt_h)
    k_sub_arr = jnp.stack(k_slices, axis=1)  # (sq_kv_heads, n_h_tot, 128, 128)
    vt_sub_arr = jnp.stack(vt_slices, axis=1)  # (sq_kv_heads, n_h_tot, 64, 128)

    # 2. Exact Q unpack (`_decode_q_pair`) from `(1, 256, sq_pairs, 2, 128)` bf16.
    q_u16 = np.asarray(lax.bitcast_convert_type(q_5d_bf16, jnp.uint16)).reshape(tq_sz * sq_pairs, 2, 128)
    q_u32 = jnp.asarray(q_u16[:, 0, :].astype(np.uint32) | (q_u16[:, 1, :].astype(np.uint32) << 16), dtype=jnp.uint32)
    lane_is_lo = lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 1) < 64

    qt_list = [None] * sq_q_heads
    for kv_h in range(sq_kv_heads):
        for gp in range(4):
            s = kv_h * 4 + gp
            u_s = q_u32[s::sq_pairs]
            even_f = lax.bitcast_convert_type(u_s << jnp.uint32(16), jnp.float32) * exp_scale
            odd_f = lax.bitcast_convert_type(u_s & jnp.uint32(0xFFFF0000), jnp.float32) * exp_scale
            if (kv_h * 4) < 4:
                pair_f = jnp.where(lane_is_lo, even_f, jnp.roll(odd_f, shift=64, axis=1))
            else:
                pair_f = jnp.where(lane_is_lo, jnp.roll(even_f, shift=64, axis=1), odd_f)
            if use_fp8_qk:
                pair_t = jnp.clip(jnp.swapaxes(pair_f, 0, 1), -448.0, 448.0).astype(jnp.float8_e4m3fn)
            else:
                pair_t = jnp.swapaxes(pair_f.astype(jnp.bfloat16), 0, 1)
            qt_list[2 * s] = pair_t[:64, :][None, :, :]
            qt_list[2 * s + 1] = pair_t[64:, :][None, :, :]

    # 3. Static lt0/lt1 and ut0/ut1 masks matching `ragged_paged_attention_v3.py`.
    r_256 = np.arange(256)[:, None]
    c_256 = np.arange(256)[None, :]
    lt_full = jnp.asarray(np.where(r_256 <= c_256, 0.0, mask_val)[None, :, :], dtype=jnp.bfloat16)
    ut_full = jnp.asarray(np.where(r_256 > c_256, 0.0, mask_val)[None, :, :], dtype=jnp.bfloat16)
    if use_d2:
        static_masks = {
            "lt0": lt_full[:, 0::2, :].astype(jnp.float32),
            "lt1": lt_full[:, 1::2, :].astype(jnp.float32),
            "ut0": ut_full[:, 0::2, :].astype(jnp.float32),
            "ut1": ut_full[:, 1::2, :].astype(jnp.float32),
        }
    else:
        static_masks = {
            "lt0": lt_full[:, :128, :].astype(jnp.float32),
            "lt1": lt_full[:, 128:, :].astype(jnp.float32),
            "ut0": ut_full[:, :128, :].astype(jnp.float32),
            "ut1": ut_full[:, 128:, :].astype(jnp.float32),
        }

    def _kq_dot(ks, qt):
        return lax.dot_general(
            ks, qt, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
        )

    def _vp_dot(vts, p):
        return lax.dot_general(
            vts, p.astype(jnp.bfloat16), (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
        )

    def _sm_eval(qk, bspec):
        if bspec != "none":
            b_arr = static_masks[bspec] if isinstance(bspec, str) else bspec
            qk = qk + b_arr
        p = jnp.exp2(qk)
        p_sub = jnp.sum(p.reshape(1, 16, 8, tq_sz), axis=1, dtype=jnp.bfloat16)
        ps = jnp.broadcast_to(
            jnp.sum(p_sub, axis=1, keepdims=True, dtype=jnp.float32),
            (1, 8, tq_sz),
        )
        return p, ps

    def _eval_mk128_head(kv_h: int, qt_1: jax.Array, h_base: int, n_slices: int, bias_specs: tuple):
        n_h = 2 * n_slices
        z_acc = jnp.zeros((1, 64, tq_sz), jnp.float32)
        if bias_specs == ("none",) * n_slices and n_h >= 4:
            qk_0 = _kq_dot(k_sub_arr[kv_h, h_base, :, :64][None], qt_1)
            p_0, ps_0 = _sm_eval(qk_0, "none")

            def _pipe(h_off, carry):
                acc_c, l_c, p_prev = carry
                qk_c = _kq_dot(k_sub_arr[kv_h, h_base + h_off, :, :64][None], qt_1)
                acc_n = acc_c + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base + h_off - 1], p_prev)
                p_c, ps_c = _sm_eval(qk_c, "none")
                return acc_n, l_c + ps_c, p_c

            acc_f, l_f, p_last = lax.fori_loop(1, n_h, _pipe, (z_acc, ps_0, p_0))
            return acc_f + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base + n_h - 1], p_last), l_f

        if n_slices == 9 and bias_specs == ("ut",) + ("none",) * 7 + ("lt",):
            qk_0 = _kq_dot(k_sub_arr[kv_h, h_base, :, :64][None], qt_1)
            p_prev, l_cur = _sm_eval(qk_0, "ut0")
            qk_1 = _kq_dot(k_sub_arr[kv_h, h_base + 1, :, :64][None], qt_1)
            acc_cur = z_acc + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base], p_prev)
            p_prev, ps_1 = _sm_eval(qk_1, "ut1")
            l_cur = l_cur + ps_1

            def _sw_mid(h_off, carry):
                acc_c, l_c, p_p = carry
                qk_c = _kq_dot(k_sub_arr[kv_h, h_base + h_off, :, :64][None], qt_1)
                acc_n = acc_c + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base + h_off - 1], p_p)
                p_c, ps_c = _sm_eval(qk_c, "none")
                return acc_n, l_c + ps_c, p_c

            acc_cur, l_cur, p_prev = lax.fori_loop(2, 16, _sw_mid, (acc_cur, l_cur, p_prev))
            for h_tail, b_tail in ((16, "lt0"), (17, "lt1")):
                qk_t = _kq_dot(k_sub_arr[kv_h, h_base + h_tail, :, :64][None], qt_1)
                acc_cur = acc_cur + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base + h_tail - 1], p_prev)
                p_prev, ps_t = _sm_eval(qk_t, b_tail)
                l_cur = l_cur + ps_t
            return acc_cur + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base + 17], p_prev), l_cur

        h_specs = []
        for bs in bias_specs:
            if bs == "none":
                h_specs.extend(("none", "none"))
            elif bs == "lt":
                h_specs.extend(("lt0", "lt1"))
            elif bs == "ut":
                h_specs.extend(("ut0", "ut1"))
            else:
                if use_d2:
                    h_specs.extend((bs[:, 0::2, :], bs[:, 1::2, :]))
                else:
                    h_specs.extend((bs[:, :128, :], bs[:, 128:, :]))
        qk_0 = _kq_dot(k_sub_arr[kv_h, h_base, :, :64][None], qt_1)
        p_prev, l_cur = _sm_eval(qk_0, h_specs[0])
        acc_cur = z_acc
        for h_off in range(1, n_h):
            qk_c = _kq_dot(k_sub_arr[kv_h, h_base + h_off, :, :64][None], qt_1)
            acc_cur = acc_cur + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base + h_off - 1], p_prev)
            p_prev, ps_c = _sm_eval(qk_c, h_specs[h_off])
            l_cur = l_cur + ps_c
        return acc_cur + _vp_dot(vt_sub_arr[kv_h : kv_h + 1, h_base + n_h - 1], p_prev), l_cur

    acc_heads = []
    l_heads = []
    for g in range(sq_q_heads):
        kv_h = g // 8
        if mode == "sw9_fast":
            s_lo = (q_first - 2048) // 256
            a_g, l_g = _eval_mk128_head(kv_h, qt_list[g], 2 * s_lo, 9, ("ut",) + ("none",) * 7 + ("lt",))
        elif mode == "sw_short":
            n_unm = q_first // 256
            a_g = jnp.zeros((1, 64, tq_sz), jnp.float32)
            l_g = jnp.zeros((1, 8, tq_sz), jnp.float32)
            if n_unm >= 2:
                a_g, l_g = _eval_mk128_head(kv_h, qt_list[g], 0, n_unm, ("none",) * n_unm)
            elif n_unm == 1:
                a_g, l_g = _eval_mk128_head(kv_h, qt_list[g], 0, 1, ("none",))
            a_lt, l_lt = _eval_mk128_head(kv_h, qt_list[g], 2 * n_unm, 1, ("lt",))
            a_g, l_g = a_g + a_lt, l_g + l_lt
        elif mode == "full_prefix_and_diag":
            n_unm = q_first // 256
            a_g, l_g = _eval_mk128_head(kv_h, qt_list[g], 0, n_unm, ("none",) * n_unm)
            a_lt, l_lt = _eval_mk128_head(kv_h, qt_list[g], 2 * n_unm, 1, ("lt",))
            a_g, l_g = a_g + a_lt, l_g + l_lt
        else:
            # General dynamic-mask path (unaligned q_first / partial kv_len)
            a_g = jnp.zeros((1, 64, tq_sz), jnp.float32)
            l_g = jnp.zeros((1, 8, tq_sz), jnp.float32)
            for s_idx in range(num_s):
                kp = s_idx * 256 + np.arange(256)[:, None]
                qp = q_first + np.arange(tq_sz)[None, :]
                vis = (kp < kv_len) & (kp <= qp)
                if sliding_window is not None:
                    vis = vis & ((qp - kp) < sliding_window)
                msk = jnp.asarray(np.where(vis, 0.0, mask_val)[None, :, :], dtype=jnp.float32)
                a_s, l_s = _eval_mk128_head(kv_h, qt_list[g], 2 * s_idx, 1, (msk,))
                a_g, l_g = a_g + a_s, l_g + l_s
        acc_heads.append(a_g[0])
        l_heads.append(l_g[0])

    # 4. Exact A3 `[tq_sz, 128 * sq_pairs]` epilogue (`[T, 1024]` on TP=8, `[T, 512]` on TP=16)
    cols = []
    for s in range(sq_pairs):
        h0, h1 = 2 * s, 2 * s + 1
        inv0 = jnp.where(l_heads[h0] > 0.0, (1.0 / l_heads[h0]), 0.0)
        inv1 = jnp.where(l_heads[h1] > 0.0, (1.0 / l_heads[h1]), 0.0)
        o0 = (acc_heads[h0].reshape(8, 8, tq_sz) * inv0[None, :, :]).reshape(64, tq_sz).astype(jnp.bfloat16)
        o1 = (acc_heads[h1].reshape(8, 8, tq_sz) * inv1[None, :, :]).reshape(64, tq_sz).astype(jnp.bfloat16)
        cols.append(jnp.swapaxes(jnp.concatenate([o0, o1], axis=0), 0, 1))
    out_a3 = jnp.concatenate(cols, axis=1)  # [tq_sz, 128 * sq_pairs] bf16

    # Reorder TP=8 head-pair interleaved A3 layout `(0,1, 2,3, ..., 14,15)` -> `[T, sq_q_heads * 64]`
    return np.asarray(out_a3.astype(jnp.float32))


class TestMuseW9RpaPrefill(unittest.TestCase):
    def test_default_off_flags_and_ast_no_bare_envfield_bool(self):
        from sglang_tpu.envs import envs

        for env_key, expected in (
            ("SGLANG_TPU_RPA_V3_PREFILL_W9", False),
            ("SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q", True),
        ):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(env_key, None)
                self.assertIs(getattr(envs, env_key).get(), expected, f"{env_key} must default to {expected}")

        for rel in _TOUCHED_FILES:
            src = (_REPO_ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(src, filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    tests = [node.test]
                elif isinstance(node, ast.BoolOp):
                    tests = list(node.values)
                elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    tests = [node.operand]
                else:
                    continue
                for t in tests:
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "envs"
                        and t.attr.startswith("SGLANG_")
                    ):
                        self.fail(f"Bare EnvField truthiness evaluation in {rel}:{t.lineno}: envs.{t.attr}")

    def test_invoke_v3_kernel_w9_scoping_and_tp16_routing(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch required for attention_pallas_rpa_v3 wrapper routing test")

        import sglang_tpu.attention_pallas_rpa_v3 as rpa_mod

        captured = {}

        def _fake_rpa(q, k, v, c, kl, pi, cq, ckv, dist, *args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return q, c

        def _call_invoke(q, k, v, c, kl, pi, *, sw=None, prepared=False, d2=False, skip_decode=True, a3_out=0):
            captured.clear()
            rpa_mod._invoke_v3_kernel(
                _fake_rpa,
                queries=q,
                keys=k,
                values=v,
                kv_cache_fused=c,
                kv_lens=kl,
                page_indices=pi,
                cu_q_lens=None,
                cu_kv_lens=None,
                distribution=None,
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=sw,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=prepared,
                kv_pack_dense2=d2,
                skip_decode=skip_decode,
                a3_out_layout=a3_out,
            )
            return dict(captured)

        # 1. Verify default-OFF (`PREFILL_W9=0`) preserves exact cand_w9_base LC_OPTS behavior,
        # while `PREFILL_W9=1` (or `scope_lc`) scopes LC_OPTS off SWA layers!
        q_tp8 = SimpleNamespace(ndim=5, shape=(1, 8192, 8, 2, 128), dtype="bfloat16")
        k_tp8 = SimpleNamespace(ndim=4, shape=(8192, 1, 4, 128))
        c_d2 = SimpleNamespace(shape=(2048, 256, 1, 4, 128), dtype="float8_e4m3fn")
        kl = SimpleNamespace(shape=(64,))
        pi_swa = SimpleNamespace(shape=(64 * 8,))
        pi_lc = SimpleNamespace(shape=(64 * 545,))

        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_KV_PACK_DENSE2": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "0",
                "SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS": "sw_w8,sw_fp8_qk",
                "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq4096,bkv8192,s2,prefix_fast,staged_qo,fp8_qk,vreg_acc1,mrb1",
            },
            clear=False,
        ):
            kw_def = _call_invoke(q_tp8, k_tp8, None, c_d2, kl, pi_swa, sw=2048, prepared=True, d2=True, a3_out=2)
            self.assertIn("mrb1", kw_def["_prefill_opts"])
            self.assertIn("vreg_acc1", kw_def["_prefill_opts"])

        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_KV_PACK_DENSE2": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS": "sw_w8,sw_fp8_qk,nofull_fp8_qk",
                "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq4096,bkv8192,s2,prefix_fast,staged_qo,fp8_qk,vreg_acc1,mrb1",
            },
            clear=False,
        ):
            kw = _call_invoke(q_tp8, k_tp8, None, c_d2, kl, pi_swa, sw=2048, prepared=True, d2=True, a3_out=2)
            swa_opts = kw["_prefill_opts"]
            self.assertNotIn("mrb1", swa_opts)
            self.assertNotIn("vreg_acc1", swa_opts)
            self.assertNotIn("vreg_acc", swa_opts)
            self.assertNotIn("sw_short_fuse", swa_opts)
            self.assertIn("mk128", swa_opts)
            self.assertIn("la2", swa_opts)
            self.assertIn("sub8_l", swa_opts)
            self.assertIn("bf16_pop", swa_opts)
            self.assertIn("sw_short_v2", swa_opts)
            self.assertIn("mrb3", swa_opts)
            self.assertIn("fp8_qk", swa_opts)
            self.assertIn("sw2048_fuse", swa_opts)
            self.assertEqual(kw["m_block_sizes"][:2], (256, 2048))
            # Verify `nofull_fp8_qk` keeps <=32K Full-Attn on bf16 QK while enabling unrolled `mk128` (#2518 §1).
            pi_std_full = SimpleNamespace(shape=(64 * 36,))
            kw_full_bf16 = _call_invoke(q_tp8, k_tp8, None, c_d2, kl, pi_std_full, sw=None, prepared=True, d2=True, a3_out=2)
            self.assertIn("mk128", kw_full_bf16["_prefill_opts"])
            self.assertNotIn("fp8_qk", kw_full_bf16["_prefill_opts"])

        # 2. Verify `SGLANG_TPU_RPA_V3_PREFILL_W9=1` activates `mk128` even under `run_e2e.py` `force_runtime_env`!
        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_KV_PACK_DENSE2": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS": "sw_w8,sw_fp8_qk",
                "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq4096,bkv8192,s2,prefix_fast,staged_qo,fp8_qk,vreg_acc1,mrb1",
            },
            clear=False,
        ):
            q_tp8_4k = SimpleNamespace(ndim=5, shape=(1, 4096, 8, 2, 128), dtype="bfloat16")
            kw = _call_invoke(q_tp8_4k, k_tp8, None, c_d2, kl, pi_lc, sw=None, prepared=True, d2=True, a3_out=2)
            lc_opts = kw["_prefill_opts"]
            self.assertIn("mk128", lc_opts)
            self.assertNotIn("gqa8", lc_opts)
            self.assertIn("fp8_qk", lc_opts)
            self.assertIn("staged_qo", lc_opts)
            self.assertIn("prefix_fast", lc_opts)
            self.assertNotIn("vreg_acc1", lc_opts)
            self.assertNotIn("mrb1", lc_opts)
            self.assertEqual(kw["m_block_sizes"][:2], (4096, 8192))

        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_KV_PACK_DENSE2": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS": "sw_w8,sw_fp8_qk,gqa8",
                "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq4096,bkv8192,s2,prefix_fast,staged_qo,fp8_qk,vreg_acc1,mrb1",
            },
            clear=False,
        ):
            q_tp8_4k = SimpleNamespace(ndim=5, shape=(1, 4096, 8, 2, 128), dtype="bfloat16")
            kw_g8 = _call_invoke(q_tp8_4k, k_tp8, None, c_d2, kl, pi_lc, sw=None, prepared=True, d2=True, a3_out=2)
            self.assertIn("gqa8", kw_g8["_prefill_opts"])

        # 3. Verify `TP=16` (`q=(1, T, 4, 2, 128)`) prepared Stationary-Q v2 + `[T, 512]` `a3_out_layout`.
        q_tp16_prep = SimpleNamespace(ndim=5, shape=(1, 8192, 4, 2, 128), dtype="bfloat16")
        c_tp16 = SimpleNamespace(shape=(2048, 512, 1, 4, 128), dtype="float8_e4m3fn")
        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
            },
            clear=False,
        ):
            kw = _call_invoke(q_tp16_prep, k_tp8, None, c_tp16, kl, pi_swa, sw=2048, prepared=True, d2=False, a3_out=-2)
            tp16_opts = kw["_prefill_opts"]
            self.assertIn("stationary_q_m0_ring_fuse", tp16_opts)
            self.assertIn("mrb3", tp16_opts)
            self.assertIn("sw2048_fuse", tp16_opts)
            self.assertIn("sw_short_v2", tp16_opts)
            self.assertIn("sub8_l", tp16_opts)
            self.assertIn("bf16_pop", tp16_opts)
            self.assertNotIn("mk128", tp16_opts)
            self.assertIn("tp16_sq", tp16_opts)
            self.assertTrue(kw.get("_a3_out_layout"))
            kw_tp16_full = _call_invoke(q_tp16_prep, k_tp8, None, c_tp16, kl, SimpleNamespace(shape=(64 * 36,)), sw=None, prepared=True, d2=False, a3_out=-2)
            self.assertIn("mk128", kw_tp16_full["_prefill_opts"])
            self.assertIn("tp16_sq", kw_tp16_full["_prefill_opts"])
            # Sub-256 TP=16 (`L61_PRUNE=1` `T=128` & `1/1K` `T=128`, #2981/#2982/#2990) pads
            # `(queries, keys, values)` to `256` -> `f32` `RPAm` (`tp16_sq`, `_a3_out_layout=True`)
            # and slices `out_final[:orig_num_q_tokens]`, while Guard A keeps `TP=8` & `skip_decode=False` untouched!
            from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import prepare_head_packed_outputs

            q_tp16_l61 = SimpleNamespace(ndim=5, shape=(1, 128, 4, 2, 128), dtype="bfloat16")
            kw_l61 = _call_invoke(q_tp16_l61, k_tp8, None, c_tp16, kl, pi_swa, sw=None, prepared=True, d2=False, a3_out=-2)
            self.assertIn("tp16_sq", kw_l61["_prefill_opts"])
            self.assertIn("stationary_q_m0_ring_fuse", kw_l61["_prefill_opts"])
            self.assertTrue(kw_l61.get("_a3_out_layout"))
            self.assertEqual(kw_l61["m_block_sizes"][0], 256)
            with patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES": "512,4096,512,512",
                    "SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES_SW": "256,2048,256,512",
                },
                clear=False,
            ):
                kw_l61_blk = _call_invoke(
                    q_tp16_l61, k_tp8, None, c_tp16, kl, pi_swa, sw=None, prepared=True, d2=False, a3_out=-2
                )
                self.assertEqual(kw_l61_blk["m_block_sizes"], (256, 4096, 256, 512))
            self.assertEqual(rpa_mod._prepared_a3_layout_spec(-2).out_head_pack, 2)
            out_l61 = prepare_head_packed_outputs(jnp.ones((1, 128, 4, 2, 128), jnp.bfloat16), -2, 8, 128)
            self.assertEqual(out_l61.shape, (128, 512))

            # Guard A (#2990 §3): TP=8 `T=128` and `skip_decode=False` unprepared decode fallback stay unpadded!
            q_tp8_l61 = SimpleNamespace(ndim=5, shape=(1, 128, 8, 2, 128), dtype="bfloat16")
            kw_tp8_l61 = _call_invoke(q_tp8_l61, k_tp8, None, c_d2, kl, pi_swa, sw=None, prepared=True, d2=True, a3_out=2)
            self.assertEqual(kw_tp8_l61.get("_fused_out_pack"), 2)
            self.assertNotIn("stationary_q_m0_ring_fuse", kw_tp8_l61.get("_prefill_opts", frozenset()))
            q_tp16_dec_fb = SimpleNamespace(ndim=3, shape=(64, 8, 64), dtype="bfloat16")
            k_tp16_dec_fb = SimpleNamespace(shape=(64, 1, 64))
            kw_dec_fb = _call_invoke(
                q_tp16_dec_fb, k_tp16_dec_fb, k_tp16_dec_fb, c_tp16, kl, pi_swa, sw=None, prepared=False, skip_decode=False
            )
            self.assertNotIn("tp16_sq", kw_dec_fb.get("_prefill_opts", frozenset()))

            # Guard B & C (#2990 §3): real JAX arrays pad `queries`, `keys`, and `values` to 256 and slice tuple `out_final[:T]`!
            seen_shapes = {}

            def _fake_rpa_shapes(q, k, v, c, kl_, pi_, cq_, ckv_, dist_, *args, **kwargs):
                seen_shapes["q"] = q.shape
                seen_shapes["k"] = None if k is None else k.shape
                seen_shapes["v"] = None if v is None else v.shape
                out_256 = (
                    jnp.ones((256, 512), jnp.bfloat16)
                    if kwargs.get("_a3_out_layout")
                    else jnp.ones((256, 8, 64), jnp.bfloat16)
                )
                return out_256, c

            out_u, c_u = rpa_mod._invoke_v3_kernel(
                _fake_rpa_shapes,
                queries=jnp.zeros((128, 8, 64), jnp.bfloat16),
                keys=jnp.zeros((128, 1, 64), jnp.float8_e4m3fn),
                values=jnp.zeros((128, 1, 64), jnp.float8_e4m3fn),
                kv_cache_fused=jnp.zeros((16, 512, 1, 4, 128), jnp.float8_e4m3fn),
                kv_lens=jnp.zeros((8,), jnp.int32),
                page_indices=jnp.zeros((64,), jnp.int32),
                cu_q_lens=jnp.zeros((9,), jnp.int32),
                cu_kv_lens=jnp.zeros((9,), jnp.int32),
                distribution=jnp.zeros((3,), jnp.int32),
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=None,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=False,
                skip_decode=True,
            )
            self.assertEqual(seen_shapes, {"q": (256, 8, 64), "k": (256, 1, 64), "v": (256, 1, 64)})
            self.assertEqual(out_u.shape, (128, 8, 64))
            self.assertEqual(c_u.shape, (16, 512, 1, 4, 128))

        # 4. Verify TP=16 unprepared routes to `tp16_sq` by default (#2390 fix), while
        #    kill-switch (`TP16_STATIONARY_Q=0`) or explicit `TP16_NATIVE_PREFILL=1` disables it cleanly.
        q_tp16_unprep = SimpleNamespace(ndim=3, shape=(8192, 8, 64), dtype="bfloat16")
        k_tp16_unprep = SimpleNamespace(shape=(8192, 1, 64))
        c_tp16_p128 = SimpleNamespace(shape=(2048, 128, 1, 4, 128), dtype="float8_e4m3fn")
        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "0",
            },
            clear=False,
        ):
            os.environ.pop("SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q", None)
            os.environ.pop("SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL", None)
            kw_def_tp16 = _call_invoke(q_tp16_unprep, k_tp16_unprep, k_tp16_unprep, c_tp16_p128, kl, pi_swa, sw=2048, prepared=False)
            self.assertIn("tp16_sq", kw_def_tp16["_prefill_opts"])
            self.assertIn("stationary_q_m0_ring_fuse", kw_def_tp16["_prefill_opts"])

        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q": "0",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
            },
            clear=False,
        ):
            kw_kill = _call_invoke(q_tp16_unprep, k_tp16_unprep, k_tp16_unprep, c_tp16_p128, kl, pi_swa, sw=2048, prepared=False)
            kill_opts = kw_kill.get("_prefill_opts", frozenset())
            self.assertNotIn("tp16_sq", kill_opts)
            self.assertNotIn("stationary_q_m0_ring_fuse", kill_opts)

        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL": "1",
                "SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q": "0",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "0",
            },
            clear=False,
        ):
            kw = _call_invoke(q_tp16_unprep, k_tp16_unprep, k_tp16_unprep, c_tp16_p128, kl, pi_swa, sw=2048, prepared=False)
            self.assertNotIn("tp16_sq", kw["_prefill_opts"])
            self.assertIn("visible_kv_tile", kw["_prefill_opts"])
            self.assertEqual(kw["m_block_sizes"], (256, 1024, 256, 256))

    def test_mk128_numerical_parity_tp8_tp16_and_outlier_nan_immunity(self):
        rng = np.random.default_rng(20261123)
        sm_scale = 0.125

        # Case A: TP=8 (16 Q / 2 KV, d2=True & d2=False) across sw9_fast, sw_short, full_prefix, and unaligned dynamic.
        for use_d2 in (True, False):
            for mode, q_first, kv_len, sw in (
                ("sw9_fast", 2048, 4352, 2048),
                ("sw_short", 512, 1024, 2048),
                ("full_prefix_and_diag", 2560, 2816, None),
                ("general_unaligned", 320, 1900, 2048),
            ):
                with self.subTest(tp=8, use_d2=use_d2, mode=mode):
                    n_k_pad = 4608
                    q_16_64 = (rng.standard_normal((256, 16, 64)) * 1.1).astype(np.float32)
                    stock_u8 = np.full((n_k_pad, 1, 4, 128), 0x7F, dtype=np.uint8)
                    stock_u8[:kv_len, 0, :2, :] = _rand_fp8_u8(rng, (kv_len, 2, 128), scale=0.45)
                    stock_u8[:kv_len, 0, 2:, :] = 0
                    kv_buf_u8 = _pack_stock_kv_to_dense2_u8(stock_u8) if use_d2 else stock_u8

                    kv_f32 = np.asarray(_u8_to_fp8(stock_u8[:kv_len, 0, :2, :]).astype(jnp.float32))
                    k_f32 = kv_f32[:, 0, :].reshape(kv_len, 2, 64)
                    v_f32 = kv_f32[:, 1, :].reshape(kv_len, 2, 64)

                    ref = _ref_attention_gqa(
                        q_16_64, k_f32, v_f32, q_first=q_first, kv_len=kv_len, sm_scale=sm_scale, sliding_window=sw
                    )
                    got = _sim_mk128_strip(
                        _pack_q_tp8_5d(q_16_64),
                        kv_buf_u8,
                        sq_kv_heads=2,
                        use_d2=use_d2,
                        use_fp8_qk=True,
                        q_first=q_first,
                        kv_len=kv_len,
                        sm_scale=sm_scale,
                        sliding_window=sw,
                        mode=mode,
                    )
                    self.assertFalse(np.isnan(got).any())
                    cos = float(np.sum(got * ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
                    self.assertGreater(cos, 0.999)
                    np.testing.assert_allclose(got, ref, rtol=4e-2, atol=4e-2)

        # Case B: TP=16 (8 Q / 1 KV, `[T, 512]` A3 layout), including #2468 extreme outlier stress test!
        for stress_outlier, mode, q_first, kv_len, sw in (
            (False, "sw9_fast", 2048, 4352, 2048),
            (False, "sw_short", 256, 768, 2048),
            (False, "full_prefix_and_diag", 3072, 3328, None),
            (True, "full_prefix_and_diag", 2048, 2304, None),
        ):
            with self.subTest(tp=16, outlier=stress_outlier, mode=mode):
                n_k_pad = 4608
                q_8_64 = (rng.standard_normal((256, 8, 64)) * 1.1).astype(np.float32)
                stock_u8 = np.full((n_k_pad, 1, 4, 128), 0x7F, dtype=np.uint8)
                stock_u8[:kv_len, 0, :2, :64] = _rand_fp8_u8(rng, (kv_len, 2, 64), scale=0.45)
                stock_u8[:kv_len, 0, :2, 64:] = 0
                stock_u8[:kv_len, 0, 2:, :] = 0

                if stress_outlier:
                    # Inject #2468-class outlier sink key & query (`q @ k * sm_scale = +32` -> `exp2(+46.2) = 8.1e13 >> bf16_max=65504`)!
                    # Stock TP=16 `compute_with_bq` overflows `l_next`/`acc_next` in bf16 -> `inf / inf = NaN`.
                    q_8_64[13, :, :16] = 2.0
                    outlier_kv = np.zeros((1, 2, 64), dtype=np.float32)
                    outlier_kv[0, 0, :16] = 8.0
                    outlier_kv[0, 1, :] = rng.standard_normal((64,)).astype(np.float32) * 1.5
                    stock_u8[7, 0, :2, :64] = _fp8_to_u8(jnp.asarray(outlier_kv, dtype=jnp.float8_e4m3fn))[0]

                kv_f32 = np.asarray(_u8_to_fp8(stock_u8[:kv_len, 0, :2, :64]).astype(jnp.float32))
                k_f32 = kv_f32[:, 0:1, :]
                v_f32 = kv_f32[:, 1:2, :]

                ref = _ref_attention_gqa(
                    q_8_64, k_f32, v_f32, q_first=q_first, kv_len=kv_len, sm_scale=sm_scale, sliding_window=sw
                )
                got = _sim_mk128_strip(
                    _pack_q_tp16_5d(q_8_64),
                    stock_u8,
                    sq_kv_heads=1,
                    use_d2=False,
                    use_fp8_qk=not stress_outlier,
                    q_first=q_first,
                    kv_len=kv_len,
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    mode=mode,
                )
                self.assertEqual(got.shape, (256, 512))
                self.assertFalse(np.isnan(got).any())
                self.assertFalse(np.isinf(got).any())
                cos = float(np.sum(got * ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
                self.assertGreater(cos, 0.999)
                np.testing.assert_allclose(got, ref, rtol=4e-2, atol=4e-2)
                if stress_outlier:
                    got_fp8 = _sim_mk128_strip(
                        _pack_q_tp16_5d(q_8_64),
                        stock_u8,
                        sq_kv_heads=1,
                        use_d2=False,
                        use_fp8_qk=True,
                        q_first=q_first,
                        kv_len=kv_len,
                        sm_scale=sm_scale,
                        sliding_window=sw,
                        mode=mode,
                    )
                    self.assertFalse(np.isnan(got_fp8).any())
                    self.assertFalse(np.isinf(got_fp8).any())
                    cos_fp8 = float(np.sum(got_fp8 * ref) / (np.linalg.norm(got_fp8) * np.linalg.norm(ref)))
                    self.assertGreater(cos_fp8, 0.998)

        # Case C (#2756/#2760 Unified `nod2` regression): when `1 <= (kv_len % 256) <= 128`
        # (odd `cdiv(kv_len, 128)`), `_up_partial` MUST zero `2 * cdiv(kv_len, 256)` halves
        # in VMEM so dirty `NaN` VMEM bits from prior decode kernels never hit `0.0 * NaN = NaN`!
        src_rpa = (_REPO_ROOT / "sglang_tpu/attention/pallas/ragged_paged_attention_v3.py").read_text(encoding="utf-8")
        self.assertIn("h_lim = 2 * cdiv(valid_keys, 256)", src_rpa)
        self.assertNotIn("h_lim = cdiv(valid_keys, 128)", src_rpa)

    def test_pallas_interpret_uninitialized_nan_scratch(self):
        """Execute real `ragged_paged_attention` under `pltpu.InterpretParams(uninitialized_memory='nan')`
        with `0x7F`-poisoned KV cache across `nod2`/`d2`, `TP=8`/`TP=16`, `Full`/`SWA`, and odd-half `kv_len` (#2770/#2773).
        """
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import ragged_paged_attention

        rng = np.random.default_rng(20261124)
        interp = pltpu.InterpretParams(uninitialized_memory="nan")
        sm_scale = 0.125

        for qh_pairs, use_d2 in ((8, False), (8, True), (4, False)):
            phys_pg = 256 if use_d2 else 512
            n_q_heads = 2 * qh_pairs
            n_kv_heads = 2 if qh_pairs == 8 else 1
            extra_tp = {"tp16_sq"} if qh_pairs == 4 else set()
            sw_mk = {"mk128", "la2", "unroll_up"} if qh_pairs == 8 else set()

            for seq_lens in ((6,), (300,), (6, 506)):
                tot = sum(seq_lens)
                t_bkt = 256 if tot <= 256 else 768
                q_heads_f32 = (rng.standard_normal((tot, n_q_heads, 64)) * 0.6).astype(np.float32)
                q_pad_f32 = np.zeros((t_bkt, n_q_heads, 64), dtype=np.float32)
                q_pad_f32[:tot] = q_heads_f32
                q_5d = _pack_q_tp8_5d(q_pad_f32) if qh_pairs == 8 else _pack_q_tp16_5d(q_pad_f32)

                kv_u8 = np.full((t_bkt, 1, 4, 128), 0x7F, dtype=np.uint8)
                k_cols = 128 if qh_pairs == 8 else 64
                kv_u8[:tot, 0, :2, :k_cols] = _rand_fp8_u8(rng, (tot, 2, k_cols), scale=0.4)
                if k_cols == 64:
                    kv_u8[:tot, 0, :2, 64:] = 0
                kv_u8[:tot, 0, 2:, :] = 0
                kv_in = _u8_to_fp8(kv_u8)

                kv_tok_f32 = np.asarray(_u8_to_fp8(kv_u8[:tot, 0, :2, :k_cols]).astype(jnp.float32))
                k_all_f32 = kv_tok_f32[:, 0, :].reshape(tot, n_kv_heads, 64)
                v_all_f32 = kv_tok_f32[:, 1, :].reshape(tot, n_kv_heads, 64)

                n_s = len(seq_lens)
                kl = np.zeros((8,), dtype=np.int32)
                cq = np.zeros((9,), dtype=np.int32)
                pps = 3
                pi = np.zeros((8 * pps,), dtype=np.int32)
                cur_q = 0
                cur_pg = 1
                for si, sl in enumerate(seq_lens):
                    kl[si] = sl
                    cur_q += sl
                    cq[si + 1] = cur_q
                    npg = (sl + 511) // 512
                    pi[si * pps : si * pps + npg] = np.arange(cur_pg, cur_pg + npg, dtype=np.int32)
                    cur_pg += npg
                cq[n_s + 1 :] = cur_q
                ckv = np.arange(9, dtype=np.int32) * (pps * 512)
                dist = np.array([0, 0, n_s], dtype=np.int32)

                for mode_tag, sw, mb, opts in (
                    (
                        "full_w94",
                        None,
                        (256, 1024, 256, 512),
                        {
                            "stationary_q_m0_ring_fuse",
                            "skip_empty_cdma",
                            "bkv_full_unroll",
                            "mixed_s3",
                            "prefix_fast",
                            "fp8_qk",
                            "mk128",
                            "la2",
                            "sub8_l",
                            "bf16_pop",
                            "unroll_up",
                        }
                        | extra_tp,
                    ),
                    (
                        "swa_w94",
                        2048,
                        (256, 2048, 256, 512),
                        {
                            "stationary_q_m0_ring_fuse",
                            "skip_empty_cdma",
                            "mixed_s3",
                            "sw2048_fuse",
                            "sw_short_v2",
                            "fp8_qk",
                            "mrb3",
                            "sub8_l",
                            "bf16_pop",
                        }
                        | sw_mk
                        | extra_tp,
                    ),
                ):
                    with self.subTest(tp=16 // (8 // qh_pairs), d2=use_d2, seqs=seq_lens, mode=mode_tag):
                        cache_poison = _u8_to_fp8(np.full((16, phys_pg, 1, 4, 128), 0x7F, dtype=np.uint8))
                        out_2d, _ = ragged_paged_attention(
                            jnp.array(q_5d),
                            jnp.array(kv_in),
                            None,
                            cache_poison,
                            jnp.asarray(kl),
                            jnp.asarray(pi),
                            jnp.asarray(cq),
                            jnp.asarray(ckv),
                            jnp.asarray(dist),
                            None,
                            None,
                            sm_scale=sm_scale,
                            sliding_window=sw,
                            k_scale=1.0,
                            v_scale=1.0,
                            m_block_sizes=mb,
                            _prefill_opts=frozenset(opts),
                            _fused_out_pack=2,
                            _prepared_inputs=True,
                            _skip_decode=True,
                            _a3_out_layout=True,
                            _kv_pack_dense2=use_d2,
                            _interpret=interp,
                        )
                        got_all = np.asarray(out_2d.astype(jnp.float32))
                        self.assertEqual(got_all.shape, (t_bkt, 128 * qh_pairs))
                        self.assertFalse(np.isnan(got_all).any(), f"NaN found in {mode_tag} {seq_lens} d2={use_d2}")
                        self.assertFalse(np.isinf(got_all).any(), f"Inf found in {mode_tag} {seq_lens} d2={use_d2}")

                        q_off = 0
                        for sl in seq_lens:
                            ref = _ref_attention_gqa(
                                q_heads_f32[q_off : q_off + sl],
                                k_all_f32[q_off : q_off + sl],
                                v_all_f32[q_off : q_off + sl],
                                q_first=0,
                                kv_len=sl,
                                sm_scale=sm_scale,
                                sliding_window=sw,
                            )
                            got_seq = got_all[q_off : q_off + sl, : 64 * n_q_heads]
                            cos = float(np.sum(got_seq * ref) / (np.linalg.norm(got_seq) * np.linalg.norm(ref)))
                            self.assertGreater(cos, 0.999)
                            q_off += sl

    def test_tp16_sub256_pad_and_stock_mixed_zero_tail_fail_to_pass(self):
        """Deterministic FAIL->PASS interpret gate (#2942, #2981, #2982, #2983, #2990) on TP=16 nod2:
        - `num_q_tokens in {1, 7, 128, 255}` (prepared `L61_PRUNE` and unprepared `1/1K`)
        - `kv_len in {1, 104, 615, 1023}` (`kv_len % 512 in {1, 104, 511}`)
        - Verifies:
          1. Legacy unmasked/bf16 stock MIXED FAILS deterministically on `0x7F` unused lanes & un-normalized V overflow (#2390).
          2. `_invoke_v3_kernel` (`<256 -> 256` pad into `f32` `RPAm`) PASSES with `0` NaNs/Infs, `cos > 0.999`,
             and bit-exact `0` KV writes outside active `cu_q_lens` slots.
          3. Patched `stock MIXED` (`zero_kv_tail` + `@pl.when(sz > 0)` + `valid_lane_dim`) ALSO PASSES `0x7F` KV poison
             and preserves multi-`bq` `sliding_window` KV cache writes without clobbering `update_sz`.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch required for attention_pallas_rpa_v3._invoke_v3_kernel")

        import sglang_tpu.attention.pallas.ragged_paged_attention_v3 as rpa_v3_mod
        import sglang_tpu.attention_pallas_rpa_v3 as rpa_wrap_mod

        rng = np.random.default_rng(20261125)
        interp = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
        sm_scale = 0.125
        page_size = 512
        pps = 4

        with patch.object(rpa_v3_mod, "get_tpu_version", return_value=7):
            # 1. Deterministic FAIL before fix:
            # (a) Unmasked lanes 64..127 (`_fused_out_pack=1` bypassing `valid_lane_dim=64`) on `0x7F` TP=16 KV cache -> NaN!
            q_fail_np = (rng.standard_normal((128, 8, 64)) * 0.5).astype(np.float32)
            kv_fail_u8 = np.full((128, 1, 4, 128), 0x7F, dtype=np.uint8)
            kv_fail_u8[0, 0, :2, :64] = _rand_fp8_u8(rng, (2, 64), scale=0.4)
            cache_fail_u8 = np.full((8, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
            kl_1 = jnp.array([1, 0, 0, 0], dtype=jnp.int32)
            pi_1 = jnp.array([1, 0, 0, 0] + [0] * 12, dtype=jnp.int32)
            cq_1 = jnp.array([0, 1, 1, 1, 1], dtype=jnp.int32)
            ckv_1 = jnp.arange(5, dtype=jnp.int32) * (pps * page_size)
            dist_1 = jnp.array([0, 0, 1], dtype=jnp.int32)
            out_legacy_lanes, _ = rpa_v3_mod.ragged_paged_attention(
                _pack_q_tp16_5d(q_fail_np),
                _u8_to_fp8(kv_fail_u8),
                None,
                _u8_to_fp8(cache_fail_u8),
                kl_1,
                pi_1,
                cq_1,
                ckv_1,
                dist_1,
                None,
                None,
                sm_scale=sm_scale,
                k_scale=1.0,
                v_scale=1.0,
                _skip_decode=True,
                _prepared_inputs=True,
                _fused_out_pack=1,
                _interpret=interp,
            )
            self.assertTrue(
                np.isnan(np.asarray(out_legacy_lanes[:1].astype(jnp.float32))).any(),
                "Expected legacy unmasked lanes 64..127 in stock MIXED to produce NaN on 0x7F KV cache",
            )

            # (b) Verify patched `stock MIXED` (`_fused_out_pack=-2` -> `valid_lane_dim=64` + `zero_kv_tail`)
            #     AND `_invoke_v3_kernel` (`<256 -> 256` `f32` `RPAm`) BOTH cure that exact `0x7F` KV cache NaN!
            out_patched_mixed, _ = rpa_v3_mod.ragged_paged_attention(
                _pack_q_tp16_5d(q_fail_np),
                _u8_to_fp8(kv_fail_u8),
                None,
                _u8_to_fp8(cache_fail_u8),
                kl_1,
                pi_1,
                cq_1,
                ckv_1,
                dist_1,
                None,
                None,
                sm_scale=sm_scale,
                k_scale=1.0,
                v_scale=1.0,
                _skip_decode=True,
                _prepared_inputs=True,
                _fused_out_pack=-2,
                _interpret=interp,
            )
            got_patched_mixed = np.asarray(out_patched_mixed[:1].astype(jnp.float32))
            self.assertEqual(out_patched_mixed.shape, (128, 512))
            self.assertFalse(np.isnan(got_patched_mixed).any())
            self.assertFalse(np.isinf(got_patched_mixed).any())

            with patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
                    "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                    "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                    "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                },
                clear=False,
            ):
                out_rpam_cure, _ = rpa_wrap_mod._invoke_v3_kernel(
                    functools.partial(rpa_v3_mod.ragged_paged_attention, _interpret=interp),
                    queries=_pack_q_tp16_5d(q_fail_np),
                    keys=_u8_to_fp8(kv_fail_u8),
                    values=None,
                    kv_cache_fused=_u8_to_fp8(cache_fail_u8),
                    kv_lens=kl_1,
                    page_indices=pi_1,
                    cu_q_lens=cq_1,
                    cu_kv_lens=ckv_1,
                    distribution=dist_1,
                    attention_sink=None,
                    sm_scale=sm_scale,
                    sliding_window=None,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    prepared_inputs=True,
                    skip_decode=True,
                    a3_out_layout=-2,
                )
                got_rpam_cure = np.asarray(out_rpam_cure[:1].astype(jnp.float32))
                self.assertEqual(out_rpam_cure.shape, (128, 512))
                self.assertFalse(np.isnan(got_rpam_cure).any())
                self.assertFalse(np.isinf(got_rpam_cure).any())
                np.testing.assert_allclose(got_rpam_cure, got_patched_mixed, rtol=3e-2, atol=3e-2)

            # 2. Full matrix PASS after fix: `num_q_tokens in {1, 7, 128, 255}` x `kv_len % 512 in {1, 104, 511}`
            #    + strict bit-exact ZERO extra KV writes check!
            with patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
                    "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                    "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                    "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                },
                clear=False,
            ):
                for num_q_tok, is_prep in ((1, True), (7, False), (128, True), (128, False), (255, True)):
                    for klen in (1, 104, 511, 615):
                        with self.subTest(num_q_tokens=num_q_tok, prepared=is_prep, kv_len=klen):
                            q_f32 = (rng.standard_normal((num_q_tok, 8, 64)) * 0.5).astype(np.float32)
                            cache_u8 = np.full((8, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
                            npg = (klen + page_size - 1) // page_size
                            pref_len = klen - 1
                            rem = pref_len
                            k_all = np.zeros((klen, 1, 64), dtype=np.float32)
                            v_all = np.zeros((klen, 1, 64), dtype=np.float32)
                            off_t = 0
                            for p_off in range(npg):
                                pg_id = 1 + p_off
                                n_in_pg = min(rem, page_size)
                                if n_in_pg > 0:
                                    kv_pg_u8 = _rand_fp8_u8(rng, (n_in_pg, 2, 64), scale=0.4)
                                    cache_u8[pg_id, :n_in_pg, 0, :2, :64] = kv_pg_u8
                                    kv_pg_f32 = np.asarray(_u8_to_fp8(kv_pg_u8).astype(jnp.float32))
                                    k_all[off_t : off_t + n_in_pg, 0, :] = kv_pg_f32[:, 0, :]
                                    v_all[off_t : off_t + n_in_pg, 0, :] = kv_pg_f32[:, 1, :]
                                    off_t += n_in_pg
                                    rem -= n_in_pg

                            new_kv_u8 = _rand_fp8_u8(rng, (2, 64), scale=0.4)
                            new_kv_f32 = np.asarray(_u8_to_fp8(new_kv_u8).astype(jnp.float32))
                            k_all[klen - 1, 0, :] = new_kv_f32[0]
                            v_all[klen - 1, 0, :] = new_kv_f32[1]

                            if is_prep:
                                q_in = _pack_q_tp16_5d(q_f32)
                                kv_buf_u8 = np.full((num_q_tok, 1, 4, 128), 0x7F, dtype=np.uint8)
                                kv_buf_u8[0, 0, :2, :64] = new_kv_u8
                                kv_buf_u8[0, 0, :2, 64:] = 0
                                kv_buf_u8[0, 0, 2:, :] = 0
                                k_in = _u8_to_fp8(kv_buf_u8)
                                v_in = None
                                a3_arg = -2
                            else:
                                q_in = jnp.asarray(q_f32, dtype=jnp.bfloat16)
                                k_buf_u8 = np.full((num_q_tok, 1, 64), 0x7F, dtype=np.uint8)
                                v_buf_u8 = np.full((num_q_tok, 1, 64), 0x7F, dtype=np.uint8)
                                k_buf_u8[0, 0, :] = new_kv_u8[0]
                                v_buf_u8[0, 0, :] = new_kv_u8[1]
                                k_in = _u8_to_fp8(k_buf_u8)
                                v_in = _u8_to_fp8(v_buf_u8)
                                a3_arg = 0

                            kl_arr = jnp.array([klen, 0, 0, 0], dtype=jnp.int32)
                            pi_list = [1 + p for p in range(npg)] + [0] * (16 - npg)
                            pi_arr = jnp.array(pi_list, dtype=jnp.int32)

                            out_res, updated_cache = rpa_wrap_mod._invoke_v3_kernel(
                                functools.partial(rpa_v3_mod.ragged_paged_attention, _interpret=interp),
                                queries=q_in,
                                keys=k_in,
                                values=v_in,
                                kv_cache_fused=_u8_to_fp8(cache_u8),
                                kv_lens=kl_arr,
                                page_indices=pi_arr,
                                cu_q_lens=cq_1,
                                cu_kv_lens=ckv_1,
                                distribution=dist_1,
                                attention_sink=None,
                                sm_scale=sm_scale,
                                sliding_window=None,
                                soft_cap=None,
                                k_scale=1.0,
                                v_scale=1.0,
                                prepared_inputs=is_prep,
                                skip_decode=True,
                                a3_out_layout=a3_arg,
                            )
                            got = np.asarray(out_res.astype(jnp.float32)).reshape(num_q_tok, 512)
                            self.assertFalse(np.isnan(got).any(), f"NaN in {num_q_tok=} {is_prep=} {klen=}")
                            self.assertFalse(np.isinf(got).any(), f"Inf in {num_q_tok=} {is_prep=} {klen=}")
                            ref = _ref_attention_gqa(
                                q_f32[:1],
                                k_all,
                                v_all,
                                q_first=klen - 1,
                                kv_len=klen,
                                sm_scale=sm_scale,
                                sliding_window=None,
                            )
                            cos = float(np.sum(got[:1] * ref) / (np.linalg.norm(got[:1]) * np.linalg.norm(ref)))
                            self.assertGreater(cos, 0.998)

                            # Gate (#2983): assert padded Q rows caused ZERO extra KV writes!
                            upd_u8 = _fp8_to_u8(updated_cache)
                            target_pg = 1 + (klen - 1) // page_size
                            target_row = (klen - 1) % page_size
                            for pg_chk in range(8):
                                if pg_chk != target_pg:
                                    np.testing.assert_array_equal(upd_u8[pg_chk], cache_u8[pg_chk])
                                else:
                                    np.testing.assert_array_equal(upd_u8[pg_chk, :target_row], cache_u8[pg_chk, :target_row])
                                    np.testing.assert_array_equal(
                                        upd_u8[pg_chk, target_row + 1 :], cache_u8[pg_chk, target_row + 1 :]
                                    )
                                    np.testing.assert_array_equal(upd_u8[pg_chk, target_row, 0, :2, :64], new_kv_u8)

            # 3. Defense-in-depth (`stock MIXED` with `zero_kv_tail` + `@pl.when(sz > 0)` + `valid_lane_dim`):
            #    Verify BOTH `0x7F` unused lanes AND multi-`bq` `sliding_window` (`q_len=96 > bq_sz=32`) KV writes!
            q_sw_f32 = (rng.standard_normal((128, 8, 64)) * 0.5).astype(np.float32)
            kv_sw_u8 = np.full((128, 1, 4, 128), 0x7F, dtype=np.uint8)
            valid_sw_u8 = _rand_fp8_u8(rng, (96, 2, 64), scale=0.4)
            kv_sw_u8[:96, 0, :2, :64] = valid_sw_u8
            cache_sw_u8 = np.full((8, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
            out_sw_mixed, upd_sw_cache = rpa_v3_mod.ragged_paged_attention(
                _pack_q_tp16_5d(q_sw_f32),
                _u8_to_fp8(kv_sw_u8),
                None,
                _u8_to_fp8(cache_sw_u8),
                jnp.array([96, 0, 0, 0], dtype=jnp.int32),
                jnp.array([1, 0, 0, 0] + [0] * 12, dtype=jnp.int32),
                jnp.array([0, 96, 96, 96, 96], dtype=jnp.int32),
                ckv_1,
                dist_1,
                None,
                None,
                sm_scale=sm_scale,
                sliding_window=64,
                k_scale=1.0,
                v_scale=1.0,
                m_block_sizes=(32, 512, 32, 512),
                _skip_decode=True,
                _prepared_inputs=True,
                _fused_out_pack=-2,
                _interpret=interp,
            )
            got_sw_mixed = np.asarray(out_sw_mixed[:96].astype(jnp.float32))
            self.assertFalse(np.isnan(got_sw_mixed).any())
            upd_sw_u8 = _fp8_to_u8(upd_sw_cache)
            np.testing.assert_array_equal(upd_sw_u8[1, :96, 0, :2, :64], valid_sw_u8)

            # 4. #3029 §1 & §2 + #3033 §1:
            #    (a) TP=8 nod2 stock MIXED 0.0-ULP bitwise identity (`_mixed_zero_kv_tail=True` vs `False`) on finite inputs!
            for num_q_tok in (1, 128):
                for klen in (1, 615):
                    q_tp8_np = (rng.standard_normal((num_q_tok, 16, 64)) * 0.5).astype(np.float32)
                    kv_tp8_u8 = _rand_fp8_u8(rng, (num_q_tok, 1, 4, 128), scale=0.4)
                    cache_tp8_u8 = _rand_fp8_u8(rng, (8, page_size, 1, 4, 128), scale=0.4)
                    npg_8 = (klen + page_size - 1) // page_size
                    kl_8 = jnp.array([klen, 0, 0, 0], dtype=jnp.int32)
                    pi_8 = jnp.array([1 + p for p in range(npg_8)] + [0] * (16 - npg_8), dtype=jnp.int32)
                    out_zt1, kv_zt1 = rpa_v3_mod.ragged_paged_attention(
                        _pack_q_tp8_5d(q_tp8_np),
                        _u8_to_fp8(kv_tp8_u8),
                        None,
                        _u8_to_fp8(cache_tp8_u8),
                        kl_8,
                        pi_8,
                        cq_1,
                        ckv_1,
                        dist_1,
                        None,
                        None,
                        sm_scale=sm_scale,
                        k_scale=1.0,
                        v_scale=1.0,
                        _skip_decode=True,
                        _prepared_inputs=True,
                        _fused_out_pack=2,
                        _mixed_zero_kv_tail=True,
                        _interpret=interp,
                    )
                    out_zt0, kv_zt0 = rpa_v3_mod.ragged_paged_attention(
                        _pack_q_tp8_5d(q_tp8_np),
                        _u8_to_fp8(kv_tp8_u8),
                        None,
                        _u8_to_fp8(cache_tp8_u8),
                        kl_8,
                        pi_8,
                        cq_1,
                        ckv_1,
                        dist_1,
                        None,
                        None,
                        sm_scale=sm_scale,
                        k_scale=1.0,
                        v_scale=1.0,
                        _skip_decode=True,
                        _prepared_inputs=True,
                        _fused_out_pack=2,
                        _mixed_zero_kv_tail=False,
                        _interpret=interp,
                    )
                    np.testing.assert_array_equal(np.asarray(out_zt1.astype(jnp.float32)), np.asarray(out_zt0.astype(jnp.float32)))
                    np.testing.assert_array_equal(_fp8_to_u8(kv_zt1), _fp8_to_u8(kv_zt0))

            #    (b) TP=16 padded-RPAm (`T=128 -> 256`) vs unpadded-RPAm (`T=256` with non-zero garbage in `[128:256]`) is 0.0-ULP!
            #    (c) #3033 §1: real JAX array call fails loudly with AssertionError if `jax_rpa` returns non-tuple!
            with patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
                    "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                    "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                    "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                },
                clear=False,
            ):
                q_256_np = (rng.standard_normal((256, 8, 64)) * 2.0).astype(np.float32)
                kv_256_u8 = _rand_fp8_u8(rng, (256, 1, 4, 128), scale=1.0)
                cache_p_u8 = np.full((8, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
                cache_p_u8[1:3, :, 0, :2, :64] = _rand_fp8_u8(rng, (2, page_size, 2, 64), scale=0.4)
                kl_p = jnp.array([615, 0, 0, 0], dtype=jnp.int32)
                pi_p = jnp.array([1, 2, 0, 0] + [0] * 12, dtype=jnp.int32)
                out_pad128, kv_pad128 = rpa_wrap_mod._invoke_v3_kernel(
                    functools.partial(rpa_v3_mod.ragged_paged_attention, _interpret=interp),
                    queries=_pack_q_tp16_5d(q_256_np[:128]),
                    keys=_u8_to_fp8(kv_256_u8[:128]),
                    values=None,
                    kv_cache_fused=_u8_to_fp8(cache_p_u8),
                    kv_lens=kl_p,
                    page_indices=pi_p,
                    cu_q_lens=cq_1,
                    cu_kv_lens=ckv_1,
                    distribution=dist_1,
                    attention_sink=None,
                    sm_scale=sm_scale,
                    sliding_window=None,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    prepared_inputs=True,
                    skip_decode=True,
                    a3_out_layout=-2,
                )
                out_unpad256, kv_unpad256 = rpa_wrap_mod._invoke_v3_kernel(
                    functools.partial(rpa_v3_mod.ragged_paged_attention, _interpret=interp),
                    queries=_pack_q_tp16_5d(q_256_np),
                    keys=_u8_to_fp8(kv_256_u8),
                    values=None,
                    kv_cache_fused=_u8_to_fp8(cache_p_u8),
                    kv_lens=kl_p,
                    page_indices=pi_p,
                    cu_q_lens=cq_1,
                    cu_kv_lens=ckv_1,
                    distribution=dist_1,
                    attention_sink=None,
                    sm_scale=sm_scale,
                    sliding_window=None,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    prepared_inputs=True,
                    skip_decode=True,
                    a3_out_layout=-2,
                )
                np.testing.assert_array_equal(
                    np.asarray(out_pad128.astype(jnp.float32)),
                    np.asarray(out_unpad256[:128].astype(jnp.float32)),
                )
                np.testing.assert_array_equal(_fp8_to_u8(kv_pad128), _fp8_to_u8(kv_unpad256))

                with self.assertRaises(AssertionError):
                    rpa_wrap_mod._invoke_v3_kernel(
                        lambda *a, **kw: None,
                        queries=_pack_q_tp16_5d(q_256_np[:128]),
                        keys=_u8_to_fp8(kv_256_u8[:128]),
                        values=None,
                        kv_cache_fused=_u8_to_fp8(cache_p_u8),
                        kv_lens=kl_p,
                        page_indices=pi_p,
                        cu_q_lens=cq_1,
                        cu_kv_lens=ckv_1,
                        distribution=dist_1,
                        attention_sink=None,
                        sm_scale=sm_scale,
                        sliding_window=None,
                        soft_cap=None,
                        k_scale=1.0,
                        v_scale=1.0,
                        prepared_inputs=True,
                        skip_decode=True,
                        a3_out_layout=-2,
                    )

    def test_tpu7x_mosaic_exports_w9_mk128_tp8_and_tp16(self):
        try:
            from jax._src import tpu_info as _tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available in lightweight CPU venv")

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import ragged_paged_attention
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        swa_opts = frozenset({
            "stationary_q_m0_ring_fuse", "skip_empty_cdma", "mixed_s3",
            "sw2048_fuse", "sw_short_v2", "fp8_qk", "mk128", "la2", "sub8_l", "bf16_pop", "unroll_up", "mrb3",
        })
        full_opts = frozenset({
            "stationary_q_m0_ring_fuse", "skip_empty_cdma", "bkv_full_unroll",
            "mixed_s3", "prefix_fast", "fp8_qk", "mk128", "la2", "sub8_l", "bf16_pop", "unroll_up",
        })
        lc_opts = frozenset({
            "stationary_q_m0_ring_fuse", "skip_empty_cdma", "bkv_full_unroll",
            "prefix_fast", "staged_qo", "fp8_qk", "mk128", "la2", "sub8_l", "bf16_pop", "unroll_up",
        })

        cases = (
            # (tag, qh, t_tok, pg_sz, pps, mb, sw, opts, use_d2, prepared, use_a3)
            ("tp8_swa_16k_mk128_p512_d2", 8, 8192, 512, 8, (256, 2048, 256, 512), 2048, swa_opts, True, True, True),
            ("tp8_full_16k_mk128_p512_d2", 8, 8192, 512, 36, (512, 4096, 256, 512), None, full_opts, True, True, True),
            ("tp8_lc_bq4k_bkv8k_mk128_p512_d2", 8, 4096, 512, 545, (4096, 8192, 256, 512), None, lc_opts, True, True, True),
            ("tp8_lc_bq4k_bkv4k_mk128_p512_d2", 8, 4096, 512, 545, (4096, 4096, 256, 512), None, lc_opts, True, True, True),
            ("tp8_lc_bq4k_bkv4k_gqa8_p512_d2", 8, 4096, 512, 545, (4096, 4096, 256, 512), None, lc_opts | {"gqa8"}, True, True, True),
            ("tp16_swa_16k_mk128_prep_p512", 4, 8192, 512, 8, (256, 2048, 256, 512), 2048, swa_opts | {"tp16_sq"}, False, True, True),
            ("tp16_full_16k_mk128_prep_p512", 4, 8192, 512, 36, (512, 4096, 256, 512), None, full_opts | {"tp16_sq"}, False, True, True),
            ("tp16_lc_bq4k_bkv8k_mk128_prep_p512", 4, 4096, 512, 545, (4096, 8192, 256, 512), None, lc_opts | {"tp16_sq"}, False, True, True),
            ("tp16_lc_bq4k_bkv8k_gqa8_prep_p512", 4, 4096, 512, 545, (4096, 8192, 256, 512), None, lc_opts | {"tp16_sq", "gqa8"}, False, True, True),
            ("tp16_lc_bq4k_bkv8k_mk128_unprep_p512", 8, 8192, 512, 545, (4096, 8192, 256, 512), None, lc_opts | {"tp16_sq"}, False, False, False),
        )
        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            for tag, qh, t_tok, pg_sz, pps, mb, sw, opts, use_d2, prepared, use_a3 in cases:
                with self.subTest(export=tag):
                    phys_pg = pg_sz // 2 if use_d2 else pg_sz
                    c_sd = jax.ShapeDtypeStruct((2048, phys_pg, 1, 4, 128), jnp.float8_e4m3fn)
                    kl_sd = jax.ShapeDtypeStruct((64,), jnp.int32)
                    pi_sd = jax.ShapeDtypeStruct((64 * pps,), jnp.int32)
                    cq_sd = jax.ShapeDtypeStruct((65,), jnp.int32)
                    ckv_sd = jax.ShapeDtypeStruct((65,), jnp.int32)
                    d_sd = jax.ShapeDtypeStruct((3,), jnp.int32)
                    if prepared:
                        exported = jax_export.export(
                            jax.jit(
                                lambda q, kv, c, kl, pi, cq, ckv, d, _mb=mb, _sw=sw, _op=opts, _d2=use_d2, _a3=use_a3: (
                                    ragged_paged_attention(
                                        q, kv, None, c, kl, pi, cq, ckv, d, None, None,
                                        sm_scale=0.125, sliding_window=_sw, k_scale=1.0, v_scale=1.0,
                                        m_block_sizes=_mb, _prefill_opts=_op, _fused_out_pack=2,
                                        _prepared_inputs=True, _skip_decode=True, _a3_out_layout=_a3,
                                        _kv_pack_dense2=_d2,
                                    )
                                )
                            ),
                            platforms=["tpu"],
                        )(
                            jax.ShapeDtypeStruct((1, t_tok, qh, 2, 128), jnp.bfloat16),
                            jax.ShapeDtypeStruct((t_tok, 1, 4, 128), jnp.float8_e4m3fn),
                            c_sd, kl_sd, pi_sd, cq_sd, ckv_sd, d_sd,
                        )
                        if use_a3:
                            self.assertEqual(exported.out_avals[0].shape, (t_tok, 128 * qh))
                    else:
                        exported = jax_export.export(
                            jax.jit(
                                lambda q, k, v, c, kl, pi, cq, ckv, d, _mb=mb, _sw=sw, _op=opts, _d2=use_d2: (
                                    ragged_paged_attention(
                                        q, k, v, c, kl, pi, cq, ckv, d, None, None,
                                        sm_scale=0.125, sliding_window=_sw, k_scale=1.0, v_scale=1.0,
                                        m_block_sizes=_mb, _prefill_opts=_op, _skip_decode=True,
                                        _kv_pack_dense2=_d2,
                                    )
                                )
                            ),
                            platforms=["tpu"],
                        )(
                            jax.ShapeDtypeStruct((t_tok, qh, 64), jnp.bfloat16),
                            jax.ShapeDtypeStruct((t_tok, 1, 64), jnp.float8_e4m3fn),
                            jax.ShapeDtypeStruct((t_tok, 1, 64), jnp.float8_e4m3fn),
                            c_sd, kl_sd, pi_sd, cq_sd, ckv_sd, d_sd,
                        )
                        self.assertEqual(exported.out_avals[0].shape, (t_tok, qh, 64))
                    self.assertIn("RPAm", exported.mlir_module())

            # Also verify TPU7x Mosaic export of defense-in-depth `stock MIXED` `-zt` (#2942 / #2982)!
            with self.subTest(export="tp16_stock_mixed_zt_p512"):
                c_sd = jax.ShapeDtypeStruct((2048, 512, 1, 4, 128), jnp.float8_e4m3fn)
                kl_sd = jax.ShapeDtypeStruct((64,), jnp.int32)
                pi_sd = jax.ShapeDtypeStruct((64 * 36,), jnp.int32)
                cq_sd = jax.ShapeDtypeStruct((65,), jnp.int32)
                ckv_sd = jax.ShapeDtypeStruct((65,), jnp.int32)
                d_sd = jax.ShapeDtypeStruct((3,), jnp.int32)
                exp_zt = jax_export.export(
                    jax.jit(
                        lambda q, kv, c, kl, pi, cq, ckv, d: ragged_paged_attention(
                            q, kv, None, c, kl, pi, cq, ckv, d, None, None,
                            sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
                            _fused_out_pack=-2, _prepared_inputs=True, _skip_decode=True,
                        )
                    ),
                    platforms=["tpu"],
                )(
                    jax.ShapeDtypeStruct((1, 128, 4, 2, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((128, 1, 4, 128), jnp.float8_e4m3fn),
                    c_sd, kl_sd, pi_sd, cq_sd, ckv_sd, d_sd,
                )
                self.assertEqual(exp_zt.out_avals[0].shape, (128, 512))
                self.assertIn("-zt", exp_zt.mlir_module())

    def test_fp8_qk_dyn_clip_and_poisoned_hbm_tp16_rpam(self):
        """Regression (#3203/#3194): fp8_qk_dyn Q clip + 0x7F KV / NaN pad HBM stay finite on TP=16 RPAm."""
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas import ragged_paged_attention_v3 as rpa_v3_mod
        from sglang_tpu import attention_pallas_rpa_v3 as rpa_wrap_mod

        rng = np.random.default_rng(3203)
        page_size = 512
        interp = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")

        cases = (
            ("fp8_qk_dyn_swa", 256, 615, 512, 2048, frozenset({
                "stationary_q_m0_ring_fuse", "skip_empty_cdma", "mixed_s3",
                "sw2048_fuse", "sw_short_v2", "fp8_qk_dyn", "mk128", "la2",
                "sub8_l", "bf16_pop", "unroll_up", "mrb3", "tp16_sq",
            })),
            ("poison_q1_kv1_t512_full", 1, 1, 512, None, None),
            ("poison_q1_kv1_t512_swa", 1, 1, 512, 2048, None),
            ("poison_l61_q1_kv615_t128", 1, 615, 128, None, None),
        )
        env_patch = {
            "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
            "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
            "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
            "SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE": "1",
            "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
            "SGLANG_TPU_MUSE_KERNEL_A3_RPA_OUT": "0",
            "SGLANG_TPU_KV_PACK_HEADS": "2",
        }
        with patch.dict(os.environ, env_patch, clear=False), patch.object(
            rpa_v3_mod, "get_tpu_version", return_value=7
        ):
            for tag, ql, kl, max_tok, sw, direct_opts in cases:
                with self.subTest(case=tag):
                    q_np = rng.standard_normal((max_tok, 8, 64)).astype(np.float32) * 0.5
                    if "fp8_qk_dyn" in tag:
                        # Mix tiny (<=1e-12 floor) and moderate values to stress 448.0 / q_amax clip without exp2 overflow.
                        q_np[:ql:3] *= 8.0
                        q_np[1:ql:3] *= 1e-13
                    q_jax = _pack_q_tp16_5d(q_np)
                    q_jax = q_jax.at[:, ql:, :, :, :].set(jnp.bfloat16(jnp.nan))

                    kv_u8 = np.array(
                        _rand_fp8_u8(rng, (max_tok, 1, 4, 128), scale=0.5), copy=True
                    )
                    kv_u8[ql:, :, :, :] = 0x7F
                    kv_jax = _u8_to_fp8(kv_u8)

                    cache_u8 = np.full((6, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
                    gap = kl - ql
                    for pg in range((gap + page_size - 1) // page_size):
                        cache_u8[pg, :, 0, :2, :64] = _rand_fp8_u8(
                            rng, (page_size, 2, 64), scale=0.4
                        )
                    kv_cache = _u8_to_fp8(cache_u8)

                    kl_arr = jnp.array([kl, 0, 0, 0], dtype=jnp.int32)
                    cq_arr = jnp.array([0, ql, ql, ql, ql], dtype=jnp.int32)
                    ckv_arr = jnp.array([0, kl, kl, kl, kl], dtype=jnp.int32)
                    pi_arr = jnp.array([0, 1, 2, 0] + [0] * 12, dtype=jnp.int32)
                    dist_arr = jnp.array([0, 0, 1], dtype=jnp.int32)

                    if direct_opts is not None:
                        out, _ = rpa_v3_mod.ragged_paged_attention(
                            q_jax,
                            kv_jax,
                            None,
                            kv_cache,
                            kl_arr,
                            pi_arr,
                            cq_arr,
                            ckv_arr,
                            dist_arr,
                            None,
                            None,
                            sm_scale=0.125,
                            sliding_window=sw,
                            k_scale=1.0,
                            v_scale=1.0,
                            m_block_sizes=(256, 2048, 256, 512),
                            _prefill_opts=direct_opts,
                            _fused_out_pack=-2,
                            _prepared_inputs=True,
                            _skip_decode=True,
                            _a3_out_layout=True,
                            _interpret=interp,
                        )
                    else:
                        out, _ = rpa_wrap_mod._invoke_v3_kernel(
                            functools.partial(rpa_v3_mod.ragged_paged_attention, _interpret=interp),
                            queries=q_jax,
                            keys=kv_jax,
                            values=None,
                            kv_cache_fused=kv_cache,
                            kv_lens=kl_arr,
                            page_indices=pi_arr,
                            cu_q_lens=cq_arr,
                            cu_kv_lens=ckv_arr,
                            distribution=dist_arr,
                            attention_sink=None,
                            sm_scale=0.125,
                            sliding_window=sw,
                            soft_cap=None,
                            k_scale=1.0,
                            v_scale=1.0,
                            prepared_inputs=True,
                            skip_decode=True,
                            a3_out_layout=-2,
                        )
                    out_np = np.asarray(out.astype(jnp.float32))
                    self.assertTrue(np.all(np.isfinite(out_np[:ql])), f"{tag}: non-finite valid output")
                    self.assertTrue(np.all(out_np[ql:] == 0.0), f"{tag}: non-zero pad rows")

    def test_w910_optin_levers_parity_and_24k_32k_exports(self):
        """Verify opt-in Full fp8_qkpv_norm, sw_bq512, TP=16 sw_mk128, and T=24K/32K RPAm (#3395)."""
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas import ragged_paged_attention_v3 as rpa_v3_mod
        from sglang_tpu import attention_pallas_rpa_v3 as rpa_wrap_mod

        # 1. Verify opt-in routing scopes (without changing default w9 behavior).
        class _FakeRng:
            def __init__(self, sh):
                self.shape = sh
                self.ndim = len(sh)
                self.dtype = "bfloat16"

        class _FakeCache:
            def __init__(self, sh):
                self.shape = sh
                self.ndim = len(sh)
                self.dtype = "float8_e4m3fn"

        def _capture_route(w7_opts, *, t_tok, qh, sw, pps):
            captured = {}

            def _stub(*args, **kwargs):
                captured.update(kwargs)
                return ("out", "kvc")

            env = {
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS": w7_opts,
                "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
                "SGLANG_TPU_KV_PACK_HEADS": "2",
            }
            with patch.dict(os.environ, env, clear=False):
                rpa_wrap_mod._invoke_v3_kernel(
                    _stub,
                    queries=_FakeRng((1, t_tok, qh, 2, 128)),
                    keys=_FakeRng((t_tok, 1, 4, 128)),
                    values=None,
                    kv_cache_fused=_FakeCache((2048, 512, 1, 4, 128)),
                    kv_lens=_FakeRng((64,)),
                    page_indices=_FakeRng((64 * pps,)),
                    cu_q_lens=_FakeRng((65,)),
                    cu_kv_lens=_FakeRng((65,)),
                    distribution=_FakeRng((3,)),
                    attention_sink=None,
                    sm_scale=0.125,
                    sliding_window=sw,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    prepared_inputs=True,
                    skip_decode=True,
                    a3_out_layout=-2,
                )
            return captured

        c_full_qkpv = _capture_route(
            "full_fp8_qkpv,full_fp8_qkpv_norm,sw_bq512,sw_mk128",
            t_tok=10240,
            qh=4,
            sw=None,
            pps=36,
        )
        self.assertEqual(c_full_qkpv["m_block_sizes"], (512, 4096, 256, 512))
        self.assertIn("fp8_qkpv", c_full_qkpv["_prefill_opts"])
        self.assertIn("fp8_qkpv_norm", c_full_qkpv["_prefill_opts"])

        c_swa_512 = _capture_route(
            "full_fp8_qkpv,full_fp8_qkpv_norm,sw_bq512,sw_mk128",
            t_tok=10240,
            qh=4,
            sw=2048,
            pps=8,
        )
        self.assertEqual(c_swa_512["m_block_sizes"], (512, 2048, 256, 512))
        self.assertIn("mk128", c_swa_512["_prefill_opts"])
        self.assertNotIn("fp8_qkpv", c_swa_512["_prefill_opts"])

        c_full_24k = _capture_route(
            "full_bq1024,full_staged_qo",
            t_tok=16384,
            qh=8,
            sw=None,
            pps=48,
        )
        self.assertEqual(c_full_24k["m_block_sizes"], (1024, 4096, 256, 512))
        self.assertIn("staged_qo", c_full_24k["_prefill_opts"])

        # 2. CPU InterpretParams(detect_races=True, uninitialized_memory="nan") + 0x7F KV poison parity.
        rng = np.random.default_rng(3395)
        page_size = 512
        interp = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")

        for tp_mode in ("tp8", "tp16"):
            qh = 8 if tp_mode == "tp8" else 4
            fused_pack = 2 if tp_mode == "tp8" else -2
            extra_opt = set() if tp_mode == "tp8" else {"tp16_sq"}

            t_tok = 512
            q_np = rng.normal(0.0, 0.4, (1, t_tok, qh, 2, 128)).astype(np.float32)
            if tp_mode == "tp16":
                q_np[..., 64:] = 0.0
            q_jax = jnp.asarray(q_np, dtype=jnp.bfloat16)
            kv_jax = jnp.asarray(rng.normal(0.0, 0.4, (t_tok, 1, 4, 128)), dtype=jnp.float8_e4m3fn)
            cache_u8 = np.full((6, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
            cache_u8[0, :, 0, :2, : (128 if tp_mode == "tp8" else 64)] = 0x30
            kv_cache = _u8_to_fp8(cache_u8)
            kl = jnp.array([768, 0, 0, 0], dtype=jnp.int32)
            cq = jnp.array([0, 512, 512, 512, 512], dtype=jnp.int32)
            ckv = jnp.array([0, 768, 768, 768, 768], dtype=jnp.int32)
            pi = jnp.array([0, 1, 2, 3] * 4, dtype=jnp.int32)
            dist = jnp.array([0, 0, 1], dtype=jnp.int32)

            base_full = frozenset(
                {
                    "stationary_q_m0_ring_fuse",
                    "skip_empty_cdma",
                    "bkv_full_unroll",
                    "mixed_s3",
                    "prefix_fast",
                    "fp8_qk",
                    "mk128",
                    "la2",
                    "sub8_l",
                    "bf16_pop",
                    "unroll_up",
                }
                | extra_opt
            )
            with patch.object(rpa_v3_mod, "get_tpu_version", return_value=7):
                out_base, _ = rpa_v3_mod.ragged_paged_attention(
                    jnp.array(q_jax), jnp.array(kv_jax), None, jnp.array(kv_cache),
                    kl, pi, cq, ckv, dist, None, None,
                    sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    m_block_sizes=(512, 4096, 256, 512), _prefill_opts=base_full,
                    _fused_out_pack=fused_pack, _prepared_inputs=True,
                    _skip_decode=True, _a3_out_layout=True, _interpret=interp,
                )
                out_fp8pv, _ = rpa_v3_mod.ragged_paged_attention(
                    jnp.array(q_jax), jnp.array(kv_jax), None, jnp.array(kv_cache),
                    kl, pi, cq, ckv, dist, None, None,
                    sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    m_block_sizes=(512, 4096, 256, 512),
                    _prefill_opts=base_full | {"fp8_qkpv", "fp8_qkpv_norm"},
                    _fused_out_pack=fused_pack, _prepared_inputs=True,
                    _skip_decode=True, _a3_out_layout=True, _interpret=interp,
                )
            b_np = np.asarray(out_base, dtype=np.float32)
            p_np = np.asarray(out_fp8pv, dtype=np.float32)
            self.assertTrue(np.all(np.isfinite(p_np)), f"{tp_mode} full_fp8_qkpv_norm non-finite")
            cos = float(np.dot(b_np.ravel(), p_np.ravel()) / (np.linalg.norm(b_np) * np.linalg.norm(p_np) + 1e-12))
            self.assertGreater(cos, 0.999, f"{tp_mode} full_fp8_qkpv_norm cos={cos}")

            kl_sw = jnp.array([2560, 0, 0, 0], dtype=jnp.int32)
            ckv_sw = jnp.array([0, 2560, 2560, 2560, 2560], dtype=jnp.int32)
            pi_sw = jnp.arange(24, dtype=jnp.int32)
            cache_sw_u8 = np.full((26, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
            cache_sw_u8[:4, :, 0, :2, : (128 if tp_mode == "tp8" else 64)] = 0x30
            kv_cache_sw = _u8_to_fp8(cache_sw_u8)
            base_swa = frozenset(
                {
                    "stationary_q_m0_ring_fuse",
                    "skip_empty_cdma",
                    "mixed_s3",
                    "sw2048_fuse",
                    "sw_short_v2",
                    "fp8_qk",
                    "sub8_l",
                    "bf16_pop",
                    "mrb3",
                }
                | ({"mk128", "la2", "unroll_up"} if tp_mode == "tp8" else {"tp16_sq"})
            )
            with patch.object(rpa_v3_mod, "get_tpu_version", return_value=7):
                out_sw_base, _ = rpa_v3_mod.ragged_paged_attention(
                    jnp.array(q_jax), jnp.array(kv_jax), None, jnp.array(kv_cache_sw),
                    kl_sw, pi_sw, cq, ckv_sw, dist, None, None,
                    sm_scale=0.125, sliding_window=2048, k_scale=1.0, v_scale=1.0,
                    m_block_sizes=(256, 2048, 256, 512), _prefill_opts=base_swa,
                    _fused_out_pack=fused_pack, _prepared_inputs=True,
                    _skip_decode=True, _a3_out_layout=True, _interpret=interp,
                )
                out_sw_512, _ = rpa_v3_mod.ragged_paged_attention(
                    jnp.array(q_jax), jnp.array(kv_jax), None, jnp.array(kv_cache_sw),
                    kl_sw, pi_sw, cq, ckv_sw, dist, None, None,
                    sm_scale=0.125, sliding_window=2048, k_scale=1.0, v_scale=1.0,
                    m_block_sizes=(512, 2048, 256, 512),
                    _prefill_opts=base_swa | {"mk128", "la2", "unroll_up"},
                    _fused_out_pack=fused_pack, _prepared_inputs=True,
                    _skip_decode=True, _a3_out_layout=True, _interpret=interp,
                )
            sb_np = np.asarray(out_sw_base, dtype=np.float32)
            s5_np = np.asarray(out_sw_512, dtype=np.float32)
            self.assertTrue(np.all(np.isfinite(s5_np)), f"{tp_mode} sw_bq512 non-finite")
            cos_sw = float(np.dot(sb_np.ravel(), s5_np.ravel()) / (np.linalg.norm(sb_np) * np.linalg.norm(s5_np) + 1e-12))
            self.assertGreater(cos_sw, 0.999, f"{tp_mode} sw_bq512 cos={cos_sw}")

        # 3. TPU7x Mosaic AOT exports (16K fp8_qkpv_norm, sw_bq512, TP16 sw_mk128, and 24K/32K Full).
        try:
            from jax._src import tpu_info as _tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available in lightweight CPU venv")

        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        swa_opts = frozenset({
            "stationary_q_m0_ring_fuse", "skip_empty_cdma", "mixed_s3",
            "sw2048_fuse", "sw_short_v2", "fp8_qk", "mk128", "la2", "sub8_l",
            "bf16_pop", "unroll_up", "mrb3",
        })
        full_opts = frozenset({
            "stationary_q_m0_ring_fuse", "skip_empty_cdma", "bkv_full_unroll",
            "mixed_s3", "prefix_fast", "fp8_qk", "mk128", "la2", "sub8_l",
            "bf16_pop", "unroll_up",
        })
        export_cases = (
            ("tp8_full_16k_fp8_qkpv_norm", 8, 10240, 36, (512, 4096, 256, 512), None, full_opts | {"fp8_qkpv", "fp8_qkpv_norm"}),
            ("tp16_full_16k_fp8_qkpv_norm", 4, 10240, 36, (512, 4096, 256, 512), None, full_opts | {"tp16_sq", "fp8_qkpv", "fp8_qkpv_norm"}),
            ("tp8_swa_16k_bq512_mk128", 8, 10240, 8, (512, 2048, 256, 512), 2048, swa_opts),
            ("tp16_swa_16k_mk128", 4, 10240, 8, (256, 2048, 256, 512), 2048, swa_opts | {"tp16_sq"}),
            ("tp16_swa_16k_bq512_mk128", 4, 10240, 8, (512, 2048, 256, 512), 2048, swa_opts | {"tp16_sq"}),
            ("tp8_full_24k_bq512", 8, 16384, 48, (512, 4096, 256, 512), None, full_opts),
            ("tp8_full_32k_bq1024_staged", 8, 16384, 64, (1024, 4096, 256, 512), None, full_opts | {"staged_qo"}),
            ("tp16_full_24k_bq512", 4, 16384, 48, (512, 4096, 256, 512), None, full_opts | {"tp16_sq"}),
            ("tp16_full_32k_bq1024_staged", 4, 16384, 64, (1024, 4096, 256, 512), None, full_opts | {"tp16_sq", "staged_qo"}),
        )
        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            for tag, qh, t_tok, pps, mb, sw, opts in export_cases:
                with self.subTest(export=tag):
                    c_sd = jax.ShapeDtypeStruct((2048, 512, 1, 4, 128), jnp.float8_e4m3fn)
                    kl_sd = jax.ShapeDtypeStruct((64,), jnp.int32)
                    pi_sd = jax.ShapeDtypeStruct((64 * pps,), jnp.int32)
                    cq_sd = jax.ShapeDtypeStruct((65,), jnp.int32)
                    ckv_sd = jax.ShapeDtypeStruct((65,), jnp.int32)
                    d_sd = jax.ShapeDtypeStruct((3,), jnp.int32)
                    exp = jax_export.export(
                        jax.jit(
                            lambda q, kv, c, kl, pi, cq, ckv, d, _mb=mb, _sw=sw, _op=opts: rpa_v3_mod.ragged_paged_attention(
                                q, kv, None, c, kl, pi, cq, ckv, d, None, None,
                                sm_scale=0.125, sliding_window=_sw, k_scale=1.0, v_scale=1.0,
                                m_block_sizes=_mb, _prefill_opts=_op, _fused_out_pack=2,
                                _prepared_inputs=True, _skip_decode=True, _a3_out_layout=True,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, t_tok, qh, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((t_tok, 1, 4, 128), jnp.float8_e4m3fn),
                        c_sd, kl_sd, pi_sd, cq_sd, ckv_sd, d_sd,
                    )
                    self.assertIn("RPAm", exp.mlir_module())

    def test_cand13_lc_routing_bkv16k_and_tq_loop_parity(self):
        """Verify LC_OPTS partial override preservation, bkv16k routing, exact tq_loop parity, and TPU7x exports."""
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas import ragged_paged_attention_v3 as rpa_v3_mod
        from sglang_tpu import attention_pallas_rpa_v3 as rpa_wrap_mod

        class _FakeRng:
            def __init__(self, sh, dtype="bfloat16"):
                self.shape = sh
                self.ndim = len(sh)
                self.dtype = dtype

        def _capture_lc(lc_opts, *, t_tok, qh, sw, pps):
            captured = {}

            def _stub(*args, **kwargs):
                captured.update(kwargs)
                return ("out", "kvc")

            env = {
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_W9": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": lc_opts,
                "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                "SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
                "SGLANG_TPU_KV_PACK_HEADS": "2",
            }
            with patch.dict(os.environ, env, clear=False):
                rpa_wrap_mod._invoke_v3_kernel(
                    _stub,
                    queries=_FakeRng((1, t_tok, qh, 2, 128)),
                    keys=_FakeRng((t_tok, 1, 4, 128), "float8_e4m3fn"),
                    values=None,
                    kv_cache_fused=_FakeRng((2048, 512, 1, 4, 128), "float8_e4m3fn"),
                    kv_lens=_FakeRng((64,)),
                    page_indices=_FakeRng((64 * pps,)),
                    cu_q_lens=_FakeRng((65,)),
                    cu_kv_lens=_FakeRng((65,)),
                    distribution=_FakeRng((3,)),
                    attention_sink=None,
                    sm_scale=0.125,
                    sliding_window=sw,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    prepared_inputs=True,
                    skip_decode=True,
                    a3_out_layout=-2,
                )
            return captured

        # Partial LC_OPTS="bq8192" on TP=16 preserves bkv8192, s2 (no mixed_s3), prefix_fast, staged_qo, and mk128.
        c_bq8k = _capture_lc("bq8192", t_tok=16384, qh=4, sw=None, pps=540)
        self.assertEqual(c_bq8k["m_block_sizes"], (8192, 8192, 256, 512))
        self.assertIn("mk128", c_bq8k["_prefill_opts"])
        self.assertIn("prefix_fast", c_bq8k["_prefill_opts"])
        self.assertIn("staged_qo", c_bq8k["_prefill_opts"])
        self.assertNotIn("mixed_s3", c_bq8k["_prefill_opts"])

        # On TP=8, LC_OPTS="bq8192" avoids injecting bkv8192 (caps to 4096), while explicit bq8192,bkv8192 raises.
        c_tp8_bq8k = _capture_lc("bq8192", t_tok=16384, qh=8, sw=None, pps=540)
        self.assertEqual(c_tp8_bq8k["m_block_sizes"], (8192, 4096, 256, 512))
        self.assertNotIn("mixed_s3", c_tp8_bq8k["_prefill_opts"])
        with self.assertRaises(ValueError):
            _capture_lc("bq8192,bkv8192", t_tok=16384, qh=8, sw=None, pps=540)

        # Opt-in bkv16384 routes cleanly on >32K Full without affecting <=32K Full or SWA.
        c_bkv16k = _capture_lc("bkv16384", t_tok=16384, qh=8, sw=None, pps=540)
        self.assertEqual(c_bkv16k["m_block_sizes"], (4096, 16384, 256, 512))
        self.assertNotIn("mixed_s3", c_bkv16k["_prefill_opts"])

        c_short = _capture_lc("bkv16384", t_tok=16384, qh=8, sw=None, pps=48)
        self.assertEqual(c_short["m_block_sizes"], (512, 4096, 256, 512))
        self.assertIn("mixed_s3", c_short["_prefill_opts"])

        # CPU InterpretParams(detect_races=True, uninitialized_memory="nan") + 0x7F KV poison
        # verifying exact tq_loop bound cdiv(valid_bq_rows, tq_sz) on partial bq=512 (ql=128 -> 1 iter).
        rng = np.random.default_rng(1313)
        page_size = 512
        interp = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
        for tp_mode in ("tp8", "tp16"):
            qh = 8 if tp_mode == "tp8" else 4
            fused_pack = 2 if tp_mode == "tp8" else -2
            tp_opt = set() if tp_mode == "tp8" else {"tp16_sq"}
            q_np = rng.normal(0.0, 0.35, (1, 512, qh, 2, 128)).astype(np.float32)
            if tp_mode == "tp16":
                q_np[..., 64:] = 0.0
            q_jax = jnp.asarray(q_np, dtype=jnp.bfloat16)
            kv_jax = jnp.asarray(rng.normal(0.0, 0.35, (512, 1, 4, 128)), dtype=jnp.float8_e4m3fn)
            cache_u8 = np.full((6, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
            cache_u8[0, :, 0, :4, :] = 0x30
            if tp_mode == "tp16":
                cache_u8[0, :, 0, :, 64:] = 0
            kv_cache = _u8_to_fp8(cache_u8)
            ql = 128  # < 256 so cdiv(valid_bq_rows, 256) == 1 inside bq=512 (2 tq slots)
            kl = jnp.array([512 + ql, 0, 0, 0], dtype=jnp.int32)
            cq = jnp.array([0, ql, ql, ql, ql], dtype=jnp.int32)
            ckv = jnp.array([0, 512 + ql, 512 + ql, 512 + ql, 512 + ql], dtype=jnp.int32)
            pi = jnp.array([0, 1, 2, 3] * 4, dtype=jnp.int32)
            dist = jnp.array([0, 0, 1], dtype=jnp.int32)
            opts = frozenset({
                "stationary_q_m0_ring_fuse", "skip_empty_cdma", "bkv_full_unroll",
                "mixed_s3", "prefix_fast", "fp8_qk", "mk128", "la2", "sub8_l",
                "bf16_pop", "unroll_up",
            } | tp_opt)
            with patch.object(rpa_v3_mod, "get_tpu_version", return_value=7):
                out_loop, _ = rpa_v3_mod.ragged_paged_attention(
                    jnp.array(q_jax), jnp.array(kv_jax), None, jnp.array(kv_cache),
                    kl, pi, cq, ckv, dist, None, None,
                    sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    m_block_sizes=(512, 1024, 256, 512), _prefill_opts=opts,
                    _fused_out_pack=fused_pack, _prepared_inputs=True,
                    _skip_decode=True, _a3_out_layout=True, _interpret=interp,
                )
                out_unroll, _ = rpa_v3_mod.ragged_paged_attention(
                    jnp.array(q_jax), jnp.array(kv_jax), None, jnp.array(kv_cache),
                    kl, pi, cq, ckv, dist, None, None,
                    sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    m_block_sizes=(512, 1024, 256, 512), _prefill_opts=opts | {"tq_unroll"},
                    _fused_out_pack=fused_pack, _prepared_inputs=True,
                    _skip_decode=True, _a3_out_layout=True, _interpret=interp,
                )
            a = np.asarray(out_loop[:ql], dtype=np.float32)
            b = np.asarray(out_unroll[:ql], dtype=np.float32)
            self.assertTrue(np.all(np.isfinite(a)), f"{tp_mode} tq_loop non-finite")
            self.assertEqual(float(np.max(np.abs(a - b))), 0.0)

        try:
            from jax._src import tpu_info as _tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available in lightweight CPU venv")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        lc_mk_opts = frozenset({
            "stationary_q_m0_ring_fuse", "skip_empty_cdma", "bkv_full_unroll",
            "prefix_fast", "fp8_qk", "mk128", "la2", "sub8_l",
            "bf16_pop", "unroll_up", "staged_qo",
        })
        export_cases = (
            ("tp8_lc_mk128_bkv16k", 8, 2, 16384, 540, (4096, 16384, 256, 512), lc_mk_opts),
            ("tp16_lc_mk128_bkv16k", 4, -2, 16384, 540, (4096, 16384, 256, 512), lc_mk_opts | {"tp16_sq"}),
        )
        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            for tag, qh, fpack, t_tok, pps, mb, opts in export_cases:
                with self.subTest(export=tag):
                    exp = jax_export.export(
                        jax.jit(
                            lambda q, kv, c, kl, pi, cq, ckv, d, _mb=mb, _op=opts, _fp=fpack: rpa_v3_mod.ragged_paged_attention(
                                q, kv, None, c, kl, pi, cq, ckv, d, None, None,
                                sm_scale=0.125, sliding_window=None, k_scale=1.0, v_scale=1.0,
                                m_block_sizes=_mb, _prefill_opts=_op, _fused_out_pack=_fp,
                                _prepared_inputs=True, _skip_decode=True, _a3_out_layout=True,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, t_tok, qh, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((t_tok, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((2048, 512, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((64,), jnp.int32),
                        jax.ShapeDtypeStruct((64 * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((65,), jnp.int32),
                        jax.ShapeDtypeStruct((65,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    )
                    self.assertIn("RPAm", exp.mlir_module())


if __name__ == "__main__":
    unittest.main()
