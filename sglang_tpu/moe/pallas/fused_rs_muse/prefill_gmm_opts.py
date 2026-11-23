# SPDX-License-Identifier: Apache-2.0
"""Wave-8 Prefill MoE (`gmm_v2_fused_rs`) Deep Optimization Primitives.

Provides modular, zero-spill TPU7x Pallas & NumPy/JAX helpers for:
  1. `GMM_SCOPED_W4A8` (`1` / `2`): Direct VMEM slice + explicit `.astype(fp8)` +
     `m_sub = 128` zero-spill sub-tiling inside `inner_kernel` (eliminates the
     monolithic 8 MiB `[512, 4096] uint32` SSA load and 96-VREG accumulator spills).
  2. `GMM_EXPERT_OUTER_FP8` (`1` / `2`):
     - Mode 1: Per-strip `FP4 (block-16/256) -> Scaled-FP8 (per-column K=4096)`
       VMEM expansion + single-pop `K=4096` `FP8 x FP8 -> F32` MXU matmul in
       `m_sub = 128` sub-tiles (collapses 16 K256 MXU pops -> 1 pop, 0 inner-K VPU
       ops, <=32 live f32 VREGs).
     - Mode 2: Pair-batched (`G_BATCH = 2` consecutive same-expert `gm` tiles = 512
       rows at `tile_m=256`) strip-major execution reusing 1x VPU `FP4 -> Scaled-FP8`
       expansion across both `gm` tiles with exact gather/staging/W1/W2 parity.
  3. `PREFILL_STATIC_FP8_RS`: Mathematical unit-RMS static-FP8 scaling helpers
     (`|x_j| <= sqrt(4096) = 64.0`, so `S = 4.0` satisfies `64 * 4 = 256 <= 448`
     with zero clipping on any unit-RMS vector).
  4. `PREFILL_CHIPMATE_SLOTS = 2`: Two-round greedy intra-chip chipmate whole/split
     rebalance (`E_34`) in both Pallas VMEM+SMEM and pure-JAX / NumPy reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_FP4_PACK = 8
_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0


def resolve_static_fp8_rs_scale(raw_scale: float | int | None) -> tuple[float, bool]:
    """Validate and normalize `SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS`.

    Returns `(abs_scale, keep_fp8_out)`:
      - `0.0`: disabled -> `(0.0, False)`.
      - `> 0` (e.g. `4.0`): FP8 ICI RS + keep `float8_e4m3fn` output buffer for `Kernel D`.
      - `< 0` (e.g. `-4.0`): FP8 ICI RS + dequantize back to `bfloat16` after RS so any
        downstream `Kernel D` variant (`D_MXU_COMBINE`, etc.) consumes `bf16` unchanged.
    """
    if raw_scale is None:
        return 0.0, False
    val = float(raw_scale)
    if val == 0.0:
        return 0.0, False
    abs_val = abs(val)
    if abs_val > 7.0:
        raise ValueError(
            f"abs(SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS)={abs_val} exceeds the "
            "guaranteed zero-clipping bound 448 / sqrt(4096) = 7.0 (recommended: 4.0 or -4.0)."
        )
    return abs_val, (val > 0.0)


# =============================================================================
# Pure-NumPy Reference Models (for fast deterministic CPU numerical tests)
# =============================================================================


def simulate_e4m3fn_numpy(x: np.ndarray) -> np.ndarray:
    """Exact IEEE-like `float8_e4m3fn` round-to-nearest-even simulator in NumPy."""
    arr = np.asarray(x, dtype=np.float32)
    sign = np.sign(arr)
    ax = np.clip(np.abs(arr), 0.0, _FP8_E4M3_MAX)
    out = np.zeros_like(ax, dtype=np.float32)

    sub_mask = (ax > 0.0) & (ax < 2.0**-6)
    if np.any(sub_mask):
        out[sub_mask] = np.rint(ax[sub_mask] * 512.0) * (1.0 / 512.0)

    norm_mask = ax >= 2.0**-6
    if np.any(norm_mask):
        v = ax[norm_mask]
        exp = np.clip(np.floor(np.log2(v)), -6.0, 8.0)
        step = (2.0**exp) * 0.125
        q = np.rint(v / step) * step
        carry = q >= (2.0 ** (exp + 1.0))
        if np.any(carry):
            exp_c = np.minimum(exp[carry] + 1.0, 8.0)
            step_c = (2.0**exp_c) * 0.125
            q[carry] = np.rint(v[carry] / step_c) * step_c
        out[norm_mask] = np.minimum(q, _FP8_E4M3_MAX)
    return (sign * out).astype(np.float32)


def scaled_fp8_1pop_strip_numpy_ref(
    lhs_fp8: np.ndarray,
    lhs_scale: np.ndarray,
    w_fp4_vals: np.ndarray,
    w_scale_blocks: np.ndarray,
    rhs_qbs: int,
) -> np.ndarray:
    """Reference for `expand_fp4_strip_to_scaled_fp8` + `single_pop_scaled_fp8_strip_acc`.

    Args:
        lhs_fp8: float32[M, K] already rounded to E4M3 codes.
        lhs_scale: float32[M, 1] per-row activation scale.
        w_fp4_vals: float32[K, N] decoded E2M1 codes in {0, +-0.5, ..., +-6.0}.
        w_scale_blocks: float32[K // rhs_qbs, N] block scales.
        rhs_qbs: block size along K (e.g. 256 or 16).
    """
    k_dim, _ = w_fp4_vals.shape
    assert w_scale_blocks.shape[0] * rhs_qbs == k_dim
    col_max = np.max(np.abs(w_scale_blocks), axis=0, keepdims=True)
    col_c = col_max * (_FP4_MAX / _FP8_E4M3_MAX)
    rel_s = np.where(
        col_c == 0.0, 0.0, w_scale_blocks / np.where(col_c == 0.0, 1.0, col_c)
    )
    w_scaled_fp8 = simulate_e4m3fn_numpy(w_fp4_vals * np.repeat(rel_s, rhs_qbs, axis=0))
    pop = lhs_fp8.astype(np.float32) @ w_scaled_fp8.astype(np.float32)
    return pop * (lhs_scale.astype(np.float32) * col_c.astype(np.float32))


def plan_prefill_chipmate_2slot_numpy_ref(
    topk_indices: np.ndarray,
    *,
    ep_size: int,
    num_local_experts: int,
    tile_m: int = 256,
    chipmate_xor: int = 1,
    min_tile_gap: int = 3,
    mode: str = "split",
    weight_load_tiles: int = 1,
    slot_xors: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact multi-round deterministic NumPy reference for `PREFILL_CHIPMATE_SLOTS=2/3`.

    Returns:
        v_ids: int32[T, topk] remapped virtual expert IDs over `ep_size * (num_local_experts + n_cm)`
            (preserving sentinel `>= ep_size * num_local_experts` mapped to `ep_size * e_virt`).
        cm_plan: int32[ep_size, 3 * n_cm] per-rank `[send_0, recv_0, src_0, ...]`.
    """
    mode_norm = (mode or "whole").strip().lower()
    if mode_norm not in ("whole", "split"):
        raise ValueError(f"Unsupported prefill chipmate mode={mode!r}")

    eff_slot_xors = (
        tuple(slot_xors) if slot_xors is not None else (chipmate_xor, chipmate_xor)
    )
    n_cm = len(eff_slot_xors)
    ids = np.asarray(topk_indices, dtype=np.int32)
    e_loc = num_local_experts
    e_virt = e_loc + n_cm
    num_in = ep_size * e_loc

    valid_mask = (ids >= 0) & (ids < num_in)
    counts_flat = np.bincount(ids[valid_mask], minlength=num_in).astype(np.int32)[
        :num_in
    ]
    counts_rs = counts_flat.reshape(ep_size, e_loc).copy()
    tiles_rs = np.where(counts_rs > 0, (counts_rs + tile_m - 1) // tile_m, 0).astype(
        np.int32
    )
    cost_rs = np.where(counts_rs > 0, tiles_rs + weight_load_tiles, 0).astype(np.int32)
    loads = np.sum(cost_rs, axis=1).astype(np.int32)

    own = ids // e_loc
    sl = ids % e_loc
    base_v = np.where(ids >= num_in, ep_size * e_virt, own * e_virt + sl).astype(
        np.int32
    )

    total_elems = ids.size
    n_blocks = 64 if (total_elems >= 64 and total_elems % 64 == 0) else ep_size
    blk_elems = total_elems // n_blocks
    v_blk = base_v.reshape(n_blocks, blk_elems).copy()

    cm_plan = np.zeros((ep_size, 3 * n_cm), dtype=np.int32)
    for rnd in range(n_cm):
        cm_plan[:, 3 * rnd] = -1

    used_mask = np.zeros((ep_size, e_loc), dtype=bool)
    for rnd, r_xor in enumerate(eff_slot_xors):
        pairs = [
            (r, r ^ r_xor)
            for r in range(ep_size)
            if 0 <= (r ^ r_xor) < ep_size and r < (r ^ r_xor)
        ]
        for r0, r1 in pairs:
            l0, l1 = int(loads[r0]), int(loads[r1])
            hi = r0 if l0 >= l1 else r1
            lo = r1 if l0 >= l1 else r0
            l_hi, l_lo = max(l0, l1), min(l0, l1)

            t_hi = np.where(used_mask[hi], 0, tiles_rs[hi])
            c_hi = np.where(t_hi > 0, t_hi + weight_load_tiles, 0)
            new_peaks = np.where(c_hi > 0, np.maximum(l_hi - c_hi, l_lo + c_hi), l_hi)
            best_s = int(np.argmin(new_peaks))
            whole_gain = l_hi - int(new_peaks[best_s])

            if mode_norm == "whole":
                do_any = whole_gain >= min_tile_gap
                use_whole = do_any
                chosen_s = best_s
                split_t = 0
                t_max = int(t_hi[best_s])
            else:
                max_s = int(np.argmax(t_hi))
                t_max = int(t_hi[max_s])
                split_raw = (l_hi - (l_lo + weight_load_tiles)) // 2
                split_t = int(np.clip(split_raw, 0, max(0, t_max - 1)))
                split_peak = max(l_hi - split_t, l_lo + split_t + weight_load_tiles)
                split_gain = (l_hi - split_peak) if split_t >= min_tile_gap else 0
                use_whole = (whole_gain >= split_gain) and (whole_gain >= min_tile_gap)
                use_split = (not use_whole) and (split_gain >= min_tile_gap)
                do_any = use_whole or use_split
                chosen_s = best_s if use_whole else max_s

            if not do_any:
                continue

            used_mask[hi, chosen_s] = True
            src_vid = hi * e_virt + chosen_s
            borrowed_vid = lo * e_virt + (e_loc + rnd)

            if use_whole:
                v_blk = np.where(v_blk == src_vid, borrowed_vid, v_blk)
                delta_c = int(c_hi[chosen_s])
                loads[hi] -= delta_c
                loads[lo] += delta_c
                tiles_rs[hi, chosen_s] = 0
            else:
                keep_rows = (t_max - split_t) * tile_m
                hit_blk = v_blk == src_vid
                blk_cnt = np.sum(hit_blk.astype(np.int32), axis=1)
                cum_blk = np.cumsum(blk_cnt) - blk_cnt
                mid_row = cum_blk + (blk_cnt >> 1)
                to_lo_blk = mid_row >= keep_rows
                v_blk = np.where(hit_blk & to_lo_blk[:, None], borrowed_vid, v_blk)
                loads[hi] -= split_t
                loads[lo] += split_t + weight_load_tiles
                tiles_rs[hi, chosen_s] = max(0, int(tiles_rs[hi, chosen_s]) - split_t)

            col_base = 3 * rnd
            cm_plan[hi, col_base + 0] = chosen_s
            cm_plan[lo, col_base + 1] = 1
            cm_plan[lo, col_base + 2] = chosen_s

    return v_blk.reshape(ids.shape), cm_plan


# =============================================================================
# TPU7x Pallas / JAX Kernel Helpers
# =============================================================================


def _slice_pq_lhs_sub(
    prequant_lhs_fp8_ref: Any,
    prequant_lhs_3d_fp8_ref: Any,
    m0: int,
    m_rows_sub: int,
    rk0: int,
    k_len: int,
):
    """Read `(m_rows_sub, k_len)` FP8 LHS slice from either 2D VMEM cache or 3D prequant gather buffer."""
    import jax.experimental.pallas as pl

    if prequant_lhs_3d_fp8_ref is not None:
        n_lanes = prequant_lhs_3d_fp8_ref.shape[-1]
        return prequant_lhs_3d_fp8_ref[
            pl.ds(m0, m_rows_sub),
            pl.ds(rk0 // n_lanes, k_len // n_lanes),
            :,
        ].reshape(m_rows_sub, k_len)
    return prequant_lhs_fp8_ref[pl.ds(m0, m_rows_sub), pl.ds(rk0, k_len)]


def unpack_3d_pq_lhs_to_2d_fp8(
    lhs_3d_fp8_ref: Any,
    dst_fp8_ref: Any,
    *,
    tile_m: int,
    tile_k: int,
    via_bf16_xpose: bool = False,
):
    """Unpack 3D `(M, K//128, 128) fp8` gather buffer -> 2D `(M, K) fp8` VMEM cache once per tile.

    Slices full `(32, 32, 128)` FP8 sublane tiles (or `(32, 16, 128)` BF16 XLU tiles when
    `via_bf16_xpose=True`) so TPU7x uses hardware XLU transpose instead of emitting
    `106k` VALU `vrot.slane` instructions inside the strip loop.
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    if via_bf16_xpose and tile_m % 32 == 0 and tile_k % 2048 == 0 and tile_k >= 2048:
        n_kh = tile_k // 2048
        for m0 in range(0, tile_m, 32):
            for kh in range(n_kh):
                dst_fp8_ref[pl.ds(m0, 32), pl.ds(kh * 2048, 2048)] = (
                    lhs_3d_fp8_ref[pl.ds(m0, 32), pl.ds(kh * 16, 16), :]
                    .astype(jnp.bfloat16)
                    .reshape(32, 2048)
                    .astype(jnp.float8_e4m3fn)
                )
        return

    if tile_m % 32 == 0 and tile_k % 4096 == 0 and tile_k >= 4096:
        n_kh = tile_k // 4096
        for m0 in range(0, tile_m, 32):
            for kh in range(n_kh):
                dst_fp8_ref[pl.ds(m0, 32), pl.ds(kh * 4096, 4096)] = lhs_3d_fp8_ref[
                    pl.ds(m0, 32), pl.ds(kh * 32, 32), :
                ].reshape(32, 4096)
        return

    m_ch = 32 if (tile_m >= 32 and tile_m % 32 == 0) else tile_m
    n_lanes = lhs_3d_fp8_ref.shape[-1]
    for m0 in range(0, tile_m, m_ch):
        dst_fp8_ref[pl.ds(m0, m_ch), pl.ds(0, tile_k)] = lhs_3d_fp8_ref[
            pl.ds(m0, m_ch), pl.ds(0, tile_k // n_lanes), :
        ].reshape(m_ch, tile_k)


def fill_pq_lhs_scales_tree(
    dst_scale_ref: Any,
    lhs_row_scales_ref: Any,
    lhs_indices_ref: Any,
    *,
    m_st: Any,
    m_en: Any,
    num_valid: Any,
    num_lanes: int,
    packed_u16_lhs: bool,
    gather_divisor: int,
    scale_b_id: int = 0,
):
    """Populate `dst_scale_ref[scale_b_id, 0:num_valid, :]` using a 3-level binary `vselect` tree."""
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax import lax

    max_r = jnp.maximum(m_en - 1, m_st)
    row_q = lax.broadcasted_iota(jnp.int32, (8, num_lanes), 0)
    b0 = (row_q & jnp.int32(1)) != 0
    b1 = (row_q & jnp.int32(2)) != 0
    b2 = (row_q & jnp.int32(4)) != 0

    def _load_sc(rk):
        if packed_u16_lhs:
            w_u = lhs_indices_ref[lax.shift_right_logical(rk, jnp.int32(1))]
            sh_u = lax.shift_left(rk & jnp.int32(1), jnp.int32(4))
            tok_k = lax.shift_right_logical(w_u, sh_u) & jnp.int32(0xFFFF)
        else:
            tok_k = lhs_indices_ref[rk]
            if gather_divisor != 1:
                tok_k = tok_k // gather_divisor
        return lhs_row_scales_ref[tok_k].astype(jnp.float32)

    def _pack_8(i0):
        s = [_load_sc(jnp.minimum(m_st + i0 + k, max_r)) for k in range(8)]
        p01 = jnp.where(b0, s[1], s[0])
        p23 = jnp.where(b0, s[3], s[2])
        p45 = jnp.where(b0, s[5], s[4])
        p67 = jnp.where(b0, s[7], s[6])
        q03 = jnp.where(b1, p23, p01)
        q47 = jnp.where(b1, p67, p45)
        dst_scale_ref[scale_b_id, pl.ds(i0, 8), :] = jnp.where(b2, q47, q03)

    tile_m_cap = dst_scale_ref.shape[1]
    if tile_m_cap >= 32 and tile_m_cap % 32 == 0:

        def _fill_32(q32, _):
            base_i = q32 * 32
            for sub in range(4):
                _pack_8(base_i + sub * 8)
            return _

        lax.fori_loop(0, (num_valid + 31) >> 5, _fill_32, jnp.int32(0))
    else:

        def _fill_8(q8, _):
            _pack_8(q8 * 8)
            return _

        lax.fori_loop(0, (num_valid + 7) >> 3, _fill_8, jnp.int32(0))


def scoped_two_level_pow2_strip_acc(
    *,
    w_weight_ref: Any,
    w_scale_ref: Any,
    rhs_fp8_strip_ref: Any,
    start_n: int,
    end_n: int,
    tile_m: int,
    tile_k: int,
    q_block_size: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
    lhs_quant_dtype: Any,
    acc_dtype: Any,
    two_level_mode: int,
    two_level_k_step: int,
    quant_otf: bool,
    otf_inv_scale_bf16: Any,
    lhs_2d_vmem_ref: Any,
    prequant_lhs_fp8_ref: Any,
    prequant_lhs_3d_fp8_ref: Any,
    has_prequant_lhs: bool,
    hoisted_lq_blocks: list[Any],
    load_lhs_scale_strip_fn: Any,
    mh_outer: bool = False,
):
    """Zero-spill (`<= 58` VREGs) Scoped VMEM Two-Level Power-of-2 W4A8 strip matmul.

    Eliminates the 8 MiB SSA weight load and all 16 inner-K VPU accumulator rescales /
    `f32 -> bf16` casts while preserving exact `0.0`-ULP `e2m1 * 2^p -> float8_e4m3fn`
    exponent shifting (`17.50 dB` MSE-calibrated SNR):
      - `two_level_k_step >= tile_k` (e.g. `4096`) with `rhs_fp8_strip_ref`: unpacks
        `FP4 * 2^p -> rhs_fp8_strip_ref[tile_k, col_size]` strictly ONCE per strip
        (`<= 36` VREGs, `32`-row aligned store), then accumulates `k_sub = 256` chunks
        in `f32` (`56` VREGs, `2048` mode uses `k_sub = 512` `8`-pop MXU).
      - `two_level_k_step == 512`: pure in-register `8`-pop `K = 512` MXU matmul (`0`
        VMEM strip round-trip, `<= 64` VREGs).
      - `two_level_k_step == 1024` (default): pure in-register decoupled VPU||MXU
        `k_sub = 256` `f32` accumulation (`0` VMEM strip round-trip, `<= 58` VREGs).
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    col_size = end_n - start_n
    num_blocks = tile_k // rhs_qbs
    scale_is_2d = len(w_scale_ref.shape) == 2
    m_step = 128 if (tile_m >= 256 and tile_m % 128 == 0) else tile_m
    num_m_halves = tile_m // m_step

    if scale_is_2d:
        s_f32 = w_scale_ref[:, pl.ds(start_n, col_size)].astype(jnp.float32)
    else:
        s_f32 = (
            w_scale_ref[:, :, pl.ds(start_n, col_size)]
            .reshape(num_blocks, col_size)
            .astype(jnp.float32)
        )
    s_max = jnp.max(s_f32, axis=0, keepdims=True)
    s_safe = jnp.where(s_max > 0.0, s_max, jnp.float32(1.0))
    if two_level_mode >= 2:
        log_r = jnp.clip(
            jnp.rint(jnp.log2(jnp.maximum(s_f32 / s_safe, jnp.float32(2.0**-11)))),
            -11.0,
            0.0,
        )
        pow2_bf16 = jnp.where(
            s_f32 > 0.0, jnp.exp2(log_r + jnp.float32(6.0)), jnp.float32(0.0)
        ).astype(jnp.bfloat16)
    else:
        pow2_bf16 = ((s_f32 / s_safe) * jnp.float32(64.0)).astype(jnp.bfloat16)
    col_sc_f32 = s_max * jnp.float32(1.0 / 64.0)

    if quant_otf:
        for qk0 in range(0, tile_k, 256):
            qk_len = min(256, tile_k - qk0)
            prequant_lhs_fp8_ref[pl.ds(0, tile_m), pl.ds(qk0, qk_len)] = (
                lhs_2d_vmem_ref[pl.ds(0, tile_m), pl.ds(qk0, qk_len)].astype(
                    jnp.bfloat16
                )
                * otf_inv_scale_bf16
            ).astype(lhs_quant_dtype)

    def _unpack_256_pow2_fp8(rk0: int) -> Any:
        r_bid = rk0 // rhs_qbs
        s_row = pow2_bf16[r_bid : r_bid + 1, :]
        parts_128 = []
        for sub_k in (0, 128):
            raw_h = w_weight_ref[
                pl.ds((rk0 + sub_k) // packing, 128 // packing),
                pl.ds(start_n, col_size),
            ]
            w_h = (
                pltpu.bitcast(raw_h, rhs_quant_dtype) if packing > 1 else raw_h
            ).astype(jnp.bfloat16)
            parts_128.append((w_h * s_row).astype(jnp.float8_e4m3fn))
        return jnp.concatenate(parts_128, axis=0)

    use_vmem_strip = (
        rhs_fp8_strip_ref is not None
        and two_level_k_step >= 2048
        and col_size <= rhs_fp8_strip_ref.shape[1]
    )
    if use_vmem_strip:
        for rk0 in range(0, tile_k, 256):
            rhs_fp8_strip_ref[pl.ds(rk0, 256), pl.ds(0, col_size)] = (
                _unpack_256_pow2_fp8(rk0)
            )

    if not mh_outer and not use_vmem_strip and num_m_halves > 1:
        acc_halves_reuse: list[Any] = [None] * num_m_halves
        use_k512_reuse = two_level_k_step == 512 and (tile_k % 512 == 0)
        k_step_r = 512 if use_k512_reuse else 256
        for l_k0 in range(0, tile_k, q_block_size):
            l_k1 = min(tile_k, l_k0 + q_block_size)
            l_bid = l_k0 // q_block_size
            sub_halves_r: list[Any] = [None] * num_m_halves
            for rk0 in range(l_k0, l_k1, k_step_r):
                w_blk = (
                    jnp.concatenate(
                        [_unpack_256_pow2_fp8(rk0), _unpack_256_pow2_fp8(rk0 + 256)],
                        axis=0,
                    )
                    if use_k512_reuse
                    else _unpack_256_pow2_fp8(rk0)
                )
                for mh in range(num_m_halves):
                    m0 = mh * m_step
                    m1 = m0 + m_step
                    x_sub = (
                        _slice_pq_lhs_sub(
                            prequant_lhs_fp8_ref,
                            prequant_lhs_3d_fp8_ref,
                            m0,
                            m_step,
                            rk0,
                            k_step_r,
                        )
                        if (has_prequant_lhs or quant_otf)
                        else hoisted_lq_blocks[l_bid][
                            m0:m1, rk0 - l_k0 : rk0 + k_step_r - l_k0
                        ]
                    )
                    p = jnp.matmul(
                        x_sub, w_blk, preferred_element_type=jnp.float32
                    ).astype(jnp.bfloat16)
                    sub_halves_r[mh] = (
                        p if sub_halves_r[mh] is None else sub_halves_r[mh] + p
                    )
            lhs_s_full = load_lhs_scale_strip_fn(l_bid, col_size)
            for mh in range(num_m_halves):
                m0 = mh * m_step
                m1 = m0 + m_step
                term = sub_halves_r[mh].astype(acc_dtype) * (
                    lhs_s_full[m0:m1, :] * col_sc_f32.astype(acc_dtype)
                )
                acc_halves_reuse[mh] = (
                    term
                    if acc_halves_reuse[mh] is None
                    else acc_halves_reuse[mh] + term
                )
        return jnp.concatenate(acc_halves_reuse, axis=0)

    acc_halves: list[Any] = []
    for mh in range(num_m_halves):
        m0 = mh * m_step
        m1 = m0 + m_step
        acc_h = None
        for l_k0 in range(0, tile_k, q_block_size):
            l_k1 = min(tile_k, l_k0 + q_block_size)
            l_bid = l_k0 // q_block_size
            if use_vmem_strip and two_level_k_step == 2048 and (l_k1 - l_k0) % 512 == 0:
                sub_bf16 = None
                for rk0 in range(l_k0, l_k1, 512):
                    x_sub = (
                        _slice_pq_lhs_sub(
                            prequant_lhs_fp8_ref,
                            prequant_lhs_3d_fp8_ref,
                            m0,
                            m_step,
                            rk0,
                            512,
                        )
                        if (has_prequant_lhs or quant_otf)
                        else hoisted_lq_blocks[l_bid][
                            m0:m1, rk0 - l_k0 : rk0 + 512 - l_k0
                        ]
                    )
                    w_sub = rhs_fp8_strip_ref[pl.ds(rk0, 512), pl.ds(0, col_size)]
                    p = jnp.matmul(
                        x_sub, w_sub, preferred_element_type=jnp.float32
                    ).astype(jnp.bfloat16)
                    sub_bf16 = p if sub_bf16 is None else sub_bf16 + p
                sub_f32 = sub_bf16.astype(jnp.float32)
            elif use_vmem_strip:
                sub_f32 = None
                for rk0 in range(l_k0, l_k1, 256):
                    x_sub = (
                        _slice_pq_lhs_sub(
                            prequant_lhs_fp8_ref,
                            prequant_lhs_3d_fp8_ref,
                            m0,
                            m_step,
                            rk0,
                            256,
                        )
                        if (has_prequant_lhs or quant_otf)
                        else hoisted_lq_blocks[l_bid][
                            m0:m1, rk0 - l_k0 : rk0 + 256 - l_k0
                        ]
                    )
                    w_sub = rhs_fp8_strip_ref[pl.ds(rk0, 256), pl.ds(0, col_size)]
                    p = jnp.matmul(x_sub, w_sub, preferred_element_type=jnp.float32)
                    sub_f32 = p if sub_f32 is None else sub_f32 + p
            elif two_level_k_step == 512 and (l_k1 - l_k0) % 512 == 0:
                sub_bf16 = None
                for rk0 in range(l_k0, l_k1, 512):
                    w_512 = jnp.concatenate(
                        [_unpack_256_pow2_fp8(rk0), _unpack_256_pow2_fp8(rk0 + 256)],
                        axis=0,
                    )
                    x_512 = (
                        _slice_pq_lhs_sub(
                            prequant_lhs_fp8_ref,
                            prequant_lhs_3d_fp8_ref,
                            m0,
                            m_step,
                            rk0,
                            512,
                        )
                        if (has_prequant_lhs or quant_otf)
                        else hoisted_lq_blocks[l_bid][
                            m0:m1, rk0 - l_k0 : rk0 + 512 - l_k0
                        ]
                    )
                    p = jnp.matmul(
                        x_512, w_512, preferred_element_type=jnp.float32
                    ).astype(jnp.bfloat16)
                    sub_bf16 = p if sub_bf16 is None else sub_bf16 + p
                sub_f32 = sub_bf16.astype(jnp.float32)
            else:
                sub_f32 = None
                for rk0 in range(l_k0, l_k1, 256):
                    w_256 = _unpack_256_pow2_fp8(rk0)
                    x_256 = (
                        _slice_pq_lhs_sub(
                            prequant_lhs_fp8_ref,
                            prequant_lhs_3d_fp8_ref,
                            m0,
                            m_step,
                            rk0,
                            256,
                        )
                        if (has_prequant_lhs or quant_otf)
                        else hoisted_lq_blocks[l_bid][
                            m0:m1, rk0 - l_k0 : rk0 + 256 - l_k0
                        ]
                    )
                    p = jnp.matmul(x_256, w_256, preferred_element_type=jnp.float32)
                    sub_f32 = p if sub_f32 is None else sub_f32 + p

            lhs_s_full = load_lhs_scale_strip_fn(l_bid, col_size)
            term = sub_f32.astype(acc_dtype) * (
                lhs_s_full[m0:m1, :] * col_sc_f32.astype(acc_dtype)
            )
            acc_h = term if acc_h is None else acc_h + term
        acc_halves.append(acc_h)

    return acc_halves[0] if num_m_halves == 1 else jnp.concatenate(acc_halves, axis=0)


def scoped_w4a8_strip_acc(
    *,
    w_weight_ref: Any,
    w_scale_ref: Any,
    start_n: int,
    end_n: int,
    tile_m: int,
    tile_k: int,
    q_block_size: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
    lhs_quant_dtype: Any,
    acc_dtype: Any,
    use_bf16_inner: bool,
    quant_otf: bool,
    otf_inv_scale_bf16: Any,
    lhs_2d_vmem_ref: Any,
    prequant_lhs_fp8_ref: Any,
    has_prequant_lhs: bool,
    hoisted_lq_blocks: list[Any],
    load_lhs_scale_strip_fn: Any,
    zero_spill_sub_m: bool = False,
    mh_outer: bool = False,
    prequant_lhs_3d_fp8_ref: Any = None,
    col_size: int | None = None,
    hoist_rhs_scale: bool = False,
    w_fp8_cache_ref: Any = None,
    prequant_lhs_scale_ref: Any = None,
    pair_k512_unpack: bool = False,
):
    """Scoped VMEM W4A8 strip matmul (`GMM_SCOPED_W4A8` modes 1, 2, 3).

    Slices `[rk0 // packing : rk1 // packing, start_n : end_n]` directly from VMEM
    instead of loading the whole 8 MiB `[512, 4096] uint32` tile into SSA, casts
    unpacked FP4 explicitly to `float8_e4m3fn`, and when `zero_spill_sub_m=True`
    at `tile_m >= 256`, splits rows into `m_sub = 128` halves (`<= 48` VREGs when
    `mh_outer=True` vs `~80` VREGs with `2x` weight reuse when `mh_outer=False`).
    Also supports `prequant_lhs_3d_fp8_ref` (`GMM_PREQUANT_LHS_FP8=1`) and zero-spill
    `ks_step <= 512` sub-chunking for wider `COMPACT_FP4_QBLOCK` (`512/1024/2048/4096`).
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    eff_col_size = (end_n - start_n) if col_size is None else col_size
    col_size = eff_col_size
    scale_is_2d = len(w_scale_ref.shape) == 2
    use_sub_m = zero_spill_sub_m and tile_m >= 256 and (tile_m % 128 == 0)
    eff_mh_outer = mh_outer or (rhs_qbs > 512 and use_sub_m)
    m_step = (
        64
        if (
            eff_mh_outer
            and use_sub_m
            and 512 < rhs_qbs < q_block_size
            and tile_m % 64 == 0
        )
        else (128 if use_sub_m else tile_m)
    )
    num_m_halves = tile_m // m_step
    num_r_blocks = tile_k // rhs_qbs
    can_hoist_rs = (
        hoist_rhs_scale
        and (tile_k % rhs_qbs == 0)
        and (num_r_blocks >= 8)
        and (num_r_blocks % 8 == 0)
        and (w_scale_ref.shape[0] == num_r_blocks)
    )
    if can_hoist_rs:
        hoisted_rhs_s_bf16 = (
            w_scale_ref[:, pl.ds(start_n, col_size)].astype(jnp.bfloat16)
            if scale_is_2d
            else w_scale_ref[:, :, pl.ds(start_n, col_size)]
            .reshape(num_r_blocks, col_size)
            .astype(jnp.bfloat16)
        )
    else:
        hoisted_rhs_s_bf16 = None

    _use_w_cache = w_fp8_cache_ref is not None and (
        not isinstance(start_n, int) or start_n + col_size <= w_fp8_cache_ref.shape[-1]
    )

    def _load_w_slice_fp8(k0: int, k_len: int):
        if _use_w_cache:
            return w_fp8_cache_ref[pl.ds(k0, k_len), pl.ds(start_n, col_size)]
        raw_w = w_weight_ref[
            pl.ds(k0 // packing, k_len // packing),
            pl.ds(start_n, col_size),
        ]
        w_fp4 = pltpu.bitcast(raw_w, rhs_quant_dtype) if packing > 1 else raw_w
        return w_fp4.astype(jnp.float8_e4m3fn)

    def _load_rhs_scale(r_bid: int):
        if hoisted_rhs_s_bf16 is not None:
            return hoisted_rhs_s_bf16[r_bid : r_bid + 1, :]
        if scale_is_2d:
            return w_scale_ref[pl.ds(r_bid, 1), pl.ds(start_n, col_size)].astype(
                jnp.bfloat16
            )
        return (
            w_scale_ref[pl.ds(r_bid, 1), :, pl.ds(start_n, col_size)]
            .reshape(1, col_size)
            .astype(jnp.bfloat16)
        )

    if eff_mh_outer and use_sub_m:
        if quant_otf:
            for qk0 in range(0, tile_k, 256):
                qk_len = min(256, tile_k - qk0)
                prequant_lhs_fp8_ref[pl.ds(0, tile_m), pl.ds(qk0, qk_len)] = (
                    lhs_2d_vmem_ref[pl.ds(0, tile_m), pl.ds(qk0, qk_len)].astype(
                        jnp.bfloat16
                    )
                    * otf_inv_scale_bf16
                ).astype(lhs_quant_dtype)
        acc_halves: list[Any] = []
        for mh in range(num_m_halves):
            m0 = mh * m_step
            m1 = m0 + m_step
            acc_h = None
            for l_k0 in range(0, tile_k, q_block_size):
                l_k1 = min(tile_k, l_k0 + q_block_size)
                l_bid = l_k0 // q_block_size
                sub_h = None
                for rk0 in range(l_k0, l_k1, rhs_qbs):
                    rk1 = min(l_k1, rk0 + rhs_qbs)
                    r_bid = rk0 // rhs_qbs
                    k_len = rk1 - rk0
                    rhs_s_bf16 = _load_rhs_scale(r_bid)
                    ks_step = 256 if (k_len > 512 and k_len % 256 == 0) else k_len
                    pop = None
                    for ks0 in range(rk0, rk1, ks_step):
                        w_fp8 = _load_w_slice_fp8(ks0, ks_step)
                        if has_prequant_lhs or quant_otf:
                            lhs_sub = _slice_pq_lhs_sub(
                                prequant_lhs_fp8_ref,
                                prequant_lhs_3d_fp8_ref,
                                m0,
                                m_step,
                                ks0,
                                ks_step,
                            )
                        else:
                            lhs_sub = hoisted_lq_blocks[l_bid][
                                m0:m1, ks0 - l_k0 : ks0 + ks_step - l_k0
                            ]
                        p_k = jnp.matmul(
                            lhs_sub, w_fp8, preferred_element_type=jnp.float32
                        )
                        pop = p_k if pop is None else pop + p_k
                    if use_bf16_inner:
                        sp = pop.astype(jnp.bfloat16) * rhs_s_bf16
                    else:
                        sp = pop.astype(acc_dtype) * rhs_s_bf16.astype(acc_dtype)
                    sub_h = sp if sub_h is None else sub_h + sp
                lhs_s_full = load_lhs_scale_strip_fn(l_bid, col_size)
                term = sub_h.astype(acc_dtype) * lhs_s_full[m0:m1, :]
                acc_h = term if acc_h is None else acc_h + term
            acc_halves.append(acc_h)
        return (
            acc_halves[0] if num_m_halves == 1 else jnp.concatenate(acc_halves, axis=0)
        )

    if (
        pair_k512_unpack
        and rhs_qbs == 256
        and packing == 8
        and not eff_mh_outer
        and (has_prequant_lhs or quant_otf)
        and not _use_w_cache
        and (tile_k % 512 == 0)
        and (q_block_size % 512 == 0)
    ):
        acc_halves = [None] * num_m_halves
        for l_k0 in range(0, tile_k, q_block_size):
            l_k1 = min(tile_k, l_k0 + q_block_size)
            l_bid = l_k0 // q_block_size
            sub_halves_p: list[Any] = [None] * num_m_halves
            for ok0 in range(l_k0, l_k1, 512):
                w_512 = _load_w_slice_fp8(ok0, 512)
                for ib in range(2):
                    rk0 = ok0 + ib * 256
                    rk1 = rk0 + 256
                    r_bid = rk0 // 256
                    w_256 = w_512[ib * 256 : (ib + 1) * 256, :]
                    rhs_s_bf16 = _load_rhs_scale(r_bid)
                    if quant_otf:
                        lhs_full_blk = (
                            lhs_2d_vmem_ref[:, rk0:rk1].astype(jnp.bfloat16)
                            * otf_inv_scale_bf16
                        ).astype(lhs_quant_dtype)
                        prequant_lhs_fp8_ref[:, rk0:rk1] = lhs_full_blk
                    else:
                        lhs_full_blk = None
                    for mh in range(num_m_halves):
                        m0 = mh * m_step
                        m1 = m0 + m_step
                        if lhs_full_blk is not None:
                            lhs_sub = lhs_full_blk[m0:m1, :]
                        else:
                            lhs_sub = _slice_pq_lhs_sub(
                                prequant_lhs_fp8_ref,
                                prequant_lhs_3d_fp8_ref,
                                m0,
                                m_step,
                                rk0,
                                256,
                            )
                        pop = jnp.matmul(
                            lhs_sub,
                            w_256,
                            preferred_element_type=jnp.float32,
                        )
                        if use_bf16_inner:
                            sp = pop.astype(jnp.bfloat16) * rhs_s_bf16
                        else:
                            sp = pop.astype(acc_dtype) * rhs_s_bf16.astype(acc_dtype)
                        sub_halves_p[mh] = (
                            sp if sub_halves_p[mh] is None else sub_halves_p[mh] + sp
                        )
            lhs_s_full = load_lhs_scale_strip_fn(l_bid, col_size)
            for mh in range(num_m_halves):
                m0 = mh * m_step
                m1 = m0 + m_step
                term = sub_halves_p[mh].astype(acc_dtype) * lhs_s_full[m0:m1, :]
                acc_halves[mh] = (
                    term if acc_halves[mh] is None else acc_halves[mh] + term
                )
        return (
            acc_halves[0] if num_m_halves == 1 else jnp.concatenate(acc_halves, axis=0)
        )

    acc_halves = [None] * num_m_halves
    for l_k0 in range(0, tile_k, q_block_size):
        l_k1 = min(tile_k, l_k0 + q_block_size)
        l_bid = l_k0 // q_block_size
        sub_halves: list[Any] = [None] * num_m_halves

        for rk0 in range(l_k0, l_k1, rhs_qbs):
            rk1 = min(l_k1, rk0 + rhs_qbs)
            r_bid = rk0 // rhs_qbs
            k_len = rk1 - rk0

            w_fp8 = _load_w_slice_fp8(rk0, k_len)
            rhs_s_bf16 = _load_rhs_scale(r_bid)

            if quant_otf:
                lhs_full_blk = (
                    lhs_2d_vmem_ref[:, rk0:rk1].astype(jnp.bfloat16)
                    * otf_inv_scale_bf16
                ).astype(lhs_quant_dtype)
                prequant_lhs_fp8_ref[:, rk0:rk1] = lhs_full_blk
            else:
                lhs_full_blk = None

            for mh in range(num_m_halves):
                m0 = mh * m_step
                m1 = m0 + m_step
                if lhs_full_blk is not None:
                    lhs_sub = lhs_full_blk[m0:m1, :]
                elif has_prequant_lhs:
                    lhs_sub = _slice_pq_lhs_sub(
                        prequant_lhs_fp8_ref,
                        prequant_lhs_3d_fp8_ref,
                        m0,
                        m_step,
                        rk0,
                        k_len,
                    )
                else:
                    lhs_sub = hoisted_lq_blocks[l_bid][m0:m1, rk0 - l_k0 : rk1 - l_k0]

                pop = jnp.matmul(
                    lhs_sub,
                    w_fp8,
                    preferred_element_type=jnp.float32,
                )
                if use_bf16_inner:
                    sp = pop.astype(jnp.bfloat16) * rhs_s_bf16
                else:
                    sp = pop.astype(acc_dtype) * rhs_s_bf16.astype(acc_dtype)
                sub_halves[mh] = sp if sub_halves[mh] is None else sub_halves[mh] + sp

        lhs_s_full = load_lhs_scale_strip_fn(l_bid, col_size)
        for mh in range(num_m_halves):
            m0 = mh * m_step
            m1 = m0 + m_step
            term = sub_halves[mh].astype(acc_dtype) * lhs_s_full[m0:m1, :]
            acc_halves[mh] = term if acc_halves[mh] is None else acc_halves[mh] + term

    return acc_halves[0] if num_m_halves == 1 else jnp.concatenate(acc_halves, axis=0)


def scoped_w4a8_dual_strip_acc(
    *,
    w_weight_ref: Any,
    w_scale_ref: Any,
    g_s: int,
    g_e: int,
    u_s: int,
    u_e: int,
    tile_m: int,
    tile_k: int,
    q_block_size: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
    acc_dtype: Any,
    use_bf16_inner: bool,
    prequant_lhs_fp8_ref: Any,
    has_prequant_lhs: bool,
    hoisted_lq_blocks: list[Any],
    load_lhs_scale_strip_fn: Any,
    zero_spill_sub_m: bool = False,
    mh_outer: bool = False,
    hoist_rhs_scale: bool = False,
    prequant_lhs_3d_fp8_ref: Any = None,
):
    """Scoped VMEM dual gate+up strip matmul (`GMM_SCOPED_W4A8` modes 1..5)."""
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    if rhs_qbs >= 512 and (hoist_rhs_scale or (zero_spill_sub_m and tile_m >= 256)):
        _dual_mh_outer = mh_outer or (rhs_qbs > 512)
        acc_g = scoped_w4a8_strip_acc(
            w_weight_ref=w_weight_ref,
            w_scale_ref=w_scale_ref,
            start_n=g_s,
            end_n=g_e,
            tile_m=tile_m,
            tile_k=tile_k,
            q_block_size=q_block_size,
            rhs_qbs=rhs_qbs,
            packing=packing,
            rhs_quant_dtype=rhs_quant_dtype,
            lhs_quant_dtype=jnp.float8_e4m3fn,
            acc_dtype=acc_dtype,
            use_bf16_inner=use_bf16_inner,
            quant_otf=False,
            otf_inv_scale_bf16=None,
            lhs_2d_vmem_ref=None,
            prequant_lhs_fp8_ref=prequant_lhs_fp8_ref,
            prequant_lhs_3d_fp8_ref=prequant_lhs_3d_fp8_ref,
            has_prequant_lhs=has_prequant_lhs,
            hoisted_lq_blocks=hoisted_lq_blocks,
            load_lhs_scale_strip_fn=load_lhs_scale_strip_fn,
            zero_spill_sub_m=zero_spill_sub_m,
            mh_outer=_dual_mh_outer,
            hoist_rhs_scale=hoist_rhs_scale,
        )
        acc_u = scoped_w4a8_strip_acc(
            w_weight_ref=w_weight_ref,
            w_scale_ref=w_scale_ref,
            start_n=u_s,
            end_n=u_e,
            tile_m=tile_m,
            tile_k=tile_k,
            q_block_size=q_block_size,
            rhs_qbs=rhs_qbs,
            packing=packing,
            rhs_quant_dtype=rhs_quant_dtype,
            lhs_quant_dtype=jnp.float8_e4m3fn,
            acc_dtype=acc_dtype,
            use_bf16_inner=use_bf16_inner,
            quant_otf=False,
            otf_inv_scale_bf16=None,
            lhs_2d_vmem_ref=None,
            prequant_lhs_fp8_ref=prequant_lhs_fp8_ref,
            prequant_lhs_3d_fp8_ref=prequant_lhs_3d_fp8_ref,
            has_prequant_lhs=has_prequant_lhs,
            hoisted_lq_blocks=hoisted_lq_blocks,
            load_lhs_scale_strip_fn=load_lhs_scale_strip_fn,
            zero_spill_sub_m=zero_spill_sub_m,
            mh_outer=_dual_mh_outer,
            hoist_rhs_scale=hoist_rhs_scale,
        )
        return acc_g, acc_u

    col_size = g_e - g_s
    scale_is_2d = len(w_scale_ref.shape) == 2
    use_sub_m = zero_spill_sub_m and tile_m >= 256 and (tile_m % 128 == 0)
    m_step = 128 if use_sub_m else tile_m
    num_m_halves = tile_m // m_step

    def _load_dual_w_and_scales(rk0: int, k_len: int, r_bid: int):
        raw_wg = w_weight_ref[
            pl.ds(rk0 // packing, k_len // packing), pl.ds(g_s, col_size)
        ]
        raw_wu = w_weight_ref[
            pl.ds(rk0 // packing, k_len // packing), pl.ds(u_s, col_size)
        ]
        wg_fp8 = (
            pltpu.bitcast(raw_wg, rhs_quant_dtype) if packing > 1 else raw_wg
        ).astype(jnp.float8_e4m3fn)
        wu_fp8 = (
            pltpu.bitcast(raw_wu, rhs_quant_dtype) if packing > 1 else raw_wu
        ).astype(jnp.float8_e4m3fn)

        if scale_is_2d:
            sg_bf16 = w_scale_ref[pl.ds(r_bid, 1), pl.ds(g_s, col_size)].astype(
                jnp.bfloat16
            )
            su_bf16 = w_scale_ref[pl.ds(r_bid, 1), pl.ds(u_s, col_size)].astype(
                jnp.bfloat16
            )
        else:
            sg_bf16 = (
                w_scale_ref[pl.ds(r_bid, 1), :, pl.ds(g_s, col_size)]
                .reshape(1, col_size)
                .astype(jnp.bfloat16)
            )
            su_bf16 = (
                w_scale_ref[pl.ds(r_bid, 1), :, pl.ds(u_s, col_size)]
                .reshape(1, col_size)
                .astype(jnp.bfloat16)
            )
        return wg_fp8, wu_fp8, sg_bf16, su_bf16

    if mh_outer and use_sub_m:
        acc_g_h: list[Any] = []
        acc_u_h: list[Any] = []
        for mh in range(num_m_halves):
            m0 = mh * m_step
            m1 = m0 + m_step
            acc_g = None
            acc_u = None
            for l_k0 in range(0, tile_k, q_block_size):
                l_k1 = min(tile_k, l_k0 + q_block_size)
                l_bid = l_k0 // q_block_size
                sub_g = None
                sub_u = None
                for rk0 in range(l_k0, l_k1, rhs_qbs):
                    rk1 = min(l_k1, rk0 + rhs_qbs)
                    r_bid = rk0 // rhs_qbs
                    k_len = rk1 - rk0
                    wg_fp8, wu_fp8, sg_bf16, su_bf16 = _load_dual_w_and_scales(
                        rk0, k_len, r_bid
                    )
                    if has_prequant_lhs:
                        lhs_sub = _slice_pq_lhs_sub(
                            prequant_lhs_fp8_ref,
                            prequant_lhs_3d_fp8_ref,
                            m0,
                            m_step,
                            rk0,
                            k_len,
                        )
                    else:
                        lhs_sub = hoisted_lq_blocks[l_bid][
                            m0:m1, rk0 - l_k0 : rk1 - l_k0
                        ]
                    pop_g = jnp.matmul(
                        lhs_sub, wg_fp8, preferred_element_type=jnp.float32
                    )
                    pop_u = jnp.matmul(
                        lhs_sub, wu_fp8, preferred_element_type=jnp.float32
                    )
                    if use_bf16_inner:
                        sp_g = pop_g.astype(jnp.bfloat16) * sg_bf16
                        sp_u = pop_u.astype(jnp.bfloat16) * su_bf16
                    else:
                        sp_g = pop_g.astype(acc_dtype) * sg_bf16.astype(acc_dtype)
                        sp_u = pop_u.astype(acc_dtype) * su_bf16.astype(acc_dtype)
                    sub_g = sp_g if sub_g is None else sub_g + sp_g
                    sub_u = sp_u if sub_u is None else sub_u + sp_u
                lhs_s_full = load_lhs_scale_strip_fn(l_bid, col_size)
                tg = sub_g.astype(acc_dtype) * lhs_s_full[m0:m1, :]
                tu = sub_u.astype(acc_dtype) * lhs_s_full[m0:m1, :]
                acc_g = tg if acc_g is None else acc_g + tg
                acc_u = tu if acc_u is None else acc_u + tu
            acc_g_h.append(acc_g)
            acc_u_h.append(acc_u)
        if num_m_halves == 1:
            return acc_g_h[0], acc_u_h[0]
        return jnp.concatenate(acc_g_h, axis=0), jnp.concatenate(acc_u_h, axis=0)

    acc_g_h = [None] * num_m_halves
    acc_u_h = [None] * num_m_halves

    for l_k0 in range(0, tile_k, q_block_size):
        l_k1 = min(tile_k, l_k0 + q_block_size)
        l_bid = l_k0 // q_block_size
        sub_g_h: list[Any] = [None] * num_m_halves
        sub_u_h: list[Any] = [None] * num_m_halves

        for rk0 in range(l_k0, l_k1, rhs_qbs):
            rk1 = min(l_k1, rk0 + rhs_qbs)
            r_bid = rk0 // rhs_qbs
            k_len = rk1 - rk0

            wg_fp8, wu_fp8, sg_bf16, su_bf16 = _load_dual_w_and_scales(
                rk0, k_len, r_bid
            )

            for mh in range(num_m_halves):
                m0 = mh * m_step
                m1 = m0 + m_step
                if has_prequant_lhs:
                    lhs_sub = _slice_pq_lhs_sub(
                        prequant_lhs_fp8_ref,
                        prequant_lhs_3d_fp8_ref,
                        m0,
                        m_step,
                        rk0,
                        k_len,
                    )
                else:
                    lhs_sub = hoisted_lq_blocks[l_bid][m0:m1, rk0 - l_k0 : rk1 - l_k0]

                pop_g = jnp.matmul(lhs_sub, wg_fp8, preferred_element_type=jnp.float32)
                pop_u = jnp.matmul(lhs_sub, wu_fp8, preferred_element_type=jnp.float32)
                if use_bf16_inner:
                    sp_g = pop_g.astype(jnp.bfloat16) * sg_bf16
                    sp_u = pop_u.astype(jnp.bfloat16) * su_bf16
                else:
                    sp_g = pop_g.astype(acc_dtype) * sg_bf16.astype(acc_dtype)
                    sp_u = pop_u.astype(acc_dtype) * su_bf16.astype(acc_dtype)
                sub_g_h[mh] = sp_g if sub_g_h[mh] is None else sub_g_h[mh] + sp_g
                sub_u_h[mh] = sp_u if sub_u_h[mh] is None else sub_u_h[mh] + sp_u

        lhs_s_full = load_lhs_scale_strip_fn(l_bid, col_size)
        for mh in range(num_m_halves):
            m0 = mh * m_step
            m1 = m0 + m_step
            tg = sub_g_h[mh].astype(acc_dtype) * lhs_s_full[m0:m1, :]
            tu = sub_u_h[mh].astype(acc_dtype) * lhs_s_full[m0:m1, :]
            acc_g_h[mh] = tg if acc_g_h[mh] is None else acc_g_h[mh] + tg
            acc_u_h[mh] = tu if acc_u_h[mh] is None else acc_u_h[mh] + tu

    if num_m_halves == 1:
        return acc_g_h[0], acc_u_h[0]
    return jnp.concatenate(acc_g_h, axis=0), jnp.concatenate(acc_u_h, axis=0)


def expand_fp4_strip_to_scaled_fp8(
    w_weight_ref: Any,
    w_scale_ref: Any,
    w_strip_fp8_ref: Any,
    *,
    start_n: int,
    col_size: int,
    tile_k: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
):
    """Expand `w_weight_ref[:, start_n:start_n+col_size]` (`FP4`) + `w_scale_ref`
    into `w_strip_fp8_ref[0:tile_k, 0:col_size]` (`float8_e4m3fn`) and return
    column scale `col_c: float32[1, col_size]`.

    Uses `<= 32` VREGs per 256-row K chunk so zero VMEM spills occur.
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    num_blocks = tile_k // rhs_qbs
    scale_is_2d = len(w_scale_ref.shape) == 2

    if scale_is_2d:
        s_all = w_scale_ref[:, pl.ds(start_n, col_size)].astype(jnp.float32)
    else:
        s_all = (
            w_scale_ref[:, :, pl.ds(start_n, col_size)]
            .reshape(num_blocks, col_size)
            .astype(jnp.float32)
        )

    col_max = jnp.max(jnp.abs(s_all), axis=0, keepdims=True)
    col_c = col_max * jnp.float32(_FP4_MAX / _FP8_E4M3_MAX)
    inv_col_c = jnp.where(col_c == 0.0, jnp.zeros_like(col_c), jnp.float32(1.0) / col_c)
    rel_s_bf16 = (s_all * inv_col_c).astype(jnp.bfloat16)

    k_chunk = 256 if rhs_qbs >= 256 else 64
    groups_per_chunk = k_chunk // rhs_qbs
    for rk0 in range(0, tile_k, k_chunk):
        raw_u32 = w_weight_ref[
            pl.ds(rk0 // packing, k_chunk // packing), pl.ds(start_n, col_size)
        ]
        w_bf16 = (
            pltpu.bitcast(raw_u32, rhs_quant_dtype) if packing > 1 else raw_u32
        ).astype(jnp.bfloat16)
        b_base = rk0 // rhs_qbs
        if groups_per_chunk <= 1:
            w_scaled = w_bf16 * rel_s_bf16[b_base : b_base + 1, :]
        else:
            s_grp = rel_s_bf16[b_base : b_base + groups_per_chunk, :]
            w_scaled = (
                w_bf16.reshape(groups_per_chunk, rhs_qbs, col_size) * s_grp[:, None, :]
            ).reshape(k_chunk, col_size)
        w_strip_fp8_ref[pl.ds(rk0, k_chunk), pl.ds(0, col_size)] = w_scaled.astype(
            jnp.float8_e4m3fn
        )

    return col_c


def dot_lhs_fp8_with_scaled_fp8_strip(
    lhs_fp8_ref: Any,
    lhs_scale_2d_ref: Any,
    w_strip_fp8_ref: Any,
    col_c: Any,
    *,
    m_rows: int,
    tile_k: int,
    col_size: int,
    k_chunk: int = 256,
):
    """Execute zero-spill chunked `FP8 x FP8 -> F32` MXU matmul in `<=128`-row halves.

    Accumulates across `k_chunk`-sized K slices (`<= 56-64` live VREGs vs `384`
    for monolithic `K=4096`), eliminating VREG spills while keeping `0` inner-K
    VPU scale multiplications (`col_c` applied once after `K` reduction).
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    m_step = 128 if (m_rows >= 256 and m_rows % 128 == 0) else m_rows
    eff_k_chunk = k_chunk if (tile_k > k_chunk and tile_k % k_chunk == 0) else tile_k
    halves = []
    for m0 in range(0, m_rows, m_step):
        acc_m = jnp.zeros((m_step, col_size), dtype=jnp.float32)
        for rk0 in range(0, tile_k, eff_k_chunk):
            x_sub = lhs_fp8_ref[pl.ds(m0, m_step), pl.ds(rk0, eff_k_chunk)]
            w_sub = w_strip_fp8_ref[pl.ds(rk0, eff_k_chunk), pl.ds(0, col_size)]
            acc_m = acc_m + jnp.matmul(x_sub, w_sub, preferred_element_type=jnp.float32)
        s128 = lhs_scale_2d_ref[pl.ds(m0, m_step), :].astype(jnp.float32)
        if col_size > s128.shape[1] and col_size % s128.shape[1] == 0:
            lhs_s = jnp.concatenate([s128] * (col_size // s128.shape[1]), axis=1)
        else:
            lhs_s = s128[:, :col_size]
        halves.append(acc_m * (lhs_s * col_c))
    return halves[0] if len(halves) == 1 else jnp.concatenate(halves, axis=0)


def quantize_3d_lhs_tile_to_fp8(
    lhs_3d_ref: Any,
    dst_fp8_ref: Any,
    dst_scale_ref: Any,
    *,
    tile_m: int,
    tile_k: int,
    scale_b_id: int | None = None,
):
    """Zero-spill, sublane-aligned 3D `(M, K//128, 128)` -> 2D `(M, K) fp8` quantizer.

    Uses `32`-row x `<=2048`-col K-half writes when `tile_m % 32 == 0` so TPU7x
    FP8 `(32, 128)` VMEM writes have zero Read-Modify-Write while keeping peak live
    VREGs `<= 48`.
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    def _write_scale(m0: int, m_ch: int, sc_m: Any):
        sc_bcast = jnp.broadcast_to(sc_m, (m_ch, dst_scale_ref.shape[-1]))
        if scale_b_id is not None:
            dst_scale_ref[scale_b_id, pl.ds(m0, m_ch), :] = sc_bcast
        else:
            dst_scale_ref[pl.ds(m0, m_ch), :] = sc_bcast

    if tile_m % 32 == 0 and tile_k % 2048 == 0 and tile_k >= 2048:
        # Sublane-native 1-pass (32, 16, 128) -> (32, 2048) xpose + 2D 128-lane tree-max:
        # cuts TPU7x LLO quantizer bundles 3,278 -> 1,678 (-48.8%), vld 2,048 -> 1,024,
        # and cvt 804 -> 36 (-95.5%) with 0-ULP bit-exact parity.
        n_kh = tile_k // 2048
        for m0 in range(0, tile_m, 32):
            h_list = []
            lmax_128 = None
            for kh in range(n_kh):
                h2d = (
                    lhs_3d_ref[pl.ds(m0, 32), pl.ds(kh * 16, 16), :]
                    .reshape(32, 2048)
                    .astype(jnp.bfloat16)
                )
                h_list.append(h2d)
                for b0 in range(0, 2048, 256):
                    mb = jnp.maximum(
                        jnp.abs(h2d[:, b0 : b0 + 128]),
                        jnp.abs(h2d[:, b0 + 128 : b0 + 256]),
                    )
                    lmax_128 = mb if lmax_128 is None else jnp.maximum(lmax_128, mb)
            amax_m = jnp.max(lmax_128, axis=1, keepdims=True).astype(jnp.float32)
            inv_m = jnp.where(
                amax_m == 0.0,
                jnp.zeros_like(amax_m),
                jnp.float32(_FP8_E4M3_MAX) / amax_m,
            ).astype(jnp.bfloat16)
            sc_m = jnp.where(
                amax_m == 0.0,
                jnp.zeros_like(amax_m),
                amax_m * jnp.float32(1.0 / _FP8_E4M3_MAX),
            )
            for kh in range(n_kh):
                dst_fp8_ref[pl.ds(m0, 32), pl.ds(kh * 2048, 2048)] = (
                    h_list[kh] * inv_m
                ).astype(jnp.float8_e4m3fn)
            _write_scale(m0, 32, sc_m)
        return

    m_ch = (
        32 if (tile_m % 32 == 0 and tile_k <= 2048) else (16 if tile_m % 16 == 0 else 8)
    )
    for m0 in range(0, tile_m, m_ch):
        blk = lhs_3d_ref[pl.ds(m0, m_ch), :, :].astype(jnp.bfloat16)
        amax_m = jnp.max(jnp.max(jnp.abs(blk), axis=1), axis=1, keepdims=True).astype(
            jnp.float32
        )
        inv_m = jnp.where(
            amax_m == 0.0, jnp.zeros_like(amax_m), jnp.float32(_FP8_E4M3_MAX) / amax_m
        ).astype(jnp.bfloat16)
        sc_m = jnp.where(
            amax_m == 0.0,
            jnp.zeros_like(amax_m),
            amax_m * jnp.float32(1.0 / _FP8_E4M3_MAX),
        )
        dst_fp8_ref[pl.ds(m0, m_ch), pl.ds(0, tile_k)] = (
            blk.reshape(m_ch, tile_k) * inv_m
        ).astype(jnp.float8_e4m3fn)
        _write_scale(m0, m_ch, sc_m)


def quantize_2d_lhs_tile_to_fp8(
    lhs_2d_ref: Any,
    dst_fp8_ref: Any,
    dst_scale_ref: Any,
    *,
    tile_m: int,
    tile_k: int,
):
    """Zero-spill 2D `(M, K) bf16` -> `(M, K) float8_e4m3fn` quantizer."""
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    lmax_bf16 = None
    for k0 in range(0, tile_k, 256):
        b2 = lhs_2d_ref[pl.ds(0, tile_m), pl.ds(k0, 256)].astype(jnp.bfloat16)
        m128 = jnp.maximum(jnp.abs(b2[:, :128]), jnp.abs(b2[:, 128:]))
        lmax_bf16 = m128 if lmax_bf16 is None else jnp.maximum(lmax_bf16, m128)
    amax_f32 = jnp.max(lmax_bf16, axis=1, keepdims=True).astype(jnp.float32)
    inv_sc = jnp.where(
        amax_f32 == 0.0, jnp.zeros_like(amax_f32), jnp.float32(_FP8_E4M3_MAX) / amax_f32
    ).astype(jnp.bfloat16)
    sc_f32 = jnp.where(
        amax_f32 == 0.0,
        jnp.zeros_like(amax_f32),
        amax_f32 * jnp.float32(1.0 / _FP8_E4M3_MAX),
    )
    dst_scale_ref[pl.ds(0, tile_m), :] = jnp.broadcast_to(
        sc_f32, (tile_m, dst_scale_ref.shape[-1])
    )
    for k0 in range(0, tile_k, 256):
        b2 = lhs_2d_ref[pl.ds(0, tile_m), pl.ds(k0, 256)].astype(jnp.bfloat16)
        dst_fp8_ref[pl.ds(0, tile_m), pl.ds(k0, 256)] = (b2 * inv_sc).astype(
            jnp.float8_e4m3fn
        )


def compute_pair_gmm1_n_tile_strip_major(
    *,
    lhs_fp8_pair_ref: Any,
    lhs_sc_pair_ref: Any,
    w1_tile_ref: Any,
    w1_scale_tile_ref: Any,
    w_strip_fp8_ref: Any,
    gmm1_pair_ref: Any,
    n_id: int,
    tile_m: int,
    tile_k1: int,
    tile_n1: int,
    strip_step: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
    gs_gate_val: Any,
    gs_up_val: Any,
    out_dtype: Any,
):
    """Strip-major GMM1 over a 2-tile (`2 * tile_m` rows) same-expert pair.

    Expands each `W1_gate` and `W1_up` strip strictly ONCE in VMEM and runs
    1-pop `K=tile_k1` FP8 MXU matmuls across both `gm` tiles (`0` and `1`).
    """
    import jax
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    for s_n in range(0, tile_n1, strip_step):
        col_w = min(strip_step, tile_n1 - s_n)
        out_col = pl.ds(n_id * tile_n1 + s_n, col_w)

        col_c_g = expand_fp4_strip_to_scaled_fp8(
            w1_tile_ref,
            w1_scale_tile_ref,
            w_strip_fp8_ref,
            start_n=s_n,
            col_size=col_w,
            tile_k=tile_k1,
            rhs_qbs=rhs_qbs,
            packing=packing,
            rhs_quant_dtype=rhs_quant_dtype,
        )
        for p_idx in range(2):
            gate_acc = dot_lhs_fp8_with_scaled_fp8_strip(
                lhs_fp8_pair_ref.at[p_idx],
                lhs_sc_pair_ref.at[p_idx],
                w_strip_fp8_ref,
                col_c_g * gs_gate_val,
                m_rows=tile_m,
                tile_k=tile_k1,
                col_size=col_w,
            )
            gmm1_pair_ref[p_idx, :, out_col] = jax.nn.silu(gate_acc).astype(out_dtype)

        col_c_u = expand_fp4_strip_to_scaled_fp8(
            w1_tile_ref,
            w1_scale_tile_ref,
            w_strip_fp8_ref,
            start_n=tile_n1 + s_n,
            col_size=col_w,
            tile_k=tile_k1,
            rhs_qbs=rhs_qbs,
            packing=packing,
            rhs_quant_dtype=rhs_quant_dtype,
        )
        for p_idx in range(2):
            up_acc = dot_lhs_fp8_with_scaled_fp8_strip(
                lhs_fp8_pair_ref.at[p_idx],
                lhs_sc_pair_ref.at[p_idx],
                w_strip_fp8_ref,
                col_c_u * gs_up_val,
                m_rows=tile_m,
                tile_k=tile_k1,
                col_size=col_w,
            )
            gate_bf16 = gmm1_pair_ref[p_idx, :, out_col].astype(jnp.float32)
            gmm1_pair_ref[p_idx, :, out_col] = (gate_bf16 * up_acc).astype(out_dtype)


def compute_pair_gmm2_n_tile_strip_major(
    *,
    lhs_fp8_pair_ref: Any,
    lhs_sc_pair_ref: Any,
    w2_tile_ref: Any,
    w2_scale_tile_ref: Any,
    w_strip_fp8_ref: Any,
    tiled_out_2x_ref: Any,
    sem0: Any,
    sem1: Any,
    n_id: int,
    tile_m: int,
    tile_k2: int,
    tile_n2: int,
    strip_step: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
    gs_w2_val: Any,
    pnw_ref: Any,
    out_dtype: Any,
):
    """Strip-major GMM2 over a 2-tile (`2 * tile_m` rows) same-expert pair.

    Expands each `W2` strip strictly ONCE in VMEM and writes tile 0 to
    `tiled_out_2x_ref[sem0]` and tile 1 to `tiled_out_2x_ref[sem1]`.
    """
    import jax.experimental.pallas as pl

    pnw_full_bf16 = pnw_ref[...].astype(out_dtype) if pnw_ref is not None else None
    n_tile_offset = n_id * tile_n2
    sems = (sem0, sem1)

    for s_n in range(0, tile_n2, strip_step):
        col_w = min(strip_step, tile_n2 - s_n)
        out_col = pl.ds(n_tile_offset + s_n, col_w)

        col_c = expand_fp4_strip_to_scaled_fp8(
            w2_tile_ref,
            w2_scale_tile_ref,
            w_strip_fp8_ref,
            start_n=s_n,
            col_size=col_w,
            tile_k=tile_k2,
            rhs_qbs=rhs_qbs,
            packing=packing,
            rhs_quant_dtype=rhs_quant_dtype,
        )
        pnw_s = (
            pnw_full_bf16[:, n_tile_offset + s_n : n_tile_offset + s_n + col_w]
            if pnw_full_bf16 is not None
            else None
        )
        for p_idx in range(2):
            acc_n = dot_lhs_fp8_with_scaled_fp8_strip(
                lhs_fp8_pair_ref.at[p_idx],
                lhs_sc_pair_ref.at[p_idx],
                w_strip_fp8_ref,
                col_c * gs_w2_val,
                m_rows=tile_m,
                tile_k=tile_k2,
                col_size=col_w,
            )
            out_bf16 = acc_n.astype(out_dtype)
            if pnw_s is not None:
                out_bf16 = (
                    out_bf16.reshape(tile_m // 8, 8, col_w) * pnw_s[None, :, :]
                ).reshape(tile_m, col_w)
            tiled_out_2x_ref[sems[p_idx], :, out_col] = out_bf16


# =============================================================================
# 2-Slot Intra-Chip Chipmate Rebalance (`PREFILL_CHIPMATE_SLOTS = 2`)
# =============================================================================


def plan_prefill_chipmate_2slot_jax(
    topk_indices: Any,
    *,
    ep_size: int,
    num_local_experts: int,
    tile_m: int = 256,
    chipmate_xor: int = 1,
    min_tile_gap: int = 3,
    mode: str = "split",
    weight_load_tiles: int = 1,
    use_pallas: bool | None = None,
    slot_xors: tuple[int, ...] | None = None,
):
    """Deterministic multi-round chipmate planner borrowing up to 2/3 experts (`E_34` / `E_35`).

    Returns:
        v_ids: int32[T, topk] remapped virtual expert IDs over `ep_size * (num_local_experts + n_cm)`.
        cm_plan: int32[ep_size, 3 * n_cm] per-rank `[send_0, recv_0, src_0, ...]`.
    """
    import jax.numpy as jnp
    from jax import lax

    from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
        _base_virtual_ids_arith,
        _can_use_tpu_pallas_prefill_router,
        _jax_bincount_2d_topk,
    )

    mode_norm = (mode or "whole").strip().lower()
    if mode_norm not in ("whole", "split"):
        raise ValueError(
            f"Unsupported prefill chipmate mode={mode!r}; expected 'whole' or 'split'"
        )
    eff_slot_xors = (
        tuple(slot_xors) if slot_xors is not None else (chipmate_xor, chipmate_xor)
    )
    total_elems_in = int(topk_indices.shape[0] * topk_indices.shape[1])
    if _can_use_tpu_pallas_prefill_router(total_elems_in, use_pallas):
        return _pallas_prefill_chipmate_2slot_route(
            topk_indices,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            chipmate_xor=chipmate_xor,
            chipmate_mode=mode_norm,
            min_tile_gap=min_tile_gap,
            tile_m=tile_m,
            weight_load_tiles=weight_load_tiles,
            slot_xors=eff_slot_xors,
        )

    e_loc = num_local_experts
    n_cm = len(eff_slot_xors)
    e_virt = e_loc + n_cm
    num_in_experts = ep_size * e_loc

    ids_i32 = topk_indices.astype(jnp.int32)
    counts_flat = _jax_bincount_2d_topk(ids_i32, num_in_experts)
    counts_rs = counts_flat.reshape(ep_size, e_loc)
    tiles_rs = jnp.where(counts_rs > 0, (counts_rs + tile_m - 1) // tile_m, 0)
    cost_rs = jnp.where(counts_rs > 0, tiles_rs + weight_load_tiles, 0)
    loads = jnp.sum(cost_rs, axis=1)

    base_v = _base_virtual_ids_arith(ids_i32, e_loc, e_virt, ep_size)
    total_elems = ids_i32.shape[0] * ids_i32.shape[1]
    n_blocks = 64 if (total_elems >= 64 and total_elems % 64 == 0) else ep_size
    blk_elems = total_elems // n_blocks
    v_blk = base_v.reshape(n_blocks, blk_elems)

    used_mask = jnp.zeros((ep_size, e_loc), dtype=jnp.bool_)
    send_cols = [jnp.full((ep_size,), -1, dtype=jnp.int32) for _ in range(n_cm)]
    recv_cols = [jnp.zeros((ep_size,), dtype=jnp.int32) for _ in range(n_cm)]
    src_cols = [jnp.zeros((ep_size,), dtype=jnp.int32) for _ in range(n_cm)]

    for rnd, r_xor in enumerate(eff_slot_xors):
        pairs = [
            (r, r ^ r_xor)
            for r in range(ep_size)
            if 0 <= (r ^ r_xor) < ep_size and r < (r ^ r_xor)
        ]
        for r0, r1 in pairs:
            l0, l1 = loads[r0], loads[r1]
            r0_heavier = l0 >= l1
            hi = jnp.where(r0_heavier, jnp.int32(r0), jnp.int32(r1))
            lo = jnp.where(r0_heavier, jnp.int32(r1), jnp.int32(r0))
            l_hi = jnp.maximum(l0, l1)
            l_lo = jnp.minimum(l0, l1)

            t_hi = jnp.where(used_mask[hi], jnp.int32(0), tiles_rs[hi])
            c_hi = jnp.where(t_hi > 0, t_hi + weight_load_tiles, jnp.int32(0))
            new_peaks = jnp.where(c_hi > 0, jnp.maximum(l_hi - c_hi, l_lo + c_hi), l_hi)
            best_s = jnp.argmin(new_peaks).astype(jnp.int32)
            whole_gain = l_hi - new_peaks[best_s]

            if mode_norm == "whole":
                do_any = whole_gain >= min_tile_gap
                use_whole = do_any
                chosen_s = best_s
                split_t = jnp.int32(0)
                t_max = t_hi[best_s]
            else:
                max_s = jnp.argmax(t_hi).astype(jnp.int32)
                t_max = t_hi[max_s]
                split_t = jnp.clip(
                    (l_hi - (l_lo + weight_load_tiles)) // 2,
                    0,
                    jnp.maximum(0, t_max - 1),
                )
                split_peak = jnp.maximum(
                    l_hi - split_t, l_lo + split_t + weight_load_tiles
                )
                split_gain = jnp.where(split_t >= min_tile_gap, l_hi - split_peak, 0)
                use_whole = (whole_gain >= split_gain) & (whole_gain >= min_tile_gap)
                use_split = (~use_whole) & (split_gain >= min_tile_gap)
                do_any = use_whole | use_split
                chosen_s = jnp.where(use_whole, best_s, max_s)

            src_vid = hi * e_virt + chosen_s
            borrowed_vid = lo * e_virt + (e_loc + rnd)
            keep_rows = jnp.where(use_whole, 0, (t_max - split_t) * tile_m)

            hit_blk = v_blk == src_vid
            blk_cnt = jnp.sum(hit_blk.astype(jnp.int32), axis=1)
            cum_blk = jnp.cumsum(blk_cnt) - blk_cnt
            mid_row = cum_blk + lax.shift_right_logical(blk_cnt, 1)
            to_lo_blk = do_any & (mid_row >= keep_rows)
            v_blk = jnp.where(hit_blk & to_lo_blk[:, None], borrowed_vid, v_blk)

            delta_hi = jnp.where(
                do_any, jnp.where(use_whole, c_hi[chosen_s], split_t), 0
            )
            delta_lo = jnp.where(
                do_any,
                jnp.where(use_whole, c_hi[chosen_s], split_t + weight_load_tiles),
                0,
            )
            loads = loads.at[hi].set(l_hi - delta_hi)
            loads = loads.at[lo].set(l_lo + delta_lo)
            new_t_hi = jnp.where(
                do_any,
                jnp.where(
                    use_whole,
                    jnp.int32(0),
                    jnp.maximum(0, tiles_rs[hi, chosen_s] - split_t),
                ),
                tiles_rs[hi, chosen_s],
            )
            tiles_rs = tiles_rs.at[hi, chosen_s].set(new_t_hi)
            used_mask = used_mask.at[hi, chosen_s].set(used_mask[hi, chosen_s] | do_any)

            send_cols[rnd] = (
                send_cols[rnd]
                .at[hi]
                .set(jnp.where(do_any, chosen_s, send_cols[rnd][hi]))
            )
            recv_cols[rnd] = (
                recv_cols[rnd]
                .at[lo]
                .set(jnp.where(do_any, jnp.int32(1), recv_cols[rnd][lo]))
            )
            src_cols[rnd] = (
                src_cols[rnd].at[lo].set(jnp.where(do_any, chosen_s, src_cols[rnd][lo]))
            )

    cols = []
    for rnd in range(n_cm):
        cols.extend([send_cols[rnd], recv_cols[rnd], src_cols[rnd]])
    cm_plan = jnp.stack(cols, axis=1)
    return v_blk.reshape(topk_indices.shape), cm_plan


def _pallas_prefill_chipmate_2slot_route(
    topk_indices: Any,
    *,
    ep_size: int,
    num_local_experts: int,
    chipmate_xor: int = 1,
    chipmate_mode: str = "split",
    min_tile_gap: int = 3,
    tile_m: int = 256,
    weight_load_tiles: int = 1,
    slot_xors: tuple[int, ...] | None = None,
):
    """Single-op TPU7x Pallas VMEM+SMEM multi-slot chipmate router (`~30 us`, 0 sorts)."""
    import jax
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
        _jax_bincount_2d_topk,
    )

    eff_slot_xors = (
        tuple(slot_xors) if slot_xors is not None else (chipmate_xor, chipmate_xor)
    )
    e_in = num_local_experts
    n_cm = len(eff_slot_xors)
    e_final = e_in + n_cm
    num_in_experts = ep_size * e_in
    e_in_pow2 = (e_in & (e_in - 1)) == 0
    e_in_shift = e_in.bit_length() - 1
    e_in_mask = e_in - 1

    ids_i32 = topk_indices.astype(jnp.int32)
    total_elems = int(ids_i32.shape[0] * ids_i32.shape[1])
    n_blocks = 64
    blk_elems = total_elems // n_blocks
    ids_blk = ids_i32.reshape(n_blocks, blk_elems)
    counts_e = _jax_bincount_2d_topk(ids_i32, num_in_experts)

    max_moves = max(1, (ep_size // 2) * n_cm)

    def _kernel(*refs):
        it = iter(refs)
        cnt_ref = next(it)
        ids_ref = next(it)
        v_out_ref = next(it)
        cm_out_ref = next(it)
        loads_s = next(it)
        tiles_s = next(it)
        used_s = next(it)
        n_moves_s = next(it)
        m_src_s = next(it)
        m_dst_s = next(it)
        m_lo_s = next(it)
        m_hi_s = next(it)
        blk_cnt_v = next(it) if chipmate_mode == "split" else None

        n_moves_s[0] = jnp.int32(0)

        @pl.loop(0, ep_size)
        def _init_ranks(r):
            loads_s[r] = jnp.int32(0)
            for _rnd in range(n_cm):
                cm_out_ref[r, 3 * _rnd + 0] = jnp.int32(-1)
                cm_out_ref[r, 3 * _rnd + 1] = jnp.int32(0)
                cm_out_ref[r, 3 * _rnd + 2] = jnp.int32(0)

        @pl.loop(0, num_in_experts)
        def _init_in(e):
            c = cnt_ref[e]
            t = jnp.where(
                c > 0,
                (c + jnp.int32(tile_m - 1)) // jnp.int32(tile_m),
                jnp.int32(0),
            )
            cost = jnp.where(c > 0, t + jnp.int32(weight_load_tiles), jnp.int32(0))
            own = (
                lax.shift_right_logical(e, jnp.int32(e_in_shift))
                if e_in_pow2
                else (e // jnp.int32(e_in))
            )
            tiles_s[e] = t
            used_s[e] = jnp.int32(0)
            loads_s[own] = loads_s[own] + cost

        for rnd, r_xor in enumerate(eff_slot_xors):
            pairs = [
                (r, r ^ r_xor)
                for r in range(ep_size)
                if 0 <= (r ^ r_xor) < ep_size and r < (r ^ r_xor)
            ]
            for r0, r1 in pairs:
                l0 = loads_s[r0]
                l1 = loads_s[r1]
                r0_heavier = l0 >= l1
                hi = jnp.where(r0_heavier, jnp.int32(r0), jnp.int32(r1))
                lo = jnp.where(r0_heavier, jnp.int32(r1), jnp.int32(r0))
                l_hi = jnp.maximum(l0, l1)
                l_lo = jnp.minimum(l0, l1)

                def _scan_slot(s, carry, _hi=hi, _l_hi=l_hi, _l_lo=l_lo):
                    b_s, b_pk, m_s, m_t = carry
                    idx = _hi * jnp.int32(e_in) + s
                    t_s = jnp.where(used_s[idx] != 0, jnp.int32(0), tiles_s[idx])
                    c_s = jnp.where(
                        t_s > 0, t_s + jnp.int32(weight_load_tiles), jnp.int32(0)
                    )
                    pk = jnp.where(
                        c_s > 0, jnp.maximum(_l_hi - c_s, _l_lo + c_s), _l_hi
                    )
                    better = pk < b_pk
                    b_s = jnp.where(better, s, b_s)
                    b_pk = jnp.where(better, pk, b_pk)
                    hotter = t_s > m_t
                    m_s = jnp.where(hotter, s, m_s)
                    m_t = jnp.where(hotter, t_s, m_t)
                    return (b_s, b_pk, m_s, m_t)

                best_s, best_pk, max_s, t_max = lax.fori_loop(
                    0,
                    e_in,
                    _scan_slot,
                    (jnp.int32(0), l_hi, jnp.int32(0), jnp.int32(0)),
                )
                whole_gain = l_hi - best_pk
                if chipmate_mode == "whole":
                    do_any = whole_gain >= jnp.int32(min_tile_gap)
                    use_whole = do_any
                    chosen_s = best_s
                    split_t = jnp.int32(0)
                    keep_rows = jnp.int32(0)
                else:
                    split_raw = (
                        l_hi - (l_lo + jnp.int32(weight_load_tiles))
                    ) // jnp.int32(2)
                    split_t = jnp.minimum(
                        jnp.maximum(jnp.int32(0), split_raw),
                        jnp.maximum(jnp.int32(0), t_max - jnp.int32(1)),
                    )
                    split_pk = jnp.maximum(
                        l_hi - split_t, l_lo + split_t + jnp.int32(weight_load_tiles)
                    )
                    split_gain = jnp.where(
                        split_t >= jnp.int32(min_tile_gap),
                        l_hi - split_pk,
                        jnp.int32(0),
                    )
                    use_whole = (whole_gain >= split_gain) & (
                        whole_gain >= jnp.int32(min_tile_gap)
                    )
                    use_split = (~use_whole) & (split_gain >= jnp.int32(min_tile_gap))
                    do_any = use_whole | use_split
                    chosen_s = jnp.where(use_whole, best_s, max_s)
                    keep_rows = jnp.where(
                        use_whole, jnp.int32(0), (t_max - split_t) * jnp.int32(tile_m)
                    )

                c_chosen_raw = tiles_s[hi * jnp.int32(e_in) + chosen_s]
                c_chosen = jnp.where(
                    c_chosen_raw > 0,
                    c_chosen_raw + jnp.int32(weight_load_tiles),
                    jnp.int32(0),
                )

                @pl.when(do_any)
                def _do_cm_rnd(
                    _rnd=rnd,
                    _hi=hi,
                    _lo=lo,
                    _cs=chosen_s,
                    _kr=keep_rows,
                    _uw=use_whole,
                    _st=split_t,
                    _cc=c_chosen,
                    _traw=c_chosen_raw,
                ):
                    src_vid = _hi * jnp.int32(e_final) + _cs
                    dst_vid = _lo * jnp.int32(e_final) + jnp.int32(e_in + _rnd)
                    mi = n_moves_s[0]
                    m_src_s[mi] = src_vid
                    m_dst_s[mi] = dst_vid
                    m_lo_s[mi] = _kr
                    m_hi_s[mi] = jnp.int32(1 << 30)
                    n_moves_s[0] = mi + jnp.int32(1)
                    col_b = 3 * _rnd
                    cm_out_ref[_hi, col_b + 0] = _cs
                    cm_out_ref[_lo, col_b + 1] = jnp.int32(1)
                    cm_out_ref[_lo, col_b + 2] = _cs
                    dh = jnp.where(_uw, _cc, _st)
                    dl = jnp.where(_uw, _cc, _st + jnp.int32(weight_load_tiles))
                    loads_s[_hi] = loads_s[_hi] - dh
                    loads_s[_lo] = loads_s[_lo] + dl
                    h_idx = _hi * jnp.int32(e_in) + _cs
                    tiles_s[h_idx] = jnp.where(
                        _uw, jnp.int32(0), jnp.maximum(jnp.int32(0), _traw - _st)
                    )
                    used_s[h_idx] = jnp.int32(1)

        @pl.loop(0, n_blocks // 8)
        def _map_base(b8):
            r_sl = pl.ds(b8 * 8, 8)
            for c0 in range(0, blk_elems, 128):
                c_sl = pl.ds(c0, 128)
                raw = ids_ref[r_sl, c_sl]
                if e_in_pow2:
                    own = lax.shift_right_logical(raw, jnp.int32(e_in_shift))
                    sl = raw & jnp.int32(e_in_mask)
                else:
                    own = raw // jnp.int32(e_in)
                    sl = raw % jnp.int32(e_in)
                mapped = own * jnp.int32(e_final) + sl
                v_out_ref[r_sl, c_sl] = jnp.where(
                    raw >= jnp.int32(num_in_experts),
                    jnp.int32(ep_size * e_final),
                    mapped,
                )

        @pl.loop(0, n_moves_s[0])
        def _apply_move(mi):
            src_vid = m_src_s[mi]
            dst_vid = m_dst_s[mi]
            lo_row = m_lo_s[mi]
            hi_row = m_hi_s[mi]
            is_whole = (lo_row == 0) & (hi_row >= jnp.int32(1 << 29))

            @pl.when(is_whole)
            def _apply_whole():
                @pl.loop(0, n_blocks // 8)
                def _w_blk(b8):
                    r_sl = pl.ds(b8 * 8, 8)
                    for c0 in range(0, blk_elems, 128):
                        c_sl = pl.ds(c0, 128)
                        cur = v_out_ref[r_sl, c_sl]
                        v_out_ref[r_sl, c_sl] = jnp.where(cur == src_vid, dst_vid, cur)

            if chipmate_mode == "split":

                @pl.when(~is_whole)
                def _apply_split():
                    @pl.loop(0, n_blocks // 8)
                    def _c_blk(b8):
                        r_sl = pl.ds(b8 * 8, 8)
                        acc = jnp.zeros((8, 128), dtype=jnp.int32)
                        for c0 in range(0, blk_elems, 128):
                            cur = v_out_ref[r_sl, pl.ds(c0, 128)]
                            acc = acc + (cur == src_vid).astype(jnp.int32)
                        rsum = jnp.sum(acc, axis=1, keepdims=True)
                        blk_cnt_v[r_sl, :] = jnp.broadcast_to(rsum, (8, 128))

                    tril_64 = (
                        lax.broadcasted_iota(jnp.int32, (n_blocks, n_blocks), 0)
                        >= lax.broadcasted_iota(jnp.int32, (n_blocks, n_blocks), 1)
                    ).astype(jnp.bfloat16)
                    cnt_all = blk_cnt_v[...]
                    lo8 = (cnt_all & jnp.int32(255)).astype(jnp.bfloat16)
                    hi8 = lax.shift_right_logical(cnt_all, jnp.int32(8)).astype(
                        jnp.bfloat16
                    )
                    cum_lo = jnp.matmul(
                        tril_64, lo8, preferred_element_type=jnp.float32
                    ).astype(jnp.int32)
                    cum_hi = jnp.matmul(
                        tril_64, hi8, preferred_element_type=jnp.float32
                    ).astype(jnp.int32)
                    cum_all = lax.shift_left(cum_hi, jnp.int32(8)) + cum_lo
                    mid_all = (cum_all - cnt_all) + lax.shift_right_logical(
                        cnt_all, jnp.int32(1)
                    )
                    blk_cnt_v[...] = ((mid_all >= lo_row) & (mid_all < hi_row)).astype(
                        jnp.int32
                    )

                    @pl.loop(0, n_blocks // 8)
                    def _s_blk(b8):
                        r_sl = pl.ds(b8 * 8, 8)
                        tk = blk_cnt_v[r_sl, :] != 0
                        for c0 in range(0, blk_elems, 128):
                            c_sl = pl.ds(c0, 128)
                            cur = v_out_ref[r_sl, c_sl]
                            v_out_ref[r_sl, c_sl] = jnp.where(
                                tk & (cur == src_vid), dst_vid, cur
                            )

    scratch_shapes = [
        pltpu.SMEM((ep_size,), jnp.int32),
        pltpu.SMEM((num_in_experts,), jnp.int32),
        pltpu.SMEM((num_in_experts,), jnp.int32),
        pltpu.SMEM((1,), jnp.int32),
        pltpu.SMEM((max_moves,), jnp.int32),
        pltpu.SMEM((max_moves,), jnp.int32),
        pltpu.SMEM((max_moves,), jnp.int32),
        pltpu.SMEM((max_moves,), jnp.int32),
    ]
    if chipmate_mode == "split":
        scratch_shapes.append(pltpu.VMEM((n_blocks, 128), jnp.int32))

    v_blk, cm_plan = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((n_blocks, blk_elems), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, 3 * n_cm), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"pallas_prefill_chipmate_{n_cm}slot_{chipmate_mode}_m{total_elems}",
    )(counts_e, ids_blk)
    return v_blk.reshape(topk_indices.shape), cm_plan
