# SPDX-License-Identifier: Apache-2.0
"""Symmetric EP(P/2)xTP2 Fuse-All Decode MoE Kernel for Muse Spark on TPU7x.

Implements arXiv:2611.17857v1 §4.6 (Algorithm 2) specialized for Muse Spark
NVFP4 decode (E=256, top_k=8, H=4096, I=4096, B in {16, 32, 64, 128, 256})
supporting:
  - EP4xTP2 on 2x2x1 (P=8 TensorCores across 4 physical chips, 64 experts/chip)
  - EP8xTP2 on 2x2x2 (P=16 TensorCores across 8 physical chips, 32 experts/chip)
  - Compact K256 AND Native K16 NVFP4 scales (with multi-pass block-diagonal K16)
  - Optional §4.5b preloaded cross-chip hot-expert replica bank (R in {0,1,2,4})
  - Dual execution modes:
      1. ``row_mode="dense"`` (auto for B <= 32 on K256): zero token-pack/gather,
         canonical 2D [B/2, 4096] vectorized bridge + RMSNorm + weighted FMA,
         and halved ep-way Reduce-Scatter!
      2. ``row_mode="sparse"`` (auto for B >= 64 and K16): speculative slot-0
         weight prime before routing, branchless VPU-filtered token packing,
         2-bucket low-bundle schedule (<28K bundles, zero instruction overlays),
         binary-peeled 4+2+1 vectorized RMSNorm + scatter-add, and spill-free
         4-row chunked Reduce-Scatter skipping self-DMA!
"""

from __future__ import annotations

import functools
import os
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_rs_muse.decode_hbm_breakers import (
    scale_free_expert_for_batch,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
    RoutingScratch,
    _bridge1_prerouted_active,
    _route_prerouted_topk_in_kernel,
    build_default_exec_plan_smem,
    build_exec_plan_from_masks_smem,
    extract_my_holder_candidates_smem,
    extract_my_holder_candidates_tuple,
    pack_active_tokens_smem,
    parse_free_expert_rows_spec,
    route_topk_and_counts_in_kernel,
    routing_scratch_shapes,
    run_replica_planner_smem,
)
from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
    _fp4_blockdiag_gmm,
    _fp4_k256_dual_gmm,
    _fp4_k256_gmm,
)

# Empirically verified on cdk-tpu-exos TPU7x (2x2x1 & 2x2x2):
# core_on_chip in {0, 1} is bit 0 (XOR 1) of global mesh rank:
# pairs (0,1), (2,3), (4,5), (6,7), ... share a physical chip with ~733 GB/s
# bidirectional VMEM->VMEM die-to-die bridge (~8.1x inter-chip ICI).
TPU7X_CHIPMATE_XOR: int = 1


def _resolve_chipmate_xor(chipmate_xor: int | None) -> int:
    xor = TPU7X_CHIPMATE_XOR if chipmate_xor is None else int(chipmate_xor)
    if xor <= 0 or (xor & (xor - 1)) != 0:
        raise ValueError(f"chipmate_xor must be a power of two, got {xor}")
    return xor


def chipmate_rank(rank, *, chipmate_xor: int | None = None):
    """Return the intra-chip partner mesh rank for ``rank`` (Python int or JAX i32)."""
    xor = _resolve_chipmate_xor(chipmate_xor)
    if isinstance(rank, int):
        return rank ^ xor
    return lax.bitwise_xor(jnp.asarray(rank, dtype=jnp.int32), jnp.int32(xor))


def rank_to_chip_and_half(rank, *, chipmate_xor: int | None = None):
    """Map global mesh ``rank`` to ``(chip_id, tp_half)`` with ``tp_half in {0, 1}``."""
    xor = _resolve_chipmate_xor(chipmate_xor)
    bit = xor.bit_length() - 1
    low_mask = xor - 1
    if isinstance(rank, int):
        tp_half = (rank >> bit) & 1
        chip_id = ((rank >> (bit + 1)) << bit) | (rank & low_mask)
        return chip_id, tp_half
    r = jnp.asarray(rank, dtype=jnp.int32)
    tp_half = lax.bitwise_and(lax.shift_right_logical(r, jnp.int32(bit)), jnp.int32(1))
    high = lax.shift_left(lax.shift_right_logical(r, jnp.int32(bit + 1)), jnp.int32(bit))
    low = lax.bitwise_and(r, jnp.int32(low_mask))
    chip_id = lax.bitwise_or(high, low)
    return chip_id, tp_half


def chip_and_half_to_rank(chip_id, tp_half, *, chipmate_xor: int | None = None):
    """Inverse of ``rank_to_chip_and_half``."""
    xor = _resolve_chipmate_xor(chipmate_xor)
    bit = xor.bit_length() - 1
    low_mask = xor - 1
    if isinstance(chip_id, int) and isinstance(tp_half, int):
        return ((chip_id >> bit) << (bit + 1)) | ((tp_half & 1) << bit) | (chip_id & low_mask)
    c = jnp.asarray(chip_id, dtype=jnp.int32)
    h = lax.bitwise_and(jnp.asarray(tp_half, dtype=jnp.int32), jnp.int32(1))
    high = lax.shift_left(lax.shift_right_logical(c, jnp.int32(bit)), jnp.int32(bit + 1))
    mid = lax.shift_left(h, jnp.int32(bit))
    low = lax.bitwise_and(c, jnp.int32(low_mask))
    return lax.bitwise_or(lax.bitwise_or(high, mid), low)


def _route_topk_and_counts_in_kernel(
    logits_ref,
    bias_ref,
    counts_vmem,
    counts_smem,
    slot_ids_vmem,
    slot_ids_smem,
    slot_w_vmem,
    slot_w_smem,
    dense_w_vmem,
    dma_sem,
    *,
    batch_size: int,
    topk: int = 8,
    route_eps: float = 1e-15,
    emit_slots_smem: bool = True,
):
    """Exact serving top-8 routing in VPU (NaN-safe, lower-ID tie-exact).

    Populates:
      - ``counts_smem[0, e]``: exact valid count (w > 0) for expert e
      - ``slot_ids_smem[s, t]`` & ``slot_w_smem[s, t]`` (when emit_slots_smem=True)
      - ``dense_w_vmem[t, e]`` (when dense_w_vmem is not None, for dense row_mode)
    """
    bias_row = bias_ref[0:1, :].astype(jnp.float32)  # (1, 256)
    rb = min(batch_size, 32)
    ids_grid = lax.broadcasted_iota(jnp.int32, (rb, 256), 1)
    counts_acc = jnp.zeros((1, 256), dtype=jnp.int32)

    pad_b = max(128, batch_size)
    num_groups = pad_b // 128
    chunks_per_group = min(batch_size, 128) // rb

    for g in range(num_groups):
        g_ids_chunks = []
        g_w_chunks = []
        for c in range(chunks_per_group):
            b0 = g * 128 + c * rb
            raw_lg = logits_ref[pl.ds(b0, rb), :].astype(jnp.float32)
            # Sanitize NaN -> -inf so best ID is always in 0..255 (never 256 OOB!)
            lg = jnp.where(jnp.isnan(raw_lg), -jnp.inf, raw_lg)
            scores = 1.0 / (1.0 + jnp.exp(-lg))
            choice = scores + bias_row

            bests = []
            sels = []
            picks = []
            for _ in range(topk):
                peak = jnp.max(choice, axis=1, keepdims=True)
                best = jnp.min(
                    jnp.where(choice == peak, ids_grid, 255), axis=1, keepdims=True
                )
                pick = ids_grid == best
                sel = jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
                bests.append(best)
                sels.append(sel)
                picks.append(pick)
                choice = jnp.where(pick, -jnp.inf, choice)

            denom = sels[0]
            for s_idx in range(1, topk):
                denom = denom + sels[s_idx]
            denom = denom + route_eps

            ws = [s / denom for s in sels]
            chunk_active = jnp.zeros((rb, 256), dtype=jnp.int32)
            dense_chunk = jnp.zeros((rb, 256), dtype=jnp.float32) if dense_w_vmem is not None else None
            for s_idx in range(topk):
                pos_w = ws[s_idx] > 0.0
                chunk_active = jnp.where(picks[s_idx] & pos_w, 1, chunk_active)
                if dense_chunk is not None:
                    dense_chunk = jnp.where(picks[s_idx] & pos_w, ws[s_idx], dense_chunk)
            counts_acc = counts_acc + jnp.sum(chunk_active, axis=0, keepdims=True)

            if dense_w_vmem is not None:
                dense_w_vmem[pl.ds(b0, rb), :] = dense_chunk

            if emit_slots_smem:
                g_ids_chunks.append(jnp.concatenate(bests, axis=1))
                g_w_chunks.append(jnp.concatenate(ws, axis=1))

        if emit_slots_smem:
            grp_ids_t = jnp.concatenate(g_ids_chunks, axis=0)
            grp_w_t = jnp.concatenate(g_w_chunks, axis=0)
            if grp_ids_t.shape[0] < 128:
                pad_rows = 128 - grp_ids_t.shape[0]
                grp_ids_t = jnp.concatenate(
                    [grp_ids_t, jnp.zeros((pad_rows, topk), dtype=jnp.int32)], axis=0
                )
                grp_w_t = jnp.concatenate(
                    [grp_w_t, jnp.zeros((pad_rows, topk), dtype=jnp.float32)], axis=0
                )
            col_slice = pl.ds(g * 128, 128)
            slot_ids_vmem[:, col_slice] = grp_ids_t.T
            slot_w_vmem[:, col_slice] = grp_w_t.T

    counts_vmem[...] = jnp.broadcast_to(counts_acc, (8, 256))
    c_cp = pltpu.make_async_copy(counts_vmem, counts_smem, dma_sem)
    c_cp.start()
    if emit_slots_smem:
        i_cp = pltpu.make_async_copy(slot_ids_vmem, slot_ids_smem, dma_sem)
        w_cp = pltpu.make_async_copy(slot_w_vmem, slot_w_smem, dma_sem)
        i_cp.start()
        w_cp.start()
        c_cp.wait()
        i_cp.wait()
        w_cp.wait()
    else:
        c_cp.wait()


def _plan_preloaded_replicas_smem_inline(
    my_owner,
    e_owner: int,
    num_owners: int,
    num_replicas: int,
    counts_smem,
    replica_meta_smem,
    local_suppress_smem,
    holder_slot_expert_smem,
    owner_load_smem,
    moved_flag_smem,
    *,
    cost_mode: str = "active",
    tile_m: int = 64,
    base_cost: int = 22,
    row_weight: int = 1,
):
    """Deterministic §4.5b SMEM replica planner."""

    def _cost(cnt):
        if cost_mode == "active":
            return jnp.where(cnt > 0, jnp.int32(1), jnp.int32(0))
        if cost_mode == "tiles":
            return jnp.where(
                cnt > 0, (cnt + jnp.int32(tile_m - 1)) // jnp.int32(tile_m), jnp.int32(0)
            )
        if cost_mode in ("weighted", "hybrid"):
            return jnp.where(
                cnt > 0, jnp.int32(base_cost) + jnp.int32(row_weight) * cnt, jnp.int32(0)
            )
        raise ValueError(f"Unsupported replica_cost_mode: {cost_mode}")

    @pl.loop(0, num_owners)
    def _init_load(o):
        base_e = o * e_owner

        def _sum_body(le, acc):
            return acc + _cost(counts_smem[0, base_e + le])

        owner_load_smem[o] = lax.fori_loop(0, e_owner, _sum_body, jnp.int32(0))

    k_rep = num_owners * num_replicas

    @pl.loop(0, k_rep)
    def _zero_moved(k):
        eid = replica_meta_smem[0, k]
        moved_flag_smem[jnp.maximum(eid, jnp.int32(0))] = jnp.int32(0)

    @pl.loop(0, e_owner)
    def _zero_sup(le):
        local_suppress_smem[le] = jnp.int32(0)

    @pl.loop(0, num_replicas)
    def _init_hslot(s):
        holder_slot_expert_smem[s] = jnp.int32(-1)

    @pl.loop(0, k_rep)
    def _consider(k):
        eid = replica_meta_smem[0, k]

        @pl.when(eid >= 0)
        def _valid():
            @pl.when(moved_flag_smem[eid] == 0)
            def _not_moved():
                cnt = counts_smem[0, eid]
                c = _cost(cnt)

                @pl.when(c > 0)
                def _active():
                    home = replica_meta_smem[1, k]
                    holder = replica_meta_smem[2, k]
                    hslot = replica_meta_smem[3, k]
                    l_home = owner_load_smem[home]
                    l_holder = owner_load_smem[holder]

                    @pl.when(l_home - c >= l_holder + c)
                    def _move():
                        owner_load_smem[home] = l_home - c
                        owner_load_smem[holder] = l_holder + c
                        moved_flag_smem[eid] = k + jnp.int32(1)

                        @pl.when(home == my_owner)
                        def _mark_sup():
                            local_suppress_smem[eid - my_owner * e_owner] = jnp.int32(1)

                        @pl.when(holder == my_owner)
                        def _mark_hold():
                            holder_slot_expert_smem[hslot] = eid


def _fp4_hoisted_blockdiag_gmm(
    xc, x_bd_buf, w_ref, scale_ref, k: int, n: int, *, bd_w4a8: bool = False
):
    """TPU7x hoisted block-diagonal K16 GMM (eliminates 15/16 of masked_x VPU ops).

    Optionally quantizes each (row, 16-group) to float8_e4m3fn (``bd_w4a8=True``)
    so the MXU runs ``fp8 x fp8`` at 4,614 TFLOP/s (2x BF16 rate) with folded FP32
    group scales!
    """
    a_rows = xc.shape[0]
    kc = 256
    gpc = 16
    rows = a_rows * gpc
    nc = min(n, 256 if a_rows > 2 else 512)
    diag = (
        lax.broadcasted_iota(jnp.int32, (rows, kc), 0) % gpc
        == lax.broadcasted_iota(jnp.int32, (rows, kc), 1) // 16
    )

    xsc_by_c0 = {}
    for c0 in range(0, k, kc):
        if bd_w4a8:
            xr = xc[:, c0 : c0 + kc].astype(jnp.float32).reshape(a_rows, gpc, 16)
            amax = jnp.max(jnp.abs(xr), axis=2, keepdims=True)
            xsc = jnp.maximum(amax * (1.0 / 448.0), 1e-12)
            xq = (xr / xsc).reshape(a_rows, kc).astype(jnp.float8_e4m3fn)
            xs = jnp.concatenate(
                [jnp.broadcast_to(xq[a : a + 1, :], (gpc, kc)) for a in range(a_rows)],
                axis=0,
            )
            x_bd_buf[pl.ds(0, rows), pl.ds(c0, kc)] = jnp.where(
                diag, xs, jnp.float8_e4m3fn(0.0)
            )
            xsc_by_c0[c0] = xsc.reshape(a_rows, 2, 8, 1)
        else:
            xs = jnp.concatenate(
                [
                    jnp.broadcast_to(xc[a : a + 1, c0 : c0 + kc], (gpc, kc))
                    for a in range(a_rows)
                ],
                axis=0,
            )
            x_bd_buf[pl.ds(0, rows), pl.ds(c0, kc)] = jnp.where(diag, xs, 0)

    cols = []
    for n0 in range(0, n, nc):
        acc = None
        for c0 in range(0, k, kc):
            xm = x_bd_buf[pl.ds(0, rows), pl.ds(c0, kc)]
            w = pltpu.bitcast(
                w_ref[pl.ds(c0 // 8, kc // 8), pl.ds(n0, nc)],
                jnp.float4_e2m1fn,
            ).astype(jnp.float8_e4m3fn)
            res = jnp.dot(xm, w, preferred_element_type=jnp.float32)
            s = scale_ref[pl.ds(c0 // 16, gpc), pl.ds(n0, nc)].astype(jnp.float32)
            s_4d = s.reshape(1, 2, 8, nc)
            if bd_w4a8:
                s_4d = s_4d * xsc_by_c0[c0]
            part = (res.reshape(a_rows, 2, 8, nc) * s_4d).sum(axis=1).reshape(
                a_rows * 8, nc
            )
            acc = part if acc is None else acc + part
        cols.append(
            jnp.concatenate(
                [
                    jnp.sum(acc[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
                    for a in range(a_rows)
                ],
                axis=0,
            )
        )
    return jnp.concatenate(cols, axis=1)


def _ep_tp2_kblock_gmm(
    x_ref, w_ref, scale_ref, k: int, n0, nc: int, block256: bool, scale_block: int
):
    """K256/K512 compact-scale or K16 dequant GMM hoisting scales & reusing K512 VREGs."""
    if not block256:
        return _fp4_k256_gmm(x_ref, w_ref, scale_ref, k, n0, nc, False, scale_block)
    kc = 256
    n_sb = k // scale_block
    can_hoist = n_sb >= 8 and (n_sb % 8 == 0)
    s_all = scale_ref[pl.ds(0, n_sb), pl.ds(n0, nc)] if can_hoist else None
    acc = None
    s_cur = None
    for c0 in range(0, k, kc):
        x = x_ref[:, c0 : c0 + kc].astype(jnp.bfloat16)
        w = pltpu.bitcast(
            w_ref[pl.ds(c0 // 8, kc // 8), pl.ds(n0, nc)], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        res = jnp.dot(x, w, preferred_element_type=jnp.float32)
        if (c0 % scale_block) == 0 or s_cur is None:
            sb_id = c0 // scale_block
            if s_all is not None:
                s_cur = s_all[sb_id : sb_id + 1, :].astype(jnp.float32)
            else:
                s_cur = scale_ref[pl.ds(sb_id, 1), pl.ds(n0, nc)].astype(jnp.float32)
        res = res * s_cur
        acc = res if acc is None else acc + res
    return acc


def _ep_tp2_kblock_dual_gmm(
    x_ref, w_ref, scale_ref, k: int, n0_g, n0_u, nc: int, block256: bool, scale_block: int
):
    """Interleaved gate + up K256/K512 GMM hoisting scales & reusing K512 VREGs."""
    if not block256:
        return _fp4_k256_dual_gmm(
            x_ref, w_ref, scale_ref, k, n0_g, n0_u, nc, False, scale_block
        )
    kc = 256
    n_sb = k // scale_block
    can_hoist = n_sb >= 8 and (n_sb % 8 == 0)
    sg_all = scale_ref[pl.ds(0, n_sb), pl.ds(n0_g, nc)] if can_hoist else None
    su_all = scale_ref[pl.ds(0, n_sb), pl.ds(n0_u, nc)] if can_hoist else None
    acc_g = None
    acc_u = None
    sg_cur = None
    su_cur = None
    for c0 in range(0, k, kc):
        x = x_ref[:, c0 : c0 + kc].astype(jnp.bfloat16)
        wg = pltpu.bitcast(
            w_ref[pl.ds(c0 // 8, kc // 8), pl.ds(n0_g, nc)], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        wu = pltpu.bitcast(
            w_ref[pl.ds(c0 // 8, kc // 8), pl.ds(n0_u, nc)], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        if (c0 % scale_block) == 0 or sg_cur is None:
            sb_id = c0 // scale_block
            if sg_all is not None:
                sg_cur = sg_all[sb_id : sb_id + 1, :].astype(jnp.float32)
                su_cur = su_all[sb_id : sb_id + 1, :].astype(jnp.float32)
            else:
                sg_cur = scale_ref[pl.ds(sb_id, 1), pl.ds(n0_g, nc)].astype(jnp.float32)
                su_cur = scale_ref[pl.ds(sb_id, 1), pl.ds(n0_u, nc)].astype(jnp.float32)
        rg = jnp.dot(x, wg, preferred_element_type=jnp.float32) * sg_cur
        ru = jnp.dot(x, wu, preferred_element_type=jnp.float32) * su_cur
        acc_g = rg if acc_g is None else acc_g + rg
        acc_u = ru if acc_u is None else acc_u + ru
    return acc_g, acc_u


def _compute_tile_gmm1_gmm2(
    x_tile_2d,
    act_ref,
    p_send_slot_ref,
    w1_ref,
    s1_ref,
    w2_ref,
    s2_ref,
    *,
    m_bucket: int,
    block: int,
    strip_nc1: int,
    strip_nc2: int,
    use_dual_gmm1: bool,
    use_blockdiag_k16: bool,
    wire_dtype,
    m_valid_bd: int = 0,
    out_2d: bool = False,
    x_bd_buf=None,
    bd_w4a8: bool = False,
    unroll_strips: bool = False,
):
    """Run GMM1 + SwiGLU + GMM2 -> p_send_slot_ref (2D [m,4096] or 3D [m,32,128])."""
    h = 4096
    i_half = 2048
    block256 = block in (256, 512)

    if use_blockdiag_k16 and (not block256) and m_valid_bd > 0:
        chunk_bd = 8 if m_valid_bd > 8 else m_valid_bd
        for r_off in range(0, m_valid_bd, chunk_bd):
            if out_2d:
                xc = x_tile_2d[pl.ds(r_off, chunk_bd), :]
            else:
                xc = jnp.concatenate(
                    [x_tile_2d[r_off + a].reshape(1, h) for a in range(chunk_bd)],
                    axis=0,
                )
            if x_bd_buf is not None:
                acc1 = _fp4_hoisted_blockdiag_gmm(
                    xc, x_bd_buf, w1_ref, s1_ref, h, 2 * i_half, bd_w4a8=bd_w4a8
                )
            else:
                acc1 = _fp4_blockdiag_gmm(xc, w1_ref, s1_ref, h, 2 * i_half)
            gate = acc1[:, :i_half]
            up = acc1[:, i_half:]
            act_bd = (
                jax.nn.silu(gate).astype(jnp.bfloat16).astype(jnp.float32) * up
            ).astype(jnp.bfloat16)
            if x_bd_buf is not None:
                acc2 = _fp4_hoisted_blockdiag_gmm(
                    act_bd, x_bd_buf, w2_ref, s2_ref, i_half, h, bd_w4a8=bd_w4a8
                )
            else:
                acc2 = _fp4_blockdiag_gmm(act_bd, w2_ref, s2_ref, i_half, h)
            if out_2d:
                p_send_slot_ref[pl.ds(r_off, chunk_bd), :] = acc2.astype(wire_dtype)
            else:
                for a in range(chunk_bd):
                    p_send_slot_ref[r_off + a] = acc2[a : a + 1, :].reshape(32, 128).astype(
                        wire_dtype
                    )
        return

    x_sub = x_tile_2d.at[pl.ds(0, m_bucket)]
    act_sub = act_ref.at[pl.ds(0, m_bucket)]

    def _gmm1_strip(j):
        col = pl.multiple_of(j * strip_nc1, strip_nc1)
        if use_dual_gmm1:
            g, u = _ep_tp2_kblock_dual_gmm(
                x_sub, w1_ref, s1_ref, h, col, i_half + col, strip_nc1, block256, block
            )
            act_sub[:, pl.ds(col, strip_nc1)] = (
                jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
            ).astype(jnp.bfloat16)
        else:
            g = _ep_tp2_kblock_gmm(x_sub, w1_ref, s1_ref, h, col, strip_nc1, block256, block)
            act_sub[:, pl.ds(col, strip_nc1)] = jax.nn.silu(g).astype(jnp.bfloat16)
            u = _ep_tp2_kblock_gmm(
                x_sub, w1_ref, s1_ref, h, i_half + col, strip_nc1, block256, block
            )
            sg = act_sub[:, pl.ds(col, strip_nc1)].astype(jnp.float32)
            act_sub[:, pl.ds(col, strip_nc1)] = (sg * u).astype(jnp.bfloat16)

    n_strips1 = i_half // strip_nc1
    if n_strips1 == 1:
        _gmm1_strip(0)
    elif unroll_strips and n_strips1 <= 4:
        for s_j in range(n_strips1):
            _gmm1_strip(s_j)
    else:
        @pl.loop(0, n_strips1)
        def _loop1(j):
            _gmm1_strip(j)

    if out_2d:
        n_strips2 = h // strip_nc2

        def _gmm2_2d(j):
            col = pl.multiple_of(j * strip_nc2, strip_nc2)
            d = _ep_tp2_kblock_gmm(
                act_sub, w2_ref, s2_ref, i_half, col, strip_nc2, block256, block
            )
            p_send_slot_ref[pl.ds(0, m_bucket), pl.ds(col, strip_nc2)] = d.astype(
                wire_dtype
            )

        if n_strips2 == 1:
            _gmm2_2d(0)
        elif unroll_strips and n_strips2 <= 4:
            for s_j in range(n_strips2):
                _gmm2_2d(s_j)
        else:
            @pl.loop(0, n_strips2)
            def _loop2_2d(j):
                _gmm2_2d(j)
        return

    min_store_cols = 2048 if wire_dtype == jnp.bfloat16 else 1024
    store_cols = max(strip_nc2, min_store_cols)
    strips_per_store = store_cols // strip_nc2
    sub_per_store = store_cols // 128
    n_stores = h // store_cols

    def _gmm2_store_group(g_idx):
        base_col = pl.multiple_of(g_idx * store_cols, store_cols)
        if strips_per_store == 1:
            part = _ep_tp2_kblock_gmm(
                act_sub, w2_ref, s2_ref, i_half, base_col, strip_nc2, block256, block
            )
        else:
            parts = [
                _ep_tp2_kblock_gmm(
                    act_sub,
                    w2_ref,
                    s2_ref,
                    i_half,
                    base_col + s_i * strip_nc2,
                    strip_nc2,
                    block256,
                    block,
                )
                for s_i in range(strips_per_store)
            ]
            part = jnp.concatenate(parts, axis=1)
        sub0 = pl.multiple_of(g_idx * sub_per_store, sub_per_store)
        p_send_slot_ref[pl.ds(0, m_bucket), pl.ds(sub0, sub_per_store), :] = (
            part.reshape(m_bucket, sub_per_store, 128).astype(wire_dtype)
        )

    if n_stores == 1:
        _gmm2_store_group(0)
    elif n_stores == 2 and wire_dtype == jnp.bfloat16:
        _gmm2_store_group(0)
        _gmm2_store_group(1)
    elif unroll_strips and n_stores <= 4:
        for g_i in range(n_stores):
            _gmm2_store_group(g_i)
    else:
        @pl.loop(0, n_stores)
        def _loop2(g_idx):
            _gmm2_store_group(g_idx)


def _select_strip_nc(m_bucket: int, block: int, override_nc: int | None) -> tuple[int, int, bool]:
    """Return ``(strip_nc1, strip_nc2, use_dual_gmm1)`` keeping live VREGs <= 32 AND bundles low!"""
    if override_nc is not None and override_nc > 0:
        nc1 = min(override_nc, 2048)
        nc2 = min(override_nc, 4096)
        use_dual = (m_bucket * (nc1 // 128) * 2) <= 32
        return nc1, nc2, use_dual
    compact_blk = block in (256, 512)
    if m_bucket <= 8:
        return (1024, 2048, True) if compact_blk else (1024, 1024, True)
    if m_bucket <= 16:
        return (1024, 1024, True) if compact_blk else (512, 1024, True)
    if m_bucket <= 32:
        return (512, 1024, True) if compact_blk else (512, 1024, False)
    return (512, 1024, False)


def _select_row_buckets(tile_m: int, block: int, bucket_policy: str) -> tuple[int, ...]:
    """Select compact row buckets keeping total bundles well below the ~66K overlay cliff."""
    if bucket_policy == "2bucket":
        if tile_m <= 8:
            return (8,)
        if tile_m <= 16:
            return (8, 16)
        if tile_m <= 32:
            return (8, 32)
        return (16, 64)
    # 4-bucket fallback
    base = tuple(b for b in (8, 16, 32, 64) if b <= tile_m)
    return base if (base and base[-1] == tile_m) else base + (tile_m,)


# =============================================================================
# MODE A: Symmetric Dense-Half TP2 Kernel (B <= 32 or explicit row_mode="dense")
#   Zero token-pack, zero 3D gather/scatter inside loop, ep-way halved RS!
# =============================================================================
def _ep_tp2_dense_kernel(
    rank_ref,
    *rest_refs,
    ep: int,
    tp: int,
    batch_size: int,
    block: int,
    chipmate_xor: int,
    has_replicas: bool,
    num_replicas: int,
    replica_cost_mode: str,
    return_sharded: bool,
    strip_nc: int | None,
    bridge_dtype_str: str,
    weight_bufs: int,
    use_blockdiag_k16: bool,
    rs_wire_dtype_str: str,
):
    idx = 0
    replica_meta_smem = rest_refs[idx] if has_replicas else None
    if has_replicas:
        idx += 1

    logits_ref = rest_refs[idx]; idx += 1
    bias_ref = rest_refs[idx]; idx += 1
    x_ref = rest_refs[idx]; idx += 1
    w1_ref = rest_refs[idx]; idx += 1
    s1_ref = rest_refs[idx]; idx += 1
    w2_ref = rest_refs[idx]; idx += 1
    s2_ref = rest_refs[idx]; idx += 1
    norm_ref = rest_refs[idx]; idx += 1

    if has_replicas:
        rep_w1_ref = rest_refs[idx]; idx += 1
        rep_s1_ref = rest_refs[idx]; idx += 1
        rep_w2_ref = rest_refs[idx]; idx += 1
        rep_s2_ref = rest_refs[idx]; idx += 1
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None

    output_ref = rest_refs[idx]; idx += 1

    w1_buf = rest_refs[idx]; idx += 1
    s1_buf = rest_refs[idx]; idx += 1
    w2_buf = rest_refs[idx]; idx += 1
    s2_buf = rest_refs[idx]; idx += 1
    w1_sem = rest_refs[idx]; idx += 1
    w2_sem = rest_refs[idx]; idx += 1

    act_buf = rest_refs[idx]; idx += 1
    dense_w_vmem = rest_refs[idx]; idx += 1
    accum_half_2d = rest_refs[idx]; idx += 1
    accum_half_3d = rest_refs[idx]; idx += 1

    p_send = rest_refs[idx]; idx += 1
    p_recv = rest_refs[idx]; idx += 1
    b_send_sem = rest_refs[idx]; idx += 1
    b_recv_sem = rest_refs[idx]; idx += 1
    b_free_sem = rest_refs[idx]; idx += 1
    chip_entry_sem = rest_refs[idx]; idx += 1

    counts_vmem = rest_refs[idx]; idx += 1
    counts_smem = rest_refs[idx]; idx += 1
    route_dma_sem = rest_refs[idx]; idx += 1

    exec_eid = rest_refs[idx]; idx += 1
    exec_w_src = rest_refs[idx]; idx += 1
    exec_w_slot = rest_refs[idx]; idx += 1

    if has_replicas:
        local_suppress_smem = rest_refs[idx]; idx += 1
        holder_slot_expert_smem = rest_refs[idx]; idx += 1
        owner_load_smem = rest_refs[idx]; idx += 1
        moved_flag_smem = rest_refs[idx]; idx += 1
    else:
        local_suppress_smem = holder_slot_expert_smem = owner_load_smem = moved_flag_smem = None

    rs_recv = rest_refs[idx]; idx += 1
    rs_send_sem = rest_refs[idx]; idx += 1
    rs_recv_sem = rest_refs[idx]; idx += 1
    owner_buf = rest_refs[idx]; idx += 1
    if rs_wire_dtype_str == "bfloat16":
        rs_send_bf16 = rest_refs[idx]; idx += 1
    else:
        rs_send_bf16 = None
    if not return_sharded:
        fan_recv = rest_refs[idx]; idx += 1
        fan_send_sem = rest_refs[idx]; idx += 1
        fan_recv_sem = rest_refs[idx]; idx += 1
    else:
        fan_recv = fan_send_sem = fan_recv_sem = None

    assert idx == len(rest_refs), f"Dense scratch mismatch: {idx} != {len(rest_refs)}"

    world_size = ep * tp
    e_chip = 256 // ep
    half_b = batch_size // 2
    rows_per_rank = batch_size // world_size
    wire_dtype = jnp.float32 if bridge_dtype_str == "float32" else jnp.bfloat16

    me = rank_ref[0]
    mate = chipmate_rank(me, chipmate_xor=chipmate_xor)
    chip_id, tp_half = rank_to_chip_and_half(me, chipmate_xor=chipmate_xor)

    pl.semaphore_signal(
        chip_entry_sem, 1, device_id={"model": mate}, device_id_type=pl.DeviceIdType.MESH
    )
    barrier = pltpu.get_barrier_semaphore()
    for dest in range(world_size):
        pl.semaphore_signal(
            barrier, device_id={"model": jnp.int32(dest)}, device_id_type=pl.DeviceIdType.MESH
        )

    w1_u32 = w1_ref.bitcast(jnp.uint32)
    w2_u32 = w2_ref.bitcast(jnp.uint32)
    if has_replicas:
        rep_w1_u32 = rep_w1_ref.bitcast(jnp.uint32)
        rep_w2_u32 = rep_w2_ref.bitcast(jnp.uint32)

    def _start_w1(buf_idx, w_src, w_slot):
        if has_replicas:
            @pl.when(w_src == 0)
            def _h1():
                pltpu.make_async_copy(w1_u32.at[w_slot], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
                pltpu.make_async_copy(s1_ref.at[w_slot], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()

            @pl.when(w_src != 0)
            def _r1():
                pltpu.make_async_copy(rep_w1_u32.at[w_slot], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
                pltpu.make_async_copy(rep_s1_ref.at[w_slot], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
        else:
            pltpu.make_async_copy(w1_u32.at[w_slot], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
            pltpu.make_async_copy(s1_ref.at[w_slot], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()

    def _start_w2(buf_idx, w_src, w_slot):
        if has_replicas:
            @pl.when(w_src == 0)
            def _h2():
                pltpu.make_async_copy(w2_u32.at[w_slot], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
                pltpu.make_async_copy(s2_ref.at[w_slot], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()

            @pl.when(w_src != 0)
            def _r2():
                pltpu.make_async_copy(rep_w2_u32.at[w_slot], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
                pltpu.make_async_copy(rep_s2_ref.at[w_slot], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
        else:
            pltpu.make_async_copy(w2_u32.at[w_slot], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
            pltpu.make_async_copy(s2_ref.at[w_slot], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()

    def _wait_w1(buf_idx):
        pltpu.make_async_copy(w1_buf.at[buf_idx], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).wait()
        pltpu.make_async_copy(s1_buf.at[buf_idx], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).wait()

    def _wait_w2(buf_idx):
        pltpu.make_async_copy(w2_buf.at[buf_idx], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).wait()
        pltpu.make_async_copy(s2_buf.at[buf_idx], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).wait()

    # Speculative prime of local slot 0 BEFORE routing so HBM starts streaming at t=0!
    _start_w1(0, jnp.int32(0), jnp.int32(0))
    if weight_bufs == 2:
        _start_w2(0, jnp.int32(0), jnp.int32(0))

    # Route in VPU -> dense_w_vmem [B, 256] + counts_smem [8, 256] (NO slot SMEM copy!)
    _route_topk_and_counts_in_kernel(
        logits_ref,
        bias_ref,
        counts_vmem,
        counts_smem,
        None,
        None,
        None,
        None,
        dense_w_vmem,
        route_dma_sem,
        batch_size=batch_size,
        emit_slots_smem=False,
    )

    if has_replicas:
        _plan_preloaded_replicas_smem_inline(
            chip_id,
            e_chip,
            ep,
            num_replicas,
            counts_smem,
            replica_meta_smem,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            moved_flag_smem,
            cost_mode=replica_cost_mode,
            tile_m=batch_size,
        )

    first_e = chip_id * e_chip

    def _scan_home_dense(le, n_act):
        eid = first_e + le
        cnt = counts_smem[0, eid]
        sup = local_suppress_smem[le] if has_replicas else jnp.int32(0)
        keep = (cnt > 0) & (sup == 0)

        @pl.when(keep)
        def _add():
            exec_eid[n_act] = eid
            exec_w_src[n_act] = jnp.int32(0)
            exec_w_slot[n_act] = le

        return n_act + jnp.where(keep, jnp.int32(1), jnp.int32(0))

    num_active = lax.fori_loop(0, e_chip, _scan_home_dense, jnp.int32(0))

    if has_replicas:
        def _scan_rep_dense(rs, n_act):
            eid = holder_slot_expert_smem[rs]
            valid = eid >= 0

            @pl.when(valid)
            def _add_r():
                exec_eid[n_act] = eid
                exec_w_src[n_act] = jnp.int32(1)
                exec_w_slot[n_act] = rs

            return n_act + jnp.where(valid, jnp.int32(1), jnp.int32(0))

        num_active = lax.fori_loop(0, num_replicas, _scan_rep_dense, num_active)

    # Reconcile speculative slot-0 prime: if expert 0 is NOT (src=0, slot=0) or chip is idle,
    # drain the speculative DMA cleanly and reissue the real expert 0!
    spec_hit = (num_active > 0) & (exec_w_src[0] == 0) & (exec_w_slot[0] == 0)

    @pl.when(~spec_hit)
    def _fix_spec():
        _wait_w1(0)
        if weight_bufs == 2:
            _wait_w2(0)

        @pl.when(num_active > 0)
        def _reissue():
            _start_w1(0, exec_w_src[0], exec_w_slot[0])
            if weight_bufs == 2:
                _start_w2(0, exec_w_src[0], exec_w_slot[0])

    # Zero accum_half_2d [half_b, 4096] in 8-row chunks
    for r0 in range(0, half_b, 8):
        accum_half_2d[pl.ds(r0, 8), :] = jnp.zeros((8, 4096), dtype=jnp.float32)

    pl.semaphore_wait(chip_entry_sem, 1)

    own_r0 = pl.multiple_of(
        jnp.where(tp_half == 0, jnp.int32(0), jnp.int32(half_b)), half_b
    )
    send_r0 = pl.multiple_of(
        jnp.where(tp_half == 0, jnp.int32(half_b), jnp.int32(0)), half_b
    )
    lane_128 = lax.broadcasted_iota(jnp.int32, (half_b, 128), 1)
    norm_gain_2d = norm_ref[0:1, :].astype(jnp.float32) + 1.0  # (1, 4096)

    nc1, nc2, use_dual = _select_strip_nc(batch_size, block, strip_nc)
    bd_dense = use_blockdiag_k16 and (block == 16) and (batch_size <= 16)

    def _finish_dense(prev_t, signal_free: bool):
        pslot = lax.bitwise_and(prev_t, jnp.int32(1))
        pltpu.make_async_copy(p_recv.at[pslot], p_recv.at[pslot], b_recv_sem.at[pslot]).wait()

        eid = exec_eid[prev_t]
        blk128 = pl.multiple_of(lax.shift_left(lax.shift_right_logical(eid, jnp.int32(7)), jnp.int32(7)), 128)
        w_sub = dense_w_vmem[pl.ds(own_r0, half_b), pl.ds(blk128, 128)]
        w_col = jnp.sum(
            jnp.where((lane_128 + blk128) == eid, w_sub, 0.0), axis=1, keepdims=True
        )  # (half_b, 1) f32

        # Process in 8-row (f32) or 16-row (bf16) chunks so live VREGs stay <= 32 AND sublane-aligned!
        step_r = 16 if wire_dtype == jnp.bfloat16 else 8
        for r_off in range(0, half_b, step_r):
            r_sl = pl.ds(r_off, step_r)
            own_sl = pl.ds(pl.multiple_of(own_r0 + r_off, step_r), step_r)
            z = p_send[pslot, own_sl, :].astype(jnp.float32) + p_recv[pslot, r_sl, :].astype(
                jnp.float32
            )
            u = z * norm_gain_2d
            inv = lax.rsqrt(jnp.mean(u * u, axis=1, keepdims=True) + 1e-8)
            w_r = w_col[r_off : r_off + step_r, :]
            accum_half_2d[r_sl, :] = accum_half_2d[r_sl, :] + jnp.where(
                w_r > 0.0, u * (inv * w_r), 0.0
            )

        if signal_free:
            pl.semaphore_signal(
                b_free_sem.at[pslot], 1, device_id={"model": mate}, device_id_type=pl.DeviceIdType.MESH
            )

    def _dense_body(t_idx, _):
        bslot = lax.bitwise_and(t_idx, jnp.int32(1))
        wslot = bslot if weight_bufs == 2 else jnp.int32(0)
        nxt_wslot = lax.bitwise_xor(wslot, jnp.int32(1)) if weight_bufs == 2 else jnp.int32(0)

        if weight_bufs == 2:
            _wait_w1(wslot)
            _wait_w2(wslot)
            @pl.when(t_idx + 1 < num_active)
            def _pf_next():
                _start_w1(nxt_wslot, exec_w_src[t_idx + 1], exec_w_slot[t_idx + 1])
                _start_w2(nxt_wslot, exec_w_src[t_idx + 1], exec_w_slot[t_idx + 1])
        else:
            _start_w2(0, exec_w_src[t_idx], exec_w_slot[t_idx])
            _wait_w1(0)
            _wait_w2(0)

        @pl.when(t_idx >= 2)
        def _drain_prev_s():
            pltpu.make_async_remote_copy(
                p_send.at[bslot, pl.ds(0, half_b)],
                p_recv.at[bslot],
                b_send_sem.at[bslot],
                b_recv_sem.at[bslot],
                device_id={"model": mate},
                device_id_type=pl.DeviceIdType.MESH,
            ).wait_send()

        _compute_tile_gmm1_gmm2(
            x_ref,
            act_buf,
            p_send.at[bslot],
            w1_buf.at[wslot],
            s1_buf.at[wslot],
            w2_buf.at[wslot],
            s2_buf.at[wslot],
            m_bucket=batch_size,
            block=block,
            strip_nc1=nc1,
            strip_nc2=nc2,
            use_dual_gmm1=use_dual,
            use_blockdiag_k16=bd_dense,
            wire_dtype=wire_dtype,
            m_valid_bd=batch_size if bd_dense else 0,
            out_2d=True,
        )

        if weight_bufs == 1:
            @pl.when(t_idx + 1 < num_active)
            def _pf_1b():
                _start_w1(0, exec_w_src[t_idx + 1], exec_w_slot[t_idx + 1])

        @pl.when(t_idx >= 2)
        def _wait_free():
            pl.semaphore_wait(b_free_sem.at[bslot], 1)

        pltpu.make_async_remote_copy(
            p_send.at[bslot, pl.ds(send_r0, half_b)],
            p_recv.at[bslot],
            b_send_sem.at[bslot],
            b_recv_sem.at[bslot],
            device_id={"model": mate},
            device_id_type=pl.DeviceIdType.MESH,
        ).start()

        @pl.when(t_idx > 0)
        def _fin_prev():
            lax.cond(
                t_idx + 1 < num_active,
                lambda: _finish_dense(t_idx - 1, True),
                lambda: _finish_dense(t_idx - 1, False),
            )

        return jnp.int32(0)

    lax.fori_loop(0, num_active, _dense_body, jnp.int32(0))

    @pl.when(num_active > 0)
    def _fin_last():
        _finish_dense(num_active - 1, False)
        last_slot = lax.bitwise_and(num_active - 1, jnp.int32(1))
        pltpu.make_async_remote_copy(
            p_send.at[last_slot, pl.ds(0, half_b)],
            p_recv.at[last_slot],
            b_send_sem.at[last_slot],
            b_recv_sem.at[last_slot],
            device_id={"model": mate},
            device_id_type=pl.DeviceIdType.MESH,
        ).wait_send()

        @pl.when(num_active >= 2)
        def _drain_prev2():
            prev_slot = lax.bitwise_xor(last_slot, jnp.int32(1))
            pltpu.make_async_remote_copy(
                p_send.at[prev_slot, pl.ds(0, half_b)],
                p_recv.at[prev_slot],
                b_send_sem.at[prev_slot],
                b_recv_sem.at[prev_slot],
                device_id={"model": mate},
                device_id_type=pl.DeviceIdType.MESH,
            ).wait_send()

    # Convert accum_half_2d [half_b, 4096] -> accum_half_3d [half_b, 32, 128] once (8-row chunks)
    for r0 in range(0, half_b, 8):
        accum_half_3d[pl.ds(r0, 8), :, :] = accum_half_2d[pl.ds(r0, 8), :].reshape(8, 32, 128)

    if rs_wire_dtype_str == "bfloat16":
        for r0 in range(0, half_b, 4):
            rs_send_bf16[pl.ds(r0, 4)] = accum_half_3d[pl.ds(r0, 4)].astype(jnp.bfloat16)
        rs_src = rs_send_bf16
    else:
        rs_src = accum_half_3d

    # Halved ep-way Reduce-Scatter!
    # tp_half=0 sends slice c -> rank c; tp_half=1 sends slice c -> rank (ep + c)!
    # Every rank d in 0..P-1 receives ONLY ep slices (one from each chip of tp_half = d // ep)!
    pl.semaphore_wait(barrier, world_size)
    base_dest = tp_half * ep
    rs_copies = [
        pltpu.make_async_remote_copy(
            rs_src.at[pl.ds(c * rows_per_rank, rows_per_rank)],
            rs_recv.at[chip_id],
            rs_send_sem.at[c],
            rs_recv_sem,
            device_id={"model": base_dest + jnp.int32(c)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        for c in range(ep)
    ]
    for op in rs_copies:
        op.start()
    pltpu.make_async_copy(rs_recv, rs_recv, rs_recv_sem).wait()

    r_step = min(rows_per_rank, 4)
    for r0 in range(0, rows_per_rank, r_step):
        r_sl = pl.ds(r0, r_step)
        tot = rs_recv[0, r_sl].astype(jnp.float32)
        for src_c in range(1, ep):
            tot = tot + rs_recv[src_c, r_sl].astype(jnp.float32)
        owner_buf[r_sl] = tot.astype(jnp.bfloat16)

    if return_sharded:
        output_ref[...] = owner_buf[...].reshape(rows_per_rank, 4096)
        for op in rs_copies:
            op.wait_send()
    else:
        fan_copies = [
            pltpu.make_async_remote_copy(
                owner_buf,
                fan_recv.at[me],
                fan_send_sem.at[dest],
                fan_recv_sem,
                device_id={"model": jnp.int32(dest)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            for dest in range(world_size)
        ]
        for op in fan_copies:
            op.start()
        pltpu.make_async_copy(fan_recv, fan_recv, fan_recv_sem).wait()
        chunk_ranks = 1 if (rows_per_rank >= 8 and rows_per_rank % 8 == 0) else (8 // rows_per_rank)
        chunk_rows = chunk_ranks * rows_per_rank
        for r0 in range(0, world_size, chunk_ranks):
            output_ref[pl.ds(r0 * rows_per_rank, chunk_rows), :] = (
                fan_recv[pl.ds(r0, chunk_ranks)].reshape(chunk_rows, 4096)
            )
        for op in fan_copies:
            op.wait_send()
        for op in rs_copies:
            op.wait_send()


def _read_ep_tp2_env_knobs(batch_size: int):
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is not None and raw.strip() != "":
            return raw.strip().lower() not in ("0", "false", "no", "off")
        try:
            from sglang_tpu.envs import envs

            ev = getattr(envs, name, None)
            if ev is not None:
                return bool(ev.get())
        except Exception:
            pass
        return default

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is not None and raw.strip() != "":
            s = raw.strip().lower()
            if s in ("true", "yes", "on"):
                return 1
            if s in ("false", "no", "off"):
                return 0
            return int(s)
        try:
            from sglang_tpu.envs import envs

            ev = getattr(envs, name, None)
            if ev is not None:
                return int(ev.get())
        except Exception:
            pass
        return default

    min_p = float(os.environ.get("SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK", "0.0") or 0.0)
    free_spec = os.environ.get("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS", "")
    try:
        from sglang_tpu.envs import envs

        if os.environ.get("SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK") is None:
            e_mp = getattr(envs, "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK", None)
            if e_mp is not None:
                min_p = float(e_mp.get())
        if os.environ.get("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS") is None:
            e_fr = getattr(envs, "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS", None)
            if e_fr is not None:
                free_spec = str(e_fr.get())
    except Exception:
        pass
    f_rows, f_max_w, f_min_s = parse_free_expert_rows_spec(free_spec)
    f_rows_eff, f_min_s_eff = scale_free_expert_for_batch(f_rows, f_min_s, batch_size)
    return (
        max(0.0, float(min_p)),
        f_rows_eff,
        f_max_w,
        f_min_s_eff,
        _env_bool("SGLANG_TPU_MUSE_KERNEL_DECODE_STATIC_HOLDER_MASK", False),
        _env_bool("SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST", False),
        _env_int("SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN", 0),
        _env_bool("SGLANG_TPU_MUSE_KERNEL_DECODE_UNROLL_STRIPS", False),
        _env_bool("SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_GS", False),
        _env_bool("SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2_PAIR_RS", False),
        _env_bool("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_BRIDGE64", False),
    )


# =============================================================================
# MODE B: Symmetric Sparse TP2 Kernel (B >= 16, K16/K256/K512, Wave-9 0-ULP!)
# =============================================================================
def _ep_tp2_sparse_kernel(
    rank_ref,
    *rest_refs,
    ep: int,
    tp: int,
    batch_size: int,
    block: int,
    chipmate_xor: int,
    has_replicas: bool,
    num_replicas: int,
    replica_cost_mode: str,
    return_sharded: bool,
    tile_m: int,
    strip_nc: int | None,
    bridge_dtype_str: str,
    gather_mode: str,
    weight_bufs: int,
    use_blockdiag_k16: bool,
    bd_w4a8: bool,
    rs_wire_dtype_str: str,
    bucket_policy: str,
    min_p_topk: float,
    free_max_rows: int,
    free_max_w: float,
    free_min_slot: int,
    static_holder_mask: bool,
    route_fast: bool,
    fast_plan: int,
    unroll_strips: bool,
    fast_gs: bool,
    pair_rs: bool,
):
    idx = 0
    replica_meta_smem = rest_refs[idx] if has_replicas else None
    if has_replicas:
        idx += 1

    logits_ref = rest_refs[idx]; idx += 1
    bias_ref = rest_refs[idx]; idx += 1
    x_ref = rest_refs[idx]; idx += 1
    w1_ref = rest_refs[idx]; idx += 1
    s1_ref = rest_refs[idx]; idx += 1
    w2_ref = rest_refs[idx]; idx += 1
    s2_ref = rest_refs[idx]; idx += 1
    norm_ref = rest_refs[idx]; idx += 1

    if has_replicas:
        rep_w1_ref = rest_refs[idx]; idx += 1
        rep_s1_ref = rest_refs[idx]; idx += 1
        rep_w2_ref = rest_refs[idx]; idx += 1
        rep_s2_ref = rest_refs[idx]; idx += 1
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None

    output_ref = rest_refs[idx]; idx += 1

    w1_buf = rest_refs[idx]; idx += 1
    s1_buf = rest_refs[idx]; idx += 1
    w2_buf = rest_refs[idx]; idx += 1
    s2_buf = rest_refs[idx]; idx += 1
    w1_sem = rest_refs[idx]; idx += 1
    w2_sem = rest_refs[idx]; idx += 1

    x_3d_buf = rest_refs[idx]; idx += 1
    x_tile_3d = rest_refs[idx]; idx += 1
    x_tile_2d = rest_refs[idx]; idx += 1
    act_buf = rest_refs[idx]; idx += 1
    norm_gain_3d = rest_refs[idx]; idx += 1
    accum_buf = rest_refs[idx]; idx += 1

    p_send = rest_refs[idx]; idx += 1
    p_recv = rest_refs[idx]; idx += 1
    b_send_sem = rest_refs[idx]; idx += 1
    b_recv_sem = rest_refs[idx]; idx += 1
    b_free_sem = rest_refs[idx]; idx += 1
    chip_entry_sem = rest_refs[idx]; idx += 1
    slot_send_len = rest_refs[idx]; idx += 1

    if gather_mode == "coop_bridge":
        x_fwd_send_sem = rest_refs[idx]; idx += 1
        x_fwd_recv_sem = rest_refs[idx]; idx += 1
        x_fwd_free_sem = rest_refs[idx]; idx += 1
    else:
        x_fwd_send_sem = x_fwd_recv_sem = x_fwd_free_sem = None

    r_scratch = RoutingScratch(*rest_refs[idx : idx + 12])
    idx += 12
    route_dma_sem = rest_refs[idx]; idx += 1

    tile_w_src = rest_refs[idx]; idx += 1
    tile_w_slot = rest_refs[idx]; idx += 1
    tile_row_base = rest_refs[idx]; idx += 1
    tile_row_cnt = rest_refs[idx]; idx += 1
    tile_is_new_exp = rest_refs[idx]; idx += 1
    tile_is_last_of_exp = rest_refs[idx]; idx += 1
    tile_next_exp_src = rest_refs[idx]; idx += 1
    tile_next_exp_slot = rest_refs[idx]; idx += 1

    if has_replicas:
        local_suppress_smem = rest_refs[idx]; idx += 1
        holder_slot_expert_smem = rest_refs[idx]; idx += 1
        owner_load_smem = rest_refs[idx]; idx += 1
        moved_flag_smem = rest_refs[idx]; idx += 1
        if not static_holder_mask:
            holder_cand_smem = rest_refs[idx]; idx += 1
        else:
            holder_cand_smem = None
    else:
        local_suppress_smem = holder_slot_expert_smem = owner_load_smem = moved_flag_smem = None
        holder_cand_smem = None

    if pair_rs:
        pair_recv = rest_refs[idx]; idx += 1
        pair_sum_buf = rest_refs[idx]; idx += 1
        pair_send_sem = rest_refs[idx]; idx += 1
        pair_recv_sem = rest_refs[idx]; idx += 1
    else:
        pair_recv = pair_sum_buf = pair_send_sem = pair_recv_sem = None

    rs_recv = rest_refs[idx]; idx += 1
    rs_send_sem = rest_refs[idx]; idx += 1
    rs_recv_sem = rest_refs[idx]; idx += 1
    owner_buf = rest_refs[idx]; idx += 1
    if (not pair_rs) and rs_wire_dtype_str == "bfloat16":
        rs_send_bf16 = rest_refs[idx]; idx += 1
    else:
        rs_send_bf16 = None
    if not return_sharded:
        fan_recv = rest_refs[idx]; idx += 1
        fan_send_sem = rest_refs[idx]; idx += 1
        fan_recv_sem = rest_refs[idx]; idx += 1
    else:
        fan_recv = fan_send_sem = fan_recv_sem = None

    if use_blockdiag_k16 and block == 16:
        x_bd_buf = rest_refs[idx]; idx += 1
    else:
        x_bd_buf = None

    assert idx == len(rest_refs), f"Sparse scratch mismatch: {idx} != {len(rest_refs)}"

    world_size = ep * tp
    e_chip = 256 // ep
    wire_dtype = jnp.float32 if bridge_dtype_str == "float32" else jnp.bfloat16
    rs_dt = jnp.float32 if rs_wire_dtype_str == "float32" else jnp.bfloat16
    me = rank_ref[0]
    mate = chipmate_rank(me, chipmate_xor=chipmate_xor)
    chip_id, tp_half = rank_to_chip_and_half(me, chipmate_xor=chipmate_xor)

    # 1. Signal entry readiness + global RS barrier early so network converges under routing!
    pl.semaphore_signal(
        chip_entry_sem, 1, device_id={"model": mate}, device_id_type=pl.DeviceIdType.MESH
    )
    barrier = pltpu.get_barrier_semaphore()
    for dest in range(world_size):
        pl.semaphore_signal(
            barrier, device_id={"model": jnp.int32(dest)}, device_id_type=pl.DeviceIdType.MESH
        )

    w1_u32 = w1_ref.bitcast(jnp.uint32)
    w2_u32 = w2_ref.bitcast(jnp.uint32)
    if has_replicas:
        rep_w1_u32 = rep_w1_ref.bitcast(jnp.uint32)
        rep_w2_u32 = rep_w2_ref.bitcast(jnp.uint32)

    def _start_w1(buf_idx, w_src, w_slot):
        if has_replicas:
            @pl.when(w_src == 0)
            def _h1():
                pltpu.make_async_copy(w1_u32.at[w_slot], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
                pltpu.make_async_copy(s1_ref.at[w_slot], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()

            @pl.when(w_src != 0)
            def _r1():
                pltpu.make_async_copy(rep_w1_u32.at[w_slot], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
                pltpu.make_async_copy(rep_s1_ref.at[w_slot], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
        else:
            pltpu.make_async_copy(w1_u32.at[w_slot], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()
            pltpu.make_async_copy(s1_ref.at[w_slot], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).start()

    def _start_w2(buf_idx, w_src, w_slot):
        if has_replicas:
            @pl.when(w_src == 0)
            def _h2():
                pltpu.make_async_copy(w2_u32.at[w_slot], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
                pltpu.make_async_copy(s2_ref.at[w_slot], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()

            @pl.when(w_src != 0)
            def _r2():
                pltpu.make_async_copy(rep_w2_u32.at[w_slot], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
                pltpu.make_async_copy(rep_s2_ref.at[w_slot], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
        else:
            pltpu.make_async_copy(w2_u32.at[w_slot], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()
            pltpu.make_async_copy(s2_ref.at[w_slot], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).start()

    def _wait_w1(buf_idx):
        pltpu.make_async_copy(w1_buf.at[buf_idx], w1_buf.at[buf_idx], w1_sem.at[buf_idx]).wait()
        pltpu.make_async_copy(s1_buf.at[buf_idx], s1_buf.at[buf_idx], w1_sem.at[buf_idx]).wait()

    def _wait_w2(buf_idx):
        pltpu.make_async_copy(w2_buf.at[buf_idx], w2_buf.at[buf_idx], w2_sem.at[buf_idx]).wait()
        pltpu.make_async_copy(s2_buf.at[buf_idx], s2_buf.at[buf_idx], w2_sem.at[buf_idx]).wait()

    # 2. Exact Wave-9 0-ULP VPU routing (with MIN_P + #2407 b64_ms16 FREE + ROUTE_FAST)
    if has_replicas:
        if static_holder_mask:
            cand_eids = extract_my_holder_candidates_tuple(
                chip_id, num_replicas, replica_meta_smem
            )
        else:
            extract_my_holder_candidates_smem(
                chip_id, num_replicas, replica_meta_smem, holder_cand_smem
            )
            cand_eids = tuple(holder_cand_smem[s] for s in range(num_replicas))
    else:
        cand_eids = ()

    if _bridge1_prerouted_active():
        _route_prerouted_topk_in_kernel(
            logits_ref,
            r_scratch,
            route_dma_sem,
            my_owner=chip_id,
            e_per_owner=e_chip,
            holder_cand_eids=cand_eids,
            hoist_holder_vecs=static_holder_mask,
            precompute_owner_loads=(has_replicas and static_holder_mask),
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=tile_m,
            topk=8,
            route_chunk_rows=min(batch_size, 32),
        )
    else:
        route_topk_and_counts_in_kernel(
            logits_ref,
            bias_ref,
            r_scratch,
            route_dma_sem,
            my_owner=chip_id,
            e_per_owner=e_chip,
            holder_cand_eids=cand_eids,
            hoist_holder_vecs=static_holder_mask,
            precompute_owner_loads=(has_replicas and static_holder_mask),
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=tile_m,
            topk=8,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=min(batch_size, 32),
            min_p_topk=min_p_topk,
            free_max_rows=free_max_rows,
            free_max_w=free_max_w,
            free_min_slot=free_min_slot,
            route_fast=route_fast,
        )

    # 3. Branchless 0-sbr SMEM replica & execution plan
    if has_replicas:
        run_replica_planner_smem(
            chip_id,
            e_chip,
            ep,
            num_replicas,
            r_scratch.counts_smem,
            replica_meta_smem,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            cost_mode=replica_cost_mode,
            tile_m=tile_m,
            moved_flag_smem=moved_flag_smem,
            vpu_precomputed_loads=static_holder_mask,
            fast_plan=fast_plan,
        )
        num_active_exp = build_exec_plan_from_masks_smem(
            chip_id,
            e_chip,
            num_replicas,
            local_suppress_smem,
            holder_slot_expert_smem,
            r_scratch,
            fast_plan=bool(fast_plan),
        )
    else:
        num_active_exp = build_default_exec_plan_smem(
            chip_id, e_chip, r_scratch, fast_plan=bool(fast_plan)
        )

    exec_w_src = r_scratch.exec_weight_src
    exec_w_slot = r_scratch.exec_weight_slot
    exec_start = r_scratch.exec_start
    packed_tok_ids = r_scratch.packed_tok_ids
    packed_tok_w = r_scratch.packed_tok_w
    max_packed_len = int(packed_tok_ids.shape[0])

    # 4. Immediate 100%-hit Expert 0 HBM prime overlapped with tile-expand, pack & VMEM setup!
    @pl.when(num_active_exp > 0)
    def _prime_exp0():
        s0 = exec_w_src[0]
        sl0 = exec_w_slot[0]
        _start_w1(0, s0, sl0)
        if weight_bufs == 2:
            _start_w2(0, s0, sl0)

    # Build flat tile schedule in SMEM (exec_start[num_active_exp] == total_rows sentinel)
    max_nt = (batch_size + tile_m - 1) // tile_m
    tm_shift = tile_m.bit_length() - 1

    def _expand_tiles(a_idx, n_tiles):
        src_a = exec_w_src[a_idx]
        slot_a = exec_w_slot[a_idx]
        base_a = exec_start[a_idx]
        end_a = exec_start[a_idx + 1]
        cnt_a = end_a - base_a
        nt_a = lax.shift_right_logical(
            cnt_a + jnp.int32(tile_m - 1), jnp.int32(tm_shift)
        )
        has_next_e = (a_idx + 1) < num_active_exp
        next_src = jnp.where(has_next_e, exec_w_src[a_idx + 1], jnp.int32(-1))
        next_slot = jnp.where(has_next_e, exec_w_slot[a_idx + 1], jnp.int32(-1))

        if fast_plan and max_nt <= 4:
            cur_t = n_tiles
            for sub_t in range(max_nt):
                act_sub = jnp.int32(sub_t) < nt_a
                r_off = jnp.int32(sub_t * tile_m)
                rcnt = jnp.minimum(cnt_a - r_off, jnp.int32(tile_m))
                is_last = (jnp.int32(sub_t + 1) == nt_a)
                tile_w_src[cur_t] = src_a
                tile_w_slot[cur_t] = slot_a
                tile_row_base[cur_t] = base_a + r_off
                tile_row_cnt[cur_t] = rcnt
                tile_is_new_exp[cur_t] = jnp.int32(1 if sub_t == 0 else 0)
                tile_is_last_of_exp[cur_t] = is_last.astype(jnp.int32)
                tile_next_exp_src[cur_t] = jnp.where(is_last, next_src, jnp.int32(-1))
                tile_next_exp_slot[cur_t] = jnp.where(is_last, next_slot, jnp.int32(-1))
                cur_t = cur_t + act_sub.astype(jnp.int32)
            return cur_t

        def _fill_one(sub_t, cur_t):
            r_off = sub_t * jnp.int32(tile_m)
            rem = cnt_a - r_off
            rcnt = jnp.minimum(rem, jnp.int32(tile_m))
            tile_w_src[cur_t] = src_a
            tile_w_slot[cur_t] = slot_a
            tile_row_base[cur_t] = base_a + r_off
            tile_row_cnt[cur_t] = rcnt
            tile_is_new_exp[cur_t] = jnp.where(sub_t == 0, jnp.int32(1), jnp.int32(0))
            is_last = sub_t + 1 == nt_a
            tile_is_last_of_exp[cur_t] = jnp.where(is_last, jnp.int32(1), jnp.int32(0))
            tile_next_exp_src[cur_t] = jnp.where(is_last, next_src, jnp.int32(-1))
            tile_next_exp_slot[cur_t] = jnp.where(is_last, next_slot, jnp.int32(-1))
            return cur_t + jnp.int32(1)

        return lax.fori_loop(0, nt_a, _fill_one, n_tiles)

    num_tiles = lax.fori_loop(0, num_active_exp, _expand_tiles, jnp.int32(0))

    @pl.when(num_active_exp > 0)
    def _pack_tokens_fast():
        pack_active_tokens_smem(
            batch_size, r_scratch, has_replicas=has_replicas, fast_gs=fast_gs
        )

    # Spill-free VMEM buffer setup (static 16-row chunks for 2D bf16 -> 0 E2003, <= 16 VREGs!)
    norm_gain_3d[...] = (norm_ref[0:1, :].astype(jnp.float32) + 1.0).reshape(32, 128)

    for b0 in range(0, batch_size, 16):
        x_3d_buf[pl.ds(b0, 16), :, :] = x_ref[pl.ds(b0, 16), :].reshape(16, 32, 128)

    z_step = 8 if (fast_gs and batch_size % 8 == 0) else 4
    z_blk = jnp.zeros((z_step, 32, 128), dtype=jnp.float32)
    for b0 in range(0, batch_size, z_step):
        accum_buf[pl.ds(b0, z_step), :, :] = z_blk
    if fast_gs:
        accum_buf[batch_size] = jnp.zeros((32, 128), dtype=jnp.float32)

    z8_bf = jnp.zeros((8, 32, 128), dtype=jnp.bfloat16)
    for r0 in range(0, tile_m, 8):
        x_tile_3d[pl.ds(r0, 8)] = z8_bf

    slot_send_len[0] = jnp.int32(0)
    slot_send_len[1] = jnp.int32(0)

    pl.semaphore_wait(chip_entry_sem, 1)

    def _row_split(t_idx, m_valid):
        split = lax.shift_right_logical(
            m_valid + jnp.int32(1) - lax.bitwise_and(t_idx, jnp.int32(1)),
            jnp.int32(1),
        )
        own_start = jnp.where(tp_half == 0, jnp.int32(0), split)
        own_len = jnp.where(tp_half == 0, split, m_valid - split)
        send_start = jnp.where(tp_half == 0, split, jnp.int32(0))
        send_len = jnp.where(tp_half == 0, m_valid - split, split)
        return own_start, own_len, send_start, send_len, own_len

    def _drain_bridge_send(slot_idx):
        slen = slot_send_len[slot_idx]

        @pl.when(slen > 0)
        def _ws():
            pltpu.make_async_remote_copy(
                p_send.at[slot_idx, pl.ds(0, slen)],
                p_recv.at[slot_idx, pl.ds(0, slen)],
                b_send_sem.at[slot_idx],
                b_recv_sem.at[slot_idx],
                device_id={"model": mate},
                device_id_type=pl.DeviceIdType.MESH,
            ).wait_send()
            slot_send_len[slot_idx] = jnp.int32(0)

    def _finish_tile(prev_t, signal_free: bool):
        prev_slot = lax.bitwise_and(prev_t, jnp.int32(1))
        prev_base = tile_row_base[prev_t]
        prev_cnt = tile_row_cnt[prev_t]
        own_start, own_len, _ss, _sl, recv_len = _row_split(prev_t, prev_cnt)

        @pl.when(recv_len > 0)
        def _wait_recv():
            pltpu.make_async_copy(
                p_recv.at[prev_slot, pl.ds(0, recv_len)],
                p_recv.at[prev_slot, pl.ds(0, recv_len)],
                b_recv_sem.at[prev_slot],
            ).wait()

            ng = norm_gain_3d[...]

            def _do_k_rows(r_start, k_cnt: int):
                us = []
                invs = []
                for k in range(k_cnt):
                    rk = r_start + jnp.int32(k)
                    po = p_send[prev_slot, own_start + rk].astype(jnp.float32)
                    pr = p_recv[prev_slot, rk].astype(jnp.float32)
                    uk = (po + pr) * ng
                    sq = jnp.sum(jnp.sum(uk * uk, axis=0, keepdims=True), axis=1, keepdims=True)
                    us.append(uk)
                    invs.append(lax.rsqrt(sq * (1.0 / 4096.0) + 1e-8))
                for k in range(k_cnt):
                    row_idx = prev_base + own_start + r_start + jnp.int32(k)
                    tok_id = packed_tok_ids[row_idx]
                    wt = packed_tok_w[row_idx]
                    accum_buf[tok_id] = accum_buf[tok_id] + us[k] * (invs[k] * wt)

            if fast_gs:
                # 0-sbr branchless 4-row RMSNorm + scatter-add covering 95%+ of tiles (own_len <= 4)!
                us4 = []
                invs4 = []
                max_rk = own_len - jnp.int32(1)
                for k in range(4):
                    rk = jnp.minimum(jnp.int32(k), max_rk)
                    po = p_send[prev_slot, own_start + rk].astype(jnp.float32)
                    pr = p_recv[prev_slot, rk].astype(jnp.float32)
                    uk = (po + pr) * ng
                    sq = jnp.sum(jnp.sum(uk * uk, axis=0, keepdims=True), axis=1, keepdims=True)
                    us4.append(uk)
                    invs4.append(lax.rsqrt(sq * (1.0 / 4096.0) + 1e-8))
                for k in range(4):
                    valid_k = jnp.int32(k) < own_len
                    safe_p = jnp.where(
                        valid_k,
                        prev_base + own_start + jnp.int32(k),
                        jnp.minimum(prev_base + own_start, jnp.int32(max_packed_len - 1)),
                    )
                    tok_id = jnp.where(valid_k, packed_tok_ids[safe_p], jnp.int32(batch_size))
                    wt = jnp.where(valid_k, packed_tok_w[safe_p], jnp.float32(0.0))
                    accum_buf[tok_id] = accum_buf[tok_id] + us4[k] * (invs4[k] * wt)

                @pl.when(own_len > 4)
                def _gt4():
                    rem_len = own_len - jnp.int32(4)
                    n4_hi = lax.shift_right_logical(rem_len, jnp.int32(2))

                    @pl.loop(0, n4_hi)
                    def _blk4_hi(bi):
                        _do_k_rows(jnp.int32(4) + lax.shift_left(bi, jnp.int32(2)), 4)

                    rem_off = jnp.int32(4) + lax.shift_left(n4_hi, jnp.int32(2))
                    has_2 = (
                        lax.bitwise_and(lax.shift_right_logical(rem_len, jnp.int32(1)), jnp.int32(1))
                        != 0
                    )

                    @pl.when(has_2)
                    def _tail2_hi():
                        _do_k_rows(rem_off, 2)

                    off_1 = rem_off + jnp.where(has_2, jnp.int32(2), jnp.int32(0))
                    has_1 = lax.bitwise_and(rem_len, jnp.int32(1)) != 0

                    @pl.when(has_1)
                    def _tail1_hi():
                        _do_k_rows(off_1, 1)
            else:
                n4 = lax.shift_right_logical(own_len, jnp.int32(2))

                @pl.loop(0, n4)
                def _blk4(bi):
                    _do_k_rows(lax.shift_left(bi, jnp.int32(2)), 4)

                rem_off = lax.shift_left(n4, jnp.int32(2))
                has_2 = lax.bitwise_and(lax.shift_right_logical(own_len, jnp.int32(1)), jnp.int32(1)) != 0

                @pl.when(has_2)
                def _tail2():
                    _do_k_rows(rem_off, 2)

                off_1 = rem_off + jnp.where(has_2, jnp.int32(2), jnp.int32(0))
                has_1 = lax.bitwise_and(own_len, jnp.int32(1)) != 0

                @pl.when(has_1)
                def _tail1():
                    _do_k_rows(off_1, 1)

        if signal_free:
            pl.semaphore_signal(
                b_free_sem.at[prev_slot],
                1,
                device_id={"model": mate},
                device_id_type=pl.DeviceIdType.MESH,
            )

    row_buckets = _select_row_buckets(tile_m, block, bucket_policy)

    def _run_bucketed_compute(bslot, wslot, m_valid):
        def _dispatch_bucket(bi: int):
            mb = row_buckets[bi]
            nc1, nc2, use_dual = _select_strip_nc(mb, block, strip_nc)

            def _do_this():
                # Spill-free 8-row chunked x_tile_3d -> x_tile_2d (only 8 VREGs live!)
                for r0 in range(0, mb, 8):
                    x_tile_2d[pl.ds(r0, 8), :] = x_tile_3d[pl.ds(r0, 8), :, :].reshape(8, 4096)
                _compute_tile_gmm1_gmm2(
                    x_tile_2d,
                    act_buf,
                    p_send.at[bslot],
                    w1_buf.at[wslot],
                    s1_buf.at[wslot],
                    w2_buf.at[wslot],
                    s2_buf.at[wslot],
                    m_bucket=mb,
                    block=block,
                    strip_nc1=nc1,
                    strip_nc2=nc2,
                    use_dual_gmm1=use_dual,
                    use_blockdiag_k16=False,
                    wire_dtype=wire_dtype,
                    unroll_strips=unroll_strips,
                )

            if bi == len(row_buckets) - 1:
                _do_this()
            else:
                lax.cond(m_valid <= mb, _do_this, lambda: _dispatch_bucket(bi + 1))

        if use_blockdiag_k16 and block == 16:
            bd_sizes = (1, 2, 4, 8)

            def _bd_branch(a_val: int):
                _compute_tile_gmm1_gmm2(
                    x_tile_3d,
                    act_buf,
                    p_send.at[bslot],
                    w1_buf.at[wslot],
                    s1_buf.at[wslot],
                    w2_buf.at[wslot],
                    s2_buf.at[wslot],
                    m_bucket=a_val,
                    block=block,
                    strip_nc1=1024,
                    strip_nc2=1024,
                    use_dual_gmm1=True,
                    use_blockdiag_k16=True,
                    wire_dtype=wire_dtype,
                    m_valid_bd=a_val,
                    x_bd_buf=x_bd_buf,
                    bd_w4a8=bd_w4a8,
                    unroll_strips=unroll_strips,
                )

            fb_bi = next((i for i, b in enumerate(row_buckets) if b > bd_sizes[-1]), len(row_buckets) - 1)

            def _dispatch_bd(idx_bd: int):
                a_v = bd_sizes[idx_bd]
                if idx_bd == len(bd_sizes) - 1:
                    lax.cond(
                        m_valid <= a_v,
                        lambda: _bd_branch(a_v),
                        lambda: _dispatch_bucket(fb_bi),
                    )
                else:
                    lax.cond(
                        m_valid <= a_v,
                        lambda: _bd_branch(a_v),
                        lambda: _dispatch_bd(idx_bd + 1),
                    )

            _dispatch_bd(0)
        else:
            _dispatch_bucket(0)

    def _tile_body(t_idx, cur_wslot):
        bslot = lax.bitwise_and(t_idx, jnp.int32(1))
        r_base = tile_row_base[t_idx]
        m_valid = tile_row_cnt[t_idx]
        is_new_e = tile_is_new_exp[t_idx]
        is_last_e = tile_is_last_of_exp[t_idx]
        nxt_src = tile_next_exp_src[t_idx]
        nxt_slot = tile_next_exp_slot[t_idx]

        if weight_bufs == 2:
            wslot = jnp.where(
                (t_idx > 0) & (is_new_e != 0),
                lax.bitwise_xor(cur_wslot, jnp.int32(1)),
                cur_wslot,
            )
            nxt_wslot = lax.bitwise_xor(wslot, jnp.int32(1))
        else:
            wslot = jnp.int32(0)
            nxt_wslot = jnp.int32(0)

        if gather_mode == "coop_bridge":
            gatherer = lax.bitwise_and(t_idx, jnp.int32(1))
            i_gather = (t_idx == 0) | (tp_half == gatherer)

            @pl.when(i_gather)
            def _do_coop_gather():
                @pl.loop(0, m_valid)
                def _g(r):
                    x_tile_3d[r] = x_3d_buf[packed_tok_ids[r_base + r]]

                @pl.when(t_idx > 0)
                def _fwd_x():
                    pl.semaphore_wait(x_fwd_free_sem, 1)
                    op = pltpu.make_async_remote_copy(
                        x_tile_3d.at[pl.ds(0, m_valid)],
                        x_tile_3d.at[pl.ds(0, m_valid)],
                        x_fwd_send_sem,
                        x_fwd_recv_sem,
                        device_id={"model": mate},
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    op.start()
                    op.wait_send()

            @pl.when(~i_gather)
            def _do_coop_recv():
                pl.semaphore_signal(
                    x_fwd_free_sem, 1, device_id={"model": mate}, device_id_type=pl.DeviceIdType.MESH
                )
                pltpu.make_async_copy(
                    x_tile_3d.at[pl.ds(0, m_valid)],
                    x_tile_3d.at[pl.ds(0, m_valid)],
                    x_fwd_recv_sem,
                ).wait()
        elif fast_gs:
            safe_base = jnp.minimum(r_base, jnp.int32(max_packed_len - 1))
            for s in range(8):
                safe_p = jnp.where(
                    jnp.int32(s) < m_valid,
                    r_base + jnp.int32(s),
                    safe_base,
                )
                tok_s = jnp.clip(packed_tok_ids[safe_p], jnp.int32(0), jnp.int32(batch_size - 1))
                x_tile_3d[s] = x_3d_buf[tok_s]

            @pl.when(m_valid > 8)
            def _g_vmem_rest():
                @pl.loop(8, m_valid)
                def _g_hi(r):
                    x_tile_3d[r] = x_3d_buf[packed_tok_ids[r_base + r]]
        else:
            @pl.loop(0, m_valid)
            def _g_vmem(r):
                x_tile_3d[r] = x_3d_buf[packed_tok_ids[r_base + r]]

        if weight_bufs == 2:
            @pl.when(is_new_e != 0)
            def _wait_cur_w():
                _wait_w1(wslot)
                _wait_w2(wslot)

            @pl.when((is_last_e != 0) & (nxt_slot >= 0))
            def _pf_next_both():
                _start_w1(nxt_wslot, nxt_src, nxt_slot)
                _start_w2(nxt_wslot, nxt_src, nxt_slot)
        else:
            @pl.when(is_new_e != 0)
            def _w_pingpong_start():
                _start_w2(0, tile_w_src[t_idx], tile_w_slot[t_idx])
                _wait_w1(0)
                _wait_w2(0)

        _drain_bridge_send(bslot)
        _run_bucketed_compute(bslot, wslot, m_valid)

        if weight_bufs == 1:
            @pl.when((is_last_e != 0) & (nxt_slot >= 0))
            def _pf_next_w1_1buf():
                _start_w1(0, nxt_src, nxt_slot)

        @pl.when(t_idx >= 2)
        def _wait_mate_free():
            pl.semaphore_wait(b_free_sem.at[bslot], 1)

        _os, _ol, send_start, send_len, _rl = _row_split(t_idx, m_valid)

        @pl.when(send_len > 0)
        def _do_bridge_send():
            pltpu.make_async_remote_copy(
                p_send.at[bslot, pl.ds(send_start, send_len)],
                p_recv.at[bslot, pl.ds(0, send_len)],
                b_send_sem.at[bslot],
                b_recv_sem.at[bslot],
                device_id={"model": mate},
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
            slot_send_len[bslot] = send_len

        @pl.when(t_idx > 0)
        def _finish_prev():
            lax.cond(
                t_idx + 1 < num_tiles,
                lambda: _finish_tile(t_idx - 1, signal_free=True),
                lambda: _finish_tile(t_idx - 1, signal_free=False),
            )

        return wslot

    lax.fori_loop(0, num_tiles, _tile_body, jnp.int32(0))

    @pl.when(num_tiles > 0)
    def _finish_last():
        _finish_tile(num_tiles - 1, signal_free=False)
        _drain_bridge_send(0)
        _drain_bridge_send(1)

    rows_per_rank = batch_size // world_size
    r_step = min(rows_per_rank, 4)

    if pair_rs:
        # Intra-chip XOR=1 FP32 swap-sum (733 GB/s die-to-die) -> halved ep-way ICI RS!
        half_b = batch_size // 2
        own_r0 = pl.multiple_of(
            jnp.where(tp_half == 0, jnp.int32(0), jnp.int32(half_b)), half_b
        )
        send_r0 = pl.multiple_of(
            jnp.where(tp_half == 0, jnp.int32(half_b), jnp.int32(0)), half_b
        )
        pl.semaphore_wait(barrier, world_size)
        pair_cp = pltpu.make_async_remote_copy(
            accum_buf.at[pl.ds(send_r0, half_b)],
            pair_recv,
            pair_send_sem,
            pair_recv_sem,
            device_id={"model": mate},
            device_id_type=pl.DeviceIdType.MESH,
        )
        pair_cp.start()
        pltpu.make_async_copy(pair_recv, pair_recv, pair_recv_sem).wait()
        p_step = min(half_b, 8)
        for r0 in range(0, half_b, p_step):
            own_sl = pl.ds(pl.multiple_of(own_r0 + r0, p_step), p_step)
            r_sl = pl.ds(r0, p_step)
            pair_sum_buf[r_sl] = (accum_buf[own_sl] + pair_recv[r_sl]).astype(rs_dt)
        pair_cp.wait_send()

        base_dest = tp_half * ep
        rs_copies = [
            pltpu.make_async_remote_copy(
                pair_sum_buf.at[pl.ds(c * rows_per_rank, rows_per_rank)],
                rs_recv.at[chip_id],
                rs_send_sem.at[c],
                rs_recv_sem,
                device_id={"model": base_dest + jnp.int32(c)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            for c in range(ep)
        ]
        for op in rs_copies:
            op.start()
        pltpu.make_async_copy(rs_recv, rs_recv, rs_recv_sem).wait()
        for r0 in range(0, rows_per_rank, r_step):
            r_sl = pl.ds(r0, r_step)
            tot = rs_recv[0, r_sl].astype(jnp.float32)
            for src_c in range(1, ep):
                tot = tot + rs_recv[src_c, r_sl].astype(jnp.float32)
            owner_buf[r_sl] = tot.astype(jnp.bfloat16)

        if return_sharded:
            output_ref[...] = owner_buf[...].reshape(rows_per_rank, 4096)
            for op in rs_copies:
                op.wait_send()
        else:
            fan_recv[me] = owner_buf[...]
            for dest in range(world_size):
                @pl.when(me != dest)
                def _send_fan_p(dest=dest):
                    pltpu.make_async_remote_copy(
                        owner_buf,
                        fan_recv.at[me],
                        fan_send_sem.at[dest],
                        fan_recv_sem,
                        device_id={"model": jnp.int32(dest)},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
            pltpu.make_async_copy(
                fan_recv.at[pl.ds(0, world_size - 1)],
                fan_recv.at[pl.ds(0, world_size - 1)],
                fan_recv_sem,
            ).wait()
            chunk_ranks = 1 if (rows_per_rank >= 8 and rows_per_rank % 8 == 0) else (8 // rows_per_rank)
            chunk_rows = chunk_ranks * rows_per_rank
            for r0 in range(0, world_size, chunk_ranks):
                output_ref[pl.ds(r0 * rows_per_rank, chunk_rows), :] = (
                    fan_recv[pl.ds(r0, chunk_ranks)].reshape(chunk_rows, 4096)
                )
            for dest in range(world_size):
                @pl.when(me != dest)
                def _ws_fan_p(dest=dest):
                    pltpu.make_async_remote_copy(
                        owner_buf,
                        fan_recv.at[me],
                        fan_send_sem.at[dest],
                        fan_recv_sem,
                        device_id={"model": jnp.int32(dest)},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait_send()
            for op in rs_copies:
                op.wait_send()
        return

    # Spill-free Reduce-Scatter skipping self-DMA (d != me) + 4-row chunked summation!
    if rs_wire_dtype_str == "bfloat16":
        c_step = 8 if (fast_gs and batch_size % 8 == 0) else 4
        for b0 in range(0, batch_size, c_step):
            rs_send_bf16[pl.ds(b0, c_step), :, :] = accum_buf[pl.ds(b0, c_step), :, :].astype(jnp.bfloat16)
        rs_src_buf = rs_send_bf16
    else:
        rs_src_buf = accum_buf

    pl.semaphore_wait(barrier, world_size)
    for d in range(world_size):
        @pl.when(me != d)
        def _send_rs(d=d):
            pltpu.make_async_remote_copy(
                rs_src_buf.at[pl.ds(d * rows_per_rank, rows_per_rank)],
                rs_recv.at[me],
                rs_send_sem.at[d],
                rs_recv_sem,
                device_id={"model": jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

    # Copy local slice into rs_recv[me] while remote peers stream into rs_recv[d != me]
    for r0 in range(0, rows_per_rank, r_step):
        rs_recv[me, pl.ds(r0, r_step)] = rs_src_buf[
            pl.ds(me * rows_per_rank + r0, r_step)
        ]

    # Wait for exactly (world_size - 1) incoming slices from peers != me!
    pltpu.make_async_copy(
        rs_recv.at[pl.ds(0, world_size - 1)],
        rs_recv.at[pl.ds(0, world_size - 1)],
        rs_recv_sem,
    ).wait()

    # Sum in 4-row chunks (<= 16 VREGs live -> 0 spills even at B=256!)
    for r0 in range(0, rows_per_rank, r_step):
        r_sl = pl.ds(r0, r_step)
        tot = rs_recv[0, r_sl].astype(jnp.float32)
        for src in range(1, world_size):
            tot = tot + rs_recv[src, r_sl].astype(jnp.float32)
        owner_buf[r_sl] = tot.astype(jnp.bfloat16)

    if return_sharded:
        output_ref[...] = owner_buf[...].reshape(rows_per_rank, 4096)
        for d in range(world_size):
            @pl.when(me != d)
            def _ws_rs(d=d):
                pltpu.make_async_remote_copy(
                    rs_src_buf.at[pl.ds(d * rows_per_rank, rows_per_rank)],
                    rs_recv.at[me],
                    rs_send_sem.at[d],
                    rs_recv_sem,
                    device_id={"model": jnp.int32(d)},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
    else:
        fan_recv[me] = owner_buf[...]
        for dest in range(world_size):
            @pl.when(me != dest)
            def _send_fan(dest=dest):
                pltpu.make_async_remote_copy(
                    owner_buf,
                    fan_recv.at[me],
                    fan_send_sem.at[dest],
                    fan_recv_sem,
                    device_id={"model": jnp.int32(dest)},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
        pltpu.make_async_copy(
            fan_recv.at[pl.ds(0, world_size - 1)],
            fan_recv.at[pl.ds(0, world_size - 1)],
            fan_recv_sem,
        ).wait()
        chunk_ranks = 1 if (rows_per_rank >= 8 and rows_per_rank % 8 == 0) else (8 // rows_per_rank)
        chunk_rows = chunk_ranks * rows_per_rank
        for r0 in range(0, world_size, chunk_ranks):
            output_ref[pl.ds(r0 * rows_per_rank, chunk_rows), :] = (
                fan_recv[pl.ds(r0, chunk_ranks)].reshape(chunk_rows, 4096)
            )
        for dest in range(world_size):
            @pl.when(me != dest)
            def _ws_fan(dest=dest):
                pltpu.make_async_remote_copy(
                    owner_buf,
                    fan_recv.at[me],
                    fan_send_sem.at[dest],
                    fan_recv_sem,
                    device_id={"model": jnp.int32(dest)},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
        for d in range(world_size):
            @pl.when(me != d)
            def _ws_rs2(d=d):
                pltpu.make_async_remote_copy(
                    rs_src_buf.at[pl.ds(d * rows_per_rank, rows_per_rank)],
                    rs_recv.at[me],
                    rs_send_sem.at[d],
                    rs_recv_sem,
                    device_id={"model": jnp.int32(d)},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()


def _auto_tile_m(batch_size: int) -> int:
    if batch_size <= 16:
        return 16
    if batch_size <= 64:
        return 32
    return 64


def fused_decode_ep_tp2_from_logits(
    rank,
    logits,
    bias,
    x,
    w1,
    s1,
    w2,
    s2,
    norm,
    rep_w1=None,
    rep_s1=None,
    rep_w2=None,
    rep_s2=None,
    replica_meta=None,
    *,
    ep: int = 4,
    tp: int = 2,
    chipmate_xor: int | None = None,
    return_sharded: bool = False,
    replica_cost_mode: str = "active",
    tile_m: int | None = None,
    strip_nc: int | None = None,
    bridge_dtype: str = "float32",
    gather_mode: str = "vmem",
    weight_bufs: int = 2,
    use_blockdiag_k16: bool | None = None,
    bd_w4a8: bool = False,
    rs_wire_dtype: str | None = None,
    row_mode: str = "auto",
    bucket_policy: str = "2bucket",
    min_p_topk: float | None = None,
    free_max_rows: int | None = None,
    free_max_w: float | None = None,
    free_min_slot: int | None = None,
    static_holder_mask: bool | None = None,
    route_fast: bool | None = None,
    fast_plan: int | None = None,
    unroll_strips: bool | None = None,
    fast_gs: bool | None = None,
    pair_rs: bool | None = None,
):
    """Symmetric EP(P/2)xTP2 Fuse-All decode MoE from pre-scaled FP32 logits."""
    if tp != 2 or ep not in (4, 8):
        raise ValueError(f"Symmetric EPxTP2 requires ep in (4, 8) and tp == 2, got ep={ep}, tp={tp}")
    xor = _resolve_chipmate_xor(chipmate_xor)
    world_size = ep * tp
    b, h = x.shape
    e_chip = 256 // ep
    i_half = 2048
    if h != 4096 or logits.shape != (b, 256) or bias.shape != (1, 256):
        raise ValueError(
            f"Expected x[B,4096], logits[B,256], bias[1,256]; got x={x.shape}, logits={logits.shape}, bias={bias.shape}"
        )
    if b <= 0 or b > 256 or (b % world_size != 0) or (b % 16 != 0):
        raise ValueError(
            f"B must be in (16..256), divisible by 16 and world_size={world_size}; got B={b}"
        )
    if norm.shape != (1, 4096):
        raise ValueError(f"Expected norm shape (1, 4096), got {norm.shape}")
    block = h // s1.shape[1]
    if block not in (16, 256, 512):
        raise ValueError(
            f"Unsupported FP4 scale block size {block}; expected 16, 256, or 512"
        )
    if (
        w1.shape != (e_chip, h, 2 * i_half)
        or w2.shape != (e_chip, i_half, h)
        or w1.dtype != jnp.float4_e2m1fn
        or w2.dtype != jnp.float4_e2m1fn
        or s1.shape != (e_chip, h // block, 2 * i_half)
        or s2.shape != (e_chip, i_half // block, h)
    ):
        raise ValueError(
            f"EP{ep}xTP2 K{block} weight shape mismatch: "
            f"w1={w1.shape}/{w1.dtype}, s1={s1.shape}, w2={w2.shape}/{w2.dtype}, s2={s2.shape}"
        )

    rep_args = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    has_replicas = any(a is not None for a in rep_args)
    num_replicas = 0
    if has_replicas:
        if not all(a is not None for a in rep_args):
            raise ValueError("All 5 replica arguments must be provided together or all None")
        num_replicas = int(rep_w1.shape[0])
        if (
            num_replicas <= 0
            or rep_w1.shape != (num_replicas, h, 2 * i_half)
            or rep_s1.shape != (num_replicas, h // block, 2 * i_half)
            or rep_w2.shape != (num_replicas, i_half, h)
            or rep_s2.shape != (num_replicas, i_half // block, h)
            or replica_meta.shape != (4, ep * num_replicas)
            or replica_meta.dtype != jnp.int32
        ):
            raise ValueError(
                f"Replica tensor shape mismatch: rep_w1={rep_w1.shape}, replica_meta={replica_meta.shape}"
            )

    (
        env_min_p,
        env_free_rows,
        env_free_w,
        env_free_slot,
        env_sm,
        env_rfast,
        env_fplan,
        env_unroll,
        env_fgs,
        env_pair_rs,
        env_br64,
    ) = _read_ep_tp2_env_knobs(b)
    eff_min_p = env_min_p if min_p_topk is None else float(min_p_topk)
    eff_free_rows = env_free_rows if free_max_rows is None else int(free_max_rows)
    eff_free_w = env_free_w if free_max_w is None else float(free_max_w)
    eff_free_slot = env_free_slot if free_min_slot is None else int(free_min_slot)
    eff_sm = env_sm if static_holder_mask is None else bool(static_holder_mask)
    eff_rfast = env_rfast if route_fast is None else bool(route_fast)
    eff_fplan = env_fplan if fast_plan is None else int(fast_plan)
    eff_unroll = env_unroll if unroll_strips is None else bool(unroll_strips)
    eff_fgs = env_fgs if fast_gs is None else bool(fast_gs)
    eff_pair_rs = env_pair_rs if pair_rs is None else bool(pair_rs)

    if bridge_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Invalid bridge_dtype: {bridge_dtype}")
    if rs_wire_dtype is None:
        rs_wire_dtype = "bfloat16" if (b >= 128 or (env_br64 and b == 64)) else "float32"
    if rs_wire_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Invalid rs_wire_dtype: {rs_wire_dtype}")
    if gather_mode not in ("vmem", "coop_bridge"):
        raise ValueError(f"Invalid gather_mode: {gather_mode}")
    if weight_bufs not in (1, 2):
        raise ValueError(f"Invalid weight_bufs: {weight_bufs}")
    if row_mode not in ("auto", "dense", "sparse"):
        raise ValueError(f"Invalid row_mode: {row_mode}")
    if bucket_policy not in ("2bucket", "4bucket"):
        raise ValueError(f"Invalid bucket_policy: {bucket_policy}")

    has_w9_route_pruning = (
        eff_min_p > 0.0
        or eff_free_rows > 0
        or eff_rfast
        or eff_fplan > 0
        or eff_fgs
        or eff_pair_rs
        or _bridge1_prerouted_active()
    )
    if row_mode == "auto":
        use_dense = (
            gather_mode == "vmem"
            and tile_m is None
            and block in (256, 512)
            and b == 16
            and not has_w9_route_pruning
        )
    else:
        use_dense = row_mode == "dense"
    # 2D bfloat16 bridge requires half_b % 16 == 0 (B >= 32); at B == 16 use 3D sparse path for bf16 bridge
    if use_dense and bridge_dtype == "bfloat16" and (b // 2) < 16:
        use_dense = False

    bd_k16 = (block == 16) if use_blockdiag_k16 is None else bool(use_blockdiag_k16)
    wire_dt = jnp.float32 if bridge_dtype == "float32" else jnp.bfloat16
    rs_dt = jnp.float32 if rs_wire_dtype == "float32" else jnp.bfloat16

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    sem_dma = pltpu.SemaphoreType.DMA
    sem_reg = pltpu.SemaphoreType.REGULAR
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_exec = e_chip + num_replicas
    rows_per_rank = b // world_size

    out_shape = (
        jax.ShapeDtypeStruct((rows_per_rank, h), jnp.bfloat16)
        if return_sharded
        else jax.ShapeDtypeStruct((b, h), jnp.bfloat16)
    )
    in_specs = (v_spec, v_spec, v_spec, h_spec, h_spec, h_spec, h_spec, v_spec)
    call_args = [rank]
    if has_replicas:
        call_args.append(replica_meta)
    call_args.extend([logits, bias, x, w1, s1, w2, s2, norm])
    if has_replicas:
        in_specs = in_specs + (h_spec, h_spec, h_spec, h_spec)
        call_args.extend([rep_w1, rep_s1, rep_w2, rep_s2])

    if use_dense:
        half_b = b // 2
        scratch_dense: list = [
            vmem((weight_bufs, h // 8, 2 * i_half), jnp.uint32),
            vmem((weight_bufs, h // block, 2 * i_half), s1.dtype),
            vmem((weight_bufs, i_half // 8, h), jnp.uint32),
            vmem((weight_bufs, i_half // block, h), s2.dtype),
            sem_dma((weight_bufs,)),
            sem_dma((weight_bufs,)),
            vmem((b, i_half), jnp.bfloat16),
            vmem((b, 256), jnp.float32),
            vmem((half_b, h), jnp.float32),
            vmem((half_b, 32, 128), jnp.float32),
            vmem((2, b, h), wire_dt),
            vmem((2, half_b, h), wire_dt),
            sem_dma((2,)),
            sem_dma((2,)),
            sem_reg((2,)),
            sem_reg,
            vmem((8, 256), jnp.int32),
            smem((8, 256), jnp.int32),
            sem_dma,
            smem((max_exec,), jnp.int32),
            smem((max_exec,), jnp.int32),
            smem((max_exec,), jnp.int32),
        ]
        if has_replicas:
            scratch_dense.extend([
                smem((e_chip,), jnp.int32),
                smem((num_replicas,), jnp.int32),
                smem((ep,), jnp.int32),
                smem((256,), jnp.int32),
            ])
        scratch_dense.extend([
            vmem((ep, rows_per_rank, 32, 128), rs_dt),
            sem_dma((ep,)),
            sem_dma,
            vmem((rows_per_rank, 32, 128), jnp.bfloat16),
        ])
        if rs_wire_dtype == "bfloat16":
            scratch_dense.append(vmem((half_b, 32, 128), jnp.bfloat16))
        if not return_sharded:
            scratch_dense.extend([
                vmem((world_size, rows_per_rank, 32, 128), jnp.bfloat16),
                sem_dma((world_size,)),
                sem_dma,
            ])

        kname = (
            f"muse_ep{ep}tp2_dense_b{b}_k{block}"
            f"{f'_r{num_replicas}' if has_replicas else ''}"
            f"{'_bf16br' if bridge_dtype == 'bfloat16' else ''}"
            f"{'_bd' if bd_k16 else ''}"
            f"{'_sh' if return_sharded else '_fan'}"
        )
        return pl.pallas_call(
            functools.partial(
                _ep_tp2_dense_kernel,
                ep=ep,
                tp=tp,
                batch_size=b,
                block=block,
                chipmate_xor=xor,
                has_replicas=has_replicas,
                num_replicas=num_replicas,
                replica_cost_mode=replica_cost_mode,
                return_sharded=return_sharded,
                strip_nc=strip_nc,
                bridge_dtype_str=bridge_dtype,
                weight_bufs=weight_bufs,
                use_blockdiag_k16=bd_k16,
                rs_wire_dtype_str=rs_wire_dtype,
            ),
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2 if has_replicas else 1,
                grid=(1,),
                in_specs=in_specs,
                out_specs=v_spec,
                scratch_shapes=tuple(scratch_dense),
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=64 * 1024**2,
                collective_id=0,
            ),
            name=kname,
        )(*call_args)

    # Sparse TP2 path
    tm = _auto_tile_m(b) if tile_m is None else int(tile_m)
    tm = min(tm, b)
    max_nt = (b + tm - 1) // tm
    max_tiles = max_exec * max_nt + max_nt + 4
    accum_rows = (b + 1) if eff_fgs else b

    scratch_shapes: list = [
        vmem((weight_bufs, h // 8, 2 * i_half), jnp.uint32),
        vmem((weight_bufs, h // block, 2 * i_half), s1.dtype),
        vmem((weight_bufs, i_half // 8, h), jnp.uint32),
        vmem((weight_bufs, i_half // block, h), s2.dtype),
        sem_dma((weight_bufs,)),
        sem_dma((weight_bufs,)),
        vmem((b, 32, 128), jnp.bfloat16),
        vmem((tm, 32, 128), jnp.bfloat16),
        vmem((tm, h), jnp.bfloat16),
        vmem((tm, i_half), jnp.bfloat16),
        vmem((32, 128), jnp.float32),
        vmem((accum_rows, 32, 128), jnp.float32),
        vmem((2, tm, 32, 128), wire_dt),
        vmem((2, tm, 32, 128), wire_dt),
        sem_dma((2,)),
        sem_dma((2,)),
        sem_reg((2,)),
        sem_reg,
        smem((2,), jnp.int32),
    ]
    if gather_mode == "coop_bridge":
        scratch_shapes.extend([sem_dma, sem_dma, sem_reg])

    r_shapes = routing_scratch_shapes(b, max_exec)
    scratch_shapes.extend(list(r_shapes))
    scratch_shapes.append(sem_dma)
    scratch_shapes.extend([
        smem((max_tiles,), jnp.int32),
        smem((max_tiles,), jnp.int32),
        smem((max_tiles,), jnp.int32),
        smem((max_tiles,), jnp.int32),
        smem((max_tiles,), jnp.int32),
        smem((max_tiles,), jnp.int32),
        smem((max_tiles,), jnp.int32),
        smem((max_tiles,), jnp.int32),
    ])
    if has_replicas:
        moved_len = 8 if eff_sm else 256
        scratch_shapes.extend([
            smem((e_chip,), jnp.int32),
            smem((num_replicas,), jnp.int32),
            smem((ep,), jnp.int32),
            smem((moved_len,), jnp.int32),
        ])
        if not eff_sm:
            scratch_shapes.append(smem((num_replicas,), jnp.int32))

    if eff_pair_rs:
        half_b = b // 2
        scratch_shapes.extend([
            vmem((half_b, 32, 128), jnp.float32),
            vmem((half_b, 32, 128), rs_dt),
            sem_dma,
            sem_dma,
            vmem((ep, rows_per_rank, 32, 128), rs_dt),
            sem_dma((ep,)),
            sem_dma,
            vmem((rows_per_rank, 32, 128), jnp.bfloat16),
        ])
    else:
        scratch_shapes.extend([
            vmem((world_size, rows_per_rank, 32, 128), rs_dt),
            sem_dma((world_size,)),
            sem_dma,
            vmem((rows_per_rank, 32, 128), jnp.bfloat16),
        ])
        if rs_wire_dtype == "bfloat16":
            scratch_shapes.append(vmem((b, 32, 128), jnp.bfloat16))
    if not return_sharded:
        scratch_shapes.extend([
            vmem((world_size, rows_per_rank, 32, 128), jnp.bfloat16),
            sem_dma((world_size,)),
            sem_dma,
        ])
    if bd_k16 and block == 16:
        bd_x_dt = jnp.float8_e4m3fn if bd_w4a8 else jnp.bfloat16
        scratch_shapes.append(vmem((128, h), bd_x_dt))

    kname = (
        f"muse_ep{ep}tp2_b{b}_k{block}_tm{tm}_{bucket_policy}"
        f"{f'_r{num_replicas}' if has_replicas else ''}"
        f"{'_rfast' if eff_rfast else ''}"
        f"{f'_fp{eff_fplan}' if eff_fplan else ''}"
        f"{'_fgs' if eff_fgs else ''}"
        f"{'_prs' if eff_pair_rs else ''}"
        f"{'_bf16br' if bridge_dtype == 'bfloat16' else ''}"
        f"{'_bf16rs' if rs_wire_dtype == 'bfloat16' else ''}"
        f"{'_coop' if gather_mode == 'coop_bridge' else ''}"
        f"{'_bd8a8' if (bd_k16 and bd_w4a8) else '_bd8' if bd_k16 else ''}"
        f"{'_sh' if return_sharded else '_fan'}"
    )
    return pl.pallas_call(
        functools.partial(
            _ep_tp2_sparse_kernel,
            ep=ep,
            tp=tp,
            batch_size=b,
            block=block,
            chipmate_xor=xor,
            has_replicas=has_replicas,
            num_replicas=num_replicas,
            replica_cost_mode=replica_cost_mode,
            return_sharded=return_sharded,
            tile_m=tm,
            strip_nc=strip_nc,
            bridge_dtype_str=bridge_dtype,
            gather_mode=gather_mode,
            weight_bufs=weight_bufs,
            use_blockdiag_k16=bd_k16,
            bd_w4a8=bool(bd_w4a8),
            rs_wire_dtype_str=rs_wire_dtype,
            bucket_policy=bucket_policy,
            min_p_topk=eff_min_p,
            free_max_rows=eff_free_rows,
            free_max_w=eff_free_w,
            free_min_slot=eff_free_slot,
            static_holder_mask=eff_sm,
            route_fast=eff_rfast,
            fast_plan=eff_fplan,
            unroll_strips=eff_unroll,
            fast_gs=eff_fgs,
            pair_rs=eff_pair_rs,
        ),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2 if has_replicas else 1,
            grid=(1,),
            in_specs=in_specs,
            out_specs=v_spec,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=kname,
    )(*call_args)
