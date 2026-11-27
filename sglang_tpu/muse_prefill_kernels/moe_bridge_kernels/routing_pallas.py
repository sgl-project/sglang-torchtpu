# SPDX-License-Identifier: Apache-2.0
"""Zero-sort Pallas VMEM+MXU routing kernels for Muse Spark prefill MoE."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128


def pallas_local_inv_perm_and_counts(
    local_topk_ids: jax.Array,
    global_topk_ids: jax.Array,
    go_val: jax.Array,
    my_id: jax.Array,
    *,
    ep_size: int,
    num_local_experts: int,
    total_id_buckets: int,
    has_pad_sentinel: bool = False,
    zero_local_start: bool = True,
    native_pack: bool | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Single-op Pallas VMEM+MXU kernel replacing local argsorts and 3D count broadcasts.

    Computes bit-exact:
      1. `inv_perm_c`: `int32[t_local, topk]` == `argsort(argsort(local_topk_ids.reshape(-1), stable=True))`
         (including duplicate sentinel IDs on padded rows).
      2. `group_sizes`: `int32[ep_size * num_local_experts]` (populated at `[go_val : go_val + num_local_experts]`).
      3. `group_start_arg`: `int32[2 + ep_size + num_local_experts * ep_size]` closed-form block-scatter header.
      4. `t_recv_val`: `int32[]` scalar incoming remote row count for this rank.
    """
    t_local, topk = local_topk_ids.shape
    t_global, g_topk = global_topk_ids.shape
    num_experts = ep_size * num_local_experts
    assert topk == g_topk and t_global == ep_size * t_local
    assert t_local % LANE == 0 and (t_local * topk) % LANE == 0
    assert num_local_experts <= LANE and ep_size <= LANE

    b_strips = (total_id_buckets + LANE - 1) // LANE
    b_pad = b_strips * LANE
    n_tok_blks = t_local // LANE
    g_rows_per_chip = (t_local * topk) // LANE

    if native_pack is None:
        from sglang_tpu.envs import envs as _tpu_envs

        use_native_pack = bool(
            topk == 8
            and _tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK.get()
        )
    else:
        use_native_pack = bool(native_pack and topk == 8)

    if use_native_pack:
        local_in = local_topk_ids.astype(jnp.int32).reshape(
            n_tok_blks * 8, LANE
        )
    else:
        local_in = jnp.pad(
            local_topk_ids.astype(jnp.int32), ((0, 0), (0, LANE - topk))
        )
    global_2d = global_topk_ids.astype(jnp.int32).reshape(
        ep_size * g_rows_per_chip, LANE
    )
    meta_sp = jnp.stack(
        [
            go_val.reshape(-1)[0].astype(jnp.int32),
            my_id.reshape(-1)[0].astype(jnp.int32),
        ]
    )
    hdr_len = 2 + ep_size + num_local_experts * ep_size
    hdr_strips = (hdr_len + LANE - 1) // LANE
    hdr_pad_len = hdr_strips * LANE
    gs_strips = (num_experts + LANE - 1) // LANE
    gs_pad_len = gs_strips * LANE

    def _kernel(
        meta_ref,
        local_pad_ref,
        global_ref,
        inv_pad_out_ref,
        gs_pad_out_ref,
        hdr_pad_out_ref,
        oh_vmem_ref,
        blk_sum_vmem_ref,
    ):
        go = meta_ref[0]
        me = meta_ref[1]
        lane_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 1)
        row_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 0)
        l_128 = (row_128 > lane_128).astype(jnp.bfloat16)
        u_128 = (row_128 < lane_128).astype(jnp.bfloat16)
        if use_native_pack:
            sub_8 = lax.broadcasted_iota(jnp.int32, (8, LANE), 0)
            base_lane = (row_128 & jnp.int32(15)) * jnp.int32(8)
            slot_masks = [lane_128 == (base_lane + k) for k in range(topk)]

            def _load_cols_k(b_idx: int):
                blk_8 = local_pad_ref[pl.ds(pl.multiple_of(b_idx * 8, 8), 8), :]
                rows_128 = jnp.concatenate(
                    [
                        jnp.broadcast_to(
                            jnp.sum(
                                jnp.where(sub_8 == r, blk_8, 0),
                                axis=0,
                                keepdims=True,
                            ),
                            (16, LANE),
                        )
                        for r in range(8)
                    ],
                    axis=0,
                )
                return [
                    jnp.sum(
                        jnp.where(slot_masks[k], rows_128, 0),
                        axis=1,
                        keepdims=True,
                    )
                    for k in range(topk)
                ]
        else:
            slot_masks = None

            def _load_cols_k(b_idx: int):
                ids_blk = local_pad_ref[pl.ds(b_idx * LANE, LANE), :]
                return [
                    jnp.sum(
                        jnp.where(lane_128 == k, ids_blk, 0),
                        axis=1,
                        keepdims=True,
                    )
                    for k in range(topk)
                ]

        # Part A: Exact VMEM+MXU counting-sort rank for `inv_perm_c`.
        if use_native_pack:
            init_cnts = tuple(
                jnp.zeros((8, LANE), dtype=jnp.int32) for _ in range(b_strips)
            )

            def _pass1_np(b, cnts_acc):
                cols_k = _load_cols_k(b)
                r0 = pl.multiple_of(b * LANE, LANE)
                new_cnts = []
                for s in range(b_strips):
                    iota_s = s * LANE + lane_128
                    oh_bs = (cols_k[0] == iota_s).astype(jnp.bfloat16)
                    for k in range(1, topk):
                        oh_bs = oh_bs + (cols_k[k] == iota_s).astype(jnp.bfloat16)
                    oh_vmem_ref[pl.ds(r0, LANE), pl.ds(s * LANE, LANE)] = oh_bs
                    b_cnt = jnp.broadcast_to(
                        jnp.sum(oh_bs.astype(jnp.int32), axis=0, keepdims=True),
                        (8, LANE),
                    )
                    new_cnts.append(cnts_acc[s] + b_cnt)
                return tuple(new_cnts)

            tot_cnts = lax.fori_loop(0, n_tok_blks, _pass1_np, init_cnts)
            starts_strips = []
            carry_total = jnp.zeros((8, 1), dtype=jnp.int32)
            for s in range(b_strips):
                cnt_s = tot_cnts[s]
                lo7 = (cnt_s & jnp.int32(127)).astype(jnp.bfloat16)
                hi7 = lax.shift_right_logical(cnt_s, jnp.int32(7)).astype(
                    jnp.bfloat16
                )
                p_lo = jnp.matmul(
                    lo7, u_128, preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                p_hi = jnp.matmul(
                    hi7, u_128, preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                starts_strips.append(
                    lax.shift_left(p_hi, jnp.int32(7)) + p_lo + carry_total
                )
                carry_total = carry_total + jnp.max(
                    jnp.sum(cnt_s, axis=1, keepdims=True), axis=0, keepdims=True
                )

            def _pass2_np(b, run_strips):
                cols_k = _load_cols_k(b)
                r0 = pl.multiple_of(b * LANE, LANE)
                rank_strips = []
                next_run = []
                for s in range(b_strips):
                    oh_bs = oh_vmem_ref[pl.ds(r0, LANE), pl.ds(s * LANE, LANE)]
                    intra_s = jnp.matmul(
                        l_128, oh_bs, preferred_element_type=jnp.float32
                    ).astype(jnp.int32)
                    run_s = run_strips[s]
                    rank_strips.append(
                        (
                            intra_s.reshape(LANE // 8, 8, LANE) + run_s[None, :, :]
                        ).reshape(LANE, LANE)
                    )
                    b_cnt = jnp.broadcast_to(
                        jnp.sum(oh_bs.astype(jnp.int32), axis=0, keepdims=True),
                        (8, LANE),
                    )
                    next_run.append(run_s + b_cnt)
                out_blk = jnp.zeros((LANE, LANE), dtype=jnp.int32)
                for k in range(topk):
                    ck = cols_k[k]
                    val_k = jnp.zeros((LANE, 1), dtype=jnp.int32)
                    for s in range(b_strips):
                        iota_s = s * LANE + lane_128
                        m_ks = ck == iota_s
                        val_k = val_k + jnp.sum(
                            jnp.where(m_ks, rank_strips[s], 0),
                            axis=1,
                            keepdims=True,
                        )
                        rank_strips[s] = jnp.where(
                            m_ks, rank_strips[s] + 1, rank_strips[s]
                        )
                    out_blk = jnp.where(
                        slot_masks[k],
                        jnp.broadcast_to(val_k, (LANE, LANE)),
                        out_blk,
                    )
                inv_pad_out_ref[pl.ds(pl.multiple_of(b * 8, 8), 8), :] = jnp.sum(
                    out_blk.reshape(8, 16, LANE), axis=1
                )
                return tuple(next_run)

            lax.fori_loop(0, n_tok_blks, _pass2_np, tuple(starts_strips))
        else:
            for b in range(n_tok_blks):
                r_sl = pl.ds(b * LANE, LANE)
                cols_k = _load_cols_k(b)
                for s in range(b_strips):
                    c_sl = pl.ds(s * LANE, LANE)
                    iota_s = s * LANE + lane_128
                    oh_bs = (cols_k[0] == iota_s).astype(jnp.bfloat16)
                    for k in range(1, topk):
                        oh_bs = oh_bs + (cols_k[k] == iota_s).astype(jnp.bfloat16)
                    oh_vmem_ref[r_sl, c_sl] = oh_bs
                    b_cnt = jnp.sum(oh_bs.astype(jnp.int32), axis=0, keepdims=True)
                    blk_sum_vmem_ref[pl.ds(b * 8, 8), c_sl] = jnp.broadcast_to(
                        b_cnt, (8, LANE)
                    )

            starts_strips = []
            carry_total = jnp.zeros((8, 1), dtype=jnp.int32)
            for s in range(b_strips):
                c_sl = pl.ds(s * LANE, LANE)
                cnt_s = blk_sum_vmem_ref[pl.ds(0, 8), c_sl]
                for b in range(1, n_tok_blks):
                    cnt_s = cnt_s + blk_sum_vmem_ref[pl.ds(b * 8, 8), c_sl]
                lo7 = (cnt_s & jnp.int32(127)).astype(jnp.bfloat16)
                hi7 = lax.shift_right_logical(cnt_s, jnp.int32(7)).astype(
                    jnp.bfloat16
                )
                p_lo = jnp.matmul(
                    lo7, u_128, preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                p_hi = jnp.matmul(
                    hi7, u_128, preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                excl_s = lax.shift_left(p_hi, jnp.int32(7)) + p_lo + carry_total
                starts_strips.append(excl_s)
                carry_total = carry_total + jnp.max(
                    jnp.sum(cnt_s, axis=1, keepdims=True), axis=0, keepdims=True
                )

            running_strips = list(starts_strips)
            for b in range(n_tok_blks):
                r_sl = pl.ds(b * LANE, LANE)
                cols_k = _load_cols_k(b)
                rank_strips = []
                for s in range(b_strips):
                    c_sl = pl.ds(s * LANE, LANE)
                    oh_bs = oh_vmem_ref[r_sl, c_sl]
                    intra_s = jnp.matmul(
                        l_128, oh_bs, preferred_element_type=jnp.float32
                    ).astype(jnp.int32)
                    run_s = running_strips[s]
                    rank_strips.append(
                        (
                            intra_s.reshape(LANE // 8, 8, LANE)
                            + run_s[None, :, :]
                        ).reshape(LANE, LANE)
                    )
                    running_strips[s] = (
                        run_s + blk_sum_vmem_ref[pl.ds(b * 8, 8), c_sl]
                    )

                out_blk = jnp.zeros((LANE, LANE), dtype=jnp.int32)
                for k in range(topk):
                    ck = cols_k[k]
                    val_k = jnp.zeros((LANE, 1), dtype=jnp.int32)
                    for s in range(b_strips):
                        iota_s = s * LANE + lane_128
                        m_ks = ck == iota_s
                        val_k = val_k + jnp.sum(
                            jnp.where(m_ks, rank_strips[s], 0),
                            axis=1,
                            keepdims=True,
                        )
                        rank_strips[s] = jnp.where(
                            m_ks, rank_strips[s] + 1, rank_strips[s]
                        )
                    out_blk = jnp.where(
                        lane_128 == k,
                        jnp.broadcast_to(val_k, (LANE, LANE)),
                        out_blk,
                    )
                inv_pad_out_ref[r_sl, :] = out_blk

        # Part B: Exact `chip_before_local`, `local_chip_counts`, `group_sizes`, `group_start_arg`.
        # Accumulate across `ep_size` inside `lax.fori_loop` rather than unrolling
        # `ep_size * num_local_experts` scalar reductions in one basic block, which triggers
        # TPU7x TensorCore scalar-program halts at `t_local <= 256`.
        lane_1d = lax.broadcasted_iota(jnp.int32, (8, LANE), 1)
        init_h = tuple(jnp.zeros((8, LANE), dtype=jnp.int32) for _ in range(hdr_strips))
        init_lgs = tuple(jnp.int32(0) for _ in range(num_local_experts))
        cb_hs_lo = 2 // LANE
        cb_hs_hi = (2 + ep_size - 1) // LANE

        def _chip_step(c, carry):
            hs_tup, lgs_tup, act_my, my_h, bef_tot = carry
            hs_l = list(hs_tup)
            lgs_l = list(lgs_tup)
            r0_c = (
                pl.multiple_of(c * g_rows_per_chip, 8)
                if g_rows_per_chip % 8 == 0
                else c * g_rows_per_chip
            )
            g_chip = global_ref[pl.ds(r0_c, g_rows_per_chip), :]
            bef_c = jnp.sum((g_chip < go).astype(jnp.int32))
            if has_pad_sentinel:
                act_c = jnp.sum((g_chip < jnp.int32(num_experts)).astype(jnp.int32))
                act_my = jnp.where(c == me, act_c, act_my)
            if not zero_local_start:
                bef_tot = bef_tot + bef_c
            idx_cb = jnp.int32(2) + c
            for hs in range(cb_hs_lo, cb_hs_hi + 1):
                hs_l[hs] = jnp.where(
                    lane_1d == (idx_cb - jnp.int32(hs * LANE)), bef_c, hs_l[hs]
                )
            rel_c = g_chip - go
            c_tot = jnp.int32(0)
            for e_loc in range(num_local_experts):
                cnt_ce = jnp.sum((rel_c == e_loc).astype(jnp.int32))
                c_tot = c_tot + cnt_ce
                lgs_l[e_loc] = lgs_l[e_loc] + cnt_ce
                f_base = 2 + ep_size + e_loc * ep_size
                f_idx = jnp.int32(f_base) + c
                for hs in range(f_base // LANE, (f_base + ep_size - 1) // LANE + 1):
                    hs_l[hs] = jnp.where(
                        lane_1d == (f_idx - jnp.int32(hs * LANE)), cnt_ce, hs_l[hs]
                    )
            my_h = jnp.where(c == me, c_tot, my_h)
            return tuple(hs_l), tuple(lgs_l), act_my, my_h, bef_tot

        hs_fin, local_gs, active_my_slots, my_hits, bef_tot_fin = lax.fori_loop(
            0,
            ep_size,
            _chip_step,
            (init_h, init_lgs, jnp.int32(0), jnp.int32(0), jnp.int32(0)),
        )

        for gs_s in range(gs_strips):
            rel_g = (gs_s * LANE + lane_1d) - go
            val_gs = jnp.zeros((8, LANE), dtype=jnp.int32)
            for e_loc in range(num_local_experts):
                val_gs = jnp.where(rel_g == e_loc, local_gs[e_loc], val_gs)
            gs_pad_out_ref[pl.ds(0, 8), pl.ds(gs_s * LANE, LANE)] = val_gs

        local_start = jnp.int32(0) if zero_local_start else bef_tot_fin
        t_recv = (
            (active_my_slots - my_hits)
            if has_pad_sentinel
            else (jnp.int32(t_local * topk) - my_hits)
        )
        h_strips = list(hs_fin)
        h_strips[0] = jnp.where(lane_1d == 0, local_start, h_strips[0])
        h_strips[0] = jnp.where(lane_1d == 1, t_recv, h_strips[0])
        for hs in range(hdr_strips):
            hdr_pad_out_ref[pl.ds(0, 8), pl.ds(hs * LANE, LANE)] = h_strips[hs]

    v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    inv_out_rows = (n_tok_blks * 8) if use_native_pack else t_local
    inv_raw, gs_pad, hdr_pad = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((inv_out_rows, LANE), jnp.int32),
            jax.ShapeDtypeStruct((8, gs_pad_len), jnp.int32),
            jax.ShapeDtypeStruct((8, hdr_pad_len), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[v_spec, v_spec],
            out_specs=[v_spec, v_spec, v_spec],
            scratch_shapes=[
                pltpu.VMEM((t_local, b_pad), jnp.bfloat16),
                pltpu.VMEM((n_tok_blks * 8, b_pad), jnp.int32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=(
                int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.94)
                if t_global > 16384
                else 32 * 1024 * 1024
            ),
        ),
        interpret=interpret,
        name=(
            f"pallas_moe_sortfree_route{'_np' if use_native_pack else ''}"
            f"_tl{t_local}_el{num_local_experts}_b{b_pad}"
            f"{'_pad' if has_pad_sentinel else ''}"
        ),
    )(meta_sp, local_in, global_2d)

    inv_perm_c = (
        inv_raw.reshape(t_local, topk) if use_native_pack else inv_raw[:, :topk]
    )
    group_sizes = gs_pad[0, :num_experts]
    group_start_arg = hdr_pad[0, :hdr_len]
    t_recv_val = group_start_arg[1]
    return inv_perm_c, group_sizes, group_start_arg, t_recv_val


def pallas_global_local_pos(
    global_topk_ids: jax.Array,
    go_val: jax.Array,
    *,
    num_local_experts: int,
    ep_size: int = 8,
    rotate_rs: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """VMEM+MXU bucket-rank kernel computing sorted slot positions `[T_global, topk]` with 0 XLA sorts.

    Maps local experts `[go_val, go_val + num_local_experts)` to contiguous destination slots
    `[0, local_total)` in `(expert, token, slot)` order (or sender-rotated chip order when
    `rotate_rs=True`) and non-local entries to `n_flat`.
    """
    t_global, topk = global_topk_ids.shape
    n_flat = t_global * topk
    assert t_global % LANE == 0 and num_local_experts < LANE

    n_blks = t_global // LANE
    blks_per_chip = max(1, n_blks // ep_size)
    use_rot = bool(rotate_rs and ep_size > 1 and (n_blks % ep_size == 0))
    g_pad = jnp.pad(global_topk_ids.astype(jnp.int32), ((0, 0), (0, LANE - topk)))
    go_sp = go_val.reshape(1).astype(jnp.int32)

    def _pos_kernel(go_ref, g_pad_ref, pos_pad_out_ref, blk_sum_vmem_ref):
        go = go_ref[0]
        me = lax.div(go, jnp.int32(num_local_experts))
        lane_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 1)
        row_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 0)
        l_128 = (row_128 > lane_128).astype(jnp.bfloat16)
        u_128 = (row_128 < lane_128).astype(jnp.bfloat16)

        # Pass 1: per-128-token block counts across local buckets `0 .. num_local_experts - 1`
        # (non-local mapped to bucket `num_local_experts`).
        def _pass1(b, tot_acc):
            r_sl = pl.ds(b * LANE, LANE)
            ids_blk = g_pad_ref[r_sl, :] - go
            oh_b = jnp.zeros((LANE, LANE), dtype=jnp.bfloat16)
            for k in range(topk):
                ck = jnp.sum(jnp.where(lane_128 == k, ids_blk, 0), axis=1, keepdims=True)
                bkt_k = jnp.where(
                    (ck >= 0) & (ck < num_local_experts),
                    ck,
                    jnp.int32(num_local_experts),
                )
                oh_b = oh_b + (bkt_k == lane_128).astype(jnp.bfloat16)
            b_cnt = jnp.broadcast_to(
                jnp.sum(oh_b.astype(jnp.int32), axis=0, keepdims=True), (8, LANE)
            )
            blk_sum_vmem_ref[pl.ds(b * 8, 8), :] = b_cnt
            return tot_acc + b_cnt

        lane_8 = lax.broadcasted_iota(jnp.int32, (8, LANE), 1)
        tot_cnt = lax.fori_loop(
            0, n_blks, _pass1, jnp.zeros((8, LANE), dtype=jnp.int32)
        )
        # Zero out non-local bucket before computing exclusive bucket starts.
        tot_local_only = jnp.where(lane_8 < num_local_experts, tot_cnt, 0)
        lo7 = (tot_local_only & jnp.int32(127)).astype(jnp.bfloat16)
        hi7 = lax.shift_right_logical(tot_local_only, jnp.int32(7)).astype(jnp.bfloat16)
        p_lo = jnp.matmul(lo7, u_128, preferred_element_type=jnp.float32).astype(
            jnp.int32
        )
        p_hi = jnp.matmul(hi7, u_128, preferred_element_type=jnp.float32).astype(
            jnp.int32
        )
        starts_init = lax.shift_left(p_hi, jnp.int32(7)) + p_lo

        # Pass 2: intra-block MXU prefix + slot output.
        def _pass2(b, running):
            phys_b = (
                jnp.mod(lax.div(b, jnp.int32(blks_per_chip)) + me + 1, ep_size)
                * jnp.int32(blks_per_chip)
                + lax.rem(b, jnp.int32(blks_per_chip))
                if use_rot
                else b
            )
            r_sl = pl.ds(phys_b * LANE, LANE)
            ids_blk = g_pad_ref[r_sl, :] - go
            bkts_k = []
            oh_b = jnp.zeros((LANE, LANE), dtype=jnp.bfloat16)
            for k in range(topk):
                ck = jnp.sum(jnp.where(lane_128 == k, ids_blk, 0), axis=1, keepdims=True)
                bkt_k = jnp.where(
                    (ck >= 0) & (ck < num_local_experts),
                    ck,
                    jnp.int32(num_local_experts),
                )
                bkts_k.append(bkt_k)
                oh_b = oh_b + (bkt_k == lane_128).astype(jnp.bfloat16)
            intra_b = jnp.matmul(
                l_128, oh_b, preferred_element_type=jnp.float32
            ).astype(jnp.int32)
            rank_b = (
                intra_b.reshape(LANE // 8, 8, LANE) + running[None, :, :]
            ).reshape(LANE, LANE)
            out_b = jnp.full((LANE, LANE), jnp.int32(n_flat), dtype=jnp.int32)
            for k in range(topk):
                bk = bkts_k[k]
                m_k = bk == lane_128
                pos_k = jnp.sum(jnp.where(m_k, rank_b, 0), axis=1, keepdims=True)
                pos_k = jnp.where(bk < num_local_experts, pos_k, jnp.int32(n_flat))
                rank_b = jnp.where(m_k, rank_b + 1, rank_b)
                out_b = jnp.where(
                    lane_128 == k, jnp.broadcast_to(pos_k, (LANE, LANE)), out_b
                )
            pos_pad_out_ref[r_sl, :] = out_b
            return running + blk_sum_vmem_ref[pl.ds(phys_b * 8, 8), :]

        lax.fori_loop(0, n_blks, _pass2, starts_init)

    v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    pos_pad = pl.pallas_call(
        _pos_kernel,
        out_shape=jax.ShapeDtypeStruct((t_global, LANE), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[v_spec],
            out_specs=v_spec,
            scratch_shapes=[pltpu.VMEM((n_blks * 8, LANE), jnp.int32)],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=(
                int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.94)
                if t_global > 16384
                else 32 * 1024 * 1024
            ),
        ),
        interpret=interpret,
        name=(
            f"pallas_moe_global_pos_tg{t_global}_el{num_local_experts}"
            f"{'_rrs' if use_rot else ''}"
        ),
    )(go_sp, g_pad)
    return pos_pad[:, :topk]


def resolve_qblock_params(
    total_elems: int, ep_size: int, quantum: int
) -> tuple[int, int]:
    """Return ``(n_blocks, eff_quantum)`` shared by Pallas and XLA ``q_block`` water-filling."""
    eff_q = min(128, int(quantum)) if int(quantum) > 128 else max(1, int(quantum))
    if total_elems >= 64 * 1024 and total_elems % 1024 == 0:
        chunks_8row = total_elems // 1024
        # Prefer coarse 48/64 macro-aligned blocks (e.g. T=24K: n_blocks=48, 4x32-row tiles
        # in each 128x128 macro-tile) when blks_per_macro is a power of 2 in [2, 16]:
        # cuts Pallas q_block steps 192->12 while improving gmm_v2 tile balance, without
        # altering non-macro shapes such as T=36K (chunks_8row=288 -> n_blocks=96).
        if total_elems % (LANE * LANE) == 0:
            n_macro = total_elems // (LANE * LANE)
            for cand in (48, 64):
                if (
                    chunks_8row % cand == 0
                    and cand % n_macro == 0
                    and (cand // n_macro) in (2, 4, 8, 16)
                ):
                    return cand, eff_q
        if chunks_8row <= 256:
            return chunks_8row, eff_q
        for cand in (256, 192, 128, 96, 64, 32):
            if chunks_8row % cand == 0:
                return cand, eff_q
    n_blocks = 64 if (total_elems >= 64 and total_elems % 64 == 0) else ep_size
    return n_blocks, eff_q


def pallas_qblock_remap_virtual_ids(
    global_topk_ids: jax.Array,
    my_id: jax.Array,
    *,
    ep_size: int = 8,
    num_local_experts: int,
    num_base_local: int = 32,
    topk: int = 8,
    quantum: int = 128,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Single-op Pallas VMEM+MXU kernel for `q_block` Zero-Glue EPLB virtual-ID remapping.

    Eliminates ~45 unfused XLA HLO elementwise/reduction ops with 0-ULP bit-exact parity
    against `remap_zeroglue_virtual_ids(..., mode='q_block')`.
    Supports `EP=8/16` (`num_base_local in {16, 32}`), vectorizing `(128, 128)` macro-tiles
    (`12` steps at `T=24K` instead of `192` `8`-row steps) whenever macro-aligned.
    """
    num_replicas = num_local_experts - num_base_local
    t_global = global_topk_ids.shape[0]
    t_local = t_global // ep_size
    total_elems = t_global * topk
    n_blocks, q_eff = resolve_qblock_params(total_elems, ep_size, quantum)
    blk_elems = total_elems // n_blocks
    assert ep_size in (8, 16) and num_base_local in (16, 32) and 1 <= num_replicas <= 16
    assert n_blocks <= 128 or num_replicas <= 12
    assert total_elems % (n_blocks * 512) == 0
    base_shift = num_base_local.bit_length() - 1
    base_mask = num_base_local - 1
    blks_per_step = 2 if blk_elems == 512 else 1
    assert (blks_per_step * blk_elems) % (8 * LANE) == 0
    assert not (blks_per_step == 2 and n_blocks > 128)
    use_two_halves = n_blocks > 128
    n_steps = n_blocks // blks_per_step
    rows_per_step = (blks_per_step * blk_elems) // LANE
    total_rows = total_elems // LANE
    rows_per_rank = (t_local * topk) // LANE
    assert rows_per_rank % 8 == 0

    use_macro_vec = (
        total_rows % LANE == 0
        and n_blocks <= 256
        and n_blocks % (total_rows // LANE) == 0
        and (n_blocks // (total_rows // LANE)) in (2, 4, 8, 16)
    )
    n_macro = total_rows // LANE if use_macro_vec else 0
    blks_per_macro = (n_blocks // n_macro) if use_macro_vec else 0
    rows_per_blk = (LANE // blks_per_macro) if use_macro_vec else 0
    row_shift = (rows_per_blk.bit_length() - 1) if use_macro_vec else 0
    blk_mask = (blks_per_macro - 1) if use_macro_vec else 0

    ids_2d = global_topk_ids.astype(jnp.int32).reshape(total_rows, LANE)
    meta_sp = jnp.asarray([my_id.reshape(-1)[0].astype(jnp.int32)], dtype=jnp.int32)

    def _qblock_kernel(meta_ref, ids_ref, v_global_out_ref, v_local_out_ref):
        me = meta_ref[0]
        lane_8 = lax.broadcasted_iota(jnp.int32, (8, LANE), 1)
        row_8 = lax.broadcasted_iota(jnp.int32, (8, LANE), 0)
        lane_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 1)
        row_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 0)
        u_128 = (row_128 < lane_128).astype(jnp.bfloat16)

        if use_two_halves:
            init_carry = (jnp.zeros((8, LANE), dtype=jnp.int32),) + tuple(
                jnp.zeros((8, LANE), dtype=jnp.int32) for _ in range(2 * num_replicas)
            )
        else:
            init_carry = (jnp.zeros((8, LANE), dtype=jnp.int32),) + tuple(
                jnp.zeros((8, LANE), dtype=jnp.int32) for _ in range(num_replicas)
            )

        if use_macro_vec:
            p_right = (
                lax.shift_right_logical(row_128, jnp.int32(row_shift))
                == (lane_128 & jnp.int32(blk_mask))
            ).astype(jnp.bfloat16)

            def _pass1(m, carry):
                base_acc = carry[0]
                r0 = pl.multiple_of(m * LANE, LANE)
                x = ids_ref[pl.ds(r0, LANE), :]
                own = lax.shift_right_logical(x, jnp.int32(base_shift))
                is_non_rep = x >= jnp.int32(num_replicas)
                for r in range(ep_size):
                    cnt_r = jnp.sum((is_non_rep & (own == r)).astype(jnp.int32))
                    base_acc = jnp.where(lane_8 == r, base_acc + cnt_r, base_acc)

                blk_base = m * jnp.int32(blks_per_macro)
                mask_lo = (lane_8 >= blk_base) & (
                    lane_8 < blk_base + jnp.int32(blks_per_macro)
                )
                if use_two_halves:
                    bcs_lo = list(carry[1 : 1 + num_replicas])
                    bcs_hi = list(carry[1 + num_replicas :])
                    mask_hi = ((lane_8 + jnp.int32(128)) >= blk_base) & (
                        (lane_8 + jnp.int32(128))
                        < blk_base + jnp.int32(blks_per_macro)
                    )
                    for k in range(num_replicas):
                        row_k = jnp.sum((x == k).astype(jnp.int32), axis=1, keepdims=True)
                        row_t = jnp.broadcast_to(
                            row_k.astype(jnp.bfloat16), (LANE, LANE)
                        ).T[:8, :]
                        cnt_8 = jnp.matmul(
                            row_t, p_right, preferred_element_type=jnp.float32
                        ).astype(jnp.int32)
                        bcs_lo[k] = jnp.where(mask_lo, cnt_8, bcs_lo[k])
                        bcs_hi[k] = jnp.where(mask_hi, cnt_8, bcs_hi[k])
                    return (base_acc, *bcs_lo, *bcs_hi)

                bcs = list(carry[1:])
                for k in range(num_replicas):
                    row_k = jnp.sum((x == k).astype(jnp.int32), axis=1, keepdims=True)
                    row_t = jnp.broadcast_to(
                        row_k.astype(jnp.bfloat16), (LANE, LANE)
                    ).T[:8, :]
                    cnt_8 = jnp.matmul(
                        row_t, p_right, preferred_element_type=jnp.float32
                    ).astype(jnp.int32)
                    bcs[k] = jnp.where(mask_lo, cnt_8, bcs[k])
                return (base_acc, *bcs)

            done_carry = lax.fori_loop(0, n_macro, _pass1, init_carry)
        else:

            def _pass1(st, carry):
                base_acc = carry[0]
                r0 = pl.multiple_of(st * rows_per_step, 8)
                x = ids_ref[pl.ds(r0, rows_per_step), :]
                own = lax.shift_right_logical(x, jnp.int32(base_shift))
                is_non_rep = x >= jnp.int32(num_replicas)
                for r in range(ep_size):
                    cnt_r = jnp.sum((is_non_rep & (own == r)).astype(jnp.int32))
                    base_acc = jnp.where(lane_8 == r, base_acc + cnt_r, base_acc)
                if use_two_halves:
                    bcs_lo = list(carry[1 : 1 + num_replicas])
                    bcs_hi = list(carry[1 + num_replicas :])
                    for k in range(num_replicas):
                        hk = jnp.sum((x == k).astype(jnp.int32))
                        bcs_lo[k] = jnp.where(lane_8 == st, hk, bcs_lo[k])
                        bcs_hi[k] = jnp.where(
                            (lane_8 + jnp.int32(128)) == st, hk, bcs_hi[k]
                        )
                    return (base_acc, *bcs_lo, *bcs_hi)
                bcs = list(carry[1:])
                for k in range(num_replicas):
                    mx = (x == k).astype(jnp.int32)
                    if blks_per_step == 1:
                        hk = jnp.sum(mx)
                        bcs[k] = jnp.where(lane_8 == st, hk, bcs[k])
                    else:
                        hk0 = jnp.sum(jnp.where(row_8 < 4, mx, 0))
                        hk1 = jnp.sum(jnp.where(row_8 >= 4, mx, 0))
                        bcs[k] = jnp.where(lane_8 == (2 * st), hk0, bcs[k])
                        bcs[k] = jnp.where(lane_8 == (2 * st + 1), hk1, bcs[k])
                return (base_acc, *bcs)

            done_carry = lax.fori_loop(0, n_steps, _pass1, init_carry)

        base_acc = done_carry[0]

        base_r = [
            jnp.max(jnp.sum(jnp.where(lane_8 == r, base_acc, 0), axis=1))
            for r in range(ep_size)
        ]
        cur_t = [
            lax.div(base_r[r] + jnp.int32(q_eff - 1), jnp.int32(q_eff))
            for r in range(ep_size)
        ]

        def _excl_prefix_128(vec):
            lo7 = (vec & jnp.int32(127)).astype(jnp.bfloat16)
            hi7 = lax.shift_right_logical(vec, jnp.int32(7)).astype(jnp.bfloat16)
            p_lo = jnp.matmul(lo7, u_128, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            p_hi = jnp.matmul(hi7, u_128, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            return lax.shift_left(p_hi, jnp.int32(7)) + p_lo

        vk_vecs_lo = []
        vk_vecs_hi = []
        for k in range(num_replicas):
            if use_two_halves:
                bc_lo = done_carry[1 + k]
                bc_hi = done_carry[1 + num_replicas + k]
                tot_lo = jnp.max(jnp.sum(bc_lo, axis=1))
                tot_hi = jnp.max(jnp.sum(bc_hi, axis=1))
                cnt_k = tot_lo + tot_hi
            else:
                bc_k = done_carry[1 + k]
                cnt_k = jnp.max(jnp.sum(bc_k, axis=1))
            n_blks_k = jnp.where(
                cnt_k > 0,
                lax.div(cnt_k + jnp.int32(q_eff - 1), jnp.int32(q_eff)),
                jnp.int32(0),
            )
            ord_r = [
                sum(
                    (
                        (cur_t[j] < cur_t[i])
                        | ((cur_t[j] == cur_t[i]) & jnp.bool_(j < i))
                    ).astype(jnp.int32)
                    for j in range(ep_size)
                )
                for i in range(ep_size)
            ]
            s_r = [
                sum(
                    jnp.where(ord_r[i] == pos, cur_t[i], jnp.int32(0))
                    for i in range(ep_size)
                )
                for pos in range(ep_size)
            ]
            cs_r = []
            acc_s = jnp.int32(0)
            for pos in range(ep_size):
                acc_s = acc_s + s_r[pos]
                cs_r.append(acc_s)
            costs = [
                jnp.int32(pos + 1) * s_r[pos] - cs_r[pos] for pos in range(ep_size)
            ]
            k_act = sum((costs[pos] <= n_blks_k).astype(jnp.int32) for pos in range(ep_size))
            sub_sum = sum(
                jnp.where(k_act == jnp.int32(pos + 1), cs_r[pos], jnp.int32(0))
                for pos in range(ep_size)
            )
            tot_act = sub_sum + n_blks_k
            base_lvl = lax.div(tot_act, k_act)
            rem_t = tot_act - base_lvl * k_act
            is_act = [(n_blks_k > 0) & (ord_r[i] < k_act) for i in range(ep_size)]
            rq = [
                jnp.where(is_act[i], base_lvl - cur_t[i], jnp.int32(0))
                for i in range(ep_size)
            ]
            cum_tq = []
            acc_tq = jnp.int32(0)
            act_pref = jnp.int32(0)
            for i in range(ep_size):
                tq_i = rq[i] + (is_act[i] & (act_pref < rem_t)).astype(jnp.int32)
                act_pref = act_pref + is_act[i].astype(jnp.int32)
                acc_tq = acc_tq + tq_i
                cum_tq.append(acc_tq)
                cur_t[i] = cur_t[i] + tq_i

            def _route_blocks(bc_vec, cum_off):
                cum_blk = _excl_prefix_128(bc_vec) + cum_off
                mid_row = cum_blk + lax.shift_right_logical(bc_vec, jnp.int32(1))
                blk_tile = lax.div(mid_row, jnp.int32(q_eff))
                rk_vec = sum(
                    (blk_tile >= cum_tq[r]).astype(jnp.int32)
                    for r in range(ep_size - 1)
                )
                return rk_vec * jnp.int32(num_local_experts) + jnp.int32(
                    num_base_local + k
                )

            if use_two_halves:
                vk_vecs_lo.append(_route_blocks(bc_lo, jnp.int32(0)))
                vk_vecs_hi.append(_route_blocks(bc_hi, tot_lo))
            else:
                vk_vecs_lo.append(_route_blocks(bc_k, jnp.int32(0)))

        if use_macro_vec:
            blk_sub = lax.shift_right_logical(row_128, jnp.int32(row_shift))

            def _pass2(m, dummy):
                r0 = pl.multiple_of(m * LANE, LANE)
                x = ids_ref[pl.ds(r0, LANE), :]
                v = lax.shift_right_logical(x, jnp.int32(base_shift)) * jnp.int32(
                    num_local_experts
                ) + (x & jnp.int32(base_mask))
                blk_of_row = m * jnp.int32(blks_per_macro) + blk_sub
                for k in range(num_replicas):
                    v_lo_128 = jnp.broadcast_to(vk_vecs_lo[k][:1, :], (LANE, LANE))
                    if use_two_halves:
                        v_hi_128 = jnp.broadcast_to(vk_vecs_hi[k][:1, :], (LANE, LANE))
                        sel_lo = lane_128 == blk_of_row
                        sel_hi = (lane_128 + jnp.int32(128)) == blk_of_row
                        vk_row = jnp.sum(
                            jnp.where(sel_lo, v_lo_128, 0)
                            + jnp.where(sel_hi, v_hi_128, 0),
                            axis=1,
                            keepdims=True,
                        )
                    else:
                        vk_row = jnp.sum(
                            jnp.where(lane_128 == blk_of_row, v_lo_128, 0),
                            axis=1,
                            keepdims=True,
                        )
                    v = jnp.where(x == k, vk_row, v)
                v_global_out_ref[pl.ds(r0, LANE), :] = v
                return dummy

            lax.fori_loop(0, n_macro, _pass2, jnp.int32(0))
        else:

            def _pass2(st, dummy):
                r0 = pl.multiple_of(st * rows_per_step, 8)
                x = ids_ref[pl.ds(r0, rows_per_step), :]
                v = lax.shift_right_logical(x, jnp.int32(base_shift)) * jnp.int32(
                    num_local_experts
                ) + (x & jnp.int32(base_mask))
                for k in range(num_replicas):
                    if use_two_halves:
                        vk_b = jnp.max(
                            jnp.sum(
                                jnp.where(lane_8 == st, vk_vecs_lo[k], 0)
                                + jnp.where(
                                    (lane_8 + jnp.int32(128)) == st,
                                    vk_vecs_hi[k],
                                    0,
                                ),
                                axis=1,
                            )
                        )
                        v = jnp.where(x == k, vk_b, v)
                    elif blks_per_step == 1:
                        vk_b = jnp.max(
                            jnp.sum(jnp.where(lane_8 == st, vk_vecs_lo[k], 0), axis=1)
                        )
                        v = jnp.where(x == k, vk_b, v)
                    else:
                        vk_b0 = jnp.max(
                            jnp.sum(
                                jnp.where(lane_8 == (2 * st), vk_vecs_lo[k], 0), axis=1
                            )
                        )
                        vk_b1 = jnp.max(
                            jnp.sum(
                                jnp.where(lane_8 == (2 * st + 1), vk_vecs_lo[k], 0),
                                axis=1,
                            )
                        )
                        vk_row = jnp.where(row_8 < 4, vk_b0, vk_b1)
                        v = jnp.where(x == k, vk_row, v)
                v_global_out_ref[pl.ds(r0, rows_per_step), :] = v
                return dummy

            lax.fori_loop(0, n_steps, _pass2, jnp.int32(0))

        my_r0 = pl.multiple_of(me * rows_per_rank, 8)
        v_local_out_ref[:, :] = v_global_out_ref[pl.ds(my_r0, rows_per_rank), :]

    if t_global > 16384:
        try:
            vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.94)
        except Exception:
            vmem_limit_bytes = 60 * 1024 * 1024
    else:
        vmem_limit_bytes = 32 * 1024 * 1024

    v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    v_glob_2d, v_loc_2d = pl.pallas_call(
        _qblock_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((total_rows, LANE), jnp.int32),
            jax.ShapeDtypeStruct((rows_per_rank, LANE), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[v_spec],
            out_specs=[v_spec, v_spec],
            scratch_shapes=[],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
        name=f"pallas_qblock_eplb_remap_tg{t_global}_ep{ep_size}_el{num_local_experts}_nb{n_blocks}_q{q_eff}",
    )(meta_sp, ids_2d)

    return v_glob_2d.reshape(global_topk_ids.shape), v_loc_2d.reshape(t_local, topk)


def pallas_compact_local_packed_keys(
    global_topk_ids: jax.Array,
    go_val: jax.Array,
    my_id: jax.Array,
    *,
    num_local_experts: int,
    ep_size: int = 8,
    rotate_rs: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """2D `(128, 128)` VMEM+MXU local-expert key compactor with lossless overflow tiering.

    Compacts active `(rel_expert << tok_bits) | sort_tok` keys (`rel_expert in [0, num_local_experts)`)
    from `global_topk_ids` (`[T_global, 8]`, `N_flat = T_global * 8`) into power-of-2 buffers:
      - `out32`: `int32[32768]` (`2^15`, Tier 0 when `tier == 0`),
      - `out_mid`: `int32[65536]` (`2^16` for `T_global <= 24576`, else `int32[131072]` = `2^17`,
        Tier 1 when `tier == 1`),
      - `tier`: scalar `int32[]` in `{0, 1, 2}` (`2` triggers exact full-batch `jnp.sort` fallback
        so arbitrary worst-case skew up to `100%` single-expert load never drops a token).

    Inactive / padding slots are filled with `sentinel = num_local_experts << tok_bits`, which
    sorts strictly after all local keys while satisfying `(sentinel & tok_mask) == 0` so trailing
    decoded indices are always in-bounds (`0`).
    """
    t_global, topk = global_topk_ids.shape
    n_flat = t_global * topk
    assert topk == 8 and n_flat >= 65536 and n_flat % (LANE * LANE) == 0
    assert 0 < num_local_experts <= 120

    n_macro = n_flat // (LANE * LANE)
    out_mid_rows = 512 if n_macro <= 12 else 1024
    r_keep1 = min(64, (out_mid_rows // n_macro) & ~7)
    assert r_keep1 >= 24 and r_keep1 % 8 == 0
    n_stage2_blks = out_mid_rows // LANE
    out32_rows = 256
    r_keep2 = out32_rows // n_stage2_blks
    assert r_keep2 % 8 == 0

    tok_bits = max(1, (t_global - 1).bit_length())
    chunk_size = t_global // ep_size
    sentinel_py = int(num_local_experts) << tok_bits

    g_2d = global_topk_ids.astype(jnp.int32).reshape(n_macro * LANE, LANE)
    meta_sp = jnp.stack(
        [
            go_val.reshape(-1)[0].astype(jnp.int32),
            my_id.reshape(-1)[0].astype(jnp.int32),
        ]
    )

    def _compact_128x128(val, sentinel_v, lane_128, row_128, u_128, l_128):
        is_act = val < sentinel_v
        dst_col = jnp.matmul(
            is_act.astype(jnp.bfloat16), u_128, preferred_element_type=jnp.float32
        ).astype(jnp.int32)
        delta = jnp.where(is_act, lane_128 - dst_col, jnp.int32(0))
        for sh in (1, 2, 4, 8, 16, 32, 64):
            r_val = jnp.roll(val, -sh, axis=1)
            r_delta = jnp.roll(delta, -sh, axis=1)
            move = (delta & sh) != 0
            stay = (val < sentinel_v) & (~move)
            r_move = ((r_delta & sh) != 0) & (lane_128 < (LANE - sh))
            val = jnp.where(r_move, r_val, jnp.where(stay, val, sentinel_v))
            delta = jnp.where(r_move, r_delta, jnp.where(stay, delta, 0))
        rot_amt = (row_128 * jnp.int32(17)) & jnp.int32(127)
        for sh in (1, 2, 4, 8, 16, 32, 64):
            val = jnp.where((rot_amt & sh) != 0, jnp.roll(val, sh, axis=1), val)
        is_act2 = val < sentinel_v
        dst_row = jnp.matmul(
            l_128, is_act2.astype(jnp.bfloat16), preferred_element_type=jnp.float32
        ).astype(jnp.int32)
        delta_r = jnp.where(is_act2, row_128 - dst_row, jnp.int32(0))
        for sh in (1, 2, 4, 8, 16, 32, 64):
            r_val = jnp.roll(val, -sh, axis=0)
            r_delta = jnp.roll(delta_r, -sh, axis=0)
            move = (delta_r & sh) != 0
            stay = (val < sentinel_v) & (~move)
            r_move = ((r_delta & sh) != 0) & (row_128 < (LANE - sh))
            val = jnp.where(r_move, r_val, jnp.where(stay, val, sentinel_v))
            delta_r = jnp.where(r_move, r_delta, jnp.where(stay, delta_r, 0))
        return val

    def _kernel(meta_ref, g_ref, out32_ref, out_mid_ref, flag_out_ref):
        sentinel = jnp.int32(sentinel_py)
        go = meta_ref[0]
        me = meta_ref[1]
        lane_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 1)
        row_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 0)
        u_128 = (row_128 < lane_128).astype(jnp.bfloat16)
        l_128 = (row_128 > lane_128).astype(jnp.bfloat16)
        chunk_base_tok = (
            lax.shift_right_logical(row_128, jnp.int32(3)) * jnp.int32(n_macro * 128)
            + (row_128 & jnp.int32(7)) * jnp.int32(16)
            + lax.shift_right_logical(lane_128, jnp.int32(3))
        )

        used_mid_rows = n_macro * r_keep1
        if used_mid_rows < out_mid_rows:
            out_mid_ref[pl.ds(used_mid_rows, out_mid_rows - used_mid_rows), :] = (
                jnp.full(
                    (out_mid_rows - used_mid_rows, LANE),
                    sentinel,
                    dtype=jnp.int32,
                )
            )

        def _stage1_body(m, ovf1):
            # Sample 16 strided 8-row slices across the global batch so replicated/hot
            # experts cannot cluster inside a single Stage-1 tile or Stage-2 quarter.
            ids_blk = jnp.concatenate(
                [
                    g_ref[pl.ds(pl.multiple_of((j * n_macro + m) * 8, 8), 8), :]
                    for j in range(16)
                ],
                axis=0,
            )
            rel = ids_blk - go
            is_loc = (rel >= 0) & (rel < jnp.int32(num_local_experts))
            tok_id = m * jnp.int32(128) + chunk_base_tok
            if rotate_rs:
                chip_r = jnp.mod(
                    lax.div(tok_id, jnp.int32(chunk_size)) + (ep_size - 1) - me,
                    ep_size,
                )
                sort_tok = chip_r * jnp.int32(chunk_size) + lax.rem(
                    tok_id, jnp.int32(chunk_size)
                )
            else:
                sort_tok = tok_id
            packed_blk = jnp.where(
                is_loc,
                lax.shift_left(rel, jnp.int32(tok_bits)) | sort_tok,
                sentinel,
            )
            comp = _compact_128x128(
                packed_blk, sentinel, lane_128, row_128, u_128, l_128
            )
            w0 = pl.multiple_of(m * r_keep1, 8)
            out_mid_ref[pl.ds(w0, r_keep1), :] = comp[:r_keep1, :]
            hit_ovf = jnp.any(comp[r_keep1 : r_keep1 + 8, :] < sentinel).astype(
                jnp.int32
            )
            return ovf1 | hit_ovf

        ovf1_any = lax.fori_loop(0, n_macro, _stage1_body, jnp.int32(0))

        def _stage2_body(b, ovf2):
            r0 = pl.multiple_of(b * LANE, LANE)
            blk = out_mid_ref[pl.ds(r0, LANE), :]
            comp2 = _compact_128x128(
                blk, sentinel, lane_128, row_128, u_128, l_128
            )
            w0 = pl.multiple_of(b * r_keep2, 8)
            out32_ref[pl.ds(w0, r_keep2), :] = comp2[:r_keep2, :]
            hit_ovf2 = jnp.any(
                comp2[r_keep2 : r_keep2 + 8, :] < sentinel
            ).astype(jnp.int32)
            return ovf2 | hit_ovf2

        ovf2_any = lax.fori_loop(0, n_stage2_blks, _stage2_body, jnp.int32(0))
        tier = jnp.where(
            ovf1_any != 0,
            jnp.int32(2),
            jnp.where(ovf2_any != 0, jnp.int32(1), jnp.int32(0)),
        )
        flag_out_ref[:, :] = jnp.broadcast_to(tier, (8, LANE))

    v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    out32, out_mid, flag_pad = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((out32_rows, LANE), jnp.int32),
            jax.ShapeDtypeStruct((out_mid_rows, LANE), jnp.int32),
            jax.ShapeDtypeStruct((8, LANE), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[v_spec],
            out_specs=[v_spec, v_spec, v_spec],
            scratch_shapes=[],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=16 * 1024 * 1024,
        ),
        interpret=interpret,
        name=(
            f"pallas_compact_local_keys_tg{t_global}_el{num_local_experts}"
            f"{'_rrs' if rotate_rs else ''}"
        ),
    )(meta_sp, g_2d)
    return out32.reshape(-1), out_mid.reshape(-1), flag_pad[0, 0]
