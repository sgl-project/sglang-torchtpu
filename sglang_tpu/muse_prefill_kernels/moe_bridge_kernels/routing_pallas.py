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

    local_pad = jnp.pad(
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

        # Part A: Exact VMEM+MXU counting-sort rank for `inv_perm_c`.
        for b in range(n_tok_blks):
            r_sl = pl.ds(b * LANE, LANE)
            ids_blk = local_pad_ref[r_sl, :]
            cols_k = [
                jnp.sum(jnp.where(lane_128 == k, ids_blk, 0), axis=1, keepdims=True)
                for k in range(topk)
            ]
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
            hi7 = lax.shift_right_logical(cnt_s, jnp.int32(7)).astype(jnp.bfloat16)
            p_lo = jnp.matmul(lo7, u_128, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            p_hi = jnp.matmul(hi7, u_128, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            excl_s = lax.shift_left(p_hi, jnp.int32(7)) + p_lo + carry_total
            starts_strips.append(excl_s)
            carry_total = carry_total + jnp.max(
                jnp.sum(cnt_s, axis=1, keepdims=True), axis=0, keepdims=True
            )

        running_strips = list(starts_strips)
        for b in range(n_tok_blks):
            r_sl = pl.ds(b * LANE, LANE)
            ids_blk = local_pad_ref[r_sl, :]
            cols_k = [
                jnp.sum(jnp.where(lane_128 == k, ids_blk, 0), axis=1, keepdims=True)
                for k in range(topk)
            ]
            rank_strips = []
            for s in range(b_strips):
                c_sl = pl.ds(s * LANE, LANE)
                oh_bs = oh_vmem_ref[r_sl, c_sl]
                intra_s = jnp.matmul(
                    l_128, oh_bs, preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                run_s = running_strips[s]
                rank_strips.append(
                    (intra_s.reshape(LANE // 8, 8, LANE) + run_s[None, :, :]).reshape(
                        LANE, LANE
                    )
                )
                running_strips[s] = run_s + blk_sum_vmem_ref[pl.ds(b * 8, 8), c_sl]

            out_blk = jnp.zeros((LANE, LANE), dtype=jnp.int32)
            for k in range(topk):
                ck = cols_k[k]
                val_k = jnp.zeros((LANE, 1), dtype=jnp.int32)
                for s in range(b_strips):
                    iota_s = s * LANE + lane_128
                    m_ks = ck == iota_s
                    val_k = val_k + jnp.sum(
                        jnp.where(m_ks, rank_strips[s], 0), axis=1, keepdims=True
                    )
                    rank_strips[s] = jnp.where(m_ks, rank_strips[s] + 1, rank_strips[s])
                out_blk = jnp.where(
                    lane_128 == k, jnp.broadcast_to(val_k, (LANE, LANE)), out_blk
                )
            inv_pad_out_ref[r_sl, :] = out_blk

        # Part B: Exact `chip_before_local`, `local_chip_counts`, `group_sizes`, `group_start_arg`.
        lane_1d = lax.broadcasted_iota(jnp.int32, (8, LANE), 1)
        bef_vals = []
        ce_vals = []
        active_my_slots = jnp.int32(0)
        my_hits = jnp.int32(0)

        for c in range(ep_size):
            g_chip = global_ref[pl.ds(c * g_rows_per_chip, g_rows_per_chip), :]
            bef_vals.append(jnp.sum((g_chip < go).astype(jnp.int32)))
            if has_pad_sentinel:
                act_c = jnp.sum((g_chip < jnp.int32(num_experts)).astype(jnp.int32))
                active_my_slots = jnp.where(c == me, act_c, active_my_slots)
            rel_c = g_chip - go
            row_ce = []
            c_tot = jnp.int32(0)
            for e_loc in range(num_local_experts):
                cnt_ce = jnp.sum((rel_c == e_loc).astype(jnp.int32))
                row_ce.append(cnt_ce)
                c_tot = c_tot + cnt_ce
            ce_vals.append(row_ce)
            my_hits = jnp.where(c == me, c_tot, my_hits)

        local_gs = [
            sum(ce_vals[c][e_loc] for c in range(ep_size))
            for e_loc in range(num_local_experts)
        ]
        for gs_s in range(gs_strips):
            rel_g = (gs_s * LANE + lane_1d) - go
            val_gs = jnp.zeros((8, LANE), dtype=jnp.int32)
            for e_loc in range(num_local_experts):
                val_gs = jnp.where(rel_g == e_loc, local_gs[e_loc], val_gs)
            gs_pad_out_ref[pl.ds(0, 8), pl.ds(gs_s * LANE, LANE)] = val_gs

        local_start = (
            jnp.int32(0) if zero_local_start else sum(bef_vals)
        )
        t_recv = (
            (active_my_slots - my_hits)
            if has_pad_sentinel
            else (jnp.int32(t_local * topk) - my_hits)
        )

        h_strips = [jnp.zeros((8, LANE), dtype=jnp.int32) for _ in range(hdr_strips)]
        h_strips[0] = jnp.where(lane_1d == 0, local_start, h_strips[0])
        h_strips[0] = jnp.where(lane_1d == 1, t_recv, h_strips[0])
        for c in range(ep_size):
            idx_cb = 2 + c
            h_strips[idx_cb // LANE] = jnp.where(
                lane_1d == (idx_cb % LANE), bef_vals[c], h_strips[idx_cb // LANE]
            )
            for e_loc in range(num_local_experts):
                f_idx = 2 + ep_size + e_loc * ep_size + c
                hs = f_idx // LANE
                h_strips[hs] = jnp.where(
                    lane_1d == (f_idx % LANE), ce_vals[c][e_loc], h_strips[hs]
                )
        for hs in range(hdr_strips):
            hdr_pad_out_ref[pl.ds(0, 8), pl.ds(hs * LANE, LANE)] = h_strips[hs]

    v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    inv_pad, gs_pad, hdr_pad = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((t_local, LANE), jnp.int32),
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
            f"pallas_moe_sortfree_route_tl{t_local}_el{num_local_experts}_b{b_pad}"
            f"{'_pad' if has_pad_sentinel else ''}"
        ),
    )(meta_sp, local_pad, global_2d)

    inv_perm_c = inv_pad[:, :topk]
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

    Eliminates ~45 unfused XLA HLO elementwise/reduction ops (~90 us/layer at R=4) with
    0-ULP bit-exact parity against `remap_zeroglue_virtual_ids(..., mode='q_block')`.
    """
    num_replicas = num_local_experts - num_base_local
    t_global = global_topk_ids.shape[0]
    t_local = t_global // ep_size
    total_elems = t_global * topk
    n_blocks = 64
    blk_elems = total_elems // n_blocks
    assert ep_size == 8 and num_base_local == 32 and 1 <= num_replicas <= 8
    assert total_elems % (n_blocks * 512) == 0
    blks_per_step = 2 if blk_elems == 512 else 1
    assert (blks_per_step * blk_elems) % (8 * LANE) == 0
    n_steps = n_blocks // blks_per_step
    rows_per_step = (blks_per_step * blk_elems) // LANE
    total_rows = total_elems // LANE
    rows_per_rank = (t_local * topk) // LANE
    assert rows_per_rank % 8 == 0

    ids_2d = global_topk_ids.astype(jnp.int32).reshape(total_rows, LANE)
    meta_sp = jnp.asarray([my_id.reshape(-1)[0].astype(jnp.int32)], dtype=jnp.int32)

    def _qblock_kernel(meta_ref, ids_ref, v_global_out_ref, v_local_out_ref):
        me = meta_ref[0]
        lane_8 = lax.broadcasted_iota(jnp.int32, (8, LANE), 1)
        row_8 = lax.broadcasted_iota(jnp.int32, (8, LANE), 0)
        lane_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 1)
        row_128 = lax.broadcasted_iota(jnp.int32, (LANE, LANE), 0)
        u_128 = (row_128 < lane_128).astype(jnp.bfloat16)

        init_carry = (jnp.zeros((8, LANE), dtype=jnp.int32),) + tuple(
            jnp.zeros((8, LANE), dtype=jnp.int32) for _ in range(num_replicas)
        )

        def _pass1(st, carry):
            base_acc = carry[0]
            bcs = list(carry[1:])
            r0 = pl.multiple_of(st * rows_per_step, 8)
            x = ids_ref[pl.ds(r0, rows_per_step), :]
            own = lax.shift_right_logical(x, jnp.int32(5))
            is_non_rep = x >= jnp.int32(num_replicas)
            for r in range(ep_size):
                cnt_r = jnp.sum((is_non_rep & (own == r)).astype(jnp.int32))
                base_acc = jnp.where(lane_8 == r, base_acc + cnt_r, base_acc)
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
        bcs = done_carry[1:]

        base_r = [
            jnp.max(jnp.sum(jnp.where(lane_8 == r, base_acc, 0), axis=1))
            for r in range(ep_size)
        ]
        cur_t = [
            lax.div(
                base_r[r]
                + jnp.int32(
                    ((num_base_local - (num_replicas if r == 0 else 0)) * quantum) // 2
                    + quantum
                    - 1
                ),
                jnp.int32(quantum),
            )
            for r in range(ep_size)
        ]

        vk_vecs = []
        for k in range(num_replicas):
            bc_k = bcs[k]
            cnt_k = jnp.max(jnp.sum(bc_k, axis=1))
            n_blks_k = jnp.where(
                cnt_k > 0,
                lax.div(cnt_k + jnp.int32(quantum - 1), jnp.int32(quantum)),
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

            lo7 = (bc_k & jnp.int32(127)).astype(jnp.bfloat16)
            hi7 = lax.shift_right_logical(bc_k, jnp.int32(7)).astype(jnp.bfloat16)
            p_lo = jnp.matmul(lo7, u_128, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            p_hi = jnp.matmul(hi7, u_128, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            cum_blk = lax.shift_left(p_hi, jnp.int32(7)) + p_lo
            mid_row = cum_blk + lax.shift_right_logical(bc_k, jnp.int32(1))
            blk_tile = lax.div(mid_row, jnp.int32(quantum))
            rk_vec = sum(
                (blk_tile >= cum_tq[r]).astype(jnp.int32) for r in range(ep_size - 1)
            )
            vk_vecs.append(
                rk_vec * jnp.int32(num_local_experts) + jnp.int32(num_base_local + k)
            )

        def _pass2(st, dummy):
            r0 = pl.multiple_of(st * rows_per_step, 8)
            x = ids_ref[pl.ds(r0, rows_per_step), :]
            v = lax.shift_right_logical(x, jnp.int32(5)) * jnp.int32(
                num_local_experts
            ) + (x & jnp.int32(31))
            for k in range(num_replicas):
                if blks_per_step == 1:
                    vk_b = jnp.max(
                        jnp.sum(jnp.where(lane_8 == st, vk_vecs[k], 0), axis=1)
                    )
                    v = jnp.where(x == k, vk_b, v)
                else:
                    vk_b0 = jnp.max(
                        jnp.sum(jnp.where(lane_8 == (2 * st), vk_vecs[k], 0), axis=1)
                    )
                    vk_b1 = jnp.max(
                        jnp.sum(
                            jnp.where(lane_8 == (2 * st + 1), vk_vecs[k], 0), axis=1
                        )
                    )
                    vk_row = jnp.where(row_8 < 4, vk_b0, vk_b1)
                    v = jnp.where(x == k, vk_row, v)
            v_global_out_ref[pl.ds(r0, rows_per_step), :] = v
            return dummy

        lax.fori_loop(0, n_steps, _pass2, jnp.int32(0))
        my_r0 = pl.multiple_of(me * rows_per_rank, 8)
        v_local_out_ref[:, :] = v_global_out_ref[pl.ds(my_r0, rows_per_rank), :]

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
            vmem_limit_bytes=(
                int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.94)
                if t_global > 16384
                else 32 * 1024 * 1024
            ),
        ),
        interpret=interpret,
        name=f"pallas_qblock_eplb_remap_tg{t_global}_el{num_local_experts}_q{quantum}",
    )(meta_sp, ids_2d)

    return v_glob_2d.reshape(global_topk_ids.shape), v_loc_2d.reshape(t_local, topk)

