# SPDX-License-Identifier: Apache-2.0
"""Sparse & low-batch-dense fuse-all Muse Spark decode MoE kernels with
Section 4.5b preloaded hot-expert copies, Section 4.3 intra-chip chipmate
weight redistribution, and Symmetric EPxTP2 replica-bank testbench.

Wave-3 Turbo Optimizations (arXiv:2611.17857v1 + LLO/roofline tuning):
  1. VPU-precomputed owner/rank loads (`precompute_imbalance_loads_vpu`): zero
     256-iteration SMEM RMW loops before weight DMA priming.
  2. Dense 2D fast-path (`row_mode="auto"` -> `"dense"` for `B in {16, 32}` on
     K256): eliminates token packing, gather, 3D<->2D copies, `lax.switch`, and
     row-by-row scatter when all `B` rows already fit 1-2 MXU tiles.
  3. 2-slot double-buffered `w1_buf[2]` + `s1_buf[2]` on K256 (`B=16..256`,
     `<= 58.2 MiB` VMEM): primes Tasks 0 & 1 upfront and prefetches Task `t+2`
     immediately after Task `t` Gate/Up (`0` exposed W1 HBM stalls).
  4. 2-bucket `(16, 64)` sparse schedule at `B >= 64` (keeps instruction bundle
     count far below the ~66K overlay cliff).
  5. Native K16 fast hoisted FP8 block-diagonal MXU path for `valid <= 8`
     (`bd_x_buf` built once in VMEM outside `@pl.loop`) + VREG-safe `nc=256`
     BF16 dequant for `valid > 8` (in both pure-EP and EPxTP2 testbench).
"""

from __future__ import annotations

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
    _fp4_k256_dual_gmm,
    _fp4_k256_gmm,
)
from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
    build_unified_worklist_smem,
    normalize_replica_cost_mode,
    plan_chipmate_move_smem,
    plan_preloaded_replicas_smem,
    precompute_imbalance_loads_vpu,
)

DEFAULT_TPU7X_CHIPMATE_XOR = 1
_NUM_EXPERTS = 256
_TOPK = 8
_H = 4096
_I = 4096
_N1 = 8192
_BD_KC = 256
_FP4_GRP = 16
_FP4_PK = 8


def _populate_bd8_x(src_2d_ref, bd_x_ref, k: int) -> None:
    """Expand 8 input rows into `(128, k)` block-diagonal group-16 masked VMEM."""
    diag = (
        lax.broadcasted_iota(jnp.int32, (8 * _FP4_GRP, _BD_KC), 0) % _FP4_GRP
        == lax.broadcasted_iota(jnp.int32, (8 * _FP4_GRP, _BD_KC), 1) // _FP4_GRP
    )
    for c0 in range(0, k, _BD_KC):
        xs = jnp.concatenate(
            [
                jnp.broadcast_to(src_2d_ref[a : a + 1, c0 : c0 + _BD_KC], (_FP4_GRP, _BD_KC))
                for a in range(8)
            ],
            axis=0,
        )
        bd_x_ref[:, c0 : c0 + _BD_KC] = jnp.where(diag, xs, jnp.bfloat16(0.0))


def _fp4_k16_bd8_dual_gmm(bd_x_ref, w_ref, scale_ref, k: int, n0_g, n0_u, nc: int):
    """Hoisted FP8 block-diagonal dual Gate+Up dot for `<= 8` valid rows on K16."""
    acc_g = None
    acc_u = None
    for c0 in range(0, k, _BD_KC):
        xb = bd_x_ref[:, c0 : c0 + _BD_KC]
        wg = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PK, _BD_KC // _FP4_PK), pl.ds(n0_g, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        wu = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PK, _BD_KC // _FP4_PK), pl.ds(n0_u, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        rg = jnp.dot(xb, wg, preferred_element_type=jnp.float32)
        ru = jnp.dot(xb, wu, preferred_element_type=jnp.float32)
        sg = scale_ref[pl.ds(c0 // _FP4_GRP, _FP4_GRP), pl.ds(n0_g, nc)].astype(jnp.float32)
        su = scale_ref[pl.ds(c0 // _FP4_GRP, _FP4_GRP), pl.ds(n0_u, nc)].astype(jnp.float32)
        pg = (rg.reshape(8, 2, 8, nc) * sg.reshape(1, 2, 8, nc)).sum(axis=1).reshape(64, nc)
        pu = (ru.reshape(8, 2, 8, nc) * su.reshape(1, 2, 8, nc)).sum(axis=1).reshape(64, nc)
        acc_g = pg if acc_g is None else acc_g + pg
        acc_u = pu if acc_u is None else acc_u + pu
    out_g = jnp.concatenate(
        [jnp.sum(acc_g[a * 8 : (a + 1) * 8, :], axis=0, keepdims=True) for a in range(8)],
        axis=0,
    )
    out_u = jnp.concatenate(
        [jnp.sum(acc_u[a * 8 : (a + 1) * 8, :], axis=0, keepdims=True) for a in range(8)],
        axis=0,
    )
    return out_g, out_u


def _fp4_k16_bd8_gmm(bd_x_ref, w_ref, scale_ref, k: int, n0, nc: int):
    """Hoisted FP8 block-diagonal Down dot for `<= 8` valid rows on K16."""
    acc = None
    for c0 in range(0, k, _BD_KC):
        xb = bd_x_ref[:, c0 : c0 + _BD_KC]
        w = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PK, _BD_KC // _FP4_PK), pl.ds(n0, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        r = jnp.dot(xb, w, preferred_element_type=jnp.float32)
        s = scale_ref[pl.ds(c0 // _FP4_GRP, _FP4_GRP), pl.ds(n0, nc)].astype(jnp.float32)
        part = (r.reshape(8, 2, 8, nc) * s.reshape(1, 2, 8, nc)).sum(axis=1).reshape(64, nc)
        acc = part if acc is None else acc + part
    return jnp.concatenate(
        [jnp.sum(acc[a * 8 : (a + 1) * 8, :], axis=0, keepdims=True) for a in range(8)],
        axis=0,
    )


def _route_topk_and_counts(
    logits_ref,
    bias_ref,
    counts_vmem,
    slot_ids_vmem,
    slot_w_vmem,
    counts_smem,
    slot_ids_smem,
    slot_w_smem,
    dma_sem,
    b: int,
    *,
    num_owners: int,
    cost_mode: str = "active",
    tile_m: int = 64,
    base_cost: int = 18,
    row_weight: int = 1,
    dense_w_vmem=None,
) -> None:
    """Exact tiled in-kernel Muse Spark top-8 routing + VPU load precomputation."""
    rb = min(b, 32)
    ids_tile = lax.broadcasted_iota(jnp.int32, (rb, _NUM_EXPERTS), 1)
    bias_row = bias_ref[...]

    counts_acc = jnp.zeros((1, _NUM_EXPERTS), jnp.int32)
    ids_blocks = []
    w_blocks = []
    dense_blocks = []

    for blk in range(b // rb):
        t0 = blk * rb
        l_blk = logits_ref[t0 : t0 + rb, :]
        l_blk = jnp.where(jnp.isnan(l_blk), jnp.float32(-jnp.inf), l_blk)
        scores = jnp.float32(1.0) / (jnp.float32(1.0) + jnp.exp(-l_blk))
        choice = scores + bias_row

        blk_ids = []
        blk_sel = []
        dense_blk = jnp.zeros((rb, _NUM_EXPERTS), jnp.float32) if dense_w_vmem is not None else None
        for _ in range(_TOPK):
            peak = jnp.max(choice, axis=1, keepdims=True)
            best = jnp.min(
                jnp.where(choice == peak, ids_tile, jnp.int32(_NUM_EXPERTS)),
                axis=1,
                keepdims=True,
            )
            pick = ids_tile == best
            sel_s = jnp.sum(jnp.where(pick, scores, jnp.float32(0.0)), axis=1)
            blk_ids.append(best[:, 0])
            blk_sel.append(sel_s)
            if dense_blk is not None:
                dense_blk = jnp.where(pick, scores, dense_blk)
            choice = jnp.where(pick, jnp.float32(-jnp.inf), choice)

        in_order = jnp.stack(blk_sel, axis=1)
        denom = jnp.sum(in_order, axis=1, keepdims=True) + jnp.float32(1e-15)
        norm_w = in_order / denom

        for s in range(_TOPK):
            w_s = norm_w[:, s : s + 1]
            id_s = blk_ids[s][:, None]
            counts_acc = counts_acc + jnp.sum(
                ((ids_tile == id_s) & (w_s > jnp.float32(0.0))).astype(jnp.int32),
                axis=0,
                keepdims=True,
            )

        if dense_blk is not None:
            dense_blocks.append(dense_blk / denom)
        else:
            ids_blocks.append(jnp.stack(blk_ids, axis=0))
            w_blocks.append(norm_w.T)

    counts_vmem[...] = precompute_imbalance_loads_vpu(
        counts_acc,
        num_owners=num_owners,
        cost_mode=cost_mode,
        tile_m=tile_m,
        base_cost=base_cost,
        row_weight=row_weight,
    )

    if dense_w_vmem is not None:
        full_dense = (
            jnp.concatenate(dense_blocks, axis=0)
            if len(dense_blocks) > 1
            else dense_blocks[0]
        )
        dense_w_vmem[...] = full_dense
        c0 = pltpu.make_async_copy(counts_vmem, counts_smem, dma_sem.at[0])
        c0.start()
        c0.wait()
        return

    b_pad = slot_ids_vmem.shape[1]
    full_ids = jnp.concatenate(ids_blocks, axis=1) if len(ids_blocks) > 1 else ids_blocks[0]
    full_w = jnp.concatenate(w_blocks, axis=1) if len(w_blocks) > 1 else w_blocks[0]
    if b < b_pad:
        pad = b_pad - b
        full_ids = jnp.concatenate([full_ids, jnp.zeros((_TOPK, pad), jnp.int32)], axis=1)
        full_w = jnp.concatenate([full_w, jnp.zeros((_TOPK, pad), jnp.float32)], axis=1)

    slot_ids_vmem[...] = full_ids
    slot_w_vmem[...] = full_w

    c0 = pltpu.make_async_copy(counts_vmem, counts_smem, dma_sem.at[0])
    c1 = pltpu.make_async_copy(slot_ids_vmem, slot_ids_smem, dma_sem.at[1])
    c2 = pltpu.make_async_copy(slot_w_vmem, slot_w_smem, dma_sem.at[2])
    c0.start()
    c1.start()
    c2.start()
    c0.wait()
    c1.wait()
    c2.wait()


def fused_decode_chipmate_from_logits(
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
    ep: int = 8,
    chipmate_xor: int | None = None,
    return_sharded: bool = False,
    replica_cost_mode: str = "active",
    enable_chipmate: bool = True,
    chipmate_min_gap: int = 2,
    chipmate_require_global_max: bool = False,
    chipmate_prefer_max_rows: bool = False,
    replica_tile_m: int = 64,
    replica_base_cost: int = 18,
    replica_row_weight: int = 1,
    replica_greedy_peak_first: bool = False,
    replica_chip_aware: bool = False,
    row_mode: str = "auto",
    use_k16_blockdiag: bool = True,
    nc: int = 1024,
):
    """Fuse-all FP4 decode MoE with Section 4.5b replicas + Section 4.3 chipmate."""
    b, h = x.shape
    if ep not in (8, 16):
        raise ValueError(f"fused_decode_chipmate supports ep in (8, 16), got {ep}")
    if h != _H or logits.shape != (b, _NUM_EXPERTS) or bias.shape != (1, _NUM_EXPERTS):
        raise ValueError(
            f"Expected x=(B,{_H}), logits=(B,{_NUM_EXPERTS}), bias=(1,{_NUM_EXPERTS}); "
            f"got x={x.shape}, logits={logits.shape}, bias={bias.shape}"
        )
    if b < ep or b % ep != 0 or b % 8 != 0 or b > 256:
        raise ValueError(
            f"Batch B={b} must be in [{ep}, 256], divisible by ep={ep} and 8"
        )
    if norm.shape != (1, _H):
        raise ValueError(f"Expected norm=(1, {_H}), got {norm.shape}")

    local = _NUM_EXPERTS // ep
    w_groups = int(w1.shape[0])
    block = h // s1.shape[1]
    if block not in (16, 256):
        raise ValueError(f"Expected K16 or K256 scales, got inferred block={block}")
    if (
        not (local <= w_groups <= local + 8)
        or w1.shape != (w_groups, _H, _N1)
        or w2.shape != (w_groups, _I, _H)
        or w1.dtype != jnp.float4_e2m1fn
        or w2.dtype != jnp.float4_e2m1fn
        or s1.shape != (w_groups, _H // block, _N1)
        or s2.shape != (w_groups, _I // block, _H)
    ):
        raise ValueError(
            f"Invalid EP{ep} K{block} weight shapes: w1={w1.shape}({w1.dtype}), "
            f"s1={s1.shape}, w2={w2.shape}({w2.dtype}), s2={s2.shape}"
        )

    rep_args = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    has_replicas = any(a is not None for a in rep_args)
    if has_replicas:
        if any(a is None for a in rep_args):
            raise ValueError(
                "rep_w1, rep_s1, rep_w2, rep_s2, and replica_meta must all be set or all None"
            )
        num_replicas = int(rep_w1.shape[0])
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if (
            rep_w1.shape != (num_replicas, _H, _N1)
            or rep_w2.shape != (num_replicas, _I, _H)
            or rep_w1.dtype != jnp.float4_e2m1fn
            or rep_w2.dtype != jnp.float4_e2m1fn
            or rep_s1.shape != (num_replicas, _H // block, _N1)
            or rep_s2.shape != (num_replicas, _I // block, _H)
            or replica_meta.shape != (4, ep * num_replicas)
            or replica_meta.dtype != jnp.int32
        ):
            raise ValueError(
                f"Invalid replica shapes: rep_w1={rep_w1.shape}, rep_s1={rep_s1.shape}, "
                f"rep_w2={rep_w2.shape}, rep_s2={rep_s2.shape}, meta={replica_meta.shape}"
            )
    else:
        num_replicas = 0

    if row_mode not in ("auto", "dense", "sparse"):
        raise ValueError(f"row_mode must be 'auto', 'dense', or 'sparse', got {row_mode!r}")
    use_dense = (row_mode == "dense") or (
        row_mode == "auto" and block == 256 and b in (16, 32)
    )
    if use_dense and b > 32:
        raise ValueError(f"row_mode='dense' supports B <= 32, got B={b}")

    cost_mode = normalize_replica_cost_mode(replica_cost_mode)
    cm_xor = DEFAULT_TPU7X_CHIPMATE_XOR if chipmate_xor is None else int(chipmate_xor)
    rows_per_rank = b // ep
    rows_shift = rows_per_rank.bit_length() - 1
    rows_is_pow2 = (rows_per_rank & (rows_per_rank - 1)) == 0
    rows_mask = rows_per_rank - 1
    b_pad = max(128, b)
    max_tile_m = min(64, b)
    n_w1_slots = 2 if block == 256 else 1
    nc_deq = nc if block == 256 else min(nc, 256)
    enable_bd8 = (block == 16) and use_k16_blockdiag and (not use_dense)
    # 2-bucket schedule at B>=64 keeps bundle count far below ~66K overlay cliff!
    if enable_bd8:
        buckets = tuple(m for m in (8, 16, 64) if m <= max_tile_m)
        if not buckets or buckets[-1] < max_tile_m:
            buckets = tuple(sorted(set(buckets + (max_tile_m,))))
    else:
        buckets = (16,) if max_tile_m <= 16 else ((16, 32) if max_tile_m == 32 else (16, 64))

    max_tasks = local + num_replicas + (1 if enable_chipmate else 0)
    max_packed = max(128, b * _TOPK)
    need_moved_flag = has_replicas and (enable_chipmate or replica_greedy_peak_first)

    # =========================================================================
    # Path A: Dense 2D Fast-Path (`B in {16, 32}` on K256)
    # =========================================================================
    if use_dense:
        packet = (8, rows_per_rank * _H // 8)
        wire = (ep, *packet)

        def _dense_kernel(*all_refs):
            idx = 0
            rank_ref = all_refs[idx]
            idx += 1
            replica_meta_ref = all_refs[idx] if has_replicas else None
            if has_replicas:
                idx += 1

            logits_ref = all_refs[idx]
            bias_ref = all_refs[idx + 1]
            x_ref = all_refs[idx + 2]
            w1_ref = all_refs[idx + 3].bitcast(jnp.uint32)
            s1_ref = all_refs[idx + 4]
            w2_ref = all_refs[idx + 5].bitcast(jnp.uint32)
            s2_ref = all_refs[idx + 6]
            norm_ref = all_refs[idx + 7]
            idx += 8

            if has_replicas:
                rep_w1_ref = all_refs[idx].bitcast(jnp.uint32)
                rep_s1_ref = all_refs[idx + 1]
                rep_w2_ref = all_refs[idx + 2].bitcast(jnp.uint32)
                rep_s2_ref = all_refs[idx + 3]
                idx += 4
            else:
                rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None

            out_ref = all_refs[idx]
            idx += 1
            if enable_chipmate:
                cm_w1_pad = all_refs[idx]
                cm_s1_pad = all_refs[idx + 1]
                cm_w2_pad = all_refs[idx + 2]
                cm_s2_pad = all_refs[idx + 3]
                idx += 4
            else:
                cm_w1_pad = cm_s1_pad = cm_w2_pad = cm_s2_pad = None

            w1_buf = all_refs[idx]
            s1_buf = all_refs[idx + 1]
            w2_buf = all_refs[idx + 2]
            s2_buf = all_refs[idx + 3]
            act_buf = all_refs[idx + 4]
            y_buf = all_refs[idx + 5]
            accum_buf = all_refs[idx + 6]
            dense_w_vmem = all_refs[idx + 7]
            counts_vmem = all_refs[idx + 8]
            counts_smem = all_refs[idx + 9]
            w1_dma_sem = all_refs[idx + 10]
            aux_dma_sem = all_refs[idx + 11]
            rs_send = all_refs[idx + 12]
            rs_recv = all_refs[idx + 13]
            rs_send_sem = all_refs[idx + 14]
            rs_recv_sem = all_refs[idx + 15]
            owner_buf = all_refs[idx + 16]
            idx += 17

            if not return_sharded:
                fan_recv = all_refs[idx]
                fan_send_sem = all_refs[idx + 1]
                fan_recv_sem = all_refs[idx + 2]
                idx += 3
            else:
                fan_recv = fan_send_sem = fan_recv_sem = None

            task_eid = all_refs[idx]
            task_src = all_refs[idx + 1]
            task_slot = all_refs[idx + 2]
            num_tasks_smem = all_refs[idx + 3]
            idx += 4

            if has_replicas:
                owner_load_smem = all_refs[idx]
                local_suppress_smem = all_refs[idx + 1]
                holder_slot_expert_smem = all_refs[idx + 2]
                idx += 3
                if need_moved_flag:
                    moved_flag_smem = all_refs[idx]
                    idx += 1
                else:
                    moved_flag_smem = None
            else:
                owner_load_smem = local_suppress_smem = None
                holder_slot_expert_smem = moved_flag_smem = None

            if enable_chipmate:
                cm_rank_active = all_refs[idx]
                cm_rank_rows = all_refs[idx + 1]
                cm_decision = all_refs[idx + 2]
                wt_send_sem = all_refs[idx + 3]
                wt_recv_sem = all_refs[idx + 4]
            else:
                cm_rank_active = cm_rank_rows = cm_decision = None
                wt_send_sem = wt_recv_sem = None

            me = rank_ref[0]
            partner = lax.bitwise_xor(me, jnp.int32(cm_xor))

            barrier = pltpu.get_barrier_semaphore()
            for dest in range(ep):
                pl.semaphore_signal(
                    barrier,
                    device_id={"model": jnp.int32(dest)},
                    device_id_type=pl.DeviceIdType.MESH,
                )

            _route_topk_and_counts(
                logits_ref,
                bias_ref,
                counts_vmem,
                None,
                None,
                counts_smem,
                None,
                None,
                aux_dma_sem,
                b,
                num_owners=ep,
                cost_mode=cost_mode,
                tile_m=replica_tile_m,
                base_cost=replica_base_cost,
                row_weight=replica_row_weight,
                dense_w_vmem=dense_w_vmem,
            )

            if has_replicas:
                plan_preloaded_replicas_smem(
                    me,
                    local,
                    ep,
                    num_replicas,
                    counts_smem,
                    replica_meta_ref,
                    local_suppress_smem,
                    holder_slot_expert_smem,
                    owner_load_smem,
                    cost_mode=cost_mode,
                    tile_m=replica_tile_m,
                    base_cost=replica_base_cost,
                    row_weight=replica_row_weight,
                    moved_flag_smem=moved_flag_smem,
                    greedy_peak_first=replica_greedy_peak_first,
                    vpu_precomputed_loads=True,
                    chip_aware=replica_chip_aware,
                    chipmate_xor=cm_xor,
                )

            if enable_chipmate:
                plan_chipmate_move_smem(
                    me,
                    ep,
                    counts_smem,
                    cm_rank_active,
                    cm_rank_rows,
                    cm_decision,
                    chipmate_xor=cm_xor,
                    replica_meta_smem=replica_meta_ref if has_replicas else None,
                    moved_flag_smem=moved_flag_smem if has_replicas else None,
                    min_active_gap=chipmate_min_gap,
                    require_global_max=chipmate_require_global_max,
                    vpu_precomputed_loads=True,
                    prefer_max_rows=chipmate_prefer_max_rows,
                )
                cm_send_slot = cm_decision[0]
                cm_recv_eid = cm_decision[1]
            else:
                cm_send_slot = None
                cm_recv_eid = None

            build_unified_worklist_smem(
                me,
                local,
                num_replicas,
                counts_smem,
                local_suppress_smem,
                holder_slot_expert_smem,
                task_eid,
                task_src,
                task_slot,
                num_tasks_smem,
                cm_send_slot=cm_send_slot,
                cm_recv_eid=cm_recv_eid,
            )
            n_tasks = num_tasks_smem[0]

            def _start_w1_slot(t_idx, s_idx):
                src_kind = task_src[t_idx]
                slot = task_slot[t_idx]

                @pl.when(src_kind == 0)
                def _fh():
                    pltpu.make_async_copy(
                        w1_ref.at[slot], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                    ).start()
                    pltpu.make_async_copy(
                        s1_ref.at[slot], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                    ).start()

                if has_replicas:

                    @pl.when(src_kind == 1)
                    def _fr():
                        pltpu.make_async_copy(
                            rep_w1_ref.at[slot], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                        ).start()
                        pltpu.make_async_copy(
                            rep_s1_ref.at[slot], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                        ).start()

                if enable_chipmate:

                    @pl.when(src_kind == 2)
                    def _fcm():
                        pltpu.make_async_copy(
                            cm_w1_pad, cm_w1_pad, wt_recv_sem.at[0]
                        ).wait()
                        pltpu.make_async_copy(
                            cm_s1_pad, cm_s1_pad, wt_recv_sem.at[1]
                        ).wait()
                        pltpu.make_async_copy(
                            cm_w1_pad, w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                        ).start()
                        pltpu.make_async_copy(
                            cm_s1_pad, s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                        ).start()

            def _start_w2(t_idx):
                src_kind = task_src[t_idx]
                slot = task_slot[t_idx]

                @pl.when(src_kind == 0)
                def _fh():
                    pltpu.make_async_copy(w2_ref.at[slot], w2_buf, aux_dma_sem.at[1]).start()
                    pltpu.make_async_copy(s2_ref.at[slot], s2_buf, aux_dma_sem.at[2]).start()

                if has_replicas:

                    @pl.when(src_kind == 1)
                    def _fr():
                        pltpu.make_async_copy(
                            rep_w2_ref.at[slot], w2_buf, aux_dma_sem.at[1]
                        ).start()
                        pltpu.make_async_copy(
                            rep_s2_ref.at[slot], s2_buf, aux_dma_sem.at[2]
                        ).start()

                if enable_chipmate:

                    @pl.when(src_kind == 2)
                    def _fcm():
                        pltpu.make_async_copy(
                            cm_w2_pad, cm_w2_pad, wt_recv_sem.at[2]
                        ).wait()
                        pltpu.make_async_copy(
                            cm_s2_pad, cm_s2_pad, wt_recv_sem.at[3]
                        ).wait()
                        pltpu.make_async_copy(cm_w2_pad, w2_buf, aux_dma_sem.at[1]).start()
                        pltpu.make_async_copy(cm_s2_pad, s2_buf, aux_dma_sem.at[2]).start()

            # Prime local/replica Task 0 & Task 1 BEFORE barrier wait so cross-chip
            # barrier sync overlaps behind 32 MiB of local HBM weight streaming!
            @pl.when((n_tasks > 0) & (task_src[0] != 2))
            def _prime_early0():
                _start_w1_slot(jnp.int32(0), jnp.int32(0))

            if n_w1_slots == 2:

                @pl.when((n_tasks > 1) & (task_src[1] != 2))
                def _prime_early1():
                    _start_w1_slot(jnp.int32(1), jnp.int32(1))

            accum_buf[...] = jnp.zeros(accum_buf.shape, jnp.float32)

            if enable_chipmate:
                pl.semaphore_wait(barrier, ep)

                @pl.when(cm_send_slot >= 0)
                def _start_cm_dma():
                    pltpu.make_async_remote_copy(
                        w1_ref.at[cm_send_slot],
                        cm_w1_pad,
                        wt_send_sem.at[0],
                        wt_recv_sem.at[0],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
                    pltpu.make_async_remote_copy(
                        s1_ref.at[cm_send_slot],
                        cm_s1_pad,
                        wt_send_sem.at[1],
                        wt_recv_sem.at[1],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
                    pltpu.make_async_remote_copy(
                        w2_ref.at[cm_send_slot],
                        cm_w2_pad,
                        wt_send_sem.at[2],
                        wt_recv_sem.at[2],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
                    pltpu.make_async_remote_copy(
                        s2_ref.at[cm_send_slot],
                        cm_s2_pad,
                        wt_send_sem.at[3],
                        wt_recv_sem.at[3],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                # Late prime if Task 0 or Task 1 WAS the borrowed chipmate expert!
                @pl.when((n_tasks > 0) & (task_src[0] == 2))
                def _prime_late0():
                    _start_w1_slot(jnp.int32(0), jnp.int32(0))

                if n_w1_slots == 2:

                    @pl.when((n_tasks > 1) & (task_src[1] == 2))
                    def _prime_late1():
                        _start_w1_slot(jnp.int32(1), jnp.int32(1))

            ids_full = lax.broadcasted_iota(jnp.int32, (b, _NUM_EXPERTS), 1)
            dw_val = dense_w_vmem[...]

            @pl.loop(0, n_tasks)
            def _dense_task(t_idx):
                _start_w2(t_idx)
                s_idx = (t_idx & jnp.int32(1)) if n_w1_slots == 2 else jnp.int32(0)
                pltpu.make_async_copy(
                    w1_buf.at[s_idx], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                ).wait()
                pltpu.make_async_copy(
                    s1_buf.at[s_idx], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                ).wait()

                @pl.loop(0, _I // nc_deq)
                def _up(j):
                    col = pl.multiple_of(j * nc_deq, nc_deq)
                    g, u = _fp4_k256_dual_gmm(
                        x_ref,
                        w1_buf.at[s_idx],
                        s1_buf.at[s_idx],
                        _H,
                        col,
                        _I + col,
                        nc_deq,
                        block == 256,
                        block,
                    )
                    act_buf[:, pl.ds(col, nc_deq)] = (
                        jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                    ).astype(jnp.bfloat16)

                next_t = t_idx + jnp.int32(n_w1_slots)

                @pl.when(next_t < n_tasks)
                def _pref_next():
                    _start_w1_slot(next_t, s_idx)

                pltpu.make_async_copy(w2_buf, w2_buf, aux_dma_sem.at[1]).wait()
                pltpu.make_async_copy(s2_buf, s2_buf, aux_dma_sem.at[2]).wait()

                @pl.loop(0, _H // nc_deq)
                def _down(j):
                    col = pl.multiple_of(j * nc_deq, nc_deq)
                    d = _fp4_k256_gmm(
                        act_buf, w2_buf, s2_buf, _I, col, nc_deq, block == 256, block
                    ).astype(jnp.bfloat16)
                    y_buf[:, pl.ds(col, nc_deq)] = d * (
                        norm_ref[:, pl.ds(col, nc_deq)].astype(jnp.float32)
                        + jnp.float32(1.0)
                    ).astype(jnp.bfloat16)

                yf = y_buf[...].astype(jnp.float32)
                rms = lax.rsqrt(
                    jnp.sum(yf * yf, axis=1, keepdims=True) * jnp.float32(1.0 / _H)
                    + jnp.float32(1e-8)
                ).astype(jnp.bfloat16)
                wy = (y_buf[...] * rms).astype(jnp.float32)
                eid = task_eid[t_idx]
                w_col = jnp.sum(
                    jnp.where(ids_full == eid, dw_val, jnp.float32(0.0)),
                    axis=1,
                    keepdims=True,
                )
                accum_buf[...] = accum_buf[...] + wy * w_col

            if enable_chipmate:

                @pl.when(cm_send_slot >= 0)
                def _drain_cm():
                    pltpu.make_async_remote_copy(
                        w1_ref.at[cm_send_slot],
                        cm_w1_pad,
                        wt_send_sem.at[0],
                        wt_recv_sem.at[0],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait_send()
                    pltpu.make_async_remote_copy(
                        s1_ref.at[cm_send_slot],
                        cm_s1_pad,
                        wt_send_sem.at[1],
                        wt_recv_sem.at[1],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait_send()
                    pltpu.make_async_remote_copy(
                        w2_ref.at[cm_send_slot],
                        cm_w2_pad,
                        wt_send_sem.at[2],
                        wt_recv_sem.at[2],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait_send()
                    pltpu.make_async_remote_copy(
                        s2_ref.at[cm_send_slot],
                        cm_s2_pad,
                        wt_send_sem.at[3],
                        wt_recv_sem.at[3],
                        device_id={"model": partner},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait_send()
            else:
                pl.semaphore_wait(barrier, ep)

            for dest in range(ep):
                rs_send[dest] = accum_buf[
                    pl.ds(dest * rows_per_rank, rows_per_rank), :
                ].reshape(packet)
            rs_copies = [
                pltpu.make_async_remote_copy(
                    rs_send.at[d],
                    rs_recv.at[me],
                    rs_send_sem.at[d],
                    rs_recv_sem,
                    device_id={"model": jnp.int32(d)},
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for d in range(ep)
            ]
            for op in rs_copies:
                op.start()
            pltpu.make_async_copy(rs_recv, rs_recv, rs_recv_sem).wait()
            total = rs_recv[0]
            for src in range(1, ep):
                total = total + rs_recv[src]
            owner_buf[...] = total.astype(jnp.bfloat16)

            if return_sharded:
                out_ref[...] = owner_buf[...].reshape(out_ref.shape)
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
                    for dest in range(ep)
                ]
                for cp in fan_copies:
                    cp.start()
                pltpu.make_async_copy(fan_recv, fan_recv, fan_recv_sem).wait()
                out_ref[...] = fan_recv[...].reshape(out_ref.shape)
                for cp in fan_copies:
                    cp.wait_send()

            for op in rs_copies:
                op.wait_send()

        vmem = pltpu.VMEM
        smem = pltpu.SMEM
        v_spec = pl.BlockSpec(memory_space=vmem)
        hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
        sem_dma = pltpu.SemaphoreType.DMA

        in_specs = [v_spec, v_spec, v_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec, v_spec]
        call_args = [rank]
        if has_replicas:
            call_args.append(replica_meta)
        call_args.extend([logits, bias, x, w1, s1, w2, s2, norm])
        if has_replicas:
            in_specs.extend([hbm_spec, hbm_spec, hbm_spec, hbm_spec])
            call_args.extend([rep_w1, rep_s1, rep_w2, rep_s2])

        main_out_shape = jax.ShapeDtypeStruct(
            (rows_per_rank, _H) if return_sharded else (b, _H), jnp.bfloat16
        )
        if enable_chipmate:
            out_shape = (
                main_out_shape,
                jax.ShapeDtypeStruct((_H // 8, _N1), jnp.uint32),
                jax.ShapeDtypeStruct(s1.shape[1:], s1.dtype),
                jax.ShapeDtypeStruct((_I // 8, _H), jnp.uint32),
                jax.ShapeDtypeStruct(s2.shape[1:], s2.dtype),
            )
            out_specs = (v_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec)
        else:
            out_shape = main_out_shape
            out_specs = v_spec

        scratch_shapes = [
            vmem((n_w1_slots, _H // 8, _N1), jnp.uint32),
            vmem((n_w1_slots, *s1.shape[1:]), s1.dtype),
            vmem((_I // 8, _H), jnp.uint32),
            vmem(s2.shape[1:], s2.dtype),
            vmem((b, _I), jnp.bfloat16),
            vmem((b, _H), jnp.bfloat16),
            vmem((b, _H), jnp.float32),
            vmem((b, _NUM_EXPERTS), jnp.float32),
            vmem((8, _NUM_EXPERTS), jnp.int32),
            smem((8, _NUM_EXPERTS), jnp.int32),
            sem_dma((n_w1_slots, 2)),
            sem_dma((3,)),
            vmem(wire, jnp.float32),
            vmem(wire, jnp.float32),
            sem_dma((ep,)),
            sem_dma,
            vmem(packet, jnp.bfloat16),
        ]
        if not return_sharded:
            scratch_shapes.extend([vmem(wire, jnp.bfloat16), sem_dma((ep,)), sem_dma])
        scratch_shapes.extend(
            [
                smem((max_tasks,), jnp.int32),
                smem((max_tasks,), jnp.int32),
                smem((max_tasks,), jnp.int32),
                smem((1,), jnp.int32),
            ]
        )
        if has_replicas:
            scratch_shapes.extend(
                [
                    smem((ep,), jnp.int32),
                    smem((local,), jnp.int32),
                    smem((num_replicas,), jnp.int32),
                ]
            )
            if need_moved_flag:
                scratch_shapes.append(smem((_NUM_EXPERTS,), jnp.int32))
        if enable_chipmate:
            scratch_shapes.extend(
                [
                    smem((ep,), jnp.int32),
                    smem((ep,), jnp.int32),
                    smem((3,), jnp.int32),
                    sem_dma((4,)),
                    sem_dma((4,)),
                ]
            )

        rep_tag = f"_r{num_replicas}_{cost_mode}" if has_replicas else "_r0"
        cm_tag = f"_cm{cm_xor}" if enable_chipmate else "_nocm"
        sh_tag = "_sh" if return_sharded else "_rep"
        kname = f"muse_decode_cm_ep{ep}_b{b}_k{block}_dns{rep_tag}{cm_tag}{sh_tag}"

        res = pl.pallas_call(
            _dense_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2 if has_replicas else 1,
                grid=(1,),
                in_specs=tuple(in_specs),
                out_specs=out_specs,
                scratch_shapes=tuple(scratch_shapes),
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=64 * 1024**2,
                collective_id=0,
            ),
            name=kname,
        )(*call_args)
        return res[0] if enable_chipmate else res

    # =========================================================================
    # Path B: Sparse Row-Bucketed Turbo Path (`B=64..256` K256 & `B=16..256` K16)
    # =========================================================================
    x_3d_in = x.reshape(b, 32, 128)

    def _sparse_kernel(*all_refs):
        idx = 0
        rank_ref = all_refs[idx]
        idx += 1
        replica_meta_ref = all_refs[idx] if has_replicas else None
        if has_replicas:
            idx += 1

        logits_ref = all_refs[idx]
        bias_ref = all_refs[idx + 1]
        x3d_ref = all_refs[idx + 2]
        w1_ref = all_refs[idx + 3].bitcast(jnp.uint32)
        s1_ref = all_refs[idx + 4]
        w2_ref = all_refs[idx + 5].bitcast(jnp.uint32)
        s2_ref = all_refs[idx + 6]
        norm_ref = all_refs[idx + 7]
        idx += 8

        if has_replicas:
            rep_w1_ref = all_refs[idx].bitcast(jnp.uint32)
            rep_s1_ref = all_refs[idx + 1]
            rep_w2_ref = all_refs[idx + 2].bitcast(jnp.uint32)
            rep_s2_ref = all_refs[idx + 3]
            idx += 4
        else:
            rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None

        out_ref = all_refs[idx]
        idx += 1
        if enable_chipmate:
            cm_w1_pad = all_refs[idx]
            cm_s1_pad = all_refs[idx + 1]
            cm_w2_pad = all_refs[idx + 2]
            cm_s2_pad = all_refs[idx + 3]
            idx += 4
        else:
            cm_w1_pad = cm_s1_pad = cm_w2_pad = cm_s2_pad = None

        w1_buf = all_refs[idx]
        s1_buf = all_refs[idx + 1]
        w2_buf = all_refs[idx + 2]
        s2_buf = all_refs[idx + 3]
        x_tile_3d = all_refs[idx + 4]
        x_tile_2d = all_refs[idx + 5]
        act_tile = all_refs[idx + 6]
        y_tile = all_refs[idx + 7]
        wy_tile_3d = all_refs[idx + 8]
        accum_3d = all_refs[idx + 9]
        counts_vmem = all_refs[idx + 10]
        slot_ids_vmem = all_refs[idx + 11]
        slot_w_vmem = all_refs[idx + 12]
        counts_smem = all_refs[idx + 13]
        slot_ids_smem = all_refs[idx + 14]
        slot_w_smem = all_refs[idx + 15]
        w1_dma_sem = all_refs[idx + 16]
        aux_dma_sem = all_refs[idx + 17]
        rs_recv = all_refs[idx + 18]
        rs_send_sem = all_refs[idx + 19]
        rs_recv_sem = all_refs[idx + 20]
        owner_buf = all_refs[idx + 21]
        idx += 22

        if enable_bd8:
            bd_x_buf = all_refs[idx]
            idx += 1
        else:
            bd_x_buf = None

        if not return_sharded:
            fan_recv = all_refs[idx]
            fan_send_sem = all_refs[idx + 1]
            fan_recv_sem = all_refs[idx + 2]
            idx += 3
        else:
            fan_recv = fan_send_sem = fan_recv_sem = None

        task_eid = all_refs[idx]
        task_src = all_refs[idx + 1]
        task_slot = all_refs[idx + 2]
        task_start = all_refs[idx + 3]
        task_end = all_refs[idx + 4]
        task_wptr = all_refs[idx + 5]
        num_tasks_smem = all_refs[idx + 6]
        expert_to_task = all_refs[idx + 7]
        packed_tok_ids = all_refs[idx + 8]
        packed_tok_w = all_refs[idx + 9]
        idx += 10

        if has_replicas:
            owner_load_smem = all_refs[idx]
            local_suppress_smem = all_refs[idx + 1]
            holder_slot_expert_smem = all_refs[idx + 2]
            idx += 3
            if need_moved_flag:
                moved_flag_smem = all_refs[idx]
                idx += 1
            else:
                moved_flag_smem = None
        else:
            owner_load_smem = local_suppress_smem = None
            holder_slot_expert_smem = moved_flag_smem = None

        if enable_chipmate:
            cm_rank_active = all_refs[idx]
            cm_rank_rows = all_refs[idx + 1]
            cm_decision = all_refs[idx + 2]
            wt_send_sem = all_refs[idx + 3]
            wt_recv_sem = all_refs[idx + 4]
        else:
            cm_rank_active = cm_rank_rows = cm_decision = None
            wt_send_sem = wt_recv_sem = None

        me = rank_ref[0]
        partner = lax.bitwise_xor(me, jnp.int32(cm_xor))

        barrier = pltpu.get_barrier_semaphore()
        for dest in range(ep):
            pl.semaphore_signal(
                barrier,
                device_id={"model": jnp.int32(dest)},
                device_id_type=pl.DeviceIdType.MESH,
            )

        _route_topk_and_counts(
            logits_ref,
            bias_ref,
            counts_vmem,
            slot_ids_vmem,
            slot_w_vmem,
            counts_smem,
            slot_ids_smem,
            slot_w_smem,
            aux_dma_sem,
            b,
            num_owners=ep,
            cost_mode=cost_mode,
            tile_m=replica_tile_m,
            base_cost=replica_base_cost,
            row_weight=replica_row_weight,
        )

        if has_replicas:
            plan_preloaded_replicas_smem(
                me,
                local,
                ep,
                num_replicas,
                counts_smem,
                replica_meta_ref,
                local_suppress_smem,
                holder_slot_expert_smem,
                owner_load_smem,
                cost_mode=cost_mode,
                tile_m=replica_tile_m,
                base_cost=replica_base_cost,
                row_weight=replica_row_weight,
                moved_flag_smem=moved_flag_smem,
                greedy_peak_first=replica_greedy_peak_first,
                vpu_precomputed_loads=True,
                chip_aware=replica_chip_aware,
                chipmate_xor=cm_xor,
            )

        if enable_chipmate:
            plan_chipmate_move_smem(
                me,
                ep,
                counts_smem,
                cm_rank_active,
                cm_rank_rows,
                cm_decision,
                chipmate_xor=cm_xor,
                replica_meta_smem=replica_meta_ref if has_replicas else None,
                moved_flag_smem=moved_flag_smem if has_replicas else None,
                min_active_gap=chipmate_min_gap,
                require_global_max=chipmate_require_global_max,
                vpu_precomputed_loads=True,
                prefer_max_rows=chipmate_prefer_max_rows,
            )
            cm_send_slot = cm_decision[0]
            cm_recv_eid = cm_decision[1]
        else:
            cm_send_slot = None
            cm_recv_eid = None

        build_unified_worklist_smem(
            me,
            local,
            num_replicas,
            counts_smem,
            local_suppress_smem,
            holder_slot_expert_smem,
            task_eid,
            task_src,
            task_slot,
            num_tasks_smem,
            cm_send_slot=cm_send_slot,
            cm_recv_eid=cm_recv_eid,
        )
        n_tasks = num_tasks_smem[0]

        def _start_w1_slot(t_idx, s_idx):
            src_kind = task_src[t_idx]
            slot = task_slot[t_idx]

            @pl.when(src_kind == 0)
            def _from_home():
                pltpu.make_async_copy(
                    w1_ref.at[slot], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                ).start()
                pltpu.make_async_copy(
                    s1_ref.at[slot], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                ).start()

            if has_replicas:

                @pl.when(src_kind == 1)
                def _from_rep():
                    pltpu.make_async_copy(
                        rep_w1_ref.at[slot], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                    ).start()
                    pltpu.make_async_copy(
                        rep_s1_ref.at[slot], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                    ).start()

            if enable_chipmate:

                @pl.when(src_kind == 2)
                def _from_cm():
                    pltpu.make_async_copy(
                        cm_w1_pad, cm_w1_pad, wt_recv_sem.at[0]
                    ).wait()
                    pltpu.make_async_copy(
                        cm_s1_pad, cm_s1_pad, wt_recv_sem.at[1]
                    ).wait()
                    pltpu.make_async_copy(
                        cm_w1_pad, w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                    ).start()
                    pltpu.make_async_copy(
                        cm_s1_pad, s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                    ).start()

        def _start_w2_for_task(t_idx):
            src_kind = task_src[t_idx]
            slot = task_slot[t_idx]

            @pl.when(src_kind == 0)
            def _from_home():
                pltpu.make_async_copy(w2_ref.at[slot], w2_buf, aux_dma_sem.at[1]).start()
                pltpu.make_async_copy(s2_ref.at[slot], s2_buf, aux_dma_sem.at[2]).start()

            if has_replicas:

                @pl.when(src_kind == 1)
                def _from_rep():
                    pltpu.make_async_copy(
                        rep_w2_ref.at[slot], w2_buf, aux_dma_sem.at[1]
                    ).start()
                    pltpu.make_async_copy(
                        rep_s2_ref.at[slot], s2_buf, aux_dma_sem.at[2]
                    ).start()

            if enable_chipmate:

                @pl.when(src_kind == 2)
                def _from_cm():
                    pltpu.make_async_copy(
                        cm_w2_pad, cm_w2_pad, wt_recv_sem.at[2]
                    ).wait()
                    pltpu.make_async_copy(
                        cm_s2_pad, cm_s2_pad, wt_recv_sem.at[3]
                    ).wait()
                    pltpu.make_async_copy(cm_w2_pad, w2_buf, aux_dma_sem.at[1]).start()
                    pltpu.make_async_copy(cm_s2_pad, s2_buf, aux_dma_sem.at[2]).start()

        # Prime local/replica Task 0 & Task 1 BEFORE barrier wait + SMEM token packing!
        @pl.when((n_tasks > 0) & (task_src[0] != 2))
        def _prime_early0():
            _start_w1_slot(jnp.int32(0), jnp.int32(0))

        if n_w1_slots == 2:

            @pl.when((n_tasks > 1) & (task_src[1] != 2))
            def _prime_early1():
                _start_w1_slot(jnp.int32(1), jnp.int32(1))

        # Pack routed token indices & weights in SMEM while W1 streams!
        @pl.loop(0, _NUM_EXPERTS)
        def _clear_e2t(e):
            expert_to_task[e] = jnp.int32(-1)

        def _prefix_body(t, off):
            eid = task_eid[t]
            cnt = counts_smem[0, eid]
            expert_to_task[eid] = t
            task_start[t] = off
            task_wptr[t] = off
            task_end[t] = off + cnt
            return off + cnt

        lax.fori_loop(0, n_tasks, _prefix_body, jnp.int32(0))

        @pl.loop(0, b)
        def _pack_tok(tok):
            for s in range(_TOPK):
                w_val = slot_w_smem[s, tok]

                @pl.when(w_val > jnp.float32(0.0))
                def _check_tok():
                    eid = slot_ids_smem[s, tok]
                    t = expert_to_task[eid]

                    @pl.when(t >= 0)
                    def _write_tok():
                        p = task_wptr[t]
                        packed_tok_ids[p] = tok
                        packed_tok_w[p] = w_val
                        task_wptr[t] = p + jnp.int32(1)

        accum_3d[...] = jnp.zeros(accum_3d.shape, jnp.float32)
        x_tile_3d[...] = jnp.zeros(x_tile_3d.shape, jnp.bfloat16)

        if enable_chipmate:
            pl.semaphore_wait(barrier, ep)

            @pl.when(cm_send_slot >= 0)
            def _start_cm_weight_dma():
                pltpu.make_async_remote_copy(
                    w1_ref.at[cm_send_slot],
                    cm_w1_pad,
                    wt_send_sem.at[0],
                    wt_recv_sem.at[0],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
                pltpu.make_async_remote_copy(
                    s1_ref.at[cm_send_slot],
                    cm_s1_pad,
                    wt_send_sem.at[1],
                    wt_recv_sem.at[1],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
                pltpu.make_async_remote_copy(
                    w2_ref.at[cm_send_slot],
                    cm_w2_pad,
                    wt_send_sem.at[2],
                    wt_recv_sem.at[2],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
                pltpu.make_async_remote_copy(
                    s2_ref.at[cm_send_slot],
                    cm_s2_pad,
                    wt_send_sem.at[3],
                    wt_recv_sem.at[3],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            @pl.when((n_tasks > 0) & (task_src[0] == 2))
            def _prime_late0():
                _start_w1_slot(jnp.int32(0), jnp.int32(0))

            if n_w1_slots == 2:

                @pl.when((n_tasks > 1) & (task_src[1] == 2))
                def _prime_late1():
                    _start_w1_slot(jnp.int32(1), jnp.int32(1))

        def _gmm1_swiglu(m_b: int, s_idx):
            for kc in range(4):
                x_tile_2d[0:m_b, kc * 1024 : (kc + 1) * 1024] = x_tile_3d[
                    0:m_b, kc * 8 : (kc + 1) * 8, :
                ].reshape(m_b, 1024)

            if enable_bd8 and m_b == 8:
                _populate_bd8_x(x_tile_2d.at[0:8, :], bd_x_buf, _H)
                nc_bd = 256

                @pl.loop(0, _I // nc_bd)
                def _up_bd(j):
                    col = pl.multiple_of(j * nc_bd, nc_bd)
                    g, u = _fp4_k16_bd8_dual_gmm(
                        bd_x_buf,
                        w1_buf.at[s_idx],
                        s1_buf.at[s_idx],
                        _H,
                        col,
                        _I + col,
                        nc_bd,
                    )
                    act_tile[0:8, pl.ds(col, nc_bd)] = (
                        jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                    ).astype(jnp.bfloat16)

                return jnp.int32(0)

            x_sub = x_tile_2d.at[0:m_b, :]

            @pl.loop(0, _I // nc_deq)
            def _up(j):
                col = pl.multiple_of(j * nc_deq, nc_deq)
                g, u = _fp4_k256_dual_gmm(
                    x_sub,
                    w1_buf.at[s_idx],
                    s1_buf.at[s_idx],
                    _H,
                    col,
                    _I + col,
                    nc_deq,
                    block == 256,
                    block,
                )
                act_tile[0:m_b, pl.ds(col, nc_deq)] = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)

            return jnp.int32(0)

        def _gmm2_norm(m_b: int):
            if enable_bd8 and m_b == 8:
                _populate_bd8_x(act_tile.at[0:8, :], bd_x_buf, _I)
                nc_bd = 256

                @pl.loop(0, _H // nc_bd)
                def _down_bd(j):
                    col = pl.multiple_of(j * nc_bd, nc_bd)
                    d = _fp4_k16_bd8_gmm(
                        bd_x_buf, w2_buf, s2_buf, _I, col, nc_bd
                    ).astype(jnp.bfloat16)
                    y_tile[0:8, pl.ds(col, nc_bd)] = d * (
                        norm_ref[:, pl.ds(col, nc_bd)].astype(jnp.float32)
                        + jnp.float32(1.0)
                    ).astype(jnp.bfloat16)

            else:
                act_sub = act_tile.at[0:m_b, :]

                @pl.loop(0, _H // nc_deq)
                def _down(j):
                    col = pl.multiple_of(j * nc_deq, nc_deq)
                    d = _fp4_k256_gmm(
                        act_sub, w2_buf, s2_buf, _I, col, nc_deq, block == 256, block
                    ).astype(jnp.bfloat16)
                    y_tile[0:m_b, pl.ds(col, nc_deq)] = d * (
                        norm_ref[:, pl.ds(col, nc_deq)].astype(jnp.float32)
                        + jnp.float32(1.0)
                    ).astype(jnp.bfloat16)

            yf = y_tile[0:m_b, :].astype(jnp.float32)
            rms = lax.rsqrt(
                jnp.sum(yf * yf, axis=1, keepdims=True) * jnp.float32(1.0 / _H)
                + jnp.float32(1e-8)
            ).astype(jnp.bfloat16)
            wy = (y_tile[0:m_b, :] * rms).astype(jnp.float32)
            for kc in range(4):
                wy_tile_3d[0:m_b, kc * 8 : (kc + 1) * 8, :] = wy[
                    :, kc * 1024 : (kc + 1) * 1024
                ].reshape(m_b, 8, 128)
            return jnp.int32(0)

        @pl.loop(0, n_tasks)
        def _task_loop(t_idx):
            _start_w2_for_task(t_idx)
            s_idx = (t_idx & jnp.int32(1)) if n_w1_slots == 2 else jnp.int32(0)

            start = task_start[t_idx]
            end = task_end[t_idx]
            cnt = end - start
            n_chunks = (cnt + jnp.int32(max_tile_m - 1)) // jnp.int32(max_tile_m)

            @pl.loop(0, n_chunks)
            def _chunk_loop(ch):
                r0 = ch * jnp.int32(max_tile_m)
                valid = jnp.minimum(jnp.int32(max_tile_m), cnt - r0)
                base_p = start + r0

                @pl.loop(0, valid)
                def _gather_x(row):
                    tok = packed_tok_ids[base_p + row]
                    x_tile_3d[row, :, :] = x3d_ref[tok, :, :]

                is_first = ch == jnp.int32(0)
                is_last = ch == (n_chunks - jnp.int32(1))

                @pl.when(is_first)
                def _wait_w1():
                    pltpu.make_async_copy(
                        w1_buf.at[s_idx], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                    ).wait()
                    pltpu.make_async_copy(
                        s1_buf.at[s_idx], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                    ).wait()

                b_idx = jnp.int32(0)
                for mb in buckets[:-1]:
                    b_idx = b_idx + (valid > jnp.int32(mb)).astype(jnp.int32)

                if len(buckets) == 1:
                    _gmm1_swiglu(buckets[0], s_idx)
                else:
                    lax.switch(
                        b_idx,
                        [(lambda m=mb: _gmm1_swiglu(m, s_idx)) for mb in buckets],
                    )

                next_t = t_idx + jnp.int32(n_w1_slots)

                @pl.when(is_last & (next_t < n_tasks))
                def _prefetch_next_w1():
                    _start_w1_slot(next_t, s_idx)

                @pl.when(is_first)
                def _wait_w2():
                    pltpu.make_async_copy(w2_buf, w2_buf, aux_dma_sem.at[1]).wait()
                    pltpu.make_async_copy(s2_buf, s2_buf, aux_dma_sem.at[2]).wait()

                if len(buckets) == 1:
                    _gmm2_norm(buckets[0])
                else:
                    lax.switch(
                        b_idx,
                        [(lambda m=mb: _gmm2_norm(m)) for mb in buckets],
                    )

                @pl.loop(0, valid)
                def _scatter_y(row):
                    tok = packed_tok_ids[base_p + row]
                    w_val = packed_tok_w[base_p + row]
                    if rows_is_pow2:
                        d_rank = lax.shift_right_logical(tok, jnp.int32(rows_shift))
                        r_off = tok & jnp.int32(rows_mask)
                    else:
                        d_rank = tok // jnp.int32(rows_per_rank)
                        r_off = tok % jnp.int32(rows_per_rank)
                    accum_3d[d_rank, r_off, :, :] = (
                        accum_3d[d_rank, r_off, :, :]
                        + wy_tile_3d[row, :, :] * w_val
                    )

        if enable_chipmate:

            @pl.when(cm_send_slot >= 0)
            def _wait_cm_weight_send():
                pltpu.make_async_remote_copy(
                    w1_ref.at[cm_send_slot],
                    cm_w1_pad,
                    wt_send_sem.at[0],
                    wt_recv_sem.at[0],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                pltpu.make_async_remote_copy(
                    s1_ref.at[cm_send_slot],
                    cm_s1_pad,
                    wt_send_sem.at[1],
                    wt_recv_sem.at[1],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                pltpu.make_async_remote_copy(
                    w2_ref.at[cm_send_slot],
                    cm_w2_pad,
                    wt_send_sem.at[2],
                    wt_recv_sem.at[2],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                pltpu.make_async_remote_copy(
                    s2_ref.at[cm_send_slot],
                    cm_s2_pad,
                    wt_send_sem.at[3],
                    wt_recv_sem.at[3],
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
        else:
            pl.semaphore_wait(barrier, ep)

        rs_copies = [
            pltpu.make_async_remote_copy(
                accum_3d.at[d],
                rs_recv.at[me],
                rs_send_sem.at[d],
                rs_recv_sem,
                device_id={"model": jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            for d in range(ep)
        ]
        for op in rs_copies:
            op.start()
        pltpu.make_async_copy(rs_recv, rs_recv, rs_recv_sem).wait()
        total = rs_recv[0]
        for src in range(1, ep):
            total = total + rs_recv[src]
        owner_buf[...] = total.astype(jnp.bfloat16)

        if return_sharded:
            out_ref[...] = owner_buf[...]
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
                for dest in range(ep)
            ]
            for cp in fan_copies:
                cp.start()
            pltpu.make_async_copy(fan_recv, fan_recv, fan_recv_sem).wait()
            out_ref[...] = fan_recv[...]
            for cp in fan_copies:
                cp.wait_send()

        for op in rs_copies:
            op.wait_send()

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    v_spec = pl.BlockSpec(memory_space=vmem)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    sem_dma = pltpu.SemaphoreType.DMA

    in_specs = [v_spec, v_spec, v_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec, v_spec]
    call_args = [rank]
    if has_replicas:
        call_args.append(replica_meta)
    call_args.extend([logits, bias, x_3d_in, w1, s1, w2, s2, norm])
    if has_replicas:
        in_specs.extend([hbm_spec, hbm_spec, hbm_spec, hbm_spec])
        call_args.extend([rep_w1, rep_s1, rep_w2, rep_s2])

    main_out_shape = (
        jax.ShapeDtypeStruct((rows_per_rank, 32, 128), jnp.bfloat16)
        if return_sharded
        else jax.ShapeDtypeStruct((ep, rows_per_rank, 32, 128), jnp.bfloat16)
    )
    if enable_chipmate:
        out_shape = (
            main_out_shape,
            jax.ShapeDtypeStruct((_H // 8, _N1), jnp.uint32),
            jax.ShapeDtypeStruct(s1.shape[1:], s1.dtype),
            jax.ShapeDtypeStruct((_I // 8, _H), jnp.uint32),
            jax.ShapeDtypeStruct(s2.shape[1:], s2.dtype),
        )
        out_specs = (v_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec)
    else:
        out_shape = main_out_shape
        out_specs = v_spec

    scratch_shapes = [
        vmem((n_w1_slots, _H // 8, _N1), jnp.uint32),
        vmem((n_w1_slots, *s1.shape[1:]), s1.dtype),
        vmem((_I // 8, _H), jnp.uint32),
        vmem(s2.shape[1:], s2.dtype),
        vmem((max_tile_m, 32, 128), jnp.bfloat16),
        vmem((max_tile_m, _H), jnp.bfloat16),
        vmem((max_tile_m, _I), jnp.bfloat16),
        vmem((max_tile_m, _H), jnp.bfloat16),
        vmem((max_tile_m, 32, 128), jnp.float32),
        vmem((ep, rows_per_rank, 32, 128), jnp.float32),
        vmem((8, _NUM_EXPERTS), jnp.int32),
        vmem((_TOPK, b_pad), jnp.int32),
        vmem((_TOPK, b_pad), jnp.float32),
        smem((8, _NUM_EXPERTS), jnp.int32),
        smem((_TOPK, b_pad), jnp.int32),
        smem((_TOPK, b_pad), jnp.float32),
        sem_dma((n_w1_slots, 2)),
        sem_dma((3,)),
        vmem((ep, rows_per_rank, 32, 128), jnp.float32),
        sem_dma((ep,)),
        sem_dma,
        vmem((rows_per_rank, 32, 128), jnp.bfloat16),
    ]
    if enable_bd8:
        scratch_shapes.append(vmem((8 * _FP4_GRP, _H), jnp.bfloat16))
    if not return_sharded:
        scratch_shapes.extend(
            [
                vmem((ep, rows_per_rank, 32, 128), jnp.bfloat16),
                sem_dma((ep,)),
                sem_dma,
            ]
        )

    scratch_shapes.extend(
        [
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((1,), jnp.int32),
            smem((_NUM_EXPERTS,), jnp.int32),
            smem((max_packed,), jnp.int32),
            smem((max_packed,), jnp.float32),
        ]
    )

    if has_replicas:
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((local,), jnp.int32),
                smem((num_replicas,), jnp.int32),
            ]
        )
        if need_moved_flag:
            scratch_shapes.append(smem((_NUM_EXPERTS,), jnp.int32))

    if enable_chipmate:
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((ep,), jnp.int32),
                smem((3,), jnp.int32),
                sem_dma((4,)),
                sem_dma((4,)),
            ]
        )

    rep_tag = f"_r{num_replicas}_{cost_mode}" if has_replicas else "_r0"
    cm_tag = f"_cm{cm_xor}" if enable_chipmate else "_nocm"
    sh_tag = "_sh" if return_sharded else "_rep"
    kname = f"muse_decode_cm_ep{ep}_b{b}_k{block}{rep_tag}{cm_tag}{sh_tag}"

    res = pl.pallas_call(
        _sparse_kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2 if has_replicas else 1,
            grid=(1,),
            in_specs=tuple(in_specs),
            out_specs=out_specs,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=kname,
    )(*call_args)

    main_out = res[0] if enable_chipmate else res
    return main_out.reshape((rows_per_rank, _H) if return_sharded else (b, _H))


def fused_decode_tp2_replica_testbench(
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
    replica_tile_m: int = 64,
    replica_base_cost: int = 18,
    replica_row_weight: int = 1,
    replica_greedy_peak_first: bool = False,
    use_k16_blockdiag: bool = True,
    nc: int = 1024,
):
    """Symmetric EPxTP2 Section 4.5b replica-bank execution testbench (`EP4xTP2` / `EP8xTP2`)."""
    b, h = x.shape
    if tp != 2 or ep not in (4, 8):
        raise ValueError(
            f"TP2 replica testbench requires ep in (4,8), tp=2; got ep={ep}, tp={tp}"
        )
    world = ep * tp
    e_chip = _NUM_EXPERTS // ep
    i_half = _I // 2
    n1_half = _N1 // 2
    if b < world or b % world != 0 or b % 8 != 0 or b > 256:
        raise ValueError(f"Batch B={b} must be in [{world}, 256] divisible by {world} and 8")

    block = h // s1.shape[1]
    if block not in (16, 256):
        raise ValueError(f"Expected K16 or K256 scales, got block={block}")
    if (
        w1.shape != (e_chip, _H, n1_half)
        or w2.shape != (e_chip, i_half, _H)
        or s1.shape != (e_chip, _H // block, n1_half)
        or s2.shape != (e_chip, i_half // block, _H)
    ):
        raise ValueError(
            f"Invalid EP{ep}xTP2 K{block} shapes: w1={w1.shape}, s1={s1.shape}, "
            f"w2={w2.shape}, s2={s2.shape}"
        )

    rep_args = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    has_replicas = any(a is not None for a in rep_args)
    if has_replicas:
        if any(a is None for a in rep_args):
            raise ValueError("All replica bank args must be set or all None")
        num_replicas = int(rep_w1.shape[0])
        if (
            rep_w1.shape != (num_replicas, _H, n1_half)
            or rep_w2.shape != (num_replicas, i_half, _H)
            or rep_s1.shape != (num_replicas, _H // block, n1_half)
            or rep_s2.shape != (num_replicas, i_half // block, _H)
            or replica_meta.shape != (4, ep * num_replicas)
        ):
            raise ValueError("Invalid TP2 replica bank shapes")
    else:
        num_replicas = 0

    cost_mode = normalize_replica_cost_mode(replica_cost_mode)
    cm_xor = DEFAULT_TPU7X_CHIPMATE_XOR if chipmate_xor is None else int(chipmate_xor)
    if cm_xor <= 0 or (cm_xor & (cm_xor - 1)) != 0:
        raise ValueError(f"chipmate_xor must be a power of 2, got {cm_xor}")
    xor_bit = cm_xor.bit_length() - 1

    rows_per_rank = b // world
    rows_shift = rows_per_rank.bit_length() - 1
    rows_is_pow2 = (rows_per_rank & (rows_per_rank - 1)) == 0
    rows_mask = rows_per_rank - 1
    b_pad = max(128, b)
    max_tile_m = min(64, b)
    # 2-slot W1 fits for BOTH K256 and K16 on TP2 because W1_half is only 8 MiB!
    n_w1_slots = 2
    nc_deq = nc if block == 256 else min(nc, 256)
    enable_bd8 = (block == 16) and use_k16_blockdiag
    if enable_bd8:
        buckets = tuple(m for m in (8, 16, 64) if m <= max_tile_m)
        if not buckets or buckets[-1] < max_tile_m:
            buckets = tuple(sorted(set(buckets + (max_tile_m,))))
    else:
        buckets = (16,) if max_tile_m <= 16 else ((16, 32) if max_tile_m == 32 else (16, 64))

    max_tasks = e_chip + num_replicas
    max_packed = max(128, b * _TOPK)
    need_moved_flag = has_replicas and replica_greedy_peak_first
    x_3d_in = x.reshape(b, 32, 128)

    def _tp2_kernel(*all_refs):
        idx = 0
        rank_ref = all_refs[idx]
        idx += 1
        replica_meta_ref = all_refs[idx] if has_replicas else None
        if has_replicas:
            idx += 1

        logits_ref = all_refs[idx]
        bias_ref = all_refs[idx + 1]
        x3d_ref = all_refs[idx + 2]
        w1_ref = all_refs[idx + 3].bitcast(jnp.uint32)
        s1_ref = all_refs[idx + 4]
        w2_ref = all_refs[idx + 5].bitcast(jnp.uint32)
        s2_ref = all_refs[idx + 6]
        norm_ref = all_refs[idx + 7]
        idx += 8

        if has_replicas:
            rep_w1_ref = all_refs[idx].bitcast(jnp.uint32)
            rep_s1_ref = all_refs[idx + 1]
            rep_w2_ref = all_refs[idx + 2].bitcast(jnp.uint32)
            rep_s2_ref = all_refs[idx + 3]
            idx += 4
        else:
            rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None

        out_ref = all_refs[idx]
        idx += 1

        w1_buf = all_refs[idx]
        s1_buf = all_refs[idx + 1]
        w2_buf = all_refs[idx + 2]
        s2_buf = all_refs[idx + 3]
        x_tile_3d = all_refs[idx + 4]
        x_tile_2d = all_refs[idx + 5]
        act_tile = all_refs[idx + 6]
        part_send = all_refs[idx + 7]
        part_recv = all_refs[idx + 8]
        wy_tile_3d = all_refs[idx + 9]
        accum_3d = all_refs[idx + 10]
        counts_vmem = all_refs[idx + 11]
        slot_ids_vmem = all_refs[idx + 12]
        slot_w_vmem = all_refs[idx + 13]
        counts_smem = all_refs[idx + 14]
        slot_ids_smem = all_refs[idx + 15]
        slot_w_smem = all_refs[idx + 16]
        w1_dma_sem = all_refs[idx + 17]
        aux_dma_sem = all_refs[idx + 18]
        br_send_sem = all_refs[idx + 19]
        br_recv_sem = all_refs[idx + 20]
        br_ack_sem = all_refs[idx + 21]
        step_counter_smem = all_refs[idx + 22]
        rs_recv = all_refs[idx + 23]
        rs_send_sem = all_refs[idx + 24]
        rs_recv_sem = all_refs[idx + 25]
        owner_buf = all_refs[idx + 26]
        idx += 27

        if enable_bd8:
            bd_x_buf = all_refs[idx]
            idx += 1
        else:
            bd_x_buf = None

        if not return_sharded:
            fan_recv = all_refs[idx]
            fan_send_sem = all_refs[idx + 1]
            fan_recv_sem = all_refs[idx + 2]
            idx += 3
        else:
            fan_recv = fan_send_sem = fan_recv_sem = None

        task_eid = all_refs[idx]
        task_src = all_refs[idx + 1]
        task_slot = all_refs[idx + 2]
        task_start = all_refs[idx + 3]
        task_end = all_refs[idx + 4]
        task_wptr = all_refs[idx + 5]
        num_tasks_smem = all_refs[idx + 6]
        expert_to_task = all_refs[idx + 7]
        packed_tok_ids = all_refs[idx + 8]
        packed_tok_w = all_refs[idx + 9]
        idx += 10

        if has_replicas:
            owner_load_smem = all_refs[idx]
            local_suppress_smem = all_refs[idx + 1]
            holder_slot_expert_smem = all_refs[idx + 2]
            moved_flag_smem = all_refs[idx + 3] if need_moved_flag else None
        else:
            owner_load_smem = local_suppress_smem = None
            holder_slot_expert_smem = moved_flag_smem = None

        me = rank_ref[0]
        partner = lax.bitwise_xor(me, jnp.int32(cm_xor))
        tp_half = lax.shift_right_logical(me, jnp.int32(xor_bit)) & jnp.int32(1)
        low_mask = jnp.int32((1 << xor_bit) - 1)
        my_chip = (me & low_mask) | lax.shift_left(
            lax.shift_right_logical(me, jnp.int32(xor_bit + 1)), jnp.int32(xor_bit)
        )

        barrier = pltpu.get_barrier_semaphore()
        for dest in range(world):
            pl.semaphore_signal(
                barrier,
                device_id={"model": jnp.int32(dest)},
                device_id_type=pl.DeviceIdType.MESH,
            )

        _route_topk_and_counts(
            logits_ref,
            bias_ref,
            counts_vmem,
            slot_ids_vmem,
            slot_w_vmem,
            counts_smem,
            slot_ids_smem,
            slot_w_smem,
            aux_dma_sem,
            b,
            num_owners=ep,
            cost_mode=cost_mode,
            tile_m=replica_tile_m,
            base_cost=replica_base_cost,
            row_weight=replica_row_weight,
        )

        if has_replicas:
            plan_preloaded_replicas_smem(
                my_chip,
                e_chip,
                ep,
                num_replicas,
                counts_smem,
                replica_meta_ref,
                local_suppress_smem,
                holder_slot_expert_smem,
                owner_load_smem,
                cost_mode=cost_mode,
                tile_m=replica_tile_m,
                base_cost=replica_base_cost,
                row_weight=replica_row_weight,
                moved_flag_smem=moved_flag_smem,
                greedy_peak_first=replica_greedy_peak_first,
                vpu_precomputed_loads=True,
            )

        build_unified_worklist_smem(
            my_chip,
            e_chip,
            num_replicas,
            counts_smem,
            local_suppress_smem,
            holder_slot_expert_smem,
            task_eid,
            task_src,
            task_slot,
            num_tasks_smem,
        )
        n_tasks = num_tasks_smem[0]
        step_counter_smem[0] = jnp.int32(0)

        def _start_w1_slot(t_idx, s_idx):
            src_kind = task_src[t_idx]
            slot = task_slot[t_idx]

            @pl.when(src_kind == 0)
            def _h():
                pltpu.make_async_copy(
                    w1_ref.at[slot], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                ).start()
                pltpu.make_async_copy(
                    s1_ref.at[slot], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                ).start()

            if has_replicas:

                @pl.when(src_kind == 1)
                def _r():
                    pltpu.make_async_copy(
                        rep_w1_ref.at[slot], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                    ).start()
                    pltpu.make_async_copy(
                        rep_s1_ref.at[slot], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                    ).start()

        def _start_w2(t_idx):
            src_kind = task_src[t_idx]
            slot = task_slot[t_idx]

            @pl.when(src_kind == 0)
            def _h():
                pltpu.make_async_copy(w2_ref.at[slot], w2_buf, aux_dma_sem.at[1]).start()
                pltpu.make_async_copy(s2_ref.at[slot], s2_buf, aux_dma_sem.at[2]).start()

            if has_replicas:

                @pl.when(src_kind == 1)
                def _r():
                    pltpu.make_async_copy(
                        rep_w2_ref.at[slot], w2_buf, aux_dma_sem.at[1]
                    ).start()
                    pltpu.make_async_copy(
                        rep_s2_ref.at[slot], s2_buf, aux_dma_sem.at[2]
                    ).start()

        @pl.when(n_tasks > 0)
        def _prime0():
            _start_w1_slot(jnp.int32(0), jnp.int32(0))

        @pl.when(n_tasks > 1)
        def _prime1():
            _start_w1_slot(jnp.int32(1), jnp.int32(1))

        @pl.loop(0, _NUM_EXPERTS)
        def _clr(e):
            expert_to_task[e] = jnp.int32(-1)

        def _pref(t, off):
            eid = task_eid[t]
            cnt = counts_smem[0, eid]
            expert_to_task[eid] = t
            task_start[t] = off
            task_wptr[t] = off
            task_end[t] = off + cnt
            return off + cnt

        lax.fori_loop(0, n_tasks, _pref, jnp.int32(0))

        @pl.loop(0, b)
        def _pack(tok):
            for s in range(_TOPK):
                w_val = slot_w_smem[s, tok]

                @pl.when(w_val > jnp.float32(0.0))
                def _c():
                    eid = slot_ids_smem[s, tok]
                    t = expert_to_task[eid]

                    @pl.when(t >= 0)
                    def _w():
                        p = task_wptr[t]
                        packed_tok_ids[p] = tok
                        packed_tok_w[p] = w_val
                        task_wptr[t] = p + jnp.int32(1)

        accum_3d[...] = jnp.zeros(accum_3d.shape, jnp.float32)
        x_tile_3d[...] = jnp.zeros(x_tile_3d.shape, jnp.bfloat16)
        pl.semaphore_wait(barrier, world)

        nc_up = min(nc_deq, i_half)

        def _tp2_gmm1(m_b: int, s_idx):
            for kc in range(4):
                x_tile_2d[0:m_b, kc * 1024 : (kc + 1) * 1024] = x_tile_3d[
                    0:m_b, kc * 8 : (kc + 1) * 8, :
                ].reshape(m_b, 1024)
            if enable_bd8 and m_b == 8:
                _populate_bd8_x(x_tile_2d.at[0:8, :], bd_x_buf, _H)
                nc_bd = 256

                @pl.loop(0, i_half // nc_bd)
                def _up_bd(j):
                    col = pl.multiple_of(j * nc_bd, nc_bd)
                    g, u = _fp4_k16_bd8_dual_gmm(
                        bd_x_buf,
                        w1_buf.at[s_idx],
                        s1_buf.at[s_idx],
                        _H,
                        col,
                        i_half + col,
                        nc_bd,
                    )
                    act_tile[0:8, pl.ds(col, nc_bd)] = (
                        jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                    ).astype(jnp.bfloat16)

                return jnp.int32(0)

            x_sub = x_tile_2d.at[0:m_b, :]

            @pl.loop(0, i_half // nc_up)
            def _up(j):
                col = pl.multiple_of(j * nc_up, nc_up)
                g, u = _fp4_k256_dual_gmm(
                    x_sub,
                    w1_buf.at[s_idx],
                    s1_buf.at[s_idx],
                    _H,
                    col,
                    i_half + col,
                    nc_up,
                    block == 256,
                    block,
                )
                act_tile[0:m_b, pl.ds(col, nc_up)] = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)

            return jnp.int32(0)

        def _tp2_gmm2(m_b: int):
            if enable_bd8 and m_b == 8:
                _populate_bd8_x(act_tile.at[0:8, :], bd_x_buf.at[:, 0:i_half], i_half)
                nc_bd = 256

                @pl.loop(0, _H // nc_bd)
                def _down_bd(j):
                    col = pl.multiple_of(j * nc_bd, nc_bd)
                    d = _fp4_k16_bd8_gmm(
                        bd_x_buf.at[:, 0:i_half], w2_buf, s2_buf, i_half, col, nc_bd
                    )
                    part_send[0:8, pl.ds(col, nc_bd)] = d

                return jnp.int32(0)

            act_sub = act_tile.at[0:m_b, :]

            @pl.loop(0, _H // nc_deq)
            def _down(j):
                col = pl.multiple_of(j * nc_deq, nc_deq)
                d = _fp4_k256_gmm(
                    act_sub, w2_buf, s2_buf, i_half, col, nc_deq, block == 256, block
                )
                part_send[0:m_b, pl.ds(col, nc_deq)] = d

            return jnp.int32(0)

        def _tp2_reduce_norm(m_b: int):
            d_sum = (part_send[0:m_b, :] + part_recv[0:m_b, :]).astype(jnp.bfloat16)
            gain = (norm_ref[...].astype(jnp.float32) + jnp.float32(1.0)).astype(
                jnp.bfloat16
            )
            y = d_sum * gain
            yf = y.astype(jnp.float32)
            rms = lax.rsqrt(
                jnp.sum(yf * yf, axis=1, keepdims=True) * jnp.float32(1.0 / _H)
                + jnp.float32(1e-8)
            ).astype(jnp.bfloat16)
            wy = (y * rms).astype(jnp.float32)
            for kc in range(4):
                wy_tile_3d[0:m_b, kc * 8 : (kc + 1) * 8, :] = wy[
                    :, kc * 1024 : (kc + 1) * 1024
                ].reshape(m_b, 8, 128)
            return jnp.int32(0)

        @pl.loop(0, n_tasks)
        def _task_loop(t_idx):
            _start_w2(t_idx)
            s_idx = t_idx & jnp.int32(1)
            start = task_start[t_idx]
            end = task_end[t_idx]
            cnt = end - start
            n_chunks = (cnt + jnp.int32(max_tile_m - 1)) // jnp.int32(max_tile_m)

            @pl.loop(0, n_chunks)
            def _chunk_loop(ch):
                r0 = ch * jnp.int32(max_tile_m)
                valid = jnp.minimum(jnp.int32(max_tile_m), cnt - r0)
                base_p = start + r0

                @pl.loop(0, valid)
                def _gather_x(row):
                    tok = packed_tok_ids[base_p + row]
                    x_tile_3d[row, :, :] = x3d_ref[tok, :, :]

                is_first = ch == jnp.int32(0)
                is_last = ch == (n_chunks - jnp.int32(1))

                @pl.when(is_first)
                def _wait_w1():
                    pltpu.make_async_copy(
                        w1_buf.at[s_idx], w1_buf.at[s_idx], w1_dma_sem.at[s_idx, 0]
                    ).wait()
                    pltpu.make_async_copy(
                        s1_buf.at[s_idx], s1_buf.at[s_idx], w1_dma_sem.at[s_idx, 1]
                    ).wait()

                b_idx = jnp.int32(0)
                for mb in buckets[:-1]:
                    b_idx = b_idx + (valid > jnp.int32(mb)).astype(jnp.int32)

                if len(buckets) == 1:
                    _tp2_gmm1(buckets[0], s_idx)
                else:
                    lax.switch(
                        b_idx, [(lambda m=mb: _tp2_gmm1(m, s_idx)) for mb in buckets]
                    )

                next_t = t_idx + jnp.int32(n_w1_slots)

                @pl.when(is_last & (next_t < n_tasks))
                def _pref_w1():
                    _start_w1_slot(next_t, s_idx)

                @pl.when(is_first)
                def _wait_w2():
                    pltpu.make_async_copy(w2_buf, w2_buf, aux_dma_sem.at[1]).wait()
                    pltpu.make_async_copy(s2_buf, s2_buf, aux_dma_sem.at[2]).wait()

                if len(buckets) == 1:
                    _tp2_gmm2(buckets[0])
                else:
                    lax.switch(b_idx, [(lambda m=mb: _tp2_gmm2(m)) for mb in buckets])

                step_id = step_counter_smem[0]

                @pl.when(step_id > 0)
                def _wait_prev_ack():
                    pl.semaphore_wait(br_ack_sem, 1)

                br_op = pltpu.make_async_remote_copy(
                    part_send,
                    part_recv,
                    br_send_sem,
                    br_recv_sem,
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                )
                br_op.start()
                pltpu.make_async_copy(part_recv, part_recv, br_recv_sem).wait()
                br_op.wait_send()

                if len(buckets) == 1:
                    _tp2_reduce_norm(buckets[0])
                else:
                    lax.switch(
                        b_idx, [(lambda m=mb: _tp2_reduce_norm(m)) for mb in buckets]
                    )

                pl.semaphore_signal(
                    br_ack_sem,
                    device_id={"model": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                )
                step_counter_smem[0] = step_id + jnp.int32(1)

                @pl.when(tp_half == 0)
                def _scatter_chip():
                    @pl.loop(0, valid)
                    def _sc(row):
                        tok = packed_tok_ids[base_p + row]
                        w_val = packed_tok_w[base_p + row]
                        if rows_is_pow2:
                            d_rank = lax.shift_right_logical(tok, jnp.int32(rows_shift))
                            r_off = tok & jnp.int32(rows_mask)
                        else:
                            d_rank = tok // jnp.int32(rows_per_rank)
                            r_off = tok % jnp.int32(rows_per_rank)
                        accum_3d[d_rank, r_off, :, :] = (
                            accum_3d[d_rank, r_off, :, :]
                            + wy_tile_3d[row, :, :] * w_val
                        )

        @pl.when(step_counter_smem[0] > 0)
        def _drain_final_ack():
            pl.semaphore_wait(br_ack_sem, 1)

        rs_copies = [
            pltpu.make_async_remote_copy(
                accum_3d.at[d],
                rs_recv.at[me],
                rs_send_sem.at[d],
                rs_recv_sem,
                device_id={"model": jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            for d in range(world)
        ]
        for op in rs_copies:
            op.start()
        pltpu.make_async_copy(rs_recv, rs_recv, rs_recv_sem).wait()
        total = rs_recv[0]
        for src in range(1, world):
            total = total + rs_recv[src]
        owner_buf[...] = total.astype(jnp.bfloat16)

        if return_sharded:
            out_ref[...] = owner_buf[...]
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
                for dest in range(world)
            ]
            for cp in fan_copies:
                cp.start()
            pltpu.make_async_copy(fan_recv, fan_recv, fan_recv_sem).wait()
            out_ref[...] = fan_recv[...]
            for cp in fan_copies:
                cp.wait_send()

        for op in rs_copies:
            op.wait_send()

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    v_spec = pl.BlockSpec(memory_space=vmem)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    sem_dma = pltpu.SemaphoreType.DMA
    sem_reg = pltpu.SemaphoreType.REGULAR

    in_specs = [v_spec, v_spec, v_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec, v_spec]
    call_args = [rank]
    if has_replicas:
        call_args.append(replica_meta)
    call_args.extend([logits, bias, x_3d_in, w1, s1, w2, s2, norm])
    if has_replicas:
        in_specs.extend([hbm_spec, hbm_spec, hbm_spec, hbm_spec])
        call_args.extend([rep_w1, rep_s1, rep_w2, rep_s2])

    out_shape = (
        jax.ShapeDtypeStruct((rows_per_rank, 32, 128), jnp.bfloat16)
        if return_sharded
        else jax.ShapeDtypeStruct((world, rows_per_rank, 32, 128), jnp.bfloat16)
    )

    scratch_shapes = [
        vmem((n_w1_slots, _H // 8, n1_half), jnp.uint32),
        vmem((n_w1_slots, *s1.shape[1:]), s1.dtype),
        vmem((i_half // 8, _H), jnp.uint32),
        vmem(s2.shape[1:], s2.dtype),
        vmem((max_tile_m, 32, 128), jnp.bfloat16),
        vmem((max_tile_m, _H), jnp.bfloat16),
        vmem((max_tile_m, i_half), jnp.bfloat16),
        vmem((max_tile_m, _H), jnp.float32),
        vmem((max_tile_m, _H), jnp.float32),
        vmem((max_tile_m, 32, 128), jnp.float32),
        vmem((world, rows_per_rank, 32, 128), jnp.float32),
        vmem((8, _NUM_EXPERTS), jnp.int32),
        vmem((_TOPK, b_pad), jnp.int32),
        vmem((_TOPK, b_pad), jnp.float32),
        smem((8, _NUM_EXPERTS), jnp.int32),
        smem((_TOPK, b_pad), jnp.int32),
        smem((_TOPK, b_pad), jnp.float32),
        sem_dma((n_w1_slots, 2)),
        sem_dma((3,)),
        sem_dma,
        sem_dma,
        sem_reg,
        smem((1,), jnp.int32),
        vmem((world, rows_per_rank, 32, 128), jnp.float32),
        sem_dma((world,)),
        sem_dma,
        vmem((rows_per_rank, 32, 128), jnp.bfloat16),
    ]
    if enable_bd8:
        scratch_shapes.append(vmem((8 * _FP4_GRP, _H), jnp.bfloat16))
    if not return_sharded:
        scratch_shapes.extend(
            [
                vmem((world, rows_per_rank, 32, 128), jnp.bfloat16),
                sem_dma((world,)),
                sem_dma,
            ]
        )

    scratch_shapes.extend(
        [
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((max_tasks,), jnp.int32),
            smem((1,), jnp.int32),
            smem((_NUM_EXPERTS,), jnp.int32),
            smem((max_packed,), jnp.int32),
            smem((max_packed,), jnp.float32),
        ]
    )

    if has_replicas:
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((e_chip,), jnp.int32),
                smem((num_replicas,), jnp.int32),
            ]
        )
        if need_moved_flag:
            scratch_shapes.append(smem((_NUM_EXPERTS,), jnp.int32))

    rep_tag = f"_r{num_replicas}_{cost_mode}" if has_replicas else "_r0"
    sh_tag = "_sh" if return_sharded else "_rep"
    kname = f"muse_decode_tp2tb_ep{ep}tp2_b{b}_k{block}{rep_tag}{sh_tag}"

    res = pl.pallas_call(
        _tp2_kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2 if has_replicas else 1,
            grid=(1,),
            in_specs=tuple(in_specs),
            out_specs=v_spec,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=kname,
    )(*call_args)

    return res.reshape((rows_per_rank, _H) if return_sharded else (b, _H))
