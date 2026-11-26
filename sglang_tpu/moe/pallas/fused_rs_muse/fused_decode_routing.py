# SPDX-License-Identifier: Apache-2.0
"""Shared in-kernel routing, global count reduction, and VPU-compacted token bucketing.

Provides tie-exact Muse Spark decode routing (sigmoid + bias, lowest-ID tie
breaking, selection-order normalization, NaN/-inf padded-row safety) inside
Pallas TPU kernels with zero extra cross-device collectives:

  1. ``route_weights(logits, bias)``: standalone JAX reference matching
     ``MuseSparkMoE`` and ``fused_decode_experts.route_weights``.
  2. ``route_dense_and_counts_in_kernel(...)``: fast dense-rows routing path for
     small decode batches (e.g. K256 B<=32), emitting ``dense_w_vmem[B, 256]``
     and ``counts_smem[0, 256]`` with only a single 8-KiB VMEM->SMEM DMA.
  3. ``route_topk_and_counts_in_kernel(...)``: VPU-filtered sparse routing that
     pre-compacts each token's local-candidate hits (home experts + optional R
     holder replica slots) in VREGs during top-k selection, cutting SMEM scatter
     work 8x-16x (zero predicated branches when R=0).
  4. ``build_default_exec_plan_smem(...)`` / ``build_exec_plan_from_masks_smem(...)``:
     32-iteration SMEM execution schedule builders (zero 256-entry clear loop)
     allowing immediate weight-ring priming before prologue/packing work.
"""

# PEP 563 stays off so TorchTPU / cpu_export_jax_op sees concrete annotations.

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def parse_free_expert_rows_spec(spec: str) -> tuple[int, float, int]:
    """Parse ``SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS`` into ``(max_rows, max_w, min_slot)``.

    Supported formats:
      - ``""`` / ``"0"`` / ``"off"`` -> ``(0, 0.0, 8)`` (disabled)
      - ``"2"`` -> ``(2, 0.06, 5)``
      - ``"2:0.05"`` -> ``(2, 0.05, 5)``
      - ``"2:0.05:6"`` -> ``(2, 0.05, 6)``
    """
    raw = str(spec or "").strip().lower()
    if not raw or raw in ("0", "false", "off", "none"):
        return 0, 0.0, 8
    parts = [p.strip() for p in raw.split(":") if p.strip()]
    max_rows = max(0, int(parts[0]))
    if max_rows <= 0:
        return 0, 0.0, 8
    max_w = float(parts[1]) if len(parts) >= 2 else 0.12
    min_slot = max(1, min(7, int(parts[2]))) if len(parts) >= 3 else 6
    return max_rows, max_w, min_slot


def _filter_topk_keep_flags(
    slot_ws: list[jax.Array],
    *,
    route_eps: float,
    min_p_topk: float | jax.Array,
    topk: int,
) -> list[jax.Array]:
    w_sum8 = sum(slot_ws) + route_eps
    if isinstance(min_p_topk, jax.Array):
        is_off = min_p_topk <= 0.0
        is_floor = (min_p_topk > 0.0) & (min_p_topk < 0.5)
        is_cap = min_p_topk >= 1.0
        # Express all 4 modes as scalar float32 thresholds (no scalar->vector i1 select_n in Mosaic).
        floor_eff = jnp.where(is_floor, min_p_topk, jnp.float32(-1.0))
        mass_eff = jnp.where(
            (~is_off) & (~is_floor) & (~is_cap), min_p_topk, jnp.float32(2.0)
        )
        cap_i32 = jnp.where(is_cap, min_p_topk.astype(jnp.int32), jnp.int32(topk + 1))
        keep_flags: list[jax.Array] = []
        cum_p = jnp.zeros_like(w_sum8)
        for s, w_s in enumerate(slot_ws):
            pos = w_s > 0.0
            if s == 0:
                keep_flags.append(pos)
            else:
                # Encode scalar cap via float32 threshold (< -1.0 impossible for positive weights).
                s_mass = jnp.where(jnp.int32(s) < cap_i32, mass_eff, jnp.float32(-1.0))
                s_floor = jnp.where(jnp.int32(s) < cap_i32, floor_eff, jnp.float32(2.0))
                ratio = w_s / w_sum8
                keep_flags.append(pos & (ratio >= s_floor) & (cum_p < s_mass))
            cum_p = cum_p + (w_s / w_sum8)
        return keep_flags
    if min_p_topk <= 0.0:
        return [w_s > 0.0 for w_s in slot_ws]
    if min_p_topk < 0.5:
        return [
            (w_s > 0.0)
            if s == 0
            else jnp.logical_and(w_s > 0.0, (w_s / w_sum8) >= min_p_topk)
            for s, w_s in enumerate(slot_ws)
        ]
    if min_p_topk >= 1.0:
        k_cap = max(1, min(topk, int(min_p_topk)))
        return [
            (w_s > 0.0) if s < k_cap else jnp.zeros_like(w_s, dtype=jnp.bool_)
            for s, w_s in enumerate(slot_ws)
        ]
    keep_flags = []
    cum_p = jnp.zeros_like(w_sum8)
    for s, w_s in enumerate(slot_ws):
        keep_s = (
            (w_s > 0.0)
            if s == 0
            else jnp.logical_and(w_s > 0.0, cum_p < min_p_topk)
        )
        keep_flags.append(keep_s)
        cum_p = cum_p + (w_s / w_sum8)
    return keep_flags


def _select_and_filter_chunk_vpu(
    lg_raw: jax.Array,
    bias_row: jax.Array,
    e_iota: jax.Array,
    num_experts: int,
    *,
    topk: int,
    route_eps: float,
    min_p_topk: float | jax.Array,
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array]]:
    """Select ``topk`` experts per row on VPU and apply per-token ``min_p_topk`` filter."""
    neg_inf = jnp.float32(-jnp.inf)
    lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
    scores = 1.0 / (1.0 + jnp.exp(-lg))
    choice = scores + bias_row
    slot_bests: list[jax.Array] = []
    slot_ws: list[jax.Array] = []
    for _ in range(topk):
        peak = jnp.max(choice, axis=1, keepdims=True)
        best = jnp.min(
            jnp.where(choice == peak, e_iota, num_experts),
            axis=1,
            keepdims=True,
        )
        pick = e_iota == best
        w_slot = jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
        slot_bests.append(best)
        slot_ws.append(w_slot)
        choice = jnp.where(pick, neg_inf, choice)

    keep_flags = _filter_topk_keep_flags(
        slot_ws, route_eps=route_eps, min_p_topk=min_p_topk, topk=topk
    )
    return slot_bests, slot_ws, keep_flags


def _droppable_flags_for_chunk(
    slot_ws: list[jax.Array],
    keep_flags: list[jax.Array],
    *,
    route_eps: float,
    free_max_w: float | jax.Array,
    free_min_slot: int | jax.Array,
) -> list[jax.Array]:
    w_sum_kept = (
        sum(jnp.where(k_s, w_s, 0.0) for w_s, k_s in zip(slot_ws, keep_flags))
        + route_eps
    )
    if isinstance(free_min_slot, jax.Array):
        eff_min_s = jnp.maximum(jnp.int32(1), free_min_slot.astype(jnp.int32))
        return [
            jnp.zeros_like(k_s)
            if s < 1
            else jnp.logical_and(
                k_s,
                (w_s / w_sum_kept)
                < jnp.where(jnp.int32(s) >= eff_min_s, free_max_w, jnp.float32(-1.0)),
            )
            for s, (w_s, k_s) in enumerate(zip(slot_ws, keep_flags))
        ]
    eff_min_slot = max(1, int(free_min_slot))
    return [
        jnp.zeros_like(k_s)
        if s < eff_min_slot
        else jnp.logical_and(k_s, (w_s / w_sum_kept) < free_max_w)
        for s, (w_s, k_s) in enumerate(zip(slot_ws, keep_flags))
    ]


def _owner_active_broadcast(
    active_row: jax.Array,
    *,
    num_owners: int,
    num_experts: int,
) -> jax.Array:
    """Broadcast per-owner active-expert counts to every expert ``(1, num_experts) int32``.

    Uses exact ``(8, 256) @ (256, 128) @ (128, 256)`` MXU tiles when
    ``num_experts == 256`` so Pallas TPU lowering never introduces unaligned
    ``(8, 32)`` VREG reshapes.
    """
    e_owner = num_experts // num_owners
    if num_experts == 256 and num_owners <= 128:
        active_8 = jnp.broadcast_to(active_row.reshape(1, num_experts).astype(jnp.float32), (8, num_experts))
        iota_e = lax.broadcasted_iota(jnp.int32, (num_experts, 128), 0)
        if e_owner > 0 and (e_owner & (e_owner - 1)) == 0:
            owner_col = lax.shift_right_logical(iota_e, jnp.int32(e_owner.bit_length() - 1))
        else:
            owner_col = iota_e // jnp.int32(e_owner)
        lane_128 = lax.broadcasted_iota(jnp.int32, (num_experts, 128), 1)
        one_hot = (
            (owner_col == lane_128) & (lane_128 < jnp.int32(num_owners))
        ).astype(jnp.float32)
        owner_sums_128 = jnp.rint(
            jnp.dot(
                active_8,
                one_hot,
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        )
        return jnp.rint(
            jnp.dot(
                owner_sums_128,
                jnp.transpose(one_hot),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        ).astype(jnp.int32)[0:1, :]
    counts_2d = active_row.reshape(num_owners, e_owner)
    per_owner = jnp.sum(counts_2d.astype(jnp.int32), axis=1, keepdims=True)
    return jnp.broadcast_to(per_owner, (num_owners, e_owner)).reshape(1, num_experts)


def _compute_load_aware_free_masks(
    tot_all: jax.Array,
    drp_base: jax.Array,
    drp_tail: jax.Array | None,
    *,
    free_max_rows: int | jax.Array,
    num_owners: int,
    num_experts: int,
    free_rank_cap: int = 0,
    free_tail_rows: int = 0,
) -> tuple[jax.Array, jax.Array | None]:
    """Compute base FREE mask and optional progressive rank-capped tail mask."""
    base_free = (
        (tot_all > 0) & (tot_all <= free_max_rows) & (drp_base == tot_all)
    )
    if free_rank_cap <= 0 or free_tail_rows <= 0 or drp_tail is None:
        return base_free, None
    cap_i32 = jnp.int32(free_rank_cap)
    after_base = jnp.where(base_free, jnp.int32(0), tot_all)
    ra0 = _owner_active_broadcast(
        after_base > 0, num_owners=num_owners, num_experts=num_experts
    )
    tf1 = (ra0 > cap_i32) & (tot_all == 1) & (drp_tail == 1)
    if free_tail_rows <= 1:
        return base_free, tf1
    after_t1 = jnp.where(base_free | tf1, jnp.int32(0), tot_all)
    ra1 = _owner_active_broadcast(
        after_t1 > 0, num_owners=num_owners, num_experts=num_experts
    )
    tf2 = (
        (ra1 > cap_i32)
        & (tot_all > 0)
        & (tot_all <= jnp.int32(free_tail_rows))
        & (drp_tail == tot_all)
    )
    return base_free, (tf1 | tf2)


def route_weights(
    logits: jax.Array,
    bias: jax.Array,
    *,
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    min_p_topk: float = 0.0,
    free_max_rows: int = 0,
    free_max_w: float = 0.06,
    free_min_slot: int = 5,
    num_owners: int = 8,
    free_rank_cap: int = 0,
    free_tail_rows: int = 0,
    free_tail_w: float = 0.0,
) -> jax.Array:
    """Exact serving router: sigmoid + bias, lower-ID tie-break, selection-order norm."""
    use_tail_cap = free_rank_cap > 0 and free_tail_rows > 0 and free_tail_w > 0.0
    has_free = free_max_rows > 0 or use_tail_cap
    if min_p_topk > 0.0 or has_free:
        num_tokens, num_experts = logits.shape
        bias_row = bias.reshape(1, -1).astype(jnp.float32)
        ids = lax.broadcasted_iota(jnp.int32, (num_tokens, num_experts), 1)
        slot_bests, slot_ws, keep_flags = _select_and_filter_chunk_vpu(
            logits.astype(jnp.float32),
            bias_row,
            ids,
            num_experts,
            topk=topk,
            route_eps=route_eps,
            min_p_topk=min_p_topk,
        )
        if has_free:
            drop_flags = _droppable_flags_for_chunk(
                slot_ws,
                keep_flags,
                route_eps=route_eps,
                free_max_w=free_max_w,
                free_min_slot=free_min_slot,
            )
            drop_tail = (
                _droppable_flags_for_chunk(
                    slot_ws,
                    keep_flags,
                    route_eps=route_eps,
                    free_max_w=max(float(free_max_w), float(free_tail_w)),
                    free_min_slot=free_min_slot,
                )
                if use_tail_cap
                else None
            )
            tot_cnt = jnp.zeros((1, num_experts), dtype=jnp.int32)
            drp_cnt = jnp.zeros((1, num_experts), dtype=jnp.int32)
            drp_tail_cnt = (
                jnp.zeros((1, num_experts), dtype=jnp.int32) if use_tail_cap else None
            )
            for idx_s, (b_s, k_s, d_s) in enumerate(zip(slot_bests, keep_flags, drop_flags)):
                pk = ids == b_s
                tot_cnt = tot_cnt + jnp.sum(
                    jnp.logical_and(pk, k_s).astype(jnp.int32),
                    axis=0,
                    keepdims=True,
                )
                drp_cnt = drp_cnt + jnp.sum(
                    jnp.logical_and(pk, d_s).astype(jnp.int32),
                    axis=0,
                    keepdims=True,
                )
                if use_tail_cap:
                    drp_tail_cnt = drp_tail_cnt + jnp.sum(
                        jnp.logical_and(pk, drop_tail[idx_s]).astype(jnp.int32),
                        axis=0,
                        keepdims=True,
                    )
            base_mask, tail_mask = _compute_load_aware_free_masks(
                tot_cnt,
                drp_cnt,
                drp_tail_cnt,
                free_max_rows=free_max_rows,
                num_owners=num_owners,
                num_experts=num_experts,
                free_rank_cap=free_rank_cap if use_tail_cap else 0,
                free_tail_rows=free_tail_rows if use_tail_cap else 0,
            )
            eff_min_s = max(1, int(free_min_slot))
            for s in range(eff_min_s, topk):
                freed_b = jnp.any(
                    (ids == slot_bests[s]) & base_mask, axis=1, keepdims=True
                )
                drop_s = drop_flags[s] & freed_b
                if tail_mask is not None:
                    freed_t = jnp.any(
                        (ids == slot_bests[s]) & tail_mask, axis=1, keepdims=True
                    )
                    drop_s = drop_s | (drop_tail[s] & freed_t)
                keep_flags[s] = keep_flags[s] & (~drop_s)
        dense = jnp.zeros((num_tokens, num_experts), dtype=jnp.float32)
        w_sum = jnp.zeros((num_tokens, 1), dtype=jnp.float32)
        for b_s, w_s, k_s in zip(slot_bests, slot_ws, keep_flags):
            pk = jnp.logical_and(ids == b_s, k_s)
            dense = jnp.where(pk, w_s, dense)
            w_sum = w_sum + jnp.where(k_s, w_s, 0.0)
        out = dense / (w_sum + route_eps)
        return out * route_scale if route_scale != 1.0 else out
    logits_raw = logits.astype(jnp.float32)
    logits_f32 = jnp.where(jnp.isnan(logits_raw), -jnp.inf, logits_raw)
    bias_row = bias.reshape(1, -1).astype(jnp.float32)
    scores = 1.0 / (1.0 + jnp.exp(-logits_f32))
    choice = scores + bias_row
    num_tokens, num_experts = scores.shape
    ids = lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    slots = lax.broadcasted_iota(jnp.int32, (num_tokens, topk), 1)
    in_order = jnp.zeros(slots.shape, jnp.float32)
    dense = jnp.zeros_like(scores)
    for slot in range(topk):
        peak = jnp.max(choice, axis=1, keepdims=True)
        best = jnp.min(
            jnp.where(choice == peak, ids, num_experts), axis=1, keepdims=True
        )
        pick = ids == best
        selected = jnp.where(pick, scores, 0.0)
        in_order = jnp.where(
            slots == slot, selected.sum(axis=1, keepdims=True), in_order
        )
        dense = jnp.where(pick, scores, dense)
        choice = jnp.where(pick, -jnp.inf, choice)
    out = dense / (in_order.sum(axis=1, keepdims=True) + route_eps)
    if route_scale != 1.0:
        out = out * route_scale
    return out


class RoutingScratch(NamedTuple):
    """Scratch refs owned by ``fused_decode_routing`` inside a sparse Pallas kernel."""

    counts_vmem: object
    counts_smem: object
    route_pack_vmem: object
    route_pack_smem: object
    exec_expert_id: object
    exec_weight_src: object
    exec_weight_slot: object
    exec_start: object
    exec_write_ptr: object
    cand_exec_idx: object
    packed_tok_ids: object
    packed_tok_w: object


class DenseRoutingScratch(NamedTuple):
    """Compact scratch refs for dense-rows decode mode (no token pack/scatter)."""

    counts_vmem: object
    counts_smem: object
    dense_w_vmem: object
    exec_expert_id: object
    exec_weight_src: object
    exec_weight_slot: object


def routing_scratch_shapes(
    batch_size: int,
    max_exec_experts: int,
    *,
    topk: int = 8,
    num_experts: int = 256,
) -> RoutingScratch:
    """Return Pallas VMEM/SMEM scratch specs matching ``RoutingScratch``."""
    pack_rows = max(8, ((batch_size + 7) // 8) * 8)
    max_packed = max(128, batch_size * topk) + 1
    return RoutingScratch(
        counts_vmem=pltpu.VMEM((8, num_experts), jnp.int32),
        counts_smem=pltpu.SMEM((8, num_experts), jnp.int32),
        route_pack_vmem=pltpu.VMEM((pack_rows, 128), jnp.int32),
        route_pack_smem=pltpu.SMEM((pack_rows, 128), jnp.int32),
        exec_expert_id=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
        exec_weight_src=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
        exec_weight_slot=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
        exec_start=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
        exec_write_ptr=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
        cand_exec_idx=pltpu.SMEM((max_exec_experts,), jnp.int32),
        packed_tok_ids=pltpu.SMEM((max_packed,), jnp.int32),
        packed_tok_w=pltpu.SMEM((max_packed,), jnp.float32),
    )


def dense_routing_scratch_shapes(
    batch_size: int,
    max_exec_experts: int,
    *,
    num_experts: int = 256,
) -> DenseRoutingScratch:
    """Return Pallas VMEM/SMEM scratch specs for dense-rows decode mode."""
    return DenseRoutingScratch(
        counts_vmem=pltpu.VMEM((8, num_experts), jnp.int32),
        counts_smem=pltpu.SMEM((8, num_experts), jnp.int32),
        dense_w_vmem=pltpu.VMEM((batch_size, num_experts), jnp.float32),
        exec_expert_id=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
        exec_weight_src=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
        exec_weight_slot=pltpu.SMEM((max_exec_experts + 1,), jnp.int32),
    )


def extract_my_holder_candidates_smem(
    my_owner: jax.Array,
    num_replicas: int,
    replica_meta_smem,
    holder_cand_smem,
) -> None:
    """Populate ``holder_cand_smem[0..R-1]`` with global expert IDs held by ``my_owner``."""
    for s in range(num_replicas):
        holder_cand_smem[s] = jnp.int32(-1)

    k_rep = replica_meta_smem.shape[1]

    def _scan_meta(idx, _):
        eid = replica_meta_smem[0, idx]
        holder = replica_meta_smem[2, idx]
        slot = replica_meta_smem[3, idx]
        mine = jnp.logical_and(eid >= 0, holder == my_owner)

        @pl.when(mine)
        def _set():
            holder_cand_smem[slot] = eid

        return _

    lax.fori_loop(0, k_rep, _scan_meta, jnp.int32(0))


def extract_my_holder_candidates_tuple(
    my_owner: jax.Array,
    num_replicas: int,
    replica_meta_smem,
) -> tuple[jax.Array, ...]:
    """Branchless scalar extraction of ``my_owner``'s ``R`` holder candidate expert IDs.

    Since ``k_rep = replica_meta_smem.shape[1]`` is Python-static and each
    ``(holder, slot)`` pair appears at most once with ``eid >= 0``, unrolling the
    exact scalar select eliminates ``holder_cand_smem``, ``scf.for``, ``@pl.when``,
    and the SALU store-to-load SMEM fence before VPU routing.
    """
    k_rep = replica_meta_smem.shape[1]
    eids = [replica_meta_smem[0, k] for k in range(k_rep)]
    holders = [replica_meta_smem[2, k] for k in range(k_rep)]
    slots = [replica_meta_smem[3, k] for k in range(k_rep)]
    return tuple(
        sum(
            jnp.where(
                (eids[k] >= 0) & (holders[k] == my_owner) & (slots[k] == s),
                eids[k] + jnp.int32(1),
                jnp.int32(0),
            )
            for k in range(k_rep)
        )
        - jnp.int32(1)
        for s in range(num_replicas)
    )


def select_non_replicated_home_slot(
    my_owner: jax.Array,
    e_per_owner: int,
    replica_meta_smem,
    *,
    preferred_slot: jax.Array | int = 0,
    num_replicas: int = 0,
) -> jax.Array:
    """Return ``preferred_slot`` if not outgoing-replicated, else first non-replicated home slot."""
    pref = jnp.asarray(preferred_slot, dtype=jnp.int32)
    if num_replicas <= 0 or replica_meta_smem is None:
        return pref
    first = my_owner * e_per_owner
    k_rep = replica_meta_smem.shape[1]
    rep_eids = [replica_meta_smem[0, k] for k in range(k_rep)]

    def _is_rep(local_slot: jax.Array) -> jax.Array:
        g_eid = first + local_slot
        hit = jnp.bool_(False)
        for eid in rep_eids:
            hit = hit | (eid == g_eid)
        return hit

    # At most R local experts on my_owner can appear in replica_meta; checking R + 1
    # distinct local slots guarantees finding a non-replicated home slot.
    chosen = jnp.int32(-1)
    chosen = jnp.where(_is_rep(pref), chosen, pref)
    for alt in range(min(e_per_owner, num_replicas + 1)):
        alt_s = jnp.int32(alt)
        chosen = jnp.where((chosen < 0) & (~_is_rep(alt_s)), alt_s, chosen)
    return jnp.where(chosen >= 0, chosen, pref)


def route_dense_and_counts_in_kernel(
    logits_ref,
    bias_ref,
    d_scratch: DenseRoutingScratch,
    dma_sem,
    *,
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    route_chunk_rows: int = 32,
    min_p_topk: float = 0.0,
    free_max_rows: int = 0,
    free_max_w: float = 0.06,
    free_min_slot: int = 5,
) -> None:
    """Compute ``dense_w_vmem[B, 256]`` and DMA ``counts_smem[0, 256]`` (1 DMA)."""
    if _bridge1_prerouted_active():
        _route_prerouted_dense_in_kernel(logits_ref, d_scratch, dma_sem, topk=topk, route_chunk_rows=route_chunk_rows)
        return
    batch_size, num_experts = logits_ref.shape
    eff_chunk = (
        max(route_chunk_rows, min(batch_size, 64))
        if free_max_rows > 0
        else route_chunk_rows
    )
    rb = min(batch_size, eff_chunk)
    if batch_size % rb != 0:
        rb = 8 if batch_size % 8 == 0 else batch_size

    bias_row = bias_ref[...].reshape(1, num_experts).astype(jnp.float32)
    e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
    slots_iota = lax.broadcasted_iota(jnp.int32, (rb, topk), 1)
    counts_init = jnp.zeros((1, num_experts), jnp.int32)
    neg_inf = jnp.float32(-jnp.inf)

    if min_p_topk > 0.0 or free_max_rows > 0:
        n_chunks = batch_size // rb
        eff_min_s = max(1, int(free_min_slot))
        free_mask_global = None
        if free_max_rows > 0 and n_chunks > 1:

            def _scan_free_dense(ci, carry):
                tot_c, drp_c = carry
                r0 = pl.multiple_of(ci * rb, rb)
                sb, sw, kf = _select_and_filter_chunk_vpu(
                    logits_ref[pl.ds(r0, rb), :].astype(jnp.float32),
                    bias_row,
                    e_iota,
                    num_experts,
                    topk=topk,
                    route_eps=route_eps,
                    min_p_topk=min_p_topk,
                )
                df = _droppable_flags_for_chunk(
                    sw,
                    kf,
                    route_eps=route_eps,
                    free_max_w=free_max_w,
                    free_min_slot=free_min_slot,
                )
                for b_s, k_s, d_s in zip(sb, kf, df):
                    pk = e_iota == b_s
                    tot_c = tot_c + jnp.sum(
                        jnp.logical_and(pk, k_s).astype(jnp.int32),
                        axis=0,
                        keepdims=True,
                    )
                    drp_c = drp_c + jnp.sum(
                        jnp.logical_and(pk, d_s).astype(jnp.int32),
                        axis=0,
                        keepdims=True,
                    )
                return tot_c, drp_c

            tot_all, drp_all = lax.fori_loop(
                0, n_chunks, _scan_free_dense, (counts_init, counts_init)
            )
            free_mask_global = (
                (tot_all > 0)
                & (tot_all <= free_max_rows)
                & (drp_all == tot_all)
            )

        def _route_chunk_dyn(r0, counts_acc):
            row_s = pl.ds(r0, rb)
            slot_bests, slot_ws, keep_flags = _select_and_filter_chunk_vpu(
                logits_ref[row_s, :].astype(jnp.float32),
                bias_row,
                e_iota,
                num_experts,
                topk=topk,
                route_eps=route_eps,
                min_p_topk=min_p_topk,
            )
            if free_max_rows > 0:
                drop_flags = _droppable_flags_for_chunk(
                    slot_ws,
                    keep_flags,
                    route_eps=route_eps,
                    free_max_w=free_max_w,
                    free_min_slot=free_min_slot,
                )
                if free_mask_global is None:
                    tot_c = counts_init
                    drp_c = counts_init
                    for b_s, k_s, d_s in zip(slot_bests, keep_flags, drop_flags):
                        pk = e_iota == b_s
                        tot_c = tot_c + jnp.sum(
                            jnp.logical_and(pk, k_s).astype(jnp.int32),
                            axis=0,
                            keepdims=True,
                        )
                        drp_c = drp_c + jnp.sum(
                            jnp.logical_and(pk, d_s).astype(jnp.int32),
                            axis=0,
                            keepdims=True,
                        )
                    fmask = (
                        (tot_c > 0)
                        & (tot_c <= free_max_rows)
                        & (drp_c == tot_c)
                    )
                else:
                    fmask = free_mask_global
                for s in range(eff_min_s, topk):
                    freed_e = jnp.any(
                        (e_iota == slot_bests[s]) & fmask, axis=1, keepdims=True
                    )
                    keep_flags[s] = keep_flags[s] & (~(drop_flags[s] & freed_e))
            dense = jnp.zeros((rb, num_experts), jnp.float32)
            hit = jnp.zeros((rb, num_experts), jnp.bool_)
            w_sum = jnp.zeros((rb, 1), jnp.float32)
            for b_s, w_s, k_s in zip(slot_bests, slot_ws, keep_flags):
                pk = jnp.logical_and(e_iota == b_s, k_s)
                dense = jnp.where(pk, w_s, dense)
                hit = jnp.logical_or(hit, pk)
                w_sum = w_sum + jnp.where(k_s, w_s, 0.0)
            w_norm = dense / (w_sum + route_eps)
            if route_scale != 1.0:
                w_norm = w_norm * route_scale
            d_scratch.dense_w_vmem[row_s, :] = w_norm
            return counts_acc + jnp.sum(
                hit.astype(jnp.int32), axis=0, keepdims=True
            )

        if n_chunks == 1:
            counts_acc = _route_chunk_dyn(0, counts_init)
        else:
            counts_acc = lax.fori_loop(
                0,
                n_chunks,
                lambda ci, acc: _route_chunk_dyn(
                    pl.multiple_of(ci * rb, rb), acc
                ),
                counts_init,
            )
    else:

        def _route_chunk(r0, counts_acc):
            row_s = pl.ds(r0, rb)
            lg_raw = logits_ref[row_s, :].astype(jnp.float32)
            lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
            scores = 1.0 / (1.0 + jnp.exp(-lg))
            choice = scores + bias_row
            in_order = jnp.zeros((rb, topk), jnp.float32)
            dense = jnp.zeros((rb, num_experts), jnp.float32)
            hit = jnp.zeros((rb, num_experts), jnp.bool_)

            for slot in range(topk):
                peak = jnp.max(choice, axis=1, keepdims=True)
                best = jnp.min(
                    jnp.where(choice == peak, e_iota, num_experts),
                    axis=1,
                    keepdims=True,
                )
                pick = e_iota == best
                selected = jnp.where(pick, scores, 0.0)
                w_slot = jnp.sum(selected, axis=1, keepdims=True)
                in_order = jnp.where(slots_iota == slot, w_slot, in_order)
                dense = jnp.where(pick, scores, dense)
                hit = jnp.logical_or(hit, jnp.logical_and(pick, scores > 0.0))
                choice = jnp.where(pick, neg_inf, choice)

            w_norm = dense / (jnp.sum(in_order, axis=1, keepdims=True) + route_eps)
            if route_scale != 1.0:
                w_norm = w_norm * route_scale
            d_scratch.dense_w_vmem[row_s, :] = w_norm
            return counts_acc + jnp.sum(
                hit.astype(jnp.int32), axis=0, keepdims=True
            )

        n_chunks = batch_size // rb
        if n_chunks == 1:
            counts_acc = _route_chunk(0, counts_init)
        else:
            counts_acc = lax.fori_loop(
                0,
                n_chunks,
                lambda ci, acc: _route_chunk(pl.multiple_of(ci * rb, rb), acc),
                counts_init,
            )

    d_scratch.counts_vmem[...] = jnp.broadcast_to(
        counts_acc, d_scratch.counts_vmem.shape
    )
    cp = pltpu.make_async_copy(
        d_scratch.counts_vmem, d_scratch.counts_smem, dma_sem
    )
    cp.start()
    cp.wait()


def _read_route_rb16_mode() -> int:
    import os

    raw = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_RB16")
    if raw is not None and raw.strip() != "":
        s = raw.strip().lower()
        if s in ("true", "yes", "on"):
            return 1
        if s in ("false", "no", "off"):
            return 0
        return int(s)
    try:
        from sglang_tpu.envs import envs

        env_v = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_RB16", None)
        if env_v is not None:
            return int(env_v.get())
    except Exception:
        pass
    return 0


def _read_route_fast_flag() -> bool:
    import os

    raw = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    try:
        from sglang_tpu.envs import envs

        env_v = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST", None)
        if env_v is not None:
            return bool(env_v.get())
    except Exception:
        pass
    return False


def _route_vertical_lane_parallel(
    logits_ref,
    bias_ref,
    scratch: RoutingScratch,
    dma_sem,
    *,
    my_owner: jax.Array,
    e_per_owner: int,
    holder_cand_eids: tuple = (),
    precompute_owner_loads: bool = False,
    replica_cost_mode: str = "active",
    replica_tile_m: int = 32,
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    min_p_topk: float = 0.0,
    free_max_rows: int = 0,
    free_max_w: float = 0.06,
    free_min_slot: int = 5,
    free_params_ref=None,
    free_rank_cap: int = 0,
    free_tail_rows: int = 0,
    free_tail_w: float = 0.0,
) -> None:
    """Zero-cross-lane-butterfly, 0.0-ULP bit-exact VPU router for E == 256.

    Processes ``128``-token vertical strips (single pass for ``B <= 128`` or
    ``free_max_rows <= 0``; 2-pass global count + mask when ``B > 128`` and
    ``free_max_rows > 0``). Transposes ``[<=128, 256]`` -> ``[256, 128]`` via
    square ``128x128`` XLU hardware ``vxpose`` so tokens sit in independent lanes
    with zero cross-lane swizzles and zero ``fori_loop`` ``sbr`` drains.
    """
    use_tail_cap = free_rank_cap > 0 and free_tail_rows > 0 and free_tail_w > 0.0
    if free_params_ref is not None:
        eff_min_p = free_params_ref[0].astype(jnp.float32)
        eff_free_rows = jnp.maximum(jnp.int32(0), free_params_ref[1].astype(jnp.int32))
        eff_free_w = free_params_ref[2].astype(jnp.float32)
        eff_free_ms = jnp.maximum(jnp.int32(1), free_params_ref[3].astype(jnp.int32))
        has_free = True
    else:
        eff_min_p = min_p_topk
        eff_free_rows = free_max_rows
        eff_free_w = free_max_w
        eff_free_ms = free_min_slot
        has_free = free_max_rows > 0 or use_tail_cap
    eff_tail_w = (
        jnp.maximum(eff_free_w, jnp.float32(free_tail_w))
        if isinstance(eff_free_w, jax.Array)
        else max(float(eff_free_w), float(free_tail_w))
    )
    batch_size, num_experts = logits_ref.shape
    first = my_owner * e_per_owner
    last_excl = first + e_per_owner
    bias_row = bias_ref[...].reshape(1, num_experts).astype(jnp.float32)
    neg_inf = jnp.float32(-jnp.inf)
    e_base = lax.broadcasted_iota(jnp.int32, (64, 128), 0)
    row16 = lax.broadcasted_iota(jnp.int32, (16, 128), 0)
    first_T = jnp.broadcast_to(first, (1, 128))
    last_excl_T = jnp.broadcast_to(last_excl, (1, 128))
    rep_T = tuple(
        jnp.broadcast_to(jnp.where(eid >= 0, eid, jnp.int32(-1)), (1, 128))
        for eid in holder_cand_eids
    )

    def _load_256x128(t0: int, chunk_rows: int, add_bias: bool) -> tuple[jax.Array, jax.Array]:
        pad_v = neg_inf if add_bias else jnp.float32(0.0)
        if chunk_rows <= 64:
            lg_r = logits_ref[pl.ds(t0, chunk_rows), :].astype(jnp.float32)
            lg_c = jnp.where(jnp.isnan(lg_r), neg_inf, lg_r)
            val = 1.0 / (1.0 + jnp.exp(-lg_c))
            if add_bias:
                val = val + bias_row
            if chunk_rows < 128:
                val = jnp.concatenate(
                    [
                        val,
                        jnp.full(
                            (128 - chunk_rows, num_experts), pad_v, jnp.float32
                        ),
                    ],
                    axis=0,
                )
        else:
            lg0 = logits_ref[pl.ds(t0, 64), :].astype(jnp.float32)
            v0 = 1.0 / (1.0 + jnp.exp(-jnp.where(jnp.isnan(lg0), neg_inf, lg0)))
            lg1 = logits_ref[pl.ds(pl.multiple_of(t0 + 64, 64), 64), :].astype(
                jnp.float32
            )
            v1 = 1.0 / (1.0 + jnp.exp(-jnp.where(jnp.isnan(lg1), neg_inf, lg1)))
            if add_bias:
                v0 = v0 + bias_row
                v1 = v1 + bias_row
            val = jnp.concatenate([v0, v1], axis=0)
        return jnp.transpose(val[:, :128]), jnp.transpose(val[:, 128:])

    def _sum_mask_chunk(m0, m1, m2, m3, chunk_rows: int) -> jax.Array:
        lo = jnp.sum(
            jnp.transpose(jnp.concatenate([m0, m1], axis=0).astype(jnp.int32))[
                :chunk_rows, :
            ],
            axis=0,
            keepdims=True,
        )
        hi = jnp.sum(
            jnp.transpose(jnp.concatenate([m2, m3], axis=0).astype(jnp.int32))[
                :chunk_rows, :
            ],
            axis=0,
            keepdims=True,
        )
        return jnp.concatenate([lo, hi], axis=1)

    def _score_vertical_chunk(t0: int, chunk_rows: int):
        ct_lo, ct_hi = _load_256x128(t0, chunk_rows, True)
        c0, c1 = ct_lo[:64, :], ct_lo[64:, :]
        c2, c3 = ct_hi[:64, :], ct_hi[64:, :]
        del ct_lo, ct_hi

        slot_bests: list[jax.Array] = []
        for _ in range(topk):
            p0 = jnp.max(c0, axis=0, keepdims=True)
            p1 = jnp.max(c1, axis=0, keepdims=True)
            p2 = jnp.max(c2, axis=0, keepdims=True)
            p3 = jnp.max(c3, axis=0, keepdims=True)
            peak = jnp.maximum(jnp.maximum(p0, p1), jnp.maximum(p2, p3))

            b0 = jnp.min(jnp.where(c0 == peak, e_base, num_experts), axis=0, keepdims=True)
            b1 = jnp.min(jnp.where(c1 == peak, e_base + 64, num_experts), axis=0, keepdims=True)
            b2 = jnp.min(jnp.where(c2 == peak, e_base + 128, num_experts), axis=0, keepdims=True)
            b3 = jnp.min(jnp.where(c3 == peak, e_base + 192, num_experts), axis=0, keepdims=True)
            best = jnp.minimum(jnp.minimum(b0, b1), jnp.minimum(b2, b3))
            slot_bests.append(best)

            c0 = jnp.where(e_base == best, neg_inf, c0)
            c1 = jnp.where((e_base + 64) == best, neg_inf, c1)
            c2 = jnp.where((e_base + 128) == best, neg_inf, c2)
            c3 = jnp.where((e_base + 192) == best, neg_inf, c3)
        del c0, c1, c2, c3

        st_lo, st_hi = _load_256x128(t0, chunk_rows, False)
        s0, s1 = st_lo[:64, :], st_lo[64:, :]
        slot_ws = [
            jnp.sum(
                jnp.where(e_base == b, s0, 0.0)
                + jnp.where((e_base + 64) == b, s1, 0.0),
                axis=0,
                keepdims=True,
            )
            for b in slot_bests
        ]
        del st_lo, s0, s1
        s2, s3 = st_hi[:64, :], st_hi[64:, :]
        slot_ws = [
            w
            + jnp.sum(
                jnp.where((e_base + 128) == b, s2, 0.0)
                + jnp.where((e_base + 192) == b, s3, 0.0),
                axis=0,
                keepdims=True,
            )
            for w, b in zip(slot_ws, slot_bests)
        ]
        del st_hi, s2, s3

        keep_flags = _filter_topk_keep_flags(
            slot_ws, route_eps=route_eps, min_p_topk=eff_min_p, topk=topk
        )

        ht0 = jnp.zeros((64, 128), dtype=jnp.bool_)
        ht1 = jnp.zeros((64, 128), dtype=jnp.bool_)
        ht2 = jnp.zeros((64, 128), dtype=jnp.bool_)
        ht3 = jnp.zeros((64, 128), dtype=jnp.bool_)
        if has_free:
            drop_flags = _droppable_flags_for_chunk(
                slot_ws,
                keep_flags,
                route_eps=route_eps,
                free_max_w=eff_free_w,
                free_min_slot=eff_free_ms,
            )
            drop_tail = (
                _droppable_flags_for_chunk(
                    slot_ws,
                    keep_flags,
                    route_eps=route_eps,
                    free_max_w=eff_tail_w,
                    free_min_slot=eff_free_ms,
                )
                if use_tail_cap
                else None
            )
            dt0 = jnp.zeros((64, 128), dtype=jnp.bool_)
            dt1 = jnp.zeros((64, 128), dtype=jnp.bool_)
            dt2 = jnp.zeros((64, 128), dtype=jnp.bool_)
            dt3 = jnp.zeros((64, 128), dtype=jnp.bool_)
            if use_tail_cap:
                tt0 = jnp.zeros((64, 128), dtype=jnp.bool_)
                tt1 = jnp.zeros((64, 128), dtype=jnp.bool_)
                tt2 = jnp.zeros((64, 128), dtype=jnp.bool_)
                tt3 = jnp.zeros((64, 128), dtype=jnp.bool_)
            for idx_s, (b_T, k_T, d_T) in enumerate(zip(slot_bests, keep_flags, drop_flags)):
                bk_T = jnp.where(k_T, b_T, jnp.int32(-1))
                ht0 = ht0 | (e_base == bk_T)
                ht1 = ht1 | ((e_base + 64) == bk_T)
                ht2 = ht2 | ((e_base + 128) == bk_T)
                ht3 = ht3 | ((e_base + 192) == bk_T)
                bd_T = jnp.where(d_T, b_T, jnp.int32(-1))
                dt0 = dt0 | (e_base == bd_T)
                dt1 = dt1 | ((e_base + 64) == bd_T)
                dt2 = dt2 | ((e_base + 128) == bd_T)
                dt3 = dt3 | ((e_base + 192) == bd_T)
                if use_tail_cap:
                    bt_T = jnp.where(drop_tail[idx_s], b_T, jnp.int32(-1))
                    tt0 = tt0 | (e_base == bt_T)
                    tt1 = tt1 | ((e_base + 64) == bt_T)
                    tt2 = tt2 | ((e_base + 128) == bt_T)
                    tt3 = tt3 | ((e_base + 192) == bt_T)
            tot_chunk = _sum_mask_chunk(ht0, ht1, ht2, ht3, chunk_rows)
            drp_chunk = _sum_mask_chunk(dt0, dt1, dt2, dt3, chunk_rows)
            drp_tail_chunk = (
                _sum_mask_chunk(tt0, tt1, tt2, tt3, chunk_rows) if use_tail_cap else None
            )
            del ht0, ht1, ht2, ht3, dt0, dt1, dt2, dt3
            if use_tail_cap:
                del tt0, tt1, tt2, tt3
            return slot_bests, slot_ws, keep_flags, drop_flags, drop_tail, tot_chunk, drp_chunk, drp_tail_chunk

        for b_T, k_T in zip(slot_bests, keep_flags):
            bk_T = jnp.where(k_T, b_T, jnp.int32(-1))
            ht0 = ht0 | (e_base == bk_T)
            ht1 = ht1 | ((e_base + 64) == bk_T)
            ht2 = ht2 | ((e_base + 128) == bk_T)
            ht3 = ht3 | ((e_base + 192) == bk_T)
        tot_chunk = _sum_mask_chunk(ht0, ht1, ht2, ht3, chunk_rows)
        del ht0, ht1, ht2, ht3
        return slot_bests, slot_ws, keep_flags, None, None, tot_chunk, None, None

    def _apply_fmask_to_chunk(slot_bests, keep_flags, drop_flags, fm_lo, fm_hi, drop_tail=None, tm_lo=None, tm_hi=None):
        eff_min_s = 1 if isinstance(eff_free_ms, jax.Array) else max(1, int(eff_free_ms))
        for s in range(eff_min_s, topk):
            b_T = slot_bests[s]
            freed_T = jnp.any(
                ((e_base == b_T) & fm_lo[:64, :])
                | (((e_base + 64) == b_T) & fm_lo[64:, :])
                | (((e_base + 128) == b_T) & fm_hi[:64, :])
                | (((e_base + 192) == b_T) & fm_hi[64:, :]),
                axis=0,
                keepdims=True,
            )
            drop_s = drop_flags[s] & freed_T
            if drop_tail is not None and tm_lo is not None and tm_hi is not None:
                freed_t_T = jnp.any(
                    ((e_base == b_T) & tm_lo[:64, :])
                    | (((e_base + 64) == b_T) & tm_lo[64:, :])
                    | (((e_base + 128) == b_T) & tm_hi[:64, :])
                    | (((e_base + 192) == b_T) & tm_hi[64:, :]),
                    axis=0,
                    keepdims=True,
                )
                drop_s = drop_s | (drop_tail[s] & freed_t_T)
            keep_flags[s] = keep_flags[s] & (~drop_s)

    def _pack_vertical_chunk(t0: int, chunk_rows: int, slot_bests, slot_ws, keep_flags):
        w_sum_T = jnp.zeros((1, 128), dtype=jnp.float32)
        for w_T, k_T in zip(slot_ws, keep_flags):
            w_sum_T = w_sum_T + jnp.where(k_T, w_T, 0.0)

        pack_ids_T = jnp.zeros((16, 128), dtype=jnp.int32)
        w_loc_raw_T = jnp.zeros((16, 128), dtype=jnp.float32)
        loc_cnt_T = jnp.zeros((1, 128), dtype=jnp.int32)

        for b_T, w_T, k_T in zip(slot_bests, slot_ws, keep_flags):
            cand_T = jnp.where(
                (b_T >= first_T) & (b_T < last_excl_T),
                b_T - first_T,
                jnp.int32(-1),
            )
            for r_slot, rv in enumerate(rep_T):
                cand_T = jnp.where(
                    b_T == rv, jnp.int32(e_per_owner + r_slot), cand_T
                )
            is_cand_T = k_T & (cand_T >= 0)
            id_row = jnp.int32(1) + loc_cnt_T
            pack_ids_T = jnp.where(is_cand_T & (row16 == id_row), cand_T, pack_ids_T)
            w_loc_raw_T = jnp.where(
                is_cand_T & (row16 == loc_cnt_T), w_T, w_loc_raw_T
            )
            loc_cnt_T = loc_cnt_T + is_cand_T.astype(jnp.int32)

        pack_ids_T = jnp.where(row16 == 0, loc_cnt_T, pack_ids_T)
        w_loc_norm_T = w_loc_raw_T / (w_sum_T + route_eps)
        if route_scale != 1.0:
            w_loc_norm_T = w_loc_norm_T * route_scale
        w_bits_T = lax.bitcast_convert_type(w_loc_norm_T, jnp.int32)
        full_pack_T = jnp.concatenate(
            [pack_ids_T, w_bits_T, jnp.zeros((96, 128), dtype=jnp.int32)], axis=0
        )
        scratch.route_pack_vmem[pl.ds(t0, chunk_rows), :] = jnp.transpose(full_pack_T)[
            :chunk_rows, :
        ]

    def _compute_free_masks(tot_all, drp_all, drp_tail_all=None):
        bmask, tmask = _compute_load_aware_free_masks(
            tot_all,
            drp_all,
            drp_tail_all,
            free_max_rows=eff_free_rows,
            num_owners=num_experts // e_per_owner,
            num_experts=num_experts,
            free_rank_cap=free_rank_cap if use_tail_cap else 0,
            free_tail_rows=free_tail_rows if use_tail_cap else 0,
        )
        fmask = bmask if tmask is None else (bmask | tmask)
        counts_acc = jnp.where(fmask, jnp.int32(0), tot_all)
        fm_lo = jnp.transpose(
            jnp.broadcast_to(bmask[:, :128].astype(jnp.int32), (128, 128))
        ) != 0
        fm_hi = jnp.transpose(
            jnp.broadcast_to(bmask[:, 128:].astype(jnp.int32), (128, 128))
        ) != 0
        if tmask is None:
            return counts_acc, fm_lo, fm_hi, None, None
        tm_lo = jnp.transpose(
            jnp.broadcast_to(tmask[:, :128].astype(jnp.int32), (128, 128))
        ) != 0
        tm_hi = jnp.transpose(
            jnp.broadcast_to(tmask[:, 128:].astype(jnp.int32), (128, 128))
        ) != 0
        return counts_acc, fm_lo, fm_hi, tm_lo, tm_hi

    if batch_size <= 128:
        s_bests, s_ws, k_flags, d_flags, dt_flags, tot_all, drp_all, drp_tail_all = _score_vertical_chunk(
            0, batch_size
        )
        if has_free:
            counts_acc, fm_lo, fm_hi, tm_lo, tm_hi = _compute_free_masks(
                tot_all, drp_all, drp_tail_all
            )
            _apply_fmask_to_chunk(
                s_bests, k_flags, d_flags, fm_lo, fm_hi, dt_flags, tm_lo, tm_hi
            )
        else:
            counts_acc = tot_all
        _pack_vertical_chunk(0, batch_size, s_bests, s_ws, k_flags)
    else:
        chunks = [
            (t0, min(128, batch_size - t0)) for t0 in range(0, batch_size, 128)
        ]
        tot_all = jnp.zeros((1, num_experts), dtype=jnp.int32)
        drp_all = (
            jnp.zeros((1, num_experts), dtype=jnp.int32)
            if has_free
            else None
        )
        drp_tail_all = (
            jnp.zeros((1, num_experts), dtype=jnp.int32)
            if (has_free and use_tail_cap)
            else None
        )
        can_cache_vmem = (
            has_free
            and (not use_tail_cap)
            and ((2 * topk) % 8 == 0)
            and all(c_rows >= 2 * topk for _, c_rows in chunks)
        )
        def _stash_chunk_to_vmem(t0, s_bests, s_ws, k_flags, d_flags):
            # Stash packed chunk routing in unused route_pack_vmem[t0:t0+2*topk, :]
            # so Pass 2 avoids re-reading logits/bias or re-running sigmoid/top-k.
            b_rows = [
                s_bests[s]
                | (k_flags[s].astype(jnp.int32) << 10)
                | (d_flags[s].astype(jnp.int32) << 11)
                for s in range(topk)
            ]
            w_rows = [
                lax.bitcast_convert_type(s_ws[s], jnp.int32) for s in range(topk)
            ]
            scratch.route_pack_vmem[pl.ds(t0, 2 * topk), :] = jnp.concatenate(
                b_rows + w_rows, axis=0
            )

        def _load_chunk_from_vmem(t0):
            st = scratch.route_pack_vmem[pl.ds(t0, 2 * topk), :]
            s_bests = [st[s : s + 1, :] & jnp.int32(0x3FF) for s in range(topk)]
            k_flags = [
                ((st[s : s + 1, :] >> 10) & jnp.int32(1)) != 0 for s in range(topk)
            ]
            d_flags = [
                ((st[s : s + 1, :] >> 11) & jnp.int32(1)) != 0 for s in range(topk)
            ]
            s_ws = [
                lax.bitcast_convert_type(
                    st[topk + s : topk + s + 1, :], jnp.float32
                )
                for s in range(topk)
            ]
            return s_bests, s_ws, k_flags, d_flags

        if can_cache_vmem and batch_size % 128 == 0:
            # Emit 1 loop body instead of 4 unrolled _score_vertical_chunk copies so
            # B>=256 FREE keeps compiled IMEM/HBM compact across all 62 layers (#3513).
            n_chunks = batch_size // 128

            def _pass1_loop(ci, carry):
                tot_acc, drp_acc = carry
                t0 = pl.multiple_of(ci * 128, 128)
                s_bests, s_ws, k_flags, d_flags, _, tot_c, drp_c, _ = _score_vertical_chunk(
                    t0, 128
                )
                _stash_chunk_to_vmem(t0, s_bests, s_ws, k_flags, d_flags)
                return tot_acc + tot_c, drp_acc + drp_c

            tot_all, drp_all = lax.fori_loop(
                0, n_chunks, _pass1_loop, (tot_all, drp_all)
            )
            counts_acc, fm_lo, fm_hi, tm_lo, tm_hi = _compute_free_masks(tot_all, drp_all)

            def _pass2_loop(ci, _):
                t0 = pl.multiple_of(ci * 128, 128)
                s_bests, s_ws, k_flags, d_flags = _load_chunk_from_vmem(t0)
                _apply_fmask_to_chunk(s_bests, k_flags, d_flags, fm_lo, fm_hi)
                _pack_vertical_chunk(t0, 128, s_bests, s_ws, k_flags)
                return _

            lax.fori_loop(0, n_chunks, _pass2_loop, jnp.int32(0))
        else:
            for t0, c_rows in chunks:
                s_bests, s_ws, k_flags, d_flags, dt_flags, tot_c, drp_c, drp_tc = _score_vertical_chunk(
                    t0, c_rows
                )
                tot_all = tot_all + tot_c
                if has_free:
                    drp_all = drp_all + drp_c
                    if use_tail_cap:
                        drp_tail_all = drp_tail_all + drp_tc
                    if can_cache_vmem:
                        _stash_chunk_to_vmem(t0, s_bests, s_ws, k_flags, d_flags)
                else:
                    _pack_vertical_chunk(t0, c_rows, s_bests, s_ws, k_flags)
            if has_free:
                counts_acc, fm_lo, fm_hi, tm_lo, tm_hi = _compute_free_masks(
                    tot_all, drp_all, drp_tail_all
                )
                for t0, c_rows in chunks:
                    if can_cache_vmem:
                        s_bests, s_ws, k_flags, d_flags = _load_chunk_from_vmem(t0)
                        dt_flags = None
                    else:
                        s_bests, s_ws, k_flags, d_flags, dt_flags, _, _, _ = _score_vertical_chunk(
                            t0, c_rows
                        )
                    _apply_fmask_to_chunk(
                        s_bests, k_flags, d_flags, fm_lo, fm_hi, dt_flags, tm_lo, tm_hi
                    )
                    _pack_vertical_chunk(t0, c_rows, s_bests, s_ws, k_flags)
            else:
                counts_acc = tot_all

    if precompute_owner_loads:
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            precompute_imbalance_loads_vpu,
        )

        scratch.counts_vmem[...] = precompute_imbalance_loads_vpu(
            counts_acc,
            num_owners=num_experts // e_per_owner,
            cost_mode=replica_cost_mode,
            tile_m=replica_tile_m,
        )
    else:
        scratch.counts_vmem[...] = jnp.broadcast_to(
            counts_acc, scratch.counts_vmem.shape
        )
    cp_counts = pltpu.make_async_copy(
        scratch.counts_vmem, scratch.counts_smem, dma_sem
    )
    cp_pack = pltpu.make_async_copy(
        scratch.route_pack_vmem, scratch.route_pack_smem, dma_sem
    )
    cp_counts.start()
    cp_pack.start()
    cp_counts.wait()
    cp_pack.wait()


def route_topk_and_counts_in_kernel(
    logits_ref,
    bias_ref,
    scratch: RoutingScratch,
    dma_sem,
    *,
    my_owner: jax.Array,
    e_per_owner: int,
    holder_cand_eids: tuple = (),
    hoist_holder_vecs: bool = False,
    precompute_owner_loads: bool = False,
    replica_cost_mode: str = "active",
    replica_tile_m: int = 32,
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    route_chunk_rows: int = 32,
    min_p_topk: float = 0.0,
    free_max_rows: int = 0,
    free_max_w: float = 0.06,
    free_min_slot: int = 5,
    route_rb16_mode: int | None = None,
    route_fast: bool | None = None,
    free_params_ref=None,
    free_rank_cap: int = 0,
    free_tail_rows: int = 0,
    free_tail_w: float = 0.0,
) -> None:
    """VPU-filtered top-k routing compacting local-candidate hits per token in VREGs.

    Layout of ``scratch.route_pack_smem[t, :]``:
      - ``col 0``: ``loc_cnt`` (``0..topk``) — number of hits on ``my_owner``'s
        candidate experts (home ``[first, first + e_per_owner)`` + ``R`` holder slots).
      - ``cols 1..8``: compacted candidate slot index ``0 .. e_per_owner + R - 1``.
      - ``cols 16..23``: IEEE-754 int32 bits of normalized FP32 weights for those
        compacted local hits (normalized over ALL ``topk`` selected experts!).
    """
    if _bridge1_prerouted_active():
        _route_prerouted_topk_in_kernel(
            logits_ref, scratch, dma_sem,
            my_owner=my_owner, e_per_owner=e_per_owner,
            holder_cand_eids=holder_cand_eids,
            hoist_holder_vecs=hoist_holder_vecs,
            precompute_owner_loads=precompute_owner_loads,
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=replica_tile_m,
            topk=topk, route_chunk_rows=route_chunk_rows,
        )
        return
    batch_size, num_experts = logits_ref.shape
    use_rfast = _read_route_fast_flag() if route_fast is None else bool(route_fast)
    if (
        use_rfast
        and batch_size >= 8
        and (
            (batch_size <= 64 and batch_size % 8 == 0)
            or (batch_size % 64 == 0)
        )
        and num_experts == 256
    ):
        _route_vertical_lane_parallel(
            logits_ref,
            bias_ref,
            scratch,
            dma_sem,
            my_owner=my_owner,
            e_per_owner=e_per_owner,
            holder_cand_eids=holder_cand_eids,
            precompute_owner_loads=precompute_owner_loads,
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=replica_tile_m,
            topk=topk,
            route_eps=route_eps,
            route_scale=route_scale,
            min_p_topk=min_p_topk,
            free_max_rows=free_max_rows,
            free_max_w=free_max_w,
            free_min_slot=free_min_slot,
            free_params_ref=free_params_ref,
            free_rank_cap=free_rank_cap,
            free_tail_rows=free_tail_rows,
            free_tail_w=free_tail_w,
        )
        return
    use_tail_cap = free_rank_cap > 0 and free_tail_rows > 0 and free_tail_w > 0.0
    if free_params_ref is not None:
        eff_min_p_fb = free_params_ref[0].astype(jnp.float32)
        eff_free_rows_fb = jnp.maximum(jnp.int32(0), free_params_ref[1].astype(jnp.int32))
        eff_free_w_fb = free_params_ref[2].astype(jnp.float32)
        eff_free_ms_fb = jnp.maximum(jnp.int32(1), free_params_ref[3].astype(jnp.int32))
        has_free_fb = True
        has_dyn_prune = True
    else:
        eff_min_p_fb = min_p_topk
        eff_free_rows_fb = free_max_rows
        eff_free_w_fb = free_max_w
        eff_free_ms_fb = free_min_slot
        has_free_fb = free_max_rows > 0 or use_tail_cap
        has_dyn_prune = (min_p_topk > 0.0) or has_free_fb
    eff_tail_w_fb = (
        jnp.maximum(eff_free_w_fb, jnp.float32(free_tail_w))
        if isinstance(eff_free_w_fb, jax.Array)
        else max(float(eff_free_w_fb), float(free_tail_w))
    )
    rb16_mode = _read_route_rb16_mode() if route_rb16_mode is None else int(route_rb16_mode)
    if rb16_mode > 0:
        eff_chunk = (
            min(route_chunk_rows, 16) if has_dyn_prune else route_chunk_rows
        )
    else:
        eff_chunk = (
            max(route_chunk_rows, min(batch_size, 64))
            if (has_free_fb and batch_size <= 64)
            else route_chunk_rows
        )
    rb = min(batch_size, eff_chunk)
    if batch_size % rb != 0:
        rb = 8 if batch_size % 8 == 0 else batch_size

    first = my_owner * e_per_owner
    last_excl = first + e_per_owner
    num_rep = len(holder_cand_eids)

    bias_row = bias_ref[...].reshape(1, num_experts).astype(jnp.float32)
    e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
    col128 = lax.broadcasted_iota(jnp.int32, (rb, 128), 1)
    counts_init = jnp.zeros((1, num_experts), jnp.int32)
    neg_inf = jnp.float32(-jnp.inf)

    if rb16_mode > 0 and has_free_fb and not use_tail_cap:
        # Opt-in 2-pass VMEM-staged routing over spill-free <=16-row chunks:
        # Pass 1 evaluates sigmoid + Top-8 ONCE per chunk into scratch.route_pack_vmem;
        # Pass 2 masks freed experts and normalizes weights without re-reading logits!
        n_chunks = batch_size // rb
        first_v = jnp.broadcast_to(first, (rb, 1))
        last_excl_v = jnp.broadcast_to(last_excl, (rb, 1))
        rep_vecs = tuple(
            jnp.broadcast_to(jnp.where(eid >= 0, eid, jnp.int32(-1)), (rb, 1))
            for eid in holder_cand_eids
        )

        def _stage_free_chunk(r0, carry):
            tot_c, drp_c = carry
            row_s = pl.ds(r0, rb)
            sb, sw, kf = _select_and_filter_chunk_vpu(
                logits_ref[row_s, :].astype(jnp.float32),
                bias_row,
                e_iota,
                num_experts,
                topk=topk,
                route_eps=route_eps,
                min_p_topk=eff_min_p_fb,
            )
            df = _droppable_flags_for_chunk(
                sw,
                kf,
                route_eps=route_eps,
                free_max_w=eff_free_w_fb,
                free_min_slot=eff_free_ms_fb,
            )
            stage_meta = jnp.zeros((rb, 128), dtype=jnp.int32)
            stage_w = jnp.zeros((rb, 128), dtype=jnp.float32)
            for s, (b_s, w_s, k_s, d_s) in enumerate(zip(sb, sw, kf, df)):
                pk = e_iota == b_s
                tot_c = tot_c + jnp.sum(
                    jnp.logical_and(pk, k_s).astype(jnp.int32),
                    axis=0,
                    keepdims=True,
                )
                drp_c = drp_c + jnp.sum(
                    jnp.logical_and(pk, d_s).astype(jnp.int32),
                    axis=0,
                    keepdims=True,
                )
                eid_store = jnp.where(k_s, b_s, jnp.int32(-1))
                drp_store = d_s.astype(jnp.int32)
                w_store = jnp.where(k_s, w_s, jnp.float32(0.0))
                stage_meta = jnp.where(col128 == s, eid_store, stage_meta)
                stage_meta = jnp.where(
                    col128 == (8 + s), drp_store, stage_meta
                )
                stage_w = jnp.where(col128 == (16 + s), w_store, stage_w)
            w_bits = lax.bitcast_convert_type(stage_w, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, stage_meta, w_bits
            )
            return tot_c, drp_c

        if n_chunks == 1:
            tot_all, drp_all = _stage_free_chunk(0, (counts_init, counts_init))
        elif rb16_mode >= 2 and n_chunks <= 4:
            tot_all, drp_all = counts_init, counts_init
            for ci in range(n_chunks):
                tot_all, drp_all = _stage_free_chunk(ci * rb, (tot_all, drp_all))
        else:
            tot_all, drp_all = lax.fori_loop(
                0,
                n_chunks,
                lambda ci, acc: _stage_free_chunk(
                    pl.multiple_of(ci * rb, rb), acc
                ),
                (counts_init, counts_init),
            )

        fmask = (
            (tot_all > 0) & (tot_all <= eff_free_rows_fb) & (drp_all == tot_all)
        )
        counts_acc = jnp.where(fmask, jnp.int32(0), tot_all)
        fmask_bcast = jnp.broadcast_to(fmask, (rb, num_experts))

        def _pack_free_chunk(r0, _):
            row_s = pl.ds(r0, rb)
            staged = scratch.route_pack_vmem[row_s, :]
            staged_f = lax.bitcast_convert_type(staged, jnp.float32)

            pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
            w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
            loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
            w_sum = jnp.zeros((rb, 1), dtype=jnp.float32)

            for s in range(topk):
                best = jnp.sum(
                    jnp.where(col128 == s, staged, jnp.int32(0)),
                    axis=1,
                    keepdims=True,
                )
                drop_s = (
                    jnp.sum(
                        jnp.where(col128 == (8 + s), staged, jnp.int32(0)),
                        axis=1,
                        keepdims=True,
                    )
                    != 0
                )
                w_slot = jnp.sum(
                    jnp.where(col128 == (16 + s), staged_f, jnp.float32(0.0)),
                    axis=1,
                    keepdims=True,
                )
                was_kept = best >= 0
                freed_e = jnp.any(
                    (e_iota == best) & fmask_bcast, axis=1, keepdims=True
                )
                keep_s = was_kept & (~(drop_s & freed_e))
                w_sum = w_sum + jnp.where(keep_s, w_slot, 0.0)

                cand_slot = jnp.where(
                    (best >= first_v) & (best < last_excl_v),
                    best - first_v,
                    jnp.int32(-1),
                )
                for r_slot, rep_v in enumerate(rep_vecs):
                    cand_slot = jnp.where(
                        best == rep_v,
                        jnp.int32(e_per_owner + r_slot),
                        cand_slot,
                    )
                is_cand = keep_s & (cand_slot >= 0)
                id_col = jnp.int32(1) + loc_cnt
                w_col = jnp.int32(16) + loc_cnt
                pack_ids = jnp.where(
                    jnp.logical_and(is_cand, col128 == id_col),
                    cand_slot,
                    pack_ids,
                )
                w_loc_raw = jnp.where(
                    jnp.logical_and(is_cand, col128 == w_col),
                    w_slot,
                    w_loc_raw,
                )
                loc_cnt = loc_cnt + is_cand.astype(jnp.int32)

            pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
            w_loc_norm = w_loc_raw / (w_sum + route_eps)
            if route_scale != 1.0:
                w_loc_norm = w_loc_norm * route_scale
            w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, pack_ids, w_bits
            )
            return _

        if n_chunks == 1:
            _pack_free_chunk(0, jnp.int32(0))
        elif rb16_mode >= 2 and n_chunks <= 4:
            for ci in range(n_chunks):
                _pack_free_chunk(ci * rb, jnp.int32(0))
        else:
            lax.fori_loop(
                0,
                n_chunks,
                lambda ci, c: _pack_free_chunk(
                    pl.multiple_of(ci * rb, rb), c
                ),
                jnp.int32(0),
            )
    elif has_dyn_prune:
        # Exact cand_w9_base (349b604d63) default routing path (rb=64 at B<=64 when free_max_rows>0).
        n_chunks = batch_size // rb
        eff_min_s = 1 if isinstance(eff_free_ms_fb, jax.Array) else max(1, int(eff_free_ms_fb))
        first_v = jnp.broadcast_to(first, (rb, 1))
        last_excl_v = jnp.broadcast_to(last_excl, (rb, 1))
        rep_vecs = tuple(
            jnp.broadcast_to(jnp.where(eid >= 0, eid, jnp.int32(-1)), (rb, 1))
            for eid in holder_cand_eids
        )
        free_mask_global = None
        tail_mask_global = None
        if has_free_fb and n_chunks > 1:

            def _scan_free_sparse(ci, carry):
                tot_c, drp_c, drp_tc = carry
                r0 = pl.multiple_of(ci * rb, rb)
                sb, sw, kf = _select_and_filter_chunk_vpu(
                    logits_ref[pl.ds(r0, rb), :].astype(jnp.float32),
                    bias_row,
                    e_iota,
                    num_experts,
                    topk=topk,
                    route_eps=route_eps,
                    min_p_topk=eff_min_p_fb,
                )
                df = _droppable_flags_for_chunk(
                    sw,
                    kf,
                    route_eps=route_eps,
                    free_max_w=eff_free_w_fb,
                    free_min_slot=eff_free_ms_fb,
                )
                dt = (
                    _droppable_flags_for_chunk(
                        sw,
                        kf,
                        route_eps=route_eps,
                        free_max_w=eff_tail_w_fb,
                        free_min_slot=eff_free_ms_fb,
                    )
                    if use_tail_cap
                    else None
                )
                for idx_s, (b_s, k_s, d_s) in enumerate(zip(sb, kf, df)):
                    pk = e_iota == b_s
                    tot_c = tot_c + jnp.sum(
                        jnp.logical_and(pk, k_s).astype(jnp.int32),
                        axis=0,
                        keepdims=True,
                    )
                    drp_c = drp_c + jnp.sum(
                        jnp.logical_and(pk, d_s).astype(jnp.int32),
                        axis=0,
                        keepdims=True,
                    )
                    if use_tail_cap:
                        drp_tc = drp_tc + jnp.sum(
                            jnp.logical_and(pk, dt[idx_s]).astype(jnp.int32),
                            axis=0,
                            keepdims=True,
                        )
                return tot_c, drp_c, drp_tc

            tot_all, drp_all, drp_tail_all = lax.fori_loop(
                0,
                n_chunks,
                _scan_free_sparse,
                (counts_init, counts_init, counts_init),
            )
            free_mask_global, tail_mask_global = _compute_load_aware_free_masks(
                tot_all,
                drp_all,
                drp_tail_all if use_tail_cap else None,
                free_max_rows=eff_free_rows_fb,
                num_owners=num_experts // e_per_owner,
                num_experts=num_experts,
                free_rank_cap=free_rank_cap if use_tail_cap else 0,
                free_tail_rows=free_tail_rows if use_tail_cap else 0,
            )

        def _route_chunk_sparse_dyn(r0, counts_acc):
            row_s = pl.ds(r0, rb)
            slot_bests, slot_ws, keep_flags = _select_and_filter_chunk_vpu(
                logits_ref[row_s, :].astype(jnp.float32),
                bias_row,
                e_iota,
                num_experts,
                topk=topk,
                route_eps=route_eps,
                min_p_topk=eff_min_p_fb,
            )
            if has_free_fb:
                drop_flags = _droppable_flags_for_chunk(
                    slot_ws,
                    keep_flags,
                    route_eps=route_eps,
                    free_max_w=eff_free_w_fb,
                    free_min_slot=eff_free_ms_fb,
                )
                drop_tail = (
                    _droppable_flags_for_chunk(
                        slot_ws,
                        keep_flags,
                        route_eps=route_eps,
                        free_max_w=eff_tail_w_fb,
                        free_min_slot=eff_free_ms_fb,
                    )
                    if use_tail_cap
                    else None
                )
                if free_mask_global is None:
                    tot_c = counts_init
                    drp_c = counts_init
                    drp_tc = counts_init if use_tail_cap else None
                    for idx_s, (b_s, k_s, d_s) in enumerate(zip(slot_bests, keep_flags, drop_flags)):
                        pk = e_iota == b_s
                        tot_c = tot_c + jnp.sum(
                            jnp.logical_and(pk, k_s).astype(jnp.int32),
                            axis=0,
                            keepdims=True,
                        )
                        drp_c = drp_c + jnp.sum(
                            jnp.logical_and(pk, d_s).astype(jnp.int32),
                            axis=0,
                            keepdims=True,
                        )
                        if use_tail_cap:
                            drp_tc = drp_tc + jnp.sum(
                                jnp.logical_and(pk, drop_tail[idx_s]).astype(jnp.int32),
                                axis=0,
                                keepdims=True,
                            )
                    fmask, tmask = _compute_load_aware_free_masks(
                        tot_c,
                        drp_c,
                        drp_tc,
                        free_max_rows=eff_free_rows_fb,
                        num_owners=num_experts // e_per_owner,
                        num_experts=num_experts,
                        free_rank_cap=free_rank_cap if use_tail_cap else 0,
                        free_tail_rows=free_tail_rows if use_tail_cap else 0,
                    )
                else:
                    fmask = free_mask_global
                    tmask = tail_mask_global
                for s in range(eff_min_s, topk):
                    freed_e = jnp.any(
                        (e_iota == slot_bests[s]) & fmask, axis=1, keepdims=True
                    )
                    drop_s = drop_flags[s] & freed_e
                    if tmask is not None and drop_tail is not None:
                        freed_t = jnp.any(
                            (e_iota == slot_bests[s]) & tmask, axis=1, keepdims=True
                        )
                        drop_s = drop_s | (drop_tail[s] & freed_t)
                    keep_flags[s] = keep_flags[s] & (~drop_s)

            pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
            w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
            loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
            hit = jnp.zeros((rb, num_experts), dtype=jnp.bool_)
            w_sum = jnp.zeros((rb, 1), dtype=jnp.float32)

            for best, w_slot, keep_s in zip(slot_bests, slot_ws, keep_flags):
                pk_kept = jnp.logical_and(e_iota == best, keep_s)
                hit = jnp.logical_or(hit, pk_kept)
                w_sum = w_sum + jnp.where(keep_s, w_slot, 0.0)

                cand_slot = jnp.where(
                    (best >= first_v) & (best < last_excl_v),
                    best - first_v,
                    jnp.int32(-1),
                )
                for r_slot, rep_v in enumerate(rep_vecs):
                    cand_slot = jnp.where(
                        best == rep_v,
                        jnp.int32(e_per_owner + r_slot),
                        cand_slot,
                    )
                is_cand = keep_s & (cand_slot >= 0)
                id_col = jnp.int32(1) + loc_cnt
                w_col = jnp.int32(16) + loc_cnt
                pack_ids = jnp.where(
                    jnp.logical_and(is_cand, col128 == id_col),
                    cand_slot,
                    pack_ids,
                )
                w_loc_raw = jnp.where(
                    jnp.logical_and(is_cand, col128 == w_col),
                    w_slot,
                    w_loc_raw,
                )
                loc_cnt = loc_cnt + is_cand.astype(jnp.int32)

            pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
            w_loc_norm = w_loc_raw / (w_sum + route_eps)
            if route_scale != 1.0:
                w_loc_norm = w_loc_norm * route_scale
            w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, pack_ids, w_bits
            )
            return counts_acc + jnp.sum(
                hit.astype(jnp.int32), axis=0, keepdims=True
            )

        if n_chunks == 1:
            counts_acc = _route_chunk_sparse_dyn(0, counts_init)
        elif (rb16_mode >= 2 or hoist_holder_vecs) and n_chunks <= 4:
            counts_acc = counts_init
            for ci in range(n_chunks):
                counts_acc = _route_chunk_sparse_dyn(ci * rb, counts_acc)
        else:
            counts_acc = lax.fori_loop(
                0,
                n_chunks,
                lambda ci, acc: _route_chunk_sparse_dyn(
                    pl.multiple_of(ci * rb, rb), acc
                ),
                counts_init,
            )
    elif hoist_holder_vecs:
        first_v0 = jnp.broadcast_to(first, (rb, 1))
        last_excl_v0 = jnp.broadcast_to(last_excl, (rb, 1))
        rep_vecs0 = tuple(
            jnp.broadcast_to(jnp.where(eid >= 0, eid, jnp.int32(-1)), (rb, 1))
            for eid in holder_cand_eids
        )

        def _route_chunk_hoisted(
            r0, counts_acc, bias_r, first_c, last_excl_c, rep_vecs_c
        ):
            row_s = pl.ds(r0, rb)
            lg_raw = logits_ref[row_s, :].astype(jnp.float32)
            lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
            scores = 1.0 / (1.0 + jnp.exp(-lg))
            choice = scores + bias_r
            pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
            w_all_8 = jnp.zeros((rb, 128), dtype=jnp.float32)
            w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
            loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
            hit = jnp.zeros((rb, num_experts), dtype=jnp.bool_)

            for slot in range(topk):
                peak = jnp.max(choice, axis=1, keepdims=True)
                best = jnp.min(
                    jnp.where(choice == peak, e_iota, num_experts),
                    axis=1,
                    keepdims=True,
                )
                pick = e_iota == best
                w_slot = jnp.sum(
                    jnp.where(pick, scores, 0.0), axis=1, keepdims=True
                )
                valid = w_slot > 0.0
                hit = jnp.logical_or(hit, jnp.logical_and(pick, scores > 0.0))
                w_all_8 = jnp.where(col128 == slot, w_slot, w_all_8)

                cand_slot = jnp.where(
                    (best >= first_c) & (best < last_excl_c),
                    best - first_c,
                    jnp.int32(-1),
                )
                for r_slot, rep_v in enumerate(rep_vecs_c):
                    cand_slot = jnp.where(
                        best == rep_v,
                        jnp.int32(e_per_owner + r_slot),
                        cand_slot,
                    )

                is_cand = valid & (cand_slot >= 0)
                id_col = jnp.int32(1) + loc_cnt
                w_col = jnp.int32(16) + loc_cnt
                pack_ids = jnp.where(
                    jnp.logical_and(is_cand, col128 == id_col),
                    cand_slot,
                    pack_ids,
                )
                w_loc_raw = jnp.where(
                    jnp.logical_and(is_cand, col128 == w_col),
                    w_slot,
                    w_loc_raw,
                )
                loc_cnt = loc_cnt + is_cand.astype(jnp.int32)
                choice = jnp.where(pick, neg_inf, choice)

            pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
            w_sum = jnp.sum(w_all_8, axis=1, keepdims=True)
            w_loc_norm = w_loc_raw / (w_sum + route_eps)
            if route_scale != 1.0:
                w_loc_norm = w_loc_norm * route_scale
            w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, pack_ids, w_bits
            )
            return counts_acc + jnp.sum(
                hit.astype(jnp.int32), axis=0, keepdims=True
            )

        n_chunks = batch_size // rb
        if n_chunks <= 4:
            counts_acc = counts_init
            for ci in range(n_chunks):
                counts_acc = _route_chunk_hoisted(
                    ci * rb,
                    counts_acc,
                    bias_row,
                    first_v0,
                    last_excl_v0,
                    rep_vecs0,
                )
        else:
            counts_acc, _, _, _, _ = lax.fori_loop(
                0,
                n_chunks,
                lambda ci, carry: (
                    _route_chunk_hoisted(
                        pl.multiple_of(ci * rb, rb),
                        carry[0],
                        carry[1],
                        carry[2],
                        carry[3],
                        carry[4],
                    ),
                    carry[1],
                    carry[2],
                    carry[3],
                    carry[4],
                ),
                (counts_init, bias_row, first_v0, last_excl_v0, rep_vecs0),
            )
    else:

        def _route_chunk(r0, counts_acc):
            row_s = pl.ds(r0, rb)
            lg_raw = logits_ref[row_s, :].astype(jnp.float32)
            lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
            scores = 1.0 / (1.0 + jnp.exp(-lg))
            choice = scores + bias_row
            pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
            w_all_8 = jnp.zeros((rb, 128), dtype=jnp.float32)
            w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
            loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
            hit = jnp.zeros((rb, num_experts), dtype=jnp.bool_)

            for slot in range(topk):
                peak = jnp.max(choice, axis=1, keepdims=True)
                best = jnp.min(
                    jnp.where(choice == peak, e_iota, num_experts),
                    axis=1,
                    keepdims=True,
                )
                pick = e_iota == best
                w_slot = jnp.sum(
                    jnp.where(pick, scores, 0.0), axis=1, keepdims=True
                )
                valid = w_slot > 0.0
                hit = jnp.logical_or(hit, jnp.logical_and(pick, scores > 0.0))
                w_all_8 = jnp.where(col128 == slot, w_slot, w_all_8)

                is_home = jnp.logical_and(
                    valid, jnp.logical_and(best >= first, best < last_excl)
                )
                cand_slot = jnp.where(is_home, best - first, jnp.int32(-1))
                for r_slot in range(num_rep):
                    rep_eid = holder_cand_eids[r_slot]
                    is_rep = jnp.logical_and(
                        valid, jnp.logical_and(rep_eid >= 0, best == rep_eid)
                    )
                    cand_slot = jnp.where(
                        is_rep, jnp.int32(e_per_owner + r_slot), cand_slot
                    )

                is_cand = cand_slot >= 0
                id_col = jnp.int32(1) + loc_cnt
                w_col = jnp.int32(16) + loc_cnt
                pack_ids = jnp.where(
                    jnp.logical_and(is_cand, col128 == id_col),
                    cand_slot,
                    pack_ids,
                )
                w_loc_raw = jnp.where(
                    jnp.logical_and(is_cand, col128 == w_col),
                    w_slot,
                    w_loc_raw,
                )
                loc_cnt = loc_cnt + is_cand.astype(jnp.int32)
                choice = jnp.where(pick, neg_inf, choice)

            pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
            w_sum = jnp.sum(w_all_8, axis=1, keepdims=True)
            w_loc_norm = w_loc_raw / (w_sum + route_eps)
            if route_scale != 1.0:
                w_loc_norm = w_loc_norm * route_scale
            w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, pack_ids, w_bits
            )
            return counts_acc + jnp.sum(
                hit.astype(jnp.int32), axis=0, keepdims=True
            )

        n_chunks = batch_size // rb
        if n_chunks == 1:
            counts_acc = _route_chunk(0, counts_init)
        else:
            counts_acc = lax.fori_loop(
                0,
                n_chunks,
                lambda ci, acc: _route_chunk(pl.multiple_of(ci * rb, rb), acc),
                counts_init,
            )

    if precompute_owner_loads:
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            precompute_imbalance_loads_vpu,
        )

        scratch.counts_vmem[...] = precompute_imbalance_loads_vpu(
            counts_acc,
            num_owners=num_experts // e_per_owner,
            cost_mode=replica_cost_mode,
            tile_m=replica_tile_m,
        )
    else:
        scratch.counts_vmem[...] = jnp.broadcast_to(
            counts_acc, scratch.counts_vmem.shape
        )
    cp_counts = pltpu.make_async_copy(
        scratch.counts_vmem, scratch.counts_smem, dma_sem
    )
    cp_pack = pltpu.make_async_copy(
        scratch.route_pack_vmem, scratch.route_pack_smem, dma_sem
    )
    cp_counts.start()
    cp_pack.start()
    cp_counts.wait()
    cp_pack.wait()


def build_dense_exec_plan_smem(
    my_owner: jax.Array,
    e_per_owner: int,
    num_replicas: int,
    local_suppress_smem,
    active_rep_eid_smem,
    d_scratch: DenseRoutingScratch,
    *,
    spec_slot: jax.Array | None = None,
    fast_plan: bool = False,
) -> jax.Array:
    """Build compact active-expert list in SMEM for dense-rows mode."""
    first = my_owner * e_per_owner
    if fast_plan and e_per_owner <= 64 and num_replicas <= 16:
        if spec_slot is not None:
            use_sp = (spec_slot >= 0) & (spec_slot < e_per_owner)
            safe_sp = jnp.where(use_sp, spec_slot, jnp.int32(0))
            g_sp = first + safe_sp
            cnt_sp = d_scratch.counts_smem[0, g_sp]
            keep_sp = use_sp & (
                jnp.logical_and(cnt_sp > 0, local_suppress_smem[safe_sp] == 0)
                if local_suppress_smem is not None
                else (cnt_sp > 0)
            )
            d_scratch.exec_expert_id[0] = g_sp
            d_scratch.exec_weight_src[0] = jnp.int32(0)
            d_scratch.exec_weight_slot[0] = safe_sp
            k = keep_sp.astype(jnp.int32)
        else:
            use_sp = jnp.bool_(False)
            safe_sp = jnp.int32(-1)
            k = jnp.int32(0)
        for le in range(e_per_owner):
            is_sp = use_sp & (jnp.int32(le) == safe_sp)
            g_e = first + jnp.int32(le)
            cnt = d_scratch.counts_smem[0, g_e]
            keep = (~is_sp) & (
                jnp.logical_and(cnt > 0, local_suppress_smem[le] == 0)
                if local_suppress_smem is not None
                else (cnt > 0)
            )
            d_scratch.exec_expert_id[k] = g_e
            d_scratch.exec_weight_src[k] = jnp.int32(0)
            d_scratch.exec_weight_slot[k] = jnp.int32(le)
            k = k + keep.astype(jnp.int32)
        if num_replicas > 0 and active_rep_eid_smem is not None:
            for slot in range(num_replicas):
                g_e = active_rep_eid_smem[slot]
                has_rep = g_e >= 0
                safe_e = jnp.where(has_rep, g_e, jnp.int32(0))
                cnt = jnp.where(has_rep, d_scratch.counts_smem[0, safe_e], jnp.int32(0))
                active = has_rep & (cnt > 0)
                d_scratch.exec_expert_id[k] = safe_e
                d_scratch.exec_weight_src[k] = jnp.int32(1)
                d_scratch.exec_weight_slot[k] = jnp.int32(slot)
                k = k + active.astype(jnp.int32)
        return k
    k_init = jnp.int32(0)
    if spec_slot is not None:
        use_sp = (spec_slot >= 0) & (spec_slot < e_per_owner)
        safe_sp = jnp.where(use_sp, spec_slot, jnp.int32(0))
        g_sp = first + safe_sp
        cnt_sp = d_scratch.counts_smem[0, g_sp]
        keep_sp = use_sp & (
            jnp.logical_and(cnt_sp > 0, local_suppress_smem[safe_sp] == 0)
            if local_suppress_smem is not None
            else (cnt_sp > 0)
        )

        @pl.when(keep_sp)
        def _add_sp_d():
            d_scratch.exec_expert_id[0] = g_sp
            d_scratch.exec_weight_src[0] = jnp.int32(0)
            d_scratch.exec_weight_slot[0] = safe_sp

        k_init = keep_sp.astype(jnp.int32)
    else:
        use_sp = jnp.bool_(False)
        safe_sp = jnp.int32(-1)

    def _scan_home(le, k):
        is_sp = use_sp & (le == safe_sp)
        g_e = first + le
        cnt = d_scratch.counts_smem[0, g_e]
        keep = (~is_sp) & (
            jnp.logical_and(cnt > 0, local_suppress_smem[le] == 0)
            if local_suppress_smem is not None
            else (cnt > 0)
        )

        @pl.when(keep)
        def _add():
            d_scratch.exec_expert_id[k] = g_e
            d_scratch.exec_weight_src[k] = jnp.int32(0)
            d_scratch.exec_weight_slot[k] = le

        return k + keep.astype(jnp.int32)

    num_exec = lax.fori_loop(0, e_per_owner, _scan_home, k_init)
    if num_replicas > 0 and active_rep_eid_smem is not None:

        def _scan_rep(slot, k):
            g_e = active_rep_eid_smem[slot]
            has_rep = g_e >= 0
            safe_e = jnp.where(has_rep, g_e, jnp.int32(0))
            cnt = jnp.where(has_rep, d_scratch.counts_smem[0, safe_e], jnp.int32(0))
            active = jnp.logical_and(has_rep, cnt > 0)

            @pl.when(active)
            def _add_rep():
                d_scratch.exec_expert_id[k] = safe_e
                d_scratch.exec_weight_src[k] = jnp.int32(1)
                d_scratch.exec_weight_slot[k] = slot

            return k + active.astype(jnp.int32)

        num_exec = lax.fori_loop(0, num_replicas, _scan_rep, num_exec)
    return num_exec


def build_default_exec_plan_smem(
    my_owner: jax.Array,
    e_per_owner: int,
    scratch: RoutingScratch,
    *,
    num_experts: int = 256,
    spec_slot: jax.Array | None = None,
    fast_plan: bool = False,
) -> jax.Array:
    """Build compact active-expert plan in SMEM in ONLY ``e_per_owner`` (32) steps!"""
    del num_experts
    first = my_owner * e_per_owner
    if fast_plan and e_per_owner <= 64:
        if spec_slot is not None:
            use_sp = (spec_slot >= 0) & (spec_slot < e_per_owner)
            safe_sp = jnp.where(use_sp, spec_slot, jnp.int32(0))
            g_sp = first + safe_sp
            cnt_sp = scratch.counts_smem[0, g_sp]
            act_sp = use_sp & (cnt_sp > 0)
            scratch.exec_expert_id[0] = g_sp
            scratch.exec_weight_src[0] = jnp.int32(0)
            scratch.exec_weight_slot[0] = safe_sp
            scratch.exec_start[0] = jnp.int32(0)
            scratch.exec_write_ptr[0] = jnp.int32(0)
            k = act_sp.astype(jnp.int32)
            cursor = jnp.where(act_sp, cnt_sp, jnp.int32(0))
        else:
            use_sp = jnp.bool_(False)
            safe_sp = jnp.int32(-1)
            act_sp = jnp.bool_(False)
            k = jnp.int32(0)
            cursor = jnp.int32(0)
        for le in range(e_per_owner):
            is_sp = use_sp & (jnp.int32(le) == safe_sp)
            g_e = first + jnp.int32(le)
            cnt = scratch.counts_smem[0, g_e]
            active = (~is_sp) & (cnt > 0)
            scratch.cand_exec_idx[le] = jnp.where(
                is_sp,
                jnp.where(act_sp, jnp.int32(0), jnp.int32(-1)),
                jnp.where(active, k, jnp.int32(-1)),
            )
            scratch.exec_expert_id[k] = g_e
            scratch.exec_weight_src[k] = jnp.int32(0)
            scratch.exec_weight_slot[k] = jnp.int32(le)
            scratch.exec_start[k] = cursor
            scratch.exec_write_ptr[k] = cursor
            k = k + active.astype(jnp.int32)
            cursor = cursor + jnp.where(active, cnt, jnp.int32(0))
        scratch.exec_start[k] = cursor
        return k
    k_init = jnp.int32(0)
    cursor_init = jnp.int32(0)
    if spec_slot is not None:
        use_sp = (spec_slot >= 0) & (spec_slot < e_per_owner)
        safe_sp = jnp.where(use_sp, spec_slot, jnp.int32(0))
        g_sp = first + safe_sp
        cnt_sp = scratch.counts_smem[0, g_sp]
        act_sp = use_sp & (cnt_sp > 0)

        @pl.when(use_sp)
        def _init_sp_idx():
            scratch.cand_exec_idx[safe_sp] = jnp.where(
                act_sp, jnp.int32(0), jnp.int32(-1)
            )

        @pl.when(act_sp)
        def _add_sp():
            scratch.exec_expert_id[0] = g_sp
            scratch.exec_weight_src[0] = jnp.int32(0)
            scratch.exec_weight_slot[0] = safe_sp
            scratch.exec_start[0] = jnp.int32(0)
            scratch.exec_write_ptr[0] = jnp.int32(0)

        k_init = act_sp.astype(jnp.int32)
        cursor_init = jnp.where(act_sp, cnt_sp, jnp.int32(0))
    else:
        use_sp = jnp.bool_(False)
        safe_sp = jnp.int32(-1)

    def _scan_local(le, carry):
        k, cursor = carry
        is_sp = use_sp & (le == safe_sp)
        g_e = first + le
        cnt = scratch.counts_smem[0, g_e]
        active = (~is_sp) & (cnt > 0)
        if spec_slot is not None:

            @pl.when(~is_sp)
            def _set_idx():
                scratch.cand_exec_idx[le] = jnp.where(active, k, jnp.int32(-1))

        else:
            scratch.cand_exec_idx[le] = jnp.where(active, k, jnp.int32(-1))

        @pl.when(active)
        def _add():
            scratch.exec_expert_id[k] = g_e
            scratch.exec_weight_src[k] = jnp.int32(0)
            scratch.exec_weight_slot[k] = le
            scratch.exec_start[k] = cursor
            scratch.exec_write_ptr[k] = cursor

        return (
            k + active.astype(jnp.int32),
            cursor + jnp.where(active, cnt, jnp.int32(0)),
        )

    num_exec, total_rows = lax.fori_loop(
        0, e_per_owner, _scan_local, (k_init, cursor_init)
    )
    scratch.exec_start[num_exec] = total_rows
    return num_exec


def build_exec_plan_from_masks_smem(
    my_owner: jax.Array,
    e_per_owner: int,
    num_replicas: int,
    local_suppress_smem,
    active_rep_eid_smem,
    scratch: RoutingScratch,
    *,
    num_experts: int = 256,
    spec_slot: jax.Array | None = None,
    fast_plan: bool = False,
) -> jax.Array:
    """Build unified active-expert plan in ONLY ``e_per_owner + R`` (<= 36) steps!"""
    del num_experts
    first = my_owner * e_per_owner
    if fast_plan and e_per_owner <= 64 and num_replicas <= 16:
        if spec_slot is not None:
            use_sp = (spec_slot >= 0) & (spec_slot < e_per_owner)
            safe_sp = jnp.where(use_sp, spec_slot, jnp.int32(0))
            g_sp = first + safe_sp
            cnt_sp = scratch.counts_smem[0, g_sp]
            keep_sp = use_sp & (cnt_sp > 0) & (local_suppress_smem[safe_sp] == 0)
            scratch.exec_expert_id[0] = g_sp
            scratch.exec_weight_src[0] = jnp.int32(0)
            scratch.exec_weight_slot[0] = safe_sp
            scratch.exec_start[0] = jnp.int32(0)
            scratch.exec_write_ptr[0] = jnp.int32(0)
            k = keep_sp.astype(jnp.int32)
            cursor = jnp.where(keep_sp, cnt_sp, jnp.int32(0))
        else:
            use_sp = jnp.bool_(False)
            safe_sp = jnp.int32(-1)
            keep_sp = jnp.bool_(False)
            k = jnp.int32(0)
            cursor = jnp.int32(0)
        for le in range(e_per_owner):
            is_sp = use_sp & (jnp.int32(le) == safe_sp)
            g_e = first + jnp.int32(le)
            cnt = scratch.counts_smem[0, g_e]
            keep = (~is_sp) & (cnt > 0) & (local_suppress_smem[le] == 0)
            scratch.cand_exec_idx[le] = jnp.where(
                is_sp,
                jnp.where(keep_sp, jnp.int32(0), jnp.int32(-1)),
                jnp.where(keep, k, jnp.int32(-1)),
            )
            scratch.exec_expert_id[k] = g_e
            scratch.exec_weight_src[k] = jnp.int32(0)
            scratch.exec_weight_slot[k] = jnp.int32(le)
            scratch.exec_start[k] = cursor
            scratch.exec_write_ptr[k] = cursor
            k = k + keep.astype(jnp.int32)
            cursor = cursor + jnp.where(keep, cnt, jnp.int32(0))
        for slot in range(num_replicas):
            g_e = active_rep_eid_smem[slot]
            has_rep = g_e >= 0
            safe_e = jnp.where(has_rep, g_e, jnp.int32(0))
            cnt = jnp.where(has_rep, scratch.counts_smem[0, safe_e], jnp.int32(0))
            active = has_rep & (cnt > 0)
            scratch.cand_exec_idx[e_per_owner + slot] = jnp.where(
                active, k, jnp.int32(-1)
            )
            scratch.exec_expert_id[k] = safe_e
            scratch.exec_weight_src[k] = jnp.int32(1)
            scratch.exec_weight_slot[k] = jnp.int32(slot)
            scratch.exec_start[k] = cursor
            scratch.exec_write_ptr[k] = cursor
            k = k + active.astype(jnp.int32)
            cursor = cursor + jnp.where(active, cnt, jnp.int32(0))
        scratch.exec_start[k] = cursor
        return k
    k_init = jnp.int32(0)
    cursor_init = jnp.int32(0)
    if spec_slot is not None:
        use_sp = (spec_slot >= 0) & (spec_slot < e_per_owner)
        safe_sp = jnp.where(use_sp, spec_slot, jnp.int32(0))
        g_sp = first + safe_sp
        cnt_sp = scratch.counts_smem[0, g_sp]
        keep_sp = use_sp & (cnt_sp > 0) & (local_suppress_smem[safe_sp] == 0)

        @pl.when(use_sp)
        def _init_sp_mask_idx():
            scratch.cand_exec_idx[safe_sp] = jnp.where(
                keep_sp, jnp.int32(0), jnp.int32(-1)
            )

        @pl.when(keep_sp)
        def _add_sp_home():
            scratch.exec_expert_id[0] = g_sp
            scratch.exec_weight_src[0] = jnp.int32(0)
            scratch.exec_weight_slot[0] = safe_sp
            scratch.exec_start[0] = jnp.int32(0)
            scratch.exec_write_ptr[0] = jnp.int32(0)

        k_init = keep_sp.astype(jnp.int32)
        cursor_init = jnp.where(keep_sp, cnt_sp, jnp.int32(0))
    else:
        use_sp = jnp.bool_(False)
        safe_sp = jnp.int32(-1)

    def _scan_home(le, carry):
        k, cursor = carry
        is_sp = use_sp & (le == safe_sp)
        g_e = first + le
        cnt = scratch.counts_smem[0, g_e]
        keep = (~is_sp) & (cnt > 0) & (local_suppress_smem[le] == 0)
        if spec_slot is not None:

            @pl.when(~is_sp)
            def _set_idx_h():
                scratch.cand_exec_idx[le] = jnp.where(keep, k, jnp.int32(-1))

        else:
            scratch.cand_exec_idx[le] = jnp.where(keep, k, jnp.int32(-1))

        @pl.when(keep)
        def _add_home():
            scratch.exec_expert_id[k] = g_e
            scratch.exec_weight_src[k] = jnp.int32(0)
            scratch.exec_weight_slot[k] = le
            scratch.exec_start[k] = cursor
            scratch.exec_write_ptr[k] = cursor

        return (
            k + keep.astype(jnp.int32),
            cursor + jnp.where(keep, cnt, jnp.int32(0)),
        )

    num_exec, total_rows = lax.fori_loop(
        0, e_per_owner, _scan_home, (k_init, cursor_init)
    )

    def _scan_rep(slot, carry):
        k, cursor = carry
        g_e = active_rep_eid_smem[slot]
        has_rep = g_e >= 0
        safe_e = jnp.where(has_rep, g_e, jnp.int32(0))
        cnt = jnp.where(has_rep, scratch.counts_smem[0, safe_e], jnp.int32(0))
        active = jnp.logical_and(has_rep, cnt > 0)
        scratch.cand_exec_idx[e_per_owner + slot] = jnp.where(
            active, k, jnp.int32(-1)
        )

        @pl.when(active)
        def _add_rep():
            scratch.exec_expert_id[k] = safe_e
            scratch.exec_weight_src[k] = jnp.int32(1)
            scratch.exec_weight_slot[k] = slot
            scratch.exec_start[k] = cursor
            scratch.exec_write_ptr[k] = cursor

        return (
            k + active.astype(jnp.int32),
            cursor + jnp.where(active, cnt, jnp.int32(0)),
        )

    num_exec, total_rows = lax.fori_loop(
        0, num_replicas, _scan_rep, (num_exec, total_rows)
    )
    scratch.exec_start[num_exec] = total_rows
    return num_exec


def fallback_plan_preloaded_replicas_smem(
    my_owner: jax.Array,
    e_owner: int,
    num_owners: int,
    num_replicas: int,
    counts_smem,
    replica_meta_smem,
    local_suppress_smem,
    holder_slot_expert_smem,
    owner_load_smem,
    *,
    cost_mode: str = "active",
    tile_m: int = 64,
    base_cost: int = 32,
    row_weight: int = 1,
    moved_flag_smem=None,
    greedy_peak_first: bool = False,
    vpu_precomputed_loads: bool = False,
    num_experts: int = 256,
) -> None:
    """Built-in §4.5b deterministic SMEM planner matching Track 3 contract."""
    del greedy_peak_first

    def _expert_cost(cnt: jax.Array) -> jax.Array:
        if cost_mode == "active":
            return (cnt > 0).astype(jnp.int32)
        if cost_mode == "tiles":
            return jnp.where(cnt > 0, (cnt + tile_m - 1) // tile_m, jnp.int32(0))
        if cost_mode == "weighted":
            return jnp.where(cnt > 0, base_cost + row_weight * cnt, jnp.int32(0))
        raise ValueError(f"Unknown replica cost_mode: {cost_mode!r}")

    if vpu_precomputed_loads and len(counts_smem.shape) == 2:
        for o in range(num_owners):
            owner_load_smem[o] = counts_smem[1, o]
    else:

        def _init_owner(o, _):
            base_e = o * e_owner

            def _sum_e(le, acc):
                return acc + _expert_cost(counts_smem[0, base_e + le])

            owner_load_smem[o] = lax.fori_loop(0, e_owner, _sum_e, jnp.int32(0))
            return _

        lax.fori_loop(0, num_owners, _init_owner, jnp.int32(0))

    moved_is_bitset = (
        moved_flag_smem is not None and int(moved_flag_smem.shape[0]) == 8
    )
    if moved_is_bitset:
        for w in range(8):
            moved_flag_smem[w] = jnp.int32(0)
    elif moved_flag_smem is not None:

        def _clear_moved(e, _):
            moved_flag_smem[e] = jnp.int32(0)
            return _

        lax.fori_loop(0, num_experts, _clear_moved, jnp.int32(0))

    def _clear_sup(le, _):
        local_suppress_smem[le] = jnp.int32(0)
        return _

    lax.fori_loop(0, e_owner, _clear_sup, jnp.int32(0))

    def _clear_rep(slot, _):
        holder_slot_expert_smem[slot] = jnp.int32(-1)
        return _

    lax.fori_loop(0, num_replicas, _clear_rep, jnp.int32(0))

    k_rep = replica_meta_smem.shape[1]

    def _eval_replica(r_idx, _):
        eid = replica_meta_smem[0, r_idx]
        home = replica_meta_smem[1, r_idx]
        holder = replica_meta_smem[2, r_idx]
        slot = replica_meta_smem[3, r_idx]
        valid = eid >= 0
        safe_e = jnp.where(valid, eid, jnp.int32(0))
        safe_home = jnp.where(valid, home, jnp.int32(0))
        safe_holder = jnp.where(valid, holder, jnp.int32(0))
        cnt = jnp.where(valid, counts_smem[0, safe_e], jnp.int32(0))
        cost = _expert_cost(cnt)
        if moved_is_bitset:
            w_idx = lax.shift_right_logical(safe_e, jnp.int32(5))
            b_msk = lax.shift_left(jnp.int32(1), safe_e & jnp.int32(31))
            already = (moved_flag_smem[w_idx] & b_msk) != 0
        elif moved_flag_smem is not None:
            already = moved_flag_smem[safe_e] != 0
        else:
            already = jnp.bool_(False)
        l_home = owner_load_smem[safe_home]
        l_holder = owner_load_smem[safe_holder]
        can_move = jnp.logical_and(
            jnp.logical_and(valid, cost > 0),
            jnp.logical_and(
                jnp.logical_not(already), (l_home - cost) >= (l_holder + cost)
            ),
        )

        @pl.when(can_move)
        def _commit_move():
            owner_load_smem[safe_home] = l_home - cost
            owner_load_smem[safe_holder] = l_holder + cost
            if moved_is_bitset:
                moved_flag_smem[w_idx] = moved_flag_smem[w_idx] | b_msk
            elif moved_flag_smem is not None:
                moved_flag_smem[safe_e] = r_idx + jnp.int32(1)

            @pl.when(safe_home == my_owner)
            def _mark_sup():
                local_suppress_smem[safe_e - my_owner * e_owner] = jnp.int32(1)

            @pl.when(safe_holder == my_owner)
            def _mark_hold():
                holder_slot_expert_smem[slot] = safe_e

        return _

    lax.fori_loop(0, k_rep, _eval_replica, jnp.int32(0))


def run_replica_planner_smem(
    my_owner: jax.Array,
    e_owner: int,
    num_owners: int,
    num_replicas: int,
    counts_smem,
    replica_meta_smem,
    local_suppress_smem,
    holder_slot_expert_smem,
    owner_load_smem,
    *,
    cost_mode: str = "active",
    tile_m: int = 64,
    moved_flag_smem=None,
    vpu_precomputed_loads: bool = False,
    fast_plan: bool = False,
) -> None:
    """Delegate to Track 3's ``plan_preloaded_replicas_smem`` when available."""
    try:
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            plan_preloaded_replicas_smem,
        )
    except ImportError:
        plan_preloaded_replicas_smem = None
    if plan_preloaded_replicas_smem is not None:
        plan_preloaded_replicas_smem(
            my_owner,
            e_owner,
            num_owners,
            num_replicas,
            counts_smem,
            replica_meta_smem,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            cost_mode=cost_mode,
            tile_m=tile_m,
            moved_flag_smem=moved_flag_smem,
            vpu_precomputed_loads=vpu_precomputed_loads,
            fast_unroll=fast_plan,
        )
    else:
        fallback_plan_preloaded_replicas_smem(
            my_owner,
            e_owner,
            num_owners,
            num_replicas,
            counts_smem,
            replica_meta_smem,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            cost_mode=cost_mode,
            tile_m=tile_m,
            moved_flag_smem=moved_flag_smem,
            vpu_precomputed_loads=vpu_precomputed_loads,
        )


def pack_active_tokens_smem(
    batch_size: int,
    scratch: RoutingScratch,
    *,
    has_replicas: bool = False,
    fast_gs: bool = False,
) -> None:
    """Scatter ONLY VPU-compacted local-candidate hits into per-expert SMEM buckets.

    When ``has_replicas=False``, every compacted local hit has ``k >= 0``
    unconditionally, executing ~B branch-free iterations total (~0.4 us at B=256).
    """
    use_branchless_rep = has_replicas and fast_gs
    if use_branchless_rep:
        trash_k = jnp.int32(int(scratch.exec_write_ptr.shape[0]) - 1)
        trash_pos = jnp.int32(int(scratch.packed_tok_ids.shape[0]) - 1)
        scratch.exec_write_ptr[trash_k] = trash_pos

    def _scan_tok(t, _):
        n_loc = scratch.route_pack_smem[t, 0]

        def _scatter_hit(j, _c):
            cand = scratch.route_pack_smem[t, 1 + j]
            k = scratch.cand_exec_idx[cand]
            w_val = lax.bitcast_convert_type(
                scratch.route_pack_smem[t, 16 + j], jnp.float32
            )
            if use_branchless_rep:
                valid_k = k >= 0
                safe_k = jnp.where(valid_k, k, trash_k)
                pos = scratch.exec_write_ptr[safe_k]
                scratch.packed_tok_ids[pos] = t
                scratch.packed_tok_w[pos] = w_val
                scratch.exec_write_ptr[safe_k] = jnp.where(
                    valid_k, pos + jnp.int32(1), trash_pos
                )
            elif has_replicas:

                @pl.when(k >= 0)
                def _store():
                    pos = scratch.exec_write_ptr[k]
                    scratch.packed_tok_ids[pos] = t
                    scratch.packed_tok_w[pos] = w_val
                    scratch.exec_write_ptr[k] = pos + jnp.int32(1)

            else:
                pos = scratch.exec_write_ptr[k]
                scratch.packed_tok_ids[pos] = t
                scratch.packed_tok_w[pos] = w_val
                scratch.exec_write_ptr[k] = pos + jnp.int32(1)
            return _c

        lax.fori_loop(0, n_loc, _scatter_hit, jnp.int32(0))
        return _

    lax.fori_loop(0, batch_size, _scan_tok, jnp.int32(0))


def _bridge1_prerouted_active() -> bool:
    """True when Bridge 1b packs pre-routed Top-8 + expert counts into ``logits [B, 256]``."""
    try:
        from sglang_tpu.envs import envs

        f_topk = getattr(envs, "SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK", None)
        col_br = getattr(envs, "SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES", None)
        return bool(
            f_topk is not None
            and f_topk.get()
            and col_br is not None
            and int(col_br.get()) in (1, 3, 4)
        )
    except Exception:
        return False


def _unpack_prerouted_counts_row(logits_ref) -> jax.Array:
    """Extract ``[1, 256] int32`` global expert hit counts from ``logits_ref[0:8, 128:256]``."""
    hdr = logits_ref[pl.ds(0, 8), pl.ds(128, 128)].astype(jnp.int32)
    return jnp.concatenate([hdr[0:1, :], hdr[1:2, :]], axis=1)


def _route_prerouted_dense_in_kernel(
    logits_ref,
    d_scratch: DenseRoutingScratch,
    dma_sem,
    *,
    topk: int = 8,
    route_chunk_rows: int = 32,
) -> None:
    """Fast dense-rows unpack from Bridge-1b packed ``[B, 256]`` (skips sigmoid + Top-8!)."""
    batch_size, num_experts = logits_ref.shape
    counts_acc = _unpack_prerouted_counts_row(logits_ref)
    d_scratch.counts_vmem[...] = jnp.broadcast_to(
        counts_acc, d_scratch.counts_vmem.shape
    )
    cp = pltpu.make_async_copy(
        d_scratch.counts_vmem, d_scratch.counts_smem, dma_sem
    )
    cp.start()

    rb = min(batch_size, route_chunk_rows)
    if batch_size % rb != 0:
        rb = 8 if batch_size % 8 == 0 else batch_size
    e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
    col128 = lax.broadcasted_iota(jnp.int32, (rb, 128), 1)

    def _unpack_chunk(r0):
        row_s = pl.ds(r0, rb)
        packed = logits_ref[row_s, pl.ds(0, 128)]
        dense = jnp.zeros((rb, num_experts), dtype=jnp.float32)
        for slot in range(topk):
            w_slot = jnp.sum(
                jnp.where(col128 == slot, packed, 0.0), axis=1, keepdims=True
            )
            best = jnp.sum(
                jnp.where(col128 == (8 + slot), packed, 0.0), axis=1, keepdims=True
            ).astype(jnp.int32)
            dense = jnp.where(e_iota == best, w_slot, dense)
        d_scratch.dense_w_vmem[row_s, :] = dense

    n_chunks = batch_size // rb
    if n_chunks == 1:
        _unpack_chunk(0)
    else:
        lax.fori_loop(
            0,
            n_chunks,
            lambda ci, _: (_unpack_chunk(pl.multiple_of(ci * rb, rb)), jnp.int32(0))[1],
            jnp.int32(0),
        )
    cp.wait()


def _route_prerouted_topk_in_kernel(
    logits_ref,
    scratch: RoutingScratch,
    dma_sem,
    *,
    my_owner: jax.Array,
    e_per_owner: int,
    holder_cand_eids: tuple = (),
    hoist_holder_vecs: bool = False,
    precompute_owner_loads: bool = False,
    replica_cost_mode: str = "active",
    replica_tile_m: int = 32,
    topk: int = 8,
    route_chunk_rows: int = 32,
) -> None:
    """32x-narrower local hit compaction from Bridge-1b packed ``[B, 256]`` (overlaps SMEM DMA!)."""
    del hoist_holder_vecs
    batch_size, num_experts = logits_ref.shape
    counts_acc = _unpack_prerouted_counts_row(logits_ref)
    if precompute_owner_loads:
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            precompute_imbalance_loads_vpu,
        )

        scratch.counts_vmem[...] = precompute_imbalance_loads_vpu(
            counts_acc,
            num_owners=num_experts // e_per_owner,
            cost_mode=replica_cost_mode,
            tile_m=replica_tile_m,
        )
    else:
        scratch.counts_vmem[...] = jnp.broadcast_to(
            counts_acc, scratch.counts_vmem.shape
        )
    # Kick off counts_smem DMA BEFORE token-hit compaction so it overlaps 100%!
    cp_counts = pltpu.make_async_copy(
        scratch.counts_vmem, scratch.counts_smem, dma_sem
    )
    cp_counts.start()

    rb = min(batch_size, route_chunk_rows)
    if batch_size % rb != 0:
        rb = 8 if batch_size % 8 == 0 else batch_size
    first = my_owner * e_per_owner
    last_excl = first + e_per_owner
    num_rep = len(holder_cand_eids)
    col128 = lax.broadcasted_iota(jnp.int32, (rb, 128), 1)

    def _compact_chunk(r0):
        row_s = pl.ds(r0, rb)
        packed = logits_ref[row_s, pl.ds(0, 128)]
        pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
        w_loc_norm = jnp.zeros((rb, 128), dtype=jnp.float32)
        loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)

        for slot in range(topk):
            w_slot = jnp.sum(
                jnp.where(col128 == slot, packed, 0.0), axis=1, keepdims=True
            )
            best = jnp.sum(
                jnp.where(col128 == (8 + slot), packed, 0.0), axis=1, keepdims=True
            ).astype(jnp.int32)
            valid = w_slot > 0.0
            is_home = jnp.logical_and(
                valid, jnp.logical_and(best >= first, best < last_excl)
            )
            cand_slot = jnp.where(is_home, best - first, jnp.int32(-1))
            for r_slot in range(num_rep):
                rep_eid = holder_cand_eids[r_slot]
                is_rep = jnp.logical_and(
                    valid, jnp.logical_and(rep_eid >= 0, best == rep_eid)
                )
                cand_slot = jnp.where(
                    is_rep, jnp.int32(e_per_owner + r_slot), cand_slot
                )
            is_cand = cand_slot >= 0
            id_col = jnp.int32(1) + loc_cnt
            w_col = jnp.int32(16) + loc_cnt
            pack_ids = jnp.where(
                jnp.logical_and(is_cand, col128 == id_col), cand_slot, pack_ids
            )
            w_loc_norm = jnp.where(
                jnp.logical_and(is_cand, col128 == w_col), w_slot, w_loc_norm
            )
            loc_cnt = loc_cnt + is_cand.astype(jnp.int32)

        pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
        w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
        scratch.route_pack_vmem[row_s, :] = jnp.where(
            col128 < 16, pack_ids, w_bits
        )

    n_chunks = batch_size // rb
    if n_chunks <= 4:
        for ci in range(n_chunks):
            _compact_chunk(ci * rb)
    else:
        lax.fori_loop(
            0,
            n_chunks,
            lambda ci, _: (_compact_chunk(pl.multiple_of(ci * rb, rb)), jnp.int32(0))[1],
            jnp.int32(0),
        )

    cp_pack = pltpu.make_async_copy(
        scratch.route_pack_vmem, scratch.route_pack_smem, dma_sem
    )
    cp_pack.start()
    cp_counts.wait()
    cp_pack.wait()
