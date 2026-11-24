# SPDX-License-Identifier: Apache-2.0
"""Zero-glue expert load balancing (EPLB) and hot-expert replication for Muse Spark prefill.

Design invariants:
1. Default-off (`SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE == ""`): zero HLO, op-name, or
   parameter-shape changes relative to `ma`.
2. Zero table-gather glue: `wrouter` columns `[8192, 256]`, `moe.gate.weight` rows `[256, 8192]`,
   `moe.e_score_correction_bias` `[256]`, and the base `[32, ...]` expert slots of each rank
   are permuted identically at load / warmup rebalance time so `col_perm[0..R-1]` hold the `R`
   hottest replicated experts and `rank_to_global[r, :32] == col_perm[r*32 : (r+1)*32]` always
   forms a complete bijection of all 256 original experts.
3. Three lossless routing modes:
   - `"perm"` (`R = 0`, `+0 GiB` HBM): tile-aware LPT + 2-opt bin-packing (`0.0 us` runtime glue).
   - `"home"` (`1 <= R <= 8`): routes `id < R` to each token's home chip `chip * (32+R) + 32 + id`
     (`2` vector ops, `0` ICI bytes for replicated experts).
   - `"q_block"` (`1 <= R <= 8`): partitions `[T, topk]` into 64 static blocks (`<= 1` tile/block)
     and assigns full `quantum`-row (`tile_m`-row) tiles across the lightest ranks via a 64-element
     prefix count (`< 5 us` XLA glue, zero partial-tile explosion on uniform/short batches, and
     uniformly spreads padded tail rows).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import jax
    import torch


def _evaluated_signature(fn):
    # This module defers annotations; cpu_export_jax_op needs real jax.Array types, not strings.
    import inspect

    import jax

    fn.__signature__ = inspect.signature(fn, eval_str=True, locals={"jax": jax})
    return fn


logger = logging.getLogger(__name__)

VALID_EPLB_MODES = (
    "",
    "off",
    "perm",
    "stripe",
    "home",
    "q_block",
    "dyn_holder",
    "dyn_split",
)
MAX_EPLB_REPLICAS = 8


def resolve_prefill_eplb_config(mode_raw: str, replicas_raw: int) -> tuple[str, int]:
    """Validate and normalize `(mode, num_replicas)` from env settings."""
    mode = (mode_raw or "").strip().lower()
    if mode in ("", "off", "0", "false", "none"):
        return "", 0
    alias_map = {
        "zg_perm": "perm",
        "zg_stripe": "stripe",
        "zg_home": "home",
        "zg_q_block": "q_block",
        "zg_qblock": "q_block",
        "zg_dyn_holder": "dyn_holder",
        "dynamic_holder": "dyn_holder",
        "zg_dyn_split": "dyn_split",
        "dynamic_split": "dyn_split",
    }
    mode = alias_map.get(mode, mode)
    if mode not in ("perm", "stripe", "home", "q_block", "dyn_holder", "dyn_split"):
        raise ValueError(
            f"SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE={mode_raw!r} must be one of "
            "('', 'perm', 'stripe', 'home', 'q_block', 'dyn_holder', 'dyn_split')"
        )
    if mode == "perm":
        if replicas_raw != 0:
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE='perm' requires "
                f"SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REPLICAS=0, got {replicas_raw}"
            )
        return "perm", 0
    replicas = int(replicas_raw) if int(replicas_raw) > 0 else 4
    if not (1 <= replicas <= MAX_EPLB_REPLICAS):
        raise ValueError(
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REPLICAS must be in "
            f"[1, {MAX_EPLB_REPLICAS}] for mode={mode!r}, got {replicas_raw}"
        )
    return mode, replicas


def compute_zeroglue_eplb_perm(
    expert_counts: Any,
    *,
    ep_size: int = 8,
    num_local_base: int = 32,
    num_replicas: int = 0,
    tile_m: int = 128,
    dual_decode: bool = False,
    decode_counts: Any | None = None,
    decode_batch_size: int = 64,
    top_k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic `(rank_to_global, col_perm)` from `[num_experts]` token counts.

    Guarantees:
    - `col_perm` (`int32[ep_size * num_local_base]`) is a bijection of `0..num_experts-1`.
    - `col_perm[:num_replicas]` are the `num_replicas` highest-count experts (stable tie-break).
    - `rank_to_global` (`int32[ep_size, num_local_base + num_replicas]`) satisfies
      `rank_to_global[r, :num_local_base] == col_perm[r * num_local_base : (r + 1) * num_local_base]`
      and `rank_to_global[r, num_local_base:] == col_perm[:num_replicas]` for every rank `r`.
    When `dual_decode=True`, packs the remaining `num_experts - num_replicas` base slots
    with a 2-key minimax objective jointly balancing prefill non-replicated tile/row load
    and colocated decode active-expert load (accounting for `col_perm[:num_replicas]` on rank 0).
    """
    counts = np.asarray(expert_counts, dtype=np.int64)
    num_experts = ep_size * num_local_base
    if counts.shape != (num_experts,):
        raise ValueError(
            f"expected expert_counts shape ({num_experts},), got {counts.shape}"
        )
    if not (0 <= num_replicas <= min(MAX_EPLB_REPLICAS, num_local_base - 1)):
        raise ValueError(
            f"invalid num_replicas={num_replicas} for num_local_base={num_local_base}"
        )
    if tile_m <= 0:
        raise ValueError(f"tile_m must be positive, got {tile_m}")

    e_local = num_local_base + num_replicas
    order_desc = np.argsort(-counts, kind="stable")
    rep_experts = [int(e) for e in order_desc[:num_replicas]]
    rep_set = set(rep_experts)

    dec_src: np.ndarray | None = None
    if decode_counts is not None:
        dec_src = np.asarray(decode_counts, dtype=np.float64)
        if dec_src.shape != (num_experts,):
            raise ValueError(
                f"expected decode_counts shape ({num_experts},), got {dec_src.shape}"
            )

    if dual_decode and (
        int(counts.sum()) > 0 or (dec_src is not None and float(dec_src.sum()) > 0.0)
    ):
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            _decode_objective_weights,
        )

        w_pre = np.zeros(num_experts, dtype=np.float64)
        tot_pre = float(np.maximum(counts, 0).sum())
        if tot_pre > 0.0:
            ref_hits = 8192.0 * float(max(1, top_k))
            for e in range(num_experts):
                if e not in rep_set and counts[e] > 0:
                    c_ref = ref_hits * (float(counts[e]) / tot_pre)
                    w_pre[e] = 0.5 * (c_ref / float(tile_m)) + 0.5 * np.ceil(
                        c_ref / float(tile_m)
                    )
        d_in = (
            np.maximum(dec_src, 0.0)
            if (dec_src is not None and float(np.maximum(dec_src, 0.0).sum()) > 0.0)
            else np.maximum(counts.astype(np.float64), 0.0)
        )
        w_dec = _decode_objective_weights(
            d_in,
            objective="decode_active",
            calib_batch=decode_batch_size,
            topk=top_k,
            tile_m=32,
        )
        n_pre = (
            w_pre * (float(ep_size) / float(w_pre.sum()))
            if float(w_pre.sum()) > 0.0
            else w_pre
        )
        n_dec = (
            w_dec * (float(ep_size) / float(w_dec.sum()))
            if float(w_dec.sum()) > 0.0
            else w_dec
        )
        rank_bins = [list(rep_experts)] + [[] for _ in range(ep_size - 1)]
        lp = np.zeros(ep_size, dtype=np.float64)
        ld = np.zeros(ep_size, dtype=np.float64)
        for rep_e in rep_experts:
            ld[0] += n_dec[rep_e]
        rem_experts = [e for e in range(num_experts) if e not in rep_set]
        rem_experts.sort(
            key=lambda e: (-max(n_pre[e], n_dec[e]), -(n_pre[e] + n_dec[e]), e)
        )
        for e in rem_experts:
            cands = [r for r in range(ep_size) if len(rank_bins[r]) < num_local_base]
            best_r = min(
                cands,
                key=lambda r: (
                    max(lp[r] + n_pre[e], ld[r] + n_dec[e]),
                    lp[r] + n_pre[e] + ld[r] + n_dec[e],
                    len(rank_bins[r]),
                    r,
                ),
            )
            rank_bins[best_r].append(e)
            lp[best_r] += n_pre[e]
            ld[best_r] += n_dec[e]
        for _ in range(128):
            scores = np.maximum(lp, ld)
            max_r = int(np.argmax(scores))
            cur_peak = float(scores[max_r])
            start_max = num_replicas if max_r == 0 else 0
            best_swap: tuple[float, float, int, int, int] | None = None
            for oth_r in range(ep_size):
                if oth_r == max_r:
                    continue
                start_oth = num_replicas if oth_r == 0 else 0
                cur_sum = float(lp[max_r] + ld[max_r] + lp[oth_r] + ld[oth_r])
                for i in range(start_max, num_local_base):
                    ei = rank_bins[max_r][i]
                    for j in range(start_oth, num_local_base):
                        ej = rank_bins[oth_r][j]
                        nlp_m = lp[max_r] - n_pre[ei] + n_pre[ej]
                        nld_m = ld[max_r] - n_dec[ei] + n_dec[ej]
                        nlp_o = lp[oth_r] - n_pre[ej] + n_pre[ei]
                        nld_o = ld[oth_r] - n_dec[ej] + n_dec[ei]
                        new_peak = max(nlp_m, nld_m, nlp_o, nld_o)
                        gain = cur_peak - new_peak
                        if gain > 1e-9:
                            tie = cur_sum - (nlp_m + nld_m + nlp_o + nld_o)
                            if best_swap is None or (gain, tie) > (
                                best_swap[0],
                                best_swap[1],
                            ):
                                best_swap = (gain, tie, oth_r, i, j)
            if best_swap is None:
                break
            _, _, oth_r, i, j = best_swap
            ei, ej = rank_bins[max_r][i], rank_bins[oth_r][j]
            rank_bins[max_r][i], rank_bins[oth_r][j] = ej, ei
            lp[max_r] += n_pre[ej] - n_pre[ei]
            ld[max_r] += n_dec[ej] - n_dec[ei]
            lp[oth_r] += n_pre[ei] - n_pre[ej]
            ld[oth_r] += n_dec[ei] - n_dec[ej]
    else:
        eff_costs = np.zeros(num_experts, dtype=np.int64)
        for e in range(num_experts):
            if e not in rep_set and counts[e] > 0:
                tiles_e = (int(counts[e]) + tile_m - 1) // tile_m
                eff_costs[e] = tiles_e * tile_m * 1024 + int(counts[e])

        rank_bins = [list(rep_experts)] + [[] for _ in range(ep_size - 1)]
        rank_loads = np.zeros(ep_size, dtype=np.int64)
        non_rep_order = [
            int(e)
            for e in np.argsort(-eff_costs, kind="stable")
            if int(e) not in rep_set
        ]
        for e in non_rep_order:
            candidates = [
                r for r in range(ep_size) if len(rank_bins[r]) < num_local_base
            ]
            best_r = min(
                candidates, key=lambda r: (int(rank_loads[r]), len(rank_bins[r]), r)
            )
            rank_bins[best_r].append(e)
            rank_loads[best_r] += eff_costs[e]

        # 2-opt pairwise swap refinement between heaviest and lightest rank (preserving rep_experts).
        for _ in range(64):
            max_r = int(np.argmax(rank_loads))
            min_r = int(np.argmin(rank_loads))
            if rank_loads[max_r] - rank_loads[min_r] <= tile_m * 1024:
                break
            start_max = num_replicas if max_r == 0 else 0
            start_min = num_replicas if min_r == 0 else 0
            best_gain = 0
            best_pair = None
            for i in range(start_max, num_local_base):
                ei = rank_bins[max_r][i]
                for j in range(start_min, num_local_base):
                    ej = rank_bins[min_r][j]
                    new_peak = max(
                        rank_loads[max_r] - eff_costs[ei] + eff_costs[ej],
                        rank_loads[min_r] - eff_costs[ej] + eff_costs[ei],
                    )
                    gain = int(rank_loads[max_r] - new_peak)
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            ei, ej = rank_bins[max_r][i], rank_bins[min_r][j]
            rank_bins[max_r][i], rank_bins[min_r][j] = ej, ei
            rank_loads[max_r] += eff_costs[ej] - eff_costs[ei]
            rank_loads[min_r] += eff_costs[ei] - eff_costs[ej]

    col_perm = np.zeros(num_experts, dtype=np.int32)
    rank_to_global = np.zeros((ep_size, e_local), dtype=np.int32)
    for r in range(ep_size):
        ordered = (
            rank_bins[0][:num_replicas] + sorted(rank_bins[0][num_replicas:])
            if r == 0
            else sorted(rank_bins[r])
        )
        for s_idx, orig_e in enumerate(ordered):
            col_perm[r * num_local_base + s_idx] = orig_e
            rank_to_global[r, s_idx] = orig_e
        for k_idx, rep_e in enumerate(rep_experts):
            rank_to_global[r, num_local_base + k_idx] = rep_e
    return rank_to_global, col_perm


def remap_zeroglue_virtual_ids(
    global_topk_ids: jax.Array,
    my_id: jax.Array,
    *,
    ep_size: int,
    num_local_experts: int,
    num_base_local: int = 32,
    mode: str = "q_block",
    quantum: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Map router column IDs `[0, ep_size * num_base_local)` to virtual IDs `[0, ep_size * num_local_experts)`."""
    import jax
    import jax.numpy as jnp

    num_replicas = num_local_experts - num_base_local
    t_local = global_topk_ids.shape[0] // ep_size
    topk = global_topk_ids.shape[1]
    if num_replicas <= 0 or mode in ("", "off", "perm", "zg_perm"):
        v_local = jax.lax.dynamic_slice_in_dim(
            global_topk_ids, my_id * t_local, t_local, axis=0
        )
        return global_topk_ids, v_local

    ids_by_chip = global_topk_ids.reshape(ep_size, t_local * topk)
    if (num_base_local & (num_base_local - 1)) == 0:
        shift = num_base_local.bit_length() - 1
        mask = jnp.int32(num_base_local - 1)
        base_owner = jax.lax.shift_right_logical(ids_by_chip, shift)
        base_slot = ids_by_chip & mask
    else:
        base_owner = ids_by_chip // num_base_local
        base_slot = ids_by_chip % num_base_local
    base_v = base_owner * num_local_experts + base_slot

    if mode in ("home", "zg_home"):
        chip_iota = jnp.arange(ep_size, dtype=jnp.int32)[:, None]
        home_v = chip_iota * num_local_experts + (num_base_local + ids_by_chip)
        v_by_chip = jnp.where(ids_by_chip < num_replicas, home_v, base_v)
        v_local = jax.lax.dynamic_index_in_dim(
            v_by_chip, my_id, axis=0, keepdims=False
        ).reshape(t_local, topk)
        return v_by_chip.reshape(global_topk_ids.shape), v_local

    if mode in ("stripe", "zg_stripe"):
        stripe_toks = max(1, quantum)
        tok_idx = jnp.arange(global_topk_ids.shape[0], dtype=jnp.int32)[:, None]
        if (stripe_toks & (stripe_toks - 1)) == 0:
            s_shift = stripe_toks.bit_length() - 1
            stripe_id = jax.lax.shift_right_logical(tok_idx, s_shift)
        else:
            stripe_id = tok_idx // stripe_toks
        if (ep_size & (ep_size - 1)) == 0:
            stripe_r = (stripe_id + global_topk_ids) & jnp.int32(ep_size - 1)
        else:
            stripe_r = (stripe_id + global_topk_ids) % ep_size
        stripe_v = (
            stripe_r * num_local_experts + (num_base_local + global_topk_ids)
        ).reshape(ep_size, t_local * topk)
        v_by_chip = jnp.where(ids_by_chip < num_replicas, stripe_v, base_v)
        v_local = jax.lax.dynamic_index_in_dim(
            v_by_chip, my_id, axis=0, keepdims=False
        ).reshape(t_local, topk)
        return v_by_chip.reshape(global_topk_ids.shape), v_local

    if mode in ("dyn_holder", "zg_dyn_holder", "dynamic_holder", "dyn_split", "zg_dyn_split", "dynamic_split"):
        is_split = mode in ("dyn_split", "zg_dyn_split", "dynamic_split")
        num_experts = ep_size * num_base_local
        iota_e = jnp.arange(num_experts, dtype=jnp.int32)[None, :]
        tok_hits = (global_topk_ids[:, 0:1] == iota_e).astype(jnp.int32)
        for kk in range(1, topk):
            tok_hits = tok_hits + (global_topk_ids[:, kk : kk + 1] == iota_e).astype(
                jnp.int32
            )
        counts_e = jnp.sum(tok_hits, axis=0)
        tiles_raw = jnp.where(counts_e > 0, (counts_e + quantum - 1) // quantum, 0)
        cost_e = jnp.where(counts_e > 0, tiles_raw + 1, 0)
        cur_loads = jnp.sum(cost_e.reshape(ep_size, num_base_local), axis=1)

        total_elems = global_topk_ids.shape[0] * topk
        n_blocks = 64 if (total_elems >= 64 and total_elems % 64 == 0) else ep_size
        blk_elems = total_elems // n_blocks
        ids_blk = ids_by_chip.reshape(n_blocks, blk_elems)
        v_blk = base_v.reshape(n_blocks, blk_elems)

        for k in range(num_replicas):
            ck = cost_e[k]
            tk = tiles_raw[k]
            holder = jnp.int32(1) + jnp.argmin(cur_loads[1:]).astype(jnp.int32)
            whole_ok = (ck > 0) & (cur_loads[0] - ck >= cur_loads[holder] + ck)
            holder_vid = holder * num_local_experts + (num_base_local + k)
            hit_blk = ids_blk == k
            if not is_split:
                target_vid = jnp.where(whole_ok, holder_vid, jnp.int32(k))
                v_blk = jnp.where(hit_blk, target_vid, v_blk)
                delta = jnp.where(whole_ok, ck, jnp.int32(0))
                cur_loads = cur_loads.at[0].add(-delta).at[holder].add(delta)
            else:
                # Exact-count all-rank water-filling with weight-load penalty:
                # keeps cold replicas on home=0 while spreading hot outliers across
                # all lighter ranks so peak matches `q_block` without cold wakeups.
                base_wo_k = cur_loads.at[0].add(-ck)
                eff_base = base_wo_k.at[1:].add(1)
                lo_t = jnp.min(eff_base).astype(jnp.float32)
                hi_t = (jnp.max(eff_base) + tk).astype(jnp.float32)

                def _dyn_bsearch(_, bnd):
                    lb, hb = bnd
                    md = 0.5 * (lb + hb)
                    cp = jnp.sum(jnp.maximum(0.0, md - eff_base.astype(jnp.float32)))
                    return (
                        jnp.where(cp < tk.astype(jnp.float32), md, lb),
                        jnp.where(cp >= tk.astype(jnp.float32), md, hb),
                    )

                _, w_lvl = jax.lax.fori_loop(0, 14, _dyn_bsearch, (lo_t, hi_t))
                rq = jnp.floor(
                    jnp.maximum(0.0, w_lvl - eff_base.astype(jnp.float32))
                ).astype(jnp.int32)
                rem_t = jnp.maximum(0, tk - jnp.sum(rq))
                prio = jnp.argsort(eff_base + rq)
                iprio = jnp.argsort(prio)
                tq = jnp.where(tk > 0, rq + (iprio < rem_t).astype(jnp.int32), 0)
                cum_tq = jnp.cumsum(tq)
                r_iota = jnp.arange(ep_size, dtype=jnp.int32)
                used_wt = (tq > 0).astype(jnp.int32)
                cur_loads = base_wo_k + tq + used_wt

                blk_cnt = jnp.sum(hit_blk.astype(jnp.int32), axis=1)
                cum_blk = jnp.cumsum(blk_cnt) - blk_cnt
                mid_row = cum_blk + jax.lax.shift_right_logical(blk_cnt, 1)
                blk_tile = mid_row // quantum
                rk_b = jnp.sum(
                    (blk_tile[:, None] >= cum_tq[:-1][None, :]).astype(jnp.int32),
                    axis=1,
                )
                vk_b = jnp.where(
                    rk_b == 0,
                    jnp.int32(k),
                    rk_b * num_local_experts + (num_base_local + k),
                )[:, None]
                v_blk = jnp.where(hit_blk & (tk > 0), vk_b, v_blk)

        v_by_chip = v_blk.reshape(ep_size, t_local * topk)
        v_local = jax.lax.dynamic_index_in_dim(
            v_by_chip, my_id, axis=0, keepdims=False
        ).reshape(t_local, topk)
        return v_by_chip.reshape(global_topk_ids.shape), v_local

    if mode not in ("q_block", "zg_q_block", "zg_qblock"):
        raise ValueError(f"Unsupported remap_zeroglue_virtual_ids mode: {mode!r}")

    total_elems = global_topk_ids.shape[0] * topk
    from sglang_tpu.envs import envs as _tpu_envs

    if (
        int(_tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get() or 0) >= 1
        and ep_size == 8
        and num_base_local == 32
        and 1 <= num_replicas <= 8
        and quantum > 0
        and total_elems >= 32768
        and total_elems % (64 * 512) == 0
        and ((t_local * topk) // 128) % 8 == 0
    ):
        from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
            pallas_qblock_remap_virtual_ids,
        )

        return pallas_qblock_remap_virtual_ids(
            global_topk_ids,
            my_id,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            num_base_local=num_base_local,
            topk=topk,
            quantum=quantum,
        )

    n_blocks = 64 if (total_elems >= 64 and total_elems % 64 == 0) else ep_size
    blk_elems = total_elems // n_blocks
    ids_blk = ids_by_chip.reshape(n_blocks, blk_elems)
    base_owner_blk = base_owner.reshape(n_blocks, blk_elems)
    v_blk = base_v.reshape(n_blocks, blk_elems)

    non_rep_owner_blk = jnp.where(ids_blk < num_replicas, jnp.int32(-1), base_owner_blk)
    ranks_1d = jnp.arange(ep_size, dtype=jnp.int32)
    base_rows_per_rank = jnp.sum(
        (non_rep_owner_blk[:, :, None] == ranks_1d[None, None, :]).astype(jnp.int32),
        axis=(0, 1),
    )
    active_e = jnp.array(
        [num_base_local - num_replicas] + [num_base_local] * (ep_size - 1),
        dtype=jnp.int32,
    )
    cur_tiles = (
        base_rows_per_rank + (active_e * quantum) // 2 + quantum - 1
    ) // quantum

    m_1d = ranks_1d + jnp.int32(1)
    for k in range(num_replicas):
        hit_blk = ids_blk == k
        blk_cnt = jnp.sum(hit_blk.astype(jnp.int32), axis=1)  # int32[n_blocks]
        cnt_k = jnp.sum(blk_cnt)
        n_blks = jnp.where(cnt_k > 0, (cnt_k + quantum - 1) // quantum, 0)

        # Closed-form exact integer water-filling across `ep_size` ranks (0 while loops, 0 sort ops).
        ord_ct = jnp.sum(
            (
                (cur_tiles[None, :] < cur_tiles[:, None])
                | (
                    (cur_tiles[None, :] == cur_tiles[:, None])
                    & (ranks_1d[None, :] < ranks_1d[:, None])
                )
            ).astype(jnp.int32),
            axis=1,
        )
        s_tiles = jnp.sum(
            jnp.where(ord_ct[:, None] == ranks_1d[None, :], cur_tiles[:, None], 0),
            axis=0,
        )
        cs_tiles = jnp.cumsum(s_tiles)
        cost = m_1d * s_tiles - cs_tiles
        k_act = jnp.sum((cost <= n_blks).astype(jnp.int32))
        sub_sum = jnp.sum(jnp.where(m_1d == k_act, cs_tiles, 0))
        tot_act = sub_sum + n_blks
        base_lvl = tot_act // k_act
        rem_t = tot_act - base_lvl * k_act
        is_act = (n_blks > 0) & (ord_ct < k_act)
        rq = jnp.where(is_act, base_lvl - cur_tiles, 0)
        act_prefix = jnp.cumsum(is_act.astype(jnp.int32)) - jnp.int32(1)
        tq = rq + (is_act & (act_prefix < rem_t)).astype(jnp.int32)
        cum_tq = jnp.cumsum(tq)
        cur_tiles = cur_tiles + tq

        cum_blk = jnp.cumsum(blk_cnt) - blk_cnt
        mid_row = cum_blk + jax.lax.shift_right_logical(blk_cnt, 1)
        blk_tile = mid_row // quantum
        rk_b = jnp.sum(
            (blk_tile[:, None] >= cum_tq[:-1][None, :]).astype(jnp.int32), axis=1
        )
        vk_b = (rk_b * num_local_experts + (num_base_local + k))[:, None]
        v_blk = jnp.where(hit_blk, vk_b, v_blk)

    v_by_chip = v_blk.reshape(ep_size, t_local * topk)
    v_local = jax.lax.dynamic_index_in_dim(
        v_by_chip, my_id, axis=0, keepdims=False
    ).reshape(t_local, topk)
    return v_by_chip.reshape(global_topk_ids.shape), v_local


def map_torch_fallback_topk_ids(
    topk_ids: torch.Tensor,
    *,
    num_replicas: int,
    ep_size: int = 8,
    num_base_local: int = 32,
) -> torch.Tensor:
    """Map `[num_tokens, top_k]` router column IDs for native fallback / decode when `R > 0`."""
    import torch

    if num_replicas <= 0:
        return topk_ids
    e_loc = num_base_local + num_replicas
    if (num_base_local & (num_base_local - 1)) == 0:
        shift = num_base_local.bit_length() - 1
        base_v = (topk_ids >> shift) * e_loc + (topk_ids & (num_base_local - 1))
    else:
        base_v = (topk_ids // num_base_local) * e_loc + (topk_ids % num_base_local)
    rep_rank = (
        torch.arange(topk_ids.shape[0], device=topk_ids.device, dtype=topk_ids.dtype)[
            :, None
        ]
        % ep_size
    )
    rep_v = rep_rank * e_loc + (num_base_local + topk_ids)
    return torch.where(topk_ids < num_replicas, rep_v, base_v)


def load_eplb_stats_file(
    path_str: str,
    *,
    num_layers: int,
    start_layer: int,
    num_experts: int = 256,
    missing_ok: bool = True,
) -> np.ndarray | None:
    """Load `[num_layers, num_experts]` expert counts from `.json`, `.npy`, or `.pt` if present."""
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        if not missing_ok:
            raise FileNotFoundError(f"Prefill EPLB stats file not found: {path}")
        return None
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            rows = [
                raw.get(str(start_layer + i), raw.get(start_layer + i))
                for i in range(num_layers)
            ]
            if any(r is None for r in rows):
                raise ValueError(
                    f"EPLB stats JSON {path} missing layer keys in [{start_layer}, {start_layer + num_layers})"
                )
            arr = np.asarray(rows, dtype=np.int64)
        else:
            arr = np.asarray(raw, dtype=np.int64)
    else:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        arr = (
            obj.detach().cpu().numpy()
            if hasattr(obj, "detach")
            else np.asarray(obj, dtype=np.int64)
        )
    if arr.shape == (start_layer + num_layers + 1, num_experts) or (
        arr.ndim == 2 and arr.shape[0] > num_layers and arr.shape[1] == num_experts
    ):
        arr = arr[start_layer : start_layer + num_layers]
    if arr.shape != (num_layers, num_experts):
        raise ValueError(
            f"EPLB stats {path} has shape {arr.shape}, expected ({num_layers}, {num_experts})"
        )
    return arr.astype(np.int64)


def save_eplb_stats_file(
    path_str: str, counts: np.ndarray, *, start_layer: int
) -> None:
    """Atomically write `[num_layers, num_experts]` expert counts on rank 0."""
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if path.suffix == ".npy":
        with open(tmp, "wb") as fh:
            np.save(fh, counts.astype(np.int64))
    elif path.suffix == ".json":
        payload = {
            str(start_layer + i): counts[i].astype(np.int64).tolist()
            for i in range(counts.shape[0])
        }
        tmp.write_text(json.dumps(payload, indent=2))
    else:
        import torch

        torch.save(torch.from_numpy(counts.astype(np.int64)), tmp)
    os.replace(tmp, path)


def is_eligible_eplb_calib_batch(
    hists_2d: Any,
    *,
    topk: int = 8,
    min_calib_tokens: int = 256,
    min_distinct_experts: int = 16,
    max_top8_share: float = 0.90,
) -> tuple[bool, str, int]:
    """Validate whether a `[num_layers, num_experts]` batch histogram came from real traffic.

    Filters out:
    1. Short / padded-dummy startup batches (`_precompile_extend` with `seq_lens=1` where
       `active_tokens == bs <= 128`, or HTTP server warmup probes `< min_calib_tokens`).
    2. Degenerate synthetic / warmup inputs (`zeros(T)`, `ones(T)`, or `cp_extend` `[1, 0, ..., 0]`)
       where Layer-0 routing collapses onto `< min_distinct_experts` experts or concentrates
       `> max_top8_share` of all `active_tokens * topk` assignments onto `<= topk` experts.
    """
    arr = np.asarray(hists_2d, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] == 0 or topk <= 0:
        return False, "empty_or_invalid_shape", 0
    row0 = arr[0]
    total_assign = int(row0.sum())
    active_tokens = total_assign // topk
    if active_tokens < max(1, int(min_calib_tokens)):
        return False, "below_min_calib_tokens", active_tokens
    distinct_l0 = int(np.count_nonzero(row0 > 0))
    if distinct_l0 < min_distinct_experts:
        return False, "degenerate_distinct_experts", active_tokens
    k = min(topk, row0.shape[0])
    top_k_sum = int(np.partition(row0, -k)[-k:].sum())
    if top_k_sum > max_top8_share * total_assign:
        return False, "degenerate_top_k_concentration", active_tokens
    return True, "ok", active_tokens


def canonicalize_eplb_batch_counts(
    hists_2d: Any, current_col_perms: Sequence[np.ndarray] | np.ndarray
) -> np.ndarray:
    """Map `[num_layers, num_experts]` counts indexed by current router columns to original expert IDs."""
    arr = np.asarray(hists_2d, dtype=np.int64)
    num_layers, _ = arr.shape
    if len(current_col_perms) != num_layers:
        raise ValueError(
            f"current_col_perms length {len(current_col_perms)} != num_layers {num_layers}"
        )
    orig = np.empty_like(arr)
    for l_idx in range(num_layers):
        perm = np.asarray(current_col_perms[l_idx], dtype=np.intp)
        orig[l_idx, perm] = arr[l_idx]
    return orig


def compute_eplb_tile_skew_stats(
    orig_counts_2d: Any,
    col_perms: Sequence[np.ndarray] | np.ndarray,
    *,
    ep_size: int = 8,
    num_local_base: int = 32,
    num_replicas: int = 0,
    mode: str = "perm",
    tile_m: int = 128,
    window_steps: int = 1,
) -> tuple[float, int]:
    """Return `(mean_layer_max_over_mean_tile_skew, total_max_rank_tiles)` under `col_perms`."""
    orig = np.asarray(orig_counts_2d, dtype=np.int64)
    num_layers, num_experts = orig.shape
    if num_experts != ep_size * num_local_base:
        raise ValueError(
            f"num_experts {num_experts} != ep_size * num_local_base ({ep_size * num_local_base})"
        )
    eff_tile = max(1, int(tile_m) * max(1, int(window_steps)))
    layer_skews: list[float] = []
    total_max_tiles = 0
    is_qblock = mode in ("q_block", "zg_q_block", "zg_qblock")

    for l_idx in range(num_layers):
        perm = np.asarray(col_perms[l_idx], dtype=np.intp)
        col_counts = orig[l_idx, perm]
        rank_tiles = np.zeros(ep_size, dtype=np.int64)
        for r in range(ep_size):
            c_lo = r * num_local_base + (num_replicas if r == 0 else 0)
            c_hi = (r + 1) * num_local_base
            sub = col_counts[c_lo:c_hi]
            pos = sub > 0
            if np.any(pos):
                rank_tiles[r] = int(np.sum((sub[pos] + eff_tile - 1) // eff_tile))
        if num_replicas > 0:
            for k in range(num_replicas):
                ck = int(col_counts[k])
                if ck <= 0:
                    continue
                if is_qblock:
                    n_blks = (ck + eff_tile - 1) // eff_tile
                    for _ in range(n_blks):
                        min_r = int(np.argmin(rank_tiles))
                        rank_tiles[min_r] += 1
                else:
                    per_rank_rows = (ck + ep_size - 1) // ep_size
                    rank_tiles += (per_rank_rows + eff_tile - 1) // eff_tile
        max_t = int(rank_tiles.max())
        mean_t = float(rank_tiles.mean())
        total_max_tiles += max_t
        layer_skews.append(max_t / mean_t if mean_t > 0.0 else 1.0)

    mean_skew = float(np.mean(layer_skews)) if layer_skews else 1.0
    return mean_skew, total_max_tiles


class OnlineEplbController:
    """Filters startup dummy warmups and manages initial + periodic skew-triggered rebalances."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int = 256,
        ep_size: int = 8,
        num_local_base: int = 32,
        num_replicas: int = 0,
        mode: str = "perm",
        calib_forwards: int = 4,
        skip_forwards: int = 0,
        min_calib_tokens: int = 256,
        rebalance_interval: int = 16,
        skew_threshold: float = 1.22,
        hysteresis_ratio: float = 0.95,
        tile_m: int = 128,
        topk: int = 8,
        initial_rebalanced: bool = False,
    ) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.num_local_base = num_local_base
        self.num_replicas = num_replicas
        self.mode = mode
        self.calib_forwards = max(0, int(calib_forwards))
        self.skip_forwards = max(0, int(skip_forwards))
        self.min_calib_tokens = max(1, int(min_calib_tokens))
        self.rebalance_interval = max(0, int(rebalance_interval))
        self.skew_threshold = float(skew_threshold)
        self.hysteresis_ratio = float(hysteresis_ratio)
        self.tile_m = tile_m
        self.topk = topk

        self.window_orig_counts = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.window_steps = 0
        self.eligible_steps = 0
        self.skipped_dummy_steps = 0
        self.skipped_warmup_steps = 0
        self.rebalance_count = 1 if initial_rebalanced else 0
        self.last_skip_reason = ""
        self.last_observed_skew = 1.0

    def observe_batch(
        self,
        hists_2d: Any,
        current_col_perms: Sequence[np.ndarray] | np.ndarray,
        *,
        in_warmup_queue: bool = False,
    ) -> tuple[np.ndarray | None, int]:
        """Consume one `EXTEND` forward's `[num_layers, 256]` histogram across all ranks.

        Returns `(canonical_orig_counts_to_apply, eff_tile_m)` when a rebalance should execute,
        or `(None, self.tile_m)` otherwise.
        """
        if in_warmup_queue:
            self.skipped_dummy_steps += 1
            self.last_skip_reason = "warmup_queue"
            return None, self.tile_m
        ok, reason, _ = is_eligible_eplb_calib_batch(
            hists_2d, topk=self.topk, min_calib_tokens=self.min_calib_tokens
        )
        if not ok:
            self.skipped_dummy_steps += 1
            self.last_skip_reason = reason
            return None, self.tile_m

        self.eligible_steps += 1
        if self.eligible_steps <= self.skip_forwards:
            self.skipped_warmup_steps += 1
            self.last_skip_reason = "skip_forwards"
            return None, self.tile_m

        if self.rebalance_count >= 1 and self.rebalance_interval <= 0:
            return None, self.tile_m

        orig_batch = canonicalize_eplb_batch_counts(hists_2d, current_col_perms)
        self.window_orig_counts += orig_batch
        self.window_steps += 1

        if self.rebalance_count == 0:
            if self.calib_forwards <= 0 or self.window_steps < self.calib_forwards:
                return None, self.tile_m
            w_steps = self.window_steps
            counts_out = self.window_orig_counts.copy()
            self.window_orig_counts.fill(0)
            self.window_steps = 0
            cur_skew, _ = compute_eplb_tile_skew_stats(
                counts_out,
                current_col_perms,
                ep_size=self.ep_size,
                num_local_base=self.num_local_base,
                num_replicas=self.num_replicas,
                mode=self.mode,
                tile_m=self.tile_m,
                window_steps=w_steps,
            )
            self.last_observed_skew = cur_skew
            self.rebalance_count += 1
            return counts_out, self.tile_m * w_steps

        if self.window_steps < self.rebalance_interval:
            return None, self.tile_m

        w_steps = self.window_steps
        counts_cand = self.window_orig_counts.copy()
        self.window_orig_counts.fill(0)
        self.window_steps = 0
        cur_skew, cur_max_tiles = compute_eplb_tile_skew_stats(
            counts_cand,
            current_col_perms,
            ep_size=self.ep_size,
            num_local_base=self.num_local_base,
            num_replicas=self.num_replicas,
            mode=self.mode,
            tile_m=self.tile_m,
            window_steps=w_steps,
        )
        self.last_observed_skew = cur_skew
        if cur_skew <= self.skew_threshold or cur_max_tiles <= 0:
            return None, self.tile_m

        eff_tm = self.tile_m * w_steps
        cand_perms = [
            compute_zeroglue_eplb_perm(
                counts_cand[l_idx],
                ep_size=self.ep_size,
                num_local_base=self.num_local_base,
                num_replicas=self.num_replicas,
                tile_m=eff_tm,
            )[1]
            for l_idx in range(self.num_layers)
        ]
        _, new_max_tiles = compute_eplb_tile_skew_stats(
            counts_cand,
            cand_perms,
            ep_size=self.ep_size,
            num_local_base=self.num_local_base,
            num_replicas=self.num_replicas,
            mode=self.mode,
            tile_m=self.tile_m,
            window_steps=w_steps,
        )
        if new_max_tiles <= self.hysteresis_ratio * cur_max_tiles:
            self.rebalance_count += 1
            return counts_cand, eff_tm
        return None, self.tile_m


def build_eplb_moe_op(
    *,
    h_q_local: int,
    h_kv_local: int,
    fused_glue: bool,
    eplb_mode: str,
    eplb_calib: bool,
    skip_pad_tokens: bool = False,
    router_hilo_pre: bool = False,
    d_tok_tile: int = 64,
    op_extra_tag: str = "",
    axis_name: str = "model",
    next_in_gather_rows_fn=None,
    next_in_gather_static_fp8: bool = False,
    causal_chunk_stream: bool = False,
    causal_chunk_spec: str = "",
):
    """Export a dedicated Torch custom op for prefill MoE when EPLB or skip-pad is enabled.

    ``next_in_gather_rows_fn`` (SGLANG_TPU_MUSE_KERNEL_A1_AG_UNDER_D): rows per rank -> the
    next layer's A1 part 0 rows; the op then returns (stream, gathered part 0, other rows).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_moe_rs_op import _get_mesh
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    if next_in_gather_rows_fn is not None and eplb_calib:
        raise ValueError(
            "A1_AG_UNDER_D / CAUSAL_CHUNK_STREAM does not support online prefill EPLB calibration"
        )
    mesh = _get_mesh()
    glue = "_fg" if fused_glue else ""
    eplb_sfx = f"_eplb_{eplb_mode}" if eplb_mode else ""
    spad_sfx = "_spad" if skip_pad_tokens else ""
    cal_sfx = "_cal" if eplb_calib else ""
    ccs_sfx = (
        "_ccs" + ("_" + causal_chunk_spec.replace(",", "_") if causal_chunk_spec else "")
        if causal_chunk_stream
        else ""
    )
    ccs_op_sfx = ccs_sfx if (ccs_sfx and ccs_sfx not in op_extra_tag) else ""
    op_name = f"sglang_tpu_pallas::muse_prefill_moe{glue}{op_extra_tag}{ccs_op_sfx}{eplb_sfx}{spad_sfx}{cal_sfx}_q{h_q_local}_kv{h_kv_local}"
    has_real_tok_arg = eplb_calib or skip_pad_tokens

    def _moe_body(*xs):
        moe_xs = xs[:22] if has_real_tok_arg else xs
        vt_arg = xs[22] if skip_pad_tokens else None
        rows0 = next_in_gather_rows_fn(xs[0].shape[0]) if next_in_gather_rows_fn else 0
        new_stream, next_attn_in, raw_local_ids, _, *gathered = (
            fused_moe_full_bridge_and_ffn(
                *moe_xs,
                mesh=mesh,
                router_mode="bf16_hilo_pre" if router_hilo_pre else "bf16_hilo",
                b2_layout="nstream",
                b2_two_half=True,
                b2_deq0_first=True,
                hoist_routing=True,
                d_layout="4slab",
                d_tok_tile=d_tok_tile,
                in_shard_map=True,
                b2_deq0_new_buffer=fused_glue,
                eplb_mode=eplb_mode,
                valid_tokens=vt_arg,
                next_in_gather_rows=rows0,
                next_in_gather_static_fp8=next_in_gather_static_fp8,
                causal_chunk_stream=causal_chunk_stream,
                causal_chunk_spec=causal_chunk_spec,
            )
        )
        if rows0:
            return new_stream, gathered[0], next_attn_in
        if not eplb_calib:
            return new_stream, next_attn_in
        real_tokens = xs[22][0]
        my_rank = lax.axis_index(axis_name)
        t_loc = raw_local_ids.shape[0]
        if causal_chunk_stream:
            from sglang_tpu.moe.pallas.fused_moe_rs_op import get_mesh_shape_product
            from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
                chunk_major_global_token_ids_jax,
                resolve_causal_chunks,
            )
            from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
                EXPERT,
                get_moe_expert_axis,
            )

            _ccs_chunks = resolve_causal_chunks(t_loc, causal_chunk_spec)
            _ep_sz = get_mesh_shape_product(mesh, get_moe_expert_axis(mesh, EXPERT))
            _tok_ids_2d = chunk_major_global_token_ids_jax(_ep_sz, _ccs_chunks)
            global_tok_idx = lax.dynamic_slice_in_dim(
                _tok_ids_2d, my_rank, 1, axis=0
            ).reshape(t_loc, 1)
        else:
            global_tok_idx = (
                my_rank * t_loc + jnp.arange(t_loc, dtype=jnp.int32)[:, None]
            )
        active_mask = global_tok_idx < real_tokens
        expert_iota = jnp.arange(256, dtype=jnp.int32)[None, :]
        masked_ids = jnp.where(active_mask, raw_local_ids, jnp.int32(-1)).reshape(-1, 1)
        local_hist = jnp.sum((masked_ids == expert_iota).astype(jnp.int32), axis=0)
        global_hist = lax.psum(local_hist, axis_name=axis_name)
        return new_stream, next_attn_in, global_hist

    n_in = 23 if has_real_tok_arg else 22
    n_out = 3 if (eplb_calib or next_in_gather_rows_fn is not None) else 2
    moe_sm = jax.shard_map(
        _moe_body,
        mesh=mesh,
        in_specs=(P(axis_name),) * n_in,
        out_specs=(P(axis_name),) * n_out,
        check_vma=False,
    )
    if next_in_gather_rows_fn is not None:
        import inspect

        def moe_agd(*xs):
            return moe_sm(*xs)

        # torch.library.custom_op rejects *args: one named Array param per input.
        moe_agd.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    f"x{i}",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=jax.Array,
                )
                for i in range(n_in)
            ],
            return_annotation=tuple[jax.Array, jax.Array, jax.Array],
        )
        return cpu_export_jax_op(
            op_name,
            moe_agd,
            mesh=mesh,
            input_partition_specs=(P(axis_name),) * n_in,
            output_partition_specs=(P(axis_name),) * 3,
        )
    if eplb_calib:

        def moe_cal(
            x0: jax.Array,
            x1: jax.Array,
            x2: jax.Array,
            x3: jax.Array,
            x4: jax.Array,
            x5: jax.Array,
            x6: jax.Array,
            x7: jax.Array,
            x8: jax.Array,
            x9: jax.Array,
            x10: jax.Array,
            x11: jax.Array,
            x12: jax.Array,
            x13: jax.Array,
            x14: jax.Array,
            x15: jax.Array,
            x16: jax.Array,
            x17: jax.Array,
            x18: jax.Array,
            x19: jax.Array,
            x20: jax.Array,
            x21: jax.Array,
            x22: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return moe_sm(
                x0,
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                x7,
                x8,
                x9,
                x10,
                x11,
                x12,
                x13,
                x14,
                x15,
                x16,
                x17,
                x18,
                x19,
                x20,
                x21,
                x22,
            )

        return cpu_export_jax_op(
            op_name,
            _evaluated_signature(moe_cal),
            mesh=mesh,
            input_partition_specs=(P(axis_name),) * 23,
            output_partition_specs=(P(axis_name),) * 3,
        )

    if skip_pad_tokens:

        def moe_spad(
            x0: jax.Array,
            x1: jax.Array,
            x2: jax.Array,
            x3: jax.Array,
            x4: jax.Array,
            x5: jax.Array,
            x6: jax.Array,
            x7: jax.Array,
            x8: jax.Array,
            x9: jax.Array,
            x10: jax.Array,
            x11: jax.Array,
            x12: jax.Array,
            x13: jax.Array,
            x14: jax.Array,
            x15: jax.Array,
            x16: jax.Array,
            x17: jax.Array,
            x18: jax.Array,
            x19: jax.Array,
            x20: jax.Array,
            x21: jax.Array,
            x22: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return moe_sm(
                x0,
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                x7,
                x8,
                x9,
                x10,
                x11,
                x12,
                x13,
                x14,
                x15,
                x16,
                x17,
                x18,
                x19,
                x20,
                x21,
                x22,
            )

        return cpu_export_jax_op(
            op_name,
            _evaluated_signature(moe_spad),
            mesh=mesh,
            input_partition_specs=(P(axis_name),) * 23,
            output_partition_specs=(P(axis_name),) * 2,
        )

    def moe_std(
        x0: jax.Array,
        x1: jax.Array,
        x2: jax.Array,
        x3: jax.Array,
        x4: jax.Array,
        x5: jax.Array,
        x6: jax.Array,
        x7: jax.Array,
        x8: jax.Array,
        x9: jax.Array,
        x10: jax.Array,
        x11: jax.Array,
        x12: jax.Array,
        x13: jax.Array,
        x14: jax.Array,
        x15: jax.Array,
        x16: jax.Array,
        x17: jax.Array,
        x18: jax.Array,
        x19: jax.Array,
        x20: jax.Array,
        x21: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return moe_sm(
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x9,
            x10,
            x11,
            x12,
            x13,
            x14,
            x15,
            x16,
            x17,
            x18,
            x19,
            x20,
            x21,
        )

    return cpu_export_jax_op(
        op_name,
        _evaluated_signature(moe_std),
        mesh=mesh,
        input_partition_specs=(P(axis_name),) * 22,
        output_partition_specs=(P(axis_name),) * 2,
    )


class PrefillEplbManager:
    """Owns pre-allocated `[32 + R, ...]` expert slabs and in-place cross-rank rebalancing."""

    _EXPERT_ATTRS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    # Local experts per all-gather in a rebalance; 8 bounds the scratch to ~2x a local slab.
    _GATHER_CHUNK = 8

    _DECODE_REP_ATTRS = (
        ("w13_weight", "_fused_decode_rep_w1"),
        ("w13_weight_scale", "_fused_decode_rep_s1"),
        ("w2_weight", "_fused_decode_rep_w2"),
        ("w2_weight_scale", "_fused_decode_rep_s2"),
    )
    _DECODE_SHADOW_FP4_ATTRS = (
        ("w13_weight", "_decode_fp4_w1"),
        ("w13_weight_scale", "_decode_fp4_s1"),
        ("w2_weight", "_decode_fp4_w2"),
        ("w2_weight_scale", "_decode_fp4_s2"),
    )

    def __init__(
        self,
        *,
        decoder: Any,
        weights_list: Any,
        gather_fn: Any,
        start: int,
        end: int,
        tp_size: int,
        tp_rank: int,
        mode: str,
        num_replicas: int,
        initial_col_perms: Sequence[np.ndarray] | None = None,
        dual_decode: bool = False,
        decode_counts_by_layer: dict[int, np.ndarray] | None = None,
        num_base_local: int | None = None,
    ) -> None:
        self.decoder = decoder
        self.weights_list = weights_list
        self.gather_fn = gather_fn
        self.start = start
        self.end = end
        self.num_layers = end - start
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.mode = mode
        self.num_replicas = num_replicas
        self.dual_decode = bool(dual_decode)
        self.decode_counts_by_layer = (
            dict(decode_counts_by_layer) if decode_counts_by_layer else {}
        )
        if num_base_local is not None:
            self.num_base_local = int(num_base_local)
        elif start < end and hasattr(decoder, "layers") and len(decoder.layers) > start:
            _exp0 = getattr(getattr(decoder.layers[start], "mlp", None), "experts", None)
            _w13 = getattr(_exp0, "w13_weight", None)
            self.num_base_local = int(_w13.shape[0]) if _w13 is not None else 32
        else:
            self.num_base_local = 32
        self.num_experts = tp_size * self.num_base_local
        self.e_local = self.num_base_local + num_replicas
        if initial_col_perms is not None:
            if len(initial_col_perms) != self.num_layers:
                raise ValueError(
                    f"initial_col_perms length {len(initial_col_perms)} != num_layers {self.num_layers}"
                )
            self.current_col_perms = [
                np.asarray(p, dtype=np.int32).copy() for p in initial_col_perms
            ]
        else:
            self.current_col_perms = [
                np.arange(self.num_experts, dtype=np.int32)
                for _ in range(self.num_layers)
            ]
        if num_replicas > 0:
            self._preallocate_replica_slots()

    def _gather_base_256(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch

        base_slice = tensor[: self.num_base_local]
        if tensor.dtype == torch.float4_e2m1fn_x2:
            return self.gather_fn(base_slice.view(torch.uint8).contiguous())
        return self.gather_fn(base_slice.contiguous())

    def _preallocate_replica_slots(self) -> None:
        """Expand each eligible layer's local expert tensors from `[32, ...]` to `[32+R, ...]` once.

        Preserves the exact identity mapping (`[:32] == [tp_rank*32 : (tp_rank+1)*32]` and
        `[32:32+R] == [0 : R]`) so `self.current_col_perms[offset] == np.arange(256)` holds
        bit-exact before the first online rebalance and across subsequent rebalances.
        """
        import torch

        from sglang.srt.layers.utils import copy_or_rebind_param
        from sglang_tpu.expert_loading import _wait_for
        from sglang_tpu.fp8_moe import _free_replaced_storage

        init_slot_idx = np.concatenate(
            [
                np.arange(
                    self.tp_rank * self.num_base_local,
                    (self.tp_rank + 1) * self.num_base_local,
                    dtype=np.int64,
                ),
                np.arange(self.num_replicas, dtype=np.int64),
            ]
        )
        for offset in range(self.num_layers):
            layer = self.decoder.layers[self.start + offset]
            moe = layer.mlp
            experts = moe.experts
            device = experts.w13_weight.device
            idx_t = torch.tensor(init_slot_idx, dtype=torch.long, device=device)
            for attr in self._EXPERT_ATTRS:
                param = getattr(experts, attr)
                old_data = param.data
                is_fp4 = old_data.dtype == torch.float4_e2m1fn_x2
                all_256 = self._gather_base_256(old_data)
                selected = all_256.index_select(0, idx_t).contiguous()
                del all_256
                if is_fp4:
                    new_buf = torch.empty(
                        (self.e_local, *old_data.shape[1:]),
                        dtype=old_data.dtype,
                        device=device,
                    )
                    # Copy through the destination tensor directly rather than
                    # .view(torch.uint8).copy_(), which leaves torch_tpu tensors
                    # unmaterialized (layout=None) and forces a batched 2x-HBM
                    # tt_jit_execute_as_strided across all 60 layers at EXTEND.
                    new_buf.copy_(selected.view(old_data.dtype))
                else:
                    new_buf = torch.empty_like(selected)
                    new_buf.copy_(selected)
                del selected
                copy_or_rebind_param(experts, attr, new_buf)
                _wait_for(getattr(experts, attr).data)
                _free_replaced_storage(old_data, getattr(experts, attr).data)
                del old_data, new_buf
            experts.num_local_experts = self.e_local
            experts._tpu_fused_rs_group_offset.fill_(self.tp_rank * self.e_local)
            _wait_for(experts._tpu_fused_rs_group_offset)
            moe.__dict__["_eplb_num_replicas"] = self.num_replicas
            moe.__dict__["_eplb_ep_size"] = self.tp_size
        if any(
            self.decoder.layers[self.start + off].mlp.__dict__.get(
                "_decode_fused_router", False
            )
            for off in range(self.num_layers)
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                get_muse_decode_router_topk_op,
            )

            get_muse_decode_router_topk_op(self.num_replicas, self.tp_size)
        if self.decoder.norm.weight.device.type != "cpu":
            torch.accelerator.synchronize()

    def _verify_counts_digest(self, counts_arr: np.ndarray, device: Any) -> None:
        """Verify all TP ranks observe identical `[num_layers, num_experts]` counts."""
        import torch

        digest_val = int(
            np.bitwise_xor.reduce(
                (
                    counts_arr.reshape(-1)
                    * np.arange(1, counts_arr.size + 1, dtype=np.int64)
                )
                & np.int64(0x7FFFFFFF)
            )
        )
        local_digest = torch.tensor([digest_val], dtype=torch.int32, device=device)
        all_digests = self.gather_fn(local_digest).cpu().tolist()
        if len(set(all_digests)) != 1:
            raise RuntimeError(
                f"Prefill EPLB rank count digest mismatch across TP ranks: {all_digests}"
            )

    def _apply_layer_perm_indices(
        self,
        offset: int,
        rel_col_idx: np.ndarray,
        rel_slot_idx: np.ndarray,
        device: Any,
    ) -> None:
        """Permute one layer's router columns, bias, wrouter buffer, and expert slabs in-place."""
        import torch

        from sglang_tpu.expert_loading import _wait_for
        from sglang_tpu.muse_prefill_kernels import integration as _integ

        layer = self.decoder.layers[self.start + offset]
        moe = layer.mlp
        experts = moe.experts
        lw = self.weights_list[offset]

        rel_col_t = torch.tensor(rel_col_idx, dtype=torch.long, device=device)

        new_gate = moe.gate.weight.data.index_select(0, rel_col_t).contiguous()
        moe.gate.weight.data.copy_(new_gate)
        del new_gate

        new_bias = moe.e_score_correction_bias.data.index_select(
            0, rel_col_t
        ).contiguous()
        moe.e_score_correction_bias.data.copy_(new_bias)
        del new_bias

        b1_bias_t = getattr(lw, "_buffers", {}).get("b1_bias_t")
        if b1_bias_t is not None:
            b1_bias_t.copy_(
                moe.e_score_correction_bias.data.float()
                .unsqueeze(1)
                .expand(-1, 128)
                .contiguous()
            )
            _wait_for(b1_bias_t)

        router_w_fn = getattr(_integ, "_router_weight", None)
        new_wrouter = (
            router_w_fn(moe.gate.weight.data)
            if router_w_fn is not None
            else moe.gate.weight.data.detach().t().float().contiguous()
        )
        lw.wrouter.copy_(new_wrouter)
        del new_wrouter
        _wait_for(moe.gate.weight.data, moe.e_score_correction_bias.data, lw.wrouter)

        moe_bufs = getattr(moe, "_buffers", {})
        gate_km = moe_bufs.get("_fused_gate_w_km")
        if gate_km is not None:
            gate_km.copy_(moe.gate.weight.data.detach().t().contiguous())
            _wait_for(gate_km)
        bias_2d = moe_bufs.get("_fused_decode_bias_2d")
        if bias_2d is not None:
            bias_2d.copy_(moe.e_score_correction_bias.data.detach().reshape(1, -1))
            _wait_for(bias_2d)

        if moe_bufs.get("_decode_fp4_w1") is not None:
            base_slot_idx = rel_slot_idx[: self.num_base_local]
            for _, sh_name in self._DECODE_SHADOW_FP4_ATTRS:
                sh_buf = moe_bufs.get(sh_name)
                if sh_buf is None:
                    continue
                sh_sel = self._gather_selected_slots(sh_buf, base_slot_idx, device)
                if sh_buf.dtype == torch.float4_e2m1fn_x2:
                    sh_buf.copy_(sh_sel.view(sh_buf.dtype))
                else:
                    sh_buf.copy_(sh_sel)
                del sh_sel
                _wait_for(sh_buf)

        for attr in self._EXPERT_ATTRS:
            param = getattr(experts, attr)
            is_fp4 = param.data.dtype == torch.float4_e2m1fn_x2
            selected = self._gather_selected_slots(param.data, rel_slot_idx, device)
            if is_fp4:
                param.data.copy_(selected.view(param.data.dtype))
            else:
                param.data.copy_(selected)
            del selected
            _wait_for(param.data)

        fp8_s1 = moe_bufs.get("_fused_decode_fp8_s1")
        if fp8_s1 is not None:
            fp8_s1.copy_(
                experts.w13_weight_scale.data[: self.num_base_local].reshape(
                    self.num_base_local, -1
                )
            )
            _wait_for(fp8_s1)
        fp8_s2 = moe_bufs.get("_fused_decode_fp8_s2")
        if fp8_s2 is not None:
            fp8_s2.copy_(
                experts.w2_weight_scale.data[: self.num_base_local].reshape(
                    self.num_base_local, -1
                )
            )
            _wait_for(fp8_s2)

    def _gather_selected_slots(self, tensor, rel_slot_idx: np.ndarray, device: Any):
        """Rows ``all_256[rel_slot_idx]`` without materializing all 256 experts at once."""
        import torch

        local = self.num_base_local
        chunk = self._GATHER_CHUNK if local % self._GATHER_CHUNK == 0 else local
        world = self.num_experts // local
        src_rank, src_local = np.divmod(np.asarray(rel_slot_idx, dtype=np.int64), local)
        base = tensor[:local]
        if tensor.dtype == torch.float4_e2m1fn_x2:
            base = base.view(torch.uint8)
        selected = None
        for start in range(0, local, chunk):
            # Same shapes every call: the eager XLA ops compile once in the warm pass.
            in_chunk = (src_local >= start) & (src_local < start + chunk)
            rows = np.where(in_chunk, src_rank * chunk + (src_local - start), 0)
            gathered = self.gather_fn(base[start : start + chunk].contiguous())
            assert gathered.shape[0] == world * chunk, gathered.shape
            picked = gathered.index_select(
                0, torch.tensor(rows, dtype=torch.long, device=device)
            )
            del gathered
            mask = torch.tensor(in_chunk, dtype=torch.bool, device=device).view(
                -1, *([1] * (picked.dim() - 1))
            )
            selected = (
                picked if selected is None else torch.where(mask, picked, selected)
            )
            del picked
        return selected.contiguous()

    def warm_rebalance_kernels(self) -> None:
        """Pre-warm eager XLA rebalance ops with a no-op identity pass before serving."""
        import torch

        from sglang_tpu.expert_loading import _wait_for

        if self.num_layers <= 0:
            return
        device = self.decoder.norm.weight.device
        self._verify_counts_digest(
            np.zeros((self.num_layers, self.num_experts), dtype=np.int64), device
        )
        rel_col_id = np.arange(self.num_experts, dtype=np.int64)
        rel_slot_id = np.concatenate(
            [
                np.arange(
                    self.tp_rank * self.num_base_local,
                    (self.tp_rank + 1) * self.num_base_local,
                    dtype=np.int64,
                ),
                np.arange(self.num_replicas, dtype=np.int64),
            ]
        )
        seen_sigs: set[tuple[Any, ...]] = set()
        for offset in range(self.num_layers):
            moe = self.decoder.layers[self.start + offset].mlp
            experts = moe.experts
            has_shadow = moe._buffers.get("_decode_fp4_w1") is not None
            sig = (
                has_shadow,
                tuple(
                    (
                        getattr(experts, attr).data.dtype,
                        tuple(getattr(experts, attr).data.shape),
                    )
                    for attr in self._EXPERT_ATTRS
                ),
            )
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            self._apply_layer_perm_indices(offset, rel_col_id, rel_slot_id, device)
            dec_r = int(moe.__dict__.get("_fused_decode_num_replicas", 0))
            if dec_r > 0 and moe._buffers.get("_fused_decode_replica_meta") is not None:
                warm_dec_idx = np.zeros(dec_r, dtype=np.int64)
                shadow_map = dict(self._DECODE_SHADOW_FP4_ATTRS)
                for src_attr, _ in self._DECODE_REP_ATTRS:
                    src_t = (
                        moe._buffers[shadow_map[src_attr]]
                        if has_shadow
                        else getattr(experts, src_attr).data
                    )
                    w_sel = self._gather_selected_slots(src_t, warm_dec_idx, device)
                    _wait_for(w_sel)
                    del w_sel
        if device.type != "cpu":
            torch.accelerator.synchronize()

    def _refresh_decode_replicas(
        self,
        offset: int,
        orig_counts: np.ndarray,
        new_col_perm: np.ndarray,
        device: Any,
    ) -> None:
        """Re-synchronize colocated `_fused_decode_rep_*` buffers and metadata in-place."""
        import torch

        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.expert_loading import _wait_for

        moe = self.decoder.layers[self.start + offset].mlp
        dec_r = int(moe.__dict__.get("_fused_decode_num_replicas", 0))
        meta_buf = moe._buffers.get("_fused_decode_replica_meta")
        if dec_r <= 0 or meta_buf is None:
            return
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            build_preloaded_replica_plan,
            validate_replica_plan,
        )

        sel_env = getattr(
            tpu_envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SELECTION_MODE", None
        )
        sel_mode = (
            str(sel_env.get() or "global_top").strip()
            if sel_env is not None
            else "global_top"
        ) or "global_top"
        layer_idx = self.start + offset
        dec_counts = self.decode_counts_by_layer.get(layer_idx, orig_counts)
        owner_to_chip = (
            [r >> 1 for r in range(self.tp_size)] if self.tp_size % 2 == 0 else None
        )
        plan = build_preloaded_replica_plan(
            dec_counts,
            num_owners=self.tp_size,
            num_replicas_per_owner=dec_r,
            placement=new_col_perm,
            owner_to_chip=owner_to_chip,
            selection_mode=sel_mode,
        )
        validate_replica_plan(plan)
        holder_placed = np.asarray(
            [
                int(x) if int(x) >= 0 else 0
                for x in plan.holder_expert_ids[self.tp_rank]
            ],
            dtype=np.int64,
        )
        experts = moe.experts
        has_shadow = moe._buffers.get("_decode_fp4_w1") is not None
        shadow_map = dict(self._DECODE_SHADOW_FP4_ATTRS)
        for src_attr, dst_name in self._DECODE_REP_ATTRS:
            dst = moe._buffers.get(dst_name)
            if dst is None:
                return
            src_t = (
                moe._buffers[shadow_map[src_attr]]
                if has_shadow
                else getattr(experts, src_attr).data
            )
            selected = self._gather_selected_slots(src_t, holder_placed, device)
            if dst.dtype == torch.float4_e2m1fn_x2:
                dst.copy_(selected.view(dst.dtype))
            else:
                dst.copy_(selected)
            del selected
            _wait_for(dst)
        meta_t = torch.from_numpy(
            np.ascontiguousarray(plan.replica_meta, dtype=np.int32)
        ).to(device=device, dtype=torch.int32)
        meta_buf.copy_(meta_t)
        _wait_for(meta_buf)

    def rebalance_from_counts(
        self, counts_2d: np.ndarray, *, tile_m: int = 128
    ) -> None:
        """Permute router columns, bias, and expert slabs in-place across all managed layers."""
        import torch

        counts_arr = np.asarray(counts_2d, dtype=np.int64)
        if counts_arr.shape != (self.num_layers, self.num_experts):
            raise ValueError(
                f"counts_2d shape {counts_arr.shape} != ({self.num_layers}, {self.num_experts})"
            )

        device = self.decoder.norm.weight.device
        self._verify_counts_digest(counts_arr, device)

        for offset in range(self.num_layers):
            layer_idx = self.start + offset
            r2g, new_col_perm = compute_zeroglue_eplb_perm(
                counts_arr[offset],
                ep_size=self.tp_size,
                num_local_base=self.num_base_local,
                num_replicas=self.num_replicas,
                tile_m=tile_m,
                dual_decode=self.dual_decode,
                decode_counts=self.decode_counts_by_layer.get(layer_idx),
            )
            cur_perm = self.current_col_perms[offset]
            if np.array_equal(cur_perm, new_col_perm) and self.num_replicas == 0:
                continue
            inv_cur = np.argsort(cur_perm).astype(np.int64)
            rel_col_idx = inv_cur[new_col_perm]
            rel_slot_idx = inv_cur[r2g[self.tp_rank]]
            self._apply_layer_perm_indices(offset, rel_col_idx, rel_slot_idx, device)
            self.current_col_perms[offset] = new_col_perm
            self._refresh_decode_replicas(
                offset, counts_arr[offset], new_col_perm, device
            )
        if device.type != "cpu":
            torch.accelerator.synchronize()
