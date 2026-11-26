# SPDX-License-Identifier: Apache-2.0
"""Offline placement/replica calibration and deterministic Pallas SMEM planners.

Implements Muse Spark decode MoE imbalance handling from arXiv:2611.17857v1:
  - Section 4.5a: Static expert placement (`build_static_placement`) compatible
    with `muse_spark.py:_expert_placement`.
  - Section 4.5b: Cross-chip preloaded hot-expert copies (`build_preloaded_replica_plan`
    + deterministic in-kernel `plan_preloaded_replicas_smem`), supporting both
    pure-EP (EP8 / EP16) and symmetric EPxTP2 (EP4xTP2 / EP8xTP2), R in {1, 2, 4}
    extra replica slots per owner, and K256/K16 FP4 weight banks.
  - Section 4.3: Intra-chip chipmate redistribution (`plan_chipmate_move_smem` +
    NumPy reference `plan_chipmate_move_ref`) for pure-EP8 / EP16, selecting at
    most 1 few-row expert per chipmate pair with active-expert gap >= 2 without
    worsening pair max rows, composable after Section 4.5b moves.

Design invariants:
  1. Pure-NumPy CPU-importable offline helpers: JAX/Pallas/Torch are only
     imported lazily inside Pallas or framework-specific helpers.
  2. Zero extra collectives at runtime: both SMEM planners consume only
     replicated `counts_smem` (`int32[8, 256]` or `int32[256]`) and replicated
     `replica_meta_smem` (`int32[4, K_rep]`), executing pure deterministic
     `int32` scalar arithmetic so all ranks/chipmates reach bit-identical
     decisions.
  3. Unique execution ownership: every active expert executes on exactly one
     owner (home unless moved to holder or chipmate; never zero times, never
     twice).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

NUM_EXPERTS_DEFAULT = 256
VALID_REPLICA_COST_MODES = ("active", "tiles", "weighted")
_COST_MODE_ALIASES = {
    "active": "active",
    "experts": "active",
    "tiles": "tiles",
    "tile": "tiles",
    "weighted": "weighted",
    "hybrid": "weighted",
    "row_weighted": "weighted",
}


def normalize_replica_cost_mode(cost_mode: str) -> str:
    """Validate and normalize a replica planner cost mode string."""
    key = (cost_mode or "active").strip().lower()
    if key not in _COST_MODE_ALIASES:
        raise ValueError(
            f"Unsupported replica cost_mode={cost_mode!r}; expected one of "
            f"{VALID_REPLICA_COST_MODES}"
        )
    return _COST_MODE_ALIASES[key]


@dataclass(frozen=True)
class PreloadedReplicaPlan:
    """Offline Section 4.5b preloaded hot-expert replica assignment.

    Attributes:
      replica_meta: `int32[4, K_rep]` replicated SMEM metadata table in placed
        (runtime) expert ID order, where `K_rep = num_owners * num_replicas_per_owner`:
          - `[0, k]`: placed expert ID in `[0, num_experts)` or `-1` if unused
          - `[1, k]`: `home_owner` in `[0, num_owners)` or `-1`
          - `[2, k]`: `holder_owner` in `[0, num_owners)` (`!= home_owner`) or `-1`
          - `[3, k]`: `holder_slot` in `[0, num_replicas_per_owner)` or `-1`
      holder_expert_ids: `int32[num_owners, R]` placed expert IDs stored in each
        owner's replica weight bank (`-1` if unused).
      holder_checkpoint_ids: `int32[num_owners, R]` checkpoint expert IDs stored
        in each owner's replica weight bank (`placement[eid]` when `placement`
        is provided, else `eid`; `-1` if unused).
      num_owners: Number of EP owners (8/16 for pure EP, 4/8 chips for EPxTP2).
      num_replicas_per_owner: Reserved replica slots `R` per owner.
      num_experts: Total experts (default 256).
      placement: Optional `int32[num_experts]` permutation (`placement[p]` is the
        checkpoint expert loaded at placed index `p`).
      owner_to_chip: Optional `int32[num_owners]` physical chip ID per owner.
      max_holders_per_expert: Maximum distinct non-home holders allowed per expert.
    """

    replica_meta: np.ndarray
    holder_expert_ids: np.ndarray
    holder_checkpoint_ids: np.ndarray
    num_owners: int
    num_replicas_per_owner: int
    num_experts: int = NUM_EXPERTS_DEFAULT
    placement: np.ndarray | None = None
    owner_to_chip: np.ndarray | None = None
    max_holders_per_expert: int = 1


def _validate_permutation(order: np.ndarray, num_experts: int) -> np.ndarray:
    perm = np.asarray(order, dtype=np.int32)
    if perm.shape != (num_experts,):
        raise ValueError(
            f"Expected placement shape ({num_experts},), got {perm.shape}"
        )
    if not np.array_equal(np.sort(perm), np.arange(num_experts, dtype=np.int32)):
        raise ValueError(f"placement is not a valid permutation of 0..{num_experts - 1}")
    return perm


def _validate_owner_to_chip(
    owner_to_chip: Sequence[int] | np.ndarray | None,
    num_owners: int,
) -> np.ndarray | None:
    if owner_to_chip is None:
        return None
    arr = np.asarray(owner_to_chip, dtype=np.int32)
    if arr.shape != (num_owners,):
        raise ValueError(
            f"Expected owner_to_chip shape ({num_owners},), got {arr.shape}"
        )
    return arr


_EMPIRICAL_STEP_SAMPLES_CACHE: dict[tuple[int, bytes], np.ndarray] = {}


def _cache_empirical_step_samples(counts_1d: np.ndarray, step_samples_2d: np.ndarray) -> None:
    c = np.ascontiguousarray(counts_1d, dtype=np.float64)
    s = np.ascontiguousarray(step_samples_2d)
    if c.ndim != 1 or s.ndim != 2 or s.shape[1] != c.size or s.shape[0] == 0:
        return
    if len(_EMPIRICAL_STEP_SAMPLES_CACHE) >= 256:
        _EMPIRICAL_STEP_SAMPLES_CACHE.clear()
    _EMPIRICAL_STEP_SAMPLES_CACHE[(int(c.size), c.tobytes())] = s


def _lookup_empirical_step_samples(counts_1d: np.ndarray) -> np.ndarray | None:
    if not _EMPIRICAL_STEP_SAMPLES_CACHE:
        return None
    c = np.ascontiguousarray(counts_1d, dtype=np.float64)
    return _EMPIRICAL_STEP_SAMPLES_CACHE.get((int(c.size), c.tobytes()))


def _topk_inclusion_probs(p_raw: np.ndarray, topk: int) -> np.ndarray:
    """Waterfill marginal top-k inclusion probabilities `q_e in [0, 1]` with `sum(q_e) == min(topk, nnz)`."""
    p = np.asarray(p_raw, dtype=np.float64)
    pos = p > 0.0
    target = float(min(max(1, int(topk)), int(pos.sum())))
    if target <= 0.0:
        return np.zeros_like(p, dtype=np.float64)
    q = np.zeros_like(p, dtype=np.float64)
    rem_mask = pos.copy()
    rem_target = target
    for _ in range(10):
        rem_sum = float(p[rem_mask].sum())
        if rem_sum <= 0.0 or rem_target <= 0.0:
            break
        scaled = p[rem_mask] * (rem_target / rem_sum)
        over = scaled >= 1.0 - 1e-12
        if not np.any(over):
            q[rem_mask] = scaled
            break
        idx_rem = np.flatnonzero(rem_mask)
        q[idx_rem[over]] = 1.0
        rem_mask[idx_rem[over]] = False
        rem_target -= float(over.sum())
    return np.clip(q, 0.0, 1.0)


def load_calibration_counts(
    path: str | Path,
    *,
    num_layers: int = 62,
    num_experts: int = NUM_EXPERTS_DEFAULT,
) -> dict[int, np.ndarray]:
    """Load per-layer expert activation counts (`float64[num_experts]`).

    Supports:
      - `eplb_stats_nd_c64.json`: `{"1": [256 counts], ..., "60": [...]}`
      - Wrapper JSON: `{"layers": {"1": [256 counts], ...}}`
      - Route-hist JSONL (`SGLANG_TPU_MUSE_MOE_ROUTE_HIST`): lines containing
        `{"first_layer": int, "counts": [[256], ...]}` summed over all forwards.
      - `.npy` (`[num_experts]` or `[L, num_experts]`) and `.npz`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration counts file not found: {p}")

    suffix = p.suffix.lower()
    out: dict[int, np.ndarray] = {}

    if suffix == ".npy":
        arr = np.asarray(np.load(p), dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape != (num_experts,):
                raise ValueError(f"{p}: expected ({num_experts},), got {arr.shape}")
            out[0] = arr
        elif arr.ndim == 2 and arr.shape[1] == num_experts:
            for layer_idx in range(min(arr.shape[0], num_layers)):
                out[layer_idx] = arr[layer_idx]
        else:
            raise ValueError(f"{p}: unsupported .npy shape {arr.shape}")
        return out

    if suffix == ".npz":
        with np.load(p) as z:
            if "counts" in z:
                arr = np.asarray(z["counts"], dtype=np.float64)
                if arr.ndim == 1:
                    out[0] = arr
                elif arr.ndim == 2 and arr.shape[1] == num_experts:
                    for layer_idx in range(min(arr.shape[0], num_layers)):
                        out[layer_idx] = arr[layer_idx]
                else:
                    raise ValueError(f"{p}: unsupported counts shape {arr.shape}")
            else:
                for k in z.files:
                    if k == "step_samples":
                        continue
                    arr = np.asarray(z[k], dtype=np.float64)
                    if arr.shape == (num_experts,):
                        layer_idx = int(k.lstrip("Llayer_")) if k.lstrip("Llayer_").isdigit() else 0
                        out[layer_idx] = arr
            if "step_samples" in z:
                steps_arr = np.asarray(z["step_samples"])
                if steps_arr.ndim == 2 and steps_arr.shape[1] == num_experts and 0 in out:
                    _cache_empirical_step_samples(out[0], steps_arr)
                elif steps_arr.ndim == 3 and steps_arr.shape[2] == num_experts:
                    for layer_idx, c_vec in out.items():
                        if 0 <= layer_idx < steps_arr.shape[1]:
                            _cache_empirical_step_samples(c_vec, steps_arr[:, layer_idx, :])
        if not out:
            raise ValueError(f"{p}: no valid [{num_experts}] count arrays found in .npz")
        return out

    if suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                first_layer = int(rec.get("first_layer", 0))
                for offset, row in enumerate(rec["counts"]):
                    vec = np.asarray(row, dtype=np.float64)
                    if vec.shape != (num_experts,):
                        raise ValueError(
                            f"{p}: layer {first_layer + offset} expected {num_experts}, got {vec.shape}"
                        )
                    layer = first_layer + offset
                    if layer in out:
                        out[layer] = out[layer] + vec
                    else:
                        out[layer] = vec.copy()
        if not out:
            raise ValueError(f"{p}: empty JSONL calibration file")
        return out

    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape == (num_experts,):
            return {0: arr}
        raise ValueError(f"{p}: expected list of length {num_experts}, got {arr.shape}")
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: unsupported JSON root type {type(raw).__name__}")
    mapping = raw["layers"] if "layers" in raw and isinstance(raw["layers"], dict) else raw
    for k, v in mapping.items():
        if not str(k).isdigit():
            continue
        vec = np.asarray(v, dtype=np.float64)
        if vec.shape != (num_experts,):
            raise ValueError(
                f"{p}: layer {k} expected ({num_experts},), got {vec.shape}"
            )
        layer_idx = int(k)
        if 0 <= layer_idx < num_layers:
            out[layer_idx] = vec
    if not out:
        raise ValueError(f"{p}: no layer count vectors found")
    steps_sidecar = p.with_suffix(".steps.npy")
    if steps_sidecar.exists():
        try:
            steps_arr = np.asarray(np.load(steps_sidecar))
            if steps_arr.ndim == 3 and steps_arr.shape[2] == num_experts:
                for layer_idx, c_vec in out.items():
                    if 0 <= layer_idx < steps_arr.shape[1]:
                        _cache_empirical_step_samples(c_vec, steps_arr[:, layer_idx, :])
        except Exception:
            pass
    return out


def _decode_objective_weights(
    counts: np.ndarray,
    *,
    objective: str = "decode_active",
    calib_batch: int = 64,
    topk: int = 8,
    tile_m: int = 32,
) -> np.ndarray:
    """Convert raw calibration counts `[E]` into per-expert scheduling weights.

    Objectives:
      - `"prefill_rows"`: raw token-assignment counts (row/compute-bound regime).
      - `"decode_active"`: expected decode activation probability averaged across
        `B in {calib_batch // 2, calib_batch, calib_batch * 2}`:
        `P(active_e | B) = 1 - (1 - min(1, topk * p_e))^B`.
        Crucial for skewed distributions (like real Muse Spark Layer 26): avoids
        stacking many medium-probability experts onto one rank opposite a single
        mega-hot expert.
      - `"decode_hybrid"`: `P(active_e) + 0.25 * E[rows_e] / tile_m` (captures both
        weight-streaming cost and multi-tile compute at B=128..256).
    """
    obj = (objective or "decode_active").strip().lower()
    if obj in ("prefill", "prefill_rows", "rows"):
        return counts.astype(np.float64, copy=True)
    total = float(counts.sum())
    if total <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    p_slot = counts.astype(np.float64) / total
    if obj in (
        "prefill_tiles",
        "prefill_hybrid",
        "colocated",
        "colocated_hybrid",
        "dual",
        "dual_tiles",
    ):
        tm = float(max(1, tile_m if tile_m >= 64 else 128))
        prefill_toks = (4096.0, 8192.0)
        q_tok = np.clip(float(topk) * p_slot, 0.0, 1.0)
        tile_costs = []
        for t_len in prefill_toks:
            exp_rows = (t_len * float(topk)) * p_slot
            p_act_t = 1.0 - np.power(1.0 - q_tok, t_len)
            # Expected compute tiles + 1 weight-load equivalent tile when active.
            tile_costs.append(np.ceil(exp_rows / tm) * p_act_t + p_act_t)
        pre_w = np.mean(tile_costs, axis=0)
        if obj in ("prefill_tiles", "prefill_hybrid"):
            return pre_w
        batches = (
            max(1, calib_batch // 2),
            max(1, calib_batch),
            max(1, calib_batch * 2),
        )
        p_act = np.mean([1.0 - np.power(1.0 - q_tok, float(b)) for b in batches], axis=0)
        pre_norm = pre_w / max(1e-9, float(pre_w.mean()))
        dec_norm = p_act / max(1e-9, float(p_act.mean()))
        return np.maximum(pre_norm, dec_norm)
    q_tok = np.clip(float(topk) * p_slot, 0.0, 1.0)
    batches = (max(1, calib_batch // 2), max(1, calib_batch), max(1, calib_batch * 2))
    p_act = np.mean([1.0 - np.power(1.0 - q_tok, float(b)) for b in batches], axis=0)
    if obj in ("decode_active", "active", "decode"):
        return p_act
    if obj in ("decode_hybrid", "hybrid", "weighted", "decode_tiles"):
        exp_tiles = (float(calib_batch * topk) * p_slot) / float(max(1, tile_m))
        return p_act + 0.25 * exp_tiles
    raise ValueError(
        f"Unsupported objective={objective!r}; expected 'decode_active', "
        "'decode_hybrid', 'prefill_tiles', 'dual', or 'prefill_rows'"
    )


def sort_placement_intra_owner_hot0(
    placement: np.ndarray,
    decode_counts: Any,
    *,
    num_owners: int = 8,
    replicated_placed_eids: Sequence[int] = (),
    sub_owners: int | None = None,
) -> np.ndarray:
    """Reorder slots WITHIN each owner (or ``sub_owners``) so slot `0` holds
    that group's hottest non-replicated decode expert (0 cross-owner perturbation!).
    """
    eff_groups = num_owners if sub_owners is None else int(sub_owners)
    if eff_groups <= 0 or eff_groups % num_owners != 0:
        raise ValueError(
            f"sub_owners={sub_owners} must be a positive multiple of num_owners={num_owners}"
        )
    perm = _validate_permutation(placement, int(np.asarray(placement).size))
    num_experts = int(perm.size)
    e_per_owner = num_experts // eff_groups
    dec = np.asarray(decode_counts, dtype=np.float64)
    placed_dec = dec[perm]
    rep_set = {int(x) for x in replicated_placed_eids if int(x) >= 0}
    out = perm.copy()
    for o in range(eff_groups):
        base = o * e_per_owner
        non_rep = [
            le for le in range(e_per_owner) if (base + le) not in rep_set
        ]
        rep_loc = [le for le in range(e_per_owner) if (base + le) in rep_set]
        non_rep.sort(key=lambda le: (-float(placed_dec[base + le]), int(perm[base + le])))
        rep_loc.sort(key=lambda le: (-float(placed_dec[base + le]), int(perm[base + le])))
        ordered_local = non_rep + rep_loc
        for slot, le in enumerate(ordered_local):
            out[base + slot] = perm[base + le]
    return _validate_permutation(out, num_experts)


def build_static_placement(
    expert_counts: Any,
    *,
    num_owners: int,
    owner_to_chip: Sequence[int] | np.ndarray | None = None,
    tile_m: int = 32,
    max_2opt_iters: int = 128,
    objective: str = "decode_active",
    calib_batch: int = 64,
    topk: int = 8,
    min_gain: float = 0.02,
    decode_counts: Any | None = None,
    hot_slot0_order: bool = False,
    sub_owners: int | None = None,
) -> np.ndarray:
    """Compute a static expert placement permutation `int32[E]` (Section 4.5a).

    Compatible with `muse_spark.py:_expert_placement`: entry `p` is the checkpoint
    expert ID loaded at runtime slot `p`, assigned to owner `p // (E // num_owners)`.
    When `owner_to_chip` is provided (pure EP8/EP16), LPT packing jointly balances
    physical chip totals and per-owner totals before 2-opt refinement. When
    `objective="dual"`, passing optional `decode_counts` optimizes the decode axis
    directly against observed decode-step routing counts while balancing prefill rows
    against `expert_counts`. When `sub_owners` (e.g. `16` for `num_owners=8`) is set,
    balances at `sub_owners` granularity first and pairs sub-bins into balanced
    `num_owners` parent groups so both EP scales (`EP=8` and `EP=16`) stay balanced.
    """
    counts = np.asarray(expert_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"Expected 1D expert_counts, got shape {counts.shape}")
    num_experts = int(counts.size)
    if num_owners <= 0 or num_experts % num_owners != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by num_owners={num_owners}"
        )
    if np.any(counts < 0):
        raise ValueError("expert_counts must be non-negative")
    dec_src = counts
    if decode_counts is not None:
        dec_src = np.asarray(decode_counts, dtype=np.float64)
        if dec_src.shape != (num_experts,):
            raise ValueError(
                f"Expected decode_counts shape ({num_experts},), got {dec_src.shape}"
            )
        if np.any(dec_src < 0):
            raise ValueError("decode_counts must be non-negative")
    e_per_owner = num_experts // num_owners
    chip_map = _validate_owner_to_chip(owner_to_chip, num_owners)

    if sub_owners is not None:
        sub_n = int(sub_owners)
        if sub_n <= num_owners or sub_n % num_owners != 0 or num_experts % sub_n != 0:
            raise ValueError(
                f"sub_owners={sub_n} must strictly exceed and divide cleanly into "
                f"num_owners={num_owners} and num_experts={num_experts}"
            )
        sub_perm = build_static_placement(
            counts,
            num_owners=sub_n,
            owner_to_chip=None,
            tile_m=tile_m,
            max_2opt_iters=max_2opt_iters,
            objective=objective,
            calib_batch=calib_batch,
            topk=topk,
            min_gain=0.0,
            decode_counts=decode_counts,
            hot_slot0_order=False,
        )
        e_per_sub = num_experts // sub_n
        subs_per_owner = sub_n // num_owners
        sub_bins = [sub_perm[s * e_per_sub : (s + 1) * e_per_sub] for s in range(sub_n)]
        obj_key_s = (objective or "decode_active").strip().lower()
        if obj_key_s in ("dual", "dual_tiles", "colocated", "colocated_hybrid"):
            pre_o = "prefill_tiles" if obj_key_s == "dual_tiles" else "prefill_rows"
            w_pre_s = _decode_objective_weights(
                counts, objective=pre_o, calib_batch=calib_batch, topk=topk, tile_m=max(128, tile_m) if obj_key_s == "dual_tiles" else tile_m
            )
            w_dec_s = _decode_objective_weights(
                dec_src if dec_src.sum() > 0.0 else counts, objective="decode_active", calib_batch=calib_batch, topk=topk, tile_m=32
            )
            w_sub = (
                w_pre_s / max(1e-12, float(w_pre_s.mean()))
                + w_dec_s / max(1e-12, float(w_dec_s.mean()))
            ) * 0.5
        else:
            w_sub = _decode_objective_weights(
                counts, objective=objective, calib_batch=calib_batch, topk=topk, tile_m=tile_m
            )
        sub_loads = [float(w_sub[b].sum()) for b in sub_bins]
        sub_order = sorted(range(sub_n), key=lambda s: (-sub_loads[s], s))
        p_bins: list[list[int]] = [[] for _ in range(num_owners)]
        p_loads = np.zeros(num_owners, dtype=np.float64)
        n_ch_s = int(chip_map.max()) + 1 if chip_map is not None else num_owners
        ch_loads = np.zeros(n_ch_s, dtype=np.float64)
        for s_idx in sub_order:
            cands = [o for o in range(num_owners) if len(p_bins[o]) < subs_per_owner]
            if chip_map is not None:
                best_o = min(cands, key=lambda o: (float(ch_loads[int(chip_map[o])]), float(p_loads[o]), len(p_bins[o]), o))
            else:
                best_o = min(cands, key=lambda o: (float(p_loads[o]), len(p_bins[o]), o))
            p_bins[best_o].append(s_idx)
            p_loads[best_o] += sub_loads[s_idx]
            if chip_map is not None:
                ch_loads[int(chip_map[best_o])] += sub_loads[s_idx]
        grouped = np.concatenate([sub_bins[s_idx] for o in range(num_owners) for s_idx in p_bins[o]]).astype(np.int32)
        grouped = _validate_permutation(grouped, num_experts)
        if min_gain > 0.0 and w_sub.sum() > 0.0:
            id_l = w_sub.reshape(sub_n, e_per_sub).sum(axis=1)
            pl_l = w_sub[grouped].reshape(sub_n, e_per_sub).sum(axis=1)
            if (float(id_l.max()) - float(pl_l.max())) / max(1e-12, float(id_l.mean())) < min_gain:
                grouped = np.arange(num_experts, dtype=np.int32)
        if hot_slot0_order:
            grouped = sort_placement_intra_owner_hot0(
                grouped,
                dec_src if dec_src.sum() > 0.0 else counts,
                num_owners=num_owners,
                sub_owners=sub_n,
            )
        return grouped

    obj_key = (objective or "decode_active").strip().lower()
    if obj_key in ("dual", "dual_tiles", "colocated", "colocated_hybrid") and (
        counts.sum() > 0.0 or dec_src.sum() > 0.0
    ):
        pre_obj = "prefill_tiles" if obj_key == "dual_tiles" else "prefill_rows"
        pre_tm = max(128, tile_m) if obj_key == "dual_tiles" else tile_m
        w_pre_raw = _decode_objective_weights(
            counts,
            objective=pre_obj,
            calib_batch=calib_batch,
            topk=topk,
            tile_m=pre_tm,
        )
        w_dec_raw = _decode_objective_weights(
            dec_src if dec_src.sum() > 0.0 else counts,
            objective="decode_active",
            calib_batch=calib_batch,
            topk=topk,
            tile_m=32,
        )
        # Normalize each objective so mean owner load == 1.0 across `num_owners`.
        w_pre = w_pre_raw * (float(num_owners) / max(1e-12, float(w_pre_raw.sum())))
        w_dec = w_dec_raw * (float(num_owners) / max(1e-12, float(w_dec_raw.sum())))

        bins_d: list[list[int]] = [[] for _ in range(num_owners)]
        own_pre = np.zeros(num_owners, dtype=np.float64)
        own_dec = np.zeros(num_owners, dtype=np.float64)
        n_ch = int(chip_map.max()) + 1 if chip_map is not None else num_owners
        ch_pre = np.zeros(n_ch, dtype=np.float64)
        ch_dec = np.zeros(n_ch, dtype=np.float64)
        ch_scale = float(n_ch) / float(num_owners)

        order_dual = np.argsort(-np.maximum(w_pre, w_dec), kind="stable")
        for e_np in order_dual:
            e = int(e_np)
            wp, wd = float(w_pre[e]), float(w_dec[e])
            cands = [o for o in range(num_owners) if len(bins_d[o]) < e_per_owner]
            if chip_map is not None:
                best_o = min(
                    cands,
                    key=lambda o: (
                        max(own_pre[o] + wp, own_dec[o] + wd),
                        max(
                            (ch_pre[int(chip_map[o])] + wp) * ch_scale,
                            (ch_dec[int(chip_map[o])] + wd) * ch_scale,
                        ),
                        own_pre[o] + own_dec[o] + wp + wd,
                        len(bins_d[o]),
                        o,
                    ),
                )
            else:
                best_o = min(
                    cands,
                    key=lambda o: (
                        max(own_pre[o] + wp, own_dec[o] + wd),
                        own_pre[o] + own_dec[o] + wp + wd,
                        len(bins_d[o]),
                        o,
                    ),
                )
            bins_d[best_o].append(e)
            own_pre[best_o] += wp
            own_dec[best_o] += wd
            if chip_map is not None:
                c_idx = int(chip_map[best_o])
                ch_pre[c_idx] += wp
                ch_dec[c_idx] += wd

        for _ in range(max_2opt_iters):
            peaks = np.maximum(own_pre, own_dec)
            max_o = int(np.argmax(peaks))
            cur_global_peak = float(peaks[max_o])
            best_gain = 1e-9
            best_swap: tuple[int, int, int] | None = None
            for other_o in range(num_owners):
                if other_o == max_o:
                    continue
                pair_old = max(peaks[max_o], peaks[other_o])
                for i, ei in enumerate(bins_d[max_o]):
                    wpi, wdi = float(w_pre[ei]), float(w_dec[ei])
                    for j, ej in enumerate(bins_d[other_o]):
                        wpj, wdj = float(w_pre[ej]), float(w_dec[ej])
                        nm_p = own_pre[max_o] - wpi + wpj
                        nm_d = own_dec[max_o] - wdi + wdj
                        no_p = own_pre[other_o] - wpj + wpi
                        no_d = own_dec[other_o] - wdj + wdi
                        pair_new = max(nm_p, nm_d, no_p, no_d)
                        gain = pair_old - pair_new
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (other_o, i, j)
            if best_swap is None or cur_global_peak <= 1.001:
                break
            other_o, i, j = best_swap
            ei, ej = bins_d[max_o][i], bins_d[other_o][j]
            bins_d[max_o][i], bins_d[other_o][j] = ej, ei
            dp = float(w_pre[ej] - w_pre[ei])
            dd = float(w_dec[ej] - w_dec[ei])
            own_pre[max_o] += dp
            own_dec[max_o] += dd
            own_pre[other_o] -= dp
            own_dec[other_o] -= dd

        placement = np.empty(num_experts, dtype=np.int32)
        for o in range(num_owners):
            for slot, e in enumerate(sorted(bins_d[o])):
                placement[o * e_per_owner + slot] = e
        placement = _validate_permutation(placement, num_experts)
        if min_gain > 0.0:
            id_pre = w_pre.reshape(num_owners, e_per_owner).sum(axis=1).max()
            id_dec = w_dec.reshape(num_owners, e_per_owner).sum(axis=1).max()
            pl_pre = w_pre[placement].reshape(num_owners, e_per_owner).sum(axis=1).max()
            pl_dec = w_dec[placement].reshape(num_owners, e_per_owner).sum(axis=1).max()
            if max(id_pre, id_dec) - max(pl_pre, pl_dec) < min_gain:
                placement = np.arange(num_experts, dtype=np.int32)
        if hot_slot0_order:
            placement = sort_placement_intra_owner_hot0(
                placement,
                dec_src if dec_src.sum() > 0.0 else counts,
                num_owners=num_owners,
            )
        return placement

    weights = _decode_objective_weights(
        counts,
        objective=objective,
        calib_batch=calib_batch,
        topk=topk,
        tile_m=tile_m,
    )

    bins: list[list[int]] = [[] for _ in range(num_owners)]
    owner_loads = np.zeros(num_owners, dtype=np.float64)
    num_chips = int(chip_map.max()) + 1 if chip_map is not None else num_owners
    chip_loads = np.zeros(num_chips, dtype=np.float64)

    order_desc = np.argsort(-weights, kind="stable")
    for e_np in order_desc:
        e = int(e_np)
        candidates = [o for o in range(num_owners) if len(bins[o]) < e_per_owner]
        if chip_map is not None:
            best_o = min(
                candidates,
                key=lambda o: (
                    float(chip_loads[int(chip_map[o])]),
                    float(owner_loads[o]),
                    len(bins[o]),
                    o,
                ),
            )
        else:
            best_o = min(
                candidates,
                key=lambda o: (float(owner_loads[o]), len(bins[o]), o),
            )
        bins[best_o].append(e)
        owner_loads[best_o] += float(weights[e])
        if chip_map is not None:
            chip_loads[int(chip_map[best_o])] += float(weights[e])

    # 2-opt pairwise swap refinement between heaviest and lightest owners.
    for _ in range(max_2opt_iters):
        max_o = int(np.argmax(owner_loads))
        min_o = int(np.argmin(owner_loads))
        if max_o == min_o or owner_loads[max_o] - owner_loads[min_o] <= 1e-9:
            break
        best_gain = 1e-9
        best_pair: tuple[int, int] | None = None
        for i, ei in enumerate(bins[max_o]):
            wi = float(weights[ei])
            for j, ej in enumerate(bins[min_o]):
                wj = float(weights[ej])
                new_max = owner_loads[max_o] - wi + wj
                new_min = owner_loads[min_o] - wj + wi
                if chip_map is not None and chip_map[max_o] != chip_map[min_o]:
                    c_max = int(chip_map[max_o])
                    c_min = int(chip_map[min_o])
                    old_chip_peak = max(chip_loads[c_max], chip_loads[c_min])
                    new_chip_peak = max(
                        chip_loads[c_max] - wi + wj,
                        chip_loads[c_min] - wj + wi,
                    )
                    if new_chip_peak > old_chip_peak + 1e-9:
                        continue
                gain = owner_loads[max_o] - max(new_max, new_min)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        ei, ej = bins[max_o][i], bins[min_o][j]
        delta = float(weights[ej]) - float(weights[ei])
        bins[max_o][i], bins[min_o][j] = ej, ei
        owner_loads[max_o] += delta
        owner_loads[min_o] -= delta
        if chip_map is not None:
            chip_loads[int(chip_map[max_o])] += delta
            chip_loads[int(chip_map[min_o])] -= delta

    placement = np.empty(num_experts, dtype=np.int32)
    for o in range(num_owners):
        for slot, e in enumerate(sorted(bins[o])):
            placement[o * e_per_owner + slot] = e
    placement = _validate_permutation(placement, num_experts)

    # Guard against noisy permutations on already-balanced layers (matches
    # upstream expert_placement.py `--min-gain`): keep identity order unless
    # peak/mean objective load improves by at least `min_gain`.
    if min_gain > 0.0 and weights.sum() > 0.0:
        ident_loads = weights.reshape(num_owners, e_per_owner).sum(axis=1)
        placed_loads = weights[placement].reshape(num_owners, e_per_owner).sum(axis=1)
        mean_l = float(ident_loads.mean())
        before_ratio = float(ident_loads.max()) / mean_l
        after_ratio = float(placed_loads.max()) / mean_l
        chip_gain = 0.0
        if chip_map is not None and len(set(chip_map.tolist())) > 1:
            n_ch = int(chip_map.max()) + 1
            id_ch = np.bincount(chip_map, weights=ident_loads, minlength=n_ch)
            pl_ch = np.bincount(chip_map, weights=placed_loads, minlength=n_ch)
            chip_gain = float(id_ch.max() - pl_ch.max()) / float(id_ch.mean())
        if max(before_ratio - after_ratio, chip_gain) < min_gain:
            placement = np.arange(num_experts, dtype=np.int32)

    if hot_slot0_order:
        placement = sort_placement_intra_owner_hot0(
            placement,
            dec_src if dec_src.sum() > 0.0 else counts,
            num_owners=num_owners,
        )
    return placement


def build_preloaded_replica_plan(
    expert_counts: Any,
    *,
    num_owners: int,
    num_replicas_per_owner: int,
    placement: Any | None = None,
    owner_to_chip: Sequence[int] | np.ndarray | None = None,
    max_holders_per_expert: int = 1,
    holder_load_alpha: float = 0.5,
    objective: str = "decode_active",
    calib_batch: int = 64,
    topk: int = 8,
    selection_mode: str = "global_top",
    tile_m: int = 32,
    empirical_step_counts: Any | None = None,
) -> PreloadedReplicaPlan:
    """Build an offline Section 4.5b preloaded hot-expert replica plan.

    Selects top hot experts by calibration activation counts and assigns each to
    a disjoint non-home holder owner minimizing aggregate holder load
    (`base_owner_load + holder_load_alpha * assigned_replica_counts`). When
    `owner_to_chip` is given and spans multiple chips, strictly prefers holders
    on a different physical chip than the expert's home owner.

    Args:
      expert_counts: 1D non-negative calibration counts `[num_experts]` in
        checkpoint expert order (mapped through `placement` if provided).
      num_owners: Number of EP owners (8 or 16 for pure EP; 4 or 8 for EPxTP2).
      num_replicas_per_owner: Extra preloaded replica slots `R` per owner
        (0, 1, 2, or 4).
      placement: Optional checkpoint-order permutation (`placement[p]` is the
        checkpoint expert loaded at runtime slot `p`).
      owner_to_chip: Optional physical chip ID for each owner `[num_owners]`.
      max_holders_per_expert: Max distinct holders per expert (default 1,
        matching arXiv:2611.17857v1 Section 4.5).
      holder_load_alpha: Weight applied to previously assigned replica counts
        when scoring candidate holder aggregate load.
    """
    counts = np.asarray(expert_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"Expected 1D expert_counts, got shape {counts.shape}")
    num_experts = int(counts.size)
    if num_owners < 2 or num_experts % num_owners != 0:
        raise ValueError(
            f"num_owners={num_owners} must be >= 2 and divide num_experts={num_experts}"
        )
    if num_replicas_per_owner < 0:
        raise ValueError(
            f"num_replicas_per_owner must be >= 0, got {num_replicas_per_owner}"
        )
    if max_holders_per_expert < 1:
        raise ValueError(
            f"max_holders_per_expert must be >= 1, got {max_holders_per_expert}"
        )
    if np.any(counts < 0):
        raise ValueError("expert_counts must be non-negative")

    e_per_owner = num_experts // num_owners
    perm = _validate_permutation(placement, num_experts) if placement is not None else None
    chip_map = _validate_owner_to_chip(owner_to_chip, num_owners)
    placed_raw = counts[perm] if perm is not None else counts
    placed_counts = _decode_objective_weights(
        placed_raw,
        objective=objective,
        calib_batch=calib_batch,
        topk=topk,
    )

    k_rep = num_owners * num_replicas_per_owner
    replica_meta = np.full((4, k_rep), -1, dtype=np.int32)
    holder_expert_ids = np.full((num_owners, num_replicas_per_owner), -1, dtype=np.int32)
    holder_checkpoint_ids = np.full((num_owners, num_replicas_per_owner), -1, dtype=np.int32)

    if num_replicas_per_owner == 0:
        plan = PreloadedReplicaPlan(
            replica_meta=replica_meta,
            holder_expert_ids=holder_expert_ids,
            holder_checkpoint_ids=holder_checkpoint_ids,
            num_owners=num_owners,
            num_replicas_per_owner=0,
            num_experts=num_experts,
            placement=perm,
            owner_to_chip=chip_map,
            max_holders_per_expert=max_holders_per_expert,
        )
        validate_replica_plan(plan)
        return plan

    base_owner_load = placed_counts.reshape(num_owners, e_per_owner).sum(axis=1)
    holder_agg_load = base_owner_load.copy()
    holder_next_slot = np.zeros(num_owners, dtype=np.int32)
    holders_of_expert: list[set[int]] = [set() for _ in range(num_experts)]

    def _pick_holder(home_owner: int, eid: int) -> int | None:
        elig = [
            d
            for d in range(num_owners)
            if d != home_owner
            and holder_next_slot[d] < num_replicas_per_owner
            and d not in holders_of_expert[eid]
        ]
        if not elig:
            return None
        if chip_map is not None and len(set(chip_map.tolist())) > 1:
            home_chip = int(chip_map[home_owner])
            elig = [d for d in elig if int(chip_map[d]) != home_chip]
            if not elig:
                return None
        return min(
            elig,
            key=lambda d: (
                float(holder_agg_load[d]),
                int(holder_next_slot[d]),
                d,
            ),
        )

    out_k = 0
    sel_mode = (selection_mode or "global_top").strip().lower()
    eff_max_holders = max(max_holders_per_expert, 2) if "_h2" in sel_mode else max_holders_per_expert
    if sel_mode.startswith("bottleneck_owner") and placed_raw.sum() > 0.0:
        # Submodular greedy selection directly minimizing E_batch[max_owner_load]
        # under the exact sequential in-kernel SMEM move inequality!
        emp_src = (
            empirical_step_counts
            if empirical_step_counts is not None
            else _lookup_empirical_step_samples(counts)
        )
        if emp_src is not None:
            emp_arr = np.asarray(emp_src, dtype=np.int32)
            if emp_arr.ndim != 2 or emp_arr.shape[1] != num_experts or emp_arr.shape[0] == 0:
                raise ValueError(
                    f"Expected empirical_step_counts shape (M, {num_experts}), got {emp_arr.shape}"
                )
            placed_steps = emp_arr[:, perm] if perm is not None else emp_arr
            step_rows = placed_steps.sum(axis=1, keepdims=True)
            tm_vec = np.where(step_rows <= 64 * max(1, topk), max(1, int(tile_m)), max(64, int(tile_m)))
            cost_m = np.where(
                placed_steps > 0, (placed_steps + tm_vec - 1) // tm_vec, 0
            ).astype(np.int32)
        else:
            rng_mc = np.random.default_rng(1729)
            p_raw = placed_raw / float(placed_raw.sum())
            q_inc = _topk_inclusion_probs(p_raw, topk)
            if "_b64" in sel_mode:
                mc_batches = (max(8, int(calib_batch)),)
                draws_per_b = 192
            else:
                mc_batches = (
                    max(8, calib_batch // 2),
                    max(16, calib_batch),
                    max(32, calib_batch * 2),
                )
                draws_per_b = 64
            cost_list = []
            for b_mc in mc_batches:
                cnt_mc = rng_mc.binomial(int(b_mc), q_inc, size=(draws_per_b, num_experts))
                tm_mc = max(1, int(tile_m)) if b_mc <= 64 else max(64, int(tile_m))
                c_mc = np.where(cnt_mc > 0, (cnt_mc + tm_mc - 1) // tm_mc, 0).astype(
                    np.int32
                )
                cost_list.append(c_mc)
            cost_m = np.concatenate(cost_list, axis=0)  # (M, num_experts)
        n_m = cost_m.shape[0]
        loads_m = (
            cost_m.reshape(n_m, num_owners, e_per_owner)
            .sum(axis=2)
            .astype(np.int32)
        )  # (M, num_owners)
        moved_m = np.zeros((n_m, num_experts), dtype=bool)
        unique_chips = set(chip_map.tolist()) if chip_map is not None else set()
        multi_chip = chip_map is not None and len(unique_chips) > 1
        enforce_distinct_holder_chips = multi_chip and eff_max_holders <= len(unique_chips) - 1
        holder_chips_of_expert: list[set[int]] = [set() for _ in range(num_experts)]

        while out_k < k_rep:
            cur_peak = loads_m.max(axis=1)  # (M,)
            other_max = np.zeros((num_owners, n_m), dtype=np.int32)
            for h_o in range(num_owners):
                cols = [o for o in range(num_owners) if o != h_o]
                other_max[h_o] = loads_m[:, cols].max(axis=1)

            best_key: tuple[float, float, float, float, int, int] | None = None
            best_pair_bn: tuple[int, int] | None = None
            for h_o in range(num_owners):
                elig_holders = [
                    d
                    for d in range(num_owners)
                    if d != h_o
                    and holder_next_slot[d] < num_replicas_per_owner
                    and (not multi_chip or int(chip_map[d]) != int(chip_map[h_o]))
                ]
                if not elig_holders:
                    continue
                base_e = h_o * e_per_owner
                avail_le = [
                    le
                    for le in range(e_per_owner)
                    if len(holders_of_expert[base_e + le]) < eff_max_holders
                ]
                if not avail_le:
                    continue
                eids_arr = np.array([base_e + le for le in avail_le], dtype=np.int32)
                ce_mat = cost_m[:, eids_arr]  # (M, E_avail)
                can_mat = (ce_mat > 0) & (~moved_m[:, eids_arr])
                lh_col = loads_m[:, h_o : h_o + 1]  # (M, 1)
                om_col = other_max[h_o][:, None]  # (M, 1)
                cp_col = cur_peak[:, None]  # (M, 1)
                ho_load = float(base_owner_load[h_o])

                for d in elig_holders:
                    ld_col = loads_m[:, d : d + 1]
                    fires_mat = can_mat & ((lh_col - ce_mat) >= (ld_col + ce_mat))
                    new_pk_mat = np.where(
                        fires_mat,
                        np.maximum(
                            om_col, np.maximum(lh_col - ce_mat, ld_col + ce_mat)
                        ),
                        cp_col,
                    )
                    gains_vec = (cp_col - new_pk_mat).mean(axis=0)  # (E_avail,)
                    d_agg = float(holder_agg_load[d])
                    d_chip = int(chip_map[d]) if multi_chip else -1
                    for idx_a, eid_val in enumerate(eids_arr.tolist()):
                        if d in holders_of_expert[eid_val]:
                            continue
                        if (
                            enforce_distinct_holder_chips
                            and d_chip in holder_chips_of_expert[eid_val]
                        ):
                            continue
                        cand_key = (
                            round(float(gains_vec[idx_a]), 9),
                            ho_load,
                            float(placed_counts[eid_val]),
                            -d_agg,
                            -eid_val,
                            -d,
                        )
                        if best_key is None or cand_key > best_key:
                            best_key = cand_key
                            best_pair_bn = (eid_val, d)

            if best_pair_bn is None:
                break
            eid, holder = best_pair_bn
            home_owner = eid // e_per_owner
            slot = int(holder_next_slot[holder])
            holder_next_slot[holder] += 1
            holders_of_expert[eid].add(holder)
            if multi_chip:
                holder_chips_of_expert[eid].add(int(chip_map[holder]))
            holder_agg_load[holder] += holder_load_alpha * float(
                placed_counts[eid]
            )
            ce_sel = cost_m[:, eid]
            lh_sel = loads_m[:, home_owner]
            ld_sel = loads_m[:, holder]
            fires = (
                (ce_sel > 0)
                & (~moved_m[:, eid])
                & ((lh_sel - ce_sel) >= (ld_sel + ce_sel))
            )
            if np.any(fires):
                ce_f = cost_m[fires, eid]
                loads_m[fires, home_owner] -= ce_f
                loads_m[fires, holder] += ce_f
                moved_m[fires, eid] = True

            replica_meta[0, out_k] = eid
            replica_meta[1, out_k] = home_owner
            replica_meta[2, out_k] = holder
            replica_meta[3, out_k] = slot
            holder_expert_ids[holder, slot] = eid
            holder_checkpoint_ids[holder, slot] = (
                int(perm[eid]) if perm is not None else eid
            )
            out_k += 1
    elif eff_max_holders == 1:
        raw_max = max(1.0, float(placed_raw.max()))
        tie_eps = 1e-4 * (placed_raw / raw_max)
        eff_scores = placed_counts + tie_eps
        home_res_load = base_owner_load.copy()
        # Iterative bottleneck-home-first selection (#435 §2.2 / #437): pick from
        # the currently heaviest eligible home owner's hottest un-replicated expert,
        # updating residual home load and aggregate holder load after each slot.
        while out_k < k_rep:
            owner_order = np.argsort(-home_res_load, kind="stable")
            chosen: tuple[int, int, int] | None = None
            for ho_np in owner_order:
                ho = int(ho_np)
                lo_e, hi_e = ho * e_per_owner, (ho + 1) * e_per_owner
                sub_order = lo_e + np.argsort(-eff_scores[lo_e:hi_e], kind="stable")
                for eid_np in sub_order:
                    eid = int(eid_np)
                    if holders_of_expert[eid] or placed_counts[eid] <= 0.0:
                        continue
                    holder = _pick_holder(ho, eid)
                    if holder is not None:
                        chosen = (eid, ho, holder)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                # Fallback if all positive-count experts were already replicated.
                for eid in range(num_experts):
                    if holders_of_expert[eid]:
                        continue
                    ho = eid // e_per_owner
                    holder = _pick_holder(ho, eid)
                    if holder is not None:
                        chosen = (eid, ho, holder)
                        break
            if chosen is None:
                break
            eid, home_owner, holder = chosen
            slot = int(holder_next_slot[holder])
            holder_next_slot[holder] += 1
            holders_of_expert[eid].add(holder)
            w_e = float(placed_counts[eid])
            home_res_load[home_owner] = max(0.0, home_res_load[home_owner] - holder_load_alpha * w_e)
            holder_agg_load[holder] += holder_load_alpha * w_e

            replica_meta[0, out_k] = eid
            replica_meta[1, out_k] = home_owner
            replica_meta[2, out_k] = holder
            replica_meta[3, out_k] = slot
            holder_expert_ids[holder, slot] = eid
            holder_checkpoint_ids[holder, slot] = int(perm[eid]) if perm is not None else eid
            out_k += 1
    else:
        # Highest-averages (D'Hondt) selection: score = count / (1 + assigned_holders).
        while out_k < k_rep:
            best_cand: tuple[float, int, int, int] | None = None
            for eid in range(num_experts):
                n_h = len(holders_of_expert[eid])
                if n_h >= eff_max_holders:
                    continue
                home_owner = eid // e_per_owner
                holder = _pick_holder(home_owner, eid)
                if holder is None:
                    continue
                score = float(placed_counts[eid]) / float(n_h + 1)
                if best_cand is None or score > best_cand[0] + 1e-12:
                    best_cand = (score, eid, home_owner, holder)
            if best_cand is None:
                break
            _, eid, home_owner, holder = best_cand
            slot = int(holder_next_slot[holder])
            holder_next_slot[holder] += 1
            holders_of_expert[eid].add(holder)
            holder_agg_load[holder] += holder_load_alpha * (
                float(placed_counts[eid]) / float(len(holders_of_expert[eid]))
            )

            replica_meta[0, out_k] = eid
            replica_meta[1, out_k] = home_owner
            replica_meta[2, out_k] = holder
            replica_meta[3, out_k] = slot
            holder_expert_ids[holder, slot] = eid
            holder_checkpoint_ids[holder, slot] = int(perm[eid]) if perm is not None else eid
            out_k += 1

    plan = PreloadedReplicaPlan(
        replica_meta=replica_meta,
        holder_expert_ids=holder_expert_ids,
        holder_checkpoint_ids=holder_checkpoint_ids,
        num_owners=num_owners,
        num_replicas_per_owner=num_replicas_per_owner,
        num_experts=num_experts,
        placement=perm,
        owner_to_chip=chip_map,
        max_holders_per_expert=eff_max_holders,
    )
    validate_replica_plan(plan)
    return plan


def select_hot_non_replicated_home_slots(
    expert_counts: Any,
    *,
    num_owners: int,
    replica_meta: Any | None = None,
    placement: Any | None = None,
) -> np.ndarray:
    """Return each owner's hottest local home slot not replicated in ``replica_meta``.

    Args:
      expert_counts: 1D calibration counts `[num_experts]` in checkpoint order.
      num_owners: Number of EP owners dividing `num_experts`.
      replica_meta: Optional `int32[4, K_rep]` table whose row 0 holds placed
        replicated expert IDs (`>= 0`).
      placement: Optional permutation mapping runtime placed slot -> checkpoint ID.
    """
    counts = np.asarray(expert_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"Expected 1D expert_counts, got shape {counts.shape}")
    num_experts = int(counts.size)
    if num_owners < 2 or num_experts % num_owners != 0:
        raise ValueError(
            f"num_owners={num_owners} must be >= 2 and divide num_experts={num_experts}"
        )
    perm = _validate_permutation(placement, num_experts) if placement is not None else None
    placed_counts = counts[perm] if perm is not None else counts

    replicated_eids: set[int] = set()
    if replica_meta is not None:
        meta = np.asarray(replica_meta)
        if meta.ndim != 2 or meta.shape[0] < 1:
            raise ValueError(f"Invalid replica_meta shape {meta.shape}")
        replicated_eids = {int(e) for e in meta[0].tolist() if int(e) >= 0}

    e_per_owner = num_experts // num_owners
    hot_slots = np.zeros((num_owners,), dtype=np.int32)
    for owner in range(num_owners):
        base = owner * e_per_owner
        cands = [
            le
            for le in range(e_per_owner)
            if (base + le) not in replicated_eids
        ]
        if not cands:
            cands = list(range(e_per_owner))
        hot_slots[owner] = max(
            cands, key=lambda le: (float(placed_counts[base + le]), -le)
        )
    return hot_slots


def validate_replica_plan(plan: PreloadedReplicaPlan) -> None:
    """Strictly validate all structural and disjointness invariants of a plan."""
    num_owners = plan.num_owners
    r = plan.num_replicas_per_owner
    num_experts = plan.num_experts
    if num_owners < 2 or num_experts % num_owners != 0:
        raise ValueError(
            f"Invalid num_owners={num_owners} for num_experts={num_experts}"
        )
    e_per_owner = num_experts // num_owners
    k_rep = num_owners * r

    meta = np.asarray(plan.replica_meta)
    h_ids = np.asarray(plan.holder_expert_ids)
    h_ckpt = np.asarray(plan.holder_checkpoint_ids)
    if meta.shape != (4, k_rep) or meta.dtype != np.int32:
        raise ValueError(
            f"replica_meta must be int32[(4, {k_rep})], got {meta.dtype}{meta.shape}"
        )
    if h_ids.shape != (num_owners, r) or h_ids.dtype != np.int32:
        raise ValueError(
            f"holder_expert_ids must be int32[({num_owners}, {r})], got {h_ids.dtype}{h_ids.shape}"
        )
    if h_ckpt.shape != (num_owners, r) or h_ckpt.dtype != np.int32:
        raise ValueError(
            f"holder_checkpoint_ids must be int32[({num_owners}, {r})], got {h_ckpt.dtype}{h_ckpt.shape}"
        )

    perm = _validate_permutation(plan.placement, num_experts) if plan.placement is not None else None
    seen_holder_slot: set[tuple[int, int]] = set()
    seen_eid_holder: set[tuple[int, int]] = set()
    eid_count: dict[int, int] = {}

    for k in range(k_rep):
        eid, home, holder, slot = (int(meta[i, k]) for i in range(4))
        if eid < 0:
            if (eid, home, holder, slot) != (-1, -1, -1, -1):
                raise ValueError(
                    f"Unused replica_meta column {k} must be (-1,-1,-1,-1), got {(eid, home, holder, slot)}"
                )
            continue
        if not (0 <= eid < num_experts):
            raise ValueError(f"Column {k}: eid={eid} out of [0, {num_experts})")
        expected_home = eid // e_per_owner
        if home != expected_home:
            raise ValueError(
                f"Column {k}: home={home} does not match eid // {e_per_owner} = {expected_home}"
            )
        if not (0 <= holder < num_owners) or holder == home:
            raise ValueError(
                f"Column {k}: invalid holder={holder} for home={home}, num_owners={num_owners}"
            )
        if not (0 <= slot < r):
            raise ValueError(f"Column {k}: slot={slot} out of [0, {r})")
        if (holder, slot) in seen_holder_slot:
            raise ValueError(f"Duplicate (holder, slot)={(holder, slot)} at column {k}")
        seen_holder_slot.add((holder, slot))
        if (eid, holder) in seen_eid_holder:
            raise ValueError(f"Duplicate (eid, holder)={(eid, holder)} at column {k}")
        seen_eid_holder.add((eid, holder))
        eid_count[eid] = eid_count.get(eid, 0) + 1
        if eid_count[eid] > plan.max_holders_per_expert:
            raise ValueError(
                f"Expert {eid} assigned {eid_count[eid]} holders > max_holders_per_expert={plan.max_holders_per_expert}"
            )
        if int(h_ids[holder, slot]) != eid:
            raise ValueError(
                f"holder_expert_ids[{holder}, {slot}]={h_ids[holder, slot]} != meta eid={eid}"
            )
        expected_ckpt = int(perm[eid]) if perm is not None else eid
        if int(h_ckpt[holder, slot]) != expected_ckpt:
            raise ValueError(
                f"holder_checkpoint_ids[{holder}, {slot}]={h_ckpt[holder, slot]} != {expected_ckpt}"
            )

    for d in range(num_owners):
        for s in range(r):
            if (d, s) not in seen_holder_slot:
                if int(h_ids[d, s]) != -1 or int(h_ckpt[d, s]) != -1:
                    raise ValueError(
                        f"Unreferenced slot ({d}, {s}) must be -1, got ({h_ids[d, s]}, {h_ckpt[d, s]})"
                    )


def slice_replica_weights_for_owner(
    full_w1: Any,
    full_s1: Any,
    full_w2: Any,
    full_s2: Any,
    plan: PreloadedReplicaPlan,
    owner_id: int,
    *,
    tp_half: int = 0,
    tp_size: int = 1,
    use_checkpoint_ids: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Build `(rep_w1, rep_s1, rep_w2, rep_s2)` for one owner (and optional TP2 half).

    Supports both whole-expert EP (`tp_size=1`) and halved-intermediate TP2
    (`tp_size=2`, `tp_half in {0, 1}`), both K256 and K16 scales, and NumPy /
    JAX / PyTorch tensors (including packed `torch.float4_e2m1fn_x2` or unpacked
    `jnp.float4_e2m1fn`).

    Under `tp_size=2`, `full_w1` has gate columns `[0 : N1//2]` and up columns
    `[N1//2 : N1]` along axis -1, and `full_w2` has reduction dimension along
    axis 1; half `tp_half` takes:
      - `rep_w1`: concat(gate_half, up_half) along axis -1
      - `rep_s1`: concat(s1_gate_half, s1_up_half) along axis -1
      - `rep_w2`: slice `full_w2[:, half_i0:half_i1, :]`
      - `rep_s2`: slice `full_s2[:, half_s0:half_s1, :]`
    """
    if tp_size not in (1, 2):
        raise ValueError(f"tp_size must be 1 or 2, got {tp_size}")
    if not (0 <= tp_half < tp_size):
        raise ValueError(f"tp_half={tp_half} out of range for tp_size={tp_size}")
    if not (0 <= owner_id < plan.num_owners):
        raise ValueError(f"owner_id={owner_id} out of [0, {plan.num_owners})")

    slot_ids = (
        plan.holder_checkpoint_ids[owner_id]
        if use_checkpoint_ids
        else plan.holder_expert_ids[owner_id]
    )
    safe_ids = np.where(slot_ids >= 0, slot_ids, 0).astype(np.int32)
    is_valid = slot_ids >= 0

    mod_name = type(full_w1).__module__
    is_torch = mod_name.startswith("torch")
    is_jax = mod_name.startswith("jax")

    def _gather_and_zero(arr: Any) -> Any:
        if is_torch:
            import torch

            idx = torch.as_tensor(safe_ids, dtype=torch.long, device=arr.device)
            gathered = arr.index_select(0, idx).clone()
            if not np.all(is_valid):
                mask = torch.as_tensor(is_valid, dtype=torch.bool, device=arr.device)
                view_shape = (len(safe_ids),) + (1,) * (gathered.ndim - 1)
                if gathered.dtype == getattr(torch, "float4_e2m1fn_x2", None):
                    raw = gathered.view(torch.uint8)
                    raw = torch.where(mask.view(view_shape), raw, torch.zeros_like(raw))
                    gathered = raw.view(gathered.dtype)
                else:
                    gathered = torch.where(mask.view(view_shape), gathered, torch.zeros_like(gathered))
            return gathered
        if is_jax:
            import jax.numpy as jnp

            gathered = arr[jnp.asarray(safe_ids, dtype=jnp.int32)]
            if not np.all(is_valid):
                mask = jnp.asarray(is_valid).reshape((len(safe_ids),) + (1,) * (gathered.ndim - 1))
                gathered = jnp.where(mask, gathered, jnp.zeros_like(gathered))
            return gathered
        gathered = np.asarray(arr)[safe_ids].copy()
        if not np.all(is_valid):
            gathered[~is_valid] = 0
        return gathered

    rw1 = _gather_and_zero(full_w1)
    rs1 = _gather_and_zero(full_s1)
    rw2 = _gather_and_zero(full_w2)
    rs2 = _gather_and_zero(full_s2)

    if tp_size == 1:
        return rw1, rs1, rw2, rs2

    # TP2 split along intermediate dimension (matching gate + up halves in W1/S1
    # and reduction rows in W2/S2).
    n1_half_gate_up = rw1.shape[-1] // 2
    n1_quarter = n1_half_gate_up // 2
    s1_half_gate_up = rs1.shape[-1] // 2
    s1_quarter = s1_half_gate_up // 2
    i_half = rw2.shape[1] // 2
    s2_half = rs2.shape[1] // 2

    g0 = tp_half * n1_quarter
    u0 = n1_half_gate_up + tp_half * n1_quarter
    sg0 = tp_half * s1_quarter
    su0 = s1_half_gate_up + tp_half * s1_quarter
    w2_0 = tp_half * i_half
    s2_0 = tp_half * s2_half

    if is_torch:
        import torch

        rw1 = torch.cat(
            [rw1[..., g0 : g0 + n1_quarter], rw1[..., u0 : u0 + n1_quarter]], dim=-1
        ).contiguous()
        rs1 = torch.cat(
            [rs1[..., sg0 : sg0 + s1_quarter], rs1[..., su0 : su0 + s1_quarter]], dim=-1
        ).contiguous()
        rw2 = rw2[:, w2_0 : w2_0 + i_half, :].contiguous()
        rs2 = rs2[:, s2_0 : s2_0 + s2_half, :].contiguous()
    elif is_jax:
        import jax.numpy as jnp

        rw1 = jnp.concatenate(
            [rw1[..., g0 : g0 + n1_quarter], rw1[..., u0 : u0 + n1_quarter]], axis=-1
        )
        rs1 = jnp.concatenate(
            [rs1[..., sg0 : sg0 + s1_quarter], rs1[..., su0 : su0 + s1_quarter]], axis=-1
        )
        rw2 = rw2[:, w2_0 : w2_0 + i_half, :]
        rs2 = rs2[:, s2_0 : s2_0 + s2_half, :]
    else:
        rw1 = np.ascontiguousarray(
            np.concatenate(
                [rw1[..., g0 : g0 + n1_quarter], rw1[..., u0 : u0 + n1_quarter]], axis=-1
            )
        )
        rs1 = np.ascontiguousarray(
            np.concatenate(
                [rs1[..., sg0 : sg0 + s1_quarter], rs1[..., su0 : su0 + s1_quarter]], axis=-1
            )
        )
        rw2 = np.ascontiguousarray(rw2[:, w2_0 : w2_0 + i_half, :])
        rs2 = np.ascontiguousarray(rs2[:, s2_0 : s2_0 + s2_half, :])

    return rw1, rs1, rw2, rs2


def build_replica_weight_bank_from_sharded(
    sharded_w1: Sequence[Any],
    sharded_s1: Sequence[Any],
    sharded_w2: Sequence[Any],
    sharded_s2: Sequence[Any],
    plan: PreloadedReplicaPlan,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Build per-owner replica banks `[owner] -> (R, ...)` from per-owner base shards.

    Works identically for pure-EP shards (`[num_owners][E_owner, ...]`) and
    pre-split TP2 shards (call once per TP half, or pass per-rank shards indexed
    by owner). Zero full-model materialization required.
    """
    num_owners = plan.num_owners
    e_per_owner = plan.num_experts // num_owners
    if len(sharded_w1) != num_owners:
        raise ValueError(
            f"Expected {num_owners} owner shards, got {len(sharded_w1)}"
        )

    is_jax = type(sharded_w1[0]).__module__.startswith("jax")
    is_torch = type(sharded_w1[0]).__module__.startswith("torch")

    def _stack(items: list[Any]) -> Any:
        if is_torch:
            import torch

            return torch.stack(items, dim=0).contiguous()
        if is_jax:
            import jax.numpy as jnp

            return jnp.stack(items, axis=0)
        return np.ascontiguousarray(np.stack(items, axis=0))

    def _zeros_like_one(sample: Any) -> Any:
        if is_torch:
            import torch

            if sample.dtype == getattr(torch, "float4_e2m1fn_x2", None):
                return torch.zeros(
                    sample.shape, dtype=torch.uint8, device=sample.device
                ).view(sample.dtype)
            return torch.zeros_like(sample)
        if is_jax:
            import jax.numpy as jnp

            return jnp.zeros_like(sample)
        return np.zeros_like(sample)

    out_w1, out_s1, out_w2, out_s2 = [], [], [], []
    for owner in range(num_owners):
        w1_slots, s1_slots, w2_slots, s2_slots = [], [], [], []
        for slot in range(plan.num_replicas_per_owner):
            eid = int(plan.holder_expert_ids[owner, slot])
            if eid < 0:
                w1_slots.append(_zeros_like_one(sharded_w1[owner][0]))
                s1_slots.append(_zeros_like_one(sharded_s1[owner][0]))
                w2_slots.append(_zeros_like_one(sharded_w2[owner][0]))
                s2_slots.append(_zeros_like_one(sharded_s2[owner][0]))
            else:
                src_owner = eid // e_per_owner
                src_local = eid % e_per_owner
                w1_slots.append(sharded_w1[src_owner][src_local])
                s1_slots.append(sharded_s1[src_owner][src_local])
                w2_slots.append(sharded_w2[src_owner][src_local])
                s2_slots.append(sharded_s2[src_owner][src_local])
        out_w1.append(_stack(w1_slots))
        out_s1.append(_stack(s1_slots))
        out_w2.append(_stack(w2_slots))
        out_s2.append(_stack(s2_slots))
    return out_w1, out_s1, out_w2, out_s2


# =============================================================================
# Pure-NumPy Deterministic Reference Planners (Bit-Exact Spec for SMEM Planners)
# =============================================================================


def _expert_cost_py(
    cnt: int,
    *,
    cost_mode: str,
    tile_m: int = 64,
    base_cost: int = 32,
    row_weight: int = 1,
) -> int:
    if cnt <= 0:
        return 0
    mode = normalize_replica_cost_mode(cost_mode)
    if mode == "active":
        return 1
    if mode == "tiles":
        return (cnt + tile_m - 1) // tile_m
    return base_cost + row_weight * cnt


@dataclass(frozen=True)
class ReplicaDecisionRef:
    """Reference output of the deterministic Section 4.5b runtime planner."""

    owner_loads_before: np.ndarray  # int32[num_owners]
    owner_loads_after: np.ndarray  # int32[num_owners]
    moved_mask: np.ndarray  # int32[num_experts], 1 if moved to holder
    exec_owner_of_expert: np.ndarray  # int32[num_experts], executing owner
    local_suppress: np.ndarray  # int32[num_owners, e_owner]
    holder_slot_expert: np.ndarray  # int32[num_owners, R], global eid or -1
    num_moves: int


def plan_preloaded_replicas_ref(
    global_counts: Any,
    replica_meta: Any,
    *,
    num_owners: int,
    num_replicas_per_owner: int,
    cost_mode: str = "active",
    tile_m: int = 64,
    base_cost: int = 32,
    row_weight: int = 1,
    greedy_peak_first: bool = False,
    chip_aware: bool = False,
    chipmate_xor: int = 1,
) -> ReplicaDecisionRef:
    """Exact NumPy reference matching `plan_preloaded_replicas_smem` bit-for-bit."""
    mode = normalize_replica_cost_mode(cost_mode)
    counts = np.asarray(global_counts, dtype=np.int32)
    if counts.ndim == 2:
        counts = counts[0]
    num_experts = int(counts.shape[0])
    e_owner = num_experts // num_owners
    meta = np.asarray(replica_meta, dtype=np.int32)
    k_rep = int(meta.shape[1])

    owner_loads = np.zeros(num_owners, dtype=np.int32)
    for o in range(num_owners):
        acc = 0
        base = o * e_owner
        for le in range(e_owner):
            acc += _expert_cost_py(
                int(counts[base + le]),
                cost_mode=mode,
                tile_m=tile_m,
                base_cost=base_cost,
                row_weight=row_weight,
            )
        owner_loads[o] = acc
    owner_loads_before = owner_loads.copy()

    moved_mask = np.zeros(num_experts, dtype=np.int32)
    exec_owner = np.arange(num_experts, dtype=np.int32) // e_owner
    local_suppress = np.zeros((num_owners, e_owner), dtype=np.int32)
    holder_slot_expert = np.full((num_owners, num_replicas_per_owner), -1, dtype=np.int32)
    num_moves = 0

    def _can_accept_py(home: int, holder: int, c: int) -> tuple[bool, int, int]:
        lh = int(owner_loads[home])
        ld = int(owner_loads[holder])
        if chip_aware:
            lh_mate = int(owner_loads[home ^ chipmate_xor])
            ld_mate = int(owner_loads[holder ^ chipmate_xor])
            chip_home = lh + lh_mate
            chip_holder = ld + ld_mate
            margin = (chip_home - c) - (chip_holder + c)
            ok = (margin >= 0) and ((ld + c) <= max(lh, lh_mate))
            return ok, max(lh, lh_mate), margin
        margin = (lh - c) - (ld + c)
        return (margin >= 0), lh, margin

    if not greedy_peak_first:
        for k in range(k_rep):
            eid = int(meta[0, k])
            if eid < 0:
                continue
            cnt = int(counts[eid])
            if cnt <= 0 or moved_mask[eid] != 0:
                continue
            home = int(meta[1, k])
            holder = int(meta[2, k])
            slot = int(meta[3, k])
            c = _expert_cost_py(
                cnt,
                cost_mode=mode,
                tile_m=tile_m,
                base_cost=base_cost,
                row_weight=row_weight,
            )
            ok, _, _ = _can_accept_py(home, holder, c)
            if ok:
                owner_loads[home] -= c
                owner_loads[holder] += c
                moved_mask[eid] = 1
                exec_owner[eid] = holder
                local_suppress[home, eid - home * e_owner] = 1
                holder_slot_expert[holder, slot] = eid
                num_moves += 1
    else:
        used_k = np.zeros(k_rep, dtype=np.int32)
        for _ in range(k_rep):
            best_k = -1
            best_lh = -1
            best_margin = -1
            for k in range(k_rep):
                if used_k[k] != 0:
                    continue
                eid = int(meta[0, k])
                if eid < 0:
                    continue
                cnt = int(counts[eid])
                if cnt <= 0 or moved_mask[eid] != 0:
                    continue
                home = int(meta[1, k])
                holder = int(meta[2, k])
                c = _expert_cost_py(
                    cnt,
                    cost_mode=mode,
                    tile_m=tile_m,
                    base_cost=base_cost,
                    row_weight=row_weight,
                )
                ok, peak_h, margin = _can_accept_py(home, holder, c)
                if ok:
                    if (peak_h > best_lh) or (peak_h == best_lh and margin > best_margin):
                        best_k = k
                        best_lh = peak_h
                        best_margin = margin
            if best_k < 0:
                break
            eid = int(meta[0, best_k])
            home = int(meta[1, best_k])
            holder = int(meta[2, best_k])
            slot = int(meta[3, best_k])
            c = _expert_cost_py(
                int(counts[eid]),
                cost_mode=mode,
                tile_m=tile_m,
                base_cost=base_cost,
                row_weight=row_weight,
            )
            owner_loads[home] -= c
            owner_loads[holder] += c
            used_k[best_k] = 1
            moved_mask[eid] = 1
            exec_owner[eid] = holder
            local_suppress[home, eid - home * e_owner] = 1
            holder_slot_expert[holder, slot] = eid
            num_moves += 1

    return ReplicaDecisionRef(
        owner_loads_before=owner_loads_before,
        owner_loads_after=owner_loads,
        moved_mask=moved_mask,
        exec_owner_of_expert=exec_owner,
        local_suppress=local_suppress,
        holder_slot_expert=holder_slot_expert,
        num_moves=num_moves,
    )


@dataclass(frozen=True)
class ChipmateDecisionRef:
    """Reference output of the deterministic Section 4.3 chipmate planner."""

    send_slot_per_rank: np.ndarray  # int32[ep], local slot or -1
    recv_eid_per_rank: np.ndarray  # int32[ep], global eid or -1
    active_before: np.ndarray  # int32[ep]
    active_after: np.ndarray  # int32[ep]
    rows_before: np.ndarray  # int32[ep]
    rows_after: np.ndarray  # int32[ep]
    num_moves: int


def plan_chipmate_move_ref(
    global_counts: Any,
    *,
    ep: int,
    chipmate_xor: int = 1,
    chipmate_map: Sequence[int] | np.ndarray | None = None,
    replica_decision: ReplicaDecisionRef | None = None,
    min_active_gap: int = 2,
    require_global_max: bool = False,
    global_max_slack: int = 0,
    max_move_rows: int = 1 << 30,
) -> ChipmateDecisionRef:
    """Exact NumPy reference matching `plan_chipmate_move_smem` bit-for-bit.

    Composed cleanly after Section 4.5b when `replica_decision` is provided:
    uses post-replica active counts and row totals per rank, and only migrates
    a non-suppressed local home expert (`moved_mask[eid] == 0`).
    """
    counts = np.asarray(global_counts, dtype=np.int32)
    if counts.ndim == 2:
        counts = counts[0]
    num_experts = int(counts.shape[0])
    e_local = num_experts // ep

    if chipmate_map is not None:
        cm_map = np.asarray(chipmate_map, dtype=np.int32)
    else:
        cm_map = np.array([r ^ chipmate_xor for r in range(ep)], dtype=np.int32)

    active = np.zeros(ep, dtype=np.int32)
    rows = np.zeros(ep, dtype=np.int32)
    moved_mask = (
        replica_decision.moved_mask
        if replica_decision is not None
        else np.zeros(num_experts, dtype=np.int32)
    )
    for eid in range(num_experts):
        c = int(counts[eid])
        if c > 0:
            owner = (
                int(replica_decision.exec_owner_of_expert[eid])
                if replica_decision is not None
                else (eid // e_local)
            )
            active[owner] += 1
            rows[owner] += c

    active_before = active.copy()
    rows_before = rows.copy()
    global_max_active = int(active.max()) if ep > 0 else 0

    send_slot = np.full(ep, -1, dtype=np.int32)
    recv_eid = np.full(ep, -1, dtype=np.int32)
    num_moves = 0

    for r in range(ep):
        partner = int(cm_map[r])
        if r >= partner:
            continue
        if active[r] >= active[partner]:
            sender, receiver = r, partner
        else:
            sender, receiver = partner, r
        gap = int(active[sender] - active[receiver])
        if gap < min_active_gap:
            continue
        if require_global_max and int(active[sender]) < global_max_active - global_max_slack:
            continue
        pair_max_rows = max(int(rows[sender]), int(rows[receiver]))
        best_le = -1
        best_cnt = 1 << 30
        base = sender * e_local
        for le in range(e_local):
            eid = base + le
            cnt = int(counts[eid])
            if cnt <= 0 or moved_mask[eid] != 0 or cnt > max_move_rows:
                continue
            if int(rows[receiver]) + cnt > pair_max_rows:
                continue
            if cnt < best_cnt:
                best_cnt = cnt
                best_le = le
        if best_le >= 0:
            eid = base + best_le
            send_slot[sender] = best_le
            recv_eid[receiver] = eid
            active[sender] -= 1
            active[receiver] += 1
            rows[sender] -= best_cnt
            rows[receiver] += best_cnt
            num_moves += 1

    return ChipmateDecisionRef(
        send_slot_per_rank=send_slot,
        recv_eid_per_rank=recv_eid,
        active_before=active_before,
        active_after=active,
        rows_before=rows_before,
        rows_after=rows,
        num_moves=num_moves,
    )


# =============================================================================
# In-Kernel Deterministic Pallas SMEM Planners
# =============================================================================


class ReplicaPlannerScratch(NamedTuple):
    """SMEM scratch specs for `plan_preloaded_replicas_smem`."""

    owner_load: Any
    local_suppress: Any
    holder_slot_expert: Any
    moved_flag: Any


class ChipmatePlannerScratch(NamedTuple):
    """SMEM scratch specs for `plan_chipmate_move_smem`."""

    rank_active: Any
    rank_rows: Any
    cm_decision: Any  # int32[3]: [0]=send_slot, [1]=recv_eid, [2]=partner_rank


def replica_planner_scratch_shapes(
    num_owners: int,
    num_replicas: int,
    e_owner: int,
    *,
    include_moved_flag: bool = True,
):
    """Return `ReplicaPlannerScratch` of `pltpu.SMEM` specs for Pallas calls."""
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    return ReplicaPlannerScratch(
        owner_load=pltpu.SMEM((num_owners,), jnp.int32),
        local_suppress=pltpu.SMEM((e_owner,), jnp.int32),
        holder_slot_expert=pltpu.SMEM((max(1, num_replicas),), jnp.int32),
        moved_flag=pltpu.SMEM((NUM_EXPERTS_DEFAULT,), jnp.int32)
        if include_moved_flag
        else None,
    )


def chipmate_planner_scratch_shapes(ep: int) -> ChipmatePlannerScratch:
    """Return `ChipmatePlannerScratch` of `pltpu.SMEM` specs for Pallas calls."""
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    return ChipmatePlannerScratch(
        rank_active=pltpu.SMEM((ep,), jnp.int32),
        rank_rows=pltpu.SMEM((ep,), jnp.int32),
        cm_decision=pltpu.SMEM((3,), jnp.int32),
    )


def _expert_cost_smem(
    cnt,
    *,
    cost_mode: str,
    tile_m: int,
    base_cost: int,
    row_weight: int,
):
    import jax.numpy as jnp
    from jax import lax

    mode = normalize_replica_cost_mode(cost_mode)
    is_active = (cnt > 0).astype(jnp.int32)
    if mode == "active":
        return is_active
    if mode == "tiles":
        if tile_m > 0 and (tile_m & (tile_m - 1)) == 0:
            shift = jnp.int32(tile_m.bit_length() - 1)
            tiles = lax.shift_right_logical(cnt + jnp.int32(tile_m - 1), shift)
        else:
            tiles = (cnt + jnp.int32(tile_m - 1)) // jnp.int32(tile_m)
        return jnp.where(cnt > 0, tiles, jnp.int32(0))
    return jnp.where(
        cnt > 0,
        jnp.int32(base_cost) + jnp.int32(row_weight) * cnt,
        jnp.int32(0),
    )


def precompute_imbalance_loads_vpu(
    counts_row,
    *,
    num_owners: int,
    cost_mode: str = "active",
    tile_m: int = 64,
    base_cost: int = 18,
    row_weight: int = 1,
):
    """Pack `(counts, owner_loads, rank_active, rank_rows)` into `(8, 256) int32`.

    Uses `moe_lowtok`'s `(8, 256) @ (256, 128)` MXU indicator dot so every
    intermediate tensor is an exact multiple of `(8, 128)` VREG tiles (zero
    unaligned `vector<8x32xi32>` reshapes or 1D pads!).
    """
    import jax.numpy as jnp
    from jax import lax

    c_row = counts_row.reshape(1, -1)
    num_experts = int(c_row.shape[1])
    e_owner = num_experts // num_owners
    mode = normalize_replica_cost_mode(cost_mode)

    c_bcast = jnp.broadcast_to(c_row, (8, num_experts))
    active_bcast = (c_bcast > 0).astype(jnp.int32)
    if mode == "active":
        cost_bcast = active_bcast
    elif mode == "tiles":
        if tile_m > 0 and (tile_m & (tile_m - 1)) == 0:
            shift = jnp.int32(tile_m.bit_length() - 1)
            tiles = lax.shift_right_logical(c_bcast + jnp.int32(tile_m - 1), shift)
        else:
            tiles = (c_bcast + jnp.int32(tile_m - 1)) // jnp.int32(tile_m)
        cost_bcast = jnp.where(c_bcast > 0, tiles, jnp.int32(0))
    else:
        cost_bcast = jnp.where(
            c_bcast > 0,
            jnp.int32(base_cost) + jnp.int32(row_weight) * c_bcast,
            jnp.int32(0),
        )

    sublane = lax.broadcasted_iota(jnp.int32, (8, num_experts), 0)
    in_8x256 = jnp.where(
        sublane == 1,
        cost_bcast.astype(jnp.float32),
        jnp.where(
            sublane == 2,
            active_bcast.astype(jnp.float32),
            jnp.where(sublane == 3, c_bcast.astype(jnp.float32), jnp.float32(0.0)),
        ),
    )

    iota_e = lax.broadcasted_iota(jnp.int32, (num_experts, 128), 0)
    if e_owner > 0 and (e_owner & (e_owner - 1)) == 0:
        e_owner_col = lax.shift_right_logical(
            iota_e, jnp.int32(e_owner.bit_length() - 1)
        )
    else:
        e_owner_col = iota_e // jnp.int32(e_owner)
    lane_128 = lax.broadcasted_iota(jnp.int32, (num_experts, 128), 1)
    one_hot_128 = (
        (e_owner_col == lane_128) & (lane_128 < jnp.int32(num_owners))
    ).astype(jnp.float32)

    loads_128 = jnp.rint(
        jnp.dot(
            in_8x256,
            one_hot_128,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
    ).astype(jnp.int32)
    loads_256 = jnp.concatenate(
        [loads_128, jnp.zeros((8, num_experts - 128), jnp.int32)], axis=1
    )
    return jnp.where(sublane == 0, c_bcast, loads_256)


def _read_count_smem(counts_smem, eid):
    if len(counts_smem.shape) == 2:
        return counts_smem[0, eid]
    return counts_smem[eid]


def _read_replica_planner_env_flags() -> tuple[bool, bool]:
    import os

    try:
        from sglang_tpu.envs import envs

        gp = getattr(envs, "SGLANG_TPU_MUSE_DECODE_REPLICA_GREEDY_PEAK", None)
        ca = getattr(envs, "SGLANG_TPU_MUSE_DECODE_REPLICA_CHIP_AWARE", None)
        return (
            bool(gp.get()) if gp is not None else False,
            bool(ca.get()) if ca is not None else False,
        )
    except Exception:
        return (
            os.environ.get("SGLANG_TPU_MUSE_DECODE_REPLICA_GREEDY_PEAK", "").strip().lower()
            in ("1", "true", "yes"),
            os.environ.get("SGLANG_TPU_MUSE_DECODE_REPLICA_CHIP_AWARE", "").strip().lower()
            in ("1", "true", "yes"),
        )


def plan_preloaded_replicas_smem(
    my_owner,
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
    base_cost: int = 18,
    row_weight: int = 1,
    moved_flag_smem=None,
    greedy_peak_first: bool | None = None,
    vpu_precomputed_loads: bool = False,
    chip_aware: bool | None = None,
    chipmate_xor: int = 1,
    fast_unroll: bool = False,
) -> None:
    """Deterministic in-kernel Section 4.5b home-vs-holder planner in Pallas SMEM.

    Signature aligns directly with Track 1 (`fused_decode_routing.py`) and
    Track 2 (`fused_decode_ep_tp2.py`). Zero collectives; executes pure `int32`
    scalar math so all ranks/chipmates make identical decisions.

    Args:
      my_owner: Scalar `int32` EP owner ID of this rank (`rank` for pure EP;
        `chip_id` for EPxTP2).
      e_owner: Compile-time number of home experts per owner (`256 // num_owners`).
      num_owners: Compile-time number of EP owners (`4`, `8`, or `16`).
      num_replicas: Compile-time replica slots `R` per owner.
      counts_smem: Replicated SMEM counts, shape `(8, 256)` or `(256,)` `int32`.
      replica_meta_smem: Replicated SMEM `int32[4, K_rep]` table.
      local_suppress_smem: Output SMEM `int32[e_owner]` (`1` = moved to holder).
      holder_slot_expert_smem: Output SMEM `int32[R]` (global expert ID or `-1`).
      owner_load_smem: Scratch SMEM `int32[num_owners]`.
      cost_mode: `"active"`, `"tiles"`, or `"weighted"` (alias `"hybrid"`).
      tile_m: Tile height for `"tiles"` cost mode.
      base_cost: Fixed per-active-expert weight-load proxy for `"weighted"` mode.
      row_weight: Per-row compute proxy weight for `"weighted"` mode.
      moved_flag_smem: Optional SMEM `int32[256]` or bitset `int32[8]`; when
        provided, prevents double moves for multi-holder tables and records
        global move state for Section 4.3 chipmate composition.
      greedy_peak_first: If True, executes heaviest-home-first greedy rounds
        (`O(K_rep^2)`, still `< 0.3 us` for `K_rep <= 16`) instead of single pass.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    if greedy_peak_first is None or chip_aware is None:
        env_gp, env_ca = _read_replica_planner_env_flags()
        if greedy_peak_first is None:
            greedy_peak_first = env_gp
        if chip_aware is None:
            chip_aware = env_ca

    mode = normalize_replica_cost_mode(cost_mode)
    k_rep = int(replica_meta_smem.shape[1])
    num_experts = num_owners * e_owner

    # 1. Initialize local outputs and per-owner aggregate costs.
    if fast_unroll and e_owner <= 32:
        for le in range(e_owner):
            local_suppress_smem[le] = jnp.int32(0)
    else:
        @pl.loop(0, e_owner)
        def _init_sup(le):
            local_suppress_smem[le] = jnp.int32(0)

    if fast_unroll and num_replicas <= 8:
        for s in range(num_replicas):
            holder_slot_expert_smem[s] = jnp.int32(-1)
    else:
        @pl.loop(0, num_replicas)
        def _init_slots(s):
            holder_slot_expert_smem[s] = jnp.int32(-1)

    moved_is_bitset = (
        moved_flag_smem is not None and int(moved_flag_smem.shape[0]) == 8
    )
    if moved_is_bitset:
        if fast_unroll:
            for w in range(8):
                moved_flag_smem[w] = jnp.int32(0)
        else:
            @pl.loop(0, 8)
            def _init_moved_bits(w):
                moved_flag_smem[w] = jnp.int32(0)

    elif moved_flag_smem is not None:

        @pl.loop(0, num_experts)
        def _init_moved(e):
            moved_flag_smem[e] = jnp.int32(0)

    if vpu_precomputed_loads and len(counts_smem.shape) == 2:
        if fast_unroll and num_owners <= 16:
            for o in range(num_owners):
                owner_load_smem[o] = counts_smem[1, o]
        else:
            @pl.loop(0, num_owners)
            def _copy_owner_vpu(o):
                owner_load_smem[o] = counts_smem[1, o]

    else:

        @pl.loop(0, num_owners)
        def _accum_owner(o):
            base = o * jnp.int32(e_owner)

            def _sum_local(le, acc):
                cnt = _read_count_smem(counts_smem, base + le)
                return acc + _expert_cost_smem(
                    cnt,
                    cost_mode=mode,
                    tile_m=tile_m,
                    base_cost=base_cost,
                    row_weight=row_weight,
                )

            owner_load_smem[o] = lax.fori_loop(0, e_owner, _sum_local, jnp.int32(0))

    if not greedy_peak_first and fast_unroll:

        def _decide_step_branchless(k):
            eid = replica_meta_smem[0, k]
            valid = eid >= 0
            safe_eid = jnp.clip(eid, jnp.int32(0), jnp.int32(num_experts - 1))
            cnt = jnp.where(valid, _read_count_smem(counts_smem, safe_eid), jnp.int32(0))
            if moved_is_bitset:
                word_idx = lax.shift_right_logical(safe_eid, jnp.int32(5))
                bit_mask = lax.shift_left(jnp.int32(1), safe_eid & jnp.int32(31))
                cur_m = moved_flag_smem[word_idx]
                not_moved = (cur_m & bit_mask) == 0
            elif moved_flag_smem is not None:
                cur_m = moved_flag_smem[safe_eid]
                not_moved = cur_m == 0
            else:
                cur_m = jnp.int32(0)
                not_moved = jnp.bool_(True)
            home = jnp.clip(
                jnp.where(valid, replica_meta_smem[1, k], jnp.int32(0)),
                jnp.int32(0),
                jnp.int32(num_owners - 1),
            )
            holder = jnp.clip(
                jnp.where(valid, replica_meta_smem[2, k], jnp.int32(0)),
                jnp.int32(0),
                jnp.int32(num_owners - 1),
            )
            slot = jnp.clip(
                jnp.where(valid, replica_meta_smem[3, k], jnp.int32(0)),
                jnp.int32(0),
                jnp.int32(max(0, num_replicas - 1)),
            )
            c = _expert_cost_smem(
                cnt,
                cost_mode=mode,
                tile_m=tile_m,
                base_cost=base_cost,
                row_weight=row_weight,
            )
            lh = owner_load_smem[home]
            ld = owner_load_smem[holder]
            if chip_aware:
                lh_mate = owner_load_smem[lax.bitwise_xor(home, jnp.int32(chipmate_xor))]
                ld_mate = owner_load_smem[lax.bitwise_xor(holder, jnp.int32(chipmate_xor))]
                chip_home = lh + lh_mate
                chip_holder = ld + ld_mate
                ineq = ((chip_home - c) >= (chip_holder + c)) & (
                    (ld + c) <= jnp.maximum(lh, lh_mate)
                )
            else:
                ineq = (lh - c) >= (ld + c)
            accept = valid & (cnt > 0) & not_moved & ineq & (home != holder)
            owner_load_smem[home] = jnp.where(accept, lh - c, lh)
            owner_load_smem[holder] = jnp.where(accept, ld + c, ld)
            if moved_is_bitset:
                moved_flag_smem[word_idx] = jnp.where(
                    accept, cur_m | bit_mask, cur_m
                )
            elif moved_flag_smem is not None:
                moved_flag_smem[safe_eid] = jnp.where(
                    accept, jnp.asarray(k, jnp.int32) + jnp.int32(1), cur_m
                )
            is_h = accept & (home == my_owner)
            h_loc = jnp.clip(
                jnp.where(is_h, safe_eid - home * jnp.int32(e_owner), jnp.int32(0)),
                jnp.int32(0),
                jnp.int32(e_owner - 1),
            )
            local_suppress_smem[h_loc] = jnp.where(
                is_h, jnp.int32(1), local_suppress_smem[h_loc]
            )
            is_d = accept & (holder == my_owner)
            holder_slot_expert_smem[slot] = jnp.where(
                is_d, safe_eid, holder_slot_expert_smem[slot]
            )

        if int(fast_unroll) >= 2 and k_rep <= 64:
            for k_i in range(k_rep):
                _decide_step_branchless(k_i)
        else:
            @pl.loop(0, k_rep)
            def _decide_fast(k):
                _decide_step_branchless(k)

    elif not greedy_peak_first:

        @pl.loop(0, k_rep)
        def _decide(k):
            eid = replica_meta_smem[0, k]

            @pl.when(eid >= 0)
            def _check_valid():
                cnt = _read_count_smem(counts_smem, eid)
                if moved_is_bitset:
                    word_idx = lax.shift_right_logical(eid, jnp.int32(5))
                    bit_mask = lax.shift_left(jnp.int32(1), eid & jnp.int32(31))
                    not_moved = (moved_flag_smem[word_idx] & bit_mask) == 0
                elif moved_flag_smem is not None:
                    not_moved = moved_flag_smem[eid] == 0
                else:
                    not_moved = jnp.bool_(True)

                @pl.when((cnt > 0) & not_moved)
                def _check_active():
                    home = replica_meta_smem[1, k]
                    holder = replica_meta_smem[2, k]
                    slot = replica_meta_smem[3, k]
                    c = _expert_cost_smem(
                        cnt,
                        cost_mode=mode,
                        tile_m=tile_m,
                        base_cost=base_cost,
                        row_weight=row_weight,
                    )
                    lh = owner_load_smem[home]
                    ld = owner_load_smem[holder]
                    if chip_aware:
                        lh_mate = owner_load_smem[
                            lax.bitwise_xor(home, jnp.int32(chipmate_xor))
                        ]
                        ld_mate = owner_load_smem[
                            lax.bitwise_xor(holder, jnp.int32(chipmate_xor))
                        ]
                        chip_home = lh + lh_mate
                        chip_holder = ld + ld_mate
                        accept = ((chip_home - c) >= (chip_holder + c)) & (
                            (ld + c) <= jnp.maximum(lh, lh_mate)
                        )
                    else:
                        # Paper Section 4.5 inequality: L_home - c >= L_holder + c.
                        accept = (lh - c) >= (ld + c)

                    @pl.when(accept)
                    def _apply_move():
                        owner_load_smem[home] = lh - c
                        owner_load_smem[holder] = ld + c
                        if moved_is_bitset:
                            moved_flag_smem[word_idx] = (
                                moved_flag_smem[word_idx] | bit_mask
                            )
                        elif moved_flag_smem is not None:
                            moved_flag_smem[eid] = k + jnp.int32(1)

                        @pl.when(home == my_owner)
                        def _mark_home():
                            local_suppress_smem[eid - home * jnp.int32(e_owner)] = (
                                jnp.int32(1)
                            )

                        @pl.when(holder == my_owner)
                        def _mark_holder():
                            holder_slot_expert_smem[slot] = eid

    else:
        # Heaviest-home-first greedy rounds. Supports both int32[256] and
        # compact int32[8] bitset moved_flag_smem plus chip_aware pooling.
        if moved_flag_smem is None:
            raise ValueError(
                "greedy_peak_first=True requires moved_flag_smem (int32[256] or int32[8] SMEM)"
            )
        from jax import lax

        def _greedy_round(_, carry):
            done = carry

            def _scan_body(k, best):
                best_k, best_lh, best_margin = best
                eid = replica_meta_smem[0, k]
                safe_eid = jnp.where(eid >= 0, eid, jnp.int32(0))
                cnt = jnp.where(eid >= 0, _read_count_smem(counts_smem, safe_eid), jnp.int32(0))
                if moved_is_bitset:
                    w_i = lax.shift_right_logical(safe_eid, jnp.int32(5))
                    b_m = lax.shift_left(jnp.int32(1), safe_eid & jnp.int32(31))
                    unmoved = (moved_flag_smem[w_i] & b_m) == 0
                else:
                    unmoved = moved_flag_smem[safe_eid] == 0
                home = jnp.where(eid >= 0, replica_meta_smem[1, k], jnp.int32(0))
                holder = jnp.where(eid >= 0, replica_meta_smem[2, k], jnp.int32(0))
                c = _expert_cost_smem(
                    cnt,
                    cost_mode=mode,
                    tile_m=tile_m,
                    base_cost=base_cost,
                    row_weight=row_weight,
                )
                lh = owner_load_smem[home]
                ld = owner_load_smem[holder]
                if chip_aware:
                    lh_mate = owner_load_smem[
                        lax.bitwise_xor(home, jnp.int32(chipmate_xor))
                    ]
                    ld_mate = owner_load_smem[
                        lax.bitwise_xor(holder, jnp.int32(chipmate_xor))
                    ]
                    peak_h = jnp.maximum(lh, lh_mate)
                    margin = ((lh + lh_mate) - c) - ((ld + ld_mate) + c)
                    accept = (margin >= 0) & ((ld + c) <= peak_h)
                else:
                    peak_h = lh
                    margin = (lh - c) - (ld + c)
                    accept = margin >= 0
                elig = (eid >= 0) & (cnt > 0) & unmoved & accept
                better = elig & (
                    (peak_h > best_lh) | ((peak_h == best_lh) & (margin > best_margin))
                )
                return (
                    jnp.where(better, k, best_k),
                    jnp.where(better, peak_h, best_lh),
                    jnp.where(better, margin, best_margin),
                )

            def _do_round():
                bk, _, _ = lax.fori_loop(
                    0,
                    k_rep,
                    _scan_body,
                    (jnp.int32(-1), jnp.int32(-1), jnp.int32(-1)),
                )

                @pl.when(bk >= 0)
                def _commit():
                    eid = replica_meta_smem[0, bk]
                    home = replica_meta_smem[1, bk]
                    holder = replica_meta_smem[2, bk]
                    slot = replica_meta_smem[3, bk]
                    cnt = _read_count_smem(counts_smem, eid)
                    c = _expert_cost_smem(
                        cnt,
                        cost_mode=mode,
                        tile_m=tile_m,
                        base_cost=base_cost,
                        row_weight=row_weight,
                    )
                    owner_load_smem[home] = owner_load_smem[home] - c
                    owner_load_smem[holder] = owner_load_smem[holder] + c
                    if moved_is_bitset:
                        w_i = lax.shift_right_logical(eid, jnp.int32(5))
                        b_m = lax.shift_left(jnp.int32(1), eid & jnp.int32(31))
                        moved_flag_smem[w_i] = moved_flag_smem[w_i] | b_m
                    else:
                        moved_flag_smem[eid] = bk + jnp.int32(1)

                    @pl.when(home == my_owner)
                    def _mark_home():
                        local_suppress_smem[eid - home * jnp.int32(e_owner)] = (
                            jnp.int32(1)
                        )

                    @pl.when(holder == my_owner)
                    def _mark_holder():
                        holder_slot_expert_smem[slot] = eid

                return bk < 0

            return lax.cond(done, lambda: jnp.bool_(True), _do_round)

        lax.fori_loop(0, k_rep, _greedy_round, jnp.bool_(False))


def plan_chipmate_move_smem(
    my_rank,
    ep: int,
    counts_smem,
    rank_active_smem,
    rank_rows_smem,
    cm_decision_smem,
    *,
    chipmate_xor: int = 1,
    chipmate_map_smem=None,
    replica_meta_smem=None,
    moved_flag_smem=None,
    min_active_gap: int = 2,
    require_global_max: bool = False,
    global_max_slack: int = 0,
    max_move_rows: int = 1 << 30,
    vpu_precomputed_loads: bool = False,
    prefer_max_rows: bool = False,
) -> None:
    """Deterministic in-kernel Section 4.3 intra-chip chipmate planner in SMEM.

    Computes post-Section-4.5b per-rank active expert counts and row totals,
    then picks at most 1 non-suppressed few-row active expert to migrate across
    `(my_rank, partner)` when `active_gap >= min_active_gap` without worsening
    the pair's maximum row count.

    Writes `cm_decision_smem` (`int32[3]`):
      - `[0]`: `cm_send_slot` (`0..e_local-1` if `my_rank` sends, else `-1`)
      - `[1]`: `cm_recv_eid` (`0..255` if `my_rank` receives, else `-1`)
      - `[2]`: `partner_rank` (`0..ep-1`)
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    num_experts = NUM_EXPERTS_DEFAULT
    e_local = num_experts // ep

    partner = (
        chipmate_map_smem[my_rank]
        if chipmate_map_smem is not None
        else lax.bitwise_xor(my_rank, jnp.int32(chipmate_xor))
    )
    cm_decision_smem[0] = jnp.int32(-1)
    cm_decision_smem[1] = jnp.int32(-1)
    cm_decision_smem[2] = partner

    # 1. Compute base per-rank active expert counts & total rows.
    if vpu_precomputed_loads and len(counts_smem.shape) == 2:

        @pl.loop(0, ep)
        def _copy_rank_vpu(r):
            rank_active_smem[r] = counts_smem[2, r]
            rank_rows_smem[r] = counts_smem[3, r]

    else:

        @pl.loop(0, ep)
        def _accum_rank(r):
            base = r * jnp.int32(e_local)

            def _sum_e(le, carry):
                act_acc, row_acc = carry
                cnt = _read_count_smem(counts_smem, base + le)
                is_act = (cnt > 0).astype(jnp.int32)
                return act_acc + is_act, row_acc + cnt

            a_r, rw_r = lax.fori_loop(
                0, e_local, _sum_e, (jnp.int32(0), jnp.int32(0))
            )
            rank_active_smem[r] = a_r
            rank_rows_smem[r] = rw_r

    # 2. If composed after Section 4.5b, adjust per-rank active & row totals
    #    for experts already moved by preloaded replicas.
    if replica_meta_smem is not None and moved_flag_smem is not None:
        k_rep = int(replica_meta_smem.shape[1])

        @pl.loop(0, k_rep)
        def _adj_rep(k):
            eid = replica_meta_smem[0, k]

            @pl.when(eid >= 0)
            def _chk():
                @pl.when(moved_flag_smem[eid] == k + jnp.int32(1))
                def _apply_adj():
                    cnt = _read_count_smem(counts_smem, eid)
                    home = replica_meta_smem[1, k]
                    holder = replica_meta_smem[2, k]
                    rank_active_smem[home] = rank_active_smem[home] - jnp.int32(1)
                    rank_active_smem[holder] = rank_active_smem[holder] + jnp.int32(1)
                    rank_rows_smem[home] = rank_rows_smem[home] - cnt
                    rank_rows_smem[holder] = rank_rows_smem[holder] + cnt

    # 3. Evaluate our own chipmate pair deterministically on both chipmates.
    lo_rank = jnp.minimum(my_rank, partner)
    hi_rank = jnp.maximum(my_rank, partner)
    a_lo = rank_active_smem[lo_rank]
    a_hi = rank_active_smem[hi_rank]
    lo_is_sender = a_lo >= a_hi
    sender = jnp.where(lo_is_sender, lo_rank, hi_rank)
    receiver = jnp.where(lo_is_sender, hi_rank, lo_rank)
    a_sender = rank_active_smem[sender]
    a_receiver = rank_active_smem[receiver]
    gap = a_sender - a_receiver

    if require_global_max:

        def _max_body(r, cur_max):
            return jnp.maximum(cur_max, rank_active_smem[r])

        global_max = lax.fori_loop(0, ep, _max_body, jnp.int32(0))
        pair_ok = (gap >= jnp.int32(min_active_gap)) & (
            a_sender >= global_max - jnp.int32(global_max_slack)
        )
    else:
        pair_ok = gap >= jnp.int32(min_active_gap)

    @pl.when(pair_ok)
    def _select_expert():
        r_sender = rank_rows_smem[sender]
        r_receiver = rank_rows_smem[receiver]
        pair_max_rows = jnp.maximum(r_sender, r_receiver)
        base = sender * jnp.int32(e_local)

        def _pick_body(le, best):
            best_le, best_cnt = best
            eid = base + le
            cnt = _read_count_smem(counts_smem, eid)
            not_moved = (
                (moved_flag_smem[eid] == 0)
                if moved_flag_smem is not None
                else jnp.bool_(True)
            )
            if prefer_max_rows:
                row_ok = (r_receiver + cnt) <= (r_sender - cnt)
                better = cnt > best_cnt
            else:
                row_ok = (r_receiver + cnt) <= pair_max_rows
                better = cnt < best_cnt
            ok = (
                (cnt > 0)
                & not_moved
                & (cnt <= jnp.int32(max_move_rows))
                & row_ok
                & better
            )
            return (
                jnp.where(ok, le, best_le),
                jnp.where(ok, cnt, best_cnt),
            )

        init_cnt = jnp.int32(0) if prefer_max_rows else jnp.int32(1 << 30)
        best_le, _ = lax.fori_loop(
            0,
            e_local,
            _pick_body,
            (jnp.int32(-1), init_cnt),
        )

        @pl.when(best_le >= 0)
        def _record():
            @pl.when(my_rank == sender)
            def _as_sender():
                cm_decision_smem[0] = best_le

            @pl.when(my_rank == receiver)
            def _as_receiver():
                cm_decision_smem[1] = base + best_le


def build_unified_worklist_smem(
    my_owner,
    e_owner: int,
    num_replicas: int,
    counts_smem,
    local_suppress_smem,
    holder_slot_expert_smem,
    task_eid_smem,
    task_src_smem,
    task_slot_smem,
    num_tasks_smem,
    *,
    cm_send_slot=None,
    cm_recv_eid=None,
) -> None:
    """Build one compact execution worklist over home, replica, and chipmate work.

    Outputs:
      - `task_eid_smem[t]`: global expert ID `0..255`
      - `task_src_smem[t]`: `0`=home (`w1`), `1`=preloaded replica (`rep_w1`),
        `2`=chipmate landing pad (`cm_w1_pad`), matching Track 1's `exec_weight_src`.
      - `task_slot_smem[t]`: slot index inside that source (`0..e_owner-1` for home,
        `0..R-1` for replica, `0` for chipmate).
      - `num_tasks_smem[0]`: total active tasks to execute locally.

    Note: Home and preloaded-replica tasks come BEFORE any borrowed chipmate task
    so the receiver maximizes overlap of incoming die-to-die weight DMA behind
    all local + replica compute!
    """
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    num_tasks_smem[0] = jnp.int32(0)
    base = my_owner * jnp.int32(e_owner)

    # 1. Active non-suppressed local home experts.
    @pl.loop(0, e_owner)
    def _add_home(le):
        eid = base + le
        cnt = _read_count_smem(counts_smem, eid)
        sup_rep = (
            (local_suppress_smem[le] != 0)
            if local_suppress_smem is not None
            else jnp.bool_(False)
        )
        sup_cm = (
            (cm_send_slot == le)
            if cm_send_slot is not None
            else jnp.bool_(False)
        )

        @pl.when((cnt > 0) & (~sup_rep) & (~sup_cm))
        def _push_home():
            t = num_tasks_smem[0]
            task_eid_smem[t] = eid
            task_src_smem[t] = jnp.int32(0)
            task_slot_smem[t] = le
            num_tasks_smem[0] = t + jnp.int32(1)

    # 2. Active preloaded replica slots owned by this device.
    if holder_slot_expert_smem is not None and num_replicas > 0:

        @pl.loop(0, num_replicas)
        def _add_rep(s):
            eid = holder_slot_expert_smem[s]

            @pl.when(eid >= 0)
            def _push_rep():
                t = num_tasks_smem[0]
                task_eid_smem[t] = eid
                task_src_smem[t] = jnp.int32(1)
                task_slot_smem[t] = s
                num_tasks_smem[0] = t + jnp.int32(1)

    # 3. Borrowed chipmate expert appended LAST to maximize DMA overlap window.
    if cm_recv_eid is not None:

        @pl.when(cm_recv_eid >= 0)
        def _push_cm():
            t = num_tasks_smem[0]
            task_eid_smem[t] = cm_recv_eid
            task_src_smem[t] = jnp.int32(2)
            task_slot_smem[t] = jnp.int32(0)
            num_tasks_smem[0] = t + jnp.int32(1)


# =============================================================================
# Prefill (T = 4096..8192) Imbalance Planners: §4.5b, §4.3, and §4.2b
# =============================================================================


def _jax_bincount_2d_topk(ids_2d, num_bins: int):
    """Exact O(T * num_bins) VPU histogram without contended scatter_add."""
    import jax.numpy as jnp

    topk = ids_2d.shape[1]
    iota = jnp.arange(num_bins, dtype=jnp.int32)[None, :]
    hits = (ids_2d[:, 0:1] == iota).astype(jnp.int32)
    for k in range(1, topk):
        hits = hits + (ids_2d[:, k : k + 1] == iota).astype(jnp.int32)
    return jnp.sum(hits, axis=0)


def _base_virtual_ids_arith(ids_i32, e_in: int, e_out: int, ep_size: int):
    """Gather-free O(1) VPU mapping from `[0, ep_size*e_in]` to `[0, ep_size*e_out]`."""
    import jax
    import jax.numpy as jnp

    if (e_in & (e_in - 1)) == 0:
        shift = e_in.bit_length() - 1
        owner = jax.lax.shift_right_logical(ids_i32, shift)
        slot = ids_i32 & jnp.int32(e_in - 1)
    else:
        owner = ids_i32 // e_in
        slot = ids_i32 % e_in
    mapped = owner * e_out + slot
    return jnp.where(
        ids_i32 >= ep_size * e_in, jnp.int32(ep_size * e_out), mapped
    )


def remap_prefill_replicas_jax(
    topk_indices,
    replica_meta,
    *,
    ep_size: int,
    num_base_local: int = 32,
    num_replicas: int,
    tile_m: int = 128,
    mode: str = "whole",
    weight_load_tiles: int = 1,
):
    """Dynamic §4.5b prefill home-vs-holder remapper under row/tile objective.

    Gather-free VPU implementation (`~30 us` at T=8192, zero collectives).
    Supports both single-holder (`max_holders_per_expert=1`) and D'Hondt
    multi-holder (`max_holders_per_expert >= 2`) replica plans:
      - `"whole"` (Paper §4.5b Fig 14 Left): moves all rows of expert `e` from
        `home` to `holder` iff `L_home - cost(e) >= L_holder + cost(e)`.
      - `"split"` (Paper §4.5b Fig 14 Right + multi-holder water-filling):
        moves full `tile_m`-row blocks from `home` to each assigned `holder`
        whenever `L_home > L_holder + weight_load_tiles`, splitting mega-hot
        outlier experts evenly across `home` and all assigned holders.
    """
    import jax
    import jax.numpy as jnp

    if num_replicas <= 0:
        return topk_indices
    mode_norm = (mode or "whole").strip().lower()
    if mode_norm not in ("whole", "split"):
        raise ValueError(
            f"Unsupported prefill replica mode={mode!r}; expected 'whole' or 'split'"
        )
    num_experts = ep_size * num_base_local
    e_loc = num_base_local + num_replicas
    ids_i32 = topk_indices.astype(jnp.int32)
    counts_e = _jax_bincount_2d_topk(ids_i32, num_experts)
    tiles_raw = jnp.where(counts_e > 0, (counts_e + tile_m - 1) // tile_m, 0)
    cost_e = jnp.where(counts_e > 0, tiles_raw + weight_load_tiles, 0)
    owner_loads = jnp.sum(cost_e.reshape(ep_size, num_base_local), axis=1)

    base_v = _base_virtual_ids_arith(ids_i32, num_base_local, e_loc, ep_size)
    k_rep = int(replica_meta.shape[1])

    if mode_norm == "whole":
        v_out = base_v
        loads = owner_loads
        moved = jnp.zeros((num_experts,), dtype=jnp.int32)
        for k in range(k_rep):
            eid = replica_meta[0, k]
            home = replica_meta[1, k]
            holder = replica_meta[2, k]
            slot = replica_meta[3, k]
            valid = (eid >= 0) & (eid < num_experts)
            eid_c = jnp.clip(eid, 0, num_experts - 1)
            home_c = jnp.clip(home, 0, ep_size - 1)
            hold_c = jnp.clip(holder, 0, ep_size - 1)
            already = moved[eid_c] != 0
            ce = cost_e[eid_c]
            accept = (
                valid
                & (~already)
                & (ce > 0)
                & (loads[home_c] - ce >= loads[hold_c] + ce)
            )
            new_vid = hold_c * e_loc + num_base_local + slot
            v_out = jnp.where(accept & (ids_i32 == eid_c), new_vid, v_out)
            delta = jnp.where(accept, ce, jnp.int32(0))
            loads = loads.at[home_c].add(-delta).at[hold_c].add(delta)
            moved = moved.at[eid_c].set(jnp.where(accept, jnp.int32(1), moved[eid_c]))
        return v_out

    # Gather-free split mode with automatic multi-holder water-filling:
    # `rem_tiles[eid]` tracks tiles still sitting on `home` (`[0, rem_tiles*tile_m)`),
    # peeling suffix slabs `[new_rem*tile_m, old_rem*tile_m)` to each holder!
    total_elems = ids_i32.shape[0] * ids_i32.shape[1]
    n_blocks = 64 if (total_elems >= 64 and total_elems % 64 == 0) else ep_size
    blk_elems = total_elems // n_blocks
    ids_blk = ids_i32.reshape(n_blocks, blk_elems)
    v_blk = base_v.reshape(n_blocks, blk_elems)

    # Pre-count how many remaining holder entries each expert has in `replica_meta`.
    all_eids = jnp.clip(replica_meta[0], 0, num_experts - 1)
    all_valid = (replica_meta[0] >= 0) & (replica_meta[0] < num_experts)
    rem_holders = jnp.sum(
        (
            (all_eids[:, None] == jnp.arange(num_experts, dtype=jnp.int32)[None, :])
            & all_valid[:, None]
        ).astype(jnp.int32),
        axis=0,
    )
    rem_tiles = tiles_raw
    loads = owner_loads

    for k in range(k_rep):
        eid = replica_meta[0, k]
        home = replica_meta[1, k]
        holder = replica_meta[2, k]
        slot = replica_meta[3, k]
        valid = (eid >= 0) & (eid < num_experts)
        eid_c = jnp.clip(eid, 0, num_experts - 1)
        home_c = jnp.clip(home, 0, ep_size - 1)
        hold_c = jnp.clip(holder, 0, ep_size - 1)

        rt = rem_tiles[eid_c]
        rh = jnp.maximum(jnp.int32(1), rem_holders[eid_c])
        ce_rem = jnp.where(rt > 0, rt + weight_load_tiles, 0)

        # Whole move only when this is the sole/last holder and eliminates home copy.
        whole_ok = (
            valid
            & (rh == 1)
            & (rt == tiles_raw[eid_c])
            & (ce_rem > 0)
            & (loads[home_c] - ce_rem >= loads[hold_c] + ce_rem)
        )
        # Multi-holder aware split: cap holder's share at `rt // (rh + 1)` when `rh > 1`
        # or `(L_home - (L_holder + w)) // 2` when `rh == 1`, balancing all `1 + rh` copies!
        gap_share = (loads[home_c] - (loads[hold_c] + weight_load_tiles)) // 2
        even_share = (rt + rh // 2) // (rh + 1)
        target_t = jnp.where(rh > 1, jnp.minimum(gap_share, jnp.maximum(even_share, gap_share // rh)), gap_share)
        split_t = jnp.clip(target_t, 0, jnp.maximum(0, rt - 1))
        do_split = valid & (~whole_ok) & (split_t >= 2)
        any_act = whole_ok | do_split

        move_t = jnp.where(whole_ok, rt, jnp.where(do_split, split_t, 0))
        new_rt = rt - move_t
        lo_row = new_rt * tile_m
        hi_row = jnp.where(rt == tiles_raw[eid_c], jnp.int32(1 << 30), rt * tile_m)
        holder_vid = hold_c * e_loc + num_base_local + slot

        hit_blk = ids_blk == eid_c
        blk_cnt = jnp.sum(hit_blk.astype(jnp.int32), axis=1)
        cum_blk = jnp.cumsum(blk_cnt) - blk_cnt
        mid_row = cum_blk + jax.lax.shift_right_logical(blk_cnt, 1)
        in_slab = any_act & (mid_row >= lo_row) & (mid_row < hi_row)
        v_blk = jnp.where(hit_blk & in_slab[:, None], holder_vid, v_blk)

        d_home = jnp.where(whole_ok, ce_rem, jnp.where(do_split, move_t, 0))
        d_hold = jnp.where(
            whole_ok, ce_rem, jnp.where(do_split, move_t + weight_load_tiles, 0)
        )
        loads = loads.at[home_c].add(-d_home).at[hold_c].add(d_hold)
        rem_tiles = rem_tiles.at[eid_c].set(jnp.where(valid, new_rt, rt))
        rem_holders = rem_holders.at[eid_c].add(jnp.where(valid, jnp.int32(-1), 0))

    return v_blk.reshape(topk_indices.shape)


def plan_prefill_chipmate_jax(
    topk_indices,
    *,
    ep_size: int,
    num_local_experts: int,
    tile_m: int = 128,
    chipmate_xor: int = 1,
    min_tile_gap: int = 3,
    mode: str = "whole",
    weight_load_tiles: int = 1,
    use_pallas: bool | None = None,
):
    """Dynamic §4.3 prefill intra-chip redistribution planner (`0` collectives, gather-free).

    Uses the 1-op Pallas VMEM+SMEM router on TPU (`~30 us` total glue at T=8192)
    and falls back to pure-JAX arithmetic on CPU / small non-aligned shapes.
    """
    import jax
    import jax.numpy as jnp

    mode_norm = (mode or "whole").strip().lower()
    if mode_norm not in ("whole", "split"):
        raise ValueError(
            f"Unsupported prefill chipmate mode={mode!r}; expected 'whole' or 'split'"
        )
    total_elems_in = int(topk_indices.shape[0] * topk_indices.shape[1])
    if _can_use_tpu_pallas_prefill_router(total_elems_in, use_pallas):
        return pallas_prefill_imbalance_route(
            topk_indices,
            ep_size=ep_size,
            num_base_local=num_local_experts,
            do_chipmate=True,
            chipmate_xor=chipmate_xor,
            chipmate_mode=mode_norm,
            min_tile_gap=min_tile_gap,
            tile_m=tile_m,
            weight_load_tiles=weight_load_tiles,
        )
    e_loc = num_local_experts
    e_virt = e_loc + 1
    num_in_experts = ep_size * e_loc

    ids_i32 = topk_indices.astype(jnp.int32)
    counts_flat = _jax_bincount_2d_topk(ids_i32, num_in_experts)
    counts_rs = counts_flat.reshape(ep_size, e_loc)
    tiles_rs = jnp.where(counts_rs > 0, (counts_rs + tile_m - 1) // tile_m, 0)
    cost_rs = jnp.where(counts_rs > 0, tiles_rs + weight_load_tiles, 0)
    loads = jnp.sum(cost_rs, axis=1)  # int32[ep_size]

    base_v = _base_virtual_ids_arith(ids_i32, e_loc, e_virt, ep_size)

    pairs = [
        (r, r ^ chipmate_xor)
        for r in range(ep_size)
        if 0 <= (r ^ chipmate_xor) < ep_size and r < (r ^ chipmate_xor)
    ]

    send_slots = jnp.full((ep_size,), -1, dtype=jnp.int32)
    recv_flags = jnp.zeros((ep_size,), dtype=jnp.int32)
    recv_src_slots = jnp.zeros((ep_size,), dtype=jnp.int32)

    if mode_norm == "whole":
        v_out = base_v
        for r0, r1 in pairs:
            l0, l1 = loads[r0], loads[r1]
            r0_heavier = l0 >= l1
            hi = jnp.where(r0_heavier, jnp.int32(r0), jnp.int32(r1))
            lo = jnp.where(r0_heavier, jnp.int32(r1), jnp.int32(r0))
            l_hi = jnp.maximum(l0, l1)
            l_lo = jnp.minimum(l0, l1)

            c_hi = cost_rs[hi]
            new_peaks = jnp.where(
                c_hi > 0, jnp.maximum(l_hi - c_hi, l_lo + c_hi), l_hi
            )
            best_s = jnp.argmin(new_peaks).astype(jnp.int32)
            gain = l_hi - new_peaks[best_s]
            do_move = gain >= min_tile_gap

            old_eid = hi * e_loc + best_s
            borrowed_vid = lo * e_virt + e_loc
            v_out = jnp.where(do_move & (ids_i32 == old_eid), borrowed_vid, v_out)
            send_slots = send_slots.at[hi].set(
                jnp.where(do_move, best_s, send_slots[hi])
            )
            recv_flags = recv_flags.at[lo].set(
                jnp.where(do_move, jnp.int32(1), recv_flags[lo])
            )
            recv_src_slots = recv_src_slots.at[lo].set(
                jnp.where(do_move, best_s, recv_src_slots[lo])
            )

        cm_plan = jnp.stack([send_slots, recv_flags, recv_src_slots], axis=1)
        return v_out, cm_plan

    # Gather-free split mode: picks larger gain between whole-expert move and outlier split.
    total_elems = ids_i32.shape[0] * ids_i32.shape[1]
    n_blocks = 64 if (total_elems >= 64 and total_elems % 64 == 0) else ep_size
    blk_elems = total_elems // n_blocks
    ids_blk = ids_i32.reshape(n_blocks, blk_elems)
    v_blk = base_v.reshape(n_blocks, blk_elems)

    for r0, r1 in pairs:
        l0, l1 = loads[r0], loads[r1]
        r0_heavier = l0 >= l1
        hi = jnp.where(r0_heavier, jnp.int32(r0), jnp.int32(r1))
        lo = jnp.where(r0_heavier, jnp.int32(r1), jnp.int32(r0))
        l_hi = jnp.maximum(l0, l1)
        l_lo = jnp.minimum(l0, l1)

        c_hi = cost_rs[hi]
        t_hi = tiles_rs[hi]
        new_peaks = jnp.where(c_hi > 0, jnp.maximum(l_hi - c_hi, l_lo + c_hi), l_hi)
        best_s = jnp.argmin(new_peaks).astype(jnp.int32)
        whole_gain = l_hi - new_peaks[best_s]

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
        chosen_eid = hi * e_loc + chosen_s
        borrowed_vid = lo * e_virt + e_loc
        keep_rows = jnp.where(use_whole, 0, (t_max - split_t) * tile_m)

        hit_blk = ids_blk == chosen_eid
        blk_cnt = jnp.sum(hit_blk.astype(jnp.int32), axis=1)
        cum_blk = jnp.cumsum(blk_cnt) - blk_cnt
        mid_row = cum_blk + jax.lax.shift_right_logical(blk_cnt, 1)
        to_lo_blk = do_any & (mid_row >= keep_rows)
        v_blk = jnp.where(hit_blk & to_lo_blk[:, None], borrowed_vid, v_blk)

        send_slots = send_slots.at[hi].set(
            jnp.where(do_any, chosen_s, send_slots[hi])
        )
        recv_flags = recv_flags.at[lo].set(
            jnp.where(do_any, jnp.int32(1), recv_flags[lo])
        )
        recv_src_slots = recv_src_slots.at[lo].set(
            jnp.where(do_any, chosen_s, recv_src_slots[lo])
        )

    cm_plan = jnp.stack([send_slots, recv_flags, recv_src_slots], axis=1)
    return v_blk.reshape(topk_indices.shape), cm_plan


def _can_use_tpu_pallas_prefill_router(total_elems: int, use_pallas: bool | None) -> bool:
    """Return True when the 1-op Pallas VMEM+SMEM prefill router is eligible."""
    if use_pallas is False:
        return False
    if total_elems < 64 * 128 or total_elems % (64 * 128) != 0 or total_elems > 262_144:
        return False
    if use_pallas is True:
        return True
    try:
        import jax
        from sglang_tpu.envs import envs as _tpu_envs

        if jax.default_backend() == "tpu":
            return True
        if _tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE.get():
            from sglang_tpu.pallas_export import get_tpu_info_or_none

            return get_tpu_info_or_none() is not None
        return False
    except Exception:
        return False


def pallas_prefill_imbalance_route(
    topk_indices,
    *,
    ep_size: int,
    num_base_local: int = 32,
    replica_meta=None,
    num_replicas: int = 0,
    replica_mode: str = "whole",
    do_chipmate: bool = False,
    chipmate_xor: int = 1,
    chipmate_mode: str = "whole",
    min_tile_gap: int = 3,
    tile_m: int = 128,
    weight_load_tiles: int = 1,
    interpret: bool = False,
):
    """Single-op Pallas VMEM+SMEM prefill imbalance planner & remapper (§4.5b + §4.3).

    Replaces ~680 scalar-dynamic-slice XLA ops and 16–20 serial full-tensor VPU
    passes with ONE `grid=(1,)` Mosaic TPU kernel:
      1. Runs the deterministic §4.5b (`whole`/`split`/multi-holder) and §4.3
         (`whole`/`split`) planners entirely on the Scalar Core in SMEM (`<1 us`),
         recording only active moves (`0` moves on balanced, `1..6` on skewed).
      2. Maps base virtual IDs once in VMEM (`(8, 128)` register-tiled, `0` spills)
         and applies only active moves in VMEM using exact bit-split MXU prefix sum
         for split moves (`~30-45 us` total vs `+963 us` pure-XLA glue).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    has_rep = replica_meta is not None and num_replicas > 0
    rep_mode_norm = (replica_mode or "whole").strip().lower()
    cm_mode_norm = (chipmate_mode or "whole").strip().lower()
    if has_rep and rep_mode_norm not in ("whole", "split"):
        raise ValueError(f"Unsupported prefill replica mode={replica_mode!r}")
    if do_chipmate and cm_mode_norm not in ("whole", "split"):
        raise ValueError(f"Unsupported prefill chipmate mode={chipmate_mode!r}")

    ids_i32 = topk_indices.astype(jnp.int32)
    total_elems = int(ids_i32.shape[0] * ids_i32.shape[1])
    n_blocks = 64
    if total_elems % (n_blocks * 128) != 0:
        raise ValueError(
            f"pallas_prefill_imbalance_route requires total_elems % 8192 == 0, got {total_elems}"
        )
    blk_elems = total_elems // n_blocks
    ids_blk = ids_i32.reshape(n_blocks, blk_elems)

    e_in = num_base_local
    e_after_rep = e_in + (num_replicas if has_rep else 0)
    e_final = e_after_rep + (1 if do_chipmate else 0)
    num_in_experts = ep_size * e_in
    num_after_rep = ep_size * e_after_rep
    k_rep = int(replica_meta.shape[1]) if has_rep else 0
    max_moves = max(1, k_rep + (ep_size // 2 if do_chipmate else 0))

    counts_e = _jax_bincount_2d_topk(ids_i32, num_in_experts)
    pairs = [
        (r, r ^ chipmate_xor)
        for r in range(ep_size)
        if 0 <= (r ^ chipmate_xor) < ep_size and r < (r ^ chipmate_xor)
    ]
    e_in_pow2 = (e_in & (e_in - 1)) == 0
    e_in_shift = e_in.bit_length() - 1
    e_in_mask = e_in - 1

    def _kernel(*refs):
        it = iter(refs)
        counts_ref = next(it)
        rep_meta_ref = next(it) if has_rep else None
        ids_ref = next(it)
        v_out_ref = next(it)
        cm_out_ref = next(it) if do_chipmate else None

        loads_s = next(it)  # SMEM (ep_size,)
        tiles_s = next(it)  # SMEM (num_after_rep,)
        tiles_raw_s = next(it)  # SMEM (num_in_experts,)
        rem_h_s = next(it) if has_rep else None  # SMEM (num_in_experts,)
        moved_s = next(it) if has_rep else None  # SMEM (num_in_experts,)
        n_moves_s = next(it)  # SMEM (1,)
        m_src_s = next(it)  # SMEM (max_moves,)
        m_dst_s = next(it)  # SMEM (max_moves,)
        m_lo_s = next(it)  # SMEM (max_moves,)
        m_hi_s = next(it)  # SMEM (max_moves,)
        blk_cnt_v = next(it)  # VMEM (64, 128) int32

        # 1. Initialize SMEM loads and tile tables on Scalar Core.
        @pl.loop(0, ep_size)
        def _zero_loads(r):
            loads_s[r] = jnp.int32(0)

        @pl.loop(0, num_after_rep)
        def _zero_tiles(i):
            tiles_s[i] = jnp.int32(0)

        @pl.loop(0, num_in_experts)
        def _init_in(e):
            c = counts_ref[e]
            t = jnp.where(c > 0, (c + jnp.int32(tile_m - 1)) // jnp.int32(tile_m), jnp.int32(0))
            cost = jnp.where(c > 0, t + jnp.int32(weight_load_tiles), jnp.int32(0))
            own = e // jnp.int32(e_in)
            sl = e % jnp.int32(e_in)
            tiles_raw_s[e] = t
            tiles_s[own * jnp.int32(e_after_rep) + sl] = t
            loads_s[own] = loads_s[own] + cost
            if has_rep:
                rem_h_s[e] = jnp.int32(0)
                moved_s[e] = jnp.int32(0)

        n_moves_s[0] = jnp.int32(0)

        # 2. Section 4.5b deterministic planner in SMEM.
        if has_rep:
            @pl.loop(0, k_rep)
            def _cnt_holders(k):
                eid = rep_meta_ref[0, k]
                valid = (eid >= 0) & (eid < jnp.int32(num_in_experts))

                @pl.when(valid)
                def _inc_h():
                    rem_h_s[eid] = rem_h_s[eid] + jnp.int32(1)

            @pl.loop(0, k_rep)
            def _plan_rep(k):
                eid = rep_meta_ref[0, k]
                home = rep_meta_ref[1, k]
                holder = rep_meta_ref[2, k]
                slot = rep_meta_ref[3, k]
                valid = (
                    (eid >= 0)
                    & (eid < jnp.int32(num_in_experts))
                    & (home >= 0)
                    & (home < jnp.int32(ep_size))
                    & (holder >= 0)
                    & (holder < jnp.int32(ep_size))
                )

                @pl.when(valid)
                def _eval_rep():
                    home_sl = eid % jnp.int32(e_in)
                    home_idx = home * jnp.int32(e_after_rep) + home_sl
                    hold_idx = holder * jnp.int32(e_after_rep) + jnp.int32(e_in) + slot
                    src_vid = home * jnp.int32(e_final) + home_sl
                    dst_vid = holder * jnp.int32(e_final) + jnp.int32(e_in) + slot
                    rt = tiles_s[home_idx]
                    rh = jnp.maximum(jnp.int32(1), rem_h_s[eid])
                    ce_rem = jnp.where(
                        rt > 0, rt + jnp.int32(weight_load_tiles), jnp.int32(0)
                    )
                    l_home = loads_s[home]
                    l_hold = loads_s[holder]

                    if rep_mode_norm == "whole":
                        accept = (
                            (moved_s[eid] == 0)
                            & (ce_rem > 0)
                            & (l_home - ce_rem >= l_hold + ce_rem)
                        )

                        @pl.when(accept)
                        def _do_whole():
                            mi = n_moves_s[0]
                            m_src_s[mi] = src_vid
                            m_dst_s[mi] = dst_vid
                            m_lo_s[mi] = jnp.int32(0)
                            m_hi_s[mi] = jnp.int32(1 << 30)
                            n_moves_s[0] = mi + jnp.int32(1)
                            loads_s[home] = l_home - ce_rem
                            loads_s[holder] = l_hold + ce_rem
                            tiles_s[home_idx] = jnp.int32(0)
                            tiles_s[hold_idx] = tiles_s[hold_idx] + rt
                            moved_s[eid] = jnp.int32(1)
                    else:
                        t_orig = tiles_raw_s[eid]
                        whole_ok = (
                            (rh == 1)
                            & (rt == t_orig)
                            & (ce_rem > 0)
                            & (l_home - ce_rem >= l_hold + ce_rem)
                        )
                        gap_share = (l_home - (l_hold + jnp.int32(weight_load_tiles))) // jnp.int32(2)
                        even_share = (rt + rh // jnp.int32(2)) // (rh + jnp.int32(1))
                        target_t = jnp.where(
                            rh > 1,
                            jnp.minimum(
                                gap_share,
                                jnp.maximum(even_share, gap_share // rh),
                            ),
                            gap_share,
                        )
                        max_split = jnp.maximum(jnp.int32(0), rt - jnp.int32(1))
                        split_t = jnp.minimum(jnp.maximum(jnp.int32(0), target_t), max_split)
                        do_split = (~whole_ok) & (split_t >= jnp.int32(2))
                        any_act = whole_ok | do_split

                        @pl.when(any_act)
                        def _do_split_or_whole():
                            move_t = jnp.where(whole_ok, rt, split_t)
                            new_rt = rt - move_t
                            lo_row = new_rt * jnp.int32(tile_m)
                            hi_row = jnp.where(
                                rt == t_orig,
                                jnp.int32(1 << 30),
                                rt * jnp.int32(tile_m),
                            )
                            mi = n_moves_s[0]
                            m_src_s[mi] = src_vid
                            m_dst_s[mi] = dst_vid
                            m_lo_s[mi] = jnp.where(whole_ok, jnp.int32(0), lo_row)
                            m_hi_s[mi] = jnp.where(whole_ok, jnp.int32(1 << 30), hi_row)
                            n_moves_s[0] = mi + jnp.int32(1)
                            d_home = jnp.where(whole_ok, ce_rem, move_t)
                            d_hold = jnp.where(
                                whole_ok,
                                ce_rem,
                                move_t + jnp.int32(weight_load_tiles),
                            )
                            loads_s[home] = l_home - d_home
                            loads_s[holder] = l_hold + d_hold
                            tiles_s[home_idx] = new_rt
                            tiles_s[hold_idx] = tiles_s[hold_idx] + move_t
                    rem_h_s[eid] = rem_h_s[eid] - jnp.int32(1)

        # 3. Section 4.3 deterministic intra-chip chipmate planner in SMEM.
        if do_chipmate:
            @pl.loop(0, ep_size)
            def _init_cm(r):
                cm_out_ref[r, 0] = jnp.int32(-1)
                cm_out_ref[r, 1] = jnp.int32(0)
                cm_out_ref[r, 2] = jnp.int32(0)

            for r0, r1 in pairs:
                l0 = loads_s[r0]
                l1 = loads_s[r1]
                r0_heavier = l0 >= l1
                hi = jnp.where(r0_heavier, jnp.int32(r0), jnp.int32(r1))
                lo = jnp.where(r0_heavier, jnp.int32(r1), jnp.int32(r0))
                l_hi = jnp.maximum(l0, l1)
                l_lo = jnp.minimum(l0, l1)

                # Scan `e_after_rep` local slots on `hi` in SMEM (~0.05 us).
                best_peak_s = n_moves_s  # reuse local Python-carried loop vars via fori_loop
                def _scan_slot(s, carry):
                    b_s, b_pk, m_s, m_t = carry
                    t_s = tiles_s[hi * jnp.int32(e_after_rep) + s]
                    c_s = jnp.where(t_s > 0, t_s + jnp.int32(weight_load_tiles), jnp.int32(0))
                    pk = jnp.where(
                        c_s > 0,
                        jnp.maximum(l_hi - c_s, l_lo + c_s),
                        l_hi,
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
                    e_after_rep,
                    _scan_slot,
                    (jnp.int32(0), l_hi, jnp.int32(0), jnp.int32(0)),
                )
                whole_gain = l_hi - best_pk
                if cm_mode_norm == "whole":
                    do_any = whole_gain >= jnp.int32(min_tile_gap)
                    use_whole = do_any
                    chosen_s = best_s
                    keep_rows = jnp.int32(0)
                else:
                    split_raw = (l_hi - (l_lo + jnp.int32(weight_load_tiles))) // jnp.int32(2)
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

                @pl.when(do_any)
                def _do_cm():
                    src_vid = hi * jnp.int32(e_final) + chosen_s
                    dst_vid = lo * jnp.int32(e_final) + jnp.int32(e_after_rep)
                    mi = n_moves_s[0]
                    m_src_s[mi] = src_vid
                    m_dst_s[mi] = dst_vid
                    m_lo_s[mi] = keep_rows
                    m_hi_s[mi] = jnp.int32(1 << 30)
                    n_moves_s[0] = mi + jnp.int32(1)
                    cm_out_ref[hi, 0] = chosen_s
                    cm_out_ref[lo, 1] = jnp.int32(1)
                    cm_out_ref[lo, 2] = chosen_s

        # 4. Base virtual ID mapping in VMEM (`(8, 128)` tiled, peak <= 4 VREGs).
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

        # 5. Apply ONLY active moves (`0` iterations on balanced!).
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

            @pl.when(~is_whole)
            def _apply_split():
                # Count per-block hits into `blk_cnt_v [64, 128]`.
                @pl.loop(0, n_blocks // 8)
                def _c_blk(b8):
                    r_sl = pl.ds(b8 * 8, 8)
                    acc = jnp.zeros((8, 128), dtype=jnp.int32)
                    for c0 in range(0, blk_elems, 128):
                        cur = v_out_ref[r_sl, pl.ds(c0, 128)]
                        acc = acc + (cur == src_vid).astype(jnp.int32)
                    rsum = jnp.sum(acc, axis=1, keepdims=True)
                    blk_cnt_v[r_sl, :] = jnp.broadcast_to(rsum, (8, 128))

                # Exact int32 prefix sum via bit-split `(hi << 8) + lo` on bf16xbf16->f32 MXU.
                tril_64 = (
                    lax.broadcasted_iota(jnp.int32, (n_blocks, n_blocks), 0)
                    >= lax.broadcasted_iota(jnp.int32, (n_blocks, n_blocks), 1)
                ).astype(jnp.bfloat16)
                cnt_all = blk_cnt_v[...]
                lo8 = (cnt_all & jnp.int32(255)).astype(jnp.bfloat16)
                hi8 = lax.shift_right_logical(cnt_all, jnp.int32(8)).astype(jnp.bfloat16)
                cum_lo = jnp.matmul(
                    tril_64, lo8, preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                cum_hi = jnp.matmul(
                    tril_64, hi8, preferred_element_type=jnp.float32
                ).astype(jnp.int32)
                cum_all = lax.shift_left(cum_hi, jnp.int32(8)) + cum_lo
                mid_all = (cum_all - cnt_all) + lax.shift_right_logical(cnt_all, jnp.int32(1))
                blk_cnt_v[...] = ((mid_all >= lo_row) & (mid_all < hi_row)).astype(jnp.int32)

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

    v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    s_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    call_args = [counts_e]
    num_sp = 1
    if has_rep:
        call_args.append(replica_meta.astype(jnp.int32))
        num_sp = 2
    call_args.append(ids_blk)

    out_shape = (
        (
            jax.ShapeDtypeStruct((n_blocks, blk_elems), jnp.int32),
            jax.ShapeDtypeStruct((ep_size, 3), jnp.int32),
        )
        if do_chipmate
        else jax.ShapeDtypeStruct((n_blocks, blk_elems), jnp.int32)
    )
    out_specs = (v_spec, s_spec) if do_chipmate else v_spec

    scratch_shapes = [
        pltpu.SMEM((ep_size,), jnp.int32),
        pltpu.SMEM((num_after_rep,), jnp.int32),
        pltpu.SMEM((num_in_experts,), jnp.int32),
    ]
    if has_rep:
        scratch_shapes.extend(
            [
                pltpu.SMEM((num_in_experts,), jnp.int32),
                pltpu.SMEM((num_in_experts,), jnp.int32),
            ]
        )
    scratch_shapes.extend(
        [
            pltpu.SMEM((1,), jnp.int32),
            pltpu.SMEM((max_moves,), jnp.int32),
            pltpu.SMEM((max_moves,), jnp.int32),
            pltpu.SMEM((max_moves,), jnp.int32),
            pltpu.SMEM((max_moves,), jnp.int32),
            pltpu.VMEM((n_blocks, 128), jnp.int32),
        ]
    )

    res = pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_sp,
            grid=(1,),
            in_specs=(v_spec,),
            out_specs=out_specs,
            scratch_shapes=tuple(scratch_shapes),
        ),
        interpret=interpret,
        name=(
            f"prefill_imb_route_ep{ep_size}_e{e_in}_r{num_replicas if has_rep else 0}"
            f"_{'cm' if do_chipmate else 'nocm'}"
        ),
    )(*call_args)

    if do_chipmate:
        v_blk_out, cm_plan = res
        return v_blk_out.reshape(topk_indices.shape), cm_plan
    return res.reshape(topk_indices.shape), None


def route_prefill_imbalance_jax(
    topk_indices,
    *,
    ep_size: int,
    num_base_local: int = 32,
    replica_meta=None,
    num_replicas: int = 0,
    replica_mode: str = "whole",
    do_chipmate: bool = False,
    chipmate_xor: int = 1,
    chipmate_mode: str = "whole",
    min_tile_gap: int = 3,
    tile_m: int = 128,
    weight_load_tiles: int = 1,
    use_pallas: bool | None = None,
):
    """Unified §4.5b + §4.3 prefill imbalance router (1-op Pallas on TPU, JAX fallback on CPU)."""
    has_rep = replica_meta is not None and num_replicas > 0
    if not has_rep and not do_chipmate:
        return topk_indices, None
    total_elems = int(topk_indices.shape[0] * topk_indices.shape[1])
    if _can_use_tpu_pallas_prefill_router(total_elems, use_pallas):
        return pallas_prefill_imbalance_route(
            topk_indices,
            ep_size=ep_size,
            num_base_local=num_base_local,
            replica_meta=replica_meta if has_rep else None,
            num_replicas=num_replicas if has_rep else 0,
            replica_mode=replica_mode,
            do_chipmate=do_chipmate,
            chipmate_xor=chipmate_xor,
            chipmate_mode=chipmate_mode,
            min_tile_gap=min_tile_gap,
            tile_m=tile_m,
            weight_load_tiles=weight_load_tiles,
        )
    v_cur = topk_indices
    e_cur = num_base_local
    if has_rep:
        v_cur = remap_prefill_replicas_jax(
            v_cur,
            replica_meta,
            ep_size=ep_size,
            num_base_local=num_base_local,
            num_replicas=num_replicas,
            tile_m=tile_m,
            mode=replica_mode,
            weight_load_tiles=weight_load_tiles,
        )
        e_cur = num_base_local + num_replicas
    if do_chipmate:
        return plan_prefill_chipmate_jax(
            v_cur,
            ep_size=ep_size,
            num_local_experts=e_cur,
            tile_m=tile_m,
            chipmate_xor=chipmate_xor,
            min_tile_gap=min_tile_gap,
            mode=chipmate_mode,
            weight_load_tiles=weight_load_tiles,
        )
    return v_cur, None


def interleave_hot_expert_tiles_jax(
    gm_group,
    tiles_per_group,
    *,
    num_slots: int,
    cand_groups: int,
    min_hot_tiles: int = 4,
    skew_ratio: int = 2,
):
    """Build §4.2b exact bijection `perm_gm[num_slots]` + `hot_slot` (`-1` if inactive).

    Alternates tiles of the busiest local expert (`0 .. cand_groups - 1`) `1:1`
    with non-hot local tiles (`H_0, C_0, H_1, C_1, ..., unmatched_tail`), while
    preserving borrowed chipmate tiles (`group >= cand_groups`) and padded slots
    in place at the end of the schedule (Paper §4.2b).
    """
    import jax.numpy as jnp

    cand_t = tiles_per_group[:cand_groups]
    hot_cand = jnp.argmax(cand_t).astype(jnp.int32)
    hot_t = cand_t[hot_cand]
    sum_cand_t = jnp.sum(cand_t)
    cold_t = sum_cand_t - hot_t
    active = (
        (hot_t >= min_hot_tiles)
        & (cold_t >= 2)
        & (hot_t * cand_groups >= skew_ratio * sum_cand_t)
    )
    hot_slot = jnp.where(active, hot_cand, jnp.int32(-1))

    gm = jnp.arange(num_slots, dtype=jnp.int32)
    in_cand_window = active & (gm < sum_cand_t)
    is_hot = in_cand_window & (gm_group == hot_slot)
    is_cold = in_cand_window & (~is_hot)

    hot_ord = jnp.cumsum(is_hot.astype(jnp.int32)) - 1
    cold_ord = jnp.cumsum(is_cold.astype(jnp.int32)) - 1
    n_pairs = jnp.minimum(
        jnp.sum(is_hot.astype(jnp.int32)), jnp.sum(is_cold.astype(jnp.int32))
    )

    hot_dst = jnp.where(hot_ord < n_pairs, 2 * hot_ord, n_pairs + hot_ord)
    cold_dst = jnp.where(cold_ord < n_pairs, 2 * cold_ord + 1, n_pairs + cold_ord)
    dst_pos = jnp.where(is_hot, hot_dst, jnp.where(is_cold, cold_dst, gm))

    perm_gm = (
        jnp.zeros((num_slots,), dtype=jnp.int32)
        .at[dst_pos]
        .set(gm, unique_indices=True)
    )
    return perm_gm, hot_slot

