# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang launch_server CLI flag emitter, structural Top-3 pruner, and solve/v1 serializer."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .chips import ChipSpec
from .hf_loader import Dtypes
from .solver import FloorTimes, ParetoPoint, PoolMeshPlan, ServingPlan

SOLVE_SCHEMA: str = "sglang.roofline.solve/v1"

_VALID_SGLANG_KV_DTYPES: frozenset[str] = frozenset(
    {
        "auto",
        "fp8_e5m2",
        "fp8_e4m3",
        "mxfp8",
        "mxint8",
        "bf16",
        "bfloat16",
        "nvfp4",
        "fp4_mx_block16",
    }
)


def _normalize_kv_dtype(kv_dtype: str, dtypes: Dtypes | None = None) -> str:
    if kv_dtype == "fp8":
        return "fp8_e4m3"
    if kv_dtype != "auto":
        if kv_dtype not in _VALID_SGLANG_KV_DTYPES:
            raise ValueError(f"Unsupported SGLang --kv-cache-dtype {kv_dtype!r}")
        return kv_dtype
    if dtypes is not None:
        if dtypes.kv <= 1.0:
            return "fp8_e4m3"
        if dtypes.kv >= 2.0:
            return "bfloat16"
    return "auto"


def pool_cli_args(
    pool: PoolMeshPlan,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
    node_rank: int = 0,
    dist_init_addr: str = "$MASTER_ADDR:29500",
) -> list[str]:
    """Render verified SGLang ServerArgs CLI flags for one mixed, prefill, or decode pool."""
    norm_kv = _normalize_kv_dtype(kv_dtype)
    mem_frac = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else pool.gpu_memory_utilization
    )
    args: list[str] = []

    # Parallelism flags (python/sglang/srt/arg_groups/fields/parallel.py; never emit no_cli enable_dp_attention).
    eff_moe_dp = pool.pcp if pool.pcp > 1 else pool.moe_dp
    if pool.dp > 1 and pool.ep > 1:
        args.extend(
            [
                "--tp-size",
                str(pool.tp * pool.dp),
                "--attn-dp-size",
                str(pool.dp),
                "--ep-size",
                str(pool.ep),
            ]
        )
        if eff_moe_dp > 1 and pool.pcp <= 1:
            args.extend(["--moe-dp-size", str(eff_moe_dp)])
        if pool.tp > 1 or pool.spec_tokens > 0:
            args.append("--enable-dp-lm-head")
    elif pool.dp > 1:
        args.extend(["--tp-size", str(pool.tp), "--dp-size", str(pool.dp)])
    else:
        args.extend(["--tp-size", str(pool.tp)])
        if pool.ep > 1:
            args.extend(["--ep-size", str(pool.ep)])
        if eff_moe_dp > 1 and pool.pcp <= 1:
            args.extend(["--moe-dp-size", str(eff_moe_dp)])

    if pool.pp > 1:
        args.extend(["--pp-size", str(pool.pp)])
    if pool.pcp > 1:
        args.extend(
            [
                "--attn-cp-size",
                str(pool.pcp),
                "--enable-prefill-cp",
                "--cp-strategy",
                "zigzag",
            ]
        )
        if eff_moe_dp > 1:
            args.extend(["--moe-dp-size", str(eff_moe_dp)])
    if pool.kv_split > 1:
        args.extend(
            [
                "--decode-context-parallel-size",
                str(pool.kv_split),
                "--dcp-comm-backend",
                "ag_rs",
            ]
        )

    # SGLang TPU requires --enable-torch-compile whenever pp > 1, pcp > 1, or kv_split > 1
    if pool.pp > 1 or pool.pcp > 1 or pool.kv_split > 1:
        args.append("--enable-torch-compile")

    # Multi-node topology flags:
    if pool.chips_per_instance > 4:
        nnodes = max(1, pool.chips_per_instance // 4)
        args.extend(
            [
                "--nnodes",
                str(nnodes),
                "--node-rank",
                str(node_rank),
                "--dist-init-addr",
                dist_init_addr,
            ]
        )

    # Speculative decoding:
    if pool.spec_tokens > 0:
        k = pool.spec_tokens
        args.extend(
            [
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                str(k),
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                str(k + 1),
            ]
        )

    if pool.role == "prefill":
        args.extend(["--disaggregation-mode", "prefill"])
    elif pool.role == "decode":
        args.extend(["--disaggregation-mode", "decode"])

    eff_page = (
        max(pool.page_size, 256)
        if (pool.pcp > 1 or pool.kv_split > 1)
        else pool.page_size
    )
    args.extend(["--page-size", str(eff_page)])
    args.extend(["--kv-cache-dtype", norm_kv])
    if mem_frac is not None:
        args.extend(["--mem-fraction-static", f"{mem_frac:g}"])

    if pool.kv_offload_gb > 0.0 and pool.kv_split == 1:
        hicache_gb = max(1, int(math.ceil(pool.kv_offload_gb)))
        args.extend(["--enable-hierarchical-cache", "--hicache-size", str(hicache_gb)])

    if pool.max_running_requests is not None:
        args.extend(["--max-running-requests", str(pool.max_running_requests)])

    return args


def _format_server_cmd(cli_args: list[str], model: str | None) -> str:
    prefix = ["python3", "-m", "sglang.launch_server"]
    if model:
        prefix.extend(["--model-path", model])
    return " ".join([*prefix, *cli_args])


def _pool_key(pool: PoolMeshPlan | None) -> tuple | None:
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep > 1,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.spec_tokens,
    )


def _family_key(pt: ParetoPoint) -> tuple:
    return (pt.mode, _pool_key(pt.prefill_pool), _pool_key(pt.decode_pool))


def top_layouts(plan: ServingPlan, k: int = 3) -> tuple[ParetoPoint, ...]:
    """Select up to k structurally distinct layouts covering throughput, interactivity, and knee roles."""
    if k <= 0:
        return ()
    ordered_sources = [*plan.frontier, *plan.candidates]
    if not ordered_sources:
        return ()

    # Group best throughput representative and best interactivity representative per structural family.
    by_tput: dict[tuple, ParetoPoint] = {}
    by_inter: dict[tuple, ParetoPoint] = {}
    family_order: list[tuple] = []
    for pt in ordered_sources:
        fkey = _family_key(pt)
        if fkey not in by_tput:
            family_order.append(fkey)
            by_tput[fkey] = pt
            by_inter[fkey] = pt
        else:
            if pt.token_throughput_per_chip > by_tput[fkey].token_throughput_per_chip:
                by_tput[fkey] = pt
            if pt.p90_interactivity_tok_s > by_inter[fkey].p90_interactivity_tok_s:
                by_inter[fkey] = pt

    if not family_order:
        return ()

    # Role 1: max_throughput
    f_tput = max(
        family_order,
        key=lambda fk: (
            by_tput[fk].token_throughput_per_chip,
            by_tput[fk].p90_interactivity_tok_s,
        ),
    )
    top1_pt = by_tput[f_tput]
    top1_tput = top1_pt.token_throughput_per_chip
    chosen: list[tuple[tuple, ParetoPoint, str]] = [(f_tput, top1_pt, "max_throughput")]
    used: set[tuple] = {f_tput}

    remaining = [fk for fk in family_order if fk not in used]
    if k >= 2 and remaining:
        # Role 2: max_interactivity among structurally distinct families
        f_inter = max(
            remaining,
            key=lambda fk: (
                by_inter[fk].p90_interactivity_tok_s,
                by_inter[fk].token_throughput_per_chip,
            ),
        )
        chosen.append((f_inter, by_inter[f_inter], "max_interactivity"))
        used.add(f_inter)

    remaining = [fk for fk in family_order if fk not in used]
    if k >= 3 and remaining:
        max_t = max(p.token_throughput_per_chip for p in by_tput.values())
        max_i = max(p.p90_interactivity_tok_s for p in by_inter.values())
        f_knee = max(
            remaining,
            key=lambda fk: (
                by_tput[fk].token_throughput_per_chip / max(1e-9, max_t)
                + by_tput[fk].p90_interactivity_tok_s / max(1e-9, max_i)
            ),
        )
        chosen.append((f_knee, by_tput[f_knee], "balanced_knee"))
        used.add(f_knee)

    if len(chosen) < k:
        extra = sorted(
            (fk for fk in family_order if fk not in used),
            key=lambda fk: -by_tput[fk].token_throughput_per_chip,
        )
        for idx, fk in enumerate(extra[: k - len(chosen)], start=len(chosen) + 1):
            chosen.append((fk, by_tput[fk], f"rank_{idx}"))

    annotated: list[ParetoPoint] = []
    for _fk, pt, role in chosen[:k]:
        margin = round(
            100.0 * (pt.token_throughput_per_chip - top1_tput) / max(1e-9, top1_tput),
            2,
        )
        annotated.append(replace(pt, frontier_role=role, margin_vs_rank1_pct=margin))
    return tuple(annotated)


def point_launch_args(
    pt: ParetoPoint,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Render SGLang launch_server flag lists and command strings for one ParetoPoint."""
    dec_args = pool_cli_args(
        pt.decode_pool,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        node_rank=0,
        dist_init_addr="$MASTER_ADDR:29500",
    )
    if pt.prefill_pool is None:
        commands: list[str] = []
        if pt.decode_pool.chips_per_instance > 4:
            nnodes = max(1, pt.decode_pool.chips_per_instance // 4)
            for r in range(nnodes):
                node_cli = pool_cli_args(
                    pt.decode_pool,
                    kv_dtype=kv_dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    node_rank=r,
                    dist_init_addr="$MASTER_ADDR:29500",
                )
                commands.append(_format_server_cmd(node_cli, model))
        else:
            commands.append(_format_server_cmd(dec_args, model))
        return {
            "mode": pt.mode,
            "cli_args": dec_args,
            "prefill_cli_args": [],
            "commands": commands,
        }
    pre_args = pool_cli_args(
        pt.prefill_pool,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        node_rank=0,
        dist_init_addr="$PREFILL_MASTER_ADDR:29500",
    )
    dec_args = pool_cli_args(
        pt.decode_pool,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        node_rank=0,
        dist_init_addr="$DECODE_MASTER_ADDR:29501",
    )
    commands = []
    if pt.prefill_pool.chips_per_instance > 4:
        nnodes = max(1, pt.prefill_pool.chips_per_instance // 4)
        for r in range(nnodes):
            node_cli = pool_cli_args(
                pt.prefill_pool,
                kv_dtype=kv_dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                node_rank=r,
                dist_init_addr="$PREFILL_MASTER_ADDR:29500",
            )
            commands.append(_format_server_cmd(node_cli, model))
    else:
        commands.append(_format_server_cmd(pre_args, model))

    if pt.decode_pool.chips_per_instance > 4:
        nnodes = max(1, pt.decode_pool.chips_per_instance // 4)
        for r in range(nnodes):
            node_cli = pool_cli_args(
                pt.decode_pool,
                kv_dtype=kv_dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                node_rank=r,
                dist_init_addr="$DECODE_MASTER_ADDR:29501",
            )
            commands.append(_format_server_cmd(node_cli, model))
    else:
        commands.append(_format_server_cmd(dec_args, model))

    return {
        "mode": pt.mode,
        "cli_args": dec_args,
        "prefill_cli_args": pre_args,
        "commands": commands,
    }


def emit_launch_args(
    plan: ServingPlan,
    *,
    dtypes: Dtypes | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Render launch commands for the recommended point and structurally distinct Top-3 layouts."""
    kv_str = _normalize_kv_dtype("auto", dtypes)
    if plan.recommended is None:
        return {
            "mode": plan.mode,
            "cli_args": [],
            "prefill_cli_args": [],
            "commands": [],
            "top_k": [],
        }
    rec_launch = point_launch_args(
        plan.recommended,
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model=model,
    )
    top_entries = []
    for pt in top_layouts(plan, k=3):
        top_entries.append(
            {
                "frontier_role": pt.frontier_role,
                "margin_vs_rank1_pct": pt.margin_vs_rank1_pct,
                **point_launch_args(
                    pt,
                    kv_dtype=kv_str,
                    gpu_memory_utilization=plan.gpu_memory_utilization,
                    model=model,
                ),
            }
        )
    return {
        **rec_launch,
        "top_k": top_entries,
    }


def _finite_or_none(val: float) -> float | None:
    return val if math.isfinite(val) else None


def _chip_dict(chip: ChipSpec) -> dict[str, Any]:
    return {
        "name": chip.name,
        "hbm_bytes": chip.hbm_bytes,
        "hbm_bw": chip.hbm_bw,
        "bf16_flops": chip.bf16_flops,
        "fp8_flops": chip.fp8_flops,
        "vmem_bytes": chip.vmem_bytes,
        "ici_bw": _finite_or_none(chip.ici_bw),
        "cores_per_chip": chip.cores_per_chip,
        "ici_link_bw": chip.ici_link_bw,
        "ici_hop_latency_s": chip.ici_hop_latency_s,
        "mxu_m_tile": chip.mxu_m_tile,
        "pcie_bw": chip.pcie_bw,
        "dcn_bw": chip.dcn_bw,
    }


def _floor_dict(fl: FloorTimes | None) -> dict[str, Any] | None:
    if fl is None:
        return None
    return {
        "hbm_ms": fl.hbm_ms,
        "matmul_ms": fl.matmul_ms,
        "ici_ms": fl.ici_ms,
        "chain_ms": fl.chain_ms,
        "hop_latency_ms": fl.hop_latency_ms,
        "hard_floor_ms": fl.hard_floor_ms,
        "serial_floor_ms": fl.serial_floor_ms,
        "binding": fl.binding,
    }


def _pool_dict(pool: PoolMeshPlan | None) -> dict[str, Any] | None:
    if pool is None:
        return None
    return {
        "role": pool.role,
        "instances": pool.instances,
        "topology": pool.topology,
        "chips_per_instance": pool.chips_per_instance,
        "tp": pool.tp,
        "dp": pool.dp,
        "ep": pool.ep,
        "moe_dp": pool.moe_dp,
        "pp": pool.pp,
        "pcp": pool.pcp,
        "kv_split": pool.kv_split,
        "dcp": pool.kv_split,
        "spec_tokens": pool.spec_tokens,
        "mesh_axes": list(pool.mesh_axes),
        "hbm_used_gb": pool.hbm_used_gb,
        "hbm_limit_gb": pool.hbm_limit_gb,
        "moe_dispatch": pool.moe_dispatch,
        "page_size": pool.page_size,
        "kv_offload_gb": pool.kv_offload_gb,
        "max_running_requests": pool.max_running_requests,
        "gpu_memory_utilization": pool.gpu_memory_utilization,
        "hbm_weights_gb": pool.hbm_weights_gb,
        "hbm_kv_gb": pool.hbm_kv_gb,
        "hbm_state_gb": pool.hbm_state_gb,
        "kv_cache_capacity_tokens": pool.kv_cache_capacity_tokens,
    }


def _point_dict(pt: ParetoPoint | None) -> dict[str, Any] | None:
    if pt is None:
        return None
    return {
        "mode": pt.mode,
        "concurrency": pt.concurrency,
        "served_concurrency": pt.served_concurrency,
        "total_chips": pt.total_chips,
        "prefill_pool": _pool_dict(pt.prefill_pool),
        "decode_pool": _pool_dict(pt.decode_pool),
        "p90_ttft_ms": pt.p90_ttft_ms,
        "p90_tpot_ms": pt.p90_tpot_ms,
        "p90_interactivity_tok_s": pt.p90_interactivity_tok_s,
        "output_throughput_per_chip": pt.output_throughput_per_chip,
        "token_throughput_per_chip": pt.token_throughput_per_chip,
        "p_mixed": pt.p_mixed,
        "binding": pt.binding,
        "slo_met": pt.slo_met,
        "decode_floor": _floor_dict(pt.decode_floor),
        "prefill_floor": _floor_dict(pt.prefill_floor),
        "kv_cache_capacity_tokens": pt.kv_cache_capacity_tokens,
        "working_set_tokens": pt.working_set_tokens,
        "effective_prefix_hit_rate": pt.effective_prefix_hit_rate,
        "pcie_hicache_ms": pt.pcie_hicache_ms,
        "dcn_transfer_ms": pt.dcn_transfer_ms,
        "queue_wait_ms": pt.queue_wait_ms,
        "frontier_role": pt.frontier_role,
        "margin_vs_rank1_pct": pt.margin_vs_rank1_pct,
    }


def build_solve_payload(
    plan: ServingPlan,
    *,
    chip: ChipSpec,
    dtypes: Dtypes | None = None,
    model: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Serialize a ServingPlan into the sglang.roofline.solve/v1 JSON schema."""
    kv_str = _normalize_kv_dtype("auto", dtypes)
    sc = plan.scenario
    scenario_dict = {
        "name": sc.name,
        "isl": sc.isl,
        "osl": sc.osl,
        "prefix_hit_rate": sc.prefix_hit_rate,
        "shared_prefix_len": sc.shared_prefix_len,
        "random_prefix_len": sc.random_prefix_len,
        "accept_len": sc.accept_len,
        "think_time_s": sc.think_time_s,
        "concurrencies": list(sc.concurrencies),
        "kv_working_set_turns": sc.kv_working_set_turns,
        "prefill_chunk": sc.prefill_chunk,
        "ttft_slo_ms": sc.ttft_slo_ms,
        "tpot_slo_ms": sc.tpot_slo_ms,
        "spec_tokens": sc.spec_tokens,
        "kv_offload_gb": sc.kv_offload_gb,
        "page_size": sc.page_size,
        "max_running_requests": sc.max_running_requests,
    }
    top_pts = top_layouts(plan, k=top_k)
    top_k_list: list[dict[str, Any]] = []
    for pt in top_pts:
        pt_d = _point_dict(pt)
        assert pt_d is not None
        launch_d = point_launch_args(
            pt,
            kv_dtype=kv_str,
            gpu_memory_utilization=plan.gpu_memory_utilization,
            model=model,
        )
        top_k_list.append(
            {
                **pt_d,
                "point": pt_d,
                "launch": launch_d,
            }
        )

    rec = plan.recommended
    hw_profile = (
        f"v7x-{rec.decode_pool.topology}"
        if rec is not None and plan.chip == "tpu7x"
        else plan.chip
    )
    target_chips = rec.total_chips if rec is not None else 0
    search_dict = (
        None
        if plan.search is None
        else {
            "enumerated": plan.search.enumerated,
            "hbm_fit": plan.search.hbm_fit,
            "slo_met": plan.search.slo_met,
            "frontier": plan.search.frontier,
            "legal_layouts": plan.search.legal_layouts,
        }
    )
    return {
        "schema": SOLVE_SCHEMA,
        "model": model,
        "model_layers": plan.model_layers,
        "chip": plan.chip,
        "hardware": hw_profile,
        "target_chips": target_chips,
        "chip_spec": _chip_dict(chip),
        "mode": plan.mode,
        "isl": sc.isl,
        "osl": sc.osl,
        "concurrency": rec.concurrency if rec is not None else sc.concurrencies[-1],
        "scenario": scenario_dict,
        "recommended": _point_dict(rec),
        "frontier": [_point_dict(p) for p in plan.frontier],
        "candidates": [_point_dict(p) for p in plan.candidates],
        "top_k": top_k_list,
        "prefix_hit_crossover_rate": plan.prefix_hit_crossover_rate,
        "winner_prefix_hit_range": (
            list(plan.winner_prefix_hit_range)
            if plan.winner_prefix_hit_range is not None
            else None
        ),
        "warnings": list(plan.warnings),
        "gpu_memory_utilization": plan.gpu_memory_utilization,
        "search": search_dict,
        "provenance": {
            "calibrated": False,
            "basis": "analytical roofline floors; p90 values are modeled, not measured",
        },
    }


def emit_ceiling_rows(plan: ServingPlan) -> list[dict[str, Any]]:
    """Export Pareto frontier ceiling rows ordered by interactivity descending."""
    rows: list[dict[str, Any]] = []
    for pt in plan.frontier:
        d = pt.decode_pool
        p = pt.prefill_pool
        rows.append(
            {
                "mode": pt.mode,
                "concurrency": pt.concurrency,
                "served_concurrency": pt.served_concurrency,
                "total_chips": pt.total_chips,
                "decode_topology": d.topology,
                "decode_tp": d.tp,
                "decode_dp": d.dp,
                "decode_ep": d.ep,
                "decode_pp": d.pp,
                "decode_dcp": d.kv_split,
                "prefill_topology": None if p is None else p.topology,
                "prefill_instances": 0 if p is None else p.instances,
                "decode_instances": d.instances,
                "p90_interactivity_tok_s": round(pt.p90_interactivity_tok_s, 4),
                "token_throughput_per_chip": round(pt.token_throughput_per_chip, 4),
                "output_throughput_per_chip": round(pt.output_throughput_per_chip, 4),
                "p90_ttft_ms": round(pt.p90_ttft_ms, 4),
                "p90_tpot_ms": round(pt.p90_tpot_ms, 4),
                "effective_prefix_hit_rate": round(pt.effective_prefix_hit_rate, 4),
                "kv_cache_capacity_tokens": pt.kv_cache_capacity_tokens,
                "working_set_tokens": pt.working_set_tokens,
                "binding": pt.binding,
            }
        )
    return rows
