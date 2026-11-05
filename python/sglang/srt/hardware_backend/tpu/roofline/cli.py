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

"""Command-line interface for sglang-tpu-roofline (solve, emit, ceiling, plan, explain)."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .chips import CHIPS
from .emit import build_solve_payload, emit_ceiling_rows, emit_launch_args
from .geometry import from_hf_config
from .hf_loader import Dtypes, dtypes_from_hf_config, load_hf_config
from .modules import Parallel, StepShape
from .report import write_html
from .serving import plan as run_serving_plan
from .solver import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    EXAMPLE_SCENARIOS,
    EXAMPLE_SLO_TIERS,
    TrafficScenario,
    build_example_scenario,
    solve_serving_plan,
)
from .step import spec_cycle_floor, step_floor

_DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "mxfp4": 0.53125,
    "nvfp4": 0.5625,
}


def parse_bench_cmd(cmd: str) -> dict[str, Any]:
    """Extract workload and server flags from an sglang bench_serving or launch_server command."""
    tokens = [tok for tok in shlex.split(cmd) if tok != "\\"]
    bp = argparse.ArgumentParser(add_help=False)
    bp.add_argument("--model-path", "--model", dest="model", default=None)
    bp.add_argument(
        "--random-input-len", "--input-len", "--isl", dest="isl", type=int, default=None
    )
    bp.add_argument(
        "--random-output-len",
        "--output-len",
        "--osl",
        dest="osl",
        type=int,
        default=None,
    )
    bp.add_argument(
        "--shared-prefix-len",
        "--gsp-system-prompt-len",
        dest="shared_prefix_len",
        type=int,
        default=None,
    )
    bp.add_argument(
        "--random-prefix-len",
        "--gsp-question-len",
        dest="random_prefix_len",
        type=int,
        default=None,
    )
    bp.add_argument(
        "--max-concurrency", "--concurrency", dest="concurrency", default=None
    )
    bp.add_argument(
        "--mem-fraction-static",
        "--gpu-memory-utilization",
        dest="gpu_memory_utilization",
        type=float,
        default=None,
    )
    bp.add_argument("--page-size", dest="page_size", type=int, default=None)
    bp.add_argument(
        "--max-running-requests", dest="max_running_requests", type=int, default=None
    )
    bp.add_argument(
        "--enable-hierarchical-cache", dest="enable_hicache", action="store_true"
    )
    bp.add_argument("--hicache-size", dest="hicache_size", type=float, default=None)
    bp.add_argument(
        "--spec-tokens",
        "--speculative-num-steps",
        dest="spec_tokens",
        type=int,
        default=None,
    )
    ns, _ = bp.parse_known_args(tokens)
    out: dict[str, Any] = {}
    if ns.model is not None:
        out["model"] = ns.model
    if ns.isl is not None:
        out["isl"] = ns.isl
    if ns.osl is not None:
        out["osl"] = ns.osl
    if ns.shared_prefix_len is not None:
        out["shared_prefix_len"] = ns.shared_prefix_len
    if ns.random_prefix_len is not None:
        out["random_prefix_len"] = ns.random_prefix_len
    if ns.concurrency is not None:
        out["concurrencies"] = tuple(
            int(x.strip()) for x in str(ns.concurrency).split(",") if x.strip()
        )
    if ns.gpu_memory_utilization is not None:
        out["gpu_memory_utilization"] = ns.gpu_memory_utilization
    if ns.page_size is not None:
        out["page_size"] = ns.page_size
    if ns.max_running_requests is not None:
        out["max_running_requests"] = ns.max_running_requests
    if ns.hicache_size is not None and (ns.enable_hicache or ns.hicache_size > 0.0):
        out["kv_offload_gb"] = float(ns.hicache_size)
    if ns.spec_tokens is not None:
        out["spec_tokens"] = ns.spec_tokens
    return out


def _add_model_source(p: argparse.ArgumentParser) -> None:
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        "--config", type=str, default=None, help="Path to HuggingFace config.json"
    )
    grp.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID, directory, or config.json",
    )
    p.add_argument(
        "--from-bench-cmd",
        type=str,
        default=None,
        help="Optional sglang bench_serving or launch_server command string to seed flags",
    )


def _add_common_args(p: argparse.ArgumentParser, *, tp_required: bool = True) -> None:
    p.add_argument("--chip", choices=sorted(CHIPS), default="tpu7x")
    p.add_argument("--tp", type=int, required=tp_required, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--pcp", type=int, default=1)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--kv-split", type=int, default=1)
    p.add_argument("--kv-merge", choices=("ag_rs", "exchange"), default="ag_rs")
    p.add_argument("--moe", choices=("tp_slice", "ep"), default="tp_slice")
    p.add_argument(
        "--moe-dispatch",
        choices=("allreduce", "ag_ep", "fused_rs"),
        default="allreduce",
    )
    p.add_argument("--residual", choices=("replicated", "sp"), default="replicated")
    p.add_argument(
        "--ar-algo", choices=("one_shot", "two_shot", "ring"), default="one_shot"
    )
    p.add_argument("--weight-dtype", choices=("bf16", "fp8"), default=None)
    p.add_argument(
        "--expert-dtype", choices=("bf16", "fp8", "nvfp4", "mxfp4"), default=None
    )
    p.add_argument("--aux-dtype", choices=("bf16", "fp8"), default=None)
    p.add_argument("--kv-dtype", choices=("bf16", "fp8"), default=None)
    p.add_argument("--state-dtype", choices=("fp32", "bf16"), default=None)
    p.add_argument("--conv-dtype", choices=("fp32", "bf16"), default=None)
    p.add_argument("--act", choices=("a8", "a16"), default=None)
    p.add_argument("--ici-bw-gbps", type=float, default=None)
    p.add_argument("--collective-latency-us", type=float, default=None)
    p.add_argument("--json", action="store_true")


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chip", choices=sorted(CHIPS), default="tpu7x")
    p.add_argument(
        "--example-scenario", choices=sorted(EXAMPLE_SCENARIOS), default=None
    )
    p.add_argument("--mode", choices=("auto", "mixed", "disagg"), default="auto")
    p.add_argument("--max-chips", type=int, default=None)
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument(
        "--example-slo-tier", choices=sorted(EXAMPLE_SLO_TIERS), default="none"
    )
    p.add_argument("--ttft-slo-ms", type=float, default=None)
    p.add_argument("--tpot-slo-ms", type=float, default=None)
    p.add_argument("--isl", type=int, default=None)
    p.add_argument("--osl", type=int, default=None)
    p.add_argument("--shared-prefix-len", type=int, default=None)
    p.add_argument("--random-prefix-len", type=int, default=None)
    p.add_argument("--concurrency", type=str, default=None)
    p.add_argument("--prefix-hit-rate", type=float, default=None)
    p.add_argument("--accept-len", type=float, default=None)
    p.add_argument("--think-time-s", type=float, default=None)
    p.add_argument("--kv-offload-gb", type=float, default=None)
    p.add_argument("--page-size", type=int, default=None)
    p.add_argument("--max-running-requests", type=int, default=None)
    p.add_argument("--spec-tokens", type=int, default=None)
    p.add_argument(
        "--mem-fraction-static",
        "--gpu-memory-utilization",
        dest="gpu_memory_utilization",
        type=float,
        default=None,
    )
    p.add_argument("--json", action="store_true")


def _resolve_model_and_config(
    ns: argparse.Namespace, bench_seed: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    src = ns.config or ns.model or bench_seed.get("model")
    if not src:
        raise ValueError(
            "One of --config, --model, or --from-bench-cmd with --model-path is required"
        )
    model_label = ns.model or bench_seed.get("model") or ns.config
    return model_label, load_hf_config(src)


def _resolve_scenario(
    ns: argparse.Namespace, bench_seed: dict[str, Any]
) -> tuple[TrafficScenario, float]:
    concs = (
        tuple(int(x.strip()) for x in ns.concurrency.split(",") if x.strip())
        if ns.concurrency is not None
        else bench_seed.get("concurrencies")
    )
    isl = ns.isl if ns.isl is not None else bench_seed.get("isl")
    osl = ns.osl if ns.osl is not None else bench_seed.get("osl")
    sp_len = (
        ns.shared_prefix_len
        if ns.shared_prefix_len is not None
        else bench_seed.get("shared_prefix_len")
    )
    rp_len = (
        ns.random_prefix_len
        if ns.random_prefix_len is not None
        else bench_seed.get("random_prefix_len")
    )
    kv_off = (
        ns.kv_offload_gb
        if ns.kv_offload_gb is not None
        else bench_seed.get("kv_offload_gb")
    )
    pg_sz = ns.page_size if ns.page_size is not None else bench_seed.get("page_size")
    spec_toks = (
        ns.spec_tokens if ns.spec_tokens is not None else bench_seed.get("spec_tokens")
    )
    max_run = (
        ns.max_running_requests
        if ns.max_running_requests is not None
        else bench_seed.get("max_running_requests")
    )
    mem_util = (
        ns.gpu_memory_utilization
        if ns.gpu_memory_utilization is not None
        else bench_seed.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)
    )

    if ns.example_scenario is not None:
        scen = build_example_scenario(
            ns.example_scenario,
            isl=isl,
            osl=osl,
            concurrencies=concs,
            prefix_hit_rate=ns.prefix_hit_rate,
            shared_prefix_len=sp_len,
            random_prefix_len=rp_len,
            accept_len=ns.accept_len,
            think_time_s=ns.think_time_s,
            slo_tier=ns.example_slo_tier,
            ttft_slo_ms=ns.ttft_slo_ms,
            tpot_slo_ms=ns.tpot_slo_ms,
            kv_offload_gb=kv_off,
            page_size=pg_sz,
            max_running_requests=max_run,
        )
        return scen, float(mem_util)

    tier_ttft, tier_tpot = EXAMPLE_SLO_TIERS[ns.example_slo_tier]
    scen = TrafficScenario(
        name="cli",
        isl=isl if isl is not None else 4096,
        osl=osl if osl is not None else 1024,
        concurrencies=concs if concs is not None else (1, 4, 16, 64),
        prefix_hit_rate=ns.prefix_hit_rate if ns.prefix_hit_rate is not None else 0.0,
        shared_prefix_len=sp_len,
        random_prefix_len=rp_len,
        accept_len=ns.accept_len if ns.accept_len is not None else 1.0,
        think_time_s=ns.think_time_s if ns.think_time_s is not None else 0.0,
        ttft_slo_ms=ns.ttft_slo_ms if ns.ttft_slo_ms is not None else tier_ttft,
        tpot_slo_ms=ns.tpot_slo_ms if ns.tpot_slo_ms is not None else tier_tpot,
        kv_offload_gb=kv_off if kv_off is not None else 0.0,
        page_size=pg_sz,
        max_running_requests=max_run,
    )
    return scen, float(mem_util)


def _apply_dtype_overrides(base: Dtypes, ns: argparse.Namespace) -> Dtypes:
    dt = base
    if ns.weight_dtype is not None:
        dt = replace(dt, w_dense=_DTYPE_BYTES[ns.weight_dtype])
    if ns.expert_dtype is not None:
        dt = replace(dt, w_expert=_DTYPE_BYTES[ns.expert_dtype])
    if ns.aux_dtype is not None:
        dt = replace(dt, w_aux=_DTYPE_BYTES[ns.aux_dtype])
    if ns.kv_dtype is not None:
        dt = replace(dt, kv=_DTYPE_BYTES[ns.kv_dtype])
    if ns.state_dtype is not None:
        dt = replace(dt, state=_DTYPE_BYTES[ns.state_dtype])
    if ns.conv_dtype is not None:
        dt = replace(dt, conv=_DTYPE_BYTES[ns.conv_dtype])
    if ns.act is not None:
        dt = replace(dt, act=ns.act)
    return dt


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for sglang-tpu-roofline CLI."""
    parser = argparse.ArgumentParser(prog="sglang-tpu-roofline")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_solve = sub.add_parser(
        "solve", help="Search single-pool and disaggregated SGLang TPU layouts"
    )
    _add_model_source(p_solve)
    _add_scenario_args(p_solve)
    p_solve.add_argument(
        "--html", type=Path, default=None, help="Write offline SVG HTML report to path"
    )

    p_emit = sub.add_parser(
        "emit", help="Emit ready-to-run python3 -m sglang.launch_server commands"
    )
    _add_model_source(p_emit)
    _add_scenario_args(p_emit)

    p_ceil = sub.add_parser("ceiling", help="Export Pareto frontier ceiling curve rows")
    _add_model_source(p_ceil)
    _add_scenario_args(p_ceil)

    p_plan = sub.add_parser(
        "plan", help="Evaluate concurrency ceiling rows for a fixed Parallel tuple"
    )
    _add_model_source(p_plan)
    _add_common_args(p_plan, tp_required=True)
    p_plan.add_argument("--concurrency", type=str, default="1,4,16,64")
    p_plan.add_argument("--spec-window", type=int, default=0)
    p_plan.add_argument("--acceptance-length", type=float, default=None)

    p_exp = sub.add_parser(
        "explain",
        help="Print per-module HBM, MXU, and collective breakdown for one step",
    )
    _add_model_source(p_exp)
    _add_common_args(p_exp, tp_required=True)
    p_exp.add_argument("--batch", type=int, default=1)
    p_exp.add_argument("--rows-per-seq", type=int, default=1)
    p_exp.add_argument("--spec-window", type=int, default=0)

    ns = parser.parse_args(argv)
    bench_seed = parse_bench_cmd(ns.from_bench_cmd) if ns.from_bench_cmd else {}
    model_label, cfg = _resolve_model_and_config(ns, bench_seed)
    geom = from_hf_config(cfg)
    base_dt = dtypes_from_hf_config(cfg)
    chip = CHIPS[ns.chip]

    if ns.subcmd in ("solve", "emit", "ceiling"):
        scen, mem_util = _resolve_scenario(ns, bench_seed)
        sp = solve_serving_plan(
            geom,
            base_dt,
            chip,
            scen,
            mode=ns.mode,
            max_chips=ns.max_chips,
            max_instances=ns.max_instances,
            gpu_memory_utilization=mem_util,
        )
        if ns.subcmd == "solve":
            payload = build_solve_payload(
                sp, chip=chip, dtypes=base_dt, model=model_label
            )
            if ns.html is not None:
                write_html(payload, ns.html)
            if ns.json:
                sys.stdout.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            else:
                rec = sp.recommended
                if rec is None:
                    sys.stdout.write("No feasible SGLang TPU layout found.\n")
                else:
                    sys.stdout.write(
                        f"Recommended ({rec.mode}, {rec.total_chips} chips, c={rec.concurrency}): "
                        f"tput/chip={rec.token_throughput_per_chip:.2f} tok/s, "
                        f"interactivity={rec.p90_interactivity_tok_s:.2f} tok/s, "
                        f"TTFT={rec.p90_ttft_ms:.1f} ms, TPOT={rec.p90_tpot_ms:.2f} ms\n"
                    )
            return 0

        if ns.subcmd == "emit":
            launched = emit_launch_args(sp, dtypes=base_dt, model=model_label)
            if ns.json:
                sys.stdout.write(json.dumps(launched, indent=2, allow_nan=False) + "\n")
            else:
                for entry in launched["top_k"]:
                    sys.stdout.write(
                        f"# {entry['frontier_role']} ({entry['margin_vs_rank1_pct']:+.2f}%)\n"
                    )
                    for cmd_line in entry["commands"]:
                        sys.stdout.write(cmd_line + "\n")
            return 0

        rows = emit_ceiling_rows(sp)
        if ns.json:
            sys.stdout.write(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        else:
            for r in rows:
                sys.stdout.write(
                    f"c={r['concurrency']:>4d}  chips={r['total_chips']:>3d}  "
                    f"inter={r['p90_interactivity_tok_s']:>7.2f}  tput/chip={r['token_throughput_per_chip']:>8.2f}\n"
                )
        return 0

    if ns.ici_bw_gbps is not None:
        chip = replace(
            chip,
            ici_bw=float(ns.ici_bw_gbps) * 1e9
            if math.isfinite(ns.ici_bw_gbps)
            else math.inf,
        )
    dt = _apply_dtype_overrides(base_dt, ns)
    par = Parallel(
        tp=ns.tp,
        pp=ns.pp,
        dp=ns.dp,
        ep=ns.ep if ns.ep > 1 else (ns.tp if ns.moe == "ep" else 1),
        pcp=ns.pcp,
        kv_split=ns.kv_split,
        kv_merge=ns.kv_merge,
        moe=ns.moe,
        moe_dispatch=ns.moe_dispatch,
        residual=ns.residual,
        ar_algo=ns.ar_algo,
    )

    if ns.subcmd == "plan":
        concs = [int(x.strip()) for x in ns.concurrency.split(",") if x.strip()]
        c_rows = run_serving_plan(
            geom,
            chip,
            par,
            dt,
            context=ns.context,
            concurrencies=concs,
            spec_window=ns.spec_window,
            acceptance_length=ns.acceptance_length,
            collective_latency_us=ns.collective_latency_us,
        )
        if ns.json:
            out_list = [
                {
                    "c": r.c,
                    "rows": r.rows,
                    "binding": r.binding,
                    "hard_floor_ms": r.hard_floor_ms,
                    "serial_floor_ms": r.serial_floor_ms,
                    "tpot_floor_ms": r.tpot_floor_ms,
                    "interactivity_ceil_tok_s": r.interactivity_ceil_tok_s,
                    "output_tput_ceil_per_chip": r.output_tput_ceil_per_chip,
                    "flag": r.flag,
                }
                for r in c_rows
            ]
            sys.stdout.write(json.dumps(out_list, indent=2, allow_nan=False) + "\n")
        else:
            for r in c_rows:
                sys.stdout.write(
                    f"c={r.c:>4d}  bind={r.binding:<6s}  tpot={r.tpot_floor_ms:>7.2f} ms  "
                    f"out_tput/chip={r.output_tput_ceil_per_chip:>8.2f}\n"
                )
        return 0

    lat_s = (ns.collective_latency_us or 0.0) * 1e-6
    if ns.spec_window > 0:
        sf = spec_cycle_floor(
            geom,
            chip,
            par,
            ns.batch,
            ns.context,
            ns.spec_window,
            dt,
            collective_latency_s=lat_s,
        )
    else:
        sf = step_floor(
            geom,
            chip,
            par,
            StepShape(batch=ns.batch, rows_per_seq=ns.rows_per_seq, context=ns.context),
            dt,
            collective_latency_s=lat_s,
        )
    exp_payload = {
        "binding": sf.binding,
        "hard_floor_ms": sf.hard_floor_s * 1000.0,
        "serial_floor_ms": sf.serial_floor_s * 1000.0,
        "hbm_ms": sf.hbm_s * 1000.0,
        "matmul_ms": sf.mm_s * 1000.0,
        "chain_ms": sf.chain_s * 1000.0,
        "modules": [
            {
                "name": m.name,
                "hbm_bytes": m.hbm_bytes,
                "fp8_flops": m.fp8_flops,
                "bf16_flops": m.bf16_flops,
                "ici_bytes": m.ici_bytes,
            }
            for m in sf.modules
        ],
    }
    if ns.json:
        sys.stdout.write(json.dumps(exp_payload, indent=2, allow_nan=False) + "\n")
    else:
        sys.stdout.write(
            f"binding={sf.binding}  hard={sf.hard_floor_s * 1000.0:.3f} ms  serial={sf.serial_floor_s * 1000.0:.3f} ms\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
