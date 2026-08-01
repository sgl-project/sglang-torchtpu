"""Join every leg of a run into one results JSON plus a markdown table.

A run scatters its output across four files per leg (bench metrics JSONL, repro
meta, metrics delta, status). Reading a nightly result should not require
knowing that, so this collapses them into:

  summary.json   one object: run provenance + one record per leg, each carrying
                 the numbers, the resolved server config, the feature counters,
                 and its status. This is the artifact CI uploads.
  summary.md     the same as a table, for $GITHUB_STEP_SUMMARY.

Legs that failed or timed out are *included* with their status and warnings --
an absent row is indistinguishable from a row nobody looked at.

    summarize.py --run-root ./results/nightly/<ts> \
        --json-out summary.json --md-out summary.md

Exit code 1 if no leg produced metrics (an empty result is a failed run, not an
empty report).

FORMAT CONTRACT (what a frontend can rely on)
---------------------------------------------
``legs[].metrics`` is ``sglang.benchmark.serving``'s own result record, keys
unchanged -- the same object that gets appended to ``<tag>.jsonl``. A renderer
written against a plain ``bench_serving --output-file`` result works on it as-is.
Two documented deviations, both additive-safe:

  * the bulk per-request arrays (see ``_DETAIL_KEYS``) are omitted, because they
    are megabytes per leg and are already in ``<tag>.jsonl``. ``metrics_omitted``
    lists them and points at that file.
  * credential-looking values are masked, reusing dump_meta.py's redaction, since
    bench_serving embeds an unredacted ``/server_info`` (which echoes ``api_key``)
    and this file is meant to be published.

Anything this script computes lives in ``legs[].derived`` and is never mixed into
``metrics``. The rest of a leg record (``status``, ``config``, ``server_info``,
``feature_counters``, ``feature_derived``, ``warnings``) is information
bench_serving does not produce at all.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from typing import Dict, List, Optional

# 3: legs[].metrics is now bench_serving's record verbatim; computed fields moved
#    to legs[].derived (2 reshaped and renamed keys inside metrics).
SCHEMA_VERSION = 3

# Leg tag: concurrency x number of long-context sessions.
_LEG_RE = re.compile(r"^c(?P<bs>\d+)-long(?P<long>\d+)$")

# Per-request arrays that `--output-details` adds. One float (or full generated
# text) per request per token; for a 2400-request leg that is megabytes, and it is
# already in <tag>.jsonl. Dropped from the merged index, never from the leg file.
_DETAIL_KEYS = (
    "ttfts",
    "itls",
    "input_lens",
    "output_lens",
    "generated_texts",
    "errors",
)


def _load_redactor():
    """dump_meta.py owns the redaction rules; load it rather than re-implement so
    both records mask the same things.

    Loaded by explicit path, NOT via sys.path: benchmark/tpu/ contains a `profile/`
    directory, so putting it on sys.path[0] would shadow the stdlib `profile`
    module for the whole process.

    A failure here is fatal on purpose. bench_serving embeds an UNREDACTED
    /server_info (it echoes api_key / admin_api_key) into every leg record, and
    summary.json is uploaded as a CI artifact -- so a silent no-op fallback would
    publish credentials the moment this import broke during a refactor.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dump_meta.py"
    )
    spec = importlib.util.spec_from_file_location("_tpu_dump_meta", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the redaction rules from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._redact_secrets


_redact_secrets = _load_redactor()


def _read_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_last_jsonl(path: str) -> Optional[Dict]:
    """bench_serving appends, so the last line is this leg's run."""
    record = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return None
    return record


def _read_status(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "missing"


def _git_head(repo_root: str) -> Dict[str, Optional[str]]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", repo_root, *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _collect_legs(run_root: str, mode: str) -> List[Dict]:
    """One record per leg found under <run_root>/<mode>/."""
    mode_dir = os.path.join(run_root, mode)
    if not os.path.isdir(mode_dir):
        return []

    tags = set()
    for pattern in ("*.status", "*.jsonl"):
        for path in glob.glob(os.path.join(mode_dir, pattern)):
            tag = os.path.basename(path).split(".")[0]
            if _LEG_RE.match(tag):
                tags.add(tag)

    def sort_key(tag: str):
        groups = _LEG_RE.match(tag).groupdict()
        return (int(groups["bs"]), int(groups["long"]))

    legs: List[Dict] = []
    for tag in sorted(tags, key=sort_key):
        groups = _LEG_RE.match(tag).groupdict()
        base = os.path.join(mode_dir, tag)
        metrics = _read_last_jsonl(f"{base}.jsonl")
        meta = _read_json(f"{base}.meta.json") or {}
        delta = _read_json(f"{base}.metrics.delta.json") or {}
        leg: Dict = {
            "tag": tag,
            "mode": mode,
            "concurrency": int(groups["bs"]),
            "num_long_sessions": int(groups["long"]),
            "status": _read_status(f"{base}.status"),
            "config": meta.get("config", {}),
            "server_info": meta.get("server_info", {}),
            "feature_counters": delta.get("counters", {}),
            "feature_derived": delta.get("derived", {}),
            "warnings": delta.get("warnings", []),
            "log": f"{mode}/{tag}.bench.log",
        }
        if metrics:
            # bench_serving's record, keys unchanged -- see the FORMAT CONTRACT in
            # the module docstring. Only the bulk per-request arrays are dropped
            # and credential-looking values masked.
            leg["metrics"] = _redact_secrets(
                {k: v for k, v in metrics.items() if k not in _DETAIL_KEYS}
            )
            leg["metrics_omitted"] = {
                "keys": [k for k in _DETAIL_KEYS if k in metrics],
                "reason": "per-request arrays; megabytes per leg",
                "available_in": f"{mode}/{tag}.jsonl",
            }
            leg["derived"] = _derive(metrics, leg["feature_counters"])
        else:
            leg["metrics"] = None
            leg["metrics_omitted"] = None
            leg["derived"] = None
        legs.append(leg)
    return legs


def _derive(metrics: Dict, counters: Dict) -> Dict:
    """Values this script computes. Kept strictly out of `metrics` so that stays
    byte-compatible with a plain bench_serving result."""
    derived: Dict = {
        # `completed` counts ROUNDS, not sessions: bench_serving flattens a
        # multi-turn conversation into one output per round. Aliased, not renamed
        # in place, because a generic renderer will show `completed`.
        "completed_rounds": metrics.get("completed"),
        # bench_serving cannot account input tokens for a multi-turn replay (it
        # passes input_requests=None to calculate_metrics), so metrics'
        # total_input_tokens / input_throughput / total_throughput are 0. They are
        # left at 0 rather than patched, and corrected here instead.
        "input_tokens_unavailable_in_metrics": True,
        # cache_report.total_prompt_tokens is rounds x the FIRST round's estimate
        # (wrap_multi_turn_request_func reuses one prompt_len for every round), so
        # it is not a prompt volume. Its device/host/storage split is meaningful.
        "cache_report_total_prompt_tokens_unreliable": bool(
            metrics.get("cache_report")
        ),
    }
    prompt_tokens = counters.get("sglang:prompt_tokens_total")
    duration = metrics.get("duration")
    if prompt_tokens and duration:
        derived["prompt_tokens"] = prompt_tokens
        derived["prompt_throughput"] = prompt_tokens / duration
        derived["total_throughput"] = (
            prompt_tokens + (metrics.get("total_output_tokens") or 0)
        ) / duration
        derived["input_source"] = "server:prompt_tokens_total"
    return derived


def _fmt(value, spec: str = ".1f", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _markdown(summary: Dict) -> str:
    lines: List[str] = []
    run = summary["run"]
    lines.append("## Qwen3-Coder-480B FP8 — TPU agentic benchmark")
    lines.append("")
    lines.append(
        f"- model: `{run['model']}`  profile: `{run['profile']}`  "
        f"mode: `{run['mode']}`"
    )
    lines.append(
        f"- commit: `{(run['git'].get('commit') or 'unknown')[:12]}`"
        f"{' (dirty)' if run['git'].get('dirty') else ''}  "
        f"host: `{run['hostname']}`  finished: {run['finished_at']}"
    )
    features = run["features"]
    lines.append(
        f"- features: prefix caching **{features['prefix_caching']}**, "
        f"KV offload **{features['kv_offload']}**, "
        f"chunked prefill **{features['chunked_prefill_per_rank']}/rank**, "
        f"server context length **{features['server_context_len']}**"
    )
    lines.append("")

    shape = run["workload"]
    lines.append(
        f"- workload: multi-turn agentic, {shape['turn_input_tokens']} in / "
        f"{shape['turn_output_tokens']} out per round, accumulating to "
        f"{shape['target_context']}; the 'long reqs' column is how many sessions "
        f"accumulate to {shape['long_target_context']} instead."
    )
    if shape.get("long_ramp_is_time_boxed"):
        lines.append(
            f"- :warning: the 1M sessions ramped at "
            f"{shape['long_ramp_tokens']} tok/round instead of "
            f"{shape['turn_input_tokens']} (time-boxed run)."
        )
    lines.append("")

    header = (
        "| mode | conc | long reqs | status | reqs | out tok/s | "
        "prompt tok/s | TTFT p50/p99 ms | ITL p50/p99 ms | prefix hit | "
        "KV offload w/r (tok) |"
    )
    sep = "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    lines += [header, sep]
    for leg in summary["legs"]:
        metrics = leg.get("metrics") or {}  # bench_serving's keys
        derived = leg.get("derived") or {}  # ours
        features = leg.get("feature_derived") or {}
        hit = features.get("prefix_cache_hit_rate")
        written = features.get("kv_offload_written_tokens")
        reloaded = features.get("kv_offload_reloaded_tokens")
        lines.append(
            f"| {leg['mode']} | {leg['concurrency']} | {leg['num_long_sessions']} "
            f"| {leg['status']} "
            f"| {_fmt(metrics.get('completed'), '.0f')} "
            f"| {_fmt(metrics.get('output_throughput'))} "
            f"| {_fmt(derived.get('prompt_throughput'))} "
            f"| {_fmt(metrics.get('median_ttft_ms'), '.0f')}/"
            f"{_fmt(metrics.get('p99_ttft_ms'), '.0f')} "
            f"| {_fmt(metrics.get('median_itl_ms'), '.1f')}/"
            f"{_fmt(metrics.get('p99_itl_ms'), '.1f')} "
            f"| {(f'{hit:.1%}' if hit is not None else '-')} "
            f"| {_fmt(written, '.0f')}/{_fmt(reloaded, '.0f')} |"
        )
    lines.append("")
    lines.append(
        "_`reqs` = requests (agentic rounds) completed. `prompt tok/s` comes from "
        "the server's `prompt_tokens_total` counter: bench_serving cannot account "
        "input tokens for a multi-turn replay._"
    )
    lines.append("")

    flagged = [
        leg
        for leg in summary["legs"]
        if leg["warnings"] or leg["status"] not in ("ok", "skipped:long>bs")
    ]
    if flagged:
        lines.append("### Legs needing attention")
        lines.append("")
        for leg in flagged:
            lines.append(
                f"- **{leg['mode']}/{leg['tag']}** — status "
                f"`{leg['status']}` (log: `{leg['log']}`)"
            )
            for warning in leg["warnings"]:
                lines.append(f"  - {warning}")
        lines.append("")

    totals = summary["totals"]
    lines.append(
        f"_{totals['legs_ok']}/{totals['legs_total']} legs ok, "
        f"{totals['legs_failed']} failed, {totals['legs_timeout']} timed out, "
        f"{totals['legs_skipped']} skipped._"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    ap.add_argument("--profile", default=os.environ.get("PROFILE", "unknown"))
    ap.add_argument("--mode", default=os.environ.get("MODE", "unknown"))
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    legs: List[Dict] = []
    for mode in ("colocated", "pd"):
        legs.extend(_collect_legs(run_root, mode))

    # Run-level facts come from any leg's meta (they are identical across legs
    # by construction) and fall back to the environment for a run whose every
    # leg died before writing one.
    first_config: Dict = {}
    for leg in legs:
        if leg["config"]:
            first_config = leg["config"]
            break

    def cfg(key: str, default=None):
        return first_config.get(key, os.environ.get(key.upper(), default))

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "benchmark": "qwen3-coder-480b-tpu-agentic",
            "profile": args.profile,
            "mode": args.mode,
            "model": cfg("model", os.environ.get("MODEL", "unknown")),
            "hostname": platform.node(),
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "run_root": run_root,
            "git": _git_head(repo_root),
            "features": {
                "prefix_caching": (
                    "on"
                    if str(cfg("prefix_cache", "1")) in ("1", "True", "true")
                    else "off"
                ),
                "kv_offload": (
                    "on"
                    if str(cfg("kv_offload", "1")) in ("1", "True", "true")
                    else "off"
                ),
                "hicache_ratio": cfg("hicache_ratio"),
                "chunked_prefill_per_rank": cfg("chunked_prefill_per_rank"),
                "server_context_len": cfg("server_context_len", 1048576),
                "quantization": "fp8",
                "kv_cache_dtype": "fp8_e4m3",
            },
            "workload": {
                "turn_input_tokens": cfg("turn_input_tokens", 4096),
                "turn_output_tokens": cfg("turn_output_tokens", 1024),
                "target_context": cfg("target_context", 16384),
                "long_target_context": cfg("long_target_context", 262144),
                "long_ramp_tokens": cfg("long_ramp_tokens"),
                "long_ramp_is_time_boxed": cfg("long_ramp_tokens")
                not in (None, cfg("turn_input_tokens", 4096)),
            },
        },
        "totals": {
            "legs_total": len(legs),
            "legs_ok": sum(1 for leg in legs if leg["status"] == "ok"),
            "legs_failed": sum(1 for leg in legs if leg["status"].startswith("failed")),
            "legs_timeout": sum(1 for leg in legs if leg["status"] == "timeout"),
            "legs_skipped": sum(
                1 for leg in legs if leg["status"].startswith("skipped")
            ),
            "legs_with_metrics": sum(1 for leg in legs if leg.get("metrics")),
            "warnings": sum(len(leg["warnings"]) for leg in legs),
        },
        "legs": legs,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    with open(args.md_out, "w") as f:
        f.write(_markdown(summary))

    print(
        f"[summarize] {summary['totals']['legs_with_metrics']}/{len(legs)} legs "
        f"with metrics -> {args.json_out}"
    )
    if summary["totals"]["legs_with_metrics"] == 0:
        print("[summarize] ERROR: no leg produced metrics", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
