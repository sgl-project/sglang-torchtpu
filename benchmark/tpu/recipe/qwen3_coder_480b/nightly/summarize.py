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
# 4: additive -- legs[].server_histograms, legs[].decode (P/D decode engine),
#    and legs[].derived.{by_context,long_session_rounds,per_request_layout}.
SCHEMA_VERSION = 4

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
        decode_delta = _read_json(f"{base}.decode.metrics.delta.json")
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
            # Per-leg server histogram deltas (see metrics_probe._HISTOGRAMS):
            # {family: {series: {count, sum, mean, p50, p90, p99, ...}}}. In P/D
            # this is the prefill engine, which owns the KV transfer histograms.
            "server_histograms": delta.get("histograms", {}),
            "warnings": delta.get("warnings", []),
            "log": f"{mode}/{tag}.bench.log",
        }
        if decode_delta is not None:
            # P/D decode engine, kept apart so its warnings do not change the
            # meaning of the leg's own `warnings`.
            leg["decode"] = {
                "counters": decode_delta.get("counters", {}),
                "derived": decode_delta.get("derived", {}),
                "server_histograms": decode_delta.get("histograms", {}),
                "warnings": decode_delta.get("warnings", []),
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
            leg["derived"] = _derive(metrics, leg["feature_counters"], leg["config"])
        else:
            leg["metrics"] = None
            leg["metrics_omitted"] = None
            leg["derived"] = None
        legs.append(leg)
    return legs


def _derive(metrics: Dict, counters: Dict, config: Optional[Dict] = None) -> Dict:
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

    rows = _request_rows(metrics)
    if rows:
        layout = _session_layout(len(rows), config or {})
        derived["per_request_layout"] = layout["status"]
        if _input_lens_vary(rows, layout):
            derived["by_context"] = _by_context(rows)
        else:
            derived["by_context_unavailable"] = (
                "input_lens repeat across a session's rounds, so they are the "
                "dataset row's estimate, not the server's per-round prompt length"
            )
        if layout["status"] == "ok" and layout["num_long"] > 0:
            derived["long_session_rounds"] = _long_session_rounds(rows, layout)
    return derived


# Upper bounds (prompt tokens) of the per-context buckets.
_CONTEXT_EDGES = (8192, 16384, 32768, 65536, 98304, 131072, 262144, 524288, 1048576)


def _percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy's default), q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _request_rows(metrics: Dict) -> List[Dict]:
    """bench_serving's parallel per-request arrays as one dict per request.

    Present only with --output-details. ``input_lens`` is the server-reported
    prompt length of that request (bench_serving's _record_server_prompt_len),
    i.e. the full accumulated context of that round. Empty if the arrays are
    missing or disagree in length.
    """
    ttfts = metrics.get("ttfts")
    if not isinstance(ttfts, list) or not ttfts:
        return []
    n = len(ttfts)
    input_lens = metrics.get("input_lens") or []
    output_lens = metrics.get("output_lens") or []
    itls = metrics.get("itls") or []
    errors = metrics.get("errors") or [""] * n
    cached = metrics.get("cached_tokens")
    if any(len(a) != n for a in (input_lens, output_lens, itls, errors)):
        return []
    if cached is not None and len(cached) != n:
        cached = None
    rows = []
    for i in range(n):
        itl = [x * 1000.0 for x in (itls[i] or []) if isinstance(x, (int, float))]
        rows.append(
            {
                "ok": not errors[i],
                "input": input_lens[i] or 0,
                "output": output_lens[i] or 0,
                "ttft_ms": (ttfts[i] or 0.0) * 1000.0,
                "itl_ms": itl,
                "cached": None if cached is None else (cached[i] or 0),
            }
        )
    return rows


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _session_layout(num_rows: int, config: Dict) -> Dict:
    """Map flat request index -> (session, round).

    bench_serving replays sessions in trace order (the long sessions first)
    and emits one output per round, failed rounds included. The round counts
    come from the leg's meta (bench_leg.sh mirrors the generator's arithmetic);
    if they do not account for exactly ``num_rows`` outputs, the mapping is
    reported unavailable rather than guessed.
    """
    num_long = _as_int(config.get("num_long_sessions"))
    num_sessions = _as_int(config.get("num_sessions"))
    long_rounds = _as_int(config.get("rounds_per_long_session"))
    normal_rounds = _as_int(config.get("rounds_per_normal_session"))
    layout = {
        "num_long": num_long or 0,
        "long_rounds": long_rounds or 0,
        "normal_rounds": normal_rounds or 0,
        "num_sessions": num_sessions or 0,
    }
    if None in (num_long, num_sessions, long_rounds, normal_rounds):
        layout["status"] = "unavailable:no-round-counts-in-meta"
        return layout
    expected = num_long * long_rounds + (num_sessions - num_long) * normal_rounds
    if expected != num_rows:
        layout["status"] = f"unavailable:expected-{expected}-outputs-got-{num_rows}"
        return layout
    layout["status"] = "ok"
    return layout


def _input_lens_vary(rows: List[Dict], layout: Dict) -> bool:
    """False if input_lens look like the dataset row's single estimate.

    Every round re-sends the history, so a session's server-reported prompt
    length strictly grows. If the first multi-round session reports one value
    for all its rounds, bench_serving fell back to the row estimate (the
    server sent no usage block) and bucketing by it would be wrong. Unknown
    layouts pass: there is nothing to check against.
    """
    if layout.get("status") != "ok":
        return True
    if layout["num_long"] > 0 and layout["long_rounds"] > 1:
        first = rows[: layout["long_rounds"]]
    elif layout["normal_rounds"] > 1:
        start = layout["num_long"] * layout["long_rounds"]
        first = rows[start : start + layout["normal_rounds"]]
    else:
        return True
    lens = [r["input"] for r in first if r["ok"]]
    return len(set(lens)) > 1 or len(lens) < 2


def _by_context(rows: List[Dict]) -> List[Dict]:
    """Latency and cache reuse per prompt-length bucket (successful requests).

    TTFT here is the client's: in P/D it includes prefill queueing, prefill,
    the P->D KV transfer and decode admission. The server_histograms split it.
    """
    buckets: Dict[int, List[Dict]] = {}
    for row in rows:
        if not row["ok"] or row["input"] <= 0:
            continue
        edge = next((e for e in _CONTEXT_EDGES if row["input"] <= e), 0)
        buckets.setdefault(edge, []).append(row)
    out = []
    for edge in sorted(buckets, key=lambda e: e or float("inf")):
        group = buckets[edge]
        ttft = [r["ttft_ms"] for r in group]
        itl = [x for r in group for x in r["itl_ms"]]
        record = {
            "ctx_le": edge or None,  # None: above the largest edge
            "requests": len(group),
            "input_tokens_mean": _mean([r["input"] for r in group]),
            "ttft_ms_p50": _percentile(ttft, 50),
            "ttft_ms_p90": _percentile(ttft, 90),
            "ttft_ms_p99": _percentile(ttft, 99),
            "ttft_ms_max": max(ttft),
            "itl_ms_p50": _percentile(itl, 50),
            "itl_ms_p99": _percentile(itl, 99),
            "itl_ms_mean": _mean(itl),
        }
        if all(r["cached"] is not None for r in group):
            total_in = sum(r["input"] for r in group)
            total_cached = sum(r["cached"] for r in group)
            record["cached_fraction"] = total_cached / total_in if total_in else None
            record["uncached_tokens_mean"] = (total_in - total_cached) / len(group)
        out.append(record)
    return out


def _long_session_rounds(rows: List[Dict], layout: Dict) -> List[Dict]:
    """Turn-by-turn view of the long sessions: what each round cost as the
    context grows. Aggregated across the leg's long sessions (they share the
    same shape); failed rounds are counted, not averaged."""
    out = []
    num_long, rounds = layout["num_long"], layout["long_rounds"]
    for r in range(rounds):
        group = [rows[s * rounds + r] for s in range(num_long)]
        ok = [row for row in group if row["ok"]]
        record: Dict = {
            "round": r + 1,
            "sessions": len(group),
            "failed": len(group) - len(ok),
        }
        if ok:
            ttft = [row["ttft_ms"] for row in ok]
            itl_means = [_mean(row["itl_ms"]) for row in ok if row["itl_ms"]]
            record.update(
                {
                    "input_tokens_mean": _mean([row["input"] for row in ok]),
                    "output_tokens_mean": _mean([row["output"] for row in ok]),
                    "ttft_ms_mean": _mean(ttft),
                    "ttft_ms_max": max(ttft),
                    "itl_ms_mean": _mean(itl_means),
                }
            )
            if all(row["cached"] is not None for row in ok):
                record["cached_tokens_mean"] = _mean([row["cached"] for row in ok])
                record["uncached_tokens_mean"] = _mean(
                    [row["input"] - row["cached"] for row in ok]
                )
        out.append(record)
    return out


def _fmt(value, spec: str = ".1f", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _hist(histograms: Dict, family: str, series: str = "all") -> Dict:
    return ((histograms or {}).get(family) or {}).get(series) or {}


# The prefill's wait for its KV transfer AFTER its forward: the stage starts
# when the request enters the transfer queue and (on TPU raiden) ends at the
# transfer's finish. This is the transfer time a request is actually exposed
# to, unlike sglang:kv_transfer_latency_ms (see _markdown_server_breakdown).
_EXPOSED_TRANSFER_STAGE = "prefill_transfer_kv_cache"

# (column header, engine, stage) for the server-side stage table. Prefill
# stages are on the prefill engine (or a colocated server); decode_* stages
# exist only on a P/D decode engine.
_STAGE_COLUMNS = (
    ("prefill wait", "prefill", "prefill_waiting"),
    ("prefill fwd", "prefill", "prefill_forward"),
    ("decode bootstrap", "decode", "decode_bootstrap"),
    ("decode wait", "decode", "decode_waiting"),
    ("decode xfer", "decode", "decode_transferred"),
)


def _markdown_server_breakdown(legs: List[Dict]) -> List[str]:
    """Where a request's time went, server side: KV transfer distribution and
    mean per-stage latency. Rendered only when some leg has the histograms.

    sglang:kv_transfer_latency_ms is a transfer WINDOW: it runs from the
    first KV send to the transfer's finish. With cached-prefix early send (on
    by default) the first send goes out before the prefill forward, so on a
    prefix hit the window includes the forward and tracks prefill time, not
    the link. The exposed column is the part left after the forward."""
    stage_family = "sglang:per_stage_req_latency_seconds"
    rows = []
    any_transfer = False
    for leg in legs:
        prefill_h = leg.get("server_histograms") or {}
        decode_h = (leg.get("decode") or {}).get("server_histograms") or {}
        if not prefill_h and not decode_h:
            continue
        latency = _hist(prefill_h, "sglang:kv_transfer_latency_ms")
        size = _hist(prefill_h, "sglang:kv_transfer_total_mb")
        speed = _hist(prefill_h, "sglang:kv_transfer_speed_gb_s")
        any_transfer = any_transfer or bool(latency)
        exposed_s = _hist(prefill_h, stage_family, _EXPOSED_TRANSFER_STAGE).get("mean")
        stage_cells = []
        for _, engine, stage in _STAGE_COLUMNS:
            source = decode_h if engine == "decode" else prefill_h
            mean_s = _hist(source, stage_family, stage).get("mean")
            stage_cells.append(_fmt(None if mean_s is None else mean_s * 1000, ".0f"))
        queue = _hist(prefill_h, "sglang:queue_time_seconds").get("mean")
        # Each TP rank observes its own shard of every transfer.
        ranks = max(1, int(latency.get("label_sets") or 1))
        transfers = latency.get("count")
        request_gb = size.get("mean")
        rows.append(
            f"| {leg['mode']} | {leg['concurrency']} | {leg['num_long_sessions']} "
            f"| {_fmt(None if transfers is None else transfers / ranks, '.0f')} "
            f"| {_fmt(latency.get('mean'), '.0f')}/{_fmt(latency.get('p50'), '.0f')}"
            f"/{_fmt(latency.get('p99'), '.0f')} "
            f"| {_fmt(None if exposed_s is None else exposed_s * 1000, '.0f')} "
            f"| {_fmt(None if request_gb is None else request_gb * ranks / 1024, '.2f')} "
            f"| {_fmt(speed.get('mean'), '.2f')} "
            f"| {_fmt(None if queue is None else queue * 1000, '.0f')} "
            f"| " + " | ".join(stage_cells) + " |"
        )
    if not rows:
        return []
    out = [
        "### Server-side breakdown",
        "",
        "| mode | conc | long reqs | KV xfers | xfer window ms mean/p50/p99 "
        "| exposed xfer ms | xfer GB/req mean | window GB/s/rank mean | queue ms | "
        + " | ".join(f"{name} ms" for name, _, _ in _STAGE_COLUMNS)
        + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(_STAGE_COLUMNS),
        *rows,
        "",
        "_Per-leg deltas of the servers' Prometheus histograms; p50/p99 are "
        "bucket-interpolated. KV transfer and prefill stages come from the "
        "prefill engine, decode stages from the decode engine; stage columns "
        "are means. Each TP rank sends its own KV shard: xfer ms and GB/s are "
        "per rank shard, KV xfers and GB/req are per request._",
        "",
        "_xfer window = first KV send to transfer done (`kv_transfer_latency_ms`). "
        "The cached prefix is sent before the prefill forward, so on prefix hits "
        "the window includes the forward, and window GB/s understates the link. "
        "exposed xfer = the prefill's wait for the transfer after its forward "
        "(`prefill_transfer_kv_cache` stage): the transfer time a request "
        "actually pays._",
        "",
    ]
    if not any_transfer:
        out.insert(-1, "_No KV transfers were observed (not a P/D run)._")
    return out


def _markdown_by_context(legs: List[Dict]) -> List[str]:
    """TTFT/ITL by prompt length for the legs that reach long context."""
    blocks = []
    for leg in legs:
        if leg["num_long_sessions"] <= 0:
            continue
        by_ctx = (leg.get("derived") or {}).get("by_context")
        if not by_ctx:
            continue
        blocks.append(f"<details><summary>{leg['mode']}/{leg['tag']}</summary>")
        blocks.append("")
        blocks.append(
            "| ctx ≤ | reqs | input mean | cached | uncached mean | "
            "TTFT p50/p90/p99 ms | ITL p50/p99 ms |"
        )
        blocks.append("|---:|---:|---:|---:|---:|---:|---:|")
        for b in by_ctx:
            cached = b.get("cached_fraction")
            blocks.append(
                f"| {b['ctx_le'] if b['ctx_le'] else 'more'} | {b['requests']} "
                f"| {_fmt(b['input_tokens_mean'], '.0f')} "
                f"| {(f'{cached:.1%}' if cached is not None else '-')} "
                f"| {_fmt(b.get('uncached_tokens_mean'), '.0f')} "
                f"| {_fmt(b['ttft_ms_p50'], '.0f')}/{_fmt(b['ttft_ms_p90'], '.0f')}"
                f"/{_fmt(b['ttft_ms_p99'], '.0f')} "
                f"| {_fmt(b['itl_ms_p50'], '.1f')}/{_fmt(b['itl_ms_p99'], '.1f')} |"
            )
        blocks.append("")
        blocks.append("</details>")
        blocks.append("")
    if not blocks:
        return []
    return [
        "### Latency by context length",
        "",
        "_Client-side, successful requests bucketed by their server-reported "
        "prompt length._",
        "",
        *blocks,
    ]


def _markdown_long_rounds(legs: List[Dict]) -> List[str]:
    """Turn-by-turn cost of the long sessions as their context grows."""
    blocks = []
    for leg in legs:
        rounds = (leg.get("derived") or {}).get("long_session_rounds")
        if not rounds:
            continue
        blocks.append(f"<details><summary>{leg['mode']}/{leg['tag']}</summary>")
        blocks.append("")
        blocks.append(
            "| round | ctx in (mean) | cached (mean) | uncached (mean) | "
            "TTFT mean/max ms | ITL mean ms | failed |"
        )
        blocks.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in rounds:
            blocks.append(
                f"| {r['round']} | {_fmt(r.get('input_tokens_mean'), '.0f')} "
                f"| {_fmt(r.get('cached_tokens_mean'), '.0f')} "
                f"| {_fmt(r.get('uncached_tokens_mean'), '.0f')} "
                f"| {_fmt(r.get('ttft_ms_mean'), '.0f')}/"
                f"{_fmt(r.get('ttft_ms_max'), '.0f')} "
                f"| {_fmt(r.get('itl_ms_mean'), '.1f')} | {r['failed']} |"
            )
        blocks.append("")
        blocks.append("</details>")
        blocks.append("")
    if not blocks:
        return []
    return [
        "### Long sessions, turn by turn",
        "",
        "_Each row averages the leg's long sessions at that round; they run "
        "concurrently with the normal sessions, so this is the cost under load._",
        "",
        *blocks,
    ]


def _markdown(summary: Dict) -> str:
    lines: List[str] = []
    run = summary["run"]
    lines.append(f"## {run.get('title') or 'TPU serving benchmark'}")
    lines.append("")
    lines.append(
        f"- model: `{run['model']}`  profile: `{run['profile']}`  mode: `{run['mode']}`"
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
        f"speculative decoding **{features['speculative_decoding']}**, "
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
            f"- :warning: the long sessions ramped at "
            f"{shape['long_ramp_tokens']} tok/round instead of "
            f"{shape['turn_input_tokens']} (time-boxed run)."
        )
    lines.append("")

    header = (
        "| mode | conc | long reqs | status | reqs | out tok/s | "
        "prompt tok/s | TTFT p50/p99 ms | ITL p50/p99 ms | prefix hit | "
        "spec accept min | spec verify | KV offload w/r (tok) |"
    )
    sep = "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines += [header, sep]
    for leg in summary["legs"]:
        metrics = leg.get("metrics") or {}  # bench_serving's keys
        derived = leg.get("derived") or {}  # ours
        features = leg.get("feature_derived") or {}
        hit = features.get("prefix_cache_hit_rate")
        spec_accept = features.get("min_spec_accept_length")
        spec_verify = features.get("spec_verify_calls")
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
            f"| {_fmt(spec_accept, '.3f')} "
            f"| {_fmt(spec_verify, '.0f')} "
            f"| {_fmt(written, '.0f')}/{_fmt(reloaded, '.0f')} |"
        )
    lines.append("")
    lines.append(
        "_`reqs` = requests (agentic rounds) completed. `prompt tok/s` comes from "
        "the server's `prompt_tokens_total` counter: bench_serving cannot account "
        "input tokens for a multi-turn replay._"
    )
    lines.append("")
    lines += _markdown_server_breakdown(summary["legs"])
    lines += _markdown_by_context(summary["legs"])
    lines += _markdown_long_rounds(summary["legs"])

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
    # Labels only -- so the benchmark matrix (which reuses this summarizer
    # for other models/recipes) does not report as the 480B run.
    ap.add_argument(
        "--benchmark-label",
        default=os.environ.get("BENCH_SUMMARY_LABEL", "qwen3-coder-480b-tpu-agentic"),
    )
    ap.add_argument(
        "--title",
        default=os.environ.get(
            "BENCH_SUMMARY_TITLE", "Qwen3-Coder-480B FP8 — TPU agentic benchmark"
        ),
    )
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    legs: List[Dict] = []
    for mode in ("colocated", "pd"):
        legs.extend(_collect_legs(run_root, mode))

    # Run-level facts come from any leg's meta (they are identical across legs
    # by construction) and fall back to the environment for a run whose every
    # leg died before writing one.
    first_config: Dict = {}
    first_server_info: Dict = {}
    for leg in legs:
        if leg["config"]:
            first_config = leg["config"]
        if leg["server_info"]:
            first_server_info = leg["server_info"]
        if first_config and first_server_info:
            break

    def cfg(key: str, default=None):
        return first_config.get(key, os.environ.get(key.upper(), default))

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "benchmark": args.benchmark_label,
            "title": args.title,
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
                "speculative_decoding": (
                    first_server_info.get("speculative_algorithm") or "off"
                ),
                "speculative_num_steps": first_server_info.get("speculative_num_steps"),
                "speculative_eagle_topk": first_server_info.get(
                    "speculative_eagle_topk"
                ),
                "speculative_num_draft_tokens": first_server_info.get(
                    "speculative_num_draft_tokens"
                ),
                "min_spec_accept_length": cfg("min_spec_accept_length", 1.0),
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
