"""Headline AgentX metrics from one AIPerf run's artifacts (stdlib only).

    python3 benchmark/tpu/recipe/configs/muse_spark/agentx_summary.py \\
        RUN/c22/aiperf_artifacts --chips 4 --concurrency 22

Definitions, as the InferenceX AgentX aggregation computes them, over the
profiling-phase requests that completed without error (warmup and failed requests
excluded):

  throughput per chip   (sum of prompt + completion tokens) / window / chips,
                        where the window runs from the first request's start to
                        the last one's end and tokens are the server's usage counts
  E2E latency           request start to last token (request_latency)
  TTFT                  request start to first token
  interactivity         1 / ITL, ITL = (E2E - TTFT) / (completion tokens - 1) per
                        request; "p90 interactivity" is 1 / (p90 ITL), the slow tail
"""

import argparse
import json
import math
from pathlib import Path

SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def percentile(values, p):
    """Linear interpolation between closest ranks."""
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p / 100
    f = int(k)
    if f + 1 >= len(values):
        return values[f]
    return values[f] + (k - f) * (values[f + 1] - values[f])


def metric(record, tag, unit=None):
    entry = record.get("metrics", {}).get(tag)
    value = entry.get("value") if isinstance(entry, dict) else entry
    if value is None:
        return None
    if unit == "s":
        # Unitless exports are milliseconds, as AIPerf's record export displays them.
        scale = entry.get("unit", "ms") if isinstance(entry, dict) else "ms"
        return float(value) * SECONDS[scale]
    return value


LATENCIES = ("request_latency", "time_to_first_token", "inter_token_latency")


def seconds(records, tag):
    """One latency per request, in seconds; missing, zero and non-finite values dropped."""
    values = (metric(r, tag, "s") for r in records)
    return [v for v in values if v is not None and math.isfinite(v) and v > 0]


def artifact_dir(path):
    path = Path(path)
    if (path / "profile_export.jsonl").is_file():
        return path
    children = sorted(
        c for c in path.iterdir() if (c / "profile_export.jsonl").is_file()
    )
    if not children:
        raise SystemExit(f"no profile_export.jsonl under {path}")
    return children[0]


def summarize(path, chips):
    path = artifact_dir(path)
    records, failed = [], 0
    with open(path / "profile_export.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            phase = record.get("metadata", {}).get("benchmark_phase")
            if phase is not None and phase != "profiling":
                continue
            if record.get("error"):
                failed += 1
                continue
            records.append(record)
    if not records:
        raise SystemExit(f"no successful profiling requests in {path}")
    starts = [int(r["metadata"]["request_start_ns"]) for r in records]
    ends = [int(r["metadata"]["request_end_ns"]) for r in records]
    window = (max(ends) - min(starts)) / 1e9
    prompt = sum(int(metric(r, "input_sequence_length") or 0) for r in records)
    completion = sum(int(metric(r, "output_sequence_length") or 0) for r in records)
    e2e, ttft, itl = (seconds(records, tag) for tag in LATENCIES)
    summary = {
        "requests": len(records),
        "failed_requests": failed,
        "failed_fraction": failed / (failed + len(records)),
        "window_s": window,
        "tok_per_s_per_chip": (prompt + completion) / window / chips,
        "output_tok_per_s_per_chip": completion / window / chips,
        "p50_e2e_s": percentile(e2e, 50),
        "p90_e2e_s": percentile(e2e, 90),
        "p90_ttft_s": percentile(ttft, 90),
        "p50_interactivity": 1 / percentile(itl, 50) if itl else None,
        "p90_interactivity": 1 / percentile(itl, 90) if itl else None,
    }
    aggregate = path / "profile_export_aiperf.json"
    if aggregate.is_file():
        meta = json.loads(aggregate.read_text()).get("metadata") or {}
        summary["submission_valid"] = meta.get("submission_valid")
        if meta.get("submission_invalid_reasons"):
            summary["submission_invalid_reasons"] = meta["submission_invalid_reasons"]
    summary["server_metrics_captured"] = has_engine_metrics(
        path / "server_metrics_export.json"
    )
    return summary


def has_engine_metrics(path, prefix=b'"sglang:'):
    """A fixed-string scan: the export can be several GiB."""
    if not path.is_file():
        return False
    tail = b""
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            if prefix in tail + chunk:
                return True
            tail = chunk[-len(prefix) :]
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifacts", help="AIPerf --output-artifact-dir")
    parser.add_argument(
        "--chips", type=int, default=4, help="TPU chips serving (tp8 = 4)"
    )
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args(argv)
    summary = summarize(args.artifacts, args.chips)
    if args.concurrency is not None:
        summary = {"concurrency": args.concurrency, **summary}
    (Path(args.artifacts).resolve().parent / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        f"conc {summary.get('concurrency', '?')}: p90 E2E {summary['p90_e2e_s']:.1f} s, "
        f"{summary['tok_per_s_per_chip']:,.0f} tok/s/chip, "
        f"p90 interactivity {summary['p90_interactivity'] or float('nan'):.1f} tok/s, "
        f"p90 TTFT {summary['p90_ttft_s']:.1f} s, "
        f"failed {summary['failed_requests']}/{summary['requests'] + summary['failed_requests']}, "
        f"submission_valid {summary.get('submission_valid')}"
    )


if __name__ == "__main__":
    main()
