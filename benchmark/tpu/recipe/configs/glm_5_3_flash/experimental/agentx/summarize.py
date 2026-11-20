"""Summarize one AgentX run directory into the README's table row. Stdlib only.

usage: python3 summarize.py [--chips 4] [--md] RUN_DIR [RUN_DIR ...]

RUN_DIR holds the client output (profile_export_aiperf.json and/or profile_export.jsonl, directly or
under aiperf_artifacts/) and, if present, server-side files: server_metrics_export.csv (written by
`agentx profile --server-metrics`), msample.txt (TP0 /metrics every 30 s), server.log,
metrics_before.txt / metrics_after.txt.

Client numbers, profiling phase only (warmup excluded):
- tok/s/chip = (prompt + completion tokens) / window / chips; prompt includes cached tokens. Window =
  first request start to last request end (profile_export.jsonl), else aiperf benchmark_duration.
- TTFT and e2e (request_latency) percentiles; interactivity = 1000 / inter-token latency (ms) at p50 / p90.
- ideal = aiperf top-level theoretical_prefix_cache_hit (not the warmup_metrics one).
Server hit = (device_hit + host_hit) / all prefill_effective_tokens on TP0; cpu = host_hit / the same.
Its source, first found: aiperf server CSV (profiling phase), msample.txt or server.log clipped to the
window, else metrics_after - metrics_before (includes the warmup, marked "incl. warmup").
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import re

SEC = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
MODES = ("device_hit", "host_hit", "input")
LBL = re.compile(r'(\w+)="([^"]*)"')


def find(d, name):
    for p in (os.path.join(d, name), os.path.join(d, "aiperf_artifacts", name)):
        if os.path.isfile(p):
            return p
    return None


def pct(xs, p):
    """numpy's default (linear) percentile, as aiperf reports."""
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def profiling_records(path):
    ok, failed = [], 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("metadata", {}).get("benchmark_phase", "profiling") != "profiling":
            continue
        if r.get("error"):
            failed += 1
        else:
            ok.append(r)
    return ok, failed


def mval(r, tag):
    e = r.get("metrics", {}).get(tag)
    if e is None or e.get("value") is None:
        return None
    return float(e["value"]) * SEC.get(e.get("unit"), 1.0)


def client(d, chips):
    agg_path, recs_path = (
        find(d, "profile_export_aiperf.json"),
        find(d, "profile_export.jsonl"),
    )
    out, window = {}, None
    if recs_path:
        recs, failed = profiling_records(recs_path)
        st = min(int(r["metadata"]["request_start_ns"]) for r in recs)
        en = max(int(r["metadata"]["request_end_ns"]) for r in recs)
        window = (st / 1e9, en / 1e9)
    if agg_path:
        j = json.load(open(agg_path))
        q = lambda k, p: j[k][p] / 1e3  # ms -> s  # noqa: E731
        dur = window[1] - window[0] if window else j["benchmark_duration"]["avg"]
        toks = j["input_sequence_length"]["sum"] + j["output_sequence_length"]["sum"]
        itl = j["inter_token_latency"]
        out = dict(
            source="profile_export_aiperf.json",
            n=int(j["request_count"]["avg"]),
            failed=sum(int(e.get("count", 0)) for e in j.get("error_summary") or []),
            window_s=dur,
            tok_s_chip=toks / dur / chips,
            ttft=(q("time_to_first_token", "p50"), q("time_to_first_token", "p90")),
            inter=(1e3 / itl["p50"], 1e3 / itl["p90"]),
            e2e=(
                q("request_latency", "p50"),
                q("request_latency", "p90"),
                q("request_latency", "p99"),
            ),
            ideal=j["theoretical_prefix_cache_hit"]["avg"],
            client_hit=(j.get("overall_usage_prompt_cache_read_pct") or {}).get("avg"),
        )
    elif recs_path:
        dur = window[1] - window[0]
        get = lambda tag: [
            v
            for v in (mval(r, tag) for r in recs)
            if v is not None and math.isfinite(v)
        ]  # noqa: E731
        ttft, itl, e2e = (
            get("time_to_first_token"),
            get("inter_token_latency"),
            get("request_latency"),
        )
        prompt = sum(mval(r, "input_sequence_length") for r in recs)
        cached = sum(mval(r, "usage_prompt_cache_read_tokens") or 0 for r in recs)
        out = dict(
            source="profile_export.jsonl",
            n=len(recs),
            failed=failed,
            window_s=dur,
            tok_s_chip=(prompt + sum(mval(r, "output_sequence_length") for r in recs))
            / dur
            / chips,
            ttft=(pct(ttft, 50), pct(ttft, 90)),
            inter=(1 / pct(itl, 50), 1 / pct(itl, 90)),
            e2e=(pct(e2e, 50), pct(e2e, 90), pct(e2e, 99)) if e2e else None,
            ideal=None,
            client_hit=100 * cached / prompt,
        )
    else:
        raise SystemExit(f"{d}: no profile_export_aiperf.json or profile_export.jsonl")
    if window is None and agg_path:
        t = lambda s: (
            dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()
        )  # noqa: E731
        window = (t(j["start_time"]), t(j["end_time"]))
    return out, window


def tp0(labels):
    return labels.get("tp_rank") in (None, "", "0")


def from_csv(path):
    tot, header = {}, None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "Endpoint":
                header = row
                continue
            d = dict(zip(header or [], row))
            if (
                d.get("Metric") == "sglang:prefill_effective_tokens"
                and "total" in d
                and tp0(d)
            ):
                tot[d["mode"]] = tot.get(d["mode"], 0.0) + float(d["total"] or 0)
    return tot


def prom(lines):
    """prefill_effective_tokens_total by mode, TP0 series."""
    out = {}
    for line in lines:
        if not line.startswith("sglang:prefill_effective_tokens_total{"):
            continue
        lbl, _, val = line.rpartition(" ")
        labels = dict(LBL.findall(lbl))
        if tp0(labels):
            out[labels.get("mode")] = out.get(labels.get("mode"), 0.0) + float(val)
    return out


def from_msample(path, t0, t1):
    samples, cur = [], None
    for line in open(path):
        if line.startswith("T "):
            cur = []
            samples.append((int(line.split()[1]), cur))
        elif cur is not None:
            cur.append(line.strip())
    a = b = prom(samples[0][1])
    for ts, lines in samples:
        if ts <= t0:
            a = prom(lines)
        if ts <= t1:
            b = prom(lines)
    return {m: b.get(m, 0.0) - a.get(m, 0.0) for m in MODES}


LOGLINE = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)([^\]]*)\] Prefill batch[^,]*, (.*)$"
)
FIELD = re.compile(r"(#[a-z-]+): ([0-9.]+)")


def from_log(path, t0, t1):
    s = {"#cached-token": 0, "#new-token": 0, "#host-hit-token": 0}
    for line in open(path, errors="replace"):
        m = LOGLINE.match(line) if "Prefill batch" in line else None
        if not m or ("TP" in m.group(2) and "TP0" not in m.group(2)):
            continue
        t = (
            dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
        if math.floor(t0) <= t <= math.ceil(t1):
            for k, v in FIELD.findall(m.group(3)):
                if k in s:
                    s[k] += int(float(v))
    host = s["#host-hit-token"]
    return {
        "device_hit": s["#cached-token"] - host,
        "host_hit": host,
        "input": s["#new-token"],
    }


def server(d, window):
    t0, t1 = window
    tries = [
        ("server CSV", find(d, "server_metrics_export.csv"), lambda p: from_csv(p)),
        ("msample", find(d, "msample.txt"), lambda p: from_msample(p, t0, t1)),
        ("server.log", find(d, "server.log"), lambda p: from_log(p, t0, t1)),
    ]
    for name, path, fn in tries:
        if path:
            tot = fn(path)
            if sum(tot.get(m, 0.0) for m in MODES) > 0:
                return name, tot
    before, after = find(d, "metrics_before.txt"), find(d, "metrics_after.txt")
    if before and after:
        a, b = prom(open(before)), prom(open(after))
        return "metrics delta, incl. warmup", {
            m: b.get(m, 0.0) - a.get(m, 0.0) for m in MODES
        }
    return None, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--chips", type=int, default=4)
    ap.add_argument("--md", action="store_true", help="print one markdown row per dir")
    args = ap.parse_args()
    for d in args.dirs:
        c, window = client(d, args.chips)
        src, tot = server(d, window)
        hit = cpu = None
        if tot:
            all_ = sum(tot.get(m, 0.0) for m in MODES)
            hit = 100 * (tot.get("device_hit", 0.0) + tot.get("host_hit", 0.0)) / all_
            cpu = 100 * tot.get("host_hit", 0.0) / all_
        f = lambda v, n: "n/a" if v is None else f"{v:,.{n}f}"  # noqa: E731
        e2e = (
            " / ".join(f(v, n) for v, n in zip(c["e2e"], (2, 2, 1)))
            if c["e2e"]
            else "n/a"
        )
        cells = [
            f(c["tok_s_chip"], 0),
            f"{c['ttft'][0]:.3f} / {c['ttft'][1]:.3f}",
            f"{c['inter'][0]:.1f} / {c['inter'][1]:.1f}",
            e2e,
            f"{f(hit, 2)} ({f(c['ideal'], 2)})",
            f(cpu, 2),
            f"{c['n']:,} ({c['failed']})",
        ]
        if args.md:
            print(f"| {d} | " + " | ".join(cells) + " |")
            continue
        print(f"{d}")
        print(
            f"  client: {c['source']}, window {c['window_s']:.0f} s; server hit: {src or 'n/a'}"
        )
        names = (
            "tok/s/chip",
            "TTFT p50 / p90 s",
            "interactivity p50 / p90",
            "e2e p50 / p90 / p99 s",
            "server hit % (ideal %)",
            "cpu (host) hit %",
            "requests (failed)",
        )
        for k, v in zip(names, cells):
            print(f"  {k:<24} {v}")
        print(f"  {'client-reported hit %':<24} {f(c['client_hit'], 2)}")


if __name__ == "__main__":
    main()
