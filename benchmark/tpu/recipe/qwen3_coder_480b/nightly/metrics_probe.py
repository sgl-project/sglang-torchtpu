"""Snapshot an sglang server's Prometheus counters and capacity, and diff two
snapshots.

Why this exists: ``--enable-hierarchical-cache`` and a live radix cache mean the
*flags* were accepted, not that anything was cached or offloaded. A leg that
silently ran with 0% prefix hits, or that quietly retracted half its requests
because the KV pool could not hold the working set, looks like a plain
throughput regression in the bench output. The counters below distinguish those
cases, so every leg records a before/after pair.

    # before the leg
    metrics_probe.py snapshot --url http://127.0.0.1:30000 --out pre.json
    # after the leg
    metrics_probe.py snapshot --url http://127.0.0.1:30000 --out post.json
    metrics_probe.py diff --pre pre.json --post post.json --out delta.json \
        --expect-context 1048576 --expect-concurrency 32

``diff`` exits 0 always (a benchmark harness must not fail on an observability
gap) and reports findings as ``warnings`` in the output JSON plus stderr lines.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Dict, List, Optional

# Counters (monotonic; the diff reports post-pre) that answer "did the feature
# actually do anything during this leg?"
_COUNTERS = (
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:cached_tokens_total",  # prefix-cache hits
    "sglang:num_requests_total",
    "sglang:num_aborted_requests_total",
    "sglang:num_retracted_requests_total",  # KV pool could not hold the batch
    "sglang:num_retracted_input_tokens_total",
    # HiCache / KV offload activity
    "sglang:evicted_tokens_total",  # device -> host
    "sglang:backuped_tokens_total",
    "sglang:load_back_tokens_total",  # host -> device (the reload path)
    "sglang:prefetched_tokens_total",
    # P/D transfer health
    "sglang:num_bootstrap_failed_reqs_total",
    "sglang:num_transfer_failed_reqs_total",
    "sglang:num_prefill_retries_total",
)

# Gauges (point-in-time; the diff reports both endpoints).
_GAUGES = (
    "sglang:max_total_num_tokens",
    "sglang:context_len",
    "sglang:page_size",
    "sglang:num_pages",
    "sglang:cache_hit_rate",
    "sglang:token_usage",
    "sglang:num_used_tokens",
    "sglang:kv_available_tokens",
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_retracted_reqs",
    "sglang:num_paused_reqs",
    "sglang:gen_throughput",
)

_WANTED = frozenset(_COUNTERS) | frozenset(_GAUGES)


def _get(url: str, timeout: int) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode()
    except Exception:
        return None


def _parse_prometheus(text: str) -> Dict[str, float]:
    """Sum each wanted metric family across its label sets.

    Summing is the right reduction for these: the counters are per-DP-rank (and
    per-engine when the URL is a router), and the totals are what a leg is
    judged on. Histograms are skipped -- their _bucket/_sum/_count series are
    not in _WANTED, and bench_serving already reports the latency distribution.
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if rest:
            value = rest.rpartition("}")[2]
        else:
            name, _, value = line.partition(" ")
        name = name.strip()
        if name not in _WANTED:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(value.strip())
        except ValueError:
            continue
    return out


def _snapshot(url: str, timeout: int) -> Dict:
    base = url.rstrip("/")
    record: Dict = {"url": base}

    metrics_text = _get(f"{base}/metrics", timeout)
    if metrics_text is None:
        record["metrics"] = {}
        record["_error"] = (
            f"could not read {base}/metrics -- was the server launched with "
            "--enable-metrics?"
        )
    else:
        record["metrics"] = _parse_prometheus(metrics_text)

    info_text = _get(f"{base}/server_info", timeout)
    if info_text is not None:
        try:
            info = json.loads(info_text)
        except json.JSONDecodeError:
            info = {}
        # A P/D router answers with {"prefill": [...], "decode": [...]}.
        for role in ("prefill", "decode"):
            if isinstance(info.get(role), list) and info[role]:
                info = info[role][0]
                break
        if isinstance(info, dict):
            states = info.get("internal_states") or [{}]
            first = states[0] if isinstance(states, list) and states else {}
            record["server"] = {
                key: info.get(key)
                for key in (
                    "max_total_num_tokens",
                    "max_prefill_tokens",
                    "chunked_prefill_size",
                    "max_running_requests",
                    "context_length",
                    "tp_size",
                    "dp_size",
                    "ep_size",
                    "enable_dp_attention",
                    "page_size",
                    "attention_backend",
                    "quantization",
                    "kv_cache_dtype",
                    "disable_radix_cache",
                    "enable_hierarchical_cache",
                    "hicache_ratio",
                    "hicache_size",
                    "hicache_io_backend",
                    "hicache_mem_layout",
                    "mem_fraction_static",
                    "disaggregation_mode",
                    "enable_metrics",
                    "enable_cache_report",
                )
                if key in info
            }
            if isinstance(first, dict):
                record["server"]["internal_states"] = {
                    key: first.get(key)
                    for key in (
                        "effective_max_running_requests_per_dp",
                        "avg_spec_accept_length",
                        "waiting_queue_len",
                    )
                    if key in first
                }
    return record


def _check(delta: Dict, pre: Dict, post: Dict, args) -> List[str]:
    """Turn the numbers into statements a human can act on."""
    warnings: List[str] = []
    counters = delta["counters"]
    server = post.get("server") or pre.get("server") or {}

    if post.get("_error") or pre.get("_error"):
        warnings.append(post.get("_error") or pre["_error"])
        return warnings

    prompt = counters.get("sglang:prompt_tokens_total", 0.0)
    cached = counters.get("sglang:cached_tokens_total", 0.0)
    if prompt <= 0:
        warnings.append(
            "no prompt tokens were counted during this leg -- the bench client "
            "may have failed before sending anything"
        )
    else:
        hit = cached / prompt
        delta["derived"]["prefix_cache_hit_rate"] = round(hit, 4)
        # A multi-turn leg re-sends the whole history every round, so every round
        # after the first should be an almost complete hit. What that adds up to
        # depends on the round count -- round 1 is always a miss, so a 3-round
        # session tops out near 56% and a 25-round one near 94%. This probe does
        # not know the shape, so it only flags a rate low enough to mean the cache
        # is not being reused at all (radix cache off, flushed mid-leg, or a pool
        # so small every prefix is evicted). The CI test derives a per-leg floor
        # from the round counts; --min-hit-rate overrides this one.
        if hit < args.min_hit_rate:
            warnings.append(
                f"prefix-cache hit rate {hit:.1%} is below {args.min_hit_rate:.0%}, "
                "which a multi-turn workload clears from 2 rounds per session "
                "onwards; check that the radix cache is enabled and the KV pool "
                "is not thrashing"
            )

    if server.get("enable_hierarchical_cache"):
        offloaded = counters.get("sglang:backuped_tokens_total", 0.0) + counters.get(
            "sglang:evicted_tokens_total", 0.0
        )
        reloaded = counters.get("sglang:load_back_tokens_total", 0.0)
        delta["derived"]["kv_offload_written_tokens"] = offloaded
        delta["derived"]["kv_offload_reloaded_tokens"] = reloaded
        if offloaded <= 0:
            warnings.append(
                "KV offload is enabled but nothing was written to the host pool "
                "in this leg: the working set fit in HBM, so the leg does not "
                "exercise offload at all"
            )
        elif reloaded <= 0:
            warnings.append(
                f"KV offload wrote {offloaded:.0f} tokens to host but reloaded "
                "none -- prefixes are being evicted and then recomputed rather "
                "than reloaded"
            )

    retracted = counters.get("sglang:num_retracted_requests_total", 0.0)
    if retracted > 0:
        warnings.append(
            f"{retracted:.0f} requests were retracted: the KV pool could not "
            "hold the admitted batch, so these numbers include recompute of "
            "already-generated tokens"
        )
    aborted = counters.get("sglang:num_aborted_requests_total", 0.0)
    if aborted > 0:
        warnings.append(f"{aborted:.0f} requests were aborted server-side")
    for key in (
        "sglang:num_bootstrap_failed_reqs_total",
        "sglang:num_transfer_failed_reqs_total",
    ):
        if counters.get(key, 0.0) > 0:
            warnings.append(
                f"{key} increased by {counters[key]:.0f} (P/D KV transfer failures)"
            )

    # Capacity: can the pool hold what this leg admits at once?
    pool = server.get("max_total_num_tokens")
    if pool and args.expect_context and args.expect_concurrency:
        need = int(args.expect_context) * int(args.expect_concurrency)
        delta["derived"]["kv_tokens_required_worst_case"] = need
        delta["derived"]["kv_tokens_available"] = pool
        if need > pool:
            warnings.append(
                f"worst-case KV demand {need} tokens "
                f"({args.expect_concurrency} x {args.expect_context}) exceeds the "
                f"device pool ({pool} tokens). Expect host offload and/or "
                "retraction; that is the intended path when KV offload is on, "
                "but it does bound throughput"
            )
    ctx = server.get("context_length")
    if ctx and args.expect_context and int(ctx) < int(args.expect_context):
        warnings.append(
            f"server --context-length {ctx} is below the leg's target context "
            f"{args.expect_context}; the last rounds of the longest sessions "
            "will be rejected"
        )
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot", help="write one metrics/capacity snapshot")
    snap.add_argument("--url", required=True)
    snap.add_argument("--out", required=True)
    snap.add_argument("--timeout", type=int, default=30)

    dif = sub.add_parser("diff", help="diff two snapshots and flag problems")
    dif.add_argument("--pre", required=True)
    dif.add_argument("--post", required=True)
    dif.add_argument("--out", required=True)
    dif.add_argument(
        "--expect-context",
        type=int,
        default=0,
        help="target context of the leg's longest session",
    )
    dif.add_argument(
        "--expect-concurrency",
        type=int,
        default=0,
        help="client concurrency of the leg",
    )
    dif.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.2,
        help="warn below this prefix-cache hit rate. Shape-independent floor: "
        "the achievable rate depends on rounds per session (~56%% at 3 rounds, "
        "~94%% at 25), so this only catches a cache that is not being reused at "
        "all (default 0.2)",
    )

    args = ap.parse_args()

    if args.cmd == "snapshot":
        record = _snapshot(args.url, args.timeout)
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
        if "_error" in record:
            print(f"[metrics_probe] WARN: {record['_error']}", file=sys.stderr)
        else:
            print(
                f"[metrics_probe] snapshot -> {args.out} "
                f"({len(record['metrics'])} metrics)"
            )
        return 0

    with open(args.pre) as f:
        pre = json.load(f)
    with open(args.post) as f:
        post = json.load(f)

    delta: Dict = {"counters": {}, "gauges": {}, "derived": {}, "warnings": []}
    for name in _COUNTERS:
        before = pre.get("metrics", {}).get(name)
        after = post.get("metrics", {}).get(name)
        if before is None and after is None:
            continue
        delta["counters"][name] = (after or 0.0) - (before or 0.0)
    for name in _GAUGES:
        if name in post.get("metrics", {}) or name in pre.get("metrics", {}):
            delta["gauges"][name] = {
                "pre": pre.get("metrics", {}).get(name),
                "post": post.get("metrics", {}).get(name),
            }
    delta["server"] = post.get("server") or pre.get("server") or {}
    delta["warnings"] = _check(delta, pre, post, args)

    with open(args.out, "w") as f:
        json.dump(delta, f, indent=2, sort_keys=True)
        f.write("\n")
    for warning in delta["warnings"]:
        print(f"[metrics_probe] WARN: {warning}", file=sys.stderr)
    print(
        f"[metrics_probe] diff -> {args.out} " f"({len(delta['warnings'])} warning(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
