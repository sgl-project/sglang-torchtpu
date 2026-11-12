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
import math
import re
import sys
import urllib.request
from typing import Dict, List, Optional

# Counters (monotonic; the diff reports post-pre) that answer "did the feature
# actually do anything during this leg?"
_COUNTERS = (
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:spec_verify_calls_total",
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

# Server-side histograms whose per-leg delta says where the time went. The
# client sees one TTFT; in P/D that folds together prefill queueing, prefill
# forward, the P->D KV transfer and decode-side admission, and only the server
# can split them. Value: the label to keep apart (None sums every label set).
#
# Which engine exports what (P/D): kv_transfer_{latency_ms,total_mb,speed_gb_s}
# are observed by the prefill engine when a request's transfer completes;
# per_stage_req_latency_seconds carries prefill_* stages on the prefill engine
# and decode_* stages on the decode engine. A colocated server exports the
# stage and queue histograms only.
_HISTOGRAMS: Dict[str, Optional[str]] = {
    "sglang:kv_transfer_latency_ms": None,
    "sglang:kv_transfer_total_mb": None,
    "sglang:kv_transfer_speed_gb_s": None,
    "sglang:kv_transfer_bootstrap_ms": None,
    "sglang:kv_transfer_alloc_ms": None,
    "sglang:queue_time_seconds": None,
    "sglang:per_stage_req_latency_seconds": "stage",
    "sglang:prompt_tokens_histogram": None,
    "sglang:uncached_prompt_tokens_histogram": None,
}
_ALL_SERIES = "all"
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


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
    judged on. Histograms are not read here (see _parse_histograms).
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


def _parse_histograms(text: str) -> Dict[str, Dict[str, Dict]]:
    """Read the _HISTOGRAMS families as cumulative bucket counts.

    Returns ``{family: {series: {"buckets": {le: cum}, "sum": s, "count": n}}}``
    where ``series`` is the value of the family's split label (``"all"`` when
    it has none). Every other label set (DP rank, engine) is summed, matching
    the counter reduction.
    """
    out: Dict[str, Dict[str, Dict]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if rest:
            label_text, _, value = rest.rpartition("}")
        else:
            name, _, value = line.partition(" ")
            label_text = ""
        name = name.strip()
        for suffix in ("_bucket", "_sum", "_count"):
            if name.endswith(suffix) and name[: -len(suffix)] in _HISTOGRAMS:
                family = name[: -len(suffix)]
                break
        else:
            continue
        try:
            number = float(value.strip().split()[0])
        except (ValueError, IndexError):
            continue
        labels = dict(_LABEL_RE.findall(label_text))
        split = _HISTOGRAMS[family]
        series_key = labels.get(split, _ALL_SERIES) if split else _ALL_SERIES
        series = out.setdefault(family, {}).setdefault(
            series_key, {"buckets": {}, "sum": 0.0, "count": 0.0}
        )
        if suffix == "_bucket":
            le = labels.get("le")
            if le is None:
                continue
            le = _norm_le(le)
            series["buckets"][le] = series["buckets"].get(le, 0.0) + number
        else:
            series[suffix[1:]] += number
            if suffix == "_count":
                # How many label sets (TP ranks, for the per-rank KV transfer
                # families) were summed into this series.
                series["label_sets"] = series.get("label_sets", 0) + 1
    return out


def _norm_le(le: str) -> str:
    """Canonical bucket bound text, so "5", "5.0" and "+Inf" key consistently."""
    try:
        bound = float(le)
    except ValueError:
        return le
    return "+Inf" if math.isinf(bound) else repr(bound)


def _quantile(bounds: List[float], counts: List[float], q: float) -> Optional[float]:
    """Prometheus-style histogram_quantile on (upper bound, cumulative count).

    Linear interpolation inside the bucket that holds the q-th observation.
    A quantile that lands in the +Inf bucket is reported as the highest finite
    bound -- a floor, flagged separately via ``above_top_bucket``.
    """
    if not counts or counts[-1] <= 0:
        return None
    rank = q * counts[-1]
    prev_bound, prev_count = 0.0, 0.0
    for bound, count in zip(bounds, counts):
        if count >= rank:
            if math.isinf(bound):
                return prev_bound
            width = count - prev_count
            if width <= 0:
                return bound
            return prev_bound + (bound - prev_bound) * (rank - prev_count) / width
        prev_bound, prev_count = bound, count
    return prev_bound


def _diff_histograms(pre: Dict, post: Dict) -> Dict[str, Dict[str, Dict]]:
    """Per-leg distribution of each histogram series: post minus pre.

    Series with no new observations are omitted. A negative delta (the server
    restarted mid-leg) drops the series rather than reporting nonsense.
    """
    out: Dict[str, Dict[str, Dict]] = {}
    for family, series_map in (post or {}).items():
        for series_key, after in series_map.items():
            before = ((pre or {}).get(family) or {}).get(series_key) or {}
            count = after.get("count", 0.0) - before.get("count", 0.0)
            total = after.get("sum", 0.0) - before.get("sum", 0.0)
            if count <= 0:
                continue
            bucket_delta = []
            for le, cum in after.get("buckets", {}).items():
                try:
                    bound = float(le)
                except ValueError:
                    continue
                bucket_delta.append(
                    (bound, cum - before.get("buckets", {}).get(le, 0.0))
                )
            bucket_delta.sort()
            if any(c < 0 for _, c in bucket_delta):
                continue
            bounds = [b for b, _ in bucket_delta]
            counts = [c for _, c in bucket_delta]
            finite = [(b, c) for b, c in bucket_delta if not math.isinf(b)]
            record = {
                "count": count,
                "sum": total,
                "mean": total / count,
                "p50": _quantile(bounds, counts, 0.50),
                "p90": _quantile(bounds, counts, 0.90),
                "p99": _quantile(bounds, counts, 0.99),
                "label_sets": after.get("label_sets", 1),
            }
            if finite:
                record["top_bucket"] = finite[-1][0]
                record["above_top_bucket"] = max(0.0, count - finite[-1][1])
            out.setdefault(family, {})[series_key] = record
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
        record["histograms"] = _parse_histograms(metrics_text)

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
                    "attn_dp_size",
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
                    "speculative_algorithm",
                    "speculative_draft_model_path",
                    "speculative_num_steps",
                    "speculative_eagle_topk",
                    "speculative_num_draft_tokens",
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
            if isinstance(states, list):
                record["server"]["scheduler_states"] = [
                    {
                        "scheduler_index": index,
                        **{
                            key: state.get(key)
                            for key in (
                                "effective_max_running_requests_per_dp",
                                "avg_spec_accept_length",
                                "waiting_queue_len",
                            )
                            if key in state
                        },
                    }
                    for index, state in enumerate(states)
                    if isinstance(state, dict)
                ]
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
    if getattr(args, "role", "unified") == "decode":
        # A P/D decode engine receives its KV from prefill; prompt and prefix
        # cache accounting live on the prefill side.
        pass
    elif prompt <= 0:
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

    if server.get("speculative_algorithm"):
        verify_calls = counters.get("sglang:spec_verify_calls_total", 0.0)
        delta["derived"]["spec_verify_calls"] = verify_calls
        if verify_calls <= 0:
            warnings.append(
                "speculative decoding is configured but spec_verify_calls_total "
                "did not increase during this leg -- the workload silently ran "
                "without draft verification"
            )

        states = server.get("scheduler_states") or []
        accept_lengths = [
            state.get("avg_spec_accept_length")
            for state in states
            if state.get("avg_spec_accept_length") is not None
        ]
        delta["derived"]["spec_accept_length_by_scheduler"] = accept_lengths
        if len(accept_lengths) != len(states) or not accept_lengths:
            warnings.append(
                "speculative decoding is configured but at least one scheduler "
                "reported no avg_spec_accept_length"
            )
        else:
            min_accept_length = min(float(value) for value in accept_lengths)
            delta["derived"]["min_spec_accept_length"] = min_accept_length
            if min_accept_length <= args.min_spec_accept_length:
                warnings.append(
                    f"minimum speculative accept length {min_accept_length:.3f} "
                    f"did not exceed the configured floor "
                    f"{args.min_spec_accept_length:.3f}"
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

    # P/D: headline transfer numbers as flat fields (the dashboard dump only
    # takes top-level numbers); the full distributions stay in "histograms".
    # Every TP rank sends (and observes) its own shard of each request's KV, so
    # latency/MB/speed are per-rank-shard values; kv_transfers and
    # kv_transfer_request_mb_mean are normalized to whole requests.
    histograms = delta.get("histograms") or {}
    for family, prefix in (
        ("sglang:kv_transfer_latency_ms", "kv_transfer_latency_ms"),
        ("sglang:kv_transfer_total_mb", "kv_transfer_mb"),
        ("sglang:kv_transfer_speed_gb_s", "kv_transfer_speed_gb_s"),
    ):
        series = (histograms.get(family) or {}).get(_ALL_SERIES)
        if not series:
            continue
        for stat in ("mean", "p50", "p99"):
            if series.get(stat) is not None:
                delta["derived"][f"{prefix}_{stat}"] = round(series[stat], 3)
        ranks = max(1, int(series.get("label_sets") or 1))
        if family == "sglang:kv_transfer_latency_ms":
            delta["derived"]["kv_transfer_ranks"] = ranks
            delta["derived"]["kv_transfers"] = series["count"] / ranks
        elif family == "sglang:kv_transfer_total_mb":
            delta["derived"]["kv_transfer_request_mb_mean"] = round(
                series["mean"] * ranks, 3
            )
    # kv_transfer_latency_ms is a window from the first KV send, which (with
    # cached-prefix early send) precedes the prefill forward on prefix hits.
    # The transfer time a request actually waits for is the prefill's
    # prefill_transfer_kv_cache stage: transfer-queue entry, after the forward,
    # to the transfer's finish.
    if "kv_transfers" in delta["derived"]:
        exposed = (histograms.get("sglang:per_stage_req_latency_seconds") or {}).get(
            "prefill_transfer_kv_cache"
        ) or {}
        if exposed.get("mean") is not None:
            delta["derived"]["kv_transfer_exposed_ms_mean"] = round(
                exposed["mean"] * 1000, 3
            )
    for family, series_map in histograms.items():
        for series_key, series in series_map.items():
            over = series.get("above_top_bucket") or 0
            if over > 0:
                label = (
                    family
                    if series_key == _ALL_SERIES
                    else (f"{family}{{{_HISTOGRAMS.get(family)}={series_key}}}")
                )
                warnings.append(
                    f"{over:.0f}/{series['count']:.0f} {label} observations "
                    f"exceeded the top bucket ({series['top_bucket']:g}); its "
                    "p50/p99 are a floor, use the mean"
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
    dif.add_argument(
        "--min-spec-accept-length",
        type=float,
        default=1.0,
        help="warn unless every scheduler's lifetime speculative accept length "
        "strictly exceeds this value (default 1.0, which proves that at least "
        "one draft token was accepted)",
    )
    dif.add_argument(
        "--role",
        choices=("prefill", "decode", "unified"),
        default="unified",
        help="which engine the snapshots came from; 'decode' skips the "
        "prompt/prefix-cache checks, which a P/D decode engine never feeds",
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
                f"({len(record['metrics'])} metrics, "
                f"{len(record.get('histograms') or {})} histograms)"
            )
        return 0

    with open(args.pre) as f:
        pre = json.load(f)
    with open(args.post) as f:
        post = json.load(f)

    delta = _diff(pre, post, args)

    with open(args.out, "w") as f:
        json.dump(delta, f, indent=2, sort_keys=True)
        f.write("\n")
    for warning in delta["warnings"]:
        print(f"[metrics_probe] WARN: {warning}", file=sys.stderr)
    print(f"[metrics_probe] diff -> {args.out} ({len(delta['warnings'])} warning(s))")
    return 0


def _diff(pre: Dict, post: Dict, args) -> Dict:
    delta: Dict = {
        "role": getattr(args, "role", "unified"),
        "counters": {},
        "gauges": {},
        "histograms": {},
        "derived": {},
        "warnings": [],
    }
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
    delta["histograms"] = _diff_histograms(
        pre.get("histograms") or {}, post.get("histograms") or {}
    )
    delta["server"] = post.get("server") or pre.get("server") or {}
    delta["warnings"] = _check(delta, pre, post, args)
    return delta


if __name__ == "__main__":
    sys.exit(main())
