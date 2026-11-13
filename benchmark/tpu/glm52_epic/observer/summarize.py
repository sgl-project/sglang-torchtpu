#!/usr/bin/env python3
"""Stdlib-only rank-local observer diagnostic; never a matched JAX metric.

  python3 summarize.py --observer-dir ARTIFACTS --requests requests.json --attn-dp-size 16
  python3 summarize.py --self-test

Requires the native client's explicit C32 measured UUID manifest and all 16 ranks.
No timestamp pooling, global median, throughput, performance ratio, or JAX claim.
"""

import argparse
import json
import statistics
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path

P, E, CHUNK = 49152, 12288, 2048
CONTRACT = {
    "marker": "TORCH_RANK_LOCAL_PREFILL_COMPLETION", "schema_version": 1,
    "scope": "rank_local_diagnostic_not_matched_jax",
    "boundary": "after_existing_copy_done_synchronize",
    "clock": "process_local_perf_counter_ns",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def measured_ids(manifest):
    require(isinstance(manifest, dict), "request manifest must be an object")
    for key, value in (("format", "torch-native-cache-hit-v1"),
                       ("prefix_len", P), ("extend_len", E), ("concurrency", 32)):
        require(manifest.get(key) == value, "request manifest mismatch: " + key)
    records = manifest.get("requests")
    require(isinstance(records, list) and len(records) == 32, "expected C32 manifest")
    ids = []
    for record in records:
        require(isinstance(record, dict), "request manifest record must be an object")
        rid = record.get("measured_rid")
        require(isinstance(rid, str) and str(uuid.UUID(rid)) == rid,
                "measured_rid must be an explicit canonical UUID")
        ids.append(rid)
    require(len(set(ids)) == 32, "duplicate measured UUID in manifest")
    require(not set(ids).intersection(r.get("warm_rid") for r in records),
            "warm and measured UUIDs overlap")
    return set(ids)


def validate(event):
    require(isinstance(event, dict), "event must be an object")
    for key, value in CONTRACT.items():
        require(type(event.get(key)) is type(value) and event[key] == value,
                "observer schema mismatch: " + key)
    require(isinstance(event.get("hostname"), str) and event["hostname"], "missing hostname")
    for key in ("pid", "tp_rank", "pp_rank", "attn_dp_rank", "bid", "forward_step"):
        require(type(event.get(key)) is int and event[key] >= 0, "invalid integer: " + key)
    require(event["pid"] > 0 and event["pp_rank"] == 0, "invalid process or PP rank")
    require(event.get("status") == "observed" and event.get("unavailable_reason") is None,
            "unavailable event: " + str(event.get("unavailable_reason")))
    require(type(event.get("completed_ns")) is int and event["completed_ns"] >= 0,
            "invalid completion timestamp")
    reqs = event.get("requests")
    require(isinstance(reqs, list), "requests must be a list")
    seen = set()
    for req in reqs:
        require(isinstance(req, dict) and isinstance(req.get("rid"), str) and req["rid"],
                "invalid request RID")
        require(req["rid"] not in seen, "duplicate RID metadata")
        seen.add(req["rid"])
        for key in ("dp", "prefix", "tokens"):
            require(type(req.get(key)) is int and req[key] >= 0, "invalid request " + key)
        require(req["dp"] == event["attn_dp_rank"], "request/rank DP mismatch")


def rank_metrics(events, selected):
    previous, samples, matched = None, [], set()
    rejected = Counter()
    for event in events:
        reqs = event["requests"]
        current = {r["rid"]: r for r in reqs}
        matched.update(current.keys() & selected)
        full = (len(current) == len(reqs) == 1
                and all(r["rid"] in selected and r["tokens"] == CHUNK for r in reqs))
        # Pinned JAX helper's CURRENT-interior rule, adapted only to local width 1.
        # The first full chunk IS a predecessor: do not require prior interior.
        interior = full and all(
            P < r["prefix"] and r["prefix"] + r["tokens"] < P + E
            and (r["prefix"] - P) % CHUNK == 0 for r in reqs)
        if interior and previous is not None:
            prior, prior_reqs = previous
            if (current.keys() == prior_reqs.keys()
                and event["forward_step"] == prior["forward_step"] + 1
                and all(r["prefix"] == prior_reqs[rid]["prefix"] + CHUNK
                        and r["dp"] == prior_reqs[rid]["dp"]
                        for rid, r in current.items())):
                duration = (event["completed_ns"] - prior["completed_ns"]) / 1e6
                require(duration > 0, "non-monotonic rank-local timestamps")
                samples.append({"bid": event["bid"], "forward_step": event["forward_step"],
                                "rid": reqs[0]["rid"], "duration_ms": duration,
                                "prefix_tokens": sorted({r["prefix"] for r in reqs})})
            else:
                rejected["request_set_step_or_prefix_changed"] += 1
        else:
            rejected["nonfull_noninterior_or_no_predecessor"] += 1
        # Unknown/warm/nonfull events MUST reset, never filter them before this.
        previous = (event, current) if full else None
    durations = sorted(s["duration_ms"] for s in samples)
    return {"measured_rids": sorted(matched), "samples": samples,
            "sample_count": len(samples), "rejections": dict(rejected),
            "median_ms": statistics.median(durations) if durations else None,
            # Exact pinned JAX nearest-index percentile (Python round).
            "p95_ms": durations[min(round((len(durations) - 1) * .95), len(durations) - 1)]
                      if durations else None}


def summarize(directory, manifest, attn_dp_size=16):
    require(type(attn_dp_size) is int and attn_dp_size == 16, "expected --attn-dp-size 16")
    selected = measured_ids(manifest)
    require(directory.is_dir(), "observer directory does not exist")
    ranks, identities, processes = defaultdict(list), {}, {}
    seen, duplicates = {}, Counter()
    for path in sorted(directory.glob("*.jsonl")):
        file_identity = None
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    validate(event)
                    rank = event["attn_dp_rank"]
                    require(rank in range(attn_dp_size), "unexpected attention-DP rank")
                    identity = (event["hostname"], event["pid"])
                    require(file_identity in (None, identity), "mixed processes in one file; no single timeline")
                    file_identity = identity
                    require(identities.get(rank, identity) == identity,
                            "per-rank process restart/mixed processes; no single timeline")
                    topology = (rank, event["tp_rank"], event["pp_rank"])
                    require(processes.get(identity, topology) == topology,
                            "process rank identity changed; no single timeline")
                    identities[rank], processes[identity] = identity, topology
                    key = (identity, event["forward_step"])
                    canonical = json.dumps(event, sort_keys=True)
                    if key in seen:
                        require(seen[key] == canonical, "conflicting duplicate forward_step/restart; no single timeline")
                        duplicates[rank] += 1
                        continue
                    seen[key] = canonical
                    if ranks[rank]:
                        prior = ranks[rank][-1]
                        require(event["forward_step"] > prior["forward_step"], "forward ordinal reset; no single timeline")
                        require(event["completed_ns"] > prior["completed_ns"], "non-monotonic process timeline")
                    ranks[rank].append(event)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{path.name}:{number}: {exc}") from exc
    result = dict(CONTRACT, status="failed", matched_jax_cadence=False,
                  workload={"prefix_len": P, "extend_len": E, "chunk_size": CHUNK,
                            "attn_dp_size": 16, "concurrency": 32, "requests_per_rank": 2},
                  missing_ranks=sorted(set(range(16)) - ranks.keys()), ranks={})
    problems, ownership = [], {}
    if result["missing_ranks"]:
        problems.append("missing expected ranks")
    for rank, events in sorted(ranks.items()):
        metrics = rank_metrics(events, selected)
        metrics.update(hostname=identities[rank][0], pid=identities[rank][1],
                       tp_rank=events[0]["tp_rank"], duplicate_events=duplicates[rank])
        result["ranks"][str(rank)] = metrics
        if len(metrics["measured_rids"]) != 2 or not metrics["samples"]:
            problems.append(f"rank {rank}: expected two measured requests and qualifying samples")
        for rid in metrics["measured_rids"]:
            require(rid not in ownership, "measured RID observed on multiple ranks")
            ownership[rid] = rank
    result["missing_measured_rids"] = sorted(selected - ownership.keys())
    if result["missing_measured_rids"]:
        problems.append("manifest measured UUIDs absent")
    result.update(status="failed" if problems else "rank_local_diagnostic_not_matched_jax",
                  errors=problems)
    return result


def self_test():
    import copy
    ids = [str(uuid.UUID(int=i + 1)) for i in range(32)]
    manifest = dict(format="torch-native-cache-hit-v1", prefix_len=P, extend_len=E,
                    concurrency=32, requests=[{"measured_rid": rid} for rid in ids])
    waves = []
    for rank in range(16):
        events = []
        for wave in range(2):
            for chunk in range(6):
                step = wave * 6 + chunk
                events.append(dict(CONTRACT, hostname=f"host{rank // 8}", pid=rank + 1,
                    tp_rank=rank, pp_rank=0, attn_dp_rank=rank, bid=step, forward_step=step,
                    status="observed", unavailable_reason=None,
                    completed_ns=rank * 10**12 + step * 1000000,
                    requests=[dict(rid=ids[rank * 2 + wave], dp=rank,
                                   prefix=P + chunk * CHUNK, tokens=CHUNK)]))
        waves.append(events)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        def run(data):
            for rank in range(16):
                (directory / f"rank{rank}.jsonl").write_text(
                    "".join(json.dumps(e) + "\n" for e in data[rank]))
            return summarize(directory, manifest)
        def fails(data):
            try:
                run(data)
            except ValueError:
                return
            raise AssertionError("invalid evidence accepted")
        good = run(waves)
        assert good["status"] == CONTRACT["scope"]
        assert all(r["sample_count"] == 8 and r["median_ms"] == r["p95_ms"] == 1
                   for r in good["ranks"].values())
        missing = copy.deepcopy(waves); missing[15] = []
        assert run(missing)["missing_ranks"] == [15]
        duplicate = copy.deepcopy(waves); duplicate[0].insert(1, duplicate[0][0])
        assert run(duplicate)["ranks"]["0"]["duplicate_events"] == 1
        for field, value in (("status", "unavailable"), ("completed_ns", None),
                             ("pid", 999), ("requests", [{"rid": "bad"}])):
            bad = copy.deepcopy(waves); bad[0][1][field] = value; fails(bad)
        bad = copy.deepcopy(duplicate); bad[0][1] = dict(bad[0][1], bid=999); fails(bad)
        assert rank_metrics(waves[0][1:6], set(ids))["sample_count"] == 3
        for kind in ("unknown", "warm", "nonfull"):
            reset = copy.deepcopy(waves[0][:6])
            if kind == "nonfull":
                reset[1]["requests"][0]["tokens"] = 1024
            else:
                reset[1]["requests"][0]["rid"] = kind
            assert rank_metrics(reset, set(ids))["sample_count"] == 2
    print("PASS: 16 processes, C32/two waves, 8 samples/rank; missing/unavailable/malformed/restart/duplicates; first predecessor and resets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-dir", type=Path)
    parser.add_argument("--requests", type=Path, help="native client measured UUID manifest")
    parser.add_argument("--attn-dp-size", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.observer_dir or not args.requests:
        parser.error("--observer-dir and --requests are required")
    try:
        result = summarize(args.observer_dir, json.loads(args.requests.read_text()), args.attn_dp_size)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = dict(scope=CONTRACT["scope"], status="failed", matched_jax_cadence=False, error=str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
