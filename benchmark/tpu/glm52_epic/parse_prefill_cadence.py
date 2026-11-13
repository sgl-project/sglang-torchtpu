#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline, stdlib-only parser for the pinned GLM-5.2 completion cadence.

Examples (no server, accelerator, or SGLang imports):
  python3 parse_prefill_cadence.py --log server.log --run-prefix UUID
  python3 parse_prefill_cadence.py --log merged.log --rids-json measured.json --stream 0
  python3 parse_prefill_cadence.py --self-test

--run-prefix is the exact request_namespace, NOT an inferred warmup namespace.
--rids-json accepts a JSON string list or {"rids": [...]}; --rids accepts a list
of explicit IDs. These are measured-request manifests, never all observed IDs.
Rank identity comes from event stream_id, host/rank, or a logger [rank=N] /
[TPN] tag. For other merged-log formats use --stream-regex with one capture.
Mixed streams require explicit --stream; clocks are never pooled. Untagged
logs must already contain one process only. Log input is scanned line by line;
the unchanged pinned helper subsequently reads the normalized temporary log.

This is host result-completion cadence, not device timing or TTFT. A successful
parse alone does not certify cache hits, completions, compilation, or workload
acceptance; see MEASUREMENT_PROTOCOL.md.
"""

import argparse
import hashlib
import io
import json
import re
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

SOURCE_COMMIT = "3695225808f0293cae422ea7adf55eddacadd43d"
SOURCE_PATH = "benchmark/glm52/bench_dsa_cache_hit.py"
SOURCE_SHA256 = "7942972af82f57a53874cc843cb79bb08cf7b9a698e2b8229cde9bce3fd97b5ddd47"
HELPER_SHA256 = "ff975d82b38202c387aa68e2629cde9b5c6857c425ebed81c355805ae46d693e"
MARKER = "PREFILL_STEP_COMPLETED "
SUFFIX = "-cache-hit-extend-decode-"


# Copied verbatim from the Apache-2.0 sglang-jax source pinned above, lines 93-98.
def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


# BEGIN VERBATIM PINNED HELPER (source lines 1063-1119; hash above)
def _steady_chunk_extend_metrics(
    server_log: Path, *, start_offset: int, request_namespace: str,
    prefix_len: int, extend_len: int, chunk_size: int, dp_size: int,
) -> dict:
    """Measure host-observed completion cadence, never divide request TTFT.

    Only consecutive, full-DP chunks of the same requests qualify. Discard
    each request group's first and final chunk, including partial tails.
    """
    with server_log.open("rb") as source:
        source.seek(start_offset)
        lines = source.read().decode(errors="replace").splitlines()
    marker = "PREFILL_STEP_COMPLETED "
    previous = None
    samples = []
    for line in lines:
        if marker not in line:
            continue
        event = json.loads(line.split(marker, 1)[1])
        reqs = event["requests"]
        current = {r["rid"]: r for r in reqs}
        full = (
            len(current) == len(reqs) == dp_size
            and {r["dp"] for r in reqs} == set(range(dp_size))
            and all(r["rid"].startswith(request_namespace + "-cache-hit-extend-decode-")
                    and r["tokens"] == chunk_size for r in reqs)
        )
        interior = full and all(
            prefix_len < r["prefix"]
            and r["prefix"] + r["tokens"] < prefix_len + extend_len
            and (r["prefix"] - prefix_len) % chunk_size == 0
            for r in reqs
        )
        if interior and previous is not None:
            prior, prior_reqs = previous
            if (
                current.keys() == prior_reqs.keys()
                and event["forward_step"] == prior["forward_step"] + 1
                and all(r["prefix"] == prior_reqs[rid]["prefix"] + chunk_size
                        and r["dp"] == prior_reqs[rid]["dp"]
                        for rid, r in current.items())
            ):
                duration_ms = (event["completed_ns"] - prior["completed_ns"]) / 1e6
                if duration_ms <= 0:
                    raise RuntimeError("non-monotonic prefill completion timestamps")
                samples.append({"bid": event["bid"], "duration_ms": duration_ms,
                                "prefix_tokens": sorted({r["prefix"] for r in reqs})})
        previous = (event, current) if full else None
    durations = [s["duration_ms"] for s in samples]
    return {
        "steady_chunk_extend_ms": statistics.median(durations) if durations else None,
        "steady_chunk_extend_p95_ms": _percentile(durations, 0.95) if durations else None,
        "steady_chunk_extend_samples": samples,
        "steady_chunk_extend_sample_count": len(samples),
        "steady_chunk_extend_definition": "consecutive_host_result_completion_intervals_excluding_first_and_final_chunks",
        "steady_chunk_extend_status": "ok" if samples else "no_qualifying_completion_intervals",
    }

# END VERBATIM PINNED HELPER


def stream_identity(event, prefix, pattern=None):
    if pattern is not None:
        match = pattern.search(prefix)
        if not match:
            raise ValueError("stream regex did not match event logger prefix")
        return match.group(1)
    if "stream_id" in event:
        return str(event["stream_id"])
    if "rank" in event:
        return f"{event.get('host', '')}:{event['rank']}" if "host" in event else str(event["rank"])
    match = re.search(r"\[(?:rank[=: ]+|TP)([0-9]+)\]", prefix, re.I)
    return match.group(1) if match else "untagged"


def validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    for key in ("bid", "forward_step", "completed_ns"):
        if type(event.get(key)) is not int or event[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    reqs = event.get("requests")
    if not isinstance(reqs, list):
        raise ValueError("requests must be a list")
    seen = set()
    for req in reqs:
        if not isinstance(req, dict) or not isinstance(req.get("rid"), str) or not req["rid"]:
            raise ValueError("each request needs a nonempty rid")
        if req["rid"] in seen:
            raise ValueError("duplicate request rid metadata")
        seen.add(req["rid"])
        for key in ("dp", "prefix", "tokens"):
            if type(req.get(key)) is not int or req[key] < 0:
                raise ValueError(f"request {key} must be a nonnegative integer")


def parse_log(source, *, rids=None, run_prefix=None, stream=None, stream_regex=None):
    """Normalize selection/transport only; all timing comes from pinned helper."""
    if (rids is None) == (run_prefix is None):
        raise ValueError("provide exactly one explicit measured RID manifest or run prefix")
    if rids is not None and (not rids or len(rids) != len(set(rids))
                             or any(not isinstance(r, str) or not r for r in rids)):
        raise ValueError("RID manifest must be a nonempty list of unique nonempty strings")
    if run_prefix is not None and not run_prefix:
        raise ValueError("run prefix cannot be empty")
    pattern = re.compile(stream_regex) if stream_regex else None
    if pattern and pattern.groups != 1:
        raise ValueError("stream regex must contain exactly one capture group")
    selected = set(rids or [])
    namespace = run_prefix if run_prefix is not None else "explicit-manifest"
    aliases = {rid: namespace + SUFFIX + str(i) for i, rid in enumerate(rids or [])}
    counts = Counter()
    streams = set()
    matched = set()
    seen = {}
    input_hash = hashlib.sha256()
    audit = []
    with tempfile.TemporaryDirectory(prefix="prefill-cadence-") as temporary:
        normalized = Path(temporary) / "events.log"
        with normalized.open("w", encoding="utf-8") as output:
            for line_number, raw in enumerate(source, 1):
                if isinstance(raw, bytes):
                    input_hash.update(raw)
                    line = raw.decode("utf-8", errors="replace")
                else:
                    input_hash.update(raw.encode("utf-8"))
                    line = raw
                counts["lines"] += 1
                if MARKER not in line:
                    continue
                counts["completion_lines"] += 1
                prefix, payload = line.split(MARKER, 1)
                try:
                    event = json.loads(payload)
                    if not isinstance(event, dict):
                        raise ValueError("event must be an object")
                    identity = stream_identity(event, prefix, pattern)
                    streams.add(identity)
                    if stream is not None and identity != stream:
                        counts["other_stream"] += 1
                        continue
                    if stream is None and len(streams) > 1:
                        raise ValueError("mixed process clocks: select --stream explicitly")
                    validate_event(event)
                    # A forward ordinal identifies one completion in a process.
                    # Same ordinal with any changed metadata is not a duplicate.
                    key = (identity, event["forward_step"])
                    canonical = json.dumps({k: event[k] for k in
                        ("bid", "forward_step", "completed_ns", "requests")}, sort_keys=True)
                    if key in seen:
                        if seen[key] != canonical:
                            raise ValueError("conflicting duplicate forward_step/event identity")
                        counts["duplicate_events"] += 1
                        continue
                    seen[key] = canonical
                    for req in event["requests"]:
                        rid = req["rid"]
                        is_selected = (rid in selected if rids is not None
                                       else rid.startswith(namespace + SUFFIX))
                        if is_selected:
                            matched.add(rid)
                        if rids is not None:
                            req["rid"] = aliases.get(rid, "unselected:" + rid)
                    # Keep unrelated/nonfull events: the literal helper must reset
                    # its predecessor on them, rather than bridging over traffic.
                    output.write(MARKER + json.dumps(event) + "\n")
                    audit.append(event)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"line {line_number}: {error}") from error
        if not matched:
            raise ValueError("no matched measured-request manifest; refusing warmup inference")
        if rids is not None and matched != selected:
            raise ValueError("manifest IDs absent from selected stream: " + repr(sorted(selected - matched)))
        result = _steady_chunk_extend_metrics(
            normalized, start_offset=0, request_namespace=namespace,
            prefix_len=49152, extend_len=12288, chunk_size=2048, dp_size=16)
    # Rejection diagnostics describe gates, never compute alternative timings.
    rejections = Counter()
    previous = None
    for event in audit:
        reqs = event["requests"]
        current = {r["rid"]: r for r in reqs}
        full = (len(current) == len(reqs) == 16
                and {r["dp"] for r in reqs} == set(range(16))
                and all(r["rid"].startswith(namespace + SUFFIX) and r["tokens"] == 2048 for r in reqs))
        interior = full and all(49152 < r["prefix"] and r["prefix"] + r["tokens"] < 61440
                               and (r["prefix"] - 49152) % 2048 == 0 for r in reqs)
        if not full:
            rejections["nonfull_or_unselected_event"] += 1
        elif not interior:
            rejections["noninterior_current_event"] += 1
        elif previous is None:
            rejections["no_full_predecessor"] += 1
        else:
            prior, prior_reqs = previous
            if current.keys() != prior_reqs.keys():
                rejections["request_set_changed"] += 1
            elif event["forward_step"] != prior["forward_step"] + 1:
                rejections["nonconsecutive_forward_step"] += 1
            elif not all(r["prefix"] == prior_reqs[rid]["prefix"] + 2048
                         and r["dp"] == prior_reqs[rid]["dp"] for rid, r in current.items()):
                rejections["prefix_or_dp_mapping_changed"] += 1
        previous = (event, current) if full else None
    median = result["steady_chunk_extend_ms"]
    result.update({
        "rejections": dict(rejections), "input_counts": dict(counts),
        "streams_observed": sorted(streams), "selected_stream": stream or next(iter(streams)),
        "matched_rids": sorted(matched), "selection": {"run_prefix": run_prefix, "rids": rids},
        "median_seconds": median / 1000 if median is not None else None,
        "throughput_tokens_per_second": 32768 / (median / 1000) if median is not None else None,
        "throughput_definition": "32768 / (steady_chunk_extend_ms / 1000)",
        "workload": {"prefix_len": 49152, "extend_len": 12288, "dp_size": 16, "chunk_size": 2048},
        "provenance": {"source_commit": SOURCE_COMMIT, "source_path": SOURCE_PATH,
                       "source_file_sha256": SOURCE_SHA256, "helper_sha256": HELPER_SHA256,
                       "source_lines": "1063-1119", "input_sha256": input_hash.hexdigest(),
                       "parser_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "acceptance": "parser_only_not_workload_or_hardware_validation",
    })
    return result


def self_test():
    rids = [f"measured-{i}" for i in range(16)]
    def event(i, ns=None, rank=0):
        return {"bid": i, "forward_step": i, "completed_ns": ns if ns is not None else
                [1000000000, 1100000000, 1300000000, 1600000000, 2000000000, 2500000000][i],
                "rank": rank, "requests": [{"rid": rid, "dp": d, "prefix": 49152 + i * 2048,
                                             "tokens": 2048} for d, rid in enumerate(rids)]}
    def log(events):
        return io.StringIO("noise\n" + "".join("logger " + MARKER + json.dumps(e) + "\n" for e in events))
    def parse(events, **kwargs):
        return parse_log(log(events), rids=rids, **kwargs)
    events = [event(i) for i in range(6)]
    result = parse(events)
    assert [s["duration_ms"] for s in result["steady_chunk_extend_samples"]] == [100, 200, 300, 400]
    assert result["steady_chunk_extend_ms"] == 250
    assert result["steady_chunk_extend_p95_ms"] == 400
    assert result["median_seconds"] == 0.25
    assert result["throughput_tokens_per_second"] == 131072
    assert parse([events[0], events[0]] + events[1:])["input_counts"]["duplicate_events"] == 1
    assert parse(events[1:])["steady_chunk_extend_sample_count"] == 3
    gap = [dict(e) for e in events]
    gap[2]["forward_step"] = 20
    assert parse(gap)["steady_chunk_extend_sample_count"] == 2
    reset = event(0)
    reset["forward_step"] = 99
    reset["requests"] = []
    assert parse(events[:1] + [reset] + events[1:])["steady_chunk_extend_sample_count"] == 3
    other = event(0, rank=1)
    assert parse([other] + events, stream="0")["steady_chunk_extend_sample_count"] == 4
    def fails(events, **kwargs):
        try:
            parse(events, **kwargs)
        except (ValueError, RuntimeError):
            return
        raise AssertionError("expected fail-closed input")
    fails([other] + events)
    fails([event(0), event(0, ns=123)])
    fails([event(0), event(1, ns=1)])
    malformed = event(0)
    malformed["requests"].append(malformed["requests"][0])
    fails([malformed])
    missing = event(0)
    missing["requests"] = []
    fails([missing])
    tail = event(5)
    tail["requests"][0]["tokens"] = 1024
    assert parse(events[:5] + [tail])["steady_chunk_extend_sample_count"] == 4
    remapped = event(2)
    remapped["requests"][0]["dp"], remapped["requests"][1]["dp"] = 1, 0
    assert parse(events[:2] + [remapped] + events[3:])["steady_chunk_extend_sample_count"] == 2
    named = []
    for e in events:
        e = json.loads(json.dumps(e))
        for i, r in enumerate(e["requests"]):
            r["rid"] = "run" + SUFFIX + str(i)
        named.append(e)
    assert parse_log(log(named), run_prefix="run")["steady_chunk_extend_sample_count"] == 4
    try:
        parse_log(log(named), run_prefix="warmup")
    except ValueError:
        pass
    else:
        raise AssertionError("must not infer measured namespace")
    text = Path(__file__).read_text()
    helper = text.split("# BEGIN VERBATIM PINNED HELPER (source lines 1063-1119; hash above)\n", 1)[1].split("# END VERBATIM PINNED HELPER", 1)[0]
    assert hashlib.sha256(helper.rstrip("\n").encode() + b"\n").hexdigest() == HELPER_SHA256
    print("self-test: PASS (exact 100/200/300/400 ms, median 250 ms; selection, reset, gaps, tails, rank clocks, dedup, malformed and provenance gates)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, help="explicit input server log (single-process or tagged merged log)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--rids", nargs="+", help="explicit measured request IDs")
    selection.add_argument("--rids-json", type=Path, help='JSON list or {"rids": [...]} measured manifest')
    selection.add_argument("--run-prefix", help="exact measured request_namespace (suffix appended by parser)")
    parser.add_argument("--stream", help="exact stream/rank identity; required for merged clocks")
    parser.add_argument("--stream-regex", help="logger-prefix regex with exactly one stream-ID capture")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.log or not (args.rids or args.rids_json or args.run_prefix):
        parser.error("--log and one of --rids / --rids-json / --run-prefix are required")
    try:
        rids = args.rids
        if args.rids_json:
            manifest = json.loads(args.rids_json.read_text())
            rids = manifest.get("rids") if isinstance(manifest, dict) else manifest
            if not isinstance(rids, list):
                raise ValueError("--rids-json requires a list or an object containing a rids list")
        with args.log.open("rb") as source:
            result = parse_log(source, rids=rids, run_prefix=args.run_prefix,
                               stream=args.stream, stream_regex=args.stream_regex)
        result["input_log"] = str(args.log.resolve())
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["steady_chunk_extend_sample_count"] else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
