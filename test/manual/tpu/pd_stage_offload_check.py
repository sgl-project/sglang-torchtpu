"""Live check of the KV offload to decode DRAM behind a pipeline-parallel
prefill, against a running P/D pair through its router. HTTP only.

    python3 test/manual/tpu/pd_stage_offload_check.py --router http://<router>:8000 \\
        --prefill http://<prefill>:30000 --decode http://<decode>:30000 \\
        [--sessions 12] [--prompt-tokens 3400] [--expect-offload]

What it does, with greedy decoding throughout:

1. References: every session's second turn computed cold (cache flushed
   first), and again resumed from the prefill's own cache right after its
   first turn. A recalled prefix resumes at the same boundary as the second,
   so with --expect-offload an answer that differs from it is a failure.
2. First turns of all sessions, one after another. With a prefill pool smaller
   than the sessions together, early sessions leave the prefill's HBM and then
   its host RAM; with the offload on they are pushed to decode DRAM first.
3. Second turns of all sessions. A session the prefill no longer holds either
   recomputes (offload off, or nothing stored) or is recalled from decode DRAM
   (cached tokens reported). Each answer must name the session's passkey
   whenever the cold reference did; the passkey sits further back than the
   sliding window, so a recalled prefix that was not the computed one loses it. Answers that differ from the reference
   in any other way are counted, not failed: a cached prefix changes how the
   prompt is chunked.
4. The second turns again with the sessions in reverse, then as concurrent
   bursts, then the same burst from an empty cache (concurrent requests share
   a batch, which can change a greedy answer without any cache involved), and
   a request aborted mid-stream.

It prints one JSON line (STAGE_OFFLOAD_CHECK) with per-step counts and exits 1
on any failure. With --expect-offload it also fails when the oldest session's
second turn was not served from cache; choose --sessions so that the sessions
together exceed the prefill's HBM and host pools, or its own host tier can
serve that turn without any recall. The engines'
[dram-offload] log lines say how many branches were pushed, stored, offered
and recalled; read them from the engine logs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time

import requests

FILLER = (
    "The autumn wind rustled softly through the quiet valley while "
    "astronomers recorded distant starlight measurements. "
)


def passkey(index: int) -> str:
    return f"MARIGOLD-{8842 + 37 * index}"


def prefix(index: int, prompt_tokens: int) -> str:
    # About 19 tokens per filler sentence.
    return (
        f"IMPORTANT RECORD {index}: The secret passkey for vault {700 + index} is "
        f"{passkey(index)}. Remember this exact code.\n\n"
        + FILLER
        * max(1, prompt_tokens // 19)
    )


def turn(index: int, prompt_tokens: int, which: int) -> str:
    """The session's first prompt, or its second, which continues the first.

    A hybrid sliding-window model resumes a cached branch only where a window
    ends, so only a prompt that extends the whole first prompt can be recalled.
    The continuation does not repeat the passkey: its only copy stays further
    back than the window."""
    first = (
        prefix(index, prompt_tokens)
        + f"\nFirst verification: State the exact secret passkey for vault "
        f"{700 + index}: The passkey is"
    )
    if which == 1:
        return first
    return (
        first
        + " on file.\nSecond verification: Write only the exact secret passkey for "
        f"vault {700 + index}, nothing else. The passkey is"
    )


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=6).hexdigest()


def generate(base_url: str, prompt: str, max_new_tokens: int = 40, timeout=1800):
    started = time.monotonic()
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    output = response.json()
    if isinstance(output, list):
        output = output[0]
    meta = output["meta_info"]
    details = meta.get("cached_tokens_details") or {}
    return {
        "text": output["text"],
        "cached": int(meta.get("cached_tokens") or 0),
        "device": int(details.get("device", 0)),
        "host": int(details.get("host", 0)),
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "seconds": round(time.monotonic() - started, 3),
    }


def metric(base_url: str, name: str) -> float:
    """The sum of a Prometheus counter over its label sets; 0 if unreadable."""
    try:
        text = requests.get(base_url.rstrip("/") + "/metrics", timeout=30).text
    except requests.RequestException:
        return 0.0
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and line[len(name) : len(name) + 1] in ("{", " "):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return total


def spec_counters(base_url: str) -> tuple:
    return (
        metric(base_url, "sglang:spec_verify_calls_total"),
        metric(base_url, "sglang:generation_tokens_total"),
    )


def flush(urls) -> None:
    for url in urls:
        for _ in range(100):
            response = requests.post(url.rstrip("/") + "/flush_cache", timeout=120)
            if response.status_code == 200:
                break
            # Refused while transfers are in flight; it can be retried. A push
            # the decode had no use for settles only at the native deadline.
            time.sleep(2)
        else:
            raise RuntimeError(f"{url} did not flush its cache: {response.text[:200]}")


def abort_mid_stream(router: str, prompt: str) -> None:
    """Start a long generation and drop the connection after the first bytes."""
    with requests.post(
        router.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": 512, "temperature": 0.0},
            "stream": True,
        },
        stream=True,
        timeout=1800,
    ) as response:
        response.raise_for_status()
        for _ in response.iter_lines():
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--router", required=True)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=3400)
    parser.add_argument("--expect-offload", action="store_true")
    parser.add_argument(
        "--fillers",
        type=int,
        default=0,
        help="first turns of this many other sessions between the first and the "
        "second turns: time for the pushes to finish (a prefill pushes only while "
        "it serves) and load that evicts the sessions from the prefill",
    )
    parser.add_argument("--burst", type=int, default=4)
    parser.add_argument("--burst-rounds", type=int, default=3)
    args = parser.parse_args()
    engines = [args.prefill, args.decode]
    sessions = list(range(args.sessions))
    failures = []
    report = {"sessions": args.sessions, "prompt_tokens": args.prompt_tokens}

    def second_turns(order, label):
        served, differing, differing_warm = [], [], []
        verify_before, generated_before = spec_counters(args.decode)
        for i in order:
            got = generate(args.router, turn(i, args.prompt_tokens, 2))
            served.append(got)
            # Against the answer resumed from the prefill's own cache where
            # there is one: resuming changes some answers by itself.
            baseline = warm.get(i, reference[i])
            if passkey(i) in baseline and passkey(i) not in got["text"]:
                failures.append(
                    f"{label}: session {i} lost its passkey: {got['text']!r}"
                )
            if got["text"] != reference[i]:
                differing.append(i)
            # Comparable only when it resumes at the same boundary; a session
            # the prefill still holds part of resumes earlier.
            if got["cached"] == warm_cached.get(i) and got["text"] != warm[i]:
                differing_warm.append(i)
                # In the first pass the cached pages are the pushed ones or
                # the prefill's own; later passes also reuse KV computed by
                # an earlier, differently chunked turn.
                if args.expect_offload and label == "second_turns":
                    failures.append(
                        f"{label}: session {i} resumed from cache ({got['host']} host, "
                        f"{got['device']} device tokens) answered {got['text'][:80]!r}, "
                        f"resumed from the prefill's own cache it answers "
                        f"{warm[i][:80]!r}"
                    )
        verify_after, generated_after = spec_counters(args.decode)
        verify_calls = verify_after - verify_before
        report[label] = {
            # With a draft: tokens generated per verify call over this pass,
            # as the decode counts them. A recalled prefix whose draft KV was
            # wrong would draft worse than a local one.
            "spec_verify_calls": verify_calls,
            "generated_tokens": generated_after - generated_before,
            "accept_length": (
                round((generated_after - generated_before) / verify_calls, 3)
                if verify_calls
                else None
            ),
            # Per session, in the order served: what the turn resumed from and
            # a digest of its answer, to compare runs of the same sessions
            # (e.g. a recalled prefix here against the prefill's own host
            # tier in a run with the offload off and a larger host pool).
            "turns": [
                [i, g["cached"], g["host"], g["device"], _digest(g["text"])]
                for i, g in zip(order, served)
            ],
            "cached_requests": sum(1 for g in served if g["cached"]),
            "cached_tokens": sum(g["cached"] for g in served),
            "host_tokens": sum(g["host"] for g in served),
            "device_tokens": sum(g["device"] for g in served),
            "seconds": round(sum(g["seconds"] for g in served), 1),
            "differ_from_reference": differing,
            "differ_from_warm_reference": differing_warm,
            # Resumed at the warm boundary with pages from host RAM (the
            # prefill's own host tier, or recalled), and how many of those
            # differ from the answer resumed from the device tier.
            "host_sourced_at_warm_boundary": sum(
                1
                for i, g in zip(order, served)
                if g["host"] and g["cached"] == warm_cached.get(i)
            ),
            "host_sourced_differ_from_warm": [
                i for i, g in zip(order, served) if g["host"] and i in differing_warm
            ],
            "resumed_at_warm_boundary": sum(
                1 for i, g in zip(order, served) if g["cached"] == warm_cached.get(i)
            ),
        }
        return served

    # 1. Reference answers, each from an empty cache.
    reference, unanswered = {}, []
    for i in sessions:
        flush(engines)
        got = generate(args.router, turn(i, args.prompt_tokens, 2))
        reference[i] = got["text"]
        if got["cached"]:
            failures.append(f"reference of session {i} hit the cache ({got['cached']})")
        if passkey(i) not in got["text"]:
            # The model did not answer with the passkey even from a cold
            # cache: this session can only be compared with its reference.
            unanswered.append(i)
    # 1b. The same second turn resumed from the prefill's own cache, right
    # after its first turn. A recalled prefix resumes at the same boundary, so
    # its answer must equal this one; the cold answer need not (seen live on
    # hybrid models: resuming from any cached prefix changes some answers).
    warm, warm_cached = {}, {}
    for i in sessions:
        flush(engines)
        generate(args.router, turn(i, args.prompt_tokens, 1))
        resumed = generate(args.router, turn(i, args.prompt_tokens, 2))
        if resumed["cached"]:
            warm[i], warm_cached[i] = resumed["text"], resumed["cached"]
    report["warm_references"] = len(warm)
    report["warm_turns"] = [[i, warm_cached[i], _digest(warm[i])] for i in sorted(warm)]
    report["cold_turns"] = [[i, _digest(reference[i])] for i in sessions]
    report["warm_differs_from_cold"] = [i for i in warm if warm[i] != reference[i]]
    report["reference_prompt_tokens"] = got["prompt_tokens"]
    report["reference_without_passkey"] = unanswered
    flush(engines)

    # 2 and 3. First turns, then second turns in the same order.
    first = [generate(args.router, turn(i, args.prompt_tokens, 1)) for i in sessions]
    report["first_turns_seconds"] = round(sum(g["seconds"] for g in first), 1)
    for filler in range(args.fillers):
        generate(args.router, turn(10_000 + filler, args.prompt_tokens, 1))
    report["fillers"] = args.fillers
    served = second_turns(sessions, "second_turns")

    # 4. Again in reverse, then concurrently, then an abort.
    second_turns(list(reversed(sessions)), "second_turns_reversed")

    # Concurrent requests share a batch, which can change a greedy answer by
    # itself: run the burst with the cache flushed too, to tell that apart.
    def burst(label):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.burst) as pool:
            got = list(
                pool.map(
                    lambda i: (
                        i,
                        generate(args.router, turn(i, args.prompt_tokens, 2)),
                    ),
                    sessions,
                )
            )
        report[label] = {
            "cached_requests": sum(1 for _, g in got if g["cached"]),
            "differ_from_reference": [i for i, g in got if g["text"] != reference[i]],
            "without_passkey": [i for i, g in got if passkey(i) not in g["text"]],
        }
        return got

    rounds = [burst(f"burst_{index}") for index in range(args.burst_rounds)]
    flush(engines)
    # Seen live with the offload off: one session's greedy answer changes in a
    # burst whatever the cache holds. Only a passkey the cold burst kept counts.
    cold = dict(burst("burst_cold_cache"))
    for index, got in enumerate(rounds):
        for i, answer in got:
            if passkey(i) in cold[i]["text"] and passkey(i) not in answer["text"]:
                failures.append(
                    f"burst {index}: session {i} lost its passkey "
                    f"(cached {answer['cached']}): {answer['text'][:120]!r}"
                )
    abort_mid_stream(args.router, turn(0, args.prompt_tokens, 1))
    after_abort = generate(args.router, turn(1, args.prompt_tokens, 2))
    if 1 not in unanswered and passkey(1) not in after_abort["text"]:
        failures.append(f"after an abort session 1 answered {after_abort['text']!r}")

    # Session 0 is the oldest when its second turn comes: with a prefill pool
    # smaller than the sessions together it has left the prefill by then.
    if args.expect_offload and not served[0]["cached"]:
        failures.append(
            "the oldest session's second turn was not served from cache: "
            "it was not recalled from decode DRAM"
        )
    report["failures"] = failures
    print("STAGE_OFFLOAD_CHECK " + json.dumps(report), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
