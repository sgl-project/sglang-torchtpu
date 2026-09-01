"""Qwen3-Coder-480B-A35B-Instruct-FP8 multi-turn agentic serving benchmark
(live-server mode), the colocated half of the team's standard 480B contract.

Its MULTIHOST_LAUNCH forms its own launch group: recipe/qwen3_coder_480b_agentic_mh.sh
on 2 hosts, i.e. tp=16 / ep=16 / dp=2 -> attention_tp=8, prefix caching + HiCache
KV offload on, 1M server context. The CI runs this file on rank-0 against the
server the JobSet brought up; it skips unless a live server is provided. DP size
is config.sh's COLOCATED_DP and is overridable, so nothing here hardcodes it --
the prefix-cache gate reads dp_size back off the run.

WHAT IS ASSERTED, and what is not. Throughput is *recorded*, never gated: the TPU
slice is shared, so a tok/s floor would be a flake generator, and regression
tracking is the dashboard's job (summary.json is emitted for it). What is gated is
that the run is trustworthy at all:
  * every planned leg finished (a leg that times out or crashes fails the test --
    a silently missing row is the worst outcome for a benchmark)
  * prefix caching actually hit (a multi-turn replay that reports ~0% means the
    radix cache is off or thrashing, which invalidates every number in the run)
  * the server admitted the work without retracting or aborting requests
Those are the invariants that make a number mean what it says.

Watch items: the tp=16 dp=2 DP-attention decode precompile at attn_tp=8 (PR #279
fixed the attn_tp=4 shape this ran at before COLOCATED_DP dropped to 2). The
weights are already on the shared model-weights-rox PVC, so no hydration step is
needed.
"""

import json
import os
import subprocess
import sys
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase, dump_metric

register_tpu_ci(est_time=1800, suite="nightly-tpu-multihost", nightly=True)

# Own launch group: the 480B agentic recipe + the shared ROX weights volume.
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/qwen3_coder_480b_agentic_mh.sh",
    "weights_pvc": "model-weights-rox",
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
RECIPE_DIR = os.path.join(
    REPO_ROOT, "benchmark", "tpu", "recipe", "qwen3_coder_480b", "nightly"
)

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "600"))
# Leave headroom under the JobSet's `timeout 3600` per file so the summary still
# gets written and printed when the sweep itself runs long. HEALTH_TIMEOUT is
# spent before this and comes out of the same 3600, hence 600 + 2700 + 300 spare.
# It must also clear the `ci` profile's own worst case -- 4 legs x
# MAX_LEG_SECONDS=420 = 1680s -- or the sweep is killed mid-run and the whole
# file fails on a budget, not on a result. (1500 did not: that was sized when
# the template gave a file 1800s and the sweep was described as two legs.)
SWEEP_TIMEOUT = int(os.environ.get("SWEEP_TIMEOUT", "2700"))
PROFILE = os.environ.get("PROFILE", "ci")
# Fraction of the REACHABLE prefix-cache hit rate a leg must reach. The reachable
# rate is set by the workload shape AND the DP count: the first D rounds of a
# session can never hit, so a 6-round session at dp=2 tops out near 51% while the
# same session against one shared cache would reach 76%. A fixed threshold is
# wrong at one end or the other -- 90% fails every 16k leg, 30% would let a 128k
# leg silently fall to a third of its hit rate and still pass. Set
# MIN_PREFIX_HIT_RATE to gate on an absolute rate instead.
MIN_PREFIX_HIT_FRACTION = float(os.environ.get("MIN_PREFIX_HIT_FRACTION", "0.6"))
MIN_PREFIX_HIT_RATE = os.environ.get("MIN_PREFIX_HIT_RATE")

# The only egress from a TPU JobSet is the streamed pod log, so the results JSON is
# printed between these markers; .github/workflows/_tpu-test.yml lifts it out into
# a run artifact + the job summary.
RESULT_BEGIN = "=== TPU_BENCH_RESULT_JSON BEGIN ==="
RESULT_END = "=== TPU_BENCH_RESULT_JSON END ==="

# Dashboard series key. The canonical HF id, not the mount path the recipe serves
# from -- remounting the weights elsewhere must not split the trend in two.
MODEL_LABEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"


def _direction(field: str) -> str:
    """Which way is an improvement, so a regression check that assumes
    higher-is-better does not read a TTFT drop as a win."""
    if "throughput" in field:
        return "higher_is_better"
    if any(s in field for s in ("ttft", "itl", "latency", "duration")):
        return "lower_is_better"
    return "informational"


def _dump_leg_metrics(leg):
    """One [METRIC] line per number, same label contract as bench_serving_runner.

    Only the per-leg output_throughput carries `workload`, which is what routes it
    to the dashboard's performance tab; everything else the leg measured is dumped
    as kind=bench_detail -- stored in the dashboard's archive forever (a field not
    recorded tonight cannot be back-filled once the job log expires). The dashboard
    still charts the bench_detail rows whose trailing field matches one of its
    dropdown families (total/decode throughput, TTFT/TPOT/ITL). The leg tag is in
    the metric name because several legs share a run, and the dashboard keys a
    night's point on (test_file, metric_name, model).
    """
    tag = leg["tag"]
    numbers = {
        **(leg.get("metrics") or {}),
        **{f"derived.{k}": v for k, v in (leg.get("derived") or {}).items()},
    }
    # bench_serving cannot account input tokens for a multi-turn replay, so its
    # total_throughput degenerates to exactly output_throughput and its
    # input_throughput to 0 (see nightly/README.md "Reading a result"). The
    # dashboard picks its Total-throughput charts by the trailing field name, so
    # dumping the degenerate value under `total_throughput` plotted a Total chart
    # identical to the Decode one. Chart the corrected total (server-side prompt
    # counter + output tokens, computed by summarize.py) under that name instead,
    # and drop the two degenerate fields -- they carry no information to archive.
    numbers.pop("input_throughput", None)
    numbers.pop("total_throughput", None)
    real_total = numbers.pop("derived.total_throughput", None)
    if real_total is not None:
        numbers["total_throughput"] = real_total
    for field, value in sorted(numbers.items()):
        if not isinstance(value, (int, float)):
            continue
        labels = {"model": MODEL_LABEL, "direction": _direction(field)}
        if field == "output_throughput":
            labels["workload"] = tag
            labels["headline"] = "1"
        else:
            labels["kind"] = "bench_detail"
            labels["workload_name"] = tag
        dump_metric(f"{tag}.{field}", value, labels=labels)


def _print_workload_digest(summary) -> None:
    """One short line per workload, printed BEFORE the results payload.

    "workload" matches the metric labels; summary.json's schema still says legs.

    The payload is the fragile half of the egress (see `_emit_result_json`), and
    on 2026-08-26 and 08-28 the job log ended on the RESULT_BEGIN marker -- so the
    gsm8k-style verdict, the failing assertion and the TIMINGS record were all
    lost behind it and the red run said nothing about why. These lines are a few
    dozen bytes each and carry exactly the fields the gates below read, so a
    triage never depends on the payload surviving.
    """
    for workload in summary.get("legs") or []:
        derived = workload.get("feature_derived") or {}
        metrics = workload.get("metrics") or {}
        print(
            f"[workload] {workload.get('tag')} status={workload.get('status')} "
            f"prefix_cache_hit_rate={derived.get('prefix_cache_hit_rate')} "
            f"output_throughput={metrics.get('output_throughput')}",
            flush=True,
        )


def _emit_result_json(summary) -> None:
    """Print the summary between the markers, shaped to survive the log stream.

    The only egress from a TPU JobSet is the streamed pod log, and containerd caps
    a single log record at 16 KiB -- a summary emitted as one long line is split
    into CRI partial records and has twice taken the rest of the job log with it.
    `indent` keeps every line short without changing what the consumer sees:
    _tpu-test.yml's awk rejoins the block with newlines, and pretty-printed JSON
    only ever breaks BETWEEN tokens, so the rejoined text parses identically.
    Chunking a single line at a fixed width would not -- it would drop a newline
    inside a string literal and corrupt the payload.
    """
    try:
        blob = json.dumps(summary, sort_keys=True, default=str, indent=1)
    except (TypeError, ValueError) as exc:
        # sort_keys raises on mixed-type keys. A lost payload is recoverable from
        # the digest above; an exception here happens in setUpClass and would take
        # down every gate in the file, reporting nothing about the sweep itself.
        print(f"{RESULT_BEGIN}\n{RESULT_END}", flush=True)
        print(f"[480b-agentic] summary not serializable, payload skipped: {exc!r}")
        return
    print(RESULT_BEGIN, flush=True)
    print(blob, flush=True)
    print(RESULT_END, flush=True)


def _reachable_prefix_hit_rate(config, dp_size):
    """Prefix-cache hit rate a leg of this shape can reach at this DP count.

    A session of R rounds re-sends its whole history every round, so with I new
    input tokens per round and O output tokens the prompt of round r is
    I + (r-1)*(I+O) and the leg sends

        prompt tokens = R*I + (I+O)*R*(R-1)/2

    How much of that a PERFECT cache can serve is set by the DP count, not by the
    shape alone. Every DP rank owns its own radix tree and dispatch is round_robin
    (no sglang load_balance_method is cache-aware), so round r lands on the rank
    that last saw round r-D and can reuse only THAT round's stored sequence -- its
    prompt plus its reply, i.e. (I+O)*(r-D) tokens:

        cached tokens = (I+O)*(R-D)*(R-D+1)/2

    the same (R-D)(R-D+1)/(R(R-1)) fraction of the single-cache ideal that
    config.sh derives when it picks COLOCATED_DP. It is ZERO whenever R <= D --
    the state PR #574 found at dp=4 against 3-round sessions, where every c1 leg
    measured 0.0%. D=1 recovers the single-cache ideal, (I+O)*R*(R-1)/2.

    Exact at concurrency 1, where a session's rounds strictly alternate ranks. At
    higher concurrency round_robin interleaves sessions and some land consecutive
    rounds on one rank, so this is a LOWER bound there (measured c16-long0 53.6%
    against 50.5% predicted) -- safe for a floor, which is all it is used for.

    Summed over the leg's normal and long sessions (the long ones ramp at
    long_ramp_tokens). Returns None if the leg meta lacks the round counts.
    """
    out = config.get("turn_output_tokens")
    sessions = config.get("num_sessions")
    num_long = config.get("num_long_sessions")
    if None in (out, sessions, num_long):
        return None

    prompt_tokens = cached_tokens = 0
    for count, new_tokens_key, rounds_key in (
        (
            int(sessions) - int(num_long),
            "turn_input_tokens",
            "rounds_per_normal_session",
        ),
        (int(num_long), "long_ramp_tokens", "rounds_per_long_session"),
    ):
        if count <= 0:
            continue
        new_tokens, rounds = config.get(new_tokens_key), config.get(rounds_key)
        if None in (new_tokens, rounds):
            return None
        new_tokens, rounds = int(new_tokens), int(rounds)
        growth = new_tokens + int(out)
        prompt_tokens += count * (
            rounds * new_tokens + growth * rounds * (rounds - 1) // 2
        )
        reuses = max(0, rounds - dp_size)
        cached_tokens += count * growth * reuses * (reuses + 1) // 2
    if prompt_tokens <= 0:
        return None
    return cached_tokens / prompt_tokens


@unittest.skipUnless(
    os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
    "multi-host live-server not provided (the JobSet launches the server)",
)
class TestTpuQwen3Coder480bMultihostAgentic(CustomTestCase):
    """Agentic sweep against the 2-host tp=16 dp=4 Qwen3-Coder-480B server."""

    base_url = BASE_URL
    summary = None

    @classmethod
    def setUpClass(cls):
        cls._wait_healthy()
        cls.summary = cls._run_sweep()

    @classmethod
    def _wait_healthy(cls):
        """The JobSet's rank-0 already gated on /health_generate; this recheck keeps
        the file runnable by hand against any live server."""
        start = time.perf_counter()
        last_note = start
        deadline = start + HEALTH_TIMEOUT
        while time.perf_counter() < deadline:
            try:
                if (
                    requests.get(cls.base_url + "/health", timeout=10).status_code
                    == 200
                ):
                    print(
                        f"[480b-agentic] server healthy after "
                        f"{time.perf_counter() - start:.0f}s",
                        flush=True,
                    )
                    return
            except requests.RequestException:
                pass
            if time.perf_counter() - last_note >= 60:
                last_note = time.perf_counter()
                print(
                    f"[480b-agentic] waiting for live server at {cls.base_url} "
                    f"({time.perf_counter() - start:.0f}s elapsed)",
                    flush=True,
                )
            time.sleep(5)
        raise TimeoutError(
            f"{cls.base_url}/health did not return 200 within {HEALTH_TIMEOUT}s"
        )

    @classmethod
    def _run_sweep(cls):
        """Drive run_nightly.sh against the live server and return its summary.json.

        MODE=external means "sweep this endpoint, launch nothing" -- the JobSet owns
        the server here. The layout is the colocated one (tp=16 dp=4), so the results
        are labelled as such; P/D is a different JobSet shape and is disabled.
        """
        run_root = os.environ.get("RUN_ROOT") or "/tmp/qwen3-coder-480b-agentic"
        env = {
            **os.environ,
            "PROFILE": PROFILE,
            "MODE": "external",
            "BASE_URL": cls.base_url,
            "EXTERNAL_MODE_LABEL": "colocated",
            "RUN_ROOT": run_root,
            # The JobSet already warmed the server by serving the health probe; a
            # second compile-bucket pass would eat the file's whole budget.
            "SKIP_WARMUP": "1",
            "HEALTH_TIMEOUT": str(HEALTH_TIMEOUT),
        }
        print(
            f"[480b-agentic] running the {PROFILE} sweep against {cls.base_url} "
            f"(deadline {SWEEP_TIMEOUT}s)",
            flush=True,
        )
        proc = subprocess.run(
            ["bash", os.path.join(RECIPE_DIR, "run_nightly.sh")],
            cwd=RECIPE_DIR,
            env=env,
            timeout=SWEEP_TIMEOUT,
        )
        summary_path = os.path.join(run_root, "summary.json")
        md_path = os.path.join(run_root, "summary.md")
        if os.path.exists(md_path):
            with open(md_path) as f:
                print(f.read(), flush=True)
        if not os.path.exists(summary_path):
            raise AssertionError(
                f"the sweep produced no summary.json (run_nightly.sh rc="
                f"{proc.returncode}); see the log above"
            )
        with open(summary_path) as f:
            summary = json.load(f)
        _print_workload_digest(summary)
        _emit_result_json(summary)
        return summary

    def _legs(self):
        """Legs the sweep actually attempted (skipped-by-design ones excluded)."""
        return [
            leg
            for leg in self.summary["legs"]
            if not leg["status"].startswith("skipped")
        ]

    def test_every_leg_completed(self):
        """A timed-out or crashed leg fails the run: partial benchmark output is
        indistinguishable from a benchmark nobody looked at."""
        legs = self._legs()
        self.assertGreater(len(legs), 0, "the sweep attempted no legs")
        bad = [(leg["tag"], leg["status"]) for leg in legs if leg["status"] != "ok"]
        self.assertEqual(bad, [], f"legs did not complete: {bad}")
        missing = [leg["tag"] for leg in legs if not leg.get("metrics")]
        self.assertEqual(missing, [], f"legs produced no metrics: {missing}")

    def test_prefix_caching_hit(self):
        """A hit rate far under what the leg can reach means the cache is off or
        thrashing, which invalidates every throughput number in the leg.

        Gated against the REACHABLE rate, not the single-cache ideal. Those differ
        by a third at dp=2, and a floor derived from the ideal can sit above what
        the deployment permits -- an assertion no correct server can pass, which is
        what reddened this file on 2026-08-30 and 08-31 (c1-long1 measured 30.2%
        against a 36.9% floor, with 30.7% reachable). Green here therefore means
        "the cache does what round_robin over dp ranks allows", NOT "prefix caching
        is being exploited": at dp=2 a 6-round session leaves a third of the ideal
        on the table no matter how well the cache behaves, and only cache-aware
        routing (sgl-model-gateway, not this repo) closes that.
        """
        for leg in self._legs():
            rate = (leg.get("feature_derived") or {}).get("prefix_cache_hit_rate")
            self.assertIsNotNone(
                rate,
                f"{leg['tag']}: no prefix-cache hit rate recorded -- was the server "
                "launched with --enable-metrics?",
            )
            if MIN_PREFIX_HIT_RATE is not None:
                floor, how = float(MIN_PREFIX_HIT_RATE), "MIN_PREFIX_HIT_RATE"
            else:
                # Read back off the run: COLOCATED_DP is overridable, and a gate
                # that hardcoded the DP count would go stale the next time the
                # parallelism moves -- silently, as an unreachable floor.
                dp_size = (leg.get("server_info") or {}).get("dp_size")
                self.assertIsNotNone(
                    dp_size,
                    f"{leg['tag']}: leg meta records no dp_size, so the reachable "
                    "prefix-hit ceiling cannot be derived; set MIN_PREFIX_HIT_RATE",
                )
                reachable = _reachable_prefix_hit_rate(
                    config=leg.get("config") or {}, dp_size=int(dp_size)
                )
                self.assertIsNotNone(
                    reachable,
                    f"{leg['tag']}: leg meta lacks the round counts needed to "
                    "derive a prefix-hit floor; set MIN_PREFIX_HIT_RATE",
                )
                self.assertGreater(
                    reachable,
                    0.0,
                    f"{leg['tag']}: no round can reach a cached prefix at "
                    f"dp={dp_size} -- round r lands on the rank that last saw "
                    f"round r-{dp_size}, so a session must outlast the DP count. "
                    "Raise the round count (TARGET_CONTEXT / LONG_TARGET_CONTEXT) "
                    "or lower COLOCATED_DP; the leg measures nothing as it is.",
                )
                floor = MIN_PREFIX_HIT_FRACTION * reachable
                how = (
                    f"{MIN_PREFIX_HIT_FRACTION:.0%} of the {reachable:.1%} this "
                    f"leg's round counts reach at dp={dp_size}"
                )
            self.assertGreaterEqual(
                rate,
                floor,
                f"{leg['tag']}: prefix-cache hit rate {rate:.1%} is below "
                f"{floor:.1%} ({how})",
            )

    def test_server_admitted_the_work(self):
        """Retractions and aborts mean the KV pool could not hold the admitted batch,
        so the leg's numbers include recompute of already-generated tokens."""
        for leg in self._legs():
            counters = leg.get("feature_counters") or {}
            for key in (
                "sglang:num_retracted_requests_total",
                "sglang:num_aborted_requests_total",
            ):
                self.assertEqual(
                    counters.get(key, 0.0),
                    0.0,
                    f"{leg['tag']}: {key} = {counters.get(key)} "
                    f"(warnings: {leg.get('warnings')})",
                )

    def test_throughput_recorded(self):
        """Not a performance gate -- just proves the numbers the dashboard ingests
        are present and non-zero, then dumps them for it."""
        for leg in self._legs():
            metrics = leg.get("metrics") or {}
            derived = leg.get("derived") or {}
            self.assertGreater(
                metrics.get("output_throughput") or 0.0,
                0.0,
                f"{leg['tag']}: no output throughput",
            )
            print(
                f"[480b-agentic] {leg['tag']}: "
                f"out {metrics.get('output_throughput'):.1f} tok/s, "
                f"prompt {derived.get('prompt_throughput', 0.0):.1f} tok/s, "
                f"TTFT p50 {metrics.get('median_ttft_ms'):.0f} ms, "
                f"ITL p50 {metrics.get('median_itl_ms'):.1f} ms",
                flush=True,
            )
            # Publish only legs that finished and had no retract/abort recompute.
            # A failed prefix-hit gate still reds the job but does not stop the
            # dump; the trend reader sees the red run either way.
            counters = leg.get("feature_counters") or {}
            clean = leg["status"] == "ok" and not any(
                counters.get(k)
                for k in (
                    "sglang:num_retracted_requests_total",
                    "sglang:num_aborted_requests_total",
                )
            )
            if clean:
                _dump_leg_metrics(leg)


class TestDumpLegMetricsShape(CustomTestCase):
    """Host-side (no server): the dashboard contract of _dump_leg_metrics.

    Guards the regression where the performance tab's Total-throughput charts
    duplicated the Decode ones (observed through 2026-08-31): bench_serving
    cannot account input tokens for a multi-turn replay, so its
    total_throughput equals output_throughput exactly, and dumping it under the
    family name charted the same series twice. The dashboard selects a chart
    family by the metric name's trailing field, so the corrected derived total
    must be the ONLY name ending in total_throughput, and a leg without the
    server-side prompt counter must dump no total at all rather than the
    degenerate one.
    """

    _LEG = {
        "tag": "c1-long1",
        "metrics": {
            "output_throughput": 35.4,
            "total_throughput": 35.4,  # degenerate: input tokens unaccounted
            "input_throughput": 0.0,  # degenerate: input tokens unaccounted
            "median_itl_ms": 25.0,
        },
        "derived": {"prompt_throughput": 1448.6, "total_throughput": 1484.0},
    }

    def _dumped(self, leg):
        records = {}
        module = sys.modules[__name__]
        real = module.dump_metric
        module.dump_metric = lambda name, value, labels=None: records.__setitem__(
            name, (value, labels)
        )
        try:
            _dump_leg_metrics(leg)
        finally:
            module.dump_metric = real
        return records

    def test_total_chart_gets_the_corrected_total(self):
        records = self._dumped(self._LEG)
        total_names = [n for n in records if n.endswith("total_throughput")]
        self.assertEqual(total_names, ["c1-long1.total_throughput"])
        self.assertEqual(records["c1-long1.total_throughput"][0], 1484.0)
        self.assertNotIn("c1-long1.input_throughput", records)

    def test_total_withheld_when_prompt_counter_missing(self):
        records = self._dumped({**self._LEG, "derived": {}})
        self.assertNotIn("c1-long1.total_throughput", records)
        self.assertEqual(records["c1-long1.output_throughput"][0], 35.4)


if __name__ == "__main__":
    unittest.main()
