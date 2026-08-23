"""Qwen3-Coder-480B-A35B-Instruct-FP8 multi-turn agentic serving benchmark
(live-server mode), the colocated half of the team's standard 480B contract.

Its MULTIHOST_LAUNCH forms its own launch group: recipe/qwen3_coder_480b_agentic_mh.sh
on 2 hosts, i.e. tp=16 / ep=16 / dp=4 -> attention_tp=4, prefix caching + HiCache
KV offload on, 1M server context. The CI runs this file on rank-0 against the
server the JobSet brought up; it skips unless a live server is provided.

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

First-run watch items: the tp=16 dp=4 DP-attention decode precompile --
attn_tp=4 is the shape PR #279 fixed. The weights are already on the shared
model-weights-rox PVC, so no hydration step is needed.
"""

import json
import os
import subprocess
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

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
# Fraction of the ACHIEVABLE prefix-cache hit rate a leg must reach. The
# achievable rate is set by the workload shape, not by the server: round 1 of a
# session is always a miss, so a 3-round session (the 16k default) tops out near
# 56% while a 25-round one reaches ~94%. A fixed threshold is wrong at one end or
# the other -- 90% fails every 16k leg, 30% would let a 128k leg silently fall to
# a third of its hit rate and still pass. Set MIN_PREFIX_HIT_RATE to gate on an
# absolute rate instead.
MIN_PREFIX_HIT_FRACTION = float(os.environ.get("MIN_PREFIX_HIT_FRACTION", "0.6"))
MIN_PREFIX_HIT_RATE = os.environ.get("MIN_PREFIX_HIT_RATE")

# The only egress from a TPU JobSet is the streamed pod log, so the results JSON is
# printed between these markers; .github/workflows/_tpu-test.yml lifts it out into
# a run artifact + the job summary.
RESULT_BEGIN = "=== TPU_BENCH_RESULT_JSON BEGIN ==="
RESULT_END = "=== TPU_BENCH_RESULT_JSON END ==="


def _achievable_prefix_hit_rate(config):
    """Prefix-cache hit rate a leg of this shape would get with a perfect cache.

    A session of R rounds re-sends its whole history every round, so with I new
    input tokens per round and O output tokens the prompt of round r is
    I + (r-1)*(I+O) and only the new I tokens can miss:

        prompt tokens = R*I + (I+O)*R*(R-1)/2      misses = R*I

    Summed over the leg's normal and long sessions (the long ones ramp at
    long_ramp_tokens). Returns None if the leg meta lacks the round counts.
    """
    out = config.get("turn_output_tokens")
    sessions = config.get("num_sessions")
    num_long = config.get("num_long_sessions")
    if None in (out, sessions, num_long):
        return None

    prompt_tokens = misses = 0
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
        misses += count * rounds * new_tokens
    if prompt_tokens <= 0:
        return None
    return 1.0 - misses / prompt_tokens


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
        # Only egress from the JobSet is this log; _tpu-test.yml lifts it out.
        print(RESULT_BEGIN, flush=True)
        print(json.dumps(summary, sort_keys=True, default=str), flush=True)
        print(RESULT_END, flush=True)
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
        """A hit rate far under what the leg's shape allows means the cache is off
        or thrashing, which invalidates every throughput number in the leg."""
        for leg in self._legs():
            rate = (leg.get("feature_derived") or {}).get("prefix_cache_hit_rate")
            self.assertIsNotNone(
                rate,
                f"{leg['tag']}: no prefix-cache hit rate recorded -- was the server "
                "launched with --enable-metrics?",
            )
            achievable = _achievable_prefix_hit_rate(leg.get("config") or {})
            if MIN_PREFIX_HIT_RATE is not None:
                floor, how = float(MIN_PREFIX_HIT_RATE), "MIN_PREFIX_HIT_RATE"
            else:
                self.assertIsNotNone(
                    achievable,
                    f"{leg['tag']}: leg meta lacks the round counts needed to "
                    "derive a prefix-hit floor; set MIN_PREFIX_HIT_RATE",
                )
                floor = MIN_PREFIX_HIT_FRACTION * achievable
                how = (
                    f"{MIN_PREFIX_HIT_FRACTION:.0%} of the {achievable:.1%} this "
                    "leg's round counts allow"
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
        are present and non-zero."""
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


if __name__ == "__main__":
    unittest.main()
