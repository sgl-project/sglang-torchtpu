"""Shared live-server machinery for TPU serving-benchmark tests.

A benchmark test file is a thin shell around build_bench_test() (or the
matrix wrapper build_matrix_test()): the file itself carries only what the CI
scanners read via AST -- register_tpu_ci(suite=...) and the literal
MULTIHOST_LAUNCH dict -- plus one factory call. Nothing in the test body
binds it to a particular TPU cluster: the body only ever sees BASE_URL, and
the cluster assignment lives entirely in the registration metadata. To move
a test to another cluster, change its suite= (which pipeline schedules it,
onto which slice shape) and give MULTIHOST_LAUNCH the weights key that
cluster's scanner reads ("weights": a subdir of the external cluster's
gcsfuse bucket; "weights_pvc" + "hosts": the in-project PVC pipeline). Both
scanners ignore keys they don't read, so a file may carry both and flip
clusters by editing only its suite.

Consumers:
  * the benchmark matrix cells (test_tpu_bench_*: GPT-OSS-120B /
    Qwen3-Coder-480B across serving recipes and workloads) via
    build_matrix_test(); their sweep harness is
    benchmark/tpu/recipe/bench_matrix/run_sweep.sh, and two files sharing a
    (suite, launch_script, weights) triple share one server;
  * test_tpu_qwen3_coder_480b_multihost_agentic (the 480B nightly) via
    build_bench_test() over its own harness
    (recipe/qwen3_coder_480b/nightly/run_nightly.sh).

WHAT IS ASSERTED, and what is not: throughput is *recorded*, never gated
(the TPU slice is shared, so a tok/s floor would be a flake generator, and
regression tracking is the dashboard's job); what is gated is that the run
is trustworthy at all:
  * every planned leg finished,
  * prefix caching actually hit where the workload makes hits reachable
    (multi-turn agentic; a single-turn 8k1k session cannot re-see a prefix,
    so there the rate is recorded but not gated),
  * the server admitted the work without retracting or aborting requests,
  * the draft model actually drafted, on the recipes that claim it (a spec config
    that silently fell back to plain decode would benchmark the wrong thing).
"""

import json
import os
import subprocess
import time
import unittest
import urllib.request

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase, dump_metric
from sglang.test.tpu import emit_result_payload

# run_suite requires every file under test/registered to carry a registration;
# same convention as debug_utils/comparator/testing_helpers.py.
register_tpu_ci(
    est_time=0,
    suite="external-nightly-2x2x2",
    nightly=True,
    disabled="helper module, no tests",
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 4))
MATRIX_SWEEP_SCRIPT = os.path.join(
    REPO_ROOT, "benchmark", "tpu", "recipe", "bench_matrix", "run_sweep.sh"
)

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)

# --- The matrix cells' slice of the time-budget chain (see config.sh's "THE
# TIME-BUDGET CHAIN"). TIMEOUT_PER_FILE_BUDGET is what every cell's
# MULTIHOST_LAUNCH declares literally as timeout_per_file (the CI scanner
# reads it via AST, so the files cannot compute it); the matrix ratchet
# asserts the literals match this constant and that the chain holds.
TIMEOUT_PER_FILE_BUDGET = 8100
# Re-checking a server the JobSet already gated healthy.
HEALTH_RECHECK_BUDGET = 600
# Headroom after the sweep for summarize + digest + result-JSON printing.
RESULT_EGRESS_MARGIN = 300
SWEEP_TIMEOUT_DEFAULT = (
    TIMEOUT_PER_FILE_BUDGET - HEALTH_RECHECK_BUDGET - RESULT_EGRESS_MARGIN
)

HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", str(HEALTH_RECHECK_BUDGET)))
# Fraction of the REACHABLE prefix-cache hit rate a gated leg must reach. The
# reachable rate is set by the workload shape AND the DP count: the first D
# rounds of a session can never hit, so a 6-round session at dp=2 tops out
# near 51% while the same session against one shared cache would reach 76%. A
# fixed threshold is wrong at one end or the other -- 90% fails every 16k
# leg, 30% would let a 128k leg silently fall to a third of its hit rate and
# still pass. Set MIN_PREFIX_HIT_RATE to gate on an absolute rate instead.
MIN_PREFIX_HIT_FRACTION = float(os.environ.get("MIN_PREFIX_HIT_FRACTION", "0.6"))
MIN_PREFIX_HIT_RATE = os.environ.get("MIN_PREFIX_HIT_RATE")

# Dashboard series keys for the matrix cells: the canonical HF ids, not the
# mount paths the recipes serve from -- restaging the weights must not split
# a trend in two.
MODEL_LABELS = {
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "qwen3_coder_480b": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "muse_spark": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
}
# The cosmetic half of the series identity, free to change (unlike the id
# derived below). Same vocabulary as the nightly-perf MODEL_CONFIGs.
MODEL_DISPLAY_NAMES = {
    "gpt_oss_120b": "GPT-OSS 120B",
    "qwen3_coder_480b": "Qwen3-Coder-480B",
    "muse_spark": "Muse Spark",
}
SPEC_RECIPES = {"1host_pc_spec_kvoff", "pd_2p2d_pc_spec_kvoff"}


def _direction(field: str) -> str:
    """Which way is an improvement, so a regression check that assumes
    higher-is-better does not read a TTFT drop as a win."""
    if "throughput" in field:
        return "higher_is_better"
    if any(s in field for s in ("ttft", "itl", "latency", "duration")):
        return "lower_is_better"
    return "informational"


def _print_workload_digest(summary) -> None:
    """One short line per leg, printed BEFORE the fragile JSON payload.

    The payload is the fragile half of the egress (see `emit_result_payload`)
    and a truncated payload has taken the verdict and the failing assertion
    with it before. These lines are a few dozen bytes each and carry exactly
    the fields the gates read, so a triage never depends on the payload
    surviving the log stream."""
    for leg in summary.get("legs") or []:
        derived = leg.get("feature_derived") or {}
        metrics = leg.get("metrics") or {}
        print(
            f"[workload] {leg.get('tag')} status={leg.get('status')} "
            f"prefix_cache_hit_rate={derived.get('prefix_cache_hit_rate')} "
            f"min_spec_accept_length={derived.get('min_spec_accept_length')} "
            f"spec_verify_calls={derived.get('spec_verify_calls')} "
            f"kv_offload_written={derived.get('kv_offload_written_tokens')} "
            f"kv_offload_reloaded={derived.get('kv_offload_reloaded_tokens')} "
            f"output_throughput={metrics.get('output_throughput')}",
            flush=True,
        )


def _dp_schedulers(server_info):
    """Number of DP schedulers, each with its own radix cache: ``dp_size``
    replicas times ``attn_dp_size`` attention-DP groups. Since --attn-dp-size,
    ``--dp-size N --enable-dp-attention`` reports dp_size=1, attn_dp_size=N.
    None when server_info records no dp_size."""
    dp_size = server_info.get("dp_size")
    if dp_size is None:
        return None
    return int(dp_size) * int(server_info.get("attn_dp_size") or 1)


def _reachable_prefix_hit_rate(config, dp_size):
    """Prefix-cache hit rate a leg of this shape can reach at this DP count.

    A session of R rounds re-sends its whole history every round, so with I new
    input tokens per round and O output tokens the prompt of round r is
    I + (r-1)*(I+O) and the leg sends

        prompt tokens = R*I + (I+O)*R*(R-1)/2

    How much of that a PERFECT cache can serve is set by the DP count, not by
    the shape alone. Every DP rank owns its own radix tree and dispatch is
    round_robin (no sglang load_balance_method is cache-aware), so round r
    lands on the rank that last saw round r-D and can reuse only THAT round's
    stored sequence -- its prompt plus its reply, i.e. (I+O)*(r-D) tokens:

        cached tokens = (I+O)*(R-D)*(R-D+1)/2

    the same (R-D)(R-D+1)/(R(R-1)) fraction of the single-cache ideal that
    the 480B nightly's config.sh derives when it picks COLOCATED_DP. It is
    ZERO whenever R <= D; D=1 recovers the single-cache ideal,
    (I+O)*R*(R-1)/2.

    Exact at concurrency 1, where a session's rounds strictly alternate
    ranks. At higher concurrency round_robin interleaves sessions and some
    land consecutive rounds on one rank, so this is a LOWER bound there --
    safe for a floor, which is all it is used for.

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


def _matrix_spec_metrics_url(bench_recipe: str, base_url: str) -> str:
    """Where the spec gauges live: the engine that decodes.

    For P/D that is the decode leader (hosts[2] of the slice), never the
    mini-LB -- the LB serves no /metrics, and a request-level probe through
    it is not a reliable oracle on this P/D setup. For 1-host recipes the
    one server is the decode.
    """
    if bench_recipe.startswith("pd_"):
        # Engine leaders follow PHYSICAL host order (serve_group.sh's
        # rendezvous writes the map on this pod), not pod-index order.
        try:
            with open("/tmp/bench_matrix_phys_hosts") as f:
                hosts = [line.strip() for line in f if line.strip()]
        except OSError:
            hosts = [
                h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h
            ]
        if len(hosts) >= 3:
            port = os.environ.get("PD_DECODE_PORT", "30002")
            return f"http://{hosts[2]}:{port}"
    return base_url


def _read_spec_accept_length(metrics_url: str) -> float:
    """Max sglang:spec_accept_length across ranks; 0.0 when spec never ran."""
    with urllib.request.urlopen(metrics_url.rstrip("/") + "/metrics", timeout=60) as r:
        text = r.read().decode()
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("sglang:spec_accept_length") and " " in line
    ]
    return max(values, default=0.0)


def build_bench_test(
    *,
    class_name: str,
    label: str,
    model_label: str,
    series_id: str,
    display_name: str,
    sweep_script: str,
    sweep_env: dict,
    run_root_default: str,
    profile: str,
    sweep_timeout_default: int,
    gate_prefix_hits: bool,
    spec_metrics_url_fn=None,
    spec_contract: dict | None = None,
    metric_namespace: str | None = None,
):
    """Return a live-server benchmark TestCase class (cluster-agnostic).

    The class health-checks BASE_URL, drives `sweep_script` (with `sweep_env`
    merged over os.environ) to a summary.json, then gates trustworthiness and
    dumps the numbers for the dashboard.

    class_name           unittest-visible name (the dashboard keys on the
                         test FILE name, so this is cosmetic).
    label                short log prefix for the [label] lines.
    model_label          dashboard model series key (canonical HF id).
    series_id            stable `id` label -- the dashboard's series join key.
                         NEVER renamed; a forced rename is recorded once in
                         .github/tpu-ci/dashboard_aliases.json.
    display_name         the cosmetic half of the identity, free to change.
    sweep_script         absolute path to the harness entry point.
    sweep_env            harness-specific env (BENCH_MODEL=..., MODE=..., ...).
    run_root_default     where summary.json lands unless RUN_ROOT is set.
    profile              sweep profile, passed as PROFILE.
    sweep_timeout_default  overridable via SWEEP_TIMEOUT; must clear the
                         harness's worst-case leg schedule and leave the
                         file's timeout room for health + result egress.
    gate_prefix_hits     gate hit rate against the reachable floor (multi-turn
                         workloads); the rate is always *recorded*.
    spec_metrics_url_fn  base_url -> metrics endpoint of the DECODING engine;
                         non-None declares the recipe claims spec decoding
                         and adds the accept-length gate.
    spec_contract        expected per-leg EAGLE server configuration and
                         minimum scheduler accept length. The supported keys
                         are algorithm, draft_model_suffix, num_steps,
                         eagle_topk, num_draft_tokens, and min_accept_length.
    metric_namespace     optional prefix for metric names, for files whose
                         legs alone don't disambiguate (the matrix prefixes
                         the workload). None keeps bare leg tags -- an
                         existing file must keep its metric names or its
                         dashboard trend splits.
    """
    sweep_timeout = int(os.environ.get("SWEEP_TIMEOUT", str(sweep_timeout_default)))

    @unittest.skipUnless(
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "live server not provided (the CI JobSet launches it)",
    )
    class BenchTest(CustomTestCase):
        base_url = BASE_URL
        summary = None

        @classmethod
        def setUpClass(cls):
            cls._wait_healthy()
            cls.summary = cls._run_sweep()

        @classmethod
        def _wait_healthy(cls):
            """The JobSet's rank-0 already gated on health; this recheck keeps
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
                            f"[{label}] server healthy after "
                            f"{time.perf_counter() - start:.0f}s",
                            flush=True,
                        )
                        return
                except requests.RequestException:
                    pass
                if time.perf_counter() - last_note >= 60:
                    last_note = time.perf_counter()
                    print(
                        f"[{label}] waiting for live server at {cls.base_url} "
                        f"({time.perf_counter() - start:.0f}s elapsed)",
                        flush=True,
                    )
                time.sleep(5)
            raise TimeoutError(
                f"{cls.base_url}/health did not return 200 within {HEALTH_TIMEOUT}s"
            )

        @classmethod
        def _run_sweep(cls):
            run_root = os.environ.get("RUN_ROOT") or run_root_default
            env = {
                **os.environ,
                **sweep_env,
                "PROFILE": profile,
                "BASE_URL": cls.base_url,
                "RUN_ROOT": run_root,
                "HEALTH_TIMEOUT": str(HEALTH_TIMEOUT),
            }
            print(
                f"[{label}] running the {profile} sweep against "
                f"{cls.base_url} (deadline {sweep_timeout}s)",
                flush=True,
            )
            proc = subprocess.run(
                ["bash", sweep_script],
                cwd=os.path.dirname(sweep_script),
                env=env,
                timeout=sweep_timeout,
            )
            summary_path = os.path.join(run_root, "summary.json")
            md_path = os.path.join(run_root, "summary.md")
            if os.path.exists(md_path):
                with open(md_path) as f:
                    print(f.read(), flush=True)
            if not os.path.exists(summary_path):
                raise AssertionError(
                    f"the sweep produced no summary.json "
                    f"({os.path.basename(sweep_script)} rc={proc.returncode}); "
                    "see the log above"
                )
            with open(summary_path) as f:
                summary = json.load(f)
            _print_workload_digest(summary)
            emit_result_payload(summary)
            return summary

        def _legs(self):
            """Legs the sweep actually attempted (skipped-by-design excluded)."""
            return [
                leg
                for leg in self.summary["legs"]
                if not leg["status"].startswith("skipped")
            ]

        def test_every_leg_completed(self):
            """A timed-out or crashed leg fails the run: partial benchmark
            output is indistinguishable from a benchmark nobody looked at."""
            legs = self._legs()
            self.assertGreater(len(legs), 0, "the sweep attempted no legs")
            bad = [(leg["tag"], leg["status"]) for leg in legs if leg["status"] != "ok"]
            self.assertEqual(bad, [], f"legs did not complete: {bad}")
            missing = [leg["tag"] for leg in legs if not leg.get("metrics")]
            self.assertEqual(missing, [], f"legs produced no metrics: {missing}")

        def test_prefix_cache_hit_rate_reported(self):
            """Every leg must RECORD a hit rate (the metrics endpoint is part
            of the contract); multi-turn workloads are additionally gated.

            A rate far under what the leg can reach means the cache is off or
            thrashing, which invalidates every throughput number in the leg.
            Gated against the REACHABLE rate, not the single-cache ideal:
            those differ by a third at dp=2, and a floor derived from the
            ideal can sit above what the deployment permits -- an assertion
            no correct server can pass. Green here means "the cache does
            what the routing allows", NOT "prefix caching is fully
            exploited": only cache-aware routing (outside this repo) closes
            the rest.
            """
            for leg in self._legs():
                rate = (leg.get("feature_derived") or {}).get("prefix_cache_hit_rate")
                self.assertIsNotNone(
                    rate,
                    f"{leg['tag']}: no prefix-cache hit rate recorded -- was the "
                    "server launched with --enable-metrics?",
                )
                if not gate_prefix_hits:
                    continue
                if MIN_PREFIX_HIT_RATE is not None:
                    floor, how = float(MIN_PREFIX_HIT_RATE), "MIN_PREFIX_HIT_RATE"
                else:
                    # Read dp back off the run: a gate that hardcoded the DP
                    # count would go stale the next time the parallelism
                    # moves -- silently, as an unreachable floor.
                    config = leg.get("config") or {}
                    dp_size = _dp_schedulers(leg.get("server_info") or {})
                    self.assertIsNotNone(
                        dp_size,
                        f"{leg['tag']}: leg meta records no dp_size, so the "
                        "reachable prefix-hit ceiling cannot be derived; set "
                        "MIN_PREFIX_HIT_RATE",
                    )
                    reachable = _reachable_prefix_hit_rate(
                        config=config, dp_size=dp_size
                    )
                    self.assertIsNotNone(
                        reachable,
                        f"{leg['tag']}: leg meta lacks the round counts needed "
                        "to derive a prefix-hit floor; set MIN_PREFIX_HIT_RATE",
                    )
                    self.assertGreater(
                        reachable,
                        0.0,
                        f"{leg['tag']}: no round can reach a cached prefix at "
                        f"dp={dp_size} -- round r lands on the rank that last "
                        f"saw round r-{dp_size}, so a session must outlast the "
                        "DP count. Raise the round count (TARGET_CONTEXT / "
                        "LONG_TARGET_CONTEXT) or lower the DP; the leg "
                        "measures nothing as it is.",
                    )
                    # P/D: the mini-LB pairs every request with a uniformly
                    # random prefill engine and each engine caches on its own,
                    # so a round lands on the engine holding its previous round
                    # with probability 1/N. A round that lands elsewhere still
                    # hits that engine's older copy of the session, so 1/N of
                    # the ideal is a floor, not an estimate.
                    prefill_engines = int(config.get("pd_prefill_engines") or 1)
                    reachable /= prefill_engines
                    routing = (
                        f" across {prefill_engines} randomly paired prefill engines"
                        if prefill_engines > 1
                        else ""
                    )
                    floor = MIN_PREFIX_HIT_FRACTION * reachable
                    how = (
                        f"{MIN_PREFIX_HIT_FRACTION:.0%} of the {reachable:.1%} "
                        f"this leg's round counts reach at dp={dp_size}{routing}"
                    )
                self.assertGreaterEqual(
                    rate,
                    floor,
                    f"{leg['tag']}: prefix-cache hit rate {rate:.1%} is below "
                    f"{floor:.1%} ({how})",
                )

        def test_server_admitted_the_work(self):
            """Retractions and aborts mean the KV pool could not hold the
            admitted batch, so the leg's numbers include recompute."""
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

        if spec_metrics_url_fn is not None:

            def test_spec_decoding_active(self):
                """The recipe claims spec; prove drafting actually ran (a
                silent fallback to plain decode would still pass every other
                gate while benchmarking the wrong configuration). Oracle: the
                decoding engine's own sglang:spec_accept_length gauge after
                the sweep -- it stays 0.0 unless spec verify actually ran, and
                needs no request through the mini-LB."""
                url = spec_metrics_url_fn(self.base_url)
                accept_length = _read_spec_accept_length(url)
                print(
                    f"[{label}] spec_accept_length={accept_length} ({url})",
                    flush=True,
                )
                self.assertGreater(
                    accept_length,
                    0.0,
                    f"sglang:spec_accept_length is 0 on {url}: speculative "
                    "decoding never ran during the sweep",
                )

        if spec_contract is not None:

            def test_spec_decoding_active_and_accepting(self):
                """Prove every leg ran the requested speculative configuration
                and every DP scheduler accepted at least one draft token."""
                expected_fields = {
                    "speculative_num_steps": spec_contract["num_steps"],
                    "speculative_eagle_topk": spec_contract["eagle_topk"],
                    "speculative_num_draft_tokens": spec_contract["num_draft_tokens"],
                }
                min_accept_length = float(spec_contract["min_accept_length"])
                for leg in self._legs():
                    server_info = leg.get("server_info") or {}
                    self.assertEqual(
                        str(server_info.get("speculative_algorithm")).upper(),
                        str(spec_contract["algorithm"]).upper(),
                        f"{leg['tag']}: server resolved the wrong speculative "
                        "algorithm",
                    )
                    self.assertTrue(
                        str(
                            server_info.get("speculative_draft_model_path", "")
                        ).endswith(spec_contract["draft_model_suffix"]),
                        f"{leg['tag']}: wrong or missing speculative draft model",
                    )
                    for field, expected in expected_fields.items():
                        self.assertEqual(
                            server_info.get(field),
                            expected,
                            f"{leg['tag']}: {field} does not match the test contract",
                        )

                    derived = leg.get("feature_derived") or {}
                    self.assertGreater(
                        derived.get("spec_verify_calls") or 0.0,
                        0.0,
                        f"{leg['tag']}: spec_verify_calls_total did not increase; "
                        "the leg ran without speculative verification",
                    )
                    accept_lengths = (
                        derived.get("spec_accept_length_by_scheduler") or []
                    )
                    dp_size = _dp_schedulers(server_info) or 0
                    self.assertGreater(
                        dp_size,
                        0,
                        f"{leg['tag']}: server_info records no positive dp_size",
                    )
                    self.assertEqual(
                        len(accept_lengths),
                        dp_size,
                        f"{leg['tag']}: expected one accept length per DP scheduler "
                        f"(dp_size={dp_size}), got {accept_lengths}",
                    )
                    for scheduler_index, accept_length in enumerate(accept_lengths):
                        self.assertGreater(
                            float(accept_length),
                            min_accept_length,
                            f"{leg['tag']}: scheduler {scheduler_index} accept "
                            f"length {accept_length} did not exceed "
                            f"{min_accept_length}",
                        )

        def test_throughput_recorded(self):
            """Not a performance gate -- proves the numbers the dashboard
            ingests are present and non-zero, then dumps them for it."""
            for leg in self._legs():
                metrics = leg.get("metrics") or {}
                derived = leg.get("derived") or {}
                self.assertGreater(
                    metrics.get("output_throughput") or 0.0,
                    0.0,
                    f"{leg['tag']}: no output throughput",
                )
                print(
                    f"[{label}] {leg['tag']}: "
                    f"out {metrics.get('output_throughput'):.1f} tok/s, "
                    f"prompt {derived.get('prompt_throughput', 0.0):.1f} tok/s, "
                    f"TTFT p50 {metrics.get('median_ttft_ms'):.0f} ms, "
                    f"ITL p50 {metrics.get('median_itl_ms'):.1f} ms",
                    flush=True,
                )
                # Publish only legs that finished with no retract/abort
                # recompute. A failed prefix-hit gate still reds the job but
                # does not stop the dump; the trend reader sees the red run
                # either way.
                counters = leg.get("feature_counters") or {}
                clean = leg["status"] == "ok" and not any(
                    counters.get(k)
                    for k in (
                        "sglang:num_retracted_requests_total",
                        "sglang:num_aborted_requests_total",
                    )
                )
                if spec_contract is not None:
                    features = leg.get("feature_derived") or {}
                    server_info = leg.get("server_info") or {}
                    accept_lengths = (
                        features.get("spec_accept_length_by_scheduler") or []
                    )
                    expected_fields = {
                        "speculative_num_steps": spec_contract["num_steps"],
                        "speculative_eagle_topk": spec_contract["eagle_topk"],
                        "speculative_num_draft_tokens": spec_contract[
                            "num_draft_tokens"
                        ],
                    }
                    clean = clean and (
                        str(server_info.get("speculative_algorithm")).upper()
                        == str(spec_contract["algorithm"]).upper()
                        and str(
                            server_info.get("speculative_draft_model_path", "")
                        ).endswith(spec_contract["draft_model_suffix"])
                        and all(
                            server_info.get(field) == expected
                            for field, expected in expected_fields.items()
                        )
                        and (features.get("spec_verify_calls") or 0.0) > 0.0
                        and len(accept_lengths) == (_dp_schedulers(server_info) or 0)
                        and len(accept_lengths) > 0
                        and all(
                            float(accept_length)
                            > float(spec_contract["min_accept_length"])
                            for accept_length in accept_lengths
                        )
                    )
                if clean:
                    self._dump_leg_metrics(leg)

        def _dump_leg_metrics(self, leg):
            """One [METRIC] line per number, same label contract as
            bench_serving_runner. Only the per-leg output_throughput carries
            kind=performance plus `workload`, which is what routes it to the
            dashboard's performance tab; everything else is kind=info --
            archived forever (a field not recorded tonight cannot be
            back-filled once the job log expires) and shown in the point's
            details panel, with the rows whose trailing field matches one of
            the dashboard's dropdown families still charted. The leg tag is
            in the metric name because several legs share a run, and the
            dashboard keys a night's point on (test_file, metric, model)."""
            tag = (
                leg["tag"]
                if metric_namespace is None
                else f"{metric_namespace}.{leg['tag']}"
            )
            numbers = {
                **(leg.get("metrics") or {}),
                **{f"derived.{k}": v for k, v in (leg.get("derived") or {}).items()},
                **{
                    f"feature.{k}": v
                    for k, v in (leg.get("feature_derived") or {}).items()
                },
            }
            # bench_serving cannot account input tokens for a multi-turn
            # trace, so its total_throughput degenerates to exactly
            # output_throughput and its input_throughput to 0. Chart the
            # corrected total (server-side prompt counter + output tokens,
            # computed by summarize.py) under that name instead, and drop
            # the two degenerate fields.
            numbers.pop("input_throughput", None)
            numbers.pop("total_throughput", None)
            real_total = numbers.pop("derived.total_throughput", None)
            if real_total is not None:
                numbers["total_throughput"] = real_total
            for field, value in sorted(numbers.items()):
                if not isinstance(value, (int, float)):
                    continue
                labels = {
                    "model": model_label,
                    "direction": _direction(field),
                    "id": series_id,
                }
                if field == "output_throughput":
                    labels["kind"] = "performance"
                    labels["workload"] = tag
                    labels["headline"] = "1"
                    labels["display_name"] = display_name
                else:
                    labels["kind"] = "info"
                    labels["workload_name"] = tag
                dump_metric(f"{tag}.{field}", value, labels=labels)

    BenchTest.__qualname__ = BenchTest.__name__ = class_name
    return BenchTest


def build_matrix_test(*, bench_model: str, bench_recipe: str, workload: str):
    """Return the TestCase class for one matrix (model, recipe, workload) cell.

    The series id is DERIVED from the cell coordinates rather than spelled out
    per file, so a new cell cannot forget one or collide with a sibling. That
    makes the coordinates load-bearing: renaming bench_model/bench_recipe/
    workload renames the id and splits the dashboard trend, so a forced rename
    is recorded in .github/tpu-ci/dashboard_aliases.json like any other.
    """
    spec_expected = bench_recipe in SPEC_RECIPES
    cell = "-".join(
        part.replace("_", "-") for part in (bench_model, bench_recipe, workload)
    )
    return build_bench_test(
        class_name=f"TestBench_{bench_model}_{bench_recipe}_{workload}",
        label=f"{bench_model}/{bench_recipe}/{workload}",
        model_label=MODEL_LABELS[bench_model],
        series_id=f"perf-{cell}",
        display_name=(
            f"{MODEL_DISPLAY_NAMES[bench_model]} "
            f"{bench_recipe.replace('_', '-')} {workload}"
        ),
        sweep_script=MATRIX_SWEEP_SCRIPT,
        sweep_env={
            "BENCH_MODEL": bench_model,
            "BENCH_RECIPE": bench_recipe,
            "WORKLOAD": workload,
        },
        run_root_default=f"/tmp/bench-{bench_model}-{bench_recipe}-{workload}",
        profile=os.environ.get("BENCH_PROFILE", "ci"),
        sweep_timeout_default=SWEEP_TIMEOUT_DEFAULT,
        # A single-turn 8k1k session never re-sees its prefix, so the hit-rate
        # gate only applies to the multi-turn agentic workload.
        gate_prefix_hits=workload == "agentic",
        spec_metrics_url_fn=(
            (lambda base_url: _matrix_spec_metrics_url(bench_recipe, base_url))
            if spec_expected
            else None
        ),
        # Several cells share a model, so the workload disambiguates the
        # metric names.
        metric_namespace=workload,
    )


if __name__ == "__main__":
    # The consumer files assign factory-built classes into their own __main__
    # and run there; this satisfies the dead-TestCase lint for the class
    # defined in this helper (nothing is collected here).
    unittest.main()
