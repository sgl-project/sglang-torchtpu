"""Unit tests for the TPU bench harness's server-side observability (no TPU).

metrics_probe.py turns two /metrics scrapes into a per-leg delta, and
summarize.py joins those with bench_serving's per-request arrays. Both run
after a multi-hour benchmark, so a parsing slip there costs a whole nightly's
analysis; these cases pin the arithmetic on synthetic inputs:

  * Prometheus histogram parsing (label-set summing, the split label, ``le``
    normalization) and the per-leg bucket delta / quantile estimate;
  * the decode-role diff not raising prefill-only warnings;
  * the per-context and long-session per-round breakdowns, including the
    fallbacks when the trace shape does not match the outputs.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3))
NIGHTLY = os.path.join(
    REPO, "benchmark", "tpu", "recipe", "qwen3_coder_480b", "nightly"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_bench_{name}", os.path.join(NIGHTLY, f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("metrics_probe")
summarize = _load("summarize")


def _exposition(latency_buckets, latency_sum, stage_counts):
    """A /metrics page with kv_transfer_latency_ms on two TP ranks and a
    per-stage histogram on two stages."""
    lines = [
        "# HELP sglang:kv_transfer_latency_ms Histogram of KV cache transfer latency in ms.",
        "# TYPE sglang:kv_transfer_latency_ms histogram",
    ]
    for rank in ("0", "1"):
        for le, cum in latency_buckets:
            lines.append(
                f'sglang:kv_transfer_latency_ms_bucket{{engine_type="prefill",'
                f'le="{le}",tp_rank="{rank}"}} {cum}'
            )
        lines.append(
            f'sglang:kv_transfer_latency_ms_sum{{engine_type="prefill",'
            f'tp_rank="{rank}"}} {latency_sum}'
        )
        lines.append(
            f'sglang:kv_transfer_latency_ms_count{{engine_type="prefill",'
            f'tp_rank="{rank}"}} {latency_buckets[-1][1]}'
        )
    for stage, (count, total) in stage_counts.items():
        lines.append(
            f'sglang:per_stage_req_latency_seconds_bucket{{le="+Inf",'
            f'stage="{stage}"}} {count}'
        )
        lines.append(
            f'sglang:per_stage_req_latency_seconds_sum{{stage="{stage}"}} {total}'
        )
        lines.append(
            f'sglang:per_stage_req_latency_seconds_count{{stage="{stage}"}} {count}'
        )
    # Not in _HISTOGRAMS; must be ignored.
    lines.append('sglang:grammar_schema_count_bucket{le="1.0"} 7.0')
    lines.append("sglang:num_requests_total 12.0")
    return "\n".join(lines) + "\n"


class TestMetricsProbeHistograms(CustomTestCase):
    def test_parse_sums_label_sets_and_splits_stage(self):
        text = _exposition(
            [("10.0", 1.0), ("100.0", 3.0), ("+Inf", 4.0)],
            latency_sum=250.0,
            stage_counts={"prefill_forward": (5.0, 2.5), "decode_waiting": (2, 0.1)},
        )
        hist = probe._parse_histograms(text)
        self.assertEqual(
            set(hist),
            {"sglang:kv_transfer_latency_ms", "sglang:per_stage_req_latency_seconds"},
        )
        latency = hist["sglang:kv_transfer_latency_ms"]["all"]
        # Two TP ranks summed.
        self.assertEqual(latency["count"], 8.0)
        self.assertEqual(latency["sum"], 500.0)
        self.assertEqual(latency["buckets"], {"10.0": 2.0, "100.0": 6.0, "+Inf": 8.0})
        self.assertEqual(latency["label_sets"], 2)
        stages = hist["sglang:per_stage_req_latency_seconds"]
        self.assertEqual(set(stages), {"prefill_forward", "decode_waiting"})
        self.assertEqual(stages["prefill_forward"]["count"], 5.0)
        self.assertEqual(stages["prefill_forward"]["label_sets"], 1)
        # The counter parser is unchanged by histogram lines.
        self.assertEqual(
            probe._parse_prometheus(text), {"sglang:num_requests_total": 12.0}
        )

    def test_le_normalization(self):
        self.assertEqual(probe._norm_le("5"), "5.0")
        self.assertEqual(probe._norm_le("5.0"), "5.0")
        self.assertEqual(probe._norm_le("+Inf"), "+Inf")

    def test_quantile_interpolates_within_bucket(self):
        bounds = [10.0, 100.0, float("inf")]
        counts = [0.0, 10.0, 10.0]
        self.assertAlmostEqual(probe._quantile(bounds, counts, 0.5), 55.0)
        self.assertAlmostEqual(probe._quantile(bounds, counts, 0.9), 91.0)
        # Landing in +Inf reports the top finite bound (a floor).
        self.assertEqual(probe._quantile(bounds, [0.0, 1.0, 10.0], 0.99), 100.0)
        self.assertIsNone(probe._quantile(bounds, [0.0, 0.0, 0.0], 0.5))

    def test_diff_is_post_minus_pre(self):
        pre = probe._parse_histograms(
            _exposition(
                [("10.0", 1.0), ("100.0", 1.0), ("+Inf", 1.0)],
                latency_sum=5.0,
                stage_counts={"prefill_forward": (1.0, 1.0)},
            )
        )
        post = probe._parse_histograms(
            _exposition(
                [("10.0", 1.0), ("100.0", 3.0), ("+Inf", 4.0)],
                latency_sum=255.0,
                stage_counts={"prefill_forward": (1.0, 1.0), "decode_waiting": (2, 1)},
            )
        )
        delta = probe._diff_histograms(pre, post)
        latency = delta["sglang:kv_transfer_latency_ms"]["all"]
        self.assertEqual(latency["count"], 6.0)
        self.assertEqual(latency["sum"], 500.0)
        self.assertAlmostEqual(latency["mean"], 500.0 / 6.0)
        self.assertEqual(latency["top_bucket"], 100.0)
        self.assertEqual(latency["above_top_bucket"], 2.0)
        # No new observations: omitted. New series: reported in full.
        stages = delta["sglang:per_stage_req_latency_seconds"]
        self.assertNotIn("prefill_forward", stages)
        self.assertEqual(stages["decode_waiting"]["count"], 2.0)

    def test_diff_drops_series_after_restart(self):
        pre = {
            "f": {"all": {"buckets": {"1.0": 5.0, "+Inf": 9.0}, "sum": 9, "count": 9}}
        }
        post = {
            "f": {
                "all": {"buckets": {"1.0": 1.0, "+Inf": 10.0}, "sum": 10, "count": 10}
            }
        }
        self.assertEqual(probe._diff_histograms(pre, post), {})

    def _snap(self, prompt, cached, histograms):
        return {
            "metrics": {
                "sglang:prompt_tokens_total": prompt,
                "sglang:cached_tokens_total": cached,
            },
            "histograms": histograms,
            "server": {},
        }

    def _args(self, role):
        return SimpleNamespace(
            role=role,
            min_hit_rate=0.2,
            min_spec_accept_length=1.0,
            expect_context=0,
            expect_concurrency=0,
        )

    def test_decode_role_skips_prompt_checks(self):
        empty = self._snap(0.0, 0.0, {})
        unified = probe._diff(empty, empty, self._args("unified"))
        self.assertTrue(any("no prompt tokens" in w for w in unified["warnings"]))
        decode = probe._diff(empty, empty, self._args("decode"))
        self.assertEqual(decode["warnings"], [])
        self.assertEqual(decode["role"], "decode")

    def test_transfer_headline_and_top_bucket_warning(self):
        # Two TP ranks, each observing its own shard of 4 requests.
        def transfer(count, total, over, mb_total):
            return {
                "sglang:kv_transfer_latency_ms": {
                    "all": {
                        "buckets": {"5000.0": count - over, "+Inf": count},
                        "sum": total,
                        "count": count,
                        "label_sets": 2,
                    }
                },
                "sglang:kv_transfer_total_mb": {
                    "all": {
                        "buckets": {"+Inf": count},
                        "sum": mb_total,
                        "count": count,
                        "label_sets": 2,
                    }
                },
                "sglang:per_stage_req_latency_seconds": {
                    "prefill_transfer_kv_cache": {
                        "buckets": {"+Inf": count},
                        "sum": 0.03 * count,
                        "count": count,
                    }
                },
            }

        pre = self._snap(100.0, 0.0, transfer(0.0, 0.0, 0.0, 0.0))
        post = self._snap(1100.0, 900.0, transfer(8.0, 52000.0, 2.0, 800.0))
        delta = probe._diff(pre, post, self._args("prefill"))
        self.assertEqual(delta["derived"]["kv_transfer_ranks"], 2)
        self.assertEqual(delta["derived"]["kv_transfers"], 4.0)
        self.assertEqual(delta["derived"]["kv_transfer_latency_ms_mean"], 6500.0)
        # The window above includes the forward; the exposed part is the stage.
        self.assertEqual(delta["derived"]["kv_transfer_exposed_ms_mean"], 30.0)
        self.assertEqual(delta["derived"]["kv_transfer_mb_mean"], 100.0)
        self.assertEqual(delta["derived"]["kv_transfer_request_mb_mean"], 200.0)
        self.assertEqual(delta["derived"]["prefix_cache_hit_rate"], 0.9)
        self.assertTrue(any("exceeded the top bucket" in w for w in delta["warnings"]))


def _leg_metrics(input_lens, cached=None, errors=None):
    n = len(input_lens)
    record = {
        "completed": n,
        "duration": 10.0,
        "total_output_tokens": 10 * n,
        "input_lens": input_lens,
        "output_lens": [10] * n,
        "ttfts": [0.001 * x for x in input_lens],  # 1 ms per 1k tokens -> x us
        "itls": [[0.01, 0.03]] * n,
        "errors": errors or [""] * n,
    }
    if cached is not None:
        record["cached_tokens"] = cached
    return record


# 1 long session x 3 rounds, then 2 normal sessions x 2 rounds.
_CONFIG = {
    "num_long_sessions": "1",
    "num_sessions": "3",
    "rounds_per_long_session": "3",
    "rounds_per_normal_session": "2",
}
_INPUTS = [8000, 40000, 100000, 3000, 6000, 3000, 6000]
_CACHED = [0, 7900, 39900, 0, 2900, 0, 2900]


class TestSummarizeBreakdowns(CustomTestCase):
    def test_long_session_rounds(self):
        derived = summarize._derive(_leg_metrics(_INPUTS, _CACHED), {}, _CONFIG)
        self.assertEqual(derived["per_request_layout"], "ok")
        rounds = derived["long_session_rounds"]
        self.assertEqual([r["round"] for r in rounds], [1, 2, 3])
        self.assertEqual(
            [r["input_tokens_mean"] for r in rounds], [8000.0, 40000.0, 100000.0]
        )
        self.assertEqual(rounds[2]["uncached_tokens_mean"], 60100.0)
        self.assertAlmostEqual(rounds[1]["ttft_ms_mean"], 40000.0)
        self.assertAlmostEqual(rounds[0]["itl_ms_mean"], 20.0)
        self.assertEqual(rounds[0]["failed"], 0)

    def test_by_context_buckets(self):
        derived = summarize._derive(_leg_metrics(_INPUTS, _CACHED), {}, _CONFIG)
        by_ctx = {b["ctx_le"]: b for b in derived["by_context"]}
        self.assertEqual(set(by_ctx), {8192, 65536, 131072})
        self.assertEqual(by_ctx[8192]["requests"], 5)
        self.assertAlmostEqual(by_ctx[131072]["cached_fraction"], 0.399)
        self.assertAlmostEqual(by_ctx[8192]["itl_ms_p50"], 20.0)

    def test_failed_rounds_counted_not_averaged(self):
        errors = ["", "", "boom", "", "", "", ""]
        derived = summarize._derive(_leg_metrics(_INPUTS, _CACHED, errors), {}, _CONFIG)
        last = derived["long_session_rounds"][2]
        self.assertEqual(last["failed"], 1)
        self.assertNotIn("ttft_ms_mean", last)
        self.assertNotIn(131072, [b["ctx_le"] for b in derived["by_context"]])

    def test_shape_mismatch_disables_round_mapping(self):
        derived = summarize._derive(_leg_metrics(_INPUTS[:-1]), {}, _CONFIG)
        self.assertTrue(derived["per_request_layout"].startswith("unavailable"))
        self.assertNotIn("long_session_rounds", derived)
        self.assertIn("by_context", derived)

    def test_constant_input_lens_are_not_bucketed(self):
        flat = [5000] * 3 + [3000] * 4
        derived = summarize._derive(_leg_metrics(flat), {}, _CONFIG)
        self.assertNotIn("by_context", derived)
        self.assertIn("by_context_unavailable", derived)

    def test_end_to_end_summary(self):
        with tempfile.TemporaryDirectory() as root:
            pd = os.path.join(root, "pd")
            os.makedirs(pd)
            tag = "c4-long1"
            with open(os.path.join(pd, f"{tag}.jsonl"), "w") as f:
                f.write(json.dumps(_leg_metrics(_INPUTS, _CACHED)) + "\n")
            with open(os.path.join(pd, f"{tag}.meta.json"), "w") as f:
                json.dump({"config": _CONFIG}, f)
            with open(os.path.join(pd, f"{tag}.status"), "w") as f:
                f.write("ok\n")
            transfer = {
                "count": 7.0,
                "sum": 700.0,
                "mean": 100.0,
                "p50": 90.0,
                "p90": 150.0,
                "p99": 200.0,
            }
            with open(os.path.join(pd, f"{tag}.metrics.delta.json"), "w") as f:
                json.dump(
                    {
                        "counters": {"sglang:prompt_tokens_total": 166000.0},
                        "derived": {},
                        "warnings": [],
                        "histograms": {
                            "sglang:kv_transfer_latency_ms": {"all": transfer},
                            "sglang:per_stage_req_latency_seconds": {
                                "prefill_transfer_kv_cache": dict(transfer, mean=0.03)
                            },
                        },
                    },
                    f,
                )
            with open(os.path.join(pd, f"{tag}.decode.metrics.delta.json"), "w") as f:
                json.dump(
                    {
                        "counters": {},
                        "derived": {},
                        "warnings": ["decode-only warning"],
                        "histograms": {
                            "sglang:per_stage_req_latency_seconds": {
                                "decode_waiting": dict(transfer, mean=0.25)
                            }
                        },
                    },
                    f,
                )
            out_json = os.path.join(root, "summary.json")
            out_md = os.path.join(root, "summary.md")
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(NIGHTLY, "summarize.py"),
                    "--run-root",
                    root,
                    "--json-out",
                    out_json,
                    "--md-out",
                    out_md,
                ],
                check=True,
                capture_output=True,
            )
            with open(out_json) as f:
                summary = json.load(f)
            leg = summary["legs"][0]
            self.assertEqual(summary["schema_version"], 4)
            self.assertEqual(leg["warnings"], [])
            self.assertEqual(leg["decode"]["warnings"], ["decode-only warning"])
            self.assertIn("sglang:kv_transfer_latency_ms", leg["server_histograms"])
            self.assertEqual(len(leg["derived"]["long_session_rounds"]), 3)
            self.assertNotIn("ttfts", leg["metrics"])
            with open(out_md) as f:
                md = f.read()
            self.assertIn("### Server-side breakdown", md)
            # xfer window ms mean/p50/p99, then exposed xfer ms
            self.assertIn("| 100/90/200 | 30 |", md)
            self.assertIn("| 250 |", md)  # decode wait mean, ms
            self.assertIn("### Latency by context length", md)
            self.assertIn("### Long sessions, turn by turn", md)


if __name__ == "__main__":
    unittest.main()
