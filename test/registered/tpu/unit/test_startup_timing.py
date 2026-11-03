"""Hardware-free tests for the cold-start precompile timing gate."""

import unittest
from unittest.mock import Mock, patch

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.evals import startup_timing
from sglang.test.tpu.evals.startup_timing import (
    HEADLINE_METRIC,
    extract_startup_metrics,
    measurement_failures,
    startup_timing_runner,
)

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


def _startup_time(**overrides):
    st = {
        "load_weight": 120.5,
        "kv_cache_allocation": 2.0,
        "scheduler_e2e": 610.0,
        "tokenizer_e2e": 640.0,
        "cuda_graph": {"prefill": 300.0, "decode": 150.0, "target_verify": 0.0},
        "compile": {
            "schema_version": 1,
            "steps": [
                {
                    "step_key": "a",
                    "component": "target",
                    "phase": "extend",
                    "duration_seconds": 40.0,
                },
                {
                    "step_key": "b",
                    "component": "target",
                    "phase": "extend",
                    "duration_seconds": 35.0,
                },
                {
                    "step_key": "c",
                    "component": "target",
                    "phase": "decode",
                    "duration_seconds": 20.0,
                },
            ],
        },
    }
    st.update(overrides)
    return st


def _response(payload):
    resp = Mock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestExtractStartupMetrics(unittest.TestCase):
    def test_headline_is_scheduler_window_minus_load_and_kv(self) -> None:
        m = extract_startup_metrics(_startup_time())
        self.assertEqual(m[HEADLINE_METRIC], 610.0 - 120.5 - 2.0)
        self.assertEqual(m["cuda_graph.total_s"], 450.0)
        self.assertEqual(m["cuda_graph.prefill"], 300.0)
        self.assertEqual(m["load_weight"], 120.5)

    def test_headline_does_not_depend_on_graph_phases(self) -> None:
        # TPU's decode-capture hook returns before the async compiles finish,
        # so the graph phases under-count; the headline must not move with them.
        m = extract_startup_metrics(_startup_time(cuda_graph={"decode": 1.0}))
        self.assertEqual(m[HEADLINE_METRIC], 610.0 - 120.5 - 2.0)

    def test_compile_steps_are_counted_and_grouped(self) -> None:
        m = extract_startup_metrics(_startup_time())
        self.assertEqual(m["compile_steps.count"], 3.0)
        self.assertEqual(m["compile_steps.sum_s"], 95.0)
        self.assertEqual(m["compile_steps.target.extend_s"], 75.0)
        self.assertEqual(m["compile_steps.target.decode_s"], 20.0)

    def test_ignores_non_numeric_and_missing(self) -> None:
        m = extract_startup_metrics(
            {
                "load_weight": "n/a",
                "cuda_graph": {"prefill": None},
                "scheduler_e2e": True,
            }
        )
        self.assertEqual(m, {})
        self.assertEqual(extract_startup_metrics(None), {})


class TestMeasurementFailures(unittest.TestCase):
    def test_complete_measurement_passes(self) -> None:
        self.assertEqual(
            measurement_failures(extract_startup_metrics(_startup_time())), []
        )

    def test_missing_pieces_are_reported(self) -> None:
        failures = measurement_failures(
            extract_startup_metrics(_startup_time(scheduler_e2e=100.0, compile=None))
        )
        self.assertEqual(len(failures), 2)
        self.assertIn(HEADLINE_METRIC, failures[0])
        self.assertIn("compile steps", failures[1])

    def test_missing_scheduler_e2e_is_reported(self) -> None:
        st = _startup_time()
        del st["scheduler_e2e"]
        failures = measurement_failures(extract_startup_metrics(st))
        self.assertEqual(failures, ["startup_time has no scheduler_e2e"])


class TestStartupTimingRunner(unittest.TestCase):
    def _run(self, get_side_effect):
        runner = startup_timing_runner(
            test_name="t",
            model="m",
            workload="cold-precompile",
            series_id="perf-x",
            display_name="X",
            extra_labels={"c": 16},
        )
        with (
            patch.object(startup_timing.requests, "get", side_effect=get_side_effect),
            patch.object(startup_timing, "dump_metric") as dump,
            patch.object(startup_timing, "emit_result_payload") as emit,
        ):
            result = runner("http://h:1")
        return result, dump, emit

    def test_clean_run_dumps_headline_with_routing_labels(self) -> None:
        result, dump, emit = self._run([_response({"startup_time": _startup_time()})])
        self.assertTrue(result["_passed"])
        self.assertEqual(result["score"], 610.0 - 120.5 - 2.0)
        calls = {c.args[0]: c.kwargs["labels"] for c in dump.call_args_list}
        headline = calls[f"t.cold-precompile.{HEADLINE_METRIC}"]
        self.assertEqual(headline["kind"], "performance")
        self.assertEqual(headline["headline"], "1")
        self.assertEqual(headline["id"], "perf-x")
        self.assertEqual(headline["c"], 16)
        self.assertEqual(headline["direction"], "lower_is_better")
        info = calls["t.cold-precompile.load_weight"]
        self.assertEqual(info["kind"], "info")
        self.assertEqual(
            calls["t.cold-precompile.compile_steps.count"]["direction"], "informational"
        )
        self.assertNotIn("smoke_failed", calls)
        emit.assert_called_once()

    def test_incomplete_measurement_dumps_cross_and_fails(self) -> None:
        result, dump, _ = self._run([_response({"startup_time": {"load_weight": 1.0}})])
        self.assertFalse(result["_passed"])
        self.assertEqual([c.args[0] for c in dump.call_args_list], ["smoke_failed"])
        self.assertIn("reason", dump.call_args.kwargs["labels"])

    def test_unreachable_server_fails_cleanly(self) -> None:
        result, dump, _ = self._run(requests.ConnectionError("refused"))
        self.assertFalse(result["_passed"])
        self.assertIn("could not read /server_info", result["_failures"][0])
        self.assertEqual(dump.call_args.args[0], "smoke_failed")


if __name__ == "__main__":
    unittest.main()
