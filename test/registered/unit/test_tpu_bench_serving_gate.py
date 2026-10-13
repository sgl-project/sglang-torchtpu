"""Unit tests for TPU eval helpers (no TPU, server, or benchmark run).

The gate is an AccuracyTestParams.runner that shells out to
benchmark/tpu/bench_serving.sh and reads the JSONL back. Two contracts with that
shell script are only checkable here, because nothing else in CI runs the script
without a TPU:

  * the SWEEP tuple it emits must be what bench_serving.sh's parser accepts
    ("c:isl:osl[:workload]"), and
  * bench_point_tag must equal the script's own per-point file stem. If those drift,
    the sweep measures fine and the gate reports a point that never ran.

The gate's own pass/fail logic is exercised against pre-written JSONL with the
subprocess stubbed, since "measurement incomplete" is the only thing it gates.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu.evals import eval_support, model_config

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

POINT = {"workload": "8k1k", "isl": 8192, "osl": 1024, "concurrency": 16}
# 16 concurrency x the runner's default 10 prompts each.
COMPLETE = {
    "completed": 160,
    "total_throughput": 12000.0,
    "median_ttft_ms": 2600.0,
}


class _FakeSampler:
    system_message = None
    model = "model"
    temperature = 1.0
    max_tokens = 16
    reasoning_model = True
    reasoning_effort = "high"

    def __init__(self, create):
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    def __call__(self, _messages):
        raise AssertionError("the bounded implementation was not installed")

    @staticmethod
    def _pack_message(role, content):
        return {"role": role, "content": content}


class TestThinkingGate(CustomTestCase):
    def test_thinking_and_effort_reach_eval_runner_together(self):
        gates = model_config.build_gates(
            {
                "model": "deepseek-ai/DeepSeek-V3.1",
                "test_name": "thinking-test",
                "gates": [
                    {
                        "dataset": "gpqa",
                        "report_only": True,
                        "thinking": True,
                        "reasoning_effort": "high",
                    }
                ],
            }
        )
        self.assertEqual(
            gates[0].extra_eval_args,
            {"sgl_eval_thinking": True, "reasoning_effort": "high"},
        )


class TestSglEvalRunner(CustomTestCase):
    def test_template_kwargs_reach_sgl_eval_and_metrics_are_read_back(self):
        from pathlib import Path

        def fake_run(cmd, check):
            out = Path(cmd[cmd.index("--out-dir") + 1]) / "sgl_eval_aime26_1"
            out.mkdir(parents=True)
            (out / "metrics.json").write_text(
                json.dumps(
                    {
                        "aggregate": {"score": 0.875},
                        "num_examples": 30,
                        "latency_seconds": 12.0,
                        "output_throughput_tps": 3.5,
                    }
                )
            )

        runner = eval_support.sgl_eval_runner(
            "aime26",
            model="gemma4-31b",
            max_tokens=32768,
            num_threads=64,
            temperature=1.0,
            top_p=0.95,
            n_repeats=4,
            chat_template_kwargs={"enable_thinking": True},
        )
        with (
            patch.object(eval_support.subprocess, "run", side_effect=fake_run) as run,
            patch.object(eval_support, "dump_metric"),
        ):
            result = runner("http://127.0.0.1:30000/")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:3], ["sgl-eval", "run", "aime26"])
        self.assertEqual(cmd[cmd.index("--base-url") + 1], "http://127.0.0.1:30000/v1")
        self.assertEqual(
            cmd[cmd.index("--chat-template-kwarg") + 1], "enable_thinking=true"
        )
        self.assertEqual(cmd[cmd.index("--n-repeats") + 1], "4")
        self.assertNotIn("--thinking", cmd)
        self.assertEqual(result["score"], 0.875)
        self.assertEqual(result["latency"], 12.0)


class TestBoundedGptOssSampler(CustomTestCase):
    @staticmethod
    def _module(sleeps):
        class BadRequestError(Exception):
            pass

        class SamplerResponse:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        return SimpleNamespace(
            openai=SimpleNamespace(BadRequestError=BadRequestError),
            time=SimpleNamespace(sleep=sleeps.append),
            SamplerResponse=SamplerResponse,
        )

    @staticmethod
    def _response(content=None, reasoning=None):
        message = SimpleNamespace(content=content, reasoning=reasoning)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage={"tokens": 16}
        )

    def test_reasoning_only_response_is_not_regenerated(self):
        create = MagicMock(return_value=self._response(reasoning="Answer: B"))
        sampler = _FakeSampler(create)
        module = self._module([])
        eval_support._bound_gpt_oss_sampler(sampler, module)
        try:
            result = sampler([{"role": "user", "content": "question"}])
        finally:
            eval_support._restore_gpt_oss_sampler(sampler)

        self.assertEqual(result.response_text, "Answer: B")
        create.assert_called_once()

    def test_transport_failure_retries_once(self):
        sleeps = []
        create = MagicMock(
            side_effect=[RuntimeError("temporary"), self._response(content="A")]
        )
        sampler = _FakeSampler(create)
        module = self._module(sleeps)
        eval_support._bound_gpt_oss_sampler(sampler, module)
        try:
            result = sampler([{"role": "user", "content": "question"}])
        finally:
            eval_support._restore_gpt_oss_sampler(sampler)

        self.assertEqual(result.response_text, "A")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(sleeps, [1])


class TestSweepArg(CustomTestCase):
    """The string handed to bench_serving.sh, whose parser is the outside spec."""

    def test_tuple_shape_and_separator(self):
        # "c:isl:osl:workload", space-separated -- the script reads the 4th field
        # as the workload name and word-splits on spaces.
        other = {"workload": "1k1k", "isl": 1024, "osl": 1024, "concurrency": 64}
        self.assertEqual(
            eval_support.bench_sweep_arg([POINT, other]),
            "16:8192:1024:8k1k 64:1024:1024:1k1k",
        )

    def test_colon_in_workload_raises(self):
        # ':' is the tuple's own field separator, so a name containing one would
        # silently shift every field after it rather than failing.
        with self.assertRaises(ValueError):
            eval_support.bench_sweep_arg([{**POINT, "workload": "8k:1k"}])

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            eval_support.bench_sweep_arg(
                [{k: v for k, v in POINT.items() if k != "osl"}]
            )

    def test_unknown_key_raises(self):
        # A typo'd shape key would otherwise be dropped and the point measured at
        # some other geometry than the one written down.
        with self.assertRaises(ValueError):
            eval_support.bench_sweep_arg([{**POINT, "batch": 4}])


class TestPointTag(CustomTestCase):
    def test_matches_bench_serving_naming(self):
        # bench_serving.sh: tag="c${bs}-isl${isl}-osl${osl}"
        self.assertEqual(
            eval_support.bench_point_tag(16, 8192, 1024), "c16-isl8192-osl1024"
        )


class TestOutputDir(CustomTestCase):
    def test_under_the_ci_collector_root(self):
        with patch.dict(os.environ, {"TPU_CI_BENCH_OUTPUT_ROOT": "/tmp/bench"}):
            self.assertEqual(eval_support.bench_output_dir("m"), "/tmp/bench/m")

    def test_empty_without_the_root(self):
        # Empty means "leave OUTPUT_DIR unset" so bench_serving.sh's own ./results
        # default applies -- a path the CI collector also globs. Inventing a
        # directory here instead would put the numbers where nothing collects them.
        env = {k: v for k, v in os.environ.items() if k != "TPU_CI_BENCH_OUTPUT_ROOT"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(eval_support.bench_output_dir("m"), "")


class TestResultPayload(CustomTestCase):
    """The framed-payload egress every perf test shares (emit_result_payload)."""

    @staticmethod
    def _emit(summary):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_support.emit_result_payload(summary)
        return buf.getvalue()

    def test_payload_is_framed_and_reparses(self):
        # Bug regression (2026-08-26/28): a summary emitted as ONE long line is
        # split by containerd's 16 KiB CRI record cap and corrupted the rest of
        # the job log. The contract is a marker-framed, pretty-printed block:
        # line breaks only ever fall BETWEEN JSON tokens, so the rejoined text
        # parses back to exactly what was emitted.
        summary = {
            "legs": [
                {
                    "workload": "8k1k",
                    "status": "clean",
                    "metrics": {"total_throughput": 12000.0},
                }
            ]
        }
        out = self._emit(summary).splitlines()
        self.assertEqual(out[0], eval_support.RESULT_BEGIN)
        self.assertEqual(out[-1], eval_support.RESULT_END)
        body = "\n".join(out[1:-1])
        self.assertGreater(len(out), 3, "payload must be multi-line, not one record")
        self.assertEqual(json.loads(body), summary)

    def test_unserializable_summary_skips_payload_without_raising(self):
        # sort_keys raises on mixed-type keys. The payload is the fragile half
        # of the egress; an exception here would take down the caller and every
        # gate behind it, reporting nothing about the measurement itself.
        out = self._emit({1: "a", "b": 2})
        self.assertIn(eval_support.RESULT_BEGIN, out)
        self.assertIn(eval_support.RESULT_END, out)
        self.assertIn("payload skipped", out)


def _run_gate(tmp, record, returncode=0, seen=None, **runner_kw):
    """Run the gate with bench_serving.sh stubbed and `record` already written.

    `seen` collects the env the stubbed subprocess would have been handed.
    """
    out = os.path.join(tmp, "m")
    os.makedirs(out, exist_ok=True)
    if record is not None:
        stem = eval_support.bench_point_tag(16, 8192, 1024)
        with open(os.path.join(out, stem + ".jsonl"), "w") as fh:
            fh.write(json.dumps(record) + "\n")
    runner = eval_support.bench_serving_runner(
        sweep=[POINT], test_name="m", **runner_kw
    )

    def fake_run(argv, **kwargs):
        if seen is not None:
            seen.update(kwargs.get("env") or {})
        return MagicMock(returncode=returncode)

    with (
        patch.dict(os.environ, {"TPU_CI_BENCH_OUTPUT_ROOT": tmp}),
        patch.object(eval_support.subprocess, "run", side_effect=fake_run),
    ):
        return runner("http://127.0.0.1:30000")


class TestGate(CustomTestCase):
    """What the gate is for: telling "slower" from "served nothing"."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_complete_measurement_passes(self):
        result = _run_gate(self.tmp, COMPLETE)
        self.assertTrue(result["_passed"], result["_failures"])
        self.assertEqual(result["score"], 12000.0)

    def test_dropped_requests_fail(self):
        # A partial run still writes a JSONL, and its throughput is not comparable
        # to a full one -- so it must not land on the trend as a normal point.
        result = _run_gate(self.tmp, {**COMPLETE, "completed": 159})
        self.assertFalse(result["_passed"])
        self.assertIn("159 of 160", " ".join(result["_failures"]))

    def test_dropped_requests_publish_only_a_cross_record(self):
        """Bug regression (2026-08-24 gsm8k 0.0): the incomplete branch recorded
        its failure, then fell through and dumped the point anyway -- putting a
        number from a half-dead server on the chart, indistinguishable from a
        measured one. An incomplete point must dump NO numbers; what it dumps
        instead is one `smoke_failed` record, which the dashboard renders as a ✕
        with the reason on hover -- a failed night is explained, not hidden."""
        dumped = []
        with patch.object(
            eval_support,
            "dump_metric",
            side_effect=lambda name, value, labels=None: dumped.append(
                (name, value, labels)
            ),
        ):
            result = _run_gate(self.tmp, {**COMPLETE, "completed": 159})
        self.assertFalse(result["_passed"])
        self.assertEqual([d[0] for d in dumped], ["smoke_failed"])
        labels = dumped[0][2]
        # kind/model/workload are what route the ✕ to the right chart; reason is
        # free text from the actual failure.
        self.assertEqual(labels["kind"], "performance")
        self.assertEqual(labels["model"], "m")
        self.assertEqual(labels["workload"], "8k1k")
        self.assertIn("159 of 160", labels["reason"])

    def test_failed_smoke_leg_withholds_the_sweep(self):
        """A red smoke leg means the measurement would be garbage: the sweep must
        not run at all (no bench subprocess), and every point must land as a ✕
        carrying the smoke reason instead of a number."""
        dumped, ran = [], []
        runner = eval_support.bench_serving_runner(
            sweep=[POINT],
            test_name="m",
            smoke_state={"failures": ["server health check timed out after 1800s"]},
        )
        with (
            patch.object(
                eval_support,
                "dump_metric",
                side_effect=lambda name, value, labels=None: dumped.append(
                    (name, labels)
                ),
            ),
            patch.object(
                eval_support.subprocess,
                "run",
                side_effect=lambda *a, **k: ran.append(a),
            ),
        ):
            result = runner("http://127.0.0.1:30000")
        self.assertFalse(result["_passed"])
        self.assertEqual(ran, [])
        self.assertEqual([d[0] for d in dumped], ["smoke_failed"])
        self.assertIn("timed out", dumped[0][1]["reason"])

    def test_headline_carries_the_series_identity(self):
        # id joins the series across nights and display_name names it; without
        # them on the headline record, a rename splits the trend in two.
        dumped = []
        with patch.object(
            eval_support,
            "dump_metric",
            side_effect=lambda name, value, labels=None: dumped.append((name, labels)),
        ):
            _run_gate(
                self.tmp,
                COMPLETE,
                series_id="perf-m",
                display_name="M 1B",
            )
        headline = [l for n, l in dumped if n.endswith(".total_throughput")]
        self.assertEqual(len(headline), 1)
        self.assertEqual(headline[0]["kind"], "performance")
        self.assertEqual(headline[0]["id"], "perf-m")
        self.assertEqual(headline[0]["display_name"], "M 1B")
        self.assertEqual(headline[0]["workload"], "8k1k")
        # The non-headline detail records stay off every chart: kind=info.
        details = {l["kind"] for n, l in dumped if "total_throughput" not in n}
        self.assertEqual(details, {"info"})

    def test_zero_throughput_fails(self):
        result = _run_gate(self.tmp, {**COMPLETE, "total_throughput": 0.0})
        self.assertFalse(result["_passed"])

    def test_missing_jsonl_fails(self):
        # The server came up and the script exited 0, but the point never wrote.
        result = _run_gate(self.tmp, None)
        self.assertFalse(result["_passed"])
        self.assertIn("never ran", " ".join(result["_failures"]))

    def test_bench_script_failure_fails(self):
        result = _run_gate(self.tmp, COMPLETE, returncode=1)
        self.assertFalse(result["_passed"])

    def test_model_is_passed_to_the_script(self):
        # bench_serving.sh exits 2 without MODEL, so a runner that forgets it turns
        # every nightly red before a single request is sent.
        env = {}
        _run_gate(self.tmp, COMPLETE, seen=env, model="org/some-model")
        self.assertEqual(env["MODEL"], "org/some-model")

    def test_bench_env_overrides_the_model(self):
        # A model served from a local checkpoint points MODEL at the path (the
        # tokenizer source) and names itself with SERVED_MODEL_NAME.
        env = {}
        _run_gate(
            self.tmp,
            COMPLETE,
            seen=env,
            model="openai/gpt-oss-120b",
            bench_env={
                "MODEL": "/weights/gpt-oss-120b",
                "SERVED_MODEL_NAME": "openai/gpt-oss-120b",
            },
        )
        self.assertEqual(env["MODEL"], "/weights/gpt-oss-120b")
        self.assertEqual(env["SERVED_MODEL_NAME"], "openai/gpt-oss-120b")


class TestSweepGateTranslation(CustomTestCase):
    """model_config turns a `sweep` gate into the runner, and rejects a graded one."""

    def test_sweep_gate_builds_a_self_gating_runner(self):
        gates = model_config.build_gates(
            {
                "model": "m",
                "test_name": "m",
                "gates": [
                    {"dataset": "bench_serving", "label": "8k1k", "sweep": [POINT]}
                ],
            }
        )
        self.assertIsNotNone(gates[0].runner)
        # No floor: run_accuracy_test would fail a gate that has neither a baseline
        # nor a self-gating runner, so this pairing is what keeps it valid.
        self.assertIsNone(gates[0].baseline_accuracy)

    def test_config_model_reaches_the_script(self):
        # The config's model is the gate's MODEL default; without this wiring
        # bench_serving.sh exits 2 for want of one.
        gates = model_config.build_gates(
            {
                "model": "org/some-model",
                "gates": [{"dataset": "bench_serving", "sweep": [POINT]}],
            }
        )
        env = {}

        def fake_run(argv, **kwargs):
            env.update(kwargs.get("env") or {})
            return MagicMock(returncode=1)  # short-circuit; MODEL is what we assert

        with patch.object(eval_support.subprocess, "run", side_effect=fake_run):
            gates[0].runner("http://127.0.0.1:30000")
        self.assertEqual(env["MODEL"], "org/some-model")

    def test_baseline_on_a_sweep_gate_raises(self):
        # A throughput floor invented before any nightly has run would either never
        # fire or fire on noise; the dashboard's trend is the regression detector.
        with self.assertRaises(ValueError):
            model_config.build_gates(
                {
                    "model": "m",
                    "gates": [
                        {"dataset": "bench_serving", "sweep": [POINT], "baseline": 1.0}
                    ],
                }
            )

    def test_bench_env_without_sweep_raises(self):
        with self.assertRaises(ValueError):
            model_config.build_gates(
                {
                    "model": "m",
                    "gates": [{"dataset": "mmlu", "baseline": 0.8, "bench_env": {}}],
                }
            )


class TestSmokeLegs(CustomTestCase):
    """Smoke legs guard the sweep: ordering, shape, and the withhold wiring."""

    @staticmethod
    def _smoke(passed=True):
        return {
            "dataset": "health",
            "smoke": True,
            "runner": lambda base_url: {
                "score": 1.0,
                "_passed": passed,
                "_failures": [] if passed else ["sanity request returned 500"],
            },
        }

    def test_smoke_after_sweep_raises(self):
        # Gates run in list order; a smoke leg after the sweep would validate a
        # measurement that has already been published.
        with self.assertRaises(ValueError):
            model_config.build_gates(
                {
                    "model": "m",
                    "gates": [
                        {"dataset": "bench_serving", "sweep": [POINT]},
                        self._smoke(),
                    ],
                }
            )

    def test_smoke_without_runner_raises(self):
        # Only a self-gating runner reports a verdict this module can observe; a
        # baseline-graded smoke leg would fail invisibly and never withhold.
        with self.assertRaises(ValueError):
            model_config.build_gates(
                {"model": "m", "gates": [{"dataset": "health", "smoke": True}]}
            )

    def test_failed_smoke_leg_withholds_the_sweep(self):
        # The wiring end to end: the smoke gate's failure text must reach the ✕
        # record the sweep dumps in place of its numbers.
        gates = model_config.build_gates(
            {
                "model": "m",
                "gates": [
                    self._smoke(passed=False),
                    {"dataset": "bench_serving", "sweep": [POINT]},
                ],
            }
        )
        gates[0].runner("http://127.0.0.1:30000")

        dumped, ran = [], []
        with (
            patch.object(
                eval_support,
                "dump_metric",
                side_effect=lambda name, value, labels=None: dumped.append(
                    (name, labels)
                ),
            ),
            patch.object(
                eval_support.subprocess,
                "run",
                side_effect=lambda *a, **k: ran.append(a),
            ),
        ):
            result = gates[1].runner("http://127.0.0.1:30000")
        self.assertFalse(result["_passed"])
        self.assertEqual(ran, [])
        self.assertEqual([d[0] for d in dumped], ["smoke_failed"])
        self.assertIn("sanity request returned 500", dumped[0][1]["reason"])


class TestSeriesIdentity(CustomTestCase):
    """id/display_name flow from MODEL_CONFIG onto every dashboard-bound record."""

    def test_accuracy_gate_labels_carry_id_and_display_name(self):
        gates = model_config.build_gates(
            {
                "model": "m",
                "id": "accuracy-m",
                "display_name": "M 1B",
                "dashboard": "accuracy",
                "gates": [{"dataset": "mmlu", "baseline": 0.8, "atol": 0.03}],
            }
        )
        labels = gates[0].metric_labels
        self.assertEqual(labels["kind"], "accuracy")
        self.assertEqual(labels["id"], "accuracy-m")
        self.assertEqual(labels["display_name"], "M 1B")


class TestPerfConfigsAreValid(CustomTestCase):
    """Every perf model's config, validated without a TPU.

    The harness rejects unknown keys and missing shapes at build time, so building
    the real configs here turns a typo in a launch env or a sweep point into a CPU
    test failure rather than a wasted nightly slice.

    Discovered by glob rather than listed: one file per model is the convention,
    and a hand-maintained list here would silently stop covering the next one.
    """

    def _perf_modules(self):
        found = list(_config_modules("nightly-perf", "test_tpu_perf_*.py"))
        self.assertTrue(found, "no perf test files found; did they move?")
        yield from found

    def test_every_config_builds(self):
        names, suites = set(), set()
        for path, mod in self._perf_modules():
            config = _model_config(mod)
            model_config.build_model(config)  # rejects unknown launch keys
            self.assertTrue(model_config.build_gates(config))
            case = model_config.build_test_case(config)
            self.assertNotIn(case.__name__, names, f"{path.name} duplicates a case")
            names.add(case.__name__)

            # One suite per model is what makes them separate nightly jobs; two
            # files sharing one would quietly serialise onto the same slice.
            suite = _registered_suite(path)
            self.assertTrue(suite, f"{path.name} registers no suite")
            self.assertNotIn(suite, suites, f"{path.name} shares a suite")
            suites.add(suite)

    def test_every_model_measures_a_named_workload(self):
        # The dashboard groups by workload name; an unnamed point would fall back to
        # its raw shape and sit in a separate dropdown entry from its siblings.
        for _path, mod in self._perf_modules():
            for gate in _model_config(mod)["gates"]:
                for point in gate.get("sweep") or []:
                    self.assertTrue(point["workload"])

    def test_every_suite_is_registered_with_run_suite(self):
        # A suite the runner does not know is a job that silently runs nothing.
        from pathlib import Path

        runner = (Path(__file__).resolve().parents[2] / "run_suite.py").read_text(
            encoding="utf-8"
        )
        for path, _mod in self._perf_modules():
            suite = _registered_suite(path)
            self.assertIn(f'"{suite}"', runner, f"{suite} missing from run_suite.py")


class TestAccuracyConfigsAreValid(CustomTestCase):
    """Every accuracy model's config, validated without a TPU.

    Same shape as the perf checks above, plus the dashboard wiring: the accuracy
    tab plots only metrics stamped kind=accuracy, so a config that forgets its
    declaration runs green forever while its scores silently never chart.
    """

    def _accuracy_modules(self):
        found = list(_config_modules("nightly-accuracy", "test_tpu_*.py"))
        self.assertTrue(found, "no accuracy test files found; did they move?")
        yield from found

    def _assert_fixture_owns_the_server(self, path, mod):
        # Without a launch entry, the file's TPULauncherTestCase boots the recipe.
        from pathlib import Path

        from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase
        from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

        # The imported bases are in the module namespace too; only the concrete
        # subclass this file defines counts.
        bases = (TPULauncherTestCase, AccuracyTestCase)
        fixtures = [
            value
            for value in vars(mod).values()
            if isinstance(value, type)
            and issubclass(value, TPULauncherTestCase)
            and value not in bases
        ]
        self.assertEqual(len(fixtures), 1, f"{path.name} needs one launcher fixture")
        recipe = (
            Path(__file__).resolve().parents[3]
            / "benchmark/tpu"
            / fixtures[0].config_module
        )
        self.assertTrue(recipe.is_file(), f"{path.name}: no {recipe}")

    def test_every_config_builds(self):
        names, suites = set(), set()
        for path, mod in self._accuracy_modules():
            config = _model_config(mod)
            if "launch" in config:
                model_config.build_model(config)  # rejects unknown launch keys
            else:
                self._assert_fixture_owns_the_server(path, mod)
            self.assertTrue(model_config.build_gates(config))
            case = model_config.build_test_case(config)
            self.assertNotIn(case.__name__, names, f"{path.name} duplicates a case")
            names.add(case.__name__)

            # One suite per model is what makes them separate nightly jobs; two
            # files sharing one would quietly serialise onto the same slice.
            suite = _registered_suite(path)
            self.assertTrue(suite, f"{path.name} registers no suite")
            # Multi-host discovery creates one job per file, even when files
            # share a suite. Single-host discovery creates one job per suite.
            if not hasattr(mod, "MULTIHOST_LAUNCH"):
                self.assertNotIn(suite, suites, f"{path.name} shares a suite")
                suites.add(suite)

    def test_every_gate_has_a_hardcoded_floor(self):
        # floor = baseline - atol (an lm_eval gate keeps both in its YAML).
        for path, mod in self._accuracy_modules():
            for gate in _model_config(mod)["gates"]:
                self.assertFalse(gate.get("report_only"), path.name)
                if "lm_eval_config" not in gate:
                    self.assertIn("baseline", gate, path.name)
                    self.assertIn("atol", gate, path.name)

    def test_gated_eval_without_a_score_fails_without_plotting_zero(self):
        from sglang.test import accuracy_test_runner as runner

        params = runner.AccuracyTestParams(
            dataset="gpqa", baseline_accuracy=0.8, atol=0.1, metric_labels={}
        )
        model = SimpleNamespace(model_path="m", variant=None)
        with (
            patch.object(runner, "served", MagicMock()),
            patch.object(runner, "_dispatch_eval", return_value=(True, None, {})),
            patch.object(runner, "dump_metric") as dump,
        ):
            result = runner.run_accuracy_test(model, params, "http://x")
        self.assertFalse(result.passed)
        dump.assert_not_called()

    def test_every_gate_reaches_the_dashboard(self):
        # "accuracy" is the literal the dashboard's metric_kind() matches on; a
        # renamed key or value empties the tab without failing anything here.
        for path, mod in self._accuracy_modules():
            self.assertEqual(_model_config(mod).get("dashboard"), "accuracy", path.name)
            for params in model_config.build_gates(_model_config(mod)):
                self.assertEqual(
                    (params.metric_labels or {}).get("kind"),
                    "accuracy",
                    f"{path.name}: gate {params.name} would never be plotted",
                )
                self.assertEqual(
                    (params.metric_labels or {}).get("model"),
                    _model_config(mod)["dashboard_model"],
                    f"{path.name}: gate {params.name} has the wrong model series",
                )

    def test_dashboard_is_declared_only_under_nightly_accuracy(self):
        # The inverse guard: a feature-validation test declaring `dashboard` would
        # put its gsm8k-style scores on the accuracy tab. Category = directory.
        from pathlib import Path

        tpu_dir = Path(__file__).resolve().parents[1] / "tpu"
        for path in sorted(tpu_dir.rglob("test_*.py")):
            if "nightly-accuracy" in path.parts or "__pycache__" in path.parts:
                continue
            self.assertNotIn(
                '"dashboard"',
                path.read_text(encoding="utf-8"),
                f"{path.relative_to(tpu_dir)} declares dashboard outside "
                "nightly-accuracy/",
            )

    def test_every_suite_is_registered_with_run_suite(self):
        # A suite the runner does not know is a job that silently runs nothing.
        from pathlib import Path

        runner = (Path(__file__).resolve().parents[2] / "run_suite.py").read_text(
            encoding="utf-8"
        )
        for path, _mod in self._accuracy_modules():
            suite = _registered_suite(path)
            self.assertIn(f'"{suite}"', runner, f"{suite} missing from run_suite.py")


def _model_config(mod):
    """The file's MODEL_CONFIG, however it declares it.

    #836's shape builds it in a function so the gates can close over values the
    fixture resolves; older files still assign the dict at import.
    """
    if hasattr(mod, "build_model_config"):
        return mod.build_model_config()
    return mod.MODEL_CONFIG


def _config_modules(subdir, pattern):
    """Load every matching MODEL_CONFIG module under test/registered/tpu/<subdir>."""
    import importlib.util
    from pathlib import Path

    for path in sorted(
        (Path(__file__).resolve().parents[1] / "tpu" / subdir).glob(pattern)
    ):
        spec = importlib.util.spec_from_file_location(f"cfg_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield path, mod


def _registered_suite(path):
    """The suite= a perf file registers, read from source rather than imported."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "register_tpu_ci"
        ):
            for kw in node.keywords:
                if kw.arg == "suite":
                    return kw.value.value
    return None


if __name__ == "__main__":
    unittest.main()
