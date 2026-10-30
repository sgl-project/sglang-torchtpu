"""Run a TPU accuracy nightly from a declarative ``MODEL_CONFIG`` dict.

Adding a model means copying a test file and editing its dict. This module turns that
dict into the launcher, the ``AccuracyTestParams`` list and the ``run_combined_tests``
call, so a test file is a register call, a dict, and two lines to run it:

    register_tpu_ci(est_time=10800, suite="nightly-tpu-4-v7", nightly=True)

    MODEL_CONFIG = {
        "model": "/weights/gpt-oss-120b",
        "test_name": "gpt-oss-120b",
        "variant": "TP8+EP8",
        "launch": {"recipe": "recipe/gpt_oss_120b/eval.sh", "tag": "gpqa-e2e"},
        "gates": [
            {"dataset": "mmlu", "baseline": 0.859, "atol": 0.03,
             "num_examples": 2000, "reasoning_effort": "low"},
        ],
    }

    if __name__ == "__main__":
        model_config_main(MODEL_CONFIG)

``model_config_main`` also takes a LIST of configs, which is how one file covers
several models -- see test_tpu_perf.py, where two models want the same sweep but
different servers. One TestCase per config, run in sequence.

Top level
    model         Model path or HF id. Reported, and sent as the model id by the eval
                  clients; the recipe is what actually sets MODEL on the server.
    gates         Evaluations to run, all sharing one server boot.
    launch        How to bring the server up (see below). Omit when a test fixture
                  owns the server and supplies its ready base_url.
    test_name     Report label, and the basis of the generated TestCase name.
                  Defaults to `model`.
    id            Stable series slug, e.g. "accuracy-gpt-oss-120b". The dashboard's
                  join key across nights: NEVER renamed, never shown. A forced
                  rename is recorded once in .github/tpu-ci/dashboard_aliases.json
                  instead of edited here. Required for nightly-accuracy and
                  nightly-perf configs (pr-gate label lint).
    display_name  What the dashboard calls the series. Cosmetic, free to change --
                  the split from `id` is what makes a rename a non-event.
    variant       Distinguishes launch configs of the same model in reports, e.g.
                  "TP8+EP8". Cosmetic while a model has one config.
    dashboard     "accuracy" -> each non-sweep gate's score is dumped as
                  `[METRIC] {label}={score}` with kind=accuracy, which is what puts
                  it on the CI dashboard's accuracy tab. Absent -> the gates still
                  gate, but nothing is plotted: a feature-validation test cannot
                  land on the tab by accident. (The performance tab works the same
                  way off the sweep gate's own workload labels.)
    dashboard_model  Model-series label for dashboard metrics. Defaults to
                  `test_name`; set it when jobs need distinct names but one series.
    skip_unless   (condition, reason) -- the config form of @unittest.skipUnless.

launch
    config        Python recipe under benchmark/tpu/recipe/configs/ that brings the
                  server up, e.g. "recipe/configs/gpt_oss_20b/spec.py". The form new
                  tests take; a human reproduces a red run with
                  `python -m benchmark.tpu.launch_server --config <it>`.
    recipe        Shell script under benchmark/tpu/ that brings the server up, e.g.
                  "recipe/gpt_oss_120b/server.sh". Legacy, and mutually exclusive
                  with `config`.
    env           Environment for the recipe. The launcher logs it on boot, so a red run
                  stays reproducible by hand. Config describing *how* the model is
                  served belongs in the recipe.
    external      True -> attach to a server an external orchestrator already started
                  (the multi-host JobSets). Takes no recipe/env.
    tag           Prefix on this model's progress lines, e.g. "gpqa-e2e".
    health_timeout  Seconds to wait for /health_generate before giving up.

gates -- one entry per evaluation
    dataset       run_eval eval name ("gsm8k", "mmlu", "gpqa", "aime25", ...), or a
                  descriptive name when `runner`/`lm_eval_config` does the scoring.
    baseline      Reference score, normally the model card's.
    atol          Tolerance below it: floor = baseline - atol.
    report_only   True -> require and publish a numeric score without comparing it
                  to a baseline. Backend and transport failures still fail the test.
    label         Name in reports and metrics. Defaults to `dataset`; set it when one
                  dataset is gated more than once.
    repeat        Passes to average. Small datasets need several before a tight
                  tolerance means anything -- see GPQA in test_tpu_gpt_oss_120b.py
                  for how to size it against the single-pass sigma.
    reasoning_effort   "low"/"medium"/"high". Cards report a different score per
                  level, so this must match whichever the baseline came from.
    thinking      Pass --thinking to sgl-eval for hybrid reasoning checkpoints.
    runner        runner(base_url) -> score or metrics dict. Scores the gate with a
                  harness this module has no dependency on.
    lm_eval_config     YAML under test/lm_eval_configs/. It owns the ground truths and
                  tolerances and is also what run_lm_eval.sh reads, so `baseline`
                  and `atol` must not be set alongside it.
    sweep         Makes this a THROUGHPUT gate instead of an accuracy one: a list of
                  {workload, isl, osl, concurrency} points handed to
                  bench_serving.sh, measured against the same server boot as the
                  model's accuracy gates. Each point's numbers reach the perf
                  dashboard, where `workload` is the name its dropdown shows. There
                  is no floor -- the gate checks the measurement is complete, and
                  the dashboard's trend is what detects a regression -- so
                  `baseline`/`atol` must not be set alongside it.
    bench_env     Extra environment for a `sweep` gate's bench_serving.sh
                  (DATASET_NAME, TOKENIZE_PROMPT, ...). Geometry comes from `sweep`.
    smoke         True marks this gate a SMOKE LEG guarding the sweep: it must be
                  listed before every sweep gate and must bring its own self-gating
                  `runner` (returning "_passed"). If a smoke leg fails, the sweep
                  dumps no numbers -- each point gets a `smoke_failed` record
                  carrying the reason, which the dashboard renders as a ✕ -- so an
                  inaccurately measured night can never look like a normal
                  datapoint. Optional: with no smoke legs, the sweep still gates
                  itself on measurement completeness, which takes the same ✕ path.
    num_examples, num_threads, max_tokens, temperature, top_p, top_k, api
                  Passed straight through to run_eval.
    spec_accept_length_threshold
                  Optional secondary threshold for every scheduler's cumulative
                  avg_spec_accept_length, read from /server_info after the eval.

Two things stay in the test file. ``register_tpu_ci(...)`` is read by parsing source
with ``ast``, so it must appear literally; it is also what lets est_time, suite and
disabled differ per model. And the ``__main__`` guard is required because ci_utils
runs each file as ``python3 <file> -f``, while importing one (tests do) must not
execute anything.

Unknown keys raise rather than being ignored -- a silently-dropped typo in a
threshold is the failure mode this shape would otherwise invite.
"""

import re
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings
from sglang.test.tpu.evals.eval_support import (
    DEFAULT_HEALTH_TIMEOUT,
    ExternalServerLauncher,
    RecipeLauncher,
    TpuModelLaunchSettings,
    bench_serving_runner,
    lm_eval_runner,
)

_TPU_BENCH_DIR = Path(__file__).resolve().parents[5] / "benchmark" / "tpu"

_CONFIG_KEYS = {
    "model",
    "test_name",
    "id",
    "display_name",
    "variant",
    "launch",
    "gates",
    "dashboard",
    "dashboard_model",
    "skip_unless",
}
_LAUNCH_KEYS = {"config", "recipe", "env", "tag", "health_timeout", "external"}
# Gate keys that map straight onto AccuracyTestParams.
_GATE_PASSTHROUGH = {
    "dataset",
    "label",
    "atol",
    "repeat",
    "num_examples",
    "num_threads",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "api",
    "spec_accept_length_threshold",
    "fail_on_recompile",
    "runner",
    "report_only",
}
# Gate keys needing translation.
_GATE_TRANSLATED = {
    "baseline",
    "reasoning_effort",
    "thinking",
    "lm_eval_config",
    "sweep",
    "bench_env",
    "smoke",
}


def _reject_unknown(got, allowed, where: str) -> None:
    unknown = set(got) - allowed
    if unknown:
        raise ValueError(
            f"unknown {where} key(s): {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def _build_launcher(launch: dict[str, Any]):
    _reject_unknown(launch, _LAUNCH_KEYS, "launch")
    timeout = launch.get("health_timeout", DEFAULT_HEALTH_TIMEOUT)
    tag = launch.get("tag", "tpu-e2e")

    if launch.get("external"):
        if {"recipe", "config", "env"} & set(launch):
            raise ValueError(
                "external launch takes no recipe/config/env: the JobSet owns it"
            )
        return ExternalServerLauncher(health_timeout=timeout, tag=tag)

    config, recipe = launch.get("config"), launch.get("recipe")
    if config and recipe:
        raise ValueError('launch takes "config" or "recipe", not both')
    if config:
        # --config resolves as a module under benchmark.tpu.recipe, so the
        # launcher runs from the repo root, not from benchmark/tpu/.
        return RecipeLauncher(
            command=[
                sys.executable,
                "-u",
                "-m",
                "benchmark.tpu.launch_server",
                "--config",
                config,
            ],
            env=launch.get("env"),
            cwd=str(_TPU_BENCH_DIR.parents[1]),
            health_timeout=timeout,
            tag=tag,
        )
    if not recipe:
        raise ValueError('launch needs "config", "recipe", or "external": True')
    script = Path(recipe)
    if not script.is_absolute():
        script = _TPU_BENCH_DIR / script
    return RecipeLauncher(
        str(script),
        env=launch.get("env"),
        cwd=str(_TPU_BENCH_DIR),
        health_timeout=timeout,
        tag=tag,
    )


def _smoke_wrapped(runner, smoke_state: dict[str, Any], name: str):
    """Record a smoke leg's failure so the sweep can withhold its numbers.

    The reason is taken from the actual failure -- the runner's own "_failures"
    text, or the exception -- because it is what the dashboard shows on the ✕.
    """

    def run(base_url: str) -> dict[str, Any]:
        try:
            result = runner(base_url)
        except Exception as exc:
            smoke_state["failures"].append(f"smoke leg {name}: {exc}")
            raise
        if result.get("_passed") is False:
            reasons = [str(r) for r in result.get("_failures") or []]
            smoke_state["failures"].extend(reasons or [f"smoke leg {name} failed"])
        return result

    return run


def _build_gate(
    gate: dict[str, Any],
    test_name: str = "",
    model: str = "",
    dashboard: str | None = None,
    dashboard_model: str | None = None,
    series_id: str = "",
    display_name: str = "",
    smoke_state: dict[str, Any] | None = None,
) -> AccuracyTestParams:
    _reject_unknown(gate, _GATE_PASSTHROUGH | _GATE_TRANSLATED, "gate")
    kwargs = {k: v for k, v in gate.items() if k in _GATE_PASSTHROUGH}

    if dashboard == "accuracy" and "sweep" not in gate:
        # The dashboard plots only metrics that declare kind=accuracy; the stamp
        # is per-config so a copied gate cannot reach the tab undeclared. A sweep
        # gate is excluded: its points carry their own performance labels.
        # `id` joins the series across nights (renames go through
        # dashboard_aliases.json, never through this value); display_name is
        # cosmetic and free to change.
        kwargs["metric_labels"] = {
            "model": dashboard_model or test_name,
            "kind": "accuracy",
        }
        if series_id:
            kwargs["metric_labels"]["id"] = series_id
        if display_name:
            kwargs["metric_labels"]["display_name"] = display_name

    if gate.get("report_only"):
        conflicting = sorted({"baseline", "atol"} & set(gate))
        if conflicting:
            raise ValueError(
                f"a report-only gate must not set {conflicting}: it records the "
                "measured score without a quality floor"
            )

    if "baseline" in gate:
        kwargs["baseline_accuracy"] = gate["baseline"]
    extra_eval_args = {}
    if "reasoning_effort" in gate:
        extra_eval_args["reasoning_effort"] = gate["reasoning_effort"]
    if "thinking" in gate:
        extra_eval_args["sgl_eval_thinking"] = gate["thinking"]
    if extra_eval_args:
        kwargs["extra_eval_args"] = extra_eval_args
    if "lm_eval_config" in gate:
        if "runner" in gate:
            raise ValueError("a gate takes either lm_eval_config or runner, not both")
        conflicting = sorted({"baseline", "atol"} & set(gate))
        if conflicting:
            raise ValueError(
                f"lm_eval gate must not set {conflicting}: put them in "
                f"{gate['lm_eval_config']} instead, which is the only copy and is "
                "what run_lm_eval.sh reads"
            )
        kwargs["runner"] = lm_eval_runner(
            gate["lm_eval_config"], report_only=gate.get("report_only", False)
        )

    if gate.get("smoke"):
        # A smoke leg guards the sweep, so its verdict must be observable here:
        # only a self-gating runner reports one ("_passed"); a baseline-graded
        # gate is scored outside this module and its failure would go unseen.
        if "sweep" in gate:
            raise ValueError("a gate is a smoke leg or a sweep, not both")
        conflicting = sorted(
            {"baseline", "atol", "lm_eval_config", "report_only"} & set(gate)
        )
        if conflicting:
            raise ValueError(
                f"a smoke gate must not set {conflicting}: it needs a self-gating "
                '"runner", whose verdict is what withholds the sweep'
            )
        if "runner" not in gate:
            raise ValueError('a smoke gate needs a self-gating "runner"')
        if smoke_state is None:
            raise ValueError("smoke gates are only valid inside a MODEL_CONFIG")
        kwargs["runner"] = _smoke_wrapped(
            gate["runner"], smoke_state, gate.get("label") or gate["dataset"]
        )

    if "sweep" in gate:
        # A throughput gate. It shares the boot with this model's accuracy gates
        # rather than paying for a second one, and it self-gates (no floor) --
        # see bench_serving_runner for what it does check.
        conflicting = sorted(
            {"baseline", "atol", "runner", "lm_eval_config"} & set(gate)
        )
        if conflicting:
            raise ValueError(
                f"a sweep gate must not set {conflicting}: it measures rather than "
                "grades, and gates itself on the measurement being complete"
            )
        kwargs["runner"] = bench_serving_runner(
            sweep=gate["sweep"],
            test_name=test_name or gate.get("label") or gate["dataset"],
            model=model,
            bench_env=gate.get("bench_env"),
            series_id=series_id,
            display_name=display_name,
            smoke_state=smoke_state,
        )
    elif "bench_env" in gate:
        raise ValueError('bench_env is only meaningful alongside "sweep"')

    return AccuracyTestParams(**kwargs)


def build_gates(config: dict[str, Any]) -> list[AccuracyTestParams]:
    """Expand the ``gates`` list. Split out so tests can assert on it directly."""
    model = str(config.get("model", ""))
    test_name = str(config.get("test_name", model))
    dashboard = config.get("dashboard")
    dashboard_model = config.get("dashboard_model", test_name)
    if dashboard not in (None, "accuracy"):
        raise ValueError(f'dashboard must be "accuracy" or absent, got {dashboard!r}')

    # Smoke legs run first because gates run in list order; a smoke leg after a
    # sweep would validate a measurement that has already been published.
    gates = config["gates"]
    sweep_indices = [i for i, g in enumerate(gates) if "sweep" in g]
    smoke_indices = [i for i, g in enumerate(gates) if g.get("smoke")]
    if smoke_indices and sweep_indices and max(smoke_indices) > min(sweep_indices):
        raise ValueError("smoke gates must be listed before every sweep gate")

    # Shared between the smoke legs (writers) and the sweep runners (readers):
    # what withholds a sweep's numbers when a leg has already failed.
    smoke_state: dict[str, Any] = {"failures": []}
    return [
        _build_gate(
            g,
            test_name=test_name,
            model=model,
            dashboard=dashboard,
            dashboard_model=dashboard_model,
            series_id=str(config.get("id", "")),
            display_name=str(config.get("display_name", "")),
            smoke_state=smoke_state,
        )
        for g in gates
    ]


def build_model(config: dict[str, Any]) -> TpuModelLaunchSettings:
    """Build the launch settings. Split out so tests can assert on it directly."""
    # No tp_size: the recipe owns parallelism and TpuModelLaunchSettings clears the
    # extra_args it would have fed, so it would only be a second copy that can drift.
    return TpuModelLaunchSettings(
        config["model"],
        launcher=_build_launcher(config.get("launch", {})),
        variant=config.get("variant"),
    )


def run_model_config(config: dict[str, Any], base_url: str | None = None) -> dict:
    """Run every gate using the configured launcher or a fixture's ready URL.

    Without a ``launch`` entry, ``base_url`` is required and its caller owns
    startup, readiness, and shutdown. Explicit launch entries retain their
    behavior even when ``base_url`` is supplied.
    """
    _reject_unknown(config, _CONFIG_KEYS, "MODEL_CONFIG")
    for required in ("model", "gates"):
        if required not in config:
            raise ValueError(f'MODEL_CONFIG needs "{required}"')

    if "launch" in config:
        model = build_model(config)
    else:
        if not base_url:
            raise ValueError('MODEL_CONFIG without "launch" needs a ready base_url')
        # The class fixture owns startup, readiness, and shutdown. Each gate
        # enters the existing server without another launch or health check.
        model = ModelLaunchSettings(
            config["model"], launcher=nullcontext, variant=config.get("variant")
        )
        model.extra_args = []

    return run_combined_tests(
        models=[model],
        test_name=config.get("test_name", config["model"]),
        base_url=base_url,
        accuracy_params=build_gates(config),
    )


def build_test_case(config: dict[str, Any], base_url: str | None = None):
    """Synthesise the unittest.TestCase for ``config``.

    ``skip_unless`` is ``(condition, reason)`` -- the config equivalent of
    ``@unittest.skipUnless``, used by the multi-host gates that only run when an
    external server was provided.
    """
    name = config.get("test_name", config["model"])

    class ModelConfigTest(unittest.TestCase):
        def test_accuracy(self):
            run_model_config(config, base_url=base_url)

    # Strip every non-alphanumeric char: unittest test IDs are dot-separated, so a
    # name like "MiMo-V2.5-Pro" would otherwise produce an ambiguous id.
    ModelConfigTest.__name__ = "Test" + "".join(
        part[:1].upper() + part[1:]
        for part in re.split(r"[^0-9A-Za-z]+", str(name))
        if part
    )
    ModelConfigTest.__qualname__ = ModelConfigTest.__name__
    ModelConfigTest.__doc__ = f"Accuracy evals for {name}, sharing one server boot."

    skip = config.get("skip_unless")
    if skip is not None:
        condition, reason = skip
        ModelConfigTest = unittest.skipUnless(condition, reason)(ModelConfigTest)
    return ModelConfigTest


def model_config_main(
    config: dict[str, Any] | list[dict[str, Any]],
    base_url: str | None = None,
) -> None:
    """Entry point for a test file's ``__main__`` block.

    Registers the synthesised case(s) in the caller's module so ``unittest.main()``
    finds them -- the loader only looks at ``__main__``, not at this module.

    A LIST of configs puts several models in one file, one TestCase each, run in
    sequence. They share ``base_url``, and RecipeLauncher evicts whoever holds it,
    so each config still gets exactly the server its own launch env describes --
    the point of a list is one file per concern, not one server for several
    configs. Note the file, not the case, is the unit CI budgets and retries: the
    whole list shares one --timeout-per-file, and failfast stops at the first
    failing case. Split the file when either of those starts to bite.
    """
    import __main__

    configs = config if isinstance(config, list) else [config]
    for entry in configs:
        case = build_test_case(entry, base_url=base_url)
        setattr(__main__, case.__name__, case)
    unittest.main()
