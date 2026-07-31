"""Run a TPU accuracy nightly from a declarative ``MODEL_CONFIG`` dict.

Adding a model means copying a test file and editing its dict. This module turns that
dict into the launcher, the ``AccuracyTestParams`` list and the ``run_combined_tests``
call, so a test file is a register call, a dict, and two lines to run it:

    register_tpu_ci(est_time=10800, suite="nightly-tpu-4-v7", nightly=True)

    MODEL_CONFIG = {
        "model": "/weights/gpt-oss-120b",
        "test_name": "gpt-oss-120b",
        "variant": "TP8+EP8",
        "launch": {"recipe": "recipe/gpt_oss_120b.sh", "tag": "gpqa-e2e"},
        "gates": [
            {"dataset": "mmlu", "baseline": 0.859, "atol": 0.03,
             "num_examples": 2000, "reasoning_effort": "low"},
        ],
    }

    if __name__ == "__main__":
        model_config_main(MODEL_CONFIG)

Top level
    model         Model path or HF id. Reported, and sent as the model id by the eval
                  clients; the recipe is what actually sets MODEL on the server.
    gates         Evaluations to run, all sharing one server boot.
    launch        How to bring the server up (see below).
    test_name     Report label, and the basis of the generated TestCase name.
                  Defaults to `model`.
    variant       Distinguishes launch configs of the same model in reports, e.g.
                  "TP8+EP8". Cosmetic while a model has one config.
    skip_unless   (condition, reason) -- the config form of @unittest.skipUnless.

launch
    recipe        Shell recipe under benchmark/tpu/, e.g. "recipe/gpt_oss_120b.sh".
                  Preferred over `env`: the recipe is what a human runs to reproduce
                  a CI failure, so keeping config there leaves only one copy.
    env           Extra environment for the recipe. Prefer the recipe itself.
    external      True -> attach to a server an external orchestrator already started
                  (the multi-host JobSets). Takes no recipe/env.
    tag           Prefix on this model's progress lines, e.g. "gpqa-e2e".
    health_timeout  Seconds to wait for /health_generate before giving up.

gates -- one entry per evaluation
    dataset       run_eval eval name ("gsm8k", "mmlu", "gpqa", "aime25", ...), or a
                  descriptive name when `runner`/`lm_eval_config` does the scoring.
    baseline      Reference score, normally the model card's.
    atol          Tolerance below it: floor = baseline - atol.
    label         Name in reports and metrics. Defaults to `dataset`; set it when one
                  dataset is gated more than once.
    repeat        Passes to average. Small datasets need several before a tight
                  tolerance means anything -- see AIME in test_tpu_gpt_oss_120b.py
                  for how to size it.
    reasoning_effort   "low"/"medium"/"high". Cards report a different score per
                  level, so this must match whichever the baseline came from.
    runner        runner(base_url) -> score or metrics dict. Scores the gate with a
                  harness this module has no dependency on.
    lm_eval_config     YAML under test/lm_eval_configs/. It owns the ground truths and
                  tolerances and is also what run_lm_eval.sh reads, so `baseline`
                  and `atol` must not be set alongside it.
    num_examples, num_threads, max_tokens, temperature, top_p, top_k, api
                  Passed straight through to run_eval.

Two things stay in the test file. ``register_tpu_ci(...)`` is read by parsing source
with ``ast``, so it must appear literally; it is also what lets est_time, suite and
disabled differ per model. And the ``__main__`` guard is required because ci_utils
runs each file as ``python3 <file> -f``, while importing one (tests do) must not
execute anything.

Unknown keys raise rather than being ignored -- a silently-dropped typo in a
threshold is the failure mode this shape would otherwise invite.
"""

import re
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.tpu.eval_support import (
    DEFAULT_HEALTH_TIMEOUT,
    ExternalServerLauncher,
    RecipeLauncher,
    TpuModelLaunchSettings,
    lm_eval_runner,
)

_TPU_BENCH_DIR = Path(__file__).resolve().parents[4] / "benchmark" / "tpu"

_CONFIG_KEYS = {"model", "test_name", "variant", "launch", "gates", "skip_unless"}
_LAUNCH_KEYS = {"recipe", "env", "tag", "health_timeout", "external"}
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
    "runner",
}
# Gate keys needing translation.
_GATE_TRANSLATED = {"baseline", "reasoning_effort", "lm_eval_config"}


def _reject_unknown(got, allowed, where: str) -> None:
    unknown = set(got) - allowed
    if unknown:
        raise ValueError(
            f"unknown {where} key(s): {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def _build_launcher(launch: Dict[str, Any]):
    _reject_unknown(launch, _LAUNCH_KEYS, "launch")
    timeout = launch.get("health_timeout", DEFAULT_HEALTH_TIMEOUT)
    tag = launch.get("tag", "tpu-e2e")

    if launch.get("external"):
        if "recipe" in launch or "env" in launch:
            raise ValueError("external launch takes no recipe/env: the JobSet owns it")
        return ExternalServerLauncher(health_timeout=timeout, tag=tag)

    recipe = launch.get("recipe")
    if not recipe:
        raise ValueError('launch needs either "recipe" or "external": True')
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


def _build_gate(gate: Dict[str, Any]) -> AccuracyTestParams:
    _reject_unknown(gate, _GATE_PASSTHROUGH | _GATE_TRANSLATED, "gate")
    kwargs = {k: v for k, v in gate.items() if k in _GATE_PASSTHROUGH}

    if "baseline" in gate:
        kwargs["baseline_accuracy"] = gate["baseline"]
    if "reasoning_effort" in gate:
        kwargs["extra_eval_args"] = {"reasoning_effort": gate["reasoning_effort"]}
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
        kwargs["runner"] = lm_eval_runner(gate["lm_eval_config"])

    return AccuracyTestParams(**kwargs)


def build_gates(config: Dict[str, Any]) -> List[AccuracyTestParams]:
    """Expand the ``gates`` list. Split out so tests can assert on it directly."""
    return [_build_gate(g) for g in config["gates"]]


def build_model(config: Dict[str, Any]) -> TpuModelLaunchSettings:
    """Build the launch settings. Split out so tests can assert on it directly."""
    # No tp_size: the recipe owns parallelism and TpuModelLaunchSettings clears the
    # extra_args it would have fed, so it would only be a second copy that can drift.
    return TpuModelLaunchSettings(
        config["model"],
        launcher=_build_launcher(config.get("launch", {})),
        variant=config.get("variant"),
    )


def run_model_config(config: Dict[str, Any], base_url: Optional[str] = None) -> dict:
    """Run every gate in ``config`` against one server boot."""
    _reject_unknown(config, _CONFIG_KEYS, "MODEL_CONFIG")
    for required in ("model", "gates"):
        if required not in config:
            raise ValueError(f'MODEL_CONFIG needs "{required}"')

    return run_combined_tests(
        models=[build_model(config)],
        test_name=config.get("test_name", config["model"]),
        base_url=base_url,
        accuracy_params=build_gates(config),
    )


def build_test_case(config: Dict[str, Any], base_url: Optional[str] = None):
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
    ModelConfigTest.__doc__ = f"Accuracy gates for {name}, sharing one server boot."

    skip = config.get("skip_unless")
    if skip is not None:
        condition, reason = skip
        ModelConfigTest = unittest.skipUnless(condition, reason)(ModelConfigTest)
    return ModelConfigTest


def model_config_main(config: Dict[str, Any], base_url: Optional[str] = None) -> None:
    """Entry point for a test file's ``__main__`` block.

    Registers the synthesised case in the caller's module so ``unittest.main()``
    finds it -- the loader only looks at ``__main__``, not at this module.
    """
    import __main__

    case = build_test_case(config, base_url=base_url)
    setattr(__main__, case.__name__, case)
    unittest.main()
