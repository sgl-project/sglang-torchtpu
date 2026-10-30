import contextlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.sgl_eval_utils import SGL_EVAL_BENCHMARKS, run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    dump_metric,
    popen_launch_server,
    write_github_step_summary,
)


@dataclass
class AccuracyTestParams:
    """Parameters for accuracy testing."""

    dataset: str  # e.g., "mgsm_en", "gsm8k", "mmmu", "gpqa"
    baseline_accuracy: Optional[float] = None  # None only if `runner` self-gates
    num_examples: Optional[int] = None
    num_threads: Optional[int] = None
    max_tokens: Optional[int] = None
    return_latency: bool = False
    # Extended parameters for special evaluations (e.g., GPQA with thinking mode)
    thinking_mode: Optional[str] = None  # e.g., "deepseek-v3"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat: Optional[int] = None
    api: Optional[str] = None  # "chat" or "completion"; defaults to "chat" in run_eval
    atol: float = 0.0  # floor = baseline_accuracy - atol
    # Replaces the dataset dispatch, so a caller can plug in a harness this module
    # has no dependency on. Returning "_passed" means it gated itself and
    # baseline_accuracy/atol are ignored.
    runner: Optional[Callable[[str], dict]] = None
    extra_eval_args: Optional[dict] = None  # merged into run_eval's namespace
    label: Optional[str] = None  # display/metric name; defaults to `dataset`
    seed: Optional[int] = None  # pin for reproducibility when temperature > 0
    # Optional secondary gate read from /server_info after the evaluation. The
    # primary score remains dataset accuracy. Every scheduler state must meet it.
    spec_accept_length_threshold: Optional[float] = None
    # When set, run_accuracy_test dumps the measured score as `[METRIC] {name}=score`
    # with these labels, whatever backend scored it. This is the one channel that
    # feeds dashboards, so only declared gates are ever plotted; the backends' own
    # dump_metric calls stay as stored-but-unplotted detail.
    metric_labels: Optional[dict] = None
    num_shots: Optional[int] = None  # few-shot count; None = run_eval's default
    # Report a completed score without comparing it to a quality floor.
    report_only: bool = False
    # Fail the gate if any serving-time XLA recompilation or cache load occurred.
    fail_on_recompile: bool = False

    def __post_init__(self) -> None:
        if self.spec_accept_length_threshold is not None and (
            not math.isfinite(self.spec_accept_length_threshold)
            or self.spec_accept_length_threshold < 0.0
        ):
            raise ValueError(
                "spec_accept_length_threshold must be finite and non-negative, got "
                f"{self.spec_accept_length_threshold}"
            )

    @property
    def name(self) -> str:
        return self.label or self.dataset

    @property
    def floor(self) -> Optional[float]:
        if self.baseline_accuracy is None:
            return None
        return self.baseline_accuracy - self.atol


@dataclass
class AccuracyTestResult:
    """Result of an accuracy test."""

    model: str
    dataset: str
    passed: bool
    score: Optional[float]
    baseline_accuracy: Optional[float]
    error: Optional[str]
    latency: Optional[float] = None
    variant: Optional[str] = None


def write_accuracy_github_summary(
    test_name: str,
    dataset: str,
    results: List[AccuracyTestResult],
) -> None:
    """Write accuracy test results to GitHub step summary."""
    summary = f"#### {test_name} - Accuracy ({dataset})\n"
    summary += "| config | status | score | baseline | error |\n"
    summary += "| ------ | ------ | ----- | -------- | ----- |\n"

    for result in results:
        status_emoji = "✅" if result.passed else "❌"
        score_str = f"{result.score:.4f}" if result.score is not None else "N/A"
        baseline_str = (
            f"{result.baseline_accuracy:.4f}"
            if result.baseline_accuracy is not None
            else "from config"
        )
        error_str = result.error if result.error else "-"
        # Use variant name if available, otherwise use model path
        config_name = result.variant if result.variant else result.model
        summary += f"| {config_name} | {status_emoji} | {score_str} | {baseline_str} | {error_str} |\n"

    write_github_step_summary(summary)


@contextlib.contextmanager
def served(model: ModelLaunchSettings, base_url: str):
    """Bring up the server for `model`, yield, then tear it down.

    `model.launcher`, when set, takes over bring-up entirely; see
    ModelLaunchSettings.launcher.
    """
    if model.launcher is not None:
        with model.launcher(base_url):
            yield
        return

    process = popen_launch_server(
        model.model_path,
        base_url,
        other_args=model.extra_args,
        timeout=model.launch_timeout or DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        env=model.env,
    )
    try:
        yield
    finally:
        kill_process_tree(process.pid)


def _get_avg_spec_accept_lengths(base_url: str) -> List[float]:
    """Read each scheduler's cumulative speculative accept length."""
    base_url = base_url.rstrip("/")
    endpoint = "/server_info"
    response = requests.get(f"{base_url}{endpoint}", timeout=30)
    if response.status_code == 404:
        # Older MiniLB releases expose only the deprecated alias. Keep the
        # fallback local to endpoint discovery so other HTTP errors stay fatal.
        endpoint = "/get_server_info"
        response = requests.get(f"{base_url}{endpoint}", timeout=30)
    response.raise_for_status()
    server_info = response.json()
    if not isinstance(server_info, dict):
        raise ValueError(f"{endpoint} did not return an object")

    states = server_info.get("internal_states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"{endpoint} returned no internal_states")

    accept_lengths = []
    for state_index, state in enumerate(states):
        value = state.get("avg_spec_accept_length") if isinstance(state, dict) else None
        if value is None:
            raise ValueError(
                f"{endpoint} scheduler {state_index} has no avg_spec_accept_length"
            )
        accept_length = float(value)
        if not math.isfinite(accept_length):
            raise ValueError(
                f"{endpoint} scheduler {state_index} returned non-finite "
                f"avg_spec_accept_length={value!r}"
            )
        accept_lengths.append(accept_length)

    return accept_lengths


def _run_simple_eval(
    model: ModelLaunchSettings,
    base_url: str,
    dataset: str,
    num_examples: Optional[int] = None,
    num_threads: Optional[int] = None,
    num_shots: Optional[int] = None,
    max_tokens: Optional[int] = None,
    return_latency: bool = False,
    thinking_mode: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat: Optional[int] = None,
    api: Optional[str] = None,
    seed: Optional[int] = None,
    extra_eval_args: Optional[dict] = None,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Run ``dataset`` through run_sgl_eval (sgl-eval benchmarks) or run_eval."""
    try:
        args = SimpleNamespace(
            base_url=base_url,
            model=model.model_path,
            eval_name=dataset,
            num_examples=num_examples,
            num_threads=num_threads or 1024,
        )

        if num_shots is not None:
            args.num_shots = num_shots

        if api is not None:
            args.api = api

        if max_tokens is not None:
            args.max_tokens = max_tokens

        if return_latency:
            args.return_latency = True

        if thinking_mode is not None:
            args.thinking_mode = thinking_mode

        if temperature is not None:
            args.temperature = temperature

        if top_p is not None:
            args.top_p = top_p

        if top_k is not None:
            args.top_k = top_k

        if repeat is not None:
            args.repeat = repeat

        if seed is not None:
            args.seed = seed

        for key, value in (extra_eval_args or {}).items():
            setattr(args, key, value)

        evaluate = run_sgl_eval if dataset in SGL_EVAL_BENCHMARKS else run_eval
        result = evaluate(args)

        # Handle result format (run_eval can return metrics or (metrics, latency))
        if return_latency and isinstance(result, tuple):
            metrics, latency = result
            metrics["latency"] = round(latency, 4)
        else:
            metrics = result

        return True, None, metrics

    except Exception as e:
        return False, f"Accuracy test exception: {str(e)}", None


# Cached uv venv for NeMo Skills (persists across variants within a process).
_nemo_venv_dir: Optional[str] = None
_nemo_data_prepared: set = set()


def _get_nemo_venv() -> Tuple[str, dict]:
    """Get or create a uv venv with nemo_skills installed.

    Returns (venv_python_path, env_dict) reusable across calls.
    """
    import os
    import subprocess
    import tempfile

    global _nemo_venv_dir

    if _nemo_venv_dir is not None:
        venv_python = f"{_nemo_venv_dir}/venv/bin/python"
        env = {
            **dict(os.environ),
            "NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK": "1",
            "OPENAI_API_KEY": "dummy",
            "VIRTUAL_ENV": f"{_nemo_venv_dir}/venv",
            "PATH": f"{_nemo_venv_dir}/venv/bin:" + os.environ.get("PATH", ""),
        }
        return venv_python, env

    _nemo_venv_dir = tempfile.mkdtemp(prefix="nemo_skills_")
    print(f"Creating NeMo Skills venv in {_nemo_venv_dir}...")

    # Create venv
    result = subprocess.run(
        ["uv", "venv", f"{_nemo_venv_dir}/venv", "--python", "3.12"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["uv", "venv", f"{_nemo_venv_dir}/venv"],
            capture_output=True,
            text=True,
        )

    # Install nemo_skills.
    # Pinned: NeMo-Skills main after PR #1433 pins litellm==1.83.14 (httpx==0.28.1),
    # which is unsatisfiable against nemo-run's transitive leptonai dep.
    nemo_skills_ref = "589294c"
    print(f"Installing nemo_skills (pinned to {nemo_skills_ref})...")
    pip_result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            f"{_nemo_venv_dir}/venv/bin/python",
            f"git+https://github.com/NVIDIA/NeMo-Skills.git@{nemo_skills_ref}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if pip_result.returncode != 0:
        raise RuntimeError(f"Failed to install nemo_skills: {pip_result.stderr[-500:]}")

    print("NeMo Skills installed successfully")
    return _get_nemo_venv()


def _ensure_nemo_data_prepared(
    venv_python: str, env: dict, dataset: str
) -> Tuple[bool, Optional[str]]:
    """Prepare NeMo Skills dataset data if not already done.

    Uses the venv python so data lands inside the venv's nemo_skills package.
    """
    import subprocess

    if dataset in _nemo_data_prepared:
        return True, None

    print(f"Preparing {dataset} data (this may take a few minutes for VLM datasets)...")
    result = subprocess.run(
        [venv_python, "-m", "nemo_skills.dataset.prepare", dataset],
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        return False, f"Failed to prepare {dataset} data (exit {result.returncode})"

    _nemo_data_prepared.add(dataset)
    return True, None


def _run_nemo_skills_eval(
    model: ModelLaunchSettings,
    base_url: str,
    dataset: str,
    max_tokens: Optional[int] = None,
    repeat: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Run evaluation using NeMo Skills (ns eval) for benchmarks like mmmu-pro.

    Uses an isolated uv venv (shared across variants) so nemo_skills dependencies
    don't interfere with the system python / sglang server.

    Returns:
        Tuple of (success, error_message, metrics_dict)
    """
    import subprocess
    import tempfile

    try:
        # Get or create the shared venv (once per process)
        venv_python, env = _get_nemo_venv()

        # Prepare dataset (once per process, cached)
        ok, err = _ensure_nemo_data_prepared(venv_python, env, dataset)
        if not ok:
            return False, err, None

        port = int(base_url.split(":")[-1])
        server_address = f"http://127.0.0.1:{port}/v1"
        repeat_val = repeat or 1
        max_tokens_val = max_tokens or 32768
        benchmark_spec = f"{dataset}:{repeat_val}"

        # Build ns eval command using venv python
        # Note: nemo_skills.pipeline.eval requires the "eval" subcommand
        output_dir = tempfile.mkdtemp(prefix="ns_eval_output_")
        cmd = [
            venv_python,
            "-m",
            "nemo_skills.pipeline.eval",
            "eval",
            f"--benchmarks={benchmark_spec}",
            "--server_type=sglang",
            f"--model={model.model_path}",
            f"--server_address={server_address}",
            f"--output_dir={output_dir}",
            f"++inference.tokens_to_generate={max_tokens_val}",
        ]

        if temperature is not None:
            cmd.append(f"++inference.temperature={temperature}")
        if top_p is not None:
            cmd.append(f"++inference.top_p={top_p}")

        # Add VLM-specific config
        if dataset in ("mmmu-pro", "mmmu_pro"):
            cmd.append("++prompt_config=vlm/mmmu-pro")
            cmd.append("++max_concurrent_requests=512")
            cmd.append("++max_samples=500")

        print(f"Running: {' '.join(cmd)}")
        eval_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )

        print(eval_result.stdout[-2000:] if eval_result.stdout else "(no stdout)")
        if eval_result.stderr:
            print(eval_result.stderr[-1000:])

        if eval_result.returncode != 0:
            return (
                False,
                f"ns eval failed (exit {eval_result.returncode}): {eval_result.stderr[-500:]}",
                None,
            )

        # Parse results
        summarize_result = subprocess.run(
            [
                venv_python,
                "-m",
                "nemo_skills.pipeline.summarize_results",
                f"{output_dir}/eval-results",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )

        output = summarize_result.stdout + "\n" + eval_result.stdout
        print(f"Summary: {summarize_result.stdout[:1000]}")

        # Parse accuracy from output (format varies, look for common patterns)
        import re

        score = None
        for line in output.split("\n"):
            match = re.search(r"(?:accuracy|score)[:\s]+([0-9.]+)", line, re.IGNORECASE)
            if match:
                score = float(match.group(1))

        if score is None:
            # Try to find it in eval-results directory
            import glob
            import json

            for result_file in glob.glob(
                f"{output_dir}/eval-results/**/*.json", recursive=True
            ):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        score = (
                            data.get("accuracy")
                            or data.get("score")
                            or data.get("mean_score")
                        )
                        if score is not None:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue

        if score is None:
            # Last resort: compute accuracy directly from JSONL output
            import glob
            import json

            for jsonl_file in sorted(
                glob.glob(f"{output_dir}/eval-results/**/*.jsonl*", recursive=True)
            ):
                correct = 0
                total = 0
                try:
                    with open(jsonl_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            expected = entry.get("expected_answer", "")
                            generation = entry.get("generation", "")
                            # Extract "Answer: X" from the end of generation
                            answer_match = re.search(
                                r"Answer:\s*([A-J])", generation, re.IGNORECASE
                            )
                            if answer_match:
                                predicted = answer_match.group(1).upper()
                                if predicted == expected.upper():
                                    correct += 1
                            total += 1
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
                if total > 0:
                    score = correct / total
                    print(
                        f"Computed accuracy from {jsonl_file}: "
                        f"{correct}/{total} = {score:.4f}"
                    )
                    break

        if score is None:
            return False, "Could not parse accuracy from ns eval output", None

        dump_metric(
            f"{dataset}_score",
            score,
            labels={"model": model.model_path, "eval": dataset, "api": "nemo-skills"},
        )

        return True, None, {"score": score}

    except subprocess.TimeoutExpired:
        return False, "NeMo Skills eval timed out", None
    except Exception as e:
        return False, f"NeMo Skills eval exception: {str(e)}", None


def _dispatch_eval(
    model: ModelLaunchSettings,
    params: AccuracyTestParams,
    base_url: str,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Pick the evaluation backend for `params` and run it against a live server.

    - caller-supplied runner: takes over completely (see AccuracyTestParams.runner)
    - NeMo Skills: mmmu-pro (and other VLM evals needing ns eval)
    - simple_eval: everything else (gsm8k, gpqa, mmlu, mmmu, etc.)
    """
    if params.runner is not None:
        try:
            metrics = params.runner(base_url)
            # A bare score is the natural thing to return; accept it rather than
            # failing later on a dict lookup.
            if isinstance(metrics, (int, float)):
                metrics = {"score": float(metrics)}
            if metrics is None:
                return False, f"Runner for {params.name} returned no metrics", None
            return True, None, metrics
        except Exception as e:
            return False, f"Runner exception: {str(e)}", None

    if params.dataset in ("mmmu-pro", "mmmu_pro"):
        return _run_nemo_skills_eval(
            model=model,
            base_url=base_url,
            dataset="mmmu-pro",
            max_tokens=params.max_tokens,
            repeat=params.repeat or 1,
            temperature=params.temperature,
            top_p=params.top_p,
        )

    return _run_simple_eval(
        model=model,
        base_url=base_url,
        dataset=params.dataset,
        num_examples=params.num_examples,
        num_threads=params.num_threads,
        num_shots=params.num_shots,
        max_tokens=params.max_tokens,
        return_latency=params.return_latency,
        thinking_mode=params.thinking_mode,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        repeat=params.repeat,
        api=params.api,
        seed=params.seed,
        extra_eval_args=params.extra_eval_args,
    )


def run_accuracy_test(
    model: ModelLaunchSettings,
    params: AccuracyTestParams,
    base_url: Optional[str] = None,
) -> AccuracyTestResult:
    """Run accuracy test for a single model."""
    base_url = base_url or DEFAULT_URL_FOR_TEST

    print(f"\n{'=' * 60}")
    print(f"Running ACCURACY test for {model.model_path}")
    print(f"  Eval: {params.name}")
    if params.floor is not None:
        print(f"  Baseline: {params.baseline_accuracy}  atol: {params.atol}")
    if params.spec_accept_length_threshold is not None:
        print(f"  Spec accept-length threshold: {params.spec_accept_length_threshold}")
    print(f"{'=' * 60}\n")

    spec_accept_lengths = None
    spec_accept_length_error = None
    recompile_failures: List[str] = []
    with served(model, base_url):
        success, error, metrics = _dispatch_eval(model, params, base_url)
        if success and params.spec_accept_length_threshold is not None:
            try:
                spec_accept_lengths = _get_avg_spec_accept_lengths(base_url)
            except (TypeError, ValueError, requests.RequestException) as exc:
                spec_accept_length_error = str(exc)
        if success and params.fail_on_recompile:
            from sglang.test.recompilation_gate import (
                check_no_serving_recompilation,
            )

            recompile_failures = check_no_serving_recompilation(base_url)

    if not success:
        print(f"✗ Accuracy test failed for {model.model_path}: {error}")
        return AccuracyTestResult(
            model=model.model_path,
            dataset=params.name,
            passed=False,
            score=None,
            baseline_accuracy=params.baseline_accuracy,
            error=error,
            variant=model.variant,
        )

    # Handle different metric key names: "score", "mean_score" (for GPQA with
    # repeat), "accuracy". `is not None` so a legitimate 0.0 is not skipped.
    score = metrics.get("score")
    if score is None:
        score = metrics.get("mean_score")
    if score is None:
        score = metrics.get("accuracy")
    # A graded gate with no score must not reach the dashboard as a 0.0.
    if score is None and (params.report_only or params.floor is not None):
        error = f"{params.name}: eval completed without a numeric score"
        print(f"✗ {error}")
        return AccuracyTestResult(
            model=model.model_path,
            dataset=params.name,
            passed=False,
            score=None,
            baseline_accuracy=None,
            error=error,
            variant=model.variant,
        )
    latency = metrics.get("latency")
    if score is None:
        score = 0.0

    # Dump per gate, not after the aggregate verdict: a sibling gate failing the
    # file must not cost this one its datapoint.
    if params.metric_labels is not None and score is not None:
        dump_metric(params.name, score, labels=params.metric_labels)

    if params.report_only:
        passed = True
        error = None
        print(f"✓ {params.name}: recorded score {score:.4f} (report only)")
    elif "_passed" in metrics:
        # The runner gated itself (it owns thresholds we cannot see from here).
        passed = metrics["_passed"]
        error = None if passed else "; ".join(metrics.get("_failures") or [])
        print(f"{'✓' if passed else '✗'} {params.name}: {error or 'all floors met'}")
    elif params.floor is None:
        # Nothing to compare against: the params carry no baseline and the runner did
        # not gate itself. Fail loudly -- a silent pass here is an ungated nightly.
        passed = False
        error = (
            f"{params.name}: no baseline_accuracy, and the runner returned no "
            f"'_passed' key -- a gate must either carry a baseline or gate itself"
        )
        print(f"✗ {error}")
    else:
        floor = params.floor
        passed = score >= floor
        if passed:
            print(f"✓ {params.name}: {score:.4f} >= floor {floor:.4f}")
        else:
            error = (
                f"{params.name}: {score:.4f} below floor {floor:.4f} "
                f"(baseline {params.baseline_accuracy} - atol {params.atol})"
            )
            print(f"✗ {error}")

    if params.spec_accept_length_threshold is not None:
        spec_threshold = params.spec_accept_length_threshold
        if spec_accept_length_error is not None:
            spec_error = (
                f"{params.name}: could not read avg_spec_accept_length: "
                f"{spec_accept_length_error}"
            )
            passed = False
            error = "; ".join(filter(None, [error, spec_error]))
            print(f"✗ {spec_error}")
        elif spec_accept_lengths is None:
            spec_error = f"{params.name}: avg_spec_accept_length was not collected"
            passed = False
            error = "; ".join(filter(None, [error, spec_error]))
            print(f"✗ {spec_error}")
        elif min(spec_accept_lengths) < spec_threshold:
            spec_error = (
                f"{params.name}: minimum avg_spec_accept_length "
                f"{min(spec_accept_lengths):.4f} below threshold "
                f"{spec_threshold:.4f}"
            )
            passed = False
            error = "; ".join(filter(None, [error, spec_error]))
            print(f"✗ {spec_error}")
        else:
            print(
                f"✓ {params.name}: avg_spec_accept_length="
                f"{[round(value, 4) for value in spec_accept_lengths]}; "
                f"minimum {min(spec_accept_lengths):.4f} >= threshold "
                f"{spec_threshold:.4f}"
            )

    if params.fail_on_recompile:
        if recompile_failures:
            recompile_error = f"{params.name}: " + "; ".join(recompile_failures)
            passed = False
            error = "; ".join(filter(None, [error, recompile_error]))
            print(f"✗ {recompile_error}")
        else:
            print(f"✓ {params.name}: zero serving-time XLA recompilations")

    return AccuracyTestResult(
        model=model.model_path,
        dataset=params.name,
        passed=passed,
        score=score,
        baseline_accuracy=params.baseline_accuracy,
        error=error if not passed else None,
        latency=latency,
        variant=model.variant,
    )
