"""Launchers and eval runners for the TPU nightly gates.

Plugs into the two extension points on the shared accuracy runner:

* ``ModelLaunchSettings.launcher`` -- a TPU server cannot be started from a bare
  argv, because chiplet visibility, the torch_tpu compile cache and the multi-host
  topology must all be exported before exec, and only ``launch_server.sh`` (plus its
  recipe wrappers) knows how.
* ``AccuracyTestParams.runner`` -- scores a gate with a harness the shared runner has
  no dependency on: lm-eval, and OpenAI's own gpt_oss.evals package.
"""

import atexit
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.test_utils import ModelLaunchSettings, dump_metric

DEFAULT_HEALTH_TIMEOUT = 1800.0


def wait_for_health(
    base_url: str,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
    process: Optional[subprocess.Popen] = None,
    endpoint: str = "/health_generate",
    tag: str = "tpu-e2e",
) -> None:
    """Block until the server answers, or raise.

    ``/health_generate`` rather than ``/health``: on TPU a server binds its port long
    before the last shape finishes compiling, and only the former runs a generation.

    Notes progress every 60s: weight load plus compile takes ~10 min single-host and
    longer multi-host, and a silent wait is indistinguishable from a hang in CI logs.
    """
    start = time.perf_counter()
    deadline = start + timeout
    last_note = start
    last = None
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            response = requests.get(base_url.rstrip("/") + endpoint, timeout=10)
            if response.status_code == 200:
                print(
                    f"[{tag}] server healthy after "
                    f"{time.perf_counter() - start:.0f}s",
                    flush=True,
                )
                return
            last = f"HTTP {response.status_code}"
        except requests.RequestException as e:
            last = str(e)
        if time.perf_counter() - last_note >= 60:
            last_note = time.perf_counter()
            print(
                f"[{tag}] waiting for {base_url}{endpoint} "
                f"({time.perf_counter() - start:.0f}s elapsed; last: {last})",
                flush=True,
            )
        time.sleep(5)
    raise TimeoutError(
        f"{base_url}{endpoint} did not return 200 within {timeout}s (last: {last})"
    )


# One server per base_url. Teardown is deferred to keep gates cheap, so a second
# launch config wanting the same URL must evict the incumbent or find the port taken.
_HOLDERS: Dict[str, "RecipeLauncher"] = {}


class RecipeLauncher:
    """Launch a TPU server from a shell recipe, and keep it up across evals.

    The shared runner brings the server up once per evaluation, but a TPU boot costs
    tens of minutes (weight load plus per-shape compile). This boots on first entry,
    attaches on later ones, and tears down at process exit, so N gates cost one boot.

    When several variants of a model run in sequence, each has its own launcher and
    they share a base_url; booting one evicts whoever currently holds that URL.
    """

    def __init__(
        self,
        script: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
        tag: str = "tpu-e2e",
    ):
        self.script = os.path.abspath(script)
        if not os.path.exists(self.script):
            raise FileNotFoundError(f"launch recipe not found: {self.script}")
        self.env = dict(env or {})
        self.cwd = cwd or os.path.dirname(self.script)
        self.health_timeout = health_timeout
        self.tag = tag
        self._process: Optional[subprocess.Popen] = None
        self._base_url: Optional[str] = None

    @contextlib.contextmanager
    def __call__(self, base_url: str):
        if self._process is None:
            incumbent = _HOLDERS.get(base_url)
            if incumbent is not None and incumbent is not self:
                print(
                    f"[{self.tag}] evicting the previous server on {base_url} "
                    f"({os.path.basename(incumbent.script)})",
                    flush=True,
                )
                incumbent.shutdown()

            env = os.environ.copy()
            env.update(self.env)
            # The whole invocation, env included: this line is what a reader reproduces
            # a red nightly with.
            overrides = " ".join(f"{k}={v!r}" for k, v in sorted(self.env.items()))
            print(
                f"[{self.tag}] boot: {overrides + ' ' if overrides else ''}"
                f"bash {self.script} (cwd={self.cwd})",
                flush=True,
            )
            self._process = subprocess.Popen(
                ["bash", self.script], cwd=self.cwd, env=env
            )
            # The recipe and launch_server.sh both `exec`, so this pid IS the
            # server process -- kill_process_tree on it does the right thing.
            _HOLDERS[base_url] = self
            self._base_url = base_url
            atexit.register(self.shutdown)
            wait_for_health(
                base_url,
                timeout=self.health_timeout,
                process=self._process,
                tag=self.tag,
            )
        else:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"server died between evals (code {self._process.returncode})"
                )
            print(f"[{self.tag}] reusing live server at {base_url}", flush=True)
        yield self._process
        # Not torn down here -- see the class docstring.

    def shutdown(self) -> None:
        if self._process is not None and self._process.poll() is None:
            print(f"[{self.tag}] shutting down server", flush=True)
            kill_process_tree(self._process.pid, wait_timeout=60)
        self._process = None
        if self._base_url is not None and _HOLDERS.get(self._base_url) is self:
            del _HOLDERS[self._base_url]
        self._base_url = None


class ExternalServerLauncher:
    """Attach to a server someone else started; never launch or kill anything.

    Used by the multi-host gates, where a JobSet brings the server up across hosts
    and the test runs on rank 0 against it.
    """

    def __init__(
        self,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
        tag: str = "tpu-e2e",
    ):
        self.health_timeout = health_timeout
        self.tag = tag
        self._checked = False

    @contextlib.contextmanager
    def __call__(self, base_url: str):
        if not self._checked:
            print(f"[{self.tag}] attaching to {base_url}", flush=True)
            wait_for_health(base_url, timeout=self.health_timeout, tag=self.tag)
            self._checked = True
        yield None


class TpuModelLaunchSettings(ModelLaunchSettings):
    """ModelLaunchSettings that delegates bring-up to a ``launcher``.

    ``launcher`` itself is a plain ModelLaunchSettings field; this subclass exists
    only to clear ``extra_args``, because the base class synthesises ``--tp`` /
    ``--enable-multimodal`` -- dead config here (the recipe derives every flag from
    environment) that would mislead in reports.
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.extra_args = []


class _LMEvalHarness(LMEvalMixin):
    """Adapter that lets a runner call ``LMEvalMixin.launch_lm_eval``.

    ``launch_lm_eval`` needs nothing but ``base_url``, and it is the one implementation
    of the lm-eval invocation -- endpoint suffix, model_args, ``env_vars`` scoping --
    shared with every other lm-eval test and with run_lm_eval.sh.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url


def lm_eval_runner(config_path: str):
    """Build an ``AccuracyTestParams.runner`` that scores via lm-eval.

    Needed because these models' reference values are the cards' standard few-shot
    numbers (5-shot ``mmlu_llama``, ``mmlu_pro``), which simple-evals does not
    reproduce. The YAML is the only copy of the ground truths and tolerances, so the
    runner gates itself and reports ``_passed``.
    """

    def run(base_url: str) -> Dict[str, Any]:
        import yaml

        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        # Prefix caching would let a repeated eval short-circuit real prefills.
        try:
            requests.get(base_url.rstrip("/") + "/flush_cache", timeout=30)
        except requests.RequestException:
            pass

        results = _LMEvalHarness(base_url).launch_lm_eval(cfg)

        # Same floor as LMEvalMixin.test_lm_eval, which asserts instead of reporting.
        rtol = cfg.get("rtol", LMEvalMixin.default_rtol)
        atol = cfg.get("atol", 0.0)
        passed = True
        failures: List[str] = []
        primary: Optional[float] = None

        for task in cfg["tasks"]:
            for metric in task["metrics"]:
                ground_truth = metric["value"]
                measured = results["results"][task["name"]][metric["name"]]
                floor = ground_truth * (1 - rtol) - atol
                if primary is None:
                    primary = measured
                print(
                    f"{task['name']} | {metric['name']}: "
                    f"ground_truth={ground_truth:.3f} | measured={measured:.3f} | "
                    f"floor={floor:.3f} | rtol={rtol} | atol={atol}"
                )
                dump_metric(
                    f"{task['name']}_{metric['name']}",
                    measured,
                    labels={
                        "model": cfg.get("model_name", ""),
                        "eval": "lm-eval",
                        "task": task["name"],
                    },
                )
                if measured < floor:
                    passed = False
                    failures.append(
                        f"{task['name']}/{metric['name']}: "
                        f"{measured:.4f} < floor {floor:.4f}"
                    )

        return {"score": primary, "_passed": passed, "_failures": failures}

    return run


def gpqa_openai_runner(
    model: str,
    n_repeats: int,
    reasoning_effort: str,
    max_tokens: int,
    n_threads: int = 32,
):
    """Build an ``AccuracyTestParams.runner`` that scores GPQA via OpenAI's harness.

    gpt_oss.evals is the only harness that reproduces the gpt-oss card's GPQA number --
    it owns the permuted-choice prompt and the grader, and temperature 1.0 below is part
    of that methodology. ``n_repeats`` is the caller's, because how many passes a tight
    floor needs depends on the tolerance it is gated at.

    ``model`` is both the id sent to the server and the metric label, so a gate cannot
    report under a name the server was never asked for.
    """

    def run(base_url: str) -> Dict[str, Any]:
        # Pure-python eval code; --no-deps so its pins never touch the pinned
        # TPU torch stack. openai/pandas/blobfile are image deps (test extra).
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "gpt-oss"],
            check=True,
        )
        # The openai client insists on a key even for a local base_url.
        os.environ.setdefault("OPENAI_API_KEY", "dummy")

        from gpt_oss.evals.chat_completions_sampler import ChatCompletionsSampler
        from gpt_oss.evals.gpqa_eval import GPQAEval

        sampler = ChatCompletionsSampler(
            model=model,
            reasoning_model=True,
            reasoning_effort=reasoning_effort,
            temperature=1.0,
            base_url=base_url.rstrip("/") + "/v1",
            # Their CLI uses 131072; the caller caps it to fit CONTEXT_LEN.
            max_tokens=max_tokens,
        )
        result = GPQAEval(n_repeats=n_repeats, n_threads=n_threads)(sampler)
        print(
            f"gpqa_diamond (openai harness, n_repeats={n_repeats}): "
            f"score={result.score:.4f} metrics={result.metrics}"
        )
        # Built-in backends dump their own metrics; a runner must do its own.
        dump_metric(
            "gpqa_diamond_openai_harness",
            result.score,
            labels={"model": model, "eval": "gpt_oss.evals"},
        )
        return {"score": result.score}

    return run


# --- perf ---------------------------------------------------------------------
# A throughput sweep is an AccuracyTestParams.runner like the two above, not a
# parallel harness: it wants the same recipe launch, the same one-boot-many-gates
# sharing, and the same reporting. run_combined_tests' own performance_params
# cannot serve here -- it launches with popen_launch_server, and a TPU server only
# comes up through launch_server.sh (chiplet visibility, compile cache, topology).

_TPU_BENCH_DIR = Path(__file__).resolve().parents[4] / "benchmark" / "tpu"


# bench_serving.sh's own per-point file stem. Its tag and this must agree, or the
# sweep would measure fine and the reader would report a point that never ran.
def bench_point_tag(concurrency: int, isl: int, osl: int) -> str:
    return f"c{concurrency}-isl{isl}-osl{osl}"


def bench_sweep_arg(points: List[Dict[str, Any]]) -> str:
    """The SWEEP string for bench_serving.sh: "c:isl:osl:workload" per point.

    ``workload`` is the name the perf dashboard's Workload dropdown shows, so the
    test that defines a shape is what names it.
    """
    required = {"workload", "isl", "osl", "concurrency"}
    tokens = []
    for point in points:
        unknown = set(point) - required
        if unknown:
            raise ValueError(f"unknown sweep point key(s): {sorted(unknown)}")
        missing = sorted(required - set(point))
        if missing:
            raise ValueError(f"sweep point is missing {missing}: {point}")
        workload = str(point["workload"])
        if not workload or ":" in workload:
            raise ValueError(
                f"workload name must be non-empty and ':'-free: {workload!r}"
            )
        tokens.append(
            f"{point['concurrency']}:{point['isl']}:{point['osl']}:{workload}"
        )
    return " ".join(tokens)


def bench_output_dir(test_name: str) -> str:
    """Where a sweep writes: under the root the CI collector reads, when set.

    Empty means "leave OUTPUT_DIR unset", so bench_serving.sh's own
    ./results/<slug>-tpu/<ts> default applies -- a path the collector also globs.
    """
    root = os.environ.get("TPU_CI_BENCH_OUTPUT_ROOT")
    return os.path.join(root, test_name) if root else ""


def read_bench_point(out_dir: str, point: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The last JSONL record a sweep point wrote, or None if it never landed.

    Searched recursively: with OUTPUT_DIR unset the file is under a generated
    ./results/<slug>-tpu/<timestamp>/ path. Last line, because serving.py appends.
    """
    stem = bench_point_tag(point["concurrency"], point["isl"], point["osl"]) + ".jsonl"
    matches = sorted(Path(out_dir).rglob(stem))
    if not matches:
        return None
    record = None
    with open(matches[-1], encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
    return record


def bench_serving_runner(
    sweep: List[Dict[str, Any]],
    test_name: str,
    model: str = "",
    bench_env: Optional[Dict[str, str]] = None,
    prompts_per_concurrency: int = 10,
):
    """Build an ``AccuracyTestParams.runner`` that measures throughput.

    The measurement is benchmark/tpu/bench_serving.sh -- the same script a human
    runs by hand -- so CI and a manual repro produce the same JSONL, and the numbers
    reach the dashboard through the collector that already reads that directory.
    Nothing is re-derived here.

    WHAT IT GATES. Not a throughput threshold: the dashboard's trend is the
    regression detector, and a floor invented before any nightly has run would
    either never fire or fire on noise. It gates that the measurement is COMPLETE --
    every point wrote its JSONL, no request errored, throughput positive -- which is
    what separates "slower" from "the server served nothing".
    """

    def run(base_url: str) -> Dict[str, Any]:
        out_dir = bench_output_dir(test_name)
        env = os.environ.copy()
        # bench_serving.sh exits 2 without MODEL, and it is the tokenizer source, so
        # default it to the config's model. A test whose weights are a local path
        # overrides MODEL in bench_env and names the model with SERVED_MODEL_NAME.
        if model:
            env["MODEL"] = model
        env.update({k: str(v) for k, v in (bench_env or {}).items()})
        env["BASE_URL"] = base_url
        env["SWEEP"] = bench_sweep_arg(sweep)
        env["PROMPTS_PER_CONCURRENCY"] = str(prompts_per_concurrency)
        if out_dir:
            env["OUTPUT_DIR"] = out_dir
            os.makedirs(out_dir, exist_ok=True)

        print(f"[perf] {test_name}: SWEEP={env['SWEEP']}", flush=True)
        # No capture_output: bench_serving.sh's own per-point block is the log a
        # human reads when a number looks wrong, and swallowing it hides the run.
        proc = subprocess.run(
            ["bash", str(_TPU_BENCH_DIR / "bench_serving.sh")],
            cwd=str(_TPU_BENCH_DIR),
            env=env,
        )
        search_dir = out_dir or str(_TPU_BENCH_DIR / "results")

        failures = []
        if proc.returncode != 0:
            failures.append(f"bench_serving.sh exited {proc.returncode}")

        headline = None
        for point in sweep:
            label = f"{test_name} {point['workload']} c{point['concurrency']}"
            record = read_bench_point(search_dir, point)
            if record is None:
                failures.append(f"{label}: no JSONL -- the point never ran")
                continue
            expected = point["concurrency"] * prompts_per_concurrency
            if record.get("completed") != expected:
                failures.append(
                    f"{label}: {record.get('completed')} of {expected} requests "
                    "completed; the rest errored, so its throughput is not comparable"
                )
                # And for the same reason it must not be published: a partial
                # point charted next to complete ones poisons the trend (the
                # 2026-08-24 gsm8k 0.0 was exactly this shape of bug).
                continue
            total = record.get("total_throughput")
            if not total or total <= 0:
                failures.append(f"{label}: total_throughput is {total}")
                continue
            headline = headline or total
            print(
                f"[perf] {label}: {total:.0f} tok/s total, "
                f"ttft {record.get('median_ttft_ms', float('nan')):.0f} ms",
                flush=True,
            )
            # Built-in backends dump their own metrics; a runner must do its own.
            dump_metric(
                f"{test_name}.{point['workload']}.total_throughput",
                total,
                labels={
                    # model keys the dashboard's series and ties the point back to
                    # the server/bench config captured for the same model.
                    "model": model or test_name,
                    "workload": point["workload"],
                    "c": point["concurrency"],
                },
            )

        return {
            "score": headline or 0.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run
