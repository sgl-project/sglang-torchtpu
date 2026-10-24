"""Launchers and eval runners for the TPU nightly gates.

Plugs into the two extension points on the shared accuracy runner:

* ``ModelLaunchSettings.launcher`` -- a TPU server cannot be started from a bare
  argv, because chiplet visibility, the torch_tpu compile cache and the multi-host
  topology must all be exported before exec. Shell recipes or the unified Python
  launcher configure these before starting the server.
* ``AccuracyTestParams.runner`` -- scores a gate with a harness the shared runner has
  no dependency on: lm-eval, and OpenAI's own gpt_oss.evals package.
"""

import atexit
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.test_utils import (
    ModelLaunchSettings,
    dump_metric,
    terminate_and_kill_process_tree,
)

DEFAULT_HEALTH_TIMEOUT = 1800.0


def wait_for_health(
    base_url: str,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
    process: subprocess.Popen | None = None,
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
                    f"[{tag}] server healthy after {time.perf_counter() - start:.0f}s",
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
_HOLDERS: dict[str, "RecipeLauncher"] = {}


def _tee_log(pipe, log):
    with pipe, log:
        for line in pipe:
            log.write(line)
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except (OSError, ValueError):
                pass  # Keep file logging if the caller's stdout has closed.


class RecipeLauncher:
    """Manage a shell recipe or an explicit launcher command.

    Class fixtures use start(), wait(), and shutdown() to own the process lifetime.
    start() does not wait for health or register deferred cleanup.

    The callable context-manager interface retains servers between evaluations,
    checks readiness on first entry, and shuts down at process exit. Starting
    another callable launcher on the same base_url evicts the previous one.
    """

    def __init__(
        self,
        script: str | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
        tag: str = "tpu-e2e",
        command: list[str] | None = None,
        log_path: str | None = None,
        start_new_session: bool = False,
        terminate_timeout: float | None = None,
    ):
        if (script is None) == (command is None):
            raise ValueError("Provide exactly one of script or command")
        self.script = os.path.abspath(script) if script is not None else None
        if self.script is not None and not os.path.exists(self.script):
            raise FileNotFoundError(f"launch recipe not found: {self.script}")
        self.command = ["bash", self.script] if self.script else list(command)
        if not self.command:
            raise ValueError("command must not be empty")
        self.env = dict(env or {})
        self.cwd = cwd or (os.path.dirname(self.script) if self.script else None)
        self.health_timeout = health_timeout
        self.tag = tag
        self.log_path = log_path
        self.start_new_session = start_new_session
        self.terminate_timeout = terminate_timeout
        self._process: subprocess.Popen | None = None
        self._log_reader: threading.Thread | None = None
        self._base_url: str | None = None

    @property
    def process(self) -> subprocess.Popen | None:
        return self._process

    def start(self) -> subprocess.Popen:
        """Start once without a health check; reject reuse of an exited process."""
        if self._process is not None:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"server died between evals (code {self._process.returncode})"
                )
            return self._process

        env = os.environ.copy()
        env.update(self.env)
        # Default shell recipes to this interpreter; explicit PYTHON wins.
        if self.script and sys.executable:
            env.setdefault("PYTHON", sys.executable)
        overrides = " ".join(f"{k}={v!r}" for k, v in sorted(self.env.items()))
        print(
            f"[{self.tag}] boot: {overrides + ' ' if overrides else ''}"
            f"{shlex.join(self.command)} (cwd={self.cwd})",
            flush=True,
        )
        with contextlib.ExitStack() as stack:
            log = (
                stack.enter_context(
                    open(self.log_path, "a", encoding="utf-8", buffering=1)
                )
                if self.log_path
                else None
            )
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=env,
                start_new_session=self.start_new_session,
                stdout=subprocess.PIPE if log else None,
                stderr=subprocess.STDOUT if log else None,
                text=bool(log),
                encoding="utf-8" if log else None,
                errors="replace" if log else None,
            )
            if log is not None:
                reader = threading.Thread(
                    target=_tee_log, args=(self._process.stdout, log), daemon=True
                )
                try:
                    reader.start()
                except BaseException:
                    self._process.stdout.close()
                    self.shutdown()
                    raise
                self._log_reader = reader
                stack.pop_all()  # The reader thread now owns the log file.
        return self._process

    def wait(self, timeout: float | None = None) -> int:
        """Wait for exit and drain captured output before the caller reads logs."""
        if self._process is None:
            raise RuntimeError("launcher has not been started")
        code = self._process.wait(timeout=timeout)
        if self._log_reader is not None:
            self._log_reader.join(timeout=5)
        return code

    @contextlib.contextmanager
    def __call__(self, base_url: str):
        if self._base_url is None:
            incumbent = _HOLDERS.get(base_url)
            if incumbent is not None and incumbent is not self:
                print(
                    f"[{self.tag}] evicting the previous server on {base_url} "
                    f"({shlex.join(incumbent.command)})",
                    flush=True,
                )
                incumbent.shutdown()

            self.start()
            _HOLDERS[base_url] = self
            self._base_url = base_url
            atexit.register(self.shutdown)
            try:
                wait_for_health(
                    base_url,
                    timeout=self.health_timeout,
                    process=self._process,
                    tag=self.tag,
                )
            except BaseException:
                self.shutdown()
                raise
        else:
            self.start()  # Check that the retained process is still alive.
            print(f"[{self.tag}] reusing live server at {base_url}", flush=True)
        yield self._process
        # Deferred teardown applies only to this evaluation context interface.

    def shutdown(self) -> None:
        try:
            if self._process is not None and self._process.poll() is None:
                print(f"[{self.tag}] shutting down server", flush=True)
                if self.terminate_timeout is None:
                    kill_process_tree(self._process.pid, wait_timeout=60)
                else:
                    terminate_and_kill_process_tree(
                        self._process, terminate_timeout=self.terminate_timeout
                    )
        finally:
            if self._log_reader is not None:
                self._log_reader.join(timeout=5)
                self._log_reader = None
            self._process = None
            if self._base_url is not None and _HOLDERS.get(self._base_url) is self:
                del _HOLDERS[self._base_url]
            self._base_url = None
            atexit.unregister(self.shutdown)


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


def lm_eval_runner(config_path: str, report_only: bool = False):
    """Build an ``AccuracyTestParams.runner`` that scores via lm-eval.

    The YAML owns the task protocol and API model arguments. Thresholds are applied
    unless ``report_only`` is set, in which case the runner returns the raw score.
    """

    def run(base_url: str) -> dict[str, Any]:
        import yaml

        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        # Prefix caching would let a repeated eval short-circuit real prefills.
        try:
            requests.get(base_url.rstrip("/") + "/flush_cache", timeout=30)
        except requests.RequestException:
            pass

        # lm_eval.simple_evaluate() calls torch.manual_seed(), which triggers
        # torch.tpu.manual_seed_all() -> PjrtBackend::EnsureInitialized() if
        # torch_tpu was autoloaded in this HTTP client process while the server
        # process owns /dev/vfio/*.
        import torch

        tpu_mod = getattr(torch, "tpu", None)
        if tpu_mod is not None and not tpu_mod.is_initialized():
            from unittest.mock import patch

            with patch.object(tpu_mod, "is_available", return_value=False):
                results = _LMEvalHarness(base_url).launch_lm_eval(cfg)
        else:
            results = _LMEvalHarness(base_url).launch_lm_eval(cfg)

        # Threshold fields are ignored when this runner is report-only.
        rtol = cfg.get("rtol", LMEvalMixin.default_rtol)
        atol = cfg.get("atol", 0.0)
        passed = True
        failures: list[str] = []
        primary: float | None = None

        for task in cfg["tasks"]:
            for metric in task["metrics"]:
                measured = results["results"][task["name"]][metric["name"]]
                if primary is None:
                    primary = measured
                if report_only:
                    print(
                        f"{task['name']} | {metric['name']}: "
                        f"measured={measured:.3f} (report only)"
                    )
                else:
                    ground_truth = metric["value"]
                    floor = ground_truth * (1 - rtol) - atol
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
                if not report_only and measured < floor:
                    passed = False
                    failures.append(
                        f"{task['name']}/{metric['name']}: "
                        f"{measured:.4f} < floor {floor:.4f}"
                    )

        if report_only:
            return {"score": primary}
        return {"score": primary, "_passed": passed, "_failures": failures}

    return run


def _bound_gpt_oss_sampler(sampler: Any, sampler_module: Any) -> None:
    """Keep a capped generation from turning into unbounded regeneration.

    The reference sampler retries forever when Chat Completions returns reasoning
    without final content. Local TPU evals intentionally cap generation, so preserve
    that reasoning as the scoreable response and bound genuine request failures.
    """

    original_call = sampler.__class__.__call__
    original_sleep = sampler_module.time.sleep

    def bounded_call(self: Any, message_list: Any) -> Any:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

        for trial in range(2):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": message_list,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.reasoning_model:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                reasoning = getattr(choice.message, "reasoning", None)
                content = choice.message.content
                if reasoning:
                    message_list.append(self._pack_message("assistant", reasoning))
                content = content or reasoning or "No response."
                return sampler_module.SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except sampler_module.openai.BadRequestError as error:
                print("Bad Request Error", error)
                break
            except Exception as error:
                if trial == 1:
                    print("Request failed after 2 attempts", error)
                    break
                delay = min(2**trial, 30)
                print(f"Request failed; retrying once after {delay}s", error)
                original_sleep(delay)

        return sampler_module.SamplerResponse(
            response_text="No response.",
            response_metadata={"usage": None},
            actual_queried_message_list=message_list,
        )

    sampler._sglang_original_call = original_call
    sampler.__class__.__call__ = bounded_call


def _restore_gpt_oss_sampler(sampler: Any) -> None:
    sampler.__class__.__call__ = sampler._sglang_original_call
    del sampler._sglang_original_call


def gpqa_openai_runner(
    model: str,
    n_repeats: int,
    reasoning_effort: str,
    max_tokens: int,
    n_threads: int = 32,
    num_examples: int | None = None,
):
    """Build an ``AccuracyTestParams.runner`` that scores GPQA via OpenAI's harness.

    ``gpt_oss.evals`` supplies the permuted-choice prompt and grader used here.
    Temperature 1.0 follows that harness's protocol; the caller owns the repeat and
    sample counts.

    ``model`` is both the id sent to the server and the metric label, so a gate cannot
    report under a name the server was never asked for.
    """

    def run(base_url: str) -> dict[str, Any]:
        # Pure-python eval code; --no-deps so its pins never touch the pinned
        # TPU torch stack. openai/pandas/blobfile are image deps (test extra).
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "gpt-oss",
            ],
            check=True,
        )
        # The openai client insists on a key even for a local base_url.
        os.environ.setdefault("OPENAI_API_KEY", "dummy")

        import gpt_oss.evals.chat_completions_sampler as sampler_module
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
        _bound_gpt_oss_sampler(sampler, sampler_module)
        try:
            result = GPQAEval(
                n_repeats=n_repeats,
                n_threads=n_threads,
                num_examples=num_examples,
            )(sampler)
        finally:
            _restore_gpt_oss_sampler(sampler)
        print(
            f"gpqa_diamond (openai harness, n_repeats={n_repeats}, "
            f"num_examples={num_examples}): "
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


def aime25_openai_runner(
    model: str,
    n_repeats: int,
    reasoning_effort: str,
    max_tokens: int,
    n_threads: int = 32,
):
    """Score AIME 2025 with OpenAI's GPT-OSS reference harness."""

    def run(base_url: str) -> dict[str, Any]:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "gpt-oss"],
            check=True,
        )
        os.environ.setdefault("OPENAI_API_KEY", "dummy")

        import gpt_oss.evals.chat_completions_sampler as sampler_module
        from gpt_oss.evals.aime_eval import AIME25Eval
        from gpt_oss.evals.chat_completions_sampler import ChatCompletionsSampler

        sampler = ChatCompletionsSampler(
            model=model,
            reasoning_model=True,
            reasoning_effort=reasoning_effort,
            temperature=1.0,
            base_url=base_url.rstrip("/") + "/v1",
            max_tokens=max_tokens,
        )
        _bound_gpt_oss_sampler(sampler, sampler_module)
        try:
            result = AIME25Eval(n_repeats=n_repeats, n_threads=n_threads)(sampler)
        finally:
            _restore_gpt_oss_sampler(sampler)
        print(
            f"aime25 (openai harness, n_repeats={n_repeats}): "
            f"score={result.score:.4f} metrics={result.metrics}"
        )
        dump_metric(
            "aime25_openai_harness",
            result.score,
            labels={"model": model, "eval": "gpt_oss.evals"},
        )
        return {"score": result.score}

    return run


def sgl_eval_args(
    eval_name: str,
    *,
    base_url: str,
    model: str,
    out_dir: str,
    max_tokens: int,
    num_threads: int,
    temperature: float,
    top_p: float,
    n_repeats: int = 1,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """The sgl-eval argv for one gate, shared by the runner and the sharder.

    One builder so a sharded eval sends the same prompt and sampling as an
    unsharded one; only the row list differs.
    """
    args = [
        "run",
        eval_name,
        "--base-url",
        base_url.rstrip("/") + "/v1",
        "--model",
        model,
        "--max-tokens",
        str(max_tokens),
        "--num-threads",
        str(num_threads),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--n-repeats",
        str(n_repeats),
        "--out-dir",
        str(out_dir),
    ]
    for key, value in (chat_template_kwargs or {}).items():
        # --thinking is the same thing: both land in chat_template_kwargs.
        args += ["--chat-template-kwarg", f"{key}={json.dumps(value)}"]
    return args


def sgl_eval_runner(
    eval_name: str,
    *,
    model: str,
    max_tokens: int,
    num_threads: int,
    temperature: float,
    top_p: float,
    n_repeats: int = 1,
    chat_template_kwargs: dict[str, Any] | None = None,
):
    """Score a gate with ``sgl-eval run <eval_name>`` over its full dataset.

    Two reasons to prefer this over run_eval's sgl-eval path: that path can only
    toggle ``chat_template_kwargs.thinking``, so a template reading any other key
    needs ``chat_template_kwargs``; and it routes ``mmmu_pro`` to the shared
    runner, which scores NeMo-Skills' 500-sample subset rather than the full set.
    """

    def run(base_url: str) -> dict[str, Any]:
        out_dir = Path(tempfile.mkdtemp(prefix=f"sgl-eval-{eval_name}-"))
        cmd = [
            "sgl-eval",
            *sgl_eval_args(
                eval_name,
                base_url=base_url,
                model=model,
                out_dir=str(out_dir),
                max_tokens=max_tokens,
                num_threads=num_threads,
                temperature=temperature,
                top_p=top_p,
                n_repeats=n_repeats,
                chat_template_kwargs=chat_template_kwargs,
            ),
        ]
        print("Running: " + shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        metrics_files = list(out_dir.glob(f"sgl_eval_{eval_name}_*/metrics.json"))
        if len(metrics_files) != 1:
            raise RuntimeError(
                f"Expected one sgl-eval metrics.json, found {metrics_files}"
            )
        payload = json.loads(metrics_files[0].read_text())
        print(
            f"{eval_name} (sgl-eval, n_repeats={n_repeats}): "
            f"score={payload['aggregate']['score']:.4f} "
            f"examples={payload.get('num_examples')} metrics={payload['aggregate']}"
        )
        dump_metric(
            f"{eval_name}_sgl_eval",
            payload["aggregate"]["score"],
            labels={"model": model, "eval": eval_name},
        )
        return {
            **payload["aggregate"],
            "latency": payload.get("latency_seconds", 0.0),
            "output_throughput": payload.get("output_throughput_tps", 0.0),
        }

    return run


# --- accuracy models ----------------------------------------------------------
# Per-model identity for the accuracy nightlies, as data: a new model is an entry
# here, not a module of its own. An entry says how its server starts -- either a
# `config_module` recipe its test's fixture boots. Serving settings are NOT here:
# a benchmark that needs a different server gets its own recipe rather than env
# patched in at launch, so the shape a score was measured at stays with the
# score. `sampling` is the model card's decoding settings.

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT))

GPT_OSS_120B = {
    "served_model_name": "gpt-oss-120b",
    "display_name": "GPT-OSS 120B",
    "variant": "TP8+EP8",
    # Per benchmark: recipe/configs/gpt_oss_120b/accuracy_<benchmark>.py.
    "config_module": "recipe/configs/gpt_oss_120b/accuracy.py",
}

# Shared by the entries below that differ only in checkpoint and recipe.
_GEMMA4_SAMPLING = {
    "max_tokens": 32768,
    "num_threads": 64,
    "temperature": 1.0,
    "top_p": 0.95,
    # The Gemma 4 template reads enable_thinking; sgl-eval's --thinking sets `thinking`.
    "chat_template_kwargs": {"enable_thinking": True},
}

GEMMA4_31B = {
    "served_model_name": "gemma4-31b",
    "display_name": "Gemma 4 31B",
    "variant": "TP8",
    "config_module": "recipe/configs/gemma4_31b/accuracy.py",
    "sampling": _GEMMA4_SAMPLING,
}

QWEN3_5_397B = {
    "served_model_name": "Qwen/Qwen3.5-397B-A17B-FP8",
    "display_name": "Qwen3.5 397B-A17B",
    "variant": "TP8+EP8-fp8",
    "config_module": "recipe/configs/qwen3_5_397b/server.py",
    "sampling": {
        "max_tokens": 32768,
        "num_threads": 128,
        "temperature": 0.6,
        "top_p": 0.95,
        # The Qwen3.5 template reads enable_thinking, not the generic thinking.
        "chat_template_kwargs": {"enable_thinking": True},
    },
}

QWEN3_8_27B = {
    "served_model_name": "Qwen/Qwen3.8-27B",
    "display_name": "Qwen3.8 27B",
    "variant": "TP8",
    "config_module": "recipe/configs/qwen3_8_27b/accuracy.py",
    # The model card's thinking-mode sampling; its top_k=20 comes from the
    # checkpoint's generation_config.json, which the server applies by default.
    "sampling": {
        # At 32768, 10.8% of AIME26 samples truncated without an answer; at
        # 81920, 3.3% (measured on tpu7x, 2026-10-12).
        "max_tokens": 81920,
        "num_threads": 64,
        "temperature": 1.0,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": True},
    },
}

KIMI_K26 = {
    "served_model_name": "/weights/Kimi-K2.6",
    "display_name": "Kimi K2.6",
    "variant": "2host-TP16+EP16+DPA4",
    "config_module": "recipe/configs/kimi_k26/mmmu.py",
    # The 2026-09-23 full-set run's settings; its template reads `thinking`.
    "sampling": {
        "max_tokens": 98304,
        "num_threads": 256,
        "temperature": 1.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"thinking": True},
    },
}

GEMMA4_26B_A4B = {
    "served_model_name": "gemma4-26b-a4b",
    "display_name": "Gemma 4 26B-A4B",
    "variant": "TP8+EP8",
    "config_module": "recipe/configs/gemma4_26b_a4b/accuracy.py",
    "sampling": _GEMMA4_SAMPLING,
}

GLM_5_3_FLASH = {
    "served_model_name": "zai-org/GLM-5.3-Flash",
    "display_name": "GLM-5.3-Flash",
    "variant": "TP8-DSA",
    "config_module": "recipe/configs/glm_5_3_flash/accuracy.py",
    # The model card's sampling; the template always thinks, with no toggle.
    "sampling": {
        "max_tokens": 290000,
        "num_threads": 32,
        "temperature": 1.0,
        "top_p": 0.95,
    },
}


def fixture_runtime_envs(base_url: str = BASE_URL) -> dict[str, str | None]:
    """Fixture environment: MODEL=None keeps a stray job MODEL from winning."""
    return {
        "MODEL": None,
        "HOST": urlsplit(base_url).hostname,
        "PORT": str(urlsplit(base_url).port),
    }


def make_runner(model: dict[str, Any], eval_name: str, *, n_repeats: int = 1):
    """Score one benchmark through sgl-eval with this model's card sampling."""
    return sgl_eval_runner(
        eval_name,
        model=model["served_model_name"],
        n_repeats=n_repeats,
        **model["sampling"],
    )


def make_model_config(
    model: dict[str, Any],
    *,
    test_name: str,
    series_id: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Build one isolated model configuration for one benchmark."""
    served = model["served_model_name"]
    return {
        "model": served,
        "test_name": test_name,
        # The dashboard's series join key: never renamed, never shown. A forced
        # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
        "id": series_id,
        # Cosmetic and free to change -- the split from `id` is what makes a
        # display rename a non-event for the trend.
        "display_name": model["display_name"],
        "variant": model["variant"],
        "dashboard": "accuracy",
        "dashboard_model": served,
        "gates": [gate],
    }


# --- perf ---------------------------------------------------------------------
# A throughput sweep is an AccuracyTestParams.runner like the two above, not a
# parallel harness: it wants the same recipe launch, the same one-boot-many-gates
# sharing, and the same reporting. run_combined_tests' own performance_params
# cannot serve here -- it launches with popen_launch_server, and a TPU server only
# comes up through launch_server.sh (chiplet visibility, compile cache, topology).

_TPU_BENCH_DIR = Path(__file__).resolve().parents[5] / "benchmark" / "tpu"


# bench_serving.sh's own per-point file stem. Its tag and this must agree, or the
# sweep would measure fine and the reader would report a point that never ran.
def bench_point_tag(concurrency: int, isl: int, osl: int) -> str:
    return f"c{concurrency}-isl{isl}-osl{osl}"


def bench_sweep_arg(points: list[dict[str, Any]]) -> str:
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


def read_bench_point(out_dir: str, point: dict[str, Any]) -> dict[str, Any] | None:
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


# The framed-payload egress every TPU perf test shares (the standard set by
# test_tpu_qwen3_coder_480b_multihost_agentic.py): metrics are dumped per-number
# via dump_metric when a leg is clean, and the WHOLE measurement summary is also
# printed once, machine-readable, between these markers. The dashboard keys on the
# marker text, so it never changes.
RESULT_BEGIN = "=== TPU_BENCH_RESULT_JSON BEGIN ==="
RESULT_END = "=== TPU_BENCH_RESULT_JSON END ==="


def emit_result_payload(summary: Any) -> None:
    """Print ``summary`` between the RESULT markers, shaped to survive the log.

    The only egress from a TPU JobSet is the streamed pod log, and containerd
    caps a single log record at 16 KiB -- a summary emitted as one long line is
    split into CRI partial records and has twice (2026-08-26, 08-28) taken the
    rest of the job log with it. ``indent`` keeps every line short without
    changing what the consumer sees: _tpu-test.yml's awk rejoins the block with
    newlines, and pretty-printed JSON only ever breaks BETWEEN tokens, so the
    rejoined text parses identically. Chunking a single line at a fixed width
    would not -- it would drop a newline inside a string literal and corrupt the
    payload.
    """
    try:
        blob = json.dumps(summary, sort_keys=True, default=str, indent=1)
    except (TypeError, ValueError) as exc:
        # sort_keys raises on mixed-type keys. A lost payload is recoverable from
        # the digest lines printed before it; an exception here would take down
        # the caller, reporting nothing about the measurement itself.
        print(f"{RESULT_BEGIN}\n{RESULT_END}", flush=True)
        print(f"[payload] summary not serializable, payload skipped: {exc!r}")
        return
    print(RESULT_BEGIN, flush=True)
    print(blob, flush=True)
    print(RESULT_END, flush=True)


# Every numeric field bench_serving writes per sweep point, and how to read it.
# Dumped in full because the JSONL never leaves the pod and the job log expires at
# ~90 days: a field not recorded tonight cannot be back-filled later without
# re-running the sweep. The cost is a few hundred bytes a night.
#
# Config echoes in the same record (backend, dataset_name, tag, server_info, the
# random_* shape, request_rate) are deliberately NOT dumped -- they are settings,
# not measurements, and the dashboard already captures them from the config block
# bench_serving.sh prints.
_HIGHER_IS_BETTER = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "max_output_tokens_per_s",
    "accept_length",  # speculative decoding; absent otherwise
)
_LOWER_IS_BETTER = tuple(
    f"{stat}_{family}"
    for family in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_latency_ms")
    for stat in ("mean", "median", "std", "p90", "p95", "p99")
) + ("duration",)
# Counters worth keeping but not worth judging. `completed` should equal the
# expected request count exactly -- more is not better, which is why it is not
# higher_is_better; the gate checks it instead. Cache counters are only
# meaningful against a run's own cache setting: this sweep disables prefix
# caching on purpose, so a rise would mean the cache leaked in, not that things
# improved.
_INFORMATIONAL = (
    "completed",
    "total_input_tokens",
    "total_input_text_tokens",
    "total_input_vision_tokens",
    "total_output_tokens",
    "total_output_tokens_retokenized",
    "total_prompt_tokens",
    "total_cached_tokens",
    "device_cached_tokens",
    "host_cached_tokens",
    "storage_cached_tokens",
    "cache_hit_rate_pct",
    "max_concurrent_requests",
)

_BENCH_METRICS = {
    **{f: "higher_is_better" for f in _HIGHER_IS_BETTER},
    **{f: "lower_is_better" for f in _LOWER_IS_BETTER},
    **{f: "informational" for f in _INFORMATIONAL},
}

# The one series per model a reader should look at first.
_HEADLINE_METRIC = "total_throughput"


def bench_serving_runner(
    sweep: list[dict[str, Any]],
    test_name: str,
    model: str = "",
    bench_env: dict[str, str] | None = None,
    prompts_per_concurrency: int = 10,
    series_id: str = "",
    display_name: str = "",
    smoke_state: dict[str, Any] | None = None,
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

    A point that fails -- a preceding smoke leg red (``smoke_state``, written by
    the smoke gates model_config builds), the script exiting non-zero, or the
    measurement incomplete -- dumps a ``smoke_failed`` record instead of numbers.
    The dashboard renders it as a ✕ on that night's chart with ``reason`` on
    hover, so a failed run is explained rather than hidden.

    ``series_id``/``display_name`` come from MODEL_CONFIG's ``id``/``display_name``
    and ride on every dashboard-bound record: ``id`` is the stable series join key
    (never renamed; forced changes go through
    .github/tpu-ci/dashboard_aliases.json), ``display_name`` is cosmetic.
    """

    def dashboard_labels(point: dict[str, Any]) -> dict[str, Any]:
        # kind/model/workload route the record to the right Performance chart;
        # id joins the series across nights, display_name is what it is called.
        labels = {
            "kind": "performance",
            "model": model or test_name,
            "workload": point["workload"],
        }
        if series_id:
            labels["id"] = series_id
        if display_name:
            labels["display_name"] = display_name
        return labels

    def dump_cross(point: dict[str, Any], reason: str) -> None:
        # The ✕ record: same routing labels as the numbers it replaces, so the
        # server/bench config panels stay attached to the failed night too.
        dump_metric(
            "smoke_failed", 1, labels={**dashboard_labels(point), "reason": reason}
        )

    def run(base_url: str) -> dict[str, Any]:
        smoke_failures = (smoke_state or {}).get("failures") or []
        if smoke_failures:
            # A smoke leg already failed: the measurement would be garbage, so no
            # numbers are plotted -- each point gets a ✕ carrying the smoke reason.
            reason = "; ".join(smoke_failures)
            for point in sweep:
                dump_cross(point, reason)
            emit_result_payload(
                {
                    "test": test_name,
                    "model": model or test_name,
                    "legs": [
                        {
                            "workload": point["workload"],
                            "concurrency": point["concurrency"],
                            "status": "withheld",
                            "reason": reason,
                        }
                        for point in sweep
                    ],
                }
            )
            return {
                "score": 0.0,
                "_passed": False,
                "_failures": [f"sweep withheld: {reason}"],
            }
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
        legs = []

        def fail_point(point, reason):
            failures.append(reason)
            dump_cross(point, reason)
            legs.append(
                {
                    "workload": point["workload"],
                    "concurrency": point["concurrency"],
                    "status": "failed",
                    "reason": reason,
                }
            )

        for point in sweep:
            label = f"{test_name} {point['workload']} c{point['concurrency']}"
            record = read_bench_point(search_dir, point)
            if record is None:
                fail_point(point, f"{label}: no JSONL -- the point never ran")
                continue
            expected = point["concurrency"] * prompts_per_concurrency
            if record.get("completed") != expected:
                # A partial point must not be published as a number: charted next
                # to complete ones it poisons the trend (the 2026-08-24 gsm8k 0.0
                # was exactly this shape of bug). The ✕ record explains the gap.
                fail_point(
                    point,
                    f"{label}: {record.get('completed')} of {expected} requests "
                    "completed; the rest errored, so its throughput is not "
                    "comparable",
                )
                continue
            total = record.get("total_throughput")
            if not total or total <= 0:
                fail_point(point, f"{label}: total_throughput is {total}")
                continue
            headline = headline or total
            legs.append(
                {
                    "workload": point["workload"],
                    "concurrency": point["concurrency"],
                    "status": "clean",
                    "metrics": record,
                }
            )
            print(
                f"[perf] {label}: {total:.0f} tok/s total, "
                f"ttft {record.get('median_ttft_ms', float('nan')):.0f} ms",
                flush=True,
            )
            # Built-in backends dump their own metrics; a runner must do its own.
            # Everything bench_serving measured is dumped, not just the headline:
            # the [METRIC] line is the ONLY thing that survives -- the JSONL dies
            # with the pod and the log expires at ~90 days, so a field not dumped
            # tonight cannot be charted later without re-running the sweep. The
            # cost of carrying all of them is a few hundred bytes a night.
            stem = f"{test_name}.{point['workload']}"
            for field, direction in _BENCH_METRICS.items():
                value = record.get(field)
                if not isinstance(value, (int, float)):
                    continue
                # `direction` says which way is an improvement, because a
                # regression check that assumes higher-is-better reads a TTFT
                # drop as a win.
                labels = {"c": point["concurrency"], "direction": direction}
                if field == _HEADLINE_METRIC:
                    # kind/model/workload route the headline to the Performance
                    # tab. The other 40-odd land as kind=info: stored in the
                    # dashboard's archive forever, shown in the point's details,
                    # but charted nowhere -- which is the point. Recording them
                    # costs a few hundred bytes a night and cannot be done
                    # retroactively once the job log expires; charting them all
                    # would bury the one number anyone opens the page for under
                    # forty series. To chart one later, promote it on the
                    # dashboard side -- the historical records already carry the data.
                    labels.update(dashboard_labels(point))
                    labels["headline"] = "1"
                else:
                    labels["kind"] = "info"
                    labels["model"] = model or test_name
                    labels["workload_name"] = point["workload"]
                    if series_id:
                        labels["id"] = series_id
                dump_metric(f"{stem}.{field}", value, labels=labels)

        # The framed one-block summary, after the per-number dumps: clean legs
        # carry their full record, failed ones their reason -- the same egress
        # shape as the 480B agentic sweep.
        emit_result_payload(
            {"test": test_name, "model": model or test_name, "legs": legs}
        )

        return {
            "score": headline or 0.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run
