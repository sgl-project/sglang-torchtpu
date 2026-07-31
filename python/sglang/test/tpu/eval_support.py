"""Launchers and eval runners for the TPU nightly gates.

Plugs into the two extension points on the shared accuracy runner:

* ``ModelLaunchSettings.launcher`` -- a TPU server cannot be started from a bare
  argv, because chiplet visibility, the torch_tpu compile cache and the multi-host
  topology must all be exported before exec, and only ``launch_server.sh`` (plus its
  recipe wrappers) knows how.
* ``AccuracyTestParams.runner`` -- scores a gate with a harness the shared runner has
  no dependency on (lm-eval here, OpenAI's own package in the gpt-oss test).
"""

import atexit
import contextlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from sglang.srt.utils import kill_process_tree
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


class RecipeLauncher:
    """Launch a TPU server from a shell recipe, and keep it up across evals.

    The shared runner brings the server up once per evaluation, but a TPU boot costs
    tens of minutes (weight load plus per-shape compile). This boots on first entry,
    attaches on later ones, and tears down at process exit, so N gates cost one boot.

    Assumes one model config per test process. Two configs would contend for the
    port; call ``shutdown()`` between them.
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

    @contextlib.contextmanager
    def __call__(self, base_url: str):
        if self._process is None:
            env = os.environ.copy()
            env.update(self.env)
            print(f"[RecipeLauncher] boot: bash {self.script} (cwd={self.cwd})")
            self._process = subprocess.Popen(
                ["bash", self.script], cwd=self.cwd, env=env
            )
            # The recipe and launch_server.sh both `exec`, so this pid IS the
            # server process -- kill_process_tree on it does the right thing.
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
            print(f"[RecipeLauncher] reusing live server at {base_url}")
        yield self._process
        # Deliberately not torn down here: see the class docstring.

    def shutdown(self) -> None:
        if self._process is not None and self._process.poll() is None:
            print("[RecipeLauncher] shutting down server")
            kill_process_tree(self._process.pid, wait_timeout=60)
        self._process = None


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
            print(f"[ExternalServerLauncher] attaching to {base_url}")
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


def lm_eval_runner(
    config_path: str,
    default_rtol: float = 0.08,
):
    """Build an ``AccuracyTestParams.runner`` that scores via lm-eval.

    Needed because these models' reference values are the cards' standard few-shot
    numbers (5-shot ``mmlu_llama``, ``mmlu_pro``), which simple-evals does not
    reproduce. The YAML is the only copy of the ground truths and tolerances, so the
    runner gates itself and reports ``_passed``.
    """

    def run(base_url: str) -> Dict[str, Any]:
        import lm_eval
        import yaml

        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        # Prefix caching would let a repeated eval short-circuit real prefills.
        try:
            requests.get(base_url.rstrip("/") + "/flush_cache", timeout=30)
        except requests.RequestException:
            pass

        backend = cfg.get("backend", "local-completions")
        api = cfg.get("api", "chat" if "chat" in backend else "completion")
        suffix = "/v1/chat/completions" if api == "chat" else "/v1/completions"
        model_args = {
            "model": cfg["model_name"],
            "base_url": base_url.rstrip("/") + suffix,
            "num_concurrent": cfg.get("num_concurrent", 1),
        }
        model_args.update(cfg.get("model_args", {}))

        results = lm_eval.simple_evaluate(
            model=backend,
            model_args=model_args,
            tasks=[task["name"] for task in cfg["tasks"]],
            num_fewshot=cfg.get("num_fewshot", 0),
            limit=cfg.get("limit", None),
            apply_chat_template=cfg.get("apply_chat_template", False),
            fewshot_as_multiturn=cfg.get("fewshot_as_multiturn", False),
            gen_kwargs=cfg.get("gen_kwargs"),
            batch_size=cfg.get("batch_size", "auto"),
        )

        rtol = cfg.get("rtol", default_rtol)
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
