"""P/D gate for the experimental spec worker: serving through the Raiden handoff
compiles no XLA program on either engine.

Requests of staggered lengths drain the decode batch through every live size, so
a program keyed on live rows instead of the batch bucket compiles mid-serving and
the recompilation checker logs it. Prefill checks its own log after the drain;
decode checks its log once the JobSet tears it down.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import requests

from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

SERVING_EVENT = "at serving time"
# The gate recipes' max_running_requests: the drain walks every live size below it.
CONCURRENCY = 16
PREFILL_PORT = int(os.environ.get("PREFILL_PORT", "30000"))
DECODE_PORT = int(os.environ.get("DECODE_PORT", "30000"))
REPO = Path(__file__).absolute().parents[2]
BENCH = REPO / "benchmark" / "tpu"
RENDER_JOBSET = REPO / ".github" / "tpu-ci" / "render_jobset.py"
EXPERIMENTAL_SPEC_ENVS = {
    "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
    "SGLANG_TPU_CHECK_RECOMPILATION": "1",
}


def serving_compiles(log_path: str, start: int = 0) -> list[str]:
    """Recompilation-checker lines logged past byte offset start."""
    with open(log_path, errors="replace") as f:
        f.seek(start)
        return [line.strip() for line in f if SERVING_EVENT in line]


def _self_hostname() -> str:
    if os.environ.get("PD_SELF_HOST"):
        return os.environ["PD_SELF_HOST"]
    hosts = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hosts[0] if hosts else socket.gethostname()


def _server_info(host: str, port: int) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://{host}:{port}/server_info", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


class PDSpecRecompileTestCase(PDSingleHostTestCase):
    """Subclasses set config_module, log paths, tag and served_algorithm."""

    # As /server_info reports it: core resolves NEXTN (MTP) to EAGLE.
    served_algorithm: ClassVar[str] = ""
    repo_root = REPO
    cwd = str(BENCH)
    self_hostname = staticmethod(_self_hostname)
    decode_host_env = "PD_DECODE_HOST"

    # A class attribute, not get_runtime_envs: the decode role launches from it.
    # CI-only FutureMap debug asserts scatter into live rows eagerly, one program
    # per live bs; these servers measure production serving.
    runtime_envs: ClassVar[dict[str, str | None]] = {"SGLANG_IS_IN_CI": None}

    @classmethod
    def check_server_configuration(cls) -> None:
        failures = []
        for role, info in (
            ("prefill", _server_info("127.0.0.1", PREFILL_PORT)),
            ("decode", _server_info(cls.decode_hostname(), DECODE_PORT)),
        ):
            for key, expected in (
                ("disaggregation_mode", role),
                ("disaggregation_transfer_backend", "raiden"),
                ("speculative_algorithm", cls.served_algorithm),
            ):
                if str(info.get(key, "")).upper() != expected.upper():
                    failures.append(
                        f"{role} {key}={info.get(key)!r}, want {expected!r}"
                    )
        if failures:
            raise RuntimeError("; ".join(failures))

    def _generate(self, index: int) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": f"Request {index}: count upward from {index * 7} in words.",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16 + 24 * index,
                    "ignore_eos": True,
                },
            },
            timeout=900,
        )
        response.raise_for_status()
        return response.json()["meta_info"]

    def test_serving_compiles_nothing(self) -> None:
        if self.role == "decode":
            self.run_evaluations()
            self.assertEqual(serving_compiles(self.decode_log_path), [])
            return
        served_from = Path(self.log_path).stat().st_size
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            metas = list(pool.map(self._generate, range(CONCURRENCY)))
        verify_count = sum(meta.get("spec_verify_ct", 0) for meta in metas)
        completion_tokens = sum(meta["completion_tokens"] for meta in metas)
        self.assertGreater(verify_count, 0, "speculative verification never ran")
        # Every verify commits at least one token, so this is accept length > 1.
        self.assertGreater(completion_tokens, verify_count, "no draft was accepted")
        self.assertEqual(serving_compiles(self.log_path, served_from), [])


class PDSpecGateConfigurationTestCase(unittest.TestCase):
    """Launch-contract checks of a P/D gate recipe; no TPU needed."""

    config_module: ClassVar[str] = ""
    expected_algorithm: ClassVar[str] = ""
    # Role -> the launcher's dry-run plan.
    plans: ClassVar[dict[str, dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.config_module:
            raise unittest.SkipTest("abstract configuration test")
        cls.plans = {
            role: json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "benchmark.tpu.launch_server",
                        "--config",
                        f"{cls.config_module}:{role.upper()}_CONFIG",
                        "--runtime-envs",
                        json.dumps(
                            {"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}
                        ),
                        "--dry-run",
                    ],
                    cwd=REPO,
                    text=True,
                )
            )
            for role in ("prefill", "decode")
        }

    def test_both_engines_serve_the_same_spec_flags(self) -> None:
        prefill, decode = (self.plans[r]["server_args"] for r in ("prefill", "decode"))
        self.assertEqual(prefill["disaggregation_mode"], "prefill")
        self.assertEqual(decode["disaggregation_mode"], "decode")
        spec = {k: v for k, v in prefill.items() if k.startswith("speculative_")}
        self.assertEqual(spec["speculative_algorithm"], self.expected_algorithm)
        self.assertEqual(
            spec, {k: v for k, v in decode.items() if k.startswith("speculative_")}
        )
        for plan in self.plans.values():
            self.assertEqual(
                plan["server_args"]["disaggregation_transfer_backend"], "raiden"
            )
            for name in EXPERIMENTAL_SPEC_ENVS:
                self.assertIn(name, plan["runtime_env_keys"])


def render_pd_jobset(test_file: str, run_name: str) -> str:
    """One prefill host plus one decode host running test_file in integration mode."""
    rel_path = os.path.relpath(test_file, REPO)
    return subprocess.check_output(
        [
            os.environ.get("PYTHON", "python3"),
            str(RENDER_JOBSET),
            "--pd-prefill-hosts",
            "1",
            "--pd-decode-hosts",
            "1",
            "--lane",
            "nightly",
            "--image",
            "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
            "--weights-pvc",
            "model-weights",
            "--name",
            run_name,
            "--files",
            rel_path,
            "--exec",
            f"python3 {rel_path} integration",
        ],
        text=True,
    )


def parse_mode() -> tuple[str, list[str]]:
    """The test file's mode and the argv left for unittest.main."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
    )
    args, extra_args = parser.parse_known_args()
    return args.mode, [sys.argv[0], *extra_args]
