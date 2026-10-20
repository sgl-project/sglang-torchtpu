"""Shared class fixture and host-side checks for the accuracy nightlies.

Subclasses supply config_module and build_model_config; server startup and
shutdown remain in the inherited class lifecycle. The companion helpers run
without TPU hardware, so a test file can assert its own serving shape in a
plain unittest.TestCase beside its evaluation.
"""

import os
import sys
import unittest
from typing import Any, ClassVar

from sglang.test.tpu.evals.eval_support import wait_for_health
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
# The multi-host JobSet starts one server across the slice and runs this file on
# rank 0 only, so there the fixture attaches rather than starting a second one.
LIVE_SERVER = os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1"
live_server_only = unittest.skipUnless(
    LIVE_SERVER, "multi-host live server not provided (the JobSet starts it)"
)


def validate_model_source(model_path: str, *, env_var: str = "MODEL") -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint.

    The shell launcher refused to start on a missing checkpoint; the Python
    launcher does not, so without this the run dies deep in weight loading.
    """
    if not os.path.isabs(model_path):
        return  # HF repo ids resolve remotely
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"checkpoint incomplete or missing at {model_path}, provision it on "
            f"the weights volume or set {env_var}"
        )


def recipe_server_args(config_module: str) -> dict[str, Any]:
    """The server_args a Python recipe resolves to.

    A test that has to agree with its recipe on a value -- a pool size its probe
    divides by, a page size it floors to -- reads it here instead of restating it.
    """
    # benchmark/ is a checkout directory, not a package: import it from the repo root.
    root = str(TPULauncherTestCase.repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmark.tpu.launch_server import load_recipe

    return load_recipe(config_module)["server_args"]


def gate_server_failures(config_module: str, *sampling: dict[str, Any]) -> list[str]:
    """Reject sampling that does not fit the server this recipe starts.

    Both ways round cost a whole job: a generation budget past the context
    length truncates every answer, and more client threads than the server
    admits queues them behind each other for the length of the eval. Takes the
    gate itself, or the catalog sampling when a runner owns the request.
    """
    args = recipe_server_args(config_module)
    asked = {k: v for spec in sampling for k, v in spec.items()}
    failures = []
    max_tokens = asked.get("max_tokens")
    context_length = args.get("context_length")
    if max_tokens and context_length and max_tokens > context_length:
        failures.append(
            f"gate asks for {max_tokens} output tokens, but {config_module} "
            f"serves context_length {context_length}"
        )
    threads = asked.get("num_threads")
    running = args.get("max_running_requests")
    if threads and running and threads > running:
        failures.append(
            f"gate drives {threads} threads, but {config_module} admits "
            f"{running} requests at a time"
        )
    return failures


class AccuracyTestCase(TPULauncherTestCase):
    """Boot this model's recipe, then score its gates against that server."""

    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    # Set when the checkpoint is a mounted path CI must have provisioned.
    model_path: ClassVar[str | None] = None

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        raise NotImplementedError("Provide this test's MODEL_CONFIG")

    @classmethod
    def get_runtime_envs(cls) -> dict[str, str | None]:
        from urllib.parse import urlsplit

        # MODEL=None keeps a stray job MODEL from overriding the recipe's path.
        return {
            "MODEL": None,
            "HOST": urlsplit(cls.base_url).hostname,
            "PORT": str(urlsplit(cls.base_url).port),
        }

    @classmethod
    def setUpClass(cls) -> None:
        if LIVE_SERVER:
            # Attach. Starting a server here would be a second copy on chips the
            # JobSet's is already using, and on a TP-spanning slice only rank 0
            # serves HTTP, so there is nothing for this process to launch.
            wait_for_health(
                cls.base_url,
                cls.health_timeout,
                endpoint=cls.health_endpoint,
                tag=cls.tag,
            )
            return
        if cls.model_path:
            validate_model_source(cls.model_path)
        super().setUpClass()

    def run_evaluations(self) -> None:
        from sglang.test.tpu import run_model_config

        run_model_config(self.build_model_config(), base_url=self.base_url)
