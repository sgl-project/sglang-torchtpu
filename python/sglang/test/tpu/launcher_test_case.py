"""Class fixtures that invoke the shared TPU launcher as a separate program."""

import json
import os
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from sglang.test.tpu.eval_support import RecipeLauncher, wait_for_health


class TPULauncherTestCase(unittest.TestCase):
    """Share one launcher across every test method in a subclass.

    Set config_module to a Python recipe and base_url to its server or router.
    Override get_runtime_envs when topology is only available at setup time.
    Set health_generate_timeout to check generation after server configuration
    validation, as P/D tests do once both engines and the router are healthy.
    repo_root defaults to the editable checkout (or Bazel runfiles workspace).
    Override it when the installed SGLang package is outside that checkout.
    Bazel tests depend on //python:tpu_launcher_test_support for the executable.
    """

    config_module: ClassVar[str] = ""
    runtime_envs: ClassVar[Mapping[str, str | None]] = {}
    base_url: ClassVar[str] = ""
    health_timeout: ClassVar[float] = 3600
    health_endpoint: ClassVar[str] = "/health"
    health_generate_timeout: ClassVar[float | None] = None
    repo_root: ClassVar[Path] = Path(__file__).absolute().parents[4]
    cwd: ClassVar[str | None] = None
    log_path: ClassVar[str | None] = None
    keep_up: ClassVar[bool] = False
    tag: ClassVar[str] = "tpu-launcher"
    _launcher: ClassVar[RecipeLauncher | None] = None

    @classmethod
    def get_runtime_envs(cls) -> dict[str, str | None]:
        return dict(cls.runtime_envs)

    @classmethod
    def check_server_configuration(cls) -> None:
        """Optionally validate the ready servers before running any tests."""

    @classmethod
    def _start_launcher(cls, config_module, runtime_envs, *, log_path=None):
        root = Path(cls.repo_root).absolute()
        if os.environ.get("RUNFILES_DIR") or os.environ.get("RUNFILES_MANIFEST_FILE"):
            from python.runfiles import runfiles

            executable = runfiles.Create().Rlocation(
                "sglang_torchtpu/benchmark/tpu/launch_server"
            )
            if not executable or not Path(executable).is_file():
                raise FileNotFoundError(
                    "Missing Bazel launcher; depend on //python:tpu_launcher_test_support"
                )
            command = [executable]
        else:
            if not (root / "benchmark/tpu/launch_server.py").is_file():
                raise FileNotFoundError(
                    f"No TPU launcher in {root}; set repo_root to the repository checkout"
                )
            command = [sys.executable, "-u", "-m", "benchmark.tpu.launch_server"]
        command.extend(
            [
                "--config",
                config_module,
                "--runtime-envs",
                json.dumps(dict(runtime_envs)),
                "--working-directory",
                os.path.abspath(cls.cwd or root / "benchmark/tpu"),
            ]
        )
        launcher = RecipeLauncher(
            command=command,
            env={"PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1")},
            cwd=str(root),
            log_path=log_path,
            start_new_session=True,
            # Allow the supervisor's 30-second child shutdown to finish.
            terminate_timeout=60,
            tag=cls.tag,
        )
        launcher.start()
        if not cls.keep_up:
            # unittest skips tearDownClass when any subclass's setUpClass fails.
            # Class cleanups still run, including for the decode-only fixture.
            # atexit would wait for the runner to exit, not for a child to crash.
            cls.addClassCleanup(cls._stop_launcher)
        return launcher

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not cls.config_module or not cls.base_url:
            raise ValueError("Set config_module and base_url on the test class")
        cls._launcher = cls._start_launcher(
            cls.config_module,
            cls.get_runtime_envs(),
            log_path=cls.log_path,
        )

        try:
            wait_for_health(
                cls.base_url,
                cls.health_timeout,
                process=cls._launcher.process,
                endpoint=cls.health_endpoint,
                tag=cls.tag,
            )
            cls.check_server_configuration()
            if cls.health_generate_timeout is not None:
                wait_for_health(
                    cls.base_url,
                    cls.health_generate_timeout,
                    process=cls._launcher.process,
                    endpoint="/health_generate",
                    tag=cls.tag,
                )
        except BaseException:
            # unittest does not call tearDownClass if setUpClass fails.
            cls._stop_launcher()
            raise

    @classmethod
    def _stop_launcher(cls) -> None:
        if cls._launcher is not None:
            try:
                cls._launcher.shutdown()
            finally:
                cls._launcher = None

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if not cls.keep_up:
                cls._stop_launcher()
        finally:
            super().tearDownClass()
