"""Shared class fixture and role dispatch for one-host-per-engine P/D tests."""

import os
import re
import signal
from pathlib import Path
from typing import Any, ClassVar

from sglang.test.tpu.pd_router import SMG, resolve_mode
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase


def peer_hostname(hostname: str, *, from_role: str, to_role: str) -> str:
    """Map this pod's DNS name onto the pod of the peer role."""
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


class PDSingleHostTestCase(TPULauncherTestCase):
    """Run evaluations on prefill while the decode pod serves independently.

    Subclasses supply config_module, log paths, self_hostname, and
    build_model_config. They may also override check_server_configuration.
    Server startup and shutdown remain in the inherited class lifecycle.
    """

    base_url = f"http://127.0.0.1:{int(os.environ.get('LB_PORT', '8000'))}"
    health_timeout = 2 * int(os.environ.get("WAIT_SECS", "3600")) + 300
    health_generate_timeout = float(os.environ.get("HEALTH_TIMEOUT", "3600"))
    keep_up = os.environ.get("KEEP_UP", "0") == "1"
    tag = "pd-singlehost"
    decode_log_path: ClassVar[str | None] = None
    decode_host_env: ClassVar[str | None] = None

    @staticmethod
    def self_hostname() -> str:
        raise NotImplementedError("Provide this test's hostname resolver")

    @classmethod
    def decode_hostname(cls) -> str:
        if cls.decode_host_env and os.environ.get(cls.decode_host_env):
            return os.environ[cls.decode_host_env]
        return peer_hostname(cls.self_hostname(), from_role="prefill", to_role="decode")

    @classmethod
    def get_runtime_envs(cls) -> dict[str, str | None]:
        return {
            **super().get_runtime_envs(),
            "SGLANG_LAUNCHER_HOSTNAMES": f"{cls.self_hostname()},{cls.decode_hostname()}",
            "SGLANG_LAUNCHER_RANK": "0",
        }

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        raise NotImplementedError("Provide this test's evaluation configuration")

    @classmethod
    def setUpClass(cls) -> None:
        cls.role = os.environ.get("CI_PD_ROLE")
        if cls.role == "prefill":
            if resolve_mode() == SMG:
                cls.health_endpoint = "/readyz"
                cls.health_generate_timeout = None
            super().setUpClass()
        elif cls.role == "decode":
            cls._launcher = cls._start_launcher(
                cls.config_module + ":DECODE_CONFIG",
                {**cls.runtime_envs, "SGLANG_LAUNCHER_RANK": "0"},
                log_path=cls.decode_log_path,
            )
        else:
            raise ValueError(f"Expected CI_PD_ROLE=prefill or decode, got {cls.role!r}")

    def run_evaluations(self) -> None:
        if self.role == "decode":
            # Match the JobSet's worker teardown policy: SIGTERM is expected, and a
            # SIGKILL is expected only when the prefill peer disconnected.
            code = self._launcher.wait()
            if code in (-signal.SIGTERM, 128 + signal.SIGTERM):
                return
            if code in (-signal.SIGKILL, 128 + signal.SIGKILL):
                log = Path(self.decode_log_path) if self.decode_log_path else None
                if (
                    log
                    and log.is_file()
                    and re.search(
                        r"Connection (closed|reset) by peer",
                        log.read_text(errors="replace"),
                    )
                ):
                    return
            self.assertEqual(code, 0)
            return

        from sglang.test.tpu import run_model_config

        run_model_config(self.build_model_config(), base_url=self.base_url)
