"""Process-local driver for generated runtime monitors."""

from __future__ import annotations

import logging
import os
import threading

from sglang.srt.raiden_runtime_verification.generated_runtime_monitor import (
    GeneratedRuntimeMonitor,
)

logger = logging.getLogger(__name__)


class RuntimeMonitorDriver:
    """Route observations to one generated monitor per bootstrap room."""

    def __init__(self) -> None:
        self.process_id = os.getpid()
        self._monitors: dict[int | None, GeneratedRuntimeMonitor] = {}
        self._violated = False
        self._lock = threading.Lock()

    def step(
        self,
        *atomic_propositions: str,
        bootstrap_room: int | None = None,
    ) -> bool:
        valuation = set(atomic_propositions)
        with self._lock:
            monitor = self._monitors.get(bootstrap_room)
            if monitor is None:
                monitor = GeneratedRuntimeMonitor(trace_id=bootstrap_room)
                self._monitors[bootstrap_room] = monitor

            logger.info(
                "RuntimeMonitorDriver observed atomic propositions for "
                "bootstrap_room=%s: %s",
                bootstrap_room,
                sorted(valuation),
            )
            accepted = monitor.step(valuation)
            if not accepted:
                self._violated = True
                logger.error(
                    "RuntimeMonitorDriver detected a violation for atomic "
                    "propositions in bootstrap_room=%s: %s. The full observed "
                    "trace is available in the preceding RuntimeMonitorDriver "
                    "logs.",
                    bootstrap_room,
                    sorted(valuation),
                )
            return accepted

    def discard(self, bootstrap_room: int) -> None:
        with self._lock:
            self._monitors.pop(bootstrap_room, None)

    @property
    def violated(self) -> bool:
        with self._lock:
            return self._violated


_driver_lock = threading.Lock()
_driver: RuntimeMonitorDriver | None = None


def get_runtime_monitor_driver() -> RuntimeMonitorDriver:
    """Return one driver per OS process, including after a fork."""
    global _driver
    process_id = os.getpid()
    with _driver_lock:
        if _driver is None or _driver.process_id != process_id:
            _driver = RuntimeMonitorDriver()
        return _driver
