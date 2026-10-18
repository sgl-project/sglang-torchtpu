"""P/D /health_generate from whether held requests produce anything (TPU plugin).

Each scheduler rank keeps one ``PDProgressHealth`` (``health_generate_scheduler``),
fed by the real requests it holds (``observe``), the last output
(``record_progress``) and the last engine-side abort (``record_failure``), and
writes its verdict to a slot file in /dev/shm that the HTTP process reads
(``health_generate_http``).

UNHEALTHY once demand has been continuous for ``window_s`` and nothing was
produced in that time; demand is a held request or an engine failure within
the window. Idle is always ready, the first request after idleness gets the
whole window, ``cold_window_s`` applies until the first output after start,
and recovery is stateless. The clock is injected for tests.
"""

from __future__ import annotations

import enum
import glob
import os
import time
from typing import Callable, Optional

import msgspec

from sglang_tpu.envs import envs as tpu_envs


class PDHealthState(str, enum.Enum):
    READY = "ready"
    UNHEALTHY = "unhealthy"


class PDHealthConfig(msgspec.Struct, frozen=True, kw_only=True):
    window_s: float = 600.0
    cold_window_s: float = 1200.0

    @classmethod
    def from_env(cls) -> PDHealthConfig:
        return cls(
            window_s=tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S.get(),
            cold_window_s=tpu_envs.SGLANG_TPU_PD_HEALTH_COLD_WINDOW_S.get(),
        )

    def validate(self) -> None:
        if self.window_s <= 0:
            raise ValueError("SGLANG_TPU_PD_HEALTH_WINDOW_S must be > 0")
        if self.cold_window_s < self.window_s:
            raise ValueError(
                "SGLANG_TPU_PD_HEALTH_COLD_WINDOW_S must be >= SGLANG_TPU_PD_HEALTH_WINDOW_S"
            )


class PDHealthSnapshot(msgspec.Struct, kw_only=True):
    """Returned in the probe's 503 body."""

    role: str = "decode"
    state: str = PDHealthState.READY.value
    reason: str = "ready"
    requests_present: int = 0
    demand_age_s: Optional[float] = None
    last_failure_age_s: Optional[float] = None
    last_progress_age_s: Optional[float] = None
    window_s: float = 600.0
    since_s: float = 0.0

    @property
    def ready(self) -> bool:
        return self.state != PDHealthState.UNHEALTHY.value

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode()

    @classmethod
    def from_json(cls, text: str) -> PDHealthSnapshot:
        return msgspec.json.decode(text, type=cls)


class PDProgressHealth:
    def __init__(
        self,
        config: Optional[PDHealthConfig] = None,
        clock: Callable[[], float] = time.monotonic,
        role: str = "decode",
    ):
        self.config = config or PDHealthConfig()
        self.config.validate()
        self._clock = clock
        self.role = role
        self.state = PDHealthState.READY
        self.version = 0
        self._requests_present = 0
        self._demand_since: Optional[float] = None
        self._last_progress_ts: Optional[float] = None
        self._last_failure_ts: Optional[float] = None
        self._state_since = clock()

    def record_progress(self) -> None:
        """A real request produced something; the window restarts from now."""
        self._last_progress_ts = self._clock()
        self._judge()

    def record_failure(self) -> None:
        """The engine concluded a real request without output; demand persists."""
        now = self._clock()
        self._last_failure_ts = now
        if self._demand_since is None:
            self._demand_since = now
        self._judge()

    def observe(self, requests_present: int) -> PDHealthState:
        """The real requests held right now; every scheduler iteration, or on
        every bookkeeping event when fed from the HTTP process."""
        self._requests_present = requests_present
        return self._refresh()

    def _refresh(self) -> PDHealthState:
        """Re-apply the demand rule at the current time with the facts in hand.

        Time alone changes the verdict: a failure ages out of the window with no
        new event to report it, so every read re-judges rather than trusting
        the state left by the last event.
        """
        now = self._clock()
        recent_failure = (
            self._last_failure_ts is not None
            and now - self._last_failure_ts < self._window()
        )
        if self._requests_present <= 0 and not recent_failure:
            self._demand_since = None
        elif self._demand_since is None:
            self._demand_since = now
        return self._judge()

    # ----- what the slot carries -----

    def health_state(self) -> str:
        return self._refresh().value

    def health_json(self) -> str:
        return self.snapshot().to_json()

    def requests_present(self) -> int:
        return self._requests_present

    def stale_after_s(self) -> float:
        # A scheduler that stops publishing with work in hand is judged on the
        # same window as one that publishes and produces nothing.
        return self._window()

    def snapshot(self) -> PDHealthSnapshot:
        self._refresh()
        now = self._clock()
        return PDHealthSnapshot(
            role=self.role,
            state=self.state.value,
            reason=self._reason(now),
            requests_present=self._requests_present,
            demand_age_s=(
                round(now - self._demand_since, 1)
                if self._demand_since is not None
                else None
            ),
            last_progress_age_s=(
                round(now - self._last_progress_ts, 1)
                if self._last_progress_ts is not None
                else None
            ),
            last_failure_age_s=(
                round(now - self._last_failure_ts, 1)
                if self._last_failure_ts is not None
                else None
            ),
            window_s=self._window(),
            since_s=round(now - self._state_since, 1),
        )

    def _window(self) -> float:
        # Nothing has ever come out of this engine: allow for a cold compile.
        if self._last_progress_ts is None:
            return self.config.cold_window_s
        return self.config.window_s

    def _judge(self) -> PDHealthState:
        now = self._clock()
        window = self._window()
        stalled = (
            self._demand_since is not None
            and now - self._demand_since >= window
            and (
                self._last_progress_ts is None or now - self._last_progress_ts >= window
            )
        )
        state = PDHealthState.UNHEALTHY if stalled else PDHealthState.READY
        if state != self.state:
            self.state = state
            self._state_since = now
            self.version += 1
        return self.state

    def _reason(self, now: float) -> str:
        if self.state == PDHealthState.UNHEALTHY:
            produced = (
                f"nothing produced for {now - self._last_progress_ts:.0f}s"
                if self._last_progress_ts is not None
                else "nothing produced since start"
            )
            failed = (
                f", last failed {now - self._last_failure_ts:.0f}s ago"
                if self._last_failure_ts is not None
                else ""
            )
            return (
                f"stalled: {self._requests_present} requests held, demand for "
                f"{now - self._demand_since:.0f}s, {produced}{failed} "
                f"(window {self._window():.0f}s)"
            )
        return "ready"


def build_pd_health_tracker(role: str) -> Optional[PDProgressHealth]:
    """None when disabled."""
    if not tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.get():
        return None
    return PDProgressHealth(config=PDHealthConfig.from_env(), role=role)


# ----- the slot: one file per scheduler rank, read by the HTTP process -----

SLOT_ROOT = "/dev/shm"


class PDHealthSlot(msgspec.Struct, kw_only=True):
    """One scheduler rank's verdict and the facts to judge a frozen rank."""

    dp_rank: int = 0
    tp_rank: int = 0
    pid: int = 0
    written_at: float = 0.0  # time.time()
    state: str = PDHealthState.READY.value
    record: str = ""  # PDHealthSnapshot JSON
    requests_present: int = 0
    stale_after_s: float = 0.0


def slot_dir(port: int) -> str:
    return os.path.join(SLOT_ROOT, f"sglang_tpu_pd_health_{port}")


def slot_path(port: int, *, dp_rank: int, tp_rank: int) -> str:
    return os.path.join(slot_dir(port), f"dp{dp_rank}_tp{tp_rank}.json")


def write_slot(path: str, slot: PDHealthSlot) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(msgspec.json.encode(slot))
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_slots(port: int) -> list[PDHealthSlot]:
    """Slots of live scheduler ranks; a dead writer's file is a leftover."""
    slots = []
    for path in sorted(glob.glob(os.path.join(slot_dir(port), "dp*_tp*.json"))):
        try:
            with open(path, "rb") as f:
                slot = msgspec.json.decode(f.read(), type=PDHealthSlot)
        except (OSError, msgspec.DecodeError):
            continue
        if _pid_alive(slot.pid):
            slots.append(slot)
    return slots
