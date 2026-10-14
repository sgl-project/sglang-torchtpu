"""P/D /health_generate from whether in-flight requests produce anything (TPU plugin).

One ``PDProgressHealth`` per tokenizer manager, fed from the HTTP process
(``health_generate_http``): real requests in flight (``observe``), the last
output (``record_progress``) and the last engine-side abort (``record_failure``).

UNHEALTHY once requests have been outstanding continuously for ``window_s``
and nothing was produced in that time; a request is outstanding while in
flight, and an engine failure keeps requests counted as outstanding for one
window. Idle is always ready, the first request after idleness gets
the whole window, ``cold_window_s`` applies until the first output after start,
client cancels count for nothing, and recovery is stateless. The clock is
injected for tests.
"""

from __future__ import annotations

import enum
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
    # Per-peer judgement (decode only); ratio 0 turns it off.
    peer_failure_ratio: float = 0.5
    peer_failure_streak: int = 3
    peer_min: int = 1

    @classmethod
    def from_env(cls) -> PDHealthConfig:
        return cls(
            window_s=tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S.get(),
            cold_window_s=tpu_envs.SGLANG_TPU_PD_HEALTH_COLD_WINDOW_S.get(),
            peer_failure_ratio=tpu_envs.SGLANG_TPU_PD_HEALTH_PEER_FAILURE_RATIO.get(),
            peer_failure_streak=tpu_envs.SGLANG_TPU_PD_HEALTH_PEER_FAILURE_STREAK.get(),
            peer_min=tpu_envs.SGLANG_TPU_PD_HEALTH_PEER_MIN.get(),
        )

    def validate(self) -> None:
        if self.window_s <= 0:
            raise ValueError("SGLANG_TPU_PD_HEALTH_WINDOW_S must be > 0")
        if self.cold_window_s < self.window_s:
            raise ValueError(
                "SGLANG_TPU_PD_HEALTH_COLD_WINDOW_S must be >= SGLANG_TPU_PD_HEALTH_WINDOW_S"
            )
        if not 0 <= self.peer_failure_ratio <= 1:
            raise ValueError(
                "SGLANG_TPU_PD_HEALTH_PEER_FAILURE_RATIO must be in [0, 1]"
            )
        if self.peer_failure_streak < 1 or self.peer_min < 1:
            raise ValueError(
                "SGLANG_TPU_PD_HEALTH_PEER_FAILURE_STREAK and _PEER_MIN must be >= 1"
            )


class PDHealthSnapshot(msgspec.Struct, kw_only=True):
    """Returned in the probe's 503 body."""

    role: str = "decode"
    state: str = PDHealthState.READY.value
    reason: str = "ready"
    requests_present: int = 0
    requests_age_s: Optional[float] = None
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
        self._requests_since: Optional[float] = None
        self._last_progress_ts: Optional[float] = None
        self._last_failure_ts: Optional[float] = None
        self._state_since = clock()

    def record_progress(self) -> None:
        """A real request produced something."""
        self._last_progress_ts = self._clock()
        self._judge()

    def record_failure(self) -> None:
        """The engine concluded a real request without output."""
        now = self._clock()
        self._last_failure_ts = now
        if self._requests_since is None:
            self._requests_since = now
        self._judge()

    def observe(self, requests_present: int) -> PDHealthState:
        """Real requests in flight right now."""
        self._requests_present = requests_present
        return self._refresh()

    def _refresh(self) -> PDHealthState:
        # Time alone changes the verdict: a failure ages out with no new event.
        now = self._clock()
        recent_failure = (
            self._last_failure_ts is not None
            and now - self._last_failure_ts < self._window()
        )
        if self._requests_present <= 0 and not recent_failure:
            self._requests_since = None
        elif self._requests_since is None:
            self._requests_since = now
        return self._judge()

    # ----- what the route reads -----

    def health_state(self) -> str:
        return self._refresh().value

    def health_json(self) -> str:
        return self.snapshot().to_json()

    def snapshot(self) -> PDHealthSnapshot:
        self._refresh()
        now = self._clock()
        return PDHealthSnapshot(
            role=self.role,
            state=self.state.value,
            reason=self._reason(now),
            requests_present=self._requests_present,
            requests_age_s=(
                round(now - self._requests_since, 1)
                if self._requests_since is not None
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
        # No output yet: allow for a cold compile.
        if self._last_progress_ts is None:
            return self.config.cold_window_s
        return self.config.window_s

    def _judge(self) -> PDHealthState:
        now = self._clock()
        window = self._window()
        stalled = (
            self._requests_since is not None
            and now - self._requests_since >= window
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
                f"stalled: {self._requests_present} requests held, outstanding for "
                f"{now - self._requests_since:.0f}s, {produced}{failed} "
                f"(window {self._window():.0f}s)"
            )
        return "ready"


def build_pd_health_tracker(role: str) -> Optional[PDProgressHealth]:
    """None when disabled."""
    if not tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.get():
        return None
    return PDProgressHealth(config=PDHealthConfig.from_env(), role=role)


# ----- per-peer judgement: which prefills a decode can still transfer from -----


class PDPeerSnapshot(msgspec.Struct, kw_only=True):
    peer: str
    state: str = "ok"  # "ok" or "failing"
    in_flight: int = 0
    failure_streak: int = 0
    last_failure_age_s: Optional[float] = None
    last_progress_age_s: Optional[float] = None


class PDPeersSnapshot(msgspec.Struct, kw_only=True):
    """Returned next to the engine record in the probe's 503 body."""

    state: str = PDHealthState.READY.value
    reason: str = "ready"
    active_peers: int = 0
    failing_peers: int = 0
    failure_ratio: float = 0.5
    failure_streak: int = 3
    peers: list[PDPeerSnapshot] = []

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode()


class _Peer:
    __slots__ = (
        "in_flight",
        "in_flight_since",
        "last_seen_ts",
        "last_failure_ts",
        "last_progress_ts",
        "failure_streak",
    )

    def __init__(self, now: float):
        self.in_flight = 0
        self.in_flight_since: Optional[float] = None
        self.last_seen_ts = now
        self.last_failure_ts: Optional[float] = None
        self.last_progress_ts: Optional[float] = None
        self.failure_streak = 0


class PDPeerHealth:
    """One record per peer the engine has exchanged KV with.

    A peer is failing after ``peer_failure_streak`` consecutive engine aborts
    of its requests with no success in between, the last one inside the
    window, or when its requests have been in flight for a window with nothing
    produced. The engine is unhealthy once failing peers reach
    ``peer_failure_ratio`` of the active peers (seen inside the window, or with
    requests in flight), with at least ``peer_min`` active. No active peers
    means nothing to judge: ready.
    """

    def __init__(
        self,
        config: Optional[PDHealthConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config or PDHealthConfig()
        self.config.validate()
        self._clock = clock
        self._peers: dict[str, _Peer] = {}

    def _peer(self, peer: str) -> _Peer:
        now = self._clock()
        record = self._peers.get(peer)
        if record is None:
            record = self._peers[peer] = _Peer(now)
        record.last_seen_ts = now
        return record

    def observe(self, in_flight: dict[str, int]) -> None:
        """Real requests in flight per peer right now."""
        now = self._clock()
        for peer in set(in_flight) | set(self._peers):
            record = self._peer(peer) if peer in in_flight else self._peers[peer]
            count = in_flight.get(peer, 0)
            if count > 0 and record.in_flight == 0:
                record.in_flight_since = now
            elif count == 0:
                record.in_flight_since = None
            record.in_flight = count

    def record_progress(self, peer: str) -> None:
        record = self._peer(peer)
        record.last_progress_ts = self._clock()
        record.failure_streak = 0

    def record_failure(self, peer: str) -> None:
        record = self._peer(peer)
        record.last_failure_ts = self._clock()
        record.failure_streak += 1

    def _active(self, record: _Peer, now: float) -> bool:
        return record.in_flight > 0 or now - record.last_seen_ts < self.config.window_s

    def _failing(self, record: _Peer, now: float) -> bool:
        aborting = (
            record.failure_streak >= self.config.peer_failure_streak
            and record.last_failure_ts is not None
            and now - record.last_failure_ts < self.config.window_s
        )
        # Requests to this peer sit in flight for a window with nothing back:
        # the same stall the engine rule sees, attributed to the peer.
        stalled = (
            record.in_flight_since is not None
            and now - record.in_flight_since >= self.config.window_s
            and (
                record.last_progress_ts is None
                or now - record.last_progress_ts >= self.config.window_s
            )
        )
        return aborting or stalled

    def _judge(self, now: float) -> tuple[int, int, bool]:
        active = [r for r in self._peers.values() if self._active(r, now)]
        failing = [r for r in active if self._failing(r, now)]
        unhealthy = (
            self.config.peer_failure_ratio > 0
            and len(active) >= self.config.peer_min
            and len(failing) / len(active) >= self.config.peer_failure_ratio
        )
        return len(active), len(failing), unhealthy

    def health_state(self) -> str:
        _, _, unhealthy = self._judge(self._clock())
        return (PDHealthState.UNHEALTHY if unhealthy else PDHealthState.READY).value

    def health_json(self) -> str:
        return self.snapshot().to_json()

    def snapshot(self) -> PDPeersSnapshot:
        now = self._clock()
        active_n, failing_n, unhealthy = self._judge(now)
        peers = []
        for name, record in sorted(self._peers.items()):
            if not self._active(record, now):
                continue
            peers.append(
                PDPeerSnapshot(
                    peer=name,
                    state="failing" if self._failing(record, now) else "ok",
                    in_flight=record.in_flight,
                    failure_streak=record.failure_streak,
                    last_failure_age_s=(
                        round(now - record.last_failure_ts, 1)
                        if record.last_failure_ts is not None
                        else None
                    ),
                    last_progress_age_s=(
                        round(now - record.last_progress_ts, 1)
                        if record.last_progress_ts is not None
                        else None
                    ),
                )
            )
        if unhealthy:
            names = ", ".join(p.peer for p in peers if p.state == "failing")
            reason = (
                f"peers failing: {failing_n} of {active_n} active "
                f"(threshold {self.config.peer_failure_ratio:.0%}, "
                f"{self.config.peer_failure_streak} aborts in a row or "
                f"{self.config.window_s:.0f}s in flight with nothing back): {names}"
            )
        else:
            reason = "ready"
        return PDPeersSnapshot(
            state=(PDHealthState.UNHEALTHY if unhealthy else PDHealthState.READY).value,
            reason=reason,
            active_peers=active_n,
            failing_peers=failing_n,
            failure_ratio=self.config.peer_failure_ratio,
            failure_streak=self.config.peer_failure_streak,
            peers=peers,
        )


def build_pd_peer_tracker(role: str) -> Optional[PDPeerHealth]:
    """Decode only: a prefill's requests do not name the decode they pair with."""
    if role != "decode" or not tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.get():
        return None
    config = PDHealthConfig.from_env()
    if config.peer_failure_ratio <= 0:
        return None
    return PDPeerHealth(config=config)
