"""Readiness of a P/D decode from what its KV transfers actually did.

The health probe runs through the fake receiver and never touches a peer, so
it cannot see a transfer failing. Real requests can. The scheduler feeds their
transfer-layer outcomes to ``PDTransferHealth``, which publishes its state in
the load snapshot; ``/health_generate`` reads it before anything else.

The rule (sglang-torchtpu #1276, agreed short-term form):

* Per peer connection, OPEN once ``failure_threshold`` transfers in a row have
  failed over at least ``failure_window_s`` with zero successes in that run.
  Both must hold, so a burst shorter than the window never opens a peer
  however large it is. One success closes the peer.
* The engine is UNHEALTHY (503) once more than ``failure_ratio`` of its known
  peers are open. One dead prefill out of two is below the default 0.5 and
  leaves the decode READY, reporting the peer as degraded.
* Recovery: a decode taken out of service gets no requests, so after
  ``failure_window_s`` with no new failure (at least
  ``probation_output_timeout_s``) it passes on PROBATION (200) and traffic
  resumes. A probation request must complete its transfer within
  ``probation_output_timeout_s`` or the decode fails again, with the next
  quiet wait doubled up to ``probation_backoff_cap_s``. Probation ends in
  READY as soon as the open-peer ratio no longer trips.

Peers are the prefill bootstrap addresses named by real requests. Health
probes (the fake receiver), client aborts and intake rejections are never
recorded; the caller filters them. The clock is injected so every rule is
testable without sleeping.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Callable, Dict, List, Optional

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class PDHealthState(str, enum.Enum):
    READY = "ready"
    UNHEALTHY = "unhealthy"
    PROBATION = "probation"


class PDHealthConfig(msgspec.Struct, frozen=True, kw_only=True):
    failure_threshold: int = 5
    failure_window_s: float = 60.0
    failure_ratio: float = 0.5
    probation_output_timeout_s: float = 60.0
    probation_backoff_cap_s: float = 1800.0
    expected_peers: int = 0

    @classmethod
    def from_env(cls) -> PDHealthConfig:
        return cls(
            failure_threshold=envs.SGLANG_PD_HEALTH_FAILURE_THRESHOLD.get(),
            failure_window_s=envs.SGLANG_PD_HEALTH_FAILURE_WINDOW_S.get(),
            failure_ratio=envs.SGLANG_PD_HEALTH_FAILURE_RATIO.get(),
            probation_output_timeout_s=envs.SGLANG_PD_HEALTH_PROBATION_OUTPUT_TIMEOUT_S.get(),
            probation_backoff_cap_s=envs.SGLANG_PD_HEALTH_PROBATION_BACKOFF_CAP_S.get(),
            expected_peers=envs.SGLANG_PD_HEALTH_EXPECTED_PEERS.get(),
        )

    def validate(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("SGLANG_PD_HEALTH_FAILURE_THRESHOLD must be >= 1")
        if self.failure_window_s < 0 or self.probation_output_timeout_s <= 0:
            raise ValueError("PD health windows must be non-negative, timeout > 0")
        if not 0.0 <= self.failure_ratio < 1.0:
            raise ValueError("SGLANG_PD_HEALTH_FAILURE_RATIO must be in [0, 1)")
        if self.probation_backoff_cap_s < self.failure_window_s:
            raise ValueError("probation backoff cap must be >= the failure window")
        if self.expected_peers < 0:
            raise ValueError("SGLANG_PD_HEALTH_EXPECTED_PEERS must be >= 0")
        if self.failure_threshold == 1 and self.failure_window_s > 0:
            # A single failure spans 0 s, so with a positive window the rule
            # "N in a row over at least T" can never hold.
            logger.warning(
                "SGLANG_PD_HEALTH_FAILURE_THRESHOLD=1 with a positive window never "
                "opens a peer; set the window to 0 to open on the first failure"
            )


class PDPeerSnapshot(msgspec.Struct, kw_only=True):
    peer: str
    open: bool = False
    consecutive_failures: int = 0
    failures_span_s: float = 0.0
    last_failure_age_s: Optional[float] = None
    last_success_age_s: Optional[float] = None


class PDHealthSnapshot(msgspec.Struct, kw_only=True):
    """Published by the scheduler; read by the probe and returned in its 503 body."""

    state: str = PDHealthState.READY.value
    reason: str = "ready"
    known_peers: int = 0
    open_peers: int = 0
    failure_ratio: float = 0.0
    failure_ratio_threshold: float = 0.5
    peers: List[PDPeerSnapshot] = []
    probation_wait_s: float = 0.0
    probation_deadline_in_s: Optional[float] = None
    since_s: float = 0.0

    @property
    def ready(self) -> bool:
        return self.state != PDHealthState.UNHEALTHY.value

    @property
    def open_peer_names(self) -> List[str]:
        return [p.peer for p in self.peers if p.open]

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode()

    @classmethod
    def from_json(cls, text: str) -> PDHealthSnapshot:
        return msgspec.json.decode(text, type=cls)


class _PeerCircuit:
    """One prefill as seen by this decode: the current run of failures."""

    def __init__(self, now: float):
        self.streak = 0
        self.first_failure_ts: Optional[float] = None
        self.last_failure_ts: Optional[float] = None
        self.last_success_ts: Optional[float] = None
        self.open = False
        self.seen_ts = now

    def fail(self, now: float, cfg: PDHealthConfig) -> None:
        self.streak += 1
        if self.first_failure_ts is None:
            self.first_failure_ts = now
        self.last_failure_ts = now
        self.seen_ts = now
        span = now - self.first_failure_ts
        if self.streak >= cfg.failure_threshold and span >= cfg.failure_window_s:
            self.open = True

    def succeed(self, now: float) -> None:
        self.streak = 0
        self.first_failure_ts = None
        self.open = False
        self.last_success_ts = now
        self.seen_ts = now

    def snapshot(self, peer: str, now: float) -> PDPeerSnapshot:
        span = (
            self.last_failure_ts - self.first_failure_ts
            if self.first_failure_ts is not None and self.last_failure_ts is not None
            else 0.0
        )
        return PDPeerSnapshot(
            peer=peer,
            open=self.open,
            consecutive_failures=self.streak,
            failures_span_s=round(span, 1),
            last_failure_age_s=(
                round(now - self.last_failure_ts, 1)
                if self.last_failure_ts is not None
                else None
            ),
            last_success_age_s=(
                round(now - self.last_success_ts, 1)
                if self.last_success_ts is not None
                else None
            ),
        )


class PDTransferHealth:
    def __init__(
        self,
        config: Optional[PDHealthConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config or PDHealthConfig()
        self.config.validate()
        self._clock = clock
        self.state = PDHealthState.READY
        self.version = 0
        self._peers: Dict[str, _PeerCircuit] = {}
        self._last_failure_ts: Optional[float] = None
        self._state_since: float = clock()
        self._probation_wait_s: float = self._base_wait()
        self._probation_deadline: Optional[float] = None

    def _base_wait(self) -> float:
        # Stay out at least as long as a probation request gets to produce
        # output; with a zero window the probe would otherwise never see a 503.
        return max(self.config.failure_window_s, self.config.probation_output_timeout_s)

    # ----- inputs from the decode queues -----

    def record_failure(self, *, peer: str) -> PDHealthState:
        now = self._clock()
        self._peer(peer, now).fail(now, self.config)
        self._last_failure_ts = now
        if self.state == PDHealthState.PROBATION:
            self._open(now, backoff=True)
        elif self.state == PDHealthState.READY and self._ratio_trips():
            self._open(now, backoff=False)
        return self.state

    def record_success(self, *, peer: str) -> PDHealthState:
        now = self._clock()
        self._peer(peer, now).succeed(now)
        if self.state != PDHealthState.READY and not self._ratio_trips():
            self._set_state(PDHealthState.READY, now)
            self._probation_wait_s = self._base_wait()
            self._probation_deadline = None
        elif self.state == PDHealthState.PROBATION:
            # One peer proved itself; the next admitted request re-arms the
            # deadline for the peers still open.
            self._probation_deadline = None
        return self.state

    def record_admitted(self) -> None:
        if self.state == PDHealthState.PROBATION and self._probation_deadline is None:
            self._probation_deadline = (
                self._clock() + self.config.probation_output_timeout_s
            )

    # ----- clock-driven transitions -----

    def tick(self) -> PDHealthState:
        now = self._clock()
        if self.state == PDHealthState.UNHEALTHY:
            assert self._last_failure_ts is not None
            if now - self._last_failure_ts >= self._probation_wait_s:
                self._set_state(PDHealthState.PROBATION, now)
                self._probation_deadline = None
        elif self.state == PDHealthState.PROBATION:
            if self._probation_deadline is not None and now > self._probation_deadline:
                self._open(now, backoff=True)
        return self.state

    def snapshot(self) -> PDHealthSnapshot:
        now = self._clock()
        self.tick()
        open_count, denom = self._open_and_denominator()
        peers = [c.snapshot(name, now) for name, c in sorted(self._peers.items())]
        return PDHealthSnapshot(
            state=self.state.value,
            reason=self._reason(open_count, denom),
            known_peers=denom,
            open_peers=open_count,
            failure_ratio=round(open_count / denom, 3) if denom else 0.0,
            failure_ratio_threshold=self.config.failure_ratio,
            peers=peers,
            probation_wait_s=self._probation_wait_s,
            probation_deadline_in_s=(
                round(self._probation_deadline - now, 1)
                if self._probation_deadline is not None
                else None
            ),
            since_s=round(now - self._state_since, 1),
        )

    # ----- internals -----

    def _peer(self, name: str, now: float) -> _PeerCircuit:
        circuit = self._peers.get(name)
        if circuit is None:
            circuit = self._peers[name] = _PeerCircuit(now)
        return circuit

    def _open_and_denominator(self) -> tuple[int, int]:
        open_count = sum(1 for c in self._peers.values() if c.open)
        return open_count, max(len(self._peers), self.config.expected_peers)

    def _ratio_trips(self) -> bool:
        open_count, denom = self._open_and_denominator()
        if denom == 0 or open_count == 0:
            return False
        return open_count / denom > self.config.failure_ratio

    def _open(self, now: float, *, backoff: bool) -> None:
        if backoff:
            self._probation_wait_s = min(
                self._probation_wait_s * 2, self.config.probation_backoff_cap_s
            )
        # A missed probation deadline is the event the quiet wait counts from;
        # without this the stale last-failure time re-admits the decode at once
        # and the request that then fails doubles the wait a second time.
        self._last_failure_ts = now
        self._probation_deadline = None
        self._set_state(PDHealthState.UNHEALTHY, now)

    def _set_state(self, state: PDHealthState, now: float) -> None:
        if state != self.state:
            self.state = state
            self._state_since = now
            self.version += 1

    def _reason(self, open_count: int, denom: int) -> str:
        names = [n for n, c in sorted(self._peers.items()) if c.open]
        threshold = self.config.failure_ratio
        if self.state == PDHealthState.UNHEALTHY:
            return (
                f"peer_failure_ratio: {open_count}/{denom} peers open (> {threshold:.2f}) "
                f"{names}; probation {self._probation_wait_s:.0f}s after the last failure"
            )
        if self.state == PDHealthState.PROBATION:
            return (
                f"probation: {open_count}/{denom} peers still open {names}; the first "
                f"admitted transfer must complete within "
                f"{self.config.probation_output_timeout_s:.0f}s"
            )
        if open_count:
            return f"degraded: {open_count}/{denom} peers open {names}"
        return "ready"
