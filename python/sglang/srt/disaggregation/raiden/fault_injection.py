"""Test-only fault and latency injection for the raiden KV transfer.

Off unless ``SGLANG_TEST_RAIDEN_FAULT_INJECTION`` names a preset of
``fault_injection.json`` on a prefill or decode server. That file holds the
presets, each a share of requests that meets a failure and a share that meets
a delay, and the tpu-sync hooks the faults are spread over.

The faults come from tpu-sync's own injector, inside the transfer code of
every scheduler process: a failure makes one step of a transfer return an
error, a delay makes it wait. A request that meets a failure gets an error
reply, as it would after a real transfer failure. A delay is placed only where
it stays inside raiden's time limit for that step, so a delay preset fails a
request only when the delay carries the whole transfer past
``SGLANG_RAIDEN_TIMEOUT_S``. A delay holds the raiden thread that runs the
step, so transfers queued behind it on that rank wait as well.

tpu-sync takes a probability per hook call, and a hook runs once per read or
once per layer of a read. A request is several reads, more for a long prompt.
So each process counts the reads and layers of the requests it serves and
turns the preset's share of requests into per-call probabilities, again as
the traffic changes. The first requests of a process run without faults, and
a long prompt is more likely to meet a fault than a short one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import msgspec

from sglang.srt.disaggregation.raiden.common import logger
from sglang.srt.environ import envs

CONFIG_PATH = Path(__file__).with_suffix(".json")

# Arbitrary: enough requests for a stable average before the first fault.
_WARMUP_REQUESTS = 20
# Arbitrary: how often the probabilities follow a change in request size.
_REFRESH_REQUESTS = 100
# Rooms remembered to tell a request's first read from its later ones.
_MAX_TRACKED_ROOMS = 4096


class FaultPreset(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    description: str
    # Expected injections per request, over both servers and all their ranks.
    fails_per_request: float
    delays_per_request: float
    # A delay lasts 0 to this long, drawn uniformly.
    max_delay_ms: int


class FaultSite(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    # A hook of tpu-sync's tpu_sync/fault_injection/hooks.h that runs with
    # either control plane backend, and an action it accepts.
    hook: str
    action: Literal["fail", "delay"]
    # The server whose scheduler processes run the hook.
    role: Literal["prefill", "decode"]
    # True: the hook runs once per layer of a read, False: once per read.
    per_layer: bool = False
    # raiden's limit on the step a delay here holds up, 10 s for the reply to
    # a pull request, None for none. A preset whose delay can reach it skips
    # the site.
    limit_ms: Optional[int] = None


class FaultConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    presets: Dict[str, FaultPreset]
    sites: Tuple[FaultSite, ...]


def load_config() -> FaultConfig:
    return msgspec.json.decode(CONFIG_PATH.read_bytes(), type=FaultConfig)


def _sites(
    *,
    sites: Tuple[FaultSite, ...],
    preset: FaultPreset,
    action: str,
    role: Optional[str] = None,
) -> List[FaultSite]:
    """Sites the preset uses for ``action``, of one role or of both."""
    return [
        site
        for site in sites
        if site.action == action
        and (role is None or site.role == role)
        and (
            action == "fail"
            or site.limit_ms is None
            or preset.max_delay_ms < site.limit_ms
        )
    ]


def _per_site(
    *, sites: Tuple[FaultSite, ...], preset: FaultPreset, action: str, ranks: int
) -> float:
    """Expected injections per request at one site of one process.

    A budget is split evenly over the sites of both roles and over the
    ``ranks`` processes of a role, so the injections of all processes add up
    to the preset's count per request.
    """
    per_request = (
        preset.fails_per_request if action == "fail" else preset.delays_per_request
    )
    count = len(_sites(sites=sites, preset=preset, action=action))
    return per_request / (count * ranks) if count else 0.0


def expected_per_request(
    *, sites: Tuple[FaultSite, ...], preset: FaultPreset, role: str, ranks: int
) -> float:
    """Expected injections per request in one process of ``role``."""
    return sum(
        _per_site(sites=sites, preset=preset, action=action, ranks=ranks)
        * len(_sites(sites=sites, preset=preset, action=action, role=role))
        for action in ("fail", "delay")
    )


def build_rules(
    *,
    sites: Tuple[FaultSite, ...],
    preset: FaultPreset,
    role: str,
    ranks: int,
    reads_per_request: float,
    layers_per_request: float,
) -> List[Dict[str, Any]]:
    """tpu-sync rules for one process of ``role``.

    ``reads_per_request`` and ``layers_per_request`` are what one process
    sees of one request.
    """
    rules: List[Dict[str, Any]] = []
    for action in ("fail", "delay"):
        per_site = _per_site(sites=sites, preset=preset, action=action, ranks=ranks)
        if per_site <= 0:
            continue
        for site in _sites(sites=sites, preset=preset, action=action, role=role):
            calls = layers_per_request if site.per_layer else reads_per_request
            rule: Dict[str, Any] = {
                "hook": site.hook,
                "action": action,
                "probability": min(1.0, per_site / calls),
            }
            if action == "delay":
                rule["min_delay_ms"] = 0
                rule["max_delay_ms"] = preset.max_delay_ms
            rules.append(rule)
    return rules


class RaidenFaultInjector:
    """Keeps one process's tpu-sync injector armed for a preset."""

    def __init__(
        self,
        *,
        native: Any,
        name: str,
        preset: FaultPreset,
        sites: Tuple[FaultSite, ...],
        role: str,
        ranks: int,
    ):
        self._native = native
        self._name = name
        self._preset = preset
        self._sites = sites
        self._role = role
        self._ranks = ranks
        self._expected_per_request = expected_per_request(
            sites=sites, preset=preset, role=role, ranks=ranks
        )
        self._lock = threading.Lock()
        self._rooms: OrderedDict[int, None] = OrderedDict()
        self._requests = 0
        self._reads = 0
        self._layers = 0
        self._arm_at = _WARMUP_REQUESTS
        # Request count at the first install, None until then.
        self._armed_from: Optional[int] = None

    def note_read(self, *, room: int, layers: int) -> None:
        """Count one read of ``room`` over ``layers`` layers, and re-arm when due."""
        with self._lock:
            self._reads += 1
            self._layers += layers
            if room not in self._rooms:
                self._rooms[room] = None
                if len(self._rooms) > _MAX_TRACKED_ROOMS:
                    self._rooms.popitem(last=False)
                self._requests += 1
            if self._requests < self._arm_at:
                return
            self._arm_at = self._requests + _REFRESH_REQUESTS
            self._arm_locked()

    def _arm_locked(self) -> None:
        reads_per_request = self._reads / self._requests
        layers_per_request = self._layers / self._requests
        rules = build_rules(
            sites=self._sites,
            preset=self._preset,
            role=self._role,
            ranks=self._ranks,
            reads_per_request=reads_per_request,
            layers_per_request=layers_per_request,
        )
        armed_requests = (
            0 if self._armed_from is None else self._requests - self._armed_from
        )
        try:
            # tpu-sync keeps its injection count across installs.
            injected = self._native.get_hit_count()
            self._native.inject_faults(rules)
        except Exception as exc:
            logger.error(
                "Raiden fault injection %s could not be armed: %s", self._name, exc
            )
            return
        logger.warning(
            "Raiden fault injection %s armed on %s: %d rules, %.1f reads and "
            "%.1f layer copies per request over %d requests, %d faults "
            "injected so far, %.1f expected",
            self._name,
            self._role,
            len(rules),
            reads_per_request,
            layers_per_request,
            self._requests,
            injected,
            armed_requests * self._expected_per_request,
        )
        if self._armed_from is None:
            self._armed_from = self._requests


_injector: Optional[RaidenFaultInjector] = None


def start(*, role: str, ranks: int) -> None:
    """Set up this process's injector when a preset is selected."""
    global _injector
    name = envs.SGLANG_TEST_RAIDEN_FAULT_INJECTION.get()
    if not name or _injector is not None:
        return
    config = load_config()
    if name not in config.presets:
        raise ValueError(
            f"SGLANG_TEST_RAIDEN_FAULT_INJECTION={name!r} is not a preset of "
            f"{CONFIG_PATH}, use one of {sorted(config.presets)}"
        )
    # TPU-only import: tpu-sync is absent on hosts without the TPU stack.
    from tpu_sync.api.torch import fault_injection as native

    _injector = RaidenFaultInjector(
        native=native,
        name=name,
        preset=config.presets[name],
        sites=config.sites,
        role=role,
        ranks=ranks,
    )
    logger.warning(
        "Raiden fault injection %s selected on %s (%s), faults start after %d requests",
        name,
        role,
        config.presets[name].description,
        _WARMUP_REQUESTS,
    )


def active() -> bool:
    return _injector is not None


def note_read(*, room: int, layers: int) -> None:
    if _injector is not None:
        _injector.note_read(room=room, layers=layers)
