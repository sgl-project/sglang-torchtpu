"""Shared health-probe identity and PD decode admission policy.

The HTTP/tokenizer layers own probe task lifetime and client validation. The
scheduler monitor owns completed-inference acknowledgements. These helpers
keep decode admission and retirement independent of rank-local probe expiry.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from sglang.srt.constants import is_health_check_request

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


class ProbeAction(Enum):
    ADMIT = auto()
    DEFER = auto()
    RETIRE = auto()


def stamp_probe_admission(req: Req, completion_seq: int) -> None:
    # Requeueing must preserve the sequence that first admitted the probe.
    if is_health_check_request(req.rid) and req.health_check_admission_seq is None:
        req.health_check_admission_seq = completion_seq


def probe_action(req: Req, completion_seq: int, *, has_traffic: bool) -> ProbeAction:
    assert req.health_check_admission_seq is not None
    if completion_seq > req.health_check_admission_seq:
        return ProbeAction.RETIRE
    return ProbeAction.DEFER if has_traffic else ProbeAction.ADMIT


def finished_probe_count(reqs: list[Req]) -> int | None:
    """Return None while a probe still runs, otherwise its finished row count."""
    count = 0
    for req in reqs:
        if is_health_check_request(req.rid):
            if not req.finished():
                return None
            count += 1
    return count


def retire_decode_probe(scheduler: Scheduler, req: Req) -> None:
    # Useful work has already acknowledged the probe. The normal abort path
    # settles ownership, host/HiSparse state, KV and final-output accounting.
    scheduler.request_retirer.abort(req, checkpoint=False)
