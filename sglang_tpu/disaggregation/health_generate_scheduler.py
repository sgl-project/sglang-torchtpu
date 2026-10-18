# SPDX-License-Identifier: Apache-2.0
"""P/D /health_generate judged in each scheduler rank, published to a slot file.

``sglang_tpu.patches`` wraps four scheduler-side methods to call the functions
below. State lives in maps keyed by the scheduler and its output streamer.
"""

from __future__ import annotations

import os
import time
import weakref
from typing import Optional

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.runtime_context import get_parallel
from sglang_tpu._logging import get_logger
from sglang_tpu.disaggregation.health_generate import (
    PDHealthSlot,
    PDProgressHealth,
    build_pd_health_tracker,
    slot_path,
    write_slot,
)

logger = get_logger(__name__)

# Refresh cadence of the slot while requests are held, so a slot that stops
# refreshing with work in it is unambiguous. Arbitrary; well under any window.
PD_HEALTH_REPUBLISH_S = 5.0


class _SchedulerHealth:
    __slots__ = (
        "tracker",
        "path",
        "published_version",
        "published_held",
        "published_ts",
        "write_failed",
    )

    def __init__(self, tracker: PDProgressHealth, path: str):
        self.tracker = tracker
        self.path = path
        self.published_version = -1
        self.published_held = -1
        self.published_ts = 0.0
        self.write_failed = False


_BY_SCHEDULER: weakref.WeakKeyDictionary[object, _SchedulerHealth] = (
    weakref.WeakKeyDictionary()
)
# The streamer is a slots dataclass without a weakref slot; one per process.
_BY_STREAMER: dict[int, _SchedulerHealth] = {}


def state_of(scheduler) -> Optional[_SchedulerHealth]:
    return _BY_SCHEDULER.get(scheduler)


def attach_tracker(scheduler) -> None:
    """One tracker per P/D rank; a fresh ready slot replaces any leftover file."""
    mode = scheduler.disaggregation_mode
    if mode not in (DisaggregationMode.DECODE, DisaggregationMode.PREFILL):
        return
    tracker = build_pd_health_tracker(mode.value)
    if tracker is None:
        return
    parallel = get_parallel()
    path = slot_path(
        scheduler.server_args.port,
        dp_rank=parallel.dp_rank or 0,
        tp_rank=parallel.tp_rank,
    )
    state = _SchedulerHealth(tracker, path)
    _BY_SCHEDULER[scheduler] = state
    _BY_STREAMER[id(scheduler.output_streamer)] = state
    _publish(state)


def requests_present(scheduler) -> int:
    """Real requests held anywhere on this rank's P/D path; probes excluded."""
    reqs = list(scheduler.waiting_queue) + list(scheduler.running_batch.reqs)
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        reqs += list(scheduler.disagg_prefill_bootstrap_queue.queue)
        reqs += list(scheduler.disagg_prefill_inflight_queue)
    else:
        reqs += [d.req for d in scheduler.disagg_decode_prealloc_queue.queue]
        reqs += [d.req for d in scheduler.disagg_decode_transfer_queue.queue]
        reqs += list(scheduler.disagg_decode_prealloc_queue.retracted_queue)
    return sum(1 for req in reqs if req.bootstrap_host != FAKE_BOOTSTRAP_HOST)


def observe_and_publish(scheduler) -> None:
    """Every loop iteration, idle ones included: a stalled rank runs no batch."""
    state = _BY_SCHEDULER.get(scheduler)
    if state is None:
        return
    held = requests_present(scheduler)
    state.tracker.observe(held)
    # The HTTP process judges a frozen slot by its held count, so the count
    # in the slot must be current: republish when it changes, not only when
    # the verdict does.
    changed = (
        state.tracker.version != state.published_version or held != state.published_held
    )
    heartbeat_due = (
        held > 0 and time.monotonic() - state.published_ts >= PD_HEALTH_REPUBLISH_S
    )
    if changed or heartbeat_due:
        _publish(state)


def _publish(state: _SchedulerHealth) -> None:
    tracker = state.tracker
    state.published_version = tracker.version
    state.published_held = tracker.requests_present()
    state.published_ts = time.monotonic()
    parallel = get_parallel()
    slot = PDHealthSlot(
        dp_rank=parallel.dp_rank or 0,
        tp_rank=parallel.tp_rank,
        pid=os.getpid(),
        written_at=time.time(),
        state=tracker.health_state(),
        record=tracker.health_json(),
        requests_present=tracker.requests_present(),
        stale_after_s=tracker.stale_after_s(),
    )
    try:
        write_slot(state.path, slot)
    except OSError as e:
        if not state.write_failed:
            logger.warning("pd health slot write failed at %s: %s", state.path, e)
        state.write_failed = True


def track_streamed(streamer, reqs) -> None:
    """Output left the rank: progress, or an engine-side abort (5xx)."""
    state = _BY_STREAMER.get(id(streamer))
    if state is None:
        return
    decode = streamer.disaggregation_mode == DisaggregationMode.DECODE
    for req in reqs:
        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            continue
        reason = req.finished_reason
        if isinstance(reason, FINISH_ABORT):
            # A client cancel or a rejected request (no code, or 4xx) is neither.
            if reason.status_code is not None and reason.status_code >= 500:
                state.tracker.record_failure()
        elif reason is not None or decode:
            # A prefill's only output is its final one; a decode streams tokens.
            state.tracker.record_progress()
