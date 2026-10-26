"""Deferred release of finished requests' multimodal input features.

Multimodal features received over shared memory are private copy-on-write
mappings of an already unlinked segment (``ShmPointerMMData.materialize``).
Dropping the last reference unmaps the segment and returns its pages to the
kernel. For a video request (tens of clips, hundreds of MB) this takes tens of
milliseconds on the scheduler thread and delays the next launch.

``SGLANG_ASYNC_MM_FEATURE_RELEASE=incremental`` parks the features of finished
requests instead and drops them a few at a time right before the scheduler
blocks on a device result, while the device is still busy. A drain is skipped
when the result is already ready, the backlog is emptied when the scheduler is
idle, and at most ``_MAX_PENDING`` features stay parked. Only the moment the
host drops its last reference changes; nothing reads the features of a
finished request.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import TYPE_CHECKING, Any

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import MultimodalInputs

logger = logging.getLogger(__name__)

# Host time spent dropping parked features before one device wait.
_DRAIN_BUDGET_NS = 4_000_000
# Beyond this many parked features the oldest are dropped at submission, as
# without deferral, so the backlog stays bounded on paths that seldom wait on
# a device result or go idle.
_MAX_PENDING = 256


class MMFeatureReleaser:
    def __init__(self, mode: str) -> None:
        mode = mode.strip().lower()
        if mode not in ("", "0", "incremental"):
            raise ValueError(
                f"Unsupported SGLANG_ASYNC_MM_FEATURE_RELEASE={mode!r}; use "
                "'incremental', or leave it unset (or '0') to disable."
            )
        self.enabled = mode == "incremental"
        self._pending: collections.deque = collections.deque()

    def release(self, mm_inputs: MultimodalInputs) -> None:
        """Release a finished request's features like ``release_features``.

        When enabled, the feature objects are parked instead of dropped here.
        """
        if not self.enabled:
            mm_inputs.release_features()
            return
        pending = self._pending
        for item in mm_inputs.mm_items:
            try:
                item.release_transport_proxies()
            except Exception:
                logger.warning(
                    "Failed to release an unused multimodal feature transport",
                    exc_info=True,
                )
            finally:
                if item.feature is not None:
                    pending.append(item.feature)
                item.feature = None
        while len(pending) > _MAX_PENDING:
            pending.popleft()

    def drain_before_wait(self, event: Any) -> None:
        """Drop parked features for up to the budget unless ``event`` is done."""
        pending = self._pending
        if not pending or event.query():
            return
        deadline = time.perf_counter_ns() + _DRAIN_BUDGET_NS
        while pending and time.perf_counter_ns() < deadline:
            pending.popleft()

    def drain_all(self) -> None:
        self._pending.clear()


MM_FEATURE_RELEASER = MMFeatureReleaser(envs.SGLANG_ASYNC_MM_FEATURE_RELEASE.get())
