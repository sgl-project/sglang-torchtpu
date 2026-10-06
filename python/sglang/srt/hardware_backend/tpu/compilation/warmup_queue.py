"""Join TPU warmups after every model runner has submitted its graph grid."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall

logger = get_logger(__name__)


@dataclass
class WarmupQueue:
    calls: list[PendingCall] = field(default_factory=list)
    # Run after every warmup finishes. TpuGraphRunner uses these callbacks to
    # snapshot and attach the target's recompilation checker (when enabled),
    # then log "TpuGraphRunner precompile done".
    completions: list[Callable[[], None]] = field(default_factory=list)

    def wait(self) -> None:
        with torch.no_grad():
            for call in self.calls:
                call.wait()
        for complete in self.completions:
            complete()


_CURRENT: ContextVar[WarmupQueue | None] = ContextVar("tpu_warmup_queue", default=None)


def current_warmup_queue() -> WarmupQueue | None:
    return _CURRENT.get()


@contextmanager
def collect_warmups() -> Iterator[WarmupQueue]:
    existing = _CURRENT.get()
    if existing is not None:
        yield existing
        return
    queue = WarmupQueue()
    token = _CURRENT.set(queue)
    start = time.perf_counter()
    try:
        yield queue
        submitted = time.perf_counter()
        logger.info("TPU global warmup: submitted %d calls; joining", len(queue.calls))
        queue.wait()
        finished = time.perf_counter()
        logger.info(
            "TPU global warmup: pending=%d submit=%.3fs wait=%.3fs total=%.3fs",
            len(queue.calls),
            submitted - start,
            finished - submitted,
            finished - start,
        )
    finally:
        _CURRENT.reset(token)
        queue.calls.clear()
        queue.completions.clear()
