"""Background KV-copy worker for the TPU HiCache offload data plane.

A single-direction FIFO worker thread -- **one instance per transfer
direction** (D2H write-back / H2D reload) -- that runs transfer closures off
the main scheduler thread. Modeled on the vLLM torch_tpu offload ``_DmaWorker``:
the FIFO queue serializes tasks, so at most one transfer per direction runs at a
time, and each ``submit`` returns a completion event set when the task returns.

Because a transfer task ends with a blocking device sync
(``_tpu_sync(..., wait=True)`` over the gathered / scattered buffers), a set
event implies the DMA has physically landed on-device -- which is exactly what
HiCache's ack queues need before a node is unlocked / made loadable.

Two worker threads must never share one instance across directions: a slow D2H
must not be able to head-of-line block an H2D (and vice versa), so each
direction gets its own ``TpuKvCopyWorker``.
"""

from __future__ import annotations

import logging
import threading
from queue import SimpleQueue
from typing import Callable

logger = logging.getLogger(__name__)


class _ThreadDoneEvent:
    """Completion event for an async KV transfer, adapted to the CUDA-event
    surface HiCache's ack queues use (``query`` / ``synchronize`` / ``record`` /
    ``wait``).

    The transfer runs on a worker thread and ends with a blocking device sync,
    so ``query`` / ``synchronize`` reflect real on-device completion -- a plain
    always-done sentinel here would let ``writing_check`` unlock a node before
    its host write landed (corruption). ``record`` / ``wait`` are no-ops:
    torch_tpu has no device streams, and the ack drain logic only ever calls
    ``query`` / ``synchronize`` on the finish event.
    """

    def __init__(self) -> None:
        self._e = threading.Event()

    def set(self) -> None:
        self._e.set()

    def query(self) -> bool:
        return self._e.is_set()

    def synchronize(self) -> None:
        self._e.wait()

    def record(self, *args, **kwargs) -> None:
        pass

    def wait(self, *args, **kwargs) -> None:
        self._e.wait()


class TpuKvCopyWorker:
    """Single background thread draining a FIFO work queue, one per direction.

    ``submit(fn, *args)`` enqueues work and returns a :class:`_ThreadDoneEvent`
    set when ``fn`` returns. The FIFO queue serializes tasks, so at most one
    transfer per direction is ever running; a caller that needs a tighter bound
    (e.g. depth-1 against a shared staging buffer) blocks on the previous
    event before submitting the next. Modeled on the vLLM torch_tpu
    ``_DmaWorker``.
    """

    def __init__(self, name: str) -> None:
        self._queue: SimpleQueue = SimpleQueue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable, *args) -> _ThreadDoneEvent:
        done = _ThreadDoneEvent()
        self._queue.put((fn, args, done))
        return done

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # shutdown sentinel
                return
            fn, args, done = item
            try:
                fn(*args)
            except Exception:
                logger.exception("[kv-offload] copy worker task failed")
            finally:
                done.set()

    def shutdown(self) -> None:
        self._queue.put(None)
