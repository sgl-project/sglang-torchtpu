"""Raiden-backed TPU<->host KV transfer substrate for HiCache.

torch_tpu has no real streams and no pinned host memory, so the CUDA HiCache
transfer path can't be ported. This module drives tpu-sync's
``KVCacheManager``: ``D2h`` / ``H2d`` do direct PJRT raw-buffer DMA between the
fused device KV buffer and raiden's DMA-pinned host pool, returning a pollable
``RaidenFuture``. A raiden "block" is one page of the fused buffer, and a single
``D2h`` / ``H2d`` call covers all registered layers; raiden issues the copies
on its own background thread, so the call returns before anything is submitted.
Completion is poll-only (``is_ready()``; never ``Await()``, which can deadlock
the running model), and a DMA that completes with an error raises at the poll.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, List

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

logger = logging.getLogger(__name__)


class RaidenFutureEvent:
    """Adapts a raiden ``RaidenFuture`` to the CUDA-event surface HiCache's
    ack queues use (``query`` / ``synchronize`` / ``record`` / ``wait``).

    Only ``query`` and ``synchronize`` carry meaning here; ``record`` / ``wait``
    are no-ops so the object can stand in for a ``device_module.Event`` in
    ``HiCacheAck`` without touching the hardware-agnostic drain logic in
    ``HiRadixCache.writing_check`` / ``loading_check``.
    """

    # Poll interval for the blocking flush path.
    _POLL_INTERVAL_S = 0.0005

    def __init__(self, future):
        self._future = future
        # A poll past this raises like a blocking wait: the copy's pages can be
        # neither reused nor released while it may still land.
        self._poll_deadline = (
            time.monotonic() + envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.get()
        )

    def _raise_if_failed(self) -> None:
        """Surface a ready DMA's error; ``ok()`` is a non-blocking probe."""
        if not self._future.ok():
            raise RuntimeError(f"raiden DMA failed: {self._future.error_message()}")

    def query(self) -> bool:
        """Non-blocking completion poll. ``True`` once the DMA has landed."""
        if self._future is None:
            return True
        if not self._future.is_ready():
            if time.monotonic() > self._poll_deadline:
                raise TimeoutError(
                    "raiden DMA still pending "
                    f"{envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.get()}s after issue"
                )
            return False
        self._raise_if_failed()
        return True

    def synchronize(self) -> None:
        """Block until complete by POLLING ``is_ready`` — never ``Await``.

        ``RaidenFuture.Await`` (``PJRT_Event_Await`` / ``xla::Future``
        ``BlockUntilReady``) can deadlock against the live model, so the flush
        primitive polls ``is_ready`` instead of awaiting. A DMA that never lands
        raises ``TimeoutError`` past the deadline rather than spinning forever.
        """
        if self._future is None:
            return
        timeout_s = envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.get()
        deadline = time.monotonic() + timeout_s
        while not self._future.is_ready():
            if time.monotonic() > deadline:
                raise TimeoutError(f"raiden DMA did not complete within {timeout_s}s")
            time.sleep(self._POLL_INTERVAL_S)
        self._raise_if_failed()

    def record(self, *args, **kwargs) -> None:  # noqa: D401 - event-API no-op
        return None

    def wait(self, *args, **kwargs) -> None:  # noqa: D401 - event-API no-op
        return None


# A no-op event used for the ``start_event`` slot of ``HiCacheAck`` (HiCache
# only ever queries/synchronizes the ``finish_event``).
NOOP_EVENT = RaidenFutureEvent(None)


class CombinedFutureEvent:
    """A single event over several ``RaidenFutureEvent``-like events.

    Presents one completion surface for an operation that fans out into
    multiple raiden DMAs (e.g. the target and draft KV pools transferred by
    one HiCache write-back): ready only when every underlying event is ready.
    """

    def __init__(self, events):
        self._events = list(events)

    def query(self) -> bool:
        # A failed DMA is reported only once every DMA has landed: the others
        # may still be copying into pages the caller would release on failure.
        done, error = True, None
        for event in self._events:
            try:
                done = event.query() and done
            except RuntimeError as e:
                error = error or e
        if not done:
            return False
        if error is not None:
            raise error
        return True

    def synchronize(self) -> None:
        # Waits for every DMA before raising the first failure, as query() does.
        error = None
        for event in self._events:
            try:
                event.synchronize()
            except RuntimeError as e:
                error = error or e
        if error is not None:
            raise error

    def record(self, *args, **kwargs) -> None:  # noqa: D401 - event-API no-op
        return None

    def wait(self, *args, **kwargs) -> None:  # noqa: D401 - event-API no-op
        return None


class LoadBackEvent(CombinedFutureEvent):
    """A HiCache load-back's DMAs, whose failure is cleaned up, not raised.

    ``error`` keeps this rank's failure. ``failed`` stays None until every rank
    agrees on the outcome (``TpuRaidenHybridCacheController.resolve_load_back``)
    and then says whether the load failed on any rank. A failed load reports
    done only once ``released`` lets its cleanup run, so ``loading_check``
    cannot ack it earlier; an unsettled failure still raises from
    ``synchronize()``.
    """

    def __init__(self, events):
        super().__init__(events)
        self.error = None
        self.failed = None
        self.released = False

    def landed(self) -> bool:
        """This rank's view: True once every DMA has landed, failed or not."""
        if self.error is None:
            try:
                if not super().query():
                    return False
            except RuntimeError as e:
                self.error = e
        return True

    def query(self) -> bool:
        if not self.landed():
            return False
        return self.released if self.failed else self.error is None

    def synchronize(self) -> None:
        if self.failed:
            return
        try:
            super().synchronize()
        except RuntimeError as e:
            self.error = e
        if self.error is not None and self.failed is None:
            raise self.error


class RaidenKVTransfer:
    """Owns a raiden ``KVCacheManager`` over a TPU fused KV pool and issues
    page-granular D2H / H2D transfers.

    The device buffers are registered with raiden by raw PJRT pointer; the
    tensor list MUST stay alive for raiden's lifetime or ``D2h``/``H2d`` will
    dereference freed buffers (segfault). We retain it on ``self``.
    """

    def __init__(
        self,
        device_pool: TpuFusedMHATokenToKVPool,
        num_host_pages: int,
    ):
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

        self.device_pool = device_pool
        self.num_host_pages = num_host_pages
        self.layer_num = device_pool.layer_num

        # outer list = layer, inner list = shard (one shard per TPU core here).
        # Keep-alive: held on self so the underlying buffers outlive raiden.
        self._device_tensors = [
            [device_pool.get_fused_kv_buffer(device_pool.start_layer + i)]
            for i in range(self.layer_num)
        ]
        # raiden issues ``d2h`` / ``h2d`` copies on a background thread only
        # when this variable is set; otherwise the calls block the scheduler
        # thread until every per-layer copy has been submitted. The manager
        # reads it in its constructor.
        os.environ["RAIDEN_ENABLE_ASYNC_DISPATCH"] = "1"
        # Local offload has no control plane; port 0 opens no control listener.
        self._mgr = KVCacheManager(
            self._device_tensors,
            local_control_port=0,
            host_blocks_to_allocate=num_host_pages,
            unsafe_skip_buffer_lock=True,
        )
        logger.info(
            "Raiden KV transfer ready: %d layers, %d host pages (page=%d tokens)",
            self.layer_num,
            num_host_pages,
            device_pool.page_size,
        )

    def backup(
        self, device_page_ids: List[int], host_page_ids: List[int]
    ) -> RaidenFutureEvent:
        """Device->host (D2H): stage device pages into raiden's host pool.

        No device sync precedes the DMA: raiden's D2H reads HBM on a separate
        hardware path that is not ordered against TC compute. Under overlap
        the caller must not issue it while a forward that may write the source
        pages is in flight; the hybrid controller defers write-through behind
        the KV free fence for that.
        """
        n = len(device_page_ids)
        assert n == len(host_page_ids)
        # raiden reads an empty page list as the whole layer buffer.
        if n == 0:
            return NOOP_EVENT
        fut = self._mgr.d2h(device_page_ids, host_page_ids, [1] * n)
        return RaidenFutureEvent(fut)

    def load(
        self, host_page_ids: List[int], device_page_ids: List[int]
    ) -> RaidenFutureEvent:
        """Host->device (H2D): load host pages back into the fused KV buffer."""
        n = len(host_page_ids)
        assert n == len(device_page_ids)
        if n == 0:
            return NOOP_EVENT
        fut = self._mgr.h2d(host_page_ids, device_page_ids, [1] * n)
        return RaidenFutureEvent(fut)
