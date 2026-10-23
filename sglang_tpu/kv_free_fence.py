"""Hold freed KV pages back from DMAs until in-flight forwards finish.

Under overlap scheduling, the scheduler processes result N-1 while forward N
still runs. A request that finished at N-1 is still in forward N's batch, so
its last pages are freed while forward N is still writing them. DMAs such as
raiden's H2D loads and KV arriving on a PD decode server are not ordered
against device compute, so a DMA into one of those pages can land before
forward N's write and be overwritten.

``KVFreeFence`` holds every page freed while a queued forward result is
unprocessed, tagged with the latest formed forward, and releases it once that
result is processed. Which pages a forward writes is never modeled: eviction
frees are held too, since a finished request's tree insert can cover the slot
the in-flight forward writes, unless every evicted node was last inserted or
unlocked before the oldest running forward's batch formed. Allocations inside
``forward_write_target()`` may take held pages, because device compute runs in
launch order and the later forward's write lands last. Every other allocation
uses free pages first and, when short, waits for the in-flight forwards before
taking held ones, so a missed scope costs a device sync, not corrupted KV.

DMA reads have the mirror hazard: a device-to-host backup issued while a forward
is in flight can copy a slot that forward has not written yet, such as the last
slot of a request that finished at N-1 and was inserted into the tree.
``after_in_flight`` runs a DMA read once the forwards formed so far are done.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Deque, Iterator, List, Optional, Tuple

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo
    from sglang_tpu.allocator_tpu import (
        TpuPagedTokenToKVPoolAllocator,
    )

logger = logging.getLogger(__name__)

# Arbitrary; one stats line a minute while forwards run.
_STATS_LOG_INTERVAL_S = 60.0


class _AfterInFlight(msgspec.Struct, frozen=True):
    """A callback awaiting every forward formed up to ``seq``."""

    seq: int
    callback: Callable[[], None]


class KVFreeFenceStats(msgspec.Struct):
    """Counts since the last stats line, summed over page pools. Device syncs
    per non-forward pool allocation rise if eviction frees stop skipping the hold."""

    pages_held: int = 0
    pages_freed_idle: int = 0
    pages_freed_past_write_horizon: int = 0
    pages_borrowed: int = 0
    non_forward_allocs: int = 0
    device_syncs: int = 0
    device_sync_s: float = 0.0


class KVFreeFence:
    """Tracks formed, unprocessed forwards for the TPU paged allocators."""

    def __init__(self, synchronize_device: Callable[[], None]) -> None:
        # Imported here: the unified cache components import the SWA allocator,
        # which imports this module on TPU.
        from sglang.srt.mem_cache.unified_cache.components.base import (
            get_and_increase_time_counter,
        )

        self._time_counter = get_and_increase_time_counter
        self._synchronize_device = synchronize_device
        self._formed = 0
        # (seq, formation time on the radix tree's counter, result) of formed
        # forwards whose results are unprocessed; result is None until launch.
        self._in_flight: Deque[Tuple[int, float, object]] = deque()
        # Latest seq a device sync has covered.
        self._device_done = 0
        self._forward_write_target = False
        self._free_write_horizon: Optional[float] = None
        self._after: Deque[_AfterInFlight] = deque()
        self._allocators: List[TpuPagedTokenToKVPoolAllocator] = []
        self.stats = KVFreeFenceStats()
        self._stats_since = time.monotonic()

    def attach(self, allocator: TpuPagedTokenToKVPoolAllocator) -> None:
        allocator.kv_free_fence = self
        self._allocators.append(allocator)

    @property
    def in_forward_write_target(self) -> bool:
        return self._forward_write_target

    def hold_tag(self, *, num_pages: int) -> Optional[int]:
        """Forward ``num_pages`` pages freed now must wait for, or None to free
        them at once."""
        oldest_running = next(
            (t for seq, t, _ in self._in_flight if seq > self._device_done), None
        )
        if oldest_running is None:
            self.stats.pages_freed_idle += num_pages
            return None
        if (
            self._free_write_horizon is not None
            and self._free_write_horizon < oldest_running
        ):
            self.stats.pages_freed_past_write_horizon += num_pages
            return None
        self.stats.pages_held += num_pages
        return self._formed

    def on_batch_formed(self) -> None:
        """Call once the next batch's requests are fixed, before any result is
        processed ahead of its launch: that processing can finish a request
        the batch still writes."""
        assert not self._in_flight or self._in_flight[-1][2] is not None
        self._formed += 1
        self._in_flight.append((self._formed, self._time_counter(), None))

    def on_forward_launched(self, result) -> None:
        """Call when the formed batch's result is queued for deferred
        processing (overlap)."""
        seq, formed_at, pending = self._in_flight.pop()
        assert pending is None
        self._in_flight.append((seq, formed_at, result))

    def on_forward_done(self, result) -> None:
        """Call after ``result`` is processed; its ``copy_done`` must cover every
        KV write of its forward. Results processed without being queued are a
        no-op."""
        if not self._in_flight or self._in_flight[0][2] is not result:
            # Queued results are processed in order; a skipped one would hold
            # its pages forever.
            assert all(queued is not result for _, _, queued in self._in_flight)
            return
        self._in_flight.popleft()
        if result.copy_done is not None:
            result.copy_done.synchronize()
        # Pages tagged with a forward still in flight stay held.
        oldest = self._in_flight[0][0] if self._in_flight else self._formed + 1
        self._release(oldest - 1)
        self._maybe_log_stats()

    def wait_for_in_flight(self) -> None:
        """Block until every in-flight forward finishes, then release the
        pages freed during them."""
        # A device sync covers every queued KV write, including ones issued
        # after a result's copy_done.
        launched = self._formed
        if self._in_flight and self._in_flight[-1][2] is None:
            launched -= 1
        start = time.perf_counter()
        self._synchronize_device()
        self.stats.device_syncs += 1
        self.stats.device_sync_s += time.perf_counter() - start
        self._device_done = launched
        self._release(launched)

    def after_in_flight(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once every forward formed so far has finished: now
        if none is running, else when that forward's result is processed or a
        device sync covers it. Callbacks run in registration order."""
        if not self._after and all(
            seq <= self._device_done for seq, _, _ in self._in_flight
        ):
            callback()
            return
        self._after.append(_AfterInFlight(self._formed, callback))

    def _release(self, completed: int) -> None:
        for allocator in self._allocators:
            allocator.release_held_pages(completed)
        while self._after and self._after[0].seq <= completed:
            self._after.popleft().callback()

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._stats_since < _STATS_LOG_INTERVAL_S:
            return
        stats, self.stats = self.stats, KVFreeFenceStats()
        window_s, self._stats_since = now - self._stats_since, now
        logger.info(
            "KV free fence over %.0fs: pages held %d, freed idle %d, freed past "
            "write horizon %d, borrowed %d, held now %d; non-forward pool "
            "allocations %d, device syncs %d (%.3f per allocation, %.1f ms)",
            window_s,
            stats.pages_held,
            stats.pages_freed_idle,
            stats.pages_freed_past_write_horizon,
            stats.pages_borrowed,
            sum(allocator.held_pages.num_pages for allocator in self._allocators),
            stats.non_forward_allocs,
            stats.device_syncs,
            stats.device_syncs / max(stats.non_forward_allocs, 1),
            stats.device_sync_s * 1e3,
        )

    @contextmanager
    def freeing_values_written_before(
        self, write_horizon: Optional[float]
    ) -> Iterator[None]:
        """Frees here are of values no forward formed after ``write_horizon``
        writes; they skip the hold if every running forward formed later."""
        prev, self._free_write_horizon = self._free_write_horizon, write_horizon
        try:
            yield
        finally:
            self._free_write_horizon = prev

    @contextmanager
    def forward_write_target(self) -> Iterator[None]:
        """Allocations here are written only by forwards, which run after the
        in-flight ones, so they may take pages those forwards still write."""
        prev, self._forward_write_target = self._forward_write_target, True
        try:
            yield
        finally:
            self._forward_write_target = prev


class _DeferredDmaEvent:
    """Completion of a DMA read whose issue waits on ``after_in_flight``.

    ``query`` reports not ready until the DMA is issued and done.
    ``synchronize`` (a blocking caller) waits for the in-flight forwards first.
    """

    _warned = False

    def __init__(self, fence: KVFreeFence, issue: Callable[[], object]) -> None:
        self._fence = fence
        self._issue_dma = issue
        self._event = None

    def issue(self) -> None:
        if self._event is None:
            self._event = self._issue_dma()

    def query(self) -> bool:
        return self._event is not None and self._event.query()

    def synchronize(self) -> None:
        if self._event is None:
            self._fence.wait_for_in_flight()
        if self._event is None:
            # Registered against a batch formed but not launched, which a
            # device sync cannot cover. Only a blocking write-back eviction
            # reaches this, and it takes the least recently used nodes, not
            # the tail just inserted; issue unfenced rather than stall.
            if not _DeferredDmaEvent._warned:
                _DeferredDmaEvent._warned = True
                logger.warning(
                    "Blocking on a KV backup before the formed batch launches; "
                    "issuing it unfenced."
                )
            self.issue()
        self._event.synchronize()

    def record(self, *args, **kwargs) -> None:
        return None

    def wait(self, *args, **kwargs) -> None:
        return None


def fence_hicache_backups(controller, fence: KVFreeFence) -> None:
    """Defer the tree's raiden HiCache backups behind in-flight forwards. Only
    the tree's controller backs up pages a running forward can still write;
    other controllers (e.g. the decode offload manager's) read finished KV and
    stay unfenced. ``controller`` is None when the tree has no HiCache
    controller (external linker)."""
    from sglang.srt.hardware_backend.tpu.raiden_hybrid_cache_controller import (
        TpuRaidenHybridCacheController,
    )

    if controller is None:
        return
    assert isinstance(controller, TpuRaidenHybridCacheController), type(
        controller
    ).__name__
    controller.kv_free_fence = fence


def deferred_dma_read(fence: Optional[KVFreeFence], issue: Callable[[], object]):
    """Issue a DMA that reads KV pages once no in-flight forward may still write
    them; returns its completion event. Without a fence, issues at once."""
    if fence is None:
        return issue()
    event = _DeferredDmaEvent(fence, issue)
    fence.after_in_flight(event.issue)
    return event


class HeldPages:
    """FIFO of (forward tag, page ids) freed while that forward was in flight."""

    def __init__(self) -> None:
        self._entries: Deque[Tuple[int, torch.Tensor]] = deque()
        self.num_pages = 0

    def push(self, tag: int, page_ids: torch.Tensor) -> None:
        if page_ids.numel() == 0:
            return
        self._entries.append((tag, page_ids))
        self.num_pages += page_ids.numel()

    def pop_released(self, completed: int) -> List[torch.Tensor]:
        """Pages whose forward has completed; tags are non-decreasing."""
        released = []
        while self._entries and self._entries[0][0] <= completed:
            released.append(self._entries.popleft()[1])
        self.num_pages -= sum(ids.numel() for ids in released)
        return released

    def borrow(self, num_pages: int) -> torch.Tensor:
        """Take ``num_pages`` held pages for a forward allocation."""
        assert 0 < num_pages <= self.num_pages
        taken = []
        remaining = num_pages
        while remaining > 0:
            tag, ids = self._entries.popleft()
            if ids.numel() > remaining:
                self._entries.appendleft((tag, ids[remaining:]))
                ids = ids[:remaining]
            taken.append(ids)
            remaining -= ids.numel()
        self.num_pages -= num_pages
        return torch.cat(taken)


def assert_not_forward_write_target(allocator: BaseTokenToKVPoolAllocator) -> None:
    """A DMA lands whenever it lands, so its target must never be a page an
    in-flight forward may still write: only forward allocations may borrow."""
    fence = allocator.kv_free_fence
    assert fence is None or not fence.in_forward_write_target, (
        "A DMA target was allocated inside forward_write_target()"
    )


def _page_pools(
    allocator: BaseTokenToKVPoolAllocator,
) -> List[BaseTokenToKVPoolAllocator]:
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang_tpu.unified_kv_pool import TpuUnifiedSWATokenToKVPoolAllocator

    if type(allocator) in (
        SWATokenToKVPoolAllocator,
        TpuUnifiedSWATokenToKVPoolAllocator,
    ):
        return [allocator.full_attn_allocator, allocator.swa_attn_allocator]
    return [allocator]


def kv_free_fence_unsupported_reason(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    spec_algorithm: SpeculativeAlgorithm | CustomSpecAlgo,
    copies_draft_tail: bool,
) -> Optional[str]:
    """Why a fence on ``allocator`` would miss KV writes, or None: it hooks only
    TPU paged pools, and needs every KV write to precede a copied result tensor."""
    from sglang_tpu.allocator_tpu import (
        TpuPagedTokenToKVPoolAllocator,
    )

    if not spec_algorithm.supports_kv_free_fence(copies_draft_tail=copies_draft_tail):
        # copy_done would complete before the KV writes that follow every copy.
        hint = (
            " unless SGLANG_TPU_ASYNC_COPY_WORKER is on"
            if spec_algorithm.supports_kv_free_fence(copies_draft_tail=True)
            else ""
        )
        return (
            f"{spec_algorithm.name} speculative decoding is not supported with "
            f"the KV free fence under overlap scheduling on TPU{hint}."
        )
    pools = _page_pools(allocator)
    if not all(isinstance(pool, TpuPagedTokenToKVPoolAllocator) for pool in pools):
        return (
            "The KV free fence needs TPU paged KV allocators; got "
            f"{[type(pool).__name__ for pool in pools]} under "
            f"{type(allocator).__name__}."
        )
    return None


def install_kv_free_fence(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    synchronize_device: Callable[[], None],
    spec_algorithm: SpeculativeAlgorithm | CustomSpecAlgo,
    copies_draft_tail: bool,
) -> KVFreeFence:
    """Attach a fence to ``allocator`` and its page pools. Raises if
    ``kv_free_fence_unsupported_reason`` gives a reason."""
    reason = kv_free_fence_unsupported_reason(
        allocator,
        spec_algorithm=spec_algorithm,
        copies_draft_tail=copies_draft_tail,
    )
    if reason is not None:
        raise ValueError(reason)
    fence = KVFreeFence(synchronize_device)
    allocator.kv_free_fence = fence
    for pool in _page_pools(allocator):
        fence.attach(pool)
    return fence
