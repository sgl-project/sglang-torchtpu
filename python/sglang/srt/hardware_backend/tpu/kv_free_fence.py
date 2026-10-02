"""Hold freed KV pages back from raiden DMAs until in-flight forwards finish.

Under overlap scheduling, the scheduler processes result N-1 while forward N
still runs. A request that finished at N-1 is still in forward N's batch, so
its last pages are freed while forward N is still writing them. raiden's H2D
DMAs are not ordered against device compute, so a HiCache load allocated one of
those pages can land before forward N's write and be overwritten.

``KVFreeFence`` holds every page freed while a queued forward result is
unprocessed, tagged with the latest formed forward, and releases it once that
result is processed. Which pages a forward writes is never modeled: eviction
frees are held too, since a finished request's tree insert can cover the slot
the in-flight forward writes, unless every evicted node was last inserted or
unlocked before the oldest running forward's batch formed. Forward allocations may
take held pages, because device compute runs in launch order and the later
forward's write lands last. Allocations inside ``dma_write_target()`` use free
pages first and, when short, wait for the in-flight forwards before taking held
ones.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Deque, Iterator, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tpu.allocator_tpu import (
        TpuPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


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
        self._dma_write_target = False
        self._free_release_time: Optional[float] = None
        self._allocators: List[TpuPagedTokenToKVPoolAllocator] = []

    def attach(self, allocator: TpuPagedTokenToKVPoolAllocator) -> None:
        allocator.kv_free_fence = self
        self._allocators.append(allocator)

    @property
    def in_dma_write_target(self) -> bool:
        return self._dma_write_target

    def hold_tag(self) -> Optional[int]:
        """Forward a page freed now must wait for, or None to free it at once."""
        oldest_running = next(
            (t for seq, t, _ in self._in_flight if seq > self._device_done), None
        )
        if oldest_running is None:
            return None
        if (
            self._free_release_time is not None
            and self._free_release_time < oldest_running
        ):
            return None
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

    def wait_for_in_flight(self) -> None:
        """Block until every in-flight forward finishes, then release the
        pages freed during them."""
        # A device sync covers every queued KV write, including ones issued
        # after a result's copy_done.
        launched = self._formed
        if self._in_flight and self._in_flight[-1][2] is None:
            launched -= 1
        self._synchronize_device()
        self._device_done = launched
        self._release(launched)

    def _release(self, completed: int) -> None:
        for allocator in self._allocators:
            allocator.release_held_pages(completed)

    @contextmanager
    def freeing_released_values(self, release_time: Optional[float]) -> Iterator[None]:
        """Frees here are of values no forward formed after ``release_time``
        writes; they skip the hold if every running forward formed later."""
        prev, self._free_release_time = self._free_release_time, release_time
        try:
            yield
        finally:
            self._free_release_time = prev

    @contextmanager
    def dma_write_target(self) -> Iterator[None]:
        """Allocations here are written by DMAs that are not ordered against
        device compute, so they only get pages no in-flight forward may write."""
        prev, self._dma_write_target = self._dma_write_target, True
        try:
            yield
        finally:
            self._dma_write_target = prev


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

    def clear(self) -> None:
        self._entries.clear()
        self.num_pages = 0


def install_kv_free_fence(
    allocator: BaseTokenToKVPoolAllocator, synchronize_device: Callable[[], None]
) -> KVFreeFence:
    """Attach a fence to ``allocator`` and its page pools. Raises unless every
    pool is a TPU paged allocator, whose frees and allocations it hooks."""
    from sglang.srt.hardware_backend.tpu.allocator_tpu import (
        TpuPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    if type(allocator) is SWATokenToKVPoolAllocator:
        pools = [allocator.full_attn_allocator, allocator.swa_attn_allocator]
    else:
        pools = [allocator]
    for pool in pools:
        if not isinstance(pool, TpuPagedTokenToKVPoolAllocator):
            raise ValueError(
                "The raiden HiCache free fence needs TPU paged KV allocators; "
                f"got {type(pool).__name__} under {type(allocator).__name__}."
            )
    fence = KVFreeFence(synchronize_device)
    allocator.kv_free_fence = fence
    for pool in pools:
        fence.attach(pool)
    return fence
