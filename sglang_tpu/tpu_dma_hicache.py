# SPDX-License-Identifier: Apache-2.0
"""
TPU HiCache is implemented with a generic "TPU DMA path" with a swappable
H2D/D2H copy engine (which uses Raiden by default).

The overall design is:
- TpuDMAHiCacheController (single-pool radix cache controller)
- TpuDMAHybridCacheController (unified full+SWA pool controller)
- TpuDMAHostKVPool (stores the host memory pool)

The swappable copy engine must adhere to the TpuDMAKVTransfer and
TransferEvent protocols located in this file. Please only use APIs
in the interface in the TPU HiCache implementation (and add more if needed).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class TransferEvent(Protocol):
    def query(self) -> bool:
        """Return whether the copy is done, without blocking; raise if it failed."""

    def synchronize(self) -> None:
        """Wait until the copy is done; raise if it failed.

        Prefer to wait by polling, not with a blocking PJRT Await (PJRT_Event_Await /
        BlockUntilReady): raiden found that can deadlock against the running model.
        """


class TpuDMAKVTransfer(Protocol):
    """Copies whole pages between one device KV sub-pool and host memory.

    The registered factory builds it as ``factory(device_pool, num_host_pages)``,
    once per device sub-pool (full, SWA, draft, recurrent state), after warm-up, on
    the thread that issues torch ops on the pool. It may bind the pool's
    ``page_major_tensors`` once, so they must only be written in place.
    It owns the host memory for ``num_host_pages`` pages (allocation, pinning, DMA
    registration); ``TpuDMAHostKVPool`` owns which pages are free.
    """

    def backup(self, device_pages: list[int], host_pages: list[int]) -> TransferEvent:
        """Start copying every layer of ``device_pages[i]`` to ``host_pages[i]``.

        Returns without waiting. The copy is not ordered against compute, so the
        caller fences it (``kv_free_fence``) and keeps torch off the pages until
        the event is done.
        """

    def load(self, host_pages: list[int], device_pages: list[int]) -> TransferEvent:
        """Start copying every layer of ``host_pages[i]`` to ``device_pages[i]``.

        Returns without waiting; the caller keeps the forward pass off the pages
        until the event is done.
        """


def page_major_tensors(device_pool) -> list:
    """The device tensors a page of ``device_pool`` spans, each indexed by page on dim 0.

    A fused pool registers its ``page_buffers()`` (one per layer, or one bundle in
    the block-major layout), the split sparse-MLA pool several buffers per layer,
    and a recurrent-state pool one tensor per layer and state kind, whose "pages"
    are state slots.
    """
    from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool
    from sglang_tpu.mamba_pool import TpuMambaPool

    if isinstance(device_pool, TpuMambaPool):
        return device_pool.get_state()
    if isinstance(device_pool, TpuSplitMLATokenToKVPool):
        layers = range(
            device_pool.start_layer, device_pool.start_layer + device_pool.layer_num
        )
        return [buf for layer in layers for buf in device_pool.get_page_buffers(layer)]
    return device_pool.page_buffers()


def register_tpu_dma_hicache_backend(
    name: str, factory: Callable[[Any, int], TpuDMAKVTransfer]
) -> None:
    """Register ``factory(device_pool, num_host_pages)`` as engine ``name``.

    Raises ``ValueError`` if ``name`` is already registered.
    """
    if name in _ENGINES:
        raise ValueError(f"TPU DMA HiCache backend {name!r} is already registered")
    _ENGINES[name] = factory


def is_tpu_dma_hicache_backend(io_backend: str | None) -> bool:
    return io_backend in _ENGINES


def tpu_dma_hicache_backends() -> list[str]:
    """The registered names, sorted, for error messages."""
    return sorted(_ENGINES)


def create_tpu_dma_kv_transfer(
    io_backend: str, device_pool, num_host_pages: int, *, draft_pools: tuple = ()
) -> TpuDMAKVTransfer:
    """Build ``io_backend``'s engine; raises ``ValueError`` if it is not registered.

    ``draft_pools`` are packed MTP draft pools on the target's page ids; an
    engine that copies them takes them as ``factory(..., draft_pools=...)``.
    """
    if io_backend not in _ENGINES:
        raise ValueError(
            f"--hicache-io-backend {io_backend!r} is not one of "
            f"{', '.join(sorted(_ENGINES))}"
        )
    if draft_pools:
        return _ENGINES[io_backend](
            device_pool, num_host_pages, draft_pools=draft_pools
        )
    return _ENGINES[io_backend](device_pool, num_host_pages)


# An already-complete event, for an empty copy and the unused ack start slot.
class _NoopEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


NOOP_EVENT = _NoopEvent()


class CombinedFutureEvent:
    """A single event over several events.

    Presents one completion surface for an operation that fans out into
    multiple engine copies (e.g. the target and draft KV pools transferred by
    one HiCache write-back): ready only when every underlying event is ready.
    """

    def __init__(self, events):
        self._events = list(events)

    def query(self) -> bool:
        return all(event.query() for event in self._events)

    def synchronize(self) -> None:
        for event in self._events:
            event.synchronize()


_ENGINES: dict[str, Callable[[Any, int], TpuDMAKVTransfer]] = {}


def _raiden(
    device_pool, num_host_pages: int, draft_pools: tuple = ()
) -> TpuDMAKVTransfer:
    # Looked up per call, so tpu-sync loads only when a raiden engine is built
    # and tests can patch RaidenKVTransfer.
    from sglang_tpu import raiden_transfer

    return raiden_transfer.RaidenKVTransfer(
        device_pool, num_host_pages, draft_pools=draft_pools
    )


register_tpu_dma_hicache_backend("raiden", _raiden)
