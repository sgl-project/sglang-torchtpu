"""Hooks of the offload to decode DRAM into the scheduler, the P/D queues and
the radix cache.

Why: core has no per-request disaggregation cache policy or queue extension.
How: the small lifecycle wrappers listed in ``_patch_targets``; core itself is
unchanged.
Upstream PR or issue: none yet; queue/cache lifecycle seams need upstreaming.
Remove when: core supplies equivalent checkpoint and decode-queue callbacks.

Installed only with SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM=1;
startup_validation.py refuses the settings the offload does not support at startup.
Sticky routing sends a session's next turn to the same decode, whose host
prefix offer lets the prefill recall it into DRAM before ordinary HBM
admission.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.disaggregation.decode_dram_offload.dram_offload_extension import (
    PrefillDramOffloadExtension,
    attach_dram_offload_extension,
)
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    deletion_orphans_host_only_child,
)
from sglang_tpu.disaggregation.decode_dram_offload.startup_validation import (
    validate_engine,
)
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

_is_installed = False


def _extension(cache: Any) -> Any:
    """The decode DRAM extension of ``cache``'s L2 controller, or None for a
    cache without one."""
    controller = cache.cache_controller
    if isinstance(controller, TpuDMAHybridCacheController):
        return controller.decode_dram_extension
    return None


def _decode_add(
    upstream: Callable, queue: DecodePreallocQueue, req: Any, *args, **kwargs
):
    """Wraps ``DecodePreallocQueue.add``: the decode caches only pushed
    sessions, never its own generation requests."""
    req.skip_radix_cache_insert = True
    return upstream(queue, req, *args, **kwargs)


def _prefill_init_kv_manager(upstream: Callable, queue: Any) -> Any:
    """Wraps ``PrefillBootstrapQueue._init_kv_manager``: refuse unsupported
    settings once the manager holds the model's KV pools, then extend the L2
    controller to decode DRAM over it."""
    manager = upstream(queue)
    scheduler = queue.scheduler
    validate_engine(
        role="prefill",
        scheduler=scheduler,
        kv_manager=manager,
        tree_cache=scheduler.tree_cache,
    )
    attach_dram_offload_extension(
        cache=scheduler.tree_cache,
        channel=manager.dram_offload_channel,
        scheduler=scheduler,
    )
    return manager


def _pop_bootstrapped(upstream: Callable, queue: Any, *args, **kwargs) -> Any:
    """Wraps ``PrefillBootstrapQueue.pop_bootstrapped``, one scheduler step of
    the bootstrap queue on every rank: drive the prefill's end before it, and
    after it let it settle the requests that left bootstrap, which are
    admitted (and their prefix matched) only later."""
    scheduler = queue.scheduler
    extension = _extension(scheduler.tree_cache)
    extension.before_bootstrap_step(
        queue.queue,
        waiting_reqs=scheduler.waiting_queue,
        serving=bool(queue.queue or scheduler.waiting_queue)
        or not scheduler.running_batch.is_empty(),
    )
    result = upstream(queue, *args, **kwargs)
    # With return_failed_reqs the result is (bootstrapped, failed).
    bootstrapped = result[0] if isinstance(result, tuple) else result
    extension.after_bootstrap_step(bootstrapped=bootstrapped, queued=queue.queue)
    return result


def _release_memory_occupation(upstream: Callable, queue: Any) -> Any:
    """Wraps ``PrefillBootstrapQueue.release_memory_occupation``: first let go
    of the pages the prefill's end holds, or raise while a read is in flight."""
    _extension(queue.scheduler.tree_cache).release_all()
    return upstream(queue)


def _record_scheduler_state_for_paused_engine(upstream: Callable, scheduler: Any):
    """Wraps ``Scheduler._record_scheduler_state_for_paused_engine``, which
    every rank runs each tick of a paused engine: the P/D queues are not
    stepped then, but transfers in flight still have to settle, or the engine
    never goes idle and cannot be flushed while paused."""
    mode = scheduler.disaggregation_mode
    if mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
        extension = _extension(scheduler.tree_cache)
        if isinstance(extension, PrefillDramOffloadExtension):
            extension.settle_host_transfers()
        elif extension is not None:
            extension.settle_host_transfers()
    return upstream(scheduler)


def _decode_init_kv_manager(upstream: Callable, queue: DecodePreallocQueue) -> Any:
    """Wraps ``DecodePreallocQueue._init_kv_manager``: as on the prefill."""
    manager = upstream(queue)
    validate_engine(
        role="decode",
        scheduler=queue.scheduler,
        kv_manager=manager,
        tree_cache=queue.tree_cache,
    )
    attach_dram_offload_extension(
        cache=queue.tree_cache,
        channel=manager.dram_offload_channel,
        scheduler=queue.scheduler,
        decode_queue=queue.transfer_queue,
    )
    return manager


def _send_kv_metadata(
    upstream: Callable,
    queue: DecodePreallocQueue,
    decode_req: Any,
    kv_indices: Any,
    page_size: int,
    *args,
    **kwargs,
) -> Any:
    """Wraps ``DecodePreallocQueue._send_kv_metadata``, which sends the
    request's PULL_REQ: first pin the pushed prefix this decode holds in RAM
    for the request, so that the PULL_REQ names its pages to the prefill.
    Every rank pre-allocates the same requests in the same order in a step,
    so the host pins agree. A request the decode holds no page of (a host
    receive included) offers nothing."""
    if kwargs.get("decode_prefix_len", 0) >= page_size:
        _extension(queue.tree_cache).source.offer_host_prefix(decode_req)
    return upstream(queue, decode_req, kv_indices, page_size, *args, **kwargs)


def _pop_transferred(upstream: Callable, queue: DecodeTransferQueue, *args, **kwargs):
    """Wraps ``DecodeTransferQueue.pop_transferred``: drop the offers of
    requests that left the queue in this step. Runs on every rank over the
    same requests, so the host pins agree."""
    result = upstream(queue, *args, **kwargs)
    _extension(queue.tree_cache).source.close_departed_offers()
    return result


def _resolve_deferred_releases(upstream: Callable, queue: DecodeTransferQueue) -> Any:
    """Wraps ``DecodeTransferQueue.resolve_deferred_releases``, which every rank
    runs each decode iteration: drive the decode's end of the extension."""
    result = upstream(queue)
    _extension(queue.tree_cache).advance()
    return result


def _has_pending_deferred_releases(
    upstream: Callable, queue: DecodeTransferQueue
) -> bool:
    """Wraps ``DecodeTransferQueue.has_pending_deferred_releases``: granted
    landing pages sit outside the tree by design too, so the idle pool leak
    check waits for them as for deferred releases."""
    extension = _extension(queue.tree_cache)
    return upstream(queue) or (
        extension is not None and extension.holds_host_pages_outside_tree()
    )


def _is_fully_idle(
    upstream: Callable,
    scheduler: Any,
    for_health_check: bool = False,
    ignore_waiting: bool = False,
) -> bool:
    """Wraps ``Scheduler.is_fully_idle``: an engine that still owns pages
    outside its tree (a prefill's source pins and landings, a decode's offered
    and landing pages) is not idle. A cache flush would otherwise pull those
    pages from under a peer's read; it is refused like any other pending work
    and retried by the caller. Neither query performs a collective. A health
    check asks whether a request is running, which those pages do not tell."""
    idle = upstream(
        scheduler, for_health_check=for_health_check, ignore_waiting=ignore_waiting
    )
    if not idle or for_health_check:
        return idle
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        extension = _extension(scheduler.tree_cache)
        return extension is None or not extension.holds_host_pages_outside_tree()
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        return (
            not scheduler.disagg_decode_transfer_queue.has_pending_deferred_releases()
        )
    return True


def _evict_device_leaf(upstream: Callable, cache: Any, node_id: Any, tracker: Any):
    """Wraps ``UnifiedRadixCache._evict_device_leaf``: skip a leaf the walk
    already deleted (it can yield an unbacked leaf twice), and keep an
    unbacked leaf on the device while its deletion would orphan a host-only
    child."""
    if deletion_orphans_host_only_child(cache, node_id) in (None, True):
        return None
    return upstream(cache, node_id, tracker)


def _checkpoint(upstream, cache, req, **kwargs):
    result = upstream(cache, req, **kwargs)
    extension = _extension(cache)
    if isinstance(extension, PrefillDramOffloadExtension):
        extension.push.track_host_leaf(req)
    return result


def _reset_cache(upstream, cache):
    extension = _extension(cache)
    if extension is not None:
        extension.before_cache_reset()
    return upstream(cache)


def _patch_targets() -> tuple[tuple[Any, str, Callable], ...]:
    # Imported here: these modules are heavy and installs run early.
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
    from sglang.srt.managers.scheduler import Scheduler

    return (
        (UnifiedRadixCache, "reset", _reset_cache),
        (UnifiedRadixCache, "checkpoint", _checkpoint),
        (UnifiedRadixCache, "_evict_device_leaf", _evict_device_leaf),
        (DecodePreallocQueue, "add", _decode_add),
        (DecodePreallocQueue, "_init_kv_manager", _decode_init_kv_manager),
        (DecodePreallocQueue, "_send_kv_metadata", _send_kv_metadata),
        (PrefillBootstrapQueue, "_init_kv_manager", _prefill_init_kv_manager),
        (PrefillBootstrapQueue, "pop_bootstrapped", _pop_bootstrapped),
        (
            PrefillBootstrapQueue,
            "release_memory_occupation",
            _release_memory_occupation,
        ),
        (Scheduler, "is_fully_idle", _is_fully_idle),
        (
            Scheduler,
            "_record_scheduler_state_for_paused_engine",
            _record_scheduler_state_for_paused_engine,
        ),
        (DecodeTransferQueue, "pop_transferred", _pop_transferred),
        (DecodeTransferQueue, "resolve_deferred_releases", _resolve_deferred_releases),
        (
            DecodeTransferQueue,
            "has_pending_deferred_releases",
            _has_pending_deferred_releases,
        ),
    )


def _wrap(owner: Any, name: str, impl: Callable) -> None:
    upstream = getattr(owner, name)

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return impl(upstream, *args, **kwargs)

    setattr(owner, name, wrapper)


def install() -> None:
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    for owner, name, impl in _patch_targets():
        _wrap(owner, name, impl)
