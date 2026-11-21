"""Hooks of the offload to decode DRAM into the scheduler, the P/D queues and
the radix cache: the lifecycle wrappers listed in ``_patch_targets``; core
itself is unchanged (why, and when to remove them: patches.py).

The L2 controller (``TpuDMAHybridCacheController``) offloads KV from device
memory to this engine's host pool. With the switch on it also holds this
engine's end of the extension to decode DRAM (``decode_dram_extension``, set
by ``attach_dram_offload_extension``): the prefill pushes cold branches to the
decode that served them (prefill_dram_victim_pusher.py) and recalls them when
they return (prefill_dram_prefix_puller.py); the decode lands pushed branches
(decode_dram_victim_receiver.py) and offers them back
(decode_dram_prefix_offerer.py). L2 knows neither requests nor peers, so the
P/D layer supplies the raiden channel to the peer rank of a request
(raiden/dram_offload_channel.py) and the bootstrap step in which a recall
must finish. The bytes move RAM to RAM through the host pool's own raiden
manager: no HBM, no load back, no staging slot on either side.

Installed only with SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM=1;
startup_validation.py refuses the settings the offload does not support.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, Union

import msgspec

from sglang.srt.disaggregation.base.conn import KVTransferDestination
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_prefix_offerer import (
    DecodeDramPrefixOfferer,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_victim_receiver import (
    DecodeDramVictimReceiver,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_prefix_puller import (
    PrefillDramPrefixPuller,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    PrefillDramVictimPusher,
    log_offload_stats,
)
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    anchor_host_pool,
    deletion_orphans_host_only_child,
    offer_limit,
    swa_host_pool,
    window_tokens,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.decode_dram_offload.reshard_host_store import (
    DecodeReshardBranchOfferer,
    DecodeReshardBranchReceiver,
    ReshardPoolSet,
    build_reshard_pool,
)
from sglang_tpu.disaggregation.decode_dram_offload.startup_validation import (
    validate_engine_args,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

_is_installed = False


class PrefillDramOffloadExtension(msgspec.Struct, frozen=True):
    """The prefill's end: recall and push, driven around each step of the
    bootstrap queue."""

    recall: PrefillDramPrefixPuller
    push: PrefillDramVictimPusher


class DecodeDramOffloadExtension(msgspec.Struct, frozen=True):
    """The decode's end: land pushed branches and offer them back, driven by
    the decode queues."""

    landing: DecodeDramVictimReceiver
    source: DecodeDramPrefixOfferer
    reshard_landing: Optional[DecodeReshardBranchReceiver] = None
    reshard_source: Optional[DecodeReshardBranchOfferer] = None


DramOffloadExtension = Union[PrefillDramOffloadExtension, DecodeDramOffloadExtension]


def _reshard_store_budget(host_pool: Any) -> int:
    configured = int(tpu_envs.SGLANG_PD_OFFLOAD_RESHARD_STORE_GB.get() * 2**30)
    if configured > 0:
        return configured
    size_per_token = int(getattr(host_pool, "size_per_token", 0) or 0)
    if size_per_token > 0:
        return int(host_pool.size) * size_per_token
    tensors = [
        t for layer in getattr(host_pool.transfer, "_device_tensors", ()) for t in layer
    ]
    if tensors:
        page_bytes = sum(int(t.nbytes) // int(t.shape[0]) for t in tensors)
        return page_bytes * int(host_pool.transfer.num_host_pages)
    return max(1, int(getattr(host_pool, "size", 1)))


def _build_reshard_workers(
    *,
    channel: Any,
    lockstep: RankLockstep,
    host_pool: Any,
    page_size: int,
    decode_queue: Any,
) -> tuple[DecodeReshardBranchReceiver, DecodeReshardBranchOfferer]:
    mgr = getattr(channel, "_mgr", None)
    rank = int(getattr(mgr, "attn_tp_rank", 0))
    tensors = getattr(host_pool.transfer, "_device_tensors", None)
    device = tensors[0][0].device if tensors else "cpu"
    pools = ReshardPoolSet(
        rank=rank,
        budget_bytes=_reshard_store_budget(host_pool),
        build_pool=functools.partial(build_reshard_pool, device=device),
    )
    bind_fn = getattr(channel, "bind_reshard_pools", None)
    if callable(bind_fn):
        bind_fn(pools)
    last_log = [0.0]
    landing = DecodeReshardBranchReceiver(
        channel=channel,
        lockstep=lockstep,
        pools=pools,
        page_size=page_size,
        log_stats=lambda stats: (
            log_offload_stats(stats, "reshard landing", last_log)
            if any(stats.values())
            else None
        ),
    )

    def owned_rooms():
        rooms = {dr.req.bootstrap_room for dr in getattr(decode_queue, "queue", ())}
        rooms.update(
            entry[0].req.bootstrap_room
            for entry in getattr(decode_queue, "_deferred_releases", ())
        )
        return rooms

    source = DecodeReshardBranchOfferer(
        channel=channel,
        lockstep=lockstep,
        receiver=landing,
        owned_rooms=owned_rooms,
    )
    return landing, source


def attach_dram_offload_extension(
    *, cache: Any, channel: Any, scheduler: Any, decode_queue: Any = None
) -> DramOffloadExtension:
    """Extend ``cache``'s L2 controller to the peer role's DRAM, reached over
    the raiden ``channel``. A decode passes its transfer queue."""
    host_pool, swa_pool = anchor_host_pool(cache), swa_host_pool(cache)
    channel.bind_host_pools(
        host_pool, swa_pool, retain_metadata=decode_queue is not None
    )
    page_size = int(scheduler.token_to_kv_pool_allocator.page_size)
    lockstep = RankLockstep.of(scheduler)
    ctx = DramOffloadWorkerContext(
        cache=cache,
        host_pool=host_pool,
        page_size=page_size,
        channel=channel,
        lockstep=lockstep,
        swa_host_pool=swa_pool,
        window_tokens=window_tokens(cache, page_size),
    )
    if decode_queue is None:
        extension: DramOffloadExtension = PrefillDramOffloadExtension(
            recall=PrefillDramPrefixPuller(ctx), push=PrefillDramVictimPusher(ctx)
        )
    else:
        reshard_landing, reshard_source = _build_reshard_workers(
            channel=channel,
            lockstep=lockstep,
            host_pool=host_pool,
            page_size=page_size,
            decode_queue=decode_queue,
        )
        extension = DecodeDramOffloadExtension(
            landing=DecodeDramVictimReceiver(ctx),
            source=DecodeDramPrefixOfferer(ctx, decode_queue),
            reshard_landing=reshard_landing,
            reshard_source=reshard_source,
        )
    cache.cache_controller.decode_dram_extension = extension
    return extension


def _extension(cache: Any) -> Optional[DramOffloadExtension]:
    controller = cache.cache_controller
    if isinstance(controller, TpuDMAHybridCacheController):
        return controller.decode_dram_extension
    return None


def _settle_host_transfers(extension: DramOffloadExtension) -> None:
    """A tick that does not step the P/D queues (a paused engine, a memory
    release) must still retire the transfers in flight, or the engine never
    goes idle."""
    if isinstance(extension, PrefillDramOffloadExtension):
        extension.recall.free_settled_host_landings()
        extension.push.retire_settled_pushes()
    else:
        extension.source.settle_offers()
        extension.landing.finish_agreed_landings()
        reshard_source = getattr(extension, "reshard_source", None)
        reshard_landing = getattr(extension, "reshard_landing", None)
        if reshard_source is not None:
            reshard_source.settle_offers()
        if reshard_landing is not None:
            reshard_landing.finish_agreed_landings()


def _holds_host_pages_outside_tree(extension: DramOffloadExtension) -> bool:
    if isinstance(extension, PrefillDramOffloadExtension):
        return (
            extension.recall.holds_host_landing_pages()
            or extension.push.has_pushes_in_flight()
        )
    reshard_source = getattr(extension, "reshard_source", None)
    reshard_landing = getattr(extension, "reshard_landing", None)
    return (
        bool(extension.source.offers)
        or extension.landing.has_host_landings()
        or bool(reshard_source and reshard_source.offers)
        or bool(reshard_landing and reshard_landing.has_host_landings())
    )


def _release_all(extension: DramOffloadExtension) -> None:
    if isinstance(extension, PrefillDramOffloadExtension):
        extension.push.release_all()
        extension.recall.release_all()
    else:
        extension.landing.release_all()
        extension.source.release_all()
        reshard_landing = getattr(extension, "reshard_landing", None)
        reshard_source = getattr(extension, "reshard_source", None)
        if reshard_landing is not None:
            reshard_landing.release_all()
        if reshard_source is not None:
            reshard_source.release_all()


def _decode_add(
    upstream: Callable, queue: DecodePreallocQueue, req: Any, *args, **kwargs
):
    """Wraps ``DecodePreallocQueue.add``: the decode caches only pushed sessions."""
    req.skip_radix_cache_insert = True
    return upstream(queue, req, *args, **kwargs)


def _prefill_init_kv_manager(upstream: Callable, queue: Any) -> Any:
    """Wraps ``PrefillBootstrapQueue._init_kv_manager``: refuse unsupported
    settings before the manager exists, then extend L2 over its channel."""
    scheduler = queue.scheduler
    validate_engine_args(role="prefill", scheduler=scheduler)
    manager = upstream(queue)
    attach_dram_offload_extension(
        cache=scheduler.tree_cache,
        channel=manager.dram_offload_channel,
        scheduler=scheduler,
    )
    return manager


def _decode_init_kv_manager(upstream: Callable, queue: DecodePreallocQueue) -> Any:
    """Wraps ``DecodePreallocQueue._init_kv_manager``: as on the prefill."""
    validate_engine_args(role="decode", scheduler=queue.scheduler)
    manager = upstream(queue)
    attach_dram_offload_extension(
        cache=queue.tree_cache,
        channel=manager.dram_offload_channel,
        scheduler=queue.scheduler,
        decode_queue=queue.transfer_queue,
    )
    return manager


def _move_bootstrapped(upstream: Callable, queue: Any, *args, **kwargs) -> Any:
    """Wraps ``PrefillBootstrapQueue.move_bootstrapped``: drive the prefill's
    end before the step, settle the requests that left bootstrap after it."""
    scheduler = queue.scheduler
    extension = _extension(scheduler.tree_cache)
    extension.recall.advance(queue.queue, waiting_reqs=scheduler.waiting_queue)
    extension.push.advance(
        serving=bool(queue.queue or scheduler.waiting_queue)
        or not scheduler.running_batch.is_empty()
    )
    bootstrapped = upstream(queue, *args, **kwargs)
    # Core all-reduces the polls, so every rank bootstraps and drops the same
    # requests in a step and the tree inserts below are in step.
    for req in bootstrapped:
        extension.push.note_decode_of(
            req, recalled=extension.recall.insert_recalled_host_prefix(req)
        )
    extension.recall.drop_departed(queue.queue)
    return bootstrapped


def _release_memory_occupation(upstream: Callable, queue: Any) -> Any:
    """Wraps ``PrefillBootstrapQueue.release_memory_occupation``: let go of
    the pages the prefill's end holds first."""
    _release_all(_extension(queue.scheduler.tree_cache))
    return upstream(queue)


def _record_scheduler_state_for_paused_engine(upstream: Callable, scheduler: Any):
    """Wraps ``Scheduler._record_scheduler_state_for_paused_engine``, a tick
    of a paused engine, which does not step the P/D queues."""
    extension = _extension(scheduler.tree_cache)
    if extension is not None:
        _settle_host_transfers(extension)
    return upstream(scheduler)


def _is_reshard_request(decode_req: Any) -> bool:
    infos = getattr(getattr(decode_req, "kv_receiver", None), "bootstrap_infos", None)
    return isinstance(infos, list) and len(infos) > 1


def _send_kv_metadata(
    upstream: Callable,
    queue: DecodePreallocQueue,
    decode_req: Any,
    kv_indices: Any,
    page_size: int,
    *args,
    **kwargs,
) -> Any:
    """Wraps ``DecodePreallocQueue._send_kv_metadata``: pin and offer the
    pushed prefix this decode holds for the request first, so that its
    PULL_REQ or RESHARD_REQ names the pages. Every rank pre-allocates the same
    requests in the same order in a step, so the host pins agree; a request the
    decode holds no page of (a host receive included) offers nothing."""
    if kwargs.get("destination") != KVTransferDestination.HOST:
        if _is_reshard_request(decode_req):
            ext = _extension(queue.tree_cache)
            if ext is not None and ext.reshard_source is not None:
                req = decode_req.req
                limit = offer_limit(req=req, page_size=page_size)
                ext.reshard_source.offer_reshard_prefix(
                    room=req.bootstrap_room,
                    tokens=list(req.origin_input_ids[:limit]),
                    scope=(req.extra_key, req.cache_salt),
                )
        elif kwargs["decode_prefix_len"] >= page_size:
            _extension(queue.tree_cache).source.offer_host_prefix(decode_req)
    return upstream(queue, decode_req, kv_indices, page_size, *args, **kwargs)


def _move_transferred_with_outcomes(
    upstream: Callable, queue: DecodeTransferQueue, *args, **kwargs
):
    """Wraps ``DecodeTransferQueue.move_transferred_with_outcomes``: close the
    offers of the requests that left the queue in this step, on every rank."""
    result = upstream(queue, *args, **kwargs)
    extension = _extension(queue.tree_cache)
    extension.source.close_departed_offers()
    reshard_source = getattr(extension, "reshard_source", None)
    if reshard_source is not None:
        reshard_source.close_departed_offers()
    return result


def _resolve_deferred_releases(upstream: Callable, queue: DecodeTransferQueue) -> Any:
    """Wraps ``DecodeTransferQueue.resolve_deferred_releases``, run every
    decode iteration on every rank: drive the decode's end."""
    result = upstream(queue)
    extension = _extension(queue.tree_cache)
    extension.source.settle_offers()
    extension.landing.advance()
    reshard_source = getattr(extension, "reshard_source", None)
    reshard_landing = getattr(extension, "reshard_landing", None)
    if reshard_source is not None:
        reshard_source.settle_offers()
    if reshard_landing is not None:
        reshard_landing.advance()
    return result


def _has_pending_deferred_releases(
    upstream: Callable, queue: DecodeTransferQueue
) -> bool:
    """Wraps ``DecodeTransferQueue.has_pending_deferred_releases``: offered
    and landing pages sit outside the tree by design too."""
    return upstream(queue) or _holds_host_pages_outside_tree(
        _extension(queue.tree_cache)
    )


def _holds_pages_outside_tree(scheduler: Any) -> bool:
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        return _holds_host_pages_outside_tree(_extension(scheduler.tree_cache))
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        return scheduler.disagg_decode_transfer_queue.has_pending_deferred_releases()
    return False


def _is_fully_idle(
    upstream: Callable,
    scheduler: Any,
    for_health_check: bool = False,
    ignore_waiting: bool = False,
) -> bool:
    """Wraps ``Scheduler.is_fully_idle``: an engine that owns pages outside
    its tree is not idle, so a cache flush is refused and retried instead of
    pulling pages from under a peer's read. A health check asks whether a
    request is running, which those pages do not tell."""
    idle = upstream(
        scheduler, for_health_check=for_health_check, ignore_waiting=ignore_waiting
    )
    if not idle or for_health_check:
        return idle
    return not _holds_pages_outside_tree(scheduler)


# Scheduler steps given to transfers in flight before a memory release: the
# transfer deadline (60 s) at 50 ms a step.
_RELEASE_SETTLE_STEPS = 1200


def _release_memory_occupation_weights(upstream: Callable, updater: Any, recv_req: Any):
    """Wraps ``SchedulerWeightUpdaterManager.release_memory_occupation``,
    which asserts an idle engine: first let the transfers in flight settle, in
    step on every rank."""
    scheduler = updater.scheduler
    for _ in range(_RELEASE_SETTLE_STEPS):
        if not _holds_pages_outside_tree(scheduler):
            break
        _settle_host_transfers(_extension(scheduler.tree_cache))
        time.sleep(0.05)
    return upstream(updater, recv_req)


def _evict_device_leaf(upstream: Callable, cache: Any, node_id: Any, tracker: Any):
    """Wraps ``UnifiedRadixCache._evict_device_leaf``: skip a node id the
    walk deleted earlier in the same walk, and keep an unbacked leaf on the
    device while its deletion would orphan a host-only child."""
    if deletion_orphans_host_only_child(cache, node_id) in (None, True):
        return None
    return upstream(cache, node_id, tracker)


def _checkpoint(upstream, cache, req, **kwargs):
    """Wraps ``UnifiedRadixCache.checkpoint``: route the request's leaf to
    the decode that served it."""
    result = upstream(cache, req, **kwargs)
    extension = _extension(cache)
    if isinstance(extension, PrefillDramOffloadExtension):
        extension.push.track_host_leaf(req)
    return result


def _reset_cache(upstream, cache):
    """Wraps ``UnifiedRadixCache.reset``: let go of every page held outside
    the tree first."""
    extension = _extension(cache)
    if extension is not None:
        _release_all(extension)
    return upstream(cache)


def _patch_targets() -> tuple[tuple[Any, str, Callable], ...]:
    # Imported here: these modules are heavy and installs run early.
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager,
    )

    return (
        (UnifiedRadixCache, "reset", _reset_cache),
        (UnifiedRadixCache, "checkpoint", _checkpoint),
        (UnifiedRadixCache, "_evict_device_leaf", _evict_device_leaf),
        (DecodePreallocQueue, "add", _decode_add),
        (DecodePreallocQueue, "_init_kv_manager", _decode_init_kv_manager),
        (DecodePreallocQueue, "_send_kv_metadata", _send_kv_metadata),
        (PrefillBootstrapQueue, "_init_kv_manager", _prefill_init_kv_manager),
        (PrefillBootstrapQueue, "move_bootstrapped", _move_bootstrapped),
        (
            PrefillBootstrapQueue,
            "release_memory_occupation",
            _release_memory_occupation,
        ),
        (Scheduler, "is_fully_idle", _is_fully_idle),
        (
            SchedulerWeightUpdaterManager,
            "release_memory_occupation",
            _release_memory_occupation_weights,
        ),
        (
            Scheduler,
            "_record_scheduler_state_for_paused_engine",
            _record_scheduler_state_for_paused_engine,
        ),
        (
            DecodeTransferQueue,
            "move_transferred_with_outcomes",
            _move_transferred_with_outcomes,
        ),
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
