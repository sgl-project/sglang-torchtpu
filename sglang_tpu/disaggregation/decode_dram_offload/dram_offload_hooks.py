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

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_prefix_offerer import (
    DecodeDramPrefixOfferer,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_victim_receiver import (
    DecodeDramVictimReceiver,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_stage_branches import (
    DecodeStageBranchOfferer,
    DecodeStageBranchReceiver,
    StagePoolSet,
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
    draft_sidecar_host_pools,
    offer_limit,
    swa_host_pool,
    window_tokens,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.decode_dram_offload.stage_agreement import (
    Verdict,
    mark,
    marked,
)
from sglang_tpu.disaggregation.decode_dram_offload.stage_prefill_workers import (
    StagePrefillWorkers,
    StagePushCore,
    StageRecallCore,
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


class PipelinePrefillOffloadExtension(msgspec.Struct, frozen=True):
    """One stage of a pipeline-parallel prefill: the decisions the stages
    vote on (stage_prefill_workers.py) and what they do (stage_prefill_ops.py).
    Its decode is tensor-parallel and uses DecodeDramOffloadExtension over the
    stage-layout workers (decode_stage_branches.py)."""

    workers: StagePrefillWorkers
    push: Any
    recall: Any


DramOffloadExtension = Union[
    PrefillDramOffloadExtension,
    DecodeDramOffloadExtension,
    PipelinePrefillOffloadExtension,
]


def attach_dram_offload_extension(
    *, cache: Any, channel: Any, scheduler: Any, decode_queue: Any = None
) -> DramOffloadExtension:
    """Extend ``cache``'s L2 controller to the peer role's DRAM, reached over
    the raiden ``channel``. A decode passes its transfer queue."""
    host_pool, swa_pool = anchor_host_pool(cache), swa_host_pool(cache)
    channel.bind_host_pools(host_pool, swa_pool)
    page_size = int(scheduler.token_to_kv_pool_allocator.page_size)
    manager = getattr(channel, "_mgr", None)
    if decode_queue is None and getattr(manager, "pp_size", 1) > 1:
        extension = _pipeline_prefill_extension(
            cache, channel, scheduler, host_pool, swa_pool, page_size
        )
        cache.cache_controller.decode_dram_extension = extension
        return extension
    if decode_queue is not None and tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get():
        extension = _stage_decode_extension(
            channel, scheduler, host_pool, page_size, decode_queue
        )
        cache.cache_controller.decode_dram_extension = extension
        return extension
    ctx = DramOffloadWorkerContext(
        cache=cache,
        host_pool=host_pool,
        page_size=page_size,
        channel=channel,
        lockstep=RankLockstep.of(scheduler),
        swa_host_pool=swa_pool,
        window_tokens=window_tokens(cache, page_size),
    )
    extension = (
        PrefillDramOffloadExtension(
            recall=PrefillDramPrefixPuller(ctx), push=PrefillDramVictimPusher(ctx)
        )
        if decode_queue is None
        else DecodeDramOffloadExtension(
            landing=DecodeDramVictimReceiver(ctx),
            source=DecodeDramPrefixOfferer(ctx, decode_queue),
        )
    )
    cache.cache_controller.decode_dram_extension = extension
    return extension


def _pipeline_prefill_extension(
    cache: Any,
    channel: Any,
    scheduler: Any,
    host_pool: Any,
    swa_pool: Any,
    page_size: int,
) -> PipelinePrefillOffloadExtension:
    # Imported here: torch-bound glue only a pipeline prefill needs.
    from sglang_tpu.disaggregation.decode_dram_offload.stage_prefill_ops import (
        PageByteCheck,
        StagePushOps,
        StageRecallOps,
    )

    manager = channel._mgr
    ctx = DramOffloadWorkerContext(
        cache=cache,
        host_pool=host_pool,
        page_size=page_size,
        channel=channel,
        # Each stage is one attention-TP rank (startup_validation.py).
        lockstep=RankLockstep(None),
        swa_host_pool=swa_pool,
        window_tokens=window_tokens(cache, page_size),
    )

    # The extension is attached while the bootstrap queue is being built, so
    # the scheduler does not hold it yet: read it when asked.
    def queued() -> list:
        return scheduler.disagg_prefill_bootstrap_queue.queue

    def serving() -> bool:
        return (
            bool(queued() or scheduler.waiting_queue)
            or not scheduler.running_batch.is_empty()
        )

    # With a draft, its sidecar pools move with the pages (opt-in).
    draft_pools = (
        draft_sidecar_host_pools(cache)
        if tpu_envs.SGLANG_PD_OFFLOAD_ALLOW_DRAFT.get()
        else (None, None)
    )
    channel.bind_host_pools(
        host_pool, swa_pool, draft=draft_pools[0], draft_swa=draft_pools[1]
    )
    byte_check = (
        PageByteCheck(page_size)
        if tpu_envs.SGLANG_PD_OFFLOAD_VERIFY_PAGES.get()
        else None
    )
    push = StagePushOps(
        ctx,
        stage=manager.pp_rank,
        stages=manager.pp_size,
        serving=serving,
        byte_check=byte_check,
        draft_pools=draft_pools,
    )
    recall = StageRecallOps(
        ctx,
        stage=manager.pp_rank,
        queued=queued,
        byte_check=byte_check,
        draft_pools=draft_pools,
    )
    return PipelinePrefillOffloadExtension(
        workers=StagePrefillWorkers(
            StagePushCore(push),
            StageRecallCore(recall, max_inflight_pages=recall.max_inflight_pages()),
        ),
        push=push,
        recall=recall,
    )


def _stage_decode_extension(
    channel: Any, scheduler: Any, host_pool: Any, page_size: int, decode_queue: Any
) -> DecodeDramOffloadExtension:
    """The decode of a pipeline-parallel prefill: pushed branches go to pools
    in the prefill stages' layouts, outside this decode's radix cache."""
    from sglang_tpu.disaggregation.decode_dram_offload.stage_pool_builder import (
        build_stage_pool,
        stage_pool_layout,
    )

    manager = channel._mgr
    own = stage_pool_layout(host_pool, stage=0, stages=1)
    budget = int(tpu_envs.SGLANG_PD_OFFLOAD_STAGE_STORE_GB.get() * 2**30) or (
        own.page_bytes * int(host_pool.transfer.num_host_pages)
    )
    device = host_pool.transfer._device_tensors[0][0].device
    pools = StagePoolSet(
        rank=manager.attn_tp_rank,
        decode_ranks=manager.attn_tp_size,
        budget_bytes=budget,
        build_pool=functools.partial(build_stage_pool, device=device),
    )
    channel.bind_stage_pools(pools)
    last_log = [0.0]
    landing = DecodeStageBranchReceiver(
        channel=channel,
        lockstep=RankLockstep.of(scheduler),
        pools=pools,
        page_size=page_size,
        log_stats=lambda stats: _log_nonzero(stats, "stage landing", last_log),
    )

    def owned_rooms():
        rooms = {dr.req.bootstrap_room for dr in decode_queue.queue}
        rooms.update(
            entry[0].req.bootstrap_room for entry in decode_queue._deferred_releases
        )
        return rooms

    return DecodeDramOffloadExtension(
        landing=landing,
        source=DecodeStageBranchOfferer(
            channel=channel,
            lockstep=RankLockstep.of(scheduler),
            receiver=landing,
            owned_rooms=owned_rooms,
        ),
    )


def _extension(cache: Any) -> Optional[DramOffloadExtension]:
    controller = cache.cache_controller
    if isinstance(controller, TpuDMAHybridCacheController):
        return controller.decode_dram_extension
    return None


def _settle_host_transfers(extension: DramOffloadExtension) -> None:
    """A tick that does not step the P/D queues (a paused engine, a memory
    release) must still retire the transfers in flight, or the engine never
    goes idle."""
    if isinstance(extension, PipelinePrefillOffloadExtension):
        # Its transfers settle on a verdict of the ring, which a tick that
        # does not step the pipeline cannot give; keep the native reports read.
        extension.push._ctx.channel.drain_host_reads()
    elif isinstance(extension, PrefillDramOffloadExtension):
        extension.recall.free_settled_host_landings()
        extension.push.retire_settled_pushes()
    else:
        extension.source.settle_offers()
        extension.landing.finish_agreed_landings()


def _holds_host_pages_outside_tree(extension: DramOffloadExtension) -> bool:
    if isinstance(extension, PipelinePrefillOffloadExtension):
        return extension.workers.busy() or extension.recall.holds_host_landing_pages()
    if isinstance(extension, PrefillDramOffloadExtension):
        return (
            extension.recall.holds_host_landing_pages()
            or extension.push.has_pushes_in_flight()
        )
    return bool(extension.source.offers) or extension.landing.has_host_landings()


def _release_all(extension: DramOffloadExtension) -> None:
    if isinstance(
        extension, (PrefillDramOffloadExtension, PipelinePrefillOffloadExtension)
    ):
        extension.push.release_all()
        extension.recall.release_all()
    else:
        extension.landing.release_all()
        extension.source.release_all()


def _decode_add(
    upstream: Callable, queue: DecodePreallocQueue, req: Any, *args, **kwargs
):
    """Wraps ``DecodePreallocQueue.add``: the decode caches only pushed
    sessions. Behind a pipeline prefill the pushed sessions are outside the
    radix cache, which then caches the decode's own requests as upstream."""
    if not tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get():
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


def _pop_bootstrapped(upstream: Callable, queue: Any, *args, **kwargs) -> Any:
    """Wraps ``PrefillBootstrapQueue.pop_bootstrapped``: drive the prefill's
    end before the step, settle the requests that left bootstrap after it."""
    scheduler = queue.scheduler
    extension = _extension(scheduler.tree_cache)
    if isinstance(extension, PipelinePrefillOffloadExtension):
        return _pop_bootstrapped_on_a_stage(upstream, queue, extension, *args, **kwargs)
    extension.recall.advance(queue.queue, waiting_reqs=scheduler.waiting_queue)
    extension.push.advance(
        serving=bool(queue.queue or scheduler.waiting_queue)
        or not scheduler.running_batch.is_empty()
    )
    result = upstream(queue, *args, **kwargs)
    # With return_failed_reqs the result is (bootstrapped, failed). Core
    # all-reduces the polls, so every rank bootstraps and drops the same
    # requests in a step and the tree inserts below are in step.
    bootstrapped = result[0] if isinstance(result, tuple) else result
    for req in bootstrapped:
        extension.push.note_decode_of(
            req, recalled=extension.recall.insert_recalled_host_prefix(req)
        )
    extension.recall.drop_departed(queue.queue)
    return result


def _pop_bootstrapped_on_a_stage(
    upstream: Callable,
    queue: Any,
    extension: PipelinePrefillOffloadExtension,
    *args,
    **kwargs,
) -> Any:
    """A pipeline stage's bootstrap step. The pipeline calls it on every stage
    with the lists the ring agreed on for a slot; the offload's verdict rides
    in them (stage_agreement.py) and is applied here, before the step."""
    scheduler = queue.scheduler
    extension.recall.unpin_departed([*queue.queue, *scheduler.waiting_queue])
    extension.workers.apply(
        Verdict.of(kwargs.get("pp_good_rids"), kwargs.get("pp_bad_rids"))
    )
    result = upstream(queue, *args, **kwargs)
    bootstrapped = result[0] if isinstance(result, tuple) else result
    for req in bootstrapped:
        extension.push.note_decode_of(
            req, recalled=req.bootstrap_room in extension.recall.recalled
        )
    extension.recall.drop_departed(queue.queue)
    _log_nonzero(
        {
            **extension.workers.counters(),
            # Why a recall was given up before any read: no host room.
            **{f"recall_{k}": v for k, v in extension.recall.stats.items()},
            **(
                extension.recall._byte_check.stats
                if extension.recall._byte_check is not None
                else {}
            ),
        },
        "stage",
        _STAGE_LOG,
    )
    return result


_STAGE_LOG = [0.0]


def _log_nonzero(stats: dict, what: str, last: list) -> None:
    """The periodic counters line, once there is something to count."""
    if any(stats.values()):
        log_offload_stats(stats, what, last)


def _get_rids(upstream: Callable, scheduler: Any, req_queue: Any, *args, **kwargs):
    """Wraps ``Scheduler.get_rids``, with which each pipeline stage collects
    its ready and failed bootstrap requests before folding them into the
    previous stage's: add this stage's votes."""
    result = upstream(scheduler, req_queue, *args, **kwargs)
    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    if queue is None or req_queue is not queue.queue:
        return result
    extension = _extension(scheduler.tree_cache)
    if not isinstance(extension, PipelinePrefillOffloadExtension):
        return result
    if not (isinstance(result, tuple) and len(result) == 2):
        return result
    votes = extension.workers.cast([req.bootstrap_room for req in req_queue])
    good, bad = result
    return [*good, *mark(votes.all_of)], [*bad, *mark(votes.any_of)]


def _process_bootstrapped_queue(
    upstream: Callable, scheduler: Any, bootstrapped_rids: Any
) -> Any:
    """Wraps ``Scheduler.process_bootstrapped_queue``: a stage passes on the
    request ids it popped, so put the offload's verdict back for the next
    stage."""
    result = upstream(scheduler, bootstrapped_rids)
    if result is None or bootstrapped_rids is None:
        return result
    good, bad = bootstrapped_rids
    carried_good, carried_bad = marked(good), marked(bad)
    if not carried_good and not carried_bad:
        return result
    return [[*result[0], *carried_good], [*result[1], *carried_bad]]


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


def _send_kv_metadata(
    upstream: Callable,
    queue: DecodePreallocQueue,
    decode_req: Any,
    kv_indices: Any,
    page_size: int,
    *args,
    **kwargs,
) -> Any:
    """Behind a pipeline prefill the stored prefix is matched in the
    stage-layout index, whatever this decode holds in its own radix cache.

    Wraps ``DecodePreallocQueue._send_kv_metadata``: pin and offer the
    pushed prefix this decode holds for the request first, so that its
    PULL_REQ names the pages. Every rank pre-allocates the same requests in the
    same order in a step, so the host pins agree; a request the decode holds no
    page of (a host receive included) offers nothing."""
    if tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get():
        source = _extension(queue.tree_cache).source
        req = decode_req.req
        limit = offer_limit(req=req, page_size=page_size)
        source.offer_stage_prefix(
            room=req.bootstrap_room,
            tokens=list(req.origin_input_ids[:limit]),
            scope=(req.extra_key, req.cache_salt),
        )
    elif kwargs["decode_prefix_len"] >= page_size:
        _extension(queue.tree_cache).source.offer_host_prefix(decode_req)
    return upstream(queue, decode_req, kv_indices, page_size, *args, **kwargs)


def _pop_transferred(upstream: Callable, queue: DecodeTransferQueue, *args, **kwargs):
    """Wraps ``DecodeTransferQueue.pop_transferred``: close the offers of the
    requests that left the queue in this step, on every rank."""
    result = upstream(queue, *args, **kwargs)
    _extension(queue.tree_cache).source.close_departed_offers()
    return result


def _resolve_deferred_releases(upstream: Callable, queue: DecodeTransferQueue) -> Any:
    """Wraps ``DecodeTransferQueue.resolve_deferred_releases``, run every
    decode iteration on every rank: drive the decode's end."""
    result = upstream(queue)
    extension = _extension(queue.tree_cache)
    extension.source.settle_offers()
    extension.landing.advance()
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
    if isinstance(
        extension, (PrefillDramOffloadExtension, PipelinePrefillOffloadExtension)
    ):
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
        (PrefillBootstrapQueue, "pop_bootstrapped", _pop_bootstrapped),
        (Scheduler, "get_rids", _get_rids),
        (Scheduler, "process_bootstrapped_queue", _process_bootstrapped_queue),
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
