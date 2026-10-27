"""Restore a host-only SWA window for device-based prefix offers.

Victim mode offers HiCache host pages directly and bypasses this restore.

A hybrid SWA model's prefix offer (``prefix_offer.lock_window_prefix``) names
only a prefix whose full KV and last sliding window are on device. With host SWA
windows (``pd_decode_swa_hicache``), the last window of a cached path may be in
RAM only. Before the offer, this module loads that window and the full KV
before it back to HBM, holds the request's offer until the load has landed on
every rank, and then lets the offer find it on device. A restore that cannot
run leaves the offer to the device part, it never fails the request.

Every decision reads tree and allocator state, the load slots, or load acks
reaped in lockstep, which all ranks share, so each rank starts, waits for and
releases the same restores in the same step.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any, Callable, List, Optional

import msgspec

from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.decode_hicache_mixin import HiCacheRestoreResult
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.disaggregation.raiden.prefix_offer import (
    deepest_device_window_prefix,
    offer_limit,
    prefix_radix_key,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pd_decode_swa_hicache import (
    _swa_host_window_pages,
    swa_window_restore,
    uses_window_host_cache,
)

logger = logging.getLogger(__name__)

# Arbitrary: bounds the windows pinned on device beyond the offer locks.
_MAX_LOADING_RESTORES = 2


class _WindowRestore(msgspec.Struct, frozen=True):
    """A request's offer restore decision. ``node`` is None when no load runs."""

    # Load anchor, locked by this restore until the offer locks the path.
    node: Any = None
    lock_receipt: Any = None


class _RestorePlan(msgspec.Struct, frozen=True):
    # Deepest node with full KV and a whole window on device or in RAM.
    node: Any
    # Host-only tokens the load moves to device, per pool.
    full_tokens: int
    swa_tokens: int


# Restore decisions per transfer queue, by rid, until the offer or queue exit.
_restores: weakref.WeakKeyDictionary[DecodeTransferQueue, dict[str, _WindowRestore]] = (
    weakref.WeakKeyDictionary()
)


def _offer_resident_prefixes(
    upstream: Callable, queue: DecodeTransferQueue, decode_reqs: List[DecodeRequest]
) -> None:
    """Wraps ``_offer_resident_prefixes``: restore host-only windows first.

    A request whose restore is loading, or waits for a load slot, is left out
    of this step's offers. The restore lock is released in the step the offer
    locks the path, so the path stays locked throughout.
    """
    if tpu_envs.SGLANG_DISAGG_PREFIX_VICTIM_PUSH.get():
        from sglang_tpu.pd_host_prefix_recall import offer_host_prefixes

        if offer_host_prefixes(queue, decode_reqs):
            return
    if not uses_window_host_cache(queue.tree_cache):
        return upstream(queue, decode_reqs)
    restores = _restores.setdefault(queue, {})
    held = _advance_restores(queue=queue, restores=restores, decode_reqs=decode_reqs)
    offered = [dr for dr in decode_reqs if dr.req.rid not in held]
    upstream(queue, offered)
    for decode_req in offered:
        if decode_req.prefix_offer_sent:
            _release_restore(queue=queue, decode_req=decode_req)


def _release_prefix_offer_lock(
    upstream: Callable, queue: DecodeTransferQueue, decode_req: DecodeRequest
) -> None:
    """Wraps ``_release_prefix_offer_lock``: an aborted or finished request also
    drops its restore lock as it leaves the transfer queue."""
    upstream(queue, decode_req)
    from sglang_tpu.pd_host_prefix_recall import release_host_offer

    release_host_offer(queue, decode_req)
    _release_restore(queue=queue, decode_req=decode_req)


def _advance_restores(
    *,
    queue: DecodeTransferQueue,
    restores: dict[str, _WindowRestore],
    decode_reqs: List[DecodeRequest],
) -> set[str]:
    """Start the restores these requests need, return the rids whose offer waits."""
    cache = queue.tree_cache
    held: set[str] = set()
    num_started = 0
    for decode_req in decode_reqs:
        if not _awaits_offer(queue=queue, decode_req=decode_req):
            continue
        rid = decode_req.req.rid
        restore = restores.get(rid)
        if restore is None:
            restore = _try_start_restore(
                queue=queue, restores=restores, decode_req=decode_req
            )
            if restore is None:
                held.add(rid)
                continue
            restores[rid] = restore
            num_started += restore.node is not None
        if _is_loading(cache=cache, restore=restore):
            held.add(rid)
    if num_started:
        cache.ready_to_load_host_cache()
    return held


def _awaits_offer(*, queue: DecodeTransferQueue, decode_req: DecodeRequest) -> bool:
    """Whether the upstream offer would describe a device prefix this step."""
    if decode_req.prefix_offer_sent or decode_req.prefix_match is None:
        return False
    if not decode_req.kv_receiver.prefix_offer_pending:
        return False
    return (
        not queue.scheduler.enable_decode_hicache
        or decode_req.hicache_restore_status == HiCacheRestoreResult.READY
    )


def _is_loading(*, cache: Any, restore: _WindowRestore) -> bool:
    # loading_check reaps an anchor only once its load landed on every rank.
    return restore.node is not None and restore.node in cache.ongoing_load_back


def _try_start_restore(
    *,
    queue: DecodeTransferQueue,
    restores: dict[str, _WindowRestore],
    decode_req: DecodeRequest,
) -> Optional[_WindowRestore]:
    """Decide and maybe start one request's restore, None to retry next step."""
    cache = queue.tree_cache
    allocator = queue.scheduler.token_to_kv_pool_allocator
    page_size = int(allocator.page_size)
    window_size = queue.scheduler.sliding_window_size
    plan = _plan_restore(
        cache=cache, req=decode_req.req, window_size=window_size, page_size=page_size
    )
    if plan is None:
        return _WindowRestore()
    num_loading = sum(_is_loading(cache=cache, restore=r) for r in restores.values())
    if num_loading >= _MAX_LOADING_RESTORES or not cache.has_free_load_back_slot():
        return None
    window_pages = _swa_host_window_pages(
        sliding_window_size=window_size, page_size=page_size
    )
    if not _has_device_room(
        cache=cache,
        allocator=allocator,
        plan=plan,
        min_swa_tokens=window_pages * page_size,
    ):
        logger.debug(
            "PD decode window restore skipped for rid=%s: no device room for "
            "%d full + %d SWA tokens",
            decode_req.req.rid,
            plan.full_tokens,
            plan.swa_tokens,
        )
        return _WindowRestore()
    with swa_window_restore():
        is_loaded = cache.load_back(plan.node)
    if not is_loaded:
        logger.debug(
            "PD decode window restore load_back declined for rid=%s, offering "
            "the device prefix",
            decode_req.req.rid,
        )
        return _WindowRestore()
    receipt = cache.inc_lock_ref(plan.node).to_dec_params()
    return _WindowRestore(node=plan.node, lock_receipt=receipt)


def _plan_restore(
    *, cache: Any, req: Any, window_size: int, page_size: int
) -> Optional[_RestorePlan]:
    """The load that makes the offer deeper, None when the device already holds
    the deepest window."""
    tree_core = cache.tree_core
    limit = offer_limit(req=req, page_size=page_size)
    if limit < page_size:
        return None
    on_device = deepest_device_window_prefix(
        tree_core=tree_core, req=req, window_size=window_size, page_size=page_size
    )
    with swa_window_restore():
        match = cache.match_prefix(
            MatchPrefixParams(
                key=prefix_radix_key(
                    req=req, num_tokens=limit, is_eagle=tree_core.is_eagle
                )
            )
        )
    node = tree_core.node_by_id(match.best_match_node)
    if _depth_tokens(node=node, root=tree_core.root_node) <= on_device:
        return None
    return _RestorePlan(
        node=match.best_match_node,
        full_tokens=_host_only_full_tokens(node),
        swa_tokens=match.swa_host_hit_length,
    )


def _depth_tokens(*, node: Any, root: Any) -> int:
    depth = 0
    while node is not root:
        depth += len(node.key)
        node = node.parent
    return depth


def _host_only_full_tokens(node: Any) -> int:
    """Full KV tokens a load-back anchored at ``node`` moves to device."""
    # Same walk as the full component's LOAD_BACK: device-evicted nodes only
    # sit below every resident one.
    num_tokens = 0
    while node.evicted:
        num_tokens += len(node.component_data[ComponentType.FULL].host_value)
        node = node.parent
    return num_tokens


def _has_device_room(
    *, cache: Any, allocator: Any, plan: _RestorePlan, min_swa_tokens: int
) -> bool:
    full_room = allocator.full_available_size() + cache.full_evictable_size()
    swa_room = allocator.swa_available_size() + cache.swa_evictable_size()
    return full_room >= plan.full_tokens and swa_room >= max(
        plan.swa_tokens, min_swa_tokens
    )


def _release_restore(*, queue: DecodeTransferQueue, decode_req: DecodeRequest) -> None:
    restores = _restores.get(queue)
    if not restores:
        return
    restore = restores.pop(decode_req.req.rid, None)
    if restore is not None and restore.node is not None:
        queue.tree_cache.dec_lock_ref(restore.node, restore.lock_receipt)


def patch_targets() -> tuple[tuple[Any, str, Callable], ...]:
    """The upstream functions this module wraps, as (owner, name, wrapper)."""
    return (
        # Restore a host-only window before the offer reads the device tree.
        (DecodeTransferQueue, "_offer_resident_prefixes", _offer_resident_prefixes),
        # Drop the restore lock when the request leaves the transfer queue.
        (
            DecodeTransferQueue,
            "_release_prefix_offer_lock",
            _release_prefix_offer_lock,
        ),
    )
