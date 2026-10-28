"""Recall victim prefixes through HiCache DRAM on both sides of a P/D pair.

The existing OFFER/PULL/READY handshake is retained. Host offers name pinned
HiCache pages, and native receives land in Prefill's existing host pools. Only
normal Prefill admission loads these pages into HBM. Device recall outside
victim mode continues to use DecodePrefixPuller.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections import defaultdict
from typing import Any, List, Optional, Sequence

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.disaggregation.raiden.prefix_offer import offer_limit, prefix_radix_key
from sglang_tpu.disaggregation.raiden.prefix_pull import (
    DecodePrefixPuller,
    PrefixPullPlan,
    _TreePin,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pd_decode_swa_hicache import swa_window_restore
from sglang_tpu.pd_host_cache import (
    _anchor_host_pool,
    _host_pages,
    _swa_host_pool,
    _window_tokens,
    insert_host_prefix,
)

logger = logging.getLogger(__name__)


def reclaim_full_host_duplicates(cache: Any, num_tokens: int) -> int:
    """Free FULL host copies of childless pages the device still holds, least
    recently used first.

    Host eviction only takes host-only leaves. Under write-through the
    prefill's RAM fills with copies of pages that are still in HBM, so a
    recall from decode RAM finds no room to land. Core reclaims such
    duplicates only under write-back; here a landing is worth more than a
    copy the device still holds, and the copy is rewritten on its next demote.
    A reclaimed node loses its backup, so write-through eviction later deletes
    it instead of demoting it, and write-through host prefixes must stay closed
    under ancestors. A node is therefore taken only while every child of it is
    device-only (chains are taken tip first), and the eviction wrapper in
    pd_victim_push keeps such a node on the device for as long as a child of
    it is host-only, so a deletion never orphans a host-only subtree. Returns
    the tokens freed (0 on a tree core without the duplicate walk).
    """
    core = cache.tree_core
    release = getattr(core, "_release_full_host_duplicate", None)
    duplicates = getattr(core, "full_host_duplicates", None)
    if release is None or duplicates is None or num_tokens <= 0:
        return 0
    tracker: dict = defaultdict(int)
    device_frees: dict = defaultdict(list)
    host_frees: dict = defaultdict(list)
    taken = []
    for node in sorted(duplicates.values(), key=lambda n: n.last_access_time):
        if tracker[ComponentType.FULL] >= num_tokens:
            break
        if not core._can_reclaim_full_host_duplicate(node) or any(
            child.component_data[ComponentType.FULL].value is None
            or child.component_data[ComponentType.FULL].host_value is not None
            for child in node.children.values()
        ):
            # Only a node whose children are all device-only: a child without
            # a device copy would be orphaned if this node, unbacked from now
            # on, were deleted by write-through eviction, and a child with a
            # host copy would sit below a host tombstone, which the next host
            # insert on this path rejects (write-through host prefixes are
            # closed under ancestors). Chains are thus taken tip first.
            continue
        release(node, tracker, device_frees, host_frees)
        taken.append(node.id)
    for node_id in taken:
        duplicates.pop(node_id, None)
    cache._free_values(device_frees, host_frees)
    return int(tracker.get(ComponentType.FULL, 0))


class HostPullPlan(PrefixPullPlan):
    """FULL landing is ``new_indices``; SWA has an independent host allocation."""

    swa_indices: Optional[torch.Tensor] = None


class HostPrefixPuller(DecodePrefixPuller):
    """Reuse the rank-agreed bootstrap lifecycle with host destinations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._host_pool = _anchor_host_pool(self._tree_cache)
        self._swa_host_pool = _swa_host_pool(self._tree_cache)
        self._mgr.bind_prefix_host_pools(self._host_pool, self._swa_host_pool)
        fraction = tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MAX_POOL_FRACTION.get()
        self._max_inflight_tokens = int(fraction * self._host_pool.size)
        self._max_inflight_swa_tokens = (
            int(fraction * self._swa_host_pool.size) if self._swa_host_pool else 0
        )

    def _plan(self, req: Any) -> None:
        room = req.bootstrap_room
        host = self._mgr.prefix_offer_is_host(room)
        offer = self._mgr.take_prefix_offer(room)
        if offer is None:
            return
        self.stats["offers"] += 1
        prefix_tokens, blocks = offer
        page = self._page_size
        want = min(prefix_tokens // page, (len(req.origin_input_ids) - 1) // page)
        swa_first, swa_blocks = self._mgr.prefix_swa_offer(room)
        window_pages = min(want, self._window_pages or 0)
        if not (
            host
            and want > 0
            and want == len(blocks)
            and prefix_tokens == want * page
            and len(swa_blocks) == window_pages
            and (not window_pages or swa_first == want - window_pages)
        ):
            self._mgr.resolve_prefix_pull(room)
            return
        key = self._radix_key(req, want * page)
        match = self._tree_cache.match_prefix(MatchPrefixParams(key=key))
        if len(match.device_indices) + match.host_hit_length >= want * page:
            self.stats["skipped_local_host"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        # Reuse a contiguous, completed FULL host prefix. Device-only pages
        # cannot be used as host indices or anchor an ordinary host insertion.
        core = self._tree_cache.tree_core
        node, have = core.root_node, 0
        while have < want * page:
            child = node.children.get(key.child_key_at(have, page))
            if child is None or child.key.match(key[have:], page_size=page) != len(
                child.key
            ):
                break
            if (
                child.component_data[ComponentType.FULL].host_value is None
                or child.write_through_pending_id is not None
            ):
                break
            node, have = child, have + len(child.key)
        # A complete FULL host hit with an SWA hole still needs a recall.
        # Re-read FULL in that case so the existing nonempty-range protocol
        # can repair the window; insertion releases all duplicate FULL pages.
        if have == want * page:
            node, have = core.root_node, 0
        first = have // page
        count = want - first
        if count < self._min_pages:
            self.stats["skipped_small"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        if (
            self._inflight_tokens() + count * page > self._max_inflight_tokens
            or self._inflight_swa_tokens() + window_pages * page
            > self._max_inflight_swa_tokens
        ):
            self.stats["over_budget"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        have_indices = []
        cursor = node
        while cursor is not core.root_node:
            have_indices.append(cursor.component_data[ComponentType.FULL].host_value)
            cursor = cursor.parent
        receipt = self._tree_cache.inc_host_lock_ref(node.id).to_dec_params()
        if not self._mgr.ask_prefix_pull(
            room, first, count, swa_first_page=swa_first, swa_num_pages=window_pages
        ):
            self._tree_cache.dec_host_lock_ref(node.id, receipt)
            self._mgr.resolve_prefix_pull(room)
            return
        self._plans[room] = HostPullPlan(
            first_page=first,
            num_pages=count,
            remote=list(blocks[first:want]),
            have_indices=torch.cat(list(reversed(have_indices)))
            if have_indices
            else torch.empty(0, dtype=torch.int64),
            new_indices=None,
            lock_node=node.id,
            lock_receipt=receipt,
            asked_at=time.time(),
            swa_first_page=swa_first,
            swa_num_pages=window_pages,
            swa_remote=list(swa_blocks),
        )

    def _allocate(self, room: int, plan: HostPullPlan) -> None:
        allocated = []
        for pool, count, component in (
            (self._host_pool, plan.num_pages, ComponentType.FULL),
            (self._swa_host_pool, plan.swa_num_pages, ComponentType.SWA),
        ):
            if count == 0:
                allocated.append(None)
                continue
            tokens = count * self._page_size
            short = self._all_ranks_reduce(
                [[max(0, tokens - pool.available_size())]], dist.ReduceOp.MAX
            )[0][0]
            if short:
                freed = self._tree_cache.evict_host(short, component)
                if freed < short and component == ComponentType.FULL:
                    reclaim_full_host_duplicates(self._tree_cache, short - freed)
            allocated.append(pool.alloc(tokens))
        full, swa = allocated
        ok = self._all_ranks_reduce(
            [[int(full is not None and (plan.swa_num_pages == 0 or swa is not None))]],
            dist.ReduceOp.MIN,
        )[0][0]
        if not ok:
            if full is not None:
                self._host_pool.free(full)
            if swa is not None:
                self._swa_host_pool.free(swa)
            self.stats["no_memory"] += 1
            self._abandon(room, plan)
            return
        plan.new_indices, plan.swa_indices = full, swa
        self._external_reads_outstanding = True

    def _maybe_start(self, req: Any, plan: HostPullPlan) -> None:
        room = req.bootstrap_room
        ready = self._mgr.prefix_pull_ready(room)
        if ready is None:
            return
        swa_ready = self._mgr.prefix_swa_ready(room) or ("", 0, "", 0, 0)
        rid, uuid, endpoint, first, count = ready
        srid, suuid, sendpoint, sfirst, scount = swa_ready
        if not self._mgr.prefix_ready_is_host(room) or (
            first,
            count,
            sfirst,
            scount,
        ) != (plan.first_page, plan.num_pages, plan.swa_first_page, plan.swa_num_pages):
            self.stats["failed"] += 1
            self._abandon(room, plan)
            return
        plan.req_id, plan.swa_req_id = rid, srid or None
        plan.state, plan.started_at = "pulling", time.time()
        try:
            for swa, read_id, read_uuid, address, remote, local, pool in (
                (
                    False,
                    rid,
                    uuid,
                    endpoint,
                    plan.remote,
                    plan.new_indices,
                    self._host_pool,
                ),
                (
                    True,
                    srid,
                    suuid,
                    sendpoint,
                    plan.swa_remote,
                    plan.swa_indices,
                    self._swa_host_pool,
                ),
            ):
                if remote:
                    self._mgr.start_host_prefix_read(
                        read_id,
                        read_uuid,
                        address,
                        remote,
                        [int(p) for p in pool._to_page_ids(local)],
                        swa=swa,
                    )
        except Exception as exc:
            # Accepted sibling reads still own their destinations. The base
            # lifecycle resolves the other ranks on the next agreement step.
            self.stats["failed"] += 1
            logger.warning(
                "Host prefix recall for room %s could not start: %s", room, exc
            )
            self._abandon(room, plan)

    def commit(self, req: Any) -> None:
        plan = self._plans.pop(req.bootstrap_room, None)
        if plan is None:
            return
        if not plan.agreed_done:
            self.stats["failed"] += 1
            self._discard(req.bootstrap_room, plan)
            return
        have = len(plan.have_indices)
        result = insert_host_prefix(
            self._tree_cache,
            self._radix_key(req, plan.end_page() * self._page_size),
            torch.cat([plan.have_indices, plan.new_indices]),
            plan.swa_indices,
            self._page_size,
            reused_tokens=have,
        )
        if result.inserted_host_node is not None and not result.host_insert_dropped:
            self._pin(req=req, plan=plan, node=result.inserted_host_node)
            self.stats["pulls"] += 1
            self.stats["pulled_tokens"] += len(plan.new_indices)
            self.stats["dedup_tokens"] += max(0, result.prefix_len - have)
        logger.info(
            "[prefix-pull] room=%s pulled_tokens=%d have_tokens=%d swa_tokens=%d "
            "inserted=%s wait_ms=%.0f read_ms=%.0f totals=%s",
            req.bootstrap_room,
            len(plan.new_indices),
            have,
            0 if plan.swa_indices is None else len(plan.swa_indices),
            result.inserted_host_node is not None and not result.host_insert_dropped,
            (plan.started_at - plan.asked_at) * 1e3,
            (time.time() - plan.started_at) * 1e3,
            self.stats,
        )
        self._tree_cache.dec_host_lock_ref(plan.lock_node, plan.lock_receipt)

    def _pin(self, *, req: Any, plan: HostPullPlan, node: Any) -> None:
        self._pins[req.bootstrap_room] = _TreePin(
            req=req,
            node=node,
            receipt=self._tree_cache.inc_host_lock_ref(node).to_dec_params(),
            root=self._tree_cache.root_node_handle(),
            num_tokens=plan.num_pages * self._page_size,
            swa_tokens=plan.swa_num_pages * self._page_size,
        )

    def _unpin(self, room: int) -> None:
        pin = self._pins.pop(room, None)
        if pin is not None and self._tree_cache.root_node_handle() == pin.root:
            self._tree_cache.dec_host_lock_ref(pin.node, pin.receipt)

    def _discard(self, room: int, plan: HostPullPlan) -> None:
        self._tree_cache.dec_host_lock_ref(plan.lock_node, plan.lock_receipt)
        self._mgr.forget_prefix_pull(room)
        if plan.new_indices is None:
            return
        if self._any_read_pending(plan.read_ids()):
            self._zombies.append(
                (plan.read_ids(), plan, plan.swa_num_pages * self._page_size)
            )
        else:
            self._free_plan(plan)

    def _free_plan(self, plan: HostPullPlan) -> None:
        self._host_pool.free(plan.new_indices)
        if plan.swa_indices is not None:
            self._swa_host_pool.free(plan.swa_indices)

    def _reap_zombies(self) -> None:
        keep = []
        for ids, plan, swa_tokens in self._zombies:
            if self._any_read_pending(ids):
                keep.append((ids, plan, swa_tokens))
            else:
                self._free_plan(plan)
        self._zombies = keep

    def _inflight_tokens(self) -> int:
        return (
            sum(p.num_pages * self._page_size for p in self._plans.values())
            + sum(p.num_tokens for p in self._pins.values())
            + sum(len(p.new_indices) for _, p, _ in self._zombies)
        )


class HostPrefixSource:
    """Keep host offers pinned until queue ownership and native reads end."""

    def __init__(self, queue: Any, manager: Any) -> None:
        self.cache, self.manager = queue.tree_cache, manager
        self.page_size = int(queue.scheduler.token_to_kv_pool_allocator.page_size)
        self.full, self.swa = _anchor_host_pool(self.cache), _swa_host_pool(self.cache)
        manager.bind_prefix_host_pools(self.full, self.swa)
        self.offers: dict[int, tuple[Any, Any]] = {}
        self.released: set[int] = set()
        self.groups = [
            g
            for g in (
                queue.scheduler.attn_tp_cpu_group,
                queue.scheduler.attn_cp_cpu_group,
            )
            if g is not None and dist.get_world_size(g) > 1
        ]

    def _reduce(self, rows: List[List[int]], op: Any) -> List[List[int]]:
        if not rows or not self.groups:
            return rows
        tensor = torch.tensor(rows, dtype=torch.int64)
        for group in self.groups:
            dist.all_reduce(tensor, op=op, group=group)
        return tensor.tolist()

    def offer(self, decode_req: Any) -> None:
        req = decode_req.req
        core = self.cache.tree_core
        limit = offer_limit(req=req, page_size=self.page_size)
        key = prefix_radix_key(req=req, num_tokens=limit, is_eagle=core.is_eagle)
        # Matching can split radix nodes but never starts a load-back. This
        # scope enables SWA host coverage only for selecting the offered M.
        with swa_window_restore():
            match = self.cache.match_prefix(MatchPrefixParams(key=key))
        node = core.node_by_id(match.best_match_node)
        window = _window_tokens(self.cache, self.page_size)
        full = swa = None
        tokens = 0
        while node is not core.root_node:
            full = _host_pages(core.root_node, self.full, node)
            tokens = len(full) * self.page_size if full is not None else 0
            swa = (
                _host_pages(
                    core.root_node,
                    self.swa,
                    node,
                    component=ComponentType.SWA,
                    tail_tokens=min(tokens, window),
                )
                if window and tokens
                else []
            )
            if tokens and swa is not None:
                break
            node = node.parent
        valid_tokens = tokens if node is not core.root_node and swa is not None else 0
        low = self._reduce([[valid_tokens]], dist.ReduceOp.MIN)
        high = self._reduce([[valid_tokens]], dist.ReduceOp.MAX)
        if low != high or not low[0][0]:
            full, swa, tokens = [], [], 0
        else:
            receipt = self.cache.inc_host_lock_ref(node.id).to_dec_params()
            self.offers[req.bootstrap_room] = (node.id, receipt)
        self.manager.offer_resident_prefix(
            req.bootstrap_room,
            full or [],
            swa_first_page=max(0, (tokens - window) // self.page_size) if swa else 0,
            swa_page_ids=swa or [],
            source="host",
        )
        decode_req.prefix_offer_sent = True

    def release(self, room: int) -> None:
        # Close before releasing the tree pin: a late PULL on the listener
        # must not register pages after the queue has relinquished them.
        with self.manager._state_lock:
            self.manager._decode.prefix_blocks.pop(room, None)
            self.manager._decode.prefix_swa_blocks.pop(room, None)
            self.manager._decode.prefix_host_offers.discard(room)
        if room in self.offers:
            self.released.add(room)

    def step(self) -> None:
        self.manager._drain_engine()
        rooms = sorted(self.released)
        done = self._reduce(
            [[int(not self.manager.host_prefix_pending(room))] for room in rooms],
            dist.ReduceOp.MIN,
        )
        for room, (settled,) in zip(rooms, done):
            if settled:
                self.cache.dec_host_lock_ref(*self.offers.pop(room))
                self.released.remove(room)


_sources: weakref.WeakKeyDictionary[Any, HostPrefixSource] = weakref.WeakKeyDictionary()


def offer_host_prefixes(queue: Any, requests: Sequence[Any]) -> bool:
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

    manager = queue.scheduler.disagg_decode_prealloc_queue.kv_manager
    if not isinstance(manager, RaidenKVManager) or not manager.prefix_pull_active:
        return False
    source = _sources.get(queue)
    if source is None:
        source = _sources[queue] = HostPrefixSource(queue, manager)
    for dr in requests:
        if not dr.prefix_offer_sent and dr.kv_receiver.prefix_offer_pending:
            source.offer(dr)
    return True


def release_host_offer(queue: Any, req: Any) -> None:
    source = _sources.get(queue)
    if source is not None:
        source.release(req.req.bootstrap_room)


def progress_host_offers(queue: Any) -> None:
    source = _sources.get(queue)
    if source is not None:
        source.step()


def host_offers_pending(queue: Any) -> bool:
    source = _sources.get(queue)
    return source is not None and bool(source.offers)


def reset_host_offers(cache: Any) -> None:
    for source in list(_sources.values()):
        if source.cache is cache:
            source.step()
            if source.offers:
                raise RuntimeError(
                    "Cannot reset KV memory while host prefix offers are pinned"
                )
