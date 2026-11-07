"""Recall victim prefixes through HiCache DRAM on both sides of a P/D pair.

The decode offers the pushed prefix it holds in RAM (``HostPrefixSource``),
naming pinned HiCache host pages in its PULL_REQ; the prefill reads the range
it lacks into its own host pools (``HostPrefixPuller``) over the Raiden
PULL/READY handshake. Only normal Prefill admission loads these pages
into HBM, so the recall uses no device page and no staging slot on either
role.
"""

from __future__ import annotations

import logging
import time
import weakref
from array import array
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import page_align_floor
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.pd_host_cache import (
    _all_ranks_reduce,
    _anchor_host_pool,
    _host_pages,
    _rank_groups,
    insert_host_prefix,
)

logger = logging.getLogger(__name__)

# Arbitrary: the share of the prefill's host pool that landings of in-flight
# recalls may hold, leaving the rest for the prefill's own backups.
_MAX_INFLIGHT_POOL_FRACTION = 0.5


def prefix_radix_key(*, req: Any, num_tokens: int) -> RadixKey:
    """The tree key of the request's first ``num_tokens`` prompt tokens of KV."""
    return RadixKey(
        token_ids=array("q", req.origin_input_ids[:num_tokens]),
        extra_key=req.extra_key,
        cache_salt=req.cache_salt,
    )


def offer_limit(*, req: Any, page_size: int) -> int:
    """Longest prompt prefix worth offering: the prefill computes at least the
    last prompt token."""
    return page_align_floor(len(req.origin_input_ids) - 1, page_size)


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
    taken_ids = set()
    # Taking a tip makes its parent eligible; repeat the LRU pass until the
    # target is met or a pass takes nothing, so a chain whose first eligible
    # tip is smaller than the landing still yields the rest of the chain.
    while tracker[ComponentType.FULL] < num_tokens:
        progress = False
        for node in sorted(duplicates.values(), key=lambda n: n.last_access_time):
            if tracker[ComponentType.FULL] >= num_tokens:
                break
            if node.id in taken_ids:
                continue
            if not core._can_reclaim_full_host_duplicate(node) or any(
                child.component_data[ComponentType.FULL].value is None
                or child.component_data[ComponentType.FULL].host_value is not None
                for child in node.children.values()
            ):
                # Only a node whose children are all device-only: a child
                # without a device copy would be orphaned if this node,
                # unbacked from now on, were deleted by write-through
                # eviction, and a child with a host copy would sit below a
                # host tombstone, which the next host insert on this path
                # rejects (write-through host prefixes are closed under
                # ancestors). Chains are thus taken tip first.
                continue
            release(node, tracker, device_frees, host_frees)
            taken.append(node.id)
            taken_ids.add(node.id)
            progress = True
        if not progress:
            break
    for node_id in taken:
        duplicates.pop(node_id, None)
    cache._free_values(device_frees, host_frees)
    return int(tracker.get(ComponentType.FULL, 0))


class HostPullPlan(msgspec.Struct):
    """One request's pull: the offered page range asked for and what it joins."""

    # Full pages read from the decode: [first_page, +num_pages), ending at M.
    first_page: int
    num_pages: int
    # The decode's page ids of the asked range.
    remote: List[int]
    # Host KV indices of the prefill's own prefix [0, first_page) the pulled
    # pages join (host-locked until commit).
    have_indices: Any
    # Host KV indices reserved for pages [first_page, M), None until every
    # rank has READY.
    new_indices: Any
    lock_node: Any
    lock_receipt: Any
    asked_at: float
    # "asked" until the decode's READY starts the read, then "pulling".
    state: str = "asked"
    req_id: Optional[str] = None
    started_at: float = 0.0
    # Set once every rank saw the read settle: True only if it succeeded everywhere.
    agreed_done: Optional[bool] = None

    def read_ids(self) -> List[str]:
        return [] if self.req_id is None else [self.req_id]

    def end_page(self) -> int:
        return self.first_page + self.num_pages


class _TreePin(msgspec.Struct, frozen=True):
    """A committed pull's tree lock, held until its request leaves the queues."""

    req: Any
    node: Any
    receipt: Any
    # The tree's root when the lock was taken: a cache flush replaces the root
    # and drops every lock with the old tree.
    root: Any
    # Pulled tokens the lock keeps unevictable.
    num_tokens: int


class HostPrefixPuller:
    """Prefill side: recall a prompt prefix the decode holds in RAM into this
    engine's own host pages, for the requests of one bootstrap queue.

    The decode names the pushed prefix it holds (in PULL_REQ), this puller
    asks for the page range the prefill lacks (PREFIX_PULL), the decode
    registers those host pages (PREFIX_READY), and the prefill reads them into
    reserved host pages and inserts them as a host-only node before the request
    leaves the bootstrap queue. Normal admission then loads them into HBM.

    Every rank must make the same tree decisions, so an offer is acted on only
    in a scheduler step where every rank holds it, and the insert runs in the
    rank-reduced step that finalizes the bootstrap. Only full-attention KV
    is recalled; hybrid sliding-window models are refused at startup.
    """

    def __init__(
        self,
        *,
        kv_manager: Any,
        scheduler: Any,
    ) -> None:
        self._mgr = kv_manager
        self._tree_cache = scheduler.tree_cache
        self._page_size = int(scheduler.token_to_kv_pool_allocator.page_size)
        self._host_pool = _anchor_host_pool(self._tree_cache)
        self._mgr.bind_prefix_host_pool(self._host_pool)
        # Landing pages of in-flight recalls sit outside the tree until
        # commit, so they are capped to leave host room for the prefill's own
        # backups.
        self._max_inflight_tokens = int(
            _MAX_INFLIGHT_POOL_FRACTION * self._host_pool.size
        )
        self._groups = _rank_groups(scheduler)
        self._plans: Dict[int, HostPullPlan] = {}
        self._pins: Dict[int, _TreePin] = {}
        # Landings of dropped requests whose reads raiden may still be
        # writing, as (read ids, plan).
        self._zombies: List[Tuple[List[str], HostPullPlan]] = []
        # Set after an all-rank allocation; only an all-rank drain may clear
        # it, even when local completion/discard has released every page.
        self._external_reads_outstanding = False
        self.stats = {
            "offers": 0,
            "pulls": 0,
            "pulled_tokens": 0,
            "dedup_tokens": 0,
            "skipped_local_host": 0,
            "skipped_small": 0,
            "no_memory": 0,
            "over_budget": 0,
            "failed": 0,
        }

    # ----- bootstrap-queue hooks (scheduler thread) ----------------------------

    def advance(self, reqs: List[Any], *, waiting_reqs: Sequence[Any] = ()) -> None:
        """One scheduler step over the bootstrap queue, in queue order on every rank.

        Ranks exchange small all-reduces so that every tree decision (take an
        offer, plan the same range, insert, give up) is taken by all of them
        in the same step. ``waiting_reqs`` is the scheduler's waiting queue.
        """
        self.resolve_deferred_releases()
        self._unpin_departed(queued_reqs=[*reqs, *waiting_reqs])
        if not reqs:
            return
        rooms = [req.bootstrap_room for req in reqs]
        agreed = self._all_ranks_reduce(
            [self._local_flags(room) for room in rooms], dist.ReduceOp.MIN
        )
        planned = []
        ready_rooms = []
        for req, room, (offered, settled, done, kept, ready) in zip(
            reqs, rooms, agreed
        ):
            plan = self._plans.get(room)
            if not kept:
                # A rank settled the room with no plan (nothing worth
                # pulling): every rank gives up alike.
                if plan is not None:
                    self._abandon(room, plan)
                else:
                    self._mgr.resolve_prefix_pull(room)
                continue
            if offered:
                self._plan(req)
                planned.append(room)
            elif plan is not None and plan.state == "pulling" and settled:
                plan.agreed_done = bool(done)
                self._mgr.resolve_prefix_pull(room)
            elif plan is not None and plan.state == "asked" and ready:
                ready_rooms.append(room)
        if planned:
            self._agree_on_plans(planned)
        # Pages are allocated only once every rank holds the decode's READY, so
        # they sit idle for the read alone, and any eviction runs in step.
        for room in ready_rooms:
            plan = self._plans.get(room)
            if plan is not None and plan.new_indices is None:
                self._allocate(room, plan)
        for req in reqs:
            plan = self._plans.get(req.bootstrap_room)
            if (
                plan is not None
                and plan.state == "asked"
                and plan.new_indices is not None
            ):
                self._maybe_start(req, plan)

    def _local_flags(self, room: int) -> List[int]:
        """[offer to take, read settled, read done, not given up, READY in] on this rank."""
        plan = self._plans.get(room)
        offered = int(plan is None and self._mgr.prefix_offer_available(room))
        settled = done = 0
        if plan is not None and plan.state == "pulling" and plan.agreed_done is None:
            state = self._mgr.prefix_read_state(plan.req_id)
            settled = int(state != "pending")
            done = int(state == "done")
        kept = int(not (plan is None and self._mgr.prefix_pull_resolved(room)))
        ready = int(
            plan is not None
            and plan.state == "asked"
            and self._mgr.prefix_pull_ready(room) is not None
        )
        return [offered, settled, done, kept, ready]

    def _agree_on_plans(self, rooms: List[int]) -> None:
        """Keep a plan only if every rank planned the same page ranges."""
        local = []
        for room in rooms:
            plan = self._plans.get(room)
            local.append([1, plan.first_page, plan.num_pages] if plan else [0, -1, -1])
        lo = self._all_ranks_reduce(local, dist.ReduceOp.MIN)
        hi = self._all_ranks_reduce(local, dist.ReduceOp.MAX)
        for room, low, high in zip(rooms, lo, hi):
            if low == high and low[0] == 1:
                continue
            plan = self._plans.get(room)
            if plan is not None:
                logger.warning(
                    "[prefix-pull] room=%s ranks planned different pulls, giving up",
                    room,
                )
                self._abandon(room, plan)
            else:
                self._mgr.resolve_prefix_pull(room)

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
            self._page_size,
            reused_tokens=have,
        )
        if result.inserted_host_node is not None and not result.host_insert_dropped:
            self._pin(req=req, plan=plan, node=result.inserted_host_node)
            self.stats["pulls"] += 1
            self.stats["pulled_tokens"] += len(plan.new_indices)
            self.stats["dedup_tokens"] += max(0, result.prefix_len - have)
        logger.info(
            "[prefix-pull] room=%s pulled_tokens=%d have_tokens=%d "
            "inserted=%s wait_ms=%.0f read_ms=%.0f totals=%s",
            req.bootstrap_room,
            len(plan.new_indices),
            have,
            result.inserted_host_node is not None and not result.host_insert_dropped,
            (plan.started_at - plan.asked_at) * 1e3,
            (time.time() - plan.started_at) * 1e3,
            self.stats,
        )
        self._tree_cache.dec_host_lock_ref(plan.lock_node, plan.lock_receipt)

    def release(self, req: Any) -> None:
        """Drop a request's plan or tree lock when it leaves the bootstrap queue."""
        plan = self._plans.pop(req.bootstrap_room, None)
        if plan is not None:
            self._discard(req.bootstrap_room, plan)
        self._unpin(req.bootstrap_room)

    def release_all(self) -> None:
        for room, plan in list(self._plans.items()):
            self._discard(room, plan)
        self._plans.clear()
        self.resolve_deferred_releases()
        if self.has_pending_deferred_releases():
            raise RuntimeError(
                "Cannot release KV memory while prefix reads are in flight; retry after they drain"
            )
        for room in list(self._pins):
            self._unpin(room)

    def has_pending_deferred_releases(self) -> bool:
        """Rank-consistent ownership of recall pages outside normal admission."""
        return self._external_reads_outstanding

    def resolve_deferred_releases(self) -> None:
        """Poll and retire external read ownership in an all-rank scheduler step.

        The paused loop also calls this: an aborted last request leaves no
        sender to poll, but its read may still write the destination pages.
        """
        if not self._external_reads_outstanding:
            return
        self._mgr.drain()
        self._reap_zombies()
        held = bool(self._zombies) or any(
            plan.new_indices is not None for plan in self._plans.values()
        )
        self._external_reads_outstanding = bool(
            self._all_ranks_reduce([[int(held)]], dist.ReduceOp.MAX)[0][0]
        )

    def _all_ranks_reduce(self, rows: List[List[int]], op) -> List[List[int]]:
        return _all_ranks_reduce(self._groups, rows, op)

    # ----- steps ---------------------------------------------------------------

    def _plan(self, req: Any) -> None:
        room = req.bootstrap_room
        blocks = self._mgr.take_prefix_offer(room)
        if blocks is None:
            return
        self.stats["offers"] += 1
        page = self._page_size
        want = min(len(blocks), (len(req.origin_input_ids) - 1) // page)
        if not (want > 0 and want == len(blocks)):
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
        first = have // page
        count = want - first
        if count <= 0:
            self.stats["skipped_small"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        if self._inflight_tokens() + count * page > self._max_inflight_tokens:
            self.stats["over_budget"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        have_indices = []
        cursor = node
        while cursor is not core.root_node:
            have_indices.append(cursor.component_data[ComponentType.FULL].host_value)
            cursor = cursor.parent
        receipt = self._tree_cache.inc_host_lock_ref(node.id).to_dec_params()
        if not self._mgr.ask_prefix_pull(room, first, count):
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
        )

    def _radix_key(self, req: Any, num_tokens: int) -> RadixKey:
        """The tree key of the prompt's first ``num_tokens`` KV slots."""
        return prefix_radix_key(req=req, num_tokens=num_tokens)

    def _inflight_tokens(self) -> int:
        return (
            sum(p.num_pages * self._page_size for p in self._plans.values())
            + sum(p.num_tokens for p in self._pins.values())
            + sum(len(p.new_indices) for _, p in self._zombies)
        )

    def _allocate(self, room: int, plan: HostPullPlan) -> None:
        pool = self._host_pool
        tokens = plan.num_pages * self._page_size
        short = self._all_ranks_reduce(
            [[max(0, tokens - pool.available_size())]], dist.ReduceOp.MAX
        )[0][0]
        if short:
            freed = self._tree_cache.evict_host(short, ComponentType.FULL)
            if freed < short:
                reclaim_full_host_duplicates(self._tree_cache, short - freed)
        landing = pool.alloc(tokens)
        ok = self._all_ranks_reduce([[int(landing is not None)]], dist.ReduceOp.MIN)[0][
            0
        ]
        if not ok:
            if landing is not None:
                pool.free(landing)
            self.stats["no_memory"] += 1
            self._abandon(room, plan)
            return
        plan.new_indices = landing
        self._external_reads_outstanding = True

    def _maybe_start(self, req: Any, plan: HostPullPlan) -> None:
        room = req.bootstrap_room
        ready = self._mgr.prefix_pull_ready(room)
        if ready is None:
            return
        rid, uuid, endpoint = ready
        if not uuid:
            self.stats["failed"] += 1
            self._abandon(room, plan)
            return
        plan.req_id = rid
        plan.state, plan.started_at = "pulling", time.time()
        try:
            self._mgr.start_host_prefix_read(
                rid,
                uuid,
                endpoint,
                plan.remote,
                [int(p) for p in self._host_pool._to_page_ids(plan.new_indices)],
            )
        except Exception as exc:
            # The lifecycle resolves the other ranks on the next agreement step.
            self.stats["failed"] += 1
            logger.warning(
                "Host prefix recall for room %s could not start: %s", room, exc
            )
            self._abandon(room, plan)

    def _pin(self, *, req: Any, plan: HostPullPlan, node: Any) -> None:
        self._pins[req.bootstrap_room] = _TreePin(
            req=req,
            node=node,
            receipt=self._tree_cache.inc_host_lock_ref(node).to_dec_params(),
            root=self._tree_cache.root_node_handle(),
            num_tokens=plan.num_pages * self._page_size,
        )

    def _unpin(self, room: int) -> None:
        pin = self._pins.pop(room, None)
        if pin is not None and self._tree_cache.root_node_handle() == pin.root:
            self._tree_cache.dec_host_lock_ref(pin.node, pin.receipt)

    def _unpin_departed(self, *, queued_reqs: List[Any]) -> None:
        """Unlock the paths of requests no longer queued: admission locked the
        path for the request before it left the waiting queue, or it is gone.
        Queue membership is the same on every rank, so the unlock is in step."""
        if not self._pins:
            return
        queued = {id(req) for req in queued_reqs}
        for room, pin in list(self._pins.items()):
            if id(pin.req) not in queued:
                self._unpin(room)

    def _abandon(self, room: int, plan: HostPullPlan) -> None:
        self._plans.pop(room, None)
        self._discard(room, plan)
        self._mgr.resolve_prefix_pull(room)

    def _discard(self, room: int, plan: HostPullPlan) -> None:
        self._tree_cache.dec_host_lock_ref(plan.lock_node, plan.lock_receipt)
        if plan.new_indices is None:
            return
        if self._any_read_pending(plan.read_ids()):
            self._zombies.append((plan.read_ids(), plan))
        else:
            self._free_plan(plan)

    def _free_plan(self, plan: HostPullPlan) -> None:
        self._host_pool.free(plan.new_indices)

    def _any_read_pending(self, read_ids: List[str]) -> bool:
        return any(self._mgr.prefix_read_state(rid) == "pending" for rid in read_ids)

    def _reap_zombies(self) -> None:
        keep = []
        for ids, plan in self._zombies:
            if self._any_read_pending(ids):
                keep.append((ids, plan))
            else:
                self._free_plan(plan)
        self._zombies = keep


class HostPrefixSource:
    """Keep host offers pinned until queue ownership and native reads end."""

    def __init__(self, queue: Any, manager: Any) -> None:
        self.cache, self.manager = queue.tree_cache, manager
        self.page_size = int(queue.scheduler.token_to_kv_pool_allocator.page_size)
        self.full = _anchor_host_pool(self.cache)
        manager.bind_prefix_host_pool(self.full)
        self.offers: dict[int, tuple[Any, Any]] = {}
        self.released: set[int] = set()
        self.groups = _rank_groups(queue.scheduler)

    def _reduce(self, rows: List[List[int]], op: Any) -> List[List[int]]:
        return _all_ranks_reduce(self.groups, rows, op)

    def offer(self, decode_req: Any) -> None:
        req = decode_req.req
        if req.bootstrap_room in self.offers:
            return
        core = self.cache.tree_core
        limit = offer_limit(req=req, page_size=self.page_size)
        key = prefix_radix_key(req=req, num_tokens=limit)
        # Matching can split radix nodes but never starts a load-back.
        match = self.cache.match_prefix(MatchPrefixParams(key=key))
        node = core.node_by_id(match.best_match_node)
        # The deepest matched prefix whose full KV is whole in host RAM.
        full = None
        while node is not core.root_node:
            full = _host_pages(core.root_node, self.full, node)
            if full:
                break
            node = node.parent
        tokens = len(full) * self.page_size if full else 0
        low = self._reduce([[tokens]], dist.ReduceOp.MIN)
        high = self._reduce([[tokens]], dist.ReduceOp.MAX)
        if low != high or not low[0][0]:
            return
        receipt = self.cache.inc_host_lock_ref(node.id).to_dec_params()
        self.offers[req.bootstrap_room] = (node.id, receipt)
        self.manager.offer_resident_prefix(req.bootstrap_room, full)

    def release(self, room: int) -> None:
        # Close before releasing the tree pin: a late PULL on the listener
        # must not register pages after the queue has relinquished them.
        self.manager.release_prefix_offer(room)
        if room in self.offers:
            self.released.add(room)

    def step(self) -> None:
        self.manager.drain()
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


def offer_host_prefix(queue: Any, decode_req: Any) -> None:
    """Pin and offer the host-held prefix of a request whose PULL_REQ is about
    to be sent. ``queue`` is the transfer queue that will own the request."""
    source = _sources.get(queue)
    if source is None:
        manager = (
            queue.scheduler.disagg_decode_prealloc_queue.kv_manager.decode_dram_offload
        )
        source = _sources[queue] = HostPrefixSource(queue, manager)
    source.offer(decode_req)


def release_departed_host_offers(queue: Any) -> None:
    """Release the offers of requests that left the transfer queue.

    Queue membership and the deferred releases change in steps every rank
    agrees on, so every rank releases the same rooms in the same step. The
    pages stay pinned until the prefill's native read settles (see
    ``HostPrefixSource.step``).
    """
    source = _sources.get(queue)
    if source is None or len(source.offers) == len(source.released):
        return
    owned = {dr.req.bootstrap_room for dr in queue.queue}
    owned.update(entry[0].req.bootstrap_room for entry in queue._deferred_releases)
    for room in list(source.offers):
        if room not in owned and room not in source.released:
            source.release(room)


def progress_host_offers(queue: Any) -> None:
    source = _sources.get(queue)
    if source is not None:
        source.step()


def host_offers_pending(queue: Any) -> bool:
    source = _sources.get(queue)
    return source is not None and bool(source.offers)


def reset_host_offers(cache: Any) -> None:
    for queue, source in list(_sources.items()):
        if source.cache is cache:
            release_departed_host_offers(queue)
            source.step()
            if source.offers:
                raise RuntimeError(
                    "Cannot reset KV memory while host prefix offers are pinned"
                )
