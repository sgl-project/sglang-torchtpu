"""Prefill-side orchestration of the decode-held prefix pull.

A PD decode matches a request's prompt against its own prefix cache (device
and host RAM) before it asks the prefill for KV, so it can hold a longer
prefix than the prefill does. The decode names those pages (PREFIX_OFFER),
this puller asks for the page range the prefill lacks (PREFIX_PULL), the
decode registers them (PREFIX_READY), and the prefill reads them into freshly
allocated pages and inserts them into its radix tree before the request
leaves the bootstrap queue, so the forward computes only the tail. The pair's
effective prefix cache is then the union of both engines' caches.

Every rank must make the same tree decisions, so an offer is acted on only in
a scheduler step where every rank holds it, and the insert runs in the
rank-reduced step that finalizes the bootstrap.

On hybrid sliding-window models the prefill can resume at the offered length M
only with the window ending at M too (see prefix_offer.py), so it takes M or
nothing: full KV [have, M) on the full engine and window KV
[max(have, M - window), M) on the SWA engine, inserted with the SWA slots
before the window marked evicted.

The inserted path stays locked from the insert until the request has left the
bootstrap and waiting queues: by then admission matched it and took its own
lock, or the request is gone. Unlocked in between, the small SWA pool's LRU or
a full eviction could drop the pulled pages before the request runs.
"""

from __future__ import annotations

import logging
import time
from array import array
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

logger = logging.getLogger(__name__)


class PrefixPullPlan(msgspec.Struct):
    """One request's pull: the offered page range asked for and what it joins."""

    # Full pages read from the decode: [first_page, +num_pages), ending at M.
    first_page: int
    num_pages: int
    # The decode's page ids of the asked range.
    remote: List[int]
    # KV indices of the prefill's own prefix [0, first_page) the pulled
    # pages join (locked until commit).
    have_indices: Any
    # KV indices allocated for pages [first_page, M), None until every
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
    # Hybrid sliding-window models: window pages [swa_first_page,
    # +swa_num_pages), ending where the full range ends, and the decode's ids
    # of them. No pages on full-only models.
    swa_first_page: int = 0
    swa_num_pages: int = 0
    swa_remote: List[int] = []
    swa_req_id: Optional[str] = None

    def read_ids(self) -> List[str]:
        return [rid for rid in (self.req_id, self.swa_req_id) if rid is not None]

    def end_page(self) -> int:
        return self.first_page + self.num_pages

    def alloc_pages(self) -> int:
        return self.num_pages


class _TreePin(msgspec.Struct, frozen=True):
    """A committed pull's tree lock, held until its request leaves the queues."""

    req: Any
    node: Any
    receipt: Any
    # The tree's root when the lock was taken: a cache flush replaces the root
    # and drops every lock with the old tree.
    root: Any
    # Pulled tokens the lock keeps unevictable, full and window.
    num_tokens: int
    swa_tokens: int


class DecodePrefixPuller:
    """Drives the prefix pull for the requests of one prefill bootstrap queue."""

    def __init__(
        self,
        *,
        kv_manager: RaidenKVManager,
        scheduler,
        swa_window: Optional[int] = None,
    ) -> None:
        self._mgr = kv_manager
        self._tree_cache = scheduler.tree_cache
        self._allocator = scheduler.token_to_kv_pool_allocator
        # Tree values follow the req_to_token table's device (host-resident on
        # TPU). The allocator hands out device-resident indices, so the insert
        # moves them there or the tree concatenates mixed devices.
        self._tree_device = scheduler.req_to_token_pool.req_to_token.device
        self._page_size = int(self._allocator.page_size)
        min_tokens = tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MIN_TOKENS.get()
        self._min_pages = max(1, -(-min_tokens // self._page_size))
        # Hybrid sliding-window models: the window in whole pages. None on
        # full-only models.
        self._window_pages = (
            None if swa_window is None else -(-swa_window // self._page_size)
        )
        hybrid = self._window_pages is not None
        pool_tokens = self._allocator.size_full if hybrid else self._allocator.size
        swa_pool_tokens = self._allocator.size_swa if hybrid else 0
        # An EAGLE tree keys token pairs, so N KV slots need N + 1 tokens.
        self._is_eagle = getattr(
            getattr(self._tree_cache, "tree_core", self._tree_cache), "is_eagle", False
        )
        # Destination pages of in-flight pulls sit outside the scheduler's
        # accounting until commit, so they are capped to leave room for admission.
        pool_fraction = tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MAX_POOL_FRACTION.get()
        self._max_inflight_tokens = int(pool_fraction * pool_tokens)
        self._max_inflight_swa_tokens = int(pool_fraction * swa_pool_tokens)
        free_fraction = tpu_envs.SGLANG_DISAGG_PREFIX_PULL_MIN_FREE_FRACTION.get()
        self._min_free_tokens = int(free_fraction * pool_tokens)
        self._min_free_swa_tokens = int(free_fraction * swa_pool_tokens)
        self._groups = [
            group
            for group in (scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group)
            if group is not None and dist.get_world_size(group) > 1
        ]
        self._plans: Dict[int, PrefixPullPlan] = {}
        self._pins: Dict[int, _TreePin] = {}
        # Pages of dropped requests whose reads raiden may still be landing, as
        # (read ids, full KV indices, window tokens).
        self._zombies: List[Tuple[List[str], Any, int]] = []
        # Set after an all-rank allocation; only an all-rank drain may clear
        # it, even when local completion/discard has released every page.
        self._external_reads_outstanding = False
        self.stats = {
            "offers": 0,
            "pulls": 0,
            "pulled_tokens": 0,
            "dedup_tokens": 0,
            "skipped_have": 0,
            "skipped_local_host": 0,
            "skipped_small": 0,
            "no_memory": 0,
            "over_budget": 0,
            "failed": 0,
        }

    # ----- bootstrap-queue hooks (scheduler thread) ----------------------------

    def advance(self, reqs: List[Req], *, waiting_reqs: Sequence[Req] = ()) -> None:
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
                # A rank settled the room with no plan (withdrawn offer or
                # nothing worth pulling): every rank gives up alike.
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
            # A hybrid pull is settled once both its reads are, done if both are.
            states = [self._mgr.prefix_read_state(rid) for rid in plan.read_ids()]
            settled = int("pending" not in states)
            done = int(all(state == "done" for state in states))
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
            row = [1, plan.first_page, plan.num_pages] if plan else [0, -1, -1]
            if self._window_pages is not None:
                row += [plan.swa_first_page, plan.swa_num_pages] if plan else [-1, -1]
            local.append(row)
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

    def commit(self, req: Req) -> None:
        """Insert the pulled pages into the tree. The request leaves bootstrap next."""
        room = req.bootstrap_room
        plan = self._plans.pop(room, None)
        if plan is None:
            return
        if plan.state != "pulling" or not plan.agreed_done:
            self.stats["failed"] += 1
            logger.warning(
                "[prefix-pull] room=%s not completed (plan=%s agreed_done=%s), "
                "computing the prompt",
                room,
                plan.state,
                plan.agreed_done,
            )
            self._discard(room, plan)
            return
        page = self._page_size
        have_tokens = plan.first_page * page
        total_tokens = plan.end_page() * page
        value = torch.cat(
            [
                plan.have_indices.to(self._tree_device),
                plan.new_indices.to(self._tree_device),
            ]
        ).to(torch.int64)
        # The allocation bound window slots only from swa_first_page on.
        evicted = (
            {ComponentType.SWA: plan.swa_first_page * page}
            if self._window_pages is not None
            else {}
        )
        # prev_prefix_len keeps the tree's own pages of [0, have) from being
        # freed as duplicates. Pulled pages that duplicate nodes the tree gained
        # meanwhile are freed by the insert. inserted_len covers the pulled range
        # so it counts no hit: write-through then backs it up to this engine's
        # host tier only once the prefill reuses it, since the decode holds a copy.
        result = self._tree_cache.insert(
            InsertParams(
                key=self._radix_key(req, total_tokens),
                value=value,
                prev_prefix_len=have_tokens,
                inserted_len=total_tokens,
                component_evicted_seqlens=evicted,
            )
        )
        # Taken before the matched prefix's lock goes, so [0, M) is never unlocked.
        self._pin(req=req, plan=plan, node=result.last_device_node)
        self._tree_cache.dec_lock_ref(plan.lock_node, plan.lock_receipt)
        # Tree nodes the insert walked past the pulled range start: KV the tree
        # gained while the pages were read.
        dedup_tokens = max(0, result.prefix_len - have_tokens)
        stats = self.stats
        stats["pulls"] += 1
        stats["pulled_tokens"] += total_tokens - have_tokens
        stats["dedup_tokens"] += dedup_tokens
        logger.info(
            "[prefix-pull] room=%s pulled_tokens=%d have_tokens=%d dedup_tokens=%d "
            "swa_pulled_tokens=%d wait_ms=%.0f read_ms=%.0f totals=%s",
            room,
            total_tokens - have_tokens,
            have_tokens,
            dedup_tokens,
            plan.swa_num_pages * page,
            (plan.started_at - plan.asked_at) * 1e3,
            (time.time() - plan.started_at) * 1e3,
            stats,
        )

    def release(self, req: Req) -> None:
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
        self._mgr._drain_engine()
        self._reap_zombies()
        held = bool(self._zombies) or any(
            plan.new_indices is not None for plan in self._plans.values()
        )
        self._external_reads_outstanding = bool(
            self._all_ranks_reduce([[int(held)]], dist.ReduceOp.MAX)[0][0]
        )

    # ----- steps ---------------------------------------------------------------

    def _all_ranks_reduce(self, rows: List[List[int]], op) -> List[List[int]]:
        if not self._groups or not rows:
            return [list(row) for row in rows]
        values = torch.tensor(rows, dtype=torch.int64)
        for group in self._groups:
            dist.all_reduce(values, op=op, group=group)
        return values.tolist()

    def _plan(self, req: Req) -> None:
        room = req.bootstrap_room
        offer = self._mgr.take_prefix_offer(room)
        if offer is None:
            return
        self.stats["offers"] += 1
        prefix_tokens, blocks = offer
        page = self._page_size
        swa_offer = None
        if self._window_pages is None:
            want_pages = min(
                prefix_tokens // page, len(blocks), len(req.origin_input_ids) // page
            )
        else:
            want_pages, swa_offer = self._window_offer(req, prefix_tokens, blocks)
        if want_pages <= 0:
            self._mgr.resolve_prefix_pull(room)
            return
        match = self._tree_cache.match_prefix(
            MatchPrefixParams(key=self._radix_key(req, want_pages * page))
        )
        have_pages = len(match.device_indices) // page
        need_pages = want_pages - have_pages
        if need_pages < self._min_pages:
            self.stats["skipped_have" if need_pages <= 0 else "skipped_small"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        # The prefill's own host tier covers the rest: a local load-back is
        # cheaper than the network, and the scheduler does it on admission.
        local_pages = (len(match.device_indices) + match.host_hit_length) // page
        if want_pages - local_pages < self._min_pages:
            self.stats["skipped_local_host"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        swa_first_page = swa_num_pages = 0
        swa_remote: List[int] = []
        if swa_offer is not None:
            offered_first, swa_blocks = swa_offer
            swa_first_page = max(have_pages, want_pages - self._window_pages)
            swa_num_pages = want_pages - swa_first_page
            swa_remote = list(swa_blocks[swa_first_page - offered_first :])
        first_page = have_pages
        lock_node = match.last_device_node
        num_pages = want_pages - first_page
        if (
            self._inflight_tokens() + num_pages * page > self._max_inflight_tokens
            or self._inflight_swa_tokens() + swa_num_pages * page
            > self._max_inflight_swa_tokens
        ):
            self.stats["over_budget"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        # Locked until commit: the destination allocation evicts any unlocked
        # leaf, the local prefix included.
        receipt = self._tree_cache.inc_lock_ref(lock_node).to_dec_params()
        if not self._mgr.ask_prefix_pull(
            room,
            first_page,
            num_pages,
            swa_first_page=swa_first_page,
            swa_num_pages=swa_num_pages,
        ):
            self._tree_cache.dec_lock_ref(lock_node, receipt)
            self.stats["failed"] += 1
            self._mgr.resolve_prefix_pull(room)
            return
        self._plans[room] = PrefixPullPlan(
            first_page=first_page,
            num_pages=num_pages,
            remote=list(blocks[first_page:want_pages]),
            have_indices=match.device_indices[: first_page * page],
            new_indices=None,
            lock_node=lock_node,
            lock_receipt=receipt,
            asked_at=time.time(),
            swa_first_page=swa_first_page,
            swa_num_pages=swa_num_pages,
            swa_remote=swa_remote,
        )

    def _radix_key(self, req: Req, num_tokens: int) -> RadixKey:
        """The tree key of the prompt's first ``num_tokens`` KV slots."""
        return RadixKey(
            token_ids=array(
                "q", req.origin_input_ids[: num_tokens + int(self._is_eagle)]
            ),
            extra_key=req.extra_key,
            is_bigram=self._is_eagle,
            cache_salt=req.cache_salt,
        )

    def _window_offer(
        self, req: Req, prefix_tokens: int, blocks: List[int]
    ) -> Tuple[int, Optional[Tuple[int, List[int]]]]:
        """A hybrid offer's pages and its (first page, ids) window, or (0, None)
        unless the window ends the offer and the prefill can resume there."""
        page = self._page_size
        want_pages = len(blocks)
        swa_first, swa_blocks = self._mgr.prefix_swa_offer(req.bootstrap_room)
        usable = (
            prefix_tokens == want_pages * page
            # The prefill computes at least the last prompt token.
            and want_pages <= (len(req.origin_input_ids) - 1) // page
            and swa_first == max(0, want_pages - self._window_pages)
            and swa_first + len(swa_blocks) == want_pages
        )
        return (want_pages, (swa_first, swa_blocks)) if usable else (0, None)

    def _inflight_tokens(self) -> int:
        """Device tokens reserved by pulls not yet admitted, zombies included."""
        held = sum(
            plan.alloc_pages() * self._page_size for plan in self._plans.values()
        )
        pinned = sum(pin.num_tokens for pin in self._pins.values())
        return held + pinned + sum(len(indices) for _, indices, _ in self._zombies)

    def _inflight_swa_tokens(self) -> int:
        """Window tokens reserved by pulls not yet admitted, zombies included."""
        held = sum(
            plan.swa_num_pages * self._page_size for plan in self._plans.values()
        )
        pinned = sum(pin.swa_tokens for pin in self._pins.values())
        return held + pinned + sum(swa_tokens for _, _, swa_tokens in self._zombies)

    def _allocate(self, room: int, plan: PrefixPullPlan) -> None:
        # Pages taken here bypass admission. Keep a reserve it can still use.
        # Native completion can retire abandoned destinations on just one
        # rank. Both the reserve decision and cache evictions must agree.
        new_indices = None
        fits = self._all_ranks_reduce(
            [[int(self._fits_with_reserve(plan))]], dist.ReduceOp.MIN
        )[0][0]
        if fits:
            new_indices = self._alloc(plan)
        allocated = self._all_ranks_reduce(
            [[int(new_indices is not None)]], dist.ReduceOp.MIN
        )[0][0]
        if not allocated:
            if new_indices is not None:
                self._allocator.free(new_indices)
            self.stats["no_memory"] += 1
            logger.info(
                "[prefix-pull] room=%s no device memory for %d pages, computing",
                room,
                plan.alloc_pages(),
            )
            self._abandon(room, plan)
            return
        plan.new_indices = new_indices
        self._external_reads_outstanding = True

    def _fits_with_reserve(self, plan: PrefixPullPlan) -> bool:
        num_tokens = plan.alloc_pages() * self._page_size
        if self._window_pages is None:
            reclaimable = (
                self._allocator.available_size()
                + self._tree_cache.full_evictable_size()
            )
            return reclaimable - num_tokens >= self._min_free_tokens
        swa_tokens = plan.swa_num_pages * self._page_size
        full_reclaimable = (
            self._allocator.full_available_size()
            + self._tree_cache.full_evictable_size()
        )
        swa_reclaimable = (
            self._allocator.swa_available_size() + self._tree_cache.swa_evictable_size()
        )
        return (
            full_reclaimable - num_tokens >= self._min_free_tokens
            and swa_reclaimable - swa_tokens >= self._min_free_swa_tokens
        )

    def _alloc(self, plan: PrefixPullPlan) -> Optional[torch.Tensor]:
        num_tokens = plan.alloc_pages() * self._page_size
        if self._window_pages is None:
            short = self._all_ranks_reduce(
                [[max(0, num_tokens - self._allocator.available_size())]],
                dist.ReduceOp.MAX,
            )[0][0]
            if short > 0:
                self._tree_cache.evict(EvictParams(num_tokens=short))
            return self._allocator.alloc(num_tokens)
        swa_tokens = plan.swa_num_pages * self._page_size
        full_short, swa_short = self._all_ranks_reduce(
            [
                [
                    max(0, num_tokens - self._allocator.full_available_size()),
                    max(0, swa_tokens - self._allocator.swa_available_size()),
                ]
            ],
            dist.ReduceOp.MAX,
        )[0]
        if full_short or swa_short:
            self._tree_cache.evict_for_alloc(
                EvictParams(num_tokens=full_short, swa_num_tokens=swa_short)
            )
        # Full slots for [have, M), window slots bound only for the tail
        # [swa_first_page, M): what the insert's SWA evicted length expects.
        device = self._allocator.device
        have_tokens = plan.first_page * self._page_size
        seq_len = have_tokens + num_tokens
        last_loc = (
            plan.have_indices[-1:].to(dtype=torch.int64, device=device)
            if have_tokens > 0
            else torch.tensor([-1], dtype=torch.int64, device=device)
        )
        return self._allocator.alloc_extend_swa_tail(
            prefix_lens=torch.tensor([have_tokens], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([have_tokens], dtype=torch.int64),
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            last_loc=last_loc,
            extend_num_tokens=num_tokens,
            swa_tail_len=swa_tokens,
        )

    def _maybe_start(self, req: Req, plan: PrefixPullPlan) -> None:
        room = req.bootstrap_room
        ready = self._mgr.prefix_pull_ready(room)
        if ready is None:
            return
        req_id, uuid, endpoint, first_page, num_pages = ready
        swa_ready = ("", 0, "", 0, 0)
        if self._window_pages is not None:
            swa_ready = self._mgr.prefix_swa_ready(room) or ("", 0, "", -1, -1)
        swa_req_id, swa_uuid, swa_endpoint, swa_first_page, swa_num_pages = swa_ready
        served = (first_page, num_pages, swa_first_page, swa_num_pages)
        asked = (
            plan.first_page,
            plan.num_pages,
            plan.swa_first_page,
            plan.swa_num_pages,
        )
        if served != asked:
            logger.error(
                "[prefix-pull] room=%s READY covers pages [%d, +%d) window "
                "[%d, +%d) but [%d, +%d) window [%d, +%d) was asked, computing "
                "the prompt",
                room,
                *served,
                *asked,
            )
            self._abandon(room, plan)
            return
        page = self._page_size
        local = [int(x) for x in kv_to_page_indices(plan.new_indices, page)]
        outcome = self._mgr.start_prefix_pull(
            room,
            req_id,
            uuid,
            endpoint,
            plan.remote,
            local,
            swa_req_id=swa_req_id,
            swa_uuid=swa_uuid,
            swa_endpoint=swa_endpoint,
            swa_remote=plan.swa_remote,
            swa_local=self._window_local_pages(plan),
        )
        if outcome == "started":
            plan.state = "pulling"
            plan.req_id = req_id
            plan.swa_req_id = swa_req_id or None
            plan.started_at = time.time()
        elif outcome == "failed":
            self.stats["failed"] += 1
            self._abandon(room, plan)
        # "retry": no host staging free yet. The next step asks again.

    def _window_local_pages(self, plan: PrefixPullPlan) -> List[int]:
        """SWA-pool page ids the allocation bound for the window pages."""
        if plan.swa_num_pages == 0:
            return []
        offset = (plan.swa_first_page - plan.first_page) * self._page_size
        swa = self._allocator.translate_swa_indices_for_transfer(
            plan.new_indices[offset:]
        )
        return [int(x) for x in kv_to_page_indices(swa, self._page_size)]

    def _pin(self, *, req: Req, plan: PrefixPullPlan, node: Any) -> None:
        receipt = self._tree_cache.inc_lock_ref(node).to_dec_params()
        self._pins[req.bootstrap_room] = _TreePin(
            req=req,
            node=node,
            receipt=receipt,
            root=self._tree_cache.root_node_handle(),
            num_tokens=plan.alloc_pages() * self._page_size,
            swa_tokens=plan.swa_num_pages * self._page_size,
        )

    def _unpin(self, room: int) -> None:
        pin = self._pins.pop(room, None)
        if pin is not None and self._tree_cache.root_node_handle() == pin.root:
            self._tree_cache.dec_lock_ref(pin.node, pin.receipt)

    def _unpin_departed(self, *, queued_reqs: List[Req]) -> None:
        """Unlock the paths of requests no longer queued: admission locked the
        path for the request before it left the waiting queue, or it is gone.
        Queue membership is the same on every rank, so the unlock is in step."""
        if not self._pins:
            return
        queued = {id(req) for req in queued_reqs}
        for room, pin in list(self._pins.items()):
            if id(pin.req) not in queued:
                self._unpin(room)

    def _abandon(self, room: int, plan: PrefixPullPlan) -> None:
        self._plans.pop(room, None)
        self._discard(room, plan)
        self._mgr.resolve_prefix_pull(room)

    def _discard(self, room: int, plan: PrefixPullPlan) -> None:
        self._tree_cache.dec_lock_ref(plan.lock_node, plan.lock_receipt)
        # A read that never got host staging must not hold the line for others.
        self._mgr.forget_prefix_pull(room)
        if plan.new_indices is None:
            return
        read_ids = plan.read_ids()
        if self._any_read_pending(read_ids):
            # Raiden cannot cancel a read, so the pages stay allocated until
            # every read settles.
            swa_tokens = plan.swa_num_pages * self._page_size
            self._zombies.append((read_ids, plan.new_indices, swa_tokens))
            return
        self._allocator.free(plan.new_indices)

    def _any_read_pending(self, read_ids: List[str]) -> bool:
        return any(self._mgr.prefix_read_state(rid) == "pending" for rid in read_ids)

    def _reap_zombies(self) -> None:
        if not self._zombies:
            return
        keep = []
        for zombie in self._zombies:
            read_ids, indices, _ = zombie
            if self._any_read_pending(read_ids):
                keep.append(zombie)
            else:
                self._allocator.free(indices)
        self._zombies = keep
