"""Prefill side of the recall: pull a prefix the decode holds in its DRAM into
the prefill's own DRAM (HiCache host pool), with the sliding window that ends
where the prefix ends on a hybrid model.

The decode names the pushed prefix it holds in its PULL_REQ
(decode_dram_prefix_offerer.py); the prefill asks for the page range it lacks
(PREFIX_PULL), the decode registers those host pages (PREFIX_READY), and the
prefill reads them into reserved host pages and inserts them as a host-only
node before the request leaves the bootstrap queue. Only normal prefill
admission loads these pages into HBM, so the recall uses no device page and no
staging slot on either role.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    log_offload_stats,
)
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    host_backed_prefix,
    host_page_ids,
    insert_host_prefix,
    lock_host_path,
    offer_limit,
    prefix_radix_key,
    reclaim_full_host_duplicates,
    unlock_host_path,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    alloc_host_pages_agreed,
)
from sglang_tpu.disaggregation.raiden.dram_offload_channel import (
    PrefixRegistration,
)

# Arbitrary: the share of the prefill's host pool that landings of in-flight
# recalls may hold, leaving the rest for the prefill's own backups.
_MAX_INFLIGHT_POOL_FRACTION = 0.5


class _PullProposal(msgspec.Struct, frozen=True):
    """What one rank could pull for a request, before the ranks agree."""

    # The decode's page ids of the whole offered prefix, and of its window.
    blocks: List[int]
    swa_blocks: List[int]
    # This rank's own host-backed prefix: its last node, its pages and its
    # host KV indices per node.
    node: Any
    have_pages: int
    have_indices: List[Any]


class DramPrefixPullPlan(msgspec.Struct):
    """One request's pull: the offered page range asked for and what it joins."""

    # Full pages read from the decode: [first_page, +num_pages), ending at M.
    first_page: int
    num_pages: int
    # The decode's page ids of the asked range.
    remote: List[int]
    # Host KV indices of the prefill's own prefix [0, first_page) the pulled
    # pages join; ``locks`` host-lock that whole path until commit.
    have_indices: Any
    locks: List[Any]
    # Host KV indices reserved for pages [first_page, M), None until every
    # rank has READY.
    new_indices: Any = None
    # Hybrid sliding-window models: the decode's page ids of the window that
    # ends at M, and the host pages reserved for it.
    swa_remote: List[int] = msgspec.field(default_factory=list)
    swa_indices: Any = None
    # The native reads that started; ``started`` once the decode's READY was
    # acted on, while the pull is only asked for before.
    read_ids: List[str] = msgspec.field(default_factory=list)
    started: bool = False
    # Set once every rank saw the read settle: True only if it succeeded everywhere.
    agreed_done: Optional[bool] = None


class _RecalledPrefixPin(msgspec.Struct, frozen=True):
    """A committed pull's tree lock, held until its request leaves the queues."""

    req: Any
    node: Any
    receipt: Any
    # Pulled tokens the lock keeps unevictable, full and window.
    num_tokens: int
    swa_tokens: int


class PrefillDramPrefixPuller:
    """Prefill side: recall a prompt prefix the decode holds in RAM into this
    engine's own host pages, for the requests of one bootstrap queue.

    Every rank must make the same tree decisions, so each step of a recall
    (ask, allocate, start, insert, give up) is taken by all ranks in the same
    scheduler step, after they reduced their local view. A hybrid
    sliding-window model's recall also reads the window that ends where the
    offered prefix ends, into the SWA host pool.
    """

    def __init__(self, ctx: DramOffloadWorkerContext) -> None:
        self._channel = ctx.channel
        self._cache = ctx.cache
        self._page_size = ctx.page_size
        self._host_pool = ctx.host_pool
        self._swa_host_pool = ctx.swa_host_pool
        self._window_pages = ctx.window_tokens // ctx.page_size
        self._lockstep = ctx.lockstep
        # Landing pages of in-flight recalls sit outside the tree until
        # commit, so they are capped to leave host room for the prefill's own
        # backups.
        self._max_inflight_tokens = int(
            _MAX_INFLIGHT_POOL_FRACTION * self._host_pool.size
        )
        self._max_inflight_swa_tokens = (
            int(_MAX_INFLIGHT_POOL_FRACTION * self._swa_host_pool.size)
            if self._swa_host_pool is not None
            else 0
        )
        self._plans: Dict[int, DramPrefixPullPlan] = {}
        # Offers taken off the channel, until every rank holds the room's:
        # the listener thread can drop a room's channel state at any time.
        self._offered: Dict[int, Tuple[List[int], List[int]]] = {}
        self._pins: Dict[int, _RecalledPrefixPin] = {}
        # Plans of dropped requests, in the same order on every rank: their
        # landings are freed once no rank's read can still write them.
        self._zombies: List[DramPrefixPullPlan] = []
        self.stats = dict.fromkeys(
            (
                "offers",
                "pulls",
                "pulled_tokens",
                "pulled_swa_tokens",
                "skipped_local_host",
                "no_memory",
                "over_budget",
                "adopted_backup_tokens",
                "failed",
            ),
            0,
        )
        self._last_log = [0.0]

    # ----- bootstrap-queue hooks (scheduler thread) ----------------------------

    def advance(self, reqs: List[Any], *, waiting_reqs: Sequence[Any]) -> None:
        """One scheduler step over the bootstrap queue, in queue order on every
        rank. ``waiting_reqs`` is the scheduler's waiting queue."""
        self.free_settled_host_landings()
        self._unpin_departed(queued_reqs=[*reqs, *waiting_reqs])
        log_offload_stats(self.stats, "recall", self._last_log)
        if not reqs:
            return
        rooms = [req.bootstrap_room for req in reqs]
        agreed = self._lockstep.reduce(
            [self._local_recall_flags(room) for room in rooms], dist.ReduceOp.MIN
        )
        proposals = {}
        ready = []
        for req, room, (offered, settled, done, ready_in) in zip(reqs, rooms, agreed):
            plan = self._plans.get(room)
            if offered:
                proposals[room] = self._propose_pull(req)
            elif settled and not plan.started:
                # A refused READY: given up before any landing page is
                # reserved, so a refusal evicts nothing from host RAM.
                self.stats["failed"] += 1
                self._abandon_pull(room)
            elif settled:
                plan.agreed_done = bool(done)
                self._channel.resolve_prefix_pull(room)
            elif ready_in:
                ready.append(room)
        if proposals:
            self._ask_decode_for_agreed_ranges(proposals)
        if ready:
            self._start_agreed_reads(ready)

    def _local_recall_flags(self, room: int) -> List[int]:
        """[offer to take, read settled, read done, READY in] on this rank."""
        plan = self._plans.get(room)
        if plan is None:
            if room not in self._offered:
                offer = self._channel.take_prefix_offer(room)
                if offer is not None:
                    self._offered[room] = offer
            return [int(room in self._offered), 0, 0, 0]
        if not plan.started:
            ready = self._channel.prefix_ready(room)
            if ready is not None and ready.refused:
                # A refused pull is settled without a read.
                return [0, 1, 0, 0]
            return [0, 0, 0, int(ready is not None)]
        if plan.agreed_done is not None:
            return [0, 0, 0, 0]
        # A hybrid pull is settled once both its reads are, done if both are.
        done = self._channel.poll_host_reads(plan.read_ids)
        return [0, int(done is not None), int(done is True), 0]

    def _propose_pull(self, req: Any) -> Optional[_PullProposal]:
        """What this rank could pull of the offer every rank holds; None when
        it needs nothing. Matching can split a node, so every rank matches."""
        blocks, swa_blocks = self._offered.pop(req.bootstrap_room)
        self.stats["offers"] += 1
        page = self._page_size
        tokens = len(blocks) * page
        # The prefill computes at least the last prompt token.
        if not 0 < tokens <= offer_limit(req=req, page_size=page):
            return None
        if not self._window_fits_offer(blocks, swa_blocks):
            return None
        key = prefix_radix_key(req=req, num_tokens=tokens)
        match = self._cache.match_prefix(MatchPrefixParams(key=key))
        # Reuse the part of the prefix this engine already holds in host RAM.
        node, have, have_indices = host_backed_prefix(self._cache, key, tokens, page)
        covered = len(match.device_indices) + match.host_hit_length >= tokens
        if covered or (have >= tokens and not swa_blocks):
            self.stats["skipped_local_host"] += 1
            return None
        if have >= tokens:
            # The full KV is here and only the window is missing: read the
            # full pages again with it, and the insert frees the duplicates.
            node, have, have_indices = self._cache.tree_core.root_node, 0, []
        return _PullProposal(
            blocks=blocks,
            swa_blocks=swa_blocks,
            node=node,
            have_pages=have // page,
            have_indices=have_indices,
        )

    def _window_fits_offer(self, blocks: List[int], swa_blocks: List[int]) -> bool:
        """A hybrid model resumes at the offered length only with the window
        that ends there: the offer must carry one whole window (or the whole
        prefix when that is shorter). A full-only model carries no window."""
        return len(swa_blocks) == min(len(blocks), self._window_pages)

    def _ask_decode_for_agreed_ranges(
        self, proposals: Dict[int, Optional[_PullProposal]]
    ) -> None:
        """Ask the decode for a room's pages only once every rank agreed on
        the range: the offered length, and the shortest local prefix of any
        rank as its start, so that every rank reads the same pages. A rank
        that holds more reads pages it has; the insert frees those."""
        page = self._page_size
        budget = (self._max_inflight_tokens - self._inflight_tokens()) // page
        swa_budget = (
            self._max_inflight_swa_tokens - self._inflight_swa_tokens()
        ) // page
        # MIN of a value and of its negation gives its minimum and maximum.
        rows = [
            [0, 0, 0, 0]
            if p is None
            else [1, len(p.blocks), -len(p.blocks), p.have_pages]
            for p in proposals.values()
        ]
        *agreed, (budget, swa_budget, _, _) = self._lockstep.reduce(
            [*rows, [budget, swa_budget, 0, 0]], dist.ReduceOp.MIN
        )
        asked = []
        for (room, proposal), (ok, want, neg_want, first) in zip(
            proposals.items(), agreed
        ):
            count = want - first
            window = min(want, self._window_pages)
            if not ok or want != -neg_want or count > budget or window > swa_budget:
                self.stats["over_budget"] += int(bool(ok) and want == -neg_want)
                self._channel.resolve_prefix_pull(room)
                continue
            budget -= count
            swa_budget -= window
            have = (
                torch.cat(proposal.have_indices)
                if proposal.have_indices
                else torch.empty(0, dtype=torch.int64)
            )
            self._plans[room] = DramPrefixPullPlan(
                first_page=first,
                num_pages=count,
                remote=list(proposal.blocks[first:want]),
                have_indices=have[: first * page],
                swa_remote=list(proposal.swa_blocks),
                locks=lock_host_path(self._cache, proposal.node),
            )
            asked.append(room)
        sent = self._lockstep.reduce(
            [
                [
                    int(
                        self._channel.ask_prefix_pull(
                            room, plan.first_page, plan.num_pages
                        )
                    )
                ]
                for room in asked
                for plan in [self._plans[room]]
            ],
            dist.ReduceOp.MIN,
        )
        for room, (ok,) in zip(asked, sent):
            if not ok:
                self._abandon_pull(room)

    def _start_agreed_reads(self, rooms: List[int]) -> None:
        """Reserve landing pages and start the reads of rooms whose READY every
        rank holds; a room any rank cannot serve is given up by all."""
        rooms = [room for room in rooms if self._reserve_host_landing(room)]
        started = self._lockstep.reduce(
            [[int(self._start_read(room))] for room in rooms], dist.ReduceOp.MIN
        )
        for room, (ok,) in zip(rooms, started):
            if not ok:
                self.stats["failed"] += 1
                self._abandon_pull(room)

    def insert_recalled_host_prefix(self, req: Any) -> bool:
        """Insert the request's recalled pages; whether a recall was inserted."""
        plan = self._plans.pop(req.bootstrap_room, None)
        if plan is None:
            return False
        if not plan.agreed_done:
            self.stats["failed"] += 1
            self._discard_plan(plan)
            return False
        node, _ = insert_host_prefix(
            self._cache,
            prefix_radix_key(
                req=req,
                num_tokens=(plan.first_page + plan.num_pages) * self._page_size,
            ),
            torch.cat([plan.have_indices, plan.new_indices]),
            self._page_size,
            swa_indices=plan.swa_indices,
            reused_tokens=len(plan.have_indices),
            stats=self.stats,
        )
        inserted = node is not None
        if inserted:
            swa_tokens = len(plan.swa_remote) * self._page_size
            self._pins[req.bootstrap_room] = _RecalledPrefixPin(
                req=req,
                node=node,
                receipt=self._cache.inc_host_lock_ref(node).to_dec_params(),
                num_tokens=plan.num_pages * self._page_size,
                swa_tokens=swa_tokens,
            )
            self.stats["pulls"] += 1
            self.stats["pulled_tokens"] += len(plan.new_indices)
            self.stats["pulled_swa_tokens"] += swa_tokens
        unlock_host_path(self._cache, plan.locks)
        return inserted

    def drop_departed(self, queued_reqs: Sequence[Any]) -> None:
        """Drop the plans of requests that left the bootstrap queue without a
        commit (failed or aborted). Queue membership is the same on every
        rank, so the drop is in step."""
        queued = {req.bootstrap_room for req in queued_reqs}
        for room in [room for room in self._plans if room not in queued]:
            self._discard_plan(self._plans.pop(room))
        for room in [room for room in self._offered if room not in queued]:
            del self._offered[room]

    def release_all(self) -> None:
        """Before a cache reset or memory release, which core issues only on
        an idle engine: no recall may be in flight; a paused engine's waiting
        requests still hold their pins."""
        if self._plans or self._zombies:
            raise RuntimeError("prefix recalls are in flight; the engine is not idle")
        self._offered.clear()
        for room in list(self._pins):
            self._unpin(room)

    def holds_host_landing_pages(self) -> bool:
        """Landing pages outside normal admission; the same on every rank."""
        return bool(self._zombies) or any(
            plan.new_indices is not None for plan in self._plans.values()
        )

    def free_settled_host_landings(self) -> None:
        """Free the landings of dropped requests in the step in which their
        read has settled on every rank: host evictions depend on the free room,
        so the ranks' pools must stay equal. The paused loop also calls this: an
        aborted last request leaves no sender to poll, but its read may still
        write the destination pages."""
        if not self.holds_host_landing_pages():
            return
        # Drained here for the step's polls of in-flight reads as well.
        self._channel.drain_host_reads()
        settled = self._lockstep.reduce(
            [[int(not self._read_still_pending(plan))] for plan in self._zombies],
            dist.ReduceOp.MIN,
        )
        keep = []
        for plan, (done,) in zip(self._zombies, settled):
            if done:
                self._free_landing(plan)
            else:
                keep.append(plan)
        self._zombies = keep

    # ----- steps ---------------------------------------------------------------

    def _inflight_tokens(self) -> int:
        return (
            sum(p.num_pages * self._page_size for p in self._plans.values())
            + sum(p.num_tokens for p in self._pins.values())
            + sum(len(p.new_indices) for p in self._zombies)
        )

    def _inflight_swa_tokens(self) -> int:
        """Window tokens reserved by pulls not yet admitted, zombies included."""
        return self._page_size * sum(
            len(p.swa_remote) for p in (*self._plans.values(), *self._zombies)
        ) + sum(p.swa_tokens for p in self._pins.values())

    def _reserve_host_landing(self, room: int) -> bool:
        """Reserve the host pages the pull lands in: the full pages, and the
        window pages of a hybrid model, each in its own pool."""
        plan = self._plans[room]
        full_tokens = plan.num_pages * self._page_size
        swa_tokens = len(plan.swa_remote) * self._page_size
        self._make_host_room(self._host_pool, full_tokens, ComponentType.FULL)
        self._make_host_room(self._swa_host_pool, swa_tokens, ComponentType.SWA)
        landings = alloc_host_pages_agreed(
            lockstep=self._lockstep,
            pools=[(self._host_pool, full_tokens), (self._swa_host_pool, swa_tokens)],
        )
        if landings is None:
            self.stats["no_memory"] += 1
            self._abandon_pull(room)
            return False
        plan.new_indices, plan.swa_indices = landings
        return True

    def _make_host_room(self, pool: Any, tokens: int, component: ComponentType) -> None:
        """Evict from ``pool`` until every rank has room for ``tokens``; full
        pages the device still holds are reclaimed when eviction is not enough."""
        if not tokens:
            return
        short = self._lockstep.max(max(0, tokens - pool.available_size()))
        if not short:
            return
        freed = self._cache.evict_host(short, component)
        if freed < short and component == ComponentType.FULL:
            reclaim_full_host_duplicates(self._cache, short - freed)

    def _start_read(self, room: int) -> bool:
        """Start the pull's native reads from the pages the decode registered:
        the full pages, then the window pages of a hybrid model."""
        plan = self._plans[room]
        plan.started = True
        # None once an abort dropped the room since the ranks agreed on READY.
        ready = self._channel.prefix_ready(room)
        if ready is None or ready.refused:
            return False
        if ready.has_window != bool(plan.swa_remote):
            return False
        # A read that started keeps its landing owned also when the other one
        # of the pull does not start.
        if not self._start_full_read(room, plan, ready):
            return False
        return not plan.swa_remote or self._start_window_read(room, plan, ready)

    def _start_full_read(
        self, room: int, plan: DramPrefixPullPlan, ready: PrefixRegistration
    ) -> bool:
        read_id = self._channel.prefix_req_id(room)
        started = self._channel.start_host_read(
            read_id,
            ready.uuid,
            ready.endpoint,
            plan.remote,
            host_page_ids(self._host_pool, plan.new_indices),
        )
        if started:
            plan.read_ids.append(read_id)
        return started

    def _start_window_read(
        self, room: int, plan: DramPrefixPullPlan, ready: PrefixRegistration
    ) -> bool:
        read_id = self._channel.prefix_req_id(room, swa=True)
        started = self._channel.start_host_read(
            read_id,
            ready.swa_uuid,
            ready.swa_endpoint,
            plan.swa_remote,
            host_page_ids(self._swa_host_pool, plan.swa_indices),
            swa=True,
        )
        if started:
            plan.read_ids.append(read_id)
        return started

    def _unpin(self, room: int) -> None:
        # A cache reset unpins first (release_all), so the node is still live.
        pin = self._pins.pop(room, None)
        if pin is not None:
            self._cache.dec_host_lock_ref(pin.node, pin.receipt)

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

    def _abandon_pull(self, room: int) -> None:
        self._discard_plan(self._plans.pop(room))
        self._channel.resolve_prefix_pull(room)

    def _discard_plan(self, plan: DramPrefixPullPlan) -> None:
        unlock_host_path(self._cache, plan.locks)
        if plan.new_indices is not None:
            # Freed by free_settled_host_landings once every rank's read settled.
            self._zombies.append(plan)

    def _free_landing(self, plan: DramPrefixPullPlan) -> None:
        self._host_pool.free(plan.new_indices)
        if plan.swa_indices is not None:
            self._swa_host_pool.free(plan.swa_indices)

    def _read_still_pending(self, plan: DramPrefixPullPlan) -> bool:
        return self._channel.poll_host_reads(plan.read_ids) is None
