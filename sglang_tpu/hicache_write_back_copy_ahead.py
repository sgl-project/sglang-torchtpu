# SPDX-License-Identifier: Apache-2.0
"""Copy the next eviction victims to host ahead of write_back evictions.

Why: under ``--hicache-write-policy write_back`` the unified cache copies a
     victim's KV to host at eviction time and waits for the copy before it
     frees the device pages. On TPU the engine D2H is a background DMA that
     is also fenced behind every in-flight forward, so each such eviction
     stalls the scheduler for a device sync plus the copy, on the allocation
     path of the batch being formed.
How: ``WriteBackCopyAhead`` keeps the next ``copy_ahead_tokens`` of KV in
     eviction order, the copy-ahead window, copied to host. Eviction still
     runs only when allocation needs pages, and the copied nodes stay on
     device until then. Once per scheduler step, after the cache's own ack
     poll, it walks the two orders the eviction takes (FULL device leaves by
     eviction priority, a parent once its last device child is covered; and,
     with its own share of the window, the SWA LRU) and issues a
     write-through style backup, locked until its ack, for every victim in
     the window without a host copy. The per-step poll reaps the acks and
     releases the locks, so an eviction demotes those victims without
     waiting; a victim past the window takes the synchronous path, which
     does not wait for window copies. An eviction that comes up short behind
     window copies waits for them once and walks again.
     AFTER UnifiedRadixCache.init_hicache (attach the window), REPLACE
     UnifiedRadixCache._evict (skip the trailing drain; retry once behind
     window copies), AFTER UnifiedRadixCache.check_hicache_events (refill the
     window), AROUND UnifiedRadixCache._finish_write_through_ack (a window
     copy's lock release leaves the write horizon alone), AROUND
     UnifiedRadixCache.writing_check and REPLACE
     UnifiedRadixCache.backup_node_for_write_back (a synchronous write_back
     wait skips window copies it does not depend on). Only TPU DMA write_back
     caches take the new arms.
Upstream PR or issue: not filed; the walk reads the Python tree core.
Remove when: core backs up the eviction frontier asynchronously itself.

Every Full host copy of a node inside the window is pinned, whoever made it, or
host pressure would reclaim it as a duplicate and the next step would copy the
same victim again. Outside the window it is an ordinary duplicate, so the host
pins at most ``copy_ahead_tokens`` of duplicates. A window of 0 keeps the
synchronous policy.
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    EvictResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        UnifiedTreeCore,
        UnifiedTreeNode,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

# Default window, in prefill chunks. A copy started in a step lands only after
# the forward in flight; measured on gpt-oss-120b TP8, two chunks still left
# 5-10 ms of stall per eviction and four left none.
_COPY_AHEAD_CHUNKS = 4
# Stands in for the chunk when chunked prefill is off.
_DEFAULT_CHUNK_TOKENS = 4096
# The SWA eviction order gets 1/_SWA_WINDOW_DIVISOR of the window. Arbitrary
# starting point; no measurement behind it yet.
_SWA_WINDOW_DIVISOR = 4


def resolve_copy_ahead_tokens(
    *, chunked_prefill_size: Optional[int], page_size: int
) -> int:
    tokens = tpu_envs.SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS.get()
    if tokens is None:
        chunk = chunked_prefill_size
        if chunk is None or chunk <= 0:
            chunk = _DEFAULT_CHUNK_TOKENS
        tokens = _COPY_AHEAD_CHUNKS * chunk
    if tokens <= 0:
        return 0
    return -(-tokens // page_size) * page_size


class WriteBackCopyAhead:
    """Keeps the eviction frontier of a write_back unified cache copied to host."""

    def __init__(self, *, cache: UnifiedRadixCache, copy_ahead_tokens: int) -> None:
        self._cache = cache
        self.copy_ahead_tokens = copy_ahead_tokens
        # ack id (= node id) -> tokens of a window copy not yet acked.
        self._pending: dict[int, int] = {}
        # The pending copies the SWA order issued; each order has its own cap
        # on copies in flight, keyed by whether it is the SWA order.
        self._swa_pending: set[int] = set()
        self._caps: dict[bool, int] = {False: copy_ahead_tokens, True: 0}
        self._inflight_tokens: dict[bool, int] = {False: 0, True: 0}
        # node id -> host lock receipt, held while the copied node is on device
        # and inside the window.
        self._host_locks: dict[int, DecLockRefParams] = {}
        self.num_backups = 0
        self.num_tokens = 0

    @property
    def pending_node_ids(self) -> set[int]:
        return set(self._pending)

    def owns_ack(self, ack_id: int) -> bool:
        return ack_id in self._pending

    def forget_ack(self, ack_id: int) -> None:
        # At the ack, not the next refill: the node's later synchronous backup
        # reuses the ack id and must not look like a window copy.
        tokens = self._pending.pop(ack_id, None)
        if tokens is not None:
            self._inflight_tokens[ack_id in self._swa_pending] -= tokens
        self._swa_pending.discard(ack_id)

    def window_ack_of(self, node_id: int) -> set[int]:
        """The window ack ``node_id`` is pending under, if any; a barrier backup
        marks only the node and unbacked ancestors below the first pending one."""
        ack_id = self._cache.tree_core.node_by_id(node_id).write_through_pending_id
        return {ack_id} if ack_id in self._pending else set()

    def wait_for_acks(self, *, window_acks: set[int] = frozenset()) -> None:
        """Finish every queued ack except those holding only window copies not
        in ``window_acks``; the step's poll reaps the rest.

        Rank-consistent: every rank queues the same acks and owns the same
        window copies, so every rank removes the same entries.
        """
        cache = self._cache
        controller = cache.cache_controller
        controller.start_writing()
        kept = []
        for ack in controller.ack_write_queue:
            if all(
                ack_id in self._pending and ack_id not in window_acks
                for ack_id in ack.node_ids
            ):
                kept.append(ack)
                continue
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                if ack_id in cache.ongoing_write_through:
                    cache._finish_write_through_ack(ack_id)
            cache._log_write_ack_metrics(ack)
        controller.ack_write_queue.clear()
        controller.ack_write_queue.extend(kept)

    def refill(self) -> None:
        """Keep the victims inside the window copied to host and pinned."""
        cache = self._cache
        if cache.disable or cache.cache_controller is None:
            return
        # Before the walk, so this step's copies can take the freed host slots.
        self._release_host_locks(window=None)
        core: UnifiedTreeCore = cache.tree_core
        # The SWA order gets a reserved share: the Full order alone fills the
        # window whenever many leaves are evictable.
        swa_share = 0
        if cache.is_swa_enabled and core.has_swa_host_pool:
            swa_share = self.copy_ahead_tokens // _SWA_WINDOW_DIVISOR
        self._prune_acked()
        window: set[int] = set()
        # Copies in flight are locked, so neither walk sees them: subtract them.
        self._caps = {False: self.copy_ahead_tokens - swa_share, True: swa_share}
        full_budget = self._caps[False] - self._inflight(swa=False)
        if full_budget > 0:
            self._cover_full(core=core, budget=full_budget, window=window)
        swa_budget = self._caps[True] - self._inflight(swa=True)
        if swa_budget > 0:
            self._cover_swa(core=core, budget=swa_budget, window=window)
        self._release_host_locks(window=window)

    def _release_host_locks(self, *, window: Optional[set[int]]) -> None:
        """Unpin copies whose node left the device and, given the walked
        ``window``, landed copies whose node is outside it."""
        core = self._cache.tree_core
        for node_id, params in list(self._host_locks.items()):
            try:
                node = core.node_by_id(node_id)
            except KeyError:
                del self._host_locks[node_id]
                continue
            outside = (
                window is not None
                and node_id not in window
                and node_id not in self._pending
            )
            if node.evicted or outside:
                self._cache.dec_host_lock_ref(node_id, params)
                del self._host_locks[node_id]

    def _prune_acked(self) -> None:
        ongoing = self._cache.ongoing_write_through
        for ack_id in [a for a in self._pending if a not in ongoing]:
            self.forget_ack(ack_id)

    def _inflight(self, *, swa: bool) -> int:
        return self._inflight_tokens[swa]

    def _cover_full(
        self, *, core: UnifiedTreeCore, budget: int, window: set[int]
    ) -> None:
        full = self._cache.components[BASE_COMPONENT_TYPE]
        full._ensure_eviction_strategy()
        priority = full.session_ref_eviction_strategy
        heap = [(priority(n), n) for n in core.evictable_device_leaves]
        heapq.heapify(heap)
        taken: set[int] = set()
        covered = 0
        while heap and covered < budget:
            _, node = heapq.heappop(heap)
            if node.id in taken:
                continue
            taken.add(node.id)
            value = node.component_data[BASE_COMPONENT_TYPE].value
            if value is None:
                continue
            covered += len(value)
            if covered > budget:
                # Crossing the window's end: left to the synchronous backup.
                break
            window.add(node.id)
            self._keep_copied(node, swa=False)
            # The parent is the next victim once its last device child is gone.
            parent = node.parent
            if (
                parent is not None
                and parent is not core.root_node
                and not parent.evicted
                and all(cd.lock_ref == 0 for cd in parent.component_data)
                and all(
                    child.id in taken
                    or child.component_data[BASE_COMPONENT_TYPE].value is None
                    for child in parent.children.values()
                )
            ):
                heapq.heappush(heap, (priority(parent), parent))

    def _cover_swa(
        self, *, core: UnifiedTreeCore, budget: int, window: set[int]
    ) -> None:
        # Mirrors SWAComponent._evict_device_next_node. Copied or pinned nodes
        # count Full tokens; the rest, and candidates too large to fit, count SWA.
        ct = ComponentType.SWA
        lru = core.lru_lists[ct]
        covered = 0
        node = lru.get_lru_no_lock()
        while node is not None and covered < budget:
            cd = node.component_data[ct]
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if cd.value is not None and full_value is not None:
                candidate = (
                    node in core.evictable_device_leaves
                    or node.id in self._host_locks
                    or (cd.host_value is None and not node.backuped)
                )
                if candidate and covered + len(full_value) <= budget:
                    covered += len(full_value)
                    window.add(node.id)
                    self._keep_copied(node, swa=True)
                else:
                    covered += len(cd.value)
            node = lru.get_prev_no_lock(node)

    def _keep_copied(self, node: UnifiedTreeNode, *, swa: bool) -> None:
        """Pin ``node``'s host copy, copying it first if it has none and the
        order's copies in flight stay within its cap."""
        if node.id in self._host_locks or node.write_through_pending_id is not None:
            return
        cache = self._cache
        if not node.backuped:
            full_tokens = len(node.component_data[BASE_COMPONENT_TYPE].value)
            if self._inflight(swa=swa) + full_tokens > self._caps[swa]:
                return
            if cache.is_swa_enabled and cache.tree_core.has_swa_host_pool:
                self._reserve_swa_host(node)
            # Locked until the ack, like a write-through backup, so no eviction
            # frees pages the DMA still reads; the ack releases the lock.
            written = cache._execute_and_commit_kv_backup(
                BackupKV(node_ids=[node.id]), write_back=False
            )
            if written <= 0:
                return
            self._pending[node.id] = written
            self._inflight_tokens[swa] += written
            if swa:
                self._swa_pending.add(node.id)
            self.num_backups += 1
            self.num_tokens += written
        # Pinned while in the window: host pressure must not reclaim a copy the
        # next eviction is about to need.
        self._host_locks[node.id] = self._pin_full_host(node)

    def _pin_full_host(self, node: UnifiedTreeNode) -> DecLockRefParams:
        # Full only: an SWA host lock would hold a window of ancestors off the
        # SWA host LRU and steer SWA host eviction onto host leaves.
        core = self._cache.tree_core
        result = IncLockRefResult(
            node_id=node.id,
            skipped_lock_components=tuple(
                c.component_type
                for c in core.components
                if c.component_type != BASE_COMPONENT_TYPE
            ),
        )
        full = self._cache.components[BASE_COMPONENT_TYPE]
        return full.acquire_component_lock(
            node=node, result=result, lock_host=True
        ).to_dec_params()

    def _reserve_swa_host(self, node: UnifiedTreeNode) -> None:
        cache = self._cache
        swa = cache.components[ComponentType.SWA]
        needed = sum(
            len(n.component_data[ComponentType.SWA].value)
            for n in swa._collect_unbacked_swa_nodes(node)
        )
        pool = cache.host_pool_group.get_pool(PoolName.SWA)
        if needed and pool.available_size() < needed:
            cache.evict_host(num_tokens=needed, component_type=ComponentType.SWA)


def attach_copy_ahead(
    *, cache: UnifiedRadixCache, chunked_prefill_size: Optional[int]
) -> None:
    """Give a TPU DMA write_back cache a copy-ahead window; others get None."""
    from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

    cache._write_back_copy_ahead = None
    controller = cache.cache_controller
    if (
        not isinstance(controller, TpuDMAHybridCacheController)
        or controller.write_policy != "write_back"
        or cache.host_memory_mode != "cache"
    ):
        return
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    if cache.pp_size > 1 or not isinstance(cache.tree_core, UnifiedTreeCore):
        logger.warning(
            "HiCache write_back copy-ahead needs the Python tree core and no "
            "pipeline parallelism; evictions back up their victims synchronously."
        )
        return
    copy_ahead_tokens = resolve_copy_ahead_tokens(
        chunked_prefill_size=chunked_prefill_size, page_size=cache.page_size
    )
    if copy_ahead_tokens <= 0:
        logger.info(
            "HiCache write_back copy-ahead off: evictions back up synchronously."
        )
        return
    cache._write_back_copy_ahead = WriteBackCopyAhead(
        cache=cache, copy_ahead_tokens=copy_ahead_tokens
    )
    logger.info(
        "HiCache write_back copy-ahead: %d tokens copied to host ahead of eviction.",
        copy_ahead_tokens,
    )


def evict_with_copy_ahead(
    cache: UnifiedRadixCache, params: EvictParams, available_size_targets=None
) -> EvictResult:
    if cache.disable:
        return EvictResult()
    start_time = time.perf_counter()
    tracker = {ct: 0 for ct in cache.tree_components}
    request_by_type = cache._evict_request_by_type(params)
    cache._evict_components(
        request_by_type, tracker, available_size_targets=available_size_targets
    )
    # Short behind window copies' locks would retract a request: wait for them
    # once and walk again with the same request (the tracker is cumulative).
    copy_ahead = cache._write_back_copy_ahead
    if copy_ahead.pending_node_ids and _is_short(
        cache, request_by_type, tracker, available_size_targets
    ):
        copy_ahead.wait_for_acks(window_acks=copy_ahead.pending_node_ids)
        cache._evict_components(
            request_by_type, tracker, available_size_targets=available_size_targets
        )
    cache.update_eviction_metrics(tracker[BASE_COMPONENT_TYPE], start_time)
    return EvictResult(
        num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
        swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
        mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
    )


def _is_short(
    cache: UnifiedRadixCache,
    request_by_type: dict[ComponentType, int],
    tracker: dict[ComponentType, int],
    available_size_targets,
) -> bool:
    for component_type in cache.tree_components:
        requested = request_by_type.get(component_type, 0)
        if requested <= 0 or tracker.get(component_type, 0) >= requested:
            continue
        target = (available_size_targets or {}).get(component_type)
        if target is not None:
            target_component, target_size = target
            if cache._component_available_size(target_component) >= target_size:
                continue
        return True
    return False


def install(cache_cls: Optional[type] = None) -> None:
    """Wrap ``UnifiedRadixCache`` so TPU DMA write_back caches copy ahead."""
    if cache_cls is None:
        from sglang.srt.mem_cache.unified_radix_cache import (
            UnifiedRadixCache as cache_cls,
        )

    orig_init_hicache = cache_cls.init_hicache
    if getattr(orig_init_hicache, "_tpu_write_back_copy_ahead", False):
        return
    orig_evict = cache_cls._evict
    orig_check_hicache_events = cache_cls.check_hicache_events
    orig_finish_ack = cache_cls._finish_write_through_ack
    orig_writing_check = cache_cls.writing_check
    orig_backup_node_for_write_back = cache_cls.backup_node_for_write_back

    def init_hicache(self, server_args, params):
        orig_init_hicache(self, server_args, params)
        attach_copy_ahead(
            cache=self, chunked_prefill_size=server_args.chunked_prefill_size
        )

    def _evict(self, params: EvictParams, available_size_targets=None) -> EvictResult:
        if self._write_back_copy_ahead is None:
            return orig_evict(self, params, available_size_targets)
        return evict_with_copy_ahead(self, params, available_size_targets)

    def check_hicache_events(self):
        orig_check_hicache_events(self)
        copy_ahead = self._write_back_copy_ahead
        if copy_ahead is not None:
            copy_ahead.refill()
            # Submit now, not at the end of the step: the copy lands a forward earlier.
            self.flush_pending_backups()

    def _finish_write_through_ack(self, ack_id):
        copy_ahead = self._write_back_copy_ahead
        if copy_ahead is None or not copy_ahead.owns_ack(ack_id):
            return orig_finish_ack(self, ack_id)
        # The lock release stamps the victim's path as just written, which
        # would hold its pages behind the in-flight forward once demoted and
        # make the next host reload wait for the device. The copy wrote
        # nothing, so the horizon stays where it was.
        saved = []
        node = self.tree_core.node_by_id(ack_id)
        while node is not None:
            saved.append((node, node.write_horizon))
            node = node.parent
        orig_finish_ack(self, ack_id)
        for node, horizon in saved:
            node.write_horizon = horizon
        copy_ahead.forget_ack(ack_id)

    def writing_check(self, write_back=False, finish_count=None):
        copy_ahead = self._write_back_copy_ahead
        if not write_back or copy_ahead is None:
            return orig_writing_check(self, write_back, finish_count)
        # A synchronous write_back wait skips the window's copies.
        copy_ahead.wait_for_acks()

    def backup_node_for_write_back(self, node_id):
        copy_ahead = self._write_back_copy_ahead
        if copy_ahead is None:
            return orig_backup_node_for_write_back(self, node_id)
        # Upstream drains every ack first in case one already covers this node;
        # of the window copies, only the node's own can.
        copy_ahead.wait_for_acks(window_acks=copy_ahead.window_ack_of(node_id))
        written = self._execute_and_commit_kv_backup(
            BackupKV(node_ids=[node_id]), write_back=True
        )
        if written == 0:
            return False
        copy_ahead.wait_for_acks()
        return True

    init_hicache._tpu_write_back_copy_ahead = True
    # Caches that never attach HiCache keep the class default.
    cache_cls._write_back_copy_ahead = None
    cache_cls.init_hicache = init_hicache
    cache_cls._evict = _evict
    cache_cls.check_hicache_events = check_hicache_events
    cache_cls._finish_write_through_ack = _finish_write_through_ack
    cache_cls.writing_check = writing_check
    cache_cls.backup_node_for_write_back = backup_node_for_write_back
