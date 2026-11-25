# SPDX-License-Identifier: Apache-2.0
"""Reduce HiCache ready counts across TP ranks without blocking the scheduler.

Why: ``UnifiedRadixCache._sync_hicache_ready_counts`` MIN-reduces the ack and
     storage-queue counts across TP ranks with a blocking gloo all-reduce on the
     scheduler thread, once per step. On TPU the ranks' hosts drift by up to a
     forward, so every rank waits for the slowest one every step (~0.8 ms per
     round at bs=1 with HiCache on).
How: REPLACE ``_sync_hicache_ready_counts`` for the TP-only case without a
     storage backend: the reduce of the write and load ack counts is issued
     asynchronously and its result is consumed on the next step, so a step acts
     on the counts reduced one step earlier. Each rank sends only the ready acks
     beyond the ones it pops, and remembers which acks those are. A blocking
     drain in between (a write_back eviction) can remove some of them and keep
     others (write_back copy-ahead keeps its window copies queued), so the
     result is spent only on the leading live acks that are among the ones it
     counted: drains remove the same acks on every rank, so every rank pops the
     same acks. PP > 1, single-rank, storage, and attention CP and TP both
     wider than one rank keep the synchronous path: storage queues hold no ack
     objects to match, and core reduces over the two attention groups one after
     the other. AROUND ``reset`` to drain the in-flight reduce.
Upstream PR or issue: not filed; CUDA and NPU keep the synchronous reduce.
Remove when: core consumes ready counts asynchronously itself
     (sgl-project/sglang#45795 proposes it).
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

ReadyCounts = tuple[int, int, tuple[int, ...], tuple["PoolName", ...]]
# cache -> (works, counts, digest, counted acks per queue) of its in-flight reduce.
_pending: weakref.WeakKeyDictionary[UnifiedRadixCache, tuple] = (
    weakref.WeakKeyDictionary()
)


def _reduce_group(cache: UnifiedRadixCache):
    """The one group core's ``_all_reduce_attn_groups`` would reduce over, else None.

    With attention CP and attention TP both wider than one rank, core reduces
    over the two groups one after the other. Two async reduces of one tensor
    would run at the same time on gloo's per-group threads and race, so that
    layout keeps the synchronous path.
    """
    groups = [
        group
        for group in (cache.attn_cp_group, cache.attn_tp_group)
        if group is not None and torch.distributed.get_world_size(group=group) > 1
    ]
    if len(groups) == 1:
        return groups[0]
    if not groups and cache.tp_world_size > 1:
        return cache.tp_group
    return None


def _leading_counted(queue, counted_acks) -> int:
    """How many leading acks of ``queue`` are in ``counted_acks``."""
    counted = {id(ack) for ack in counted_acks}
    popped = 0
    for ack in queue:
        if id(ack) not in counted:
            break
        popped += 1
    return popped


def deferred_ready_counts(cache: UnifiedRadixCache, group) -> ReadyCounts:
    cc = cache.cache_controller
    queues = (cc.ack_write_queue, cc.ack_load_queue)
    ready = [cache._count_ready_acks(queue) for queue in queues]
    digest = cache.tree_core.write_back_duplicate_reclaim_digest
    pops = [0, 0]
    previous = _pending.pop(cache, None)
    if previous is not None:
        works, counts, prev_digest, reported = previous
        for work in works:
            work.wait()
        values = list(map(int, counts.tolist()))
        assert prev_digest == values[2] and prev_digest == -values[3], (
            "write_back duplicate-reclaim victims diverged across TP ranks"
        )
        # The reduced count is the length of the prefix of ``reported`` that was
        # ready on every rank, and ``reported`` lists the same acks on every rank.
        pops = [
            _leading_counted(queue, acks[:count])
            for queue, acks, count in zip(queues, reported, values)
        ]
    # The caller pops ``pops`` acks from the queue heads right after this
    # returns; send only the ready acks beyond them, or an ack is counted twice.
    counts = torch.tensor(
        [ready[0] - pops[0], ready[1] - pops[1], digest, -digest],
        dtype=torch.int64,
        device="cpu",
    )
    _pending[cache] = (
        [
            torch.distributed.all_reduce(
                counts, op=torch.distributed.ReduceOp.MIN, group=group, async_op=True
            )
        ],
        counts,
        digest,
        tuple(tuple(q[pop:n]) for q, pop, n in zip(queues, pops, ready)),
    )
    return pops[0], pops[1], (), ()


def drain_pending(cache: UnifiedRadixCache) -> None:
    pending = _pending.pop(cache, None)
    if pending is not None:
        for work in pending[0]:
            work.wait()


def install(cache_cls: Optional[type] = None) -> None:
    """Wrap ``UnifiedRadixCache`` so TP-only caches defer the ready-count reduce."""
    if cache_cls is None:
        from sglang.srt.mem_cache.unified_radix_cache import (
            UnifiedRadixCache as cache_cls,
        )

    orig_sync = cache_cls._sync_hicache_ready_counts
    if getattr(orig_sync, "_tpu_async_ready_counts", False):
        return
    orig_reset = cache_cls.reset

    def _sync_hicache_ready_counts(self) -> ReadyCounts:
        group = None
        if (
            self.cache_controller is not None
            and self.pp_size == 1
            and self.tp_world_size > 1
            and not self.enable_storage
        ):
            group = _reduce_group(self)
        if group is None:
            # A storage backend attached at runtime switches to the synchronous
            # path; settle the reduce in flight first. Acks it counted stay
            # queued, and the synchronous reduce counts them again.
            drain_pending(self)
            return orig_sync(self)
        return deferred_ready_counts(self, group)

    def reset(self) -> None:
        drain_pending(self)
        orig_reset(self)

    _sync_hicache_ready_counts._tpu_async_ready_counts = True
    cache_cls._sync_hicache_ready_counts = _sync_hicache_ready_counts
    cache_cls.reset = reset
