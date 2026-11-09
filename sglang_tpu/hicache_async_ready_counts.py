# SPDX-License-Identifier: Apache-2.0
"""Reduce HiCache ready counts across TP ranks without blocking the scheduler.

Why: ``UnifiedRadixCache._sync_hicache_ready_counts`` MIN-reduces the ack and
     storage-queue counts across TP ranks with a blocking gloo all-reduce on the
     scheduler thread, once per step. On TPU the ranks' hosts drift by up to a
     forward, so every rank waits for the slowest one every step (~0.8 ms per
     round at bs=1 with HiCache on).
How: REPLACE ``_sync_hicache_ready_counts`` for the TP-only case: the reduce is
     issued asynchronously and its result is consumed on the next step, so a
     step acts on the counts reduced one step earlier. Queues only grow until
     the caller drains them, so what is sent is what remains after last step's
     result was drained; a blocking drain (write_back eviction) may have
     emptied a queue since, so the previous result is clamped to the live
     count, identically on every rank. PP > 1 and single-rank keep the
     synchronous path. AROUND ``reset`` to drain the in-flight reduce.
Upstream PR or issue: not filed; CUDA and NPU keep the synchronous reduce.
Remove when: core consumes ready counts asynchronously itself.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

ReadyCounts = tuple[int, int, tuple[int, ...], tuple["PoolName", ...]]
# cache -> (works, counts, digest, pool names) of its in-flight reduce.
_pending: "weakref.WeakKeyDictionary[UnifiedRadixCache, tuple]" = (
    weakref.WeakKeyDictionary()
)


def _all_reduce_attn_groups_async(cache: "UnifiedRadixCache", tensor, op) -> list:
    works = []
    for group in (cache.attn_cp_group, cache.attn_tp_group):
        if group is not None and torch.distributed.get_world_size(group=group) > 1:
            works.append(
                torch.distributed.all_reduce(tensor, op=op, group=group, async_op=True)
            )
    if not works and cache.tp_world_size > 1:
        works.append(
            torch.distributed.all_reduce(
                tensor, op=op, group=cache.tp_group, async_op=True
            )
        )
    return works


def _local_counts(cache: "UnifiedRadixCache") -> tuple[list[int], tuple]:
    cc = cache.cache_controller
    extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
    extra_pool_names = tuple(extra_release_queues) if cache.enable_storage else ()
    write_acks = cache._count_ready_acks(cc.ack_write_queue)
    load_acks = cache._count_ready_acks(cc.ack_load_queue)
    storage_queue_sizes = (
        (
            cc.prefetch_hit_queue.qsize(),
            cc.ack_prefetch_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            *(extra_release_queues[name].qsize() for name in extra_pool_names),
        )
        if cache.enable_storage
        else ()
    )
    return [write_acks, load_acks, *storage_queue_sizes], extra_pool_names


def _result(ready_counts: torch.Tensor, digest: int, extra_pool_names) -> ReadyCounts:
    values = list(map(int, ready_counts.tolist()))
    assert digest == values[-2] and digest == -values[-1], (
        "write_back duplicate-reclaim victims diverged across TP ranks"
    )
    return values[0], values[1], tuple(values[2:-2]), extra_pool_names


def deferred_ready_counts(cache: "UnifiedRadixCache") -> ReadyCounts:
    local_counts, extra_pool_names = _local_counts(cache)
    digest = cache.tree_core.write_back_duplicate_reclaim_digest
    previous = _pending.get(cache)
    if previous is None:
        drained = [0] * len(local_counts)
    else:
        works, counts, prev_digest, prev_names = previous
        for work in works:
            work.wait()
        prev = _result(counts, prev_digest, prev_names)
        drained = [prev[0], prev[1], *prev[2]]
        if len(drained) != len(local_counts):
            drained = [0] * len(local_counts)
    # Every rank drains alike, so the clamp agrees across ranks.
    drained = [min(done, now) for now, done in zip(local_counts, drained)]
    remaining = [now - done for now, done in zip(local_counts, drained)]
    ready_counts = torch.tensor(
        [*remaining, digest, -digest], dtype=torch.int64, device="cpu"
    )
    _pending[cache] = (
        _all_reduce_attn_groups_async(
            cache, ready_counts, torch.distributed.ReduceOp.MIN
        ),
        ready_counts,
        digest,
        extra_pool_names,
    )
    return drained[0], drained[1], tuple(drained[2:]), extra_pool_names


def drain_pending(cache: "UnifiedRadixCache") -> None:
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
        if self.cache_controller is None or self.pp_size > 1 or self.tp_world_size <= 1:
            return orig_sync(self)
        return deferred_ready_counts(self)

    def reset(self) -> None:
        drain_pending(self)
        orig_reset(self)

    _sync_hicache_ready_counts._tpu_async_ready_counts = True
    cache_cls._sync_hicache_ready_counts = _sync_hicache_ready_counts
    cache_cls.reset = reset
