# SPDX-License-Identifier: Apache-2.0
"""The P/D prefix pull: the decode reports, holds and anchors its pullable prefix;
the prefill fetches it.

Why: a P/D prefill recomputes prompt prefixes whose SWA window it lost, while the
     decode still holds their full KV and often the window. The decode tells the
     prefill how much it can serve, keeps that prefix locked until the request's
     KV lands, and keeps prompt-end windows, where the next turn diverges.
How: AFTER DecodeHiCachePreallocMixin._build_decode_prefix_match computes the
     pullable length (prefix_pull_decode.decode_pullable), queues it for the
     Raiden pipeline request and, with SGLANG_TPU_PD_PREFIX_PULL, locks it. AFTER
     DecodePreallocQueue._release_matched_prefix_lock and AFTER
     mem_cache.common.release_kv_cache drop the lock with the request's; AFTER
     ScheduleBatchDisaggregationDecodeMixin.process_prebuilt drops it once the KV landed and,
     with SGLANG_TPU_PD_DECODE_PROMPT_END_ANCHORS > 0, anchors the prompt-end
     window the prebuilt checkpoint just inserted. AROUND
     DecodePreallocQueue._release_cap_anchor and ScheduleBatch.check_decode_mem
     release anchors under memory pressure after upstream's CAP anchors. AFTER
     UnifiedRadixCache.reset forgets the locks a reset dropped. Prefill: AFTER
     PrefillBootstrapQueue.__init__ sets its prefix_fetcher to a
     PrefixPullFetcher (prefix_pull_prefill.py) with SGLANG_TPU_PD_PREFIX_PULL
     on a Raiden prefill; AFTER PrefillAdder._account_prefill_cache_admission
     and AFTER release_aborted_prefill_waiting_req drop the lock it keeps on a
     pulled prefix until the request is admitted.
Upstream PR or issue: none.
Remove when: core grows a cache-owned anchor seam the plugin can fill.
"""

from __future__ import annotations

import weakref

from sglang_tpu.disaggregation.prefix_pull import pullable_by_room
from sglang_tpu.disaggregation.prefix_pull_decode import (
    decode_pullable,
    prompt_end_anchor_node,
    prompt_end_anchors,
    pull_holds,
)
from sglang_tpu.hooks.pd_prefix_pull_metrics import (
    PullOutcomeMetrics,
    note_decode_match,
)


def _is_warmup(req) -> bool:
    from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

    # The PD warmup request runs on fake transfers: no prefill pulls from it.
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST


def on_decode_match(result, queue, req, *args, **kwargs):
    from sglang_tpu.envs import envs

    if _is_warmup(req):
        return result
    pullable = decode_pullable(queue.tree_cache, req)
    note_decode_match(
        req, decode_prefix_len=result.decode_prefix_len, pullable_len=pullable.length
    )
    held = envs.SGLANG_TPU_PD_PREFIX_PULL.get() and pull_holds.acquire(
        queue.tree_cache, req, pullable
    )
    if req.bootstrap_room is not None:
        # The Raiden pipeline request carries it to the prefill.
        pullable_by_room.put(req.bootstrap_room, pullable.length, held)
    return result


def release_hold_with_match(result, queue, req, *args, **kwargs):
    pull_holds.release(req.rid)
    return result


def release_hold_with_kv(result, req, *args, **kwargs):
    pull_holds.release(req.rid)
    return result


def on_prebuilt(result, batch, *args, **kwargs):
    from sglang_tpu.envs import envs

    budget = envs.SGLANG_TPU_PD_DECODE_PROMPT_END_ANCHORS.get()
    for req in batch.reqs:
        # The KV landed, so any pull of the prefix is done.
        pull_holds.release(req.rid)
        if budget > 0 and not _is_warmup(req):
            node = prompt_end_anchor_node(batch.tree_cache, req)
            if node is not None:
                prompt_end_anchors.pin(batch.tree_cache, node.id, budget)
    return result


def release_anchor_for_prealloc(original, queue, *args, **kwargs):
    return original(queue, *args, **kwargs) or prompt_end_anchors.release_lru(
        queue.tree_cache
    )


def release_anchors_for_decode_step(original, batch, *args, **kwargs):
    ok = original(batch, *args, **kwargs)
    while not ok and prompt_end_anchors.release_lru(batch.tree_cache):
        ok = original(batch, *args, **kwargs)
    return ok


def forget_on_reset(result, tree_cache, *args, **kwargs):
    pull_holds.forget_cache(tree_cache)
    prompt_end_anchors.forget_cache(tree_cache)
    return result


# ----- prefill ------------------------------------------------------------

# Each prefill scheduler's fetcher, by its tree cache.
_fetchers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def attach_fetcher(result, queue, *args, **kwargs):
    from sglang_tpu.disaggregation.prefix_pull_prefill import PrefixPullFetcher
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
    from sglang_tpu.envs import envs

    if envs.SGLANG_TPU_PD_PREFIX_PULL.get() and isinstance(
        queue.kv_manager, RaidenKVManager
    ):
        fetcher = PrefixPullFetcher(queue, PullOutcomeMetrics(queue.scheduler))
        queue.prefix_fetcher = fetcher
        _fetchers[queue.scheduler.tree_cache] = fetcher
    return result


def release_prefix_lock_at_admission(result, adder, req, *args, **kwargs):
    fetcher = _fetchers.get(adder.tree_cache)
    if fetcher is not None:
        fetcher.release_prefix_lock(req)
    return result


def release_prefix_lock_on_abort(result, scheduler, req, *args, **kwargs):
    fetcher = _fetchers.get(scheduler.tree_cache)
    if fetcher is not None:
        fetcher.release_prefix_lock(req)
    return result


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    queue = "sglang.srt.disaggregation.decode.DecodePreallocQueue"
    return [
        (
            "sglang.srt.disaggregation.decode_hicache_mixin."
            "DecodeHiCachePreallocMixin._build_decode_prefix_match",
            on_decode_match,
            HookType.AFTER,
        ),
        (
            f"{queue}._release_matched_prefix_lock",
            release_hold_with_match,
            HookType.AFTER,
        ),
        (
            "sglang.srt.mem_cache.common.release_kv_cache",
            release_hold_with_kv,
            HookType.AFTER,
        ),
        (
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "ScheduleBatchDisaggregationDecodeMixin.process_prebuilt",
            on_prebuilt,
            HookType.AFTER,
        ),
        (f"{queue}._release_cap_anchor", release_anchor_for_prealloc, HookType.AROUND),
        (
            "sglang.srt.managers.schedule_batch.ScheduleBatch.check_decode_mem",
            release_anchors_for_decode_step,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.reset",
            forget_on_reset,
            HookType.AFTER,
        ),
        (
            "sglang.srt.disaggregation.prefill.PrefillBootstrapQueue.__init__",
            attach_fetcher,
            HookType.AFTER,
        ),
        (
            "sglang.srt.managers.schedule_policy.PrefillAdder."
            "_account_prefill_cache_admission",
            release_prefix_lock_at_admission,
            HookType.AFTER,
        ),
        (
            "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin."
            "release_aborted_prefill_waiting_req",
            release_prefix_lock_on_abort,
            HookType.AFTER,
        ),
    ]
