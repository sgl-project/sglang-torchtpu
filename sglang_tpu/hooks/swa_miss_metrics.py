# SPDX-License-Identifier: Apache-2.0
"""Count prompt tokens recomputed only because their SWA window was gone.

Why: on a hybrid full + SWA cache, a match stops at the first node whose SWA
     window is missing even when full KV is cached further on. No metric splits
     those misses from full-KV misses, which decides whether more SWA space,
     more full-KV space, or neither would raise the hit rate.
How: AFTER UnifiedRadixCache.match_prefix records the request's full-KV hit
     length and the deepest prefix every validator accepted (the depth of
     best_match_node). AFTER PrefillAdder._account_prefill_cache_admission fixes
     the request's miss range [max(admitted prefix, reusable prefix), full-KV hit)
     and charges the scheduled chunk's overlap with it; AROUND
     PrefillAdder.add_chunked_req and add_dllm_staging_req charge continuation
     chunks. AFTER PrefillStats.from_adder snapshots the batch total, and AFTER
     SchedulerMetricsReporter.report_prefill_stats adds it to
     sglang:prefill_swa_miss_tokens_total, which an AFTER hook on
     SchedulerMetricsCollector.__init__ creates. Counted only on the Python
     TreeCore, for caches whose components are exactly full KV + SWA and that
     have no re-prefill tail.
Upstream PR or issue: none.
Remove when: core attributes prefix misses to the component that cut the match.
"""

from __future__ import annotations

import weakref

import msgspec


class SwaPrefixMatch(msgspec.Struct, frozen=True):
    """Match-time prefix lengths, in tokens."""

    full_kv_hit_length: int = 0
    # Deepest prefix every validator accepted, including host-restorable windows and
    # backed-up FULL; a load-back declined past it is a capacity loss, not an SWA miss.
    reusable_length: int = 0


# Plugin-side state, keyed weakly so it dies with the request or adder.
_matches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_miss_ranges: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_adder_tokens: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# PrefillStats is an unhashable dataclass; key by id and drop the entry when it dies.
_stats_tokens: dict[int, int] = {}
_counting_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _qualifies(tree_cache) -> bool:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    # A re-prefill tail caps the match before the walk, hiding the recomputed window.
    return (
        isinstance(tree_cache, UnifiedRadixCache)
        and isinstance(tree_cache.tree_core, UnifiedTreeCore)
        and set(tree_cache.tree_components) == {ComponentType.FULL, ComponentType.SWA}
        and tree_cache.swa_reprefill_tail_tokens() == 0
    )


def counts_swa_misses(tree_cache) -> bool:
    # Fixed by the cache's construction; checked on every match, so computed once.
    if tree_cache not in _counting_caches:
        _counting_caches[tree_cache] = _qualifies(tree_cache)
    return _counting_caches[tree_cache]


def prefix_depth(tree_core, node_id) -> int:
    node = tree_core.node_by_id(node_id)
    depth = 0
    while node.parent is not None:
        depth += len(node.key)
        node = node.parent
    return depth


def swa_prefix_match(tree_core, match_result) -> SwaPrefixMatch:
    reusable = max(
        prefix_depth(tree_core, match_result.best_match_node),
        len(match_result.device_indices) + match_result.host_hit_length,
    )
    # Producers that skip the tree walk leave the full-KV length at 0;
    # whatever the match reuses has full KV too.
    return SwaPrefixMatch(
        full_kv_hit_length=max(match_result.full_kv_hit_length, reusable),
        reusable_length=reusable,
    )


def swa_miss_range(*, match: SwaPrefixMatch, prefix_len: int) -> tuple[int, int]:
    """Token range `[start, end)` with full KV cached but recomputed because the
    SWA window was missing. `prefix_len` is the admitted prefix after load-back, so
    a window that load-back restored is reused, not missed. Empty when start >= end."""
    return max(prefix_len, match.reusable_length), match.full_kv_hit_length


def _charge_scheduled_chunk(adder, req) -> None:
    # Charged per scheduled chunk, like input tokens, so a prefill aborted between
    # chunks counts only what it recomputed.
    if req.retracted_stain or req not in _miss_ranges:
        return
    start, end = _miss_ranges[req]
    tokens = max(0, min(end, req.extend_range.end) - max(start, req.extend_range.start))
    if tokens:
        _adder_tokens[adder] = _adder_tokens.get(adder, 0) + tokens


def record_match(result, cache, params, *args, **kwargs):
    from sglang.srt.environ import envs

    req = params.req
    if req is None or envs.SGLANG_RADIX_FORCE_MISS.get():
        return result
    # A streaming session's own match carries a sentinel node, not a tree NodeId;
    # it reuses session KV only, so nothing in it is an SWA miss.
    if counts_swa_misses(cache) and isinstance(result.best_match_node, int):
        _matches[req] = swa_prefix_match(cache.tree_core, result)
    else:
        _matches.pop(req, None)
    return result


def charge_admission(result, adder, req, prefix_len, *args, **kwargs):
    # A retracted re-admission was already counted on its first admission.
    if req.retracted_stain or req not in _matches:
        return result
    _miss_ranges[req] = swa_miss_range(match=_matches[req], prefix_len=prefix_len)
    _charge_scheduled_chunk(adder, req)
    return result


def charge_continuation_chunk(original, adder, req, *args, **kwargs):
    scheduled_before = len(adder.can_run_list)
    out = original(adder, req, *args, **kwargs)
    if len(adder.can_run_list) > scheduled_before and adder.can_run_list[-1] is req:
        _charge_scheduled_chunk(adder, req)
    return out


def snapshot_batch(result, cls, adder, *args, **kwargs):
    tokens = _adder_tokens.pop(adder, 0)
    if tokens:
        key = id(result)
        _stats_tokens[key] = tokens
        weakref.finalize(result, _stats_tokens.pop, key, None)
    return result


def add_swa_miss_counter(result, collector, *args, **kwargs):
    from prometheus_client import Counter as _PromCounter

    Counter = collector._counter_cls or _PromCounter
    # Kept out of prefill_effective_tokens_total, whose modes sum to the hit-rate
    # denominator; full-KV misses are its input mode minus these.
    collector.prefill_swa_miss_tokens_total = Counter(
        name="sglang:prefill_swa_miss_tokens_total",
        documentation=(
            "Prefix tokens whose full-attention KV was cached on device or "
            "host but recomputed because their SWA window was missing."
        ),
        labelnames=list(collector.labels.keys()),
    )
    return result


def report_swa_miss_tokens(result, reporter, *args, **kwargs):
    prefill_stats = kwargs["prefill_stats"] if "prefill_stats" in kwargs else args[1]
    if not reporter.current_scheduler_metrics_enabled:
        return result
    if not counts_swa_misses(reporter.scheduler.tree_cache):
        return result
    collector = reporter.metrics_collector
    # Zeros are added too, so the series exists once a counting cache has run a
    # prefill batch; dashboard ratios rely on that.
    collector.prefill_swa_miss_tokens_total.labels(**collector.labels).inc(
        _stats_tokens.pop(id(prefill_stats), 0)
    )
    return result


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    policy = "sglang.srt.managers.schedule_policy.PrefillAdder"
    reporter = "sglang.srt.managers.scheduler_components.metrics_reporter"
    return [
        (
            "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.match_prefix",
            record_match,
            HookType.AFTER,
        ),
        (
            f"{policy}._account_prefill_cache_admission",
            charge_admission,
            HookType.AFTER,
        ),
        (f"{policy}.add_chunked_req", charge_continuation_chunk, HookType.AROUND),
        (f"{policy}.add_dllm_staging_req", charge_continuation_chunk, HookType.AROUND),
        (f"{reporter}.PrefillStats.from_adder", snapshot_batch, HookType.AFTER),
        (
            "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.__init__",
            add_swa_miss_counter,
            HookType.AFTER,
        ),
        (
            f"{reporter}.SchedulerMetricsReporter.report_prefill_stats",
            report_swa_miss_tokens,
            HookType.AFTER,
        ),
    ]
