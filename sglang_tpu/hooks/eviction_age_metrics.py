# SPDX-License-Identifier: Apache-2.0
"""Report how long evicted cache entries went unused, per device and host pool.

Why: under LRU each pool forgets an idle session after a time limit, and a
     returning turn hits only if its gap is shorter than the limit of the pool
     holding its last copy. Little's law (cached tokens / eviction rate) counts
     the time an entry was being reused too, so it overstates that limit; the
     time since last use at eviction measures it. Splitting evictions by whether
     another tier keeps a copy shows which pool bounds hits, for example host
     rather than device under HiCache write_through. Entries moved to the
     eviction end because no request needs them (stale SWA windows) leave early
     and short-aged, in both tiers, so they are labeled stale and kept out of
     the limit.
How: the tree core keeps wall-clock recency stamps (UnifiedTreeNode.last_access_wall
     and lru_wall). AFTER UnifiedTreeCore.evict_device_start and
     evict_device_end, and AROUND UnifiedTreeCore.drive_host_eviction, track
     which component's walk is running. AFTER
     UnifiedTreeCore._evict_component_and_detach_lru records each eviction by
     that component's own walk, so cascade frees of other components are left
     out. AFTER UnifiedLRUList.reset_node_lru records when a node was moved to
     the eviction end, and AFTER UnifiedTreeCore._split_node gives the new
     prefix node its mark, and AFTER TreeComponent._refresh_session_partition
     drops it; it is stale at eviction if not used since. AFTER
     UnifiedRadixCache.update_eviction_metrics (the end of every device
     eviction, including replacements of _evict) and evict_host add the totals
     to two counters, which an AFTER hook on RadixCacheMetricsCollector.__init__
     creates.
     Python TreeCore only; full KV and SWA pools only.
Upstream PR or issue: none.
Remove when: core reports eviction age per pool.
"""

from __future__ import annotations

import time
import weakref

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

_FULL = ComponentType.FULL
_POOL_NAMES = {ComponentType.FULL: "kv", ComponentType.SWA: "swa"}

# Plugin-side state per tree core, keyed weakly so it dies with the cache.
# _walks maps a tree core to (component_type, is_host_walk) of the running walk.
_walks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_pending: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Node -> {LRU slot: wall time of its last move to that list's eviction end}.
_demoted_at: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _pool_name(component_type) -> str | None:
    return _POOL_NAMES.get(component_type)


def last_use(tree_core, node, component_type) -> float:
    if component_type == _FULL:
        return node.last_access_wall
    # SWA recency is its device LRU slot. Entering the host LRU when the device
    # copy is evicted is not a use, so host victims are aged from the device slot.
    return node.lru_wall[tree_core.lru_lists[component_type]._pt]


def _is_stale(tree_core, node, component_type, last_used: float) -> bool:
    pt = tree_core.lru_lists[component_type]._pt
    demoted = _demoted_at.get(node, {}).get(pt)
    return demoted is not None and demoted >= last_used


def _add(
    tree_core, pool: str, last_copy: bool, stale: bool, tokens: int, age: float
) -> None:
    totals = _pending.setdefault(tree_core, {})
    entry = totals.setdefault((pool, last_copy, stale), [0, 0.0])
    entry[0] += tokens
    entry[1] += tokens * max(0.0, age)


def start_device_walk(result, tree_core, component_type, *args, **kwargs):
    _walks[tree_core] = (component_type, False)
    return result


def end_device_walk(result, tree_core, *args, **kwargs):
    _walks.pop(tree_core, None)
    return result


def run_host_walk(original, tree_core, component_type, *args, **kwargs):
    # A host walk can run inside a device walk; restore the outer walk after it.
    outer = _walks.get(tree_core)
    _walks[tree_core] = (component_type, True)
    try:
        return original(tree_core, component_type, *args, **kwargs)
    finally:
        if outer is None:
            _walks.pop(tree_core, None)
        else:
            _walks[tree_core] = outer


def record_demotion(result, lru, node, *args, **kwargs):
    # The list leaves a session-referenced node in place; that is no demotion.
    if lru._is_referenced is None or not lru._is_referenced(node):
        _demoted_at.setdefault(node, {})[lru._pt] = time.monotonic()
    return result


def copy_demotion_to_split_prefix(result, tree_core, key, child, *args, **kwargs):
    marks = _demoted_at.get(child)
    if marks:
        _demoted_at[result[0]] = dict(marks)
    return result


def clear_demotion_on_session_move(result, comp, node, *args, **kwargs):
    # A session-coverage move takes the node off the eviction end.
    marks = _demoted_at.get(node)
    if marks:
        marks.pop(comp.tree_core.lru_lists[comp.component_type]._pt, None)
    return result


def record_eviction(result, tree_core, node, comp, *args, **kwargs):
    component_type = comp.component_type
    walk = _walks.get(tree_core)
    if walk is None or walk[0] != component_type:
        return result
    is_host_walk = walk[1]
    pool = _pool_name(component_type)
    if pool is None:
        return result
    device_freed, host_freed = result
    cd = node.component_data[component_type]
    last_used = last_use(tree_core, node, component_type)
    age = time.monotonic() - last_used
    # Read from the device slot's mark, which outlives the device copy,
    # so a window demoted on device is stale when host evicts it too.
    stale = _is_stale(tree_core, node, component_type, last_used)
    # last_copy: no other tier still holds the entry, so the eviction forgets it.
    if device_freed > 0 and not is_host_walk:
        _add(tree_core, pool, cd.host_value is None, stale, device_freed, age)
    if host_freed > 0 and is_host_walk:
        _add(tree_core, f"host_{pool}", cd.value is None, stale, host_freed, age)
    return result


def flush(result, cache, *args, **kwargs):
    totals = _pending.pop(cache.tree_core, None)
    collector = cache.metrics_collector
    if not totals or collector is None:
        return result
    for (pool, last_copy, stale), (tokens, token_seconds) in totals.items():
        labels = dict(
            collector.labels,
            pool=pool,
            last_copy=str(last_copy).lower(),
            stale=str(stale).lower(),
        )
        collector.eviction_age_tokens_total.labels(**labels).inc(tokens)
        collector.eviction_age_token_seconds_total.labels(**labels).inc(token_seconds)
    return result


def add_eviction_age_counters(result, collector, *args, **kwargs):
    from prometheus_client import Counter as _PromCounter

    Counter = collector._counter_cls or _PromCounter
    labelnames = list(collector.labels.keys()) + ["pool", "last_copy", "stale"]
    # Mean age at eviction is token_seconds_total / tokens_total. With LRU it is
    # the pool's idle limit; last_copy="false" evictions leave a copy in the
    # other tier and do not cost hits, and stale="true" ones were demoted early.
    collector.eviction_age_tokens_total = Counter(
        name="sglang:eviction_age_tokens_total",
        documentation="Token slots evicted by each pool's own eviction walk "
        "(pool: kv, swa, host_kv, host_swa; cascade frees of other pools "
        "excluded); last_copy says whether no other tier still holds them; "
        "stale says whether they were moved to the eviction end, unused since.",
        labelnames=labelnames,
    )
    collector.eviction_age_token_seconds_total = Counter(
        name="sglang:eviction_age_token_seconds_total",
        documentation="Sum over evicted token slots of the wall time since "
        "their last use (full KV: last access of the node; SWA: last move to "
        "the most-recent end of the device SWA LRU).",
        labelnames=labelnames,
    )
    return result


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    core = "sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore"
    lru = "sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedLRUList"
    cache = "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache"
    return [
        (f"{lru}.reset_node_lru", record_demotion, HookType.AFTER),
        (f"{core}._split_node", copy_demotion_to_split_prefix, HookType.AFTER),
        (
            "sglang.srt.mem_cache.unified_cache.components.base.TreeComponent."
            "_refresh_session_partition",
            clear_demotion_on_session_move,
            HookType.AFTER,
        ),
        (f"{core}.evict_device_start", start_device_walk, HookType.AFTER),
        (f"{core}.evict_device_end", end_device_walk, HookType.AFTER),
        (f"{core}.drive_host_eviction", run_host_walk, HookType.AROUND),
        (f"{core}._evict_component_and_detach_lru", record_eviction, HookType.AFTER),
        (f"{cache}.update_eviction_metrics", flush, HookType.AFTER),
        (f"{cache}.evict_host", flush, HookType.AFTER),
        (
            "sglang.srt.observability.metrics_collector.RadixCacheMetricsCollector.__init__",
            add_eviction_age_counters,
            HookType.AFTER,
        ),
    ]
