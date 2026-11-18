# SPDX-License-Identifier: Apache-2.0
"""Count device evictions by the pool that ran short: KV pages or state slots.

Why: sglang:evicted_tokens_total does not say whether the KV pool or the
     recurrent-state (GDN/KDA) checkpoint slots forced an eviction, which decides
     whether more KV pool, more state slots, or a host tier would raise the hit
     rate of a hybrid linear-attention model.
How: AFTER RadixCacheMetricsCollector.__init__ adds the counters
     sglang:evicted_tokens_by_cause_total and sglang:evicted_states_by_cause_total
     (label cause = kv | state | kv_and_state); AFTER UnifiedRadixCache.evict and
     evict_for_alloc add the KV tokens and states a call freed under the cause its
     EvictParams asked for.
Upstream PR or issue: none.
Remove when: core labels evictions by the pool that ran short.
"""

from __future__ import annotations


def eviction_cause(params) -> str | None:
    """``kv``, ``state`` or ``kv_and_state`` from what the call asked to free."""
    kv = (params.num_tokens or 0) > 0 or (params.swa_num_tokens or 0) > 0
    state = (params.mamba_num or 0) > 0
    if kv and state:
        return "kv_and_state"
    return "kv" if kv else "state" if state else None


def add_cause_counters(result, collector, *args, **kwargs):
    from prometheus_client import Counter as _PromCounter

    Counter = collector._counter_cls or _PromCounter
    labelnames = [*collector.labels.keys(), "cause"]
    collector.evicted_tokens_by_cause = Counter(
        name="sglang:evicted_tokens_by_cause",
        documentation="Device KV token slots freed by eviction, by the pool whose "
        "shortfall asked for it (kv, state = recurrent-state slots, or both).",
        labelnames=labelnames,
    )
    collector.evicted_states_by_cause = Counter(
        name="sglang:evicted_states_by_cause",
        documentation="Recurrent-state slots freed by eviction, by the pool whose "
        "shortfall asked for it.",
        labelnames=labelnames,
    )
    return result


def count_eviction(result, cache, *args, **kwargs):
    collector = getattr(cache, "metrics_collector", None)
    params = args[0] if args else kwargs.get("params")
    if result is None or params is None or collector is None:
        return result
    if not hasattr(collector, "evicted_tokens_by_cause"):
        return result
    cause = eviction_cause(params)
    if cause is None:
        return result
    labels = {**collector.labels, "cause": cause}
    if result.num_tokens_evicted:
        collector.evicted_tokens_by_cause.labels(**labels).inc(
            result.num_tokens_evicted
        )
    if result.mamba_num_evicted:
        collector.evicted_states_by_cause.labels(**labels).inc(result.mamba_num_evicted)
    return result


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    cache = "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache"
    return [
        (
            "sglang.srt.observability.metrics_collector.RadixCacheMetricsCollector.__init__",
            add_cause_counters,
            HookType.AFTER,
        ),
        (f"{cache}.evict", count_eviction, HookType.AFTER),
        (f"{cache}.evict_for_alloc", count_eviction, HookType.AFTER),
    ]
