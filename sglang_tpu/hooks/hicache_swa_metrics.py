# SPDX-License-Identifier: Apache-2.0
"""Export the HiCache host SWA pool's used and total tokens.

Why: sglang:hicache_host_used_tokens and sglang:hicache_host_total_tokens cover
     only the full-KV host pool, so a hybrid SWA model's host SWA occupancy is
     invisible when sizing --hicache-size or --swa-full-tokens-ratio.
How: AFTER SchedulerMetricsCollector.__init__ adds the gauges
     sglang:hicache_host_swa_used_tokens and sglang:hicache_host_swa_total_tokens
     when HiCache is on; AFTER SchedulerMetricsReporter._log_hicache_stats sets
     them when the cache has an SWA host pool.
Upstream PR or issue: none.
Remove when: core exports host SWA pool occupancy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.base import HostKVCache


def swa_host_pool(tree_cache) -> HostKVCache | None:
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if not isinstance(tree_cache, UnifiedRadixCache):
        return None
    if not tree_cache.tree_core.has_swa_host_pool:
        return None
    return tree_cache.host_pool_group.get_pool(PoolName.SWA)


def used_tokens(pool: HostKVCache) -> int:
    from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool

    if isinstance(pool, UnifiedPageEnvelopeHostPool):
        # Free bytes are shared with the peer side, so count this side's live pages.
        backing = pool._shared_backing
        with backing.lock:
            side = backing.sides[pool.pool_label]
            return side.live_page_count * side.page_size
    return pool.logical_size - pool.available_size()


def add_swa_host_gauges(result, collector, *args, **kwargs):
    if not collector.enable_hierarchical_cache:
        return result
    from prometheus_client import Gauge as _PromGauge

    Gauge = collector._gauge_cls or _PromGauge
    collector.hicache_host_swa_used_tokens = Gauge(
        name="sglang:hicache_host_swa_used_tokens",
        documentation="Number of tokens currently used in the host SWA KV cache.",
        labelnames=collector.labels.keys(),
        multiprocess_mode="mostrecent",
    )
    collector.hicache_host_swa_total_tokens = Gauge(
        name="sglang:hicache_host_swa_total_tokens",
        documentation="Logical capacity of the host SWA KV cache in tokens; "
        "on a page-envelope host backing the bytes are shared with full KV.",
        labelnames=collector.labels.keys(),
        multiprocess_mode="mostrecent",
    )
    return result


def log_swa_host_stats(result, reporter, *args, **kwargs):
    if not reporter.scheduler.enable_hierarchical_cache:
        return result
    pool = swa_host_pool(reporter.scheduler.tree_cache)
    if pool is None:
        return result
    collector = reporter.metrics_collector
    collector._log_gauge(collector.hicache_host_swa_used_tokens, used_tokens(pool))
    collector._log_gauge(collector.hicache_host_swa_total_tokens, pool.logical_size)
    return result


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.__init__",
            add_swa_host_gauges,
            HookType.AFTER,
        ),
        (
            "sglang.srt.managers.scheduler_components.metrics_reporter.SchedulerMetricsReporter._log_hicache_stats",
            log_swa_host_stats,
            HookType.AFTER,
        ),
    ]
