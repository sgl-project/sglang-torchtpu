"""Shared host-cache directory operations for victim spill and prefix recall."""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase

_FULL = ComponentType.FULL
_SWA = ComponentType.SWA


def _host_pages(
    root: Any, host_pool: Any, leaf: Any, *, component=_FULL, tail_tokens=None
) -> Optional[List[int]]:
    """Read a host-backed path (or its trailing window), in token order.

    A window may start inside a radix node; only its intersecting suffix is
    exported. Pending backups cannot serve either component yet.
    """
    segments = []
    remaining = tail_tokens
    node = leaf
    while node is not root and (remaining is None or remaining > 0):
        data = node.component_data[component]
        if data.host_value is None or node.write_through_pending_id is not None:
            return None
        count = len(node.key) if remaining is None else min(remaining, len(node.key))
        segments.append(host_pool._to_page_ids(data.host_value[-count:]))
        if remaining is not None:
            remaining -= count
        node = node.parent
    if remaining:
        return None
    return [int(p) for segment in reversed(segments) for p in segment]


def _swa_host_pool(cache: Any) -> Any:
    if not cache.supports_swa():
        return None
    return cache.host_pool_group.get_pool(PoolName.SWA)


def _window_tokens(cache: Any, page_size: int) -> int:
    if not cache.supports_swa():
        return 0
    window = cache.components[_SWA].sliding_window_size
    return -(-window // page_size) * page_size


def _page_hashes(tokens: List[int], page_size: int) -> List[str]:
    try:
        from sglang.srt.mem_cache.utils import get_hash_str

        hashes = get_hash_str(tokens, None, page_size)
        return [hashes] if isinstance(hashes, str) else list(hashes)
    except Exception:  # noqa: BLE001 - hashes only feed storage and events
        return [""] * (len(tokens) // page_size)


def _anchor_host_pool(cache: Any) -> Any:
    """The HiCache host pool that holds full KV, and its raiden manager."""
    return cache.cache_controller.mem_pool_host.anchor_entry.host_pool


def insert_host_prefix(
    cache: Any,
    key: RadixKey,
    indices: torch.Tensor,
    swa_indices: Optional[torch.Tensor],
    page_size: int,
    *,
    reused_tokens: int = 0,
) -> Any:
    """Publish settled FULL/SWA receives using HiCache's prefetch commit.

    ``indices[:reused_tokens]`` is an already-owned, host-pinned prefix. Only
    newly allocated duplicate slices may be released here. Neither component
    is made visible before the caller has agreed on native receive completion.
    """
    core = cache.tree_core
    full_pool = _anchor_host_pool(cache)
    result = core.insert_host(
        core.root_node.id,
        key,
        indices,
        _page_hashes(list(key.token_ids[: len(key)]), page_size),
    )
    cache._apply_cache_actions(result.cache_actions)
    if result.host_insert_dropped or result.inserted_host_node is None:
        full_pool.free(indices[reused_tokens:])
        if swa_indices is not None:
            _swa_host_pool(cache).free(swa_indices)
        return result
    duplicates = max(reused_tokens, result.prefix_len)
    if duplicates > reused_tokens:
        full_pool.free(indices[reused_tokens:duplicates])
    if swa_indices is not None:
        actions = []
        core.commit_hicache_transfers(
            core.root_node.id,
            CacheTransferPhase.PREFETCH,
            {_SWA: [PoolTransfer(name=PoolName.SWA, host_indices=swa_indices)]},
            cache_actions=actions,
            insert_result=result,
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=len(indices) // page_size,
                extra_pool_hit_pages={PoolName.SWA: len(swa_indices) // page_size},
            ),
        )
        cache._apply_cache_actions(actions)
    return result
