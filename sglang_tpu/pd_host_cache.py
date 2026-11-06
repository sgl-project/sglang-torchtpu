"""Shared host-cache directory operations for victim spill and prefix recall."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

_FULL = ComponentType.FULL


def _rank_groups(scheduler: Any) -> List[Any]:
    # The CPU groups whose ranks must make the same tree decisions.
    return [
        group
        for group in (scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group)
        if group is not None and dist.get_world_size(group) > 1
    ]


def _all_ranks_reduce(
    groups: Sequence[Any], rows: List[List[int]], op: Any
) -> List[List[int]]:
    if not groups or not rows:
        return [list(row) for row in rows]
    values = torch.tensor(rows, dtype=torch.int64)
    for group in groups:
        dist.all_reduce(values, op=op, group=group)
    return values.tolist()


def _host_pages(root: Any, host_pool: Any, leaf: Any) -> Optional[List[int]]:
    """The host page ids of a path's full KV in token order, or None unless
    every node of it is backed up. A pending backup cannot be served yet."""
    segments = []
    node = leaf
    while node is not root:
        data = node.component_data[_FULL]
        if data.host_value is None or node.write_through_pending_id is not None:
            return None
        segments.append(host_pool._to_page_ids(data.host_value))
        node = node.parent
    return [int(p) for segment in reversed(segments) for p in segment]


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
    page_size: int,
    *,
    reused_tokens: int = 0,
) -> Any:
    """Publish a settled full-KV receive as a host-only path.

    ``indices[:reused_tokens]`` is an already-owned, host-pinned prefix. Only
    newly allocated duplicate slices may be released here. Nothing is made
    visible before the caller has agreed on native receive completion.
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
        return result
    duplicates = max(reused_tokens, result.prefix_len)
    if duplicates > reused_tokens:
        full_pool.free(indices[reused_tokens:duplicates])
    return result
