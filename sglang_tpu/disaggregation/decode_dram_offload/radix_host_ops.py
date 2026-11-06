"""Host-tier operations on the unified radix tree for the offload to decode DRAM.

Everything in the offload that reads or changes the tree's host-side internals
lives here, so one module follows the tree's invariants.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.utils import get_hash_str

_FULL = ComponentType.FULL


def depth_tokens(cache: Any, node: Any) -> int:
    """Tokens on the path from the root to ``node``."""
    root = cache.tree_core.root_node
    depth = 0
    while node is not root:
        depth += len(node.key)
        node = node.parent
    return depth


def path_tokens(cache: Any, leaf: Any) -> List[int]:
    """The token ids on the path from the root to ``leaf``."""
    root = cache.tree_core.root_node
    parts = []
    node = leaf
    while node is not root:
        parts.append(node.key.token_ids)
        node = node.parent
    tokens = []
    for part in reversed(parts):
        tokens.extend(part)
    return tokens


def host_pages(cache: Any, host_pool: Any, leaf: Any) -> Optional[List[int]]:
    """The host page ids of a path's full KV in token order, or None unless
    every node of it is backed up. A pending backup cannot be served yet."""
    root = cache.tree_core.root_node
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
    # The hashes only feed L3 storage and cache events, neither of which the
    # offload uses; without the native hash extension they are left empty.
    try:
        hashes = get_hash_str(tokens, None, page_size)
    except RuntimeError:
        return [""] * (len(tokens) // page_size)
    return [hashes] if isinstance(hashes, str) else list(hashes)


def lock_host_path(cache: Any, node: Any) -> List[Any]:
    """Host-lock every node from ``node`` up to the root, for pages of the
    whole path that a peer reads or that an insert will reuse. Core's host
    lock protects one node, and under write-back any host eviction reclaims
    the host copy of an unlocked node that is also on the device. Returns
    what ``unlock_host_path`` takes."""
    root = cache.tree_core.root_node
    locks = []
    while node is not root:
        locks.append((node.id, cache.inc_host_lock_ref(node.id).to_dec_params()))
        node = node.parent
    return locks


def unlock_host_path(cache: Any, locks: List[Any]) -> None:
    for node_id, receipt in locks:
        cache.dec_host_lock_ref(node_id, receipt)


def anchor_host_pool(cache: Any) -> Any:
    """The HiCache host pool that holds full KV, and its raiden manager."""
    return cache.cache_controller.mem_pool_host.anchor_entry.host_pool


def insert_host_prefix(
    cache: Any,
    key: RadixKey,
    indices: torch.Tensor,
    page_size: int,
) -> Any:
    """Publish a settled full-KV receive as a host-only path, freeing the
    slices the tree already held. Nothing is made visible before the caller
    has agreed on native receive completion.
    """
    core = cache.tree_core
    full_pool = anchor_host_pool(cache)
    result = core.insert_host(
        core.root_node.id,
        key,
        indices,
        _page_hashes(list(key.token_ids[: len(key)]), page_size),
    )
    cache._apply_cache_actions(result.cache_actions)
    if result.host_insert_dropped or result.inserted_host_node is None:
        full_pool.free(indices)
        return result
    if result.prefix_len:
        full_pool.free(indices[: result.prefix_len])
    return result


def leaf_of(cache: Any, node_id: Any) -> Any:
    """The node ``node_id`` when it is a leaf below the root, else None."""
    core = cache.tree_core
    node = core.node_by_id(node_id)
    if node is core.root_node or node.children:
        return None
    return node


def is_live_host_backed_leaf(cache: Any, node: Any) -> Optional[bool]:
    """Whether ``node`` is still a leaf of the tree with a FULL host copy;
    None once the tree dropped it."""
    try:
        if cache.tree_core.node_by_id(node.id) is not node:
            return False
    except KeyError:
        return None
    return not node.children and node.component_data[_FULL].host_value is not None


def mark_next_victim(cache: Any, anchor_id: Any, stamp: float) -> float:
    """Make the path of ``anchor_id`` the oldest in every eviction order, below
    any branch other sessions share: its subtree and its single-child
    ancestors. Returns the last stamp used.

    Device eviction heaps order leaves by last_access_time and break ties by
    set iteration order, which differs per rank, so every node gets a unique
    stamp, counting up from ``stamp``; the caller keeps them below any live
    node's counter value.
    """
    core = cache.tree_core
    anchor = core.node_by_id(anchor_id)
    nodes = []
    stack = [anchor]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children.values())
    node = anchor.parent
    while node is not core.root_node and len(node.children) == 1:
        nodes.append(node)
        node = node.parent
    lrus = [*core.lru_lists.values(), *core.host_lru_lists.values()]
    for node in nodes:
        stamp += 1.0
        node.last_access_time = stamp
        for lru in lrus:
            if lru.in_list(node):
                lru.reset_node_lru(node)
    return stamp
