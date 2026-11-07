"""Host-tier operations on the unified radix tree for the offload to decode DRAM.

Everything in the offload that reads or changes the tree's host-side internals
lives here, so one module follows the tree's invariants.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.common import page_align_floor
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.utils import get_hash_str

_FULL = ComponentType.FULL


def prefix_radix_key(*, req: Any, num_tokens: int) -> RadixKey:
    """The tree key of the request's first ``num_tokens`` prompt tokens of KV."""
    return RadixKey(
        token_ids=array("q", req.origin_input_ids[:num_tokens]),
        extra_key=req.extra_key,
        cache_salt=req.cache_salt,
    )


def offer_limit(*, req: Any, page_size: int) -> int:
    """Longest prompt prefix worth offering: the prefill computes at least the
    last prompt token."""
    return page_align_floor(len(req.origin_input_ids) - 1, page_size)


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
    *,
    reused_tokens: int = 0,
) -> Any:
    """Publish a settled full-KV receive as a host-only path.

    ``indices[:reused_tokens]`` is an already-owned, host-pinned prefix. Only
    newly allocated duplicate slices may be released here. Nothing is made
    visible before the caller has agreed on native receive completion.
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
        full_pool.free(indices[reused_tokens:])
        return result
    duplicates = max(reused_tokens, result.prefix_len)
    if duplicates > reused_tokens:
        full_pool.free(indices[reused_tokens:duplicates])
    return result


def reclaim_full_host_duplicates(cache: Any, num_tokens: int) -> int:
    """Free FULL host copies of pages the device still holds, least recently
    used first, to make room for a recall landing. Returns the tokens freed.

    Host eviction only takes host-only leaves, and core reclaims duplicates
    only under write-back. A reclaimed node loses its backup, so write-through
    eviction later deletes it instead of demoting it. A node is therefore taken
    only while every child of it is device-only: a child without a device copy
    would be orphaned by that deletion, and a child with a host copy would sit
    below a host tombstone, which the next host insert on the path rejects.
    """
    core = cache.tree_core
    tracker: dict = defaultdict(int)
    device_frees: dict = defaultdict(list)
    host_frees: dict = defaultdict(list)
    taken = set()
    # Taking a tip makes its parent eligible: repeat the LRU pass until the
    # target is met or a pass takes nothing.
    while tracker[_FULL] < num_tokens:
        progress = False
        for node in sorted(
            core.full_host_duplicates.values(), key=lambda n: n.last_access_time
        ):
            if tracker[_FULL] >= num_tokens:
                break
            if (
                node.id in taken
                or not core._can_reclaim_full_host_duplicate(node)
                or any(
                    child.component_data[_FULL].value is None
                    or child.component_data[_FULL].host_value is not None
                    for child in node.children.values()
                )
            ):
                continue
            core._release_full_host_duplicate(node, tracker, device_frees, host_frees)
            taken.add(node.id)
            progress = True
        if not progress:
            break
    for node_id in taken:
        core.full_host_duplicates.pop(node_id, None)
    cache._free_values(device_frees, host_frees)
    return int(tracker[_FULL])


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


def host_backed_prefix(cache: Any, key: RadixKey, num_tokens: int, page_size: int):
    """The longest prefix of ``key``, up to ``num_tokens``, whose FULL KV is a
    contiguous, completed host copy: (its last node, its length in tokens, its
    host indices in token order). Device-only pages cannot be used as host
    indices or anchor an ordinary host insertion."""
    core = cache.tree_core
    node, have = core.root_node, 0
    while have < num_tokens:
        child = node.children.get(key.child_key_at(have, page_size))
        if child is None or child.key.match(key[have:], page_size=page_size) != len(
            child.key
        ):
            break
        if (
            child.component_data[_FULL].host_value is None
            or child.write_through_pending_id is not None
        ):
            break
        node, have = child, have + len(child.key)
    indices = []
    cursor = node
    while cursor is not core.root_node:
        indices.append(cursor.component_data[_FULL].host_value)
        cursor = cursor.parent
    return node, have, list(reversed(indices))


def deepest_host_backed_path(cache: Any, host_pool: Any, node_id: Any):
    """Walking up from ``node_id``: the first node whose whole path has its
    FULL KV in host RAM, and that path's host page ids; (root, None) if none."""
    core = cache.tree_core
    node = core.node_by_id(node_id)
    while node is not core.root_node:
        pages = host_pages(cache, host_pool, node)
        if pages:
            return node, pages
        node = node.parent
    return node, None


def deletion_orphans_host_only_child(cache: Any, node_id: Any) -> Optional[bool]:
    """Whether a write-through eviction of ``node_id`` would delete an
    unbacked node that still has a host-only child; None once the tree dropped
    the node.

    Reclaiming a host copy for a recall landing leaves a device leaf without a
    backup, which write-through eviction deletes outright. Under write_back the
    upstream step backs the node up before the demote, so the child keeps a
    backed ancestor.
    """
    try:
        node = cache.tree_core.node_by_id(node_id)
    except KeyError:
        return None
    return (
        not cache.is_write_back
        and not node.backuped
        and any(
            child.component_data[_FULL].value is None
            for child in node.children.values()
        )
    )
