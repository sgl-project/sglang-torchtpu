"""Host-tier operations on the unified radix tree for the offload to decode DRAM.

Everything in the offload that reads or changes the tree's host-side internals
lives here, so one module follows the tree's invariants.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from itertools import takewhile
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.common import page_align_floor
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentHostSlot
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase
from sglang.srt.mem_cache.utils import get_hash_str

_FULL = ComponentType.FULL
_SWA = ComponentType.SWA


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


def _path_to_root(cache: Any, node: Any) -> Iterator[Any]:
    """The nodes from ``node`` up to, and excluding, the root."""
    root = cache.tree_core.root_node
    while node is not root:
        yield node
        node = node.parent


def path_tokens(cache: Any, leaf: Any) -> List[int]:
    """The token ids on the path from the root to ``leaf``."""
    path = list(_path_to_root(cache, leaf))
    return [t for node in reversed(path) for t in node.key.token_ids]


def host_page_ids(host_pool: Any, indices: Any) -> List[int]:
    """The host page ids of the host KV ``indices``."""
    return [int(p) for p in host_pool._to_page_ids(indices)]


def host_pages(
    cache: Any,
    host_pool: Any,
    leaf: Any,
    *,
    component: ComponentType = _FULL,
    tail_tokens: Optional[int] = None,
) -> Optional[List[int]]:
    """The host page ids of a path's KV in token order (of its last
    ``tail_tokens`` with them, which may start inside a node), or None unless
    every node of it is backed up. A pending backup cannot be served yet."""
    segments = []
    remaining = tail_tokens
    for node in _path_to_root(cache, leaf):
        if remaining is not None and remaining <= 0:
            break
        data = node.component_data[component]
        if data.host_value is None or node.write_through_pending_id is not None:
            return None
        count = len(node.key) if remaining is None else min(remaining, len(node.key))
        segments.append(host_page_ids(host_pool, data.host_value[-count:]))
        if remaining is not None:
            remaining -= count
    if remaining:
        return None
    return [p for segment in reversed(segments) for p in segment]


def window_host_pages(
    cache: Any, swa_host_pool: Any, leaf: Any, *, tokens: int, window_tokens: int
) -> Optional[List[int]]:
    """The SWA host pages of the window that ends ``tokens`` deep at ``leaf``;
    [] on a full-only model, None if the window is not whole in host RAM."""
    if not window_tokens:
        return []
    return host_pages(
        cache,
        swa_host_pool,
        leaf,
        component=_SWA,
        tail_tokens=min(tokens, window_tokens),
    )


def swa_host_pool(cache: Any) -> Any:
    """The host pool of SWA windows; None on a full-only model."""
    if not cache.supports_swa():
        return None
    return cache.host_pool_group.get_pool(PoolName.SWA)


def window_tokens(cache: Any, page_size: int) -> int:
    """The sliding window in tokens, rounded up to whole pages; 0 on a
    full-only model."""
    if not cache.supports_swa():
        return 0
    window = cache.components[_SWA].sliding_window_size
    return -(-window // page_size) * page_size


def _page_hashes(core: Any, tokens: List[int], page_size: int) -> List[str]:
    """Page hashes for a host insert, computed only when the tree keeps them
    (an L3 storage backend or an external cache linker), as core does for its
    own inserts. Otherwise the native hash extension is never loaded: loading
    it on every recall rebuilt it each time on a host whose ranks share the
    extension's build directory."""
    if not (core.enable_storage or core.enable_external_cache_linker):
        return [""] * (len(tokens) // page_size)
    hashes = get_hash_str(tokens, None, page_size)
    return [hashes] if isinstance(hashes, str) else list(hashes)


def lock_host_path(cache: Any, node: Any) -> List[Any]:
    """Host-lock every node from ``node`` up to the root; returns what
    ``unlock_host_path`` takes. Core's host lock protects one node, and under
    write-back any host eviction reclaims the host copy of an unlocked node
    that is also on the device."""
    return [
        (n.id, cache.inc_host_lock_ref(n.id).to_dec_params())
        for n in _path_to_root(cache, node)
    ]


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
    swa_indices: Optional[torch.Tensor] = None,
    reused_tokens: int = 0,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[Any], int]:
    """Publish a settled receive as a host-only path: its full KV, and with
    ``swa_indices`` the SWA window that ends where it ends. Returns (the id of
    its last node, or None if the tree dropped it; the tokens of ``key`` the
    tree already held). ``indices[:reused_tokens]`` is an already-owned,
    host-pinned prefix and is never freed here; under write-back a received
    page whose node is on the device without a host copy becomes that node's
    backup (``adopt_received_backups``), counted in ``stats``."""
    core = cache.tree_core
    full_pool = anchor_host_pool(cache)
    hashes = _page_hashes(core, list(key.token_ids[: len(key)]), page_size)
    result = core.insert_host(core.root_node.id, key, indices, hashes)
    cache._apply_cache_actions(result.cache_actions)
    node_id = None if result.host_insert_dropped else result.inserted_host_node
    if node_id is None:
        full_pool.free(indices[reused_tokens:])
        if swa_indices is not None:
            swa_host_pool(cache).free(swa_indices)
        return None, result.prefix_len
    duplicates = max(reused_tokens, result.prefix_len)
    if duplicates > reused_tokens:
        adopted = adopt_received_backups(
            cache,
            node_id,
            indices,
            hashes,
            start=reused_tokens,
            end=duplicates,
            page_size=page_size,
        )
        if stats is not None:
            stats["adopted_backup_tokens"] += adopted
    if swa_indices is not None:
        attach_window_to_path(
            cache,
            result,
            full_pages=len(indices) // page_size,
            swa_indices=swa_indices,
            page_size=page_size,
        )
    return node_id, result.prefix_len


def attach_window_to_path(
    cache: Any,
    insert_result: Any,
    *,
    full_pages: int,
    swa_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Make the received window (``swa_indices``, in the SWA host pool) the
    host SWA copy of the path ``insert_result`` just inserted, through
    HiCache's prefetch commit, which is how upstream attaches fetched windows."""
    core = cache.tree_core
    actions = []
    core.commit_hicache_transfers(
        core.root_node.id,
        CacheTransferPhase.PREFETCH,
        {_SWA: [PoolTransfer(name=PoolName.SWA, host_indices=swa_indices)]},
        cache_actions=actions,
        insert_result=insert_result,
        pool_storage_result=PoolTransferResult(
            kv_hit_pages=full_pages,
            extra_pool_hit_pages={PoolName.SWA: len(swa_indices) // page_size},
        ),
    )
    # The commit drops the slices the path already holds a window for. Core
    # releases them through the L3 release queue, which an engine without a
    # storage backend never drains: free them here. They were never visible
    # and their read has settled.
    kept = []
    for action in actions:
        if isinstance(action, FreeComponentHostSlot) and action.component_type == _SWA:
            cache.components[_SWA].free_host_values(action.host_indices)
        else:
            kept.append(action)
    cache._apply_cache_actions(kept)


def adopt_received_backups(
    cache: Any,
    end_node_id: Any,
    indices: torch.Tensor,
    hashes: List[str],
    *,
    start: int,
    end: int,
    page_size: int,
) -> int:
    """Make ``indices[start:end]`` the host backup of the matched path's nodes
    that are on the device without a host copy, and free the pages of nodes
    already backed up; returns the tokens adopted. A host-only branch below a
    device-only node is the one shape write-back cannot evict under host
    pressure, and core keeps a received copy only for a node already backed."""
    core = cache.tree_core
    full_pool = anchor_host_pool(cache)
    path = list(_path_to_root(cache, core.node_by_id(end_node_id)))
    adopted = 0
    offset = 0
    for node in reversed(path):
        length = len(node.key)
        lo, hi = max(offset, start), min(offset + length, end)
        offset += length
        if lo >= hi:
            if offset >= end:
                break
            continue
        data = node.component_data[_FULL]
        if (
            core.is_write_back
            and lo == offset - length
            and hi == offset
            and data.value is not None
            and data.host_value is None
            and node.write_through_pending_id is None
        ):
            core.commit_backup(node.id, indices[lo:hi], {})
            if node.hash_value is None:
                node.hash_value = hashes[lo // page_size : hi // page_size]
            marked = core.mark_write_through_pending([node.id], ack_id=node.id)
            core.finish_write_through(marked, node.id)
            core._update_evictable_leaf_sets(node)
            if node.parent is not None:
                core._update_evictable_leaf_sets(node.parent)
            adopted += hi - lo
        else:
            full_pool.free(indices[lo:hi])
        if offset >= end:
            break
    return adopted


def reclaim_full_host_duplicates(cache: Any, num_tokens: int) -> int:
    """Free FULL host copies of pages the device still holds, least recently
    used first, for a recall landing; returns the tokens freed. A node is taken
    only while every child is device-only: a reclaimed node loses its backup,
    so write-through eviction later deletes it, which would orphan a host-only
    child or leave a host child below a tombstone."""
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
    """Make the path of ``anchor_id`` (its subtree and single-child ancestors)
    the oldest in every eviction order, each node with a unique stamp counting
    up from ``stamp`` since heap ties pop in per-rank set order; returns the
    last stamp used."""
    core = cache.tree_core
    anchor = core.node_by_id(anchor_id)
    nodes = []
    stack = [anchor]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children.values())
    nodes.extend(
        takewhile(lambda n: len(n.children) == 1, _path_to_root(cache, anchor.parent))
    )
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
    indices = [n.component_data[_FULL].host_value for n in _path_to_root(cache, node)]
    return node, have, indices[::-1]


def deepest_host_backed_path(
    cache: Any,
    host_pool: Any,
    node_id: Any,
    *,
    swa_host_pool: Any = None,
    window_tokens: int = 0,
    page_size: int = 1,
):
    """Walking up from ``node_id``: the first node whose whole path has its
    FULL KV in host RAM and, on a hybrid model, the window that ends there
    too; with that path's host page ids and window page ids. (root, None, [])
    if there is none."""
    core = cache.tree_core
    for node in _path_to_root(cache, core.node_by_id(node_id)):
        pages = host_pages(cache, host_pool, node)
        if pages:
            window = window_host_pages(
                cache,
                swa_host_pool,
                node,
                tokens=len(pages) * page_size,
                window_tokens=window_tokens,
            )
            if window is not None:
                return node, pages, window
    return core.root_node, None, []


def deletion_orphans_host_only_child(cache: Any, node_id: Any) -> Optional[bool]:
    """Whether a write-through eviction of ``node_id`` would delete an
    unbacked node that still has a host-only child (a host copy reclaimed for
    a landing leaves such a leaf); None once the tree dropped the node. Under
    write-back the upstream step backs the node up before the demote."""
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
