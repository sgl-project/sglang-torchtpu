# SPDX-License-Identifier: Apache-2.0
"""Decode side of the P/D prefix pull: what it can serve and what it keeps.

- decode_pullable: the longest page-aligned prefix of a prompt whose full KV is on
  device and whose SWA window [m - W, m) is resident. It reads the tree past the
  decode's own match cap (the prompt's closing window), since a pull only needs
  the window, not the decode's reuse of it.
- PullHolds: a lock on that prefix from the match until the request's KV lands, so
  a prefill pull never reads freed pages.
- PromptEndAnchors: the window ending at a prompt's page-floored end, kept locked
  by the cache once the prompt is inserted. A later turn that re-renders the reply
  diverges right there, so it is the window a pull needs. Budgeted, least recently
  used released first, and released under memory pressure like CAP media anchors.
"""

from __future__ import annotations

import collections
import logging
import weakref
from array import array
from dataclasses import dataclass
from typing import Any, Optional

from sglang_tpu.disaggregation.prefix_pull import (
    iter_path_segments,
    longest_pullable_len,
    window_start,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PullablePrefix:
    length: int
    # Deepest node of the device match; the pullable prefix lies on its path.
    node_id: Any = None


def qualifies(tree_cache: Any) -> bool:
    """A Python-core unified cache with full + SWA components."""
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return (
        isinstance(tree_cache, UnifiedRadixCache)
        and isinstance(tree_cache.tree_core, UnifiedTreeCore)
        and ComponentType.SWA in tree_cache.tree_components
        # Bigram keys shift every position by one token; not served.
        and not tree_cache.tree_core.is_eagle
    )


def decode_pullable(tree_cache: Any, req: Any) -> PullablePrefix:
    """Longest page-aligned m <= len(prompt) - 1 the decode can serve for req."""
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    if not qualifies(tree_cache):
        return PullablePrefix(0)
    prompt = req.origin_input_ids
    # The prefill computes at least the last prompt token.
    key = RadixKey(
        array("q", prompt[: len(prompt) - 1]),
        req.extra_key,
        is_bigram=False,
        cache_salt=req.cache_salt,
    )
    tree_core = tree_cache.tree_core
    matched_len, node_id, _ = tree_core.match_full_device_prefix(key)
    if matched_len <= 0:
        return PullablePrefix(0)
    length = longest_pullable_len(
        list(iter_path_segments(tree_core, node_id, ComponentType.SWA)),
        limit=matched_len,
        window=tree_cache.sliding_window_size,
        page_size=tree_cache.page_size,
    )
    return PullablePrefix(length, node_id if length > 0 else None)


@dataclass(frozen=True)
class HeldPrefix:
    """Decode pages of a held prefix [0, end), read by a prefill's pull."""

    end: int
    page_size: int
    # Full-attention page ids of [0, end), in token order.
    full_pages: tuple[int, ...]
    # First token of the window ending at end, and its SWA page ids.
    window_start: int
    swa_pages: tuple[int, ...]

    def full_pages_from(self, start: int) -> list[int]:
        return list(self.full_pages[start // self.page_size :])


def _page_ids(token_indices: Any, page_size: int) -> tuple[int, ...]:
    return tuple(int(i) // page_size for i in token_indices[::page_size].tolist())


def _held_prefix(tree_cache: Any, node: Any, end: int) -> HeldPrefix:
    import torch

    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    page_size = tree_cache.page_size
    tree_core = tree_cache.tree_core
    full = tree_core.collect_full_device_indices(node.id, tree_cache.root_node_handle())
    first = window_start(end, tree_cache.sliding_window_size, page_size)
    chunks = []
    node_end = end
    cur = node
    while cur.parent is not None and node_end > first:
        node_start = node_end - len(cur.key)
        value = cur.component_data[ComponentType.SWA].value
        lo = max(first, node_start) - node_start
        chunks.append(value[lo:])
        node_end, cur = node_start, cur.parent
    swa = torch.cat(list(reversed(chunks)))
    return HeldPrefix(
        end=end,
        page_size=page_size,
        full_pages=_page_ids(full, page_size),
        window_start=first,
        swa_pages=_page_ids(swa, page_size),
    )


def _node_ending_at(tree_cache: Any, node_id: Any, position: int) -> Optional[Any]:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    swa = tree_cache.components[ComponentType.SWA]
    return swa.node_ending_at(tree_cache.tree_core.node_by_id(node_id), position)


class PullHolds:
    """Per-request locks on the pullable prefix, keyed by rid; the prefill's pull
    finds a hold's pages by the request's bootstrap room."""

    def __init__(self) -> None:
        self._holds: dict[str, tuple[Any, Any, Any]] = {}
        # Read by the Raiden listener thread; entries are replaced, never mutated.
        self._by_room: dict[int, HeldPrefix] = {}
        self._room_of: dict[str, int] = {}

    def prefix_for_room(self, room: int) -> Optional[HeldPrefix]:
        return self._by_room.get(room)

    def __contains__(self, rid: str) -> bool:
        return rid in self._holds

    def acquire(self, tree_cache: Any, req: Any, pullable: PullablePrefix) -> bool:
        self.release(req.rid)
        if pullable.length <= 0:
            return False
        node = _node_ending_at(tree_cache, pullable.node_id, pullable.length)
        if node is None:
            return False
        receipt = tree_cache.inc_lock_ref(node.id).to_dec_params()
        self._holds[req.rid] = (tree_cache, node.id, receipt)
        if req.bootstrap_room is not None:
            self._by_room[req.bootstrap_room] = _held_prefix(
                tree_cache, node, pullable.length
            )
            self._room_of[req.rid] = req.bootstrap_room
        return True

    def release(self, rid: str) -> None:
        room = self._room_of.pop(rid, None)
        if room is not None:
            self._by_room.pop(room, None)
        held = self._holds.pop(rid, None)
        if held is not None:
            tree_cache, node_id, receipt = held
            tree_cache.dec_lock_ref(node_id, receipt)

    def forget_cache(self, tree_cache: Any) -> None:
        """The tree was reset; its locks went with it."""
        for rid in [r for r, held in self._holds.items() if held[0] is tree_cache]:
            del self._holds[rid]
            room = self._room_of.pop(rid, None)
            if room is not None:
                self._by_room.pop(room, None)


class PromptEndAnchors:
    """Cache-owned locks on prompt-end windows, least recently used first."""

    def __init__(self) -> None:
        self._by_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _anchors(self, tree_cache: Any) -> collections.OrderedDict:
        anchors = self._by_cache.get(tree_cache)
        if anchors is None:
            anchors = self._by_cache[tree_cache] = collections.OrderedDict()
        return anchors

    def __len__(self) -> int:
        return sum(len(a) for a in self._by_cache.values())

    def held(self, tree_cache: Any) -> list:
        return list(self._anchors(tree_cache))

    def pin(self, tree_cache: Any, node_id: Any, budget: int) -> None:
        anchors = self._anchors(tree_cache)
        if node_id in anchors:
            anchors.move_to_end(node_id)
            return
        # A later turn of the same conversation supersedes the anchors above it.
        node = tree_cache.tree_core.node_by_id(node_id).parent
        while node is not None:
            receipt = anchors.pop(node.id, None)
            if receipt is not None:
                tree_cache.dec_lock_ref(node.id, receipt)
            node = node.parent
        anchors[node_id] = tree_cache.inc_lock_ref(node_id).to_dec_params()
        while len(anchors) > budget:
            self.release_lru(tree_cache)

    def release_lru(self, tree_cache: Any) -> bool:
        anchors = self._by_cache.get(tree_cache)
        if not anchors:
            return False
        node_id, receipt = anchors.popitem(last=False)
        tree_cache.dec_lock_ref(node_id, receipt)
        return True

    def forget_cache(self, tree_cache: Any) -> None:
        self._by_cache.pop(tree_cache, None)


def prompt_end_anchor_node(tree_cache: Any, req: Any) -> Optional[Any]:
    """The node ending at req's page-floored prompt end, if its window is resident."""
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    if req.last_node is None or not qualifies(tree_cache):
        return None
    page_size = tree_cache.page_size
    prompt_end = len(req.origin_input_ids) // page_size * page_size
    tree_core = tree_cache.tree_core
    segments = list(iter_path_segments(tree_core, req.last_node, ComponentType.SWA))
    if not segments or segments[-1].end < prompt_end or prompt_end <= 0:
        return None
    window = tree_cache.sliding_window_size
    if longest_pullable_len(segments, prompt_end, window, page_size) != prompt_end:
        return None
    return _node_ending_at(tree_cache, req.last_node, prompt_end)


pull_holds = PullHolds()
prompt_end_anchors = PromptEndAnchors()
