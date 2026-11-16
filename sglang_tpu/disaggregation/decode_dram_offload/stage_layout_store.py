"""Decode-side store for the branches of a pipeline-parallel prefill.

Why: the offload moves whole host blocks between two pools with the same block
     layout. A pipeline stage's block holds every KV head of the stage's own
     layers; a tensor-parallel decode rank's block holds one head slice of
     every layer. Neither maps onto the other, so such a decode cannot keep a
     pushed branch in its own host pool, and it could not load it into HBM
     either.
How: the decode keeps one host-only pool per source stage, in that stage's
     layout, and never reads it itself: a page goes back to the stage it came
     from, block for block. Every pool has the same number of pages and one
     page id names the same page in all of them, so this index, which is all
     the ranks have to agree on, is the same on every decode rank whichever
     stages a rank hosts. The branches are outside the decode's radix cache:
     its own prefix match never counts them, and the prefill sends the decode
     the whole prompt as it does for any request.

A branch is stored as a path of pages, shared with the branches that begin the
same way. A hybrid sliding-window model resumes a branch only where a window
ends, so a window (its own pool, its own ids) hangs off the last page of each
pushed length.

This module holds no transport and no torch; ``decide`` methods are
deterministic so that ranks applying the same calls hold the same state.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

# Arbitrary: full pages the store holds per page of sliding window, i.e. a
# stored branch is taken to be this many windows long. A fixed share of the
# budget starved models whose window pages are large (seen live: 5 windows).
DEFAULT_PAGES_PER_WINDOW_PAGE = 4


def hosting_rank(stage: int, decode_ranks: int) -> int:
    """The decode rank that holds the pool of prefill stage ``stage``: stages
    spread over the ranks in order, so over the decode's hosts too."""
    return stage % decode_ranks


def hosted_stages(rank: int, stages: int, decode_ranks: int) -> List[int]:
    return [s for s in range(stages) if hosting_rank(s, decode_ranks) == rank]


@dataclass(frozen=True)
class StagePoolLayout:
    """The block layout of one host pool of one prefill stage: per tensor of a
    page, its dtype name and the shape of one page."""

    stage: int
    stages: int
    tensors: Tuple[Tuple[str, Tuple[int, ...]], ...]
    # Bytes of one page over all tensors.
    page_bytes: int
    window: bool = False

    def fingerprint(self) -> int:
        """A non-negative int64 naming the layout; a store holds one layout
        per pool and declines pages in any other."""
        text = repr(
            (self.stage, self.stages, self.tensors, self.page_bytes, self.window)
        )
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") >> 1


def size_store(
    *,
    budget_bytes: int,
    page_bytes: int,
    window_page_bytes: int = 0,
    window_pages: int = 0,
    pages_per_window_page: int = DEFAULT_PAGES_PER_WINDOW_PAGE,
) -> Tuple[int, int]:
    """(pages, windows) a store of ``budget_bytes`` per decode rank holds.
    ``page_bytes`` and ``window_page_bytes`` are the bytes of one page over the
    stage pools of the rank that hosts the most, so every rank fits. With
    windows, the budget buys whole branches: one window and
    ``pages_per_window_page`` full pages per page of it."""
    if page_bytes <= 0:
        raise ValueError("a stage page has at least one byte")
    if not window_pages:
        return budget_bytes // page_bytes, 0
    branch_pages = pages_per_window_page * window_pages
    branch_bytes = window_pages * window_page_bytes + branch_pages * page_bytes
    windows = budget_bytes // branch_bytes
    return windows * branch_pages, windows


class _Ids:
    """Hands out the lowest free id, so equal histories give equal ids."""

    def __init__(self, count: int) -> None:
        self.count = count
        self._next = 0
        self._freed: List[int] = []

    def free(self) -> int:
        return self.count - self._next + len(self._freed)

    def take(self, n: int) -> List[int]:
        if n > self.free():
            raise ValueError(f"{n} ids wanted, {self.free()} free")
        self._freed.sort(reverse=True)
        out = [self._freed.pop() for _ in range(min(n, len(self._freed)))]
        fresh = n - len(out)
        out.extend(range(self._next, self._next + fresh))
        self._next += fresh
        return out

    def give(self, ids: Sequence[int]) -> None:
        self._freed.extend(ids)


@dataclass(eq=False)
class _Page:
    digest: bytes
    parent: Optional[_Page]
    page_id: int
    depth: int
    children: Dict[bytes, _Page] = field(default_factory=dict)
    # Ids of the window that ends with this page, in the window pool.
    window: Optional[List[int]] = None
    stamp: int = 0
    pins: int = 0
    # Reserved for a landing whose reads have not settled: not offered.
    landing: bool = False


@dataclass
class Landing:
    """Pages reserved for a pushed branch. ``new_pages`` receive the pushed
    pages from ``first_new_page`` on; the store already holds the ones before.
    ``window`` receives the pushed window, None if it is held or none came;
    a branch shorter than a window fills only the first ids of it."""

    end: _Page
    first_new_page: int
    new_pages: List[int]
    window: Optional[List[int]]
    _new_nodes: List[_Page] = field(default_factory=list)


@dataclass
class Match:
    """The stored prefix of a request: its page ids in path order and the
    window that ends with the last of them."""

    end: _Page
    pages: List[int]
    window: List[int]

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def _root_digest(scope: Hashable) -> bytes:
    return hashlib.blake2b(repr(scope).encode(), digest_size=16).digest()


def _page_digests(
    scope: Hashable, tokens: Sequence[int], page_size: int
) -> List[bytes]:
    """One digest per whole page of ``tokens``, each chained to the pages
    before it, under ``scope`` (what else must be equal for KV to be reused)."""
    digest = _root_digest(scope)
    out = []
    for start in range(0, len(tokens) - page_size + 1, page_size):
        page = tokens[start : start + page_size]
        packed = struct.pack(f"<{page_size}q", *page)
        digest = hashlib.blake2b(digest + packed, digest_size=16).digest()
        out.append(digest)
    return out


class StageLayoutStore:
    """The index of the branches a decode keeps in stage-layout pools."""

    def __init__(
        self, *, page_size: int, pages: int, window_pages: int = 0, windows: int = 0
    ) -> None:
        if page_size < 1 or pages < 0 or window_pages < 0 or windows < 0:
            raise ValueError("sizes of a stage-layout store are not negative")
        self.page_size = page_size
        self.window_pages = window_pages
        self._pages = _Ids(pages)
        self._windows = _Ids(windows * window_pages)
        self._roots: Dict[bytes, Dict[bytes, _Page]] = {}
        # Pages without children: where eviction starts.
        self._leaves: set = set()
        # Paths pinned now, by an offer or a landing.
        self._open_pins = 0
        self._clock = 0
        self.stats = dict.fromkeys(
            (
                "landed",
                "landed_pages",
                "reused_pages",
                "evicted_pages",
                "evicted_windows",
                "no_room",
                "busy",
                "aborted",
            ),
            0,
        )

    # ----- sizes ---------------------------------------------------------------

    @property
    def capacity_pages(self) -> int:
        return self._pages.count

    @property
    def used_pages(self) -> int:
        return self._pages.count - self._pages.free()

    @property
    def used_windows(self) -> int:
        if not self.window_pages:
            return 0
        return (self._windows.count - self._windows.free()) // self.window_pages

    def counters(self) -> Dict[str, int]:
        """For the periodic ``[dram-offload]`` line."""
        return dict(
            self.stats,
            pages=self.capacity_pages,
            used_pages=self.used_pages,
            windows=(
                self._windows.count // self.window_pages if self.window_pages else 0
            ),
            used_windows=self.used_windows,
        )

    # ----- landing ---------------------------------------------------------------

    def reserve(
        self, scope: Hashable, tokens: Sequence[int], *, with_window: bool = False
    ) -> Optional[Landing]:
        """Reserve pages for a pushed branch of whole pages, evicting the
        least recently used branches for room. None, with nothing changed
        beyond evictions, when the branch brings nothing new, when it meets a
        landing in flight, or when room cannot be made."""
        if with_window and not self.window_pages:
            raise ValueError("this store holds no sliding windows")
        digests = _page_digests(scope, tokens, self.page_size)
        if not digests or len(digests) * self.page_size != len(tokens):
            raise ValueError("a pushed branch is a positive number of whole pages")
        level = self._roots.setdefault(_root_digest(scope), {})
        path: List[_Page] = []
        for digest in digests:
            node = level.get(digest)
            if node is None:
                break
            if node.landing:
                self.stats["busy"] += 1
                return None
            path.append(node)
            level = node.children
        held = len(path)
        want_window = with_window and not (held == len(digests) and path[-1].window)
        if held == len(digests) and not want_window:
            self._touch(path[-1])
            return None
        # The held prefix must survive the evictions that make room for the rest.
        for node in path:
            node.pins += 1
        try:
            if not self._make_room(
                len(digests) - held, self.window_pages if want_window else 0
            ):
                self.stats["no_room"] += 1
                return None
        finally:
            for node in path:
                node.pins -= 1
        new_pages = self._pages.take(len(digests) - held)
        parent = path[-1] if path else None
        level = parent.children if parent else self._roots[_root_digest(scope)]
        new_nodes = []
        for digest, page_id in zip(digests[held:], new_pages):
            node = _Page(
                digest=digest,
                parent=parent,
                page_id=page_id,
                depth=(parent.depth if parent else 0) + 1,
                landing=True,
            )
            level[digest] = node
            self._leaves.discard(parent)
            self._leaves.add(node)
            new_nodes.append(node)
            parent, level = node, node.children
        end = parent
        # Pinned until the landing commits or aborts.
        self._pin(end)
        return Landing(
            end=end,
            first_new_page=held,
            new_pages=new_pages,
            window=self._windows.take(self.window_pages) if want_window else None,
            _new_nodes=new_nodes,
        )

    def holds(
        self, scope: Hashable, tokens: Sequence[int], *, with_window: bool = False
    ) -> bool:
        """Whether a push of this branch would bring nothing new."""
        level = self._roots.get(_root_digest(scope), {})
        node = None
        for digest in _page_digests(scope, tokens, self.page_size):
            node = level.get(digest)
            if node is None or node.landing:
                return False
            level = node.children
        return node is not None and (not with_window or node.window is not None)

    def commit(self, landing: Landing) -> None:
        """The landing's reads settled on every rank: its pages are offered."""
        for node in landing._new_nodes:
            node.landing = False
        if landing.window is not None:
            landing.end.window = landing.window
        self._unpin(landing.end)
        self._touch(landing.end)
        self.stats["landed"] += 1
        self.stats["landed_pages"] += len(landing.new_pages)
        self.stats["reused_pages"] += landing.first_new_page

    def abort(self, landing: Landing) -> None:
        """A read failed on some rank: give the reserved pages back."""
        self._unpin(landing.end)
        for node in reversed(landing._new_nodes):
            self._drop(node)
        if landing.window is not None:
            self._windows.give(landing.window)
        self.stats["aborted"] += 1

    # ----- recall ----------------------------------------------------------------

    def match(self, scope: Hashable, tokens: Sequence[int]) -> Optional[Match]:
        """The longest stored prefix of ``tokens`` a prefill can resume from:
        with windows, the longest that ends where a stored window ends."""
        level = self._roots.get(_root_digest(scope), {})
        path: List[_Page] = []
        for digest in _page_digests(scope, tokens, self.page_size):
            node = level.get(digest)
            if node is None or node.landing:
                break
            path.append(node)
            level = node.children
        if self.window_pages:
            while path and path[-1].window is None:
                path.pop()
        if not path:
            return None
        self._touch(path[-1])
        return Match(
            end=path[-1],
            pages=[node.page_id for node in path],
            # A branch shorter than a window holds only its own pages of it.
            window=list(path[-1].window or ())[: len(path)],
        )

    def pin(self, match: Match) -> None:
        """Keep an offered prefix until ``unpin``: a prefill may be reading it."""
        self._pin(match.end)

    def unpin(self, match: Match) -> None:
        self._unpin(match.end)

    def has_pins(self) -> bool:
        return self._open_pins > 0

    def clear(self) -> None:
        """Forget every branch; only with nothing pinned or landing."""
        if self.has_pins():
            raise RuntimeError("stage-layout pages are pinned; the engine is not idle")
        self._roots.clear()
        self._leaves.clear()
        self._pages = _Ids(self._pages.count)
        self._windows = _Ids(self._windows.count)

    # ----- internals -------------------------------------------------------------

    def _pin(self, node: Optional[_Page]) -> None:
        self._open_pins += 1
        while node is not None:
            node.pins += 1
            node = node.parent

    def _unpin(self, node: Optional[_Page]) -> None:
        self._open_pins -= 1
        while node is not None:
            if node.pins < 1:
                raise RuntimeError("stage-layout page unpinned more often than pinned")
            node.pins -= 1
            node = node.parent

    def _touch(self, node: Optional[_Page]) -> None:
        self._clock += 1
        while node is not None:
            node.stamp = self._clock
            node = node.parent

    def _drop(self, node: _Page) -> None:
        level = node.parent.children if node.parent else self._root_level(node)
        del level[node.digest]
        self._leaves.discard(node)
        if node.parent is not None and not node.parent.children:
            self._leaves.add(node.parent)
        self._pages.give([node.page_id])
        if node.window is not None:
            self._windows.give(node.window)
            node.window = None

    def _root_level(self, node: _Page) -> Dict[bytes, _Page]:
        for level in self._roots.values():
            if level.get(node.digest) is node:
                return level
        raise RuntimeError("stage-layout page is in no tree")

    def _make_room(self, pages: int, window_ids: int) -> bool:
        """Evict least recently used unpinned leaves until ``pages`` pages and
        ``window_ids`` window ids are free. With windows, a leaf left without
        one cannot be resumed from and goes too."""
        if pages > self._pages.count or window_ids > self._windows.count:
            return False
        while self._pages.free() < pages or self._windows.free() < window_ids:
            leaves = [
                node for node in self._leaves if not node.pins and not node.landing
            ]
            if self._windows.free() < window_ids and self._pages.free() >= pages:
                # Only windows are short: a leaf without one frees none.
                leaves = [node for node in leaves if node.window is not None]
            if not leaves:
                return False
            node = min(leaves, key=lambda n: (n.stamp, n.depth, n.digest))
            while True:
                parent = node.parent
                self.stats["evicted_pages"] += 1
                self.stats["evicted_windows"] += node.window is not None
                self._drop(node)
                if (
                    not self.window_pages
                    or parent is None
                    or parent.children
                    or parent.pins
                    or parent.landing
                    or parent.window is not None
                ):
                    break
                node = parent
        return True
