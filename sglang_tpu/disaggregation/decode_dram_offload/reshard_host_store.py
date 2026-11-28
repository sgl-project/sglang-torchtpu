"""Decode-side shard-layout host store for asymmetric attention-TP offload.

When prefill runs at a wider attention TP than decode (``src_tp > dst_tp``,
``writers = src_tp // dst_tp``), each prefill rank's host block holds ``1 /
writers`` of the KV heads of a decode rank's block, and native Raiden host
reads move whole blocks only between managers of identical block layout.

Decode rank ``d`` therefore keeps ``writers`` host-only Raiden pools (built
lazily on the first push over 1-page placeholder tensors matching prefill
writer ``w``'s block layout, for ``w in 0..writers-1``), plus ``writers``
window pools on a hybrid sliding-window model. Every pool on every decode rank
holds the same page count, and all ``dst_tp`` decode ranks maintain an
identical deterministic ``ReshardLayoutStore`` index in lockstep over
``attn_tp_cpu_group``. Prefill rank ``s = d * writers + w`` pushes its pages to
writer pool ``w`` on decode rank ``d`` and recalls them block-for-block into
its own HiCache host pool on subsequent turns.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)

# Full pages budgeted per page of sliding window when sizing a hybrid store.
DEFAULT_PAGES_PER_WINDOW_PAGE = 4
# Decode iterations between agreement rounds, matching DecodeDramVictimReceiver.
_CADENCE = 16
# Seconds an incomplete multi-writer push waits before being dropped.
_MAX_WAIT_S = 20.0
MAX_INFLIGHT = 4

PushKey = Tuple[str, int]

# RaidenKVManager sets TPU_RAIDEN_DYNAMIC_HOST_STAGING=1 in-process; host-read
# pools require static host block allocation during KVCacheManager construction.
_DYNAMIC_HOST_STAGING = "TPU_RAIDEN_DYNAMIC_HOST_STAGING"
_DYNAMIC_HOST_STAGING_LOCK = threading.Lock()


def reshard_transfer_id(
    prefill: str, seq: int, writer: int, *, swa: bool = False
) -> str:
    return f"victim:{prefill}:{seq}:w{writer}" + (":swa" if swa else "")


@dataclass(frozen=True)
class ReshardPoolLayout:
    """Block layout of one prefill writer's HiCache host pool."""

    writer: int
    writers: int
    tensors: Tuple[Tuple[str, Tuple[int, ...]], ...]
    page_bytes: int
    window: bool = False

    def fingerprint(self) -> int:
        text = repr(
            (self.writer, self.writers, self.tensors, self.page_bytes, self.window)
        )
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") >> 1


def reshard_pool_layout(
    host_pool: Any, *, writer: int, writers: int, window: bool = False
) -> ReshardPoolLayout:
    """Describe ``host_pool``'s per-page tensor shapes and byte size for the wire."""
    tensors = [t for layer in host_pool.transfer._device_tensors for t in layer]
    return ReshardPoolLayout(
        writer=writer,
        writers=writers,
        tensors=tuple(
            (str(t.dtype).removeprefix("torch."), tuple(int(x) for x in t.shape[1:]))
            for t in tensors
        ),
        page_bytes=sum(int(t.nbytes) // int(t.shape[0]) for t in tensors),
        window=window,
    )


class _PlaceholderPool:
    """Minimal device-pool surface read by RaidenKVTransfer."""

    def __init__(self, buffers: List[Any]) -> None:
        self._buffers = buffers

    def page_buffers(self) -> List[Any]:
        return self._buffers


@contextlib.contextmanager
def _static_host_pool():
    with _DYNAMIC_HOST_STAGING_LOCK:
        previous = os.environ.get(_DYNAMIC_HOST_STAGING)
        os.environ[_DYNAMIC_HOST_STAGING] = "0"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(_DYNAMIC_HOST_STAGING, None)
            else:
                os.environ[_DYNAMIC_HOST_STAGING] = previous


def build_reshard_pool(layout: ReshardPoolLayout, pages: int, *, device: Any) -> Any:
    """Build a host-only RaidenKVTransfer of ``pages`` blocks in ``layout``."""
    import torch

    from sglang_tpu.raiden_transfer import RaidenKVTransfer

    buffers = [
        torch.zeros((1, *shape), dtype=getattr(torch, dtype), device=device)
        for dtype, shape in layout.tensors
    ]
    page_bytes = sum(int(t.nbytes) for t in buffers)
    if page_bytes != layout.page_bytes:
        raise ValueError(
            f"writer {layout.writer} pages are {layout.page_bytes} bytes, "
            f"its tensors describe {page_bytes}"
        )
    with _static_host_pool():
        return RaidenKVTransfer(_PlaceholderPool(buffers), num_host_pages=pages)


def size_reshard_store(
    *,
    budget_bytes: int,
    page_bytes: int,
    window_page_bytes: int = 0,
    window_pages: int = 0,
    pages_per_window_page: int = DEFAULT_PAGES_PER_WINDOW_PAGE,
) -> Tuple[int, int]:
    """Return ``(pages, windows)`` fitting within ``budget_bytes`` per decode rank."""
    if page_bytes <= 0:
        raise ValueError("a reshard page must have positive byte size")
    if not window_pages:
        return budget_bytes // page_bytes, 0
    if pages_per_window_page <= 0:
        raise ValueError(
            f"pages_per_window_page must be positive, got {pages_per_window_page}"
        )
    branch_pages = pages_per_window_page * window_pages
    branch_bytes = window_pages * window_page_bytes + branch_pages * page_bytes
    windows = budget_bytes // branch_bytes
    return windows * branch_pages, windows


class _Ids:
    """Deterministic lowest-free-id allocator so lockstep ranks pick identical IDs."""

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
    window: Optional[List[int]] = None
    stamp: int = 0
    pins: int = 0
    landing: bool = False
    window_landing: bool = False


@dataclass
class ReshardLanding:
    end: _Page
    first_new_page: int
    new_pages: List[int]
    window: Optional[List[int]]
    _new_nodes: List[_Page] = field(default_factory=list)


@dataclass
class ReshardMatch:
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
    digest = _root_digest(scope)
    out = []
    for start in range(0, len(tokens) - page_size + 1, page_size):
        page = tokens[start : start + page_size]
        packed = struct.pack(f"<{page_size}q", *page)
        digest = hashlib.blake2b(digest + packed, digest_size=16).digest()
        out.append(digest)
    return out


class ReshardLayoutStore:
    """Deterministic prefix index of branches stored in per-writer host pools."""

    def __init__(
        self,
        *,
        page_size: int,
        pages: int,
        window_pages: int = 0,
        windows: int = 0,
        on_free_pages: Optional[Callable[[Sequence[int]], None]] = None,
        on_free_windows: Optional[Callable[[Sequence[int]], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
    ) -> None:
        if page_size < 1 or pages < 0 or window_pages < 0 or windows < 0:
            raise ValueError("reshard store dimensions must be non-negative")
        self.page_size = page_size
        self.window_pages = window_pages
        self._pages = _Ids(pages)
        self._windows = _Ids(windows * window_pages)
        self._on_free_pages = on_free_pages
        self._on_free_windows = on_free_windows
        self._on_clear = on_clear
        self._roots: Dict[bytes, Dict[bytes, _Page]] = {}
        self._leaves: set = set()
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
        return dict(
            self.stats,
            pages=self.capacity_pages,
            used_pages=self.used_pages,
            windows=(
                self._windows.count // self.window_pages if self.window_pages else 0
            ),
            used_windows=self.used_windows,
        )

    def reserve(
        self, scope: Hashable, tokens: Sequence[int], *, with_window: bool = False
    ) -> Optional[ReshardLanding]:
        if with_window and not self.window_pages:
            raise ValueError("this store holds no sliding windows")
        digests = _page_digests(scope, tokens, self.page_size)
        if not digests or len(digests) * self.page_size != len(tokens):
            raise ValueError("a pushed branch must be a positive number of whole pages")
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
        if held == len(digests) and with_window and path[-1].window_landing:
            self.stats["busy"] += 1
            return None
        want_window = with_window and not (held == len(digests) and path[-1].window)
        if held == len(digests) and not want_window:
            self._touch(path[-1])
            return None
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
        if want_window:
            end.window_landing = True
        self._pin(end)
        return ReshardLanding(
            end=end,
            first_new_page=held,
            new_pages=new_pages,
            window=self._windows.take(self.window_pages) if want_window else None,
            _new_nodes=new_nodes,
        )

    def holds(
        self, scope: Hashable, tokens: Sequence[int], *, with_window: bool = False
    ) -> bool:
        level = self._roots.get(_root_digest(scope), {})
        node = None
        for digest in _page_digests(scope, tokens, self.page_size):
            node = level.get(digest)
            if node is None or node.landing:
                return False
            level = node.children
        return node is not None and (not with_window or node.window is not None)

    def commit(self, landing: ReshardLanding) -> None:
        for node in landing._new_nodes:
            node.landing = False
        if landing.window is not None:
            landing.end.window_landing = False
            landing.end.window = landing.window
        self._unpin(landing.end)
        self._touch(landing.end)
        self.stats["landed"] += 1
        self.stats["landed_pages"] += len(landing.new_pages)
        self.stats["reused_pages"] += landing.first_new_page

    def abort(self, landing: ReshardLanding) -> None:
        self._unpin(landing.end)
        if landing.window is not None:
            landing.end.window_landing = False
            self._windows.give(landing.window)
            if self._on_free_windows is not None:
                self._on_free_windows(landing.window)
        for node in reversed(landing._new_nodes):
            self._drop(node)
        self.stats["aborted"] += 1

    def match(self, scope: Hashable, tokens: Sequence[int]) -> Optional[ReshardMatch]:
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
        return ReshardMatch(
            end=path[-1],
            pages=[node.page_id for node in path],
            window=list(path[-1].window or ())[: len(path)],
        )

    def pin(self, match: ReshardMatch) -> None:
        self._pin(match.end)

    def unpin(self, match: ReshardMatch) -> None:
        self._unpin(match.end)

    def has_pins(self) -> bool:
        return self._open_pins > 0

    def clear(self) -> None:
        if self.has_pins():
            raise RuntimeError(
                "reshard-layout pages are pinned; the engine is not idle"
            )
        self._roots.clear()
        self._leaves.clear()
        self._pages = _Ids(self._pages.count)
        self._windows = _Ids(self._windows.count)
        if self._on_clear is not None:
            self._on_clear()

    def _pin(self, node: Optional[_Page]) -> None:
        self._open_pins += 1
        while node is not None:
            node.pins += 1
            node = node.parent

    def _unpin(self, node: Optional[_Page]) -> None:
        self._open_pins -= 1
        while node is not None:
            if node.pins < 1:
                raise RuntimeError(
                    "reshard-layout page unpinned more often than pinned"
                )
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
        if self._on_free_pages is not None:
            self._on_free_pages([node.page_id])
        if node.window is not None:
            window = node.window
            self._windows.give(window)
            node.window = None
            if self._on_free_windows is not None:
                self._on_free_windows(window)

    def _root_level(self, node: _Page) -> Dict[bytes, _Page]:
        for level in self._roots.values():
            if level.get(node.digest) is node:
                return level
        raise RuntimeError("reshard-layout page is in no tree")

    def _make_room(self, pages: int, window_ids: int) -> bool:
        if pages > self._pages.count or window_ids > self._windows.count:
            return False
        while self._pages.free() < pages or self._windows.free() < window_ids:
            leaves = [
                node for node in self._leaves if not node.pins and not node.landing
            ]
            if self._windows.free() < window_ids and self._pages.free() >= pages:
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


def _manifest(header: Any) -> Tuple[int, int]:
    text = repr(
        (
            tuple(header.tokens),
            header.extra_key,
            header.cache_salt,
            header.writers,
            header.window_pages,
        )
    )
    digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "little") >> 1 for i in (0, 8))


@dataclass
class ReshardPush:
    """Decode rank's view of one asymmetric victim push across ``writers`` sources."""

    header: Any = None
    parts: Dict[int, Any] = field(default_factory=dict)
    manifest: Optional[Tuple[int, int]] = None

    def window_len(self, page_size: int) -> int:
        return min(len(self.header.tokens) // page_size, self.header.window_pages)


class ReshardPoolSet:
    """Per-writer host-only Raiden pools on one decode rank."""

    def __init__(
        self,
        *,
        rank: int,
        budget_bytes: int,
        build_pool: Callable[[ReshardPoolLayout, int], Any],
    ) -> None:
        self.rank = rank
        self._budget_bytes = budget_bytes
        self._build_pool = build_pool
        self._pools: Dict[Tuple[int, bool], Any] = {}
        self._fingerprints: Dict[Tuple[int, bool], int] = {}
        self._page_counts: Dict[Tuple[int, bool], int] = {}
        self._metadata: Dict[Tuple[int, bool], Dict[int, bytes]] = {}
        self._metadata_lock = threading.Lock()
        self.writers: int = 0
        self.sizes: Optional[Tuple[int, int]] = None
        self.refusal: Optional[str] = None

    def transfer(self, writer: int, *, swa: bool = False) -> Any:
        return self._pools[(writer, swa)]

    def has(self, writer: int, *, swa: bool = False) -> bool:
        return (writer, swa) in self._pools

    def _validate_metadata_pages(
        self, key: Tuple[int, bool], pages: Sequence[int], rows: Sequence[bytes]
    ) -> None:
        if rows and len(rows) != len(pages):
            raise ValueError("host page metadata count does not match the transfer")
        limit = self._page_counts.get(key)
        if limit is None:
            raise ValueError(f"no reshard pool registered for {key}")
        if any(page < 0 or page >= limit for page in pages):
            raise ValueError("host page metadata index is outside the pool")

    def restore_metadata(
        self,
        writer: int,
        pages: Sequence[int],
        rows: Sequence[bytes],
        *,
        swa: bool = False,
    ) -> None:
        if not pages and not rows:
            return
        key = (writer, swa)
        with self._metadata_lock:
            self._validate_metadata_pages(key, pages, rows)
            table = self._metadata.setdefault(key, {})
            if not rows:
                for page in pages:
                    table.pop(page, None)
                return
            for page, row in zip(pages, rows, strict=True):
                if row:
                    table[page] = bytes(row)
                else:
                    table.pop(page, None)

    def export_metadata(
        self, writer: int, pages: Sequence[int], *, swa: bool = False
    ) -> List[bytes]:
        if not pages:
            return []
        key = (writer, swa)
        with self._metadata_lock:
            self._validate_metadata_pages(key, pages, ())
            table = self._metadata.get(key)
            if not table:
                return []
            rows = [table.get(page, b"") for page in pages]
            return rows if any(rows) else []

    def discard_metadata(
        self,
        pages: Optional[Sequence[int]] = None,
        *,
        writer: Optional[int] = None,
        swa: Optional[bool] = None,
    ) -> None:
        with self._metadata_lock:
            if writer is None and swa is None and pages is None:
                self._metadata.clear()
                return
            page_list = None if pages is None else list(pages)
            for key, table in list(self._metadata.items()):
                k_writer, k_swa = key
                if writer is not None and k_writer != writer:
                    continue
                if swa is not None and k_swa != swa:
                    continue
                if page_list is None:
                    table.clear()
                else:
                    for page in page_list:
                        table.pop(page, None)

    def clear_metadata(self) -> None:
        with self._metadata_lock:
            self._metadata.clear()

    def fits(self, parts: Dict[int, Any]) -> bool:
        if len(parts) != self.writers:
            return False
        for writer, part in parts.items():
            if self._fingerprints.get((writer, False)) != part.layout.fingerprint():
                return False
            if part.swa_blocks and (
                self._fingerprints.get((writer, True)) != part.swa_layout.fingerprint()
            ):
                return False
        return True

    def build(
        self, parts: Dict[int, Any], *, window_pages: int, lockstep: Any
    ) -> Optional[Tuple[int, int]]:
        page_bytes = lockstep.max(sum(p.layout.page_bytes for p in parts.values()))
        swa_bytes = lockstep.max(
            sum(p.swa_layout.page_bytes for p in parts.values()) if window_pages else 0
        )
        from sglang_tpu.envs import envs as tpu_envs

        ratio = tpu_envs.SGLANG_PD_OFFLOAD_RESHARD_PAGES_PER_WINDOW_PAGE.get()
        error = None
        pages, windows = 0, 0
        if ratio < 0:
            error = (
                f"{tpu_envs.SGLANG_PD_OFFLOAD_RESHARD_PAGES_PER_WINDOW_PAGE.name} "
                f"must be non-negative (0 uses default {DEFAULT_PAGES_PER_WINDOW_PAGE}), "
                f"got {ratio}"
            )
        else:
            pages, windows = size_reshard_store(
                budget_bytes=self._budget_bytes,
                page_bytes=max(page_bytes, 1),
                window_page_bytes=swa_bytes,
                window_pages=window_pages if swa_bytes else 0,
                pages_per_window_page=ratio or DEFAULT_PAGES_PER_WINDOW_PAGE,
            )
            if pages < 1 or (window_pages and windows < 1):
                error = (
                    f"budget of {self._budget_bytes} bytes holds no page or no window"
                )
            else:
                try:
                    for writer, part in sorted(parts.items()):
                        self._add(writer, False, part.layout, pages)
                        if window_pages:
                            self._add(
                                writer, True, part.swa_layout, windows * window_pages
                            )
                except Exception as exc:  # noqa: BLE001 - decline offload cleanly
                    error = f"cannot build a reshard-layout pool: {exc}"
        if error is not None:
            logger.error(
                "[dram-offload] reshard store on rank %d: %s", self.rank, error
            )
        if not lockstep.all(error is None):
            self.refusal = error or "another decode rank could not build its pools"
            return None
        self.writers = len(parts)
        self.sizes = (pages, windows)
        logger.info(
            "[dram-offload] reshard store ready on rank %d: %d writers, %d pages "
            "(%d bytes across writers), %d windows of %d pages",
            self.rank,
            self.writers,
            pages,
            page_bytes,
            windows,
            window_pages,
        )
        return self.sizes

    def _add(
        self, writer: int, swa: bool, layout: ReshardPoolLayout, pages: int
    ) -> None:
        self._pools[(writer, swa)] = self._build_pool(layout, pages)
        self._fingerprints[(writer, swa)] = layout.fingerprint()
        self._page_counts[(writer, swa)] = pages


@dataclass
class _ReshardLandingEntry:
    push: ReshardPush
    landing: ReshardLanding
    read_ids: List[str] = field(default_factory=list)
    failed: bool = False


class DecodeReshardBranchReceiver:
    """Lands branches pushed by a wider-TP prefill into per-writer host pools."""

    def __init__(
        self,
        *,
        channel: Any,
        lockstep: Any,
        pools: ReshardPoolSet,
        page_size: int,
        log_stats: Callable[[Dict[str, int]], None] = lambda stats: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._channel = channel
        self._lockstep = lockstep
        self._pools = pools
        self._page_size = page_size
        self._log_stats = log_stats
        self._clock = clock
        self.store: Optional[ReshardLayoutStore] = None
        self._iterations = 0
        self._rounds = 0
        self._first_seen: Dict[PushKey, float] = {}
        self._landings: Dict[PushKey, _ReshardLandingEntry] = {}
        self.stats = dict.fromkeys(
            ("received", "granted", "declined", "held", "failed", "dropped", "layout"),
            0,
        )

    def has_host_landings(self) -> bool:
        return bool(self._landings)

    def release_all(self) -> None:
        if self._landings:
            raise RuntimeError("reshard pushes are landing; the engine is not idle")
        if self.store is not None:
            self.store.clear()

    def counters(self) -> Dict[str, int]:
        return dict(self.stats, **(self.store.counters() if self.store else {}))

    def advance(self) -> None:
        self._iterations += 1
        if self._iterations % _CADENCE:
            return
        self._rounds += 1
        self._admit_agreed_pushes()
        self.finish_agreed_landings()
        self._log_stats(self.counters())

    def _whole(self, push: ReshardPush) -> bool:
        header = push.header
        if header is None or header.writers <= 1:
            return False
        pages = len(header.tokens) // self._page_size
        win_len = push.window_len(self._page_size)
        return all(
            writer in push.parts
            and len(push.parts[writer].blocks) == pages
            and len(push.parts[writer].swa_blocks) == win_len
            for writer in range(header.writers)
        )

    def _admit_agreed_pushes(self) -> None:
        received = {
            key: push
            for key, push in self._channel.received_reshard_pushes().items()
            if key not in self._landings
        }
        now = self._clock()
        for key in received:
            if key not in self._first_seen:
                self._first_seen[key] = now
                self.stats["received"] += 1
        manifests = {}
        for key, push in received.items():
            if self._whole(push):
                if push.manifest is None:
                    push.manifest = _manifest(push.header)
                manifests[(*key, *push.manifest)] = key
        for manifest in self._lockstep.common_to_all_ranks(list(manifests)):
            key = manifests[manifest]
            push = self._channel.take_reshard_push(key)
            self._first_seen.pop(key, None)
            stored = self._admit(key, push)
            if stored is not None:
                self._ack(push, stored=stored)
        for key, first in list(self._first_seen.items()):
            if now - first > _MAX_WAIT_S:
                self._first_seen.pop(key)
                if self._channel.take_reshard_push(key) is not None:
                    self.stats["dropped"] += 1

    def _admit(self, key: PushKey, push: ReshardPush) -> Optional[bool]:
        header = push.header
        parts = {w: push.parts[w] for w in range(header.writers)}
        if not self._ensure_store(parts, header.window_pages):
            self.stats["declined"] += 1
            return False
        if not self._lockstep.all(
            self._pools.fits(parts)
            and header.window_pages in (0, self.store.window_pages)
        ):
            self.stats["layout"] += 1
            return False
        scope = (header.extra_key, header.cache_salt)
        with_window = bool(header.window_pages)
        if self.store.holds(scope, header.tokens, with_window=with_window):
            self.stats["held"] += 1
            return True
        if len(self._landings) >= MAX_INFLIGHT:
            self.stats["declined"] += 1
            return False
        landing = self.store.reserve(scope, header.tokens, with_window=with_window)
        if landing is None:
            self.stats["declined"] += 1
            return False
        entry = _ReshardLandingEntry(push=push, landing=landing)
        started = all(
            self._start_reads(key, writer, part, entry)
            for writer, part in sorted(parts.items())
        )
        entry.failed = not self._lockstep.all(started)
        self._landings[key] = entry
        self.stats["granted"] += 1
        return None

    def _ensure_store(self, parts: Dict[int, Any], window_pages: int) -> bool:
        if self.store is not None:
            return True
        if self._pools.refusal is not None:
            return False
        sizes = self._pools.build(
            parts, window_pages=window_pages, lockstep=self._lockstep
        )
        if sizes is None:
            return False
        self.store = ReshardLayoutStore(
            page_size=self._page_size,
            pages=sizes[0],
            window_pages=window_pages,
            windows=sizes[1],
            on_free_pages=lambda ids: self._pools.discard_metadata(ids, swa=False),
            on_free_windows=lambda ids: self._pools.discard_metadata(ids, swa=True),
            on_clear=self._pools.clear_metadata,
        )
        return True

    def _start_reads(
        self, key: PushKey, writer: int, part: Any, entry: _ReshardLandingEntry
    ) -> bool:
        landing = entry.landing
        reads = []
        if landing.new_pages:
            reads.append(
                (
                    False,
                    part.uuid,
                    part.address,
                    part.blocks[landing.first_new_page :],
                    landing.new_pages,
                )
            )
        if landing.window is not None:
            reads.append(
                (
                    True,
                    part.swa_uuid or part.uuid,
                    part.swa_address,
                    part.swa_blocks,
                    landing.window[: len(part.swa_blocks)],
                )
            )
        for swa, uuid, address, remote, local in reads:
            read_id = reshard_transfer_id(*key, writer, swa=swa)
            if not self._channel.start_reshard_read(
                read_id,
                uuid,
                address,
                list(remote),
                list(local),
                writer=writer,
                swa=swa,
            ):
                return False
            entry.read_ids.append(read_id)
        return True

    def finish_agreed_landings(self) -> None:
        if not self._landings:
            return
        self._channel.drain_host_reads()
        keys = sorted(self._landings)
        row = []
        for key in keys:
            entry = self._landings[key]
            received = self._channel.poll_host_reads(entry.read_ids)
            row += [
                int(received is not None),
                int(received is True and not entry.failed),
            ]
        low, _ = self._lockstep.min_max(row)
        for index, key in enumerate(keys):
            settled, received = low[2 * index], low[2 * index + 1]
            if not settled:
                continue
            entry = self._landings.pop(key)
            self._channel.forget_host_reads(entry.read_ids)
            if received:
                restored = False
                try:
                    self._restore_landing_metadata(entry.push, entry.landing)
                    restored = True
                except Exception:  # noqa: BLE001 - decline on every rank
                    logger.exception("Cannot restore reshard victim push host metadata")
                received = self._lockstep.all(restored)
            if received:
                self.store.commit(entry.landing)
            else:
                self.store.abort(entry.landing)
                self.stats["failed"] += 1
            self._ack(entry.push, stored=bool(received))

    def _restore_landing_metadata(
        self, push: ReshardPush, landing: ReshardLanding
    ) -> None:
        header = push.header
        for writer in range(header.writers):
            part = push.parts[writer]
            full_meta = getattr(part, "page_metadata", ())
            if full_meta and len(full_meta) != len(part.blocks):
                raise ValueError("host page metadata count does not match the transfer")
            if landing.new_pages:
                new_meta = full_meta[landing.first_new_page :] if full_meta else ()
                self._pools.restore_metadata(
                    writer, landing.new_pages, new_meta, swa=False
                )
            if landing.window is not None:
                swa_meta = getattr(part, "swa_page_metadata", ())
                win_len = len(part.swa_blocks)
                if swa_meta and len(swa_meta) != win_len:
                    raise ValueError(
                        "host page metadata count does not match the transfer"
                    )
                self._pools.restore_metadata(
                    writer, landing.window[:win_len], swa_meta, swa=True
                )

    def _ack(self, push: ReshardPush, *, stored: bool) -> None:
        for part in push.parts.values():
            self._channel.send_victim_ack(
                endpoint=part.ack_endpoint, seq=part.seq, stored=stored
            )


class DecodeReshardBranchOfferer:
    """Offers prefixes from ReshardLayoutStore on asymmetric RESHARD_REQ turns."""

    def __init__(
        self,
        *,
        channel: Any,
        lockstep: Any,
        receiver: DecodeReshardBranchReceiver,
        owned_rooms: Callable[[], Iterable[int]],
    ) -> None:
        self._channel = channel
        self._lockstep = lockstep
        self._receiver = receiver
        self._owned_rooms = owned_rooms
        self.offers: Dict[int, ReshardMatch] = {}
        self._released: set = set()
        self.stats = dict.fromkeys(("offers", "offered_pages"), 0)

    def offer_reshard_prefix(self, *, room: int, tokens: List[int], scope: Any) -> None:
        store = self._receiver.store
        if store is None or room in self.offers or not tokens:
            return
        match = store.match(scope, tokens)
        if match is None:
            return
        store.pin(match)
        self.offers[room] = match
        self.stats["offers"] += 1
        self.stats["offered_pages"] += match.num_pages
        self._channel.offer_reshard_prefix(room, match.pages, match.window)

    def settle_offers(self) -> None:
        self.close_departed_offers()
        self.unpin_settled_offers()

    def close_departed_offers(self) -> None:
        if len(self.offers) == len(self._released):
            return
        owned = set(self._owned_rooms())
        for room in self.offers:
            if room not in owned and room not in self._released:
                self._channel.release_prefix_offer(room)
                self._released.add(room)

    def unpin_settled_offers(self) -> None:
        if not self._released:
            return
        self._channel.drain_host_reads()
        rooms = sorted(self._released)
        low, _ = self._lockstep.min_max(
            [int(not self._channel.host_prefix_pending(room)) for room in rooms]
        )
        for room, done in zip(rooms, low):
            if done:
                self._receiver.store.unpin(self.offers.pop(room))
                self._released.remove(room)

    def release_all(self) -> None:
        if self.offers:
            raise RuntimeError(
                "reshard prefix offers are pinned; the engine is not idle"
            )
