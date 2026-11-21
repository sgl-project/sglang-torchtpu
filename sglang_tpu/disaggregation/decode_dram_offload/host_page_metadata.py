"""Opaque page metadata follows the host KV bytes through spill and recall.

A page owner may register a codec (for example, an integrity checker). Decode
retains the bytes without interpreting them. Its table is bounded by host page
capacity; receiving a new occupant replaces or clears the old entry.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable
from weakref import WeakKeyDictionary

_lock = threading.RLock()
_codecs: WeakKeyDictionary = WeakKeyDictionary()
_retained_rows: WeakKeyDictionary = WeakKeyDictionary()


def _retained_tables_for(pool: Any) -> list[dict[int, bytes]]:
    tables: list[dict[int, bytes]] = []
    own = _retained_rows.get(pool)
    if own is not None:
        tables.append(own)
    domain = getattr(pool, "shared_allocation_domain", None)
    if domain is None:
        return tables
    for candidate in list(_retained_rows.keys()):
        if candidate is pool:
            continue
        if getattr(candidate, "shared_allocation_domain", None) is domain:
            rows = _retained_rows.get(candidate)
            if rows is not None:
                tables.append(rows)
    return tables


def discard_host_page_metadata(pool: Any, pages: Iterable[int] | None = None) -> None:
    """Invalidate opaque tags before their host pages are freed or rewritten.

    Registered codecs own their own page lifecycle; this only clears rows
    retained on behalf of a peer when this pool has no metadata owner. When
    ``pages`` is given for a pool backed by a ``shared_allocation_domain``
    (e.g. unified full/SWA pools sharing one physical host arena), sibling
    tables sharing that domain also drop stale entries for those physical pages.
    """
    with _lock:
        if pages is None:
            rows = _retained_rows.get(pool)
            if rows is not None:
                rows.clear()
            return
        page_list = list(pages)
        if not page_list:
            return
        for rows in _retained_tables_for(pool):
            for page in page_list:
                rows.pop(page, None)


def register_host_page_metadata(
    pool: Any,
    *,
    export: Callable[[list[int]], list[bytes]],
    restore: Callable[[list[int], list[bytes]], None],
) -> None:
    """Bind a pool's metadata owner, independently of channel creation order."""
    with _lock:
        if pool in _codecs:
            raise RuntimeError("host page metadata owner registered twice")
        _codecs[pool] = (export, restore)


class HostPageMetadata:
    def __init__(self, pool: Any, *, retain: bool = True) -> None:
        self.pool = pool
        self._retain = retain
        with _lock:
            self._rows: dict[int, bytes] = (
                _retained_rows.setdefault(pool, {})
                if retain and pool is not None
                else {}
            )

    def export(self, pages: list[int]) -> list[bytes]:
        with _lock:
            codec = _codecs.get(self.pool)
            if codec is None:
                rows = [self._rows.get(page, b"") for page in pages]
        if codec is not None:
            rows = codec[0](pages)
        self._validate(pages, rows)
        return rows if any(rows) else []

    def restore(self, pages: list[int], rows: list[bytes]) -> None:
        self._validate(pages, rows)
        with _lock:
            codec = _codecs.get(self.pool)
        if codec is not None:
            # An owner must reject missing metadata rather than keep stale tags.
            codec[1](pages, rows)
            return
        if not self._retain:
            # An unchecked prefill overwrites host pages through ordinary
            # HiCache too; it must not forward metadata from a prior recall.
            return
        with _lock:
            discard_host_page_metadata(self.pool, pages)
            for page, row in zip(pages, rows or [b""] * len(pages)):
                if row:
                    self._rows[page] = row
                else:
                    self._rows.pop(page, None)

    def _page_limit(self) -> int:
        limit = getattr(self.pool, "transfer_page_limit", None)
        if isinstance(limit, int) and not isinstance(limit, bool):
            return limit
        return self.pool.page_num

    def _validate(self, pages: list[int], rows: list[bytes]) -> None:
        if rows and len(rows) != len(pages):
            raise ValueError("host page metadata count does not match the transfer")
        page_limit = self._page_limit()
        if any(page < 0 or page >= page_limit for page in pages):
            raise ValueError("host page metadata index is outside the pool")

    def clear(self) -> None:
        with _lock:
            self._rows.clear()
