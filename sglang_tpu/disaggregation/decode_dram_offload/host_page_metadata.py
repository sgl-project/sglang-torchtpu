"""Opaque page metadata follows the host KV bytes through spill and recall.

A page owner may register a codec (for example, an integrity checker). Decode
retains the bytes without interpreting them. Its table is bounded by host page
capacity; receiving a new occupant replaces or clears the old entry.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable
from weakref import WeakKeyDictionary

_codecs: WeakKeyDictionary = WeakKeyDictionary()
_retained_rows: WeakKeyDictionary = WeakKeyDictionary()


def discard_host_page_metadata(pool: Any, pages: Iterable[int] | None = None) -> None:
    """Invalidate opaque tags before their host pages are freed or rewritten.

    Registered codecs own their own page lifecycle; this only clears rows
    retained on behalf of a peer when this pool has no metadata owner.
    """
    rows = _retained_rows.get(pool)
    if rows is None:
        return
    if pages is None:
        rows.clear()
    else:
        for page in pages:
            rows.pop(page, None)


def register_host_page_metadata(
    pool: Any,
    *,
    export: Callable[[list[int]], list[bytes]],
    restore: Callable[[list[int], list[bytes]], None],
) -> None:
    """Bind a pool's metadata owner, independently of channel creation order."""
    if pool in _codecs:
        raise RuntimeError("host page metadata owner registered twice")
    _codecs[pool] = (export, restore)


class HostPageMetadata:
    def __init__(self, pool: Any, *, retain: bool = True) -> None:
        self.pool = pool
        self._retain = retain
        self._rows: dict[int, bytes] = (
            _retained_rows.setdefault(pool, {}) if retain and pool is not None else {}
        )

    def export(self, pages: list[int]) -> list[bytes]:
        codec = _codecs.get(self.pool)
        if codec is not None:
            rows = codec[0](pages)
        else:
            rows = [self._rows.get(page, b"") for page in pages]
        self._validate(pages, rows)
        return rows if any(rows) else []

    def restore(self, pages: list[int], rows: list[bytes]) -> None:
        self._validate(pages, rows)
        codec = _codecs.get(self.pool)
        if codec is not None:
            # An owner must reject missing metadata rather than keep stale tags.
            codec[1](pages, rows)
            return
        if not self._retain:
            # An unchecked prefill overwrites host pages through ordinary
            # HiCache too; it must not forward metadata from a prior recall.
            return
        for page, row in zip(pages, rows or [b""] * len(pages)):
            if row:
                self._rows[page] = row
            else:
                self._rows.pop(page, None)

    def _validate(self, pages: list[int], rows: list[bytes]) -> None:
        if rows and len(rows) != len(pages):
            raise ValueError("host page metadata count does not match the transfer")
        if any(page < 0 or page >= self.pool.page_num for page in pages):
            raise ValueError("host page metadata index is outside the pool")

    def clear(self) -> None:
        self._rows.clear()
