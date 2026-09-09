"""Pairing math for pulling a prefill chunk before the prefill finishes.

The prefill announces its pages as each chunked-prefill step commits them, so
the decode learns the page list in ordered pieces. A piece covers prefill pages
``[start_page, start_page + len(blocks))``; only part of it may be wanted,
because the decode can already hold the leading pages from its own prefix cache
and pairs the two lists from the tail.
"""

from __future__ import annotations

from typing import NamedTuple


class PullWindow(NamedTuple):
    """Slice of one announced chunk that the decode still has to pull."""

    chunk_offset: int
    local_offset: int
    length: int

    def __bool__(self) -> bool:
        return self.length > 0


EMPTY_WINDOW = PullWindow(0, 0, 0)


def chunk_pull_window(
    start_page: int, chunk_pages: int, total_pages: int, local_pages: int
) -> PullWindow:
    """Part of one announced chunk the decode needs, in both sides' indices.

    ``total_pages`` is what the prefill will produce for the request and
    ``local_pages`` is what the decode allocated. The decode holds the last
    ``local_pages`` of the prefill's list, so prefill page ``i`` pairs with
    local page ``i - (total_pages - local_pages)``; pages before that are
    already cached on the decode, and a page the prefill never fills (the
    reserved decode token) is past the end.
    """
    if min(chunk_pages, total_pages, local_pages) <= 0 or start_page < 0:
        return EMPTY_WINDOW
    skip = max(0, total_pages - local_pages)
    paired = min(total_pages, local_pages)
    lo = max(start_page, skip)
    hi = min(start_page + chunk_pages, skip + paired)
    if hi <= lo:
        return EMPTY_WINDOW
    return PullWindow(lo - start_page, lo - skip, hi - lo)


def whole_pull_registration(blocks: list[int], local_pages: int) -> list[int]:
    """Returns the producer KV blocks that the decode consumer will read.

    The decode pairs announced blocks from the tail against its local allocation,
    so leading pages already resident in its prefix cache are not requested.
    Returns an empty list on a full prefix hit, where no read is issued.
    """
    window = chunk_pull_window(0, len(blocks), len(blocks), local_pages)
    if not window:
        return []
    return blocks[window.chunk_offset : window.chunk_offset + window.length]
