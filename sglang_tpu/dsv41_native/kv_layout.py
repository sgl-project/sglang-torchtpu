"""Where the native DeepSeek-V4.1 caches keep a token: its KV-pool location (`loc`).

Every per-token cache row is indexed by the token's location in the token-to-KV pool, the value
the allocator puts in `out_cache_loc` and `req_to_token`. A radix hit hands a new request the
locations of the cached prefix, so the prefix rows are found without copying any cache.

A compressed group of `ratio` positions j*ratio .. j*ratio + ratio-1 is stored at
`loc(last position) // ratio`. The paged allocator puts position p at offset p % page_size of its
page and radix matches whole pages, so with page_size % ratio == 0 a group never straddles a page
and the division gives one row per group. Page 0 is the allocator's padding page; padded tokens and
tokens that complete no group write rows inside it, which no real token reads.
"""

from __future__ import annotations

import torch


def num_locs(*, max_total_tokens: int, page_size: int) -> int:
    """Rows of a loc-indexed cache: the pool's pages rounded up, plus the padding page."""
    pages = (max_total_tokens + page_size - 1) // page_size
    return (pages + 1) * page_size


def check_page_layout(*, page_size: int, ratios) -> None:
    for ratio in sorted({r for r in ratios if r > 0}):
        assert page_size % ratio == 0, (
            f"page_size {page_size} must be a multiple of compress ratio {ratio}: "
            "a compressed group has to lie inside one page"
        )


def group_tail_positions(groups: torch.Tensor, ratio: int) -> torch.Tensor:
    """Position of the last token of each compressed group (groups may be clamped at 0 first)."""
    return groups * ratio + (ratio - 1)


# Arbitrary floor on the copy size; buckets keep the eager index_put to a few shapes.
_MIN_PREFIX_COPY = 1024


def prefix_loc_writes(
    *, req_to_token: torch.Tensor, slots: list[int], prefix_lens: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host (rows, cols, locs) int32 that copy each request's cached-prefix locations from
    req_to_token into the model's loc table. The length is padded to a power of two with writes
    of location 0 to (0, 0), the padding slot, so prefix lengths do not each compile a program."""
    parts = [(s, n) for s, n in zip(slots, prefix_lens) if n > 0]
    total = sum(n for _, n in parts)
    size = max(_MIN_PREFIX_COPY, 1 << max(total - 1, 0).bit_length())
    rows = torch.zeros(size, dtype=torch.int32)
    cols = torch.zeros(size, dtype=torch.int32)
    locs = torch.zeros(size, dtype=torch.int32)
    offset = 0
    for slot, n in parts:
        rows[offset : offset + n] = slot
        cols[offset : offset + n] = torch.arange(n, dtype=torch.int32)
        locs[offset : offset + n] = req_to_token[slot, :n].to("cpu", torch.int32)
        offset += n
    return rows, cols, locs
