# SPDX-License-Identifier: Apache-2.0
"""Request-owned SWA storage of the DeepSeek-V4.1 TPU pool (encoder SWA replay).

Under ``--enable-encoder-swa-bounded-replay`` the radix cache keeps no SWA KV: a
prefix hit's last window is recomputed by the replay rows core folds in front of
its extend. The core ``RequestWindow`` moves FlashMLA-packed rows; CSA2 instead
reads bf16 ``[pages, 64, 512]`` buffers addressed by position (the row of
position ``p`` is ``table[b, p // 64] * 64 + p % 64``), so the TPU pool builds
this class in its place. One buffer per layer, in 64-token pages:

- pages ``[0, 4)``: reserved, as in the paged SWA pool (page 0 is the null page);
- one ring of ``RING_PAGES`` pages per request slot, holding the request's last
  window; position ``p`` of slot ``s`` lives in ring page ``p // 64 % RING_PAGES``;
- a workspace one extend forward fills with position-aligned pages.

Decode attends the ring directly. An extend chunk may span more pages than the
ring, so each request gets workspace pages covering its visible history and the
chunk; the history the chunk may see is copied in from the ring and the chunk's
last ``WINDOW_TOKENS`` rows are written back to the ring. Both moves are extra
records of the layer's one ``publish_swa`` call (``WindowPlan``).
"""

from typing import NamedTuple

import torch

from sglang.srt.utils import ceil_div

SWA_PAGE_TOKENS = 64
WINDOW_TOKENS = 128
# A 128-token window spans at most 3 consecutive 64-token pages, so 3 ring pages
# never alias; 4 is arbitrary headroom.
RING_PAGES = 4
RING_TOKENS = RING_PAGES * SWA_PAGE_TOKENS
RESERVED_PAGES = 4
# History plus chunk span [first - 128, first + count): at most
# (count + 127) / 64 + 2 pages, under count / 64 + 4.
EXTRA_WORKSPACE_PAGES_PER_REQUEST = 4


class WindowPlan(NamedTuple):
    """Row moves of one extend forward, each ``[B * WINDOW_TOKENS]``."""

    # int64 buffer row holding a history position (0 where unused), and int32
    # workspace row to copy it to (-1 where unused).
    history_src: torch.Tensor
    history_dst: torch.Tensor
    # int64 batch row of a chunk token kept in the ring (0 where unused), and
    # int32 ring row to write it to (-1 where unused).
    commit_token: torch.Tensor
    commit_dst: torch.Tensor


def _workspace_pages(*, num_slots: int, chunk_tokens: int) -> int:
    return ceil_div(chunk_tokens, SWA_PAGE_TOKENS) + (
        EXTRA_WORKSPACE_PAGES_PER_REQUEST * num_slots
    )


def window_buffer_pages(*, num_slots: int, chunk_tokens: int) -> int:
    """64-token pages of one layer's buffer: reserved, rings, workspace."""
    return (
        RESERVED_PAGES
        + RING_PAGES * num_slots
        + _workspace_pages(num_slots=num_slots, chunk_tokens=chunk_tokens)
    )


def _ring_rows(slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    page = (
        RESERVED_PAGES + slots * RING_PAGES + positions // SWA_PAGE_TOKENS % RING_PAGES
    )
    return page * SWA_PAGE_TOKENS + positions % SWA_PAGE_TOKENS


def decode_window_pages(slots: torch.Tensor, columns: int) -> torch.Tensor:
    """[B, columns] int64 SWA table pointing every logical page at the slot ring."""
    logical = torch.arange(columns, dtype=torch.int64)
    return RESERVED_PAGES + slots[:, None] * RING_PAGES + logical[None, :] % RING_PAGES


def _assign_workspace_pages(
    *,
    first: torch.Tensor,
    counts: torch.Tensor,
    columns: int,
    base_page: int,
    capacity_pages: int,
) -> torch.Tensor:
    """[B, columns] int64 table giving each request fresh pages over
    ``[first - 128, first + count)``; unused columns stay on the null page."""
    pages = torch.zeros(first.shape[0], columns, dtype=torch.int64)
    used = 0
    for b, (start, count) in enumerate(zip(first.tolist(), counts.tolist())):
        if not count:
            continue
        low = max(start - WINDOW_TOKENS, 0) // SWA_PAGE_TOKENS
        high = (start + count - 1) // SWA_PAGE_TOKENS
        if high >= columns:
            raise RuntimeError(f"position {start + count - 1} is past the SWA table")
        width = high - low + 1
        pages[b, low : high + 1] = torch.arange(
            base_page + used, base_page + used + width
        )
        used += width
    if used > capacity_pages:
        raise RuntimeError(
            f"the SWA request-window workspace holds {capacity_pages} pages, this "
            f"extend needs {used}; it is sized from --chunked-prefill-size"
        )
    return pages


def extend_window_pages(
    *,
    slots: torch.Tensor,
    first: torch.Tensor,
    counts: torch.Tensor,
    minimum: torch.Tensor,
    columns: int,
    base_page: int,
    capacity_pages: int,
) -> tuple[torch.Tensor, WindowPlan]:
    """Workspace SWA table and row moves of an extend forward.

    ``slots``, ``first`` (first position of the chunk), ``counts`` (chunk rows)
    and ``minimum`` (first visible position) are ``[B]`` int64 host tensors.
    """
    pages = _assign_workspace_pages(
        first=first,
        counts=counts,
        columns=columns,
        base_page=base_page,
        capacity_pages=capacity_pages,
    )
    lane = torch.arange(WINDOW_TOKENS, dtype=torch.int64)[None, :]
    live = (counts > 0)[:, None]
    slot = slots[:, None]

    history = first[:, None] - WINDOW_TOKENS + lane
    history_ok = live & (history >= 0) & (history >= minimum[:, None])
    safe = history.clamp(min=0)
    page = pages.gather(1, (safe // SWA_PAGE_TOKENS).clamp(max=columns - 1))
    history_dst = torch.where(
        history_ok, page * SWA_PAGE_TOKENS + safe % SWA_PAGE_TOKENS, -1
    )
    history_src = torch.where(history_ok, _ring_rows(slot, safe), 0)

    kept = (first + counts)[:, None] - WINDOW_TOKENS + lane
    kept_ok = live & (kept >= first[:, None])
    row_start = (torch.cumsum(counts, 0) - counts)[:, None]
    commit_token = torch.where(kept_ok, row_start + kept - first[:, None], 0)
    commit_dst = torch.where(kept_ok, _ring_rows(slot, kept.clamp(min=0)), -1)

    plan = WindowPlan(
        history_src=history_src.reshape(-1),
        history_dst=history_dst.reshape(-1).to(torch.int32),
        commit_token=commit_token.reshape(-1),
        commit_dst=commit_dst.reshape(-1).to(torch.int32),
    )
    return pages, plan


class DSV41TPURequestWindow:
    """Built by the core pool constructor in place of ``RequestWindow``.

    ``pool_factory(tokens, layers)`` is the pool's SWA factory; the tables and
    row moves are built per forward by the attention backend.
    """

    def __init__(
        self,
        pool_factory,
        *,
        num_slots: int,
        layers: int,
        page_size: int,
        capacity: int,
        workspace_rows=None,
    ):
        from sglang.srt.runtime_context import get_schedule

        del page_size, workspace_rows
        if capacity > RING_TOKENS:
            raise ValueError(
                f"the V4.1 TPU request window keeps {RING_TOKENS} tokens per slot, "
                f"{capacity} were asked for"
            )
        chunk_tokens = get_schedule().chunked_prefill_size or 0
        self.num_slots = num_slots
        self.workspace_first_page = RESERVED_PAGES + RING_PAGES * num_slots
        self.workspace_pages = _workspace_pages(
            num_slots=num_slots, chunk_tokens=chunk_tokens
        )
        pages = window_buffer_pages(num_slots=num_slots, chunk_tokens=chunk_tokens)
        # The SWA pool adds one reserved 256-token logical page to the size.
        self.state = pool_factory((pages - RESERVED_PAGES) * SWA_PAGE_TOKENS, layers)
        assert self.state.num_pages == pages, (self.state.num_pages, pages)

    def reset(self, slots: torch.Tensor) -> None:
        # A restarted window reads no ring row: the fold floors it at its first row.
        del slots

    def initialize_dummy_history(self) -> None:
        pass
