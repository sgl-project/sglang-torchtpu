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

import logging
from typing import NamedTuple, Optional

import torch

from sglang.srt.utils import ceil_div

logger = logging.getLogger(__name__)
_LOGGED_TRIM = set()

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
    """Row moves of one extend forward, each ``[R * WINDOW_TOKENS]``: R = B, or
    the live-request bound under SGLANG_TPU_DSV41_SWA_PLAN_TRIM (plan_requests)."""

    # int64 buffer row holding a history position (0 where unused), and int32
    # workspace row to copy it to (-1 where unused).
    history_src: torch.Tensor
    history_dst: torch.Tensor
    # int64 batch row of a chunk token kept in the ring (0 where unused), and
    # int32 ring row to write it to (-1 where unused).
    commit_token: torch.Tensor
    commit_dst: torch.Tensor


def window_chunk_tokens(chunked_prefill_size) -> int:
    """Rows one forward plans windows for: the prefill chunk, plus the rows a fused
    mixed prefill + verify step may add on its fine token bucket
    (SGLANG_TPU_MIXED_VERIFY_FINE_BUCKET; mixed_step.FINE_BUCKET_ROWS)."""
    from sglang_tpu.experimental.speculative.mixed_step import fine_bucket_extra_rows

    return (chunked_prefill_size or 0) + fine_bucket_extra_rows()


def extend_batch_rows() -> int:
    """Requests in a TPU extend graph's fixed batch (torch_compile_max_bs): the
    precompile warms each extend graph with that many one-token requests, one
    fresh workspace page each, whatever the scheduler's request slots."""
    from sglang.srt.runtime_context import get_exec

    return get_exec().graph.torch_compile_max_bs or 0


def _workspace_pages(*, num_slots: int, chunk_tokens: int, batch_rows: int = 0) -> int:
    # A real extend: one chunk plus each request's history span. The fixed-batch
    # warmup: one page per batch row. With few slots (a small
    # --max-running-requests) the second is the larger.
    return max(
        ceil_div(chunk_tokens, SWA_PAGE_TOKENS)
        + EXTRA_WORKSPACE_PAGES_PER_REQUEST * num_slots,
        batch_rows,
    )


def window_buffer_pages(
    *, num_slots: int, chunk_tokens: int, batch_rows: int = 0
) -> int:
    """64-token pages of one layer's buffer: reserved, rings, workspace."""
    return (
        RESERVED_PAGES
        + RING_PAGES * num_slots
        + _workspace_pages(
            num_slots=num_slots, chunk_tokens=chunk_tokens, batch_rows=batch_rows
        )
    )


def plan_requests(num_slots: int) -> Optional[int]:
    """Request rows of an extend's WindowPlan under SGLANG_TPU_DSV41_SWA_PLAN_TRIM,
    else None (one per batch row).

    Only a live request moves rows: one with rows, on a nonzero slot, the first of
    its slot. Distinct nonzero slots number at most ``num_slots - 1``, so that
    many plan rows hold every move while the fixed extend batch (64) pads the
    publish of each layer with unused records."""
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_DSV41_SWA_PLAN_TRIM.get():
        return None
    return max(num_slots - 1, 1)


def live_plan_rows(
    live: torch.Tensor, requests: Optional[int]
) -> Optional[torch.Tensor]:
    """``[min(B, requests)]`` batch rows holding every live request first (stable),
    or None to keep all B rows. Static in shape, so the device build uses it too.

    A batch with more live requests than ``requests`` has a slot past the pool
    (precompile dummies only); its extra requests move no rows."""
    if requests is None or requests >= live.shape[0]:
        return None
    if live.shape[0] not in _LOGGED_TRIM:
        # Once per batch size and process: the engagement proof of the trim.
        _LOGGED_TRIM.add(live.shape[0])
        logger.info(
            "DSV4.1 SWA plan trim: on, %d -> %d request rows, %d -> %d window "
            "records per layer publish",
            live.shape[0],
            requests,
            2 * live.shape[0] * WINDOW_TOKENS,
            2 * requests * WINDOW_TOKENS,
        )
    # Stable partition by scatter, not a sort: rank of each row in (live, rest).
    # int32 cumsum: an int64 one does not lower on the TPU vector units.
    on = live.to(torch.int32)
    off = 1 - on
    rank = torch.where(
        live,
        torch.cumsum(on, 0, dtype=torch.int32) - 1,
        on.sum(dtype=torch.int32) + torch.cumsum(off, 0, dtype=torch.int32) - 1,
    )
    rows = torch.arange(live.shape[0], dtype=torch.int32, device=live.device)
    return torch.empty_like(rows).scatter_(0, rank.long(), rows)[:requests]


def ring_rows(slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
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
    """[B, columns] int32 table giving each request fresh pages over
    ``[first - 128, first + count)``; unused columns stay on the null page.

    int32, the upload's dtype: a dense [B, 4 * max_context / 256] table is MBs,
    and an int64 build then a conversion copies it twice per extend."""
    pages = torch.zeros(first.shape[0], columns, dtype=torch.int32)
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
            base_page + used, base_page + used + width, dtype=torch.int32
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
    requests: Optional[int] = None,
) -> tuple[torch.Tensor, WindowPlan]:
    """Workspace SWA table and row moves of an extend forward.

    ``slots``, ``first`` (first position of the chunk), ``counts`` (chunk rows)
    and ``minimum`` (first visible position) are ``[B]`` int64 host tensors.
    ``requests`` (plan_requests) caps the plan's request rows.
    """
    pages = _assign_workspace_pages(
        first=first,
        counts=counts,
        columns=columns,
        base_page=base_page,
        capacity_pages=capacity_pages,
    )
    lane = torch.arange(WINDOW_TOKENS, dtype=torch.int64)[None, :]
    # A slot's ring belongs to one request per forward: the padding slot 0
    # (idle and bucket-padding rows) owns none, and a slot repeated across
    # requests (dummy and warmup batches) moves rows for its first request only.
    # Otherwise the SWA publish gets duplicate write records, which
    # write_swa_pages does not allow.
    first_of_slot = torch.ones_like(slots, dtype=torch.bool)
    if slots.numel() > 1:
        order = torch.argsort(slots, stable=True)
        sorted_slots = slots[order]
        repeat = torch.zeros_like(first_of_slot)
        repeat[1:] = sorted_slots[1:] == sorted_slots[:-1]
        first_of_slot[order] = ~repeat
    live = ((counts > 0) & (slots != 0) & first_of_slot)[:, None]
    slot = slots[:, None]

    history = first[:, None] - WINDOW_TOKENS + lane
    history_ok = live & (history >= 0) & (history >= minimum[:, None])
    safe = history.clamp(min=0)
    page = pages.gather(1, (safe // SWA_PAGE_TOKENS).clamp(max=columns - 1))
    history_dst = torch.where(
        history_ok, page * SWA_PAGE_TOKENS + safe % SWA_PAGE_TOKENS, -1
    )
    history_src = torch.where(history_ok, ring_rows(slot, safe), 0)

    kept = (first + counts)[:, None] - WINDOW_TOKENS + lane
    kept_ok = live & (kept >= first[:, None])
    row_start = (torch.cumsum(counts, 0) - counts)[:, None]
    commit_token = torch.where(kept_ok, row_start + kept - first[:, None], 0)
    commit_dst = torch.where(kept_ok, ring_rows(slot, kept.clamp(min=0)), -1)
    keep = live_plan_rows(live[:, 0], requests)
    if keep is not None:
        history_src, history_dst, commit_token, commit_dst = (
            t[keep] for t in (history_src, history_dst, commit_token, commit_dst)
        )

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
        # Rows per slot the core compares against a draft ring's.
        self.capacity = capacity
        chunk_tokens = window_chunk_tokens(get_schedule().chunked_prefill_size)
        batch_rows = extend_batch_rows()
        self.workspace_first_page = RESERVED_PAGES + RING_PAGES * num_slots
        self.plan_requests = plan_requests(num_slots)
        self.workspace_pages = _workspace_pages(
            num_slots=num_slots, chunk_tokens=chunk_tokens, batch_rows=batch_rows
        )
        pages = window_buffer_pages(
            num_slots=num_slots, chunk_tokens=chunk_tokens, batch_rows=batch_rows
        )
        # The SWA pool adds one reserved 256-token logical page to the size.
        self.state = pool_factory((pages - RESERVED_PAGES) * SWA_PAGE_TOKENS, layers)
        assert self.state.num_pages == pages, (self.state.num_pages, pages)

    def reset(self, slots: torch.Tensor) -> None:
        # A restarted window reads no ring row: the fold floors it at its first row.
        del slots

    def initialize_dummy_history(self) -> None:
        pass
