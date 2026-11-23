# SPDX-License-Identifier: Apache-2.0
"""Row metadata of a spec draft / verify forward from its device positions.

A spec step's rows sit at the device sequence lengths, which the host would only
learn by reading them back, and that read waits for the previous verify. The
fields that depend on the positions are elementwise on the positions and cache
slots, so they are built here with device ops; whatever depends only on the
batch layout (row counts, slots) still comes from the host. Every function gives
the same values as its host counterpart in attention_backend / request_window.
"""

from typing import Tuple

import torch

from sglang_tpu.models.deepseek_v41.request_window import (
    SWA_PAGE_TOKENS,
    WINDOW_TOKENS,
    WindowPlan,
    ring_rows,
)


def token_rows(
    *,
    host_rows: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """attention_backend._token_rows with the cache slots on the device.

    ``host_rows``: [T] batch row of each request-contiguous row (-1 past the
    counts) and ``req_pool_indices``: [B], both already on the slots' device.
    """
    valid = (host_rows >= 0) & (out_cache_loc > 0)
    rows = host_rows.masked_fill(~valid, -1)
    req = req_pool_indices[rows.clamp_min(0)].masked_fill(~valid, 0)
    return rows, valid, req


def low_ratio_rows(
    *,
    ratio: int,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    valid: torch.Tensor,
    compress_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """attention_backend._low_ratio_rows in the inputs' dtype (int32 on TPU)."""
    seq = positions + 1
    completes = (seq % ratio == 0) & compress_valid
    out_loc = torch.where(completes, out_cache_loc // ratio, -1)
    return out_loc, torch.where(valid, seq // ratio, 0)


def window_live(slots: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """[B] bool on the host: the request moves window rows (extend_window_pages'
    rule: it has rows, a real slot, and is its slot's first request)."""
    first_of_slot = torch.ones_like(slots, dtype=torch.bool)
    if slots.numel() > 1:
        order = torch.argsort(slots, stable=True)
        sorted_slots = slots[order]
        repeat = torch.zeros_like(first_of_slot)
        repeat[1:] = sorted_slots[1:] == sorted_slots[:-1]
        first_of_slot[order] = ~repeat
    return (counts > 0) & (slots != 0) & first_of_slot


def window_pages(
    *,
    first: torch.Tensor,
    counts: torch.Tensor,
    live: torch.Tensor,
    slots: torch.Tensor,
    minimum: torch.Tensor,
    columns: int,
    base_page: int,
) -> Tuple[torch.Tensor, WindowPlan]:
    """request_window.extend_window_pages with ``first`` on the device.

    ``counts``, ``slots``, ``minimum`` ([B]) and ``live`` ([B] bool: the slot's
    request moves rows) come from the host layout, uploaded to ``first``'s device.
    The workspace capacity and SWA-table bound checks of the host function are
    the caller's: spec rows add at most a few pages per request.
    """
    lane = torch.arange(WINDOW_TOKENS, dtype=first.dtype, device=first.device)[None, :]
    pages = _workspace_pages(
        first=first, counts=counts, columns=columns, base_page=base_page
    )
    live = live[:, None]
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
    # cumsum would promote int32 to int64, which the TPU vector units do not lower.
    row_start = (torch.cumsum(counts, 0, dtype=counts.dtype) - counts)[:, None]
    commit_token = torch.where(kept_ok, row_start + kept - first[:, None], 0)
    commit_dst = torch.where(kept_ok, ring_rows(slot, kept.clamp(min=0)), -1)

    # All int32: casting to int64 on the device does not lower on TPU, so with
    # device rows the host plan is uploaded as int32 too (window_plan_upload).
    # Fresh 1-D tensors, not views of the [B, 128] results: TorchTPU compiles a
    # view input as its base buffer, unlike the host plan's uploads.
    plan = WindowPlan(
        *(
            t.to(torch.int32).reshape(-1).clone()
            for t in (history_src, history_dst, commit_token, commit_dst)
        )
    )
    return pages, plan


def _workspace_pages(
    *, first: torch.Tensor, counts: torch.Tensor, columns: int, base_page: int
) -> torch.Tensor:
    """request_window._assign_workspace_pages without the per-request loop: each
    request with rows takes the next fresh pages over [first - 128, first + count)."""
    has_rows = counts > 0
    low = (first - WINDOW_TOKENS).clamp(min=0) // SWA_PAGE_TOKENS
    high = (first + counts - 1) // SWA_PAGE_TOKENS
    width = torch.where(has_rows, high - low + 1, 0)
    offset = torch.cumsum(width, 0, dtype=width.dtype) - width
    col = torch.arange(columns, dtype=first.dtype, device=first.device)[None, :]
    inside = has_rows[:, None] & (col >= low[:, None]) & (col <= high[:, None])
    return torch.where(inside, base_page + offset[:, None] + col - low[:, None], 0)
