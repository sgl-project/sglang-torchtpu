# SPDX-License-Identifier: Apache-2.0
"""Overlap cached-prefix KV transfers with TPU prefill computation.

Before computing a request's uncached suffix, ``run_batch`` can offer its
cached prefix to the transfer backend. However, a prefix cache hit does not
always mean the KV is ready: overlap scheduling can insert a chunk into the
radix tree before its TPU computation finishes. Another request can then match
those same pages while they are still being written.

Raiden reads directly from TPU memory, so a transfer must avoid pages that an
unfinished forward may write. The scheduler tracks their physical page IDs in
``tpu_inflight_kv_write_page_ids``. The overlap prefill loop uses two helpers:

- Before launching a forward, ``tpu_clip_cached_prefix_send_end`` limits each
  early send to the prefix before the first page the previous forward may write.
  Pages it holds back go out with the request's next send.
- After processing the previous result, ``tpu_update_inflight_kv_write_pages``
  replaces the tracked IDs with those of the newly launched forward, plus the
  pages its HiCache load-backs fill.

A HiCache load-back has the same problem within one batch, in every prefill
loop. It commits its pages to the radix tree when it is queued, so a request
admitted later in the same pass matches them as device-resident. The batch's
forward waits for the load, but the early send runs first.
``tpu_clip_hicache_load_send_end`` stops the send before those pages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


def tpu_clip_cached_prefix_send_end(
    scheduler: Scheduler, req: Req, candidate_send_end: int
) -> int:
    """Limit a cached-prefix send to pages the unfinished forward cannot write.

    ``maybe_send_cached_prefix_chunk`` calls this before each TPU early send
    under overlap scheduling. ``candidate_send_end`` is an exclusive token
    position, already aligned to a page boundary. The returned position stops
    before the first conflicting page, or keeps the proposed end if none match.
    """
    if (
        scheduler.tpu_inflight_kv_write_page_ids is None
        or scheduler.tpu_inflight_kv_write_page_ids.numel() == 0
    ):
        return candidate_send_end
    return _find_safe_prefix_send_end(
        prefix_indices=scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :candidate_send_end
        ],
        start_send_idx=req.start_send_idx,
        candidate_send_end=candidate_send_end,
        inflight_write_page_ids=scheduler.tpu_inflight_kv_write_page_ids,
        page_size=scheduler.token_to_kv_pool_allocator.page_size,
    )


def tpu_hicache_load_page_ids(
    scheduler: Scheduler, load_indices: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Convert a batch's load-back KV slot indices to unique page IDs on CPU.

    Divides by the allocator's page size, which DCP widens beyond the radix page
    size, the same unit the early-send clips use for prefix slots.
    """
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    # A page's slots are adjacent within a load, so collapse each run first;
    # the final sort then sees about one entry per page.
    page_ids = [
        torch.unique_consecutive(indices.cpu() // page_size).to(torch.int64)
        for indices in load_indices
    ]
    return torch.unique(torch.cat(page_ids))


def tpu_clip_hicache_load_send_end(
    scheduler: Scheduler,
    req: Req,
    candidate_send_end: int,
    hicache_load_page_ids: Optional[torch.Tensor],
) -> int:
    """Limit a cached-prefix send to pages this batch's HiCache loads do not fill.

    ``hicache_load_page_ids`` holds the KV page IDs of the load-backs queued for
    the batch about to launch. Same contract as ``tpu_clip_cached_prefix_send_end``.
    """
    if hicache_load_page_ids is None or hicache_load_page_ids.numel() == 0:
        return candidate_send_end
    return _find_safe_prefix_send_end(
        prefix_indices=scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :candidate_send_end
        ],
        start_send_idx=req.start_send_idx,
        candidate_send_end=candidate_send_end,
        inflight_write_page_ids=hicache_load_page_ids,
        page_size=scheduler.token_to_kv_pool_allocator.page_size,
    )


def tpu_update_inflight_kv_write_pages(
    scheduler: Scheduler, inflight_batch: Optional[ScheduleBatch]
) -> None:
    """Track the current forward's write pages, or clear them if no batch launched.

    Call each overlap iteration with early send enabled, after processing the
    previous result. These page IDs limit the next iteration's early send.
    """
    if inflight_batch is None:
        scheduler.tpu_inflight_kv_write_page_ids = None
        return
    page_ids = _collect_prefill_write_page_ids(
        reqs=inflight_batch.reqs,
        prefix_lens=inflight_batch.prefix_lens,
        extend_lens=inflight_batch.extend_lens,
        req_to_token_table=scheduler.req_to_token_pool.req_to_token,
        page_size=scheduler.token_to_kv_pool_allocator.page_size,
    )
    if inflight_batch.hicache_load_page_ids is not None:
        # A torch_tpu HiCache load scatters into these pages on device,
        # queued with the forward, so it may not have landed yet either.
        page_ids = torch.cat([page_ids, inflight_batch.hicache_load_page_ids])
    scheduler.tpu_inflight_kv_write_page_ids = page_ids


def _find_safe_prefix_send_end(
    prefix_indices: torch.Tensor,
    start_send_idx: int,
    candidate_send_end: int,
    inflight_write_page_ids: Optional[torch.Tensor],
    page_size: int,
) -> int:
    """Find where a prefix send must stop to avoid pending KV writes.

    The send covers token positions ``[start_send_idx, candidate_send_end)``,
    and the proposed end must be page-aligned. ``prefix_indices`` maps each token
    position to a physical KV slot. Return the start of the first page whose
    ID is tracked as a pending write, or the proposed end if no page conflicts.

    Example: page_size=4, start_send_idx=0, candidate_send_end=12::

        Token positions:    [0, 4)    [4, 8)    [8, 12)
        Physical page IDs:     5         9          2
        Pending write IDs: {9}

    Return 4: only tokens [0, 4) can be sent now. Page 2 is also ready, but
    the send must stop before page 9 to keep the token range contiguous.
    With no matching pending writes, return the proposed end, 12.
    """
    if inflight_write_page_ids is None or inflight_write_page_ids.numel() == 0:
        return candidate_send_end
    # Include the page containing the first unsent token, even if that token
    # is not at the start of the page.
    start = start_send_idx - start_send_idx % page_size
    if candidate_send_end <= start:
        return candidate_send_end
    if prefix_indices.device.type != "cpu":
        prefix_indices = prefix_indices.cpu()
    # Token positions and KV slots share page boundaries. Sample one slot per
    # page and divide by page_size to obtain its physical page ID.
    page_ids = (
        prefix_indices[start:candidate_send_end:page_size].to(torch.int64) // page_size
    )
    hit = torch.isin(page_ids, inflight_write_page_ids)
    if not bool(hit.any()):
        return candidate_send_end
    return start + int(hit.nonzero()[0].item()) * page_size


def _collect_prefill_write_page_ids(
    reqs: Sequence[Req],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    req_to_token_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Collect physical KV page IDs touched by the requests' extend ranges.

    Each extend range contains the token positions computed by this forward.
    Look up one KV slot per touched page, including partial first and last
    pages. Return the IDs on CPU. Duplicates are harmless for membership checks.
    """
    if not reqs:
        return torch.empty(0, dtype=torch.int64)
    rows = []
    for req, start, tokens in zip(reqs, prefix_lens, extend_lens, strict=True):
        end = start + tokens
        if tokens == 0:
            continue
        # Sample the first token in the range, then each later page boundary.
        # This includes an unaligned first page without scanning every token.
        # Example: page_size=4, range [2, 9) samples positions [2, 4, 8].
        # If their KV slots are [14, 24, 32], the page IDs are [3, 6, 8].
        first_boundary = (start // page_size + 1) * page_size
        positions = torch.tensor(
            [start, *range(first_boundary, end, page_size)],
            device=req_to_token_table.device,
        )
        rows.append(req_to_token_table[req.kv.req_pool_idx, positions])
    if not rows:
        return torch.empty(0, dtype=torch.int64)
    slots = torch.cat(rows)
    if slots.device.type != "cpu":
        # TPU keeps this table on CPU by default
        # (SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1). If configured on device instead,
        # copy the sampled slots to CPU for the page checks, which can block.
        slots = slots.cpu()
    return slots.to(torch.int64) // page_size
