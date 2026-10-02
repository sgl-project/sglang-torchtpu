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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.disaggregation.utils import find_prefix_send_end_before_pages

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
    return find_prefix_send_end_before_pages(
        prefix_indices=req.prefix_indices,
        start_send_idx=req.start_send_idx,
        candidate_send_end=candidate_send_end,
        unready_page_ids=scheduler.tpu_inflight_kv_write_page_ids,
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
        req_to_token_table=scheduler.req_to_token_pool.req_to_token,
        page_size=scheduler.token_to_kv_pool_allocator.page_size,
    )
    if inflight_batch.hicache_load_page_ids is not None:
        # A torch_tpu HiCache load scatters into these pages on device,
        # queued with the forward, so it may not have landed yet either.
        page_ids = torch.cat([page_ids, inflight_batch.hicache_load_page_ids])
    scheduler.tpu_inflight_kv_write_page_ids = page_ids


def _collect_prefill_write_page_ids(
    reqs: Sequence[Req], req_to_token_table: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Collect physical KV page IDs touched by the requests' extend ranges.

    Each extend range contains the token positions computed by this forward.
    Look up one KV slot per touched page, including partial first and last
    pages. Return the IDs on CPU. Duplicates are harmless for membership checks.
    """
    rows = []
    for req in reqs:
        rng = req.extend_range
        if rng is None or rng.end <= rng.start:
            continue
        # Sample the first token in the range, then each later page boundary.
        # This includes an unaligned first page without scanning every token.
        # Example: page_size=4, range [2, 9) samples positions [2, 4, 8].
        # If their KV slots are [14, 24, 32], the page IDs are [3, 6, 8].
        first_boundary = (rng.start // page_size + 1) * page_size
        positions = torch.tensor(
            [rng.start, *range(first_boundary, rng.end, page_size)],
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
