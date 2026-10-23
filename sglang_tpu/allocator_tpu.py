# SPDX-License-Identifier: Apache-2.0
"""TPU paged memory allocator.

CPU-resident free-page bookkeeping.

Why CPU bookkeeping on TPU
--------------------------
PJRT compiles a fresh HLO graph for every unique tensor shape it sees.
The upstream allocator keeps ``self.free_pages`` as a device tensor
that shrinks by one entry every page-boundary allocation
(``self.free_pages = self.free_pages[num_new_pages:]``). The gather
``self.free_pages[start_new_pages]`` inside ``alloc_decode`` therefore
sees a different input shape every ``page_size`` decode tokens, which
triggers a ~25 ms TPU recompile per page boundary (~25% of long-decode
wall time at page_size=32).

This subclass keeps the free/release pools as host-resident **CPU torch
tensors** (adapted from the numpy-based CPU-resident allocator in
``sglang-jax``; ported to torch tensors for the torch_tpu backend).
``alloc_decode`` precomputes the
assigned page-IDs on CPU and ships a fixed-shape ``(bs,)`` tensor to the
device-side formula with a direct ``cpu_tensor.to(device)`` — no
``from_numpy`` round-trip. The compiled HLO graph has constant input
shapes regardless of how many pages have been consumed, so PJRT reuses
its cache.

The fix only affects the TPU subclass; the parent
:class:`PagedTokenToKVPoolAllocator` is unchanged for other backends.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_num_new_pages
from sglang_tpu.kv_free_fence import HeldPages


def alloc_extend_vectorized_cpu(
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fill ``out_indices`` like ``alloc_extend_naive`` without a per-request loop.

    Each request contributes three contiguous runs of token slots: the rest of
    its current partial page (continuing from ``last_loc``), whole new pages,
    and a trailing partial page. All inputs are CPU tensors.

    Example with page_size=4, prefix_lens=[3, 0, 5], seq_lens=[9, 2, 8],
    last_loc=[22, -1, 48], and free_pages=[10, 27, 35, 42]:
      prefix_page_end = [4, 0, 8]
      num1 = [1, 0, 3]  # Continue the current page.
      num2 = [4, 0, 0]  # Fill whole new pages.
      num3 = [1, 2, 0]  # Fill a final partial new page.
    Request 0 gets [23, 40, 41, 42, 43, 108], request 1 gets [140, 141],
    and request 2 gets [49, 50, 51]. Free page 42 is unused.
    """
    prefix_lens = prefix_lens.to(torch.int64)
    seq_lens = seq_lens.to(torch.int64)
    last_loc = last_loc.to(torch.int64)
    free_pages = free_pages.to(torch.int64)

    extend_lens = seq_lens - prefix_lens
    total = int(extend_lens.sum())
    if total == 0:
        return

    prefix_page_end = (prefix_lens + page_size - 1) // page_size * page_size
    num1 = torch.minimum(seq_lens, prefix_page_end) - prefix_lens
    remaining = extend_lens - num1
    num2 = remaining // page_size * page_size
    num3 = remaining - num2
    num_new_pages = num2 // page_size + (num3 > 0).to(torch.int64)
    page_start = torch.cumsum(num_new_pages, 0) - num_new_pages

    # Segment k of request i (k = 0: partial current page, 1: full new pages,
    # 2: partial last page) covers ``seg_len[i, k]`` slots. Segments 0 and 2
    # each continue one page from a base slot; segment 1 spans several pages
    # that need not be contiguous, so its slots look their page up directly.
    seg_len = torch.stack([num1, num2, num3], dim=1)
    zeros = torch.zeros_like(num3)
    if free_pages.numel() > 0:
        last_page_idx = (page_start + num_new_pages - 1).clamp(
            min=0, max=free_pages.numel() - 1
        )
        last_page_base = torch.where(
            num3 > 0, free_pages[last_page_idx] * page_size, zeros
        )
    else:
        last_page_base = zeros
    seg_base = torch.stack([last_loc + 1, zeros, last_page_base], dim=1)

    seg_len_flat = seg_len.reshape(-1)
    seg_base_flat = seg_base.reshape(-1)
    seg_start = torch.cumsum(seg_len_flat, 0) - seg_len_flat
    within = torch.arange(total, dtype=torch.int64) - torch.repeat_interleave(
        seg_start, seg_len_flat, output_size=total
    )
    values = (
        torch.repeat_interleave(seg_base_flat, seg_len_flat, output_size=total) + within
    )
    seg_id = torch.repeat_interleave(
        torch.arange(seg_len_flat.numel(), dtype=torch.int64),
        seg_len_flat,
        output_size=total,
    )
    is_full_seg = seg_id % 3 == 1
    if bool(is_full_seg.any()):
        req_of_slot = seg_id // 3
        page_idx = (page_start[req_of_slot] + within // page_size).clamp(
            max=free_pages.numel() - 1
        )
        full_value = free_pages[page_idx] * page_size + within % page_size
        values = torch.where(is_full_seg, full_value, values)
    out_indices[:total] = values.to(out_indices.dtype)


class TpuPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    # Set by the non-spec host-padding policy; wrapped allocators retain
    # their existing device-return contract.
    defer_decode_upload: bool = False

    def clear(self) -> None:
        # Slot 0 is reserved as the dummy-padding page, so page IDs start at 1.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int32, device="cpu"
        )
        # None outside a free_group_begin()/free_group_end() pair, as in the parent.
        self.free_group = None
        self.free_page_ids_group = []
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0
        # Pages freed while a forward was in flight; see kv_free_fence.py.
        self.held_pages = HeldPages()
        # True while a page sits in free_pages, staged_pages or held_pages; a
        # second release would let a later allocation hand it to two owners.
        self.page_is_free = torch.ones(self.num_pages + 1, dtype=torch.bool)
        self.page_is_free[0] = False

    def available_size(self):
        return super().available_size() + self.held_pages.num_pages * self.page_size

    def _ensure_free_pages(self, num_pages: int) -> bool:
        """Make ``free_pages`` hold at least ``num_pages``, taking held pages
        when short."""
        if (
            self.kv_free_fence is not None
            and not self.kv_free_fence.in_forward_write_target
        ):
            self.kv_free_fence.stats.non_forward_allocs += 1
        if num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        short = num_pages - len(self.free_pages)
        if short <= 0:
            return True
        if short > self.held_pages.num_pages:
            return False
        if self.kv_free_fence.in_forward_write_target:
            # Device compute runs in launch order, so this forward writes last.
            self.kv_free_fence.stats.pages_borrowed += short
            self.free_pages = torch.cat(
                (self.held_pages.borrow(short), self.free_pages)
            )
        else:
            # Other allocations may be DMA targets, which are not ordered after
            # the forward that may still write these.
            self.kv_free_fence.wait_for_in_flight()
            self.merge_and_sort_free()
        return num_pages <= len(self.free_pages)

    def alloc(self, need_size: int):
        # Page-aligned bulk allocation, returns contiguous device indices.
        assert need_size % self.page_size == 0, (
            "The allocation size should be page-aligned"
        )
        num_pages = need_size // self.page_size
        if not self._ensure_free_pages(num_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self._mark_handed_out(out_pages)

        # Expand to per-token indices on CPU, then one async H2D.
        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, dtype=torch.int32)
        ).reshape(-1)
        return out_indices.to(self.device, non_blocking=True)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
        out_indices_cpu: torch.Tensor = None,
        return_host: bool = False,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if num_new_pages is None:
            num_new_pages_item = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        else:
            num_new_pages_item = num_new_pages

        if not self._ensure_free_pages(num_new_pages_item):
            return None

        # Compute the per-token page indices entirely on host, then do a
        # single H2D copy. Batch-wide tensor operations avoid the per-request
        # Python loop in ``alloc_extend_naive``. ``prefix_lens_cpu`` and
        # ``seq_lens_cpu`` are already mirrors of the device tensors, so only
        # ``last_loc`` needs a D2H copy when it is not already on the host.
        if num_new_pages_item > 0:
            free_pages_cpu = self.free_pages[:num_new_pages_item]
        else:
            free_pages_cpu = torch.empty((0,), dtype=torch.int32, device="cpu")

        if out_indices_cpu is None:
            out_indices_cpu = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int32,
                device="cpu",
            )
        alloc_extend_vectorized_cpu(
            prefix_lens_cpu,
            seq_lens_cpu,
            last_loc.cpu(),
            free_pages_cpu,
            out_indices_cpu,
            self.page_size,
        )
        out_indices = (
            out_indices_cpu
            if return_host
            else out_indices_cpu.to(self.device, non_blocking=True)
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        self._mark_handed_out(free_pages_cpu)
        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        out_indices_cpu: torch.Tensor | None = None,
    ):
        """``out_indices_cpu``: optional caller-owned host (bs,) buffer filled
        with the returned locations, so host consumers skip a D2H readback.
        Requires a host-resident ``last_loc``."""
        if out_indices_cpu is not None and last_loc.device.type != "cpu":
            raise ValueError("out_indices_cpu requires a host-resident last_loc")
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        # Everything below is computed on CPU torch tensors; we never
        # index into ``self.free_pages`` from a device kernel, so its
        # size doesn't affect any HLO.
        need_new_pages = (seq_lens_cpu % self.page_size == 1).to(torch.int32)
        num_new_pages = int(need_new_pages.sum())

        if not self._ensure_free_pages(num_new_pages):
            return None

        bs = int(seq_lens.shape[0])
        # ``per_req_new_page[i]`` = page-ID assigned to req i if it
        # needs a fresh page, else 0 (multiplied by 0 in the formula
        # below — the sentinel is harmless).
        per_req_new_page = torch.zeros(bs, dtype=torch.int32, device="cpu")
        if num_new_pages > 0:
            new_page_ids = self.free_pages[:num_new_pages]
            start_new_pages = torch.cumsum(need_new_pages, 0) - need_new_pages
            mask = need_new_pages.to(torch.bool)
            per_req_new_page[mask] = new_page_ids[start_new_pages[mask].long()]
            self.free_pages = self.free_pages[num_new_pages:]
            self._mark_handed_out(new_page_ids)

        #   need_new_pages[i] == 1  →  out_indices[i] = new_page * page_size
        #   need_new_pages[i] == 0  →  out_indices[i] = last_loc + 1
        if last_loc.device.type == "cpu":
            # Host-resident: last_loc is host, so run the formula on CPU and
            # H2D once (no per-bucket device compile of the allocator formula).
            host_indices = (last_loc + 1) * (1 - need_new_pages) + (
                per_req_new_page * self.page_size * need_new_pages
            )
            if out_indices_cpu is not None:
                out_indices_cpu.copy_(host_indices)
            out_indices = (
                host_indices
                if self.defer_decode_upload
                else host_indices.to(self.device, non_blocking=True)
            )
        else:
            # Device formula on fixed-shape (bs,) inputs (bs is bucketed).
            per_req_new_page_dev = per_req_new_page.to(self.device, non_blocking=True)
            need_new_pages_dev = need_new_pages.to(self.device, non_blocking=True)
            out_indices = (last_loc + 1) * (1 - need_new_pages_dev) + (
                per_req_new_page_dev * self.page_size * need_new_pages_dev
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        return out_indices.int()

    def _mark_handed_out(self, page_ids: torch.Tensor) -> None:
        if page_ids.numel() == 0:
            return
        ids = page_ids.long()
        if not bool(self.page_is_free[ids].all()):
            raise RuntimeError(
                "KV page allocator handed out pages that are not free: "
                f"{ids[~self.page_is_free[ids]].tolist()}"
            )
        self.page_is_free[ids] = False

    def _mark_released(self, page_ids: torch.Tensor) -> None:
        if page_ids.numel() == 0:
            return
        ids = page_ids.long()
        already_free = self.page_is_free[ids]
        if bool(already_free.any()):
            raise RuntimeError(
                "KV page allocator released pages that are already free: "
                f"{ids[already_free].tolist()}"
            )
        unique_ids, counts = torch.unique(ids, return_counts=True)
        if unique_ids.numel() != ids.numel():
            raise RuntimeError(
                "KV page allocator released pages twice in one call: "
                f"{unique_ids[counts > 1].tolist()}"
            )
        self.page_is_free[ids] = True

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        """Keep every page-id release path on the host pool's int32 dtype."""
        normalized = tuple(
            page_id.to(device="cpu", dtype=torch.int32) for page_id in page_ids
        )
        self._mark_released(torch.cat(normalized))
        tag = (
            None
            if self.kv_free_fence is None
            else self.kv_free_fence.hold_tag(
                num_pages=sum(page_id.numel() for page_id in normalized)
            )
        )
        if tag is None:
            super()._release_page_ids(*normalized)
        else:
            self.held_pages.push(tag, torch.cat(normalized))

    def release_held_pages(self, completed: int) -> None:
        released = self.held_pages.pop_released(completed)
        if released:
            super()._release_page_ids(*released)

    def free(self, free_index: torch.Tensor) -> None:
        # Page pools are host-resident int32 tensors. Scheduler locations can
        # be int64, so normalize before the parent derives and reinserts page
        # IDs; otherwise torch.cat promotes the pool and the next decode-page
        # assignment fails against its int32 destination.
        if free_index.device.type != "cpu":
            free_index = free_index.cpu()
        free_index = free_index.to(dtype=torch.int32)
        super().free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        # Radix-cache node values are int64. Normalize to host-resident int32 so
        # torch.cat does not promote the release pool, keeping the free pages
        # int32 for subsequent decode assignments.
        if free_index.device.type != "cpu":
            free_index = free_index.cpu()
        free_index = free_index.to(dtype=torch.int32)
        super().free_segment(free_index, start_pos=start_pos)


def host_return_allocators(allocator) -> list[TpuPagedTokenToKVPoolAllocator]:
    """TPU allocators behind ``allocator`` that can return host-resident indices.

    Empty when any inner allocator would keep its device-return contract. An SWA
    allocator qualifies when both inner allocators are plain TPU allocators; its
    full-to-SWA mapping is host-resident on TPU.
    """
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    if isinstance(allocator, TpuPagedTokenToKVPoolAllocator):
        return [allocator]
    if isinstance(allocator, SWATokenToKVPoolAllocator):
        inner = [allocator.full_attn_allocator, allocator.swa_attn_allocator]
        if all(isinstance(a, TpuPagedTokenToKVPoolAllocator) for a in inner) and (
            allocator.full_to_swa_index_mapping.device.type == "cpu"
        ):
            return inner
    return []
