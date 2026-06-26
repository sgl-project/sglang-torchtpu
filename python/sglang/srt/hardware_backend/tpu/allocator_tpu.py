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
tensors** (torchtpu-vllm's style). ``alloc_decode`` precomputes the
assigned page-IDs on CPU and ships a fixed-shape ``(bs,)`` tensor to the
device-side formula with a direct ``cpu_tensor.to(device)`` — no
``from_numpy`` round-trip. The compiled HLO graph has constant input
shapes regardless of how many pages have been consumed, so PJRT reuses
its cache.

The fix only affects the TPU subclass; the parent
:class:`PagedTokenToKVPoolAllocator` is unchanged for other backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class TpuPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):

    def clear(self) -> None:
        # Slot 0 is reserved as the dummy-padding page, so page IDs start at 1.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int32, device="cpu"
        )
        self.release_pages = torch.empty((0,), dtype=torch.int32, device="cpu")
        self.is_not_in_free_group = True
        self.free_group = []

    def alloc(self, need_size: int):
        # Page-aligned bulk allocation, returns contiguous device indices.
        assert (
            need_size % self.page_size == 0
        ), "The allocation size should be page-aligned"
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            # Pools are host-resident; keep release_pages on CPU (the parent
            # defaults to self.device, which is the accelerator here).
            self.merge_and_sort_free(device="cpu")
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

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

        if self.need_sort and num_new_pages_item > len(self.free_pages):
            # Pools are host-resident; keep release_pages on CPU (the parent
            # defaults to self.device, which is the accelerator here).
            self.merge_and_sort_free(device="cpu")

        if num_new_pages_item > len(self.free_pages):
            return None

        # Compute the per-token page indices entirely on host, then do a
        # single H2D copy. The shared ``alloc_extend_naive`` iterates over
        # ``range(len(prefix_lens))`` and reads scalar elements off its
        # tensor inputs each iteration — if those tensors live on the TPU,
        # the XLA backend lowers each scalar ``//`` to its own one-element
        # HLO module per iteration (~64 tiny compiles per extend) and every
        # scalar read forces a device→host sync. Running the loop on CPU
        # avoids both. ``prefix_lens_cpu`` / ``seq_lens_cpu`` are already
        # mirrors of the device tensors, so the only extra D2H is the tiny
        # ``last_loc`` vector (one int32 per sequence).
        if num_new_pages_item > 0:
            free_pages_cpu = self.free_pages[:num_new_pages_item]
        else:
            free_pages_cpu = torch.empty((0,), dtype=torch.int32, device="cpu")

        out_indices_cpu = torch.empty(
            (extend_num_tokens,),
            dtype=torch.int32,
            device="cpu",
        )
        alloc_extend_naive(
            prefix_lens_cpu,
            seq_lens_cpu,
            last_loc.cpu(),
            free_pages_cpu,
            out_indices_cpu,
            self.page_size,
            "cpu",
        )
        out_indices = out_indices_cpu.to(self.device, non_blocking=True)

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        # Everything below is computed on CPU torch tensors; we never
        # index into ``self.free_pages`` from a device kernel, so its
        # size doesn't affect any HLO.
        need_new_pages = (seq_lens_cpu % self.page_size == 1).to(torch.int32)
        num_new_pages = int(need_new_pages.sum())

        if num_new_pages > len(self.free_pages):
            # Pools are host-resident; keep release_pages on CPU (the parent
            # defaults to self.device, which is the accelerator here).
            self.merge_and_sort_free(device="cpu")
        if num_new_pages > len(self.free_pages):
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

        #   need_new_pages[i] == 1  →  out_indices[i] = new_page * page_size
        #   need_new_pages[i] == 0  →  out_indices[i] = last_loc + 1
        if last_loc.device.type == "cpu":
            # Host-resident: last_loc is host, so run the formula on CPU and
            # H2D once (no per-bucket device compile of the allocator formula).
            out_indices_cpu = (last_loc + 1) * (1 - need_new_pages) + (
                per_req_new_page * self.page_size * need_new_pages
            )
            out_indices = out_indices_cpu.to(self.device, non_blocking=True)
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

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            # D2H copy of the freed token indices, then everything
            # stays on host until merge_and_sort_free flushes them
            # back into self.free_pages.
            free_page_indices = torch.unique(free_index.cpu() // self.page_size).to(
                torch.int32
            )
            if self.need_sort:
                self.release_pages = torch.cat([free_page_indices, self.release_pages])
            else:
                self.free_pages = torch.cat([free_page_indices, self.free_pages])
        else:
            # Aggregated frees still come in as device tensors; flush
            # at free_group_end -> free.
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)
