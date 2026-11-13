# SPDX-License-Identifier: Apache-2.0
"""The K3 decode's KV page allocator under attention DP2 (``sglang_dp``).

A request's pages all come from one DP group's region (pages ``1 .. H`` for
group 0, ``H + 1 .. 2H`` for group 1): a new request takes the group with the
fewest live requests (then the most free pages), a growing request the group
of its last page. Landing pages (``sglang_dp.DpPages.transfer_rows``) come
from the other region and go back once the swap has moved their rows.

Everything else is :class:`TpuPagedTokenToKVPoolAllocator`'s host-resident
bookkeeping: one sorted free list, filtered by region per allocation.
"""

from __future__ import annotations

import torch

from sglang.srt.utils import get_num_new_pages
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
    alloc_extend_vectorized_cpu,
)
from sglang_tpu.k3_programs import sglang_dp as dp

_ACTIVE: list = []


def active() -> "K3DpPagedAllocator | None":
    """The process's DP2 allocator (the decode's), once built."""
    return _ACTIVE[-1] if _ACTIVE else None


class K3DpPagedAllocator(TpuPagedTokenToKVPoolAllocator):
    """Region-aware pages for the two DP groups."""

    def clear(self) -> None:
        super().clear()
        if self.num_pages % dp.GROUPS:
            # An odd last page stays out of both regions.
            self.free_pages = self.free_pages[
                self.free_pages <= self.num_pages // 2 * 2
            ]
            self.page_is_free[self.num_pages] = False
        self.group_requests = [0] * dp.GROUPS
        self.request_group: dict[str, int] = {}
        if self not in _ACTIVE:
            _ACTIVE.append(self)

    # ------------------------------------------------------------ regions
    @property
    def per_group(self) -> int:
        return self.num_pages // dp.GROUPS

    @property
    def dp_pages(self) -> int:
        return self.per_group * dp.GROUPS

    def group_of_page(self, page: int) -> int:
        return (int(page) - 1) // self.per_group

    def _region_mask(self, pages: torch.Tensor, group: int) -> torch.Tensor:
        low = 1 + group * self.per_group
        return (pages >= low) & (pages < low + self.per_group)

    def free_in_group(self, group: int, *, held: bool = False) -> int:
        """Free pages of ``group``'s region (``held``: and those the free
        fence holds until in-flight forwards finish)."""
        lists = [self.free_pages, *getattr(self, "staged_pages", ())]
        if self.release_pages is not None and len(self.release_pages):
            lists.append(self.release_pages)
        if held:
            lists += [ids for _, ids in self.held_pages._entries]
        return sum(int(self._region_mask(pages, group).sum()) for pages in lists)

    def available_size(self):
        """What any one request can count on: the fuller group's free pages
        bound both groups (all of them when idle, as the leak check expects,
        an odd last page included)."""
        per_group = min(self.free_in_group(g, held=True) for g in range(dp.GROUPS))
        odd = self.num_pages - self.dp_pages
        return (dp.GROUPS * per_group + odd) * self.page_size

    def pick_group(self) -> int:
        """A new request's group: fewest live requests, then most free pages."""
        return min(
            range(dp.GROUPS),
            key=lambda g: (self.group_requests[g], -self.free_in_group(g)),
        )

    def _take(self, group: int, num_pages: int) -> torch.Tensor | None:
        """``num_pages`` free pages of ``group``'s region, handed out."""
        if num_pages <= 0:
            return torch.empty((0,), dtype=torch.int32)
        mask = self._region_mask(self.free_pages, group)
        if int(mask.sum()) < num_pages:
            self.merge_and_sort_free()
            mask = self._region_mask(self.free_pages, group)
        if int(mask.sum()) < num_pages and self.held_pages.num_pages:
            if self.kv_free_fence is not None:
                self.kv_free_fence.wait_for_in_flight()
            self.merge_and_sort_free()
            mask = self._region_mask(self.free_pages, group)
        index = torch.nonzero(mask).flatten()[:num_pages]
        if len(index) < num_pages:
            return None
        out = self.free_pages[index]
        keep = torch.ones(len(self.free_pages), dtype=torch.bool)
        keep[index] = False
        self.free_pages = self.free_pages[keep]
        self._mark_handed_out(out)
        return out

    # ------------------------------------------------------------ requests
    def assign(self, rid: str, group: int | None = None) -> int:
        """Place request ``rid`` in a group (the picked one by default)."""
        group = self.pick_group() if group is None else group
        if rid in self.request_group:
            return self.request_group[rid]
        self.request_group[rid] = group
        self.group_requests[group] += 1
        return group

    def release(self, rid: str) -> None:
        group = self.request_group.pop(rid, None)
        if group is not None:
            self.group_requests[group] -= 1

    def alloc_landing(self, group: int, num_pages: int) -> torch.Tensor | None:
        """Landing pages of ``group``'s region (the other group's request
        lands there before the swap)."""
        return self._take(group, num_pages)

    def free_landing(self, pages: torch.Tensor) -> None:
        if len(pages):
            self._release_page_ids(pages.to(dtype=torch.int32))

    # ------------------------------------------------------------ allocation
    def _group_for(self, last_loc: int) -> int:
        page = int(last_loc) // self.page_size if int(last_loc) >= 0 else 0
        if page >= 1:
            return self.group_of_page(page)
        return (
            self.pending_group if self.pending_group is not None else self.pick_group()
        )

    pending_group: int | None = None

    def alloc(self, need_size: int):
        assert need_size % self.page_size == 0
        group = (
            self.pending_group if self.pending_group is not None else self.pick_group()
        )
        pages = self._take(group, need_size // self.page_size)
        if pages is None:
            return None
        out = (
            pages[:, None] * self.page_size
            + torch.arange(self.page_size, dtype=torch.int32)
        ).reshape(-1)
        return out.to(self.device, non_blocking=True)

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
        last = last_loc.cpu()
        per_request = [
            int(
                get_num_new_pages(
                    seq_lens=seq_lens_cpu[i : i + 1],
                    page_size=self.page_size,
                    prefix_lens=prefix_lens_cpu[i : i + 1],
                )
            )
            for i in range(len(seq_lens_cpu))
        ]
        taken: list[torch.Tensor] = []
        for i, count in enumerate(per_request):
            pages = self._take(self._group_for(int(last[i])), count)
            if pages is None:
                for got in taken:  # all or nothing
                    self.free_landing(got)
                return None
            taken.append(pages)
        free_pages_cpu = (
            torch.cat(taken) if taken else torch.empty((0,), dtype=torch.int32)
        )
        if out_indices_cpu is None:
            out_indices_cpu = torch.empty((extend_num_tokens,), dtype=torch.int32)
        alloc_extend_vectorized_cpu(
            prefix_lens_cpu,
            seq_lens_cpu,
            last,
            free_pages_cpu,
            out_indices_cpu,
            self.page_size,
        )
        out = (
            out_indices_cpu
            if return_host
            else out_indices_cpu.to(self.device, non_blocking=True)
        )
        return out.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        out_indices_cpu: torch.Tensor | None = None,
    ):
        last = last_loc.cpu()
        need = (seq_lens_cpu % self.page_size == 1).to(torch.int32)
        new_page = torch.zeros(len(seq_lens_cpu), dtype=torch.int32)
        taken: list[torch.Tensor] = []
        for i in torch.nonzero(need).flatten().tolist():
            pages = self._take(self._group_for(int(last[i])), 1)
            if pages is None:
                for got in taken:
                    self.free_landing(got)
                return None
            taken.append(pages)
            new_page[i] = pages[0]
        host = (last + 1) * (1 - need) + new_page * self.page_size * need
        if out_indices_cpu is not None:
            out_indices_cpu.copy_(host)
        if last_loc.device.type == "cpu" and self.defer_decode_upload:
            return host.int()
        return host.to(self.device, non_blocking=True).int()
