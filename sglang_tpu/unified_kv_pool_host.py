# SPDX-License-Identifier: Apache-2.0
"""HiCache host pools for the unified full/sliding-window KV pool on TPU.

The unified device pool has one set of per-layer buffers and one page-id space
for both attention kinds, so one raiden engine over those buffers and one host
page list serve both: a host page, like a device page, holds either a full page
or a sliding-window page, and host capacity moves to whichever kind needs it.
HiCache keeps one host pool entry per kind, because the tree stores separate
full and sliding-window host indices; both entries' host pools are facades over
the one shared raiden host pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_memory
from sglang_tpu.kv_pool_host import TpuRaidenHostKVCache

if TYPE_CHECKING:
    from sglang_tpu.raiden_transfer import RaidenFutureEvent
    from sglang_tpu.unified_kv_pool import _SharedBufferView


class TpuRaidenUnifiedHostKVCache(TpuRaidenHostKVCache):
    """One attention kind's host pool over the unified pool's shared raiden
    host pool, which it builds if it is the first of the two.

    A transfer copies a page's row in every shared buffer, whichever kind owns
    the page; rows in the buffers the owning kind does not use are dead bytes on
    both tiers. ``size``, ``page_num`` and ``size_per_token`` are therefore the
    shared pool's, and allocation, release and transfers go to the shared pool
    without remapping page ids. ``device_pool`` and the layer range are this
    kind's view. On the sliding-window side of a pool with several groups, a
    page takes one host page per group, and every transfer moves all of them.
    """

    def __init__(
        self,
        device_pool: _SharedBufferView,
        host_to_device_ratio: float,
        host_size: float,
        page_size: int,
        layout: str,
        pin_memory: bool = False,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        pool_label: str = "kv",
        mtp_draft_device_pools: tuple = (),
    ):
        assert not mtp_draft_device_pools, "the unified KV pool has no MTP drafts"
        # Like the parent, skip HostKVCache.__init__; unlike it, build no raiden
        # engine or page list: the shared pool owns both.
        backing = device_pool.backing
        if backing.host_pool is None:
            # One pool serves both kinds, so it takes the whole --hicache-size
            # budget; host_size is only this kind's share of it.
            backing.host_pool = TpuRaidenHostKVCache(
                device_pool=backing,
                host_to_device_ratio=host_to_device_ratio,
                host_size=get_memory().hicache_size,
                page_size=page_size,
                layout=layout,
                pool_label="unified",
            )
        self._shared = backing.host_pool

        self.device_pool = device_pool
        self.pool_label = pool_label
        self.page_size = self._shared.page_size
        self.layout = self._shared.layout
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = self._shared.dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num

        self.size_per_token = self._shared.size_per_token
        self.size = self._shared.size
        self.page_num = self._shared.page_num

        # A sliding-window page of a pool with several groups is one page per
        # group on both tiers: [g - 1, p] is group g's host page of host page p,
        # as device_pool.group_pages is on the device.
        self._device_group_pages = device_pool.group_pages
        self._host_group_pages: Optional[torch.Tensor] = None
        self._pages_per_id = 1
        if self._device_group_pages is not None:
            num_extra = self._device_group_pages.shape[0]
            self._host_group_pages = torch.zeros(
                (num_extra, self.page_num), dtype=torch.int64
            )
            self._pages_per_id = num_extra + 1

    def clear(self) -> None:
        # Resets the page list both kinds share, so clearing both facades
        # resets it twice to the same state.
        self._shared.clear()

    def available_size(self) -> int:
        free_pages = self._shared.available_size() // self.page_size
        return free_pages // self._pages_per_id * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._host_group_pages is None:
            return self._shared.alloc(need_size)
        out = self._shared.alloc(need_size * self._pages_per_id)
        if out is None:
            return None
        pages = out[:: self.page_size] // self.page_size
        num_pages = need_size // self.page_size
        self._host_group_pages[:, pages[:num_pages]] = pages[num_pages:].view(
            -1, num_pages
        )
        return out[:need_size]

    def free(self, indices: torch.Tensor) -> int:
        if self._host_group_pages is None:
            return self._shared.free(indices)
        pages = torch.unique(indices.cpu() // self.page_size)
        self._shared.free(
            torch.cat([indices.cpu(), self._tokens(self._host_group_pages[:, pages])])
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> RaidenFutureEvent:
        if self._host_group_pages is None:
            return self._shared.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )
        device_pages, host_pages = self._group_page_pairs(device_indices, host_indices)
        return self._shared.transfer.backup(device_pages, host_pages)

    def load_all_layer(
        self, device_pool, host_indices, device_indices
    ) -> RaidenFutureEvent:
        if self._host_group_pages is None:
            return self._shared.load_all_layer(
                device_pool, host_indices, device_indices
            )
        device_pages, host_pages = self._group_page_pairs(device_indices, host_indices)
        return self._shared.transfer.load(host_pages, device_pages)

    def _group_page_pairs(self, device_indices, host_indices):
        """Device and host page ids of every group, pairwise in one list each."""
        # Whole-page token runs, one page id per run, as the shared pool reads them.
        device_pages = device_indices.cpu()[:: self.page_size].long() // self.page_size
        host_pages = host_indices.cpu()[:: self.page_size].long() // self.page_size
        device_all = torch.cat(
            [device_pages, self._device_group_pages[:, device_pages].reshape(-1)]
        )
        host_all = torch.cat(
            [host_pages, self._host_group_pages[:, host_pages].reshape(-1)]
        )
        return device_all.tolist(), host_all.tolist()

    def _tokens(self, pages: torch.Tensor) -> torch.Tensor:
        pages = pages.reshape(-1, 1)
        return (pages * self.page_size + torch.arange(self.page_size)).reshape(-1)
