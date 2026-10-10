# SPDX-License-Identifier: Apache-2.0
"""Page-granular retraction backup for the TPU fused KV pool over raiden.

A retracted request's pages are staged into raiden's pinned host pool by DMA
and loaded back into the pages allocated at resume. The bytes never change
representation -- device pages out, the same pages back -- so no kernel runs,
nothing recompiles, and the transfer moves every layer of a page in one DMA.

Enabled by ``SGLANG_RAIDEN_RETRACT_HOST_GB`` (a pinned-host budget; ``0``
keeps the plain row-copy path). A request whose rows do not fill its pages
sequentially falls back to the row path at save time; a page saved here is
restored whole, which is sound because retraction allocates its resume pages
exclusively and row offsets are preserved by the page-aligned allocator.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def sequential_page_runs(
    indices: torch.Tensor, page_size: int
) -> Optional[Tuple[List[int], int, int]]:
    """The pages ``indices`` occupy, when they fill them sequentially.

    Returns ``(page_ids, first_offset, num_rows)`` when row ``k`` sits at
    offset ``(first_offset + k) % page_size`` of its page and pages change
    exactly at page boundaries -- the shape a retracted request's slots have.
    The pages themselves need not be adjacent. None otherwise; such an index
    set needs the row path.
    """
    n = indices.numel()
    if n == 0:
        return None
    idx = indices.to("cpu", copy=False).long()
    first = int(idx[0])
    first_offset = first % page_size
    pages = idx // page_size
    offsets = idx % page_size
    expected_off = (first_offset + torch.arange(n)) % page_size
    if not torch.equal(offsets, expected_off):
        return None
    boundaries = torch.nonzero(offsets == 0).flatten()
    changes = torch.nonzero(pages[1:] != pages[:-1]).flatten() + 1
    if not torch.equal(changes, boundaries[boundaries > 0]):
        return None
    page_ids = pages[torch.cat([torch.tensor([0]), changes])].tolist()
    if len(set(page_ids)) != len(page_ids):
        return None
    return page_ids, first_offset, n


class RaidenKVBackupHandle:
    """Host-resident pages of one retracted request.

    Frees its host pages when the framework drops it (``del req.kv_cache_cpu``
    on resume or abort).
    """

    def __init__(
        self,
        backup: RaidenKVBackup,
        host_pages: List[int],
        first_offset: int,
        num_rows: int,
    ):
        self._backup = backup
        self.host_pages = host_pages
        self.first_offset = first_offset
        self.num_rows = num_rows

    def release(self) -> None:
        if self._backup is not None:
            self._backup._free_pages(self.host_pages)
            self._backup = None
            self.host_pages = []

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class RaidenKVBackup:
    """Pinned host staging for retracted requests, moved by raiden DMA."""

    def __init__(self, device_pool, num_host_pages: int):
        from sglang.srt.hardware_backend.tpu.raiden_transfer import RaidenKVTransfer

        self._transfer = RaidenKVTransfer(device_pool, num_host_pages)
        self._page_size = device_pool.page_size
        self._free: List[int] = list(range(num_host_pages))

    def _alloc_pages(self, n: int) -> Optional[List[int]]:
        if len(self._free) < n:
            return None
        pages, self._free = self._free[:n], self._free[n:]
        return pages

    def _free_pages(self, pages: List[int]) -> None:
        self._free.extend(pages)

    def try_save(self, indices: torch.Tensor) -> Optional[RaidenKVBackupHandle]:
        """Stages the pages holding ``indices``; None means use the row path
        (irregular rows, or the host pool is full right now)."""
        runs = sequential_page_runs(indices, self._page_size)
        if runs is None or runs[1] != 0:
            # Rows starting mid-page could share that page with another
            # request; restoring the whole page would put stale bytes under
            # the other owner. Page-aligned allocations never do this.
            return None
        page_ids, first_offset, num_rows = runs
        host_pages = self._alloc_pages(len(page_ids))
        if host_pages is None:
            logger.warning(
                "raiden retraction pool full (%d pages needed); falling back "
                "to the row path for this request",
                len(page_ids),
            )
            return None
        self._transfer.backup(page_ids, host_pages).synchronize()
        return RaidenKVBackupHandle(self, host_pages, first_offset, num_rows)

    def restore(self, handle: RaidenKVBackupHandle, indices: torch.Tensor) -> None:
        """Loads the handle's pages into the pages holding ``indices``."""
        runs = sequential_page_runs(indices, self._page_size)
        if runs is None:
            raise ValueError(
                "raiden retraction restore needs sequentially paged rows; "
                "got an irregular index set"
            )
        page_ids, first_offset, num_rows = runs
        if num_rows != handle.num_rows or first_offset != handle.first_offset:
            raise ValueError(
                f"raiden retraction restore shape mismatch: saved "
                f"{handle.num_rows} rows at offset {handle.first_offset}, "
                f"asked to restore {num_rows} at {first_offset}"
            )
        self._transfer.load(handle.host_pages, page_ids).synchronize()
        handle.release()


def backup_pool_pages(env_gb: float, page_host_bytes: int) -> int:
    """Host pages the configured budget affords."""
    return int(env_gb * 1024**3) // max(1, page_host_bytes)


def maybe_build_backup(device_pool) -> Optional[RaidenKVBackup]:
    """The pool's retraction backup, when the operator budgeted one."""
    gb = float(os.environ.get("SGLANG_RAIDEN_RETRACT_HOST_GB", "0") or 0)
    if gb <= 0:
        return None
    page_bytes = 0
    for layer_id in range(device_pool.layer_num):
        buf = device_pool.get_fused_kv_buffer(device_pool.start_layer + layer_id)
        page_bytes += buf[0].numel() * buf.element_size()
    pages = backup_pool_pages(gb, page_bytes)
    if pages <= 0:
        raise ValueError(
            f"SGLANG_RAIDEN_RETRACT_HOST_GB={gb} affords no page "
            f"({page_bytes} bytes per page across layers)"
        )
    logger.info(
        "raiden retraction backup: %d host pages (%.2f GB budget, %.1f MB/page)",
        pages,
        gb,
        page_bytes / 2**20,
    )
    return RaidenKVBackup(device_pool, pages)
