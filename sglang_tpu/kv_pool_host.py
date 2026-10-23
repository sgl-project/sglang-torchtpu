"""Host-side KV pool for TPU HiCache, backed by raiden's DMA-pinned pool.

``MHATokenToKVPoolHost`` (the CUDA host pool) allocates split ``[layer, token,
head, dim]`` K/V tensors and moves data with ``sgl_kernel`` kernels — neither
works on TPU. This subclass keeps the HiCache ``HostKVCache`` interface but
delegates storage and transfer to ``RaidenKVTransfer``: raiden owns the host
memory (DMA-pinned) and copies whole pages of the fused device buffer via
direct PJRT DMA.

HiCache addresses the host pool at **token** granularity, but raiden addresses
**pages**. A page is ``page_size`` consecutive tokens. To keep the two
consistent under fragmentation, this pool allocates and frees at page
granularity and only ever hands out whole, page-aligned token runs, so
``host_index // page_size`` is an exact raiden page id.

Scope is L2 (TPU<->host) only; the L3 flat-page accessors raise so a future
storage-backend port has an explicit, loud extension point.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes

if TYPE_CHECKING:
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
    from sglang_tpu.raiden_transfer import RaidenFutureEvent

logger = logging.getLogger(__name__)

# Marketing name surfaced as ``layout`` for logs / compatibility reads.
RAIDEN_LAYOUT = "fused_raiden"


class TpuRaidenHostKVCache(HostKVCache):
    """Page-granular host KV pool whose storage + transfer live in raiden.

    One instance mirrors one fused device pool: a hybrid sliding-window model
    builds two of these (one per ``SWAKVPool`` sub-pool), each owning a raiden
    engine over just its sub-pool's layers and page-id space.
    """

    device_pool: TpuFusedMHATokenToKVPool
    # The hybrid HostPoolGroup reads this directly off every entry; the JIT
    # write-back path is a CUDA kernel feature.
    can_use_write_back_jit = False

    def __init__(
        self,
        device_pool: TpuFusedMHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = False,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        pool_label: str = "kv",
        mtp_draft_device_pools: tuple = (),
    ):
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "MTP draft KV offload is not supported on the TPU raiden path."
            )
        # Raiden owns the DMA-pinned storage, so skip the base allocator setup
        # but retain its sizing and host-memory admission check.
        self.device_pool = device_pool
        self.pool_label = pool_label
        self.page_size = page_size
        self.layout = RAIDEN_LAYOUT
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num

        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align the pool up to a whole number of pages.
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache %s host pool (%d tokens) is smaller than the device "
                "pool (%d tokens); L2 cache effectiveness is reduced. "
                "Consider increasing --hicache-ratio (or --hicache-size) for "
                "higher L2 cache hit rate.",
                pool_label,
                self.size,
                device_pool.size,
            )

        requested_bytes = self.size * self.size_per_token
        available_bytes = host_memory_budget_bytes(requested_bytes)
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating %s hierarchical KV host pool: %d tokens, %.2f GB host memory.",
            pool_label,
            self.size,
            requested_bytes / 1e9,
        )

        from sglang_tpu.raiden_transfer import RaidenKVTransfer

        self.transfer = RaidenKVTransfer(device_pool, self.page_num)

        self.lock = threading.RLock()
        self.clear()

    # ------------------------------------------------------------------ #
    # Sizing
    # ------------------------------------------------------------------ #

    def get_size_per_token(self) -> int:
        """Host bytes per cached token = fused device page bytes / page_size.

        Uses the *real* (padded) fused-page byte size so the host_size GB
        budget matches raiden's actual per-page allocation, not the logical
        ``2 * head_num * head_dim`` lower bound.
        """
        fused = self.device_pool.get_fused_kv_buffer(self.start_layer)
        page_bytes = int(np.prod(fused.shape[1:])) * fused.element_size()
        per_token = page_bytes // self.page_size
        return per_token * self.layer_num

    def init_kv_buffer(self):
        # raiden owns host memory; there is no torch host buffer.
        return None

    # ------------------------------------------------------------------ #
    # Page-granular allocation (token-addressed interface)
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        with self.lock:
            # Free list of host *pages*. Page 0..page_num-1.
            self.free_pages = deque(range(self.page_num))

    def available_size(self) -> int:
        return len(self.free_pages) * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate ``need_size`` token slots as whole pages.

        Returns page-aligned token indices (int64) so ``// page_size`` yields
        exact raiden page ids, or ``None`` if the pool is full.
        """
        assert need_size % self.page_size == 0, (
            "host alloc size must be a multiple of page_size"
        )
        num_pages = need_size // self.page_size
        with self.lock:
            if num_pages > len(self.free_pages):
                return None
            pages = [self.free_pages.popleft() for _ in range(num_pages)]
        base = np.repeat(
            np.array(pages, dtype=np.int64) * self.page_size, self.page_size
        )
        offsets = np.tile(np.arange(self.page_size, dtype=np.int64), num_pages)
        return torch.from_numpy(base + offsets)

    def free(self, indices: torch.Tensor) -> int:
        page_ids = torch.unique(indices.cpu() // self.page_size).tolist()
        with self.lock:
            self.free_pages.extend(page_ids)
        return len(indices)

    # ------------------------------------------------------------------ #
    # Transfer (token indices -> raiden page ids)
    # ------------------------------------------------------------------ #

    def _to_page_ids(self, token_indices: torch.Tensor) -> List[int]:
        """Whole-page token runs -> page ids, one per page, order preserved."""
        # Indices are host-resident (raiden runs with
        # SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1), so
        # this is a pure host slice+divide. Take the first token of each
        # page-sized run; the runs are page-aligned and contiguous within a page
        # by construction (see ``alloc``).
        return (token_indices[:: self.page_size] // self.page_size).tolist()

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> RaidenFutureEvent:
        """D2H for all layers in one raiden call; returns its future-event."""
        device_pages = self._to_page_ids(device_indices)
        host_pages = self._to_page_ids(host_indices)
        return self.transfer.backup(device_pages, host_pages)

    def load_all_layer(
        self, device_pool, host_indices, device_indices
    ) -> RaidenFutureEvent:
        """H2D for all layers in one raiden call; returns its future-event."""
        host_pages = self._to_page_ids(host_indices)
        device_pages = self._to_page_ids(device_indices)
        return self.transfer.load(host_pages, device_pages)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ) -> None:
        # The raiden controller issues a single all-layer ``load_all_layer``;
        # the per-layer path must never be reached on this pool.
        raise NotImplementedError(
            "TpuRaidenHostKVCache uses all-layer loads via load_all_layer(); "
            "load_to_device_per_layer should not be called on the raiden path."
        )

    # ------------------------------------------------------------------ #
    # L3 storage seams (not yet supported on TPU)
    # ------------------------------------------------------------------ #

    def get_data_page(self, index, flat: bool = True):
        raise NotImplementedError("L3 storage not yet supported on TPU raiden path")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("L3 storage not yet supported on TPU raiden path")

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError("L3 storage not yet supported on TPU raiden path")


class _SharedRaidenHostState:
    """Single Raiden host transfer engine and logical-to-physical page table shared by Full and SWA views."""

    def __init__(
        self,
        parent_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
    ):
        from sglang_tpu.raiden_transfer import RaidenKVTransfer

        backing_pool = parent_pool.full_kv_pool
        self.num_swa_groups = parent_pool.num_swa_groups
        fused = backing_pool.get_fused_kv_buffer(backing_pool.start_layer)
        page_bytes = int(np.prod(fused.shape[1:])) * fused.element_size()
        self.size_per_token = (page_bytes // page_size) * backing_pool.layer_num

        if host_size > 0:
            raw_tokens = int(host_size * 1e9 // self.size_per_token)
        else:
            raw_tokens = int(parent_pool.size * host_to_device_ratio)
        usable_pages = max(raw_tokens // page_size, self.num_swa_groups)
        self.size = usable_pages * page_size
        # Page 0 is reserved as the dummy page.
        self.page_num = usable_pages + 1

        requested_bytes = self.page_num * page_size * self.size_per_token
        available_bytes = host_memory_budget_bytes(requested_bytes)
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating unified hierarchical KV host pool: %d tokens (%d pages, "
            "%d SWA groups), %.2f GB host memory.",
            self.size,
            usable_pages,
            self.num_swa_groups,
            requested_bytes / 1e9,
        )

        self.transfer = RaidenKVTransfer(backing_pool, self.page_num)
        self.swa_group_pages = torch.zeros(
            (max(0, self.num_swa_groups - 1), self.page_num),
            dtype=torch.int32,
            device="cpu",
        )
        self.free_pages = torch.arange(1, self.page_num, dtype=torch.int32)
        self.lock = threading.RLock()

    def clear(self) -> None:
        self.free_pages = torch.arange(1, self.page_num, dtype=torch.int32)
        self.swa_group_pages.zero_()


class TpuRaidenUnifiedHostKVCache(TpuRaidenHostKVCache):
    """Logical-to-physical host KV pool sharing one ``RaidenKVTransfer`` between Full and SWA."""

    def __init__(
        self,
        device_pool: TpuFusedMHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = False,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        pool_label: str = "kv",
        mtp_draft_device_pools: tuple = (),
    ):
        del layout, allocator_type
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "MTP draft KV offload is not supported on the TPU raiden path."
            )
        parent_pool = getattr(device_pool, "parent_pool", None)
        if parent_pool is None:
            raise ValueError(
                "TpuRaidenUnifiedHostKVCache requires a _SharedBufferView with parent_pool."
            )
        state = getattr(parent_pool, "_shared_raiden_state", None)
        if state is None:
            state = _SharedRaidenHostState(
                parent_pool=parent_pool,
                host_to_device_ratio=host_to_device_ratio,
                host_size=host_size,
                page_size=page_size,
            )
            parent_pool._shared_raiden_state = state

        self.device_pool = device_pool
        self.pool_label = pool_label
        self.page_size = page_size
        self.layout = RAIDEN_LAYOUT
        self.pin_memory = pin_memory
        self.device = device
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num

        self._state = state
        self.is_swa = getattr(device_pool, "allocation_label", "") == "SWA"
        self._m = state.num_swa_groups if self.is_swa else 1
        self._host_group_pages = state.swa_group_pages if self.is_swa else None
        self.transfer = state.transfer
        self.lock = state.lock
        self.page_num = state.page_num
        self.size = state.size // self._m
        self.size_per_token = state.size_per_token * self._m

    def clear(self) -> None:
        with self.lock:
            self._state.clear()

    def available_size(self) -> int:
        with self.lock:
            free_count = len(self._state.free_pages)
            pages = free_count // self._m
            return pages * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert need_size % self.page_size == 0, (
            "host alloc size must be a multiple of page_size"
        )
        num_pages = need_size // self.page_size
        if num_pages == 0:
            return torch.empty((0,), dtype=torch.int64)
        total_needed = num_pages * self._m
        with self.lock:
            if total_needed > len(self._state.free_pages):
                return None
            primary_pages = self._state.free_pages[:num_pages]
            if self._m > 1:
                companions = self._state.free_pages[num_pages:total_needed]
                self._host_group_pages[:, primary_pages.long()] = companions.view(
                    self._m - 1, num_pages
                )
            self._state.free_pages = self._state.free_pages[total_needed:]
        pages = primary_pages.to(torch.int64)
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        return (pages[:, None] * self.page_size + offsets[None, :]).reshape(-1)

    def free(self, indices: torch.Tensor) -> int:
        if indices.numel() == 0:
            return 0
        page_ids = torch.unique(indices.cpu() // self.page_size).to(torch.int32)
        page_ids = page_ids[page_ids > 0]
        if page_ids.numel() == 0:
            return len(indices)
        with self.lock:
            if self._m > 1:
                companions = self._host_group_pages[:, page_ids.long()].reshape(-1)
                released = torch.cat([page_ids, companions])
            else:
                released = page_ids
            self._state.free_pages = torch.cat([self._state.free_pages, released])
        return len(indices)

    def _expand_pages(
        self, token_indices: torch.Tensor, group_pages: Optional[torch.Tensor]
    ) -> List[int]:
        primary_pages = (token_indices[:: self.page_size] // self.page_size).tolist()
        if self._m <= 1 or group_pages is None:
            return primary_pages
        expanded: List[int] = []
        for p in primary_pages:
            expanded.append(p)
            expanded.extend(group_pages[:, p].tolist())
        return expanded

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> RaidenFutureEvent:
        del io_backend
        dev_group_pages = getattr(device_pool, "group_pages", None)
        device_pages = self._expand_pages(device_indices, dev_group_pages)
        host_pages = self._expand_pages(host_indices, self._host_group_pages)
        return self.transfer.backup(device_pages, host_pages)

    def load_all_layer(
        self, device_pool, host_indices, device_indices
    ) -> RaidenFutureEvent:
        dev_group_pages = getattr(device_pool, "group_pages", None)
        host_pages = self._expand_pages(host_indices, self._host_group_pages)
        device_pages = self._expand_pages(device_indices, dev_group_pages)
        return self.transfer.load(host_pages, device_pages)
