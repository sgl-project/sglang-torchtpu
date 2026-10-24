"""Host-side KV pools for TPU HiCache.

``TpuFusedMHATokenToKVPoolHost`` backs the torch_tpu io backend with plain torch
copies; ``TpuDMAHostKVPool`` backs the TPU DMA io backends with the backend's
host memory. The rest of this docstring describes the TPU DMA pool.

``MHATokenToKVPoolHost`` (the CUDA host pool) allocates split ``[layer, token,
head, dim]`` K/V tensors and moves data with ``sgl_kernel`` kernels — neither
works on TPU. This subclass keeps the HiCache ``HostKVCache`` interface but
delegates storage and transfer to a ``TpuDMAKVTransfer`` engine (raiden unless
``--hicache-io-backend`` names another): the engine owns the host memory and
copies whole pages of the fused device buffer.

HiCache addresses the host pool at **token** granularity, but engines address
**pages**. A page is ``page_size`` consecutive tokens. To keep the two
consistent under fragmentation, this pool allocates and frees at page
granularity and only ever hands out whole, page-aligned token runs, so
``host_index // page_size`` is an exact engine page id.

Scope is L2 (TPU<->host) only; the L3 flat-page accessors raise so a future
storage-backend port has an explicit, loud extension point.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.utils import next_power_of_2
from sglang_tpu.tpu_dma_hicache import TransferEvent, create_tpu_dma_kv_transfer

if TYPE_CHECKING:
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

logger = logging.getLogger(__name__)

# Marketing name surfaced as ``layout`` for logs / compatibility reads.
TPU_DMA_LAYOUT = "fused_tpu_dma"


class TpuFusedMHATokenToKVPoolHost(HostKVCache):
    """Host (L2) pool for a TPU fused MHA device pool.

    Stores the device's fused page VERBATIM, page-major
    (``page_first_fused``)::

        kv_buffer[page_num, layer_num, *device_page_shape]

    so a page's all-layer KV round-trips as a contiguous per-page copy with no
    de-interleave / un-pad.

    ``device_page_shape`` is read off the live device tensor rather than
    rebuilt, because it differs per KV layout -- token-major is
    ``(page_size, kv_heads_packed, kv_packing, padded_head_dim)`` while
    seq-along-lane is ``(2*kv_heads, head_dim_lanes // kv_packing, kv_packing,
    page_size)``. The two have the same byte count, so a hardcoded shape sizes
    the pool correctly and then fails at the first scatter.

    INVARIANT this relies on: the host side addresses **whole pages only**.
    Every ``kv_buffer[...]`` here indexes dim 0, and ``_to_page_ids`` discards
    any sub-page offset, so the page interior is never interpreted -- which is
    what lets an opaque blob of either layout round-trip. Token-granular host
    access would break that: under seq-along-lane the token axis is innermost,
    so one token's KV is strided across the whole page rather than contiguous.

    DATA PLANE: plain torch indexing -- no Pallas kernels and no pre-allocated
    staging buffer. Reload (``load_to_device_all``) runs on the
    controller's producer thread: one bulk CPU gather of all layers, one H2D,
    then a native in-place ``index_put`` scatter into each live per-layer device
    buffer (the device pool is a per-layer list, so the write is the only
    per-layer step). One barrier materialises every scatter before the round is
    signalled (no-stale); CompiledForwardShim gates the whole forward on it. The
    H2D overlaps the prior batch's forward (needs TPU_PREMAPPED_BUFFER_SIZE).
    Backup mirrors it: per-layer gather stacked on device, then one D2H + one
    host scatter. pow2 padding keeps the scatter/gather HLOs shape-bucketed.
    """

    device_pool: TpuFusedMHATokenToKVPool

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = False,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        pool_label: str = "kv",
    ):
        assert layout == "page_first_fused", layout
        assert not pin_memory
        # Upstream HostKVCache packs MTP-draft KV layers behind the target's
        # (layer_num = target + draft); TPU has no packed draft pools, so the
        # target layer count is the whole pool.
        self.target_layer_num = device_pool.layer_num
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            pool_label=pool_label,
        )

    def _device_page_shape(self) -> tuple:
        """Per-page dims of the device's fused KV tensor, layout and all."""
        dp = self.device_pool
        fused = dp.get_fused_kv_buffer(dp.start_layer)
        return tuple(int(d) for d in fused.shape[1:])

    def get_size_per_token(self):
        dp = self.device_pool
        self.layer_num = dp.layer_num
        # Real (padded) page bytes off the live tensor, so the budget matches
        # what the device actually allocated under either layout.
        fused = dp.get_fused_kv_buffer(dp.start_layer)
        page_bytes = math.prod(self._device_page_shape()) * fused.element_size()
        return page_bytes // self.page_size * self.layer_num

    def init_kv_buffer(self):
        # layout is asserted page_first_fused in __init__, so no re-check here.
        dims = (self.page_num, self.layer_num) + self._device_page_shape()
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        return alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )

    def _to_page_ids(self, token_indices: torch.Tensor) -> torch.Tensor:
        """token-granular, page-aligned indices -> page-granular ids."""
        return token_indices[:: self.page_size] // self.page_size

    def _padded_ids(self, device_indices: torch.Tensor):
        # Page-id prep with NO device->host round-trip. device_indices is
        # host-born allocator output that was pushed to the device; pulling it
        # back with .cpu() forced a hard sync (single in-order PJRT queue must
        # flush behind the running forward)
        pages = device_indices[:: self.page_size] // self.page_size
        num_pages = pages.shape[0]
        if num_pages == 0:
            return None, 0, 0
        num_padded_pages = next_power_of_2(num_pages)
        if num_padded_pages > num_pages:
            pages = torch.cat(
                [
                    pages,
                    pages[num_pages - 1 : num_pages].expand(
                        num_padded_pages - num_pages
                    ),
                ],
                dim=0,
            )
        return pages, num_pages, num_padded_pages

    def load_to_device_all(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
    ):
        """Reload evicted KV host->device in ONE bulk pass, on the HiCache
        producer thread.

        One CPU gather of all layers, one H2D, then a native in-place
        ``index_put`` scatter into each live per-layer device buffer (the device
        pool is a per-layer list, so the write is the only per-layer step -- no
        Pallas kernel, no staging). One barrier materialises every scatter
        before the round is signalled (no-stale); CompiledForwardShim gates the
        whole forward on it. Runs while the prior batch's forward executes, so
        the H2D overlaps that compute (needs TPU_PREMAPPED_BUFFER_SIZE set).
        pow2 padding keeps the scatter HLO shape-bucketed (no per-page-count
        recompile).
        """
        if io_backend != "torch_tpu":
            raise ValueError(
                "TpuFusedMHATokenToKVPoolHost.load_to_device_all only "
                f"supports io_backend='torch_tpu', got {io_backend!r}"
            )

        staged = self.load_stage(device_pool, host_indices, device_indices)
        self.load_commit(device_pool, staged)

    def load_stage(self, device_pool, host_indices, device_indices):
        """
        Returns ``(src_all, dst_ids, host_all)`` or None if nothing to load.
        """
        dst_ids, num_pages, num_padded_pages = self._padded_ids(device_indices)
        if num_pages == 0:
            return None

        # host_indices is host-born (move_indices keeps it on CPU); index the
        # CPU host buffer directly. ``.cpu()`` is a no-op here (no sync).
        host_pages = self._to_page_ids(host_indices).cpu()

        # One bulk gather of ALL layers from the host pool + one H2D. The padded
        # tail repeats the last page so the scatter rewrites it idempotently.
        host_all = self.kv_buffer[host_pages]  # [num_pages, layer_num, page, ...]
        if num_padded_pages > num_pages:
            pad = host_all[-1:].expand(
                (num_padded_pages - num_pages,) + tuple(host_all.shape[1:])
            )
            host_all = torch.cat([host_all, pad], dim=0)  # [num_padded_pages, ...]
        # Single async H2D (DMA engine, overlaps the prior forward).
        src_all = host_all.to(device_pool.kv_buffer[0].device, non_blocking=True)

        # host_all rides along to stay alive until the commit (the DMA may
        # still be reading it).
        return src_all, dst_ids, host_all

    def load_commit(self, device_pool, staged):
        """Scheduler-thread half of the reload: scatter the staged tensor into
        the live per-layer device buffers. MUST run on the scheduler thread
        with no forward in flight (BatchDoneCounter drains commits at its gate
        points): the single in-order PJRT queue then orders the scatter before
        the next forward's reads, and the scatter's dependency on ``src_all``
        orders it after the staging DMA -- no host sync anywhere."""
        if staged is None:
            return
        src_all, dst_ids, _host_all = staged
        for layer_id in range(self.layer_num):
            device_pool.kv_buffer[layer_id][dst_ids] = src_all[:, layer_id]

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        # The torch_tpu path uses load_to_device_all (threaded, all
        # layers at once). This per-layer entry point is kept to satisfy the
        # abstract base class; it is not called on this backend.
        raise NotImplementedError(
            "TpuFusedMHATokenToKVPoolHost uses load_to_device_all; "
            "load_to_device_per_layer is not supported on the torch_tpu backend."
        )

    def backup_gather(self, device_pool, device_indices):
        """Gather the evicted pages off-device into ONE
        ``[num_padded_pages, layer_num, ...]`` donation-safe HBM tensor and
        return it for backup_copy_d2h(), or None if nothing to back up.

        Runs INLINE on the scheduler thread (start_writing), so it must not
        block: the donation-safety is provided by PJRT queue ordering, not a
        host sync (see below).
        """
        src_ids, num_pages, num_padded_pages = self._padded_ids(device_indices)
        if num_pages == 0:
            return None
        # Native gather per layer, stacked into one tensor matching the host
        # layout. The stack is the donation-safe HBM copy -- a fresh HBM buffer
        # (advanced-index + stack both copy), decoupled from kv_buffer, so a
        # later forward overwriting those KV slots can't touch it.
        gathered = torch.stack(
            [
                device_pool.kv_buffer[layer_id][src_ids]
                for layer_id in range(self.layer_num)
            ],
            dim=1,
        )
        return gathered, num_pages, num_padded_pages

    def backup_copy_d2h(self, staged, host_indices):
        """Copy the staged (donation-safe) HBM tensor host-ward in ONE D2H and
        scatter it into the host pool. This D2H overlaps the running forward.
        """
        gathered, num_pages, num_padded_pages = staged  # [num_padded_pages, ...]
        host_pages = self._to_page_ids(host_indices).cpu()
        host_all = gathered[:num_pages].to("cpu")  # single D2H, [num_pages, ...]
        self.kv_buffer[host_pages] = host_all

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        # Synchronous gather+drain (tests / non-overlapped path); the controller
        # calls the two phases directly to run the D2H async.
        if io_backend != "torch_tpu":
            raise ValueError(
                "TpuFusedMHATokenToKVPoolHost supports io_backend='torch_tpu', "
                f"got {io_backend!r}"
            )
        staged = self.backup_gather(device_pool, device_indices)
        if staged is None:
            return
        self.backup_copy_d2h(staged, host_indices)

    # -- L3 storage hooks (not supported in the 2-tier TPU phase) -------- #

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise NotImplementedError(
            "L3 storage is not supported by the TPU fused host pool yet."
        )

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise NotImplementedError(
            "L3 storage is not supported by the TPU fused host pool yet."
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError(
            "L3 storage is not supported by the TPU fused host pool yet."
        )


class TpuDMAHostKVPool(HostKVCache):
    """Page-granular host KV pool whose storage + transfer live in an engine.

    One instance mirrors one fused device pool: a hybrid sliding-window model
    builds two of these (one per ``SWAKVPool`` sub-pool), each owning an
    engine over just its sub-pool's layers and page-id space.
    """

    # The --hicache-io-backend engine; tpu_dma_host_kv_pool_cls binds another.
    io_backend = "raiden"

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
                "MTP draft KV offload is not supported on --hicache-io-backend "
                f"{self.io_backend}."
            )
        # The engine owns the host storage, so skip the base allocator setup
        # but retain its sizing and host-memory admission check.
        self.device_pool = device_pool
        self.pool_label = pool_label
        self.page_size = page_size
        self.layout = TPU_DMA_LAYOUT
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

        self.transfer = create_tpu_dma_kv_transfer(
            self.io_backend, device_pool, self.page_num
        )

        self.lock = threading.RLock()
        self.clear()

    # ------------------------------------------------------------------ #
    # Sizing
    # ------------------------------------------------------------------ #

    def get_size_per_token(self) -> int:
        """Host bytes per cached token = fused device page bytes / page_size.

        Uses the *real* (padded) fused-page byte size so the host_size GB
        budget matches the engine's actual per-page allocation, not the logical
        ``2 * head_num * head_dim`` lower bound.
        """
        fused = self.device_pool.get_fused_kv_buffer(self.start_layer)
        page_bytes = int(np.prod(fused.shape[1:])) * fused.element_size()
        per_token = page_bytes // self.page_size
        return per_token * self.layer_num

    def init_kv_buffer(self):
        # The engine owns host memory; there is no torch host buffer.
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
        exact engine page ids, or ``None`` if the pool is full.
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
    # Transfer (token indices -> engine page ids)
    # ------------------------------------------------------------------ #

    def _to_page_ids(self, token_indices: torch.Tensor) -> List[int]:
        """Whole-page token runs -> page ids, one per page, order preserved."""
        # Indices are host-resident (the transfer path runs with
        # SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1), so
        # this is a pure host slice+divide. Take the first token of each
        # page-sized run; the runs are page-aligned and contiguous within a page
        # by construction (see ``alloc``).
        return (token_indices[:: self.page_size] // self.page_size).tolist()

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> TransferEvent:
        """D2H for all layers in one engine call; returns its future-event."""
        device_pages = self._to_page_ids(device_indices)
        host_pages = self._to_page_ids(host_indices)
        return self.transfer.backup(device_pages, host_pages)

    def load_all_layer(
        self, device_pool, host_indices, device_indices
    ) -> TransferEvent:
        """H2D for all layers in one engine call; returns its future-event."""
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
        # The TPU controllers issue a single all-layer ``load_all_layer``;
        # the per-layer path must never be reached on this pool.
        raise NotImplementedError(
            "TpuDMAHostKVPool uses all-layer loads via load_all_layer(); "
            "load_to_device_per_layer should not be called on this path."
        )

    # ------------------------------------------------------------------ #
    # L3 storage seams (not yet supported on TPU)
    # ------------------------------------------------------------------ #

    def get_data_page(self, index, flat: bool = True):
        raise NotImplementedError(
            f"L3 storage is not supported on --hicache-io-backend {self.io_backend}"
        )

    def get_dummy_flat_data_page(self):
        raise NotImplementedError(
            f"L3 storage is not supported on --hicache-io-backend {self.io_backend}"
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError(
            f"L3 storage is not supported on --hicache-io-backend {self.io_backend}"
        )


@functools.cache
def tpu_dma_host_kv_pool_cls(io_backend: str) -> type[TpuDMAHostKVPool]:
    """The ``TpuDMAHostKVPool`` class whose instances copy with ``io_backend``."""
    if io_backend == TpuDMAHostKVPool.io_backend:
        return TpuDMAHostKVPool
    return type(
        f"TpuDMAHostKVPool[{io_backend}]",
        (TpuDMAHostKVPool,),
        {"io_backend": io_backend},
    )
