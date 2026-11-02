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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hicache_auto_size import register_sizeable_pool
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HostPoolGroup,
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    StackStrategy,
    build_pool_entry,
    register_stack_strategy,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.base import (
    host_memory_budget_bytes,
    sync_fixed_hicache_size,
)
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import next_power_of_2
from sglang_tpu.tpu_dma_hicache import (
    NOOP_EVENT,
    TransferEvent,
    create_tpu_dma_kv_transfer,
    is_tpu_dma_hicache_backend,
    page_major_tensors,
)
from sglang_tpu.unified_kv_pool import TpuUnifiedSWAKVPool

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
        self.storage_format_tag = self._storage_format_tag()

    def _storage_format_tag(self) -> str:
        """Storage-key tag for everything that fixes a page's bytes on this
        pool: the device layout with its packing, the store dtype and the page
        dims. A page written under another tag is a miss, never wrong KV. The
        fp8 scales are per-layer model constants, keyed by the model name."""
        shape_tag = "x".join(str(d) for d in self._device_page_shape())
        dtype_tag = str(self.dtype).removeprefix("torch.")
        return f"tpu-{self.device_pool.kv_layout_tag()}-{dtype_tag}-{shape_tag}"

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

    # -- L3 storage seams ------------------------------------------------ #
    # The host buffer is page-major, so one page's all-layer KV is the
    # contiguous row kv_buffer[page_id] of shape (layer_num, *device_page_shape).
    # Storage backends move that row as one opaque flat blob; the interior keeps
    # the device's packed layout and is never interpreted on the host.

    def _page_id(self, index) -> int:
        # Callers pass the first token slot of a page; drop any sub-page
        # offset, consistent with _to_page_ids.
        return int(index) // self.page_size

    def _page_dims(self) -> tuple:
        return tuple(self.kv_buffer.shape[1:])  # (layer_num, *device_page_shape)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        data_page = self.kv_buffer[self._page_id(index)]
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self._page_dims(), dtype=self.dtype, device=self.device
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.kv_buffer[self._page_id(index)] = data_page.reshape(self._page_dims())


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
        # MLA flat-layout row width; the engine copies the device pages as they
        # are, so the width comes from page_major_tensors.
        override_kv_cache_dim: Optional[int] = None,
    ):
        del override_kv_cache_dim
        # Packed MTP draft pools share the target's page ids, so one engine
        # moves a page of the target and of every draft together.
        self.draft_pools = tuple(mtp_draft_device_pools)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num
        self._init_pages(
            device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            pin_memory=pin_memory,
            device=device,
            pool_label=pool_label,
        )

    def _init_pages(
        self,
        device_pool,
        *,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        pin_memory: bool,
        device: str,
        pool_label: str,
    ) -> None:
        # The engine owns the host storage, so skip the base allocator setup
        # but retain its sizing and host-memory admission check.
        self.device_pool = device_pool
        self.pool_label = pool_label
        self.page_size = page_size
        self.layout = TPU_DMA_LAYOUT
        self.pin_memory = pin_memory
        self.device = device

        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
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
            self.io_backend, device_pool, self.page_num, draft_pools=self.draft_pools
        )

        self.lock = threading.RLock()
        self.clear()

    # ------------------------------------------------------------------ #
    # Sizing
    # ------------------------------------------------------------------ #

    def get_size_per_token(self) -> int:
        """Host bytes per cached token = device page bytes / page_size.

        Uses the *real* (padded) page byte size of every page-major tensor so
        the host_size GB budget matches the engine's actual per-page allocation,
        not the logical ``2 * head_num * head_dim`` lower bound.
        """
        tensors = [
            t
            for pool in (self.device_pool, *self.draft_pools)
            for t in page_major_tensors(pool)
        ]
        page_bytes = sum(int(np.prod(t.shape[1:])) * t.element_size() for t in tensors)
        return page_bytes // self.page_size

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


class TpuDMAMambaPoolHost(TpuDMAHostKVPool):
    """Host store for recurrent-state (GDN/KDA) slots, copied by the DMA engine.

    ``TpuMambaPool`` keeps one ``[state_slots, ...]`` tensor per layer and state
    kind, so a slot is one row of each: the engine moves it as a page of one row.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        *,
        allocator_type: str = "default",
        layout: str = TPU_DMA_LAYOUT,
    ):
        del allocator_type, layout
        self.num_mamba_layers = device_pool.num_mamba_layers
        self.draft_pools = ()
        self._init_pages(
            device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=1,
            pin_memory=False,
            device="cpu",
            pool_label="mamba",
        )


def tpu_dma_mamba_pool_host_cls(io_backend: str) -> type[TpuDMAMambaPoolHost]:
    """The ``TpuDMAMambaPoolHost`` class whose instances copy with ``io_backend``."""
    if io_backend == TpuDMAMambaPoolHost.io_backend:
        return TpuDMAMambaPoolHost
    return type(
        f"TpuDMAMambaPoolHost[{io_backend}]",
        (TpuDMAMambaPoolHost,),
        {"io_backend": io_backend},
    )


@dataclass
class _UnifiedHostSide:
    pages_per_logical: int
    live_pages: set[int] = field(default_factory=set)


class _SharedUnifiedDMAHostState:
    """Shared host page arena and single DMA engine for ``TpuUnifiedSWAKVPool``."""

    def __init__(
        self,
        *,
        io_backend: str,
        unified_pool: TpuUnifiedSWAKVPool,
        host_to_device_ratio: float,
        host_size: float,
        page_size: int,
    ):
        self.io_backend = io_backend
        self.page_size = page_size
        self.num_swa_groups = unified_pool.layout.num_swa_groups
        backing = unified_pool.kv_pool
        self.bytes_per_page = sum(buf[0].nbytes for buf in backing.page_buffers())
        if self.bytes_per_page % page_size != 0:
            raise ValueError(
                f"Unified host page bytes ({self.bytes_per_page}) must be "
                f"divisible by page_size ({page_size})."
            )
        self.full_size_per_token = self.bytes_per_page // page_size

        if host_size > 0:
            tokens = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.full_size_per_token), host_size
            )
        else:
            tokens = int(unified_pool.size * host_to_device_ratio)
        self.page_num = tokens // page_size + 1
        if self.page_num < self.num_swa_groups:
            raise ValueError(
                f"Unified host KV pool ({self.page_num} physical pages) must fit "
                f"at least one SWA page ({self.num_swa_groups} physical pages)."
            )

        full_tokens = self.page_num * page_size
        if full_tokens <= unified_pool.size:
            logger.warning(
                "HiCache unified host pool (%d tokens) is smaller than the device "
                "pool (%d tokens); L2 cache effectiveness is reduced. "
                "Consider increasing --hicache-ratio (or --hicache-size) for "
                "higher L2 cache hit rate.",
                full_tokens,
                unified_pool.size,
            )

        requested_bytes = self.page_num * self.bytes_per_page
        available_bytes = host_memory_budget_bytes(requested_bytes)
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating unified (%s) hierarchical KV host pool: %d physical pages "
            "(%d full-equiv tokens), %.2f GB host memory.",
            io_backend,
            self.page_num,
            full_tokens,
            requested_bytes / 1e9,
        )

        self.transfer = create_tpu_dma_kv_transfer(io_backend, backing, self.page_num)
        self.lock = threading.RLock()
        self.free_pages: deque[int] = deque(range(self.page_num))
        self.swa_group_pages = torch.zeros(
            (max(0, self.num_swa_groups - 1), self.page_num), dtype=torch.int32
        )
        self.sides: dict[str, _UnifiedHostSide] = {}
        self.users = 0
        self.destroyed = False

    def register_side(self, label: str, pages_per_logical: int) -> None:
        with self.lock:
            self.sides[label] = _UnifiedHostSide(pages_per_logical=pages_per_logical)
            self.users += 1

    def release(self) -> None:
        with self.lock:
            self.users = max(0, self.users - 1)
            if self.users > 0 or self.destroyed:
                return
            self.destroyed = True
            destroy_fn = getattr(self.transfer, "destroy", None)
        if destroy_fn is not None:
            destroy_fn()

    def acquire_layout(self) -> None:
        return None

    def release_layout(self, finish_event=None, transfer_key=None) -> None:
        return None

    def free_bytes(self) -> int:
        with self.lock:
            return len(self.free_pages) * self.bytes_per_page

    def available_tokens(self, label: str) -> int:
        with self.lock:
            pages = len(self.free_pages) // self.sides[label].pages_per_logical
            return pages * self.page_size

    def live_tokens(self, label: str) -> int:
        with self.lock:
            return len(self.sides[label].live_pages) * self.page_size

    def _required_physical_pages(
        self, requests: Sequence[tuple[str, int]]
    ) -> tuple[list[int], int]:
        logical_counts: list[int] = []
        total_phys = 0
        for label, need_size in requests:
            assert need_size >= 0 and need_size % self.page_size == 0, (
                f"host alloc size must be a non-negative multiple of page_size, got {need_size}"
            )
            num_logical = need_size // self.page_size
            logical_counts.append(num_logical)
            total_phys += num_logical * self.sides[label].pages_per_logical
        return logical_counts, total_phys

    def can_fit_many_then(
        self,
        requests: Sequence[tuple[str, int]],
        following_requests: Sequence[tuple[str, int]],
        *,
        empty: bool = False,
    ) -> bool:
        with self.lock:
            _, first_phys = self._required_physical_pages(requests)
            _, next_phys = self._required_physical_pages(following_requests)
            avail = self.page_num if empty else len(self.free_pages)
            return first_phys + next_phys <= avail

    def alloc_many(
        self, requests: Sequence[tuple[str, int]]
    ) -> Optional[list[torch.Tensor]]:
        with self.lock:
            logical_counts, total_phys = self._required_physical_pages(requests)
            if total_phys > len(self.free_pages):
                return None
            offsets = torch.arange(self.page_size, dtype=torch.int64)
            out: list[torch.Tensor] = []
            for (label, _), num_logical in zip(requests, logical_counts, strict=True):
                if num_logical == 0:
                    out.append(torch.empty(0, dtype=torch.int64))
                    continue
                side = self.sides[label]
                primaries = [self.free_pages.popleft() for _ in range(num_logical)]
                if side.pages_per_logical > 1:
                    extra = side.pages_per_logical - 1
                    companions = [
                        self.free_pages.popleft() for _ in range(num_logical * extra)
                    ]
                    comp_tensor = (
                        torch.tensor(companions, dtype=torch.int32)
                        .reshape(num_logical, extra)
                        .T
                    )
                    self.swa_group_pages[:, primaries] = comp_tensor
                side.live_pages.update(primaries)
                page_ids = torch.tensor(primaries, dtype=torch.int64)
                out.append((page_ids[:, None] * self.page_size + offsets).reshape(-1))
            return out

    def alloc(self, label: str, need_size: int) -> Optional[torch.Tensor]:
        res = self.alloc_many(((label, need_size),))
        return None if res is None else res[0]

    def free(self, label: str, indices: torch.Tensor) -> int:
        if indices.numel() == 0:
            return 0
        page_ids = torch.unique(indices.cpu() // self.page_size).tolist()
        with self.lock:
            side = self.sides[label]
            for pid in page_ids:
                side.live_pages.remove(pid)
                self.free_pages.append(pid)
                if side.pages_per_logical > 1:
                    self.free_pages.extend(self.swa_group_pages[:, pid].tolist())
                    self.swa_group_pages[:, pid] = 0
        return len(indices)

    def clear(self, label: str) -> None:
        with self.lock:
            side = self.sides[label]
            for pid in sorted(side.live_pages):
                self.free_pages.append(pid)
                if side.pages_per_logical > 1:
                    self.free_pages.extend(self.swa_group_pages[:, pid].tolist())
                    self.swa_group_pages[:, pid] = 0
            side.live_pages.clear()
            if all(not s.live_pages for s in self.sides.values()):
                self.free_pages = deque(range(self.page_num))


class TpuDMAUnifiedHostKVPool(TpuDMAHostKVPool):
    """Full- or SWA-side host KV endpoint sharing one TPU DMA host page arena."""

    def __init__(
        self,
        unified_pool: TpuUnifiedSWAKVPool,
        state: _SharedUnifiedDMAHostState,
        *,
        is_swa: bool,
        pool_label: str,
    ):
        self.io_backend = state.io_backend
        self.device_pool = unified_pool
        self.draft_pools = ()
        self._unified_pool = unified_pool
        self._is_swa = is_swa
        self._pages_per_logical = unified_pool.layout.num_swa_groups if is_swa else 1
        self.pool_label = pool_label
        self.page_size = state.page_size
        self.layout = TPU_DMA_LAYOUT
        self.pin_memory = False
        self.device = "cpu"
        self.dtype = unified_pool.kv_pool.store_dtype
        self.start_layer = 0
        self.layer_num = (
            unified_pool.swa.num_layers if is_swa else unified_pool.full.num_layers
        )
        self.end_layer = self.layer_num

        state.register_side(pool_label, self._pages_per_logical)
        self._state = state
        self.transfer = state.transfer
        self.lock = state.lock
        self.size_per_token = state.full_size_per_token * self._pages_per_logical
        self.page_num = state.page_num // self._pages_per_logical
        self.size = self.page_num * self.page_size
        self._released = False

    def get_size_per_token(self) -> int:
        return self.size_per_token

    @classmethod
    def build_pair(
        cls,
        unified_pool: TpuUnifiedSWAKVPool,
        *,
        io_backend: str,
        host_to_device_ratio: float,
        host_size: float,
        page_size: int,
    ) -> tuple[TpuDMAUnifiedHostKVPool, TpuDMAUnifiedHostKVPool]:
        state = _SharedUnifiedDMAHostState(
            io_backend=io_backend,
            unified_pool=unified_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
        )
        return (
            cls(unified_pool, state, is_swa=False, pool_label="full"),
            cls(unified_pool, state, is_swa=True, pool_label="swa"),
        )

    @property
    def shared_allocation_domain(self) -> _SharedUnifiedDMAHostState:
        return self._state

    def live_tokens(self) -> int:
        return self._state.live_tokens(self.pool_label)

    def available_size(self) -> int:
        return self._state.available_tokens(self.pool_label)

    def clear(self) -> None:
        self._state.clear(self.pool_label)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self._state.alloc(self.pool_label, need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self._state.free(self.pool_label, indices)

    def destroy(self) -> None:
        if not self._released:
            self._released = True
            self._state.release()
        super().destroy()

    def _expand_pages(
        self, token_indices: torch.Tensor, group_pages: Optional[torch.Tensor]
    ) -> List[int]:
        primaries = (token_indices[:: self.page_size].cpu() // self.page_size).to(
            torch.long
        )
        if not self._is_swa or group_pages is None or group_pages.shape[0] == 0:
            return primaries.tolist()
        companions = group_pages[:, primaries].to(torch.long)
        expanded = torch.cat([primaries.unsqueeze(0), companions], dim=0)
        return expanded.T.reshape(-1).tolist()

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> TransferEvent:
        if host_indices.numel() == 0 or device_indices.numel() == 0:
            return NOOP_EVENT
        device_pages = self._expand_pages(
            device_indices, self._unified_pool.swa_group_pages
        )
        host_pages = self._expand_pages(host_indices, self._state.swa_group_pages)
        return self.transfer.backup(device_pages, host_pages)

    def load_all_layer(
        self, device_pool, host_indices, device_indices
    ) -> TransferEvent:
        if host_indices.numel() == 0 or device_indices.numel() == 0:
            return NOOP_EVENT
        host_pages = self._expand_pages(host_indices, self._state.swa_group_pages)
        device_pages = self._expand_pages(
            device_indices, self._unified_pool.swa_group_pages
        )
        return self.transfer.load(host_pages, device_pages)


def _unified_swa_layer_mappings(
    kvcache: TpuUnifiedSWAKVPool,
) -> tuple[dict[int, int], dict[int, int]]:
    full = {
        gid - kvcache.start_layer: lid
        for gid, (lid, is_swa) in kvcache.layers_mapping.items()
        if not is_swa
    }
    swa = {
        gid - kvcache.start_layer: lid
        for gid, (lid, is_swa) in kvcache.layers_mapping.items()
        if is_swa
    }
    return full, swa


def _evict_swa_for_device_alloc(cache, required_size: int) -> None:
    available_size = cache.token_to_kv_pool_allocator.swa_available_size()
    shortfall = max(0, required_size - available_size)
    if shortfall > 0:
        cache.evict_for_alloc(EvictParams(swa_num_tokens=shortfall))


class _TpuUnifiedSwaStrategy(StackStrategy):
    def matches(self, kvcache, components) -> bool:
        return isinstance(kvcache, TpuUnifiedSWAKVPool) and components == {
            ComponentType.FULL,
            ComponentType.SWA,
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        if params.mtp_draft_device_pools:
            raise NotImplementedError(
                "MTP draft KV offload is not supported on TpuUnifiedSWAKVPool."
            )
        if storage_backend is not None:
            raise ValueError(
                "TpuUnifiedSWAKVPool does not support L3 storage "
                f"(--hicache-storage-backend={storage_backend!r})."
            )
        memory = get_memory()
        if not is_tpu_dma_hicache_backend(memory.hicache_io_backend):
            raise ValueError(
                "TpuUnifiedSWAKVPool with --enable-hierarchical-cache requires "
                "a TPU DMA --hicache-io-backend (e.g. 'raiden'), got "
                f"{memory.hicache_io_backend!r}."
            )
        full_layer_mapping, swa_layer_mapping = _unified_swa_layer_mappings(kvcache)
        transfer_layer_id_max = (
            max(full_layer_mapping.keys() | swa_layer_mapping.keys()) + 1
        )
        kv_host_pool, swa_host_pool = TpuDMAUnifiedHostKVPool.build_pair(
            kvcache,
            io_backend=memory.hicache_io_backend,
            host_to_device_ratio=memory.hicache_ratio,
            host_size=memory.hicache_size,
            page_size=params.page_size,
        )
        swa_alloc = params.token_to_kv_pool_allocator.swa_attn_allocator
        host_pool_group = HostPoolGroup(
            [
                build_pool_entry(
                    name=PoolName.KV,
                    host_pool=kv_host_pool,
                    device_pool=kvcache,
                    layer_mapping=full_layer_mapping,
                    transfer_layer_id_max=transfer_layer_id_max,
                    is_anchor=True,
                    host_evict_fn=lambda n: cache.evict_host(n, ComponentType.FULL),
                ),
                build_pool_entry(
                    name=PoolName.SWA,
                    host_pool=swa_host_pool,
                    device_pool=kvcache,
                    layer_mapping=swa_layer_mapping,
                    transfer_layer_id_max=transfer_layer_id_max,
                    host_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
                    device_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
                    device_alloc_fn=swa_alloc.alloc,
                    device_free_fn=swa_alloc.free,
                ),
            ]
        )
        cache_controller = HybridCacheController.create_for_backend(
            params.token_to_kv_pool_allocator,
            host_pool_group,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            pp_group=params.pp_cache_group,
            write_policy=memory.hicache_write_policy,
            io_backend=memory.hicache_io_backend,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            transfer_layer_id_max=transfer_layer_id_max,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=memory.hicache_host_memory_mode,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: kv_host_pool,
                ComponentType.SWA: swa_host_pool,
            },
            pools_desc="Full + SWA (TPU Unified)",
        )


_TPU_UNIFIED_SWA_STRATEGY = _TpuUnifiedSwaStrategy()


def register_unified_hicache_strategy() -> None:
    """Register the TPU unified Full+SWA HiCache stack strategy and sizing entry."""
    register_sizeable_pool(TpuUnifiedSWAKVPool)
    register_stack_strategy(_TPU_UNIFIED_SWA_STRATEGY)
