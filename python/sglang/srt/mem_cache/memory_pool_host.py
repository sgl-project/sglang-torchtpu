from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.srt.utils import (
    is_cuda,
    is_hip,
    is_mps,
    is_npu,
    is_tpu,
    is_xpu,
    next_power_of_2,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_tpu = is_tpu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if _is_xpu:
    from sgl_kernel import transfer_cache_dsv4_mla
else:
    from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )

if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)


from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    host_memory_budget_bytes,
    synchronized,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
    make_kernel_ptr_table,
)
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin


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


# ---- V4 Compressed KV Host Pools ----


class LogicalHostPool:
    """Pure-logical anchor pool for V4 HiCache.

    The pool manages page-aligned token slots but holds no KV tensor. V4
    compressed side pools use these logical FULL indices as stable page anchors.
    """

    shared_allocation_domain = None
    stores_page_envelope = False
    storage_format_tag = None

    def __init__(self, size: int, page_size: int, layout: str = "layer_first"):
        if size % page_size != 0:
            raise ValueError(
                "LogicalHostPool size must be page-aligned, "
                f"got size={size}, page_size={page_size}"
            )
        self.size = size
        # Stands in for a host pool (and group anchor); DCP never widens it.
        self.logical_size = size
        self.page_size = page_size
        self.device = "cpu"
        self.layout = layout
        self.dtype = torch.uint8
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.kv_buffer = None
        self.size_per_token = 0
        self.allocator = None
        self.can_use_write_back_jit = True
        self.lock = threading.RLock()
        self.clear()

    @synchronized
    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)
        # Match HostKVCache's lazy release path: defer large free-list merges
        # until an allocation needs the released slots.
        self.release_slots = []
        self.num_release_slots = 0

    def destroy(self) -> None:
        """Logical anchors own no backing buffers or registrations to release."""
        return None

    def available_size(self):
        return len(self.free_slots) + self.num_release_slots

    def _merge_release_slots(self):
        if self.num_release_slots == 0:
            return

        if len(self.free_slots) == 0 and len(self.release_slots) == 1:
            self.free_slots = self.release_slots[0]
        else:
            self.free_slots = torch.cat([self.free_slots, *self.release_slots])

        self.release_slots = []
        self.num_release_slots = 0

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={self.page_size}"
            )
        if need_size > self.available_size():
            return None

        if need_size > len(self.free_slots):
            self._merge_release_slots()

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool free must be page-aligned, "
                f"got len(indices)={len(indices)}, page_size={self.page_size}"
            )
        indices_cpu = indices.to(dtype=torch.int64, device="cpu").flatten()
        if indices_cpu.numel() == 0:
            return 0

        self.release_slots.append(indices_cpu)
        self.num_release_slots += len(indices_cpu)
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        pass

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        pass

    def get_data_page(self, index, flat=True):
        return torch.empty(0, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        pass

    def get_page_buffer_meta(self, indices):
        return None

    def get_page_buffer_element_size(self, split_factor: int = 1):
        return None

    def get_ksize_per_token(self):
        return 0


class DeepSeekV4PagedHostPool(HiSparseHostPoolMixin, HostKVCache):
    """Host mirror for a DeepSeek V4 paged KV/indexer sub-pool."""

    def __init__(
        self,
        pool_name: str,
        device_buffers: list[torch.Tensor],
        item_bytes: int,
        num_host_pages: int,
        slot_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
        page_aligned_only: bool = False,
    ):
        self.pool_name = pool_name
        self.layer_num = len(device_buffers)
        self.item_bytes = item_bytes
        # The token-granular copy below addresses the fused DSv4 C4 row. Neither
        # the FP4 indexer, whose page rows group their slots, nor unified_kv,
        # whose page row is a flat run of tokens, has that layout, so those may
        # only move whole pages.
        self.page_aligned_only = page_aligned_only
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = slot_page_size
        self.size = num_host_pages * slot_page_size
        self.layout = layout
        self.size_per_token = item_bytes
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.device_buffers = device_buffers
        self.gpu_device = device_buffers[0].device if device_buffers else device

        requested_bytes = self.layer_num * num_host_pages * self.item_bytes
        available_bytes = host_memory_budget_bytes()
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 paged pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        # ALLOC_MEMORY_FUNCS is keyed by device *type* string ("npu"/"musa"/...),
        # not torch.device objects; a torch.device key silently falls back to
        # cudaHostRegister, which fails on NPU. Resolve the alloc func by type str.
        _alloc_key = (
            self.gpu_device.type
            if isinstance(self.gpu_device, torch.device)
            else str(self.gpu_device)
        )
        alloc_func = ALLOC_MEMORY_FUNCS[_alloc_key]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.item_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=self.layer_num * self.item_bytes,
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=self.layer_num * self.item_bytes,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        logger.info(
            "Allocating %.2f GB host memory for V4 paged pool '%s' "
            "(layers=%d, pages=%d, item_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.item_bytes,
            self.layout,
        )

        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_buffers],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            make_kernel_ptr_table(
                self.data_refs,
                self.gpu_device,
                host_memory_registered=self.pin_memory,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()
        self.clear()

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.item_bytes * self.dtype.itemsize,
        )
        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def _host_page_view(self, l: int) -> torch.Tensor:
        """View a host layer as ``[pages, 1, P, 1, dim]``."""
        if self.layout == "layer_first":
            layer_buffer = self.kv_buffer[l]
        elif self.layout == "page_first":
            layer_buffer = self.kv_buffer[:, l, :]
        elif self.layout == "page_first_direct":
            layer_buffer = self.kv_buffer[:, l, 0, :]
        else:
            raise ValueError(
                f"{self.pool_name} _host_page_view: kernel_ascend requires "
                "layer_first/page_first/page_first_direct layout, "
                f"got {self.layout!r}"
            )
        device_buffer = self.device_buffers[l]
        return layer_buffer.view(device_buffer.dtype).view(
            self.num_host_pages,
            1,
            device_buffer.shape[1],
            1,
            device_buffer.shape[-1],
        )

    def get_contiguous_buf_infos(self):
        """Return per-layer page-row buffers for PD direct-to-host transfer."""
        data_ptrs = [tensor.data_ptr() for tensor in self.data_refs]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.item_bytes * self.dtype.itemsize] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.reshape(-1, self.slot_page_size)[:, 0] // self.slot_page_size

    def _to_native_page_indices(
        self, indices: torch.Tensor, native_page_size: int
    ) -> torch.Tensor:
        """Expand logical page indices into a device buffer's native slots."""
        rows = self._to_page_indices(indices)
        offsets = torch.arange(native_page_size, dtype=rows.dtype, device=rows.device)
        return (rows[:, None] * native_page_size + offsets).reshape(-1)

    def _unaligned_transfer_error(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> ValueError:
        return ValueError(
            f"{self.pool_name} expects page-aligned indices: got "
            f"{host_indices.numel()} host and {device_indices.numel()} device "
            f"indices for page size {self.slot_page_size}."
        )

    def _has_transfer_indices(
        self, host_indices: torch.Tensor | None, device_indices: torch.Tensor | None
    ) -> bool:
        if host_indices is None or device_indices is None:
            return False
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                f"{self.pool_name} transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}"
            )
        return host_indices.numel() > 0

    def get_size_per_token(self):
        return self.item_bytes

    def get_ksize_per_token(self):
        return self.item_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)
        self.release_slots = []
        self.num_release_slots = 0

    def available_size(self):
        return len(self.free_slots) + self.num_release_slots

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        need_size = (
            (need_size + self.slot_page_size - 1) // self.slot_page_size
        ) * self.slot_page_size
        if need_size > self.available_size():
            return None

        if need_size > len(self.free_slots):
            self._merge_release_slots()

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        if indices_cpu.numel() == 0:
            return 0

        self.release_slots.append(indices_cpu)
        self.num_release_slots += len(indices_cpu)
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # Whole C4 pages can use the normal HiCache page-row copy below.
            # Token-granular DSV4 C4 copy needs this helper because a token is
            # not one contiguous byte range in the paged row:
            # [value0..value63][scale0..scale63].
            if self.page_aligned_only:
                raise self._unaligned_transfer_error(host_indices, device_indices)
            transfer_cache_dsv4_mla(
                src_ptrs=self.device_ptrs,
                dst_ptrs=self.data_ptrs,
                src_indices=device_indices.to(dtype=torch.int64),
                dst_indices=host_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.item_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.item_bytes,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.item_bytes,
                    dst_layout_dim=self.layer_num * self.item_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_buffers,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_buffers,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # HiCache keeps KV-derived pools in the FULL logical coordinate
            # space. Ascend consumes the native slots of the shaped device
            # buffer, so translate the already resolved page rows only at this
            # backend boundary.
            native_page_size = int(self.device_buffers[0].shape[1])
            native_device_indices = self._to_native_page_indices(
                device_indices, native_page_size
            )
            native_host_indices = self._to_native_page_indices(
                host_indices, native_page_size
            )
            for l in range(self.layer_num):
                dev_view = self.device_buffers[l].unsqueeze(0)
                # dev_view: [1, dev_pages, native_page_size, 1, kv_dim]
                host_view = self._host_page_view(l)
                # host_view: [num_host_pages, 1, native_page_size, 1, kv_dim]
                transfer_kv_dim_exchange(
                    device_k=dev_view,
                    host_k=host_view,
                    device_v=torch.empty(0, device=dev_view.device),
                    host_v=torch.empty(0, device="cpu"),
                    device_indices=native_device_indices,
                    host_indices=native_host_indices,
                    page_size=native_page_size,
                    direction=TransferDirection.D2H,
                )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # Same DSV4 C4 layout issue as backup: this is token-granular
            # preload, so it cannot use the normal HiCache page-row copy.
            if self.page_aligned_only:
                raise self._unaligned_transfer_error(host_indices, device_indices)
            transfer_cache_dsv4_mla(
                src_ptrs=self.data_ptrs[layer_id : layer_id + 1],
                dst_ptrs=self.device_ptrs[layer_id : layer_id + 1],
                src_indices=host_indices.to(dtype=torch.int64),
                dst_indices=device_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)

        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.item_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.item_bytes,
                src_layout_dim=self.layer_num * self.item_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # NPU whole-page H2D via Ascend dim-exchange op, for layer_id only.
            native_page_size = int(self.device_buffers[layer_id].shape[1])
            native_device_indices = self._to_native_page_indices(
                device_indices, native_page_size
            )
            native_host_indices = self._to_native_page_indices(
                host_indices, native_page_size
            )
            dev_view = self.device_buffers[layer_id].unsqueeze(0)
            # dev_view: [1, dev_pages, native_page_size, 1, kv_dim]
            host_view = self._host_page_view(layer_id)
            # host_view: [num_host_pages, 1, native_page_size, 1, kv_dim]
            transfer_kv_dim_exchange(
                device_k=dev_view,
                host_k=host_view,
                device_v=torch.empty(0, device=dev_view.device),
                host_v=torch.empty(0, device="cpu"),
                device_indices=native_device_indices,
                host_indices=native_host_indices,
                page_size=native_page_size,
                direction=TransferDirection.H2D,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, 1, self.item_bytes)
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.item_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.item_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and page_bytes % page_size_bytes == 0
        )


class DeepSeekV4StateHostPool(HostKVCache):
    """Host pool for V4 CompressStatePool page rows."""

    def __init__(
        self,
        pool_name: str,
        state_pools: list,
        num_host_pages: int,
        swa_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        if any(pool is None for pool in state_pools):
            raise ValueError(f"{pool_name} state_pools must not contain None")

        self.pool_name = pool_name
        self.state_pools = state_pools
        self.layer_num = len(state_pools)
        self.num_host_pages = num_host_pages
        self.swa_page_size = swa_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = swa_page_size
        self.size = num_host_pages * swa_page_size
        self.layout = layout
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.ring_size = 0
        self.state_page_bytes = 0
        self.device_page_views = []
        self.gpu_device = device
        self._init_device_page_views()
        self.size_per_token = self.state_page_bytes

        requested_bytes = self.layer_num * num_host_pages * self.state_page_bytes
        available_bytes = host_memory_budget_bytes()
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 state pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        # ALLOC_MEMORY_FUNCS is keyed by device *type* string ("npu"/"musa"/...),
        # not torch.device objects; resolve the key the same way PagedHostPool does.
        _state_alloc_key = (
            self.gpu_device.type
            if isinstance(self.gpu_device, torch.device)
            else str(self.gpu_device)
        )
        alloc_func = ALLOC_MEMORY_FUNCS[_state_alloc_key]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.state_page_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=(self.layer_num * self.state_page_bytes),
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=(self.layer_num * self.state_page_bytes),
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        logger.info(
            "Allocating %.2f GB host memory for V4 state pool '%s' "
            "(layers=%d, pages=%d, state_page_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.state_page_bytes,
            self.layout,
        )
        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_page_views],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            make_kernel_ptr_table(
                self.data_refs,
                self.gpu_device,
                host_memory_registered=self.pin_memory,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()

    def _init_device_page_views(self) -> None:
        expected_ring_size = None
        expected_state_page_bytes = None
        for pool in self.state_pools:
            state_tensor = pool.kv_score_buffer.kv_score
            if not state_tensor.is_contiguous():
                raise ValueError(f"{self.pool_name} state tensor must be contiguous")
            ring_size = pool.ring_size
            slot_bytes = state_tensor[0].nbytes
            state_page_bytes = ring_size * slot_bytes
            if expected_ring_size is None:
                expected_ring_size = ring_size
                expected_state_page_bytes = state_page_bytes
                self.gpu_device = state_tensor.device
            elif (
                expected_ring_size != ring_size
                or expected_state_page_bytes != state_page_bytes
            ):
                raise ValueError(
                    f"{self.pool_name} state pools must share ring size and slot bytes"
                )

            state_bytes = state_tensor.view(torch.uint8).reshape(
                state_tensor.shape[0], -1
            )
            usable_slots = (state_tensor.shape[0] // ring_size) * ring_size
            self.device_page_views.append(
                state_bytes[:usable_slots].reshape(-1, state_page_bytes)
            )

        self.ring_size = expected_ring_size or 0
        self.state_page_bytes = expected_state_page_bytes or 0

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.state_page_bytes * self.dtype.itemsize,
        )
        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.swa_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be SWA-page-aligned, "
                f"got numel={indices.numel()}, swa_page_size={self.swa_page_size}"
            )
        return indices.reshape(-1, self.swa_page_size)[:, 0] // self.swa_page_size

    def _ring_op_indices(self, rows: torch.Tensor) -> torch.Tensor:
        """Expand each SWA page row into ``ring_size`` operator indices.

        For example, row ``r`` maps to ``r * ring_size + arange(ring_size)``."""
        ar = torch.arange(self.ring_size, device=rows.device, dtype=rows.dtype)
        return (rows.reshape(-1, 1) * self.ring_size + ar).reshape(-1)

    def _state_host_page_view(self, l: int) -> torch.Tensor:
        """View host state layer ``l`` as ``[pages, 1, R, 1, last_dim]``.

        ``last_dim = state_page_bytes // R // state_dtype.itemsize``."""
        state_dtype = self.state_pools[l].kv_score_buffer.kv_score.dtype
        last_dim = self.state_page_bytes // self.ring_size // state_dtype.itemsize
        if self.layout == "layer_first":
            layer_buffer = self.kv_buffer[l]
        elif self.layout == "page_first":
            layer_buffer = self.kv_buffer[:, l, :]
        elif self.layout == "page_first_direct":
            layer_buffer = self.kv_buffer[:, l, 0, :]
        else:
            raise ValueError(
                f"{self.pool_name} _state_host_page_view: kernel_ascend requires "
                "layer_first/page_first/page_first_direct layout, "
                f"got {self.layout!r}"
            )
        return layer_buffer.view(state_dtype).view(
            self.num_host_pages, 1, self.ring_size, 1, last_dim
        )

    def get_size_per_token(self):
        return self.state_page_bytes

    def get_ksize_per_token(self):
        return self.state_page_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        pass

    def available_size(self):
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no free list"
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            assert self.data_ptrs is not None
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.state_page_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.state_page_bytes,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.state_page_bytes,
                    dst_layout_dim=self.layer_num * self.state_page_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_page_views,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_page_views,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # Ascend copies ring_size state slots for each page-aligned SWA row.
            # _ring_op_indices expands device and host rows into operator indices.
            dev_op_indices = self._ring_op_indices(device_rows)
            host_op_indices = self._ring_op_indices(host_rows)
            for l in range(self.layer_num):
                state_dtype = self.state_pools[l].kv_score_buffer.kv_score.dtype
                last_dim = (
                    self.state_page_bytes // self.ring_size // state_dtype.itemsize
                )
                num_dev_pages = self.device_page_views[l].shape[0]
                dev_view = (
                    self.device_page_views[l]
                    .view(state_dtype)
                    .view(num_dev_pages, self.ring_size, 1, last_dim)
                    .unsqueeze(0)
                )
                # dev_view: [1, num_dev_pages, R, 1, last_dim]
                host_view = self._state_host_page_view(l)
                # host_view: [num_host_pages, 1, R, 1, last_dim]
                transfer_kv_dim_exchange(
                    device_k=dev_view,
                    host_k=host_view,
                    device_v=torch.empty(0, device=dev_view.device),
                    host_v=torch.empty(0, device="cpu"),
                    device_indices=dev_op_indices,
                    host_indices=host_op_indices,
                    page_size=self.ring_size,
                    direction=TransferDirection.D2H,
                )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.state_page_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.state_page_bytes,
                src_layout_dim=self.layer_num * self.state_page_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # NPU whole-page H2D via Ascend dim-exchange op, for layer_id only.
            # See backup_from_device_all_layer: indices must be ring-row indices
            # (ring_size entries per SWA page), not raw SWA locs.
            R = self.ring_size
            dev_op_indices = self._ring_op_indices(device_rows)
            host_op_indices = self._ring_op_indices(host_rows)
            state_dtype = self.state_pools[layer_id].kv_score_buffer.kv_score.dtype
            last_dim = self.state_page_bytes // R // state_dtype.itemsize
            num_dev_pages = self.device_page_views[layer_id].shape[0]
            dev_view = (
                self.device_page_views[layer_id]
                .view(state_dtype)
                .view(num_dev_pages, R, 1, last_dim)
                .unsqueeze(0)
            )
            # dev_view: [1, num_dev_pages, R, 1, last_dim]
            host_view = self._state_host_page_view(layer_id)
            # host_view: [num_host_pages, 1, R, 1, last_dim]
            transfer_kv_dim_exchange(
                device_k=dev_view,
                host_k=host_view,
                device_v=torch.empty(0, device=dev_view.device),
                host_v=torch.empty(0, device="cpu"),
                device_indices=dev_op_indices,
                host_indices=host_op_indices,
                page_size=R,
                direction=TransferDirection.H2D,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(
                self.layer_num, self.state_page_bytes
            )
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, self.state_page_bytes
                )
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, 1, self.state_page_bytes
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.state_page_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.state_page_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and page_bytes % page_size_bytes == 0
        )
