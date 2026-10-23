# SPDX-License-Identifier: Apache-2.0
"""Logical-to-Physical Unified KV cache pool for hybrid Full + SWA models on TPU.

Shares ``G = num_full_layers`` physical 5D Pallas KV buffers across Full-Attention
(1 physical page per logical page) and Sliding-Window Attention
(``M_swa = ceil(num_swa_layers / num_full_layers)`` groups) with zero external
fragmentation. Physical page 0 is reserved as the dummy-write sink for padded tokens.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.kv_free_fence import HeldPages
from sglang_tpu.kv_pool import (
    GB,
    TpuFusedMHATokenToKVPool,
    _cdiv,
    _torch_dtype_packing,
)
from sglang_tpu.unified_kv_heads import merge_swa_kv_heads, split_swa_output

logger = logging.getLogger(__name__)


def num_swa_groups(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of SWA groups ``M_swa = ceil(num_swa_layers / num_full_layers)``."""
    if num_full_layers <= 0 or num_swa_layers <= 0:
        raise ValueError(
            f"Unified KV pool requires positive full ({num_full_layers}) and "
            f"SWA ({num_swa_layers}) layer counts."
        )
    return _cdiv(num_swa_layers, num_full_layers)


def _fused_page_shape(
    page_size: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    store_dtype: torch.dtype,
) -> Tuple[int, int, int, int]:
    """Return per-page 4D shape ``(page_size, kv_heads_packed, kv_packing, padded_head_dim)``."""
    if v_head_dim > head_dim:
        raise NotImplementedError(
            "TpuUnifiedSWAKVPool requires v_head_dim <= head_dim "
            f"(got v_head_dim={v_head_dim} > head_dim={head_dim})."
        )
    align = TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
    padded_head_dim = _cdiv(head_dim, align) * align
    kv_packing = _torch_dtype_packing(store_dtype)
    kv_heads_packed = _cdiv(head_num * 2, kv_packing)
    return (page_size, kv_heads_packed, kv_packing, padded_head_dim)


def resolve_unified_page_shape(
    *,
    page_size: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    store_dtype: torch.dtype,
) -> Tuple[Tuple[int, int, int, int], bool]:
    """Validate Full/SWA page compatibility and return ``(page_shape, swa_kv_heads_merged)``."""
    full_shape = _fused_page_shape(
        page_size, head_num, head_dim, v_head_dim, store_dtype
    )
    swa_shape = _fused_page_shape(
        page_size, swa_head_num, swa_head_dim, swa_v_head_dim, store_dtype
    )
    if full_shape == swa_shape:
        return full_shape, False
    if swa_head_num > 1 and swa_v_head_dim == swa_head_dim:
        merged_dim = swa_head_num * swa_head_dim
        if full_shape == _fused_page_shape(
            page_size, 1, merged_dim, merged_dim, store_dtype
        ):
            return full_shape, True
    raise ValueError(
        "TpuUnifiedSWAKVPool requires Full and SWA layers to share an identical "
        f"per-page fused shape (directly or after merging SWA KV heads), "
        f"but got full={full_shape} (heads={head_num}, dim={head_dim}) vs "
        f"swa={swa_shape} (heads={swa_head_num}, dim={swa_head_dim})."
    )


class _SharedBufferView(TpuFusedMHATokenToKVPool):
    """``TpuFusedMHATokenToKVPool`` that either owns the shared buffers or views them."""

    PACKS_HEADS = False

    def __init__(
        self,
        *,
        shared_buffers: Optional[List[torch.Tensor]] = None,
        layer_groups: Optional[List[int]] = None,
        group_pages: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._shared_buffers = None if shared_buffers is None else list(shared_buffers)
        if self._shared_buffers is not None:
            kwargs["layer_num"] = len(self._shared_buffers)
        self.layer_groups = (
            list(layer_groups)
            if layer_groups is not None
            else [0] * kwargs["layer_num"]
        )
        self.group_pages = group_pages
        self._has_group_offsets = group_pages is not None and any(
            g > 0 for g in self.layer_groups
        )
        super().__init__(**kwargs)

    def _create_buffers(self):
        if self._shared_buffers is None:
            super()._create_buffers()
            return
        self.kv_buffer = self._shared_buffers
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [int(np.prod(x.shape[1:])) * x.dtype.itemsize for x in self.kv_buffer],
            device=self.device,
        )

    def _finalize_allocation_log(self, size: int) -> None:
        if self._shared_buffers is None:
            super()._finalize_allocation_log(size)

    def _raiden_backup(self):
        return None if self._has_group_offsets else super()._raiden_backup()

    def _map_group_token_indices(
        self, group: int, indices_cpu: torch.Tensor
    ) -> torch.Tensor:
        if group == 0 or self.group_pages is None:
            return indices_cpu
        phys_pages = self.group_pages[group - 1, indices_cpu // self.page_size].to(
            torch.long
        )
        return phys_pages * self.page_size + (indices_cpu % self.page_size)

    @contextlib.contextmanager
    def _group_buffers(self, buffers: List[torch.Tensor]):
        orig_buffers, orig_num = self.kv_buffer, self.layer_num
        self.kv_buffer, self.layer_num = buffers, len(buffers)
        try:
            yield
        finally:
            self.kv_buffer, self.layer_num = orig_buffers, orig_num

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        if not self._has_group_offsets:
            return super().get_cpu_copy(indices, mamba_indices, req_pool_index)
        indices_cpu = indices.to(device="cpu", dtype=torch.long)
        out: list = [None] * self.layer_num
        for g in range(max(self.layer_groups) + 1):
            lids = [i for i, grp in enumerate(self.layer_groups) if grp == g]
            with self._group_buffers([self.kv_buffer[i] for i in lids]):
                grp_out = super().get_cpu_copy(
                    self._map_group_token_indices(g, indices_cpu),
                    mamba_indices,
                    req_pool_index,
                )
            for i, lid in enumerate(lids):
                out[lid] = grp_out[i]
        return out

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        if not self._has_group_offsets:
            return super().load_cpu_copy(
                kv_cache_cpu, indices, mamba_indices, req_pool_index
            )
        indices_cpu = indices.to(device="cpu", dtype=torch.long)
        for g in range(max(self.layer_groups) + 1):
            lids = [i for i, grp in enumerate(self.layer_groups) if grp == g]
            with self._group_buffers([self.kv_buffer[i] for i in lids]):
                super().load_cpu_copy(
                    [kv_cache_cpu[i] for i in lids],
                    self._map_group_token_indices(g, indices_cpu),
                    mamba_indices,
                    req_pool_index,
                )


class TpuUnifiedSWAKVPool(SWAKVPool):
    """Unified Hybrid Full/SWA KV pool backed by ``G = num_full_layers`` shared 5D buffers."""

    def __init__(
        self,
        size: int,
        size_swa: Optional[int] = None,
        *,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        device: str,
        full_kv_pool_class: Optional[type] = None,
        swa_kv_pool_class: Optional[type] = None,
        full_kv_pool_kwargs: Optional[dict] = None,
        swa_kv_pool_kwargs: Optional[dict] = None,
        start_layer: int = 0,
        enable_memory_saver: bool = False,
        **kwargs,
    ):
        del full_kv_pool_class, swa_kv_pool_class
        del full_kv_pool_kwargs, swa_kv_pool_kwargs
        self.size = size
        self.size_swa = size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = start_layer
        self.page_size = page_size
        self.layer_transfer_counter = None
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        v_head_dim = kwargs.get("v_head_dim") or head_dim
        swa_head_num = kwargs.get("swa_head_num") or head_num
        swa_head_dim = kwargs.get("swa_head_dim") or head_dim
        swa_v_head_dim = kwargs.get("swa_v_head_dim") or swa_head_dim
        _, self.swa_kv_heads_merged = resolve_unified_page_shape(
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            store_dtype=dtype,
        )
        self.swa_raw_head_num = swa_head_num
        self.swa_raw_head_dim = swa_head_dim
        swa_view_heads = 1 if self.swa_kv_heads_merged else swa_head_num
        swa_view_dim = (
            swa_head_num * swa_head_dim if self.swa_kv_heads_merged else swa_head_dim
        )
        swa_view_v_dim = (
            swa_head_num * swa_v_head_dim
            if self.swa_kv_heads_merged
            else swa_v_head_dim
        )

        G = self.full_layer_nums
        self.num_shared_buffers = G
        self.num_swa_groups = num_swa_groups(G, self.swa_layer_nums)
        num_blocks = size // page_size + 1
        self.swa_group_pages = torch.zeros(
            (max(0, self.num_swa_groups - 1), num_blocks), dtype=torch.int32
        )

        common = dict(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        self.full_kv_pool = _SharedBufferView(
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            layer_num=G,
            allocation_label="Full",
            **common,
        )
        bufs = self.full_kv_pool.kv_buffer
        self.swa_kv_pool = _SharedBufferView(
            shared_buffers=[bufs[j % G] for j in range(self.swa_layer_nums)],
            head_num=swa_view_heads,
            head_dim=swa_view_dim,
            v_head_dim=swa_view_v_dim,
            allocation_label="SWA",
            layer_groups=[j // G for j in range(self.swa_layer_nums)],
            group_pages=self.swa_group_pages if self.num_swa_groups > 1 else None,
            **common,
        )

        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        self._swa_layer_group: Dict[int, int] = {}
        for i, gid in enumerate(full_attention_layer_ids):
            self.layers_mapping[gid] = (i, False)
        for j, gid in enumerate(swa_attention_layer_ids):
            self.layers_mapping[gid] = (j, True)
            self._swa_layer_group[gid] = j // G

        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            "TpuUnifiedSWAKVPool mem usage: %.2f GB, unified size: %d tokens "
            "(shared buffers=%d, swa_groups=%d, swa_heads_merged=%s)",
            self.mem_usage,
            self.size,
            G,
            self.num_swa_groups,
            self.swa_kv_heads_merged,
        )

    def swa_page_group(self, layer_id: int) -> int:
        return self._swa_layer_group.get(layer_id, 0)

    def get_state(self) -> list[torch.Tensor]:
        return self.full_kv_pool.get_state()

    def set_state(self, state: list[torch.Tensor]) -> None:
        G = self.num_shared_buffers
        self.full_kv_pool.set_state(state)
        self.swa_kv_pool.set_state([state[j % G] for j in range(self.swa_layer_nums)])

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return self.full_kv_pool.get_kv_size_bytes()

    def get_state_buf_infos(self):
        return [], [], []


class _SharedPageStore:
    """Shared CPU physical page pool (1 .. num_pages) drawn from by Full and SWA."""

    def __init__(self):
        self.free_pages: Optional[torch.Tensor] = None
        self.page_is_free: Optional[torch.Tensor] = None
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0
        self.held_pages = HeldPages()

    def unheld_free_pages(self) -> int:
        return (
            0 if self.free_pages is None else len(self.free_pages)
        ) + self.num_staged_pages

    def num_free_pages(self) -> int:
        return self.unheld_free_pages() + self.held_pages.num_pages


def _store_attr(name: str):
    return property(
        lambda self: getattr(self._page_store, name),
        lambda self, val: setattr(self._page_store, name, val),
    )


class _SharedPagesAllocator(TpuPagedTokenToKVPoolAllocator):
    """One side (Full or SWA) of the zero-fragmentation Logical-to-Physical unified allocator."""

    free_pages = _store_attr("free_pages")
    staged_pages = _store_attr("staged_pages")
    num_staged_pages = _store_attr("num_staged_pages")
    held_pages = _store_attr("held_pages")
    page_is_free = _store_attr("page_is_free")

    def __init__(
        self,
        *,
        page_store: _SharedPageStore,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
        group_pages: Optional[torch.Tensor] = None,
    ):
        self._page_store = page_store
        self.allocated_pages = 0
        self._group_pages = group_pages
        self._pages_per_id = 1 if group_pages is None else group_pages.shape[0] + 1
        self._free_pages_before_take: Optional[torch.Tensor] = None
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

    def available_size(self) -> int:
        return (
            self._page_store.num_free_pages() // self._pages_per_id
        ) * self.page_size

    def allocated_count(self) -> int:
        return (self.allocated_pages // self._pages_per_id) * self.page_size

    def alloc(self, need_size: int):
        return self._count_taken_pages(super().alloc, need_size)

    def alloc_extend(self, *args, **kwargs):
        return self._count_taken_pages(super().alloc_extend, *args, **kwargs)

    def alloc_decode(self, *args, **kwargs):
        return self._count_taken_pages(super().alloc_decode, *args, **kwargs)

    def _count_taken_pages(self, take, *args, **kwargs):
        free_before = self._page_store.num_free_pages()
        out = take(*args, **kwargs)
        if self._group_pages is not None:
            self._take_group_pages()
        self.allocated_pages += free_before - self._page_store.num_free_pages()
        return out

    def _ensure_free_pages(self, num_pages: int) -> bool:
        if self._group_pages is None:
            return super()._ensure_free_pages(num_pages)
        ok = super()._ensure_free_pages(num_pages * self._pages_per_id)
        self._free_pages_before_take = self.free_pages if ok else None
        return ok

    def _take_group_pages(self) -> None:
        before = self._free_pages_before_take
        self._free_pages_before_take = None
        if before is None:
            return
        num_taken = len(before) - len(self.free_pages)
        if num_taken == 0:
            return
        num_extra = self._group_pages.shape[0] * num_taken
        extra = self.free_pages[:num_extra]
        self._mark_handed_out(extra)
        self._group_pages[:, before[:num_taken].long()] = extra.view(-1, num_taken)
        self.free_pages = self.free_pages[num_extra:]

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        page_ids = tuple(
            ids[ids > 0]
            for ids in (
                p.reshape(-1).to(device="cpu", dtype=torch.int32) for p in page_ids
            )
        )
        if self._group_pages is not None and page_ids:
            ids = torch.cat([p.to(torch.int64) for p in page_ids])
            if ids.numel() > 0:
                page_ids = (*page_ids, self._group_pages[:, ids].reshape(-1))
        total_released = sum(ids.numel() for ids in page_ids)
        if total_released == 0:
            return
        self.allocated_pages -= total_released
        super()._release_page_ids(*page_ids)

    def clear(self) -> None:
        super().clear()
        self.allocated_pages = 0


class TpuUnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Logical-to-Physical Unified KV allocator for Full + SWA attention on TPU."""

    def __init__(
        self,
        size: int,
        size_swa: Optional[int] = None,
        *,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: TpuUnifiedSWAKVPool,
        need_sort: bool,
        req_to_token_pool=None,
    ):
        del size_swa
        if page_size <= 1:
            raise ValueError(
                f"TpuUnifiedSWATokenToKVPoolAllocator requires page_size > 1, got {page_size}."
            )
        assert isinstance(kvcache, TpuUnifiedSWAKVPool)
        self._page_store = _SharedPageStore()
        self._swa_groups = getattr(kvcache, "num_swa_groups", 1)
        self.need_sort = need_sort
        self._kvcache = kvcache
        self._req_to_token_pool = req_to_token_pool
        super().__init__(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
            req_to_token_pool=req_to_token_pool,
        )
        assert not self._swa_req_ring
        self.full_attn_allocator = _SharedPagesAllocator(
            page_store=self._page_store,
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache.full_kv_pool,
            need_sort=need_sort,
        )
        self.swa_attn_allocator = _SharedPagesAllocator(
            page_store=self._page_store,
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache.swa_kv_pool,
            need_sort=need_sort,
            group_pages=kvcache.swa_group_pages if self._swa_groups > 1 else None,
        )
        self.clear()

    def clear(self) -> None:
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.free_group = None
        self.swa_free_group = []
        self.swa_page_ids_group = []

    def _free_page_count(self) -> int:
        return self._page_store.num_free_pages()

    def _page_ids_needed(self, num_full_pages: int, num_swa_pages: int) -> int:
        return num_full_pages + self._swa_groups * num_swa_pages

    def max_extend_tokens(self) -> int:
        return (self._free_page_count() // (1 + self._swa_groups)) * self.page_size

    def available_size(self) -> int:
        return self.max_extend_tokens()

    def full_available_size(self) -> int:
        return self._free_page_count() * self.page_size

    def swa_available_size(self) -> int:
        return (self._free_page_count() // self._swa_groups) * self.page_size

    def conserve_full_available_size(self) -> int:
        return self._size_full - self.full_attn_allocator.allocated_count()

    def conserve_swa_available_size(self) -> int:
        return self._size_swa - self.swa_attn_allocator.allocated_count()

    _conserve_full_available_size = conserve_full_available_size
    _conserve_swa_available_size = conserve_swa_available_size

    @property
    def current_full_capacity(self) -> int:
        return self.full_available_size() + self.full_attn_allocator.allocated_count()

    @property
    def current_swa_capacity(self) -> int:
        return self.swa_available_size() + self.swa_attn_allocator.allocated_count()

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        del full_capacity, swa_capacity
        return (
            (self.current_full_capacity, self.full_available_size()),
            (self.current_swa_capacity, self.swa_available_size()),
        )

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        from sglang.srt.mem_cache.prefill_budget import SharedSWAPrefillBudget

        return SharedSWAPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            min(num_full_pages, num_swa_pages) >= 0
            and self._page_ids_needed(num_full_pages, num_swa_pages)
            <= self._free_page_count()
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        out_indices_cpu: Optional[torch.Tensor] = None,
    ):
        num_new_pages = int((seq_lens_cpu % self.page_size == 1).sum())
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None
        return super().alloc_decode(
            seq_lens, seq_lens_cpu, last_loc, out_indices_cpu=out_indices_cpu
        )

    def reclaim_plan(
        self,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        empty_pool: bool = False,
    ) -> Optional[Tuple[int, int]]:
        if full_tokens < 0 or swa_tokens < 0:
            return None
        ps = self.page_size
        need = self._page_ids_needed(
            (math.ceil(full_tokens) + ps - 1) // ps,
            (math.ceil(swa_tokens) + ps - 1) // ps,
        )
        if empty_pool:
            return (0, 0) if need <= self.size_full // ps else None
        free = self._free_page_count()
        if need <= free:
            return (0, 0)
        short = need - free
        ms = self._swa_groups
        max_f = min(
            self.full_attn_allocator.allocated_pages,
            max(0, int(full_evictable_tokens)) // ps,
        )
        max_s = min(
            self.swa_attn_allocator.allocated_pages // ms,
            max(0, int(swa_evictable_tokens)) // ps,
        )
        if self._page_ids_needed(max_f, max_s) < short:
            return None
        f_reclaim = min(max_f, short)
        rem = short - f_reclaim
        return (f_reclaim * ps, (_cdiv(rem, ms) if rem > 0 else 0) * ps)

    def can_reserve(
        self,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        empty_pool: bool = False,
        require_token_slack: bool = False,
    ) -> bool:
        if require_token_slack and (
            full_tokens >= self.size_full or swa_tokens >= self.size_swa
        ):
            return False
        return (
            self.reclaim_plan(
                full_tokens,
                swa_tokens,
                full_evictable_tokens=full_evictable_tokens,
                swa_evictable_tokens=swa_evictable_tokens,
                empty_pool=empty_pool,
            )
            is not None
        )

    def max_new_tokens_for_memory(
        self,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
    ) -> int | None:
        del token_capacity
        from sglang.srt.mem_cache.prefill_budget import estimate_swa_kv_tokens

        def fits(n: int) -> bool:
            return self.can_reserve(
                input_tokens + n + self.page_size,
                estimate_swa_kv_tokens(
                    input_tokens,
                    n,
                    sliding_window_size=sliding_window_size,
                    page_size=self.page_size,
                    allocation_limit=chunk_size,
                ),
                empty_pool=True,
                require_token_slack=True,
            )

        if not fits(0):
            return None
        if fits(max_new_tokens):
            return max_new_tokens
        lo, hi = 0, max_new_tokens
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
        return lo

    def evict_to_free_tokens(
        self, tree_cache, num_tokens: int, *, swa_num_tokens: Optional[int] = None
    ) -> bool | None:
        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return None
        req_swa = num_tokens if swa_num_tokens is None else swa_num_tokens
        ps = self.page_size
        need_f, need_s = _cdiv(max(0, int(num_tokens)), ps), _cdiv(
            max(0, int(req_swa)), ps
        )
        if hasattr(tree_cache, "evict_for_alloc"):
            short = self._page_ids_needed(need_f, need_s) - self._free_page_count()
            if short > 0:
                tree_cache.evict_for_alloc(
                    EvictParams(
                        num_tokens=short * ps,
                        swa_num_tokens=_cdiv(short, self._swa_groups) * ps,
                    )
                )
        else:
            plan = self.reclaim_plan(
                num_tokens,
                req_swa,
                full_evictable_tokens=tree_cache.full_evictable_size(),
                swa_evictable_tokens=tree_cache.swa_evictable_size(),
            )
            if plan is None:
                return False
            if plan[0] or plan[1]:
                tree_cache.evict(
                    EvictParams(num_tokens=plan[0], swa_num_tokens=plan[1])
                )
        return self.new_pages_available(need_f, need_s)

    def reclaim_for_prealloc(
        self, tree_cache, full_tokens: int, swa_tokens: int
    ) -> str | None:
        ready = self.evict_to_free_tokens(
            tree_cache, full_tokens, swa_num_tokens=swa_tokens
        )
        if ready is None:
            ps = self.page_size
            ready = self.new_pages_available(
                _cdiv(max(0, int(full_tokens)), ps),
                _cdiv(max(0, int(swa_tokens)), ps),
            )
        return (
            None
            if ready
            else f"Unified FULL/SWA reclamation insufficient: needed=({full_tokens}, {swa_tokens})"
        )

    def debug_print(self) -> str:
        return (
            f"#free-pages: {self._free_page_count()}, "
            f"#full-pages: {self.full_attn_allocator.allocated_pages}, "
            f"#swa-pages: {self.swa_attn_allocator.allocated_pages}, "
        )


