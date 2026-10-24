# SPDX-License-Identifier: Apache-2.0
"""Logical-to-Physical Unified KV cache pool for hybrid Full + SWA models on TPU.

Shares ``G = num_full_layers`` physical 5D Pallas KV buffers across Full-Attention
(1 physical page per logical page) and Sliding-Window Attention
(``M_swa = ceil(num_swa_layers / num_full_layers)`` groups) with zero external
fragmentation. Physical page 0 is reserved as the dummy-write sink for padded tokens.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.model_executor.pool_configurator import (
    HybridSWAPoolConfigurator,
    MemoryPoolConfig,
    MemoryPoolConfigurator,
    SWAChunkCapPoolConfigurator,
)
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
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


def _ctx_attr(getter, name: str, default=None):
    try:
        return getattr(getter(), name, default)
    except Exception:
        return default


def _unsupported_reason(kvc) -> Optional[str]:
    """Return a human-readable reason why ``kvc`` cannot use the TPU unified KV pool,
    or ``None`` when supported."""
    spec_algo = getattr(kvc, "spec_algorithm", None)
    disagg_mode = _ctx_attr(get_disagg, "disaggregation_mode", "null")
    hicache_backend = _ctx_attr(get_memory, "hicache_io_backend", None)
    pool_cls = getattr(kvc, "token_to_kv_pool_class", None)
    checks = (
        (not getattr(kvc, "is_hybrid_swa", False), "requires a hybrid SWA model."),
        (getattr(kvc, "use_mla_backend", False), "does not support MLA models."),
        (
            getattr(kvc, "is_hybrid_gdn", False)
            or getattr(kvc, "has_mamba_state", False),
            "does not support hybrid linear-attention or Mamba models.",
        ),
        (
            getattr(kvc, "is_draft_worker", False)
            or (spec_algo is not None and not spec_algo.is_none()),
            "does not support speculative decoding.",
        ),
        (
            _ctx_attr(get_parallel, "pp_size", 1) > 1 or getattr(kvc, "pp_size", 1) > 1,
            "does not support pipeline parallelism.",
        ),
        (
            _ctx_attr(get_parallel, "attn_cp_size", 1) > 1,
            "does not support prefill context parallelism.",
        ),
        (
            getattr(kvc, "dcp_enabled", False)
            or _ctx_attr(get_parallel, "attn_dcp_size", 1) > 1,
            "does not support decode context parallelism.",
        ),
        (
            disagg_mode != "null",
            f"does not support PD disaggregation (disaggregation_mode={disagg_mode!r}).",
        ),
        (
            _ctx_attr(get_memory, "enable_hierarchical_cache", False)
            and hicache_backend != "raiden",
            f"with --enable-hierarchical-cache requires --hicache-io-backend raiden, got {hicache_backend!r}.",
        ),
        (
            _ctx_attr(get_memory, "enable_unified_cache_external_linker", False),
            "does not support --enable-unified-cache-external-linker.",
        ),
        (
            pool_cls is not None and pool_cls is not TpuFusedMHATokenToKVPool,
            f"requires a fused MHA KV pool, got {getattr(pool_cls, '__name__', pool_cls)}.",
        ),
        (
            getattr(kvc, "kv_cache_dtype_str", None) == "mxint8",
            "does not support mxint8 KV cache.",
        ),
        (
            getattr(kvc, "pool_page_size", 0) <= 1,
            f"requires page_size > 1, got {getattr(kvc, 'pool_page_size', None)}.",
        ),
        (
            getattr(kvc, "custom_mem_pool_head_num", None) is not None
            or getattr(kvc, "custom_mem_pool_head_dim", None) is not None,
            "does not support custom pool head shapes.",
        ),
    )
    for bad, msg in checks:
        if bad:
            return f"--enable-unified-memory on TPU {msg}"
    n_full = len(kvc.layer_info.full_attention_layer_ids)
    n_swa = len(kvc.layer_info.swa_attention_layer_ids)
    if n_full <= 0 or n_swa <= 0:
        return (
            "--enable-unified-memory on TPU requires a hybrid model with both "
            f"full ({n_full}) and SWA ({n_swa}) layers."
        )
    tp = _ctx_attr(get_parallel, "attn_tp_size", 1)
    mc = kvc.model_config
    head_dim = mc.head_dim
    swa_dim = getattr(mc, "swa_head_dim", None) or head_dim
    try:
        resolve_unified_page_shape(
            page_size=kvc.pool_page_size,
            head_num=mc.get_num_kv_heads(tp),
            head_dim=head_dim,
            v_head_dim=getattr(mc, "v_head_dim", None) or head_dim,
            swa_head_num=mc.get_swa_num_kv_heads(tp),
            swa_head_dim=swa_dim,
            swa_v_head_dim=getattr(mc, "swa_v_head_dim", None) or swa_dim,
            store_dtype=kvc.kv_cache_dtype,
        )
    except (ValueError, NotImplementedError) as exc:
        return str(exc)
    return None


def _raise_if_unsupported(kvc) -> None:
    reason = _unsupported_reason(kvc)
    if reason is not None:
        if "mxint8" in reason or "v_head_dim" in reason:
            raise NotImplementedError(reason)
        raise ValueError(reason)


def unified_kv_pool_enabled(kvc) -> bool:
    """Return True if ``--enable-unified-memory`` is active and validate ``kvc``."""
    if not get_memory().tpu_unified_kv_pool:
        return False
    _raise_if_unsupported(kvc)
    return True


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


@dataclass
class EvictionCandidate:
    """Candidate RadixCache node/window for Unified KV Pool eviction."""

    node_id: Any
    kind: str  # "full" (leaf) or "swa" (window)
    tokens: int
    last_access_time: float
    has_coupled_swa: bool = True
    num_children: int = 0
    prefix_tokens: int = 0
    is_stale_swa: bool = False
    parent_id: Any = None


@dataclass
class UnifiedKVPoolMetrics:
    """Observability snapshot for the Unified KV Pool physical page breakdown and hit rates."""

    total_usable_physical_pages: int
    full_locked_physical_pages: int
    full_cached_evictable_physical_pages: int
    swa_locked_physical_pages: int
    swa_cached_evictable_physical_pages: int
    fenced_physical_pages: int
    free_physical_pages: int
    kv_page_utilization_ratio: float
    eviction_policy: str
    prefix_cache_hit_rate: float
    prefix_cache_hit_rates_by_policy: Dict[str, float]


class TpuUnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Logical-to-Physical Unified KV allocator for Full + SWA attention on TPU."""

    SUPPORTED_EVICTION_POLICIES = (
        "full_first",
        "swa_first",
        "timestamp_interleaved",
        "stale_swa_then_whole_branch",
        "cost_benefit_weighted",
    )

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
        eviction_policy: str = "full_first",
    ):
        del size_swa
        if page_size <= 1:
            raise ValueError(
                f"TpuUnifiedSWATokenToKVPoolAllocator requires page_size > 1, got {page_size}."
            )
        if eviction_policy not in self.SUPPORTED_EVICTION_POLICIES:
            raise ValueError(
                f"Unsupported eviction_policy={eviction_policy!r}; expected one of {self.SUPPORTED_EVICTION_POLICIES}."
            )
        assert isinstance(kvcache, TpuUnifiedSWAKVPool)
        self.eviction_policy = eviction_policy
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
        self._pingpong_swa_windows: Dict[
            object, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = {}
        self._prefix_hit_stats: Dict[str, List[int]] = {
            p: [0, 0] for p in self.SUPPORTED_EVICTION_POLICIES
        }
        self.clear()

    def clear(self) -> None:
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.free_group = None
        self.swa_free_group = []
        self.swa_page_ids_group = []
        self._pingpong_swa_windows = {}

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
        credit_swa_cascade: bool = False,
        policy: Optional[str] = None,
        candidates: Optional[Sequence[EvictionCandidate]] = None,
    ) -> Optional[Tuple[int, int]]:
        if full_tokens < 0 or swa_tokens < 0:
            return None
        active_policy = policy or self.eviction_policy
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
        if active_policy == "timestamp_interleaved" and candidates:
            interleaved = self.plan_interleaved_eviction(candidates, short)
            if interleaved is None:
                return None
            return (interleaved[0], interleaved[1])
        if active_policy == "stale_swa_then_whole_branch" and candidates:
            planned = self.plan_stale_swa_then_whole_branch_eviction(candidates, short)
            if planned is None:
                return None
            return (planned[0], planned[1])
        if active_policy == "cost_benefit_weighted" and candidates:
            planned = self.plan_cost_benefit_eviction(candidates, short)
            if planned is None:
                return None
            return (planned[0], planned[1])
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
        if active_policy == "swa_first":
            s_reclaim = min(max_s, _cdiv(short, ms))
            rem = max(0, short - s_reclaim * ms)
            f_reclaim = min(max_f, rem)
            return (f_reclaim * ps, s_reclaim * ps)
        if credit_swa_cascade or active_policy == "timestamp_interleaved":
            coupled = min(max_f, max_s)
            coupled_take = min(coupled, _cdiv(short, 1 + ms))
            rem = max(0, short - coupled_take * (1 + ms))
            extra_f = min(max_f - coupled_take, rem)
            rem -= extra_f
            extra_s = _cdiv(rem, ms) if rem > 0 else 0
            return ((coupled_take + extra_f) * ps, extra_s * ps)
        f_reclaim = min(max_f, short)
        rem = short - f_reclaim
        return (f_reclaim * ps, (_cdiv(rem, ms) if rem > 0 else 0) * ps)

    def plan_interleaved_eviction(
        self,
        candidates: Sequence[EvictionCandidate],
        short_physical_pages: int,
    ) -> Optional[Tuple[int, int, List[EvictionCandidate]]]:
        """Interleave Full leaf and SWA window candidates by oldest ``last_access_time``."""
        if short_physical_pages <= 0:
            return (0, 0, [])
        ps = self.page_size
        ms = self._swa_groups
        ordered = sorted(
            candidates,
            key=lambda c: (c.last_access_time, 0 if c.kind == "swa" else 1),
        )
        swa_reclaimed_nodes: Set[Any] = set()
        full_reclaimed_nodes: Set[Any] = set()
        freed_phys = 0
        full_tokens_evicted = 0
        swa_tokens_evicted = 0
        chosen: List[EvictionCandidate] = []
        for cand in ordered:
            if freed_phys >= short_physical_pages:
                break
            pages = max(0, int(cand.tokens)) // ps
            if pages <= 0:
                continue
            if cand.kind == "swa":
                if cand.node_id in swa_reclaimed_nodes or cand.node_id in full_reclaimed_nodes:
                    continue
                swa_reclaimed_nodes.add(cand.node_id)
                freed_phys += pages * ms
                swa_tokens_evicted += pages * ps
                chosen.append(cand)
            elif cand.kind == "full":
                if cand.node_id in full_reclaimed_nodes:
                    continue
                full_reclaimed_nodes.add(cand.node_id)
                has_swa = cand.has_coupled_swa and cand.node_id not in swa_reclaimed_nodes
                if has_swa:
                    swa_reclaimed_nodes.add(cand.node_id)
                    freed_phys += pages * (1 + ms)
                else:
                    freed_phys += pages
                full_tokens_evicted += pages * ps
                chosen.append(cand)
        if freed_phys < short_physical_pages:
            return None
        return (full_tokens_evicted, swa_tokens_evicted, chosen)

    def plan_stale_swa_then_whole_branch_eviction(
        self,
        candidates: Sequence[EvictionCandidate],
        short_physical_pages: int,
    ) -> Optional[Tuple[int, int, List[EvictionCandidate]]]:
        """Evict unbranched stale SWA windows first, then zombie leaves, then whole branches with parent zombie sweep."""
        if short_physical_pages <= 0:
            return (0, 0, [])
        ps = self.page_size
        ms = self._swa_groups
        swa_reclaimed: Set[Any] = set()
        full_reclaimed: Set[Any] = set()
        freed_phys = 0
        full_tokens_evicted = 0
        swa_tokens_evicted = 0
        chosen: List[EvictionCandidate] = []

        # Index full candidates and child counts for parent zombie sweep
        full_by_id: Dict[Any, EvictionCandidate] = {}
        live_children: Dict[Any, int] = {}
        for c in candidates:
            if c.kind == "full":
                full_by_id[c.node_id] = c
                live_children[c.node_id] = c.num_children

        # Phase 1: Unbranched interior stale SWA windows (num_children == 1).
        # Fork points (num_children > 1) are preserved so new prompts matching only the shared prefix still hit.
        stale_swa_cands = sorted(
            [
                c
                for c in candidates
                if c.kind == "swa" and c.is_stale_swa and c.num_children == 1
            ],
            key=lambda c: c.last_access_time,
        )
        for cand in stale_swa_cands:
            if freed_phys >= short_physical_pages:
                break
            pages = max(0, int(cand.tokens)) // ps
            if pages <= 0 or cand.node_id in swa_reclaimed:
                continue
            swa_reclaimed.add(cand.node_id)
            freed_phys += pages * ms
            swa_tokens_evicted += pages * ps
            chosen.append(cand)

        # Phase 2: Spotlight-less "Zombie Full-KV" leaves (num_children == 0 and no live SWA window).
        zombie_leaves = sorted(
            [
                c
                for c in candidates
                if c.kind == "full"
                and live_children.get(c.node_id, c.num_children) == 0
                and (not c.has_coupled_swa or c.node_id in swa_reclaimed)
            ],
            key=lambda c: c.last_access_time,
        )
        for cand in zombie_leaves:
            if freed_phys >= short_physical_pages:
                break
            pages = max(0, int(cand.tokens)) // ps
            if pages <= 0 or cand.node_id in full_reclaimed:
                continue
            full_reclaimed.add(cand.node_id)
            freed_phys += pages
            full_tokens_evicted += pages * ps
            chosen.append(cand)
            if cand.parent_id in live_children:
                live_children[cand.parent_id] = max(0, live_children[cand.parent_id] - 1)

        # Phase 3: Live leaves (coupled Full + SWA) with automatic parent zombie sweep.
        live_leaves = sorted(
            [
                c
                for c in candidates
                if c.kind == "full"
                and c.node_id not in full_reclaimed
                and live_children.get(c.node_id, c.num_children) == 0
            ],
            key=lambda c: c.last_access_time,
        )
        for cand in live_leaves:
            if freed_phys >= short_physical_pages:
                break
            pages = max(0, int(cand.tokens)) // ps
            if pages <= 0 or cand.node_id in full_reclaimed:
                continue
            full_reclaimed.add(cand.node_id)
            has_swa = cand.has_coupled_swa and cand.node_id not in swa_reclaimed
            if has_swa:
                swa_reclaimed.add(cand.node_id)
                freed_phys += pages * (1 + ms)
            else:
                freed_phys += pages
            full_tokens_evicted += pages * ps
            chosen.append(cand)
            # Walk up parent chain: if parent becomes a spotlight-less Zombie leaf, sweep it immediately!
            pid = cand.parent_id
            while pid is not None and pid in live_children:
                live_children[pid] = max(0, live_children[pid] - 1)
                parent_cand = full_by_id.get(pid)
                if (
                    live_children[pid] == 0
                    and parent_cand is not None
                    and pid not in full_reclaimed
                    and (not parent_cand.has_coupled_swa or pid in swa_reclaimed)
                ):
                    p_pages = max(0, int(parent_cand.tokens)) // ps
                    if p_pages > 0:
                        full_reclaimed.add(pid)
                        freed_phys += p_pages
                        full_tokens_evicted += p_pages * ps
                        chosen.append(parent_cand)
                    pid = parent_cand.parent_id
                else:
                    break

        if freed_phys < short_physical_pages:
            return None
        return (full_tokens_evicted, swa_tokens_evicted, chosen)

    def score_eviction_candidate(
        self,
        cand: EvictionCandidate,
        *,
        swa_reclaimed: Optional[Set[Any]] = None,
    ) -> float:
        """Return ``recompute_cost / physical_pages_freed`` (lower score => evict first)."""
        ps = self.page_size
        ms = self._swa_groups
        pages = max(1, max(0, int(cand.tokens)) // ps)
        reclaimed = swa_reclaimed or set()
        if cand.kind == "swa":
            if cand.is_stale_swa and cand.num_children == 1:
                return 0.0
            phys_freed = pages * ms
            fanout = max(1, cand.num_children)
            depth = max(cand.tokens, cand.prefix_tokens)
            recompute_work = float(depth) * float(depth) * float(fanout)
            return recompute_work / float(phys_freed)
        # Full candidate
        has_swa = cand.has_coupled_swa and cand.node_id not in reclaimed
        if not has_swa and cand.num_children == 0:
            return 0.0
        phys_freed = pages * (1 + ms if has_swa else 1)
        depth = max(cand.tokens, cand.prefix_tokens)
        # Quadratic prefill recompute cost proportional to N * depth
        recompute_work = float(cand.tokens) * float(depth)
        return recompute_work / float(phys_freed)

    def plan_cost_benefit_eviction(
        self,
        candidates: Sequence[EvictionCandidate],
        short_physical_pages: int,
    ) -> Optional[Tuple[int, int, List[EvictionCandidate]]]:
        """Evict candidates in ascending order of ``(recompute_cost / physical_pages_freed, last_access_time)``."""
        if short_physical_pages <= 0:
            return (0, 0, [])
        ps = self.page_size
        ms = self._swa_groups
        swa_reclaimed: Set[Any] = set()
        full_reclaimed: Set[Any] = set()
        freed_phys = 0
        full_tokens_evicted = 0
        swa_tokens_evicted = 0
        chosen: List[EvictionCandidate] = []

        full_by_id: Dict[Any, EvictionCandidate] = {}
        live_children: Dict[Any, int] = {}
        for c in candidates:
            if c.kind == "full":
                full_by_id[c.node_id] = c
                live_children[c.node_id] = c.num_children

        remaining = list(candidates)
        while remaining and freed_phys < short_physical_pages:
            eligible = [
                c
                for c in remaining
                if (
                    (c.kind == "swa" and c.node_id not in swa_reclaimed and c.node_id not in full_reclaimed)
                    or (
                        c.kind == "full"
                        and c.node_id not in full_reclaimed
                        and live_children.get(c.node_id, c.num_children) == 0
                    )
                )
                and max(0, int(c.tokens)) // ps > 0
            ]
            if not eligible:
                break
            eligible.sort(
                key=lambda c: (
                    self.score_eviction_candidate(c, swa_reclaimed=swa_reclaimed),
                    c.last_access_time,
                )
            )
            cand = eligible[0]
            remaining.remove(cand)
            pages = max(0, int(cand.tokens)) // ps
            if cand.kind == "swa":
                swa_reclaimed.add(cand.node_id)
                freed_phys += pages * ms
                swa_tokens_evicted += pages * ps
                chosen.append(cand)
            else:
                full_reclaimed.add(cand.node_id)
                has_swa = cand.has_coupled_swa and cand.node_id not in swa_reclaimed
                if has_swa:
                    swa_reclaimed.add(cand.node_id)
                    freed_phys += pages * (1 + ms)
                else:
                    freed_phys += pages
                full_tokens_evicted += pages * ps
                chosen.append(cand)
                if cand.parent_id in live_children:
                    live_children[cand.parent_id] = max(0, live_children[cand.parent_id] - 1)

        if freed_phys < short_physical_pages:
            return None
        return (full_tokens_evicted, swa_tokens_evicted, chosen)

    def tombstone_swa_pages(self, full_indices: torch.Tensor) -> int:
        """Evict SWA pages for ``full_indices`` while keeping Full KV pages intact (tombstoning)."""
        if full_indices.numel() == 0:
            return 0
        before = self._free_page_count()
        self.free_swa(full_indices)
        return self._free_page_count() - before

    def validate_swa_prefix_window(
        self, full_indices: torch.Tensor, sliding_window_size: int
    ) -> int:
        """Return longest page-aligned prefix length whose trailing SWA window is non-tombstoned."""
        n = int(full_indices.numel())
        if n == 0 or sliding_window_size <= 0:
            return 0
        ps = self.page_size
        aligned_n = (n // ps) * ps
        if aligned_n == 0:
            return 0
        mapped = self.translate_loc_from_full_to_swa(full_indices[:aligned_n])
        for end in range(aligned_n, 0, -ps):
            win_start = max(0, end - sliding_window_size)
            if bool((mapped[win_start:end] > 0).all()):
                return end
        return 0

    def record_prefix_cache_lookup(
        self,
        matched_tokens: int,
        total_tokens: int,
        *,
        policy: Optional[str] = None,
    ) -> None:
        """Record prefix cache hit/lookup tokens under the active or specified eviction policy."""
        if total_tokens <= 0:
            return
        pol = policy or self.eviction_policy
        bucket = self._prefix_hit_stats.setdefault(pol, [0, 0])
        bucket[0] += max(0, min(int(matched_tokens), int(total_tokens)))
        bucket[1] += int(total_tokens)

    def get_observability_metrics(
        self,
        tree_cache=None,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
    ) -> UnifiedKVPoolMetrics:
        """Return 6-way physical page breakdown, utilization ratio, and per-strategy hit rates."""
        ps = self.page_size
        ms = self._swa_groups
        total_phys = self.size_full // ps
        fenced_phys = int(self._page_store.held_pages.num_pages)
        free_phys = int(self._page_store.unheld_free_pages())
        if tree_cache is not None:
            if hasattr(tree_cache, "full_evictable_size"):
                full_evictable_tokens = int(tree_cache.full_evictable_size())
            if hasattr(tree_cache, "swa_evictable_size"):
                swa_evictable_tokens = int(tree_cache.swa_evictable_size())
        full_alloc_phys = int(self.full_attn_allocator.allocated_pages)
        swa_alloc_phys = int(self.swa_attn_allocator.allocated_pages)
        full_cached_phys = min(full_alloc_phys, max(0, int(full_evictable_tokens)) // ps)
        full_locked_phys = max(0, full_alloc_phys - full_cached_phys)
        swa_cached_phys = min(
            swa_alloc_phys, (max(0, int(swa_evictable_tokens)) // ps) * ms
        )
        swa_locked_phys = max(0, swa_alloc_phys - swa_cached_phys)
        used_phys = max(0, total_phys - free_phys)
        util = float(used_phys) / float(total_phys) if total_phys > 0 else 0.0
        hit_rates: Dict[str, float] = {}
        for pol, (hit_tok, tot_tok) in self._prefix_hit_stats.items():
            hit_rates[pol] = float(hit_tok) / float(tot_tok) if tot_tok > 0 else 0.0
        active_hit_rate = hit_rates.get(self.eviction_policy, 0.0)
        return UnifiedKVPoolMetrics(
            total_usable_physical_pages=total_phys,
            full_locked_physical_pages=full_locked_phys,
            full_cached_evictable_physical_pages=full_cached_phys,
            swa_locked_physical_pages=swa_locked_phys,
            swa_cached_evictable_physical_pages=swa_cached_phys,
            fenced_physical_pages=fenced_phys,
            free_physical_pages=free_phys,
            kv_page_utilization_ratio=util,
            eviction_policy=self.eviction_policy,
            prefix_cache_hit_rate=active_hit_rate,
            prefix_cache_hit_rates_by_policy=hit_rates,
        )

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

    def can_reserve_pingpong_prefill(
        self,
        extend_tokens: int,
        sliding_window_size: int,
        *,
        has_prev_window: bool = False,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
    ) -> bool:
        """Check if HBM can admit a prefill chunk using 2-window ping-pong SWA allocation."""
        if extend_tokens < 0 or sliding_window_size <= 0:
            return False
        curr_swa_tail = min(extend_tokens, sliding_window_size)
        prev_swa_tail = sliding_window_size if has_prev_window else 0
        return self.can_reserve(
            extend_tokens,
            curr_swa_tail + prev_swa_tail,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
        )

    def alloc_extend_pingpong_swa(
        self,
        req_id: object,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        sliding_window_size: int,
    ) -> Optional[torch.Tensor]:
        """Allocate Full KV for the whole prefill chunk and SWA KV only for the tail window,
        ping-ponging at most 2 consecutive SWA tail windows across chunks."""
        if sliding_window_size <= 0:
            raise ValueError(f"sliding_window_size must be > 0, got {sliding_window_size}")
        swa_tail_len = min(extend_num_tokens, sliding_window_size)
        prev_win, curr_win = self._pingpong_swa_windows.get(req_id, (None, None))
        if prev_win is not None:
            self.free_swa(prev_win)
            prev_win = None
        alloc_full = self.alloc_extend_swa_tail(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            swa_tail_len,
        )
        if alloc_full is None:
            self._pingpong_swa_windows[req_id] = (prev_win, curr_win)
            return None
        new_tail = alloc_full[-swa_tail_len:].detach().cpu() if swa_tail_len > 0 else None
        self._pingpong_swa_windows[req_id] = (curr_win, new_tail)
        return alloc_full

    def advance_pingpong_swa_window(
        self, req_id: object, *, release_all: bool = False
    ) -> int:
        """Release the older ping-pong SWA window after a chunk completes (or both if release_all)."""
        prev_win, curr_win = self._pingpong_swa_windows.get(req_id, (None, None))
        freed_tokens = 0
        if prev_win is not None:
            freed_tokens += int(prev_win.numel())
            self.free_swa(prev_win)
            prev_win = None
        if release_all and curr_win is not None:
            freed_tokens += int(curr_win.numel())
            self.free_swa(curr_win)
            curr_win = None
        if prev_win is None and curr_win is None:
            self._pingpong_swa_windows.pop(req_id, None)
        else:
            self._pingpong_swa_windows[req_id] = (prev_win, curr_win)
        return freed_tokens

    def debug_print(self) -> str:
        return (
            f"#free-pages: {self._free_page_count()}, "
            f"#full-pages: {self.full_attn_allocator.allocated_pages}, "
            f"#swa-pages: {self.swa_attn_allocator.allocated_pages}, "
        )


class _UnifiedSplitMixin:
    """Prices SWA tokens at ``M_swa * L_full`` shared buffer slots and reserves page 0."""

    def __init__(self, kvc):
        super().__init__(kvc)
        m_swa = num_swa_groups(self._full_layers_num, self._swa_layers_num)
        self._swa_per_token = self._full_per_token
        self._swa_layers_num = self._full_layers_num * m_swa
        self._recompute_cell_size()

    def _dummy_page_bytes(self, page_size: int) -> int:
        return page_size * self._full_per_token * self._full_layers_num

    @staticmethod
    def validate_swa_pool_size(
        swa_tokens: int, sliding_window_size: Optional[int], page_size: int
    ) -> None:
        pass


class _SharedPagesRatioSplit(_UnifiedSplitMixin, HybridSWAPoolConfigurator):
    pass


class _UnifiedChunkCapPoolConfigurator(_UnifiedSplitMixin, SWAChunkCapPoolConfigurator):
    pass


class TpuUnifiedSWAPoolConfigurator(MemoryPoolConfigurator):
    """Sizes the Logical-to-Physical unified Full+SWA KV pool from ``G = L_full``."""

    def __init__(self, kvc):
        _raise_if_unsupported(kvc)
        n_full = len(kvc.layer_info.full_attention_layer_ids)
        n_swa = len(kvc.layer_info.swa_attention_layer_ids)
        self._sliding_window_size = kvc.sliding_window_size
        self._num_swa_groups = num_swa_groups(n_full, n_swa)
        split_cls = (
            _UnifiedChunkCapPoolConfigurator
            if SWAChunkCapPoolConfigurator.is_applicable(kvc)
            else _SharedPagesRatioSplit
        )
        self._split = split_cls(kvc)
        self._page_bytes = kvc.pool_page_size * self._split._full_per_token * n_full
        self._reserved_bytes = self._page_bytes
        self._align_pages = 8

    def _unify(self, cfg: MemoryPoolConfig, page_size: int) -> MemoryPoolConfig:
        full_tokens = cfg.full_max_total_num_tokens or 0
        swa_tokens = cfg.swa_max_total_num_tokens or 0
        raw_shared = full_tokens + swa_tokens * self._num_swa_groups
        align = self._align_pages * page_size
        shared_tokens = (raw_shared // align) * align
        if self._sliding_window_size is not None:
            min_tokens = (1 + self._num_swa_groups) * (
                self._sliding_window_size + page_size
            )
            if shared_tokens <= min_tokens:
                raise ValueError(
                    f"Unified KV pool ({shared_tokens} page-id tokens) cannot hold "
                    f"even one request: need > {min_tokens} tokens."
                )
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=shared_tokens,
            swa_max_total_num_tokens=shared_tokens,
        )

    def calculate_total_physical_pages(
        self, available_bytes: int, page_size: int
    ) -> int:
        """Compute aligned physical page count directly from free HBM quota and Full layers."""
        usable = max(0, available_bytes - self._reserved_bytes)
        page_bytes = (
            page_size * self._split._full_per_token * self._split._full_layers_num
        )
        raw_pages = usable // page_bytes if page_bytes > 0 else 0
        return (raw_pages // self._align_pages) * self._align_pages

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._unify(
            self._split.calculate_pool_sizes(available_bytes, page_size), page_size
        )

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._unify(
            self._split.calculate_pool_sizes_from_max_tokens(
                max_total_num_tokens, page_size
            ),
            page_size,
        )


