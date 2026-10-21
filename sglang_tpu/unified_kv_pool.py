# SPDX-License-Identifier: Apache-2.0
"""One KV pool shared by the full-attention and sliding-window layers on TPU.

The pool keeps one per-layer buffer per full-attention layer, and full layer i
uses buffer i. The sliding-window layers are split by stride into
ceil(#sliding-window / #full) groups of at most #full layers, and the k-th layer
of a group uses buffer k. A page id is a row in every buffer: a full page takes
one id and a sliding-window page one id per group, all from a single free list.
Each id belongs to at most one side at a time, so two layers sharing a buffer
never touch the same rows, and capacity moves to whichever side needs it. When
a full-attention layer has one KV head as wide as all of a sliding-window
layer's KV heads together, the buffers keep the full layer's shape and the
sliding-window layers run as that one merged head. The page tables, the
full->SWA token mapping and the attention kernels are those the static pools
use; the mapping points into shared page ids, and every further sliding-window
group gets its own page table.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.prefill_budget import (
    SharedSWAPrefillBudget,
    estimate_swa_kv_tokens,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.pool_configurator import (
    HybridSWAPoolConfigurator,
    MemoryPoolConfig,
    MemoryPoolConfigurator,
    SWAChunkCapPoolConfigurator,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
from sglang.srt.utils.common import get_num_new_pages
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_free_fence import HeldPages
from sglang_tpu.kv_pool import (
    TpuFusedMHATokenToKVPool,
    _cdiv,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang_tpu.kv_pool_host import TpuRaidenHostKVCache

logger = logging.getLogger(__name__)

_EVICTION_POLICIES = ("full_first", "swa_first", "static_shadow", "compact")

# KVCacheConfigurator._derive_pool_sizes rounds a TPU SWA pool down to this many
# pages; the unified pool is that pool, so it is sized in the same unit.
_TPU_SWA_PAGE_ALIGNMENT = 8


def num_swa_groups(*, num_full_layers: int, num_swa_layers: int) -> int:
    """How many page ids a sliding-window page takes: its layers form this many
    groups of at most ``num_full_layers``, one buffer row per layer."""
    return max(1, _cdiv(num_swa_layers, num_full_layers))


def _fused_page_shape(
    *, page_size: int, num_kv_heads: int, head_dim: int, itemsize: int
) -> Tuple[int, int, int, int]:
    """One layer's page in the fused RPA layout, as TpuFusedMHATokenToKVPool
    builds it: K and V interleaved on the heads axis and packed in 32-bit lanes."""
    packing = 4 // itemsize
    heads_x2 = _cdiv(num_kv_heads * 2, packing) * packing
    alignment = TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
    return (
        page_size,
        heads_x2 // packing,
        packing,
        _cdiv(head_dim, alignment) * alignment,
    )


def _merged_swa_heads_fit(
    *, full_page: Tuple[int, ...], swa_page: Tuple[int, ...], swa_kv_heads: int
) -> bool:
    """Whether the full layer's page is one KV head holding all of the
    sliding-window layer's KV heads side by side, so the sliding-window layers
    can run as that one head (merge_swa_kv_heads), in no more bytes than their
    own page; the static budget the shared pool is sized from counts those."""
    page_size, full_packed_heads, packing, full_head_dim = full_page
    return (
        full_packed_heads == 1
        and full_head_dim == swa_kv_heads * swa_page[3]
        and swa_page == (page_size, swa_kv_heads * 2 // packing, packing, swa_page[3])
        and math.prod(full_page) == math.prod(swa_page)
    )


def unified_kv_pool_enabled(kvc: KVCacheConfigurator) -> bool:
    """Whether the KV cache is the unified pool (--enable-unified-memory on TPU).

    Raises instead of returning False when the flag is set on a configuration
    the pool cannot serve, so the flag is never a silent no-op.
    """
    if not get_memory().tpu_unified_kv_pool:
        return False
    _raise_if_unsupported(kvc)
    return True


def _raise_if_unsupported(kvc: KVCacheConfigurator) -> None:
    reason = _unsupported_reason(kvc)
    if reason is not None:
        raise ValueError(f"--enable-unified-memory on TPU does not support {reason}.")


def _unsupported_reason(kvc: KVCacheConfigurator) -> Optional[str]:
    # First: the reasons below assume a hybrid model and would misdescribe others.
    if not kvc.is_hybrid_swa:
        return (
            "models without sliding-window layers or with "
            "--disable-hybrid-swa-memory (it shares pages between "
            "full-attention and sliding-window KV)"
        )
    if kvc.use_mla_backend or kvc.mambaish_config is not None:
        return "MLA or recurrent-state models (only MHA layers share one layout)"
    pool_cls = current_platform.get_mha_kv_pool_cls()
    if not (
        issubclass(pool_cls, TpuFusedMHATokenToKVPool)
        and pool_cls.supports_unified_kv_pool
    ):
        return (
            f"the {pool_cls.__name__} KV pool (the fused RPA layout of "
            "pallas_tpu_rpa_v3, or pallas_tpu_rpa_batched without "
            "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV, and pools that set "
            "supports_unified_kv_pool are supported)"
        )
    if kvc.kv_cache_dtype_str == "mxint8":
        return "the mxint8 KV cache"
    if kvc.page_size == 1:
        return "page_size 1 (the shared free list is paged)"
    model_config = kvc.model_config
    tp_size = get_parallel().attn_tp_size
    # Merging sliding-window heads relies on the fused layout's page shape.
    if pool_cls is not TpuFusedMHATokenToKVPool and (
        model_config.get_num_kv_heads(tp_size),
        model_config.head_dim,
        model_config.v_head_dim,
    ) != (
        model_config.get_swa_num_kv_heads(tp_size),
        model_config.swa_head_dim,
        model_config.swa_v_head_dim,
    ):
        return (
            "sliding-window KV heads other than the full-attention ones in the "
            f"{pool_cls.__name__} KV pool"
        )
    itemsize = kvc.kv_cache_dtype.itemsize
    full_page = _fused_page_shape(
        page_size=kvc.page_size,
        num_kv_heads=model_config.get_num_kv_heads(tp_size),
        head_dim=model_config.head_dim,
        itemsize=itemsize,
    )
    swa_page = _fused_page_shape(
        page_size=kvc.page_size,
        num_kv_heads=model_config.get_swa_num_kv_heads(tp_size),
        head_dim=model_config.swa_head_dim,
        itemsize=itemsize,
    )
    if full_page != swa_page and not _merged_swa_heads_fit(
        full_page=full_page,
        swa_page=swa_page,
        swa_kv_heads=model_config.get_swa_num_kv_heads(tp_size),
    ):
        return (
            "sliding-window KV pages that neither match the full-attention ones "
            "nor fit as one full-attention KV head "
            f"(KV page shapes {full_page} and {swa_page} per rank)"
        )
    layer_info = kvc.layer_info
    if (
        not layer_info.full_attention_layer_ids
        or not layer_info.swa_attention_layer_ids
    ):
        return "a rank without both full-attention and sliding-window layers"
    if not kvc.spec_algorithm.is_none() or kvc.is_draft_worker:
        return "speculative decoding"
    # Pipeline-parallel PD uses raiden's pipeline transfer, which registers each
    # buffer the two views share once per view.
    if get_disagg().disaggregation_mode != "null" and get_parallel().pp_size > 1:
        return "PD disaggregation with pipeline parallelism"
    # A decode fed by a pipeline prefill takes that transfer as well, and its
    # per-layer pools would receive every sliding-window group at group 0's pages.
    if (
        get_disagg().disaggregation_mode == "decode"
        and envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get()
    ):
        return "PD disaggregation with SGLANG_RAIDEN_PIPELINE_TRANSFER"
    # A sliding-window page moves as its page id in every group
    # (swa_transfer_pages); merged sliding-window heads are not validated.
    if get_disagg().disaggregation_mode != "null" and full_page != swa_page:
        return "PD disaggregation with merged sliding-window heads"
    if get_parallel().pp_size > 1:
        return "pipeline parallelism"
    if get_parallel().attn_cp_size > 1 or get_parallel().dcp_enabled:
        return "context parallelism"
    # Only raiden's host pool can be shared by both views; torch_tpu's mirrors
    # one sub-pool's buffers.
    if (
        get_memory().enable_hierarchical_cache
        and get_memory().hicache_io_backend != "raiden"
    ):
        return "the hierarchical cache with an io backend other than raiden"
    if get_memory().enable_unified_cache_external_linker:
        return "the unified-cache external linker"
    return None


@functools.cache
def _unified_pool_classes(pool_cls: type) -> Tuple[type, type]:
    """The backing pool and view classes of a unified pool over ``pool_cls``,
    the platform's MHA pool class: they keep its buffer layout and the methods
    its attention backends call on the pool."""
    if pool_cls is TpuFusedMHATokenToKVPool:
        return _UnifiedBackingPool, _SharedBufferView
    # The unified classes come first, so their buffer sharing overrides the
    # pool class's own buffer creation and state.
    return (
        type(
            f"_UnifiedBacking{pool_cls.__name__}", (_UnifiedBackingPool, pool_cls), {}
        ),
        type(
            f"_SharedBufferView{pool_cls.__name__}", (_SharedBufferView, pool_cls), {}
        ),
    )


class _UnifiedBackingPool(TpuFusedMHATokenToKVPool):
    """The per-layer buffers of the unified pool, which both views share."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The raiden host pool shared by both views' HiCache host pools; built
        # with the first of them.
        self.host_pool: Optional[TpuRaidenHostKVCache] = None


class _SharedBufferView(TpuFusedMHATokenToKVPool):
    """One attention kind's view of the unified pool: its layer i is the
    unified pool's buffer ``buffer_ids[i]``, and it owns no storage of its own.
    ``group_pages`` is the unified pool's swa_group_pages on the sliding-window
    view of a pool with several groups, else None."""

    def __init__(
        self,
        *,
        backing: _UnifiedBackingPool,
        buffer_ids: Sequence[int],
        group_pages: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Set first: the parent constructor calls _create_buffers().
        self._backing = backing
        self._buffer_ids = tuple(buffer_ids)
        self.group_pages = group_pages
        self._view_kwargs = kwargs
        self._group_views: Optional[List[_SharedBufferView]] = None
        super().__init__(**kwargs)

    @property
    def backing(self) -> _UnifiedBackingPool:
        return self._backing

    @property
    def kv_buffer(self) -> List[torch.Tensor]:
        # Read through on every access: the unified pool's set_state() swaps
        # the buffer list while compiled graphs are traced.
        buffers = self._backing.kv_buffer
        return [buffers[i] for i in self._buffer_ids]

    def get_fused_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self._backing.kv_buffer[self._buffer_ids[layer_id - self.start_layer]]

    def _create_buffers(self):
        backing = self._backing
        assert len(self._buffer_ids) == self.layer_num
        assert max(self._buffer_ids, default=-1) < backing.layer_num
        assert (self.num_blocks, self.page_size, self.store_dtype) == (
            backing.num_blocks,
            backing.page_size,
            backing.store_dtype,
        ), "a view must have the pages of the buffers it shares"
        assert self.num_kv_heads_x2 * self.padded_head_dim == (
            backing.num_kv_heads_x2 * backing.padded_head_dim
        ), "a view's page must fill a shared buffer page"
        # Gathered on the host: the device tables are uint64.
        ids = list(self._buffer_ids)
        self.data_ptrs = backing.data_ptrs.cpu()[ids].to(backing.data_ptrs.device)
        self.data_strides = backing.data_strides.cpu()[ids].to(
            backing.data_strides.device
        )

    def _clear_buffers(self):
        # The unified pool owns the buffers.
        pass

    def get_state(self) -> list[torch.Tensor]:
        # The unified pool reports each shared buffer once.
        return []

    def set_state(self, state: list[torch.Tensor]) -> None:
        assert not state, "set the unified pool's state, not a view's"

    def _finalize_allocation_log(self, num_tokens: int):
        logger.info(
            "%s KV cache shares the unified KV buffers: layers=%d, #tokens=%d",
            self.allocation_label,
            self.layer_num,
            num_tokens,
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # ``indices`` are group-0 slots; layer j of a grouped view lives at
        # group j % groups' companion slots.
        if self.group_pages is None:
            return super().get_cpu_copy(indices, mamba_indices, req_pool_index)
        groups = self._groups(indices)
        rows = [view.get_cpu_copy(slots) for view, slots in groups]
        return [rows[j % len(groups)][j // len(groups)] for j in range(self.layer_num)]

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        if self.group_pages is None:
            return super().load_cpu_copy(
                kv_cache_cpu, indices, mamba_indices, req_pool_index
            )
        groups = self._groups(indices)
        for g, (view, slots) in enumerate(groups):
            view.load_cpu_copy(kv_cache_cpu[g :: len(groups)], slots)

    def _groups(self, indices) -> List[Tuple[_SharedBufferView, torch.Tensor]]:
        """(view of group g's layers, their slots) for every group g, from the
        group-0 token slots ``indices``."""
        if self._group_views is None:
            num_groups = self.group_pages.shape[0] + 1
            self._group_views = [
                type(self)(
                    backing=self._backing,
                    buffer_ids=self._buffer_ids[g::num_groups],
                    **{
                        **self._view_kwargs,
                        "layer_num": len(self._buffer_ids[g::num_groups]),
                    },
                )
                for g in range(num_groups)
            ]
        slots = indices.to("cpu", torch.int64)
        pages, offsets = slots // self.page_size, slots % self.page_size
        return [(self._group_views[0], slots)] + [
            (
                view,
                self.group_pages[g - 1, pages].to(torch.int64) * self.page_size
                + offsets,
            )
            for g, view in enumerate(self._group_views[1:], start=1)
        ]


class TpuUnifiedSWAKVPool(SWAKVPool):
    """Hybrid full/sliding-window pool whose two sides share per-layer buffers.

    ``size`` is the whole shared pool in tokens; either side may use any page.
    Layer routing, the full->SWA mapping and host copies are inherited and run
    through the two views. ``num_swa_groups`` page tables serve the
    sliding-window layers; the allocator records in ``swa_group_pages`` which
    page each further group uses alongside each group-0 sliding-window page.
    The buffers and views take the layout of ``token_to_kv_pool_class``.
    """

    def __init__(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_head_num: int,
        swa_head_dim: int,
        swa_v_head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        device: str,
        start_layer: int,
        token_to_kv_pool_class: type = TpuFusedMHATokenToKVPool,
    ):
        num_full = len(full_attention_layer_ids)
        num_swa = len(swa_attention_layer_ids)
        self.num_swa_groups = num_swa_groups(
            num_full_layers=num_full, num_swa_layers=num_swa
        )
        groups = self.num_swa_groups
        # i32[num_swa_groups - 1, pages + 1]: [g - 1, p] is group g's page of the
        # sliding-window page whose group-0 id is p.
        self.swa_group_pages = torch.zeros(
            (groups - 1, size // page_size + 1), dtype=torch.int32
        )
        itemsize = torch.empty((), dtype=dtype).element_size()
        full_page = _fused_page_shape(
            page_size=page_size,
            num_kv_heads=head_num,
            head_dim=head_dim,
            itemsize=itemsize,
        )
        swa_page = _fused_page_shape(
            page_size=page_size,
            num_kv_heads=swa_head_num,
            head_dim=swa_head_dim,
            itemsize=itemsize,
        )
        # The buffers keep the full layers' shape; sliding-window layers of
        # another shape run as one merged head of it (merge_swa_kv_heads).
        self.swa_kv_heads_merged = full_page != swa_page
        assert not self.swa_kv_heads_merged or _merged_swa_heads_fit(
            full_page=full_page, swa_page=swa_page, swa_kv_heads=swa_head_num
        )
        backing_cls, view_cls = _unified_pool_classes(token_to_kv_pool_class)
        # Set first: SWAKVPool.__init__ calls get_kv_size_bytes().
        self._backing = backing_cls(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=max(num_full, _cdiv(num_swa, groups)),
            device=device,
            enable_memory_saver=False,
            allocation_label="Unified",
        )
        view = functools.partial(view_cls, backing=self._backing)
        # Group g holds sliding-window layers g, g + groups, ...
        swa_buffer_ids = [j // groups for j in range(num_swa)]
        self._swa_buffer_ids = sorted(set(swa_buffer_ids))
        super().__init__(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            full_kv_pool_class=functools.partial(view, buffer_ids=range(num_full)),
            swa_kv_pool_class=functools.partial(
                view,
                buffer_ids=swa_buffer_ids,
                group_pages=self.swa_group_pages if groups > 1 else None,
            ),
            start_layer=start_layer,
        )
        # HiCache sizes host memory from these weights; both views alias the
        # backing buffers, so they are counted once.
        self.full_kv_pool.host_capacity_bytes = sum(self._backing.get_kv_size_bytes())
        self.swa_kv_pool.host_capacity_bytes = 0

    def swa_page_group(self, layer_id: int) -> int:
        """The page table, 0 to num_swa_groups - 1, that sliding-window layer
        ``layer_id`` reads."""
        index, is_swa = self.layers_mapping[layer_id]
        assert is_swa
        return index % self.num_swa_groups

    def swa_transfer_buffers(self) -> List[torch.Tensor]:
        """The buffers the sliding-window layers use, each once, in buffer
        order. Copying a page id's row of each moves that page id's KV for
        every sliding-window layer that reads it."""
        buffers = self._backing.kv_buffer
        return [buffers[i] for i in self._swa_buffer_ids]

    def swa_transfer_pages(self, page_ids: Sequence[int]) -> List[int]:
        """The page ids that hold the sliding-window pages ``page_ids`` (their
        group-0 ids), page by page: [p0 in group 0, ..., p0 in the last group,
        p1 in group 0, ...], so a tail of the list is a tail of the pages."""
        ids = torch.as_tensor(page_ids, dtype=torch.int64).reshape(1, -1)
        if self.num_swa_groups > 1:
            ids = torch.cat([ids, self.swa_group_pages[:, ids[0]].to(torch.int64)])
        return ids.t().reshape(-1).tolist()

    def get_state(self) -> list[torch.Tensor]:
        return self._backing.get_state()

    def set_state(self, state: list[torch.Tensor]) -> None:
        self._backing.set_state(state)

    def get_kv_size_bytes(self):
        # The views report the buffers they reference, so their sum would count
        # every buffer both sides use twice.
        return self._backing.get_kv_size_bytes()


class _SharedPageStore:
    """Free pages both sides of the unified allocator draw from."""

    def __init__(self):
        self.free_pages: Optional[torch.Tensor] = None
        self.page_is_free: Optional[torch.Tensor] = None
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0
        self.held_pages = HeldPages()

    def num_free_pages(self) -> int:
        return len(self.free_pages) + self.num_staged_pages + self.held_pages.num_pages


class _SharedPagesAllocator(TpuPagedTokenToKVPoolAllocator):
    """One side (full or sliding-window) of the unified allocator.

    Its free list and page bitmap live in the shared page store, so every
    inherited take and release path moves pages between this side and the store,
    and a page released by one side and taken by the other is tracked once.
    ``allocated_pages``
    counts the pages this side currently holds. With ``group_pages`` (the
    sliding-window side of a pool with several groups), every page id this side
    hands out brings one page per further group, recorded in ``group_pages`` and
    released with it.
    """

    def __init__(
        self,
        *,
        page_store: _SharedPageStore,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: TpuFusedMHATokenToKVPool,
        need_sort: bool,
        group_pages: Optional[torch.Tensor] = None,
    ):
        # Set first: the parent constructor assigns free_pages and calls clear().
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

    @property
    def free_pages(self) -> Optional[torch.Tensor]:
        return self._page_store.free_pages

    @free_pages.setter
    def free_pages(self, value: Optional[torch.Tensor]) -> None:
        self._page_store.free_pages = value

    @property
    def staged_pages(self) -> list[torch.Tensor]:
        return self._page_store.staged_pages

    @staged_pages.setter
    def staged_pages(self, value: list[torch.Tensor]) -> None:
        self._page_store.staged_pages = value

    @property
    def num_staged_pages(self) -> int:
        return self._page_store.num_staged_pages

    @num_staged_pages.setter
    def num_staged_pages(self, value: int) -> None:
        self._page_store.num_staged_pages = value

    @property
    def held_pages(self) -> HeldPages:
        return self._page_store.held_pages

    @held_pages.setter
    def held_pages(self, value: HeldPages) -> None:
        self._page_store.held_pages = value

    @property
    def page_is_free(self) -> Optional[torch.Tensor]:
        return self._page_store.page_is_free

    @page_is_free.setter
    def page_is_free(self, value: Optional[torch.Tensor]) -> None:
        self._page_store.page_is_free = value

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

    def available_size(self):
        return self._page_store.num_free_pages() // self._pages_per_id * self.page_size

    def _ensure_free_pages(self, num_pages: int) -> bool:
        if self._group_pages is None:
            return super()._ensure_free_pages(num_pages)
        ok = super()._ensure_free_pages(num_pages * self._pages_per_id)
        # Every take path slices the ids it hands out off the front of this list.
        self._free_pages_before_take = self.free_pages
        return ok

    def _take_group_pages(self) -> None:
        before = self._free_pages_before_take
        self._free_pages_before_take = None
        if before is None:
            return
        num_taken = len(before) - len(self.free_pages)
        if num_taken == 0:
            return
        # _ensure_free_pages kept the further groups' pages behind the taken ids.
        num_extra = self._group_pages.shape[0] * num_taken
        extra = self.free_pages[:num_extra]
        self._mark_handed_out(extra)
        self._group_pages[:, before[:num_taken].long()] = extra.view(-1, num_taken)
        self.free_pages = self.free_pages[num_extra:]

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        # Every release path of the paged allocators, grouped or not, ends here.
        # Page 0 takes the padding writes and is never handed out, but an unmapped
        # SWA slot translates to it; on the shared list it would reach both sides.
        page_ids = tuple(
            ids[ids > 0] for ids in (p.reshape(-1).to("cpu") for p in page_ids)
        )
        if self._group_pages is not None:
            ids = torch.cat([p.reshape(-1).to("cpu", torch.int64) for p in page_ids])
            page_ids = (*page_ids, self._group_pages[:, ids].reshape(-1))
        self.allocated_pages -= sum(ids.numel() for ids in page_ids)
        super()._release_page_ids(*page_ids)

    def clear(self) -> None:
        # Resets the shared store; only ever called for both sides together.
        super().clear()
        self.allocated_pages = 0


class TpuUnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Hybrid SWA allocator whose full and sliding-window sides share pages.

    Both sides allocate from one free list, so a free page can become either a
    full page or an SWA page, and the capacity queries answer for both sides
    against that one list. An SWA page costs one page id per group of the
    pool's sliding-window layers; a new token can need one page on each side.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: TpuUnifiedSWAKVPool,
        need_sort: bool,
        req_to_token_pool=None,
        static_full_tokens: Optional[int] = None,
    ):
        assert page_size > 1, "the unified allocator pages its shared free list"
        assert isinstance(kvcache, TpuUnifiedSWAKVPool)
        assert size == kvcache.size, "the allocator hands out the whole shared pool"
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
        self._page_store = _SharedPageStore()
        self._swa_groups = kvcache.num_swa_groups
        self._eviction_policy, self._full_share, self._swa_share = self._eviction_setup(
            kvcache, size // page_size, static_full_tokens
        )
        self._compact_retain_tokens = (
            tpu_envs.SGLANG_TPU_UNIFIED_KV_COMPACT_RETAIN_TOKENS.get()
        )
        self.full_attn_allocator = self._build_side(kvcache.full_kv_pool)
        self.swa_attn_allocator = self._build_side(
            kvcache.swa_kv_pool,
            group_pages=kvcache.swa_group_pages if self._swa_groups > 1 else None,
        )
        self.clear()
        logger.info(
            "Unified KV allocator: %d pages of %d tokens shared by the "
            "full-attention and sliding-window layers; an SWA page takes %d",
            self._total_pages(),
            page_size,
            self._swa_groups,
        )

    def _eviction_setup(
        self,
        kvcache: TpuUnifiedSWAKVPool,
        total_pages: int,
        static_full_tokens: Optional[int],
    ) -> Tuple[str, int, int]:
        """The pressure policy (SGLANG_TPU_UNIFIED_KV_EVICTION) and each kind's
        share in page ids: the pages the static full pool and SWA pool (times
        the groups) would get from the same bytes."""
        policy = tpu_envs.SGLANG_TPU_UNIFIED_KV_EVICTION.get()
        if policy not in _EVICTION_POLICIES:
            raise ValueError(
                f"SGLANG_TPU_UNIFIED_KV_EVICTION={policy!r}; expected one of "
                f"{', '.join(_EVICTION_POLICIES)}"
            )
        # With requests freeing out-of-window slots, cached nodes keep only the
        # windows a match needs and the full side keeps evicting first.
        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            policy = "full_first"
        if static_full_tokens is None:
            # Without the static split only the share-free orders apply.
            if policy not in ("full_first", "swa_first"):
                policy = "full_first"
            return policy, total_pages, 0
        full_share = static_full_tokens // self.page_size
        # The configurator sized the shared buffers from the static pools:
        # buffers x shared pages = full layers x full pages + SWA layers x SWA pages.
        swa_pages = (
            total_pages * kvcache.full_kv_pool.backing.layer_num
            - full_share * kvcache.full_layer_nums
        ) // kvcache.swa_layer_nums
        logger.info(
            "Unified KV eviction under pressure: %s (static shares: %d full, "
            "%d SWA page ids)",
            policy,
            full_share,
            swa_pages * self._swa_groups,
        )
        return policy, full_share, swa_pages * self._swa_groups

    def _build_side(
        self,
        kvcache: TpuFusedMHATokenToKVPool,
        group_pages: Optional[torch.Tensor] = None,
    ) -> _SharedPagesAllocator:
        return _SharedPagesAllocator(
            page_store=self._page_store,
            size=self._size_full,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=kvcache,
            need_sort=self.need_sort,
            group_pages=group_pages,
        )

    def _total_pages(self) -> int:
        return self.size_full // self.page_size

    def _free_page_count(self) -> int:
        return self._page_store.num_free_pages()

    def _page_ids_needed(self, num_full_pages: int, num_swa_pages: int) -> int:
        return num_full_pages + self._swa_groups * num_swa_pages

    def available_size(self):
        return self._free_page_count() // (1 + self._swa_groups) * self.page_size

    def full_available_size(self):
        # Either side alone can take every free page.
        return self._free_page_count() * self.page_size

    def swa_available_size(self):
        return self._free_page_count() // self._swa_groups * self.page_size

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            self._page_ids_needed(num_full_pages, num_swa_pages)
            <= self._free_page_count()
        )

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        # A side's capacity is what it holds plus every free page it could
        # take, which keeps available + held == capacity for the
        # pool-conservation checks.
        free_pages = self._free_page_count()
        page_size = self.page_size
        groups = self._swa_groups
        return (
            (
                (free_pages + self.full_attn_allocator.allocated_pages) * page_size,
                free_pages * page_size,
            ),
            (
                (
                    free_pages // groups
                    + self.swa_attn_allocator.allocated_pages // groups
                )
                * page_size,
                free_pages // groups * page_size,
            ),
        )

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        return SharedSWAPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
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
        # Demand is in tokens (fractional ones round up); the budget is the free
        # plus the evictable pages, or every page for an empty pool.
        if full_tokens < 0 or swa_tokens < 0:
            return False
        if require_token_slack and (
            full_tokens >= self.size_full or swa_tokens >= self.size_swa
        ):
            return False
        page_size = self.page_size
        needed_pages = self._page_ids_needed(
            (math.ceil(full_tokens) + page_size - 1) // page_size,
            (math.ceil(swa_tokens) + page_size - 1) // page_size,
        )
        if empty_pool:
            budget_pages = self._total_pages()
        else:
            budget_pages = self._free_page_count() + self._page_ids_needed(
                int(full_evictable_tokens) // page_size,
                int(swa_evictable_tokens) // page_size,
            )
        return needed_pages <= budget_pages

    def max_new_tokens_for_memory(
        self,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
    ) -> int | None:
        def fits(candidate: int) -> bool:
            return self.can_reserve(
                input_tokens + candidate + self.page_size,
                estimate_swa_kv_tokens(
                    input_tokens,
                    candidate,
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
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> bool | None:
        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return None
        num_pages = -(-num_tokens // self.page_size)
        self._evict_shortfall(tree_cache, num_pages, num_pages)
        return self._page_ids_needed(num_pages, num_pages) <= self._free_page_count()

    def _evict_shortfall(self, tree_cache, full_pages: int, swa_pages: int) -> None:
        need_pages = self._page_ids_needed(full_pages, swa_pages)
        if need_pages <= self._free_page_count():
            return
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        page_size = self.page_size
        groups = self._swa_groups
        if self._eviction_policy == "full_first":
            # Both availabilities read the one free count, so the targets name
            # the same pages: the tree evicts full KV first and moves on to SWA
            # KV, which frees one page per group, only if that did not free enough.
            short = need_pages - self._free_page_count()
            tree_cache.evict_for_alloc(
                EvictParams(
                    num_tokens=short * page_size,
                    swa_num_tokens=-(-short // groups) * page_size,
                )
            )
            return
        if self._eviction_policy == "swa_first":
            # Full KV no match can use, then the SWA walk for the whole
            # shortfall (LRU interior nodes tombstoned, LRU leaves whole), then
            # full-first LRU, each for the shortfall left; shares not consulted.
            if (
                isinstance(tree_cache, UnifiedRadixCache)
                and get_disagg().disaggregation_mode != "decode"
            ):
                tree_cache.evict_unmatchable_device_leaves(
                    lambda: self._free_page_count() >= need_pages
                )
            for with_full in (False, True):
                short = need_pages - self._free_page_count()
                if short <= 0:
                    return
                tree_cache.evict_for_alloc(
                    EvictParams(
                        num_tokens=short * page_size if with_full else 0,
                        swa_num_tokens=-(-short // groups) * page_size,
                    )
                )
            return

        def shortfall() -> int:
            return need_pages - self._free_page_count()

        def done() -> bool:
            return shortfall() <= 0

        def overshoots() -> Tuple[int, int]:
            # Page ids each kind would hold past its static share with this
            # request's pages in: what the static pools would evict from it.
            return (
                self.full_attn_allocator.allocated_pages
                + full_pages
                - self._full_share,
                self.swa_attn_allocator.allocated_pages
                + groups * swa_pages
                - self._swa_share,
            )

        def evict(full_ids: int, swa_ids: int) -> None:
            full_tokens = max(full_ids, 0) * page_size
            swa_tokens = -(-max(swa_ids, 0) // groups) * page_size
            if full_tokens and swa_tokens:
                # Both kinds give: the static pools' own call. One walk, whose
                # full leaves count towards the SWA tokens, and per-kind counts:
                # on the shared list the full pass would stop at the SWA ids its
                # leaves free and leave the SWA walk more to tombstone.
                tree_cache.evict(
                    EvictParams(num_tokens=full_tokens, swa_num_tokens=swa_tokens)
                )
            elif full_tokens or swa_tokens:
                # One kind gives: a prefix of the same walk, stopping once the
                # shared list holds the request.
                tree_cache.evict_for_alloc(
                    EvictParams(num_tokens=full_tokens, swa_num_tokens=swa_tokens)
                )

        # A PD decode's matches cross SWA tombstones: no full KV is dead there.
        compact = (
            self._eviction_policy == "compact"
            and isinstance(tree_cache, UnifiedRadixCache)
            and get_disagg().disaggregation_mode != "decode"
        )
        if compact:
            tree_cache.evict_unmatchable_device_leaves(done)
        # The static pools' eviction, each kind capped at the shortfall: what
        # each kind holds past its share. While the shares tile the pool the
        # shortfall is the sum of the overshoots, so a kind under its share
        # gives nothing and two over it give exactly their overshoots.
        over_full, over_swa = overshoots()
        if compact and over_swa > 0 and not done():
            evict(min(over_full, shortfall()), 0)
            tree_cache.compact_swa(self._compact_retain_tokens, done)
            over_full, over_swa = 0, overshoots()[1]
        if not done():
            evict(min(over_full, shortfall()), min(over_swa, shortfall()))
        # Past the overshoots (the shares overlap by the grouped SWA pages'
        # rounding, or the overshooting kind is locked): the kind nearer the end
        # of its share first, then the other.
        full_held, swa_held = (
            self.full_attn_allocator.allocated_pages + full_pages,
            self.swa_attn_allocator.allocated_pages + groups * swa_pages,
        )
        swa_nearer = swa_held * self._full_share > full_held * self._swa_share
        for kind_is_swa in (swa_nearer, not swa_nearer):
            if done():
                return
            if kind_is_swa and compact:
                tree_cache.compact_swa(self._compact_retain_tokens, done)
                if done():
                    return
            if kind_is_swa:
                evict(0, shortfall())
            else:
                evict(shortfall(), 0)

    def prealloc_fits_assumes_reclaim(self) -> bool:
        return True

    def prealloc_ceiling_fits(self, full_tokens: int, swa_tokens: int) -> bool:
        return self.can_reserve(full_tokens, swa_tokens, empty_pool=True)

    def prealloc_fits(
        self,
        tree_cache,
        full_tokens: int,
        swa_tokens: int,
        *,
        full_budget_tokens: int,
        swa_budget_tokens: int | None = None,
    ) -> bool:
        # Each budget is this side's free plus evictable tokens less the decode's
        # headroom; add that headroom to the demand and price both sides at once.
        full_evictable_tokens = tree_cache.full_evictable_size()
        swa_evictable_tokens = tree_cache.swa_evictable_size()
        full_tokens += (
            self.full_available_size() + full_evictable_tokens - full_budget_tokens
        )
        if swa_budget_tokens is not None:
            swa_tokens += (
                self.swa_available_size() + swa_evictable_tokens - swa_budget_tokens
            )
        return self.can_reserve(
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
        )

    def reclaim_for_prealloc(
        self, tree_cache, full_tokens: int, swa_tokens: int
    ) -> str | None:
        # Both sides draw from the one free list; reclaim pages for both.
        page_size = self.page_size
        full_pages, swa_pages = (
            -(-full_tokens // page_size),
            -(-swa_tokens // page_size),
        )
        need_pages = self._page_ids_needed(full_pages, swa_pages)
        self._evict_shortfall(tree_cache, full_pages, swa_pages)
        free_pages = self._free_page_count()
        if need_pages > free_pages:
            return (
                f"Unified KV eviction insufficient: needed={need_pages} pages, "
                f"free={free_pages} pages"
            )
        return None

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc, out_indices_cpu=None):
        # The parent allocates the full side before the SWA side; on one free
        # list the SWA side could then fail with the full pages already taken.
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
        )
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None
        return super().alloc_decode(
            seq_lens, seq_lens_cpu, last_loc, out_indices_cpu=out_indices_cpu
        )

    def debug_print(self) -> str:
        return (
            f"#free-pages: {self._free_page_count()}, "
            f"#full-pages: {self.full_attn_allocator.allocated_pages}, "
            f"#swa-pages: {self.swa_attn_allocator.allocated_pages}, "
        )


class _SharedPagesRatioSplit(HybridSWAPoolConfigurator):
    """The static ratio split; the unified pool applies its own floor to the
    shared size."""

    @staticmethod
    def validate_swa_pool_size(
        swa_tokens: int, sliding_window_size: Optional[int], page_size: int
    ) -> None:
        return None


class TpuUnifiedSWAPoolConfigurator(MemoryPoolConfigurator):
    """Sizes the unified pool from the static full and sliding-window pools.

    The shared pool gets the bytes the two static pools would get for the same
    flags, and max_total_num_tokens stays their full-attention token count.
    """

    def __init__(self, kvc: KVCacheConfigurator):
        _raise_if_unsupported(kvc)
        # The policy create_memory_pool_configurator picks for a hybrid
        # sliding-window model without the flag.
        self._static = (
            SWAChunkCapPoolConfigurator(kvc)
            if SWAChunkCapPoolConfigurator.is_applicable(kvc)
            else _SharedPagesRatioSplit(kvc)
        )
        self._full_layers_num = len(kvc.layer_info.full_attention_layer_ids)
        self._swa_layers_num = len(kvc.layer_info.swa_attention_layer_ids)
        self._swa_groups = num_swa_groups(
            num_full_layers=self._full_layers_num,
            num_swa_layers=self._swa_layers_num,
        )
        self._sliding_window_size = kvc.sliding_window_size

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._share(
            self._static.calculate_pool_sizes(available_bytes, page_size), page_size
        )

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._share(
            self._static.calculate_pool_sizes_from_max_tokens(
                max_total_num_tokens, page_size
            ),
            page_size,
        )

    def finalize_with_max_running_requests(
        self, config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        return self._static.finalize_with_max_running_requests(config)

    def _share(self, config: MemoryPoolConfig, page_size: int) -> MemoryPoolConfig:
        full_tokens = config.full_max_total_num_tokens
        swa_tokens = config.swa_max_total_num_tokens
        # A shared page holds a row in each of the pool's buffers, whichever
        # side owns it.
        num_buffers = max(
            self._full_layers_num, _cdiv(self._swa_layers_num, self._swa_groups)
        )
        shared_tokens = (
            full_tokens * self._full_layers_num + swa_tokens * self._swa_layers_num
        ) // num_buffers
        align_tokens = page_size * _TPU_SWA_PAGE_ALIGNMENT
        shared_tokens = shared_tokens // align_tokens * align_tokens
        self._validate_shared_pool_size(shared_tokens, page_size)
        logger.info(
            "Use the unified sliding-window KV pool: full_layer_tokens=%d, "
            "swa_layer_tokens=%d, shared_tokens=%d",
            full_tokens,
            swa_tokens,
            shared_tokens,
        )
        return dataclasses.replace(
            config,
            full_max_total_num_tokens=shared_tokens,
            swa_max_total_num_tokens=shared_tokens,
        )

    def _validate_shared_pool_size(self, tokens: int, page_size: int) -> None:
        window = self._sliding_window_size
        if window is None:
            return
        # The SWA admission floor of validate_swa_pool_size, held once by the
        # full side and once per group by the SWA side: below it every request
        # waits forever instead of failing here.
        copies = 1 + self._swa_groups
        min_tokens = copies * (window + page_size)
        if min_tokens >= tokens:
            raise ValueError(
                f"Unified KV pool ({tokens} tokens) cannot hold even one request: "
                f"it needs more than {copies} * (sliding_window_size ({window}) "
                f"+ page_size ({page_size})) = {min_tokens} tokens. Increase the "
                "KV budget (--mem-fraction-static) or --max-total-tokens."
            )
