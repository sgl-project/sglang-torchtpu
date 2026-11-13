# SPDX-License-Identifier: Apache-2.0
"""A host-DRAM tier (HiCache) for the Kimi-K3 programs' unified pools.

A K3_PROGRAM prefix hit needs the prefix's KV pages and a KDA checkpoint at its
end (sglang's unified radix cache with FULL and MAMBA components). Without a
host tier, an evicted prefix is computed again; with it, the radix cache backs
pages and checkpoints up to host memory and loads them back on a hit.

Both live in the same unified pools (``sglang_bridge.UnifiedPoolPlan``): KV page
``p`` is row ``p`` of every pool (the 24 unified pools and the DSpark draft's),
each rank holding its share; KDA checkpoint slot ``s`` is the three rows
``state_row(s) .. state_row(s) + 2`` of every pool (the draft pools' rows there
hold the slot's DSpark ring). One TPU DMA engine (raiden by default) copies rows
of all the pools between the device and its host blocks: host blocks
``0 .. P - 1`` are KV pages, ``P + 3 h .. P + 3 h + 2`` host checkpoint ``h``.
The prefix tails and colocated DP2's B rows are never copied.

The TPU hybrid controller drives both pools with all-pool engine copies;
:class:`K3ProgramWorker` (``sglang_worker``) waits on the load counter before
the program reads a reloaded prefix, as it never runs the target worker's
forward, where sglang waits.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    StackStrategy,
    _evict_mamba_for_device_alloc,
    _stage_local_layer_mapping,
    build_pool_entry,
    register_stack_strategy,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang_tpu.k3_programs.sglang_bridge import KDA_PER_POOL
from sglang_tpu.kv_pool_host import TPU_DMA_LAYOUT, TpuDMAHostKVPool
from sglang_tpu.tpu_dma_hicache import create_tpu_dma_kv_transfer
from sglang_tpu.tpu_dma_hybrid_cache_controller import (
    _KV_PAGE_POOLS,
    TpuDMAHybridCacheController,
)

logger = logging.getLogger(__name__)


class K3PoolView:
    """The unified pools as a TPU DMA engine's fused device pool: pool ``i``
    is layer ``i`` and a row (block) is a page."""

    def __init__(self, pools: list[torch.Tensor], page_size: int, device: Any) -> None:
        self.pools = list(pools)
        self.layer_num = len(self.pools)
        self.start_layer = 0
        self.end_layer = self.layer_num - 1
        self.page_size = int(page_size)
        self.store_dtype = self.pools[0].dtype
        self.device = device
        self.size = int(self.pools[0].shape[0]) * self.page_size

    def get_fused_kv_buffer(self, layer: int) -> torch.Tensor:
        return self.pools[layer]

    @property
    def block_bytes(self) -> int:
        """Bytes of one row across every pool (a host block)."""
        return sum(pool[0].numel() * pool.element_size() for pool in self.pools)


def host_blocks(
    hicache_gb: float,
    hicache_ratio: float,
    device_pages: int,
    checkpoints: int,
    block_bytes: int,
) -> tuple[int, int]:
    """``(host KV pages, host checkpoints)``: ``hicache_gb`` of host blocks a
    rank (else ``hicache_ratio`` times the device KV pages), of which each
    checkpoint takes three."""
    if checkpoints < 1:
        raise ValueError(f"K3 host tier: {checkpoints} host checkpoints")
    if hicache_gb > 0:
        pages = int(hicache_gb * 1e9 // block_bytes) - KDA_PER_POOL * checkpoints
    else:
        pages = int(device_pages * hicache_ratio)
    if pages < 1:
        raise ValueError(
            f"K3 host tier: {hicache_gb} GB holds no KV page beside "
            f"{checkpoints} checkpoints ({block_bytes} bytes a block)"
        )
    return pages, int(checkpoints)


class K3HostKVPool(TpuDMAHostKVPool):
    """Host KV pages over the shared engine's blocks ``0 .. num_pages - 1``:
    page-granular allocation and copies as ``TpuDMAHostKVPool``."""

    def __init__(
        self,
        device_pool: Any,
        view: K3PoolView,
        transfer: Any,
        num_pages: int,
        page_size: int,
    ) -> None:
        # The engine is shared with the checkpoint pool: built by the strategy.
        self.device_pool = device_pool
        self.view = view
        self.pool_label = "k3 kv"
        self.page_size = int(page_size)
        self.layout = TPU_DMA_LAYOUT
        self.pin_memory = False
        self.device = "cpu"
        self.dtype = view.store_dtype
        self.start_layer = view.start_layer
        self.end_layer = view.end_layer
        self.layer_num = view.layer_num
        self.size_per_token = self.get_size_per_token()
        self.page_num = int(num_pages)
        self.size = self.page_num * self.page_size
        self.transfer = transfer
        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self) -> int:
        return self.view.block_bytes // self.page_size


class K3HostStatePool:
    """Host KDA checkpoints over the shared engine's blocks from
    ``block_base``: slot-granular (sglang's Mamba host pool interface, a
    slot a radix node), three blocks each."""

    layout = TPU_DMA_LAYOUT
    page_size = 1
    dcp_size = 1
    pool_label = "k3 kda"
    can_use_write_back_jit = False
    shared_allocation_domain = None
    stores_page_envelope = False
    storage_format_tag = None

    def __init__(
        self,
        device_pool: Any,
        transfer: Any,
        num_slots: int,
        block_base: int,
        block_bytes: int,
    ) -> None:
        self.device_pool = device_pool
        self.transfer = transfer
        self.size = int(num_slots)
        self.block_base = int(block_base)
        self.size_per_token = KDA_PER_POOL * int(block_bytes)
        self.device = "cpu"
        self.lock = threading.RLock()
        self.clear()

    @property
    def logical_size(self) -> int:
        return self.size

    def clear(self) -> None:
        with self.lock:
            self.free_slots = deque(range(self.size))

    def destroy(self) -> None:
        return None

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with self.lock:
            if need_size > len(self.free_slots):
                return None
            slots = [self.free_slots.popleft() for _ in range(need_size)]
        return torch.tensor(slots, dtype=torch.int64)

    def free(self, indices: torch.Tensor) -> int:
        slots = [int(s) for s in torch.as_tensor(indices).reshape(-1).tolist()]
        with self.lock:
            self.free_slots.extend(slots)
        return len(slots)

    def _device_rows(self, slots) -> list[int]:
        plan = self.device_pool.plan
        return [
            plan.state_row(int(s)) + g
            for s in torch.as_tensor(slots).reshape(-1).tolist()
            for g in range(KDA_PER_POOL)
        ]

    def _host_blocks(self, slots) -> list[int]:
        return [
            self.block_base + KDA_PER_POOL * int(h) + g
            for h in torch.as_tensor(slots).reshape(-1).tolist()
            for g in range(KDA_PER_POOL)
        ]

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        return self.transfer.backup(
            self._device_rows(device_indices), self._host_blocks(host_indices)
        )

    def load_all_layer(self, device_pool, host_indices, device_indices):
        return self.transfer.load(
            self._host_blocks(host_indices), self._device_rows(device_indices)
        )

    def load_to_device_per_layer(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "K3 host checkpoints load all pools at once (load_all_layer)"
        )

    def get_data_page(self, index, flat: bool = True):
        raise NotImplementedError("K3 host tier: no L3 storage")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("K3 host tier: no L3 storage")

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        raise NotImplementedError("K3 host tier: no L3 storage")


class K3HybridCacheController(TpuDMAHybridCacheController):
    """The TPU hybrid controller over the K3 KV pages and KDA checkpoints."""

    offloaded_pools = _KV_PAGE_POOLS | {PoolName.MAMBA}


class K3Strategy(StackStrategy):
    """The host stack for a K3_PROGRAM unified radix cache (FULL + MAMBA over
    the unified pools)."""

    def matches(self, kvcache, components) -> bool:
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool

        return (
            isinstance(kvcache, HybridLinearKVPool)
            and isinstance(getattr(kvcache, "full_kv_pool", None), K3UnifiedKVPool)
            and components == {ComponentType.FULL, ComponentType.MAMBA}
        )

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
    ) -> StackBuildResult:
        from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes
        from sglang.srt.runtime_context import get_memory
        from sglang_tpu.envs import envs as tpu_envs

        if storage_backend is not None:
            raise NotImplementedError("K3 host tier: no L3 storage backend")
        memory = get_memory()
        kv_pool = kvcache.full_kv_pool
        req_pool = params.req_to_token_pool
        state_pool = req_pool.mamba_pool
        view = K3PoolView(kv_pool.raw_pools(), params.page_size, kv_pool.device)
        pages, slots = host_blocks(
            float(memory.hicache_size or 0),
            float(memory.hicache_ratio or 0),
            kv_pool.plan.kv_blocks,
            tpu_envs.SGLANG_TPU_K3_PROGRAM_HICACHE_CHECKPOINTS.get(),
            view.block_bytes,
        )
        blocks = pages + KDA_PER_POOL * slots
        requested = blocks * view.block_bytes
        if requested > host_memory_budget_bytes(requested):
            raise ValueError(
                f"K3 host tier: {requested / 1e9:.1f} GB of host blocks exceed this "
                "rank's host memory budget"
            )
        transfer = create_tpu_dma_kv_transfer(memory.hicache_io_backend, view, blocks)
        kv_host = K3HostKVPool(kv_pool, view, transfer, pages, params.page_size)
        state_host = K3HostStatePool(
            state_pool, transfer, slots, pages, view.block_bytes
        )
        full_mapping = _stage_local_layer_mapping(
            kvcache.full_attention_layer_id_mapping, kvcache.start_layer
        )
        mamba_mapping = _stage_local_layer_mapping(
            req_pool.mamba_map, kvcache.start_layer
        )
        transfer_layer_id_max = max(full_mapping.keys() | mamba_mapping.keys()) + 1
        allocator = req_pool.mamba_allocator
        group = HostPoolGroup(
            [
                build_pool_entry(
                    name=PoolName.KV,
                    host_pool=kv_host,
                    device_pool=kv_pool,
                    layer_mapping=full_mapping,
                    transfer_layer_id_max=transfer_layer_id_max,
                    is_anchor=True,
                ),
                build_pool_entry(
                    name=PoolName.MAMBA,
                    host_pool=state_host,
                    device_pool=state_pool,
                    layer_mapping=mamba_mapping,
                    transfer_layer_id_max=transfer_layer_id_max,
                    host_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
                    device_evict_fn=lambda n: _evict_mamba_for_device_alloc(cache, n),
                    device_alloc_fn=allocator.alloc,
                    device_free_fn=allocator.free,
                ),
            ]
        )
        controller = K3HybridCacheController(
            params.token_to_kv_pool_allocator,
            group,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            pp_group=params.pp_cache_group,
            write_policy=memory.hicache_write_policy,
            io_backend=memory.hicache_io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            transfer_layer_id_max=transfer_layer_id_max,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=memory.hicache_host_memory_mode,
        )
        logger.info(
            "K3 host tier: %d KV pages and %d KDA checkpoints in %.1f GB of host blocks",
            pages,
            slots,
            requested / 1e9,
        )
        return StackBuildResult(
            host_pool_group=group,
            cache_controller=controller,
            component_host_pools={
                ComponentType.FULL: kv_host,
                ComponentType.MAMBA: state_host,
            },
            register_req_to_token_counter=True,
            pools_desc="K3 KV + KDA checkpoints",
            # Both pools above; HybridLinearKVPool's declarations name sglang's.
            pool_declarations=(),
        )


_INSTALLED: list = []


def install() -> None:
    """Put :class:`K3Strategy` ahead of the built-in strategies (idempotent)."""
    if not _INSTALLED:
        register_stack_strategy(K3Strategy())
        _INSTALLED.append(True)
