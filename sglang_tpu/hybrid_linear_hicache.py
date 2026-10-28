# SPDX-License-Identifier: Apache-2.0
"""HiCache host stack for hybrid linear-attention models (GDN/KDA) on a TPU DMA engine.

The upstream mamba stack builds the CUDA host pools (``MLATokenToKVPoolHost`` /
``MHATokenToKVPoolHost`` and ``MambaPoolHost``), which read stacked GPU-layout
tensors. TPU keeps per-layer kernel-layout tensors (and, for sparse MLA, several
page buffers per layer), which only a DMA engine copies as they are. This
strategy, registered ahead of the built-in ones, builds the same pool entries
with TPU DMA host pools and ``TpuDMAHybridCacheController``.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_memory
from sglang_tpu.tpu_dma_hicache import is_tpu_dma_hicache_backend

_is_installed = False


class TpuDmaMambaStrategy(hybrid_pool_assembler.StackStrategy):
    """KV + recurrent-state host stack of a ``HybridLinearKVPool`` over a TPU mamba
    pool, under a DMA ``--hicache-io-backend``."""

    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool:
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
        from sglang_tpu.mamba_pool import TpuMambaPool

        return (
            is_tpu_dma_hicache_backend(get_memory().hicache_io_backend)
            and isinstance(kvcache, HybridLinearKVPool)
            and not isinstance(kvcache, QSATokenToKVPool)
            and isinstance(kvcache.mamba_pool, TpuMambaPool)
            and components == {ComponentType.FULL, ComponentType.MAMBA}
        )

    def build(
        self,
        *,
        cache: Any,
        kvcache: Any,
        params: Any,
        server_args: Any,
        load_cache_event: Any,
        storage_backend: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        enable_storage_metrics: bool = False,
    ) -> hybrid_pool_assembler.StackBuildResult:
        from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
            with_packed_draft_layer_mapping,
        )
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang_tpu.kv_pool_host import (
            tpu_dma_host_kv_pool_cls,
            tpu_dma_mamba_pool_host_cls,
        )
        from sglang_tpu.tpu_dma_hybrid_cache_controller import (
            TpuDMAHybridCacheController,
        )

        memory = get_memory()
        io_backend = memory.hicache_io_backend
        kv_pool, mamba_pool = kvcache.full_kv_pool, params.req_to_token_pool.mamba_pool
        full_layer_mapping = hybrid_pool_assembler._stage_local_layer_mapping(
            kvcache.full_attention_layer_id_mapping, kvcache.start_layer
        )
        mamba_layer_mapping = hybrid_pool_assembler._stage_local_layer_mapping(
            params.req_to_token_pool.mamba_map, kvcache.start_layer
        )
        transfer_layer_id_max = (
            max(full_layer_mapping.keys() | mamba_layer_mapping.keys()) + 1
        )
        # MTP draft KV is packed into the KV entry, as in the upstream mamba stack.
        draft_pools = tuple(
            pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool
            for pool in params.mtp_draft_device_pools
        )
        if draft_pools:
            full_layer_mapping = with_packed_draft_layer_mapping(
                full_layer_mapping,
                transfer_layer_start=transfer_layer_id_max,
                target_device_layer_num=kv_pool.layer_num,
                draft_layer_num=len(draft_pools),
            )
        kv_host_size, mamba_host_size = memory.hicache_size, 0
        if memory.hicache_size > 0:
            kv_host_size, mamba_host_size = hybrid_pool_assembler._split_hicache_size(
                memory.hicache_size, (kv_pool, mamba_pool)
            )
        kv_host_pool = tpu_dma_host_kv_pool_cls(io_backend)(
            kv_pool,
            memory.hicache_ratio,
            kv_host_size,
            params.page_size,
            memory.hicache_mem_layout,
            mtp_draft_device_pools=draft_pools,
        )
        mamba_host_pool = tpu_dma_mamba_pool_host_cls(io_backend)(
            mamba_pool, memory.hicache_ratio, mamba_host_size
        )
        mamba_allocator = params.req_to_token_pool.mamba_allocator
        host_pool_group = HostPoolGroup(
            [
                hybrid_pool_assembler.build_pool_entry(
                    name=PoolName.KV,
                    host_pool=kv_host_pool,
                    device_pool=kv_pool,
                    layer_mapping=full_layer_mapping,
                    transfer_layer_id_max=transfer_layer_id_max + len(draft_pools),
                    is_anchor=True,
                    packed_draft_device_pools=draft_pools,
                ),
                hybrid_pool_assembler.build_pool_entry(
                    name=PoolName.MAMBA,
                    host_pool=mamba_host_pool,
                    device_pool=mamba_pool,
                    layer_mapping=mamba_layer_mapping,
                    transfer_layer_id_max=transfer_layer_id_max,
                    host_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
                    device_evict_fn=lambda n: (
                        hybrid_pool_assembler._evict_mamba_for_device_alloc(cache, n)
                    ),
                    device_alloc_fn=mamba_allocator.alloc,
                    device_free_fn=mamba_allocator.free,
                ),
            ]
        )
        cache_controller = TpuDMAHybridCacheController(
            params.token_to_kv_pool_allocator,
            host_pool_group,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            pp_group=params.pp_cache_group,
            write_policy=memory.hicache_write_policy,
            io_backend=io_backend,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            transfer_layer_id_max=transfer_layer_id_max,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=memory.hicache_host_memory_mode,
        )
        return hybrid_pool_assembler.StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: kv_host_pool,
                ComponentType.MAMBA: mamba_host_pool,
            },
            register_req_to_token_counter=True,
            pools_desc="KV + MAMBA (TPU DMA)",
        )


def install() -> None:
    """Register the strategy ahead of the built-in ones, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    hybrid_pool_assembler.register_stack_strategy(TpuDmaMambaStrategy())
