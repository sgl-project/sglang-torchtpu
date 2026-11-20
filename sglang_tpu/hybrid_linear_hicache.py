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

from typing import Any

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
    with_packed_draft_layer_mapping,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_memory
from sglang_tpu.tpu_dma_hicache import is_tpu_dma_hicache_backend
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

_is_installed = False


class TpuDmaMambaStrategy(hybrid_pool_assembler.MambaStrategy):
    """KV + recurrent-state host stack of a ``HybridLinearKVPool`` over a TPU mamba
    pool, under a DMA ``--hicache-io-backend``."""

    cache_controller_cls = TpuDMAHybridCacheController

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
    ) -> hybrid_pool_assembler.StackBuildResult:
        """The upstream stack captures buffer infos from the device pools to
        build CUDA host pools and sidecars; a DMA host page holds every buffer
        of a TPU page as it is, so the TPU pools build from the device pools."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if getattr(params, "mtp_draft_mamba_pools", ()):
            raise NotImplementedError(
                "MTP draft recurrent state offload is not supported on the TPU "
                "DMA HiCache path."
            )
        kv_pool = kvcache.full_kv_pool
        state = params.token_to_kv_pool_allocator.state
        mamba_pool = state.pool
        start_layer = kvcache.start_layer
        full_layer_mapping = hybrid_pool_assembler._stage_local_layer_mapping(
            kvcache.full_attention_layer_id_mapping, start_layer
        )
        mamba_layer_mapping = {
            layer - start_layer: index
            for index, layer in enumerate(mamba_pool.mamba_layer_ids)
        }
        transfer_layer_id_max = (
            max(full_layer_mapping.keys() | mamba_layer_mapping.keys()) + 1
        )
        drafts = tuple(
            pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool
            for pool in params.mtp_draft_device_pools
        )
        memory = get_memory()
        kv_host_size, mamba_host_size = None, 0
        if memory.hicache_size > 0:
            kv_host_size, mamba_host_size = hybrid_pool_assembler._split_hicache_size(
                memory.hicache_size, (kv_pool, mamba_pool)
            )
        kv_host_pool = self.kv_host_pool_fn(
            kv_pool=kv_pool,
            page_size=params.page_size,
            host_size=kv_host_size,
            mtp_draft_device_pools=drafts,
        )
        mamba_host_pool = self.mamba_host_pool_fn(
            mamba_pool=mamba_pool, host_size=mamba_host_size
        )
        slots = params.token_to_kv_pool_allocator.state_slots
        host_pool_group = HostPoolGroup(
            [
                hybrid_pool_assembler.build_pool_entry(
                    name=PoolName.KV,
                    host_pool=kv_host_pool,
                    device_pool=kv_pool,
                    layer_mapping=with_packed_draft_layer_mapping(
                        full_layer_mapping,
                        transfer_layer_start=transfer_layer_id_max,
                        target_device_layer_num=kv_pool.layer_num,
                        draft_layer_num=len(drafts),
                    ),
                    transfer_layer_id_max=transfer_layer_id_max + len(drafts),
                    is_anchor=True,
                    packed_draft_device_pools=drafts,
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
                    device_alloc_fn=slots.alloc,
                    device_free_fn=slots.free,
                ),
            ]
        )
        cache_controller = hybrid_pool_assembler._build_cache_controller(
            params=params,
            host_pool_group=host_pool_group,
            transfer_layer_id_max=transfer_layer_id_max,
            cache_controller_cls=self.cache_controller_cls,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return hybrid_pool_assembler.StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: kv_host_pool,
                ComponentType.MAMBA: mamba_host_pool,
            },
            device_pool_names=(PoolName.KV,),
            loads_state_layer_wise=True,
            pools_desc="KV + MAMBA",
        )

    def kv_host_pool_fn(
        self,
        *,
        kv_pool: Any,
        page_size: int,
        host_size: float | None,
        mtp_draft_device_pools: tuple,
    ) -> Any:
        from sglang_tpu.kv_pool_host import tpu_dma_host_kv_pool_cls

        memory = get_memory()
        return tpu_dma_host_kv_pool_cls(memory.hicache_io_backend)(
            kv_pool,
            memory.hicache_ratio,
            memory.hicache_size if host_size is None else host_size,
            page_size,
            memory.hicache_mem_layout,
            mtp_draft_device_pools=mtp_draft_device_pools,
        )

    def mamba_host_pool_fn(self, *, mamba_pool: Any, host_size: float) -> Any:
        from sglang_tpu.kv_pool_host import tpu_dma_mamba_pool_host_cls

        memory = get_memory()
        return tpu_dma_mamba_pool_host_cls(memory.hicache_io_backend)(
            mamba_pool, memory.hicache_ratio, host_size
        )


def install() -> None:
    """Register the strategy ahead of the built-in ones, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    hybrid_pool_assembler.register_stack_strategy(TpuDmaMambaStrategy())
