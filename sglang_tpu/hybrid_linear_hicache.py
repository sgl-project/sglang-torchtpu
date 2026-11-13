# SPDX-License-Identifier: Apache-2.0
"""HiCache host stack for hybrid linear-attention models (GDN/KDA) on a TPU DMA engine.

The upstream mamba stack builds the CUDA host pools (``MLATokenToKVPoolHost`` /
``MHATokenToKVPoolHost`` and ``MambaPoolHost``), which read stacked GPU-layout
tensors. TPU keeps per-layer kernel-layout tensors (and, for sparse MLA, several
page buffers per layer), which only a DMA engine copies as they are. This
strategy, registered ahead of the built-in ones, builds the same pool entries
with TPU DMA host pools and ``TpuDMAHybridCacheController``. A hybrid SWA model
with mamba-pool state gets the same over its KV + SWA + state stack, with a
speculative draft's KV packed onto the KV and SWA pages.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
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

    def kv_host_pool_fn(self, **kwargs: Any) -> Any:
        return _dma_kv_host_pool(**kwargs)

    def mamba_host_pool_fn(self, *, mamba_pool: Any, host_size: float) -> Any:
        return _dma_mamba_host_pool(mamba_pool=mamba_pool, host_size=host_size)


class TpuDmaMambaSwaStrategy(hybrid_pool_assembler._MambaSwaStrategy):
    """KV + SWA + state host stack of an ``SWAKVPool`` model, under a DMA
    ``--hicache-io-backend``; every TPU mamba pool is a ``TpuMambaPool``."""

    cache_controller_cls = TpuDMAHybridCacheController

    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool:
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            is_tpu_dma_hicache_backend(get_memory().hicache_io_backend)
            and isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components
            == {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA}
        )

    def kv_host_pool_fn(self, **kwargs: Any) -> Any:
        # The engine copies pages as they are, so draft rows need not match target rows.
        return _dma_kv_host_pool(**kwargs)

    def mamba_host_pool_fn(self, *, mamba_pool: Any, host_size: float) -> Any:
        return _dma_mamba_host_pool(mamba_pool=mamba_pool, host_size=host_size)


def packs_draft_kv(target_runner: Any, draft_pools: tuple) -> bool:
    """Whether the KV + SWA + state stack carries these draft pools packed.

    A draft KV pool indexed by the target's cache locations (its window layers
    by the target's SWA slots) rides the target's host pages: one engine call
    per pool moves target and draft KV, and a host hit restores both.
    """
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    kv_pool = target_runner.token_to_kv_pool
    return (
        is_tpu_dma_hicache_backend(get_memory().hicache_io_backend)
        and isinstance(kv_pool, SWAKVPool)
        and not isinstance(kv_pool, DeepSeekV4TokenToKVPool)
        and getattr(target_runner.req_to_token_pool, "mamba_pool", None) is not None
        and len(draft_pools) == 1
        and isinstance(draft_pools[0], (SWAKVPool, TpuFusedMHATokenToKVPool))
    )


def _dma_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    host_size: float | None,
    mtp_draft_device_pools: tuple,
    pool_label: str = "kv",
) -> Any:
    from sglang_tpu.kv_pool_host import tpu_dma_host_kv_pool_cls

    memory = get_memory()
    return tpu_dma_host_kv_pool_cls(memory.hicache_io_backend)(
        kv_pool,
        memory.hicache_ratio,
        memory.hicache_size if host_size is None else host_size,
        page_size,
        memory.hicache_mem_layout,
        pool_label=pool_label,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )


def _dma_mamba_host_pool(*, mamba_pool: Any, host_size: float) -> Any:
    from sglang_tpu.kv_pool_host import tpu_dma_mamba_pool_host_cls

    memory = get_memory()
    return tpu_dma_mamba_pool_host_cls(memory.hicache_io_backend)(
        mamba_pool, memory.hicache_ratio, host_size
    )


def install() -> None:
    """Register the strategies ahead of the built-in ones, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    hybrid_pool_assembler.register_stack_strategy(TpuDmaMambaStrategy())
    hybrid_pool_assembler.register_stack_strategy(TpuDmaMambaSwaStrategy())
