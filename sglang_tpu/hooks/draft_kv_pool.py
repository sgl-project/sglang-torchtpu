# SPDX-License-Identifier: Apache-2.0
"""The draft runner's MHA KV pool follows the draft's own attention backend.

draft_mha_pool_class, draft_cell_size_per_token
  Why: KVCacheConfigurator._tpu_mha_pool_class asks the platform, which keys the
       pool class on the process attention backend. A DFlash-family draft on its
       own backend (a dense DSpark draft on pallas_tpu_rpa_v3 beside a
       pallas_tpu_mla Kimi-K3 target) then gets the generic MHA pool instead of
       the fused layout its RPA backend and the draft KV writer address. The
       target's KV budget likewise reserves the generic K + V bytes for the
       draft, while the fused pool pads head_dim to 128 lanes and the KV-head
       pair to the dtype packing (2x for the K3 draft's head_dim 64).
  How: AROUND KVCacheConfigurator._tpu_mha_pool_class: a draft runner with a
       resolved draft attention backend takes that backend's pool class.
       AROUND dflash_utils.dflash_draft_cell_size_per_token: when that pool is
       the fused layout, reserve its padded bytes per token.
  Upstream PR or issue: not filed yet.
  Remove when: the configurator asks the platform for the pool class, and its
       bytes per token, of the runner's own attention backend.

derive_unified_draft_pool_sizes, attach_unified_draft_kv_pool
  Why: TpuUnifiedSWATokenToKVPoolAllocator hands out logical slot IDs in
       [0, size_full) and allocates M_T physical pages per SWA logical page,
       so a speculative draft paired with it must size its Full sub-pool to
       size_full, its compact SWA sub-pool to size_full // M_T, and bind its
       SWA index mapping to the unified allocator's _DraftSwaSidecar.
  How: AROUND KVCacheConfigurator._derive_pool_sizes replaces the draft's
       MemoryPoolConfig token capacities via allocator.derive_draft_pool_sizes;
       AROUND KVCacheConfigurator._build_token_to_kv_pool_allocator delegates
       draft pool attachment to allocator.attach_draft_kv_pool.
  Upstream PR or issue: sgl-project/sglang-torchtpu#2250.
  Remove when: BaseTokenToKVPoolAllocator declares draft pool sizing and
       attachment hooks upstream.

build_unified_hicache_draft_plan, bind_unified_swa_draft_host_pool, price_unified_hicache_draft_bytes
  Why: TpuUnifiedSWAKVPool shares one buffer across Full and M_T-group SWA pages
       without sub-pool attributes, so multi-runner MTP HiCache offload must be
       rejected, compact DRAFT_SWA host/device pages require TpuDMAHostKVPool
       page translators bound after build_swa_draft_pools, and HiCache
       auto-sizing must scale SWA draft bytes by target SWA logical capacity.
  How: AROUND BaseSpecWorker._build_hicache_draft_plan rejects multi-runner MTP
       sidecars on TpuUnifiedSWAKVPool; AFTER build_swa_draft_pools calls
       target_swa_host_pool.bind_draft_swa_sidecar; AROUND
       hicache_auto_size._draft_bytes prices unified target sidecars against
       target Full or SWA logical capacity.
  Upstream PR or issue: sgl-project/sglang-torchtpu#2250.
  Remove when: BaseSWAKVPool and BaseKVPoolHost expose first-class HiCache
       sidecar sizing and page-translator binding protocols upstream.
"""

from __future__ import annotations


def _draft_pool_cls(backend: str | None):
    if backend is None:
        return None
    from sglang_tpu.tpu_platform import _mha_kv_pool_cls

    return _mha_kv_pool_cls(backend)


def draft_mha_pool_class(original, configurator):
    pool_cls = _draft_pool_cls(
        configurator.speculative_draft_attention_backend
        if configurator.is_draft_worker
        else None
    )
    return original(configurator) if pool_cls is None else pool_cls


def draft_cell_size_per_token(
    original,
    *,
    draft_model_config,
    draft_num_layers: int,
    draft_kv_cache_dtype,
    tp_size: int,
) -> int:
    generic = original(
        draft_model_config=draft_model_config,
        draft_num_layers=draft_num_layers,
        draft_kv_cache_dtype=draft_kv_cache_dtype,
        tp_size=tp_size,
    )
    from sglang.srt.runtime_context import get_spec
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    pool_cls = _draft_pool_cls(get_spec().speculative_draft_attention_backend)
    if (
        pool_cls is None
        or not issubclass(pool_cls, TpuFusedMHATokenToKVPool)
        or draft_num_layers <= 0
    ):
        return generic
    per_layer = pool_cls.kv_bytes_per_token_per_layer(
        num_kv_heads=draft_model_config.get_num_kv_heads(tp_size),
        head_dim=draft_model_config.head_dim,
        v_head_dim=draft_model_config.v_head_dim,
        store_dtype=draft_kv_cache_dtype,
    )
    # Never below the generic count: the fused pool only adds padding.
    return max(generic, int(per_layer) * int(draft_num_layers))


def derive_unified_draft_pool_sizes(original, configurator, *, config):
    alloc = getattr(configurator, "token_to_kv_pool_allocator", None)
    if getattr(configurator, "is_draft_worker", False) and alloc is not None:
        from sglang_tpu.unified_kv_pool import TpuUnifiedSWATokenToKVPoolAllocator

        if isinstance(alloc, TpuUnifiedSWATokenToKVPoolAllocator):
            import dataclasses
            import types

            max_tok, full_tok, swa_tok = alloc.derive_draft_pool_sizes(
                configurator,
                max_total_num_tokens=config.max_total_num_tokens,
                full_max_total_num_tokens=config.full_max_total_num_tokens,
                swa_max_total_num_tokens=config.swa_max_total_num_tokens,
            )
            updates = {
                "max_total_num_tokens": max_tok,
                "full_max_total_num_tokens": full_tok,
                "swa_max_total_num_tokens": swa_tok,
            }
            config = (
                dataclasses.replace(config, **updates)
                if dataclasses.is_dataclass(config)
                else types.SimpleNamespace(**{**vars(config), **updates})
            )
    return original(configurator, config=config)


def attach_unified_draft_kv_pool(original, configurator, *args, **kwargs):
    alloc = kwargs.get("token_to_kv_pool_allocator")
    if getattr(configurator, "is_draft_worker", False) and alloc is not None:
        from sglang_tpu.unified_kv_pool import TpuUnifiedSWATokenToKVPoolAllocator

        if isinstance(alloc, TpuUnifiedSWATokenToKVPoolAllocator):
            alloc.attach_draft_kv_pool(configurator, kwargs["token_to_kv_pool"])
            return alloc
    return original(configurator, *args, **kwargs)


def build_unified_hicache_draft_plan(original, worker):
    plan = original(worker)
    if plan.mode.value == "sidecar":
        target_runner = getattr(
            getattr(worker, "target_worker", None), "model_runner", None
        )
        target_pool = getattr(target_runner, "token_to_kv_pool", None)
        if target_pool is not None:
            from sglang_tpu.unified_kv_pool import TpuUnifiedSWAKVPool

            if (
                isinstance(target_pool, TpuUnifiedSWAKVPool)
                and len(worker._draft_model_runners()) > 1
            ):
                raise NotImplementedError(
                    "Multi-runner MTP HiCache offload is not supported on unified SWA KV pools."
                )
    return plan


def price_unified_hicache_draft_bytes(original, target, draft) -> int:
    from sglang_tpu.unified_kv_pool import TpuUnifiedSWAKVPool

    if not isinstance(target, TpuUnifiedSWAKVPool):
        return original(target, draft)
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.hicache_auto_size import _pool_bytes

    if isinstance(draft, BaseSWAKVPool):
        draft_swa = getattr(draft, "swa_kv_pool", None)
        if draft_swa is None or draft_swa.layer_num == 0 or draft_swa.size <= 0:
            return 0
        swa_pages = (target.size // target.page_size) // target.layout.num_swa_groups
        target_swa_tokens = swa_pages * target.page_size
        return _pool_bytes(draft_swa) * target_swa_tokens // draft_swa.size
    if getattr(draft, "layer_num", 0) == 0 or draft.size <= 0:
        return 0
    return _pool_bytes(draft) * target.size // draft.size


def bind_unified_swa_draft_host_pool(
    result, *, draft_kv_pool, tree_cache, target_window_host
):
    _, entries = result
    if entries:
        from sglang_tpu.kv_pool_host import TpuDMAUnifiedHostKVPool

        # The target window's host pool, which the draft SWA sidecar follows.
        swa_entry = tree_cache.cache_controller.mem_pool_host.entry_map.get(
            target_window_host
        )
        if swa_entry is not None and isinstance(
            swa_entry.host_pool, TpuDMAUnifiedHostKVPool
        ):
            swa_entry.host_pool.bind_draft_swa_sidecar(entries[0].host_pool)
    return result
