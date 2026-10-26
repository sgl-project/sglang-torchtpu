"""PD decode prefix cache with host-RAM offload for sliding-window models.

Upstream keeps the PD decode prefix cache of sliding-window models such as
GPT-OSS in HBM only. On TPU with TPU DMA HiCache, this module also offloads it
to host RAM:

- Full-attention KV goes to RAM. A request's own restore loads only full KV,
  and prefill sends each request's SWA window fresh.
- Decode cache hits stop where the SWA window starts, so prefill always sends
  the window and the new tokens.
- With SGLANG_TPU_PD_DECODE_SWA_HOST_WINDOWS > 0, RAM also keeps the last SWA
  window of each cached path for host victim storage. Prefix recall is added
  separately.
- With a draft (EAGLE3, DFlash, or DSpark on the experimental worker), the draft
  KV goes to RAM together with the full-attention KV.
- With SGLANG_TPU_ENABLE_PD_PREFILL_SWA_PATH_END_BACKUP, a PD prefill's RAM also
  keeps only the last SWA window of each path.

For every raiden PD decode prefix cache, sliding-window or not, a request that
reuses pages of an unfinished RAM restore waits for it. Restores that load only
auxiliary state (no KV pages) are not tracked.

``install()`` wraps a few upstream functions and registers a host stack
strategy. Except for the restore hold, each behaves exactly like upstream unless
the decode cache is one of these caches.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import weakref
from typing import Any, Callable, Iterator, Optional

import torch

from sglang.srt.disaggregation import decode, decode_hicache_mixin
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.hicache_auto_size import auto_size_hicache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.registry import TreeCacheBuildContext
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    select_tree_core_backend,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_disagg, get_memory
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode, HiCacheDraftPlan
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.tpu_dma_hicache import (
    is_tpu_dma_hicache_backend,
    tpu_dma_hicache_backends,
)
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController

logger = logging.getLogger(__name__)

# Set by install(), which runs once per scheduler process.
_is_installed = False
# Decode caches whose requests restore only full KV from RAM, added by the
# host stack builder.
_full_only_host_caches: weakref.WeakSet[UnifiedRadixCache] = weakref.WeakSet()
# The subset that also keeps the last SWA window of each path in RAM.
_window_host_caches: weakref.WeakSet[UnifiedRadixCache] = weakref.WeakSet()
# PD prefill caches whose RAM keeps only the last SWA window of each path.
_path_end_prefill_caches: weakref.WeakSet[UnifiedRadixCache] = weakref.WeakSet()
# Set only while the prefix offer restores a window (swa_window_restore()). The
# SWA load-back and the strict match run only then: a request's own restore
# anchors at its window start, usually an SWA tombstone, which LOAD_BACK asserts on.
_swa_window_restore = False
# Raiden PD decode prefix caches whose requests wait for unfinished restores.
_restore_gated_caches: weakref.WeakSet[UnifiedRadixCache] = weakref.WeakSet()
# Owned by _build_kv_cache for the duration of one cache build. While set, the
# tree cache factory enables HiCache and _FullOnlySwaStrategy matches.
_is_building_full_only_host_cache = False
# The draft HiCache plan of that build, None without speculative decoding.
_building_draft_plan: Optional[HiCacheDraftPlan] = None
# Device pages of each full KV load a restore-gated cache's controller issued,
# by anchor node id. An entry is in flight while the anchor is in
# ongoing_load_back.
_load_device_pages: weakref.WeakKeyDictionary[Any, dict[int, torch.Tensor]] = (
    weakref.WeakKeyDictionary()
)


def uses_full_only_host_cache(tree_cache: Any) -> bool:
    """Whether ``tree_cache`` is a decode cache whose requests restore only full KV."""
    return tree_cache in _full_only_host_caches


def uses_window_host_cache(tree_cache: Any) -> bool:
    """Whether ``tree_cache`` also keeps the last SWA window of each path in RAM."""
    return tree_cache in _window_host_caches


@contextlib.contextmanager
def swa_window_restore() -> Iterator[None]:
    """Match and load back SWA windows as upstream does, for this block only."""
    global _swa_window_restore
    assert not _swa_window_restore, "nested SWA window restore"
    _swa_window_restore = True
    try:
        yield
    finally:
        _swa_window_restore = False


@contextlib.contextmanager
def _full_only_host_cache_build(
    draft_plan: Optional[HiCacheDraftPlan],
) -> Iterator[None]:
    global _is_building_full_only_host_cache, _building_draft_plan
    assert not _is_building_full_only_host_cache, "nested full-only cache build"
    _is_building_full_only_host_cache = True
    _building_draft_plan = draft_plan
    try:
        yield
    finally:
        _is_building_full_only_host_cache = False
        _building_draft_plan = None


def _bigram_boundary(tree_cache: UnifiedRadixCache) -> int:
    """Extra token a prefix match needs to cover its last page under EAGLE.

    EAGLE keys the cache by token pairs, so N tokens make only N - 1 keys.
    """
    return 1 if tree_cache.tree_core.is_eagle else 0


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------


def _should_use_full_only_host_cache(
    *, enable_hierarchical_cache: bool, is_hybrid_swa: bool
) -> bool:
    disagg = get_disagg()
    return (
        enable_hierarchical_cache
        and is_hybrid_swa
        and disagg.disaggregation_mode == "decode"
        and disagg.disaggregation_decode_enable_radix_cache
    )


def _validate_full_only_host_cache_config(
    *,
    page_size: int,
    spec_algorithm: Any,
    hicache_draft_plan: Optional[HiCacheDraftPlan],
) -> None:
    """Raise for settings the full-only cache does not support."""
    memory = get_memory()
    requirements = (
        (
            is_tpu_dma_hicache_backend(memory.hicache_io_backend),
            "--hicache-io-backend must be one of "
            + ", ".join(tpu_dma_hicache_backends()),
        ),
        (
            memory.hicache_host_memory_mode == "cache",
            "--hicache-host-memory-mode must be cache to keep reusable host KV",
        ),
        (
            memory.hicache_storage_backend is None,
            "--hicache-storage-backend (L3) restores are not supported",
        ),
        (page_size > 1, "--page-size must be > 1 for the page-aligned SWA tail"),
        (
            spec_algorithm.supports_pd_decode_swa_hicache(),
            f"speculative algorithm {spec_algorithm.name} is not supported in this mode",
        ),
        # EAGLE3 and DFlash-family draft KV reaches RAM through upstream's draft sidecar pool.
        (
            not (spec_algorithm.is_eagle3() or spec_algorithm.is_dflash_family())
            or (
                hicache_draft_plan is not None
                and hicache_draft_plan.mode == HiCacheDraftMode.SIDECAR
            ),
            "EAGLE3, DFlash and DSpark draft KV must use the HiCache draft sidecar",
        ),
        # A packed draft lives inside the full KV pool, which this build does
        # not lay out for.
        (
            hicache_draft_plan is None
            or hicache_draft_plan.mode != HiCacheDraftMode.PACKED,
            "A packed draft KV pool is not supported",
        ),
        # Upstream auto-sizes the host pool for it, and so would this build.
        (
            get_disagg().disaggregation_decode_retraction_backup != "host_pool",
            "--disaggregation-decode-retraction-backup host_pool is not supported",
        ),
    )
    for is_met, requirement in requirements:
        if not is_met:
            raise ValueError(
                "PD decode radix cache + HiCache for sliding-window models: "
                f"{requirement}."
            )


def _build_kv_cache(
    upstream: Callable,
    *,
    enable_hierarchical_cache: bool,
    tp_worker: Any,
    page_size: int,
    spec_algorithm: Any,
    hicache_draft_plan: Optional[HiCacheDraftPlan] = None,
    **other_kwargs,
):
    """Wraps ``build_kv_cache``: build the decode cache with a full-only RAM pool.

    Upstream rejects HiCache for this setup, so the flag goes in as False and
    ``_create_tree_cache`` turns HiCache back on, draft KV included.
    ``test_upstream_flag_uses_are_known`` fails if upstream starts reading the
    flag somewhere new.
    """
    kwargs = dict(
        tp_worker=tp_worker,
        page_size=page_size,
        spec_algorithm=spec_algorithm,
        hicache_draft_plan=hicache_draft_plan,
        **other_kwargs,
    )
    if not _should_use_full_only_host_cache(
        enable_hierarchical_cache=enable_hierarchical_cache,
        is_hybrid_swa=tp_worker.is_hybrid_swa,
    ):
        return upstream(enable_hierarchical_cache=enable_hierarchical_cache, **kwargs)
    _validate_full_only_host_cache_config(
        page_size=page_size,
        spec_algorithm=spec_algorithm,
        hicache_draft_plan=hicache_draft_plan,
    )
    logger.info(
        "PD decode SWA HiCache: requests restore full-attention KV from RAM, "
        "their SWA window comes from prefill. Host SWA windows: %d.",
        tpu_envs.SGLANG_TPU_PD_DECODE_SWA_HOST_WINDOWS.get(),
    )
    # Upstream rejects decode SWA radix + HiCache before it builds the tree.
    # _create_tree_cache turns HiCache back on for this build.
    with _full_only_host_cache_build(hicache_draft_plan):
        return upstream(enable_hierarchical_cache=False, **kwargs)


def _create_tree_cache(upstream: Callable, ctx: TreeCacheBuildContext):
    """Wraps ``create_tree_cache``: turn HiCache back on for a full-only build,
    draft KV included, and register raiden PD decode caches for the restore hold."""
    if not _is_building_full_only_host_cache:
        cache = upstream(ctx)
        _gate_decode_restores(cache)
        _maybe_enable_prefill_path_end_backup(cache)
        return cache
    draft_plan = _building_draft_plan
    # The match policy below is implemented for the Python tree core only.
    # Upstream defaults to the Rust core, which the registry resolves back to
    # Python for a TPU allocator; check what it resolves to, not the setting.
    tree_core = (
        select_tree_core_backend(ctx.params)
        if ctx.params is not None
        else envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get()
    )
    if tree_core != "python":
        raise ValueError(
            "PD decode radix cache + HiCache for sliding-window models: "
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND must resolve to python, "
            f"got {tree_core!r}."
        )
    # build_kv_cache skips auto-sizing and draft registration with the flag
    # off. Sizing against the full + SWA device bytes leaves the full-only
    # host pool within budget.
    with auto_size_hicache(ctx.params, draft_plan, enabled=True):
        cache = upstream(dataclasses.replace(ctx, enable_hierarchical_cache=True))
        if not uses_full_only_host_cache(cache):
            raise ValueError(
                "PD decode SWA HiCache needs the unified radix tree with a "
                f"full-attention + SWA pool, got {type(cache).__name__}."
            )
        if draft_plan is not None:
            kv_cache_builder.maybe_register_hicache_draft(
                tree_cache=cache, draft_plan=draft_plan
            )
    return cache


def _create_cache_controller(
    *,
    params: Any,
    host_pool_group: HostPoolGroup,
    transfer_layer_id_max: int,
    load_cache_event: Any,
    storage_backend: Optional[str],
    storage_backend_extra_config: Optional[dict],
    prefetch_threshold: int,
    model_name: Optional[str],
    enable_storage_metrics: bool,
) -> HybridCacheController:
    memory = get_memory()
    return HybridCacheController.create_for_backend(
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


def _register_decode_host_cache(
    cache: UnifiedRadixCache, cache_controller: HybridCacheController
) -> None:
    _full_only_host_caches.add(cache)
    _restore_gated_caches.add(cache)
    _load_device_pages[cache_controller] = {}


def _build_full_only_host_stack(
    *,
    cache: UnifiedRadixCache,
    kvcache: Any,
    params: Any,
    **controller_kwargs,
) -> hybrid_pool_assembler.StackBuildResult:
    """Build the RAM pool and its controller for full-attention KV only.

    Like ``build_hybrid_swa_stack`` without the SWA pool, so the full pool gets the
    whole host budget.
    """
    # Draft KV registers as a sidecar after this build, never packed.
    assert not params.mtp_draft_device_pools, "packed MTP draft KV is not supported"
    full_layer_mapping, swa_layer_mapping = hybrid_pool_assembler._swa_layer_mappings(
        kvcache
    )
    # Full-attention layers interleave with SWA layers. Transfers keep the
    # stage-wide layer span even though only full-attention layers move.
    transfer_layer_id_max = (
        max(full_layer_mapping.keys() | swa_layer_mapping.keys()) + 1
    )
    memory = get_memory()
    full_host_pool = hybrid_pool_assembler.build_kv_host_pool(
        kv_pool=kvcache.full_kv_pool,
        page_size=params.page_size,
        use_mla=False,
        host_size=memory.hicache_size if memory.hicache_size > 0 else None,
        pool_label="full",
    )
    host_pool_group = HostPoolGroup(
        [
            hybrid_pool_assembler.build_pool_entry(
                name=PoolName.KV,
                host_pool=full_host_pool,
                device_pool=kvcache.full_kv_pool,
                layer_mapping=full_layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max,
                is_anchor=True,
            )
        ]
    )
    cache_controller = _create_cache_controller(
        params=params,
        host_pool_group=host_pool_group,
        transfer_layer_id_max=transfer_layer_id_max,
        **controller_kwargs,
    )
    _register_decode_host_cache(cache, cache_controller)
    return hybrid_pool_assembler.StackBuildResult(
        host_pool_group=host_pool_group,
        cache_controller=cache_controller,
        component_host_pools={ComponentType.FULL: full_host_pool},
        pools_desc="Full (SWA window from prefill)",
    )


def _swa_host_window_pages(*, sliding_window_size: int, page_size: int) -> int:
    """Pages one host window takes: a window that starts mid-page spans one more."""
    return -(-(sliding_window_size + page_size - 1) // page_size)


def _host_size_gb(kv_pool: Any, num_tokens: int) -> float:
    """``--hicache-size`` style budget (GB) that holds ``num_tokens`` of ``kv_pool``."""
    size_bytes = kv_pool.get_kv_size_bytes()
    if isinstance(size_bytes, tuple):
        size_bytes = sum(size_bytes)
    # Device bytes per token include the padding pages, so this rounds up.
    return num_tokens * size_bytes / kv_pool.size / 1e9


def _build_full_swa_host_stack(
    *,
    cache: UnifiedRadixCache,
    kvcache: Any,
    params: Any,
    num_windows: int,
    **controller_kwargs,
) -> hybrid_pool_assembler.StackBuildResult:
    """Build the full pool with the full-only budget plus an SWA pool for
    ``num_windows`` windows, with the callbacks upstream's SWA strategy uses.

    ``build_hybrid_swa_stack`` splits one budget by device bytes, so this builds
    its group directly to size the SWA pool in windows.
    """
    assert not params.mtp_draft_device_pools, "packed MTP draft KV is not supported"
    full_layer_mapping, swa_layer_mapping = hybrid_pool_assembler._swa_layer_mappings(
        kvcache
    )
    transfer_layer_id_max = (
        max(full_layer_mapping.keys() | swa_layer_mapping.keys()) + 1
    )
    allocator = params.token_to_kv_pool_allocator
    window_pages = _swa_host_window_pages(
        sliding_window_size=params.sliding_window_size, page_size=params.page_size
    )
    enable_unified_memory = get_memory().enable_unified_memory
    host_pool_group = hybrid_pool_assembler.build_hybrid_swa_group(
        page_size=params.page_size,
        full_kv_pool=kvcache.full_kv_pool,
        swa_kv_pool=kvcache.swa_kv_pool,
        full_layer_mapping=full_layer_mapping,
        swa_layer_mapping=swa_layer_mapping,
        use_mla=False,
        # None: --hicache-size, else --hicache-ratio, as the full-only pool.
        kv_host_size=None,
        swa_host_size=_host_size_gb(
            kvcache.swa_kv_pool, num_windows * window_pages * params.page_size
        ),
        host_full_evict_fn=lambda n: cache.evict_host(n, ComponentType.FULL),
        host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
        device_swa_evict_fn=lambda n: hybrid_pool_assembler._evict_swa_for_device_alloc(
            cache, n
        ),
        swa_attn_allocator=allocator.swa_attn_allocator,
        swa_indices_from_anchor_fn=(
            allocator.bind_swa_for_loaded_rows if enable_unified_memory else None
        ),
        swa_free_from_anchor_fn=allocator.free_swa if enable_unified_memory else None,
    )
    cache_controller = _create_cache_controller(
        params=params,
        host_pool_group=host_pool_group,
        transfer_layer_id_max=transfer_layer_id_max,
        **controller_kwargs,
    )
    _register_decode_host_cache(cache, cache_controller)
    _window_host_caches.add(cache)
    return hybrid_pool_assembler.StackBuildResult(
        host_pool_group=host_pool_group,
        cache_controller=cache_controller,
        component_host_pools={
            ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),
        },
        pools_desc=f"Full + SWA ({num_windows} path-end windows)",
    )


class _FullOnlySwaStrategy(hybrid_pool_assembler.StackStrategy):
    """Host stack of a full-only decode cache: full-attention KV, plus SWA
    windows when SGLANG_TPU_PD_DECODE_SWA_HOST_WINDOWS > 0.

    ``install()`` registers it ahead of the built-in strategies. It matches only
    while ``_build_kv_cache`` builds a full-only cache, so every other cache keeps
    its upstream stack.
    """

    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool:
        # Same pool check as upstream's SWA strategy.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            _is_building_full_only_host_cache
            and isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components == {ComponentType.FULL, ComponentType.SWA}
        )

    def build(
        self,
        *,
        cache: UnifiedRadixCache,
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
        controller_kwargs = dict(
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            storage_backend_extra_config=storage_backend_extra_config,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            enable_storage_metrics=enable_storage_metrics,
        )
        num_windows = tpu_envs.SGLANG_TPU_PD_DECODE_SWA_HOST_WINDOWS.get()
        if num_windows <= 0:
            return _build_full_only_host_stack(
                cache=cache, kvcache=kvcache, params=params, **controller_kwargs
            )
        return _build_full_swa_host_stack(
            cache=cache,
            kvcache=kvcache,
            params=params,
            num_windows=num_windows,
            **controller_kwargs,
        )


_FULL_ONLY_SWA_STRATEGY = _FullOnlySwaStrategy()


# ---------------------------------------------------------------------------
# Prefix matching
# ---------------------------------------------------------------------------


def _create_swa_match_validator(
    upstream: Callable, component: SWAComponent, match_device_only: bool = False
) -> Callable:
    """Wraps ``SWAComponent.create_match_validator``: SWA KV does not limit a match,
    except in a window restore."""
    if not uses_full_only_host_cache(component.cache) or (
        _swa_window_restore and uses_window_host_cache(component.cache)
    ):
        return upstream(component, match_device_only)
    component_type = component.component_type

    def validator(node) -> bool:
        swa_data = node.component_data[component_type]
        # Decode never reads SWA KV from the tree in this mode, so a missing or
        # partial window does not end the full-attention match.
        has_swa_kv = swa_data.value is not None or (
            swa_data.host_value is not None and not match_device_only
        )
        if has_swa_kv:
            return True
        # Same rule as upstream's per-request SWA ring: a tombstoned window is a
        # valid boundary while the node's full KV is on device or backed up.
        return node.backuped or not node.evicted

    return validator


def _finalize_swa_match_result(
    upstream: Callable, component: SWAComponent, *args, **kwargs
):
    """Wraps ``SWAComponent.finalize_match_result_in_tree_core``: report no SWA
    host hit outside a window restore, which alone loads SWA KV back."""
    result = upstream(component, *args, **kwargs)
    if _swa_window_restore or not uses_window_host_cache(component.cache):
        return result
    return result._replace(swa_host_hit_length=0)


def _build_swa_hicache_transfers(
    upstream: Callable,
    component: SWAComponent,
    node: Any,
    phase: CacheTransferPhase,
    **kwargs,
):
    """Wraps ``SWAComponent.build_hicache_transfers``: no SWA load-back outside a
    window restore."""
    if (
        phase == CacheTransferPhase.LOAD_BACK
        and not _swa_window_restore
        and uses_window_host_cache(component.cache)
    ):
        return None
    return upstream(component, node, phase, **kwargs)


def _match_prefix_and_lock(
    upstream: Callable, queue: DecodePreallocQueue, req: Any
) -> DecodePrefixMatch:
    """Wraps ``_match_prefix_and_lock``: match only up to the SWA window start."""
    if not uses_full_only_host_cache(queue.tree_cache):
        return upstream(queue, req)
    assert queue._uses_swa_tail_prealloc(), "full-only caches need SWA tail prealloc"
    fill_len = queue._pre_alloc_fill_len(req)
    swa_window_start = fill_len - queue._swa_tail_len(fill_len)
    # Device and host hits stop at the window start. Prefill sends the window,
    # so decode reserves and restores only the prefix before it.
    match_end = swa_window_start + _bigram_boundary(queue.tree_cache)
    result = match_prefix_for_req(
        queue.tree_cache,
        req,
        req.origin_input_ids[:match_end],
        cow_mamba=False,
        include_req=True,
    )
    req.lock_receipt = queue.tree_cache.inc_lock_ref(
        result.last_device_node
    ).to_dec_params()
    return queue._build_decode_prefix_match(req, result)


# ---------------------------------------------------------------------------
# Host SWA windows
# ---------------------------------------------------------------------------


def _collect_unbacked_swa_nodes(
    upstream: Callable, component: SWAComponent, node: Any
) -> list:
    """Wraps ``SWAComponent._collect_unbacked_swa_nodes``: back up SWA KV only
    in the last window of a path.

    A node is in it when it is a leaf, or when live SWA KV leads from it down
    to a leaf in fewer than a window of tokens. Any other node's window is
    stale, so eviction tombstones it without a copy.
    """
    tree_core = component.tree_core
    cache = component.cache
    if (
        not (uses_window_host_cache(cache) or cache in _path_end_prefill_caches)
        or not tree_core.has_swa_host_pool
        or tree_core.is_host_memory_buffer_only
    ):
        return upstream(component, node)
    window = component.sliding_window_size
    tail = _live_swa_tail_to_leaf(
        node, component_type=component.component_type, window=window
    )
    if tail is None:
        return []
    return _unbacked_swa_nodes_within(
        node,
        component_type=component.component_type,
        root=tree_core.root_node,
        num_tokens=window - tail,
    )


def _live_swa_tail_to_leaf(
    node: Any, *, component_type: ComponentType, window: int
) -> Optional[int]:
    """Fewest tokens below ``node`` on a live-SWA path to a leaf, or None when
    every such path is a window or longer."""
    best: Optional[int] = None
    stack = [(node, 0)]
    while stack:
        cur, below = stack.pop()
        if not cur.children:
            best = below if best is None else min(best, below)
            continue
        for child in cur.children.values():
            child_below = below + len(child.key)
            if (
                child.component_data[component_type].value is not None
                and child_below < window
            ):
                stack.append((child, child_below))
    return best


def _unbacked_swa_nodes_within(
    node: Any, *, component_type: ComponentType, root: Any, num_tokens: int
) -> list:
    """Device-only SWA nodes in the ``num_tokens`` ending at ``node``, deepest
    first. Same walk as upstream's window collection, with a shorter span."""
    covered = 0
    unbacked = []
    cur = node
    while cur is not root and covered < num_tokens:
        if cur.write_through_pending_id is not None:
            break
        data = cur.component_data[component_type]
        value = data.value if data.value is not None else data.host_value
        if value is None:
            break
        covered += len(value)
        if data.value is not None and data.host_value is None:
            unbacked.append(cur)
        cur = cur.parent
    return unbacked


# ---------------------------------------------------------------------------
# Allocation and restore
# ---------------------------------------------------------------------------


def _alloc_for_decode_prealloc(
    upstream: Callable,
    allocator: Any,
    *,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
    **other_kwargs,
) -> torch.Tensor:
    """Wraps ``alloc_for_decode_prealloc``: allocate new KV after the restored range.

    Upstream's SWA-tail branch passes ``prefix_lens=[prefix_len]`` (the HBM hit)
    next to ``extend_num_tokens=fill_len - total_prefix_len``. The two disagree
    once a RAM restore makes ``total_prefix_len > prefix_len``. Upstream never
    gets there, because it has no decode RAM offload for SWA models and so the
    two lengths are always equal. This wrapper starts at ``total_prefix_len``.
    """
    kwargs = dict(uses_swa_tail=uses_swa_tail, **other_kwargs)
    has_host_restore = total_prefix_len > prefix_len
    if not (uses_swa_tail and has_host_restore):
        return upstream(
            allocator,
            prefix_len=prefix_len,
            total_prefix_len=total_prefix_len,
            prefix_indices=prefix_indices,
            **kwargs,
        )
    # The restore range ends on a page, so the allocator never extends a
    # partial page from last_loc. Upstream reads only prefix_indices[-1:] to
    # build last_loc, so the device hit's tail (or -1 without one) still
    # supplies it.
    assert total_prefix_len % allocator.page_size == 0, (
        f"host restore must end on a page, got {total_prefix_len=}"
    )
    last_loc_source = (
        prefix_indices
        if prefix_len > 0
        else torch.tensor([-1], dtype=torch.int64, device=allocator.device)
    )
    return upstream(
        allocator,
        prefix_len=total_prefix_len,
        total_prefix_len=total_prefix_len,
        prefix_indices=last_loc_source,
        **kwargs,
    )


def _restore_rematch(
    upstream: Callable,
    tree_cache: Any,
    req: Any,
    token_ids: Optional[Any] = None,
    **kwargs,
):
    """Wraps ``match_prefix_for_req`` in the restore: keep the last page under EAGLE.

    Without the extra token the restore's re-match comes up one page short and
    the restore fails.
    """
    boundary = (
        _bigram_boundary(tree_cache) if uses_full_only_host_cache(tree_cache) else 0
    )
    if boundary and token_ids is not None:
        token_ids = req.origin_input_ids[: len(token_ids) + boundary]
        # Upstream also caps the key at the decode prefix; widen it the same way.
        if kwargs.get("max_prefix_len") is not None:
            kwargs["max_prefix_len"] += boundary
    return upstream(tree_cache, req, token_ids, **kwargs)


def _commit_hicache_local_restore_to_req(
    upstream: Callable, queue: DecodeHiCacheTransferMixin, decode_req: DecodeRequest
) -> None:
    """Wraps ``_commit_hicache_local_restore_to_req``: also drop the SWA lock.

    The SWA window comes from prefill, so the request releases the SWA part of the
    restored lock, as it does after admission.
    """
    prefix_match = decode_req.prefix_match
    needs_restore = prefix_match is not None and prefix_match.needs_local_restore
    upstream(queue, decode_req)
    if not (uses_full_only_host_cache(queue.tree_cache) and needs_restore):
        return
    req = decode_req.req
    queue.tree_cache.dec_swa_lock_only(req.last_node, req.lock_receipt)
    req.swa_prefix_lock_released = True


def _gate_decode_restores(cache: Any) -> None:
    """Register a TPU DMA PD decode prefix cache for the restore hold."""
    disagg = get_disagg()
    if (
        disagg.disaggregation_mode == "decode"
        and disagg.disaggregation_decode_enable_radix_cache
        and isinstance(cache, UnifiedRadixCache)
        and isinstance(cache.cache_controller, TpuDMAHybridCacheController)
    ):
        _restore_gated_caches.add(cache)
        _load_device_pages[cache.cache_controller] = {}


def _maybe_enable_prefill_path_end_backup(cache: Any) -> None:
    """Register a PD prefill cache with a host SWA pool for path-end-only SWA backup."""
    if (
        not tpu_envs.SGLANG_TPU_ENABLE_PD_PREFILL_SWA_PATH_END_BACKUP.get()
        or get_disagg().disaggregation_mode != "prefill"
        or not isinstance(cache, UnifiedRadixCache)
    ):
        return
    if not cache.tree_core.has_swa_host_pool:
        logger.warning(
            "SGLANG_TPU_ENABLE_PD_PREFILL_SWA_PATH_END_BACKUP has no effect: "
            "this prefill cache has no host SWA pool."
        )
        return
    _path_end_prefill_caches.add(cache)
    logger.info(
        "PD prefill SWA HiCache: host RAM keeps only the last SWA window of each path."
    )


def _controller_load(
    upstream: Callable,
    controller: TpuDMAHybridCacheController,
    host_indices: torch.Tensor,
    priority: Optional[int] = None,
    node_id: int = -1,
    extra_pools: Optional[list] = None,
) -> Optional[torch.Tensor]:
    """Wraps the hybrid controller's ``load``: remember which device pages it fills."""
    device_indices = upstream(
        controller,
        host_indices=host_indices,
        priority=priority,
        node_id=node_id,
        extra_pools=extra_pools,
    )
    loads = _load_device_pages.get(controller)
    # An auxiliary-only restore (e.g. Mamba state) fills no KV pages.
    if loads is not None and device_indices is not None and device_indices.numel():
        loads[node_id] = _page_ids(device_indices, controller.page_size)
    return device_indices


def _page_ids(kv_indices: torch.Tensor, page_size: int) -> torch.Tensor:
    # Tree values and load allocations are whole pages, so every page_size-th
    # token index names one page.
    return kv_indices[::page_size].to(torch.int64) // page_size


def _inflight_page_mask(cache: UnifiedRadixCache) -> Optional[torch.Tensor]:
    """Bool mask over device page ids a RAM restore is still filling, or None."""
    loads = _load_device_pages.get(cache.cache_controller)
    if not loads:
        return None
    # loading_check pops an anchor only after its DMA has landed.
    for node_id in [n for n in loads if n not in cache.ongoing_load_back]:
        del loads[node_id]
    if not loads:
        return None
    pages = torch.cat(tuple(loads.values()))
    # Built once per tick, so each queued request costs one lookup instead of
    # an isin against every loading page.
    mask = torch.zeros(int(pages.max()) + 1, dtype=torch.bool, device=pages.device)
    mask[pages] = True
    return mask


def _reads_any_page(
    decode_req: DecodeRequest, inflight_mask: torch.Tensor, page_size: int
) -> bool:
    """Whether the request's cached prefix uses a page in ``inflight_mask``."""
    prefix_parts = [decode_req.prefix_match.prefix_indices]
    if decode_req.hicache_restored_kv_indices is not None:
        prefix_parts.append(decode_req.hicache_restored_kv_indices)
    prefix_pages = torch.cat([_page_ids(part, page_size) for part in prefix_parts])
    # Pages past the end of the mask are not loading.
    prefix_pages = prefix_pages[prefix_pages < inflight_mask.numel()]
    return bool(inflight_mask[prefix_pages].any())


def _process_hicache_local_restores(
    upstream: Callable, queue: DecodeHiCacheTransferMixin, decode_reqs: list
) -> None:
    """Wraps ``_process_hicache_local_restores``: wait for restores into reused pages.

    A restore puts its pages in the cache before the copy lands. A request that
    reuses them looks ready to upstream, so it is held until the restore is acked.
    """
    upstream(queue, decode_reqs)
    if queue.tree_cache not in _restore_gated_caches:
        return
    inflight_mask = _inflight_page_mask(queue.tree_cache)
    if inflight_mask is None:
        return
    page_size = queue.tree_cache.page_size
    for decode_req in decode_reqs:
        if (
            decode_req.hicache_restore_status == HiCacheRestoreResult.READY
            and decode_req.prefix_match is not None
            and _reads_any_page(decode_req, inflight_mask, page_size)
        ):
            decode_req.hicache_restore_status = HiCacheRestoreResult.PENDING


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def _patch_targets() -> tuple[tuple[Any, str, Callable], ...]:
    """The upstream functions this module wraps, as (owner, name, wrapper).

    Callers look these functions up on their module at call time, so replacing
    the module attribute reaches all of them.
    """
    return (
        # Upstream rejects decode SWA radix + HiCache at startup. Build the
        # full-only RAM cache instead.
        (kv_cache_builder, "build_kv_cache", _build_kv_cache),
        # That build passes the HiCache flag as False to get past the check,
        # so turn HiCache back on when the tree cache is created.
        (kv_cache_builder, "create_tree_cache", _create_tree_cache),
        # Decode never reads SWA KV from its cache, so missing SWA KV must not
        # end a match of full-attention KV.
        (SWAComponent, "create_match_validator", _create_swa_match_validator),
        # A request's own restore loads only full KV: no SWA host hit and no
        # SWA load-back outside a window restore.
        (
            SWAComponent,
            "finalize_match_result_in_tree_core",
            _finalize_swa_match_result,
        ),
        (SWAComponent, "build_hicache_transfers", _build_swa_hicache_transfers),
        # Host RAM keeps only the last SWA window of each path, on a decode
        # with host windows and an opted-in prefill.
        (
            SWAComponent,
            "_collect_unbacked_swa_nodes",
            _collect_unbacked_swa_nodes,
        ),
        # Stop decode cache hits where the SWA window starts, so prefill always
        # sends the window.
        (DecodePreallocQueue, "_match_prefix_and_lock", _match_prefix_and_lock),
        # Allocate new KV after the RAM-restored range. Upstream, which has no
        # decode RAM offload for SWA models, starts at the HBM hit instead.
        (decode, "alloc_for_decode_prealloc", _alloc_for_decode_prealloc),
        # Under EAGLE the restore's re-match comes up one page short and the
        # restore fails, so add the token that covers the last page.
        (decode_hicache_mixin, "match_prefix_for_req", _restore_rematch),
        # When the restore commits, also drop the SWA part of the request's
        # lock, since the SWA window comes from prefill.
        (
            DecodeHiCacheTransferMixin,
            "_commit_hicache_local_restore_to_req",
            _commit_hicache_local_restore_to_req,
        ),
        # Hold a request that reuses pages of an unfinished restore. Upstream
        # lets it decode before the copy lands.
        (
            DecodeHiCacheTransferMixin,
            "_process_hicache_local_restores",
            _process_hicache_local_restores,
        ),
        # Record the device pages each restore fills, for the hold above.
        (TpuDMAHybridCacheController, "load", _controller_load),
    )


def _wrap(owner: Any, name: str, impl: Callable) -> None:
    """Replace ``owner.name`` with ``impl(upstream, *args, **kwargs)``."""
    upstream = getattr(owner, name)

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return impl(upstream, *args, **kwargs)

    setattr(owner, name, wrapper)


def install() -> None:
    """Install every wrapper and the host stack strategy, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    for owner, name, impl in _patch_targets():
        _wrap(owner, name, impl)
    # Builds a RAM pool for full-attention KV only, with no SWA pool.
    hybrid_pool_assembler.register_stack_strategy(_FULL_ONLY_SWA_STRATEGY)
