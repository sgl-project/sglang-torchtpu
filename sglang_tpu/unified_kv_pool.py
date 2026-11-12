# SPDX-License-Identifier: Apache-2.0
"""Logical-to-Physical Unified KV cache pool for hybrid Full + SWA models on TPU.

Shares ``G = full.num_layers`` physical 5D Pallas KV buffers across Full-Attention
(1 physical page per logical page) and Sliding-Window Attention
(``M_swa = ceil(swa.num_layers / G)`` groups) with zero external fragmentation for
models where ``H_swa == H_full`` and ``D_swa == D_full``. Physical page 0 is
reserved as the dummy-write sink for padded tokens.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.model_executor.pool_configurator import (
    MemoryPoolConfig,
    MemoryPoolConfigurator,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_server_args,
    get_spec,
)
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_free_fence import HeldPages
from sglang_tpu.kv_mxint8 import GROUP_SIZE as MXINT8_DEFAULT_GROUP_SIZE
from sglang_tpu.kv_mxint8 import cache_bytes_per_token as mxint8_cache_bytes_per_token
from sglang_tpu.kv_pool import (
    GB,
    TpuFusedMHATokenToKVPool,
    _cdiv,
)
from sglang_tpu.tpu_dma_hicache import is_tpu_dma_hicache_backend

logger = logging.getLogger(__name__)


def num_swa_groups(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of SWA groups ``M_swa = ceil(num_swa_layers / num_full_layers)``."""
    if num_full_layers <= 0 or num_swa_layers <= 0:
        raise ValueError(
            f"Unified KV pool requires positive full ({num_full_layers}) and "
            f"SWA ({num_swa_layers}) layer counts."
        )
    return _cdiv(num_swa_layers, num_full_layers)


def _window_tokens(sliding_window_size: Optional[int], page_size: int) -> int:
    if sliding_window_size is None:
        return 0
    return _cdiv(sliding_window_size + page_size, page_size) * page_size


def _max_request_tokens(
    shared_tokens: int,
    page_size: int,
    sliding_window_size: Optional[int],
    num_swa_groups: int,
) -> int:
    if sliding_window_size is None:
        unit = (1 + num_swa_groups) * page_size
        return (max(0, shared_tokens) // unit) * page_size
    return max(
        0,
        shared_tokens - num_swa_groups * _window_tokens(sliding_window_size, page_size),
    )


@dataclass(frozen=True)
class AttnLayerSpec:
    """Per-attention-type geometry for either Full or SWA layers.

    Attributes:
        head_num: KV head count on this TP rank (post-TP partitioning).
        head_dim: Unpadded per-head K/V dimension (must satisfy K == V).
        layer_ids: Global model layer indices belonging to this attention type,
            in ascending model execution order.
    """

    head_num: int
    head_dim: int
    layer_ids: Sequence[int]

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)


@dataclass(frozen=True)
class UnifiedPageLayout:
    """Resolved physical page layout for a unified Full + SWA KV pool.

    Attributes:
        page_head_num: Physical KV heads per page (requires ``swa.head_num == full.head_num``).
        page_head_dim: Physical head dimension per page (requires ``swa.head_dim == full.head_dim``).
        num_shared_buffers: Number of shared 5D physical KV tensors ``G = full.num_layers``.
        num_swa_groups: Number of physical SWA page groups per logical SWA token
            ``M_swa = ceil(swa.num_layers / G)``.
    """

    page_head_num: int
    page_head_dim: int
    num_shared_buffers: int
    num_swa_groups: int


def resolve_unified_page_layout(
    full: AttnLayerSpec,
    swa: AttnLayerSpec,
) -> UnifiedPageLayout:
    """Validate matching Full/SWA KV geometry and compute SWA group counts."""
    if (
        full.head_num <= 0
        or full.head_dim <= 0
        or swa.head_num <= 0
        or swa.head_dim <= 0
    ):
        raise ValueError(
            f"Invalid head geometry: full=({full.head_num}, {full.head_dim}), "
            f"swa=({swa.head_num}, {swa.head_dim})."
        )
    if swa.head_num != full.head_num or swa.head_dim != full.head_dim:
        raise ValueError(
            "TpuUnifiedSWAKVPool requires matching Full and SWA KV head geometry "
            f"(H_swa == H_full and D_swa == D_full), got full=({full.head_num}, "
            f"{full.head_dim}), swa=({swa.head_num}, {swa.head_dim})."
        )
    return UnifiedPageLayout(
        page_head_num=full.head_num,
        page_head_dim=full.head_dim,
        num_shared_buffers=full.num_layers,
        num_swa_groups=num_swa_groups(full.num_layers, swa.num_layers),
    )


def _kvc_unified_layout(kvc) -> UnifiedPageLayout:
    tp = get_parallel().attn_tp_size
    mc = kvc.model_config
    return resolve_unified_page_layout(
        full=AttnLayerSpec(
            head_num=mc.get_num_kv_heads(tp),
            head_dim=mc.head_dim,
            layer_ids=kvc.layer_info.full_attention_layer_ids,
        ),
        swa=AttnLayerSpec(
            head_num=mc.get_swa_num_kv_heads(tp),
            head_dim=mc.swa_head_dim or mc.head_dim,
            layer_ids=kvc.layer_info.swa_attention_layer_ids,
        ),
    )


def _mxint8_unified_bytes_per_token(kvc, layout: UnifiedPageLayout) -> int:
    return mxint8_cache_bytes_per_token(
        layout.page_head_num,
        layout.page_head_dim,
        kvc.pool_page_size,
        group_size=kvc.kv_group_size or MXINT8_DEFAULT_GROUP_SIZE,
        layout="blocked",
    )


def _draft_swa_exceeds_target_window(
    *,
    target_window: Optional[int],
    target_cfg_window: Optional[int],
    draft_window: Optional[int],
    draft_cfg_window: Optional[int],
) -> bool:
    """Return True when the draft's SWA window outlives Target's SWA eviction window."""
    t_left = (
        int(target_window)
        if target_window is not None
        else (
            max(0, int(target_cfg_window) - 1)
            if target_cfg_window is not None
            else None
        )
    )
    if t_left is None:
        return False
    d_left = (
        int(draft_window)
        if draft_window is not None
        else (
            max(0, int(draft_cfg_window) - 1) if draft_cfg_window is not None else None
        )
    )
    return d_left is None or d_left > t_left


_DRAFT_MC_CACHE: tuple[tuple, Any] | None = None


def _is_supported_eagle_spec(spec_algo) -> bool:
    return (
        spec_algo is not None
        and spec_algo.is_eagle()
        and not spec_algo.is_frozen_kv_mtp()
    )


def _is_supported_draft_spec(spec_algo) -> bool:
    return _is_supported_eagle_spec(spec_algo) or (
        spec_algo is not None and spec_algo.is_dflash_family()
    )


def _draft_num_layers(spec_algo, spec_aux) -> int:
    if spec_algo is None or spec_aux is None:
        return 0
    if spec_algo.is_dflash_family():
        return int(getattr(spec_aux, "dflash_draft_num_layers", 0) or 0)
    if _is_supported_eagle_spec(spec_algo):
        return int(getattr(spec_aux, "eagle_draft_num_layers", 0) or 0)
    return 0


def _dflash_swa_window_left(draft_mc) -> Optional[int]:
    hf_cfg = getattr(draft_mc, "hf_config", None)
    if hf_cfg is None:
        return None
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
    )

    try:
        return get_dflash_attention_sliding_window_size(hf_cfg)
    except ValueError:
        return None


def _resolve_draft_attention_info(kvc, draft_mc):
    from sglang_tpu.tpu_platform import _mha_kv_pool_cls

    spec_algo = kvc.spec_algorithm
    draft_attn_backend = (
        getattr(kvc, "speculative_draft_attention_backend", None)
        or get_spec().speculative_draft_attention_backend
        or get_exec().kernel.attention_backend
    )
    draft_pool_cls = _mha_kv_pool_cls(draft_attn_backend)
    model_dtype = getattr(draft_mc, "dtype", None) or getattr(
        kvc, "model_dtype", torch.bfloat16
    )
    draft_dtype_str, draft_dtype = configure_kv_cache_dtype(
        server_args_kv_cache_dtype=get_model().kv_cache_dtype,
        speculative_draft_kv_cache_dtype=get_spec().speculative_draft_kv_cache_dtype,
        model=None,
        model_dtype=model_dtype,
        is_draft_worker=True,
        is_dflash=spec_algo.is_dflash_family(),
        speculative_draft_attention_backend=draft_attn_backend or "",
    )
    return draft_attn_backend, draft_pool_cls, draft_dtype_str, draft_dtype


def _resolve_draft_model_config(kvc):
    """Resolve the paired speculative draft ``ModelConfig`` for ``kvc``, or ``None``."""
    global _DRAFT_MC_CACHE
    spec_algo = getattr(kvc, "spec_algorithm", None)
    if getattr(kvc, "is_draft_worker", False) or not _is_supported_draft_spec(
        spec_algo
    ):
        return None
    spec_aux = getattr(kvc, "spec_aux_config", None)
    if _draft_num_layers(spec_algo, spec_aux) <= 0:
        return None
    if (cached := getattr(spec_aux, "draft_model_config", None)) is not None:
        return cached
    spec_ctx = get_spec()
    server_args = getattr(kvc, "server_args", None)
    explicit_draft_path = getattr(spec_ctx, "speculative_draft_model_path", None)
    draft_path = explicit_draft_path or (
        None
        if spec_algo.is_dflash_family()
        else getattr(server_args, "model_path", None)
    )
    if draft_path and server_args is not None:
        draft_rev = getattr(spec_ctx, "speculative_draft_model_revision", None) or (
            None if explicit_draft_path else getattr(server_args, "revision", None)
        )
        cache_key = (
            str(draft_path),
            draft_rev,
            getattr(server_args, "context_length", None),
            getattr(server_args, "dtype", None),
            getattr(server_args, "json_model_override_args", None),
        )
        if _DRAFT_MC_CACHE is not None and _DRAFT_MC_CACHE[0] == cache_key:
            return _DRAFT_MC_CACHE[1]
        from sglang.srt.configs.model_config import ModelConfig

        draft_mc = ModelConfig.from_server_args(
            server_args,
            model_path=draft_path,
            model_revision=draft_rev,
            is_draft_model=True,
        )
        _DRAFT_MC_CACHE = (cache_key, draft_mc)
        return draft_mc
    return None if spec_algo.is_dflash_family() else getattr(kvc, "model_config", None)


def _dflash_unsupported_reason(kvc, draft_mc) -> Optional[str]:
    spec_algo = kvc.spec_algorithm
    if not spec_algo.is_dflash_family():
        return None
    from sglang_tpu.experimental import (
        is_experimental_spec_worker,
        pd_decode_draft_prefill,
    )

    spec_ctx = get_spec()
    if getattr(spec_ctx, "speculative_draft_window_size", None) is not None:
        return (
            "does not support --speculative-draft-window-size; configure an "
            "SWA draft model instead."
        )
    if pd_decode_draft_prefill():
        return "does not support SGLANG_TPU_PD_DECODE_DRAFT_PREFILL."
    exp_worker = is_experimental_spec_worker(spec_algo)
    if spec_algo.is_dspark() and not exp_worker:
        return "with DSPARK requires SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1."
    if draft_mc is None:
        return None
    is_swa_draft = draft_mc.is_hybrid_swa and bool(draft_mc.swa_attention_layer_ids)
    if is_swa_draft and not exp_worker:
        return (
            f"with hybrid-SWA {spec_algo.name} drafts requires "
            "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1."
        )
    _, draft_pool_cls, draft_dtype_str, _ = _resolve_draft_attention_info(kvc, draft_mc)
    if draft_pool_cls is not TpuFusedMHATokenToKVPool or draft_dtype_str == "mxint8":
        return (
            f"with {spec_algo.name} requires an unpacked non-MXINT8 "
            "TpuFusedMHATokenToKVPool draft pool."
        )
    return None


def _unsupported_reason(kvc) -> Optional[str]:
    """Return a human-readable reason why ``kvc`` cannot use the TPU unified KV pool,
    or ``None`` when supported."""
    parallel = get_parallel()
    memory = get_memory()
    disagg = get_disagg()
    disagg_mode = disagg.disaggregation_mode
    disagg_backend = getattr(disagg, "disaggregation_transfer_backend", None)
    pool_cls = kvc._tpu_mha_pool_class()
    spec_algo = kvc.spec_algorithm
    supported_spec = spec_algo.is_none() or _is_supported_draft_spec(spec_algo)
    draft_mc = _resolve_draft_model_config(kvc)
    draft_pool_cls = (
        _resolve_draft_attention_info(kvc, draft_mc)[1]
        if draft_mc is not None
        else TpuFusedMHATokenToKVPool
    )
    spec_aux = getattr(kvc, "spec_aux_config", None)
    draft_total_layers = _draft_num_layers(spec_algo, spec_aux)
    draft_swa_layers = int(getattr(spec_aux, "eagle_draft_swa_num_layers", 0) or 0)
    mixed_draft_layers = (
        draft_mc is not None
        and draft_mc.is_hybrid_swa
        and bool(draft_mc.full_attention_layer_ids)
        and bool(draft_mc.swa_attention_layer_ids)
    ) or (
        draft_mc is None
        and _is_supported_eagle_spec(spec_algo)
        and 0 < draft_swa_layers < draft_total_layers
    )
    dflash_reason = _dflash_unsupported_reason(kvc, draft_mc)
    draft_window_left = (
        _dflash_swa_window_left(draft_mc)
        if (
            draft_mc is not None
            and spec_algo.is_dflash_family()
            and draft_mc.sliding_window_size is None
        )
        else None
    )
    checks = (
        (kvc.is_draft_worker, "is target-only; draft workers use standard KV pools."),
        (not kvc.is_hybrid_swa, "requires a hybrid SWA model."),
        (kvc.use_mla_backend, "does not support MLA models."),
        (
            kvc.model_config.has_asymmetric_kv,
            "does not support asymmetric K/V head dimensions.",
        ),
        (
            kvc.mambaish_config is not None
            or kvc.hybrid_gdn_config is not None
            or kvc.hybrid_kda_config is not None,
            "does not support hybrid linear-attention or Mamba models.",
        ),
        (
            not supported_spec,
            f"does not support speculative_algorithm={spec_algo.name!r}.",
        ),
        (dflash_reason is not None, dflash_reason or ""),
        (
            not spec_algo.is_none()
            and (memory.enable_hierarchical_cache or disagg_mode != "null"),
            "does not yet support speculative draft KV caches with HiCache or PD disaggregation.",
        ),
        (
            draft_mc is not None
            and not (
                isinstance(draft_pool_cls, type)
                and issubclass(draft_pool_cls, TpuFusedMHATokenToKVPool)
            ),
            f"requires a TpuFusedMHATokenToKVPool draft pool subclass, got {draft_pool_cls}.",
        ),
        (
            draft_mc is not None
            and (
                draft_mc.attention_arch == AttentionArch.MLA
                or draft_mc.has_asymmetric_kv
                or mambaish_config(draft_mc) is not None
            ),
            "does not support MLA, Mamba, or asymmetric-KV speculative draft models.",
        ),
        (
            mixed_draft_layers,
            "does not support speculative draft models that mix full and "
            "sliding-window attention layers.",
        ),
        (
            draft_mc is not None
            and draft_mc.is_hybrid_swa
            and bool(draft_mc.swa_attention_layer_ids)
            and _draft_swa_exceeds_target_window(
                target_window=kvc.sliding_window_size,
                target_cfg_window=kvc.model_config.sliding_window_size,
                draft_window=draft_window_left,
                draft_cfg_window=draft_mc.sliding_window_size,
            ),
            "does not support speculative draft models whose sliding window exceeds "
            "the target sliding window.",
        ),
        (
            parallel.pp_size > 1 or kvc.pp_size > 1,
            "does not support pipeline parallelism.",
        ),
        (
            parallel.attn_cp_size > 1,
            "does not support prefill context parallelism.",
        ),
        (
            parallel.dcp_enabled or parallel.attn_dcp_size > 1,
            "does not support decode context parallelism.",
        ),
        (
            disagg_mode not in ("null", "prefill", "decode"),
            f"does not support disaggregation_mode={disagg_mode!r}.",
        ),
        (
            disagg_mode != "null" and disagg_backend != "raiden",
            "with PD disaggregation requires --disaggregation-transfer-backend raiden, "
            f"got {disagg_backend!r}.",
        ),
        (
            getattr(disagg, "enable_pd_role_switch", False),
            "does not support --enable-pd-role-switch.",
        ),
        (
            getattr(disagg, "disaggregation_decode_enable_offload_kvcache", False),
            "does not support --disaggregation-decode-enable-offload-kvcache.",
        ),
        (
            getattr(disagg, "disaggregation_decode_retraction_backup", None)
            == "host_pool",
            "does not support --disaggregation-decode-retraction-backup host_pool.",
        ),
        (
            getattr(disagg, "disaggregation_decode_host_receive_threshold", 0.0) > 0,
            "does not support --disaggregation-decode-host-receive-threshold > 0.",
        ),
        (
            disagg_mode != "null" and tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get(),
            "does not support SGLANG_RAIDEN_PIPELINE_TRANSFER.",
        ),
        (
            memory.enable_hierarchical_cache
            and not is_tpu_dma_hicache_backend(
                getattr(memory, "hicache_io_backend", None)
            ),
            "with --enable-hierarchical-cache requires a TPU DMA "
            f"--hicache-io-backend (e.g. 'raiden'), got "
            f"{getattr(memory, 'hicache_io_backend', None)!r}.",
        ),
        (
            memory.enable_hierarchical_cache
            and getattr(memory, "hicache_storage_backend", None) is not None,
            "with --enable-hierarchical-cache does not support L3 storage "
            f"(--hicache-storage-backend={getattr(memory, 'hicache_storage_backend', None)!r}).",
        ),
        (
            memory.enable_hierarchical_cache
            and getattr(memory, "hicache_host_memory_mode", "cache") == "buffer_only",
            "with --enable-hierarchical-cache does not support "
            "--hicache-host-memory-mode buffer_only.",
        ),
        (
            memory.enable_unified_cache_external_linker,
            "does not support --enable-unified-cache-external-linker.",
        ),
        (
            tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get(),
            "does not support SGLANG_TPU_KV_PACK_HEADS.",
        ),
        (
            tpu_envs.SGLANG_TPU_USE_BLOCK_MAJOR_KV.get(),
            "does not support SGLANG_TPU_USE_BLOCK_MAJOR_KV.",
        ),
        (
            not (
                isinstance(pool_cls, type)
                and issubclass(pool_cls, TpuFusedMHATokenToKVPool)
            ),
            f"requires a TpuFusedMHATokenToKVPool subclass, got {pool_cls}.",
        ),
        (
            kvc.pool_page_size <= 1,
            f"requires page_size > 1, got {kvc.pool_page_size}.",
        ),
    )
    for bad, msg in checks:
        if bad:
            return f"--enable-unified-memory on TPU {msg}"
    try:
        layout = _kvc_unified_layout(kvc)
        if kvc.kv_cache_dtype_str == "mxint8":
            _mxint8_unified_bytes_per_token(kvc, layout)
    except (ValueError, NotImplementedError) as exc:
        return str(exc)
    return None


def _raise_if_unsupported(kvc) -> None:
    reason = _unsupported_reason(kvc)
    if reason is not None:
        raise ValueError(reason)


UNIFIED_KV_POOL_RESOLUTION_SOURCE = "tpu.unified_kv_pool"


def unified_kv_pool_requested(server_args=None) -> bool:
    """Return True if ``--enable-unified-memory`` (or ``SGLANG_TPU_UNIFIED_KV_POOL=1``)
    requested the TPU unified Full+SWA KV pool."""
    if tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL.get():
        return True
    if server_args is None:
        try:
            server_args = get_server_args()
        except ValueError:
            return False
    overrides = getattr(server_args, "_resolved_overrides", None) or ()
    return any(source == UNIFIED_KV_POOL_RESOLUTION_SOURCE for source, _ in overrides)


def unified_kv_pool_enabled(kvc) -> bool:
    """Return True if ``--enable-unified-memory`` is active on a target worker."""
    if getattr(kvc, "is_draft_worker", False):
        return False
    if not unified_kv_pool_requested(getattr(kvc, "server_args", None)):
        return False
    _raise_if_unsupported(kvc)
    return True


class TpuUnifiedSWAKVPool(BaseSWAKVPool):
    """Unified Hybrid Full/SWA KV pool backed by ``G = full.num_layers`` shared 5D buffers."""

    AttnSpec = AttnLayerSpec

    def __init__(
        self,
        size: int,
        *,
        page_size: int,
        dtype: torch.dtype,
        full: AttnLayerSpec,
        swa: AttnLayerSpec,
        device: str,
        token_to_kv_pool_class: type = TpuFusedMHATokenToKVPool,
        pool_kwargs: Optional[dict] = None,
        start_layer: int = 0,
        enable_memory_saver: bool = False,
    ):
        if tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get():
            raise ValueError(
                "TpuUnifiedSWAKVPool does not support SGLANG_TPU_KV_PACK_HEADS."
            )
        if not (
            isinstance(token_to_kv_pool_class, type)
            and issubclass(token_to_kv_pool_class, TpuFusedMHATokenToKVPool)
        ):
            raise ValueError(
                "TpuUnifiedSWAKVPool requires a TpuFusedMHATokenToKVPool subclass, "
                f"got {token_to_kv_pool_class}."
            )
        self.full = full
        self.swa = swa
        self.layout = resolve_unified_page_layout(full, swa)
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=full.num_layers + swa.num_layers,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
        )
        self.head_num = self.layout.page_head_num
        self.head_dim = self.layout.page_head_dim

        G = self.layout.num_shared_buffers
        num_blocks = size // page_size + 1
        self.swa_group_pages = torch.zeros(
            (max(0, self.layout.num_swa_groups - 1), num_blocks), dtype=torch.int32
        )

        extra_kwargs = dict(pool_kwargs) if pool_kwargs else {}
        if extra_kwargs.get("kv_cache_dtype_str") == "mxint8":
            extra_kwargs["kv_layout"] = "blocked"
        self.kv_pool: TpuFusedMHATokenToKVPool = token_to_kv_pool_class(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=self.layout.page_head_num,
            head_dim=self.layout.page_head_dim,
            layer_num=G,
            device=device,
            enable_memory_saver=enable_memory_saver,
            allocation_label="Unified",
            **extra_kwargs,
        )
        if "kv_layout" in extra_kwargs:
            self.kv_layout = extra_kwargs["kv_layout"]
        self._swa_copy_kwargs = (
            {"allow_raiden_backup": False}
            if "allow_raiden_backup"
            in inspect.signature(self.kv_pool.get_cpu_copy).parameters
            else {}
        )

        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        self._swa_layer_group: Dict[int, int] = {}
        for i, gid in enumerate(full.layer_ids):
            self.layers_mapping[gid] = (i, False)
        for j, gid in enumerate(swa.layer_ids):
            self.layers_mapping[gid] = (j, True)
            self._swa_layer_group[gid] = j // G

        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            "TpuUnifiedSWAKVPool mem usage: %.2f GB, unified size: %d tokens "
            "(shared buffers=%d, swa_groups=%d)",
            self.mem_usage,
            self.size,
            G,
            self.layout.num_swa_groups,
        )

    def _buffer_idx(self, layer_id: int) -> int:
        idx, is_swa = self.layers_mapping[layer_id]
        return idx % self.layout.num_shared_buffers if is_swa else idx

    def get_fused_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_fused_kv_buffer(self._buffer_idx(layer_id))

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_key_buffer(self._buffer_idx(layer_id))

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_value_buffer(self._buffer_idx(layer_id))

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_pool.get_kv_buffer(self._buffer_idx(layer_id))

    def set_kv_buffer(self, *args, **kwargs) -> None:
        self.kv_pool.set_kv_buffer(*args, **kwargs)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_swa_index_mapping is not None
        mapping = self.full_to_swa_index_mapping
        return (
            mapping[kv_indices.to(mapping.device)].to(torch.int32).to(kv_indices.device)
        )

    def swa_page_group(self, layer_id: int) -> int:
        return self._swa_layer_group.get(layer_id, 0)

    def get_state(self) -> list[torch.Tensor]:
        return self.kv_pool.get_state()

    def set_state(self, state: list[torch.Tensor]) -> None:
        self.kv_pool.set_state(state)

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return self.kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return self.kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return [], [], []

    def expand_swa_pages(self, page_ids: Sequence[int]) -> List[int]:
        """Expand logical SWA page IDs into ``[token_page, group_slot]`` physical page IDs."""
        if not page_ids:
            return []
        if self.swa_group_pages.shape[0] == 0:
            return [int(x) for x in page_ids]
        base = torch.as_tensor(page_ids, dtype=torch.long)
        companions = self.swa_group_pages[:, base].to(torch.long)
        expanded = torch.cat([base.unsqueeze(0), companions], dim=0)
        return expanded.T.reshape(-1).tolist()

    def _map_swa_group_indices(
        self, group: int, swa_indices_cpu: torch.Tensor
    ) -> torch.Tensor:
        if group == 0:
            return swa_indices_cpu
        phys_pages = self.swa_group_pages[
            group - 1, swa_indices_cpu // self.page_size
        ].to(torch.long)
        return phys_pages * self.page_size + (swa_indices_cpu % self.page_size)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        full_kv_cpu = self.kv_pool.get_cpu_copy(
            indices, mamba_indices=mamba_indices, req_pool_index=req_pool_index
        )
        swa_kv_cpu = None
        swa_mask = None
        if self.full_to_swa_index_mapping is not None:
            mapping = self.full_to_swa_index_mapping
            swa_indices = mapping[indices.to(mapping.device)]
            swa_mask = swa_indices > 0
            if torch.any(swa_mask):
                valid_swa = swa_indices[swa_mask].to(device="cpu", dtype=torch.long)
                swa_kv_cpu = [
                    self.kv_pool.get_cpu_copy(
                        self._map_swa_group_indices(g, valid_swa),
                        **self._swa_copy_kwargs,
                    )
                    for g in range(self.layout.num_swa_groups)
                ]
                swa_mask = swa_mask.cpu()
        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        self.kv_pool.load_cpu_copy(
            kv_cache_cpu["full"],
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )
        swa_kv_cpu = kv_cache_cpu["swa"]
        if swa_kv_cpu is None or self.full_to_swa_index_mapping is None:
            return
        mapping = self.full_to_swa_index_mapping
        swa_indices = mapping[indices.to(mapping.device)]
        new_swa_mask = swa_indices > 0
        old_swa_mask = kv_cache_cpu.get("swa_mask")
        if old_swa_mask is not None:
            old_swa_mask = old_swa_mask.to(indices.device)
            row_mask = new_swa_mask[old_swa_mask].cpu()
            swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
        else:
            row_mask = new_swa_mask.cpu()
            swa_indices = swa_indices[new_swa_mask]
        if swa_indices.numel() == 0:
            return
        valid_swa = swa_indices.to(device="cpu", dtype=torch.long)
        need_filter = not bool(torch.all(row_mask).item())
        for g, grp_cpu in enumerate(swa_kv_cpu):
            if need_filter:
                grp_cpu = self.kv_pool.filter_cpu_copy(grp_cpu, row_mask)
            self.kv_pool.load_cpu_copy(
                grp_cpu, self._map_swa_group_indices(g, valid_swa)
            )


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


class _DraftSwaSidecar:
    """Compact CPU page mapper binding Target SWA primary pages to Draft SWA pages."""

    def __init__(
        self,
        *,
        target_size_full: int,
        draft_size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache=None,
        need_sort: bool = False,
    ):
        assert draft_size_swa > 0 and draft_size_swa % page_size == 0
        self.page_size = page_size
        self.num_draft_pages = draft_size_swa // page_size
        self.allocator = TpuPagedTokenToKVPoolAllocator(
            size=draft_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.target_to_draft_page = torch.zeros(
            target_size_full // page_size + 1, dtype=torch.int64, device="cpu"
        )
        self.mapping = torch.zeros(
            target_size_full + page_size + 1, dtype=torch.int64, device="cpu"
        )
        self.mapping[-1] = -1

    def bind_target_pages(self, target_pages: torch.Tensor) -> None:
        ids = target_pages.reshape(-1).to(device="cpu", dtype=torch.int64)
        ids = ids[ids > 0]
        n = int(ids.numel())
        if n == 0:
            return
        ok = self.allocator._ensure_free_pages(n)
        assert ok, f"Draft SWA sidecar ran out of pages (requested {n})."
        draft_pages = self.allocator.free_pages[:n].to(torch.int64)
        self.allocator.free_pages = self.allocator.free_pages[n:]
        self.allocator._mark_handed_out(draft_pages)
        self.target_to_draft_page[ids] = draft_pages

    def release_target_pages(self, target_pages: torch.Tensor) -> None:
        ids = target_pages.reshape(-1).to(device="cpu", dtype=torch.int64)
        ids = ids[ids > 0]
        if ids.numel() == 0:
            return
        draft_pages = self.target_to_draft_page[ids]
        self.target_to_draft_page[ids] = 0
        valid = draft_pages[draft_pages > 0]
        if valid.numel() > 0:
            self.allocator._release_page_ids(valid)

    def translate_slots(self, target_swa_slots: torch.Tensor) -> torch.Tensor:
        slots = target_swa_slots.to(device="cpu", dtype=torch.int64)
        pages = self.target_to_draft_page[slots.clamp(min=0) // self.page_size]
        return torch.where(
            slots > 0,
            pages * self.page_size + (slots % self.page_size),
            torch.zeros_like(slots),
        )

    def set_mapping(
        self, full_indices: torch.Tensor, target_swa_indices: torch.Tensor
    ) -> None:
        if full_indices.numel() == 0:
            return
        idx = full_indices.to(device="cpu", dtype=torch.int64)
        self.mapping[idx] = self.translate_slots(target_swa_indices)

    def clear_mapping(self, full_indices: torch.Tensor) -> None:
        if full_indices.numel() == 0:
            return
        self.mapping[full_indices.to(device="cpu", dtype=torch.int64)] = 0

    def clear(self) -> None:
        self.allocator.clear()
        self.target_to_draft_page.zero_()
        self.mapping[:-1].fill_(0)
        self.mapping[-1] = -1


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
        self._sidecar: Optional[_DraftSwaSidecar] = None
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

    def available_size(self) -> int:
        avail = (
            self._page_store.num_free_pages() // self._pages_per_id
        ) * self.page_size
        if self._sidecar is not None:
            avail = min(avail, self._sidecar.allocator.available_size())
        return avail

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
        if self._group_pages is not None or self._sidecar is not None:
            self._take_group_pages()
        self.allocated_pages += free_before - self._page_store.num_free_pages()
        return out

    def _ensure_free_pages(self, num_pages: int) -> bool:
        if (
            self._sidecar is not None
            and self._sidecar.allocator.available_size() < num_pages * self.page_size
        ):
            return False
        if self._group_pages is None and self._sidecar is None:
            return super()._ensure_free_pages(num_pages)
        ok = super()._ensure_free_pages(num_pages * self._pages_per_id)
        if ok and self._sidecar is not None:
            ok = self._sidecar.allocator._ensure_free_pages(num_pages)
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
        primary = before[:num_taken]
        if self._group_pages is not None:
            num_extra = self._group_pages.shape[0] * num_taken
            extra = self.free_pages[:num_extra]
            self._mark_handed_out(extra)
            self._group_pages[:, primary.long()] = extra.view(-1, num_taken)
            self.free_pages = self.free_pages[num_extra:]
        if self._sidecar is not None:
            self._sidecar.bind_target_pages(primary)

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        page_ids = tuple(
            ids[ids > 0]
            for ids in (
                p.reshape(-1).to(device="cpu", dtype=torch.int32) for p in page_ids
            )
        )
        if (self._group_pages is not None or self._sidecar is not None) and page_ids:
            ids = torch.cat([p.to(torch.int64) for p in page_ids])
            if ids.numel() > 0:
                if self._sidecar is not None:
                    self._sidecar.release_target_pages(ids)
                if self._group_pages is not None:
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
        *,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: TpuUnifiedSWAKVPool,
        need_sort: bool,
        req_to_token_pool=None,
        sliding_window_size: Optional[int] = None,
        cfg_sliding_window: Optional[int] = None,
    ):
        if page_size <= 1:
            raise ValueError(
                f"TpuUnifiedSWATokenToKVPoolAllocator requires page_size > 1, got {page_size}."
            )
        assert isinstance(kvcache, TpuUnifiedSWAKVPool)
        self._page_store = _SharedPageStore()
        self._swa_groups = kvcache.layout.num_swa_groups
        self.need_sort = need_sort
        self._kvcache = kvcache
        self._req_to_token_pool = req_to_token_pool
        self._target_sliding_window_size = sliding_window_size
        self._target_cfg_sliding_window = cfg_sliding_window
        self._draft_swa_sidecar: Optional[_DraftSwaSidecar] = None
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
            kvcache=kvcache.kv_pool,
            need_sort=need_sort,
        )
        self.swa_attn_allocator = _SharedPagesAllocator(
            page_store=self._page_store,
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache.kv_pool,
            need_sort=need_sort,
            group_pages=kvcache.swa_group_pages if self._swa_groups > 1 else None,
        )
        self.clear()

    @property
    def draft_swa_sidecar(self) -> Optional[_DraftSwaSidecar]:
        return self._draft_swa_sidecar

    def clear(self) -> None:
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.free_group = None
        self.swa_free_group = []
        self.swa_page_ids_group = []
        if self._draft_swa_sidecar is not None:
            self._draft_swa_sidecar.clear()

    def _free_page_count(self) -> int:
        return self._page_store.num_free_pages()

    def _page_ids_needed(self, num_full_pages: int, num_swa_pages: int) -> int:
        return num_full_pages + self._swa_groups * num_swa_pages

    def max_extend_tokens(self) -> int:
        return (self._free_page_count() // (1 + self._swa_groups)) * self.page_size

    def max_request_tokens(self, sliding_window_size: Optional[int]) -> int:
        """Return the maximum single-request context length that fits in this pool."""
        return _max_request_tokens(
            self.size_full,
            self.page_size,
            sliding_window_size,
            self._swa_groups,
        )

    def available_size(self) -> int:
        return self.max_extend_tokens()

    def full_available_size(self) -> int:
        return self._free_page_count() * self.page_size

    def swa_available_size(self) -> int:
        avail = (self._free_page_count() // self._swa_groups) * self.page_size
        if self._draft_swa_sidecar is not None:
            avail = min(avail, self._draft_swa_sidecar.allocator.available_size())
        return avail

    def conserve_full_available_size(self) -> int:
        return self._size_full - self.full_attn_allocator.allocated_count()

    def conserve_swa_available_size(self) -> int:
        avail = self._size_swa - self.swa_attn_allocator.allocated_count()
        if self._draft_swa_sidecar is not None:
            avail = min(avail, self._draft_swa_sidecar.allocator.available_size())
        return avail

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
        if min(num_full_pages, num_swa_pages) < 0:
            return False
        if (
            self._draft_swa_sidecar is not None
            and num_swa_pages * self.page_size
            > self._draft_swa_sidecar.allocator.available_size()
        ):
            return False
        return (
            self._page_ids_needed(num_full_pages, num_swa_pages)
            <= self._free_page_count()
        )

    def prealloc_fits_assumes_reclaim(self) -> bool:
        """Whether ``prealloc_fits`` already accounts for evictable RadixCache pages via ``can_reserve``."""
        return True

    def prealloc_ceiling_fits(
        self, full_tokens: int | float, swa_tokens: int | float
    ) -> bool:
        """Return True if a request needing ``(full_tokens, swa_tokens)`` can ever fit in an empty pool."""
        return self.can_reserve(
            full_tokens, swa_tokens, empty_pool=True, require_token_slack=True
        )

    def prealloc_fits(
        self,
        tree_cache,
        full_tokens: int,
        swa_tokens: int,
        *,
        full_budget_tokens: int,
        swa_budget_tokens: int | None = None,
    ) -> bool:
        """Price Full and SWA preallocation together against the shared page store."""
        if full_tokens > full_budget_tokens or (
            swa_budget_tokens is not None and swa_tokens > swa_budget_tokens
        ):
            return False
        full_evictable_tokens = (
            0 if tree_cache is None else tree_cache.full_evictable_size()
        )
        swa_evictable_tokens = (
            0 if tree_cache is None else tree_cache.swa_evictable_size()
        )
        full_tokens += max(
            0, self.full_available_size() + full_evictable_tokens - full_budget_tokens
        )
        if swa_budget_tokens is not None:
            swa_tokens += max(
                0,
                self.swa_available_size() + swa_evictable_tokens - swa_budget_tokens,
            )
        return self.can_reserve(
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
        )

    def _validate_draft_swa_window(self, draft_kvc) -> None:
        layer_info = getattr(draft_kvc, "layer_info", None)
        if bool(getattr(layer_info, "full_attention_layer_ids", None)) and bool(
            getattr(layer_info, "swa_attention_layer_ids", None)
        ):
            raise ValueError(
                "Mixed full and sliding-window draft layers are not supported on "
                "TpuUnifiedSWAKVPool."
            )
        draft_mc = getattr(draft_kvc, "model_config", None)
        draft_cfg_window = getattr(draft_mc, "sliding_window_size", None)
        draft_window = getattr(draft_kvc, "sliding_window_size", None)
        if draft_window is None and draft_cfg_window is None and draft_mc is not None:
            draft_window = _dflash_swa_window_left(draft_mc)
        if getattr(
            draft_kvc, "draft_swa_full_capacity", False
        ) or _draft_swa_exceeds_target_window(
            target_window=self._target_sliding_window_size,
            target_cfg_window=self._target_cfg_sliding_window,
            draft_window=draft_window,
            draft_cfg_window=draft_cfg_window,
        ):
            raise ValueError(
                "Draft sliding window must not exceed target sliding window on "
                "TpuUnifiedSWAKVPool."
            )

    def derive_draft_pool_sizes(
        self,
        draft_kvc,
        *,
        max_total_num_tokens: int,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ) -> Tuple[int, Optional[int], Optional[int]]:
        del max_total_num_tokens, full_max_total_num_tokens, swa_max_total_num_tokens
        full_tokens = self.size_full
        if not getattr(draft_kvc, "is_hybrid_swa", False):
            return full_tokens, None, None
        self._validate_draft_swa_window(draft_kvc)
        if not getattr(draft_kvc.layer_info, "swa_attention_layer_ids", None):
            return full_tokens, full_tokens, full_tokens
        draft_swa_pages = _compute_draft_swa_pages(
            full_tokens // self.page_size,
            page_size=self.page_size,
            num_swa_groups=self._swa_groups,
            sliding_window_size=(
                self._target_sliding_window_size
                if self._target_sliding_window_size is not None
                else self._target_cfg_sliding_window
            ),
        )
        swa_tokens = min(full_tokens, draft_swa_pages * self.page_size)
        return full_tokens, full_tokens, swa_tokens

    def attach_draft_kv_pool(self, draft_kvc, draft_pool) -> None:
        if not getattr(draft_kvc, "is_hybrid_swa", False):
            return
        self._validate_draft_swa_window(draft_kvc)
        if (
            getattr(draft_pool, "swa_layer_nums", 0) == 0
            or draft_pool.size_swa >= self.size_full
        ):
            draft_pool.register_mapping(self.full_to_swa_index_mapping)
            return
        if self._draft_swa_sidecar is None:
            assert self.swa_attn_allocator.allocated_pages == 0, (
                "attach_draft_kv_pool must run before SWA pages are allocated."
            )
            self._draft_swa_sidecar = _DraftSwaSidecar(
                target_size_full=self.size_full,
                draft_size_swa=draft_pool.size_swa,
                page_size=self.page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=getattr(draft_pool, "swa_kv_pool", None),
                need_sort=self.need_sort,
            )
            self.swa_attn_allocator._sidecar = self._draft_swa_sidecar
            if self.kv_free_fence is not None:
                self.kv_free_fence.attach(self._draft_swa_sidecar.allocator)
        draft_pool.register_mapping(self._draft_swa_sidecar.mapping)

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        super().set_full_to_swa_mapping(full_indices, swa_indices)
        if self._draft_swa_sidecar is not None:
            self._draft_swa_sidecar.set_mapping(full_indices, swa_indices)

    def clear_full_to_swa_mapping(self, full_indices: torch.Tensor) -> None:
        super().clear_full_to_swa_mapping(full_indices)
        if self._draft_swa_sidecar is not None:
            self._draft_swa_sidecar.clear_mapping(full_indices)

    def free_swa(self, free_index: torch.Tensor):
        if self._draft_swa_sidecar is not None and free_index.numel() > 0:
            self._draft_swa_sidecar.clear_mapping(free_index)
        return super().free_swa(free_index)

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
        out = super().alloc_decode(
            seq_lens, seq_lens_cpu, last_loc, out_indices_cpu=out_indices_cpu
        )
        if out is not None and self._draft_swa_sidecar is not None:
            full_idx = (out_indices_cpu if out_indices_cpu is not None else out).to(
                device="cpu", dtype=torch.int64
            )
            self._draft_swa_sidecar.set_mapping(
                full_idx, self.full_to_swa_index_mapping[full_idx]
            )
        return out

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
        if full_tokens < 0 or swa_tokens < 0:
            return False
        ps = self.page_size
        sidecar = self._draft_swa_sidecar
        swa_cap_tokens = (
            min(self.size_swa, sidecar.num_draft_pages * ps)
            if sidecar is not None
            else self.size_swa
        )
        if require_token_slack and (
            full_tokens >= self.size_full or swa_tokens >= swa_cap_tokens
        ):
            return False
        f_pages = (math.ceil(full_tokens) + ps - 1) // ps
        s_pages = (math.ceil(swa_tokens) + ps - 1) // ps
        need = self._page_ids_needed(f_pages, s_pages)
        if empty_pool:
            if sidecar is not None and s_pages > sidecar.num_draft_pages:
                return False
            return need <= self.size_full // ps
        ms = self._swa_groups
        max_s = min(
            self.swa_attn_allocator.allocated_pages // ms,
            max(0, int(swa_evictable_tokens)) // ps,
        )
        if sidecar is not None:
            sc_free = sidecar.allocator.available_size() // ps
            if s_pages > sc_free and s_pages > sc_free + max_s:
                return False
        free = self._free_page_count()
        if need <= free:
            return True
        max_f = min(
            self.full_attn_allocator.allocated_pages,
            max(0, int(full_evictable_tokens)) // ps,
        )
        return need <= free + self._page_ids_needed(max_f, max_s)

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
        need_f, need_s = (
            _cdiv(max(0, int(num_tokens)), ps),
            _cdiv(max(0, int(req_swa)), ps),
        )
        if self._draft_swa_sidecar is not None:
            sc_free = self._draft_swa_sidecar.allocator.available_size() // ps
            sc_short = need_s - sc_free
            if sc_short > 0:
                tree_cache.evict_for_alloc(
                    EvictParams(num_tokens=0, swa_num_tokens=sc_short * ps)
                )
        needed_pages = self._page_ids_needed(need_f, need_s)
        short = needed_pages - self._free_page_count()
        if short > 0:
            tree_cache.evict_for_alloc(
                EvictParams(num_tokens=short * ps, swa_num_tokens=0)
            )
        short = needed_pages - self._free_page_count()
        if short > 0:
            tree_cache.evict_for_alloc(
                EvictParams(
                    num_tokens=0,
                    swa_num_tokens=_cdiv(short, self._swa_groups) * ps,
                )
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


@dataclass(frozen=True)
class DraftKVPricing:
    """Per-token KV cache byte cost of an attached speculative draft model."""

    full_per_token: int = 0
    swa_compact_per_token: int = 0


def resolve_unified_draft_kv_pricing(kvc) -> DraftKVPricing:
    """Resolve exact per-token KV cache bytes for the draft worker paired with ``kvc``."""
    spec_algo = getattr(kvc, "spec_algorithm", None)
    spec_aux = getattr(kvc, "spec_aux_config", None)
    total_layers = _draft_num_layers(spec_algo, spec_aux)
    draft_mc = _resolve_draft_model_config(kvc)
    if spec_aux is None or total_layers <= 0 or draft_mc is None:
        return DraftKVPricing()

    _, draft_pool_cls, draft_dtype_str, draft_dtype = _resolve_draft_attention_info(
        kvc, draft_mc
    )
    group_size = getattr(kvc, "kv_group_size", None) or getattr(
        getattr(kvc, "server_args", None),
        "tpu_mxint8_group_size",
        MXINT8_DEFAULT_GROUP_SIZE,
    )
    draft_full_mx_layout = (
        getattr(getattr(kvc, "server_args", None), "tpu_mxint8_full_kv_layout", None)
        or "merged"
    )

    def _layer_bytes(heads: int, dim: int, v_dim: int, *, is_swa: bool) -> int:
        if draft_dtype_str == "mxint8":
            return mxint8_cache_bytes_per_token(
                heads,
                dim,
                kvc.pool_page_size,
                group_size=group_size,
                layout="merged" if is_swa else draft_full_mx_layout,
            )
        if hasattr(draft_pool_cls, "kv_bytes_per_token_per_layer"):
            return draft_pool_cls.kv_bytes_per_token_per_layer(
                num_kv_heads=heads,
                head_dim=dim,
                v_head_dim=v_dim,
                store_dtype=draft_dtype,
            )
        return heads * (dim + v_dim) * torch._utils._element_size(draft_dtype)

    tp = get_parallel().attn_tp_size
    if draft_mc.is_hybrid_swa and bool(draft_mc.swa_attention_layer_ids):
        swa_dim = draft_mc.swa_head_dim or draft_mc.head_dim
        swa_bpt = _layer_bytes(
            draft_mc.get_swa_num_kv_heads(tp),
            swa_dim,
            draft_mc.swa_v_head_dim or swa_dim,
            is_swa=True,
        )
        return DraftKVPricing(swa_compact_per_token=total_layers * swa_bpt)
    full_bpt = _layer_bytes(
        draft_mc.get_num_kv_heads(tp),
        draft_mc.head_dim,
        draft_mc.v_head_dim or draft_mc.head_dim,
        is_swa=False,
    )
    return DraftKVPricing(full_per_token=total_layers * full_bpt)


_DRAFT_SWA_HICACHE_WINDOW_MULT = 4
_DRAFT_SWA_HICACHE_CHUNK_MULT = 2


def _draft_swa_runtime_budget() -> Tuple[bool, int, int, int]:
    """Return ``(enable_hicache, per_worker_mrr, spec_draft_tokens, chunked_prefill_size)``."""
    from sglang_tpu.compilation.compiled_forward_shim import DEFAULT_TPU_COMPILE_MAX_BS

    try:
        enable_hicache = bool(get_memory().enable_hierarchical_cache)
        sched = get_schedule()
        dp = max(int(get_parallel().attn_dp_size or 1), 1)
        mrr = getattr(sched, "max_running_requests", None)
        per_worker_mrr = (
            max(int(mrr) // dp, 1) if mrr is not None else DEFAULT_TPU_COMPILE_MAX_BS
        )
        chunk_tokens = max(int(getattr(sched, "chunked_prefill_size", 0) or 0), 0)
        spec_ctx = get_spec()
        draft_tokens = max(
            int(
                getattr(spec_ctx, "speculative_num_draft_tokens", 0)
                or getattr(spec_ctx, "speculative_dflash_block_size", 0)
                or 0
            ),
            0,
        )
        return enable_hicache, per_worker_mrr, draft_tokens, chunk_tokens
    except ValueError:
        return False, DEFAULT_TPU_COMPILE_MAX_BS, 0, 0


def _compute_draft_swa_pages(
    total_target_pages: int,
    *,
    page_size: int,
    num_swa_groups: int,
    sliding_window_size: Optional[int],
) -> int:
    """Compute compact SWA draft page count for ``total_target_pages`` unified pages.

    Device-only uses the lossless bound ``floor(N_phys / (1 + M_T))`` so SWA leaf
    eviction never cascade-deletes cached Full prefixes. With HiCache enabled,
    cold SWA leaf windows spill to host RAM, so device HBM only needs
    ``4 * B * w_win + 2 * w_chunk`` pages (clamped by the lossless bound).
    """
    if total_target_pages <= 0:
        return 0
    raw_ceil = max(1, total_target_pages // (1 + num_swa_groups))
    ceil_pages = (raw_ceil // 8) * 8 if raw_ceil >= 8 else raw_ceil
    enable_hicache, per_worker_mrr, draft_tokens, chunk_tokens = (
        _draft_swa_runtime_budget()
    )
    if not enable_hicache or sliding_window_size is None:
        return ceil_pages
    w_win = _cdiv(int(sliding_window_size) + draft_tokens + page_size, page_size)
    w_chunk = _cdiv(chunk_tokens, page_size)
    working_pages = (
        _DRAFT_SWA_HICACHE_WINDOW_MULT * per_worker_mrr * w_win
        + _DRAFT_SWA_HICACHE_CHUNK_MULT * w_chunk
    )
    aligned_working = _cdiv(max(8, working_pages), 8) * 8
    return min(aligned_working, ceil_pages)


class TpuUnifiedSWAPoolConfigurator(MemoryPoolConfigurator):
    """Sizes the Logical-to-Physical unified Full+SWA KV pool from ``G = L_full``."""

    def __init__(self, kvc):
        _raise_if_unsupported(kvc)
        layout = _kvc_unified_layout(kvc)
        self._sliding_window_size = (
            kvc.sliding_window_size
            if kvc.sliding_window_size is not None
            else getattr(
                getattr(kvc, "model_config", None), "sliding_window_size", None
            )
        )
        self._num_swa_groups = layout.num_swa_groups
        self._full_layers_num = layout.num_shared_buffers
        self._full_per_token = self._compute_full_per_token(kvc, layout)
        draft_pricing = resolve_unified_draft_kv_pricing(kvc)
        self._draft_full_per_token = draft_pricing.full_per_token
        self._draft_swa_per_token = draft_pricing.swa_compact_per_token
        ps = kvc.pool_page_size
        self._page_bytes = ps * (
            self._full_per_token * self._full_layers_num + self._draft_full_per_token
        )
        self._draft_swa_page_bytes = ps * self._draft_swa_per_token
        self._reserved_bytes = self._page_bytes + self._draft_swa_page_bytes
        self._align_pages = 8

    @staticmethod
    def _compute_full_per_token(kvc, layout: UnifiedPageLayout) -> int:
        if kvc.kv_cache_dtype_str == "mxint8":
            return _mxint8_unified_bytes_per_token(kvc, layout)
        return kvc._tpu_mha_pool_class().kv_bytes_per_token_per_layer(
            num_kv_heads=layout.page_head_num,
            head_dim=layout.page_head_dim,
            v_head_dim=layout.page_head_dim,
            store_dtype=kvc.kv_cache_dtype,
        )

    def _window_overhead(self, page_size: int) -> int:
        return self._num_swa_groups * _window_tokens(
            self._sliding_window_size, page_size
        )

    def _validate_min_tokens(self, shared_tokens: int, page_size: int) -> None:
        if self._sliding_window_size is not None:
            min_tokens = (1 + self._num_swa_groups) * _window_tokens(
                self._sliding_window_size, page_size
            )
        else:
            min_tokens = (1 + self._num_swa_groups) * page_size
        if shared_tokens <= min_tokens:
            raise ValueError(
                f"Unified KV pool ({shared_tokens} page-id tokens) cannot hold "
                f"even one request: need > {min_tokens} tokens."
            )

    def calculate_total_physical_pages(
        self, available_bytes: int, page_size: int
    ) -> int:
        """Compute aligned physical page count directly from free HBM quota and Full layers."""
        from bisect import bisect_right

        usable = max(0, available_bytes - self._reserved_bytes)
        base_page_bytes = page_size * (
            self._full_per_token * self._full_layers_num + self._draft_full_per_token
        )
        if base_page_bytes <= 0:
            return 0
        draft_swa_page_bytes = page_size * self._draft_swa_per_token
        if draft_swa_page_bytes <= 0:
            return (
                (usable // base_page_bytes) // self._align_pages
            ) * self._align_pages

        def unit_cost(units: int) -> int:
            if units <= 0:
                return 0
            n_phys = units * self._align_pages
            d_pages = _compute_draft_swa_pages(
                n_phys,
                page_size=page_size,
                num_swa_groups=self._num_swa_groups,
                sliding_window_size=self._sliding_window_size,
            )
            return n_phys * base_page_bytes + d_pages * draft_swa_page_bytes

        max_units = usable // base_page_bytes // self._align_pages
        best_units = bisect_right(range(max_units + 1), usable, key=unit_cost) - 1
        return max(best_units, 0) * self._align_pages

    def max_request_tokens(self, shared_tokens: int, page_size: int) -> int:
        """Return the maximum single-request context length that fits in ``shared_tokens``
        (reserving ``M_swa`` groups only for the trailing sliding window, enabling 256K+ sessions)."""
        return _max_request_tokens(
            shared_tokens,
            page_size,
            self._sliding_window_size,
            self._num_swa_groups,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        shared_tokens = (
            self.calculate_total_physical_pages(available_bytes, page_size) * page_size
        )
        self._validate_min_tokens(shared_tokens, page_size)
        return MemoryPoolConfig(
            max_total_num_tokens=self.max_request_tokens(shared_tokens, page_size),
            full_max_total_num_tokens=shared_tokens,
            swa_max_total_num_tokens=shared_tokens,
        )

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        full_tokens = (max(0, max_total_num_tokens) // page_size) * page_size
        raw_shared = (
            full_tokens + self._window_overhead(page_size)
            if self._sliding_window_size is not None
            else full_tokens * (1 + self._num_swa_groups)
        )
        align = self._align_pages * page_size
        shared_tokens = _cdiv(raw_shared, align) * align
        self._validate_min_tokens(shared_tokens, page_size)
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=shared_tokens,
            swa_max_total_num_tokens=shared_tokens,
        )


def maybe_create_unified_pool_configurator(kvc) -> Optional[MemoryPoolConfigurator]:
    """Return ``TpuUnifiedSWAPoolConfigurator(kvc)`` when unified KV pool is enabled, else ``None``."""
    if not unified_kv_pool_enabled(kvc):
        return None
    return TpuUnifiedSWAPoolConfigurator(kvc)


def maybe_build_unified_kv_pool(
    kvc,
    *,
    full_max_total_num_tokens: Optional[int],
    swa_max_total_num_tokens: Optional[int],
) -> Optional[TpuUnifiedSWAKVPool]:
    """Build ``TpuUnifiedSWAKVPool`` for ``kvc`` when unified KV pool is enabled, else ``None``."""
    if not unified_kv_pool_enabled(kvc):
        return None
    assert full_max_total_num_tokens == swa_max_total_num_tokens
    tp = get_parallel().attn_tp_size
    dcp = get_parallel().attn_dcp_size
    head_dim = kvc.model_config.head_dim
    return TpuUnifiedSWAKVPool(
        size=full_max_total_num_tokens,
        page_size=kvc.pool_page_size,
        dtype=kvc.kv_cache_dtype,
        full=AttnLayerSpec(
            head_num=kvc.model_config.get_num_kv_heads(tp, dcp),
            head_dim=head_dim,
            layer_ids=kvc.layer_info.full_attention_layer_ids,
        ),
        swa=AttnLayerSpec(
            head_num=kvc.model_config.get_swa_num_kv_heads(tp, dcp),
            head_dim=kvc.model_config.swa_head_dim or head_dim,
            layer_ids=kvc.layer_info.swa_attention_layer_ids,
        ),
        device=kvc.device,
        token_to_kv_pool_class=kvc._tpu_mha_pool_class(),
        pool_kwargs=(
            (kvc._mxint8_pool_kwargs() or None)
            if hasattr(kvc, "_mxint8_pool_kwargs")
            else None
        ),
        start_layer=kvc.layer_info.start_layer,
        enable_memory_saver=get_exec().features.enable_memory_saver,
    )


def maybe_build_unified_kv_pool_allocator(
    kvc,
    *,
    sizes,
    token_to_kv_pool,
    req_to_token_pool,
    need_sort: bool,
) -> Optional[TpuUnifiedSWATokenToKVPoolAllocator]:
    """Build ``TpuUnifiedSWATokenToKVPoolAllocator`` for ``kvc`` when unified KV pool is enabled, else ``None``."""
    if not unified_kv_pool_enabled(kvc):
        return None
    return TpuUnifiedSWATokenToKVPoolAllocator(
        sizes.full_max_total_num_tokens,
        page_size=get_schedule().page_size,
        dtype=kvc.kv_cache_dtype,
        device=kvc.device,
        kvcache=token_to_kv_pool,
        need_sort=need_sort,
        req_to_token_pool=req_to_token_pool,
        sliding_window_size=getattr(kvc, "sliding_window_size", None),
        cfg_sliding_window=getattr(
            getattr(kvc, "model_config", None), "sliding_window_size", None
        ),
    )
