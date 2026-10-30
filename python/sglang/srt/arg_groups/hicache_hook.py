# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the hierarchical KV cache."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    declare_resolution,
    model_config_of,
    resolving_view,
    use_mla_backend,
)

logger = logging.getLogger(__name__)


def handle_hicache(server_args: Any):
    """Normalize hicache-related knobs into a valid runtime configuration.

    Resolution order:
    1) Layout <-> I/O compatibility for direct conflicts.
    2) Storage <-> layout compatibility (may rewrite layout).
    """
    cfg = resolving_view(server_args)
    if cfg.enable_linker_mla_dedup and (
        not cfg.enable_unified_cache_external_linker
        or cfg.unified_cache_external_linker_backend != "mooncake"
    ):
        raise ValueError("--enable-linker-mla-dedup requires the Mooncake linker.")
    if cfg.enable_unified_cache_external_linker:
        if cfg.enable_hierarchical_cache:
            raise ValueError(
                "--enable-unified-cache-external-linker and "
                "--enable-hierarchical-cache are mutually exclusive."
            )
        if cfg.hicache_storage_backend is not None:
            raise ValueError(
                "--enable-unified-cache-external-linker does not use "
                "--hicache-storage-backend."
            )
        return

    # Skip all normalization when neither hicache nor decode-offload path is active.
    if not (
        cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
        or (
            cfg.disaggregation_mode == "decode"
            and cfg.disaggregation_decode_retraction_backup in (None, "host_pool")
        )
    ):
        return

    validate_hicache_host_memory_mode(server_args)

    if cfg.disaggregation_decode_host_receive_threshold > 0:
        # The shared host pool must match KV transfer's per-layer buffers.
        declare_resolution(
            server_args, "handle_hicache", hicache_mem_layout="layer_first"
        )

    # Step 1: Initial layout-io compatibility normalization.
    resolve_layout_io_compatibility(server_args)

    # Step 2: Storage-layout normalization without changing io backend.
    resolve_storage_layout_compatibility(server_args)
    if (
        cfg.disaggregation_decode_host_receive_threshold > 0
        and cfg.hicache_mem_layout != "layer_first"
    ):
        raise ValueError(
            f"The resolved HiCache storage layout {cfg.hicache_mem_layout!r} "
            "cannot share the layer_first decode host pool used by KV transfer"
        )

    # Step 3: DCP compatibility for the L2 (device<->host) path.
    resolve_hicache_dcp_compatibility(server_args)


def handle_hicache_ratio_default(server_args: Any):
    """Default the host/device ratio per host memory mode.

    Runs before the dummy-model boundary: direct HostKVCache consumers
    (unit fixtures, dummy-model launches) must never see a None ratio.
    buffer_only stages in flight rather than retaining, so it needs only
    enough to cover the write backlog plus parked prefetches.

    A decode server keeps the ratio unset here: kv_cache_builder resolves
    it against the retraction-backup backend (1.0 for host_pool, else 2.0).

    An explicit --hicache-ratio or --hicache-size is honored as given, so it
    resolves --hicache-host-memory-fraction to None (auto-sizing off).
    """
    cfg = resolving_view(server_args)
    fraction = cfg.hicache_host_memory_fraction
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("--hicache-host-memory-fraction must be in (0, 1].")
    fields = {}
    if cfg.hicache_ratio is None and cfg.disaggregation_mode != "decode":
        fields["hicache_ratio"] = (
            1.2 if cfg.hicache_host_memory_mode == "buffer_only" else 2.0
        )
    if cfg.hicache_ratio is not None or cfg.hicache_size > 0:
        fields["hicache_host_memory_fraction"] = None
    if fields:
        declare_resolution(server_args, "_handle_hicache_ratio_default", **fields)


def resolve_hicache_dcp_compatibility(server_args: Any):

    cfg = resolving_view(server_args)
    if cfg.dcp_size <= 1 or not cfg.enable_hierarchical_cache:
        return
    if cfg.hicache_storage_backend is not None:
        raise NotImplementedError(
            "--hicache-storage-backend (L3) with --dcp-size > 1 is not "
            "supported yet: under DCP each rank holds a distinct "
            "interleaved MLA KV shard, so the rank-0-only replicated-MLA "
            "backup and the storage keys must become dcp_rank-aware "
            "first. Run HiCache+DCP with L1/L2 only."
        )
    if cfg.speculative_algorithm not in (None, "DSPARK"):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 only supports DSPARK speculative "
            "decoding; other draft-model host pools have no DCP index "
            "translation."
        )
    if cfg.enable_lmcache:
        raise NotImplementedError(
            "--enable-lmcache with --dcp-size > 1 is not supported: "
            "LMCache has no DCP-aware index translation."
        )
    if cfg.enable_hisparse:
        raise NotImplementedError(
            "--enable-hisparse with --dcp-size > 1 is not supported: the "
            "HiSparse host pool is constructed without DCP translation."
        )
    if not use_mla_backend(server_args):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 is only supported for MLA models: "
            "the index translation lives in MLATokenToKVPoolHost, and the "
            "MHA host pool has none."
        )
    logger.info(
        "HiCache + DCP enabled (L1/L2 only): host pool uses widened "
        "logical slot accounting with per-rank physical translation at "
        "the transfer boundary (dcp_size=%d).",
        cfg.dcp_size,
    )


def resolve_layout_io_compatibility(server_args: Any):
    from sglang_tpu.tpu_dma_hicache import is_tpu_dma_hicache_backend

    cfg = resolving_view(server_args)
    tpu_dma = is_tpu_dma_hicache_backend(cfg.hicache_io_backend)
    if cfg.hicache_io_backend == "torch_tpu" or tpu_dma:
        # The TPU fused host pool (torch_tpu) and raiden both mirror the fused
        # device pool, which only supports the "page_first_fused" layout.
        if cfg.hicache_mem_layout != "page_first_fused":
            raise ValueError(
                f"{cfg.hicache_io_backend} io backend requires "
                "--hicache-mem-layout page_first_fused, but got "
                f"{cfg.hicache_mem_layout!r}. "
                "Re-run with --hicache-mem-layout page_first_fused."
            )

        # TODO: support more backend but for now left a guard for other configs
        if use_mla_backend(server_args):
            raise ValueError(
                f"{cfg.hicache_io_backend} KV offloading supports only dense "
                "MHA models; this is an MLA model, which has no fused TPU host "
                "pool. Disable --enable-hierarchical-cache, or serve without "
                "the TPU KV-offload io backend."
            )

        # Offload copies the device KV pool, so its class declares the io
        # backends that can copy its layout. Map backend -> pool statically
        # (the runtime context does not exist yet), and read both helpers
        # through the module, because plugins patch them to register their
        # own backends and pools.
        from sglang_tpu import tpu_platform

        pool_cls = tpu_platform._mha_kv_pool_cls(cfg.attention_backend)
        if cfg.hicache_io_backend not in getattr(
            pool_cls, "kv_offload_io_backends", ()
        ):
            raise ValueError(
                f"{cfg.hicache_io_backend} KV offloading does not support the "
                f"{pool_cls.__name__} KV pool of attention backend "
                f"{cfg.attention_backend!r}: the pool does not list this io "
                "backend in kv_offload_io_backends. Re-run with an attention "
                "backend whose pool does (in-tree: pallas_tpu_rpa_v3 or "
                "pallas_tpu_rpa_batched), or disable KV offload."
            )

        # Reloaded KV lands through the reload-commit hook in the shared RPA
        # backend base, so every backend that reads the pool must be an RPA
        # backend. It must also read the pool's layout: the pool is built for
        # the base backend only, and split prefill/decode backends share it.
        for backend in dict.fromkeys(
            (cfg.attention_backend, *attention_backends_of(cfg))
        ):
            if tpu_platform._rpa_kernel_spec(backend) is None:
                raise ValueError(
                    f"{cfg.hicache_io_backend} KV offloading does not support "
                    f"attention backend {backend!r}: it is not a Pallas RPA "
                    "backend, so it never runs the HiCache reload-commit hook. "
                    "Re-run with RPA attention backends, or disable KV offload."
                )
            backend_pool_cls = tpu_platform._mha_kv_pool_cls(backend)
            if backend_pool_cls is not pool_cls:
                raise ValueError(
                    f"{cfg.hicache_io_backend} KV offloading does not support "
                    f"attention backend {backend!r} with --attention-backend "
                    f"{cfg.attention_backend!r}: it reads "
                    f"{backend_pool_cls.__name__}, but "
                    f"the shared KV pool is {pool_cls.__name__}. Re-run with "
                    "backends that use the same KV pool, or disable KV offload."
                )

        if (
            cfg.hicache_io_backend == "torch_tpu"
            and model_config_of(server_args).is_hybrid_swa
        ):
            raise ValueError(
                "torch_tpu KV offloading does not support hybrid "
                "sliding-window models: its host pool mirrors only the "
                "full-attention pool, and the SWA sub-pool has its own "
                "page-id space with no host storage. Re-run with "
                "--hicache-io-backend raiden (one host pool per sub-pool), "
                "with --disable-hybrid-swa-memory (sliding layers then "
                "share the full-size pool), or without "
                "--enable-hierarchical-cache."
            )

    if tpu_dma and cfg.hicache_storage_backend is not None:
        raise ValueError(
            f"{cfg.hicache_io_backend} io backend supports only L2 (TPU<->host) "
            "offload and has no L3 storage path. Remove --hicache-storage-backend "
            f"({cfg.hicache_storage_backend!r}) or choose a different "
            "--hicache-io-backend."
        )

    if (
        cfg.hicache_mem_layout == "page_first_direct"
        and cfg.hicache_io_backend == "kernel"
    ):
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_io_backend="direct",
        )
        logger.warning(
            "Kernel io backend does not support page first direct layout, switching to direct io backend"
        )

    if cfg.hicache_mem_layout == "page_first" and cfg.hicache_io_backend == "direct":
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_mem_layout="page_first_direct",
        )
        logger.warning(
            "Page first layout is not supported with direct IO backend, switching to page first direct layout"
        )


def resolve_storage_layout_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_storage_backend not in ("mooncake", "npu_memcache")
        or cfg.hicache_mem_layout != "layer_first"
    ):
        return

    if cfg.hicache_io_backend == "direct":
        new_layout = "page_first_direct"
    elif cfg.hicache_io_backend == "kernel":
        new_layout = "page_first"
    else:
        # Keep current behavior for unknown backends (e.g., kernel_ascend).
        new_layout = cfg.hicache_mem_layout

    declare_resolution(
        server_args,
        "_resolve_storage_layout_compatibility",
        hicache_mem_layout=new_layout,
    )
    logger.warning(
        f"Mooncake/Ascend MemCache storage backend does not support layer_first layout, "
        f"switching to {new_layout} layout for {cfg.hicache_io_backend} io backend"
    )


def validate_hicache_host_memory_mode(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.hicache_host_memory_mode not in ("cache", "buffer_only"):
        raise ValueError(
            "hicache_host_memory_mode must be 'cache' or 'buffer_only', "
            f"got {cfg.hicache_host_memory_mode!r}"
        )

    # Both modes are defaulted upstream (a decode server resolves the
    # ratio later, in kv_cache_builder), so this fires only if that
    # defaulting regresses -- never build an unsized host pool.
    if (
        cfg.hicache_size <= 0
        and cfg.hicache_ratio is None
        and cfg.disaggregation_mode != "decode"
    ):
        raise ValueError(
            f"--hicache-host-memory-mode {cfg.hicache_host_memory_mode} "
            "requires a host pool size: pass --hicache-size or "
            "--hicache-ratio."
        )

    if cfg.hicache_host_memory_mode == "cache":
        return

    if cfg.hicache_storage_backend is None:
        raise ValueError(
            "--hicache-host-memory-mode buffer_only requires a storage backend "
            "(--hicache-storage-backend): host memory is only a staging buffer "
            "and all cached data lives in storage."
        )
    if cfg.hicache_write_policy == "write_back":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only does not support "
            "--hicache-write-policy write_back; use write_through or "
            "write_through_selective."
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only is not supported on "
            "decode instances: the decode-side prefetch and offload paths "
            "bypass the buffer-mode pipeline, fetching without its prefix "
            "context and never consuming its staged holds. Prefill "
            "instances share the standard scheduler path and are supported."
        )
