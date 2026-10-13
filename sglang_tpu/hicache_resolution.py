# SPDX-License-Identifier: Apache-2.0
"""HiCache server-argument checks for the TPU io backends (torch_tpu, raiden).

Registered as an override of the core ``handle_hicache`` resolution step, so the
core step knows nothing about TPU pools and kernels. ``tpu_platform`` imports
this module, which puts the override in place whenever the TPU platform is
loaded, before the resolution pipeline reaches ``handle_hicache``.

Only ``resolution_hooks`` (no sglang imports of its own) is imported at module
level: ``tpu_platform`` is loaded during platform discovery, and the rest of
``arg_groups`` reads ``current_platform`` on import, which would re-enter
discovery before ``TpuSRTPlatform`` exists.
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

_TPU_HICACHE_IO_BACKENDS = ("torch_tpu", "raiden")


@register_resolution_hook("handle_hicache")
def handle_tpu_hicache(server_args: Any, previous: Callable[[Any], None]) -> None:
    from sglang.srt.arg_groups.hicache_hook import hicache_active
    from sglang.srt.arg_groups.overrides import resolving_view

    previous(server_args)
    cfg = resolving_view(server_args)
    if cfg.hicache_io_backend in _TPU_HICACHE_IO_BACKENDS and hicache_active(
        server_args
    ):
        _validate_tpu_hicache_io_backend(server_args)


def _validate_tpu_hicache_io_backend(server_args: Any) -> None:
    from sglang.srt.arg_groups.overrides import (
        attention_backends_of,
        model_config_of,
        resolving_view,
        use_mla_backend,
    )

    cfg = resolving_view(server_args)
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
    if cfg.hicache_io_backend not in getattr(pool_cls, "kv_offload_io_backends", ()):
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
    for backend in dict.fromkeys((cfg.attention_backend, *attention_backends_of(cfg))):
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

    if cfg.hicache_io_backend == "raiden" and cfg.hicache_storage_backend is not None:
        raise ValueError(
            "raiden io backend supports only L2 (TPU<->host) offload and has "
            "no L3 storage path. Remove --hicache-storage-backend "
            f"({cfg.hicache_storage_backend!r}) or choose a different "
            "--hicache-io-backend."
        )
