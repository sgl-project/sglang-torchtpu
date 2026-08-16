# SPDX-License-Identifier: Apache-2.0
"""TPU platform backend for SGLang SRT using torch_tpu.

Implements the SRTPlatform interface so that SGLang can run inference
on Google Cloud TPUs via the torch_tpu (PyTorch privateuse1) backend.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Type

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.server_args import ServerArgs

logger = get_logger(__name__)


# Default size for libtpu's premapped (pinned, DMA-mapped) host staging pool.
_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
# Flush torch_tpu's lazy host->device weight-copy graph once this much work accumulates,
# so it materializes in chunks rather than all at once, which may cause HBM OOMs.
_LOAD_WEIGHT_SYNC_BYTES = 2 << 30  # 2 GiB
_LOAD_WEIGHT_SYNC_TENSORS = 64


def _sync_weight_stream(weights):
    """Yield weights and periodically drain torch_tpu pending work.

    ``torch.tpu.synchronize()`` blocks until async device work and deferred
    operations on the TPU have completed, which lets transient buffers be freed.
    This bounds HBM peak usage only when the loader copies parameters while
    iterating.

    Warning: when ``weights`` is a one-pass iterator/generator, some loaders call
    ``list(weights)`` before loading parameters. That consumes the entire stream
    before any real parameter copy happens, so the periodic syncs cannot bound the
    HBM peak.
    """
    pending_bytes = 0
    pending_tensors = 0
    for name, tensor in weights:
        yield name, tensor
        pending_tensors += 1
        pending_bytes += tensor.numel() * tensor.element_size()
        if (
            pending_bytes >= _LOAD_WEIGHT_SYNC_BYTES
            or pending_tensors >= _LOAD_WEIGHT_SYNC_TENSORS
        ):
            torch.tpu.synchronize()
            pending_bytes = 0
            pending_tensors = 0


# Page size used when the user does not pick one and the attention backend has
# no floor of its own. Token-granular pages (SGLang's global default of 1) turn
# each KV fetch into many tiny DMAs, which is fine for GPU kernels and wasteful
# here.
_DEFAULT_PAGE_SIZE = 64


def _rpa_kernel_spec(backend: str | None):
    """``backend``'s ``RpaKernelSpec``, or None if it is not an RPA backend.

    The Pallas RPA backends share one kernel shell, so what the platform needs
    to know about them -- the page-size floor, whether they read fp8 KV -- is
    declared on the spec rather than restated here.
    """
    if backend == "pallas_tpu_rpa_batched":
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_batched import (
            RPA_BATCHED_SPEC as spec,
        )
    elif backend == "pallas_tpu_rpa_v3":
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
            RPA_V3_SPEC as spec,
        )
    else:
        return None
    return spec


def _uses_seq_along_lane_kv() -> bool:
    """Ask the backend, so the pool cannot disagree with the kernel."""
    from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_batched import (
        uses_seq_along_lane_kv,
    )

    return uses_seq_along_lane_kv()


def _apply_page_size_default(server_args) -> None:
    """Resolve ``page_size`` against the attention backend's declared floor.

    A backend whose kernel cannot address small pages -- the SEQ_ALONG_LANE
    path tiles the page axis onto lanes and needs a multiple of the 128-lane
    width -- must not silently receive the generic default, or it fails deep
    inside the Pallas trace with a shape error that reads like a kernel bug.
    """
    from sglang.srt.arg_groups.overrides import declare_late_resolution

    spec = _rpa_kernel_spec(server_args.attention_backend)
    floor = spec.min_page_size if spec is not None else None
    if server_args.page_size is None:
        default = max(_DEFAULT_PAGE_SIZE, floor or 0)
        declare_late_resolution(
            server_args, "tpu.apply_server_args_defaults", page_size=default
        )
        return
    if floor is not None and server_args.page_size < floor:
        raise ValueError(
            f"--attention-backend {server_args.attention_backend} needs "
            f"--page-size >= {floor}, got {server_args.page_size}."
        )


def _apply_kv_cache_dtype_default(server_args) -> None:
    """Default the KV cache to fp8 for the Pallas RPA backends.

    Those kernels dequantize KV on read, and at the head counts TPU serving
    runs (tensor parallel leaves one KV head per rank) fp8 halves KV bytes per
    token. That doubles the context or the batch that fits in HBM, which is the
    binding constraint at long context -- decode there is bound by streaming
    KV, not by arithmetic. "auto" (model dtype) is the generic SGLang default
    and gives up both.

    Only those backends: ``pallas_tpu_mla`` rejects an fp8 KV cache outright,
    and ``pallas_tpu_flash`` has no dequantization scale, so defaulting either
    to fp8 breaks or silently degrades it.
    """
    from sglang.srt.arg_groups.overrides import declare_late_resolution

    if server_args.kv_cache_dtype != "auto":
        return
    if _rpa_kernel_spec(server_args.attention_backend) is None:
        return
    declare_late_resolution(
        server_args, "tpu.apply_server_args_defaults", kv_cache_dtype="fp8_e4m3"
    )


class TpuSRTPlatform(SRTPlatform):
    """SGLang SRT platform for Google Cloud TPUs via torch_tpu."""

    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    supported_quantization: list[str] = []

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args: ServerArgs) -> None:
        # Pre-publish resolution writes go through declare_late_resolution.
        from sglang.srt.arg_groups.overrides import declare_late_resolution
        from sglang.srt.hardware_backend.tpu.patches import _prepare_tpu_env

        # Under Ray the slicebuilder/topology env is built per-rank after the
        # cluster's real node IPs are discovered (multihost: per-actor from Ray
        # IPs in _prepare_tpu_env; single-host: seeded once in
        # RayEngine._launch_scheduler_processes). Building it here -- in the
        # orchestrator, before discovery -- would emit a bogus single-host
        # (localhost) env that never applies to the actual ranks. Still apply
        # the libtpu/env overrides now so actors inherit them before importing
        # torch_tpu.
        _prepare_tpu_env(
            server_args.tp_size, setup_slicebuilder=not server_args.use_ray
        )

        if server_args.dtype is None or server_args.dtype == "auto":
            declare_late_resolution(
                server_args, "tpu.apply_server_args_defaults", dtype="bfloat16"
            )

        if server_args.attention_backend is None:
            declare_late_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                attention_backend=self.get_default_attention_backend(),
            )

        if server_args.sampling_backend is None:
            declare_late_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                sampling_backend="tpu",
            )

        declare_late_resolution(
            server_args,
            "tpu.apply_server_args_defaults",
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
        )
        # ``enable_torch_compile`` is honored on TPU and wired up by
        # ``TpuGraphRunner`` (see ``ModelRunner.init_device_graphs``). Off by
        # default (the SGLang-wide default); opt in with ``--enable-torch-compile``.
        # TPU runs SGLang's regular event-based overlap scheduling, minus the
        # CUDA-style forward/copy/schedule streams: torch_tpu serialises on a
        # single PJRT queue so cross-stream ordering is a no-op (see
        # ``Scheduler.init_overlap``'s ``use_overlap_streams``). Overlap comes
        # from PJRT's async dispatch and a scoped ``torch.tpu.Event`` D2H fence.
        # Leave it enabled by default; users can opt out via
        # ``--disable-overlap-schedule``.

        if server_args.mem_fraction_static is None:
            declare_late_resolution(
                server_args, "tpu.apply_server_args_defaults", mem_fraction_static=0.40
            )

        # raiden's load-back keeps the reloaded prefix's slot indices on CPU
        # (TpuRaidenHiCacheController.load), so req_to_token must stay
        # host-resident: a device-resident req_to_token would put the other
        # nodes' prefix indices on device and mix devices in the prefix cat. Pin
        # SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1 for raiden.
        if server_args.hicache_io_backend == "raiden":
            if os.environ.get("SGLANG_TPU_REQ_TO_TOKEN_ON_HOST") == "0":
                logger.warning(
                    "Overriding SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=0 -> 1: the "
                    "raiden KV-offload backend requires host-resident "
                    "req_to_token."
                )
            os.environ["SGLANG_TPU_REQ_TO_TOKEN_ON_HOST"] = "1"

        # The default TPU backend (Pallas RPA v3) expects a paged KV cache where
        # page_size is the HBM<->VMEM DMA chunk and the fused KV buffer's page
        # axis. SGLang's global page_size=1 default is valid for token-granular
        # GPU kernels, but is inefficient here: it makes page_indices
        # token-granular and turns KV fetches into many tiny DMAs.
        # Builtin TPU speculative decoding currently supports EAGLE3 only.
        # Registered algorithms own their validation and backend integration.
        spec_algo = server_args.speculative_algorithm
        if spec_algo is not None:
            if server_args.attention_backend == "pallas_tpu_rpa_batched":
                # It has no draft-backend wiring in speculative.draft_utils,
                # so reject here with a clear message rather than failing later
                # during draft-backend creation.
                raise NotImplementedError(
                    "Speculative decoding is not supported with "
                    "--attention-backend pallas_tpu_rpa_batched yet. "
                    "Use --attention-backend pallas_tpu_rpa_v3 for speculative "
                    "decoding."
                )
            if spec_algo == "EAGLE3":
                if server_args.speculative_use_rejection_sampling:
                    raise NotImplementedError(
                        "EAGLE3 rejection sampling is not supported on TPU yet."
                    )
                if (server_args.speculative_eagle_topk or 1) > 1:
                    raise NotImplementedError(
                        "EAGLE3 topk > 1 is not supported on TPU yet."
                    )
            else:
                from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

                # Unknown names may be registered by a plugin after platform
                # defaults run. The shared speculative registry validates them
                # later; only reject builtins the TPU backend knows it cannot run.
                if spec_algo.upper() in SpeculativeAlgorithm.__members__:
                    raise NotImplementedError(
                        f"Speculative algorithm {spec_algo!r} is not supported on TPU "
                        "yet (only EAGLE3 among builtin algorithms)."
                    )

        if server_args.enable_dp_attention:
            attn_tp_size = server_args.tp_size // server_args.dp_size
            if attn_tp_size > 1 and not server_args.enable_dp_lm_head:
                raise ValueError(
                    f"When attention TP > 1 under DP attention on TPU (attn_tp_size={attn_tp_size} "
                    f"from tp_size={server_args.tp_size} and dp_size={server_args.dp_size}), "
                    "--enable-dp-lm-head must be enabled."
                )

        _apply_page_size_default(server_args)
        _apply_kv_cache_dtype_default(server_args)

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        # Three Pallas backends are registered for TPU:
        #   - "pallas_tpu_rpa_v3" (default): ragged_paged_attention v3 kernel
        #     in `pallas/ragged_paged_attention_v3.py`. Single unified kernel
        #     for prefill + decode, fused KV cache (interleaved K/V, packed),
        #     decode-native batched call.
        #   - "pallas_tpu_rpa_batched": the batched RPA kernel in
        #     `pallas/batched_rpa`, a thin subclass of the same shared base as
        #     v3 (see `attention_pallas_rpa_base`). Same fused KV cache, batches
        #     sequences for higher throughput. No speculative decoding.
        #   - "pallas_tpu_flash": flash_attention for prefill/extend +
        #     `jax.experimental.pallas.ops.tpu.paged_attention` for decode.
        #     Original implementation from PR #5, works against the flat
        #     `MHATokenToKVPool` layout (no fused-pool swap required).
        # All are listed in `server_args.ATTENTION_BACKEND_CHOICES` and
        # registered in `layers.attention.attention_registry` (same place as
        # `ascend` / `intel_xpu`). Factories use lazy imports so non-TPU runs
        # don't pull in jax/torch_tpu.
        # Override to the reference path with `--attention-backend torch_native`.
        return "pallas_tpu_rpa_v3"

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.model_executor.tpu_graph_runner import TpuGraphRunner

        return TpuGraphRunner

    def get_quantization_config(
        self, quantization: str
    ) -> Optional[Type[QuantizationConfig]]:
        if quantization == "modelopt_fp4":
            from sglang.srt.layers.quantization.modelopt_quant import (
                TpuModelOptCompatibleFp4Config,
            )

            return TpuModelOptCompatibleFp4Config
        if quantization == "modelopt_mixed":
            from sglang.srt.layers.quantization.modelopt_quant import (
                TpuModelOptMixedPrecisionConfig,
            )

            return TpuModelOptMixedPrecisionConfig
        return None

    def get_mha_kv_pool_cls(self) -> type:
        # RPA v3 uses the fused 5D KV layout; pallas_tpu_flash and torch_native
        # read the flat MHATokenToKVPool. Pick per the resolved attention backend
        # — it's on the (scheduler-child) global server args by pool-creation
        # time (set in ModelRunner.__init__, before _init_pools).
        from sglang.srt.server_args import get_global_server_args

        backend = get_global_server_args().attention_backend
        if backend == "pallas_tpu_rpa_batched" and _uses_seq_along_lane_kv():
            # SEQ_ALONG_LANE puts the token index on the lane axis, so the
            # buffer shape differs from the fused one the other backends read.
            from sglang.srt.hardware_backend.tpu.kv_pool import (
                TpuSeqAlongLaneKVPool,
            )

            return TpuSeqAlongLaneKVPool

        if backend in (
            "pallas_tpu_rpa_v3",
            "pallas_tpu_rpa_batched",
        ):
            from sglang.srt.hardware_backend.tpu.kv_pool import (
                TpuFusedMHATokenToKVPool,
            )

            return TpuFusedMHATokenToKVPool

        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_mamba_pool_cls(self) -> type:
        # Hybrid GDN models (Qwen3.5 / Qwen3-Next) store their recurrent state
        # in a TPU-native, Pallas-kernel-layout pool.
        from sglang.srt.hardware_backend.tpu.mamba_pool import TpuMambaPool

        return TpuMambaPool

    def get_mha_kv_pool_host_cls(self, device_pool) -> type:
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
        from sglang.srt.mem_cache.memory_pool_host import (
            TpuFusedMHATokenToKVPoolHost,
        )

        # Upstream relocated MHATokenToKVPoolHost from mem_cache.memory_pool_host
        # into the mem_cache.pool_host package; TpuFusedMHATokenToKVPoolHost is the
        # fork's own class and still lives in memory_pool_host.
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.server_args import get_global_server_args

        # raiden offloads through tpu-raiden's DMA-pinned host pool instead of the
        # torch-copy fused host pool; native (torch_tpu) keeps the fused pool.
        if get_global_server_args().hicache_io_backend == "raiden":
            from sglang.srt.hardware_backend.tpu.kv_pool_host import (
                TpuRaidenHostKVCache,
            )

            return TpuRaidenHostKVCache

        if isinstance(device_pool, TpuFusedMHATokenToKVPool):
            return TpuFusedMHATokenToKVPoolHost
        return MHATokenToKVPoolHost

    def get_mla_kv_pool_cls(self) -> type:
        # pallas_tpu_mla uses the paged fused latent layout the MLA v2 kernel
        # addresses; other backends (torch_native reference) keep the flat
        # upstream pool.
        from sglang.srt.runtime_context import get_exec

        if get_exec().kernel.attention_backend == "pallas_tpu_mla":
            from sglang.srt.hardware_backend.tpu.kv_pool import (
                TpuFusedMLATokenToKVPool,
            )

            return TpuFusedMLATokenToKVPool

        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_nsa_kv_pool_cls(self) -> type:
        raise NotImplementedError("NSA is not supported on TPU.")

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.hardware_backend.tpu.allocator_tpu import (
            TpuPagedTokenToKVPoolAllocator,
        )

        return TpuPagedTokenToKVPoolAllocator

    def get_compile_backend(self, mode: str | None = None) -> str:
        return "tpu"

    def supports_torch_compile(self) -> bool:
        return True

    def get_piecewise_backend_cls(self) -> type:
        raise NotImplementedError("TPU piecewise backend not yet implemented.")

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        return False

    def is_pin_memory_available(self, device=None) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def supports_overlap_streams(self) -> bool:
        # torch_tpu serialises all device work on a single implicit PJRT queue:
        # cross-stream ordering (wait_stream / wait_event) is a no-op and a
        # non-default stream only relabels async-copy fences. Overlap comes from
        # async PJRT dispatch + a scoped torch.tpu.Event D2H fence instead. Flip
        # to True once torch_tpu exposes real multi-stream execution.
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        if getattr(self, "_backend_initialized", False):
            return

        world_size = os.environ.get("WORLD_SIZE")
        if world_size is None:
            raise RuntimeError(
                "TorchTPU backend initialization requires worker WORLD_SIZE. "
                "Initialize it from the scheduler worker bootstrap."
            )

        from sglang.srt.hardware_backend.tpu.compile_cache import (
            add_pallas_kernel_sources_to_compile_cache_key,
        )

        add_pallas_kernel_sources_to_compile_cache_key()

        # Keep c10d_functional collectives inside the compiled graph. torch_tpu
        # defaults to materializing collective tensors by forcing graph breaks, which
        # crashes under torch.compile(fullgraph=True) at TP>1. Setting this to 0/false
        # disables that guardrail for SPMD TP graphs.
        os.environ.setdefault(
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false"
        )

        # Size libtpu's premapped (pinned, DMA-mapped) host staging pool. This is
        # read by libtpu at init, so it MUST be set before `import torch_tpu`.
        os.environ.setdefault(
            "TPU_PREMAPPED_BUFFER_SIZE", str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE)
        )
        premapped_bytes = int(os.environ["TPU_PREMAPPED_BUFFER_SIZE"])
        # Log the effective size so it's in hand when debugging KV-offload OOMs.
        logger.info(
            "libtpu premapped host staging pool: %.2f GiB "
            "(TPU_PREMAPPED_BUFFER_SIZE=%d); raise it if KV-offload H2D OOMs.",
            premapped_bytes / (1024 * 1024 * 1024),
            premapped_bytes,
        )

        from sglang.srt.hardware_backend.tpu.patches import (
            _prepare_tpu_env,
            apply_tpu_patches,
        )

        # WORLD_SIZE is authoritative only inside a scheduler worker. Validate
        # the inherited launch environment here; any TorchTPU-dependent
        # topology discovery is therefore confined to the worker.
        _prepare_tpu_env(int(world_size), resolve_single_host_topology=True)

        import torch_tpu  # noqa: F401

        apply_tpu_patches()
        self._backend_initialized = True

    # ------------------------------------------------------------------
    # MultiPlatformOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        return "tpu"

    # ------------------------------------------------------------------
    # Device identity & operations (DeviceMixin)
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        # Per-process HBM limit. Each worker is pinned to one core
        # (TPU_VISIBLE_CHIPS=local_rank), so this already reflects v7 megacore
        # sharing (2 cores split a chip's HBM) — no cores-per-chip math needed.
        _, limit = torch.accelerator.get_memory_info(self.get_device(device_id))
        return int(limit)

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        if device is None:
            device = self.get_device(0)
        free, total = torch.accelerator.get_memory_info(device)
        return float(total - free)

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device("tpu", device_id)

    def set_device(self, device: torch.device) -> None:
        torch.tpu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        try:
            from torch_tpu._internal.utils import hardware

            return hardware.get_tpu_device_name()
        except Exception:
            return "tpu"

    def get_device_uuid(self, device_id: int = 0) -> str:
        return f"tpu:{device_id}"

    def get_device_capability(self, device_id: int = 0):
        return None

    def synchronize(self) -> None:
        torch.tpu.synchronize()

    @contextmanager
    def weight_loading_context(self, weights):
        # No eager drain at load exit: torch.tpu.synchronize() here deadlocks
        # EAGLE3 draft init under TP>1 -- rank 0 blocks on an async buffer while
        # peers reach the post-load barrier. Let it resolve at the first aligned
        # sync instead. Periodic syncs in _sync_weight_stream still bound HBM
        # during large loads.
        yield _sync_weight_stream(weights)

    def empty_cache(self) -> None:
        pass

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        total = self.get_device_total_memory(device_id)
        used = int(self.get_current_memory_usage())
        return (total - used, total)

    def get_torch_distributed_backend_str(self) -> str:
        return "tpu_dist"

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.tpu.manual_seed_all(seed)
