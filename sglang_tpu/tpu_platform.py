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

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang_tpu._logging import get_logger

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

    ``torch.accelerator.synchronize()`` blocks until async device work and deferred
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
            torch.accelerator.synchronize()
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
        from sglang_tpu.attention_pallas_rpa_batched import RPA_BATCHED_SPEC as spec
    elif backend == "pallas_tpu_rpa_v3":
        from sglang_tpu.attention_pallas_rpa_v3 import RPA_V3_SPEC as spec
    else:
        return None
    return spec


def _uses_seq_along_lane_kv() -> bool:
    """Ask the backend, so the pool cannot disagree with the kernel."""
    from sglang_tpu.attention_pallas_rpa_batched import (
        uses_seq_along_lane_kv,
    )

    return uses_seq_along_lane_kv()


def _mha_kv_pool_cls(backend: str | None) -> type:
    """The MHA KV pool class ``backend`` allocates.

    A function of the backend name alone, so server-args validation can ask
    before the runtime context exists; ``get_mha_kv_pool_cls`` delegates here
    so the two cannot disagree. A plugin that routes its backend to its own
    pool patches this function.
    """
    if backend == "pallas_tpu_rpa_batched" and _uses_seq_along_lane_kv():
        # SEQ_ALONG_LANE puts the token index on the lane axis, so the
        # buffer shape differs from the fused one the other backends read.
        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool

        return TpuSeqAlongLaneKVPool

    # RPA v3 and batched use the fused 5D KV layout; pallas_tpu_flash and
    # torch_native read the generic MHA pool.
    if backend in (
        "pallas_tpu_rpa_v3",
        "pallas_tpu_rpa_batched",
    ):
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        return TpuFusedMHATokenToKVPool

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool


def _validate_pipeline_parallel_args(server_args: ServerArgs) -> None:
    """Reject server configurations pipeline parallelism on TPU cannot run,
    and pin the scheduler order the stage hand-off needs.

    Every pipeline stage is one process on one device. Activations move over
    ICI as one collective-permute launched by every stage in lockstep, so a
    stage's forward waits on launches the other stages issue at the top of
    their next iteration. The scheduler must therefore receive and forward
    the last stage's outputs before it launches a forward, not after: that is
    the async-batch-depth 1 order (depth 0 does the ring exchange after the
    forward and the two waits form a cycle). Tensor parallelism inside a stage
    would need the stages to share a second device collective and is rejected.
    """
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    pp = cfg.pp_size
    if cfg.pp_async_batch_depth > 1:
        raise NotImplementedError(
            "Pipeline parallelism on TPU requires --pp-async-batch-depth 1 "
            "so output delivery and activation hand-offs keep the same pace."
        )
    if cfg.pp_async_batch_depth < 1:
        logger.info(
            "Pipeline parallelism on TPU runs with --pp-async-batch-depth 1 "
            "(got %d): the ICI hand-off needs the last stage's outputs "
            "received before each forward.",
            cfg.pp_async_batch_depth,
        )
        declare_resolution(
            server_args, "tpu.apply_server_args_defaults", pp_async_batch_depth=1
        )
    if cfg.tp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism on TPU requires --tp-size 1 "
            f"(got --tp-size {cfg.tp_size} with --pp-size {pp}): "
            "tensor parallelism inside a pipeline stage is not supported."
        )
    if cfg.speculative_algorithm is not None:
        raise NotImplementedError(
            "Speculative decoding "
            f"(--speculative-algorithm {cfg.speculative_algorithm}) "
            f"is not supported with pipeline parallelism on TPU (--pp-size {pp})."
        )
    if cfg.enable_dynamic_chunking:
        raise NotImplementedError(
            "--enable-dynamic-chunking is not supported with pipeline "
            f"parallelism on TPU (--pp-size {pp}): its chunk-latency profiling "
            "relies on a device synchronize that does not wait for TPU "
            "execution."
        )
    if not cfg.enable_torch_compile:
        raise NotImplementedError(
            "Pipeline parallelism on TPU requires --enable-torch-compile "
            f"(--pp-size {pp}): only the compiled forward trims the received "
            "activations to the batch's padded token count."
        )


def _validate_raiden_optimistic_prefill(
    *,
    disaggregation_mode: str,
    transfer_backend: str,
    optimistic_prefill_attempts: int,
) -> None:
    if (
        disaggregation_mode == "null"
        or transfer_backend != "raiden"
        or optimistic_prefill_attempts == 0
    ):
        return
    raise ValueError(
        "--optimistic-prefill-attempts must be 0 with "
        "--disaggregation-transfer-backend raiden "
        f"(got {optimistic_prefill_attempts}). A raiden prefill must not compute "
        "a request before its decode has reserved the KV pages for it. "
        "Optimistic prefill computes it earlier and parks its KV, which under "
        "load fills the prefill pool with requests the decodes have no pages "
        "for and deadlocks prefill and decode."
    )


def _apply_page_size_default(server_args) -> None:
    """Resolve ``page_size`` against the attention backend's declared floor.

    A backend whose kernel cannot address small pages -- the SEQ_ALONG_LANE
    path tiles the page axis onto lanes and needs a multiple of the 128-lane
    width -- must not silently receive the generic default, or it fails deep
    inside the Pallas trace with a shape error that reads like a kernel bug.
    """
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    # The backend may be a declaration (record left pristine); read it back through the stash.
    cfg = resolving_view(server_args)
    spec = _rpa_kernel_spec(cfg.attention_backend)
    floor = spec.min_page_size if spec is not None else None
    if cfg.page_size is None:
        default = max(_DEFAULT_PAGE_SIZE, floor or 0)
        declare_resolution(
            server_args, "tpu.apply_server_args_defaults", page_size=default
        )
        return
    if floor is not None and cfg.page_size < floor:
        raise ValueError(
            f"--attention-backend {cfg.attention_backend} needs "
            f"--page-size >= {floor}, got {cfg.page_size}."
        )


def _apply_kv_cache_dtype_default(server_args) -> None:
    """Default the KV cache to fp8 for the Pallas RPA backends.

    Those kernels dequantize KV on read, and at the head counts TPU serving
    runs (tensor parallel leaves one KV head per rank) fp8 halves KV bytes per
    token. That doubles the context or the batch that fits in HBM, which is the
    binding constraint at long context -- decode there is bound by streaming
    KV, not by arithmetic. "auto" (model dtype) is the generic SGLang default
    and gives up both.

    Only those backends: ``pallas_tpu_mla`` accepts fp8 KV but stays opt-in
    until its accuracy is validated, and ``pallas_tpu_flash`` has no
    dequantization scale, so defaulting it to fp8 silently degrades it.
    """
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    # Same as above: the RPA default backend is a declaration, not a field.
    cfg = resolving_view(server_args)
    if cfg.kv_cache_dtype != "auto":
        return
    if _rpa_kernel_spec(cfg.attention_backend) is None:
        return
    declare_resolution(
        server_args, "tpu.apply_server_args_defaults", kv_cache_dtype="fp8_e4m3"
    )


def _resolve_num_hidden_layers(server_args) -> int:
    """Layer count from the model's HF config (lru-cached load), 0 when
    unresolvable — an unloadable model fails later with its real error, and
    the prefetch containment stays off rather than guessing."""
    try:
        from sglang.srt.utils.hf_transformers.config import get_config

        hf_config = get_config(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        return int(getattr(hf_config, "num_hidden_layers", 0))
    except Exception:
        return 0


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
        # Pre-publish resolution declares rather than writes, so every read below goes
        # through the live declaration view, this hook's own declarations included.
        from sglang.srt.arg_groups.overrides import (
            declare_resolution,
            resolving_view,
        )
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            with_phase,
        )
        from sglang_tpu.patches import _prepare_tpu_env

        cfg = resolving_view(server_args)
        if cfg.pp_size > 1:
            _validate_pipeline_parallel_args(server_args)
        _validate_raiden_optimistic_prefill(
            disaggregation_mode=cfg.disaggregation_mode,
            transfer_backend=cfg.disaggregation_transfer_backend,
            optimistic_prefill_attempts=cfg.optimistic_prefill_attempts,
        )

        # Under Ray the slicebuilder/topology env is built per-rank after the
        # cluster's real node IPs are discovered (multihost: per-actor from Ray
        # IPs in _prepare_tpu_env; single-host: seeded once in
        # RayEngine._launch_scheduler_processes). Building it here -- in the
        # orchestrator, before discovery -- would emit a bogus single-host
        # (localhost) env that never applies to the actual ranks. Still apply
        # the libtpu/env overrides now so actors inherit them before importing
        # torch_tpu.
        # One device per scheduler process across all pipeline stages.
        _prepare_tpu_env(cfg.tp_size * cfg.pp_size, setup_slicebuilder=not cfg.use_ray)

        from sglang_tpu.env_override import (
            apply_small_kv_pool_vmem_prefetch_override,
        )

        apply_small_kv_pool_vmem_prefetch_override(
            max_total_tokens=cfg.max_total_tokens,
            num_hidden_layers=_resolve_num_hidden_layers(cfg),
        )

        if cfg.dtype is None or cfg.dtype == "auto":
            declare_resolution(
                server_args, "tpu.apply_server_args_defaults", dtype="bfloat16"
            )

        if cfg.attention_backend is None:
            declare_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                attention_backend=self.get_default_attention_backend(),
            )

        if cfg.sampling_backend is None:
            declare_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                sampling_backend="tpu",
            )

        # Graph configuration is resolved before platform defaults. Updating the
        # legacy flag alone cannot disable the already-resolved phase runners.
        graph_config = cfg.cuda_graph_config
        for phase in Phase.ALL:
            graph_config = with_phase(graph_config, phase, backend=Backend.DISABLED)
        declare_resolution(
            server_args,
            "tpu.apply_server_args_defaults",
            disable_cuda_graph=True,
            cuda_graph_config=graph_config,
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

        if cfg.mem_fraction_static is None:
            declare_resolution(
                server_args, "tpu.apply_server_args_defaults", mem_fraction_static=0.40
            )

        # raiden's load-back keeps the reloaded prefix's slot indices on CPU
        # (TpuRaidenHiCacheController.load), so req_to_token must stay
        # host-resident: a device-resident req_to_token would put the other
        # nodes' prefix indices on device and mix devices in the prefix cat. Pin
        # SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1 for raiden.
        if cfg.hicache_io_backend == "raiden":
            if os.environ.get("SGLANG_TPU_REQ_TO_TOKEN_ON_HOST") == "0":
                logger.warning(
                    "Overriding SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=0 -> 1: the "
                    "raiden KV-offload backend requires host-resident "
                    "req_to_token."
                )
            os.environ["SGLANG_TPU_REQ_TO_TOKEN_ON_HOST"] = "1"

        # There is no CUDA coredump on TPU: on a crash the handler would trigger
        # a no-op dump, then sleep SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS
        # (60 s) with the dead server still up. An explicit setting still wins.
        os.environ.setdefault("SGLANG_CUDA_COREDUMP_BEFORE_CRASH", "0")

        # The default TPU backend (Pallas RPA v3) expects a paged KV cache where
        # page_size is the HBM<->VMEM DMA chunk and the fused KV buffer's page
        # axis. SGLang's global page_size=1 default is valid for token-granular
        # GPU kernels, but is inefficient here: it makes page_indices
        # token-granular and turns KV fetches into many tiny DMAs.
        # Builtin TPU speculative decoding currently supports EAGLE3 only.
        # Registered algorithms own their validation and backend integration.
        spec_algo = cfg.speculative_algorithm
        if spec_algo is not None:
            if spec_algo == "EAGLE3":
                if cfg.speculative_use_rejection_sampling:
                    raise NotImplementedError(
                        "EAGLE3 rejection sampling is not supported on TPU yet."
                    )
                if (cfg.speculative_eagle_topk or 1) > 1:
                    raise NotImplementedError(
                        "EAGLE3 topk > 1 is not supported on TPU yet."
                    )
            elif spec_algo == "DFLASH":
                pass
            else:
                from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

                # Unknown names may be registered by a plugin after platform
                # defaults run. The shared speculative registry validates them
                # later; only reject builtins the TPU backend knows it cannot run.
                if spec_algo.upper() in SpeculativeAlgorithm.__members__:
                    raise NotImplementedError(
                        f"Speculative algorithm {spec_algo!r} is not supported on TPU "
                        "yet (only EAGLE3 and DFLASH among builtin algorithms)."
                    )

        # Attention DP is declared by width (--attn-dp-size); the deprecated
        # --dp-size N --enable-dp-attention spelling may not have resolved yet here.
        attn_dp_size = cfg.attn_dp_size
        if attn_dp_size == 1 and cfg.enable_dp_attention:
            attn_dp_size = cfg.dp_size
        if attn_dp_size > 1:
            attn_tp_size = cfg.tp_size // attn_dp_size
            if attn_tp_size > 1 and not cfg.enable_dp_lm_head:
                raise ValueError(
                    f"When attention TP > 1 under DP attention on TPU (attn_tp_size={attn_tp_size} "
                    f"from tp_size={cfg.tp_size} and attn_dp_size={attn_dp_size}), "
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
        #     sequences for higher throughput. Supports EAGLE3 speculative
        #     decoding through the shared RPA draft-backend implementation.
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
        from sglang_tpu.graph_runner.tpu_graph_runner import TpuGraphRunner

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
        # Pick per the resolved backend: the raw field is None when --attention-backend is omitted.
        from sglang.srt.runtime_context import get_exec

        return _mha_kv_pool_cls(get_exec().kernel.attention_backend)

    def get_mamba_pool_cls(self) -> type:
        # Hybrid GDN models (Qwen3.5 / Qwen3-Next) store their recurrent state
        # in a TPU-native, Pallas-kernel-layout pool.
        from sglang_tpu.mamba_pool import TpuMambaPool

        return TpuMambaPool

    def get_mha_kv_pool_host_cls(self, device_pool) -> type:
        from sglang.srt.mem_cache.memory_pool_host import (
            TpuFusedMHATokenToKVPoolHost,
        )

        # Upstream relocated MHATokenToKVPoolHost from mem_cache.memory_pool_host
        # into the mem_cache.pool_host package; TpuFusedMHATokenToKVPoolHost is the
        # fork's own class and still lives in memory_pool_host.
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.runtime_context import get_memory
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        # raiden offloads through tpu-sync's DMA-pinned host pool instead of the
        # torch-copy fused host pool; native (torch_tpu) keeps the fused pool.
        if get_memory().hicache_io_backend == "raiden":
            from sglang_tpu.kv_pool_host import (
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
            from sglang_tpu.kv_pool import (
                TpuFusedMLATokenToKVPool,
            )

            return TpuFusedMLATokenToKVPool

        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_nsa_kv_pool_cls(self) -> type:
        raise NotImplementedError("NSA is not supported on TPU.")

    def get_paged_allocator_cls(self) -> type:
        from sglang_tpu.allocator_tpu import (
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

        from sglang_tpu.compile_cache import (
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

        from sglang_tpu.patches import (
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
        # Guard against premature PjRt initialization in the parent/tokenizer
        # process before init_backend() runs in worker subprocesses.
        if not getattr(self, "_backend_initialized", False):
            return 0
        _, limit = torch.accelerator.get_memory_info(self.get_device(device_id))
        return int(limit)

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        if not getattr(self, "_backend_initialized", False):
            return 0.0
        if device is None:
            device = self.get_device(0)
        free, total = torch.accelerator.get_memory_info(device)
        return float(total - free)

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device("tpu", device_id)

    def set_device(self, device: torch.device) -> None:
        torch.accelerator.set_device_index(device)

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
        torch.accelerator.synchronize()

    @contextmanager
    def weight_loading_context(self, weights):
        from sglang.srt.runtime_context import get_parallel
        from sglang_tpu.expert_loading import (
            expert_loading_context,
        )

        if get_parallel().pp_size > 1:
            with expert_loading_context():
                yield _sync_weight_stream(weights)
            return

        # No eager drain at load exit: torch.accelerator.synchronize() here deadlocks
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
