"""TPU platform backend for SGLang SRT using torch_tpu.

Implements the SRTPlatform interface so that SGLang can run inference
on Google Cloud TPUs via the torch_tpu (PyTorch privateuse1) backend.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class TpuSRTPlatform(SRTPlatform):
    """SGLang SRT platform for Google Cloud TPUs via torch_tpu."""

    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    supported_quantization: list[str] = []

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        from sglang.srt.hardware_backend.tpu.patches import (
            _prepare_tpu_env,
            set_active_attention_backend,
        )

        _prepare_tpu_env(server_args.tp_size)

        if server_args.dtype is None or server_args.dtype == "auto":
            server_args.dtype = "bfloat16"

        if server_args.attention_backend is None:
            server_args.attention_backend = self.get_default_attention_backend()

        # Record for ``_patch_mha_kv_pool_for_rpa_v3``: the pool swap runs at
        # model_runner module import (well after this point) and needs to
        # know which TPU Pallas variant the server resolved to.
        set_active_attention_backend(server_args.attention_backend)

        server_args.disable_cuda_graph = True
        server_args.disable_piecewise_cuda_graph = True
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
            server_args.mem_fraction_static = 0.40

        # The default TPU backend (Pallas RPA v3) expects a paged KV cache where
        # page_size is the HBM<->VMEM DMA chunk and the fused KV buffer's page
        # axis. SGLang's global page_size=1 default is valid for token-granular
        # GPU kernels, but is inefficient here: it makes page_indices
        # token-granular and turns KV fetches into many tiny DMAs.
        if server_args.speculative_algorithm is not None:
            # Speculative decoding currently requires page_size=1 on TPU:
            # the verify eviction kernels (align_evict_mask_to_page_size,
            # get_src_tgt_cache_loc) are Triton-only and have no PyTorch
            # fallback yet.
            # TODO(pyc96): add PyTorch eviction kernels and fused move_kv_cache
            # to enable page_size > 1 with speculative decoding on TPU.
            if server_args.page_size is not None and server_args.page_size != 1:
                logger.warning(
                    "Overriding page_size=%d to 1 for speculative decoding on TPU.",
                    server_args.page_size,
                )
            server_args.page_size = 1
        elif server_args.page_size is None:
            server_args.page_size = 64

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        # Two Pallas backends are registered for TPU:
        #   - "pallas_tpu_rpa_v3" (default): ragged_paged_attention v3 kernel
        #     in `pallas/ragged_paged_attention_v3.py`. Single unified kernel
        #     for prefill + decode, fused KV cache (interleaved K/V, packed),
        #     decode-native batched call.
        #   - "pallas_tpu_flash": flash_attention for prefill/extend +
        #     `jax.experimental.pallas.ops.tpu.paged_attention` for decode.
        #     Original implementation from PR #5; works against the flat
        #     `MHATokenToKVPool` layout (no fused-pool swap required).
        # Both are listed in `server_args.ATTENTION_BACKEND_CHOICES` and
        # registered in `layers.attention.attention_registry` (same place as
        # `ascend` / `intel_xpu`; factories use lazy imports so non-TPU runs
        # don't pull in jax/torch_tpu).
        # Override to the reference path with `--attention-backend torch_native`.
        return "pallas_tpu_rpa_v3"

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.model_executor.tpu_graph_runner import TpuGraphRunner

        return TpuGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
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

    def get_piecewise_backend_cls(self) -> type:
        raise NotImplementedError("TPU piecewise backend not yet implemented.")

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        return False

    def is_pin_memory_available(self) -> bool:
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
        # Keep c10d_functional collectives inside the compiled graph. torch_tpu
        # defaults to materializing collective tensors by forcing graph breaks, which
        # crashes under torch.compile(fullgraph=True) at TP>1. Setting this to 0/false
        # disables that guardrail for SPMD TP graphs.
        os.environ.setdefault(
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false"
        )

        # Size libtpu's premapped (pinned, DMA-mapped) host staging pool. This is
        # read by libtpu at init, so it MUST be set before `import torch_tpu`.
        # Without it, host<->HBM copy_ falls back to slow per-call pinning (~1
        # GB/s); with it, copies run at DMA speed (several GB/s) — which the
        # HiCache (pallas_tpu) layer-wise reload relies on to fit the host->HBM
        # copy under the prior forward's compute. setdefault: the launch env /
        # configmap can override (raise it for heavy KV-offload workloads).
        os.environ.setdefault("TPU_PREMAPPED_BUFFER_SIZE", str(4 * 1024 * 1024 * 1024))

        import torch_tpu  # noqa: F401

        from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches

        apply_tpu_patches()

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
        return 0.0

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("tpu", local_rank)

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
