"""TPU platform backend for SGLang SRT using torch_tpu.

Implements the SRTPlatform interface so that SGLang can run inference
on Google Cloud TPUs via the torch_tpu (PyTorch privateuse1) backend.

Class layout mirrors the CUDA / ROCm split (see ``platforms/cuda.py``):

    DeviceMixin
    └── TpuDeviceMixin           # device identity + DeviceMixin overrides
        └── TpuSRTPlatform(TpuDeviceMixin, SRTPlatform)
                                  # + SRT-specific factories & lifecycle

Keeping the device-ops layer separate lets a future ``TpuMMPlatform``
(multimodal) reuse the same identity / memory / sync code without
pulling in the SRT-specific factories.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)


def _prepare_tpu_env(world_size: int = 1) -> None:
    """Set TORCH_TPU_* env vars needed by PjRt for multi-process init.

    Ported from torchtpu-vllm: _prepare_singlehost_tpu_env().
    Slicebuilder addresses must match exactly the number of TP workers
    (world_size), not the total process count — detokenizer and tokenizer
    processes do not participate in PjRt coordination.

    Idempotent: returns early if TORCH_TPU_SLICEBUILDER_ADDRESSES is set.
    """
    if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
        return

    try:
        import portpicker

        sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
    except ImportError:
        import socket

        # Hold sockets open until env var is written so OS doesn't reassign.
        _held = [
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for _ in range(world_size)
        ]
        for s in _held:
            s.bind(("", 0))
        sb_ports = [s.getsockname()[1] for s in _held]
        for s in _held:
            s.close()

    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        f"localhost:{p}" for p in sb_ports
    )

    if world_size == 1:
        os.environ.setdefault("TORCH_TPU_TOPOLOGY", "1,1,1")
    else:
        try:
            from torch_tpu._internal.utils import hardware

            topo = hardware.get_tpu_topology(world_size)
            if topo:
                os.environ.setdefault("TORCH_TPU_TOPOLOGY", topo)
        except Exception:
            pass

    logger.info(
        "TPU env: TORCH_TPU_SLICEBUILDER_ADDRESSES=%s, TORCH_TPU_TOPOLOGY=%s",
        os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES"),
        os.environ.get("TORCH_TPU_TOPOLOGY"),
    )


class TpuDeviceMixin(DeviceMixin):
    """TPU implementation of the shared device operations (via torch_tpu)."""

    _enum: PlatformEnum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    # HBM capacities by reported TPU device name.  Used as a fallback for
    # get_device_total_memory since torch_tpu does not expose a property
    # equivalent to torch.cuda.get_device_properties(...).total_memory.
    _TPU_HBM_BYTES = {
        "TPU v4": 32 * (1 << 30),
        "TPU v5p": 95 * (1 << 30),
        "TPU v5e": 16 * (1 << 30),
        "TPU v6e": 32 * (1 << 30),
        "TPU v7": 192 * (1 << 30),
    }

    def get_device_total_memory(self, device_id: int = 0) -> int:
        name = self.get_device_name(device_id)
        hbm = self._TPU_HBM_BYTES.get(name)
        if hbm is not None:
            return hbm
        return 32 * (1 << 30)

    def get_current_memory_usage(
        self, device: Optional[torch.device] = None
    ) -> float:
        return 0.0

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("tpu", local_rank)

    def set_device(self, device: torch.device | int) -> None:
        """Set the current TPU device, tolerating PjRt chiplet-bind mismatch.

        sglang's ``ModelRunner.init_torch_distributed`` calls this with
        ``self.gpu_id``. torch_tpu auto-binds each process to a chiplet at
        first PJRT init based on env vars set by ``run_scheduler_process``;
        if the auto-bound chiplet index doesn't match sglang's gpu_id (e.g.
        on v7 where PJRT may assign chiplet 5 to local_rank=3 depending on
        slicebuilder coordination), ``torch.tpu.set_device`` raises:
            "Cannot set TPU device to index X, current process is bound to
             device index Y".

        Accept the existing binding — chiplet assignment is owned by
        torch_tpu/PJRT, sglang's gpu_id is just a label.
        """
        try:
            torch.tpu.set_device(device)
        except ValueError as e:
            if "Cannot set TPU device" not in str(e):
                raise
            idx = (
                device.index
                if hasattr(device, "index") and device.index is not None
                else int(device)
            )
            logger.warning("TPU set_device(idx=%s) skipped: %s", idx, e)

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
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.tpu.manual_seed_all(seed)


class TpuSRTPlatform(TpuDeviceMixin, SRTPlatform):
    """SGLang SRT platform for Google Cloud TPUs via torch_tpu."""

    supported_quantization: list[str] = []

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        _prepare_tpu_env(server_args.tp_size)

        if server_args.dtype is None or server_args.dtype == "auto":
            server_args.dtype = "bfloat16"

        if server_args.attention_backend is None:
            server_args.attention_backend = self.get_default_attention_backend()

        server_args.disable_cuda_graph = True
        server_args.disable_piecewise_cuda_graph = True
        server_args.enable_torch_compile = False
        server_args.disable_overlap_schedule = True

        if server_args.mem_fraction_static is None:
            server_args.mem_fraction_static = 0.40

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        # Pallas backend: flash_attention for prefill/extend, jax pallas
        # paged_attention for decode (decode-native: no q>=128 padding, batched
        # single call, native GQA). Registered in
        # server_args.py:ATTENTION_BACKEND_CHOICES alongside other hardware-
        # specific backends (intel_amx, ascend, intel_xpu), so it is also
        # accepted via --attention-backend / --prefill-attention-backend /
        # --decode-attention-backend. Override to the reference path with
        # `--attention-backend torch_native`.
        return "pallas_tpu"

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.hardware_backend.tpu.graph_runner.tpu_graph_runner import (
            TpuGraphRunner,
        )

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
        raise NotImplementedError(
            "TPU piecewise backend not yet implemented."
        )

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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        import torch_tpu  # noqa: F401
        from torch_tpu._internal import execution_mode

        from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches
        from sglang.srt.layers.attention.attention_registry import (
            register_attention_backend,
        )

        execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)
        logger.info("TorchTPU eager mode set to DEFER_AND_FUSE.")

        def _create_pallas_tpu_backend(runner):
            from sglang.srt.hardware_backend.tpu.attention_pallas import (
                TpuPallasAttnBackend,
            )

            return TpuPallasAttnBackend(runner)

        register_attention_backend("pallas_tpu")(_create_pallas_tpu_backend)
        logger.info("Registered 'pallas_tpu' attention backend.")

        apply_tpu_patches()

    # ------------------------------------------------------------------
    # MultiPlatformOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        return "tpu"
