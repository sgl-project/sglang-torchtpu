# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/interface.py
from __future__ import annotations

import contextlib
import enum
import os
import random
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )
    from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

logger = init_logger(__name__)


class AttentionBackendEnum(enum.Enum):
    FA2 = enum.auto()
    FA = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    TORCH_SDPA = enum.auto()
    SAGE_ATTN = enum.auto()
    SAGE_ATTN_3 = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    SPARSE_VIDEO_GEN_2_ATTN = enum.auto()
    VMOBA_ATTN = enum.auto()
    AITER = enum.auto()
    AITER_SAGE = enum.auto()
    SLA_ATTN = enum.auto()
    SAGE_SLA_ATTN = enum.auto()
    LASER_ATTN = enum.auto()
    BLOCK_SPARSE_ATTN = enum.auto()
    RAIN_FUSION_ATTN = enum.auto()
    TPU_PALLAS_FLASH = enum.auto()
    NO_ATTENTION = enum.auto()

    def __str__(self):
        return self.name.lower()

    @property
    def is_sparse(self) -> bool:
        return self in {
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.SLA_ATTN,
            AttentionBackendEnum.SAGE_SLA_ATTN,
            AttentionBackendEnum.LASER_ATTN,
            AttentionBackendEnum.BLOCK_SPARSE_ATTN,
            AttentionBackendEnum.RAIN_FUSION_ATTN,
        }


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    CPU = enum.auto()
    MPS = enum.auto()
    NPU = enum.auto()
    MUSA = enum.auto()
    XPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    UNSPECIFIED = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class Platform:
    _enum: PlatformEnum
    device_name: str
    device_type: str
    device: torch.device | None = None  # Dummy attribute for compatibility

    # available dispatch keys:
    # check https://github.com/pytorch/pytorch/blob/313dac6c1ca0fa0cde32477509cce32089f8532a/torchgen/model.py#L134 # noqa
    # use "CPU" as a fallback for platforms not registered in PyTorch
    dispatch_key: str = "CPU"

    # The torch.compile backend for compiling simple and
    # standalone functions. The default value is "inductor" to keep
    # the same behavior as PyTorch.
    # NOTE: for the forward part of the model, vLLM has another separate
    # compilation strategy.
    simple_compile_backend: str = "inductor"

    supported_quantization: list[str] = []

    @lru_cache(maxsize=1)
    def is_cuda(self) -> bool:
        return self.is_cuda_static()

    @lru_cache(maxsize=1)
    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    @lru_cache(maxsize=1)
    def is_rocm(self) -> bool:
        return self.is_rocm_static()

    @lru_cache(maxsize=1)
    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    @lru_cache(maxsize=1)
    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    @classmethod
    @lru_cache(maxsize=1)
    def is_blackwell(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability()[0] == 10

    @classmethod
    @lru_cache(maxsize=1)
    def is_hopper(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability() == (9, 0)

    @classmethod
    @lru_cache(maxsize=1)
    def is_sm120(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability()[0] == 12

    @classmethod
    def is_cuda_static(cls) -> bool:
        return getattr(cls, "_enum", None) == PlatformEnum.CUDA

    @classmethod
    def is_rocm_static(cls) -> bool:
        return getattr(cls, "_enum", None) == PlatformEnum.ROCM

    @lru_cache(maxsize=1)
    def is_hpu(self) -> bool:
        return hasattr(torch, "hpu") and torch.hpu.is_available()

    @lru_cache(maxsize=1)
    def is_xpu(self) -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @lru_cache(maxsize=1)
    def is_npu(self) -> bool:
        return hasattr(torch, "npu") and torch.npu.is_available()

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    @lru_cache(maxsize=1)
    def is_cuda_alike(self) -> bool:
        """Stateless version of :func:`torch.cuda.is_available`."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM, PlatformEnum.MUSA)

    @lru_cache(maxsize=1)
    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    @lru_cache(maxsize=1)
    def is_musa(self):
        try:
            return hasattr(torch, "musa") and torch.musa.is_available()
        except ModuleNotFoundError:
            return False

    @lru_cache(maxsize=1)
    def is_hip(self) -> bool:
        return self.is_rocm()

    @classmethod
    @lru_cache(maxsize=1)
    def is_amp_supported(cls) -> bool:
        return True

    @classmethod
    @lru_cache(maxsize=1)
    def is_float64_supported(cls) -> bool:
        return True

    @classmethod
    def get_modelopt_fp4_quantize_op(cls) -> Callable | None:
        return None

    @classmethod
    def get_modelopt_fp4_gemm_op(cls) -> tuple[Callable | None, str | None]:
        return None, None

    @classmethod
    def get_modelopt_flashinfer_fp4_backend(cls) -> str:
        return "auto"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        raise NotImplementedError

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """Get the attention backend class of a device."""
        return ""

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        """Stateless version of :func:`torch.cuda.get_device_capability`."""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """
        Test whether this platform is compatible with a device capability.

        The ``capability`` argument can either be:

        - A tuple ``(major, minor)``.
        - An integer ``<major><minor>``. (See :meth:`DeviceCapability.to_int`)
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False

        if isinstance(capability, tuple):
            return current_capability >= capability

        return current_capability.to_int() >= capability

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a device."""
        raise NotImplementedError

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the uuid of a device, e.g. the PCI bus ID."""
        raise NotImplementedError

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of a device in bytes."""
        raise NotImplementedError

    @lru_cache(maxsize=1)
    def get_device(self, local_rank: int) -> torch.device:
        if self.is_cuda() or self.is_rocm():
            return torch.device("cuda", local_rank)
        elif self.is_npu():
            return torch.device("npu", local_rank)
        elif self.is_xpu():
            return torch.device("xpu", local_rank)
        elif self.is_musa():
            return torch.device("musa", local_rank)
        elif self.is_mps():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @lru_cache(maxsize=1)
    def get_torch_distributed_backend_str(self) -> str:
        if self.is_cuda_alike():
            return "nccl"
        elif self.is_npu():
            return "hccl"
        elif self.is_musa():
            return "mccl"
        elif self.is_mps():
            return "gloo"
        elif self.is_cpu():
            return "gloo"
        elif self.is_xpu():
            return "xccl"
        else:
            raise NotImplementedError(
                "No Accelerators(AMD/NV/MTT GPU, AMD MI instinct accelerators) available"
            )

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """
        Check if the current platform supports async output.
        """
        raise NotImplementedError

    @classmethod
    def inference_mode(cls):
        """A device-specific wrapper of `torch.inference_mode`.

        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """
        Set the seed of each random module.
        `torch.manual_seed` will set seed on all devices.

        Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.get_device_module().manual_seed_all(seed)

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """
        Verify whether the current platform supports the specified model
        architecture.

        - This will raise an Error or Warning based on the model support on
        the current platform.
        - By default all models are considered supported.
        """
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """
        Verify whether the quantization is supported by the current platform.
        """
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in "
                f"{cls.device_name}."
            )

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """
        Return the memory usage in bytes.
        """
        raise NotImplementedError

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        """
        Return the available memory in GiB.
        """
        raise NotImplementedError

    @classmethod
    def empty_cache(cls) -> None:
        """Release cached device memory back to the allocator."""
        torch.get_device_module().empty_cache()

    @classmethod
    def reset_peak_memory_stats(cls) -> None:
        """Reset the peak device-memory counters for the current device."""
        torch.get_device_module().reset_peak_memory_stats()

    @classmethod
    def get_memory_snapshot(cls) -> MemorySnapshot:
        """Capture current/peak allocated and reserved device memory."""
        # Local import: perf_logger imports current_platform at module level.
        from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

        device_module = torch.get_device_module()
        if not device_module.is_available():
            return MemorySnapshot(
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_allocated_mb=0.0,
                peak_reserved_mb=0.0,
            )
        mib = 1024**2
        return MemorySnapshot(
            allocated_mb=device_module.memory_allocated() / mib,
            reserved_mb=device_module.memory_reserved() / mib,
            peak_allocated_mb=device_module.max_memory_allocated() / mib,
            peak_reserved_mb=device_module.max_memory_reserved() / mib,
        )

    @classmethod
    def init_worker_device_env(
        cls, *, local_rank: int, rank: int, world_size: int, master_port: int
    ) -> None:
        """Bind the worker process to its device and export the single-host
        torch.distributed rendezvous environment."""
        torch.get_device_module().set_device(local_rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

    @classmethod
    def supports_op_level_torch_compile(cls) -> bool:
        """Whether op-level @torch.compile decorations (inductor) are usable."""
        return True

    @classmethod
    def supports_p2p_communication(cls) -> bool:
        """Whether the distributed backend implements point-to-point
        send/recv (required e.g. for spatial-parallel VAE halo exchange)."""
        return True

    @classmethod
    def supports_vae_decode_torch_compile(cls) -> bool:
        """Whether the VAE decode path may be wrapped in torch.compile."""
        return True

    @classmethod
    def vae_decode_context(cls, pipeline: Any) -> Any:
        """Context manager scoping the VAE decode (e.g. to free device memory
        for decode on memory-constrained platforms). Default: no-op."""
        return contextlib.nullcontext()

    @classmethod
    def denoise_loop_context(cls) -> Any:
        """Context manager wrapping the denoising loop (e.g. to select a
        platform execution mode for the loop). Default: no-op."""
        return contextlib.nullcontext()

    @classmethod
    def on_denoise_step_end(cls) -> None:
        """Called after each denoising step (e.g. to bound in-flight lazy
        work on deferred-execution platforms). Default: no-op."""

    @classmethod
    def wrap_forward_for_compile(cls, forward: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a module forward before torch.compile is applied to it
        (e.g. to keep collectives traceable). Default: identity."""
        return forward

    @classmethod
    def use_functional_seq_all_to_all(cls) -> bool:
        """Whether sequence-parallel attention should use the synchronous
        functional-collectives all-to-all instead of the stream-overlapped
        autograd path (which requires CUDA-style streams and async_op)."""
        return False

    @classmethod
    def create_aux_profiler(cls, log_dir: str) -> Any | None:
        """Return a context manager capturing platform-native device traces
        alongside torch.profiler, or None if torch.profiler suffices."""
        return None

    @classmethod
    def prepare_server_environment(cls, *, world_size: int) -> None:
        """One-time parent-process environment preparation before workers
        are spawned. Default: no-op."""

    @classmethod
    def worker_spawn_env(
        cls, *, rank: int, world_size: int, master_port: int
    ) -> dict[str, str]:
        """Environment variables that must be present at worker-process birth
        (for runtimes that latch env vars during import). Default: none."""
        return {}

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """Get the CPU architecture of the current platform."""
        return CpuArchEnum.UNSPECIFIED

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        """Whether to enable DIT layerwise offload by default on the current platform."""
        return True

    @classmethod
    def optimize_vae(cls, vae: torch.nn.Module) -> torch.nn.Module:
        """Apply platform-specific optimizations to VAE after loading."""
        return vae

    def get_attn_backend(self, *args, **kwargs) -> AttentionImpl:
        attention_cls_str = self.get_attn_backend_cls_str(*args, **kwargs)
        return resolve_obj_by_qualname(attention_cls_str)


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_type = ""
