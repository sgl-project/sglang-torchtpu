# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/interface.py
from __future__ import annotations

import enum
import random
from collections.abc import Callable
from functools import lru_cache
from pkgutil import resolve_name
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )
    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

logger = init_logger(__name__)


class AttentionBackendEnum(enum.Enum):
    FA2 = enum.auto()
    FA = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    TORCH_SDPA = enum.auto()
    TORCH_CUDNN_SDPA = enum.auto()
    DYNAMIC_CUDNN_SDPA = enum.auto()
    SAGE_ATTN = enum.auto()
    SAGE_ATTN_3 = enum.auto()
    SPARGE_ATTN = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    VIDEO_SPARSE_ATTN_H3 = enum.auto()
    HYBRID_WINDOW_ATTN_H3 = enum.auto()
    SPARSE_VIDEO_GEN_2_ATTN = enum.auto()
    VMOBA_ATTN = enum.auto()
    AITER = enum.auto()
    AITER_SAGE = enum.auto()
    SLA_ATTN = enum.auto()
    SAGE_SLA_ATTN = enum.auto()
    LASER_ATTN = enum.auto()
    BLOCK_SPARSE_ATTN = enum.auto()
    RAIN_FUSION_ATTN = enum.auto()
    SOL_ATTN = enum.auto()
    SUBBLOCK_SPARSE_ATTN = enum.auto()
    CUBE_SPARSE_ATTN = enum.auto()
    FP8_FA_SM120 = enum.auto()
    NO_ATTENTION = enum.auto()
    # An out-of-tree platform's own backend: the platform names it
    # (Platform.get_oot_attn_backend_name) and resolves its class. Appended so
    # earlier members keep their auto() values.
    OOT = enum.auto()

    def __str__(self):
        return self.name.lower()

    @property
    def is_sparse(self) -> bool:
        return self in {
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
            AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3,
            AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.SLA_ATTN,
            AttentionBackendEnum.SAGE_SLA_ATTN,
            AttentionBackendEnum.SPARGE_ATTN,
            AttentionBackendEnum.LASER_ATTN,
            AttentionBackendEnum.BLOCK_SPARSE_ATTN,
            AttentionBackendEnum.RAIN_FUSION_ATTN,
            AttentionBackendEnum.SOL_ATTN,
            AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
            AttentionBackendEnum.CUBE_SPARSE_ATTN,
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
    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    device_name: str
    device_type: str
    device: torch.device | None = None  # Dummy attribute for compatibility

    # available dispatch keys:
    # check https://github.com/pytorch/pytorch/blob/313dac6c1ca0fa0cde32477509cce32089f8532a/torchgen/model.py#L134 # noqa
    dispatch_key: str = ""

    # The torch.compile backend for compiling simple and
    # standalone functions. The default value is "inductor" to keep
    # the same behavior as PyTorch.
    # NOTE: for the forward part of the model, vLLM has another separate
    # compilation strategy.
    simple_compile_backend: str = "inductor"

    supported_quantization: list[str] = []

    def init_backend(self) -> None:
        """One-time backend initialization, in each worker; raising aborts startup.

        Where out-of-tree platforms register their custom-op forwards.
        """
        pass

    def apply_server_args_defaults(self, server_args: ServerArgs) -> None:
        """Apply defaults before argument normalization and validation."""
        pass

    def supports_teacache_with_torch_compile(self) -> bool:
        """Whether TeaCache may run under --enable-torch-compile.

        TeaCache updates module state every denoising step; a backend that
        recompiles on that (rather than graph-breaking cheaply) says False and
        requests drop TeaCache with a warning.
        """
        return True

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the backend used to compile diffusion modules."""
        return self.simple_compile_backend

    def prepare_spmd_region_wrapper(self) -> Callable | None:
        """Decorator for --regional-compile regions that every rank runs with the
        same graph (the model sets ``_compile_regions_spmd_safe``). The regions
        should trace as one graph each. May configure process-wide compile
        state the wrapper needs (e.g. a cache-key tag); call it once, before
        compiling."""
        return None

    def get_compile_options(self, module: torch.nn.Module) -> dict[str, object] | None:
        """Return backend-specific options for a diffusion module."""
        return None

    def configure_compile_cache(self) -> None:
        """Configure torch.compile caching for this worker.

        Called once per worker after distributed initialization, before any
        compilation.
        """
        pass

    def get_dispatch_key_name(self) -> str:
        """Return the behavioral dispatch key used by :class:`CustomOp`.

        This is intentionally separate from ``dispatch_key``, which names a
        PyTorch dispatcher key such as ``PrivateUse1``. An out-of-tree backend
        can return an existing key such as ``cuda`` to reuse compatible
        ``forward_cuda`` implementations, or a vendor key backed by registered
        forwards and ``forward_<key>`` methods.
        """
        return "native"

    def get_torch_library_dispatch_key(self) -> str:
        """Return the key used for direct ``torch.library`` registrations."""
        if self.is_out_of_tree():
            if not self.dispatch_key:
                raise NotImplementedError(
                    "Out-of-tree diffusion platforms must define dispatch_key"
                )
            return self.dispatch_key
        return "PrivateUse1" if self.is_npu() else "CUDA"

    @lru_cache(maxsize=1)
    def is_cuda(self) -> bool:
        return self.is_cuda_static()

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
        return cls._enum == PlatformEnum.CUDA

    @classmethod
    def is_rocm_static(cls) -> bool:
        return cls._enum == PlatformEnum.ROCM

    @lru_cache(maxsize=1)
    def is_hpu(self) -> bool:
        return hasattr(torch, "hpu") and torch.hpu.is_available()

    @lru_cache(maxsize=1)
    def is_xpu(self) -> bool:
        return (
            not self.is_out_of_tree()
            and hasattr(torch, "xpu")
            and torch.xpu.is_available()
        )

    @lru_cache(maxsize=1)
    def is_npu(self) -> bool:
        return (
            not self.is_out_of_tree()
            and hasattr(torch, "npu")
            and torch.npu.is_available()
        )

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    @lru_cache(maxsize=1)
    def is_cuda_alike(self) -> bool:
        """Stateless version of :func:`torch.cuda.is_available`."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM, PlatformEnum.MUSA)

    def is_device_type(self, device_type: str | None) -> bool:
        """Return whether a device type belongs to this platform."""
        return device_type == self.device_type

    @lru_cache(maxsize=1)
    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    @lru_cache(maxsize=1)
    def is_musa(self):
        try:
            return (
                not self.is_out_of_tree()
                and hasattr(torch, "musa")
                and torch.musa.is_available()
            )
        except ModuleNotFoundError:
            return False

    @lru_cache(maxsize=1)
    def is_hip(self) -> bool:
        return self.is_rocm()

    @classmethod
    def is_amp_supported(cls) -> bool:
        return True

    @classmethod
    @lru_cache(maxsize=1)
    def is_float64_supported(cls) -> bool:
        """Whether the device computes float64 elementwise.

        Says nothing about matmul, which an accelerator may downcast anyway.
        """
        return True

    @classmethod
    def supports_distributed_p2p(cls) -> bool:
        return True

    @classmethod
    def supports_device_tensor_pickle(cls) -> bool:
        return True

    @classmethod
    def supports_safetensors_device_load(cls) -> bool:
        """Whether safetensors can open tensors directly on this platform's device."""
        return True

    @classmethod
    def prefers_head_slab_output_exchange(cls) -> bool:
        """Whether the Ulysses output all-to-all sends a heads-major attention
        output as per-head slabs instead of relayouting it to sequence-major
        first. Both give bit-identical values."""
        return False

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
    def set_device(cls, device: torch.device) -> None:
        torch.get_device_module(device).set_device(device)

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
    def get_oot_attn_backend_name(cls) -> str | None:
        """The ``--attention-backend`` name of this platform's own backend
        (``AttentionBackendEnum.OOT``), or None when it has none."""
        return None

    @classmethod
    def oot_attn_backend_serves(
        cls, supported_attention_backends: set[AttentionBackendEnum]
    ) -> bool:
        """Whether automatic selection may give the OOT backend to a layer that
        lists only ``supported_attention_backends``."""
        return False

    @classmethod
    def get_ring_capable_attention_backends(cls) -> tuple[str, ...] | None:
        """``--attention-backend`` names that run ring attention on this
        platform, the first being the ``--ring-degree`` default; None keeps the
        core list."""
        return None

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
        if self.is_out_of_tree():
            raise NotImplementedError(
                "Out-of-tree diffusion platforms must implement get_device()"
            )
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

    def supports_distributed_device_id(self) -> bool:
        """Whether torch.distributed accepts this platform's device ID."""
        return not (
            self.is_out_of_tree()
            or self.is_mps()
            or self.is_musa()
            or self.is_npu()
            or self.is_cpu()
            or self.is_xpu()
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
                f"{quant} quantization is currently not supported in {cls.device_name}."
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
    def get_device_communicator_cls(cls) -> str:
        """Return the platform's default device communicator class."""
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa

    @classmethod
    def get_all_to_all_communicator_cls(cls) -> str:
        """Return the communicator used by ``all_to_all_4D``."""
        qualname = cls.get_device_communicator_cls()
        if (
            cls._enum is PlatformEnum.OOT
            and qualname == Platform.get_device_communicator_cls()
        ):
            raise NotImplementedError(
                "Out-of-tree diffusion platforms must implement "
                "get_all_to_all_communicator_cls()"
            )
        return qualname

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """Get the CPU architecture of the current platform."""
        return CpuArchEnum.UNSPECIFIED

    @classmethod
    def supports_vae_torch_compile(cls) -> bool:
        """Whether ``--enable-torch-compile`` also compiles the VAE decode."""
        return True

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        """Whether automatic DiT layerwise offload is enabled on this platform."""
        return True

    @classmethod
    def device_shares_host_memory(cls) -> bool:
        """Whether the accelerator draws from the same physical pool as the host.

        On such a part (DGX Spark's GB10, Jetson) a device allocation is host
        memory the kernel no longer has, and a host copy of a mapped weight is
        a second copy of bytes the page cache already holds.
        """
        return False

    @classmethod
    def optimize_vae(cls, vae: torch.nn.Module) -> torch.nn.Module:
        """Apply platform-specific optimizations to VAE after loading."""
        return vae

    @classmethod
    def drain_device_queue_after_request(cls) -> bool:
        """Whether every rank drains its device queue when a request finishes.

        For platforms where queued collectives can otherwise reach into the
        next request, so a host-to-device copy there waits on a peer.
        """
        return False

    @classmethod
    def fusion_barrier(cls, x: torch.Tensor) -> torch.Tensor:
        """Pass ``x`` through a point the graph compiler may not fuse across.

        The values are unchanged; platforms without such a primitive return
        ``x`` itself.
        """
        return x

    def get_attn_backend(self, *args, **kwargs) -> AttentionImpl:
        attention_cls_str = self.get_attn_backend_cls_str(*args, **kwargs)
        return resolve_name(attention_cls_str)

    def tensor_on_device(self, t: torch.Tensor) -> bool:
        """Check if a tensor is on the current platform's device."""
        return t.is_cuda


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_type = ""
