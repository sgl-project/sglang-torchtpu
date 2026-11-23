# SPDX-License-Identifier: Apache-2.0
"""multimodal_gen's TPU platform, backed by torch_tpu (PrivateUse1/PJRT), as an
out-of-tree plugin. One device per worker.

The ``tpu_oot`` entry point of ``sglang.multimodal_gen.platforms`` selects
:class:`TpuOOTPlatform` on TPU hosts (see :func:`sglang_tpu.multimodal_gen.activate`).
It is a ``PlatformEnum.OOT`` platform whose ``is_tpu()`` stays true, so core TPU
paths keep applying.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.compile_cache import append_compile_cache_key_tag
from sglang_tpu.multimodal_gen import worker

logger = init_logger(__name__)

_DEVICE_MODULE_SHIMS = {
    "empty_cache": lambda: None,
    "reset_peak_memory_stats": lambda: (
        torch.accelerator.reset_peak_memory_stats()
        if torch.tpu.is_initialized()
        else None
    ),
    "memory_allocated": lambda: _memory_stat("allocated_bytes.all.current"),
    "max_memory_allocated": lambda: _memory_stat("allocated_bytes.all.peak"),
    # gpu_worker sizes residency off reserved; PJRT has no caching allocator,
    # so reserved is allocated and the pool overhead it computes is zero.
    "memory_reserved": lambda: _memory_stat("allocated_bytes.all.current"),
    "max_memory_reserved": lambda: _memory_stat("allocated_bytes.all.peak"),
}


_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE = 4 * 1024 * 1024 * 1024


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _memory_stat(key: str) -> int:
    if not torch.tpu.is_initialized():
        return 0
    return int(torch.accelerator.memory_stats().get(key, 0))


def _install_device_module_shims() -> None:
    for name, fn in _DEVICE_MODULE_SHIMS.items():
        if not hasattr(torch.tpu, name):
            setattr(torch.tpu, name, fn)


# The launcher and benchmarks reach these through torch.get_device_module()
# without entering the worker hook. Every shim is is_initialized()-guarded.
if hasattr(torch, "tpu"):
    _install_device_module_shims()

_total_memory_cache: dict[int, int] = {}


def _init_torch_tpu_runtime() -> None:
    """Bring torch_tpu up in this worker and touch its device."""
    from sglang_tpu.patches import (
        _patch_set_device_tolerate_mismatch,
        _prepare_tpu_env,
    )

    # srt's activate() selects CPU-only JAX for TorchTPU processes; the
    # worker cannot rely on that having run in the parent (entry-point
    # discovery), and _prepare_tpu_env refuses an unset value once JAX is
    # imported.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # Host staging pool for multi-host DMA, sized as srt sizes it.
    os.environ.setdefault(
        "TPU_PREMAPPED_BUFFER_SIZE", str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE)
    )
    _prepare_tpu_env(world_size=_world_size(), resolve_single_host_topology=True)
    import torch_tpu  # noqa: F401

    # Registers the "tpu" dynamo backend that simple_compile_backend names.
    import torch_tpu._internal.compile  # noqa: F401

    # PJRT owns the chiplet binding; local_rank is only a label.
    _patch_set_device_tolerate_mismatch()

    _install_device_module_shims()
    torch.zeros(1, device="tpu")


# The --attention-backend name of the Pallas backend (AttentionBackendEnum.OOT).
SPLASH_ATTN_BACKEND_NAME = "tpu_splash_attn"
_SPLASH_ATTN_BACKEND_CLS = (
    "sglang_tpu.multimodal_gen.tpu_splash_attn.TpuSplashAttentionBackend"
)


class TpuOOTPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    simple_compile_backend: str = "tpu"

    _worker_initialized: bool = False

    def is_tpu(self) -> bool:
        # Kernel fallbacks, CustomOp.forward_tpu and the TPU argument checks
        # key on is_tpu().
        return True

    def get_dispatch_key_name(self) -> str:
        # CustomOp dispatch takes forward_tpu where an op defines it.
        return "tpu"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device("tpu", 0)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # Real kind ("TPU v7"); the query reads the host config, no PJRT init.
        try:
            from torch_tpu._internal.utils import hardware

            return hardware.get_tpu_device_name() or "tpu"
        except Exception:
            return "tpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # Constant per device; server_args asks several times at startup.
        cached = _total_memory_cache.get(device_id)
        if cached is not None:
            return cached
        if torch.tpu.is_initialized():
            _free, total = torch.accelerator.get_memory_info(
                torch.device("tpu", device_id)
            )
            total = int(total)
        else:
            # Pre-init query for offload sizing; get_device_properties() would
            # initialize PJRT.
            from torch_tpu._internal.utils import hardware

            total = hardware.get_hbm_bytes_per_device()
            if total is None:
                raise RuntimeError("No recognized TPU attached.")
        _total_memory_cache[device_id] = total
        return total

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group=None,
    ) -> float:
        total = cls.get_device_total_memory(device_id or 0)
        return (total - cls.get_current_memory_usage()) / (1 << 30)

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        if not torch.tpu.is_initialized():
            return 0.0
        free, total = torch.accelerator.get_memory_info(device)
        return float(total - free)

    @classmethod
    def is_float64_supported(cls) -> bool:
        # tpu7x: elementwise matches a host reference to ~1e-13, matmul
        # downcasts to ~1e-7.
        return True

    @classmethod
    def supports_distributed_p2p(cls) -> bool:
        # tpu_dist implements collectives only; send/recv raise.
        return False

    def supports_distributed_device_id(self) -> bool:
        # PJRT owns the chiplet binding.
        return False

    @classmethod
    def supports_device_tensor_pickle(cls) -> bool:
        # Reducing a PrivateUse1 storage aborts in tensor_to_buffer.cc.
        return False

    @classmethod
    def inference_mode(cls):
        # torch_tpu's lazy tensors do not support torch.inference_mode.
        return torch.no_grad()

    def get_device(self, local_rank: int) -> torch.device:
        return self.get_local_torch_device()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        if not cls._worker_initialized:
            cls._init_worker_process()
            cls._worker_initialized = True
        torch.accelerator.set_device_index(device)

    @classmethod
    def _init_worker_process(cls) -> None:
        _init_torch_tpu_runtime()
        # The torch_tpu backend compiles static shapes only. The out-of-tree
        # compile path leaves dynamic=None, so keep a recompile for a new shape
        # static too.
        import torch._dynamo

        torch._dynamo.config.automatic_dynamic_shapes = False
        worker.install_runtime_patches()

    def get_torch_distributed_backend_str(self) -> str:
        # tpu_dist needs one addressable PJRT device; a lone worker has many.
        return "gloo" if _world_size() == 1 else "tpu_dist"

    @classmethod
    def get_all_to_all_communicator_cls(cls) -> str:
        # TPU keeps the base communicator; core makes an OOT platform say so.
        return Platform.get_device_communicator_cls()

    @classmethod
    def get_oot_attn_backend_name(cls) -> str | None:
        return SPLASH_ATTN_BACKEND_NAME

    @classmethod
    def oot_attn_backend_serves(
        cls, supported_attention_backends: set[AttentionBackendEnum]
    ) -> bool:
        # A ring launch selects the splash backend globally, so every layer
        # that takes TORCH_SDPA (DiT and text encoder alike) runs it, as with
        # an explicit --attention-backend tpu_splash_attn, which the recipes
        # pass. Shapes its kernels do not cover, such as prompts up to 512
        # tokens and ragged causal attention, run SDPA.
        return AttentionBackendEnum.TORCH_SDPA in supported_attention_backends

    @classmethod
    def get_ring_capable_attention_backends(cls) -> tuple[str, ...]:
        # Rings by K/V all-gather; TPU has no p2p for the core ring backends.
        return (SPLASH_ATTN_BACKEND_NAME,)

    @classmethod
    def supports_safetensors_device_load(cls) -> bool:
        # safetensors rejects 'tpu:N'.
        return False

    @classmethod
    def prefers_head_slab_output_exchange(cls) -> bool:
        # torch_tpu relayouts a sequence-major staging copy around the
        # collective; dense head slabs keep (8, 128) tiles unpadded.
        return True

    @classmethod
    def supports_rank_subset_collectives(cls) -> bool:
        # torch_tpu launches a subgroup's collectives on every device of the
        # slice, so every rank has to issue them together.
        return False

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend is AttentionBackendEnum.OOT:
            logger.info(
                "Using TPU Pallas splash attention backend (falls back to SDPA "
                "per shape, or entirely off TPU v7x)."
            )
            return _SPLASH_ATTN_BACKEND_CLS
        if selected_backend in (None, AttentionBackendEnum.TORCH_SDPA):
            logger.info("Using Torch SDPA backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "sdpa.SDPABackend"
            )
        raise ValueError(
            f"Attention backend {selected_backend} is not supported on TPU; "
            f"the supported backends are TORCH_SDPA (default) and "
            f"{SPLASH_ATTN_BACKEND_NAME}."
        )

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        return False

    def configure_compile_cache(self) -> None:
        # torch_tpu persists XLA executables in AOTAutogradCache, whose key sees
        # neither the device assignment behind collectives, opaque Pallas kernel
        # sources nor the XLA flags: a stale hit can hang or ignore new flags.
        worker.configure_compile_cache()

    def get_compile_backend(self, mode: str | None = None) -> str:
        if mode is not None:
            logger.info_once(
                "torch.compile mode=%s is ignored by the torch_tpu backend.", mode
            )
        return super().get_compile_backend(mode)

    def prepare_spmd_region_wrapper(self) -> Callable | None:
        # torch_tpu splits a compiled graph at every collective and re-lays out
        # tensors at each split. Marking a region SPMD-safe keeps its
        # collectives in one program. A whole-DiT program needs too much
        # scratch, so only regions. The ranks must run identical programs: the
        # fused collectives otherwise hang or exchange the wrong data.
        try:
            from torch_tpu._internal.distributed.spmd_util import spmd_safe
        except ImportError:
            # A private API: without it, regions keep split collectives.
            logger.warning_once(
                "torch_tpu has no spmd_safe; regional-compile programs keep "
                "their collectives split."
            )
            return None

        # The region annotation is not part of the compile cache key; keep
        # persisted executables of the two region policies apart.
        append_compile_cache_key_tag("spmd-safe-regions-v1")
        return spmd_safe

    def apply_server_args_defaults(self, server_args) -> None:
        # The compile warmup layerwise-offloads the DiT, which TPU cannot run.
        if (
            server_args.is_arg_explicitly_set("offload_during_compile")
            and server_args.offload_during_compile
        ):
            logger.warning(
                "--offload-during-compile is not supported on TPU (the compile "
                "warmup cannot layerwise-offload the DiT); disabling it."
            )
        server_args.offload_during_compile = False

    def supports_teacache_with_torch_compile(self) -> bool:
        # Every TeaCache state update is a full XLA recompile under the
        # torch_tpu backend, up to the recompile limit.
        return False

    @classmethod
    def supports_vae_torch_compile(cls) -> bool:
        # Off for every VAE: the torch_tpu backend fails to compile the diffusers
        # AutoencoderKL decoder (torch_tpu 0.2.0rc4, libtpu 0.0.50: view() cannot
        # turn a channels-last [1, 512, 128, 128] activation into [1, 32, 262144]),
        # and the other VAEs are untested under it.
        return False

    @classmethod
    def runs_dit_attention_eager_under_compile(cls) -> bool:
        # The TPU attention path traces, and a compiled DiT block must stay
        # one program (#1886); an eager attention island splits it.
        return False

    @classmethod
    def drain_device_queue_after_request(cls) -> bool:
        # Workaround for torch_tpu 0.2.0rc4 and 0.2.0: seeding torch.Generator("tpu")
        # does a blocking host-to-device copy that waits behind queued collectives.
        # Wan2.2-A14B's second warmup request hung with 7 of 8 ranks in that copy
        # while the 8th waited in a host all_reduce (4 of 4 runs; warm latency is
        # unchanged). test_diffusion_wan22_ti2v_5b.py seeds on the CPU for the same
        # copy; remove both once a TPU Generator seeds without blocking.
        return True

    @classmethod
    def fusion_barrier(cls, x: torch.Tensor) -> torch.Tensor:
        # Only a compiled XLA program fuses; in eager the barrier would copy x.
        if x.device.type != "tpu" or not torch.compiler.is_compiling():
            return x
        return torch.tpu.ops.optimization_barrier([x])[0]
