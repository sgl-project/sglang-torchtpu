# SPDX-License-Identifier: Apache-2.0
"""TPU platform backed by torch_tpu (PrivateUse1/PJRT). One device per worker."""

from __future__ import annotations

import functools
import os

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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


def _add_compiler_flags_to_compile_cache_key() -> None:
    import hashlib

    import torch.compiler.config as compiler_config

    flags = "\0".join(
        os.environ.get(name, "")
        for name in (
            "LIBTPU_INIT_ARGS",
            "XLA_FLAGS",
            "TORCH_TPU_INTERNAL_XLA_OPTIONS",
            # Read at compile time: where torch_tpu splits graphs at collectives.
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
        )
    )
    tag = f"tpu-compiler-flags-v1:{hashlib.sha256(flags.encode()).hexdigest()[:16]}"
    # '|'-joined and idempotent, like srt's compile_cache tags.
    existing_tag = str(compiler_config.cache_key_tag or "")
    if tag not in existing_tag.split("|"):
        compiler_config.cache_key_tag = f"{existing_tag}|{tag}" if existing_tag else tag


def _install_device_module_shims() -> None:
    for name, fn in _DEVICE_MODULE_SHIMS.items():
        if not hasattr(torch.tpu, name):
            setattr(torch.tpu, name, fn)


def _expand_query_broadcast_mask(query, attn_mask):
    """Expand a mask's size-1 query dim to the query length on TPU.

    torch_tpu SDPA returns NaN for masks that broadcast over queries, e.g.
    [B, 1, 1, S] key padding (0.1.1.dev20260924 and
    0.1.1+release.2026.9.22; 0.1.1.dev20260817 is correct). Remove once
    test_diffusion_tpu_sdpa_mask's canary stops reproducing it.
    """
    if (
        attn_mask is not None
        and query.device.type == "tpu"
        and attn_mask.dim() >= 2
        and attn_mask.shape[-2] == 1
        and query.shape[-2] > 1
    ):
        attn_mask = attn_mask.expand(
            *attn_mask.shape[:-2], query.shape[-2], attn_mask.shape[-1]
        )
    return attn_mask


def _patch_sdpa_query_broadcast_mask() -> None:
    # Module-level aliases bound before worker init keep the unpatched
    # function; hunyuan3d_vae's is the only one and passes no mask.
    sdpa = torch.nn.functional.scaled_dot_product_attention
    if getattr(sdpa, "_sglang_tpu_mask_patch", False):
        return

    @functools.wraps(sdpa)
    def scaled_dot_product_attention(query, key, value, attn_mask=None, *a, **kw):
        attn_mask = _expand_query_broadcast_mask(query, attn_mask)
        return sdpa(query, key, value, attn_mask, *a, **kw)

    scaled_dot_product_attention._sglang_tpu_mask_patch = True
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention


# The launcher and benchmarks reach these through torch.get_device_module()
# without entering the worker hook. Every shim is is_initialized()-guarded.
if hasattr(torch, "tpu"):
    _install_device_module_shims()

_total_memory_cache: dict[int, int] = {}


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    simple_compile_backend: str = "tpu"

    _worker_initialized: bool = False

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
    def supports_safetensors_device_load(cls) -> bool:
        # safetensors rejects 'tpu:N'.
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
        from sglang.srt.hardware_backend.tpu.patches import (
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
        _patch_sdpa_query_broadcast_mask()
        torch.zeros(1, device="tpu")

    def configure_compile_cache(self) -> None:
        # torch_tpu persists XLA executables in AOTAutogradCache, whose key sees
        # neither the device assignment behind collectives, opaque Pallas kernel
        # sources nor the XLA flags: a stale hit can hang or ignore new flags.
        from sglang.srt.hardware_backend.tpu.compile_cache import (
            add_pallas_kernel_sources_to_compile_cache_key,
            add_tpu_rank_assignment_to_compile_cache_key,
        )

        add_tpu_rank_assignment_to_compile_cache_key()
        add_pallas_kernel_sources_to_compile_cache_key()
        _add_compiler_flags_to_compile_cache_key()

    def get_torch_distributed_backend_str(self) -> str:
        # tpu_dist needs one addressable PJRT device; a lone worker has many.
        return "gloo" if _world_size() == 1 else "tpu_dist"

    def apply_server_args_defaults(self, server_args) -> None:
        # The compile warmup layerwise-offloads the DiT, which TPU cannot run.
        if not server_args.is_arg_explicitly_set("offload_during_compile"):
            server_args.offload_during_compile = False

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend in (None, AttentionBackendEnum.TORCH_SDPA):
            logger.info("Using Torch SDPA backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "sdpa.SDPABackend"
            )
        if selected_backend is AttentionBackendEnum.TPU_SPLASH_ATTN:
            logger.info("Using TPU Pallas splash attention backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "tpu_splash_attn.TpuSplashAttentionBackend"
            )
        raise ValueError(
            f"Attention backend {selected_backend} is not supported on TPU; "
            "the supported backends are TORCH_SDPA (default) and "
            "TPU_SPLASH_ATTN."
        )

    @classmethod
    def supports_vae_torch_compile(cls) -> bool:
        # Off for every VAE: the torch_tpu backend fails to compile the diffusers
        # AutoencoderKL decoder (torch-tpu 0.1.1+release.2026.9.22), and the other
        # VAEs are untested under it.
        return False

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        return False
