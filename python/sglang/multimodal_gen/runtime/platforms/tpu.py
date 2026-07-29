# SPDX-License-Identifier: Apache-2.0
"""TPU platform for sglang-diffusion, backed by torch_tpu (PrivateUse1/PJRT).

All TPU-specific behavior for the diffusion runtime lives in this module; the
shared runtime reaches it exclusively through the Platform hooks defined in
`interface.py`. torch_tpu imports are deferred to call time so this module
imports cleanly on machines without torch_tpu installed.

Execution model notes (torch_tpu is a lazy/deferred backend):
- One chip per process. The chip is chosen by the slicebuilder env computed in
  the parent process (`prepare_server_environment`); the process-local device
  index is always 0.
- torch_tpu latches RANK/LOCAL_RANK/WORLD_SIZE at import (GetEnvOnce), and a
  worker's import chain touches torch_tpu before the worker's own setup runs,
  so those variables are exported at process birth via `worker_spawn_env`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

logger = init_logger(__name__)

_MIB = 1024**2

# DiT modules evicted to host while the VAE decodes. Wan names its (optionally
# two-expert) transformer this way; extend when other pipelines are validated
# on TPU.
_VAE_DECODE_EVICT_MODULES = ("transformer", "transformer_2")

# Env set by jax when it loads first (it is pulled in transitively by the
# pallas custom-op path). libtpu then initializes in JAX mode and torch_tpu's
# multi-process PJRT init fails on the libtpu lockfile.
_CONFLICTING_JAX_ENV = ("TPU_ML_PLATFORM", "TPU_ML_PLATFORM_VERSION")


def _scrub_conflicting_jax_env() -> None:
    """Remove jax-owned libtpu env before the first TPU device touch."""
    for name in _CONFLICTING_JAX_ENV:
        os.environ.pop(name, None)


class _DitEvictionContext:
    """Evicts the DiT(s) to host for the duration of the VAE decode.

    A 14B-class DiT plus decode activations exceed HBM at 720p; PJRT has no
    swappable allocator, so the working set must be moved explicitly.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self._evicted: list[torch.nn.Module] = []

    def __enter__(self) -> "_DitEvictionContext":
        if self._pipeline is None:
            return self
        for name in _VAE_DECODE_EVICT_MODULES:
            module = self._pipeline.get_module(name)
            if isinstance(module, torch.nn.Module):
                module.cpu()
                self._evicted.append(module)
        if self._evicted:
            torch.tpu.synchronize()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._evicted:
            torch.tpu.synchronize()
            device = TpuPlatform.get_local_torch_device()
            for module in self._evicted:
                module.to(device)
            self._evicted.clear()


class _XprofSession:
    """torch_tpu xprof capture wrapped as a context manager.

    torch.profiler cannot see TPU device activity; xprof runs alongside it and
    writes a trace viewable with xprof/tensorboard into ``log_dir``.
    """

    def __init__(self, log_dir: str) -> None:
        self._log_dir = log_dir
        self._session: Any = None

    def __enter__(self) -> "_XprofSession":
        from torch_tpu._internal.profiler import profiler_api

        self._session = profiler_api.profile(
            activities=[
                profiler_api.ProfilerActivity.TPU,
                profiler_api.ProfilerActivity.CPU,
            ],
            on_trace_ready=profiler_api.xprof_trace_handler(dir_name=self._log_dir),
        )
        self._session.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._session is None:
            return
        # Flush outstanding lazy work so device activity lands in the trace.
        torch.tpu.synchronize()
        self._session.__exit__(*exc_info)
        self._session = None


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    # torch_tpu registers the "tpu" dynamo backend on import.
    simple_compile_backend: str = "tpu"

    # ------------------------------------------------------------------
    # Device identity and memory
    # ------------------------------------------------------------------

    # Identity and memory queries must NEVER initialize PJRT as a side
    # effect: server-args processing calls them in the parent process before
    # workers spawn, and the chips belong to the workers. Anything needing
    # live allocator stats gates on torch.tpu.is_initialized().

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        # One chip per process; the process-local index is always 0.
        return torch.device("tpu", 0)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # Static: torch.tpu.get_device_name() would initialize the runtime.
        return "tpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        if torch.tpu.is_initialized():
            _free, total = torch.accelerator.get_memory_info()
            return total
        # Pre-initialization estimate for server-args threshold checks.
        # ponytail: static floor for current targets (tpu7x ~95 GiB); replace
        # with a topology lookup if smaller-HBM TPUs are targeted.
        return 90 * (1 << 30)

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if not torch.tpu.is_initialized():
            return cls.get_device_total_memory() / (1 << 30)
        free, _total = torch.accelerator.get_memory_info()
        return free / (1 << 30)

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        if not torch.tpu.is_initialized():
            return 0.0
        return float(
            torch.accelerator.memory_stats().get("allocated_bytes.all.current", 0)
        )

    @classmethod
    def empty_cache(cls) -> None:
        pass  # PJRT owns device memory; there is no cache to release.

    @classmethod
    def reset_peak_memory_stats(cls) -> None:
        if torch.tpu.is_initialized():
            torch.accelerator.reset_peak_memory_stats()

    @classmethod
    def get_memory_snapshot(cls) -> MemorySnapshot:
        from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

        if not torch.tpu.is_initialized():
            return MemorySnapshot(
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_allocated_mb=0.0,
                peak_reserved_mb=0.0,
            )
        stats = torch.accelerator.memory_stats()
        # PJRT reports allocated/peak-allocated; bytes_reserved is not
        # populated by the TPU client and reads as 0.
        return MemorySnapshot(
            allocated_mb=stats.get("allocated_bytes.all.current", 0) / _MIB,
            reserved_mb=stats.get("reserved_bytes.all.current", 0) / _MIB,
            peak_allocated_mb=stats.get("allocated_bytes.all.peak", 0) / _MIB,
            peak_reserved_mb=stats.get("reserved_bytes.all.peak", 0) / _MIB,
        )

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def is_float64_supported(cls) -> bool:
        return False  # TPUs have no f64 compute units.

    @classmethod
    def inference_mode(cls):
        # torch_tpu's lazy tensors do not support torch.inference_mode.
        return torch.no_grad()

    # ------------------------------------------------------------------
    # Process / distributed environment
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        return self.get_local_torch_device()

    def get_torch_distributed_backend_str(self) -> str:
        # torch_tpu's native c10d backend: collectives become deferred
        # in-graph StableHLO ops matching the SPMD programs torch_tpu
        # compiles per process.
        return "tpu_dist"

    @classmethod
    def prepare_server_environment(cls, *, world_size: int) -> None:
        _scrub_conflicting_jax_env()
        # Reuses the srt TPU launcher helper: applies LIBTPU_INIT_ARGS
        # overrides and writes the single-host slicebuilder/topology env that
        # spawned workers inherit so each binds its own chip via PJRT.
        from sglang.srt.hardware_backend.tpu.patches import _prepare_tpu_env

        _prepare_tpu_env(world_size=world_size)

    @classmethod
    def worker_spawn_env(
        cls, *, rank: int, world_size: int, master_port: int
    ) -> dict[str, str]:
        # torch_tpu latches these via GetEnvOnce during import, which the
        # child's import chain reaches before init_worker_device_env runs.
        return {
            "LOCAL_RANK": str(rank),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }

    @classmethod
    def init_worker_device_env(
        cls, *, local_rank: int, rank: int, world_size: int, master_port: int
    ) -> None:
        # Order matters: env scrub and libtpu flag overrides must precede the
        # first device touch (PJRT initialization is deferred until then).
        _scrub_conflicting_jax_env()
        from sglang.srt.hardware_backend.tpu.env_override import (
            apply_tpu_env_overrides,
        )

        apply_tpu_env_overrides()

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # One chip per process; the chip itself is selected by the
        # slicebuilder env, so the local index is always 0.
        torch.tpu.set_device(0)

        # Touch the device now, while all ranks are at the same point in
        # startup, so the single-host slicebuilder ranks rendezvous together
        # instead of first touching the device mid-model-load.
        torch.zeros(1, device="tpu")

    # ------------------------------------------------------------------
    # Attention and compile
    # ------------------------------------------------------------------

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend in (None, AttentionBackendEnum.TPU_PALLAS_FLASH):
            logger.info("Using TPU pallas flash attention backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "tpu_pallas.TpuPallasAttentionBackend"
            )
        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "sdpa.SDPABackend"
            )
        raise ValueError(
            f"Attention backend {selected_backend} is not supported on TPU; "
            "supported backends: TPU_PALLAS_FLASH (default), TORCH_SDPA."
        )

    @classmethod
    def supports_op_level_torch_compile(cls) -> bool:
        return False  # the op decorators target inductor, which cannot lower to TPU

    @classmethod
    def supports_vae_decode_torch_compile(cls) -> bool:
        # The wanvae frame loop feeds its temporal feature-cache concat with
        # per-chunk shapes; under torch.compile the graph-break resume regions
        # recompile per shape on a static-shape backend. Lazy decode is the
        # validated path.
        return False

    @classmethod
    def wrap_forward_for_compile(cls, forward: Callable[..., Any]) -> Callable[..., Any]:
        # Keeps collectives inside the captured graph so torch_tpu's
        # SplitCompiler can partition compute+collective+compute as designed.
        from torch_tpu._internal.distributed.spmd_util import spmd_safe

        return spmd_safe(forward)

    # ------------------------------------------------------------------
    # Distributed capabilities
    # ------------------------------------------------------------------

    @classmethod
    def supports_p2p_communication(cls) -> bool:
        # tpu_dist implements collectives; send/recv exists only as
        # experimental ops and is not wired into the c10d backend.
        return False

    @classmethod
    def use_functional_seq_all_to_all(cls) -> bool:
        # No CUDA-style streams on TPU, and the functional collective stays
        # traceable in compiled graphs.
        return True

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "sglang.multimodal_gen.runtime.distributed.device_communicators."
            "cpu_communicator.CpuCommunicator"
        )

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        return False

    @classmethod
    def vae_decode_context(cls, pipeline: Any) -> Any:
        return _DitEvictionContext(pipeline)

    @classmethod
    def denoise_loop_context(cls) -> Any:
        # DEFER_AND_FUSE (default) batches the step's ops into fused programs.
        # Scoped to the denoising loop: the VAE's feature-cache decode
        # regresses under this mode. SGLANG_DIFFUSION_TPU_EAGER_MODE selects
        # another torch_tpu EagerMode (e.g. DEFER_NEVER for op-by-op runs).
        from torch_tpu._internal.execution_mode import EagerMode, set_eager_mode

        from sglang.multimodal_gen import envs

        mode_name = envs.SGLANG_DIFFUSION_TPU_EAGER_MODE
        # EagerMode is a pybind11 enum: look up via __members__.
        mode = EagerMode.__members__.get(mode_name)
        if mode is None:
            valid = ", ".join(EagerMode.__members__)
            raise ValueError(
                f"SGLANG_DIFFUSION_TPU_EAGER_MODE={mode_name!r} is not a "
                f"torch_tpu EagerMode; valid values: {valid}."
            )
        return set_eager_mode(mode)

    @classmethod
    def on_denoise_step_end(cls) -> None:
        # Bound the in-flight lazy queue to one step's programs; letting it
        # grow across steps exhausts HBM with pending intermediaries.
        torch.tpu.synchronize()

    @classmethod
    def create_aux_profiler(cls, log_dir: str) -> Any | None:
        return _XprofSession(log_dir)
