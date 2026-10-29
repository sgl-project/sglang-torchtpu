# SPDX-License-Identifier: Apache-2.0
"""multimodal_gen's TPU platform as an out-of-tree plugin.

The ``tpu_oot`` entry point of ``sglang.multimodal_gen.platforms`` selects
:class:`TpuOOTPlatform` on TPU hosts (see :func:`sglang_tpu.multimodal_gen.activate`).
It is the in-tree ``TpuPlatform`` with ``PlatformEnum.OOT`` plus the plugin's
TPU behavior. ``is_tpu()`` stays true, so core TPU paths keep applying.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.multimodal_gen import worker

logger = init_logger(__name__)

# The --attention-backend name of the Pallas backend (AttentionBackendEnum.OOT).
SPLASH_ATTN_BACKEND_NAME = "tpu_splash_attn"
_SPLASH_ATTN_BACKEND_CLS = (
    "sglang_tpu.multimodal_gen.tpu_splash_attn.TpuSplashAttentionBackend"
)


class TpuOOTPlatform(TpuPlatform):
    _enum = PlatformEnum.OOT

    def is_tpu(self) -> bool:
        # Kernel fallbacks, CustomOp.forward_tpu and the TPU argument checks
        # key on is_tpu().
        return True

    def get_dispatch_key_name(self) -> str:
        # CustomOp dispatch takes forward_tpu where an op defines it, as in-tree.
        return "tpu"

    @classmethod
    def get_all_to_all_communicator_cls(cls) -> str:
        # The base communicator's all-to-all, as the in-tree TPU platform uses.
        return Platform.get_device_communicator_cls()

    @classmethod
    def get_oot_attn_backend_name(cls) -> str | None:
        return SPLASH_ATTN_BACKEND_NAME

    @classmethod
    def oot_attn_backend_serves(
        cls, supported_attention_backends: set[AttentionBackendEnum]
    ) -> bool:
        # The splash backend runs SDPA for shapes its kernels do not cover, so a
        # ring launch (which selects it automatically) can serve any layer that
        # takes TORCH_SDPA.
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
        return super().get_attn_backend_cls_str(selected_backend, head_size, dtype)

    @classmethod
    def _init_worker_process(cls) -> None:
        super()._init_worker_process()
        # The torch_tpu backend compiles static shapes only. The out-of-tree
        # compile path leaves dynamic=None, so keep a recompile for a new shape
        # static too.
        import torch._dynamo

        torch._dynamo.config.automatic_dynamic_shapes = False
        worker.install_runtime_patches()

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
        import torch.compiler.config as compiler_config

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
        tag = "spmd-safe-regions-v1"
        existing_tag = str(compiler_config.cache_key_tag or "")
        if tag not in existing_tag.split("|"):
            compiler_config.cache_key_tag = (
                f"{existing_tag}|{tag}" if existing_tag else tag
            )
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
    def drain_device_queue_after_request(cls) -> bool:
        # Without it Wan2.2-A14B hangs in its second warmup request: 7 of 8
        # ranks block in a host-to-device copy behind collectives one rank
        # never queued (4 of 4 runs, torch_tpu 0.2.0rc4 and 0.2.0). Output
        # and warm latency are unchanged.
        return True

    @classmethod
    def fusion_barrier(cls, x: torch.Tensor) -> torch.Tensor:
        # Only a compiled XLA program fuses; in eager the barrier would copy x.
        if x.device.type != "tpu" or not torch.compiler.is_compiling():
            return x
        barrier = getattr(getattr(torch.tpu, "ops", None), "optimization_barrier", None)
        if barrier is None:
            # torch_tpu releases before torch.tpu.ops expose only the raw op.
            barrier = torch.ops.tpu.optimization_barrier
        return barrier([x])[0]
