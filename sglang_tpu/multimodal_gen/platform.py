# SPDX-License-Identifier: Apache-2.0
"""multimodal_gen's TPU platform as an out-of-tree plugin.

The ``tpu_oot`` entry point of ``sglang.multimodal_gen.platforms`` selects
:class:`TpuOOTPlatform` on TPU hosts (see :func:`sglang_tpu.multimodal_gen.activate`).
It is the in-tree ``TpuPlatform`` with ``PlatformEnum.OOT`` plus the plugin's
TPU behavior. ``is_tpu()`` stays true, so core TPU paths keep applying.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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
    def supports_safetensors_device_load(cls) -> bool:
        # safetensors rejects 'tpu:N'.
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
        return super().get_attn_backend_cls_str(selected_backend, head_size, dtype)
