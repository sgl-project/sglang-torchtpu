# SPDX-License-Identifier: Apache-2.0
"""TPU implementations for sglang.multimodal_gen.

:func:`activate` is the ``tpu_oot`` entry point of
``sglang.multimodal_gen.platforms``.
"""

PLATFORM_CLASS_PATH = "sglang_tpu.multimodal_gen.platform.TpuOOTPlatform"


def activate() -> str | None:
    """Select :class:`~sglang_tpu.multimodal_gen.platform.TpuOOTPlatform` on TPU."""
    import torch

    # torch_tpu autoloads with torch and registers torch.tpu only when it finds
    # hardware; the built-in TPU detection checks the same.
    if getattr(torch, "tpu", None) is None:
        return None
    return PLATFORM_CLASS_PATH
