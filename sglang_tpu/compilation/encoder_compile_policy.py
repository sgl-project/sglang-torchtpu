# SPDX-License-Identifier: Apache-2.0
"""Compile-policy installation for a TPU vision encoder that has no ModelRunner."""

from __future__ import annotations

import logging

import torch

from sglang.srt.compilation.support_torch_compile import get_torch_compile_policy
from sglang_tpu.graph_runner.vit_tpu_compile_runner import VisionTpuCompilePolicy

logger = logging.getLogger(__name__)


def install_vision_compile_policies(
    model: torch.nn.Module,
) -> list[VisionTpuCompilePolicy]:
    """Install and warm every TPU vision compile policy attached to model.

    The EPD encode server's counterpart of TpuGraphRunner's policy loop; that
    process has no language model, so decoder policies are left alone.

    Raises:
        RuntimeError: No vision policy installed, so every image would run eager.
    """
    installed: list[VisionTpuCompilePolicy] = []
    pending_warmups = []
    annotated = 0
    for name, module in model.named_modules():
        policy_cls = get_torch_compile_policy(module)
        if policy_cls is None or not issubclass(policy_cls, VisionTpuCompilePolicy):
            continue
        annotated += 1
        policy = policy_cls(
            owner=module,
            method_name="forward",
            method=module.forward,
            name=f"encoder/{name + '.' if name else ''}forward",
            label=type(module).__name__,
        )
        if not policy.install():
            continue
        installed.append(policy)
        pending_warmups.extend(policy.precompile() or ())

    if not installed:
        raise RuntimeError(
            f"{type(model).__name__} installed no TPU vision compile policy "
            f"({annotated} annotated module(s)); the encoder would run eager. "
            "Check that the ViT attention backend supports segment IDs."
        )
    for warmup in pending_warmups:
        warmup.wait()
    logger.info(
        "Installed %d TPU vision compile policies on the encoder: %s.",
        len(installed),
        ", ".join(type(policy).__name__ for policy in installed),
    )
    return installed
