# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 vision tower on TPU: host-staged inputs, compiled per patch bucket.

attach_k3_vision_runner, bind_k3_vision_runner, encode_with_k3_vision_runner,
pass_through_projected_features
  Why: Kimi-K3's tower runs eagerly on TPU, so every new image grid compiled
       the patch embedding, position-embedding interpolation, RoPE lookup,
       encoder layers, merger and projector while serving (the TODO in
       KimiK3ForConditionalGeneration.get_image_feature). The TPU runner and
       its compile policy (sglang_tpu.graph_runner.kimi_k3_vit_tpu_compile_runner)
       must stay out of core, which cannot import sglang_tpu.
  How: AFTER kimi_k3_vl.MoonViT3dEncoder.__init__ attaches the runner as the
       encoder's graph_runners, where KimiK3VisionTpuCompilePolicy (declared in
       core_compile_annotations) finds and compiles it. AFTER
       KimiK3ForConditionalGeneration.__init__ binds it the tower and the
       projector. AROUND KimiK3VisionTower._forward_eager stages each image on
       the host and runs tower + projector on the runner -- compiled per padded
       bucket under --enable-torch-compile, the same math eagerly otherwise --
       and AROUND KimiK3MultiModalProjector.forward passes those already
       projected features through. Video grids and images past the ladder keep
       the tower's own eager path.
  Upstream PR or issue: none; the CUDA path has its own graph runner
       (KimiK3ViTCudaGraphRunner) selected inside the tower.
  Remove when: the tower selects its accelerator runner through a platform
       seam instead of an is_cuda branch.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch


class ProjectedImageFeatures(list):
    """Per-image projector outputs (host tensors) the TPU runner produced."""


def attach_k3_vision_runner(result: None, encoder: Any, *args, **kwargs) -> None:
    """AFTER MoonViT3dEncoder.__init__ (Kimi-K3's): give the encoder its TPU runner."""
    from sglang_tpu.graph_runner.kimi_k3_vit_tpu_compile_runner import (
        KimiK3VitTpuCompileRunner,
    )

    encoder.graph_runners = KimiK3VitTpuCompileRunner(encoder)


def bind_k3_vision_runner(result: None, model: Any, *args, **kwargs) -> None:
    """AFTER KimiK3ForConditionalGeneration.__init__: the runner runs the tower's
    patch embedding and the wrapper's projector too."""
    tower = getattr(model, "vision_tower", None)
    projector = getattr(model, "mm_projector", None)
    runner = getattr(getattr(tower, "encoder", None), "graph_runners", None)
    if runner is not None and projector is not None:
        runner.bind(tower, projector)


def encode_with_k3_vision_runner(
    original_fn: Callable,
    tower: Any,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    *,
    grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
    forward_metadata: Any = None,
) -> Any:
    """AROUND KimiK3VisionTower._forward_eager."""
    from sglang.srt.models.kimi_k3_vl import _resolve_grid_thw_list

    def eager():
        return original_fn(
            tower,
            pixel_values,
            grid_thws,
            grid_thw_list=grid_thw_list,
            forward_metadata=forward_metadata,
        )

    runner = getattr(tower.encoder, "graph_runners", None)
    if runner is None or not runner.ready or forward_metadata is not None:
        return eager()
    shapes = _resolve_grid_thw_list(grid_thws, grid_thw_list)
    # One device-to-host copy; the runner stages every image on the host.
    outputs = runner.run_images(pixel_values.cpu(), shapes)
    if outputs is None:
        return eager()
    return ProjectedImageFeatures(outputs)


def pass_through_projected_features(
    original_fn: Callable, projector: Any, image_features: Any
) -> Any:
    """AROUND KimiK3MultiModalProjector.forward: the runner already projected."""
    if not isinstance(image_features, ProjectedImageFeatures):
        return original_fn(projector, image_features)
    features = torch.cat(list(image_features))
    from sglang.srt.environ import envs

    if envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get():
        features = features.to(projector.post_norm.weight.device)
    return features
