# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash multimodal TPU compile boundaries and vision runner hooks.

attach_glm5_next_vision_runner, compile_glm5_next_decoder_only,
defer_host_staged_mm_feature_move, encode_with_glm5_next_vision_runner,
encode_glm5_next_vision_on_host
  Why: Glm5NextForConditionalGeneration and Glm5NextVisionModel declare no TPU
       compile boundaries in core (which cannot import sglang_tpu). Without
       them, TpuGraphRunner's whole-model fallback compiles the outer forward
       with text-only warmup batches and crashes under fail_on_recompile on the
       first image request, while Glm5NextVisionModel runs eagerly across
       variable patch counts and triggers Inductor recompiles on
       @torch.compile-decorated swiglu_clamped. In addition, get_image_feature
       and get_video_feature return TPU device tensors while the default TPU
       multimodal embedding path (embed_mm_on_host) stages rows on CPU, and
       mm_schedule._can_skip_pre_embed_feature_move otherwise uploads pixels to
       TPU before the host-staged vision runner immediately copies them back.
  How: AFTER Glm5NextVisionModel.__init__ attaches Glm5NextVitTpuCompileRunner
       as visual.graph_runners, unwraps glm5_next.swiglu_clamped on TPU, and
       annotates Glm5NextVisionModel via core_compile_annotations. AFTER
       Glm5NextForConditionalGeneration.__init__ sets
       model.keeps_mm_features_on_host = not use_dp_encoder when model.visual
       is present and registers Qwen3VLDecoderTpuCompilePolicy on model.model
       (Glm5NextModel) unless mm_config.language_model_only or
       --enable-dp-attention is set (covering both standard multimodal and EPD
       language-only servers). Under --language-model-only without DP
       attention it registers Glm5NextDecoderTpuCompilePolicy (spmd_safe) on
       the whole model; --encoder-only and --enable-dp-attention keep the
       default whole-model fallback. AROUND
       mm_schedule._can_skip_pre_embed_feature_move honors
       owner.keeps_mm_features_on_host so non-DP host-staged vision avoids an
       H2D -> blocking D2H -> H2D round-trip while DP encoder keeps device H2D.
       AROUND Glm5NextVisionModel.forward stages each frame on the host at a
       padded patch bucket and runs the compiled runner (falling back to eager
       for oversized grids). AROUND
       Glm5NextForConditionalGeneration.get_{image,video}_feature ensures
       outputs land on CPU unless SGLANG_TPU_MM_EMBED_ON_DEVICE is enabled.
  Upstream PR or issue: none.
  Remove when: glm5_next declares its own TPU multimodal compile boundaries.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable, List

import torch

logger = logging.getLogger(__name__)


def _force_eager():
    return getattr(torch.compiler, "set_stance", lambda _s: nullcontext())(
        "force_eager"
    )


def attach_glm5_next_vision_runner(result: None, visual: Any, *args, **kwargs) -> None:
    """AFTER Glm5NextVisionModel.__init__: attach the TPU compile runner."""
    from sglang.srt.models import glm5_next
    from sglang_tpu.compilation.core_compile_annotations import (
        annotate_core_compile_boundaries,
    )
    from sglang_tpu.graph_runner.glm5_next_vit_tpu_compile_runner import (
        Glm5NextVitTpuCompileRunner,
    )

    swiglu = getattr(glm5_next, "swiglu_clamped", None)
    if hasattr(swiglu, "__wrapped__"):
        glm5_next.swiglu_clamped = swiglu.__wrapped__
    visual.graph_runners = Glm5NextVitTpuCompileRunner(visual)
    annotate_core_compile_boundaries()


def compile_glm5_next_decoder_only(result: None, model: Any, *args, **kwargs) -> None:
    """AFTER Glm5NextForConditionalGeneration.__init__."""
    visual = getattr(model, "visual", None)
    if visual is not None:
        # Non-DP vision runs host-staged bucketed frames; keeping features on
        # CPU avoids an H2D -> blocking D2H -> H2D round-trip. DP encoder
        # broadcasts allocate on pixel_values.device and must keep device H2D.
        use_dp_encoder = bool(
            getattr(model, "use_data_parallel", False)
            or getattr(visual, "use_data_parallel", False)
        )
        model.keeps_mm_features_on_host = not use_dp_encoder

    if getattr(model, "model", None) is None:
        return  # encoder-only: no decoder.

    from sglang.srt.compilation.support_torch_compile import support_torch_compile
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    language_model_only = bool(
        getattr(getattr(model, "mm_config", None), "language_model_only", False)
    )
    if language_model_only:
        if not is_dp_attention_enabled():
            from sglang_tpu.models.glm5_next_compile_policy import (
                Glm5NextDecoderTpuCompilePolicy,
            )

            # The whole-model boundary of the fallback, declared spmd_safe.
            support_torch_compile(model, policy=Glm5NextDecoderTpuCompilePolicy)
        return

    if is_dp_attention_enabled():
        logger.warning(
            "GLM-5.3-Flash multimodal TPU compilation does not support "
            "--enable-dp-attention; falling back to whole-model compilation. "
            "Pass --language-model-only for text-only serving with DP attention."
        )
        return

    from sglang_tpu.compilation.decoder_compile_policy import (
        Qwen3VLDecoderTpuCompilePolicy,
    )

    # Read by the VL decoder policy: decode feeds input_embeds from
    # general_mm_embed_routine, not input_ids.
    model.decode_embeds_in_language_model = False
    support_torch_compile(model.model, policy=Qwen3VLDecoderTpuCompilePolicy)


def defer_host_staged_mm_feature_move(
    original_fn: Callable,
    data_embedding_func: Any,
) -> bool:
    """AROUND mm_schedule._can_skip_pre_embed_feature_move."""
    owner = getattr(data_embedding_func, "__self__", None)
    if getattr(owner, "keeps_mm_features_on_host", False) and getattr(
        data_embedding_func, "__name__", None
    ) in ("get_image_feature", "get_video_feature"):
        return True
    return original_fn(data_embedding_func)


def encode_with_glm5_next_vision_runner(
    original_fn: Callable,
    visual: Any,
    x: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """AROUND Glm5NextVisionModel.forward."""
    runner = getattr(visual, "graph_runners", None)
    if runner is None or not runner.ready:
        with _force_eager():
            return original_fn(visual, x, grid_thw)
    output = runner.run_grid(x, grid_thw)
    if output is None:
        with _force_eager():
            return original_fn(visual, x, grid_thw)
    return output


def encode_glm5_next_vision_on_host(
    original_fn: Callable,
    model: Any,
    items: List[Any],
) -> torch.Tensor:
    """AROUND Glm5NextForConditionalGeneration.get_{image,video}_feature."""
    features = original_fn(model, items)
    if not isinstance(features, torch.Tensor):
        return features
    from sglang.srt.environ import envs

    if envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get():
        target_device = getattr(getattr(model, "visual", None), "device", None)
        if target_device is not None and features.device != target_device:
            return features.to(target_device)
        return features
    if features.device.type != "cpu":
        return features.cpu()
    return features
