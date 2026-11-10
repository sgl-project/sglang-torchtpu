# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash multimodal TPU compile boundary and host feature staging.

compile_glm5_next_decoder_only, encode_glm5_next_vision_on_host
  Why: Glm5NextForConditionalGeneration declares no TPU compile boundary, so
       TpuGraphRunner falls back to compiling the outer forward with text-only
       warmup batches (mm_inputs=None). When the vision encoder is enabled
       (no --language-model-only), the first image request violates Dynamo's
       contains_mm_inputs() == False guard and triggers a serve-time recompile
       that crashes under fail_on_recompile. In addition, get_image_feature and
       get_video_feature return TPU device tensors while the default TPU
       multimodal embedding path (embed_mm_on_host) stages rows on CPU.
  How: AFTER Glm5NextForConditionalGeneration.__init__ registers
       Qwen3VLDecoderTpuCompilePolicy on model.model (Glm5NextModel) and sets
       decode_embeds_in_language_model = False whenever model.visual is
       present, keeping --language-model-only on the whole-model fallback.
       AROUND Glm5NextForConditionalGeneration.get_image_feature and
       get_video_feature lands vision outputs on CPU unless
       SGLANG_TPU_MM_EMBED_ON_DEVICE is enabled.
  Upstream PR or issue: none.
  Remove when: glm5_next declares its own TPU multimodal compile boundaries.
"""

from __future__ import annotations

from typing import Any, Callable, List

import torch


def compile_glm5_next_decoder_only(
    result: None, model: Any, *args, **kwargs
) -> None:
    """AFTER Glm5NextForConditionalGeneration.__init__."""
    if (
        getattr(model, "visual", None) is None
        or getattr(model, "model", None) is None
    ):
        return  # --language-model-only or encoder-only: keep default behavior.
    from sglang.srt.compilation.support_torch_compile import support_torch_compile
    from sglang_tpu.compilation.decoder_compile_policy import (
        Qwen3VLDecoderTpuCompilePolicy,
    )

    # Read by the VL decoder policy: decode feeds input_embeds from
    # general_mm_embed_routine, not input_ids.
    model.decode_embeds_in_language_model = False
    support_torch_compile(model.model, policy=Qwen3VLDecoderTpuCompilePolicy)


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
