# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash multimodal TPU compile boundaries and vision runner hooks.

attach_glm5_next_vision_runner, compile_glm5_next_decoder_only,
forward_glm5_next_multimodal, attach_glm5_next_nextn_multimodal_forward,
defer_host_staged_mm_feature_move, encode_with_glm5_next_vision_runner,
encode_glm5_next_vision_on_host
  Why: Glm5NextForConditionalGeneration and Glm5NextVisionModel declare no TPU
       compile boundaries in core (which cannot import sglang_tpu). Without
       them, TpuGraphRunner's whole-model fallback compiles the outer forward
       with text-only warmup batches and crashes under fail_on_recompile on the
       first image request, while Glm5NextVisionModel runs eagerly across
       variable patch counts and triggers Inductor recompiles on
       @torch.compile-decorated swiglu_clamped. Compiling only Glm5NextModel
       with Qwen3VLDecoderTpuCompilePolicy also compiles dead text_ids graphs,
       leaves embeddings and logits eager outside the XLA graph, breaks
       --enable-attn-tp-input-scattered (guard miss + tp_size vision scaling),
       uses core embed_mm_inputs instead of embed_mm_inputs_tpu, and drops
       padded mm_input_embeds needed by multimodal NEXTN MTP (while
       Glm5NextForConditionalGenerationNextN inherits DeepseekV3ForCausalLMNextN
       whose outer forward omits input_embeds). In addition,
       get_image_feature and get_video_feature return TPU device tensors while
       the default TPU multimodal embedding path (embed_mm_on_host) stages rows
       on CPU, and mm_schedule._can_skip_pre_embed_feature_move otherwise
       uploads pixels to TPU before the host-staged vision runner immediately
       copies them back.
  How: AFTER Glm5NextVisionModel.__init__ attaches Glm5NextVitTpuCompileRunner
       as visual.graph_runners, unwraps glm5_next.swiglu_clamped on TPU, and
       annotates Glm5NextVisionModel via core_compile_annotations. AFTER
       Glm5NextForConditionalGeneration.__init__ sets
       model.keeps_mm_features_on_host = not use_dp_encoder when model.visual
       is present, sets model.decode_embeds_in_language_model = True, attaches
       model.language_model = _Glm5NextLanguageModel(model) (compiling
       maybe_input_scattered + attn_rank0_scale + Glm5NextModel +
       logits_processor together), and registers Glm5NextVLDecoderTpuCompilePolicy
       on model.language_model unless mm_config.language_model_only or
       --enable-dp-attention is set (covering both standard multimodal and EPD
       language-only servers). Under --language-model-only without DP attention
       it registers Glm5NextDecoderTpuCompilePolicy (spmd_safe) on the whole
       model; --encoder-only and --enable-dp-attention keep the default
       whole-model fallback. AROUND
       Glm5NextForConditionalGeneration.forward pins non-verify EXTEND to
       FULL_VOCAB logits (avoiding duplicate prefill compiles), routes DECODE,
       TARGET_VERIFY, and SGLANG_TPU_VL_TEXT_ONLY_EXTEND calls directly into
       model.language_model with keyword defaults, and routes EXTEND through
       general_mm_embed_routine(..., mm_embedder=embed_mm_inputs_tpu) outside
       maybe_input_scattered. AFTER
       Glm5NextForConditionalGenerationNextN.__init__ binds an input_embeds-aware
       forward so multimodal NEXTN MTP draft prompt extends forward folded
       multimodal embeddings into DeepseekModelNextN. AROUND
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
import types
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


class _Glm5NextLanguageModel(torch.nn.Module):
    """Compile the GLM-5.3-Flash decoder and logits together outside ViT prep."""

    def __init__(self, owner: Any) -> None:
        super().__init__()
        # Pre-import before Dynamo traces forward() so sys.modules already holds
        # both modules and no parameters are duplicated on model.language_model.
        import sglang.srt.layers.dp_attention  # noqa: F401
        import sglang.srt.layers.layer_boundary.adapters.attention  # noqa: F401

        object.__setattr__(self, "_owner", owner)

    @property
    def pp_group(self) -> Any:
        return self._owner.pp_group

    @property
    def logits_processor(self) -> Any:
        return self._owner.logits_processor

    @property
    def layers_to_capture(self) -> Any:
        return getattr(self._owner.model, "layers_to_capture", [])

    def get_input_embeddings(self) -> Any:
        return self._owner.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Any = None,
    ) -> Any:
        from sglang.srt.layers.dp_attention import attn_rank0_scale
        from sglang.srt.layers.layer_boundary.adapters.attention import (
            get_attn_tp_context,
        )

        owner = self._owner
        attn_tp_ctx = get_attn_tp_context()
        with attn_tp_ctx.maybe_input_scattered(forward_batch):
            if input_embeds is not None and attn_tp_ctx.input_scattered:
                input_embeds = input_embeds * attn_rank0_scale(input_embeds)
            hidden_states = owner.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        aux_hidden_states = None
        if owner.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if owner.pp_group.is_last_rank:
            logits_input_ids = (
                input_ids if input_ids is not None else forward_batch.input_ids
            )
            return owner.logits_processor(
                logits_input_ids,
                hidden_states,
                owner.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        return hidden_states


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
        Glm5NextVLDecoderTpuCompilePolicy,
    )

    # Decode and verify pass input_ids directly into _Glm5NextLanguageModel so
    # VocabParallelEmbedding and LogitsProcessor compile inside one XLA graph.
    model.decode_embeds_in_language_model = True
    model.language_model = _Glm5NextLanguageModel(model)
    support_torch_compile(
        model.language_model, policy=Glm5NextVLDecoderTpuCompilePolicy
    )


def forward_glm5_next_multimodal(
    original_fn: Callable,
    model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: Any,
    input_embeds: torch.Tensor | None = None,
    pp_proxy_tensors: Any = None,
) -> Any:
    """AROUND Glm5NextForConditionalGeneration.forward."""
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return original_fn(
            model,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    from sglang.srt.managers.mm_utils import general_mm_embed_routine
    from sglang.srt.model_executor.forward_batch_info import LogitsOutputMode
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.mm_embed_tpu import embed_mm_inputs_tpu

    with torch.no_grad():
        if getattr(model, "is_mrope_enabled", False):
            positions = forward_batch.mrope_positions

        forward_mode = forward_batch.forward_mode
        if forward_mode.is_extend() and not forward_mode.is_target_verify():
            forward_batch.logits_output_mode = LogitsOutputMode.FULL_VOCAB

        if (
            input_embeds is not None
            or input_ids is None
            or forward_mode.is_decode()
            or forward_mode.is_target_verify()
            or tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.get()
        ):
            lm_input_ids = None if input_embeds is not None else input_ids
            return language_model(
                lm_input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=language_model,
            multimodal_model=model,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
            mm_embedder=embed_mm_inputs_tpu,
        )


@torch.no_grad()
def _forward_glm5_next_nextn(
    model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: Any,
    input_embeds: torch.Tensor | None = None,
    pp_proxy_tensors: Any = None,
) -> Any:
    if input_embeds is not None:
        hidden_states = model.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
        )
    else:
        hidden_states = model.model(input_ids, positions, forward_batch)
    return model.logits_processor(
        input_ids, hidden_states, model.lm_head, forward_batch
    )


def attach_glm5_next_nextn_multimodal_forward(
    result: None, draft_model: Any, *args, **kwargs
) -> None:
    """AFTER Glm5NextForConditionalGenerationNextN.__init__."""
    draft_model.forward = types.MethodType(_forward_glm5_next_nextn, draft_model)


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
