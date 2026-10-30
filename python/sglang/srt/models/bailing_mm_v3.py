# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang implementation of Bailing/Ling 3 VL image and video inference."""

import logging
from array import array
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.configs.bailing_hybrid import is_bailing_multi_gate_enabled
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe_v3 import (
    BailingMoeV3ForCausalLM,
)
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.multimodal.mm_utils import materialize_multimodal_features
from sglang.srt.runtime_context import get_mm, get_parallel
from sglang.srt.utils import add_prefix, is_tpu

logger = logging.getLogger(__name__)

_is_tpu = is_tpu()

_PUBLIC_VISION_PARAMETER_PREFIXES = (
    "visual.patch_embed.",
    "visual.pos_embed.",
    "visual.blocks.",
    "visual.merger.",
)
_NON_TEXT_PREFIX_MAPPING = (
    ("model.visual.", "visual."),
    ("linear_proj.", "linear_proj."),
    # Compatibility with the private training-checkpoint wrapper.
    ("model.linear_proj.", "linear_proj."),
)


class BailingMoeV3VLModel(BailingMoeV3ForCausalLM):
    """VL-specific decoder, so the boundary-table policy on this class stays
    off the text-only BailingMoeV3ForCausalLM (same split as
    Qwen3MoeLLMModel vs Qwen3MoeModel)."""


class BailingMoeV3VLForConditionalGeneration(nn.Module):
    """Bailing MoE V3 language model with Qwen3 vision encoding."""

    # Decode / verify embed inside the compiled language model instead of
    # an eager general_mm_embed_routine in front of it; prefill keeps the
    # routine, routing both image and text extends onto the policy-warmed
    # input_embeds face (the Qwen3-VL boundary -- the text_ids warmup bakes
    # 1D positions, so a direct mrope-shaped call cannot reuse it).
    decode_embeds_in_language_model: bool = _is_tpu

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return BailingMoeV3ForCausalLM.shared_experts_fusion_disable_reason(
            hf_config.text_config, quant_config
        )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.quant_config = quant_config
        self.norm_query_embeds = getattr(config, "norm_query_embeds", False)
        self.use_data_parallel = get_mm().mm_enable_dp_encoder

        text_config = config.text_config
        self.multi_gate_enabled = is_bailing_multi_gate_enabled(text_config)
        self.model = BailingMoeV3VLModel(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.vision_config is None:
            raise ValueError("BailingMoeV3VL requires vision_config")
        self._build_mm_encoders = self.pp_group.is_first_rank
        if self._build_mm_encoders:
            self.visual = Qwen3VLMoeVisionModel(
                config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
                use_data_parallel=self.use_data_parallel,
            )
        else:
            self.visual = PPMissingLayer()

        self.disable_merger_proj = getattr(
            config.vision_config, "disable_merger_proj", False
        )
        self.deepstack_visual_indexes = tuple(
            getattr(config.vision_config, "deepstack_visual_indexes", None) or ()
        )
        self.vision_out_dim = (
            config.vision_config.hidden_size
            * config.vision_config.spatial_merge_size**2
            if self.disable_merger_proj
            else config.vision_config.out_hidden_size
        )
        if self._build_mm_encoders:
            self.linear_proj = nn.Sequential(
                nn.Linear(self.vision_out_dim, text_config.hidden_size),
                nn.GELU(),
                nn.Linear(text_config.hidden_size, text_config.hidden_size),
            )
        else:
            self.linear_proj = PPMissingLayer()

        self.is_mrope_enabled = "mrope_section" in text_config.rope_parameters
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def _materialize_items(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return materialize_multimodal_features(
            [item.feature for item in items],
            device=self.visual.device,
            dtype=self.visual.dtype,
        )

    def _get_vision_feature(
        self, items: List[MultimodalDataItem], grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if self.use_data_parallel:
            if _is_tpu:
                # The DP runner shards and all-gathers on device, which is
                # incompatible with the host-staged TPU encoder path.
                raise NotImplementedError(
                    "TPU vision encoding does not support --mm-enable-dp-encoder"
                )
            from sglang.srt.multimodal.mm_utils import (
                run_dp_sharded_mrope_vision_model,
            )

            pixel_values = self._materialize_items(items)
            vision_embeds = run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )
        elif _is_tpu:
            # Assemble on the host at the bucketed patch length so the
            # encoder prologue sees a static shape; forward_with_tpu_graph
            # does the single host->device transfer.
            # Pre-install buckets are empty, so this is an identity pad and
            # the runner executes the blocks eagerly.
            features = [item.feature for item in items]
            if not features or any(f is None for f in features):
                raise ValueError(
                    "TPU vision staging requires a precomputed feature on "
                    "every image item"
                )
            # The gateway renders one batch on one device, so the first
            # feature's device speaks for every item.
            if features[0].device.type == "tpu":
                # A plain-buffer slice copy_ races its own async TPU->host
                # transfer and loads garbage; .cpu() forces the transfer to
                # complete before the host copy is read.
                features = [f.cpu() for f in features]
            total_patches = int(grid_thw.prod(dim=1).sum())
            pixel_values = materialize_multimodal_features(
                features,
                device=torch.device("cpu"),
                dtype=self.visual.dtype,
                pad_to=self.visual.graph_runners.bucket_for_patches(total_patches),
            )
            # forward_with_tpu_graph hard-asserts CPU inputs.
            grid_thw = grid_thw.cpu()
            vision_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        else:
            pixel_values = self._materialize_items(items)
            vision_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        if self.deepstack_visual_indexes:
            expected_dim = (
                len(self.deepstack_visual_indexes) + 1
            ) * self.vision_out_dim
            if vision_embeds.shape[-1] != expected_dim:
                raise ValueError(
                    "Unexpected Bailing vision embedding width: "
                    f"expected={expected_dim}, got={vision_embeds.shape[-1]}"
                )
            vision_embeds = vision_embeds[..., : self.vision_out_dim]

        # The TPU ViT runner stages its output on the host; the bridge
        # projection weights live on the device. Backends that keep the
        # embeds and the weights on the same device skip the move.
        if vision_embeds.device != self.linear_proj[0].weight.device:
            vision_embeds = vision_embeds.to(self.linear_proj[0].weight.device)
        vision_embeds = self.linear_proj(vision_embeds)
        if self.norm_query_embeds:
            vision_embeds = F.normalize(vision_embeds, dim=-1)
        # The TPU embed merge (mm_embed_tpu._place_mm_rows) index_puts these
        # rows into a host matrix unless SGLANG_TPU_MM_EMBED_ON_DEVICE is set.
        if _is_tpu and not envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get():
            return vision_embeds.to("cpu")
        return vision_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if not self._build_mm_encoders:
            raise RuntimeError("Vision encoder is only available on the first PP stage")
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        return self._get_vision_feature(items, image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if not self._build_mm_encoders:
            raise RuntimeError("Vision encoder is only available on the first PP stage")
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        return self._get_vision_feature(items, video_grid_thw)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        del input_embeds, get_embedding
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        if self.decode_embeds_in_language_model and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            # Embed in-graph: the compile policy warms decode, verify and
            # text-only extend as input_ids calls without input_embeds.
            return self.model(
                input_ids, positions, forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return BailingMoeV3ForCausalLM.get_model_config_for_expert_location(
            config.text_config
        )

    def _load_non_text_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
    ) -> Optional[str]:
        # Public layout: model.visual.* for the encoder and top-level
        # linear_proj.* for the bridge. Keep the private nested bridge alias.
        for checkpoint_prefix, parameter_prefix in _NON_TEXT_PREFIX_MAPPING:
            if name.startswith(checkpoint_prefix):
                name = parameter_prefix + name[len(checkpoint_prefix) :]
                break
        else:
            return None
        if name.startswith("visual."):
            name = name.replace("attn.qkv.", "attn.qkv_proj.")
        if name not in params_dict:
            return None
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name

    def _required_non_text_weights(self, params_dict: dict) -> Set[str]:
        required_prefixes = list(_PUBLIC_VISION_PARAMETER_PREFIXES)
        if self.deepstack_visual_indexes:
            required_prefixes.append("visual.deepstack_merger_list.")
        required_prefixes.append("linear_proj.")
        return {
            name
            for name in params_dict
            if self._build_mm_encoders and name.startswith(tuple(required_prefixes))
        }

    def _required_router_weights(self) -> Set[str]:
        gate_names = (
            ("gate", "image_gate", "audio_gate")
            if self.multi_gate_enabled
            else ("gate",)
        )
        suffixes = tuple(
            f".mlp.{gate_name}.{parameter_name}"
            for gate_name in gate_names
            for parameter_name in ("weight", "expert_bias")
        )
        return {
            name for name, _ in self.model.named_parameters() if name.endswith(suffixes)
        }

    @staticmethod
    def _map_text_weight_name(name: str) -> Optional[str]:
        if name == "lm_head.weight":
            return name
        if not name.startswith("model.") or name.startswith(
            ("model.visual.", "model.linear_proj.")
        ):
            return None
        # The public checkpoint is already model.layers/model.norm/
        # model.word_embeddings; retain the private model.model alias.
        return name.replace("model.model.", "model.", 1)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_non_text: Set[str] = set()
        unexpected_non_text = []

        def dispatch_text_weights():
            for name, loaded_weight in weights:
                text_name = self._map_text_weight_name(name)
                if text_name is not None:
                    yield text_name, loaded_weight
                    continue
                loaded_name = self._load_non_text_weight(
                    name, loaded_weight, params_dict
                )
                if loaded_name is None:
                    unexpected_non_text.append(name)
                else:
                    loaded_non_text.add(loaded_name)

        loaded_text = self.model.load_weights(dispatch_text_weights())
        required_non_text = self._required_non_text_weights(params_dict)
        missing_non_text = required_non_text - loaded_non_text
        if missing_non_text:
            raise RuntimeError(
                f"Missing required Bailing VL weights: {sorted(missing_non_text)[:20]}"
            )

        required_router_weights = self._required_router_weights()
        missing_router_weights = required_router_weights - loaded_text
        if missing_router_weights:
            raise RuntimeError(
                "Missing required Bailing VL router weights: "
                f"{sorted(missing_router_weights)[:20]}"
            )
        if unexpected_non_text:
            logger.warning(
                "Skipped %d non-text Bailing checkpoint tensors; examples: %s",
                len(unexpected_non_text),
                unexpected_non_text[:10],
            )
        logger.info(
            "Loaded Bailing VL weights: %d text tensors and %d vision/projection tensors",
            len(loaded_text),
            len(loaded_non_text),
        )
        return set(loaded_text) | loaded_non_text


EntryClass = [BailingMoeV3VLForConditionalGeneration]
