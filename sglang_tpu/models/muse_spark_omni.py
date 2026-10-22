# SPDX-License-Identifier: Apache-2.0
"""Muse Spark image/video inference on top of the TPU text decoder.

Adapted from sgl-project/sglang PR #44666 (26305be7f410). The vision encoder is
replicated and images are distributed across attention TP ranks; 16 vision
heads cannot be tensor-sharded over the 32 ranks of a four-host engine.
"""

import itertools
import re

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.managers import mm_utils
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.multimodal.mm_utils import (
    get_dp_encoder_lb_assignment,
)
from sglang.srt.runtime_context import get_mm, get_parallel
from sglang_tpu.mm_embed_tpu import embed_mm_inputs_tpu
from sglang_tpu.mm_embedding import PaddedEmbedding
from sglang_tpu.models.muse_spark import MuseSparkForConditionalGeneration as TextModel
from sglang_tpu.models.muse_spark import _rms_normalize, _to_interleaved_rope
from sglang_tpu.models.muse_spark_vision import (
    NaViT,
    NaViTAdapter,
    prepare_vision_input,
)

_VISION_PREFIX = "model.vision_"
_QKV_RE = re.compile(r"\.(?:self_attn|attn)\.([qkv])_proj\.(?:weight|bias)$")


def _pad_dp_embeddings(local: torch.Tensor, rows: int) -> torch.Tensor:
    # Give every rank an initialized exchange buffer, including full owners.
    # Uninitialized padding produced invalid gathered rows and stalled D2H in
    # TPU32 image prefill; this allocation/copy contract passes the same-size
    # and unequal-size multihost reproduction.
    padded = torch.zeros(
        (rows, *local.shape[1:]), device=local.device, dtype=local.dtype
    )
    if local.shape[0]:
        padded[: local.shape[0]].copy_(local)
    return padded


def _run_dp_sharded_encode(
    features: list[torch.Tensor],
    encode_cost_per_item: list[int],
    tokens_per_item: list[int],
    encode_fn,
    out_hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Mirrors run_dp_sharded_mrope_vision_model in sglang.srt.multimodal.mm_utils.
    parallel = get_parallel()
    tp_size = parallel.attn_tp_size
    tp_rank = parallel.attn_tp_rank
    if tp_size == 1:
        return encode_fn(features)

    image_to_tp_rank, gpu_sample_counts, _ = get_dp_encoder_lb_assignment(
        encode_cost_per_item, tp_size
    )
    cum_counts = [0, *itertools.accumulate(gpu_sample_counts)]
    local_idxs = image_to_tp_rank[cum_counts[tp_rank] : cum_counts[tp_rank + 1]]

    local_embeds = (
        encode_fn([features[i] for i in local_idxs])
        if local_idxs
        else torch.empty((0, out_hidden_size), device=device, dtype=dtype)
    )

    # Single-item fast path: bit-identical to the all-gather below.
    if len(features) == 1:
        if tp_rank != 0:
            local_embeds = torch.empty(
                (tokens_per_item[0], out_hidden_size), device=device, dtype=dtype
            )
        get_parallel().attn_tp_group.broadcast(local_embeds, src=0)
        return local_embeds

    grouped_out_lengths = []
    offset = 0
    for count in gpu_sample_counts:
        grouped_out_lengths.append(
            sum(tokens_per_item[i] for i in image_to_tp_rank[offset : offset + count])
        )
        offset += count
    max_len_per_rank = max(grouped_out_lengths)

    padded = _pad_dp_embeddings(local_embeds, max_len_per_rank)
    gathered = parallel.attn_tp_group.all_gather(padded, dim=0)

    per_rank = []
    for rank in range(tp_size):
        start = rank * max_len_per_rank
        per_rank.append(gathered[start : start + grouped_out_lengths[rank]])

    ordered = [None] * len(features)
    offset = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        rank_images = image_to_tp_rank[offset : offset + count]
        rank_embed = per_rank[rank]
        embed_start = 0
        for item_idx in rank_images:
            n_tok = tokens_per_item[item_idx]
            ordered[item_idx] = rank_embed[embed_start : embed_start + n_tok]
            embed_start += n_tok
        offset += count
    return torch.cat(ordered, dim=0)


class _RMSNormWithoutScale(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), None, self.eps)


class _NormalizedEmbedding(nn.Module):
    """Token embeddings with the decoder's weightless input RMS norm applied."""

    def __init__(self, embedding: nn.Module, eps: float) -> None:
        super().__init__()
        object.__setattr__(self, "_embedding", embedding)
        self.eps = eps

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        value = self._embedding(input_ids)
        return _rms_normalize(value, self.eps).to(value.dtype)

    @property
    def weight(self):
        return self._embedding.weight

    @property
    def embedding_dim(self):
        return self._embedding.embedding_dim

    @property
    def num_embeddings(self):
        return self._embedding.num_embeddings


class _Decoder(nn.Module):
    """Compile the decoder and logits together, outside media preparation."""

    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    @property
    def logits_processor(self):
        return self._owner.logits_processor

    def get_input_embeddings(self):
        return self._owner.get_input_embeddings()

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return TextModel.forward(
            self._owner, input_ids, positions, forward_batch, input_embeds
        )


class _VisionWeightLoader:
    """Loads the vision tower, adapter and projection tensors of a checkpoint."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._params = {
            name: param
            for name, param in model.named_parameters()
            if name.startswith(_VISION_PREFIX)
        }
        self._loaded: set[str] = set()
        self._qkv_shards: set[tuple[str, str]] = set()

    def filter_text(self, weights):
        """Load vision tensors as they stream by; yield everything else."""
        for name, weight in weights:
            if name.startswith(_VISION_PREFIX):
                self._load(name, weight)
            else:
                yield name, weight

    def _load(self, name: str, weight: torch.Tensor) -> None:
        if "rotary_emb" in name:
            return
        match = _QKV_RE.search(name)
        if match is None:
            if name not in self._params:
                raise ValueError(f"Unexpected Muse Spark vision weight: {name}")
            param = self._params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            self._loaded.add(name)
            return
        shard = match.group(1)
        if shard in ("q", "k"):
            # Same half-split to interleaved RoPE reorder as the decoder's q/k.
            heads = self._model.model.vision_tower.config.num_attention_heads
            weight = _to_interleaved_rope(weight, heads)
        target = name.replace(f".{shard}_proj.", ".qkv_proj.")
        param = self._params[target]
        param.weight_loader(param, weight, shard)
        self._qkv_shards.add((target, shard))
        if all((target, s) in self._qkv_shards for s in "qkv"):
            self._loaded.add(target)

    def finish(self) -> None:
        missing = sorted(set(self._params) - self._loaded)
        if missing:
            raise ValueError(f"Missing Muse Spark vision weights: {missing}")


class MuseSparkForConditionalGeneration(TextModel):
    decode_embeds_in_language_model = True
    keeps_mm_features_on_host = True
    # The decoder's checkpoint check skips these; _VisionWeightLoader owns them.
    tower_param_prefixes = (_VISION_PREFIX,)

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        self._normalized_embedding = _NormalizedEmbedding(
            self.model.embed_tokens, self.text_config.rms_norm_eps
        )
        self.vision_enabled = bool(get_mm().enable_multimodal)
        if not self.vision_enabled:
            return
        from sglang.srt.compilation.support_torch_compile import support_torch_compile
        from sglang_tpu.compilation.decoder_compile_policy import (
            Qwen3VLDecoderTpuCompilePolicy,
        )
        from sglang_tpu.models.muse_spark_vision_compile import MuseVisionCompilePolicy

        self.model.vision_tower = NaViT(
            config.vision_config, prefix="model.vision_tower"
        )
        if config.out_hidden_size != self.model.vision_tower.output_size:
            raise ValueError("Muse vision output size does not match checkpoint config")
        if config.projector_hidden_act != "gelu":
            raise ValueError("Muse vision projector requires gelu")
        self.model.vision_adapter = NaViTAdapter(
            config.out_hidden_size,
            config.projector_hidden_size,
            "model.vision_adapter",
        )
        self.model.vision_projection = ReplicatedLinear(
            config.projector_hidden_size,
            self.text_config.hidden_size,
            bias=False,
            prefix="model.vision_projection",
        )
        self.perception_emb_norm = _RMSNormWithoutScale()
        self.language_model = _Decoder(self)
        support_torch_compile(
            self.language_model, policy=Qwen3VLDecoderTpuCompilePolicy
        )
        support_torch_compile(self.model.vision_tower, policy=MuseVisionCompilePolicy)

    def get_input_embeddings(self):
        return self._normalized_embedding

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if not self.vision_enabled:
            return super().forward(input_ids, positions, forward_batch, input_embeds)
        if (
            input_ids is None
            or forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            # Match the compile policy's positional token-ID warmup. Passing
            # a fourth positional None creates a different Dynamo call shape.
            if input_embeds is None:
                return self.language_model(input_ids, positions, forward_batch)
            return self.language_model(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        return mm_utils.general_mm_embed_routine(
            input_ids,
            forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_video_feature,
            },
            positions=positions,
            mm_embedder=embed_mm_inputs_tpu,
        )

    def _encode_vision(self, features):
        param = next(self.model.vision_tower.parameters())
        outputs = []
        for feature in features:
            clips = (
                [feature]
                if feature.ndim == 3
                else list(feature.split(self.model.vision_tower.patch_temporal))
            )
            for clip in clips:
                inputs, count = prepare_vision_input(
                    clip, self.model.vision_tower.config
                )
                device_inputs = tuple(
                    t.to(device=param.device, dtype=param.dtype if i == 0 else t.dtype)
                    for i, t in enumerate(inputs)
                )
                value = self.model.vision_tower(*device_inputs)[:count]
                value = self.model.vision_adapter(value)
                value, _ = self.model.vision_projection(value)
                outputs.append(self.perception_emb_norm(value))
        if not outputs:
            return param.new_empty((0, self.text_config.hidden_size))
        return torch.cat(outputs)

    def _features(self, items):
        features = [item.feature for item in items]
        config = self.model.vision_tower.config
        costs = []
        for value in features:
            if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
                raise ValueError(
                    "Muse vision requires preprocessed image/video tensors"
                )
            clips = 1 if value.ndim == 3 else value.shape[0] // config.patch_temporal
            costs.append(
                clips
                * (value.shape[-2] // config.patch_size)
                * (value.shape[-1] // config.patch_size)
            )
        counts = [n // config.merge_size**2 for n in costs]
        param = next(self.model.vision_tower.parameters())
        embedding = _run_dp_sharded_encode(
            features,
            costs,
            counts,
            self._encode_vision,
            self.text_config.hidden_size,
            param.device,
            param.dtype,
        )
        # Preserve device ownership through caching/chunking. The TPU embedder
        # materializes these rows on the host only when host placement is used.
        return PaddedEmbedding.from_buffer(embedding, sum(counts))

    def get_image_feature(self, items):
        return self._features(items)

    def get_video_feature(self, items):
        return self._features(items)

    def load_weights(self, weights):
        if not self.vision_enabled:
            return super().load_weights(weights)
        vision = _VisionWeightLoader(self)
        super().load_weights(vision.filter_text(weights))
        vision.finish()

    def pad_input_ids(self, input_ids, mm_inputs):
        return (
            mm_utils.MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
                input_ids, mm_inputs
            )
        )


EntryClass = MuseSparkForConditionalGeneration
