# SPDX-License-Identifier: Apache-2.0
"""The part of a dense DFlash draft that writes its context KV, for a PD prefill.

A PD prefill never drafts: it projects the target features of each prompt row
into the draft's context K/V (dflash_draft_kv_write) and Raiden ships that KV
to the decode. So it holds fc, hidden_norm and per layer the K/V projection,
k_norm and rope; the MLPs, Q/O projections, convolutions, selector and norm
stay on disk. On Muse Spark DFlash2 that is 0.78 of 9.1 GiB.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import (
    is_nemotron_35_draft_config,
    parse_dflash_draft_config,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from sglang_tpu.models.dflash_mla import is_dflash_mla_draft

# Dense GQA drafts whose context KV is k_norm(k_proj(hidden_norm(fc(x)))) roped, v_proj(...).
_CONTEXT_KV_ARCHITECTURES = frozenset(
    {"DFlashDraftModel", "DFlash2DraftModel", "MuseGlimmerAssistantModel"}
)
# Checkpoint names of the native export's encoder, as DFlashDraftModel aliases them.
_ENCODER_ALIASES = {
    "encoder.fc.weight": "fc.weight",
    "encoder.output_norm_enc.weight": "hidden_norm.weight",
}
_KV_PROJ_NAME = re.compile(r"layers\.(\d+)\.self_attn\.([kv])_proj\.(weight|bias)")


def context_kv_unsupported_reason(model_config: ModelConfig) -> str | None:
    """Why this draft cannot load as DFlashContextKVModel, or None when it can."""
    hf_config = model_config.hf_config
    architectures = list(hf_config.architectures or [])
    if len(architectures) != 1 or architectures[0] not in _CONTEXT_KV_ARCHITECTURES:
        return f"architectures {architectures} are not a dense GQA DFlash draft"
    if is_dflash_mla_draft(hf_config):
        return "an MLA draft ropes its K inside the latent projection"
    if is_nemotron_35_draft_config(hf_config):
        return "a Nemotron 3.5 draft's fc is quantized and its embedding is its own"
    return None


def as_context_kv_model_config(model_config: ModelConfig) -> ModelConfig:
    """A copy of the draft's ModelConfig that loads DFlashContextKVModel."""
    reason = context_kv_unsupported_reason(model_config)
    if reason is not None:
        raise NotImplementedError(f"DFlashContextKVModel: {reason}.")
    out = copy.copy(model_config)
    # get_config caches the HF config per path: never rename the shared object.
    hf_config = copy.copy(model_config.hf_config)
    hf_config.architectures = [DFlashContextKVModel.__name__]
    if model_config.hf_text_config is model_config.hf_config:
        out.hf_text_config = hf_config
    out.hf_config = hf_config
    return out


class DFlashContextKVAttention(nn.Module):
    """The K/V half of DFlashAttention, with the same projection, norm, rope and V scale."""

    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        del layer_id
        hidden_size = int(config.hidden_size)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))
        tp_size = int(get_parallel().tp_size)
        if total_num_kv_heads % tp_size != 0:
            raise NotImplementedError(
                f"DFlashContextKVModel shards {total_num_kv_heads} draft KV heads "
                f"over TP {tp_size}; it does not replicate KV heads."
            )
        self.num_kv_heads = total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.kv_size = self.num_kv_heads * head_dim
        # Output [k | v] per rank, as DFlashAttention's fused QKV weight slices them.
        self.kv_proj = MergedColumnParallelLinear(
            hidden_size,
            [total_num_kv_heads * head_dim] * 2,
            bias=bool(getattr(config, "attention_bias", False)),
        )
        self.k_norm = RMSNorm(
            head_dim, eps=float(getattr(config, "rms_norm_eps", 1e-6))
        )
        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=bool(
                getattr(
                    config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
                )
            ),
            partial_rotary_factor=float(getattr(config, "partial_rotary_factor", 1.0)),
        )
        self.v_scale = parse_dflash_draft_config(
            draft_hf_config=config
        ).attention_value_scale

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv, _ = self.kv_proj(hidden_states)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        if self.v_scale is not None:
            v = v * self.v_scale
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        return self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        _, k = self.rotary_emb(positions, k.new_empty(k.shape), k)
        return k


class DFlashContextKVLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        self.self_attn = DFlashContextKVAttention(config, layer_id)


class DFlashContextKVModel(DFlashDraftModel):
    """A dense DFlash draft holding only what dflash_write_draft_kv reads.

    Not built by DFlashDraftModel.__init__, which would allocate every layer;
    forward raises, since a server with this model never drafts.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        nn.Module.__init__(self)
        del quant_config, prefix
        self.config = config
        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        self.rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        draft_config = self.draft_config = parse_dflash_draft_config(
            draft_hf_config=config
        )
        if draft_config.is_domino:
            raise NotImplementedError(
                "DFlashContextKVModel does not serve Domino drafts."
            )
        self.block_size = draft_config.resolve_block_size(default=16)
        self.projector_type = draft_config.projector_type
        self.shift_label = draft_config.shift_label
        self.candidate_selector = None
        self.lilicorr = None
        self.is_nemotron_35_draft = False
        self.quantizable = False
        self.embed_tokens = None
        self.lm_head = None
        self.prefix_gru = None
        self.embed_proj = None
        self.layers = nn.ModuleList(
            [DFlashContextKVLayer(config, i) for i in range(num_layers)]
        )
        if draft_config.num_target_layers is not None:
            target_num_layers = int(draft_config.num_target_layers)
        elif draft_config.target_layer_ids is not None:
            target_num_layers = max(draft_config.target_layer_ids) + 1
        else:
            target_num_layers = num_layers
        self.num_context_features = len(
            draft_config.resolve_target_layer_ids(
                target_num_layers=target_num_layers, draft_num_layers=num_layers
            )
        )
        # The same fc class as DFlashDraftModel, so its weight loads and lowers alike.
        fc_in = self.num_context_features * hidden_size
        if envs.SGLANG_DFLASH_ROW_PARALLEL_FC.get():
            self.fc = RowParallelLinear(
                fc_in, hidden_size, bias=False, input_is_parallel=False
            )
        elif envs.SGLANG_DFLASH_COLUMN_PARALLEL_FC.get():
            self.fc = ColumnParallelLinear(
                fc_in, hidden_size, bias=False, gather_output=True
            )
        else:
            self.fc = nn.Linear(fc_in, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, eps=self.rms_norm_eps)

    def set_block_size(self, block_size: int) -> None:
        # No convolutions: they only run in the block forward.
        self.block_size = int(block_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        raise NotImplementedError(
            "DFlashContextKVModel only writes context KV; a PD prefill never drafts."
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            key = name.removeprefix("model.")
            key = _ENCODER_ALIASES.get(key, key)
            match = _KV_PROJ_NAME.fullmatch(key)
            if match is not None:
                layer, kv, kind = match.groups()
                param_name = f"layers.{layer}.self_attn.kv_proj.{kind}"
                param = params[param_name]
                param.weight_loader(param, weight, 0 if kv == "k" else 1)
                # Both shards of the merged [k | v] parameter must arrive.
                loaded.add(f"{param_name}[{kv}]")
                continue
            # Weights outside the context KV path stay on disk.
            if key not in params:
                continue
            param = params[key]
            if key == "fc.weight":
                expected = (int(self.config.hidden_size), self._fc_in_features())
                if tuple(weight.shape) != expected:
                    raise ValueError(
                        f"DFLASH fc.weight shape {tuple(weight.shape)} != {expected} "
                        f"(num_context_features={self.num_context_features})."
                    )
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(key)
        required = {
            f"{name}[{kv}]" if ".kv_proj." in name else name
            for name in params
            for kv in "kv"
        }
        missing = sorted(required - loaded)
        if missing:
            raise ValueError(f"DFlashContextKVModel: checkpoint lacks {missing}.")


EntryClass = [DFlashContextKVModel]
