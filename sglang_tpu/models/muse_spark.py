# SPDX-License-Identifier: Apache-2.0
"""Muse Spark text decoder on TPU.

Each layer is ``pre-norm -> gated attention -> post-norm -> gated residual``, then
the same around a compressed MoE: ``pre_expert_proj`` maps the hidden state to
``moe_hidden_size``, experts run in that space, each expert output is
RMS-normalized before the router-weighted sum, and ``post_expert_proj`` maps the
result back. The residual stream is kept in fp32.

NVFP4 experts run through the Pallas fused-RS kernel, which applies the
post-expert norm, the weighted combine and the cross-rank reduction. The BF16
experts of the first and last layer are quantized to FP8 at load and take the
FP8 fused-RS kernel. This module is the text decoder; ``muse_spark_omni`` adds
the vision tower on top of it and is the registered architecture. Audio tower
weights are skipped.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.muse_spark import MuseSparkConfig, MuseSparkTextConfig
from sglang.srt.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptQuantConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel, process_model_config
from sglang.srt.utils import add_prefix
from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

# Everything else in the checkpoint belongs to the vision and audio towers.
_TEXT_PREFIX = "model.language_model."
# Static activation scales; the NVFP4 experts run weight-only (W4A16) on TPU.
_UNUSED_SUFFIXES = (".gate_up_proj_input_scale", ".down_proj_input_scale")
# Checkpoint suffix -> parameter suffix, first match wins.
_RENAMES = (
    (".experts.gate_up_proj_weight_scale_2", ".experts.w13_weight_scale_2"),
    (".experts.gate_up_proj_weight_scale", ".experts.w13_weight_scale"),
    (".experts.gate_up_proj", ".experts.w13_weight"),
    (".experts.down_proj_weight_scale_2", ".experts.w2_weight_scale_2"),
    (".experts.down_proj_weight_scale", ".experts.w2_weight_scale"),
    (".experts.down_proj", ".experts.w2_weight"),
    (".experts.post_expert_norm.weight", ".post_expert_norm.weight"),
    (".gate.e_score_correction_bias", ".e_score_correction_bias"),
)
_QKV_SHARDS = ((".q_proj.", "q"), (".k_proj.", "k"), (".v_proj.", "v"))
# The fused-RS kernels apply ``rms_norm((w + 1) * x, 1e-8)`` per expert slot.
_FUSED_RS_POST_EXPERT_NORM_EPS = 1e-8
# Sigmoid router scores are renormalized over the top-k with this epsilon.
_ROUTE_EPS = 1e-15


def _param_name(checkpoint_name: str) -> Optional[str]:
    """Model parameter for a checkpoint tensor, or None when the text decoder does not use it."""
    if checkpoint_name == "lm_head.weight":
        return checkpoint_name
    if not checkpoint_name.startswith(_TEXT_PREFIX):
        return None
    name = "model." + checkpoint_name.removeprefix(_TEXT_PREFIX)
    if name.endswith(_UNUSED_SUFFIXES):
        return None
    for old, new in _RENAMES:
        if name.endswith(old):
            return name.removesuffix(old) + new
    return name


def _layer_index(name: str) -> int:
    """Decoder layer index of a ``model.layers.<i>.`` parameter name, else -1."""
    if not name.startswith("model.layers."):
        return -1
    return int(name.split(".", 3)[2])


def _to_interleaved_rope(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """q/k projection rows from the checkpoint's half-split RoPE layout to interleaved pairs."""
    rows = weight.shape[0]
    return (
        weight.reshape(num_heads, 2, rows // num_heads // 2, *weight.shape[1:])
        .transpose(1, 2)
        .reshape(weight.shape)
        .contiguous()
    )


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMS normalization over the last dim, computed in fp32."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class MuseSparkRMSNorm(nn.Module):
    """RMSNorm with the checkpoint's gain conventions.

    ``gain_center`` is added to the stored weight (zero-centered gammas).
    ``scale_input`` applies the gain before normalizing, as the post-FFN and
    post-expert norms do; otherwise the gain scales the normalized output.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        has_weight: bool = True,
        gain_center: float = 0.0,
        scale_input: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.gain_center = gain_center
        self.scale_input = scale_input
        self.weight = (
            nn.Parameter(torch.empty(hidden_size), requires_grad=False)
            if has_weight
            else None
        )

    def gain(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight.float() + self.gain_center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = self.gain()
        if gain is not None and self.scale_input:
            return _rms_normalize(x.float() * gain, self.eps).to(x.dtype)
        out = _rms_normalize(x, self.eps)
        if gain is not None:
            out = out * gain
        return out.to(x.dtype)


class MuseSparkGatedResidual(nn.Module):
    """Norm-preserving gated residual: ``alpha * stream + beta * branch``.

    With ``s = gate / temperature``: ``beta = sigmoid(s)`` and
    ``alpha = sqrt(max(sigmoid(-s) * (1 + beta), eps))``, both per channel.
    """

    def __init__(self, dim: int, temperature: float, eps: float = 1e-3):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.gate = nn.Parameter(torch.empty(dim), requires_grad=False)
        self.register_buffer("alpha", torch.ones(dim), persistent=False)
        self.register_buffer("beta", torch.ones(dim), persistent=False)

    @torch.no_grad()
    def precompute_coefficients(self) -> None:
        scaled = self.gate.float() / self.temperature
        beta = torch.sigmoid(scaled)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1.0 + beta), self.eps))
        self.alpha = alpha.to(self.gate.device)
        self.beta = beta.to(self.gate.device)

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return self.alpha * stream + self.beta * branch.to(stream.dtype)


class MuseSparkInterleavedRope(nn.Module):
    """GPT-J style (interleaved pairs) rotary embedding evaluated in fp32."""

    def __init__(self, head_dim: int, max_position: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        )
        angles = torch.arange(max_position, dtype=torch.float64)[:, None] * inv_freq
        # [max_position, head_dim // 2] each, fp32.
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)
        self.head_dim = head_dim

    def _rotate(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        pairs = x.float().view(num_tokens, -1, self.head_dim // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        return rotated.view(x.shape).to(x.dtype)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [num_tokens, 1, head_dim // 2] broadcasts over heads.
        cos = self.cos.index_select(0, positions).unsqueeze(1)
        sin = self.sin.index_select(0, positions).unsqueeze(1)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class MuseSparkAttention(nn.Module):
    """QK-normed attention with a weightless output norm and a sigmoid output gate."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        rotary_emb: MuseSparkInterleavedRope,
        prefix: str,
    ):
        super().__init__()
        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        tp_rank = parallel.attn_tp_rank
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.norm_eps = config.rms_norm_eps
        self.query_scale = config.qk_scale_factor
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.gate_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=None,
            prefix=add_prefix("gate_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=None,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # Shared across layers; NoPE layers (theta 0) skip it.
        self.rotary_emb = rotary_emb if config.layer_rope_theta[layer_idx] > 0 else None
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            sliding_window_size=config.sliding_window - 1 if sliding else -1,
            prefix=add_prefix("attn", prefix),
        )

    def _per_head_normalize(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        heads = x.view(num_tokens, -1, self.head_dim)
        return _rms_normalize(heads, self.norm_eps).view(x.shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = (self._per_head_normalize(q) * self.query_scale).to(qkv.dtype)
        k = self._per_head_normalize(k).to(qkv.dtype)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = self._per_head_normalize(attn_output).to(qkv.dtype)
        gate, _ = self.gate_proj(hidden_states)
        output, _ = self.o_proj(torch.sigmoid(gate) * attn_output)
        return output


def _uses_quantized_experts(
    quant_config: Optional[QuantizationConfig], experts_prefix: str
) -> bool:
    """Whether a layer's experts are serialized NVFP4.

    The checkpoint excludes whole layers with dotted substrings such as
    ``"layers.0."``, which ModelOpt's component-wise matcher does not accept.
    """
    if quant_config is None:
        return False
    if not isinstance(quant_config, ModelOptQuantConfig):
        raise ValueError(
            f"Muse Spark supports BF16 or ModelOpt NVFP4, got {quant_config.get_name()}"
        )
    return not any(
        pattern in experts_prefix + "." for pattern in quant_config.exclude_modules
    )


class MuseSparkMoE(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        parallel = get_parallel()
        # Either expert parallelism (whole experts per rank) or expert TP (every
        # rank holds an intermediate slice of every expert).
        if parallel.moe_tp_size != 1 and parallel.moe_ep_size != 1:
            raise ValueError("Muse Spark MoE supports EP or expert TP, not both")
        if config.post_norm_eps != _FUSED_RS_POST_EXPERT_NORM_EPS:
            raise ValueError("The fused-RS post-expert norm assumes eps=1e-8")
        self.tp_size = parallel.tp_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.route_eps = _ROUTE_EPS
        self.router_multiplier = config.output_multiplier
        self.expert_hidden_size = config.moe_hidden_size

        # Replicated: the fp32 weight is 8 MiB per layer, and at small batch the
        # all-gather a sharded router would need costs more than the weight read.
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False
        )
        from sglang_tpu.moe.expert_tp import is_fused_moe_rs_expert_tp

        # With expert TP, the small decode-sized projection outputs are gathered
        # by the one-shot Pallas all-gather, as the fused-RS output is.
        self.oneshot_gather = is_fused_moe_rs_expert_tp(
            parallel.moe_ep_size, parallel.moe_tp_size, parallel.tp_size
        )
        if self.oneshot_gather:
            from sglang_tpu.oneshot_collectives import get_oneshot_all_gather_op

            get_oneshot_all_gather_op(self.tp_size)
            self.register_buffer(
                "_tp_rank_t",
                torch.tensor([parallel.tp_rank], dtype=torch.int32),
                persistent=False,
            )
        # Sharded by output channel and gathered: every token needs the full
        # projected vector (experts live on all ranks), and a replicated copy
        # would re-read both 8192x4096 matrices on every rank each step.
        self.pre_expert_proj = ColumnParallelLinear(
            config.hidden_size,
            self.expert_hidden_size,
            bias=False,
            gather_output=not self.oneshot_gather,
            quant_config=None,
            prefix=add_prefix("pre_expert_proj", prefix),
        )
        self.post_expert_proj = ColumnParallelLinear(
            self.expert_hidden_size,
            config.hidden_size,
            bias=False,
            gather_output=not self.oneshot_gather,
            quant_config=None,
            prefix=add_prefix("post_expert_proj", prefix),
        )
        self.pre_expert_norm = MuseSparkRMSNorm(
            self.expert_hidden_size, eps=config.rms_norm_eps, gain_center=1.0
        )
        self.post_expert_norm = MuseSparkRMSNorm(
            self.expert_hidden_size,
            eps=config.post_norm_eps,
            gain_center=1.0,
            scale_input=True,
        )
        experts_prefix = add_prefix("experts", prefix)
        self.quantized_experts = _uses_quantized_experts(quant_config, experts_prefix)
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            hidden_size=self.expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_idx,
            top_k=self.top_k,
            reduce_results=False,
            # BF16 experts are quantized to FP8 at load: the BF16 grouped GEMM in
            # the same executable as fused-RS and the distributed TopK sampler
            # core-halts the TPU at startup.
            quant_config=(
                quant_config
                if self.quantized_experts
                else Fp8Config(is_checkpoint_fp8_serialized=False)
            ),
            prefix=experts_prefix,
            activation="silu",
            # Per-slot outputs: the post-expert norm runs before the combine.
            no_combine=True,
        )
        self.experts.supports_tpu_fused_rs_post_expert_norm = True
        self.experts.tpu_fused_rs_post_expert_norm_weight = lambda: (
            self.post_expert_norm.weight.data
        )

    def _route(self, router_input: torch.Tensor) -> StandardTopKOutput:
        router_logits, _ = self.gate(router_input)
        scores = torch.sigmoid(router_logits.float() * self.router_multiplier)
        _, topk_ids = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1
        )
        # One-hot select, not gather: TPU runs the row gather as a slow serial loop.
        expert_ids = torch.arange(scores.shape[-1], device=scores.device)
        topk_scores = torch.where(
            topk_ids.unsqueeze(-1) == expert_ids, scores.unsqueeze(1), 0.0
        ).sum(dim=-1)
        topk_weights = (
            topk_scores
            / (topk_scores.sum(dim=-1, keepdim=True) + self.route_eps)
            * self.route_scale
        )
        return StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            router_logits=router_logits,
        )

    def _combine_slots(
        self, slot_outputs: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        # slot_outputs: [num_tokens, top_k, expert_hidden]; non-local slots are 0,
        # and a zero row stays zero through the norm.
        normed = self.post_expert_norm(slot_outputs)
        combined = (normed * topk_weights.unsqueeze(-1)).to(normed.dtype).sum(dim=1)
        if self.tp_size > 1:
            combined = tensor_model_parallel_all_reduce(combined)
        return combined

    def _gather_columns(self, shard: torch.Tensor) -> torch.Tensor:
        """[T, N/tp] column shard -> [T, N]."""
        if shard.numel() * shard.element_size() > 256 * 1024:
            # Prefill-sized shards: XLA's bandwidth-oriented all-gather.
            return tensor_model_parallel_all_gather(shard.contiguous())
        from sglang_tpu.oneshot_collectives import oneshot_all_gather

        tokens, cols = shard.shape
        full = oneshot_all_gather(
            shard.contiguous(), rank_input=self._tp_rank_t, world_size=self.tp_size
        )
        return (
            full.view(self.tp_size, tokens, cols)
            .transpose(0, 1)
            .reshape(tokens, self.tp_size * cols)
        )

    def forward(
        self, hidden_states: torch.Tensor, router_input: torch.Tensor
    ) -> torch.Tensor:
        topk_output = self._route(router_input)
        expert_input, _ = self.pre_expert_proj(hidden_states)
        if self.oneshot_gather:
            expert_input = self._gather_columns(expert_input)
        output = self.experts(self.pre_expert_norm(expert_input), topk_output)
        if not self.experts.handles_moe_reduction:
            output = self._combine_slots(output, topk_output.topk_weights)
        output, _ = self.post_expert_proj(output)
        if self.oneshot_gather:
            output = self._gather_columns(output)
        return output


class MuseSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        rotary_emb: MuseSparkInterleavedRope,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        hidden = config.hidden_size
        self.input_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.rms_norm_eps, gain_center=1.0
        )
        self.self_attn = MuseSparkAttention(
            config, layer_idx, rotary_emb, add_prefix("self_attn", prefix)
        )
        self.post_attention_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.post_norm_eps, has_weight=False
        )
        self.post_attention_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=config.residual_gate_temperature
        )
        self.pre_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.rms_norm_eps, gain_center=1.0
        )
        self.mlp = MuseSparkMoE(
            config, layer_idx, quant_config, add_prefix("mlp", prefix)
        )
        self.post_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.post_norm_eps, gain_center=1.0, scale_input=True
        )
        self.post_feedforward_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=config.residual_gate_temperature
        )

    def forward(
        self, positions: torch.Tensor, stream: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        attn_input = self.input_layernorm(stream).to(torch.bfloat16)
        attn_output = self.self_attn(positions, attn_input, forward_batch)
        stream = self.post_attention_residual_gate(
            stream, self.post_attention_layernorm(attn_output)
        )
        router_input = self.pre_feedforward_layernorm(stream)
        ffn_output = self.mlp(router_input.to(torch.bfloat16), router_input)
        return self.post_feedforward_residual_gate(
            stream, self.post_feedforward_layernorm(ffn_output)
        )


class MuseSparkTextModel(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        max_position: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.config = config
        # Layers whose output stream feeds a DFlash draft.
        self.layers_to_capture: List[int] = []
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
        )
        rope_thetas = {theta for theta in config.layer_rope_theta if theta > 0}
        if len(rope_thetas) != 1:
            raise NotImplementedError("Muse Spark RoPE layers must share one theta")
        self.rotary_emb = MuseSparkInterleavedRope(
            config.head_dim, max_position=max_position, base=rope_thetas.pop()
        )
        self.layers = nn.ModuleList(
            MuseSparkDecoderLayer(
                config,
                layer_idx,
                self.rotary_emb,
                quant_config,
                add_prefix(f"layers.{layer_idx}", prefix),
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        # The output norm's gain is centered on zero, unlike the per-layer norms.
        self.norm = MuseSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            # The decoder reads weightless-RMS-normalized token embeddings.
            embeds = self.embed_tokens(input_ids)
            input_embeds = _rms_normalize(embeds, self.config.rms_norm_eps).to(
                embeds.dtype
            )
        stream = input_embeds.float()
        aux_hidden_states = []
        for layer_idx, layer in enumerate(self.layers):
            stream = layer(positions, stream, forward_batch)
            if layer_idx in self.layers_to_capture:
                # Drafts read the operator dtype, not the fp32 residual.
                aux_hidden_states.append(stream.to(torch.bfloat16))
        hidden = self.norm(stream.to(torch.bfloat16))
        if not aux_hidden_states:
            return hidden
        # One tensor, not a list: packing a list happens outside the compiled
        # graph and costs a separate eager program on every decode step.
        return hidden, torch.cat(aux_hidden_states, dim=-1)


class MuseSparkForConditionalGeneration(nn.Module):
    """Muse Spark served text-only."""

    # Parameters a subclass loads itself (perception towers); the checkpoint
    # completeness check here skips them.
    tower_param_prefixes: Tuple[str, ...] = ()

    def __init__(
        self,
        config: MuseSparkConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        text_config = config.text_config
        if set(text_config.mlp_layer_types) != {"sparse"}:
            raise NotImplementedError("Muse Spark on TPU supports MoE layers only")
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config
        # The fp32 RoPE table must cover the served context: positions past its
        # end rotate with the last row. 2^17 rows = 32 MiB floor.
        max_position = min(
            text_config.max_position_embeddings,
            max(1 << 17, process_model_config().context_len),
        )
        self.model = MuseSparkTextModel(
            text_config, max_position, quant_config, add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
        )
        if text_config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            text_config, logit_scale=text_config.output_multiplier
        )
        # The checkpoint's output GEMM accumulates in fp32.
        self.logits_processor.use_fp32_lm_head = True
        self.capture_aux_hidden_states = False
        self._megakernel_enabled = tpu_envs.SGLANG_TPU_MUSE_MEGAKERNEL.get()
        self._megakernel = None

    def precompile_kernels_after_loading(self) -> None:
        """Build the megakernel's weights once quant post-processing has laid out the
        experts it reads, and before the KV pool is sized, so the budget counts them."""
        if not self._megakernel_enabled:
            return
        from sglang_tpu.fp8_lm_head import TpuFp8LmHeadMethod
        from sglang_tpu.muse_megakernel.ops import MuseMegakernel

        self._megakernel = MuseMegakernel.from_model(self)
        # The megakernel path also takes the fp8 weight-only LM head.
        self.lm_head.quant_method = TpuFp8LmHeadMethod(self.lm_head)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return self.text_config.sliding_window - 1

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        megakernel = self._megakernel
        if (
            megakernel is not None
            and input_embeds is None
            and megakernel.handles(forward_batch, input_ids.shape[0])
        ):
            hidden_states, aux_hidden_states = megakernel.forward(
                embeds=self.model.embed_tokens(input_ids),
                positions=positions,
                forward_batch=forward_batch,
                capture=(
                    tuple(sorted(self.model.layers_to_capture))
                    if self.capture_aux_hidden_states
                    else ()
                ),
            )
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
        if not layer_ids:
            raise ValueError("DFLASH requires explicit layer_ids for aux capture.")
        num_layers = len(self.model.layers)
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(f"Capture layer ids {bad} outside [0, {num_layers}).")
        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = list(layer_ids)

    def _load_expert_stack(
        self, param: torch.Tensor, loaded_weight: torch.Tensor, name: str
    ) -> None:
        """Copy this rank's slice of a ``[num_experts, ...]`` expert tensor.

        EP slices whole experts. Expert TP slices the intermediate dim: rows of
        each gate/up half of ``w13*`` (``[E, 2I, ...]``) and the last dim of
        ``w2*`` (``[E, H, I/pack]``).
        """
        parallel = get_parallel()
        num_experts = loaded_weight.shape[0]
        if num_experts == param.shape[0]:
            local = loaded_weight
        else:
            local_count = num_experts // parallel.moe_ep_size
            local = loaded_weight.narrow(
                0, parallel.moe_ep_rank * local_count, local_count
            )
        tp_size, tp_rank = parallel.moe_tp_size, parallel.moe_tp_rank
        if tp_size > 1 and local.dim() == 3:
            if ".w13_" in name and local.shape[1] == param.shape[1] * tp_size:
                half, shard = local.shape[1] // 2, param.shape[1] // 2
                local = torch.cat(
                    (
                        local.narrow(1, tp_rank * shard, shard),
                        local.narrow(1, half + tp_rank * shard, shard),
                    ),
                    dim=1,
                )
            elif ".w2_" in name and local.shape[2] == param.shape[2] * tp_size:
                shard = param.shape[2]
                local = local.narrow(2, tp_rank * shard, shard)
        if tuple(local.shape) != tuple(param.shape):
            raise RuntimeError(
                f"Expert tensor {name}: parameter {tuple(param.shape)} vs "
                f"checkpoint slice {tuple(local.shape)}"
            )
        param.data.copy_(local.to(param.dtype))

    def _is_online_quant_scale(self, name: str) -> bool:
        """Scales that online FP8 quantization creates for the BF16 experts."""
        index = _layer_index(name)
        if index < 0 or ".experts." not in name or not name.endswith("_scale"):
            return False
        return not self.model.layers[index].mlp.quantized_experts

    def _load_qkv_shard(
        self, params: dict, name: str, loaded_weight: torch.Tensor
    ) -> Optional[str]:
        """Load a q/k/v projection into ``qkv_proj``; returns its parameter name."""
        for weight_part, shard_id in _QKV_SHARDS:
            if weight_part not in name:
                continue
            if shard_id != "v":
                text = self.text_config
                heads = (
                    text.num_attention_heads
                    if shard_id == "q"
                    else text.num_key_value_heads
                )
                loaded_weight = _to_interleaved_rope(loaded_weight, heads)
            param_name = name.replace(weight_part, ".qkv_proj.")
            param = params[param_name]
            param.weight_loader(param, loaded_weight, shard_id)
            return param_name
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        loaded = set()
        skipped = 0
        for checkpoint_name, loaded_weight in weights:
            name = _param_name(checkpoint_name)
            if name is None or _layer_index(name) >= num_layers:
                # Tower tensors, unused scales, or layers cut by a num_hidden_layers override.
                skipped += 1
                continue
            if self.text_config.tie_word_embeddings and name == "lm_head.weight":
                continue
            qkv_name = self._load_qkv_shard(params, name, loaded_weight)
            if qkv_name is not None:
                loaded.add(qkv_name)
                continue
            param = params.get(name)
            if param is None:
                raise ValueError(
                    f"Checkpoint tensor {checkpoint_name} has no model parameter"
                )
            if ".experts.w" in name:
                self._load_expert_stack(param, loaded_weight, name)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
            loaded.add(name)
        missing = sorted(
            name
            for name in params
            if name not in loaded
            and not name.startswith(self.tower_param_prefixes)
            and not name.endswith(("w13_input_scale", "w2_input_scale"))
            and not self._is_online_quant_scale(name)
        )
        if missing:
            raise RuntimeError(
                f"Muse Spark checkpoint is missing {len(missing)} tensors, "
                f"e.g. {missing[:5]}"
            )
        if skipped:
            logger.info("Skipped %d non-text or unused checkpoint tensors", skipped)
        for layer in self.model.layers:
            layer.post_attention_residual_gate.precompute_coefficients()
            layer.post_feedforward_residual_gate.precompute_coefficients()
