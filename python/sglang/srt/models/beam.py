# Copyright 2026 SGLang Team
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
# ==============================================================================
"""Reflection Beam (Beam-1.0-501B-A23B): hybrid SWA/NoPE MoE decoder.

Platform-neutral port of the reference GPU implementation. The five
Beam-specific blocks are ``BaseFusedOp`` subclasses whose ``forward_native``
is the reference math, rounding points included:

- ``BeamSandwichResidualRMSNormOp``: post-norm -> depth-scaled residual -> next
  pre-norm (two per layer).
- ``BeamQKNormPositionOp``: per-head q/k RMSNorm, then NeoX RoPE (sliding
  layers) or the SSMax query scale (full/NoPE layers).
- ``BeamAttentionGateOp``: ``attn * sigmoid(gate + gate_bias[head])``.
- ``BeamGroupedTopKOp``: grouped sigmoid top-k router (TopK's routing function).
- ``BeamRMSNormOp``: the one standalone norm per PP stage (first input norm).

An accelerated implementation is installed without touching this file:
``BaseFusedOp.register_oot_forward(OpCls, fn, key)`` where ``key`` is
``current_platform.get_dispatch_key_name()``. ``_BeamFusedOp`` consults that
registry on in-tree platforms too, so one registration from a platform
plugin's ``init_backend()`` covers both. ``fn(op, *args)`` must keep the
``forward_native`` signature and return contract.
"""

import functools
import logging
import math
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from sglang.kernels.fused_op import BaseFusedOp
from sglang.srt.configs.beam import BeamConfig
from sglang.srt.distributed import (
    moe_expert_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, biased_grouped_topk_impl
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.compressed_tensors.online_fp8 import (
    online_fp8_weights,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_exec, get_lora, get_parallel
from sglang.srt.utils import add_prefix, make_pp_layers, set_weight_attrs

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Reference math (pure torch; the ground truth for every accelerated path)
# --------------------------------------------------------------------------


def depth_scaled_residual_scales(
    layer_id: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """(branch, residual) scales of layer ``layer_id``'s attention and MLP updates."""
    attn_depth = 2 * (layer_id + 1)
    mlp_depth = attn_depth + 1
    attn = (
        1.0 / math.sqrt(attn_depth),
        math.sqrt(attn_depth - 1) / math.sqrt(attn_depth),
    )
    mlp = (1.0 / math.sqrt(mlp_depth), math.sqrt(mlp_depth - 1) / math.sqrt(mlp_depth))
    return attn, mlp


def compute_ssmax_scale(positions: torch.Tensor) -> torch.Tensor:
    """Per-token FP32 query scale of the NoPE layers: (1 + (pos + 1) / 2048) ** 0.1."""
    return torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain (not 1+w) RMSNorm over the last dim, FP32 inside, one output rounding."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * weight).to(x.dtype)


def sandwich_residual_ref(
    branch: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    next_weight: Optional[torch.Tensor],
    eps: float,
    scales: Tuple[float, float],
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """``r' = rnd(rnd(post_norm(b) * s0) + rnd(r * s1))``; returns ``(next_norm(r'), r')``.

    ``rnd`` rounds to the branch dtype (bf16); ``next_norm`` is skipped (None)
    when ``next_weight`` is None.
    """
    dtype = branch.dtype
    normed = rms_norm_ref(branch, post_weight, eps)
    scaled_branch = (normed.float() * scales[0]).to(dtype)
    scaled_residual = (residual.to(dtype).float() * scales[1]).to(dtype)
    combined = (scaled_branch.float() + scaled_residual.float()).to(dtype)
    out = rms_norm_ref(combined, next_weight, eps) if next_weight is not None else None
    return out, combined


def neox_rope_ref(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Split-half (NeoX) rotation of ``x [T, H, D]`` by FP32 ``cos/sin [T, D/2]``.

    Rotates in FP32 and rounds once to ``x.dtype``.
    """
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    cos = cos.float().unsqueeze(-2)
    sin = sin.float().unsqueeze(-2)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


def qk_norm_position_ref(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cos_sin_cache: Optional[torch.Tensor],
    ssmax_scale: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-head q/k RMSNorm, then RoPE (sliding) or SSMax on q (full).

    ``qkv [T, >= (Hq + 2*Hkv)*D]`` starts with q|k (trailing v|gate columns are
    ignored); returns ``q [T, Hq*D]``, ``k [T, Hkv*D]``. Exactly one of
    ``cos_sin_cache`` (``[max_pos, D]`` FP32, cos|sin halves) and
    ``ssmax_scale`` (``[T]`` FP32) is given. The norm output is rounded before
    the rotation or scale, which then round once more.
    """
    num_tokens = qkv.shape[0]
    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    q = qkv[:, :q_size].reshape(num_tokens, num_q_heads, head_dim)
    k = qkv[:, q_size : q_size + kv_size].reshape(num_tokens, num_kv_heads, head_dim)
    qh = rms_norm_ref(q, q_weight, eps)
    kh = rms_norm_ref(k, k_weight, eps)
    if cos_sin_cache is not None:
        cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
        qh = neox_rope_ref(qh, cos, sin)
        kh = neox_rope_ref(kh, cos, sin)
    else:
        qh = (qh.float() * ssmax_scale[:, None, None]).to(qkv.dtype)
    return qh.reshape(num_tokens, -1), kh.reshape(num_tokens, -1)


def attention_gate_ref(
    attention_output: torch.Tensor, gate: torch.Tensor, gate_bias: torch.Tensor
) -> torch.Tensor:
    """``attn * sigmoid(gate + gate_bias[head])``, each step rounded to the input dtype."""
    heads = gate_bias.numel()
    num_tokens = attention_output.shape[0]
    multiplier = torch.sigmoid(
        gate.reshape(num_tokens, heads, -1) + gate_bias.to(gate.dtype)[None, :, None]
    )
    return (attention_output.reshape(num_tokens, heads, -1) * multiplier).reshape(
        num_tokens, -1
    )


# --------------------------------------------------------------------------
# Pluggable ops
# --------------------------------------------------------------------------


class _BeamFusedOp(BaseFusedOp):
    """A Beam op whose accelerated forward comes from the BaseFusedOp registry."""

    def _resolve_forward_method(self):
        key = current_platform.get_dispatch_key_name()
        registered = self._oot_forward_registry.get(key, {}).get(type(self))
        if registered is not None:
            return registered.__get__(self)
        return super()._resolve_forward_method()

    def _torch_compile_forward(self, num_tokens: int):
        # A registered kernel stays active under an outer torch.compile.
        return None


class BeamSandwichResidualRMSNormOp(_BeamFusedOp):
    op = "beam.sandwich_residual_rmsnorm"

    def forward_native(self, branch, residual, post_weight, next_weight, eps, scales):
        return sandwich_residual_ref(
            branch, residual, post_weight, next_weight, eps, scales
        )


class BeamRMSNormOp(_BeamFusedOp):
    """Stage-entry ``input_layernorm``; later norms ride the sandwich transitions."""

    op = "beam.rmsnorm"

    def forward_native(self, x, weight, eps):
        return rms_norm_ref(x, weight, eps)


class BeamQKNormPositionOp(_BeamFusedOp):
    op = "beam.qk_norm_position"

    def forward_native(
        self,
        qkv,
        positions,
        q_weight,
        k_weight,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        cos_sin_cache,
        ssmax_scale,
    ):
        return qk_norm_position_ref(
            qkv,
            positions,
            q_weight,
            k_weight,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            cos_sin_cache,
            ssmax_scale,
        )


class BeamGroupedTopKOp(_BeamFusedOp):
    """DeepSeek-V3 grouped sigmoid top-k with correction bias; unscaled weights.

    Same signature as ``biased_grouped_topk_impl``; plugged into ``TopK`` as its
    ``custom_routing_function`` so TopK's padding and EPLB post-processing stay.
    """

    op = "beam.grouped_topk"

    def forward_native(
        self,
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
    ):
        return biased_grouped_topk_impl(
            hidden_states,
            gating_output,
            correction_bias,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )


class BeamAttentionGateOp(_BeamFusedOp):
    op = "beam.attention_gate"

    def forward_native(self, attention_output, gate, gate_bias):
        return attention_gate_ref(attention_output, gate, gate_bias)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class BeamRotaryEmbedding(nn.Module):
    """FP32 NeoX cos|sin table shared by every sliding layer (theta 1e4, full 128 dims)."""

    def __init__(self, config: BeamConfig):
        super().__init__()
        head_dim = config.head_dim
        base = config.rope_parameters["sliding_attention"]["rope_theta"]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        t = torch.arange(config.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos(), freqs.sin()), dim=-1),
            persistent=False,
        )


class BeamMLP(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class BeamMoE(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.tp_size = get_parallel().moe_tp_size
        self.ep_size = get_parallel().moe_ep_size
        self.routed_scaling_factor = config.routed_scaling_factor
        if not get_moe_a2a_backend().is_none():
            raise NotImplementedError("Beam does not support an MoE all-to-all backend")
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32), requires_grad=False
        )
        # Routing weights stay unscaled; the routed scale is applied to the
        # combined expert output in forward(), on every platform.
        self.router = BeamGroupedTopKOp()
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=True,
            scoring_func="sigmoid",
            custom_routing_function=functools.partial(
                self.router,
                correction_bias=self.gate.e_score_correction_bias,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
            ),
            quant_config=quant_config,
        )
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            routed_scaling_factor=1.0,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            prefix=add_prefix("experts", prefix),
        )
        # This block owns the post-expert reduction; a backend that fuses it
        # (fused reduce-scatter) reports handles_moe_reduction after loading.
        self.experts.allow_backend_reduction = True

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch]
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            router_logits, _ = self.gate(hidden_states.float())
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=(
                    forward_batch.moe_num_token_non_padded()
                    if forward_batch is not None
                    else None
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )
        output = self.experts(hidden_states, topk_output)
        output = output * self.routed_scaling_factor
        if self.experts.handles_moe_reduction:
            return output
        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False
        ):
            output = moe_expert_parallel_all_reduce(output)
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True
        ):
            output = moe_tensor_model_parallel_all_reduce(output)
        return output


class BeamAttention(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        rotary_emb: Optional[BeamRotaryEmbedding] = None,
    ):
        super().__init__()
        self.tp_rank = get_parallel().attn_tp_rank
        self.tp_size = get_parallel().attn_tp_size
        total_heads = config.num_attention_heads
        total_kv_heads = config.num_key_value_heads
        if total_heads % self.tp_size:
            raise ValueError("Beam query heads must be divisible by attention TP size")
        if total_kv_heads >= self.tp_size:
            if total_kv_heads % self.tp_size:
                raise ValueError("Beam KV heads must be divisible by attention TP size")
        elif self.tp_size % total_kv_heads:
            raise ValueError("Beam attention TP size must be divisible by KV heads")
        self.num_heads = total_heads // self.tp_size
        self.num_kv_heads = max(1, total_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps

        # Sliding layers use RoPE; full-attention layers use NoPE with SSMax.
        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"
        # HF window 512 is inclusive of the current token; SGLang stores the
        # left span (RPA adds the 1 back).
        self.sliding_window = config.sliding_window - 1 if self.is_sliding else -1

        # q|k|v|gate share one input: one GEMM when every rank owns whole KV
        # heads (MergedColumnParallelLinear shards each slice by head).
        self.use_packed_projection = total_kv_heads >= self.tp_size and not (
            get_lora().enable_lora or get_lora().lora_paths
        )
        if self.use_packed_projection:
            self.qkv_gate_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [
                    total_heads * self.head_dim,
                    total_kv_heads * self.head_dim,
                    total_kv_heads * self.head_dim,
                    total_heads * self.head_dim,
                ],
                bias=False,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_gate_proj", prefix),
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                config.hidden_size,
                self.head_dim,
                total_heads,
                total_kv_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size,
                total_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("gate_proj", prefix),
            )
        self.gate_bias = nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
        set_weight_attrs(self.gate_bias, {"weight_loader": self.load_gate_bias})
        self.o_proj = RowParallelLinear(
            total_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.is_sliding and rotary_emb is None:
            rotary_emb = BeamRotaryEmbedding(config)
        # Shared across layers: keep out of this module's parameters/buffers.
        self._rotary_emb = (rotary_emb,) if self.is_sliding else (None,)
        self.qk_norm_position = BeamQKNormPositionOp()
        self.attention_gate = BeamAttentionGateOp()
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def load_gate_bias(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        default_weight_loader(
            param,
            loaded_weight.narrow(0, self.tp_rank * self.num_heads, self.num_heads),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        ssmax_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv_end = self.q_size + 2 * self.kv_size
        if self.use_packed_projection:
            qkv, _ = self.qkv_gate_proj(hidden_states)
            gate = qkv[:, qkv_end:]
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            gate, _ = self.gate_proj(hidden_states)
        v = qkv[:, self.q_size + self.kv_size : qkv_end]
        rotary_emb = self._rotary_emb[0]
        q, k = self.qk_norm_position(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            self.eps,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            rotary_emb.cos_sin_cache if rotary_emb is not None else None,
            None if rotary_emb is not None else ssmax_scale,
        )
        attention_output = self.attn(q, k, v, forward_batch)
        attention_output = self.attention_gate(
            attention_output.reshape(-1, self.q_size),
            gate.reshape(-1, self.q_size),
            self.gate_bias,
        )
        output, _ = self.o_proj(attention_output)
        return output


class BeamDecoderLayer(nn.Module):
    """One layer as two sandwich transitions; the stream between layers is the residual."""

    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        rotary_emb: Optional[BeamRotaryEmbedding] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.eps = config.rms_norm_eps
        self.attn_residual_scales, self.mlp_residual_scales = (
            depth_scaled_residual_scales(layer_id)
        )
        self.self_attn = BeamAttention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            rotary_emb=rotary_emb,
        )
        self.is_moe = config.mlp_layer_types[layer_id] == "sparse"
        if self.is_moe:
            self.mlp = BeamMoE(
                config,
                layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = BeamMLP(
                config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.sandwich_residual_norm = BeamSandwichResidualRMSNormOp()

    def forward(
        self,
        positions: torch.Tensor,
        normed: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        next_norm_weight: Optional[torch.Tensor],
        ssmax_scale: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """``normed`` is ``input_layernorm(residual)``.

        Returns ``(next_norm(residual'), residual')``; the first element is None
        when ``next_norm_weight`` is None (last layer of a non-final PP stage).
        """
        attn = self.self_attn(positions, normed, forward_batch, ssmax_scale)
        normed, residual = self.sandwich_residual_norm(
            attn,
            residual,
            self.post_attention_layernorm.weight,
            self.pre_feedforward_layernorm.weight,
            self.eps,
            self.attn_residual_scales,
        )
        mlp = self.mlp(normed, forward_batch) if self.is_moe else self.mlp(normed)
        return self.sandwich_residual_norm(
            mlp,
            residual,
            self.post_feedforward_layernorm.weight,
            next_norm_weight,
            self.eps,
            self.mlp_residual_scales,
        )


class BeamModel(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        rotary_emb = (
            BeamRotaryEmbedding(config)
            if "sliding_attention" in config.layer_types
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            layer_fn=lambda idx, prefix: BeamDecoderLayer(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                rotary_emb=rotary_emb,
            ),
            prefix=add_prefix("layers", prefix),
        )
        # Registered once here (layers only hold a non-module reference).
        self.rotary_emb = rotary_emb
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.has_full_attention = any(
            config.layer_types[i] == "full_attention"
            for i in range(self.start_layer, self.end_layer)
        )
        self.stage_input_norm = BeamRMSNormOp()
        # "Before layer i" ids: capturing at i is the output of layer i - 1.
        self.layers_to_capture: List[int] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank:
            residual = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
        else:
            residual = pp_proxy_tensors["hidden_states"]
        ssmax_scale = (
            compute_ssmax_scale(positions) if self.has_full_attention else None
        )
        aux_hidden_states = AuxHiddenStateList() if self.layers_to_capture else None

        first_norm = self.layers[self.start_layer].input_layernorm
        normed = self.stage_input_norm(
            residual, first_norm.weight, first_norm.variance_epsilon
        )
        for i in range(self.start_layer, self.end_layer):
            if aux_hidden_states is not None and i in self.layers_to_capture:
                aux_hidden_states.capture(residual)
            if i + 1 < self.end_layer:
                next_norm_weight = self.layers[i + 1].input_layernorm.weight
            elif self.pp_group.is_last_rank:
                next_norm_weight = self.norm.weight
            else:
                next_norm_weight = None
            normed, residual = self.layers[i](
                positions,
                normed,
                residual,
                forward_batch,
                next_norm_weight,
                ssmax_scale,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": residual})
        if aux_hidden_states is not None and self.end_layer in self.layers_to_capture:
            aux_hidden_states.capture(residual)
        # ``normed`` already holds ``self.norm(residual)``.
        if aux_hidden_states is not None:
            return normed, aux_hidden_states
        return normed


def get_beam_quantization(quant_config: Optional[QuantizationConfig]) -> Optional[str]:
    """``"bf16"``, ``"fp8"`` (compressed-tensors FP8_DYNAMIC), ``"nvfp4"``
    (ModelOpt NVFP4 experts) or None if unsupported."""
    if quant_config is None:
        return "bf16"
    if quant_config.get_name() == "modelopt_fp4":
        # Only the routed experts are FP4; the checkpoint's ignore list keeps
        # attention, the router, the dense MLPs and lm_head in BF16.
        if getattr(
            quant_config, "is_checkpoint_nvfp4_serialized", False
        ) and not getattr(quant_config, "is_w4a16", False):
            return "nvfp4"
        return None
    if quant_config.get_name() != "compressed_tensors":
        return None
    if (
        quant_config.quant_format != "float-quantized"
        or getattr(quant_config, "linear_fp8_config", None) is not None
        or not quant_config.target_scheme_map
    ):
        return None
    for scheme in quant_config.target_scheme_map.values():
        weights = scheme.get("weights")
        activations = scheme.get("input_activations")
        if not (
            all(
                q is not None and q.type == "float" and q.num_bits == 8 and q.symmetric
                for q in (weights, activations)
            )
            and weights.strategy == "channel"
            and not weights.dynamic
            and activations.strategy == "token"
            and activations.dynamic is True
            and scheme.get("format") in (None, "float-quantized")
        ):
            return None
    return "fp8"


class BeamForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_gate_proj": ["q_proj", "k_proj", "v_proj", "gate_proj"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # Every ``.experts.<id>.`` checkpoint key loads into FusedMoE expert <id>.
    supports_tpu_ep_local_expert_reads = True

    def __init__(
        self,
        config: BeamConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if get_beam_quantization(quant_config) is None:
            raise NotImplementedError(
                "Beam supports BF16, compressed-tensors FP8 (per-channel "
                "weights, dynamic per-token activations) or ModelOpt NVFP4 "
                "checkpoints"
            )
        if get_parallel().attn_dp_size > 1:
            raise NotImplementedError(
                "Beam does not support DP attention yet; use pure TP attention"
            )
        if (
            get_exec().moe.enable_eplb
            or get_exec().moe.init_expert_location != "trivial"
            or get_exec().moe.ep_num_redundant_experts > 0
        ):
            raise NotImplementedError(
                "Beam has not been validated with expert relocation or redundant "
                "experts; use trivial expert placement"
            )
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.model = BeamModel(config, quant_config, prefix=add_prefix("model", prefix))
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()
        # Applies config.final_logit_softcapping (30) to the logits.
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states=aux_hidden_states,
        )

    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
        """Capture the outputs (post-layer residual stream) of decoder ``layer_ids``."""
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        layer_ids = list(layer_ids)
        if (
            not layer_ids
            or layer_ids != sorted(set(layer_ids))
            or any(i < 0 or i >= self.config.num_hidden_layers for i in layer_ids)
        ):
            raise ValueError(
                "DFlash capture layers must be sorted, unique, and in range"
            )
        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [i + 1 for i in layer_ids]

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        current_platform.empty_cache()
        current_platform.synchronize()

    def get_attention_sliding_window_size(self) -> Optional[int]:
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=config.n_group,
        )

    def _is_other_pp_rank_weight(self, name: str) -> bool:
        layer_id = get_layer_id(name)
        if layer_id is not None:
            return not self.start_layer <= layer_id < self.end_layer
        if name.startswith("model.embed_tokens."):
            return not self.pp_group.is_first_rank
        if name.startswith("model.norm.") or name.startswith("lm_head."):
            return not self.pp_group.is_last_rank
        return False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        quant_config = getattr(self, "quant_config", None)
        if get_beam_quantization(quant_config) == "fp8":
            # BF16 checkpoint + FP8_DYNAMIC quantization_config override: quantize
            # on the host as tensors stream. FP8 checkpoint tensors pass through.
            ignore = quant_config.ignore
            weights = online_fp8_weights(
                weights, lambda module: not check_equal_or_regex_match(module, ignore)
            )
        attention_shards = (
            [
                (".self_attn.q_proj.", ".self_attn.qkv_gate_proj.", 0),
                (".self_attn.k_proj.", ".self_attn.qkv_gate_proj.", 1),
                (".self_attn.v_proj.", ".self_attn.qkv_gate_proj.", 2),
                (".self_attn.gate_proj.", ".self_attn.qkv_gate_proj.", 3),
            ]
            if self._uses_packed_projection()
            else [
                (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"),
                (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
                (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
            ]
        )
        stacked = attention_shards + [
            (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
            (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
        ]
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params = dict(self.named_parameters())
        loaded: Set[str] = set()

        def param_for(target: str, source: str) -> Optional[nn.Parameter]:
            if target in params:
                return params[target]
            if self._is_other_pp_rank_weight(source):
                return None
            raise KeyError(
                f"Beam checkpoint tensor {source!r} has no parameter {target!r}"
            )

        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for source, target, shard_id in stacked:
                if source in name:
                    target_name = name.replace(source, target)
                    param = param_for(target_name, name)
                    if param is not None:
                        param.weight_loader(param, weight, shard_id)
                        loaded.add(target_name)
                    break
            else:
                for target, source, expert_id, shard_id in expert_mapping:
                    if source in name:
                        target_name = name.replace(source, target)
                        param = param_for(target_name, name)
                        if param is not None:
                            param.weight_loader(
                                param,
                                weight,
                                target_name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                            loaded.add(target_name)
                        break
                else:
                    param = param_for(name, name)
                    if param is not None:
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, weight)
                        loaded.add(name)
        return loaded

    def _uses_packed_projection(self) -> bool:
        for i in range(self.start_layer, self.end_layer):
            return self.model.layers[i].self_attn.use_packed_projection
        return True


EntryClass = BeamForCausalLM
