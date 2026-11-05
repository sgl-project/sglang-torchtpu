# SPDX-License-Identifier: Apache-2.0
"""Inkling2 text decoder on TPU.

Each layer: Block AttnRes over completed residual blocks -> RMSNorm -> attention
(K/V short convolutions, L2-clipped Q/K/R, learned relative logits, output gate
and per-token sink) -> attention-output sconv -> Block AttnRes -> RMSNorm -> MoE
-> MLP-output sconv.

Attention runs on the RPA v3 Pallas kernel with its relative-bias and
per-token-sink inputs; short convolutions run on the TPU sconv backend; experts
run through FusedMoE with the checkpoint's per-layer NVFP4 / MXFP8 precision
(sglang_tpu/models/inkling2_quant.py). Vision and the MTP draft are not served.
"""

from __future__ import annotations

import logging
from typing import Iterable, NamedTuple, Optional


import torch
from torch import nn

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary.ops import _dp_scatter_step, dp_gather, to_dp_local
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers import mm_utils
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.inkling_common.sconv import SconvType
from sglang.srt.runtime_context import get_mm, get_parallel
from sglang.srt.utils import add_prefix
from sglang.tml.config import Inkling2MMConfig, Inkling2ModelConfig
from sglang_tpu.compilation.hybrid_linear_direct_compile import ConvStateModule
from sglang_tpu.envs import envs
from sglang_tpu.inkling2_attn_res import (
    attn_res_norm_tpu,
    register_inkling2_attn_res_op,
    supports_attn_res_norm,
)
from sglang_tpu.mm_embed_tpu import embed_mm_inputs_tpu
from sglang_tpu.mm_embedding import EmbeddingSpan, PaddedEmbedding
from sglang_tpu.models.inkling2_quant import FP8_BLOCK, TpuInkling2QuantConfig
from sglang_tpu.models.inkling2_vision import (
    Inkling2VisionEncoderTpu,
    prepare_vision_input,
)

logger = logging.getLogger(__name__)

# Checkpoint projection order inside the fused qkvr projection.
_QKVR_PROJECTIONS = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")
_ATTN_RES_EPS = 1e-6
_L2_CLIP_EPS = 1e-6
# Prefill-sized batches route 1/tp of the rows per rank; below this the gather costs more.
# Arbitrary; measured only at 8192.
_SHARDED_ROUTE_MIN_TOKENS = 1024


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class Inkling2RMSNorm(nn.Module):
    """RMSNorm with an optional learned scale before normalization."""

    def __init__(self, hidden_size, eps, *, pre_affine=False, affine=True, out_dtype=None):
        super().__init__()
        self.eps = eps
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(torch.ones(hidden_size)) if affine else None
        self.pre_weight = nn.Parameter(torch.ones(hidden_size)) if pre_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.out_dtype or x.dtype
        if self.pre_weight is not None:
            x = x * self.pre_weight
        y = _rms(x, self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(dtype)


def _l2_clip(x, threshold, scale=None, tau=None):
    """x * min(1, threshold / |scale * x|), then * tau per token. x: [T, H, D]."""
    value = x.float()
    if scale is not None:
        value = value * scale.float()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    value = value * (threshold / (norm + _L2_CLIP_EPS)).clamp(max=1.0)
    out = value.to(x.dtype)
    if tau is not None:
        out = (out.float() * tau[:, None, None]).to(x.dtype)
    return out


def _log_scaling_tau(positions, n_floor, alpha):
    return 1.0 + alpha * torch.log(torch.clamp((positions + 1).float() / n_floor, min=1.0))


# ----------------------------------------------------------------------------
# Block attention residuals
# ----------------------------------------------------------------------------


def _inv_rms(x: torch.Tensor) -> torch.Tensor:
    """[T, D] -> [T] fp32 inverse RMS."""
    return torch.rsqrt(x.float().square().mean(dim=-1) + _ATTN_RES_EPS)


def _block_attn_res(blocks, block_inv_rms, partial, queries, *, normalize_values, addend=None):
    """Attention over residual sources at each token; one output per query row.

    ``block_inv_rms`` holds the cached inverse RMS of the leading ``blocks`` (it may be shorter;
    the rest are computed). Sources are read in place: stacking them into one fp32 tensor per
    call cost several full passes over every block.
    """
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    last = partial.float() if addend is None else partial.float() + addend
    sources = [*blocks, last]
    inv = [*block_inv_rms, *(_inv_rms(s) for s in sources[len(block_inv_rms) :])]
    q = queries.float().T * partial.shape[-1] ** -0.5
    # [T, Q, n] logits: one small matmul per source.
    logits = torch.stack([(s.float() @ q) * r[:, None] for s, r in zip(sources, inv)], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    out = None
    for i, (s, r) in enumerate(zip(sources, inv)):
        value = s.float() * r[:, None] if normalize_values else s.float()
        term = probs[:, :, i, None] * value[:, None, :]
        out = term if out is None else out + term
    return tuple(out.unbind(1))


class Inkling2AttnRes(nn.Module):
    def __init__(self, config: Inkling2ModelConfig):
        super().__init__()
        n, h = config.num_hidden_layers, config.hidden_size
        self.block_size = config.n_layers_per_attnres_block
        self.local_layer_ids = frozenset(config.local_layer_ids)
        self.normalize_values = config.attn_res_normalize_values
        # attn_queries[i] belongs to layer i + 1; layer 0 attends to the embedding only.
        self.attn_queries = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(h if i in self.local_layer_ids else (3, h)),
                    requires_grad=False,
                )
                for i in range(1, n)
            ]
        )
        self.mlp_queries = nn.ParameterList(
            [nn.Parameter(torch.zeros(h), requires_grad=False) for _ in range(n)]
        )
        self.output_query = nn.Parameter(torch.zeros(h), requires_grad=False)
        # The kernel implements the RMSNorm without a pre-affine scale.
        self.use_pallas = envs.SGLANG_TPU_INKLING2_ATTN_RES_PALLAS.get() and not config.pre_norm_pre_affine
        if self.use_pallas:
            register_inkling2_attn_res_op()

    def is_block_boundary(self, layer_idx: int) -> bool:
        return layer_idx % self.block_size == 0


class _AttnResState:
    """Residual blocks of one forward pass; layers append in order."""

    def __init__(self, attn_res: Inkling2AttnRes, partial: torch.Tensor):
        self.attn_res = attn_res
        self.partial = partial
        self.blocks: list[torch.Tensor] = []
        # Inverse RMS of each finished block, computed once when it is appended.
        self.block_inv_rms: list[torch.Tensor] = []

    def _mix(self, blocks, partial, query, addend=None):
        return _block_attn_res(
            blocks,
            self.block_inv_rms,
            partial,
            query,
            normalize_values=self.attn_res.normalize_values,
            addend=addend,
        )

    def attention_input(self, layer_idx: int) -> tuple:
        if not self.blocks:
            # Only the embedding exists: every pseudo-query sees the same single source.
            value = self.partial.float()
            if self.attn_res.normalize_values:
                value = _rms(value, _ATTN_RES_EPS)
            count = 1 if layer_idx in self.attn_res.local_layer_ids else 3
            return (value,) * count
        return self._mix(self.blocks, self.partial, self.attn_res.attn_queries[layer_idx - 1])

    def _mix_normed(self, sources, addend, query, norms):
        """Normed bf16 sub-layer inputs from the Pallas kernel, or None when it does not apply."""
        if not self.attn_res.use_pallas or not supports_attn_res_norm(self.partial.shape[0], len(sources)):
            return None
        queries = query.float().reshape(len(norms), -1) * self.partial.shape[-1] ** -0.5
        weights = torch.stack([norm.weight.float() for norm in norms])
        out = attn_res_norm_tpu(
            sources,
            addend,
            queries,
            weights,
            normalize=self.attn_res.normalize_values,
            eps_res=_ATTN_RES_EPS,
            eps_norm=norms[0].eps,
        )
        return tuple(out.unbind(0))

    def attention_input_normed(self, layer_idx: int, norms) -> tuple:
        if self.blocks:
            query = self.attn_res.attn_queries[layer_idx - 1]
            out = self._mix_normed([*self.blocks, self.partial], None, query, norms)
            if out is not None:
                return out
        return tuple(norm(stream) for norm, stream in zip(norms, self.attention_input(layer_idx)))

    def mlp_input_normed(self, layer_idx: int, attn_output: torch.Tensor, norm) -> torch.Tensor:
        query = self.attn_res.mlp_queries[layer_idx]
        if self.attn_res.is_block_boundary(layer_idx):
            out = self._mix_normed([*self.blocks, self.partial, attn_output], None, query, (norm,))
        else:
            out = self._mix_normed([*self.blocks, self.partial], attn_output, query, (norm,))
        if out is not None:
            return out[0]
        return norm(self.mlp_input(layer_idx, attn_output))

    def mlp_input(self, layer_idx: int, attn_output: torch.Tensor) -> torch.Tensor:
        query = self.attn_res.mlp_queries[layer_idx]
        if self.attn_res.is_block_boundary(layer_idx):
            return self._mix([*self.blocks, self.partial], attn_output, query)[0]
        return self._mix(self.blocks, self.partial, query, addend=attn_output.float())[0]

    def finalize_layer(self, layer_idx, attn_output, mlp_output) -> None:
        if self.attn_res.is_block_boundary(layer_idx):
            self.blocks.append(self.partial)
            self.block_inv_rms.append(_inv_rms(self.partial))
            self.partial = (attn_output.float() + mlp_output.float()).to(attn_output.dtype)
        else:
            self.partial = (
                self.partial.float() + attn_output.float() + mlp_output.float()
            ).to(self.partial.dtype)

    def output(self) -> torch.Tensor:
        return self._mix(self.blocks, self.partial, self.attn_res.output_query)[0]


# ----------------------------------------------------------------------------
# Short convolution
# ----------------------------------------------------------------------------


class Inkling2ShortConv(ConvStateModule):
    """Causal conv over the last ``kernel_size`` inputs of one stream, cached per request.

    Static streams load ``weight`` [D, 1, W] (tap order, lag = W-1-tap); dynamic
    ones load ``static_w`` [W, D] (lag order) and ``weight_proj`` [heads*W, H].
    """

    def __init__(self, *, dim, kernel_size, stream, layer_id, shard, n_heads=1, dynamic_input_size=None):
        super().__init__()
        self.dim, self.kernel_size = dim, kernel_size
        self.stream, self.layer_id = stream, layer_id
        self.shard = shard  # (tp_rank, tp_size) of the channel dim, (0, 1) when replicated
        self.n_heads = n_heads
        self.is_dynamic = dynamic_input_size is not None
        if self.is_dynamic:
            self.static_w = nn.Parameter(torch.empty(kernel_size, dim), requires_grad=False)
            self.weight_proj = nn.Parameter(
                torch.empty(n_heads * kernel_size, dynamic_input_size), requires_grad=False
            )
            self.static_w.weight_loader = self._load_cols
            self.weight_proj.weight_loader = self._load_rows
        else:
            self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size), requires_grad=False)
            self.weight.weight_loader = self._load_rows

    def _load_rows(self, param, loaded):
        rank, size = self.shard
        if size > 1 and loaded.shape[0] != param.shape[0]:
            loaded = loaded.narrow(0, rank * param.shape[0], param.shape[0])
        param.data.copy_(loaded)

    def _load_cols(self, param, loaded):
        rank, size = self.shard
        if size > 1 and loaded.shape[1] != param.shape[1]:
            loaded = loaded.narrow(1, rank * param.shape[1], param.shape[1])
        param.data.copy_(loaded)

    def forward(self, x: torch.Tensor, dynamic_input: Optional[torch.Tensor] = None):
        # The kernel adds x_t and scales by (W+1)^-0.5, or (2W+1)^-0.5 with dyn.
        backend = get_attn_backend()
        if not self.is_dynamic:
            return backend.run_sconv(
                x, layer_id=self.layer_id, stream=int(self.stream), weight=self.weight
            )
        dyn = torch.nn.functional.linear(dynamic_input.to(x.dtype), self.weight_proj.to(x.dtype))
        return backend.run_sconv(
            x,
            layer_id=self.layer_id,
            stream=int(self.stream),
            weight=self.static_w,
            dyn=dyn.view(x.shape[0], self.n_heads, self.kernel_size),
        )


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------


class AttentionSources(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


class Inkling2Attention(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, layer_id: int, is_local: bool, prefix: str):
        super().__init__()
        parallel = get_parallel()
        self.tp_rank, self.tp_size = parallel.attn_tp_rank, parallel.attn_tp_size
        heads = config.swa_num_attention_heads if is_local else config.num_attention_heads
        kv_heads = config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        if heads % self.tp_size or kv_heads % self.tp_size:
            raise ValueError("Attention TP must divide the query and KV head counts")
        self.heads, self.kv_heads = heads // self.tp_size, kv_heads // self.tp_size
        self.d_rel = config.d_rel
        self.is_local = is_local
        self.scaling = self.head_dim**-0.5
        self.qk_clip, self.rel_clip = config.qk_l2_clip_threshold, config.rel_l2_clip_threshold
        self.tau_floor, self.tau_alpha = config.log_scaling_n_floor, config.log_scaling_alpha
        self.split_sizes = [
            self.heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.heads * self.d_rel,
            self.heads * self.head_dim,
            self.heads,
        ]
        pad = -sum(self.split_sizes) % 8
        if pad:
            self.split_sizes.append(pad)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * self.tp_size for size in self.split_sizes],
            bias=False,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("qkvr", prefix),
        )
        # Under DP attention the sum stays inside this DP group's attention-TP ranks.
        self.wo_ud = RowParallelLinear(
            heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=True,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            use_dp_attention_reduce=is_dp_attention_enabled(),
            prefix=add_prefix("wo_ud", prefix),
        )
        rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.rel_logits_proj = nn.Module()
        self.rel_logits_proj.proj = nn.Parameter(
            torch.empty(self.d_rel, rel_extent), requires_grad=False
        )
        self.q_l2_clip = nn.Module()
        self.q_l2_clip.scale = self._head_param(self.heads, self.head_dim)
        self.rel_l2_clip = nn.Module()
        self.rel_l2_clip.scale = self._head_param(self.heads, self.d_rel)
        self.v_norm_eps = config.rms_norm_eps if config.use_value_norm_no_gamma else None
        conv = dict(
            dim=self.kv_heads * self.head_dim,
            kernel_size=config.sconv_kernel_size,
            layer_id=layer_id,
            shard=(self.tp_rank, self.tp_size),
            n_heads=self.kv_heads,
            dynamic_input_size=None if is_local else config.hidden_size,
        )
        self.k_sconv = Inkling2ShortConv(
            stream=SconvType.K_LOCAL if is_local else SconvType.K_FULL, **conv
        )
        self.v_sconv = Inkling2ShortConv(
            stream=SconvType.V_LOCAL if is_local else SconvType.V_FULL, **conv
        )
        self.attn = RadixAttention(
            self.heads,
            self.head_dim,
            self.scaling,
            self.kv_heads,
            layer_id,
            sliding_window_size=config.sliding_window_size - 1 if is_local else -1,
            prefix=add_prefix("attn", prefix),
        )

    def _head_param(self, heads, dim):
        param = nn.Parameter(torch.empty(heads, dim), requires_grad=False)

        def load(p, loaded):
            if loaded.shape != p.shape:
                loaded = loaded.narrow(0, self.tp_rank * p.shape[0], p.shape[0])
            p.data.copy_(loaded)

        param.weight_loader = load
        return param

    def _project(self, x):
        out, _ = self.qkvr(x)
        return out.split(self.split_sizes, dim=-1)[:6]

    def _project_sources(self, sources: AttentionSources):
        q_proj = self._project(sources.q)
        k_proj = q_proj if sources.k is sources.q else self._project(sources.k)
        v_proj = q_proj if sources.v is sources.q else self._project(sources.v)
        q, _, _, r, gate, sink = q_proj
        return q, k_proj[1], v_proj[2], r, gate, sink

    def forward(self, sources: AttentionSources, positions, forward_batch: ForwardBatch):
        q, k, v, r, gate, sink = self._project_sources(sources)
        tokens = q.shape[0]
        k = self.k_sconv(k, None if self.is_local else sources.k)
        v = self.v_sconv(v, None if self.is_local else sources.v)
        tau = None if self.is_local else _log_scaling_tau(positions, self.tau_floor, self.tau_alpha)
        q = _l2_clip(q.view(tokens, self.heads, self.head_dim), self.qk_clip, self.q_l2_clip.scale, tau)
        k = _l2_clip(k.view(tokens, self.kv_heads, self.head_dim), self.qk_clip)
        v = v.view(tokens, self.kv_heads, self.head_dim)
        if self.v_norm_eps is not None:
            v = _rms(v, self.v_norm_eps).to(q.dtype)
        r = _l2_clip(r.view(tokens, self.heads, self.d_rel), self.rel_clip, self.rel_l2_clip.scale, tau)
        out = self.attn(
            q.reshape(tokens, -1),
            k.reshape(tokens, -1),
            v.reshape(tokens, -1),
            forward_batch,
            # Unscaled: FA4 scale-after-add computes sm_scale * qk + bias (it adds bias / sm_scale first).
            rel_r=r,
            rel_proj=self.rel_logits_proj.proj,
            token_sink=sink,
        )
        out = (out.float() * torch.sigmoid(gate.float())).to(q.dtype)
        out, _ = self.wo_ud(out)
        return out


# ----------------------------------------------------------------------------
# MoE
# ----------------------------------------------------------------------------


class Inkling2MoE(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, layer_id: int, quant_config, prefix: str):
        super().__init__()
        if config.n_shared_experts:
            raise NotImplementedError("Inkling2 shared experts are not served on TPU yet")
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.selection_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False
        )
        # A buffer (graph input) keeps the rank out of the traced graph.
        self.register_buffer(
            "_route_rank", torch.tensor([get_parallel().attn_tp_rank], dtype=torch.int32), persistent=False
        )
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            top_k=self.top_k,
            reduce_results=True,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="situ",
            gemm1_alpha=config.mlp_gate_softcap,
            gemm1_clamp_limit=config.mlp_up_softcap,
        )

    def _route(self, x: torch.Tensor) -> StandardTopKOutput:
        logits = torch.nn.functional.linear(x.float(), self.gate.weight.float())
        scores = torch.sigmoid(logits)
        _, topk_ids = torch.topk(scores + self.selection_bias, self.top_k, dim=-1)
        # One-hot select, not gather: TPU runs the row gather as a slow serial loop.
        expert_ids = torch.arange(self.num_experts, device=x.device)
        topk_scores = torch.where(
            topk_ids.unsqueeze(-1) == expert_ids, scores.unsqueeze(1), 0.0
        ).sum(dim=-1)
        weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True) * self.route_scale
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=topk_ids.to(torch.int32), router_logits=logits
        )

    def _route_sharded(self, x: torch.Tensor) -> StandardTopKOutput:
        """_route on this rank's 1/tp of the rows, gathered: routing is per row, so bit-identical."""
        group = get_parallel().attn_tp_group
        rows = x.shape[0] // group.world_size
        mine = x.view(group.world_size, rows, x.shape[-1]).index_select(0, self._route_rank).squeeze(0)
        topk = self._route(mine)
        return StandardTopKOutput(
            topk_weights=group.all_gather(topk.topk_weights, dim=0),
            topk_ids=group.all_gather(topk.topk_ids, dim=0),
            router_logits=None,
        )

    def forward(self, x: torch.Tensor, num_token_non_padded: Optional[torch.Tensor]) -> torch.Tensor:
        tp = get_parallel().attn_tp_size
        if tp > 1 and x.shape[0] >= _SHARDED_ROUTE_MIN_TOKENS and x.shape[0] % tp == 0:
            topk = self._route_sharded(x)
        else:
            topk = self._route(x)
        if num_token_non_padded is not None:
            # Bucket-padding rows hold stale values; under EP they must not reach the experts.
            pad = (torch.arange(x.shape[0], device=x.device) >= num_token_non_padded)[:, None]
            topk = StandardTopKOutput(
                topk_weights=topk.topk_weights.masked_fill(pad, 0.0),
                topk_ids=topk.topk_ids.masked_fill(pad, -1),
                router_logits=topk.router_logits,
            )
        return self.experts(x, topk)


# ----------------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------------


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, layer_id: int, quant_config, prefix: str):
        super().__init__()
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_ids
        eps, pre = config.rms_norm_eps, config.pre_norm_pre_affine
        self.attn = Inkling2Attention(config, layer_id, self.is_local, add_prefix("attn", prefix))
        self.mlp = Inkling2MoE(config, layer_id, quant_config, add_prefix("mlp", prefix))
        norm = lambda: Inkling2RMSNorm(config.hidden_size, eps, pre_affine=pre, out_dtype=torch.bfloat16)
        if self.is_local:
            self.attn_norm = norm()
        else:
            self.q_input_norm, self.k_input_norm, self.v_input_norm = norm(), norm(), norm()
        self.mlp_norm = norm()
        full = dict(dim=config.hidden_size, kernel_size=config.sconv_kernel_size, layer_id=layer_id, shard=(0, 1))
        self.attn_sconv = Inkling2ShortConv(stream=SconvType.ATTN, **full)
        self.mlp_sconv = Inkling2ShortConv(stream=SconvType.MLP, **full)

    def forward(self, state: _AttnResState, positions, forward_batch: ForwardBatch):
        if self.is_local:
            (x,) = state.attention_input_normed(self.layer_id, (self.attn_norm,))
            sources = AttentionSources(x, x, x)
        else:
            sources = AttentionSources(
                *state.attention_input_normed(
                    self.layer_id, (self.q_input_norm, self.k_input_norm, self.v_input_norm)
                )
            )
        attention = self.attn_sconv(self.attn(sources, positions, forward_batch))
        mlp_input = state.mlp_input_normed(self.layer_id, attention, self.mlp_norm)
        if is_dp_attention_enabled():
            # Experts run on every DP group's tokens; sconv state is per request, so scatter back first.
            # Gathered padding rows are zero, so no pad mask (the local count would mask other groups).
            moe_out = to_dp_local(_dp_scatter_step, forward_batch, self.mlp(dp_gather(mlp_input, forward_batch), None))
        else:
            moe_out = self.mlp(mlp_input, forward_batch.num_token_non_padded)
        mlp = self.mlp_sconv(moe_out)
        state.finalize_layer(self.layer_id, attention, mlp)


class Inkling2TextModel(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, quant_config, prefix: str):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                Inkling2DecoderLayer(config, i, quant_config, add_prefix(f"layers.{i}", prefix))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            pre_affine=config.final_norm_pre_affine,
            out_dtype=torch.bfloat16,
        )
        # Draft features: the residual partial entering each listed layer; the layer
        # count stands for the final-norm hidden.
        self.layers_to_capture: list[int] = []

    def embed(self, input_ids):
        return self.embed_norm(self.embed_tokens(input_ids))

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden = self.embed(input_ids) if input_embeds is None else input_embeds
        state = _AttnResState(self.attn_res, hidden)
        captures = []
        for i, layer in enumerate(self.layers):
            if i in self.layers_to_capture:
                captures.append(state.partial.to(torch.bfloat16))
            layer(state, positions, forward_batch)
        hidden = self.norm(state.output())
        if not self.layers_to_capture:
            return hidden
        captures.append(hidden)
        return hidden, captures


class _NormalizedEmbedding(nn.Module):
    """Token embeddings with the decoder's input RMS norm applied; image rows skip it."""

    def __init__(self, llm: Inkling2TextModel) -> None:
        super().__init__()
        # Registered, so a draft sharing this module traces its weights as graph inputs.
        self.embed_tokens = llm.embed_tokens
        self.embed_norm = llm.embed_norm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    @property
    def weight(self):
        return self.embed_tokens.weight

    @property
    def embedding_dim(self):
        return self.embed_tokens.embedding_dim

    @property
    def num_embeddings(self):
        return self.embed_tokens.num_embeddings


class _Decoder(nn.Module):
    """The compiled decoder-and-logits boundary, outside image preparation."""

    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    @property
    def logits_processor(self):
        return self._owner.logits_processor

    def get_input_embeddings(self):
        return self._owner.get_input_embeddings()

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return self._owner._forward_text(input_ids, positions, forward_batch, input_embeds)


class Inkling2ForConditionalGeneration(nn.Module):
    """Inkling2 on TPU: text decoder, optional PE-Lang image encoder."""

    # Decode and verify embed inside the compiled decoder graph.
    decode_embeds_in_language_model = True
    # Patches stay on the host; prepare_vision_input pads them into the bucket.
    keeps_mm_features_on_host = True

    def __init__(self, config: Inkling2MMConfig, quant_config=None, prefix: str = ""):
        super().__init__()
        if quant_config is not None and not isinstance(quant_config, TpuInkling2QuantConfig):
            raise ValueError("Inkling2 requires its checkpoint precision metadata")
        if get_parallel().pp_group.world_size != 1:
            raise ValueError("Inkling2 attention residuals do not support pipeline parallelism")
        text = config.text_config
        self.config, self.text_config, self.quant_config = config, text, quant_config
        self.llm = Inkling2TextModel(text, quant_config, add_prefix("llm", prefix))
        self.lm_head = ParallelLMHead(
            text.padded_vocab_size,
            text.hidden_size,
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
            prefix=add_prefix("lm_head", prefix),
        )
        # Slices logits to the real vocab and applies the final softcap.
        self.logits_processor = LogitsProcessor(text)
        self.logits_processor.use_fp32_lm_head = True
        # Checkpoint pieces that are combined once all parts of a tensor arrived.
        self._pending: dict[str, torch.Tensor] = {}
        self._normalized_embedding = _NormalizedEmbedding(self.llm)
        self.visual = None
        self.language_model = None
        self.vision_enabled = bool(get_mm().enable_multimodal)
        if self.vision_enabled:
            if config.vision_config is None:
                raise ValueError("Inkling2 image inputs require vision_config")
            from sglang.srt.compilation.support_torch_compile import support_torch_compile
            from sglang_tpu.compilation.decoder_compile_policy import (
                Qwen3VLDecoderTpuCompilePolicy,
            )
            from sglang_tpu.models.inkling2_vision_compile import (
                Inkling2VisionCompilePolicy,
            )

            self.visual = Inkling2VisionEncoderTpu(
                config.vision_config, text.hidden_size, add_prefix("visual", prefix)
            )
            self.language_model = _Decoder(self)
            support_torch_compile(self.language_model, policy=Qwen3VLDecoderTpuCompilePolicy)
            support_torch_compile(self.visual, policy=Inkling2VisionCompilePolicy)

    def get_input_embeddings(self) -> nn.Module:
        return self._normalized_embedding

    def get_attention_sliding_window_size(self) -> int:
        return self.text_config.sliding_window_size - 1

    @property
    def capture_aux_hidden_states(self) -> bool:
        return bool(self.llm.layers_to_capture)

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        num_layers = len(self.llm.layers)
        if (
            not layer_ids
            or layer_ids != sorted(set(layer_ids))
            or layer_ids[0] < 0
            or layer_ids[-1] != num_layers
        ):
            raise ValueError(
                "Inkling2 draft feature layers must be sorted, unique, and end "
                f"with {num_layers}, the post-final-norm hidden state"
            )
        self.llm.layers_to_capture = list(layer_ids[:-1])

    def _forward_text(self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None):
        hidden = self.llm(input_ids, positions, forward_batch, input_embeds)
        aux = None
        if self.capture_aux_hidden_states:
            hidden, aux = hidden
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch, aux)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None, **kwargs):
        if not self.vision_enabled:
            return self._forward_text(input_ids, positions, forward_batch, input_embeds)
        if (
            input_ids is None
            or forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            # Match the compile policy's positional token-ID warmup call shape.
            if input_embeds is None:
                return self.language_model(input_ids, positions, forward_batch)
            return self.language_model(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        return mm_utils.general_mm_embed_routine(
            input_ids,
            forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
            mm_embedder=embed_mm_inputs_tpu,
        )

    def get_image_feature(self, items: list[MultimodalDataItem]):
        """Encode each image in its own bucket; rows in placeholder order."""
        visual = self.visual
        param = visual.proj_out.weight
        spans = []
        for item in items:
            grid = item.model_specific_data["vision_grid_thw"].tolist()
            if len(grid) != 1 or grid[0][0] != 1:
                raise ValueError("Inkling2 vision expects one still image per item")
            _, h, w = grid[0]
            patches = item.feature
            if patches.ndim != 5 or patches.shape[0] != h * w:
                raise ValueError("Inkling2 image patches do not match the image grid")
            inputs, count = prepare_vision_input(
                patches.to(torch.bfloat16),
                [(int(h), int(w))],
                patch_size=visual.patch_size,
                merge=visual.merge_size,
                head_dim=visual.head_dim,
                posemb_grid=visual.posemb_grid_size,
            )
            device_inputs = tuple(t.to(param.device) for t in inputs)
            spans.append(EmbeddingSpan(visual(*device_inputs), 0, count))
        if not spans:
            return param.new_empty((0, self.text_config.hidden_size))
        # The TPU embedder copies only the valid rows to the host for placement.
        return PaddedEmbedding(tuple(spans))

    def pad_input_ids(self, input_ids, mm_inputs):
        return mm_utils.MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        loaded: set[str] = set()
        for checkpoint_name, tensor in weights:
            if checkpoint_name.startswith("model.visual."):
                if self.visual is not None:
                    self._load_vision(params, checkpoint_name[len("model."):], tensor, loaded)
                continue
            name = _param_name(checkpoint_name)
            if name is None or _beyond_layers(name, num_layers):
                continue
            for shard, projection in enumerate(_QKVR_PROJECTIONS):
                if f".attn.{projection}." in name:
                    name = name.replace(f".{projection}.", ".qkvr.")
                    params[name].weight_loader(params[name], tensor, shard)
                    break
            else:
                if ".mlp.experts." in name:
                    self._load_experts(params, name, tensor, loaded)
                    continue
                param = params.get(name)
                if param is None:
                    raise ValueError(f"Unrecognized checkpoint tensor: {checkpoint_name}")
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
            loaded.add(name)
        missing = sorted(
            n for n in params if n not in loaded and not _is_derived_param(n)
        )
        if missing:
            raise RuntimeError(f"Inkling2 checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
        for module in self.modules():
            if isinstance(module, Inkling2Attention) and len(module.split_sizes) > len(_QKVR_PROJECTIONS):
                # The alignment pad shard has no checkpoint tensor; left as torch.empty it holds NaN/Inf.
                module.qkvr.weight.data[-module.split_sizes[-1] :].zero_()

    def _load_vision(self, params, name, tensor, loaded) -> None:
        """visual.* tensors; q/k/v projections fill the fused qkv_proj, out_proj is proj."""
        for shard in ("q", "k", "v"):
            if f".self_attn.{shard}_proj." in name:
                target = name.replace(f".{shard}_proj.", ".qkv_proj.")
                params[target].weight_loader(params[target], tensor, shard)
                loaded.add(target)
                return
        name = name.replace(".self_attn.out_proj.", ".self_attn.proj.")
        param = params.get(name)
        if param is None:
            raise ValueError(f"Unrecognized vision checkpoint tensor: {name}")
        getattr(param, "weight_loader", default_weight_loader)(param, tensor)
        loaded.add(name)

    def _load_experts(self, params, name, tensor, loaded) -> None:
        """Stacked [E, ...] expert tensors -> this rank's FusedMoE parameters."""
        prefix = name.split(".mlp.experts.", 1)[0] + ".mlp.experts."
        leaf = name[len(prefix):]
        precision = self.quant_config.expert_precision(prefix) if self.quant_config else "bf16"
        # Narrow to this rank's experts first: the conversions below then touch 1/EP of the stack.
        tensor = _ep_slice(tensor, self.text_config.n_routed_experts)
        if precision == "fp8_block":  # stored ready to use: FP8 weights and their block scales
            _copy_expert_slice(params[name], tensor)
            loaded.add(name)
            return
        if precision == "nvfp4_packed":
            self._load_packed_nvfp4(params, prefix, leaf, tensor, loaded)
            return
        if precision == "nvfp4":
            dequantized = self._take_nvfp4(prefix, leaf, tensor)
            if dequantized is None:
                return
            leaf, tensor = dequantized
        elif precision == "mxfp8":
            # Weight and its UE8M0 scale may arrive in either order.
            self._pending[name] = tensor
            weight_leaf = leaf.removesuffix("_scale")
            weight = self._pending.get(prefix + weight_leaf)
            scale = self._pending.get(prefix + weight_leaf + "_scale")
            if weight is None or scale is None:
                return
            del self._pending[prefix + weight_leaf], self._pending[prefix + weight_leaf + "_scale"]
            leaf, tensor = weight_leaf, _dequant_mxfp8(weight, scale)
        target = prefix + leaf
        if leaf.startswith("w13"):
            tensor = _deinterleave_gate_up(tensor)
        if precision in ("nvfp4", "mxfp8"):
            tensor, scale_inv = _quant_block_fp8(tensor)
            _copy_expert_slice(params[target + "_scale_inv"], scale_inv)
            loaded.add(target + "_scale_inv")
        _copy_expert_slice(params[target], tensor)
        loaded.add(target)

    def _load_packed_nvfp4(self, params, prefix, leaf, tensor, loaded) -> None:
        """Packed E2M1 weights and FP8 block scales as stored; amax / denominator -> weight_scale_2."""
        if leaf.endswith(("_amax", "_outer_scale_denominator")):
            self._pending[prefix + leaf] = tensor.float()
            for w in ("w13", "w2"):
                amax = self._pending.get(f"{prefix}{w}_weight_amax")
                den = self._pending.get(f"{prefix}{w}_weight_outer_scale_denominator")
                target = f"{prefix}{w}_weight_scale_2"
                if amax is None or den is None or target in loaded:
                    continue
                scale = amax / den.reshape(-1)[0]
                param = params[target]
                if param.dim() == 2:  # w13 holds one scale each for gate and up
                    scale = scale[:, None].expand(-1, param.shape[1])
                _copy_expert_slice(param, scale)
                loaded.add(target)
            return
        if leaf.startswith("w13"):
            tensor = _deinterleave_gate_up(tensor)
        _copy_expert_slice(params[prefix + leaf], tensor)
        loaded.add(prefix + leaf)

    def _take_nvfp4(self, prefix, leaf, tensor):
        """(weight leaf, BF16 [E, N, K]) once weight, block scale, amax and denominator arrived."""
        weight_leaf = leaf.split("_weight", 1)[0] + "_weight"
        self._pending[prefix + leaf] = tensor
        parts = [prefix + weight_leaf + s for s in ("", "_scale", "_amax", "_outer_scale_denominator")]
        if any(p not in self._pending for p in parts):
            return None
        weight, scale, amax, den = (self._pending.pop(p) for p in parts)
        return weight_leaf, _dequant_nvfp4(weight, scale, amax.float() / den.float().reshape(-1)[0])


def _param_name(checkpoint_name: str) -> Optional[str]:
    """Model parameter for a checkpoint tensor, None for the vision tower."""
    if not checkpoint_name.startswith("model.llm."):
        return None
    name = "llm." + checkpoint_name[len("model.llm."):]
    if name == "llm.unembed.weight":
        return "lm_head.weight"
    name = name.replace("llm.embed.weight", "llm.embed_tokens.weight")
    name = name.replace(".mlp.gate.bias", ".mlp.selection_bias")
    return name


def _beyond_layers(name: str, num_layers: int) -> bool:
    """Tensors of layers cut by a num_hidden_layers override."""
    if ".layers." in name:
        return int(name.split(".layers.", 1)[1].split(".", 1)[0]) >= num_layers
    if ".attn_res.attn_queries." in name:  # attn_queries.{i} belongs to layer i + 1
        return int(name.rsplit(".", 1)[1]) + 1 >= num_layers
    if ".attn_res.mlp_queries." in name:
        return int(name.rsplit(".", 1)[1]) >= num_layers
    return False


def _is_derived_param(name: str) -> bool:
    """Parameters the checkpoint does not carry: pads, unused input scales, online FP8 scales."""
    return name.endswith(("_input_scale", "_weight_scale_inv", "_weight_scale_2", "_weight_scale")) and ".experts." in name


def _quant_block_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [E, N, K] -> (FP8 E4M3 [E, N, K], FP32 scale_inv [E, N/bn, K/bk]) per FP8_BLOCK block."""
    bn, bk = FP8_BLOCK
    e, n, k = w.shape
    blocks = w.float().reshape(e, n // bn, bn, k // bk, bk)
    scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max
    q = (blocks / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return q.reshape(e, n, k), scale


def _deinterleave_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """[E, 2I, ...] with gate/up rows interleaved -> [E, 2I, ...] gate rows then up rows."""
    return torch.cat((w13[:, 0::2], w13[:, 1::2]), dim=1)


def _dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 [E, N, K] with UE8M0 scales [E, N, K/32] -> BF16."""
    factor = torch.exp2(scale.to(torch.float32) - 127.0)
    w = weight.to(torch.float32).unflatten(-1, (-1, 32)) * factor.unsqueeze(-1)
    return w.flatten(-2).to(torch.bfloat16)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _dequant_nvfp4(
    weight: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Packed E2M1 [E, N, K/2] (low nibble first), FP8 block-16 scales [E, N, K/16] -> BF16."""
    codes = torch.stack((weight & 0xF, weight >> 4), -1).flatten(-2).long()
    vals = torch.tensor(_E2M1)[codes].unflatten(-1, (-1, 16))
    w = vals * scale.float().unsqueeze(-1) * global_scale.reshape(-1, 1, 1, 1)
    return w.flatten(-2).to(torch.bfloat16)


def _ep_slice(tensor: torch.Tensor, num_experts: int) -> torch.Tensor:
    """This rank's experts of a stacked [E, ...] tensor; other tensors unchanged."""
    parallel = get_parallel()
    if tensor.dim() == 0 or tensor.shape[0] != num_experts or parallel.moe_ep_size == 1:
        return tensor
    count = num_experts // parallel.moe_ep_size
    return tensor.narrow(0, parallel.moe_ep_rank * count, count)


def _copy_expert_slice(param: torch.Tensor, tensor: torch.Tensor) -> None:
    """Copy this rank's experts (EP slices whole experts along dim 0)."""
    parallel = get_parallel()
    if tensor.shape[0] != param.shape[0]:
        count = tensor.shape[0] // parallel.moe_ep_size
        tensor = tensor.narrow(0, parallel.moe_ep_rank * count, count)
    if tuple(tensor.shape) != tuple(param.shape):
        raise RuntimeError(f"Expert tensor shape {tuple(tensor.shape)} vs parameter {tuple(param.shape)}")
    param.data.copy_(tensor.to(param.dtype))


EntryClass = Inkling2ForConditionalGeneration
