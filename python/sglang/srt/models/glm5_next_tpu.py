# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text serving on TPU.

Projection and checkpoint mappings adapted from SGLang PR #36507. This initial
implementation supports bounded-context text generation: below index_topk
tokens DSA selects the entire causal prefix, which can be evaluated exactly using
ordinary attention. KDA runs on the fused GDN v3 Pallas kernel in KDA mode
(sglang_tpu.attention_kda_pallas, shared with Kimi-K3).
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.distributed import divide
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils.common import set_weight_attrs


class TpuRMSNormGated(nn.Module):
    def __init__(self, size, eps, activation="sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float() * torch.sigmoid(gate.float())).to(x.dtype)


class Glm5NextLinearAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        reduce_results: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = get_parallel().tp_size
        head_shard_size = get_parallel().attn_tp_size
        head_shard_rank = get_parallel().attn_tp_rank

        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.linear_attn_config["head_dim"]
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % head_shard_size == 0
        self.local_num_heads = divide(self.num_heads, head_shard_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_k_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
            prefix=f"{prefix}.qkv_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_b_proj",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, head_shard_size), dtype=torch.float32)
        )

        set_weight_attrs(
            self.dt_bias,
            {"weight_loader": sharded_weight_loader(0)},
        )

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.qkv_conv1d",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )
        # ColumnParallelLinear's loader cannot reshape conv1d weights, so add the
        # singleton dimension after construction.
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(
            self.A_log,
            {"weight_loader": sharded_weight_loader(2)},
        )

        self.o_norm = TpuRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=reduce_results,
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.local_num_heads,
            num_k_heads=self.local_num_heads,
            num_v_heads=self.local_num_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.attn.lower_bound = config.linear_attn_config.get("gate_lower_bound", None)

    def bind_tpu_attn_weights(self) -> None:
        """Bind the weights the TPU KDA kernel reads, after load (as Kimi-K3 does).

        The conv weight is the depthwise [dim, 1, K] parameter itself: torch_tpu
        would re-materialize a view of it on every call. A_log is an [n_v] copy,
        and the kernel reads conv / dt_bias / A_log from one packed f32 operand,
        so online weight updates are rejected (see the model's load_weights).
        """
        from sglang_tpu.kda_layout import pack_kda_weights

        attn = self.attn
        attn.conv_weights = self.qkv_conv1d.weight
        attn.A_log = nn.Parameter(
            self.A_log.detach().reshape(-1).clone(), requires_grad=False
        )
        attn.dt_bias = self.dt_bias
        # The packed operand has no conv bias; qkv_conv1d is built without one.
        assert attn.bias is None, "packed KDA weights carry no conv bias"
        attn.packed_kernel_weights = pack_kda_weights(
            attn.conv_weights,
            attn.dt_bias,
            attn.A_log,
            attn.num_v_heads,
            attn.head_k_dim,
        ).to(attn.conv_weights.device)

    def forward_qkvbfg(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        qkv, _ = self.qkv_proj(hidden_states)

        beta = self.b_proj(hidden_states)[0]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
            hidden_states, forward_batch
        )

        if not forward_batch.forward_mode.is_decode():
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))
        # The TPU backend returns [1, tokens, heads, dim]; drop the batch axis
        # only, so a one-token decode keeps its token axis.
        core_attn_out = core_attn_out.reshape(norm_gate.shape)
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.flatten(-2)

        return self.o_proj(core_attn_out)[0]


class Glm5NextDenseAttention(nn.Module):
    """Unabsorbed MLA, exact while all causal keys fit the indexer budget."""

    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.heads = divide(config.num_attention_heads, get_parallel().attn_tp_size)
        self.q_dim = config.qk_nope_head_dim
        self.v_dim = config.v_head_dim
        self.q_rank = config.q_lora_rank
        self.kv_rank = config.kv_lora_rank
        if config.qk_rope_head_dim != 0:
            raise ValueError("GLM-5.3-Flash TPU expects NoPE attention")
        self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.q_rank + self.kv_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(self.q_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_rank,
            config.num_attention_heads * self.q_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_rank,
            config.num_attention_heads * (self.q_dim + self.v_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = RadixAttention(
            self.heads,
            self.q_dim,
            self.q_dim**-0.5,
            num_kv_heads=self.heads,
            layer_id=layer_id,
            v_head_dim=self.v_dim,
        )

    def forward(self, hidden_states, forward_batch, **kwargs):
        q_latent, kv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
            [self.q_rank, self.kv_rank], dim=-1
        )
        q = self.q_b_proj(self.q_a_layernorm(q_latent))[0].view(
            -1, self.heads, self.q_dim
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_latent))[0].view(
            -1, self.heads, self.q_dim + self.v_dim
        )
        k, v = kv.split([self.q_dim, self.v_dim], dim=-1)
        out = self.attn(q, k.contiguous(), v.contiguous(), forward_batch)
        return self.o_proj(out.reshape(-1, self.heads * self.v_dim))[0]


def hc_pre(x, fn, scale, base, config):
    """mHC equations from the checkpoint's Transformers reference, in fp32."""
    h = config.hc_mult
    flat = x.flatten(-2).float()
    flat = flat * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    pre, post, comb = F.linear(flat, fn.float()).split([h, h, h * h], dim=-1)
    pre = torch.sigmoid(pre * scale[0] + base[:h]) + config.hc_eps
    post = 2 * torch.sigmoid(post * scale[1] + base[h : 2 * h])
    comb = (
        torch.softmax((comb * scale[2] + base[2 * h :]).view(-1, h, h), dim=-1)
        + config.hc_eps
    )
    comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    for _ in range(config.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + config.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    return (pre[..., None] * x.float()).sum(-2).to(x.dtype), post, comb


def hc_post(output, residual, post, comb):
    return (
        post[..., None] * output.float().unsqueeze(-2)
        + torch.einsum("tij,tid->tjd", comb, residual.float())
    ).to(residual.dtype)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.config = config
        if config.is_kda_layer(layer_id):
            self.self_attn = Glm5NextLinearAttention(
                layer_id,
                config.hidden_size,
                config,
                quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=True,
            )
        else:
            self.self_attn = Glm5NextDenseAttention(
                config, layer_id, quant_config, f"{prefix}.self_attn"
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if layer_id < config.first_k_dense_replace:
            self.mlp = DeepseekV2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        else:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                layer_id=layer_id,
            )
        for site in ("attn", "ffn"):
            h = config.hc_mult
            for suffix, shape in (
                ("base", ((2 + h) * h,)),
                ("scale", (3,)),
                ("fn", ((2 + h) * h, h * config.hidden_size)),
            ):
                setattr(
                    self,
                    f"hc_{site}_{suffix}",
                    nn.Parameter(torch.empty(shape, dtype=torch.float32)),
                )

    def forward(self, x, forward_batch):
        residual = x
        x, post, comb = hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.config
        )
        x = self.self_attn(self.input_layernorm(x), forward_batch)
        x = hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.config
        )
        x = self.mlp(self.post_attention_layernorm(x), forward_batch)
        return hc_post(x, residual, post, comb)


class Glm5NextModel(nn.Module):
    def __init__(self, config, quant_config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(config, i, quant_config, f"model.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.start_layer, self.end_layer = 0, config.num_hidden_layers

    def forward(self, input_ids, forward_batch, input_embeds=None):
        x = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        x = x.unsqueeze(-2).expand(-1, self.config.hc_mult, -1)
        for layer in self.layers:
            x = layer(x, forward_batch)
        return self.norm(x.mean(-2))


class Glm5NextForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if not current_platform.is_tpu():
            raise NotImplementedError("This GLM-5.3-Flash implementation requires TPU")
        args = get_server_args()
        self.config = config.get_text_config()
        c = self.config
        dense_limit = c.index_topk
        if not args.language_model_only:
            raise ValueError("GLM-5.3-Flash TPU requires --language-model-only")
        if getattr(c, "index_kpool", 1) > 1 and not getattr(
            c, "index_kpool_always_select_tail", False
        ):
            raise ValueError(
                "Dense DSA requires the checkpoint's always-select-tail setting"
            )
        if args.context_length is None or args.context_length > dense_limit:
            raise ValueError(
                f"GLM-5.3-Flash TPU requires --context-length <= {dense_limit}; sparse top-k is not implemented"
            )
        if not args.disable_radix_cache:
            raise ValueError(
                "GLM-5.3-Flash TPU currently requires --disable-radix-cache"
            )
        if args.speculative_algorithm or args.enable_dp_attention or args.pp_size != 1:
            raise ValueError(
                "GLM-5.3-Flash TPU currently supports TP/EP text serving without speculative decoding or DP/PP"
            )
        if not c.mhc or c.qk_rope_head_dim != 0:
            raise ValueError("Expected the GLM-5.3-Flash mHC/NoPE checkpoint")
        self.quant_config = quant_config
        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)
        self.model = Glm5NextModel(c, quant_config)
        self.lm_head = ParallelLMHead(
            c.vocab_size, c.hidden_size, quant_config=quant_config, prefix="lm_head"
        )
        self.logits_processor = LogitsProcessor(c)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        hidden = self.model(input_ids, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        config = config.get_text_config()
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
        )

    def load_weights(self, weights):
        """Load the language checkpoint, explicitly excluding vision, MTP and the inactive indexer."""
        params = dict(self.named_parameters())
        stacked = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("qkv_conv1d", "q_conv1d", 0),
            ("qkv_conv1d", "k_conv1d", 1),
            ("qkv_conv1d", "v_conv1d", 2),
        ]
        expert_mapping = FusedMoE.make_expert_params_mapping(
            "gate_proj", "down_proj", "up_proj", self.config.n_routed_experts
        )
        cached = {}
        loaded = set()
        for name, weight in weights:
            name = name.replace("model.language_model.", "model.")
            if name.startswith("model.visual.") or ".indexer." in name:
                continue
            if (
                name.startswith("model.layers.")
                and int(name.split(".")[2]) >= self.config.num_hidden_layers
            ):
                continue
            if ".experts." in name:
                for target, source, expert_id, shard_id in expert_mapping:
                    if source in name:
                        name = name.replace(source, target)
                        param = params[name]
                        param.weight_loader(
                            param, weight, name, shard_id=shard_id, expert_id=expert_id
                        )
                        loaded.add(name)
                        break
                else:
                    raise KeyError(f"Unrecognized expert weight: {name}")
                continue
            for target, source, shard_id in stacked:
                candidate = name.replace(source, target)
                if source in name and candidate in params:
                    params[candidate].weight_loader(params[candidate], weight, shard_id)
                    loaded.add(candidate)
                    break
            else:
                if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
                    cached[name] = weight
                    qname = name.replace("kv_a_proj_with_mqa", "q_a_proj")
                    kvname = name.replace("q_a_proj", "kv_a_proj_with_mqa")
                    if qname not in cached or kvname not in cached:
                        continue
                    name = qname.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    weight = torch.cat([cached.pop(qname), cached.pop(kvname)], dim=0)
                if name.endswith(".A_log"):
                    weight = weight.view(1, 1, -1, 1)
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
        if cached:
            raise ValueError(f"Unpaired attention projections: {list(cached)}")
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"Missing GLM-5.3-Flash weights: {sorted(missing)}")
        # After loaders that rebind storage (the TPU FP8 preparation path uses
        # this convention), bind what the KDA kernel reads.
        for layer in self.model.layers:
            attn = layer.self_attn
            if isinstance(attn, Glm5NextLinearAttention):
                attn.bind_tpu_attn_weights()
        # Writes that skip load_weights would leave the bound copies stale.
        self._derived_weight_cache_error = (
            "Online weight updates are not supported for GLM-5.3-Flash on TPU: "
            "the KDA kernel reads conv weight / dt_bias / A_log copies built at "
            "load time. Restart the server to change weights."
        )


# Replaces the upstream glm5_next implementation on TPU.
EntryClass = [Glm5NextForConditionalGeneration] if current_platform.is_tpu() else []
