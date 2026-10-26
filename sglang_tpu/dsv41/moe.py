# SPDX-License-Identifier: Apache-2.0
"""DP-local tokens and an EP-owned routed FFN, with replicated shared experts."""

import logging
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, MoEGate
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix
from sglang_tpu.dsv41.quantization import prepare_expert_matrix

logger = logging.getLogger(__name__)


@cache
def _build_ep_op(
    ep_size, topk, renormalize, scoring, routed_scale, is_hash, logical_inter
):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.moe.pallas.fused_moe_v2 import WeightFormat, fused_ep_moe_v2
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    parallel = get_parallel()
    if parallel.world_group.world_size != ep_size:
        raise ValueError("V4.1 fused EP must span the complete distributed world")
    mesh = get_pallas_mesh(("ep",), (ep_size,))

    def call(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        route: jax.Array,
    ) -> jax.Array:
        return fused_ep_moe_v2(
            x,
            w1,
            w2,
            s1,
            s2,
            gating.astype(jnp.float32),
            topk=topk,
            renormalize=renormalize,
            mesh=mesh,
            capacity=128,
            weight_format=WeightFormat.FP4,
            rhs_qb=512,
            act_fn="silu_and_mul_with_clamp",
            rank=rank,
            mesh_ep_ranks=tuple(range(ep_size)),
            sharded_plan=True,
            scoring_fn=scoring,
            routed_scaling_factor=routed_scale,
            score_bias=None if is_hash else route,
            hash_topk_ids=route if is_hash else None,
            logical_intermediate_size=logical_inter,
        )

    # These inputs are rank-local, so their exported global axis 0 is EP.
    # Correction bias is indexed by global expert id and is replicated.
    spec = P("ep")
    op = cpu_export_jax_op(
        f"sglang_tpu_pallas::dsv41_fused_ep_{int(is_hash)}",
        call,
        mesh=mesh,
        input_partition_specs=(spec,) * 7 + ((spec,) if is_hash else (P(),)),
        output_partition_specs=(spec,),
    )
    op.register_fake(lambda x, *_: torch.empty_like(x))
    return op


class _ExpertQuantMethod:
    def process_weights_after_loading(self, layer):
        if layer.prepared:
            return
        if layer.ep_rank == 0:
            logger.info("Preparing V4.1 FP4 experts for one layer")
        h, i = layer.hidden_size, layer.intermediate_size
        h_pad, i_pad = (h + 511) // 512 * 512, (i + 511) // 512 * 512
        for name, gate_up in (("w13", True), ("w2", False)):
            weight, scales = prepare_expert_matrix(
                getattr(layer, f"{name}_weight"),
                getattr(layer, f"{name}_weight_scale_inv"),
                gate_up=gate_up,
                padded_hidden=h_pad,
                padded_inter=i_pad,
            )
            setattr(layer, f"{name}_weight", nn.Parameter(weight, requires_grad=False))
            delattr(layer, f"{name}_weight_scale_inv")
            layer.register_buffer(f"{name}_scale", scales, persistent=False)
        layer.prepared = True
        if layer.ep_rank == 0:
            logger.info("Finished preparing V4.1 FP4 experts for one layer")


class FusedEPExperts(nn.Module):
    """Retain the core checkpoint loader's names and expert-to-rank mapping."""

    def __init__(self, config):
        super().__init__()
        parallel = get_parallel()
        self.hidden_size = h = config.hidden_size
        self.intermediate_size = i = config.moe_intermediate_size
        self.ep_size, self.ep_rank = parallel.moe_ep_size, parallel.moe_ep_rank
        if config.n_routed_experts % self.ep_size:
            raise ValueError("Expert count must divide the EP world")
        self.num_local_experts = e = config.n_routed_experts // self.ep_size
        self.prepared = False
        self.quant_method = _ExpertQuantMethod()
        for name, n, k in (("w13", 2 * i, h), ("w2", h, i)):
            weight = nn.Parameter(torch.empty(e, n, k // 2, dtype=torch.int8), False)
            scale = nn.Parameter(torch.empty(e, n, k // 32, dtype=torch.float32), False)
            weight.weight_loader = self.weight_loader
            scale.weight_loader = self.weight_loader
            self.register_parameter(f"{name}_weight", weight)
            self.register_parameter(f"{name}_weight_scale_inv", scale)

    def weight_loader(self, param, loaded, name, *, shard_id, expert_id):
        local = expert_id - self.ep_rank * self.num_local_experts
        if not 0 <= local < self.num_local_experts:
            return
        if name.endswith("weight_scale_inv"):
            loaded = (
                torch.exp2(loaded.float() - 127)
                if loaded.dtype == torch.uint8
                else loaded.float()
            )
        elif loaded.dtype == torch.uint8:
            loaded = loaded.view(torch.int8)
        target = param.data[local]
        if shard_id in ("w1", "w3"):
            target = target.narrow(
                0, 0 if shard_id == "w1" else target.shape[0] // 2, target.shape[0] // 2
            )
        if target.shape != loaded.shape:
            raise ValueError(
                f"{name}: expected {tuple(target.shape)}, got {tuple(loaded.shape)}"
            )
        target.copy_(loaded)


class SharedExperts(DeepseekV2MLP):
    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.float().chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        activation = (gate * torch.sigmoid(gate) * up).to(x.dtype)
        return self.down_proj(activation)[0]


class DSV41MoE(nn.Module):
    """The fused EP operation owns dispatch, routed scale, and combine once."""

    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        parallel = get_parallel()
        if not (
            parallel.attn_tp_size == 1
            and parallel.attn_dp_size == parallel.moe_ep_size == parallel.tp_size
            and parallel.pp_size == parallel.attn_cp_size == parallel.attn_dcp_size == 1
        ):
            raise ValueError("V4.1 requires attention TP1 and DP=EP=world, PP1/CP1")
        if float(config.swiglu_limit) != 10.0:
            raise ValueError(
                "The selected EPIC routed activation requires swiglu_limit=10"
            )
        if not getattr(quant_config, "is_fp4_experts", False):
            raise ValueError("V4.1 fused EP expects the Flash MXFP4 expert checkpoint")
        self.is_hash = layer_id < getattr(config, "num_hash_layers", 0)
        self.gate = MoEGate(
            config, quant_config, add_prefix("gate", prefix), self.is_hash, True
        )
        self.experts = FusedEPExperts(config)
        self.topk = (
            HashTopK(
                config.num_experts_per_tok,
                config.n_routed_experts,
                0,
                config.vocab_size,
                layer_id=layer_id,
            )
            if self.is_hash
            else None
        )
        if not self.is_hash and self.gate.e_score_correction_bias is None:
            raise ValueError("V4.1 top-k routing requires correction bias")
        self.shared_experts = SharedExperts(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_experts", prefix),
            tp_rank=0,
            tp_size=1,
            swiglu_limit=float(config.swiglu_limit),
        )
        self.ep_op = _build_ep_op(
            parallel.moe_ep_size,
            config.num_experts_per_tok,
            bool(config.norm_topk_prob),
            config.scoring_func,
            float(config.routed_scaling_factor),
            self.is_hash,
            config.moe_intermediate_size,
        )
        self.register_buffer(
            "ep_rank",
            torch.tensor([[parallel.moe_ep_rank]], dtype=torch.int32),
            persistent=False,
        )

    def forward(self, x, input_ids, valid):
        expert = self.experts
        if not expert.prepared:
            raise RuntimeError("V4.1 expert weights have not been prepared")
        gating = self.gate(x).float().masked_fill(~valid[:, None], float("-inf"))
        route = (
            self.topk.tid2eid[input_ids.long()]
            if self.is_hash
            else self.gate.e_score_correction_bias
        )
        padded_x = F.pad(x, (0, -x.shape[-1] % 512))
        routed = self.ep_op(
            padded_x,
            expert.w13_weight,
            expert.w2_weight,
            expert.w13_scale,
            expert.w2_scale,
            gating,
            self.ep_rank,
            route,
        )[:, : x.shape[-1]]
        return (routed + self.shared_experts(x)).masked_fill(~valid[:, None], 0)

    def get_moe_weights(self):
        return [self.experts.w13_weight, self.experts.w2_weight]
