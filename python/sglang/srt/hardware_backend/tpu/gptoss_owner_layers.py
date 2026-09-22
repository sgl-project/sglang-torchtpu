# SPDX-License-Identifier: Apache-2.0
"""Experimental GPT-OSS decode layers with token-owned residuals and norms."""

import torch


def prepare_owner_model(model):
    """Validate the layout and retain the logical rank as device input data."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel, get_spec

    parallel = get_parallel()
    attn, tp = parallel.attn_tp_group, parallel.tp_group
    if (
        parallel.pp_group.world_size != 1
        or parallel.attn_dcp_size != 1
        or parallel.attn_cp_group.world_size != 1
        or parallel.dwdp_size != 1
        or parallel.moe_ep_size != tp.world_size
        or parallel.moe_tp_size != 1
        or parallel.moe_dp_group.world_size != 1
        or attn.world_size not in (4, 8)
    ):
        raise NotImplementedError(
            "Token-owner layers require attention TP4/8, EP=TP, MoE-TP1/DP1 and PP1/DCP1/CP1/DWDP1."
        )
    if get_spec().speculative_algorithm is not None:
        raise NotImplementedError(
            "Token-owner layers do not capture speculative features."
        )
    start = (tp.rank_in_group // attn.world_size) * attn.world_size
    if list(attn.ranks) != list(tp.ranks[start : start + attn.world_size]):
        raise ValueError(
            "Token-owner attention islands must be contiguous in EP order."
        )
    config = model.layers[model.start_layer].config
    if config.hidden_size != 2880 or config.num_local_experts != 128:
        raise ValueError("Token-owner prototype requires the GPT-OSS-120B dimensions.")
    model._tpu_owner_island_size = attn.world_size
    model._tpu_owner_fused_attention = (
        envs.SGLANG_TPU_GPTOSS_FUSED_ATTN_BOUNDARIES.get()
    )
    model._tpu_owner_baseline_moe = envs.SGLANG_TPU_GPTOSS_OWNER_BASELINE_MOE.get()
    if model._tpu_owner_baseline_moe:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher

        for layer in model.layers:
            experts = layer.mlp.experts
            if (
                experts.reduce_results
                or experts.handles_moe_reduction
                or experts.moe_runner_config.no_combine
                or not isinstance(experts.dispatcher, StandardDispatcher)
            ):
                raise ValueError(
                    "Owner baseline MoE requires unreduced local expert sums."
                )
    model.register_buffer(
        "_tpu_owner_rank",
        torch.tensor(
            [attn.rank_in_group],
            dtype=torch.int64,
            device=model.layers[model.start_layer].input_layernorm.weight.device,
        ),
        persistent=False,
    )


def prepare_owner_attention_weights(model):
    """Prepare fused projections only after original checkpoint weights load."""
    if not model._tpu_owner_fused_attention:
        return
    from sglang.srt.hardware_backend.tpu.attention_ag_qkv import (
        prepare_attention_ag_qkv,
    )
    from sglang.srt.hardware_backend.tpu.attention_o_rs import prepare_attention_o_rs

    for layer in model.layers:
        prepare_attention_ag_qkv(layer.self_attn.qkv_proj)
        prepare_attention_o_rs(layer.self_attn.o_proj)


def can_use_owner_layers(model, forward_batch, hidden_states):
    """Select whole decode graphs whose physical rows divide across the island."""
    return (
        forward_batch.forward_mode.is_decode()
        and forward_batch.spec_info is None
        and not model.layers_to_capture
        and hidden_states.shape[0] > 0
        and hidden_states.shape[0] % model._tpu_owner_island_size == 0
        and (
            not model._tpu_owner_fused_attention
            or (hidden_states.shape[0] // model._tpu_owner_island_size) % 8 == 0
        )
    )


def select_owner_rows(model, hidden_states):
    """Select contiguous owner rows without rank-dependent graph structure."""
    owners = model._tpu_owner_island_size
    rows, width = hidden_states.shape
    return (
        hidden_states.view(owners, rows // owners, width)
        .index_select(0, model._tpu_owner_rank)
        .reshape(rows // owners, width)
        .contiguous()
    )


def forward_owner_layer(
    layer,
    positions,
    hidden_states,
    residual,
    forward_batch,
    *,
    fused_attention,
    baseline_moe=False,
):
    """Decode one layer, preserving owner rows at both residual/norm boundaries."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    island = parallel.attn_tp_group
    owner_rows = hidden_states.shape[0]
    if residual is None:
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
    else:
        normalized, residual = layer.input_layernorm(hidden_states, residual)
    attention = layer.self_attn
    if fused_attention:
        from sglang.srt.hardware_backend.tpu.attention_ag_qkv import attention_ag_qkv

        qkv = attention_ag_qkv(normalized, attention.qkv_proj, chunks=4)
    else:
        island_input = island.all_gather(normalized, dim=0)
        qkv, _ = attention.qkv_proj(island_input)
    q, k, v = qkv.split(
        [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
    )
    q, k = attention.rotary_emb(positions, q, k)
    attention_output = attention.attn(
        q, k, v, forward_batch, sinks=attention.sinks, save_kv_cache=True
    )
    if fused_attention:
        from sglang.srt.hardware_backend.tpu.attention_o_rs import attention_o_rs

        projected = attention_o_rs(attention_output, attention.o_proj, chunks=4)
    else:
        projected, _ = attention.o_proj(attention_output)
        projected = island.reduce_scatter_along_dim(projected, dim=0).contiguous()
    if projected.shape != residual.shape:
        raise ValueError("Attention output and residual must share owner rows.")
    normalized, residual = layer.post_attention_layernorm(projected, residual)
    if not baseline_moe and getattr(layer.mlp, "_tpu_native_packed_moe", False):
        from sglang.srt.hardware_backend.tpu.gptoss_packed_mxfp4 import (
            forward_owner_native_mxfp4_mlp,
        )

        result = forward_owner_native_mxfp4_mlp(normalized, layer.mlp)
    else:
        global_input = parallel.tp_group.all_gather(normalized, dim=0)
        if baseline_moe:
            router_logits, _ = layer.mlp.router(global_input)
            topk_output = layer.mlp.topk(global_input, router_logits)
            partial = layer.mlp.experts(global_input, topk_output)
            result = parallel.tp_group.reduce_scatter_along_dim(
                partial, dim=0
            ).contiguous()
        else:
            result = layer.mlp.forward_normal(
                global_input, token_owned=True, return_owner=True
            )
    if result.shape != (owner_rows, layer.hidden_size):
        raise ValueError("MoE direct output must preserve global token-owner rows.")
    return result, residual
