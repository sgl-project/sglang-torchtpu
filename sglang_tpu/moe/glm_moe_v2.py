# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM selected-route EP MoE preparation at loader finalization."""

import logging

import torch

logger = logging.getLogger(__name__)


def unsupported_reason(moe, parallel, exec_config):
    if moe.is_nextn:
        return "MTP draft keeps its attention-TP MoE backend"
    if not moe.allow_tpu_fused_moe_decode:
        return "the decoder does not expose the summed-by-compute FFN contract"
    if parallel.pp_size != 1 or parallel.moe_ep_size != parallel.tp_size:
        return "requires PP1 and EP=TP"
    if parallel.moe_tp_size != 1 or parallel.attn_dp_size != parallel.tp_size:
        return "requires fully data-parallel attention and unsharded experts"
    if not parallel.attn_dp_enabled:
        return "requires DP attention"
    if moe._enable_a2a_moe or exec_config.moe.enable_eplb:
        return "external all-to-all and EPLB are unsupported"
    if moe.num_fused_shared_experts or getattr(moe, "is_hash", False):
        return "requires ordinary selected routed ids and external shared weights"
    layer = moe.experts
    if getattr(layer, "weight_block_size", None) is not None:
        return "requires channelwise FP8 weights"
    if layer.w13_weight.dtype != torch.float8_e4m3fn:
        return "requires FP8 e4m3 weights"
    if getattr(layer.moe_runner_config, "swiglu_limit", None) is not None:
        return "the GLM candidate currently uses unclamped SiLU"
    if layer.moe_runner_config.activation != "silu":
        return "requires SiLU activation"
    if (
        layer.moe_runner_config.apply_router_weight_on_input
        or layer.moe_runner_config.no_combine
    ):
        return "requires output-weighted combined rows"
    return None


def _replace_data(param, value):
    # Preserve loader/shard attributes on the canonical Parameter. Packing is
    # owned by this backend, once per load, and holds no duplicate expert copy.
    old = param.data
    param.data = value
    if old.untyped_storage().data_ptr() != value.untyped_storage().data_ptr():
        old.untyped_storage().resize_(0)


def _prepare_shared(moe, parallel):
    """Replicate the existing shared MLP weights once, without requantizing.

    TP shards split the gate/up *separately*. Concatenating packed gate/up
    shards would interleave different ranks' gates with their up projections.
    Row-parallel down scales are already replicated output-channel vectors.
    """
    if not hasattr(moe, "shared_experts"):
        return False
    shared = moe.shared_experts
    gate_up, down = shared.gate_up_proj, shared.down_proj
    if (
        gate_up.weight.dtype != torch.float8_e4m3fn
        or down.weight.dtype != torch.float8_e4m3fn
    ):
        logger.warning(
            "GLM fused EP shared expert keeps external MLP: requires FP8 weights"
        )
        return False
    if (
        getattr(gate_up, "weight_block_size", None) is not None
        or getattr(down, "weight_block_size", None) is not None
    ):
        logger.warning(
            "GLM fused EP shared expert keeps external MLP: requires channel scales"
        )
        return False
    gate, up = gate_up.weight.data.chunk(2, dim=1)
    gate_scale, up_scale = gate_up.weight_scale_inv.data.reshape(-1).chunk(2)
    down_weight = down.weight.data
    if not moe._shared_expert_tp1:
        group = parallel.tp_group
        gate = group.all_gather(gate.contiguous(), dim=1)
        up = group.all_gather(up.contiguous(), dim=1)
        gate_scale = group.all_gather(gate_scale.contiguous(), dim=0)
        up_scale = group.all_gather(up_scale.contiguous(), dim=0)
        down_weight = group.all_gather(down_weight.contiguous(), dim=0)
    moe._glm_v2_sw13 = torch.stack((gate, up)).contiguous()
    moe._glm_v2_sw2 = down_weight.contiguous()
    moe._glm_v2_ss13 = torch.stack((gate_scale, up_scale)).float().contiguous()
    moe._glm_v2_ss2 = down.weight_scale_inv.data.reshape(-1).float().contiguous()
    return True


def prepare_model(model):
    from sglang.srt.layers.moe.utils import (
        get_moe_runner_backend,
        is_sbo_enabled,
        is_tbo_enabled,
    )
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang_tpu.envs import envs
    from sglang_tpu.moe.pallas.glm_moe_v2.op import (
        get_op,
        pack_gate_up,
        pack_gate_up_scale,
    )

    if not envs.SGLANG_TPU_USE_GLM_MOE_V2.get():
        return
    # This adapter targets the GLM DSA model; unrelated models retain their
    # established backend even if the process-level expert toggle is inherited.
    architectures = getattr(getattr(model, "config", None), "architectures", ()) or ()
    if (
        "GlmMoeDsaForCausalLMNextN" in architectures
        or type(model).__name__ == "GlmMoeDsaForCausalLMNextN"
    ):
        logger.info("GLM_MOE_V2_DRAFT_KEEP_BASELINE %s", type(model).__name__)
        return 0
    if "GlmMoeDsaForCausalLM" not in architectures:
        logger.info("GLM fused EP MoE v2: skipping %s", type(model).__name__)
        return
    from sglang_tpu.moe.glm_moe_v2_runtime import GlmSelectedEPMoE

    parallel = get_parallel()
    modules = [
        (name, moe)
        for name, moe in model.named_modules()
        if isinstance(moe, DeepseekV2MoE)
    ]
    draft = sum(bool(moe.is_nextn) for _, moe in modules)
    candidates = [(name, moe) for name, moe in modules if not moe.is_nextn]
    if candidates and not get_moe_runner_backend().is_pallas_tpu():
        raise ValueError("GLM MoE v2 requires --moe-runner-backend pallas_tpu")
    if candidates and (is_sbo_enabled() or is_tbo_enabled()):
        raise ValueError("GLM MoE v2 does not support decomposed SBO/TBO forwards")
    prepared = shared_count = max_post_drain_used = 0
    for name, moe in candidates:
        if isinstance(moe, GlmSelectedEPMoE):
            prepared += 1
            shared_count += int(moe._glm_v2_shared)
            continue
        reason = unsupported_reason(moe, parallel, get_exec())
        if reason:
            logger.warning("GLM fused EP MoE v2 skips %s: %s", name, reason)
            continue
        layer = moe.experts
        w13 = pack_gate_up(layer.w13_weight.data)
        s13 = pack_gate_up_scale(layer.w13_weight_scale.data)
        _replace_data(layer.w13_weight, w13)
        _replace_data(layer.w13_weight_scale, s13)
        layer.w13_weight_scale_inv = layer.w13_weight_scale
        moe.register_buffer(
            "_glm_v2_rank",
            torch.tensor(
                [parallel.moe_ep_rank],
                device=layer.w13_weight.device,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        moe._glm_v2_ep_size = parallel.moe_ep_size
        moe._glm_v2_shared = False
        # Stable placeholders keep the custom op schema independent of fusion.
        for attr in ("sw13", "sw2", "ss13", "ss2"):
            moe.register_buffer(
                f"_glm_v2_{attr}",
                torch.empty(0, device=layer.w13_weight.device, dtype=torch.float32),
                persistent=False,
            )
        moe._glm_v2_routed_op = get_op(parallel.moe_ep_size, False)
        moe._glm_v2_op = moe._glm_v2_routed_op
        moe.__class__ = GlmSelectedEPMoE
        if envs.SGLANG_TPU_GLM_MOE_V2_SHARED.get():
            moe._glm_v2_shared = _prepare_shared(moe, parallel)
            if moe._glm_v2_shared:
                moe._glm_v2_op = get_op(parallel.moe_ep_size, True)
        # This full-model pass is outside the loader's per-quant-module drain.
        # Materialize this layer's pack/gathers before the next layer retains
        # another old layout through its deferred DeviceBufferRef readers.
        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info(layer.w13_weight.device)
        max_post_drain_used = max(max_post_drain_used, int(total - free))
        prepared += 1
        shared_count += int(moe._glm_v2_shared)
        logger.info(
            "GLM fused EP MoE v2 prepared %s: EP%d, shared=%s",
            name,
            parallel.moe_ep_size,
            moe._glm_v2_shared,
        )
    summary = dict(
        expected=len(candidates),
        prepared=prepared,
        shared=shared_count,
        draft_skipped=draft,
        max_observed_post_drain_used_bytes=max_post_drain_used,
    )
    model._glm_v2_preparation_summary = summary
    logger.info("GLM_MOE_V2_PREPARED %s", summary)
    if prepared != len(candidates):
        raise RuntimeError(f"GLM MoE v2 did not bind every target MoE layer: {summary}")
    return prepared
