# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 TPU KV pool, memory-budget sizing, compressor, and config hooks.

build_tpu_dsv4_kv_pool, get_tpu_dsv4_bytes_per_swa_token,
get_tpu_dsv4_bytes_per_full_token, get_tpu_dsv4_c128_state_fixed_bytes,
get_tpu_dsv4_fixed_c4_state_bytes
  Why: KVCacheConfigurator._build_dsv4_kv_pool and DSV4PoolConfigurator hardcode
       the GPU/NPU DeepSeekV4TokenToKVPool class and FlashMLA/PA_ND byte-per-token
       and compressor-state formulas. On TPU, DeepSeek-V4 uses 4D page-major
       uint8 Pallas buffers (DSV4TPUTokenToKVPool) with 1024 B/token SWA/HCA,
       640 B/token CSA, 256 B/token Indexer K, and uniform 32-page per-request
       ring + 2048-page scratch compressor state pools.
  How: AROUND KVCacheConfigurator._build_dsv4_kv_pool binds
       DeepSeekV4TokenToKVPool to DSV4TPUTokenToKVPool for the call and registers
       the per-request C4/C128 state-shift reset callback on ReqToTokenPool.
       AROUND DSV4PoolConfigurator._get_bytes_per_swa_token,
       _get_bytes_per_full_token, _get_c128_state_fixed_bytes, and
       _fixed_c4_state_bytes delegate the byte budget to DSV4TPUTokenToKVPool.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1119.
  Remove when: KVCacheConfigurator and DSV4PoolConfigurator query
       current_platform for the DSV4 KV pool class and its byte-sizing methods.

skip_tpu_dsv4_ape_hotfix, forward_tpu_dsv4_compressor
  Why: Compressor._apply_ape_hotfix permutes the overlap APE table for the CUDA
       Triton compressor layout, whereas the TPU Pallas compressor kernels read
       the unpermuted (ratio, head_dim) APE table directly. Compressor.forward_native
       assumes a CUDA kv_score_buffer and calls attn_backend.forward_compress with
       GPU keyword arguments instead of (compressor, x, forward_batch).
  How: AROUND Compressor._apply_ape_hotfix marks ape_converted = True without
       permuting ape. AROUND Compressor.forward_native handles idle batches and
       delegates to backend.forward_compress(compressor, x, forward_batch).
  Upstream PR or issue: sgl-project/sglang-torchtpu#1119.
  Remove when: Compressor delegates APE layout and forward_compress dispatch
       through a platform-agnostic CompressorBackendMixin contract.

forward_tpu_dsv4_hash_topk_torch, forward_tpu_dsv4_hash_topk
  Why: HashTopK._forward_torch gathers router scores with int32 indices, whereas
       torch_tpu (aten.gather) requires int64 indices, and returns router-dtype
       weights unless _is_hip or _is_npu. HashTopK.forward defaults to the CUDA
       JIT hash_topk kernel when SGLANG_OPT_USE_FUSED_HASH_TOPK is enabled and
       masks padded tokens with in-place dynamic boolean indexing
       (_mask_topk_ids_padded_region), which breaks torch.compile(fullgraph=True)
       on TPU and does not zero padded topk_weights.
  How: AROUND HashTopK._forward_torch casts gather indices to int64 and returns
       float32 topk_weights. AROUND HashTopK.forward overrides
       SGLANG_OPT_USE_FUSED_HASH_TOPK to False, runs original with
       num_token_non_padded=None to bypass in-place boolean indexing, and masks
       padded topk_ids (-1) and topk_weights (0.0) out-of-place via torch.where.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1119.
  Remove when: HashTopK uses int64 gather indices, float32 topk_weights, and
       static-shape torch.where padded-token masking upstream.

tpu_dsv4_silu_and_mul_clamp
  Why: sglang.kernels.ops.moe.dsv4.silu_and_mul_clamp dispatches to sgl_kernel
       on XPU or a CUDA JIT kernel otherwise, which cannot compile or run on TPU.
  How: AROUND sglang.kernels.ops.moe.dsv4.silu_and_mul_clamp and
       sglang.srt.models.deepseek_v2.silu_and_mul_clamp computes clamped SwiGLU
       in float32 PyTorch ops and copies the result into output.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1119.
  Remove when: silu_and_mul_clamp is converted to a BaseFusedOp with a
       forward_native fallback.

tpu_dsv4_current_mhc_post_fusion
  Why: sglang.srt.layers.moe.mhc_post_fusion.current_mhc_post_fusion reads a
       ContextVar that causes a Dynamo graph break under torch.compile(fullgraph=True)
       on TPU, and CUDA/HIP stream-overlap MHC post-fusion is unused on TPU.
  How: AROUND sglang.srt.layers.moe.mhc_post_fusion.current_mhc_post_fusion
       returns None when torch.compiler.is_compiling().
  Upstream PR or issue: sgl-project/sglang-torchtpu#1119.
  Remove when: current_mhc_post_fusion is marked constant or skipped when
       compiling without MHC post-fusion enabled.

disable_tpu_dsv4_wo_a_fp8_gemm, get_tpu_dsv4_kernel_num_heads,
skip_tpu_dsv4_prewarm_mhc_kernels
  Why: deepseek_v4.wo_a_fp8_gemm_enabled reads module-import-time _FP8_WO_A_GEMM
       for a DeepGEMM/aiter FP8 absorb GEMM not supported on TPU;
       MqaAttentionBase._kernel_num_heads pads n_local_heads <= 64 to 64 for
       CUDA FlashMLA when attn_tp_size > 1, whereas TPU Pallas attention kernels
       take unpadded n_local_heads; DeepseekV4ForCausalLM._prewarm_mhc_kernels
       prewarms CUDA TileLang MHC kernels.
  How: AROUND wo_a_fp8_gemm_enabled (in deepseek_v4 and deepseek_v4_nextn)
       returns False; AROUND MqaAttentionBase._kernel_num_heads returns
       attn.n_local_heads; AROUND DeepseekV4ForCausalLM._prewarm_mhc_kernels
       marks _mhc_prewarmed_at_load = True and returns.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1119.
  Remove when: wo_a_fp8_gemm_enabled, _kernel_num_heads, and _prewarm_mhc_kernels
       query platform capabilities or attention backend properties.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch


def build_tpu_dsv4_kv_pool(original, kvc, *args, **kwargs):
    import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    with patch.object(kvc_mod, "DeepSeekV4TokenToKVPool", DSV4TPUTokenToKVPool):
        pool = original(kvc, *args, **kwargs)
    req_to_token_pool = kwargs["req_to_token_pool"]
    if not kvc.is_draft_worker and not pool._unified_kv:
        req_to_token_pool.register_on_alloc_rows(pool.clear_c4_req_states)
    return pool


def get_tpu_dsv4_bytes_per_swa_token(original, configurator) -> float:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    return DSV4TPUTokenToKVPool.bytes_per_swa_token(
        num_layers_total=configurator.num_layers_total
    )


def get_tpu_dsv4_bytes_per_full_token(original, configurator) -> float:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    swa_ratio = (
        0
        if configurator._unified or configurator.swa_cap_tokens is not None
        else configurator.swa_ratio
    )
    return DSV4TPUTokenToKVPool.bytes_per_full_token(
        swa_ratio=swa_ratio,
        bytes_per_swa_token=configurator.bytes_per_swa_token,
        c4_shrink_factor=configurator.c4_shrink_factor,
        num_c4_layers=configurator.num_layers(4),
        num_c128_layers=configurator.num_layers(128),
    )


def get_tpu_dsv4_c128_state_fixed_bytes(
    original, configurator, max_running_requests: int
) -> int:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    return DSV4TPUTokenToKVPool.c128_state_fixed_bytes(
        num_req_slots=configurator._get_num_req_slots(max_running_requests),
        attn_head_dim=configurator.attn_head_dim,
        num_c128_layers=configurator.num_layers(128),
    )


def get_tpu_dsv4_fixed_c4_state_bytes(
    original, configurator, max_running_requests: int
) -> int:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    return DSV4TPUTokenToKVPool.c4_state_fixed_bytes(
        num_req_slots=configurator._get_num_req_slots(max_running_requests),
        attn_head_dim=configurator.attn_head_dim,
        indexer_head_dim=configurator.indexer_head_dim,
        num_c4_layers=configurator.num_layers(4),
    )


def skip_tpu_dsv4_ape_hotfix(original, compressor) -> None:
    del original
    compressor.ape_converted = True


def forward_tpu_dsv4_compressor(
    original, compressor, x, forward_batch, attn_backend=None
):
    del original
    if forward_batch.forward_mode.is_idle():
        assert x.shape[0] == 0
        return x.new_empty(0, compressor.head_dim)
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = attn_backend if attn_backend is not None else get_attn_backend()
    return backend.forward_compress(compressor, x, forward_batch)


def forward_tpu_dsv4_hash_topk_torch(
    original, topk_layer, router_logits: torch.Tensor, input_ids: torch.Tensor
):
    if topk_layer.num_fused_shared_experts == 1:
        topk_weights, topk_ids = original(topk_layer, router_logits, input_ids)
        return topk_weights.to(torch.float32), topk_ids
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

    if topk_layer.score_func == "softmax":
        scores = router_logits.softmax(dim=-1)
    elif topk_layer.score_func == "sigmoid":
        scores = router_logits.sigmoid()
    else:
        scores = torch.nn.functional.softplus(router_logits).sqrt()
    topk_ids = topk_layer.tid2eid[input_ids].to(torch.int32)
    topk_weights = scores.gather(1, topk_ids.to(torch.int64))
    if topk_layer.score_func != "softmax":
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + _RENORMALIZE_SUM_EPSILON
        )
    return topk_weights.to(torch.float32), topk_ids


def forward_tpu_dsv4_hash_topk(
    original,
    topk_layer,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    num_token_non_padded: torch.Tensor | None = None,
    expert_location_dispatch_info: Any = None,
):
    from sglang.srt.environ import envs

    if envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
        with envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False):
            topk_output = original(
                topk_layer,
                hidden_states,
                router_logits,
                input_ids,
                None,
                expert_location_dispatch_info,
            )
    else:
        topk_output = original(
            topk_layer,
            hidden_states,
            router_logits,
            input_ids,
            None,
            expert_location_dispatch_info,
        )
    if num_token_non_padded is not None:
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        valid = (
            torch.arange(topk_ids.shape[0], device=topk_ids.device).unsqueeze(1)
            < num_token_non_padded
        )
        topk_ids = torch.where(valid, topk_ids, torch.full_like(topk_ids, -1))
        topk_weights = torch.where(
            valid, topk_weights, torch.zeros_like(topk_weights)
        )
        topk_output = topk_output._replace(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
    return topk_output


def tpu_dsv4_silu_and_mul_clamp(
    original, input: torch.Tensor, output: torch.Tensor, swiglu_limit: float
) -> None:
    del original
    limit = float(swiglu_limit)
    gate, up = input.float().chunk(2, dim=-1)
    res = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(
        min=-limit, max=limit
    )
    output.copy_(res.to(output.dtype))


def tpu_dsv4_current_mhc_post_fusion(original):
    if torch.compiler.is_compiling():
        return None
    return original()


def disable_tpu_dsv4_wo_a_fp8_gemm(original, quant_config) -> bool:
    del original, quant_config
    return False


def get_tpu_dsv4_kernel_num_heads(original, attn, num_tokens: int) -> int:
    del original, num_tokens
    return attn.n_local_heads


def skip_tpu_dsv4_prewarm_mhc_kernels(original, model) -> None:
    del original
    model._mhc_prewarmed_at_load = True


def _tpu_deepseek_v4_overrides(server_args: Any, hf_config: Any) -> dict:
    """Declare TPU DeepSeek-V4 ServerArgs defaults via register_model_override."""
    del hf_config
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(server_args)
    if cfg.device != "tpu":
        return {}
    return {
        "prefill_attention_backend": "dsv4",
        "decode_attention_backend": "dsv4",
        "disable_shared_experts_fusion": True,
    }


def _tpu_deepseek_v4_env_adjustments(server_args: Any, previous) -> None:
    """Apply TPU DeepSeek-V4 environment flags after handle_model_specific_adjustments."""
    previous(server_args)
    from sglang.srt.arg_groups.model_override_base import (
        model_config_of,
        resolving_view,
    )
    from sglang.srt.connector import ConnectorType
    from sglang.srt.environ import envs
    from sglang.srt.utils.common import parse_connector_type

    cfg = resolving_view(server_args)
    if cfg.device != "tpu":
        return
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return
    hf_config = model_config_of(server_args).hf_config
    if (hf_config.architectures or [None])[0] != "DeepseekV4ForCausalLM":
        return
    envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
    envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
    envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
    envs.SGLANG_OPT_USE_TILELANG_MHC_POST.set(False)
    envs.SGLANG_OPT_FUSE_MHC_POST_PRE.set(False)
    envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.set(False)
    envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.set(False)
    envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.set(True)


def register_tpu_dsv4_resolution_hooks() -> None:
    """Idempotently register TPU DeepSeek-V4 model overrides and resolution hooks."""
    import sglang.srt.arg_groups.model_overrides  # noqa: F401
    from sglang.srt.arg_groups import model_override_base, resolution_hooks

    arch = "DeepseekV4ForCausalLM"
    fns = model_override_base._MODEL_OVERRIDE_FNS.setdefault(arch, [])
    if _tpu_deepseek_v4_overrides not in fns:
        fns.append(_tpu_deepseek_v4_overrides)

    step = "handle_model_specific_adjustments"
    hooks = resolution_hooks._HOOKS.setdefault(step, [])
    if _tpu_deepseek_v4_env_adjustments not in hooks:
        hooks.append(_tpu_deepseek_v4_env_adjustments)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    kvc = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    cfg = "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator"
    comp = "sglang.srt.layers.attention.dsv4.compressor.Compressor"
    htk = "sglang.srt.layers.moe.hash_topk.HashTopK"
    dsv4 = "sglang.srt.models.deepseek_v4"
    return [
        (
            f"{kvc}._build_dsv4_kv_pool",
            build_tpu_dsv4_kv_pool,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_bytes_per_swa_token",
            get_tpu_dsv4_bytes_per_swa_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_bytes_per_full_token",
            get_tpu_dsv4_bytes_per_full_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_c128_state_fixed_bytes",
            get_tpu_dsv4_c128_state_fixed_bytes,
            HookType.AROUND,
        ),
        (
            f"{cfg}._fixed_c4_state_bytes",
            get_tpu_dsv4_fixed_c4_state_bytes,
            HookType.AROUND,
        ),
        (
            f"{comp}._apply_ape_hotfix",
            skip_tpu_dsv4_ape_hotfix,
            HookType.AROUND,
        ),
        (
            f"{comp}.forward_native",
            forward_tpu_dsv4_compressor,
            HookType.AROUND,
        ),
        (
            f"{htk}._forward_torch",
            forward_tpu_dsv4_hash_topk_torch,
            HookType.AROUND,
        ),
        (
            f"{htk}.forward",
            forward_tpu_dsv4_hash_topk,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.moe.dsv4.silu_and_mul_clamp",
            tpu_dsv4_silu_and_mul_clamp,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v2.silu_and_mul_clamp",
            tpu_dsv4_silu_and_mul_clamp,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.moe.mhc_post_fusion.current_mhc_post_fusion",
            tpu_dsv4_current_mhc_post_fusion,
            HookType.AROUND,
        ),
        (
            f"{dsv4}.wo_a_fp8_gemm_enabled",
            disable_tpu_dsv4_wo_a_fp8_gemm,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4_nextn.wo_a_fp8_gemm_enabled",
            disable_tpu_dsv4_wo_a_fp8_gemm,
            HookType.AROUND,
        ),
        (
            f"{dsv4}.MqaAttentionBase._kernel_num_heads",
            get_tpu_dsv4_kernel_num_heads,
            HookType.AROUND,
        ),
        (
            f"{dsv4}.DeepseekV4ForCausalLM._prewarm_mhc_kernels",
            skip_tpu_dsv4_prewarm_mhc_kernels,
            HookType.AROUND,
        ),
    ]
