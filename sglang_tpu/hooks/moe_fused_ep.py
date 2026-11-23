# SPDX-License-Identifier: Apache-2.0
"""Fused expert-parallel MoE preparation after the quant methods' weight pass.

Why: a fused EP kernel needs its own expert layout, converted from the gmm_v2
     expert tensors, which exist only after the quant methods'
     process_weights_after_loading. The platform's process_model_after_loading
     is only called from a model's post_load_weights (Kimi-K3 today), which
     runs inside load_weights, before DefaultModelLoader.postprocess_weights:
     too early.
How: registered only when SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL is set. AFTER
     DefaultModelLoader.postprocess_weights (after
     prepare_model_for_fused_moe_decode, before precompile),
     sglang_tpu.moe.fused_ep.prepare_model_for_fused_ep hands the model to
     every fused EP provider, which converts its eligible layers' experts and
     sets their mlp.dp_local_forward.
Upstream PR or issue: not filed.
Remove when: the loader offers a platform step after quant-method weight
     processing.
"""


def prepare_tpu_moe_fused_ep(result, model, target_device):
    if target_device.type == "tpu":
        from sglang_tpu.moe.fused_ep import prepare_model_for_fused_ep

        prepare_model_for_fused_ep(model)
    return result


def hook_specs():
    """Nothing unless the impl env is set. Never raises: load_plugins logs a
    plugin import error and drops every TPU hook; check_fused_ep_env rejects a
    bad value while the server args resolve."""
    from sglang.srt.plugins.hook_registry import HookType
    from sglang_tpu.moe.fused_ep import is_fused_ep_requested

    if not is_fused_ep_requested():
        return []
    return [
        (
            "sglang.srt.model_loader.loader.DefaultModelLoader.postprocess_weights",
            prepare_tpu_moe_fused_ep,
            HookType.AFTER,
        ),
    ]
