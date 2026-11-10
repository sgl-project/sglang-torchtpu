# SPDX-License-Identifier: Apache-2.0
"""Bind TPU fused_moe_decode helpers onto Qwen2MoeSparseMoeBlock.

Why: Qwen2MoeSparseMoeBlock lives under python/sglang/srt/models/, where direct
     sglang_tpu imports or is_tpu checks add plugin debt, and
     prepare_moe_layer_for_fused_decode runs lazily on the first forward pass
     rather than at startup.
How: AFTER Qwen2MoeSparseMoeBlock.__init__ binds _tpu_fused_moe_decode_should_use
     and _tpu_fused_moe_decode_apply onto the module instance. Imports happen
     inside the hook so CPU hosts without JAX can register hooks cleanly.
Upstream PR or issue: Qwen2MoeSparseMoeBlock TPU backend seam (plugin debt
     avoidance alongside DeepseekV2MoE).
Remove when: Qwen2MoeSparseMoeBlock dispatches through an upstream FusedMoE or
     MoE backend registry seam.
"""


def bind_qwen_fused_moe_decode(result, moe_layer, *args, **kwargs):
    import importlib.util
    import sys

    mod_name = "sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op"
    if mod_name not in sys.modules and (
        importlib.util.find_spec("torch_tpu") is None
        or importlib.util.find_spec("jax") is None
    ):
        return result
    from sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op import (
        apply_fused_moe_decode,
        should_use_fused_moe_decode,
    )

    moe_layer._tpu_fused_moe_decode_should_use = should_use_fused_moe_decode
    moe_layer._tpu_fused_moe_decode_apply = apply_fused_moe_decode
    return result
