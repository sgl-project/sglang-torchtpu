# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash and Kimi-K3 KDA layers: kernel weights bound after load.

bind_glm5_next_kda_weights
  Why: Glm5NextLinearAttention binds a squeezed view of its [dim, 1, K] conv
       weight, which the TPU backend unsqueezes again, and passes conv, dt_bias
       and A_log as three small operands that XLA stages one copy each, every
       KDA layer, every forward. Kimi-K3 binds the parameter itself and one
       packed f32 operand post-load (kimi_k3._bind_tpu_attn_weights).
  How: AFTER Glm5NextForConditionalGeneration.load_weights, each KDA layer's
       attention takes the [dim, 1, K] parameter and the packed weights
       (kda_layout.pack_kda_weights), as Kimi-K3 does. The conv and batched
       f/g up-projection weights, built through views at load, are re-homed
       through the host so torch_tpu does not relay them out on every call.
       Weight updates that go through load_weights rerun the binding; a
       path that writes these parameters in place would leave the packed
       copy stale.
  Upstream PR or issue: none yet (model-side binding, as in kimi_k3.py).
  Remove when: glm5_next.py binds its TPU kernel weights post-load.

rehome_kimi_k3_kda_weights
  Why: kimi_k3.KimiK3DeltaAttention._bind_tpu_attn_weights binds the conv
       weight (an unsqueeze view made at load), dt_bias, A_log and an A_log
       copy cloned on device. torch_tpu 0.2 relays such device-built tensors
       out with an execute_as_strided program before every compiled call
       (about 0.1 ms of host dispatch each); at TP32 that doubled Kimi-K3's
       decode TPOT (7.86 -> 15.66 ms).
  How: AFTER KimiK3LinearForCausalLM.post_load_weights, each KDA layer's
       conv weight, dt_bias, A_log and bound A_log copy are re-homed through
       the host, as for GLM-5.3-Flash above, and the packed kernel operand is
       rebuilt from them. The parameters keep their identity, so the
       attention's references to them stay valid.
  Upstream PR or issue: none yet (model-side binding in kimi_k3.py).
  Remove when: kimi_k3.py re-homes these weights when it binds them, or
       torch_tpu stops relaying out device-built inputs.
"""

from __future__ import annotations

from typing import Any


def bind_glm5_next_kda_weights(result: None, model: Any, *args, **kwargs) -> None:
    from sglang.srt.models.glm5_next import Glm5NextLinearAttention
    from sglang_tpu.kda_layout import pack_kda_weights

    for module in model.modules():
        if not isinstance(module, Glm5NextLinearAttention):
            continue
        attn = module.attn
        if attn.bias is not None:
            continue  # the packed operand carries no conv bias
        # Weights reshaped or filled through views after load: re-home them
        # through the host, or torch_tpu relays them out at every compiled call.
        for name in ("qkv_conv1d", "fused_fg_b_proj"):
            proj = getattr(module, name, None)
            if proj is not None:
                proj.weight.data = proj.weight.data.cpu().to(proj.weight.device)
        attn.conv_weights = module.qkv_conv1d.weight
        attn.packed_kernel_weights = pack_kda_weights(
            module.qkv_conv1d.weight,
            module.dt_bias,
            module.A_log,
            attn.num_v_heads,
            attn.head_k_dim,
        ).to(module.qkv_conv1d.weight.device)
    return result


def _rehome(tensor) -> None:
    tensor.data = tensor.data.cpu().to(tensor.device)


def rehome_kimi_k3_kda_weights(result: None, causal_lm: Any, *args, **kwargs) -> None:
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention

    for layer in causal_lm.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        attn_layer = layer.self_attn
        if not isinstance(attn_layer, KimiK3DeltaAttention):
            continue
        attn = attn_layer.attn
        if getattr(attn, "packed_kernel_weights", None) is None:
            continue  # not bound for the TPU kernel
        for tensor in (
            attn_layer.qkv_conv1d.weight,
            attn_layer.dt_bias,
            attn_layer.A_log,
            attn.A_log,
        ):
            _rehome(tensor)
        attn.packed_kernel_weights = attn_layer._pack_tpu_kda_weights()
    return result
