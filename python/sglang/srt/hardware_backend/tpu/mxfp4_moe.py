"""TPU MXFP4 MoE helpers (GPT-OSS).

Mirrors the FP8 backend-dispatch shape in ``hardware_backend/tpu/fp8_moe.py``:
GPT-OSS MXFP4 expert weights are formatted once after loading (by
``mxfp4_tpu_util.prepare_tpu_mxfp4_moe_weights``), then the runtime calls the
shared Pallas ``gmm_v2`` fused grouped GEMM via ``fused_moe_call`` here — the
same backend the unquant and FP8 MoE paths use. Keeping this glue out of the
generic ``Mxfp4MoEMethod`` follows the PR #75 split that moved TPU-specific
quant logic out of the quantization methods.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts


def apply_tpu_mxfp4_moe(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    apply_router_weight_on_input: bool,
):
    """Run TPU GPT-OSS MXFP4 MoE through the shared Pallas fused grouped-GEMM.

    Weights arrive in the load-time-processed layout (see
    ``mxfp4_tpu_util.prepare_tpu_mxfp4_moe_weights``). ``fused_moe_call`` dispatches on
    weight dtype at JAX trace time, so the Qwen3 (BF16) trace and the GPT-OSS
    trace are independent jaxprs.
    """
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported on TPU yet."
        )

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    # Under EP, StandardDispatcher marks non-local experts with id -1. The
    # Pallas fused-MoE kernel requires non-negative local expert ids, so route
    # those slots to expert 0 and zero their routing weight — otherwise the
    # kernel indexes experts out of bounds and emits NaN. Mirrors the FP8 MoE
    # path (fp8_moe.apply_tpu_fp8_moe).
    topk_weights, topk_ids = mask_tpu_moe_non_local_experts(topk_weights, topk_ids)

    output = fused_moe_call(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        w1_bias=layer.w13_weight_bias,
        w2_bias=layer.w2_weight_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=int(topk_ids.shape[-1]),
        activation=activation,
    )
    return StandardCombineInput(hidden_states=output)
