# SPDX-License-Identifier: Apache-2.0
"""MoE Bridge & Combine/Unpool Pallas kernels (Kernel B1, Kernel B2, Kernel D) for Muse Spark."""

from .kernel_b1 import (
    pallas_postattn_res_prenorm_router_topk,
    pallas_postattn_res_prenorm_router_topk_fp8,
)
from .kernel_b2 import (
    pallas_wpool_prenorm_int8_quant_fp8_nstream,
    pallas_wpool_prenorm_int8_quant_fp8_rowtile,
    pallas_wpool_prenorm_int8_quant_nstream,
    pallas_wpool_prenorm_int8_quant_rowtile,
)
from .kernel_d import (
    _tc_combine_bf16_4chunk,
    pallas_combine_unpool_residual_nextnorm_4slab,
    pallas_combine_unpool_residual_nextnorm_nstream,
)
from .pipeline import (
    baseline_unfused_sp_moe_layer,
    compute_block_scatter_routing,
    default_b1_b2_replicated_bf16,
    default_d_replicated_bf16,
    fused_bridge_b1_b2_ag_routing,
    fused_d_standalone,
    fused_moe_full_bridge_and_ffn,
    reference_moe_bridge_and_ffn,
)

__all__ = [
    "pallas_postattn_res_prenorm_router_topk",
    "pallas_postattn_res_prenorm_router_topk_fp8",
    "pallas_wpool_prenorm_int8_quant_nstream",
    "pallas_wpool_prenorm_int8_quant_rowtile",
    "pallas_wpool_prenorm_int8_quant_fp8_nstream",
    "pallas_wpool_prenorm_int8_quant_fp8_rowtile",
    "_tc_combine_bf16_4chunk",
    "pallas_combine_unpool_residual_nextnorm_4slab",
    "pallas_combine_unpool_residual_nextnorm_nstream",
    "compute_block_scatter_routing",
    "reference_moe_bridge_and_ffn",
    "baseline_unfused_sp_moe_layer",
    "default_b1_b2_replicated_bf16",
    "default_d_replicated_bf16",
    "fused_d_standalone",
    "fused_bridge_b1_b2_ag_routing",
    "fused_moe_full_bridge_and_ffn",
]
