# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE + Shared Expert + All-Reduce decode kernel for TPU v7x."""

from sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_kernel import (
    fused_moe_decode,
    plan_weight_ring,
    vmem_bytes,
)
from sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op import (
    apply_fused_moe_decode,
    apply_routed_fused_moe_decode,
    get_fused_moe_decode_max_tokens,
    is_fused_moe_decode_enabled,
    prepare_fp8_fused_moe_decode,
    prepare_model_for_fused_moe_decode,
    prepare_moe_layer_for_fused_decode,
    should_use_fused_moe_decode,
    should_use_routed_fused_moe_decode,
)

__all__ = [
    "apply_fused_moe_decode",
    "apply_routed_fused_moe_decode",
    "fused_moe_decode",
    "get_fused_moe_decode_max_tokens",
    "is_fused_moe_decode_enabled",
    "plan_weight_ring",
    "prepare_fp8_fused_moe_decode",
    "prepare_model_for_fused_moe_decode",
    "prepare_moe_layer_for_fused_decode",
    "should_use_fused_moe_decode",
    "should_use_routed_fused_moe_decode",
    "vmem_bytes",
]
