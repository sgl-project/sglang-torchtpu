# SPDX-License-Identifier: Apache-2.0
"""MoE experts: the engine's fused MoE op with its combine in slot order."""

import torch

from sglang_tpu.envs import envs
from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op
from sglang_tpu.rl.true_on_policy.common import refuse_flags


def refuse_moe_rounding_flags():
    """Refuse the MoE flags that change rounding."""
    flags = (
        "SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL",
        "SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE",
    )
    refuse_flags(flags, "MoE")


_fused_moe = _build_fused_moe_op()


def moe_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
):
    """The routed experts of one MoE block, [M, H] -> [M, H].

    w1 [E, H, 2 * I] (gate || up) and w2 [E, I, H], every expert; topk_weights [M, k] in the activation dtype and
    topk_ids [M, k] int32, in the router's slot order. Forward only.
    """
    refuse_moe_rounding_flags()
    with envs.SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL.override(False):
        return _fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            topk=topk_ids.shape[1],
            activation=activation,
            global_num_experts=w1.shape[0],
            use_sc_gather=0,
            permute_dense_threshold=0,
            gather_permute_with_sc_combine=0,
        )
