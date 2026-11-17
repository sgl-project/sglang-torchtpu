# SPDX-License-Identifier: Apache-2.0
"""GLM MoE runtime adapter; installed after loader quant postprocessing."""

import torch

from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_parallel


class GlmSelectedEPMoE(DeepseekV2MoE):
    def use_tpu_fused_moe_decode(
        self, hidden_states, forward_batch, skip_shared_experts=False
    ):
        # This is the existing FFN-boundary contract, for every candidate shape:
        # the returned rows already include the complete EP contribution.
        return True

    def forward(
        self,
        hidden_states,
        forward_batch=None,
        gemm_output_zero_allocator=None,
        input_ids=None,
        input_ids_global=None,
        skip_shared_experts=False,
    ):
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        logits = self.gate(hidden_states, gemm_output_zero_allocator)
        selected = self.topk(
            hidden_states,
            logits,
            num_token_non_padded=(
                forward_batch.moe_num_token_non_padded()
                if forward_batch is not None
                else None
            ),
        )
        weights, ids, _ = selected
        # The baseline TPU model applies routed_scaling_factor after its MoE;
        # apply it only to routed coefficients, leaving the shared expert alone.
        weights = weights.to(hidden_states.dtype).float()
        if not self.experts.should_fuse_routed_scaling_factor_in_topk:
            weights = weights * self.routed_scaling_factor
        has_shared = self._glm_v2_shared and not skip_shared_experts
        op = self._glm_v2_op if has_shared else self._glm_v2_routed_op
        output = op(
            hidden_states,
            ids.to(torch.int32),
            weights,
            self.experts.w13_weight,
            self.experts.w2_weight,
            self.experts.w13_weight_scale,
            self.experts.w2_weight_scale,
            self._glm_v2_rank,
            self._glm_v2_sw13,
            self._glm_v2_sw2,
            self._glm_v2_ss13,
            self._glm_v2_ss2,
        )
        if (
            not has_shared
            and not skip_shared_experts
            and hasattr(self, "shared_experts")
        ):
            shared = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator
            )
            if not self._shared_expert_tp1:
                shared = get_parallel().tp_group.all_reduce(shared)
            output = output + shared
        return output
