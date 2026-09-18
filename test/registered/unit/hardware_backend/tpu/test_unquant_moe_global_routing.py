"""BF16 routing preserves rank partials, invalid slots, and no_combine."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestUnquantGlobalRouting(CustomTestCase):
    def test_rank_sum_and_per_expert_outputs_preserve_router_semantics(self):
        x = torch.tensor([[1.0], [0.5]])
        ids = torch.tensor([[0, 3, 7, -1], [4, 6, 8, -1]], dtype=torch.int32)
        weights = torch.tensor([[0.1, 0.0, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        original_ids, original_weights = ids.clone(), weights.clone()
        valid = (ids >= 0) & (ids < 8)
        expected = (weights * valid * (ids + 1)).sum(-1, keepdim=True) * x

        def kernel(**kwargs):
            routed_ids = kwargs["topk_ids"]
            self.assertTrue(torch.all(routed_ids >= 0))
            local_ids = routed_ids - kwargs["expert_group_offset"]
            local_experts = kwargs["w1"].shape[0]
            local = (local_ids >= 0) & (local_ids < local_experts)
            expert_values = kwargs["w1"][:, 0, 0][local_ids.clamp(0, local_experts - 1)]
            values = expert_values.unsqueeze(-1) * kwargs["hidden_states"].unsqueeze(1)
            route_weights = kwargs["topk_weights"] * local
            if kwargs["no_combine"]:
                # Global routing masks ownership, not weights. In particular,
                # sanitized invalid IDs still produce expert-0 output on rank 0.
                mask = (
                    local
                    if kwargs["global_num_experts"] > local_experts
                    else route_weights != 0
                )
                return torch.where(mask.unsqueeze(-1), values, 0)
            return (values * route_weights.unsqueeze(-1)).sum(1)

        for sparsecore in (False, True):
            with self.subTest(sparsecore=sparsecore), patch(
                "sglang.srt.layers.quantization.unquant.envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get",
                return_value=sparsecore,
            ):
                self.check_routing(x, ids, weights, expected, kernel)
        torch.testing.assert_close(ids, original_ids)
        torch.testing.assert_close(weights, original_weights)

    def check_routing(self, x, ids, weights, expected, kernel):
        for ep_size in (1, 4):
            for no_combine in (False, True):
                with self.subTest(ep_size=ep_size, no_combine=no_combine):
                    partials = []
                    for rank in range(ep_size):
                        local_experts = 8 // ep_size
                        layer = SimpleNamespace(
                            num_local_experts=local_experts,
                            w13_weight=torch.arange(
                                rank * local_experts + 1, (rank + 1) * local_experts + 1
                            ).reshape(-1, 1, 1),
                            w2_weight=torch.ones(local_experts, 1, 1),
                        )
                        method = SimpleNamespace(
                            moe_runner_config=SimpleNamespace(
                                apply_router_weight_on_input=False,
                                num_fused_shared_experts=0,
                                no_combine=no_combine,
                                activation="silu",
                            )
                        )
                        dispatch = SimpleNamespace(
                            hidden_states=x, topk_output=(weights, ids, None)
                        )
                        # Test routing independently of the TPU kernel bridge.
                        # Importing the real bridge would require torch_tpu even
                        # though every tensor and the kernel reference are CPU.
                        module_name = "sglang.srt.layers.moe.pallas.fused_moe_op"
                        bridge = ModuleType(module_name)
                        call = bridge.fused_moe_call = Mock(side_effect=kernel)
                        with patch(
                            "sglang.srt.layers.quantization.unquant.get_parallel",
                            return_value=SimpleNamespace(
                                moe_ep_size=ep_size, moe_ep_rank=rank
                            ),
                        ), patch.dict(sys.modules, {module_name: bridge}):
                            output = UnquantizedFusedMoEMethod.forward_tpu(
                                method, layer, dispatch
                            ).hidden_states
                        self.assertEqual(
                            call.call_args.kwargs["global_num_experts"],
                            8,
                        )
                        self.assertEqual(
                            call.call_args.kwargs["expert_group_offset"],
                            rank * local_experts,
                        )
                        if no_combine:
                            local = (ids >= rank * local_experts) & (
                                ids < (rank + 1) * local_experts
                            )
                            expected_slots = torch.where(
                                (local & (weights != 0)).unsqueeze(-1),
                                (ids + 1).unsqueeze(-1) * x.unsqueeze(1),
                                0,
                            )
                            torch.testing.assert_close(output, expected_slots)
                        partials.append(
                            (output * weights.unsqueeze(-1)).sum(1)
                            if no_combine
                            else output
                        )
                    torch.testing.assert_close(torch.stack(partials).sum(0), expected)


if __name__ == "__main__":
    unittest.main()
