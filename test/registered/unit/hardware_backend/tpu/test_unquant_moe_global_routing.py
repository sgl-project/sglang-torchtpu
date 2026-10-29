"""BF16 global-routing and SparseCore permute contracts."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestUnquantGlobalRouting(CustomTestCase):
    def test_sparse_core_permute_gathers_full_sorted_range(self):
        module_name = "sglang_tpu.moe.pallas.fused_moe_op"
        torch_tpu = ModuleType("torch_tpu")
        torch_tpu_internal = ModuleType("torch_tpu._internal")
        torch_tpu_pallas = ModuleType("torch_tpu._internal.pallas")
        torch_tpu_pallas.jax_op = Mock()
        torch_tpu._internal = torch_tpu_internal
        torch_tpu_internal.pallas = torch_tpu_pallas

        with patch.dict(
            sys.modules,
            {
                "torch_tpu": torch_tpu,
                "torch_tpu._internal": torch_tpu_internal,
                "torch_tpu._internal.pallas": torch_tpu_pallas,
            },
        ):
            fused_moe_op = importlib.import_module(module_name)

        jax = ModuleType("jax")
        jnp = ModuleType("jax.numpy")
        for name in ("uint8", "int32"):
            setattr(jnp, name, getattr(np, name))
        for name in ("argsort", "repeat", "arange", "sum", "array", "where"):
            setattr(jnp, name, getattr(np, name))
        jax.numpy = jnp

        gmm_v2 = ModuleType("sglang_tpu.moe.pallas.gmm_v2")
        gmm_v2.TileSizes = object
        gmm_v2.apply_act_fn = Mock()
        gmm_v2.calculate_tiling = Mock()
        gmm_v2.gmm_v2 = Mock()
        gmm_v2.lhs_sublane_tiling = Mock(return_value=16)

        gather_module = ModuleType("sglang_tpu.moe.pallas.sparse_core.ragged_gather_v2")
        captured = {}

        class GatherCalled(Exception):
            pass

        def capture_gather(_hidden_states, sorted_token_ids, start, end):
            captured["n_sorted"] = sorted_token_ids.shape[0]
            captured["start"] = int(start[0])
            captured["end"] = int(end[0])
            raise GatherCalled

        gather_module.ragged_gather_v2 = capture_gather
        fake_modules = {
            "jax": jax,
            "jax.numpy": jnp,
            "sglang_tpu.moe.pallas.gmm_v2": gmm_v2,
            "sglang_tpu.moe.pallas.sparse_core.ragged_gather_v2": (gather_module),
        }
        topk_ids = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 2, 4, 6],
                [1, 3, 5, 7],
            ],
            dtype=np.int32,
        )
        with (
            patch.dict(sys.modules, fake_modules),
            self.assertRaises(GatherCalled),
        ):
            fused_moe_op._fused_moe_jax(
                hidden_states=np.ones((4, 2), dtype=np.float32),
                w1=np.ones((2, 2, 4), dtype=np.float32),
                w2=np.ones((2, 2, 2), dtype=np.float32),
                topk_weights=np.ones((4, 4), dtype=np.float32),
                topk_ids=topk_ids,
                w1_scale=None,
                w2_scale=None,
                w1_bias=None,
                w2_bias=None,
                topk=4,
                activation_name="silu",
                global_num_experts=8,
                expert_group_offset=2,
                use_sc_gather=1,
            )

        self.assertEqual(
            captured,
            {"n_sorted": topk_ids.size, "start": 0, "end": topk_ids.size},
        )

    def test_rank_sum_and_per_expert_outputs_preserve_router_semantics(self):
        x = torch.tensor([[1.0], [0.5]])
        ids = torch.tensor([[0, 3, 7, -1], [4, 6, 8, -1]], dtype=torch.int32)
        weights = torch.tensor([[0.1, 0.0, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        original_ids, original_weights = ids.clone(), weights.clone()
        valid = (ids >= 0) & (ids < 8)
        expected = (weights * valid * (ids + 1)).sum(-1, keepdim=True) * x

        def kernel(**kwargs):
            routed_ids = kwargs["topk_ids"]
            # Router ids arrive unchanged; the kernel drops out-of-range ids.
            torch.testing.assert_close(routed_ids, ids)
            in_range = (routed_ids >= 0) & (routed_ids < kwargs["global_num_experts"])
            local_ids = routed_ids - kwargs["expert_group_offset"]
            local_experts = kwargs["w1"].shape[0]
            local = in_range & (local_ids >= 0) & (local_ids < local_experts)
            expert_values = kwargs["w1"][:, 0, 0][local_ids.clamp(0, local_experts - 1)]
            values = expert_values.unsqueeze(-1) * kwargs["hidden_states"].unsqueeze(1)
            route_weights = torch.where(local, kwargs["topk_weights"], 0)
            if kwargs["no_combine"]:
                # Global routing masks ownership, not weights.
                mask = (
                    local
                    if kwargs["global_num_experts"] > local_experts
                    else route_weights != 0
                )
                return torch.where(mask.unsqueeze(-1), values, 0)
            return (values * route_weights.unsqueeze(-1)).sum(1)

        for sparsecore in (False, True):
            with (
                self.subTest(sparsecore=sparsecore),
                patch(
                    "sglang_tpu.envs.envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get",
                    return_value=sparsecore,
                ),
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
                                gemm1_alpha=None,
                                gemm1_clamp_limit=None,
                                routed_scaling_factor=None,
                            )
                        )
                        dispatch = SimpleNamespace(
                            hidden_states=x, topk_output=(weights, ids, None)
                        )
                        # Test routing independently of the TPU kernel bridge.
                        # Importing the real bridge would require torch_tpu even
                        # though every tensor and the kernel reference are CPU.
                        module_name = "sglang_tpu.moe.pallas.fused_moe_op"
                        bridge = ModuleType(module_name)
                        call = bridge.fused_moe_call = Mock(side_effect=kernel)
                        with (
                            patch(
                                "sglang.srt.layers.quantization.unquant.get_parallel",
                                return_value=SimpleNamespace(
                                    moe_ep_size=ep_size, moe_ep_rank=rank
                                ),
                            ),
                            patch.dict(sys.modules, {module_name: bridge}),
                        ):
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
