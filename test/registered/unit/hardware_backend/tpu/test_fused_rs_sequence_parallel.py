# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS model-facing contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.layers.moe.fused_rs_sequence_parallel import (
    DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN,
    FusedMoeSequenceParallelPlan,
    get_fused_moe_sequence_parallel_plan,
    get_tpu_fused_rs_post_expert_norm_weight,
    prepare_fused_moe_sequence_parallel_input,
    prepare_tpu_fused_rs_post_expert_norm,
    use_fused_moe_sequence_parallel_plan,
)

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


class TestFusedRsSequenceParallel(CustomTestCase):
    def test_plan_is_forward_scoped_and_restored(self):
        outer = FusedMoeSequenceParallelPlan(True, True, True)
        inner = FusedMoeSequenceParallelPlan(False, True, True)

        self.assertEqual(
            get_fused_moe_sequence_parallel_plan(),
            DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN,
        )
        with use_fused_moe_sequence_parallel_plan(outer):
            self.assertEqual(get_fused_moe_sequence_parallel_plan(), outer)
            with use_fused_moe_sequence_parallel_plan(inner):
                self.assertEqual(get_fused_moe_sequence_parallel_plan(), inner)
            self.assertEqual(get_fused_moe_sequence_parallel_plan(), outer)
        self.assertEqual(
            get_fused_moe_sequence_parallel_plan(),
            DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN,
        )

    def test_disabled_plan_is_identity(self):
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.ones(2, 4),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.int32),
                torch.ones(2, 1),
            ),
        )

        prepared = prepare_fused_moe_sequence_parallel_input(
            SimpleNamespace(), dispatch_output
        )

        self.assertIs(prepared.dispatch_output, dispatch_output)
        self.assertEqual(prepared.output_token_count, 2)

    def test_gathers_expert_inputs_and_drops_router_logits(self):
        hidden_local = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        weights_local = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        ids_local = torch.arange(4, dtype=torch.int32).reshape(2, 2)
        hidden_global = torch.cat((hidden_local, hidden_local + 8))
        weights_global = torch.cat((weights_local, weights_local + 4))
        ids_global = torch.cat((ids_local, ids_local + 4))
        group = SimpleNamespace(
            world_size=2,
            ranks=(0, 1),
            all_gather=Mock(side_effect=(hidden_global, weights_global, ids_global)),
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_local,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                weights_local,
                ids_local,
                torch.ones(2, 4),
            ),
        )
        layer = SimpleNamespace(moe_ep_size=2)
        plan = FusedMoeSequenceParallelPlan(True, True, True)

        with (
            use_fused_moe_sequence_parallel_plan(plan),
            patch(
                "sglang_tpu.layers.moe.fused_rs_sequence_parallel.get_parallel",
                return_value=SimpleNamespace(attn_tp_group=group, tp_group=group),
            ),
        ):
            prepared = prepare_fused_moe_sequence_parallel_input(layer, dispatch_output)

        self.assertEqual(prepared.output_token_count, 2)
        self.assertEqual(prepared.dispatch_output.hidden_states.dtype, torch.bfloat16)
        torch.testing.assert_close(
            prepared.dispatch_output.hidden_states, hidden_global
        )
        torch.testing.assert_close(
            prepared.dispatch_output.topk_output.topk_weights, weights_global
        )
        torch.testing.assert_close(
            prepared.dispatch_output.topk_output.topk_ids, ids_global
        )
        self.assertIsNone(prepared.dispatch_output.topk_output.router_logits)


class TestFusedRsModelContract(CustomTestCase):
    def test_prepares_model_owned_post_expert_norm(self):
        layer = torch.nn.Module()
        layer.hidden_size = 4
        layer.w2_weight = torch.empty(1, 4, 4)
        layer.allow_backend_reduction = False
        layer.supports_tpu_fused_rs_post_expert_norm = True
        norm_weight = torch.ones(4)
        layer.tpu_fused_rs_post_expert_norm_weight = lambda: norm_weight

        prepare_tpu_fused_rs_post_expert_norm(layer)

        self.assertTrue(layer.allow_backend_reduction)
        self.assertIs(get_tpu_fused_rs_post_expert_norm_weight(layer), norm_weight)
        self.assertIn(
            "_tpu_fused_rs_post_expert_norm_weight",
            layer._non_persistent_buffers_set,
        )

    def test_model_without_contract_remains_unchanged(self):
        layer = torch.nn.Module()
        layer.allow_backend_reduction = False

        prepare_tpu_fused_rs_post_expert_norm(layer)

        self.assertFalse(layer.allow_backend_reduction)
        self.assertIsNone(get_tpu_fused_rs_post_expert_norm_weight(layer))


if __name__ == "__main__":
    unittest.main()
