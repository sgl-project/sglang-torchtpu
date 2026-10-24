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
from sglang_tpu.fused_rs_sequence_parallel import (
    DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN,
    FusedMoeSequenceParallelPlan,
    fused_moe_entry_group,
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
                "sglang_tpu.fused_rs_sequence_parallel.get_parallel",
                return_value=SimpleNamespace(
                    attn_dp_size=1, attn_tp_group=group, tp_group=group
                ),
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

    def _dp_attention_input(self):
        """Two attention-DP groups of two ranks; this rank holds two rows."""
        hidden_local = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        weights_local = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
        # Expert IDs above 2**8 check the packed fp32 gather keeps them exact.
        ids_local = torch.tensor([[3, 300], [1023, 0]], dtype=torch.int32)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_local,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights_local, ids_local, torch.ones(2, 4)),
        )
        tp_group = SimpleNamespace(
            world_size=4,
            ranks=(0, 1, 2, 3),
            all_gather=Mock(side_effect=lambda x, dim: torch.cat([x] * 4, dim=dim)),
        )
        attn_tp_group = SimpleNamespace(
            world_size=2, ranks=(0, 1), all_gather=Mock(side_effect=AssertionError)
        )
        parallel = SimpleNamespace(
            attn_dp_size=2, attn_tp_group=attn_tp_group, tp_group=tp_group
        )
        return dispatch_output, parallel

    def test_dp_attention_gathers_over_tp_group_and_keeps_own_rows(self):
        """Under attention DP the entry spans every group's rows and each rank
        keeps an output block the size of the rows it passed."""
        dispatch_output, parallel = self._dp_attention_input()
        topk_local = dispatch_output.topk_output
        layer = SimpleNamespace(moe_ep_size=4, tpu_fused_rs_dp_attention_layout=True)
        plan = FusedMoeSequenceParallelPlan(True, True, True)

        with (
            use_fused_moe_sequence_parallel_plan(plan),
            patch(
                "sglang_tpu.fused_rs_sequence_parallel.get_parallel",
                return_value=parallel,
            ),
        ):
            self.assertIs(fused_moe_entry_group(), parallel.tp_group)
            prepared = prepare_fused_moe_sequence_parallel_input(layer, dispatch_output)

        self.assertEqual(prepared.output_token_count, 2)
        # Hidden states plus one packed top-k gather, all over the TP group.
        self.assertEqual(parallel.tp_group.all_gather.call_count, 2)
        topk = prepared.dispatch_output.topk_output
        self.assertEqual(topk.topk_weights.dtype, topk_local.topk_weights.dtype)
        self.assertEqual(topk.topk_ids.dtype, torch.int32)
        torch.testing.assert_close(
            topk.topk_weights, torch.cat([topk_local.topk_weights] * 4)
        )
        torch.testing.assert_close(topk.topk_ids, torch.cat([topk_local.topk_ids] * 4))

    def test_dp_attention_requires_model_layout(self):
        """Without the model's gathered layout, attention DP must fail closed:
        the kernel would sum partials of different groups' rows."""
        dispatch_output, parallel = self._dp_attention_input()
        layer = SimpleNamespace(moe_ep_size=4)

        with (
            use_fused_moe_sequence_parallel_plan(
                FusedMoeSequenceParallelPlan(True, True, True)
            ),
            patch(
                "sglang_tpu.fused_rs_sequence_parallel.get_parallel",
                return_value=parallel,
            ),
            self.assertRaisesRegex(RuntimeError, "gathered DP-attention layout"),
        ):
            prepare_fused_moe_sequence_parallel_input(layer, dispatch_output)
        parallel.tp_group.all_gather.assert_not_called()


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
