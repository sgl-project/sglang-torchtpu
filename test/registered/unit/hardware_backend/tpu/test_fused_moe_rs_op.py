# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS Torch wrapper."""

import importlib.util
import os
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.pallas import fused_moe_rs_op
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestFusedMoeRsOp(CustomTestCase):
    def test_padding_satisfies_ep_and_routed_row_alignment(self):
        cases = (
            # num_tokens, topk, ep_size, padded_tokens
            (8, 1, 8, 16),
            (4, 2, 4, 8),
            (16, 8, 16, 16),
            (17, 8, 16, 32),
        )
        for num_tokens, topk, ep_size, expected in cases:
            with self.subTest(num_tokens=num_tokens, topk=topk, ep_size=ep_size):
                actual = fused_moe_rs_op.fused_moe_rs_padded_tokens(
                    num_tokens, topk, ep_size
                )
                self.assertEqual(actual, expected)
                self.assertEqual(actual % ep_size, 0)
                self.assertEqual((actual * topk) % 16, 0)

    def test_prepare_sets_physical_rank_metadata(self):
        layer = torch.nn.Module()
        layer.num_local_experts = 16
        layer.w13_weight = torch.empty(1)

        with (
            patch.dict(
                os.environ,
                {"TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false"},
            ),
            patch.object(
                fused_moe_rs_op,
                "_get_physical_rank_metadata",
                return_value=(4, (0, 1, 6, 7, 2, 3, 4, 5)),
            ),
            patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=Mock(rank=2),
            ),
            patch.object(fused_moe_rs_op, "_build_fused_moe_rs_op") as build,
        ):
            fused_moe_rs_op.prepare_fused_moe_rs(layer)
            fused_moe_rs_op.prepare_fused_moe_rs(layer)
            self.assertEqual(
                os.environ["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"],
                "false",
            )

        torch.testing.assert_close(
            layer._tpu_fused_rs_physical_order,
            torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], dtype=torch.int64),
        )
        torch.testing.assert_close(
            layer._tpu_fused_rs_group_offset,
            torch.tensor([[64, 4]], dtype=torch.int32),
        )
        self.assertIn("_tpu_fused_rs_physical_order", layer._buffers)
        self.assertIn("_tpu_fused_rs_group_offset", layer._buffers)
        self.assertIn("_tpu_fused_rs_physical_order", layer._non_persistent_buffers_set)
        self.assertIn("_tpu_fused_rs_group_offset", layer._non_persistent_buffers_set)
        self.assertEqual(build.call_count, 2)

    def test_gather_reorders_tp_ranks_to_physical_device_order(self):
        rank_order_shards = torch.tensor([[0], [1], [4], [5], [6], [7], [2], [3]])
        physical_order_to_tp_rank = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5])
        tp_group = Mock(
            world_size=8,
            all_gather=Mock(return_value=rank_order_shards),
        )
        token_shard = torch.zeros(1, 1)

        with (
            patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=tp_group,
            ),
            patch.object(
                fused_moe_rs_op,
                "fused_moe_rs_call",
                return_value=token_shard,
            ),
        ):
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=torch.zeros(8, 1),
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=torch.empty(1),
                topk_ids=torch.empty(1),
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=1,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=True,
                physical_order_to_tp_rank=physical_order_to_tp_rank,
            )

        torch.testing.assert_close(output, torch.arange(8).reshape(8, 1))
        tp_group.all_gather.assert_called_once_with(token_shard, dim=0)

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_unmocked_build_fused_moe_rs_op_exports_mlir(self):
        from sglang.srt.hardware_backend.tpu import pallas_export

        pallas_export.configure_jax_export_env(8)
        op = fused_moe_rs_op._build_fused_moe_rs_op()
        self.assertIsNotNone(op)
        self.assertTrue(hasattr(torch.ops.sglang_tpu_pallas, "fused_moe_rs"))

    @unittest.skipUnless(
        importlib.util.find_spec("torch_tpu"),
        "requires TorchTPU",
    )
    def test_dynamo_faketensor_export_fused_moe_rs(self):
        class FusedMoeRsModule(torch.nn.Module):
            def forward(
                self,
                hidden_states: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                w1_scale: torch.Tensor,
                w2_scale: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                post_expert_norm_weight: torch.Tensor,
            ) -> torch.Tensor:
                return fused_moe_rs_op.fused_moe_rs_call(
                    hidden_states=hidden_states,
                    w1=w1,
                    w2=w2,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    post_expert_norm_weight=post_expert_norm_weight,
                    group_offset=None,
                    topk=8,
                    activation="silu",
                    has_post_expert_norm=False,
                    fp8_direct_write=True,
                )

        fused_moe_rs_op._build_fused_moe_rs_op()
        mod = FusedMoeRsModule()
        tokens = 128
        hidden_size = 6144
        num_local_experts = 20
        inter_size = 2560
        topk = 8
        example_inputs = (
            torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="meta"),
            torch.randn(
                num_local_experts,
                inter_size * 2,
                hidden_size,
                dtype=torch.float8_e4m3fn,
                device="meta",
            ),
            torch.randn(
                num_local_experts,
                hidden_size,
                inter_size,
                dtype=torch.float8_e4m3fn,
                device="meta",
            ),
            torch.randn(num_local_experts, dtype=torch.float32, device="meta"),
            torch.randn(num_local_experts, dtype=torch.float32, device="meta"),
            torch.randn(tokens, topk, dtype=torch.float32, device="meta"),
            torch.zeros(tokens, topk, dtype=torch.int32, device="meta"),
            torch.ones(1, hidden_size, dtype=torch.bfloat16, device="meta"),
        )
        exported = torch.export.export(mod, example_inputs)
        self.assertIsNotNone(exported)


if __name__ == "__main__":
    unittest.main()



