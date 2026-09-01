# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS Torch wrapper."""

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
            patch.object(
                fused_moe_rs_op,
                "_get_tp_rank_to_device_id_mapping",
                return_value=(4, (0, 1, 6, 7, 2, 3, 4, 5)),
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=Mock(moe_ep_rank=2),
            ),
            patch.object(fused_moe_rs_op, "_build_fused_moe_rs_op") as build,
        ):
            fused_moe_rs_op.prepare_fused_moe_rs(layer)
            fused_moe_rs_op.prepare_fused_moe_rs(layer)

        torch.testing.assert_close(
            layer._tpu_fused_rs_tp_rank_to_device_id,
            torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], dtype=torch.int32),
        )
        torch.testing.assert_close(
            layer._tpu_fused_rs_group_offset,
            torch.tensor([32, 2], dtype=torch.int32),
        )
        self.assertIn("_tpu_fused_rs_tp_rank_to_device_id", layer._buffers)
        self.assertIn("_tpu_fused_rs_group_offset", layer._buffers)
        self.assertIn(
            "_tpu_fused_rs_tp_rank_to_device_id", layer._non_persistent_buffers_set
        )
        self.assertIn("_tpu_fused_rs_group_offset", layer._non_persistent_buffers_set)
        self.assertEqual(build.call_count, 2)

    def test_gather_reorders_tp_ranks_to_physical_device_order(self):
        rank_order_shards = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]])
        tp_rank_to_device_id = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], dtype=torch.int32)
        tp_group = Mock(
            world_size=8,
            all_gather=Mock(return_value=rank_order_shards),
        )
        token_shard = torch.zeros(2, 1)

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
                tp_rank_to_device_id=tp_rank_to_device_id,
            )

        torch.testing.assert_close(output, torch.arange(8).reshape(8, 1))
        tp_group.all_gather.assert_called_once_with(token_shard, dim=0)

    def test_keep_token_shard_skips_gather(self):
        token_shard = torch.tensor([[4], [5]], dtype=torch.bfloat16)
        tp_group = Mock(world_size=8)

        with (
            patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
            patch.object(
                fused_moe_rs_op,
                "fused_moe_rs_call",
                return_value=token_shard,
            ),
        ):
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=torch.zeros(16, 1),
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=torch.ones(16, 1),
                topk_ids=torch.zeros(16, 1, dtype=torch.int32),
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=8,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=False,
                tp_rank_to_device_id=torch.arange(8, dtype=torch.int32),
                keep_token_shard=True,
            )

        torch.testing.assert_close(output, token_shard)
        tp_group.all_gather.assert_not_called()

    def test_keep_token_shard_pads_each_logical_rank_block(self):
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(8, 1)
        topk_weights = hidden_states.clone()
        topk_ids = torch.arange(8, dtype=torch.int32).reshape(8, 1)
        token_shard = torch.tensor([[5], [0]], dtype=torch.bfloat16)

        with patch.object(
            fused_moe_rs_op,
            "fused_moe_rs_call",
            return_value=token_shard,
        ) as fused:
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=hidden_states,
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=1,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=False,
                tp_rank_to_device_id=torch.arange(8, dtype=torch.int32),
                keep_token_shard=True,
            )

        expected = torch.tensor(
            [
                [0],
                [0],
                [1],
                [0],
                [2],
                [0],
                [3],
                [0],
                [4],
                [0],
                [5],
                [0],
                [6],
                [0],
                [7],
                [0],
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(fused.call_args.kwargs["hidden_states"], expected)
        torch.testing.assert_close(fused.call_args.kwargs["topk_weights"], expected)
        torch.testing.assert_close(
            fused.call_args.kwargs["topk_ids"], expected.to(torch.int32)
        )
        torch.testing.assert_close(output, torch.tensor([[5]], dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
