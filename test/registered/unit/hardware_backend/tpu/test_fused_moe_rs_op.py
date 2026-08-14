# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS Torch wrapper."""

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
                return_value=(4, (0, 1, 4, 5, 6, 7, 2, 3), (0, 1, 6, 7, 2, 3, 4, 5)),
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_expert_parallel_rank",
                return_value=2,
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
            torch.tensor([0, 1, 4, 5, 6, 7, 2, 3], dtype=torch.int64),
        )
        torch.testing.assert_close(
            layer._tpu_fused_rs_group_offset,
            torch.tensor([[32, 4, 0, 1, 6, 7, 2, 3, 4, 5]], dtype=torch.int32),
        )
        self.assertIn("_tpu_fused_rs_physical_order", layer._buffers)
        self.assertIn("_tpu_fused_rs_group_offset", layer._buffers)
        self.assertIn("_tpu_fused_rs_physical_order", layer._non_persistent_buffers_set)
        self.assertIn("_tpu_fused_rs_group_offset", layer._non_persistent_buffers_set)
        self.assertEqual(build.call_count, 2)

    def test_physical_rank_metadata_rejects_invalid_device_ids(self):
        tp_group = Mock(world_size=4, cpu_group=object())

        def gather_duplicate_id(output, _device_id, *, group):
            self.assertIs(group, tp_group.cpu_group)
            output[:] = [0, 1, 1, 3]

        fused_moe_rs_op._PHYSICAL_RANK_METADATA_CACHE.clear()
        try:
            with (
                patch.object(
                    fused_moe_rs_op,
                    "_get_current_global_device_id",
                    return_value=0,
                ),
                patch(
                    "sglang.srt.distributed.get_tp_group",
                    return_value=tp_group,
                ),
                patch("torch.distributed.is_initialized", return_value=True),
                patch(
                    "torch.distributed.all_gather_object",
                    side_effect=gather_duplicate_id,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "physical device IDs"):
                    fused_moe_rs_op._get_physical_rank_metadata()
        finally:
            fused_moe_rs_op._PHYSICAL_RANK_METADATA_CACHE.clear()

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
                fp8_direct_write=False,
                physical_order_to_tp_rank=physical_order_to_tp_rank,
            )

        torch.testing.assert_close(output, torch.arange(8).reshape(8, 1))
        tp_group.all_gather.assert_called_once_with(token_shard, dim=0)


if __name__ == "__main__":
    unittest.main()
