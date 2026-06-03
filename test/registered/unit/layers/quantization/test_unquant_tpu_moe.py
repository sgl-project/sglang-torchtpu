"""Unit tests for TPU unquantized MoE routing helpers."""

import unittest

import torch

from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnquantTpuMoe(CustomTestCase):
    def test_masks_ep_non_local_expert_slots(self):
        topk_weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32)
        topk_ids = torch.tensor([[3, -1], [-1, 2]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids
        )

        torch.testing.assert_close(
            masked_weights,
            torch.tensor([[0.25, 0.0], [0.0, 0.40]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            masked_ids,
            torch.tensor([[3, 0], [0, 2]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            topk_weights,
            torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            topk_ids,
            torch.tensor([[3, -1], [-1, 2]], dtype=torch.int32),
        )

    def test_non_ep_routing_is_unchanged(self):
        topk_weights = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
        topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids
        )

        torch.testing.assert_close(masked_weights, topk_weights)
        torch.testing.assert_close(masked_ids, topk_ids)


if __name__ == "__main__":
    unittest.main()
