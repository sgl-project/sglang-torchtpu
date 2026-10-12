"""Unit tests for TPU unquantized MoE routing helpers."""

import unittest

import torch

from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnquantTpuMoe(CustomTestCase):
    def test_translates_global_ids_and_masks_non_local_slots(self):
        # 8 experts over EP size 4 -> 2 local experts per rank; rank 1 owns
        # global ids 2 and 3.
        topk_weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32)
        topk_ids = torch.tensor([[3, 6], [0, 2]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=2, moe_ep_rank=1
        )

        torch.testing.assert_close(
            masked_weights,
            torch.tensor([[0.25, 0.0], [0.0, 0.40]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            masked_ids,
            torch.tensor([[1, -1], [-1, 0]], dtype=torch.int32),
        )
        # Inputs are not mutated.
        torch.testing.assert_close(
            topk_weights,
            torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            topk_ids,
            torch.tensor([[3, 6], [0, 2]], dtype=torch.int32),
        )

    def test_rank0_full_local_range_is_identity(self):
        # EP rank 0 with all ids inside the local range: ids pass through and
        # no weight is zeroed.
        topk_weights = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
        topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=4, moe_ep_rank=0
        )

        torch.testing.assert_close(masked_weights, topk_weights)
        torch.testing.assert_close(masked_ids, topk_ids)

    def test_last_rank_boundary(self):
        # Ids one below and one at the start of the last rank's range.
        topk_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        topk_ids = torch.tensor([[5, 6]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=2, moe_ep_rank=3
        )

        torch.testing.assert_close(
            masked_weights, torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        )
        torch.testing.assert_close(
            masked_ids, torch.tensor([[-1, 0]], dtype=torch.int32)
        )

    def test_padded_slots_stay_dropped(self):
        # TopK marks padded tokens -1; they stay -1 with zero weight on rank 0.
        topk_weights = torch.tensor([[0.0, 0.0], [0.3, 0.7]], dtype=torch.float32)
        topk_ids = torch.tensor([[-1, -1], [1, 0]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=2, moe_ep_rank=0
        )

        torch.testing.assert_close(masked_weights, topk_weights)
        torch.testing.assert_close(
            masked_ids, torch.tensor([[-1, -1], [1, 0]], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
