"""Unit tests for TPU unquantized MoE routing helpers."""

import unittest
from unittest import mock

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
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
            torch.tensor([[1, 0], [0, 0]], dtype=torch.int32),
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
            masked_ids, torch.tensor([[0, 0]], dtype=torch.int32)
        )


class TestUnquantTpuMoeWeightProcessing(CustomTestCase):
    """``UnquantizedFusedMoEMethod.process_weights_after_loading`` transposes the
    expert weights to the gmm_v2 ``(E, k, n)`` layout at load time and frees the
    pre-transpose storage.

    Regression: for a square expert matrix (``n == k``) the transpose is
    shape-preserving, so the previous ``copy_or_rebind_param()`` path copied the
    transposed weights into -- and the following ``resize_(0)`` then freed -- the
    param's own storage, leaving a 0-byte parameter that failed at forward.
    """

    @staticmethod
    def _layer(w13_shape, w2_shape):
        layer = torch.nn.Module()
        layer.w13_weight = Parameter(torch.randn(*w13_shape), requires_grad=False)
        layer.w2_weight = Parameter(torch.randn(*w2_shape), requires_grad=False)
        return layer

    def _run(self, w13_shape, w2_shape):
        layer = self._layer(w13_shape, w2_shape)
        want = {
            name: getattr(layer, name).data.transpose(1, 2).contiguous().clone()
            for name in ("w13_weight", "w2_weight")
        }
        # The TPU branch uses no instance state, so an unbound call is enough.
        with mock.patch.object(unquant, "_is_tpu", True):
            UnquantizedFusedMoEMethod.process_weights_after_loading(None, layer)

        for name in ("w13_weight", "w2_weight"):
            param = getattr(layer, name)
            self.assertGreater(
                param.data.untyped_storage().nbytes(),
                0,
                msg=f"{name} storage was freed (0-byte parameter)",
            )
            self.assertEqual(tuple(param.shape), tuple(want[name].shape))
            torch.testing.assert_close(param.data, want[name])

    def test_square_expert_weight_is_not_zeroed(self):
        # w2 is square (n == k) -- the regression case; w13 is non-square.
        self._run(w13_shape=(4, 16, 8), w2_shape=(4, 8, 8))

    def test_non_square_expert_weights(self):
        self._run(w13_shape=(4, 16, 8), w2_shape=(4, 8, 16))


if __name__ == "__main__":
    unittest.main()
