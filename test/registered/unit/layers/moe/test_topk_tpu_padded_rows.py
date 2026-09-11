import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _post_process(topk_ids, topk_weights, num_token_non_padded):
    topk_config = TopKConfig(top_k=2, allow_routed_experts_capture=False)
    router_logits = torch.zeros((topk_ids.shape[0], 8), dtype=torch.float32)
    with (
        patch.object(topk_module, "_is_cuda", False),
        patch.object(topk_module, "_is_hip", False),
        patch.object(topk_module, "_is_tpu", True),
    ):
        return topk_module._post_process_topk_ids(
            topk_ids,
            topk_weights,
            topk_config,
            router_logits,
            layer_id=0,
            num_token_non_padded=num_token_non_padded,
        )


class TestTopkTpuPaddedRows(CustomTestCase):
    def test_pad_rows_route_to_minus_one_with_zero_weight(self):
        topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 0]], dtype=torch.int32)
        topk_weights = torch.full((4, 2), 0.5, dtype=torch.float32)
        ids_in, weights_in = topk_ids.clone(), topk_weights.clone()

        out_ids, out_weights, _ = _post_process(
            topk_ids, topk_weights, torch.tensor(2, dtype=torch.int32)
        )

        self.assertTrue(torch.equal(out_ids[:2], topk_ids[:2]))
        self.assertTrue(torch.equal(out_weights[:2], topk_weights[:2]))
        self.assertTrue(
            torch.equal(out_ids[2:], torch.full((2, 2), -1, dtype=torch.int32))
        )
        self.assertTrue(torch.equal(out_weights[2:], torch.zeros(2, 2)))
        # Out of place: inputs untouched (they may alias graph buffers).
        self.assertTrue(torch.equal(topk_ids, ids_in))
        self.assertTrue(torch.equal(topk_weights, weights_in))

    def test_no_padding_is_identity(self):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        topk_weights = torch.rand(2, 2)
        out_ids, out_weights, _ = _post_process(
            topk_ids, topk_weights, torch.tensor(2, dtype=torch.int32)
        )
        self.assertTrue(torch.equal(out_ids, topk_ids))
        self.assertTrue(torch.equal(out_weights, topk_weights))

    def test_none_count_skips_the_branch(self):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        topk_weights = torch.rand(2, 2)
        out_ids, out_weights, _ = _post_process(topk_ids, topk_weights, None)
        self.assertIs(out_ids, topk_ids)
        self.assertIs(out_weights, topk_weights)


if __name__ == "__main__":
    unittest.main()
