import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import (
    _balanced_routing_num_ranks,
    _make_round_robin_expert_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_EXPERTS = 256
TOPK = 8


class TestSimulateRoundRobinExperts(CustomTestCase):
    """SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS must not leave EP ranks idle.

    Spacing each token's slots by num_experts // topk balances ranks only while
    num_tokens * topk >= num_experts. Below that, a rank whose expert block is
    narrower than the spacing is never selected -- at num_experts=256, topk=8,
    EP16 every odd rank drew zero tokens for any batch under 32.
    """

    def _route(self, num_tokens, num_ranks, *, topk=TOPK, layer_id=None):
        with patch(
            "sglang.srt.layers.moe.topk._balanced_routing_num_ranks",
            return_value=num_ranks,
        ):
            return _make_round_robin_expert_ids(
                num_tokens,
                topk,
                NUM_EXPERTS,
                device=torch.device("cpu"),
                dtype=torch.int32,
                layer_id=layer_id,
            )

    def _per_rank_counts(self, ids, num_ranks):
        block = NUM_EXPERTS // num_ranks
        return torch.bincount(
            ids.flatten().to(torch.int64) // block, minlength=num_ranks
        ).tolist()

    def test_every_rank_draws_the_same_token_count(self):
        for num_ranks in (1, 2, 4, 8, 16):
            for num_tokens in (1, 2, 4, 8, 16, 32, 300):
                for layer_id in (None, 3):
                    with self.subTest(
                        num_ranks=num_ranks, num_tokens=num_tokens, layer_id=layer_id
                    ):
                        ids = self._route(num_tokens, num_ranks, layer_id=layer_id)
                        counts = self._per_rank_counts(ids, num_ranks)
                        assignments = num_tokens * TOPK
                        self.assertEqual(sum(counts), assignments)
                        # Off by at most one only when the assignments cannot
                        # divide across the ranks; exactly equal otherwise.
                        self.assertLessEqual(max(counts) - min(counts), 1)
                        if assignments % num_ranks == 0:
                            self.assertEqual(min(counts), assignments // num_ranks)

    def test_activated_expert_count(self):
        for num_tokens in (1, 4, 32, 300):
            with self.subTest(num_tokens=num_tokens):
                ids = self._route(num_tokens, 16)
                self.assertEqual(
                    ids.unique().numel(), min(num_tokens * TOPK, NUM_EXPERTS)
                )

    def test_each_token_gets_distinct_experts(self):
        for num_ranks in (1, 4, 16):
            with self.subTest(num_ranks=num_ranks):
                ids = self._route(64, num_ranks)
                for row in ids:
                    self.assertEqual(row.unique().numel(), TOPK)

    def test_ids_stay_in_range_and_keep_dtype(self):
        for num_ranks in (1, 4, 16):
            with self.subTest(num_ranks=num_ranks):
                ids = self._route(300, num_ranks)
                self.assertEqual(ids.shape, (300, TOPK))
                self.assertEqual(ids.dtype, torch.int32)
                self.assertGreaterEqual(int(ids.min()), 0)
                self.assertLess(int(ids.max()), NUM_EXPERTS)

    def test_zero_topk_is_empty(self):
        ids = self._route(4, 16, topk=0)
        self.assertEqual(ids.shape, (4, 0))

    def test_num_ranks_is_one_without_distributed(self):
        # Standalone benchmark scripts never create the EP group; dealing must
        # fall back to a single rank instead of asserting in get_parallel().
        self.assertFalse(torch.distributed.is_initialized())
        self.assertEqual(_balanced_routing_num_ranks(NUM_EXPERTS), 1)

    def test_num_ranks_is_one_when_ep_size_does_not_divide_experts(self):
        with patch.object(torch.distributed, "is_initialized", return_value=True):
            with patch("sglang.srt.layers.moe.topk.get_parallel") as mock_get_parallel:
                mock_get_parallel.return_value.moe_ep_size = 16
                self.assertEqual(_balanced_routing_num_ranks(256), 16)
                self.assertEqual(_balanced_routing_num_ranks(250), 1)


if __name__ == "__main__":
    unittest.main()
