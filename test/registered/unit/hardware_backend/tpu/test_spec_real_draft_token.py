"""SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token inputs for TpuSpecWorker's forced accept."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.spec_ops import (
    simulated_real_draft_accept_inputs,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSimulatedRealDraftAcceptInputs(CustomTestCase):
    def test_bonus_is_target_token_at_forced_row(self):
        # bs=3, N=3 drafts -> 4 verify rows per request.
        target = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]], dtype=torch.int32
        )
        drafts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int64)
        forced = torch.tensor([2, 2, 2], dtype=torch.int64)
        num_verify = torch.tensor([3, 3, 3], dtype=torch.int32)
        num_correct, bonus = simulated_real_draft_accept_inputs(
            target, drafts, forced, num_verify
        )
        self.assertEqual(num_correct.tolist(), [2, 2, 2])
        self.assertEqual(bonus.tolist(), [12, 22, 32])
        self.assertEqual(bonus.dtype, drafts.dtype)
        self.assertEqual(num_correct.dtype, forced.dtype)

    def test_forced_length_is_clamped_to_verifiable_drafts(self):
        target = torch.arange(8, dtype=torch.int32).reshape(2, 4) + 100
        drafts = torch.zeros((2, 3), dtype=torch.int64)
        forced = torch.tensor([3, 3], dtype=torch.int64)
        # The second request can only verify one draft (e.g. at its length limit).
        num_verify = torch.tensor([3, 1], dtype=torch.int32)
        num_correct, bonus = simulated_real_draft_accept_inputs(
            target, drafts, forced, num_verify
        )
        self.assertEqual(num_correct.tolist(), [3, 1])
        self.assertEqual(bonus.tolist(), [103, 105])

    def test_zero_accept_takes_first_row(self):
        target = torch.tensor([[7, 8, 9, 10]], dtype=torch.int32)
        drafts = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        num_correct, bonus = simulated_real_draft_accept_inputs(
            target,
            drafts,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([3], dtype=torch.int32),
        )
        self.assertEqual(num_correct.tolist(), [0])
        self.assertEqual(bonus.tolist(), [7])


if __name__ == "__main__":
    unittest.main()
