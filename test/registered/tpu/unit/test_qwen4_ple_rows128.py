"""Gathering Qwen4-Exp PLE rows from the lane-aligned [N * 160 / 128, 128] view
returns exactly table[ids] (sglang_tpu.hooks.qwen4_exp.gather_ple_rows128)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang_tpu.hooks.qwen4_exp import gather_ple_rows128, ple_rows128_supported


class TestPleRows128(unittest.TestCase):
    def test_matches_direct_row_lookup(self):
        torch.manual_seed(0)
        n = 4096
        table = torch.randint(0, 256, (n, 160), dtype=torch.uint8)
        ids = torch.randint(0, n, (37, 16))
        ids[0, :4] = torch.tensor([0, 1, 2, 3])
        ids[1, :4] = torch.tensor([n - 4, n - 3, n - 2, n - 1])
        got = gather_ple_rows128(table.view(-1, 128), ids)
        self.assertEqual(tuple(got.shape), (37, 16, 160))
        self.assertTrue(torch.equal(got, table[ids]))

    def test_supported_shapes(self):
        self.assertTrue(ple_rows128_supported(torch.empty(1024, 160)))
        self.assertFalse(ple_rows128_supported(torch.empty(6, 160)))
        self.assertFalse(ple_rows128_supported(torch.empty(8, 128)))


if __name__ == "__main__":
    unittest.main()
