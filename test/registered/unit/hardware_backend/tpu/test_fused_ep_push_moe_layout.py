# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the gmm_v2 -> push-kernel FP4 weight and scale conversion."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu import fused_ep_push_moe as m


def _codes_kn(packed_kn: torch.Tensor) -> torch.Tensor:
    """uint8 [E, K/2, N] (two K codes per byte) -> int codes [E, K, N]."""
    lo, hi = packed_kn & 0x0F, packed_kn >> 4
    return torch.stack((lo, hi), dim=2).flatten(1, 2)


def _codes_nk(packed_n: torch.Tensor) -> torch.Tensor:
    """uint8 [E, K, N/2] (two N codes per byte) -> int codes [E, K, N]."""
    lo, hi = packed_n & 0x0F, packed_n >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


class TestFusedEpPushMoeLayout(CustomTestCase):
    def test_repack_keeps_every_code_in_place(self):
        g = torch.Generator().manual_seed(0)
        packed = torch.randint(0, 256, (3, 8, 6), dtype=torch.uint8, generator=g)
        out = m.repack_k_to_n(packed)
        self.assertEqual(tuple(out.shape), (3, 16, 3))
        torch.testing.assert_close(_codes_nk(out), _codes_kn(packed))

    def test_repack_rejects_odd_n(self):
        with self.assertRaises(ValueError):
            m.repack_k_to_n(torch.zeros((1, 2, 3), dtype=torch.uint8))

    def test_flat_scales_pad_each_expert_to_the_sublane_tile(self):
        k, n, experts = 5 * 512, 4, 2
        scale = torch.arange(experts * 5 * n, dtype=torch.float32).reshape(
            experts, 5, 1, n
        )
        flat = m.flat_padded_scales(scale, k)
        self.assertEqual(m.scale_rows(k), 8)
        self.assertEqual(tuple(flat.shape), (experts * 8, n))
        view = flat.view(experts, 8, n)
        torch.testing.assert_close(view[:, :5], scale[:, :, 0])
        self.assertTrue(torch.all(view[:, 5:] == 0))

    def test_flat_scales_reject_a_mismatched_block_count(self):
        with self.assertRaises(ValueError):
            m.flat_padded_scales(torch.zeros((1, 4, 1, 2)), 5 * 512)


if __name__ == "__main__":
    unittest.main()
