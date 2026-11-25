# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DSV4 RoPE kernels' token tile: below the full extent it is
the sublane axis of the (tile_n, rotary_dim) cos/sin block, so it must be a multiple
of 8 (regression: 24 requests x 5 draft tokens = 120 tokens tiled as 60 and Mosaic
rejected `rope`: "Slice sizes along tiled dimensions must be aligned to tiles")."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dsv4.rope import SUBLANE, largest_divisor, token_tile

CAPS = (64, 128)  # qnorm_rope and rope / rope_quant
# Verify rows per request: DSpark draft (5) and target (6).
TOKENS_PER_REQUEST = (5, 6)


class TestTokenTile(CustomTestCase):
    def test_verify_shapes_aligned(self):
        for cap in CAPS:
            for bs in range(1, 65):
                for k in TOKENS_PER_REQUEST:
                    n = bs * k
                    t = token_tile(n, cap)
                    with self.subTest(cap=cap, bs=bs, tokens=n):
                        self.assertEqual(n % t, 0)
                        self.assertLessEqual(t, cap)
                        if n % SUBLANE == 0 or n <= cap:
                            # Full extent, or a sublane-aligned tile.
                            self.assertTrue(t == n or t % SUBLANE == 0, t)

    def test_regression_shapes(self):
        self.assertEqual(token_tile(120, 64), 40)
        self.assertEqual(token_tile(200, 64), 40)
        self.assertEqual(token_tile(240, 64), 48)
        self.assertEqual(token_tile(200, 128), 40)

    def test_existing_shapes_unchanged(self):
        # Buckets 1..64 (powers of two) and prefill chunks keep their old tile.
        shapes = [k * bs for bs in (1, 2, 4, 8, 16, 32, 64) for k in (5, 6)]
        shapes += [512, 1024, 2048, 4096, 8192, 16384]
        for cap in CAPS:
            for n in shapes:
                with self.subTest(cap=cap, tokens=n):
                    self.assertEqual(token_tile(n, cap), largest_divisor(n, cap))


if __name__ == "__main__":
    unittest.main()
