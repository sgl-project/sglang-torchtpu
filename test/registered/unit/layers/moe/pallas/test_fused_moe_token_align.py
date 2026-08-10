"""Unit tests for the Pallas fused-MoE token alignment.

`_padded_token_count` derives the token padding from gmm_v2's constraint on the
*routed* row count (`m_pad * topk`), not on the token count. The derivation is
what these tests pin: a rewrite that "looks equivalent" (going back to a flat 16,
or dropping the gcd) has to stay correct for every topk, and has to keep the
padding no tighter than it was.
"""

import math
import unittest

from sglang.srt.layers.moe.pallas.fused_moe_op import _padded_token_count
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# gmm_v2's LHS sublane tiling for a bf16 lhs. 8-bit is 32 and 4-bit is 64, so a
# change of lhs dtype invalidates the alignment, not just this constant.
SUBLANE_TILING = 16
TOPK_RANGE = range(1, 33)
TOKEN_COUNTS = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 64, 80, 1024, 4096)


class TestPaddedTokenCount(CustomTestCase):
    def test_routed_rows_satisfy_the_kernel_constraint(self):
        # The property the padding exists for: after the permute the kernel sees
        # m_pad * topk rows, and that must be a multiple of the sublane tiling.
        for topk in TOPK_RANGE:
            for m in TOKEN_COUNTS:
                with self.subTest(topk=topk, m=m):
                    m_pad = _padded_token_count(m, topk)
                    self.assertEqual((m_pad * topk) % SUBLANE_TILING, 0)

    def test_padding_never_drops_tokens(self):
        for topk in TOPK_RANGE:
            for m in TOKEN_COUNTS:
                with self.subTest(topk=topk, m=m):
                    self.assertGreaterEqual(_padded_token_count(m, topk), m)

    def test_is_the_minimal_padding(self):
        # Minimality is the point of the change: any smaller m_pad >= m must
        # violate the constraint, otherwise we are still over-padding.
        for topk in TOPK_RANGE:
            for m in TOKEN_COUNTS:
                m_pad = _padded_token_count(m, topk)
                with self.subTest(topk=topk, m=m):
                    for candidate in range(m, m_pad):
                        self.assertNotEqual(
                            (candidate * topk) % SUBLANE_TILING,
                            0,
                            f"m_pad={m_pad} over-pads: {candidate} also satisfies "
                            "the constraint",
                        )

    def test_never_tighter_than_the_previous_flat_16(self):
        # The change must only ever loosen padding: a regression that tightens it
        # would silently grow every MoE shape.
        for topk in TOPK_RANGE:
            for m in TOKEN_COUNTS:
                flat_16 = ((m + 15) // 16) * 16
                with self.subTest(topk=topk, m=m):
                    self.assertLessEqual(_padded_token_count(m, topk), flat_16)

    def test_odd_topk_still_aligns_to_the_full_tiling(self):
        # gcd(16, odd) == 1, so nothing is gained and the flat 16 must remain --
        # this is the branch where loosening would be incorrect.
        for topk in (1, 3, 5, 7, 9, 15, 31):
            with self.subTest(topk=topk):
                self.assertEqual(_padded_token_count(9, topk), 16)

    def test_qwen35_decode_batch_stops_padding_to_16(self):
        # The measured case: topk=10 -> gcd 2 -> alignment 8, so a decode batch
        # of 8 pads to 8 and the permute sees 80 routed rows instead of 160.
        self.assertEqual(math.gcd(16, 10), 2)
        self.assertEqual(_padded_token_count(8, topk=10), 8)
        self.assertEqual(_padded_token_count(8, topk=10) * 10, 80)


if __name__ == "__main__":
    unittest.main()
