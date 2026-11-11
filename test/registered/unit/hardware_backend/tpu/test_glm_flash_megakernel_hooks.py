"""CPU checks of the GLM-5.3-Flash megakernel's serving-side bookkeeping."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.glm_flash_megakernel import ops

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlmFlashMegakernelBuckets(CustomTestCase):
    def test_buckets_cover_every_batch_up_to_max(self):
        """Every live batch size maps to a compiled bucket; the largest is max_req itself."""
        for max_req in (1, 3, 8, 16):
            bs = ops._buckets(max_req)
            self.assertEqual(bs[-1], max_req)
            for n in range(1, max_req + 1):
                self.assertTrue(any(b >= n for b in bs), (max_req, n))


class TestServingGuards(CustomTestCase):
    def test_expert_parallel_is_rejected_at_startup(self):
        """EP>1 ranks hold whole experts, which the kernel's per-rank expert DMA cannot
        stream; the hook must refuse at startup, not fail Mosaic verification mid-serve."""
        ops.require_expert_tp(1)
        with self.assertRaises(ValueError):
            ops.require_expert_tp(8)


class TestPoolBookkeeping(CustomTestCase):
    def test_record_bytes_match_the_pool_layout(self):
        """The kernel's record stride is the pool's: a mismatch reads the wrong pooled key."""
        from sglang_tpu.glm_flash_megakernel import layout as PL
        from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool

        for d in (64, 128, 256):
            self.assertEqual(
                PL.record_bytes(d), TpuSplitMLATokenToKVPool.index_key_record_bytes(d)
            )

    def test_closing_pools_are_exactly_those_ending_in_the_step(self):
        """A pool closes when its last token is one of the step's rows; the step keys
        and stores those and no others."""
        import numpy as np

        from sglang_tpu.glm_flash_megakernel import layout as PL

        kp = 4
        for Q in (1, 2, 4):
            for pos in range(0, 40):
                first, count = PL.closing_pools(np.int32(pos), Q, kp)
                want = [
                    p for p in range(0, 20) if pos <= p * kp + kp - 1 <= pos + Q - 1
                ]
                got = list(range(int(first), int(first) + int(count)))
                self.assertEqual(got, want, (Q, pos))


if __name__ == "__main__":
    unittest.main()
