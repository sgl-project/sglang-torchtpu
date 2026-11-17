"""TpuSplitMLATokenToKVPool for a model without rope (qk_rope_head_dim == 0).

The sparse MLA ops never write or read the rope cache of such a model, so the
pool allocates a one-row placeholder, leaves it out of the page buffers (P/D
transfer, HiCache), and its bytes per token leave it out of the KV budget.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase


class TestSplitMlaZeroRopeBytes(CustomTestCase):
    def test_bytes_per_token(self):
        from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool as Pool

        kwargs = dict(
            index_head_dim=128, index_key_layers=[True, False, True], index_kpool=4
        )
        with_rope = Pool.bytes_per_token(qk_rope_head_dim=64, **kwargs)
        no_rope = Pool.bytes_per_token(qk_rope_head_dim=0, **kwargs)
        self.assertEqual(with_rope - no_rope, 3 * Pool.LANE_BYTES)
        self.assertEqual(Pool.rope_bytes_per_token(64), 128)
        self.assertEqual(Pool.rope_bytes_per_token(0), 0)


if __name__ == "__main__":
    unittest.main()
