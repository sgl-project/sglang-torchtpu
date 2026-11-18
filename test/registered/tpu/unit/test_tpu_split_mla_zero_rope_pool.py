"""TpuSplitMLATokenToKVPool for a model without rope (qk_rope_head_dim == 0).

The sparse MLA ops never write or read the rope cache of such a model, so the
pool allocates a one-row placeholder, leaves it out of the page buffers (P/D
transfer, HiCache), and its bytes per token leave it out of the KV budget.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase


class TestSplitMlaZeroRopeBytes(CustomTestCase):
    def test_bytes_per_token(self):
        from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool as Pool

        for raw_ring in (False, True):
            kwargs = dict(
                index_head_dim=128,
                index_key_layers=[True, False, True],
                index_kpool=4,
                raw_ring=raw_ring,
            )
            with_rope = Pool.bytes_per_token(qk_rope_head_dim=64, **kwargs)
            default_rope = Pool.bytes_per_token(**kwargs)
            no_rope = Pool.bytes_per_token(qk_rope_head_dim=0, **kwargs)
            self.assertEqual(default_rope, with_rope)
            self.assertEqual(with_rope - no_rope, 3 * Pool.LANE_BYTES)
        self.assertEqual(Pool.rope_bytes_per_token(64), 128)
        self.assertEqual(Pool.rope_bytes_per_token(0), 0)

    def test_qk_rope_head_dim_lookup_falls_back_safely(self):
        from sglang_tpu.hooks.dsa_kv_pool import _qk_rope_head_dim

        # Direct attribute on model_config (standard MLA/DSA ModelConfig).
        self.assertEqual(
            _qk_rope_head_dim(SimpleNamespace(qk_rope_head_dim=0)), 0
        )
        # Missing on model_config; falls back to hf_config / hf_config.text_config.
        self.assertEqual(
            _qk_rope_head_dim(
                SimpleNamespace(hf_config=SimpleNamespace(qk_rope_head_dim=0))
            ),
            0,
        )
        self.assertEqual(
            _qk_rope_head_dim(
                SimpleNamespace(
                    hf_config=SimpleNamespace(
                        text_config=SimpleNamespace(qk_rope_head_dim=0)
                    )
                )
            ),
            0,
        )
        # Unset everywhere -> conservatively budgets the RoPE lane (> 0).
        self.assertGreater(
            _qk_rope_head_dim(SimpleNamespace(hf_config=SimpleNamespace())), 0
        )


if __name__ == "__main__":
    unittest.main()
