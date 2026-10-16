# SPDX-License-Identifier: Apache-2.0
"""Where speculative target verify runs on the MLA backends.

Dense ``pallas_tpu_mla`` runs fixed-width verify on the MLA kernel's decode
passes with ``decode_query_size`` rows per sequence, as RPA v3 does. The sparse
backend's kernel has no fixed-width decode pass, so its verify keeps the extend
layout.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla import TpuPallasMlaAttnBackend
from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend
from sglang_tpu.attention_pallas_rpa_base import _resolve_decode_query_size

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMlaSpecVerifyRange(CustomTestCase):
    def test_dense_mla_verifies_on_decode(self):
        self.assertTrue(TpuPallasMlaAttnBackend.kernel_spec.supports_decode_query_size)

    def test_sparse_mla_keeps_verify_on_extend(self):
        sparse = TpuPallasSparseMlaAttnBackend.kernel_spec.supports_decode_query_size
        self.assertFalse(sparse)
        self.assertEqual(
            _resolve_decode_query_size(
                supports_decode_query_size=sparse, spec_enabled=True
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
