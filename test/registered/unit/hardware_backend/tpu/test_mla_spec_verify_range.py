# SPDX-License-Identifier: Apache-2.0
"""Speculative target verify runs on the decode passes of both MLA backends.

Dense ``pallas_tpu_mla`` and sparse ``pallas_tpu_mla_sparse`` (whose DSA
indexer runs StreamIndex) take ``decode_query_size`` rows per sequence on the
decode passes, as RPA v3 does.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.layers.attention.attention_pallas_mla import TpuPallasMlaAttnBackend
from sglang_tpu.layers.attention.attention_pallas_mla_sparse import (
    TpuPallasSparseMlaAttnBackend,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMlaSpecVerifyRange(CustomTestCase):
    def test_both_mla_backends_verify_on_decode(self):
        for backend in (TpuPallasMlaAttnBackend, TpuPallasSparseMlaAttnBackend):
            with self.subTest(backend=backend.__name__):
                self.assertTrue(backend.kernel_spec.supports_decode_query_size)


if __name__ == "__main__":
    unittest.main()
