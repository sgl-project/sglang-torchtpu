# SPDX-License-Identifier: Apache-2.0
"""Speculative target verify on pallas_tpu_mla stays on the kernel's MIXED pass.

The MLA v2 kernel's DECODE range attends one query row per sequence. RPA v3
routes fixed-width verify to DECODE (decode_query_size = draft width), which on
MLA would read only each window's first query row: output past the first
draft token would be garbage. MLA turns that routing off.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla import TpuPallasMlaAttnBackend
from sglang_tpu.attention_pallas_rpa_base import _resolve_decode_query_size
from sglang_tpu.attention_pallas_rpa_v3 import TpuPallasRpaV3AttnBackend

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMlaSpecVerifyRange(CustomTestCase):
    def test_mla_verify_stays_on_mixed(self):
        mla = TpuPallasMlaAttnBackend.kernel_spec.supports_decode_query_size
        self.assertFalse(mla)
        self.assertEqual(
            _resolve_decode_query_size(supports_decode_query_size=mla, spec_enabled=True),
            1,
        )

    def test_rpa_v3_keeps_decode_routing(self):
        # The MLA spec is derived from RPA v3's; only this flag differs.
        self.assertTrue(TpuPallasRpaV3AttnBackend.kernel_spec.supports_decode_query_size)


if __name__ == "__main__":
    unittest.main()
