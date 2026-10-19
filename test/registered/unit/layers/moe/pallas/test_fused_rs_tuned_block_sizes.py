"""Regression tests for fused-RS tuned-table composition."""

import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "python" / "sglang").is_dir()
)
_TABLE_PATH = (
    _REPO_ROOT
    / "sglang_tpu/layers/moe/pallas/fused_rs/v2_fused_rs_tuned_block_sizes.py"
)
_SPEC = importlib.util.spec_from_file_location("fused_rs_tuned_table", _TABLE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TABLE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TABLE_MODULE)
FUSED_RS_TUNED_BLOCK_SIZES = _TABLE_MODULE.FUSED_RS_TUNED_BLOCK_SIZES


class TestFusedRsTunedBlockSizes(unittest.TestCase):
    def test_ep16_w4a8_uses_half_n_schedule_through_8k_tokens(self):
        dtype_table = FUSED_RS_TUNED_BLOCK_SIZES["TPU7x"][
            "lhs_bfloat16_rhs_float4_e2m1fn"
        ]
        shape_table = dtype_table[
            "k-4096_n-8192_g-16_lhs_quant_rhs_bs-256"
            "+k-4096_n-4096_g-16_lhs_quant_rhs_bs-256"
        ]
        expected = (64, 4096, 2048, 4096, 2048, 2, 2)
        for token_count in (1024, 2048, 4096, 8192):
            with self.subTest(token_count=token_count):
                routed_tokens = token_count * 8
                self.assertEqual(shape_table[f"{routed_tokens}_silu"], expected)

        # Do not retune buckets above the bounded 8K-token serving call.
        self.assertEqual(
            shape_table["131072_silu"],
            (128, 4096, 4096, 4096, 4096, 2, 1),
        )

        ep8_shape_table = dtype_table[
            "k-4096_n-8192_g-32_lhs_quant_rhs_bs-256"
            "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-256"
        ]
        self.assertEqual(
            ep8_shape_table["16384_silu"],
            (128, 4096, 4096, 4096, 4096, 2, 1),
        )


if __name__ == "__main__":
    unittest.main()
