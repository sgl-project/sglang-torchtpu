"""Unit tests for the sequence-parallel hierarchical reduce-scatter seam.

sglang_tpu/sp_hier_reduce_scatter.py arms kernels/hier_all_reduce.py's
row-streaming reduce-scatter for the reduce-scatter that ends every stage of an
input-scattered batch. These CPU tests pin which shapes may take it (the kernel
asserts on anything else) and that the hook keeps XLA's reduce-scatter unless
the kernel was armed; the kernel itself is checked on TPU by
test/registered/tpu/unit/test_sp_hier_reduce_scatter_kernel.py.
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import sp_hier_reduce_scatter as hier
from sglang_tpu.hooks.sp_hier_reduce_scatter import sp_reduce_scatter

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestShapeSupport(CustomTestCase):
    def test_scratch_matches_kernel_scratch_shapes(self):
        # Two [3, 4, 128, H] + [3, 2, 2, 64, H] + [3, 2, 64, H] bf16 buffers.
        h = 4096
        expected = (2 * 3 * 4 * 128 + 3 * 2 * 2 * 64 + 3 * 2 * 64) * h * 2
        self.assertEqual(hier.scratch_bytes(h, 2), expected)

    def test_rows_must_split_into_128_row_blocks_per_device(self):
        for rows in (1024, 2048, 4096, 8192):
            self.assertTrue(hier.shape_supported(rows, 4096, torch.bfloat16), rows)
        for rows in (8, 512, 1000, 1536, 3072 + 8):
            self.assertFalse(hier.shape_supported(rows, 4096, torch.bfloat16), rows)

    def test_dtype_and_hidden(self):
        self.assertFalse(hier.shape_supported(1024, 4096, torch.float32))
        self.assertFalse(hier.shape_supported(1024, 4000, torch.bfloat16))
        # 6144 still fits the kernel's 56 MiB VMEM limit; 8192 does not.
        self.assertTrue(hier.shape_supported(1024, 6144, torch.bfloat16))
        self.assertFalse(hier.shape_supported(1024, 8192, torch.bfloat16))

    def test_not_eligible_until_armed(self):
        x = torch.zeros(1024, 4096, dtype=torch.bfloat16)
        with mock.patch.object(hier, "_READY", False):
            self.assertFalse(hier.eligible(x))
        with mock.patch.object(hier, "_READY", True):
            self.assertTrue(hier.eligible(x))


class TestHook(CustomTestCase):
    def test_unarmed_keeps_original(self):
        original = mock.Mock()
        x = torch.randn(1024, 256).to(torch.bfloat16)
        shard = x.tensor_split(8)[0]
        with mock.patch.object(hier, "_READY", False):
            sp_reduce_scatter(original, shard, x)
        original.assert_called_once_with(shard, x)

    def test_ineligible_shape_keeps_original(self):
        original = mock.Mock()
        x = torch.randn(512, 256).to(torch.bfloat16)
        with mock.patch.object(hier, "_READY", True):
            sp_reduce_scatter(original, x.tensor_split(8)[0], x)
        original.assert_called_once()

    def test_armed_writes_kernel_output_into_shard(self):
        original = mock.Mock()
        x = torch.randn(1024, 256).to(torch.bfloat16)
        shard = x.tensor_split(8)[0]
        out = torch.full((128, 256), 3.0, dtype=torch.bfloat16)
        with (
            mock.patch.object(hier, "_READY", True),
            mock.patch.object(hier, "_OP", lambda t: out),
        ):
            sp_reduce_scatter(original, shard, x)
        original.assert_not_called()
        self.assertTrue(torch.equal(x[:128], out))


if __name__ == "__main__":
    unittest.main()
