"""CPU tests for MXINT8 geometry and physical storage accounting."""

import itertools
import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.mem_cache.kv_mxint8 import (
    EXPONENT_BIAS,
    EXPONENT_PAD_BYTE,
    GROUP_SIZE,
    QUANT_DTYPE,
    ROW_ALIGNMENT,
    _validate_geometry,
    cache_bytes_per_token,
    merged_cache_shape,
    merged_rows,
    payload_rows,
    scale_rows,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _align(rows: int) -> int:
    return math.ceil(rows / ROW_ALIGNMENT) * ROW_ALIGNMENT


class TestMXINT8Geometry(CustomTestCase):
    def test_valid_geometry_matrix(self):
        for group_size, head_num, head_dim, page_size, num_blocks in itertools.product(
            (16, 32, 64), (1, 3, 4, 5, 8, 9), (64, 128), (128, 256), (1, 3)
        ):
            with self.subTest(
                group=group_size,
                heads=head_num,
                dim=head_dim,
                page=page_size,
                blocks=num_blocks,
            ):
                self.assertIsNone(
                    _validate_geometry(head_num, head_dim, page_size, group_size)
                )
                payload = 2 * head_num * _align(head_dim)
                scales = 2 * head_num * head_dim // group_size
                self.assertEqual(payload_rows(head_num, head_dim), payload)
                self.assertEqual(scale_rows(head_num, head_dim, group_size), scales)
                rows = payload + _align(scales)
                self.assertEqual(merged_rows(head_num, head_dim, group_size), rows)
                shape = merged_cache_shape(
                    num_blocks, head_num, head_dim, page_size, group_size
                )
                self.assertEqual(shape, (num_blocks, rows, page_size))
                # The merged page carries the payload and the exponent bytes in
                # one INT8 tensor, so its size is the whole physical footprint.
                cache = torch.empty(shape, dtype=QUANT_DTYPE)
                self.assertEqual(
                    cache_bytes_per_token(head_num, head_dim, page_size, group_size)
                    * num_blocks
                    * page_size,
                    cache.nbytes,
                )

    def test_default_group_and_exponent_row_boundaries(self):
        self.assertEqual(GROUP_SIZE, 32)
        self.assertEqual(ROW_ALIGNMENT, 32)
        self.assertEqual(EXPONENT_BIAS, 127)
        # Padding exponent bytes decode to 2**0 so untouched rows are neutral.
        self.assertEqual(EXPONENT_PAD_BYTE, EXPONENT_BIAS)
        self.assertIs(QUANT_DTYPE, torch.int8)
        for head_num, expected_scale_rows, expected_bytes in (
            (7, 32, 928),
            (8, 32, 1056),
            (9, 64, 1216),
        ):
            with self.subTest(heads=head_num):
                self.assertEqual(
                    merged_cache_shape(2, head_num, 64, 128),
                    (2, 2 * head_num * 64 + expected_scale_rows, 128),
                )
                self.assertEqual(
                    cache_bytes_per_token(head_num, 64, 128), expected_bytes
                )

    def test_group32_supports_other_divisible_head_dimensions(self):
        for head_dim, expected_bytes in ((32, 96), (96, 224), (160, 352)):
            with self.subTest(dim=head_dim):
                self.assertEqual(
                    cache_bytes_per_token(1, head_dim, 128), expected_bytes
                )

    def test_invalid_geometry_matrix(self):
        invalid = [
            (1, 64, 128, group, "group_size must be") for group in (0, 8, 24, 128)
        ]
        invalid += [
            (heads, dim, 128, 32, "positive heads and head_dim divisible")
            for heads, dim in ((0, 64), (-1, 64), (1, 0), (1, -32), (1, 63))
        ]
        invalid += [
            (1, dim, 128, group, "requires head_dim 64 or 128")
            for group, dim in ((16, 16), (16, 32), (16, 96), (64, 192), (64, 256))
        ]
        invalid += [
            (1, 64, page, 32, "power of two and at least 128")
            for page in (-128, 0, 1, 64, 127, 129, 192)
        ]
        for heads, dim, page, group, message in invalid:
            for function in (_validate_geometry, cache_bytes_per_token):
                with self.subTest(
                    function=function.__name__, geometry=(heads, dim, page, group)
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        function(heads, dim, page, group)
            with self.subTest(
                function="merged_cache_shape", geometry=(heads, dim, page, group)
            ):
                with self.assertRaisesRegex(ValueError, message):
                    merged_cache_shape(1, heads, dim, page, group)

    def test_requires_dummy_page_even_without_allocatable_blocks(self):
        for num_blocks in (0, -1):
            with (
                self.subTest(blocks=num_blocks),
                self.assertRaisesRegex(ValueError, "num_blocks must be positive"),
            ):
                merged_cache_shape(num_blocks, 1, 64, 128)
        self.assertEqual(merged_cache_shape(1, 1, 64, 128), (1, 160, 128))


if __name__ == "__main__":
    unittest.main()
