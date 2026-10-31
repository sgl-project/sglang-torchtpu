# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sglang_tpu's diffusion RoPE pair swap (sglang_tpu.multimodal_gen.rope)."""

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    apply_rotary_embedding_native,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.multimodal_gen import rope  # noqa: E402


def _rope_inputs(dtype, head_dim, interleaved):
    x = torch.randn(33, 4, head_dim, dtype=dtype)
    angles = torch.randn(33, head_dim // 2)
    cos, sin = angles.cos(), angles.sin()
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    return x, cos, sin


class TestRopePairSwap(CustomTestCase):
    def test_bf16_pair_swap_matches_the_cuda_kernel(self):
        # The CUDA kernel's contract: float32 tables and math, one rounding.
        torch.manual_seed(0)
        for head_dim in (64, 128):
            for interleaved in (False, True):
                with (
                    self.subTest(head_dim=head_dim, interleaved=interleaved),
                    patch.object(
                        rope, "_pair_swap_matrix", wraps=rope._pair_swap_matrix
                    ) as pair_swap_matrix,
                ):
                    x, cos, sin = _rope_inputs(torch.bfloat16, head_dim, interleaved)
                    self.assertTrue(
                        torch.equal(
                            rope.apply_rotary_embedding_pair_swap(
                                x, cos, sin, interleaved
                            ),
                            apply_rotary_embedding_native(
                                x.float(), cos, sin, interleaved
                            ).to(x.dtype),
                        )
                    )
                    pair_swap_matrix.assert_called_once()

    def test_other_dtypes_keep_the_strided_form(self):
        # TPU float16/float32 matmuls are not exact, so only bf16 takes the swap.
        torch.manual_seed(0)
        with patch.object(
            rope, "_pair_swap_matrix", side_effect=AssertionError("took the matmul")
        ):
            for dtype in (torch.float16, torch.float32):
                for interleaved in (False, True):
                    with self.subTest(dtype=dtype, interleaved=interleaved):
                        x, cos, sin = _rope_inputs(dtype, 128, interleaved)
                        self.assertTrue(
                            torch.equal(
                                rope.apply_rotary_embedding_pair_swap(
                                    x, cos, sin, interleaved
                                ),
                                apply_rotary_embedding_native(x, cos, sin, interleaved),
                            )
                        )

    def test_matrix_is_built_once_in_eager_and_traced_when_compiling(self):
        rope._PAIR_SWAP_MATRICES.clear()
        x = torch.arange(8.0).bfloat16()
        rot = rope._pair_swap_matrix(8, torch.bfloat16, "cpu")
        self.assertIs(rope._pair_swap_matrix(8, torch.bfloat16, "cpu"), rot)
        swapped = torch.tensor([-1.0, 0, -3, 2, -5, 4, -7, 6]).bfloat16()
        self.assertTrue(torch.equal(x @ rot, swapped))

        rope._PAIR_SWAP_MATRICES.clear()
        compiled = torch.compile(
            lambda t: t @ rope._pair_swap_matrix(8, t.dtype, t.device),
            backend="eager",
            fullgraph=True,
        )
        self.assertTrue(torch.equal(compiled(x), swapped))
        self.assertEqual(rope._PAIR_SWAP_MATRICES, {})

    def test_install_rebinds_the_rotary_helper(self):
        # Stub modules: the real package imports the whole rotary stack.
        package_name = "sglang.multimodal_gen.runtime.layers.rotary_embedding"
        utils = ModuleType(f"{package_name}.utils")
        utils.apply_rotary_embedding = apply_rotary_embedding_native
        package = ModuleType(package_name)
        package.utils = utils
        with patch.dict(sys.modules, {package_name: package, utils.__name__: utils}):
            rope.install()
        self.assertIs(
            utils.apply_rotary_embedding, rope.apply_rotary_embedding_pair_swap
        )


if __name__ == "__main__":
    unittest.main()
