# SPDX-License-Identifier: Apache-2.0
"""On-device check that the TPU RoPE pair swap matches the CUDA kernel's numerics."""

import importlib.util
import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.utils import is_in_ci
from sglang_tpu.multimodal_gen.rope import (
    _build_pair_swap_matrix,
    apply_rotary_embedding_pair_swap,
)

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)


class TestRopePairSwapTpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_tpu") is None:
            if is_in_ci():
                raise RuntimeError("TPU CI requires torch_tpu and a TPU device")
            raise unittest.SkipTest("torch_tpu is not installed")
        from sglang_tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu  # noqa: F401

        # Do not turn initialization/lowering failures into a successful skip.
        torch.zeros(1, device="tpu").cpu()

    @staticmethod
    def _cuda_reference(x, cos, sin, interleaved):
        # The CUDA Triton kernel: float32 tables and math, one rounding to x.dtype.
        if interleaved:
            cos, sin = cos[..., ::2], sin[..., ::2]
        x = x.cpu().float()
        c, s = cos.cpu().float().unsqueeze(-2), sin.cpu().float().unsqueeze(-2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)

    def _assert_matches_cuda(self, actual, expected_fp32):
        expected = expected_fp32.bfloat16().float()
        actual = actual.cpu().float()
        # Only float32 rounding-order ties (FMA or not) may flip a bf16 rounding.
        ulp = expected.abs().clamp_min(2**-126) * 2**-7
        self.assertTrue(bool(((actual - expected).abs() <= ulp).all()))
        self.assertLess((actual != expected).float().mean().item(), 1e-3)

    def test_bf16_eager_matches_the_cuda_kernel(self):
        torch.manual_seed(0)
        for head_dim in (64, 128):
            for interleaved in (False, True):
                with self.subTest(head_dim=head_dim, interleaved=interleaved):
                    x = (torch.randn(33, 4, head_dim) * 3).bfloat16().to("tpu")
                    angles = torch.randn(33, head_dim // 2) * 100
                    cos, sin = angles.cos(), angles.sin()
                    if interleaved:
                        cos = cos.repeat_interleave(2, dim=-1)
                        sin = sin.repeat_interleave(2, dim=-1)
                    cos, sin = cos.to("tpu"), sin.to("tpu")
                    actual = apply_rotary_embedding_pair_swap(x, cos, sin, interleaved)
                    self.assertEqual(actual.device.type, "tpu")
                    self._assert_matches_cuda(
                        actual, self._cuda_reference(x, cos, sin, interleaved)
                    )

    def test_bf16_compiled_matches_the_cuda_kernel(self):
        # XLA may fuse the float32 math (FMA or not) under torch.compile.
        import torch_tpu._internal.compile  # noqa: F401

        torch.manual_seed(0)
        head_dim = 128
        x = torch.randn(33, 4, head_dim).bfloat16().to("tpu")
        angles = torch.randn(33, head_dim // 2)
        cos, sin = angles.cos().to("tpu"), angles.sin().to("tpu")

        # x @ R is a signed permutation: exact in bf16.
        swap = torch.compile(
            lambda x: torch.matmul(
                x, _build_pair_swap_matrix(head_dim, x.dtype, x.device)
            ),
            backend="tpu",
            dynamic=False,
        )
        x_cpu = x.cpu()
        expected_swap = torch.stack((-x_cpu[..., 1::2], x_cpu[..., ::2]), dim=-1)
        torch.testing.assert_close(
            swap(x).cpu(), expected_swap.flatten(-2), rtol=0, atol=0
        )

        pair_swap = torch.compile(
            apply_rotary_embedding_pair_swap, backend="tpu", dynamic=False
        )
        self._assert_matches_cuda(
            pair_swap(x, cos, sin, False), self._cuda_reference(x, cos, sin, False)
        )


if __name__ == "__main__":
    unittest.main()
