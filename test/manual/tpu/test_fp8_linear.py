"""Manual TPU tests for dense FP8 linear."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.fp8_linear import (
    apply_tpu_fp8_linear,
    prebuild_tpu_fp8_linear_op,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.test.test_utils import CustomTestCase


class _FakeBlockFp8Config:
    weight_block_size = [128, 128]
    is_checkpoint_fp8_serialized = True
    use_mxfp8 = False


class TestTpuFp8Linear(CustomTestCase):
    def test_fp8_linear_runs_eager_and_compiled(self):
        prebuild_tpu_fp8_linear_op()

        device = torch.device("tpu")
        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        weight_fp32 = torch.randn(128, 32, dtype=torch.float32)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        weight_scale = (
            weight_fp32.abs().amax(dim=0).clamp(min=1e-12) / fp8_max
        ).float()
        weight = (
            (weight_fp32 / weight_scale)
            .clamp(min=-fp8_max, max=fp8_max)
            .to(torch.float8_e4m3fn)
            .to(device)
        )
        weight_scale = weight_scale.to(device)

        eager_out = apply_tpu_fp8_linear(x, weight, weight_scale)

        def run_fp8_linear(x, weight, weight_scale):
            return apply_tpu_fp8_linear(x, weight, weight_scale)

        compiled = torch.compile(
            run_fp8_linear,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        compiled_out = compiled(x, weight, weight_scale)

        self.assertEqual(eager_out.shape, (4, 32))
        self.assertEqual(eager_out.dtype, torch.bfloat16)
        torch.testing.assert_close(compiled_out.cpu(), eager_out.cpu())

    def test_block_fp8_linear_requantizes_and_runs(self):
        prebuild_tpu_fp8_linear_op()

        device = torch.device("tpu")
        out_size = 256
        in_size = 256
        block = 128
        weight_fp32 = torch.randn(out_size, in_size, dtype=torch.float32) * 0.1
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        block_scale = (
            weight_fp32.reshape(2, block, 2, block)
            .abs()
            .amax(dim=(1, 3))
            .clamp(min=1e-12)
            / fp8_max
        )
        weight_fp8 = (
            (weight_fp32.reshape(2, block, 2, block) / block_scale[:, None, :, None])
            .clamp(min=-fp8_max, max=fp8_max)
            .reshape(out_size, in_size)
            .to(torch.float8_e4m3fn)
            .to(device)
        )

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            block_scale.to(device), requires_grad=False
        )
        layer.input_scale = None

        method = Fp8LinearMethod(_FakeBlockFp8Config())
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (in_size, out_size))
        self.assertEqual(layer.weight_scale.shape, (out_size,))

        x = torch.randn(3, in_size, dtype=torch.bfloat16, device=device)
        out = method.apply(layer, x)
        ref = x.cpu().float() @ weight_fp32.t()

        self.assertEqual(out.shape, (3, out_size))
        torch.testing.assert_close(out.cpu().float(), ref, rtol=0.25, atol=0.25)


if __name__ == "__main__":
    unittest.main()
