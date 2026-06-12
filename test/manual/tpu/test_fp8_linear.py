"""Manual TPU tests for dense FP8 linear."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.fp8_linear import (
    apply_tpu_fp8_linear,
    tpu_w8a8_block_fp8_linear,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.test.test_utils import CustomTestCase


class _FakeBlockFp8Config:
    weight_block_size = [128, 128]
    is_checkpoint_fp8_serialized = True
    use_mxfp8 = False


class TestTpuFp8Linear(CustomTestCase):
    def test_fp8_linear_runs_eager_and_compiled(self):
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
        x_scale = torch.clamp(
            x.abs().amax(dim=1, keepdim=True).float() / fp8_max, min=1e-12
        )
        x_q = torch.clamp(x.float() / x_scale, min=-fp8_max, max=fp8_max).to(
            torch.float8_e4m3fn
        )
        ref_x = x_q.float() * x_scale
        ref_weight = weight.float() * weight_scale.reshape(1, -1).float()
        ref = (ref_x @ ref_weight).to(torch.bfloat16)

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
        torch.testing.assert_close(eager_out.cpu(), ref.cpu(), rtol=0.25, atol=0.25)
        torch.testing.assert_close(compiled_out.cpu(), eager_out.cpu())

    def test_fp8_linear_rejects_unaligned_weight(self):
        x = torch.randn(1, 15, dtype=torch.bfloat16)
        weight = torch.empty(15, 16, dtype=torch.float8_e4m3fn)
        weight_scale = torch.ones(16, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            apply_tpu_fp8_linear(x, weight, weight_scale)

    def test_block_fp8_linear_uses_block_scales_at_runtime(self):
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

        self.assertEqual(layer.weight.shape, (out_size, in_size))
        self.assertEqual(layer.weight_scale_inv.shape, (2, 2))

        x = torch.randn(3, in_size, dtype=torch.bfloat16, device=device)
        out = method.apply(layer, x)
        ref = self._block_fp8_reference(
            x.cpu(), weight_fp8.cpu(), block_scale, [block, block]
        )

        self.assertEqual(out.shape, (3, out_size))
        torch.testing.assert_close(out.cpu().float(), ref, rtol=0.25, atol=0.25)

    def test_tpu_block_fp8_dispatch_wrapper_runs(self):
        device = torch.device("tpu")
        x = torch.randn(2, 256, dtype=torch.bfloat16, device=device)
        weight_fp32 = torch.randn(256, 256, dtype=torch.float32) * 0.1
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        block = 128
        weight_scale = (
            weight_fp32.reshape(2, block, 2, block)
            .abs()
            .amax(dim=(1, 3))
            .clamp(min=1e-12)
            / fp8_max
        )
        weight = (
            (weight_fp32.reshape(2, block, 2, block) / weight_scale[:, None, :, None])
            .clamp(min=-fp8_max, max=fp8_max)
            .reshape(256, 256)
            .to(torch.float8_e4m3fn)
            .to(device)
        )

        out = tpu_w8a8_block_fp8_linear(
            input=x,
            weight=weight,
            block_size=[128, 128],
            weight_scale=weight_scale.to(device),
        )

        def run_block_fp8_linear(x, weight, weight_scale):
            return tpu_w8a8_block_fp8_linear(
                input=x,
                weight=weight,
                block_size=[128, 128],
                weight_scale=weight_scale,
            )

        compiled = torch.compile(
            run_block_fp8_linear,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        compiled_out = compiled(x, weight, weight_scale.to(device))
        ref = self._block_fp8_reference(x.cpu(), weight.cpu(), weight_scale, [128, 128])

        self.assertEqual(out.shape, (2, 256))
        torch.testing.assert_close(out.cpu().float(), ref, rtol=0.25, atol=0.25)
        torch.testing.assert_close(compiled_out.cpu(), out.cpu())

    @staticmethod
    def _block_fp8_reference(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: list[int],
    ) -> torch.Tensor:
        block_n, block_k = block_size
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        output = torch.zeros(x.shape[0], weight.shape[0], dtype=torch.float32)
        for k_block_idx, k_start in enumerate(range(0, weight.shape[1], block_k)):
            x_block = x[:, k_start : k_start + block_k]
            x_scale = torch.clamp(
                x_block.abs().amax(dim=1, keepdim=True).float() / fp8_max, min=1e-12
            )
            x_q = torch.clamp(x_block.float() / x_scale, -fp8_max, fp8_max).to(
                torch.float8_e4m3fn
            )
            raw = x_q.float() @ weight[:, k_start : k_start + block_k].float().t()
            scale = (
                weight_scale[:, k_block_idx]
                .repeat_interleave(block_n)
                .reshape(1, weight.shape[0])
            )
            output += raw * x_scale * scale.float()
        return output


if __name__ == "__main__":
    unittest.main()
