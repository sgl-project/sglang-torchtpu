"""Manual TPU tests for dense FP8 linear."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.fp8_linear import (
    apply_tpu_fp8_linear,
    prebuild_tpu_fp8_linear_op,
)
from sglang.srt.layers.quantization.fp8 import (
    Fp8LinearMethod,
    _requantize_tpu_fp8_block_linear_weight,
)
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

    def test_partial_tp_block_fp8_linear_requantizes(self):
        out_size = 12
        in_size = 12
        block = 4
        tp_rank = 1
        local_width = 6
        local_input_width = 6
        global_output_start = tp_rank * local_width
        global_input_start = tp_rank * local_input_width
        weight_fp32 = torch.randn(out_size, in_size, dtype=torch.float32) * 0.1
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        block_scale = (
            weight_fp32.reshape(out_size // block, block, in_size // block, block)
            .abs()
            .amax(dim=(1, 3))
            .clamp(min=1e-12)
            / fp8_max
        )
        weight_fp8 = (
            (
                weight_fp32.reshape(out_size // block, block, in_size // block, block)
                / block_scale[:, None, :, None]
            )
            .clamp(min=-fp8_max, max=fp8_max)
            .reshape(out_size, in_size)
            .to(torch.float8_e4m3fn)
        )

        local_weight = weight_fp8[
            global_output_start : global_output_start + local_width,
            global_input_start : global_input_start + local_input_width,
        ]
        n_scale_start = global_output_start // block
        n_scale_end = (global_output_start + local_width + block - 1) // block
        k_scale_start = global_input_start // block
        k_scale_end = (global_input_start + local_input_width + block - 1) // block
        local_scale = block_scale[n_scale_start:n_scale_end, k_scale_start:k_scale_end]

        qweight, weight_scale = _requantize_tpu_fp8_block_linear_weight(
            local_weight,
            local_scale,
            [block, block],
            logical_widths=[local_width],
            tp_rank=tp_rank,
            input_offset=global_input_start,
        )
        dequant = qweight.t().float() * weight_scale[:, None]
        ref = weight_fp32[
            global_output_start : global_output_start + local_width,
            global_input_start : global_input_start + local_input_width,
        ]

        self.assertEqual(qweight.shape, (local_input_width, local_width))
        self.assertEqual(weight_scale.shape, (local_width,))
        torch.testing.assert_close(dequant, ref, rtol=0.2, atol=0.2)

    def test_row_parallel_block_fp8_linear_requantizes(self):
        out_size = 12
        in_size = 12
        block = 4
        tp_rank = 1
        local_input_width = 6
        global_input_start = tp_rank * local_input_width
        weight_fp32 = torch.randn(out_size, in_size, dtype=torch.float32) * 0.1
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        block_scale = (
            weight_fp32.reshape(out_size // block, block, in_size // block, block)
            .abs()
            .amax(dim=(1, 3))
            .clamp(min=1e-12)
            / fp8_max
        )
        weight_fp8 = (
            (
                weight_fp32.reshape(out_size // block, block, in_size // block, block)
                / block_scale[:, None, :, None]
            )
            .clamp(min=-fp8_max, max=fp8_max)
            .reshape(out_size, in_size)
            .to(torch.float8_e4m3fn)
        )

        local_weight = weight_fp8[
            :, global_input_start : global_input_start + local_input_width
        ]
        k_scale_start = global_input_start // block
        k_scale_end = (global_input_start + local_input_width + block - 1) // block
        local_scale = block_scale[:, k_scale_start:k_scale_end]

        qweight, weight_scale = _requantize_tpu_fp8_block_linear_weight(
            local_weight,
            local_scale,
            [block, block],
            logical_widths=[out_size],
            tp_rank=0,
            input_offset=global_input_start,
        )
        dequant = qweight.t().float() * weight_scale[:, None]
        ref = weight_fp32[
            :, global_input_start : global_input_start + local_input_width
        ]

        self.assertEqual(qweight.shape, (local_input_width, out_size))
        self.assertEqual(weight_scale.shape, (out_size,))
        torch.testing.assert_close(dequant, ref, rtol=0.2, atol=0.2)


if __name__ == "__main__":
    unittest.main()
