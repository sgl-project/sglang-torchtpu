"""Manual TPU tests for dense FP8 linear."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.fp8_linear import (
    _requantize_tpu_block_fp8_to_channelwise,
    _small_m_supports,
    apply_tpu_fp8_linear,
    compute_tpu_fp8_linear_scale_shape,
    prepare_tpu_fp8_linear_weights,
    register_fp8_linear_small_m_op,
    tpu_w8a8_block_fp8_linear,
)
from sglang_tpu.fp8_moe import (
    _requantize_tpu_fp8_moe_block_to_channelwise,
)
from sglang_tpu.quantization.dense_fp8 import TpuFp8LinearMethod

_SMALL_M_KERNEL = tpu_envs.SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL


class _FakeBlockFp8Config:
    weight_block_size = [128, 128]
    is_checkpoint_fp8_serialized = True
    use_mxfp8 = False


class TestTpuFp8Linear(CustomTestCase):
    def setUp(self):
        # publish() resolves the record; seed the leaf through the context.
        override = get_context().override_server_args(tpu_fp8_runtime="blockwise")
        override.install()
        self.addCleanup(override.restore)

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

    def _small_m_inputs(self, gen, tokens, k, n):
        x = torch.randn(tokens, k, generator=gen).bfloat16().to("tpu")
        w = (torch.randn(k, n, generator=gen) * 0.5).to(torch.float8_e4m3fn).to("tpu")
        ws = (torch.rand(n, generator=gen) * 0.01 + 1e-3).to("tpu")
        return x, w, ws

    def _xla_path(self, x, w, ws):
        with _SMALL_M_KERNEL.override(False):
            return apply_tpu_fp8_linear(x, w, ws).cpu()

    def test_fp8_linear_small_m_kernel_matches_xla_path(self):
        register_fp8_linear_small_m_op()
        gen = torch.Generator().manual_seed(3)
        # 5 tokens: a row block that is not a sublane multiple, with no pad.
        for tokens in (1, 5, 16):
            for k, n in ((4096, 512), (256, 4096), (4096, 2048), (1536, 4096)):
                with self.subTest(tokens=tokens, k=k, n=n):
                    x, w, ws = self._small_m_inputs(gen, tokens, k, n)
                    expected = self._xla_path(x, w, ws)
                    with _SMALL_M_KERNEL.override(True):
                        actual = apply_tpu_fp8_linear(x, w, ws).cpu()
                    self.assertTrue(torch.equal(actual, expected))

    def test_fp8_linear_small_m_kernel_scale_shapes(self):
        register_fp8_linear_small_m_op()
        gen = torch.Generator().manual_seed(4)
        x, w, ws = self._small_m_inputs(gen, 4, 1536, 4096)
        for name, scale in (
            ("per-channel [N, 1]", ws.reshape(-1, 1)),
            ("per-tensor scalar", ws[0].clone()),
        ):
            with self.subTest(scale=name):
                expected = self._xla_path(x, w, scale)
                with _SMALL_M_KERNEL.override(True):
                    actual = apply_tpu_fp8_linear(x, w, scale).cpu()
                self.assertTrue(torch.equal(actual, expected))

    def test_fp8_linear_small_m_kernel_compiled(self):
        register_fp8_linear_small_m_op()
        gen = torch.Generator().manual_seed(5)

        def run_small_m(x, weight, weight_scale):
            return apply_tpu_fp8_linear(x, weight, weight_scale)

        with _SMALL_M_KERNEL.override(True):
            compiled = torch.compile(
                run_small_m, backend="tpu", fullgraph=True, dynamic=False
            )
            for tokens in (1, 16):
                with self.subTest(tokens=tokens):
                    x, w, ws = self._small_m_inputs(gen, tokens, 4096, 2048)
                    actual = compiled(x, w, ws).cpu()
                    self.assertTrue(torch.equal(actual, self._xla_path(x, w, ws)))

    def test_fp8_linear_small_m_declines_unsupported_shapes(self):
        self.assertTrue(_small_m_supports(16, 4096, 512))
        self.assertFalse(_small_m_supports(17, 4096, 512))
        self.assertFalse(_small_m_supports(1, 4096, 100))
        self.assertFalse(_small_m_supports(1, 4000, 512))
        # No N tile whose double-buffered FP8 block fits the VMEM budget.
        self.assertFalse(_small_m_supports(1, 131072, 128))

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

        method = TpuFp8LinearMethod(_FakeBlockFp8Config())
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

    def test_block_fp8_linear_rejects_non_block_aligned_k(self):
        x = torch.randn(2, 192, dtype=torch.bfloat16)
        weight = torch.empty(256, 192, dtype=torch.float8_e4m3fn)
        weight_scale = torch.ones(2, 2, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "divisible by block_size"):
            tpu_w8a8_block_fp8_linear(
                input=x,
                weight=weight,
                block_size=[128, 128],
                weight_scale=weight_scale,
            )

    def test_block_fp8_linear_rejects_mismatched_scale_shape(self):
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        weight = torch.empty(256, 256, dtype=torch.float8_e4m3fn)
        weight_scale = torch.ones(2, 1, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "weight_scale shape mismatch"):
            tpu_w8a8_block_fp8_linear(
                input=x,
                weight=weight,
                block_size=[128, 128],
                weight_scale=weight_scale,
            )

    def test_dense_block_to_channelwise_releases_block_scale(self):
        out_size = 256
        in_size = 256
        block_size = [128, 128]
        weight, weight_scale = self._make_block_fp8_weight(
            (out_size, in_size), block_size
        )
        expected = block_quant_dequant(weight, weight_scale, block_size, torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            weight_scale.clone(), requires_grad=False
        )
        layer.input_scale = None
        stale_scale = layer.weight_scale_inv

        _requantize_tpu_block_fp8_to_channelwise(layer, block_size)

        self.assertIsNone(layer.weight_scale_inv)
        self.assertEqual(stale_scale.data.numel(), 0)
        self.assertEqual(layer.weight.shape, (in_size, out_size))
        self.assertEqual(layer.weight_scale.shape, (out_size, 1))
        recovered = (layer.weight.float() * layer.weight_scale.reshape(1, -1)).t()
        torch.testing.assert_close(recovered, expected, rtol=0.1, atol=0.1)

    def test_deepseek_v4_pro_shared_experts_subblock_tp16_sharding(self):
        """Verify sub-block TP=16 sharding for DeepSeek-V4-Pro shared_experts (3072 // 16 = 192)."""
        block_size = [128, 128]

        # 1. MergedColumnParallelLinear (gate_up_proj): logical_widths=[192, 192], in_size=256
        gate_w, gate_s = self._make_block_fp8_weight((384, 256), block_size)
        up_w, up_s = self._make_block_fp8_weight((384, 256), block_size)
        gate_deq = block_quant_dequant(gate_w, gate_s, block_size, torch.float32)
        up_deq = block_quant_dequant(up_w, up_s, block_size, torch.float32)

        for tp_rank in (0, 1):
            start = tp_rank * 192
            end = start + 192
            s_start = start // 128
            s_end = (end + 127) // 128
            layer = torch.nn.Module()
            layer.tp_rank = tp_rank
            layer.logical_widths = [192, 192]
            scale_rows, scale_cols = compute_tpu_fp8_linear_scale_shape(
                layer,
                input_size=256,
                input_size_per_partition=256,
                output_size=768,
                output_size_per_partition=384,
                output_partition_sizes=[192, 192],
                block_n=128,
                block_k=128,
            )
            self.assertEqual((scale_rows, scale_cols), (2 * (s_end - s_start), 2))
            layer.weight = torch.nn.Parameter(
                torch.cat([gate_w[start:end], up_w[start:end]], dim=0),
                requires_grad=False,
            )
            layer.weight_scale_inv = torch.nn.Parameter(
                torch.cat([gate_s[s_start:s_end], up_s[s_start:s_end]], dim=0),
                requires_grad=False,
            )
            layer.input_scale = None

            prepare_tpu_fp8_linear_weights(layer, weight_block_size=block_size)
            recovered = (layer.weight.float() * layer.weight_scale.reshape(1, -1)).t()
            expected = torch.cat([gate_deq[start:end], up_deq[start:end]], dim=0)
            torch.testing.assert_close(recovered, expected, rtol=0.1, atol=0.1)

        # 2. RowParallelLinear (down_proj): out_size=256, in_size=192
        down_w, down_s = self._make_block_fp8_weight((256, 384), block_size)
        down_deq = block_quant_dequant(down_w, down_s, block_size, torch.float32)
        for tp_rank in (0, 1):
            start = tp_rank * 192
            end = start + 192
            s_start = start // 128
            s_end = (end + 127) // 128
            layer = torch.nn.Module()
            layer.tp_rank = tp_rank
            layer.logical_widths = [256]
            scale_rows, scale_cols = compute_tpu_fp8_linear_scale_shape(
                layer,
                input_size=384,
                input_size_per_partition=192,
                output_size=256,
                output_size_per_partition=256,
                output_partition_sizes=[256],
                block_n=128,
                block_k=128,
            )
            self.assertEqual((scale_rows, scale_cols), (2, s_end - s_start))
            layer.weight = torch.nn.Parameter(
                down_w[:, start:end].contiguous(), requires_grad=False
            )
            layer.weight_scale_inv = torch.nn.Parameter(
                down_s[:, s_start:s_end].contiguous(), requires_grad=False
            )
            layer.input_scale = None

            prepare_tpu_fp8_linear_weights(layer, weight_block_size=block_size)
            recovered = (layer.weight.float() * layer.weight_scale.reshape(1, -1)).t()
            expected = down_deq[:, start:end]
            torch.testing.assert_close(recovered, expected, rtol=0.1, atol=0.1)

    def test_moe_block_to_channelwise_releases_block_scales(self):
        num_experts = 2
        out_size = 256
        in_size = 256
        block_size = [128, 128]
        w13, w13_scale = self._make_block_fp8_weight(
            (num_experts, out_size, in_size), block_size
        )
        w2, w2_scale = self._make_block_fp8_weight(
            (num_experts, out_size, in_size), block_size
        )
        expected_w13 = block_quant_dequant(w13, w13_scale, block_size, torch.float32)
        expected_w2 = block_quant_dequant(w2, w2_scale, block_size, torch.float32)

        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            w13_scale.clone(), requires_grad=False
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            w2_scale.clone(), requires_grad=False
        )
        stale_w13_scale = layer.w13_weight_scale_inv
        stale_w2_scale = layer.w2_weight_scale_inv

        _requantize_tpu_fp8_moe_block_to_channelwise(layer, block_size)

        self.assertIsNone(layer.w13_weight_scale_inv)
        self.assertIsNone(layer.w2_weight_scale_inv)
        self.assertEqual(stale_w13_scale.data.numel(), 0)
        self.assertEqual(stale_w2_scale.data.numel(), 0)
        self.assertEqual(layer.w13_weight_scale.shape, (num_experts, out_size))
        self.assertEqual(layer.w2_weight_scale.shape, (num_experts, out_size))
        recovered_w13 = layer.w13_weight.float() * layer.w13_weight_scale.unsqueeze(-1)
        recovered_w2 = layer.w2_weight.float() * layer.w2_weight_scale.unsqueeze(-1)
        torch.testing.assert_close(recovered_w13, expected_w13, rtol=0.1, atol=0.1)
        torch.testing.assert_close(recovered_w2, expected_w2, rtol=0.1, atol=0.1)

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

    @staticmethod
    def _make_block_fp8_weight(
        shape: tuple[int, ...],
        block_size: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_n, block_k = block_size
        *prefix, n, k = shape
        assert n % block_n == 0
        assert k % block_k == 0
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        weight = torch.randn(shape, dtype=torch.float32) * 0.1
        blocked = weight.reshape(*prefix, n // block_n, block_n, k // block_k, block_k)
        scale = blocked.abs().amax(dim=(-3, -1)).clamp(min=1e-12) / fp8_max
        weight_q = (
            (blocked / scale[..., :, None, :, None])
            .clamp(min=-fp8_max, max=fp8_max)
            .reshape(shape)
            .to(torch.float8_e4m3fn)
        )
        return weight_q, scale


if __name__ == "__main__":
    unittest.main()
