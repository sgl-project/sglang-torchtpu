# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for plugin MoE selection, checkpoint storage and post-load order."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.plugins.oot_classes import clear_oot_replacements, oot_base_class
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.quantization.moe import (
    TpuCompressedTensorsConfig,
    TpuCompressedTensorsWNA16MoE,
    TpuFp8MoEMethod,
    TpuMxfp4MoEMethod,
    TpuUnquantizedFusedMoEMethod,
    _quantize_fp8_weight,
    register_moe_quantization_methods,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTpuMoeMethods(CustomTestCase):
    def setUp(self):
        clear_oot_replacements()
        runtime = SimpleNamespace(
            moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="bf16")
        )
        for target, value in (
            ("sglang.srt.layers.quantization.mxfp4.get_exec", runtime),
            ("sglang_tpu.fp8_moe.get_parallel", SimpleNamespace(pp_size=1)),
        ):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(clear_oot_replacements)
        flags = get_flags().moe
        backend = flags.runner_backend
        flags.runner_backend = MoeRunnerBackend.PALLAS_TPU
        self.addCleanup(setattr, flags, "runner_backend", backend)

    @staticmethod
    def _ct_config():
        return TpuCompressedTensorsConfig({}, [], "pack-quantized", {}, [])

    @staticmethod
    def _int4(**kwargs):
        return QuantizationArgs(
            num_bits=4, strategy=QuantizationStrategy.GROUP, group_size=128, **kwargs
        )

    def test_registered_methods_cover_direct_construction_and_loader_identity(self):
        config = Fp8Config(is_checkpoint_fp8_serialized=True)
        register_moe_quantization_methods()
        constructors = (
            (UnquantizedFusedMoEMethod, (), TpuUnquantizedFusedMoEMethod),
            (Fp8MoEMethod, (config,), TpuFp8MoEMethod),
            (Mxfp4MoEMethod, ("experts",), TpuMxfp4MoEMethod),
            (
                CompressedTensorsWNA16MoE,
                (self._ct_config(), self._int4()),
                TpuCompressedTensorsWNA16MoE,
            ),
        )
        for base, args, replacement in constructors:
            with self.subTest(method=base.__name__):
                obj = base(*args)
                self.assertIs(type(obj), replacement)
                self.assertIs(oot_base_class(type(obj)), base)
        clear_oot_replacements()
        self.assertIs(type(UnquantizedFusedMoEMethod()), UnquantizedFusedMoEMethod)
        self.assertIs(type(Fp8MoEMethod(config)), Fp8MoEMethod)

    def test_each_moe_method_registers_its_op_before_model_tracing(self):
        register_moe_quantization_methods()
        methods = (
            UnquantizedFusedMoEMethod(),
            Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True)),
            Mxfp4MoEMethod("experts"),
            CompressedTensorsWNA16MoE(self._ct_config(), self._int4()),
        )
        for method in methods:
            config = MoeRunnerConfig()
            with (
                self.subTest(method=type(method).__name__),
                patch(
                    "sglang_tpu.moe.pallas.fused_moe_op._build_fused_moe_op"
                ) as register_op,
            ):
                method.create_moe_runner(torch.nn.Module(), config)
                register_op.assert_called_once_with()
                self.assertIs(method.moe_runner_config, config)

    def test_square_transpose_retains_live_parameter_storage(self):
        method = TpuUnquantizedFusedMoEMethod()
        layer = torch.nn.Module()
        w13 = torch.arange(2 * 256 * 128).reshape(2, 256, 128).float()
        w2 = torch.arange(2 * 128 * 128).reshape(2, 128, 128).float()
        layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
        method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.w13_weight, w13.transpose(1, 2))
        torch.testing.assert_close(layer.w2_weight, w2.transpose(1, 2))
        self.assertEqual(layer.w2_weight.untyped_storage().nbytes(), w2.numel() * 4)

    def test_mxfp4_checkpoint_shard_padding_and_loader_metadata(self):
        # Preserve the existing optional Triton-package precedence as well as
        # the TPU-only 32-element checkpoint shard alignment.
        for triton_available, padded in ((False, 736), (True, 768)):
            with (
                self.subTest(triton_available=triton_available),
                patch(
                    "sglang.srt.layers.quantization.mxfp4.has_triton_kernels",
                    triton_available,
                ),
            ):
                method = TpuMxfp4MoEMethod("experts")
                layer = torch.nn.Module()
                layer.num_local_experts = 2
                layer.hidden_size = 128
                layer.intermediate_size_per_partition = 720
                loader = object()
                method.create_weights(
                    layer, 2, 128, 720, torch.bfloat16, weight_loader=loader
                )
                self.assertEqual(method.intermediate_size_per_partition, padded)
                self.assertEqual(layer.intermediate_size_per_partition, 720)
                self.assertEqual(layer.w13_weight.shape, (2, 2 * padded, 64))
                self.assertEqual(layer.w2_weight.shape, (2, 128, padded // 2))
                self.assertIs(layer.w13_weight.weight_loader, loader)
                self.assertTrue(torch.all(layer.w2_weight_scale == 127))

    def test_wna16_selection_validates_schema_before_packed_loading(self):
        config = self._ct_config()
        register_moe_quantization_methods()
        scheme = config._get_wna16_moe_scheme(self._int4(), None)
        self.assertIsInstance(scheme, TpuCompressedTensorsWNA16MoE)
        layer = torch.nn.Module()
        layer.moe_tp_size = 1
        loader = object()
        scheme.create_weights(layer, 2, 128, 128, torch.bfloat16, weight_loader=loader)
        self.assertEqual(layer.w13_weight_packed.shape, (2, 16, 256))
        self.assertEqual(layer.w2_weight_packed.shape, (2, 16, 128))
        self.assertIs(layer.w13_weight_packed.weight_loader, loader)
        self.assertTrue(layer.w13_weight_packed.is_transposed)
        for kwargs in ({"symmetric": False}, {"actorder": "group"}):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, "only symmetric 4-bit"),
            ):
                config._get_wna16_moe_scheme(self._int4(**kwargs), None)

    def test_host_fp8_quantization_uses_common_gate_up_scale(self):
        weights = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        quantized, scale = _quantize_fp8_weight(weights)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(
            quantized.float() * scale, weights, atol=0.02, rtol=0
        )
        requantized, used_scale = _quantize_fp8_weight(weights / 2, scale)
        self.assertEqual(used_scale, scale)
        torch.testing.assert_close(
            requantized.float() * scale, weights / 2, atol=0.02, rtol=0
        )
        zeros, zero_scale = _quantize_fp8_weight(torch.zeros(4))
        self.assertTrue(torch.isfinite(zero_scale) & (zero_scale > 0))
        self.assertTrue(torch.all(zeros.float() == 0))

    def test_fp8_dispatcher_observes_prepared_weights(self):
        config = Fp8Config(is_checkpoint_fp8_serialized=False)
        method = TpuFp8MoEMethod(config)
        method.moe_runner_config = MoeRunnerConfig()
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        layer.w13_weight = torch.nn.Parameter(
            torch.ones(2, 256, 128), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.ones(2, 128, 128), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        observed = []

        def publish(config):
            observed.append(
                (layer.w13_weight.shape, layer.w13_weight_scale.shape, config)
            )

        layer.dispatcher = SimpleNamespace(set_quant_config=publish)
        method.process_weights_after_loading(layer)
        self.assertEqual(observed[0][0], (2, 128, 256))
        self.assertEqual(observed[0][1], (2, 1, 1, 256))
        self.assertEqual(observed[0][2]["weight_dtype"], torch.float8_e4m3fn)

    def test_explicit_fused_rs_rejects_unsupported_layout(self):
        get_flags().moe.runner_backend = MoeRunnerBackend.PALLAS_TPU_FUSED_RS
        method = TpuFp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
        layer = SimpleNamespace(_tpu_fused_rs_unsupported_reason="requires EP")
        with (
            patch("sglang_tpu.fp8_moe.prepare_tpu_fp8_moe_weights"),
            patch(
                "sglang_tpu.fp8_moe_fused_rs.prepare_fp8_fused_rs", return_value=False
            ),
            self.assertRaisesRegex(ValueError, "requires EP"),
        ):
            method._prepare_weights_for_backend(layer)


if __name__ == "__main__":
    unittest.main()
