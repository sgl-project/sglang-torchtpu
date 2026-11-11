# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for postquant Q-B binding and the Torch/JAX shape bridge.

These tests use the real core MQALayer type, but initialize only its metadata
and small postquant tensors: no model loading, TorchTPU or device execution.
JAX abstract tracing is not a TPU lowering or numerical kernel test.
"""

import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import preprocess
from sglang_tpu.models.deepseek_v41.ops import preprocess as ops

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _DerivedMQA(MQALayer):
    pass


def _layer(cls=_DerivedMQA):
    # Skip heavyweight model construction, not core type matching or binding.
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.is_dsv41 = True
    layer.q_head_norm = False
    layer.head_dim = 512
    layer.qk_rope_head_dim = 64
    layer.n_local_heads = 8
    layer.q_lora_rank = 128
    layer.q_norm = nn.Module()
    layer.q_norm.weight = nn.Parameter(torch.ones(128, dtype=torch.bfloat16), False)
    layer.wq_b = nn.Module()
    layer.wq_b.bias = None
    layer.wq_b.weight = nn.Parameter(
        torch.zeros((128, 4096), dtype=torch.float8_e4m3fn), False
    )
    # Nonuniform, exactly representable FP32 values catch reciprocal,
    # broadcasting, column-order and dtype conversions during preparation.
    layer.wq_b.register_buffer(
        "weight_scale", (torch.arange(4096, dtype=torch.float32) + 1).reshape(-1, 1)
    )
    return layer


def _model(layer):
    model = nn.Module()
    model.attention = layer
    return model


class TestQBPostquantBinding(CustomTestCase):
    def test_adapter_preserves_ownership_scale_values_and_rebinds(self):
        """A prepared view must not duplicate state or outlive a weight reload."""
        layer = _layer()
        model = _model(layer)
        state_before = dict(model.state_dict())
        params_before = dict(model.named_parameters(remove_duplicate=False))
        buffers_before = dict(model.named_buffers(remove_duplicate=False))
        expected_scale = layer.wq_b.weight_scale[:, 0].clone()

        self.assertEqual(preprocess.bind_model_preprocessing(model), 1)
        first = layer._dsv41_qb
        self.assertIs(first.weight, layer.wq_b.weight)
        self.assertEqual((first.rank, first.heads), (128, 8))
        self.assertEqual(first.scale.shape, (4096,))
        self.assertEqual(first.scale.dtype, torch.float32)
        self.assertTrue(first.scale.is_contiguous())
        self.assertFalse(first.scale.requires_grad)
        torch.testing.assert_close(first.scale, expected_scale, rtol=0, atol=0)
        self.assertEqual(set(model.state_dict()), set(state_before))
        for name, tensor in model.state_dict().items():
            self.assertEqual(tensor.data_ptr(), state_before[name].data_ptr())
        self.assertEqual(
            dict(model.named_parameters(remove_duplicate=False)), params_before
        )
        self.assertEqual(
            dict(model.named_buffers(remove_duplicate=False)), buffers_before
        )

        replacement = nn.Parameter(torch.ones_like(first.weight), False)
        layer.wq_b.weight = replacement
        layer.wq_b.weight_scale = torch.arange(4096, dtype=torch.float32) / 8 + 0.5
        self.assertEqual(preprocess.bind_model_preprocessing(model), 1)
        self.assertIs(layer._dsv41_qb.weight, replacement)
        self.assertIsNot(layer._dsv41_qb.weight, first.weight)
        torch.testing.assert_close(
            layer._dsv41_qb.scale, layer.wq_b.weight_scale, rtol=0, atol=0
        )

    def test_unsupported_postquant_state_clears_previously_bound_adapter(self):
        """Reloading unsupported quantization must restore generic fallback."""
        cases = (
            ("weight_block_size", [128, 128], "blockwise"),
            ("weight_scale_inv", torch.ones(1), "blockwise"),
            ("bias", nn.Parameter(torch.zeros(4096), False), "bias"),
            ("weight", nn.Parameter(torch.zeros(128, 4096), False), "FP8"),
            ("weight_scale", torch.ones(4096, dtype=torch.bfloat16), "FP32"),
            ("weight_scale", torch.ones(32, 32), "per-column"),
        )
        for name, value, reason in cases:
            with self.subTest(name=name, reason=reason):
                layer = _layer()
                model = _model(layer)
                self.assertEqual(preprocess.bind_model_preprocessing(model), 1)
                setattr(layer.wq_b, name, value)
                self.assertIn(reason, preprocess.qb_eligibility_reason(layer))
                self.assertEqual(preprocess.bind_model_preprocessing(model), 0)
                self.assertIsNone(layer._dsv41_qb)

    def test_core_identity_and_v41_normalization_gate(self):
        """Duck-typed namesakes and incompatible core layers must not opt in."""
        namesake = type("MQALayer", (nn.Module,), {})
        unrelated = _layer(namesake)
        self.assertIn("not a core", preprocess.qb_eligibility_reason(unrelated))
        self.assertEqual(preprocess.bind_model_preprocessing(_model(unrelated)), 0)
        self.assertFalse(hasattr(unrelated, "_dsv41_qb"))

        layer = _layer(MQALayer)
        layer.is_dsv41 = False
        model = _model(layer)
        self.assertEqual(preprocess.bind_model_preprocessing(model), 0)
        # Wrapper configs identify V4.1 even if an upstream layer has no flag.
        model.config = SimpleNamespace(
            text_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        )
        self.assertEqual(preprocess.bind_model_preprocessing(model), 1)
        layer.q_head_norm = True
        self.assertEqual(preprocess.bind_model_preprocessing(model), 0)
        self.assertIsNone(layer._dsv41_qb)
        layer.q_head_norm = False
        layer.q_norm.weight = nn.Parameter(torch.ones(128), False)
        self.assertEqual(preprocess.bind_model_preprocessing(model), 0)
        self.assertIsNone(layer._dsv41_qb)


class TestQBBridgeContracts(CustomTestCase):
    @staticmethod
    def _inputs(tokens, device="meta"):
        return (
            torch.empty((tokens, 128), dtype=torch.bfloat16, device=device),
            torch.empty((128, 4096), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((4096,), dtype=torch.float32, device=device),
            torch.empty((257, 64), dtype=torch.float32, device=device),
            torch.empty((tokens,), dtype=torch.int64, device=device),
        )

    def test_real_jax_bridge_matches_fake_for_empty_ragged_and_aligned_rows(self):
        """Internal 32-row padding must not leak into the compiler's Q shape."""
        from sglang_tpu import pallas_export
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        tuning = importlib.import_module(
            "sglang_tpu.models.deepseek_v41.kernels.quantized_matmul.tuned_block_sizes"
        )
        for tokens in (0, 1, 16, 128):
            with self.subTest(tokens=tokens):
                specs = (
                    jax.ShapeDtypeStruct((tokens, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((128, 4096), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((4096,), jnp.float32),
                    jax.ShapeDtypeStruct((257, 64), jnp.float32),
                    jax.ShapeDtypeStruct((tokens,), jnp.int32),
                )
                # Hardware discovery only: all bridge/gather/Pallas shape logic
                # is real. This traces, but does not lower or execute on TPU.
                with (
                    mock.patch.object(
                        pallas_export,
                        "get_pallas_tpu_device_kind",
                        return_value="TPU7x",
                    ),
                    mock.patch.object(tuning, "get_tpu_version", return_value="tpu7x"),
                    tpu_info_for_cpu_export(),
                ):
                    traced = jax.eval_shape(ops._qb_rope, *specs)
                fake = ops._fake_qb_rope(*self._inputs(tokens))
                self.assertEqual(traced.shape, (tokens, 8, 512))
                self.assertEqual(traced.dtype, jnp.bfloat16)
                self.assertEqual(tuple(fake.shape), traced.shape)
                self.assertEqual(fake.dtype, torch.bfloat16)

    def test_rope_rows_keep_packed_cos_sin_order_and_clamp_padding(self):
        """The fused kernel consumes per-token [cos|sin], not interleaved rows."""
        # This executes the real CPU JAX gather used by the bridge. Distinct
        # coefficients and shuffled/out-of-range positions detect wrong axis,
        # wrong table layout, and accidental negative-index wraparound.
        theta = np.arange(5)[:, None] * np.linspace(0.01, 0.9, 32)[None, :]
        table = np.concatenate((np.cos(theta), np.sin(theta)), axis=-1).astype(
            np.float32
        )
        positions = np.array([3, -1, 0, 99, 1], dtype=np.int32)
        rows = ops.gather_rows(jnp.asarray(table), jnp.asarray(positions))
        np.testing.assert_array_equal(np.asarray(rows), table[[3, 0, 0, 4, 1]])
        self.assertEqual(rows.shape, (5, 64))

    def test_empty_torch_output_avoids_export_registration(self):
        # A zero-token rank must preserve the attention geometry without
        # entering the nonempty Pallas kernel or loading the TorchTPU bridge.
        with mock.patch.object(ops, "_op", side_effect=AssertionError("empty export")):
            result = ops.qb_rope(*self._inputs(0, device="cpu"))
        self.assertEqual(result.shape, (0, 8, 512))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.device.type, "cpu")

    def test_wrapper_rejects_incompatible_inputs_before_export(self):
        cases = (
            (0, torch.empty((1, 128), dtype=torch.float32, device="meta"), "qr"),
            (
                1,
                torch.empty((4096, 128), dtype=torch.float8_e4m3fn, device="meta"),
                "weight",
            ),
            (2, torch.empty((4096, 1), device="meta"), "scale"),
            (3, torch.empty((257, 32), device="meta"), "cos_sin"),
            (3, torch.empty((0, 64), device="meta"), "cos_sin"),
            (4, torch.empty((1,), dtype=torch.float32, device="meta"), "positions"),
        )
        for index, invalid, message in cases:
            with self.subTest(index=index, shape=invalid.shape):
                inputs = list(self._inputs(1))
                inputs[index] = invalid
                with self.assertRaisesRegex(ValueError, message):
                    ops.qb_rope(*inputs)


if __name__ == "__main__":
    unittest.main()
