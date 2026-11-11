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
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn

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


class TestQBBackendIntegration(CustomTestCase):
    def test_backend_normalizes_before_fused_dispatch_and_preserves_output_owner(self):
        """Binding changes Q dispatch, not normalized indexer rows or KV/RoPE."""
        from sglang_tpu.models.deepseek_v41.attention_backend import (
            DSV41TPUAttnBackend,
            rope_reference,
        )

        layer = _layer()
        layer.layer_id = 3
        layer.q_norm = nn.RMSNorm(128, eps=0, dtype=torch.bfloat16)
        layer.kv_norm = nn.Identity()
        # Only one active K row keeps the CPU reference small and exact.
        weight = torch.zeros((128, 4096))
        weight[0] = 1
        layer.wq_b.weight = nn.Parameter(weight.to(torch.float8_e4m3fn), False)
        layer.wq_b.weight_scale = torch.arange(4096, dtype=torch.float32) / 4096 + 1
        angles = torch.arange(4, dtype=torch.float32)[:, None] * torch.full(
            (1, 32), 0.3
        )
        layer.freqs_cis = torch.cat((angles.cos(), angles.sin()), -1)
        model = _model(layer)
        self.assertEqual(preprocess.bind_model_preprocessing(model), 1)
        backend = DSV41TPUAttnBackend.__new__(DSV41TPUAttnBackend)
        backend.rope_tables = {}
        raw_q = (
            torch.arange(1, 257, dtype=torch.float32).reshape(2, 128).to(torch.bfloat16)
        )
        expected_normalized = (
            raw_q.float() * raw_q.float().square().mean(-1, keepdim=True).rsqrt()
        ).to(torch.bfloat16)
        positions = torch.tensor([3, 1], dtype=torch.int64)
        kv = torch.arange(1024, dtype=torch.float32).reshape(2, 512).to(torch.bfloat16)
        qkv_a = torch.cat((raw_q, kv), -1)
        output_owner = torch.full((2, 8, 512), -7, dtype=torch.bfloat16)
        seen_rows = []

        def fused_reference(rows, canonical_weight, scale, table, token_positions):
            seen_rows.append(rows)
            projected = (rows.float() @ canonical_weight.float()) * scale
            return rope_reference(
                projected.to(torch.bfloat16).view(2, 8, 512), token_positions, table
            )

        # Fused dispatch must never evaluate the generic linear. Its numerical
        # replacement is only the hardware boundary; backend/norm/RoPE are real.
        with (
            mock.patch.object(ops, "qb_rope", side_effect=fused_reference),
            mock.patch.object(
                layer.wq_b, "forward", side_effect=AssertionError("generic Q-B")
            ),
        ):
            normalized, result, result_kv = backend.prepare_mqa_qkv(
                layer, torch.empty(2, 1), raw_q, qkv_a, positions, output_owner
            )
        expected_projected = (
            (expected_normalized[:, :1].float() * layer.wq_b.weight_scale)
            .to(torch.bfloat16)
            .view(2, 8, 512)
        )
        expected_q = rope_reference(expected_projected, positions, layer.freqs_cis)
        torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)
        self.assertIs(normalized, seen_rows[0])
        self.assertIs(result, output_owner)
        torch.testing.assert_close(result, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(
            result_kv, rope_reference(kv, positions, layer.freqs_cis), rtol=0, atol=0
        )
        torch.testing.assert_close(backend.rope_tables[3], layer.freqs_cis)

        # A distinct value-bearing generic projection detects an accidental
        # unconditional fused dispatch after the adapter is removed.
        layer._dsv41_qb = None
        generic_columns = torch.arange(4096, dtype=torch.float32)[None, :] / 1024 + 5

        def generic_projection(rows):
            return (generic_columns + rows[:, :1].float()).to(torch.bfloat16), None

        with (
            mock.patch.object(layer.wq_b, "forward", side_effect=generic_projection),
            mock.patch.object(
                ops, "qb_rope", side_effect=AssertionError("fused fallback")
            ),
        ):
            fallback_rows, fallback_q, fallback_kv = backend.prepare_mqa_qkv(
                layer, torch.empty(2, 1), raw_q, qkv_a, positions
            )
        expected_generic = (
            (generic_columns + expected_normalized[:, :1].float())
            .to(torch.bfloat16)
            .view(2, 8, 512)
        )
        torch.testing.assert_close(fallback_rows, normalized, rtol=0, atol=0)
        torch.testing.assert_close(
            fallback_q,
            rope_reference(expected_generic, positions, layer.freqs_cis),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(fallback_kv, result_kv, rtol=0, atol=0)
        self.assertFalse(torch.equal(fallback_q, result))

    def test_loader_binds_the_model_returned_after_quant_postprocessing(self):
        """Binding before the parent canonicalizes weights loses fused Q-B."""
        from sglang_tpu.models.deepseek_v41.loader import (
            DSV41RunaiModelStreamerLoader,
            RunaiModelStreamerLoader,
        )

        layer = _layer()
        model = _model(layer)
        canonical_weight = layer.wq_b.weight
        layer.wq_b.weight = nn.Parameter(torch.zeros(4096, 128), False)
        layer.wq_b.weight_block_size = [128, 128]

        def postprocess(*, model_config, device_config):
            layer.wq_b.weight = canonical_weight
            layer.wq_b.weight_block_size = None
            return model

        loader = DSV41RunaiModelStreamerLoader.__new__(DSV41RunaiModelStreamerLoader)
        with mock.patch.object(
            RunaiModelStreamerLoader, "load_model", side_effect=postprocess
        ):
            result = loader.load_model(model_config=object(), device_config=object())
        self.assertIs(result, model)
        self.assertIs(result.attention._dsv41_qb.weight, canonical_weight)
        torch.testing.assert_close(
            result.attention._dsv41_qb.scale,
            layer.wq_b.weight_scale.flatten(),
            rtol=0,
            atol=0,
        )


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
