# SPDX-License-Identifier: Apache-2.0
"""CPU QA contracts only; no early-QKV connection or TPU numerical claim.

Separate from Q-B tests because QA adapters have an independent seam/owner.
"""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn

from sglang_tpu.models.deepseek_v41 import qa_preprocess as binder
from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing import qa_rope
from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing.tuned_block_sizes import (
    QARopeConfig,
)
from sglang_tpu.models.deepseek_v41.ops import qa_preprocess as ops

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _layer(cls=MQALayer):
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.is_dsv41 = True
    layer.fuse_wqa_wkv = True
    layer.hidden_size = 5120
    layer.q_lora_rank = 1280
    layer.head_dim = 512
    layer.qk_rope_head_dim = 64
    layer.wqkv_a = nn.Module()
    layer.wqkv_a.weight = nn.Parameter(
        torch.zeros(5120, 1792, dtype=torch.float8_e4m3fn), False
    )
    layer.wqkv_a.register_buffer(
        "weight_scale", (torch.arange(1792, dtype=torch.float32) + 1).reshape(-1, 1)
    )
    for name, width in (("q_norm", 1280), ("kv_norm", 512)):
        norm = nn.Module()
        norm.weight = nn.Parameter(
            (torch.arange(width, dtype=torch.float32) / width + 0.5).bfloat16(), False
        )
        norm.variance_epsilon = 1e-6
        setattr(layer, name, norm)
    return layer


class TestQABinding(CustomTestCase):
    def test_bind_upcasts_exactly_without_duplicate_state_and_refreshes(self):
        """Gamma rounding must not move before the multiply; reload refreshes it."""
        layer = _layer()
        before = {name: t.data_ptr() for name, t in layer.state_dict().items()}
        self.assertEqual(binder.bind_model_qa(layer), 1)
        prepared = layer._dsv41_qa
        self.assertIs(prepared.weight, layer.wqkv_a.weight)
        torch.testing.assert_close(prepared.scale, layer.wqkv_a.weight_scale[:, 0])
        for actual, norm in (
            (prepared.gamma_qr, layer.q_norm),
            (prepared.gamma_kv, layer.kv_norm),
        ):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, norm.weight.float(), rtol=0, atol=0)
        self.assertEqual(
            before, {name: t.data_ptr() for name, t in layer.state_dict().items()}
        )
        layer.q_norm.weight = nn.Parameter(
            torch.ones(1280, dtype=torch.bfloat16), False
        )
        self.assertEqual(binder.bind_model_qa(layer), 1)
        torch.testing.assert_close(layer._dsv41_qa.gamma_qr, torch.ones(1280))
        self.assertIsNot(layer._dsv41_qa.gamma_qr, prepared.gamma_qr)
        layer.kv_norm.variance_epsilon = 1e-5
        self.assertEqual(binder.bind_model_qa(layer), 0)
        self.assertIsNone(layer._dsv41_qa)
        layer.q_norm.variance_epsilon = 1e-5
        self.assertEqual(binder.bind_model_qa(layer), 1)
        self.assertEqual(layer._dsv41_qa.eps, 1e-5)

    def test_unsupported_state_never_binds(self):
        """Namesakes, wqa-only, noncanonical quantization and norm changes fall back."""
        namesake = type("MQALayer", (nn.Module,), {})
        self.assertIn("not a core", binder.qa_eligibility_reason(_layer(namesake)))
        cases = (
            ("is_dsv41", False),
            ("fuse_wqa_wkv", False),
            ("wqkv_a.weight_block_size", [128, 128]),
            ("wqkv_a.weight_scale_inv", torch.ones(1)),
            ("wqkv_a.input_scale", torch.ones(1)),
            ("wqkv_a.weight_scale", torch.ones(14, 128)),
            ("q_norm.variance_epsilon", 1e-5),
            ("q_norm.cast_x_before_out_mul", True),
            ("q_norm.weight", nn.Parameter(torch.ones(1280), False)),
        )
        for path, value in cases:
            with self.subTest(path=path):
                layer = _layer()
                target = layer
                parts = path.split(".")
                for part in parts[:-1]:
                    target = getattr(target, part)
                setattr(target, parts[-1], value)
                self.assertIsNone(binder.prepare_qa_adapter(layer))

    def test_fp32_gamma_matches_real_core_norm_boundary(self):
        """BF16 gamma would introduce a pre-multiply rounding absent in core."""
        norm = RMSNorm(1280, eps=1e-6)
        norm.weight = nn.Parameter(
            (torch.arange(1280).float() / 1280 + 0.5).bfloat16(), False
        )
        values = (torch.arange(2560).reshape(2, 1280).float() / 113 - 10).bfloat16()
        expected = norm.forward_native(values).float().numpy()
        actual = qa_rope.native_rms_norm(
            jnp.asarray(values.float().numpy()),
            jnp.asarray(norm.weight.float().numpy()),
            1e-6,
        )
        np.testing.assert_array_equal(np.asarray(actual.astype(jnp.float32)), expected)
        rounded = qa_rope.native_rms_norm(
            jnp.asarray(values.float().numpy()),
            jnp.asarray(norm.weight.float().numpy(), dtype=jnp.bfloat16),
            1e-6,
        )
        self.assertTrue(np.any(np.asarray(rounded.astype(jnp.float32)) != expected))


class TestQABridge(CustomTestCase):
    def test_empty_and_fake_outputs_bypass_export(self):
        layer = _layer()
        prepared = binder.prepare_qa_adapter(layer)
        args = (
            torch.empty(0, 5120, dtype=torch.bfloat16),
            prepared.weight,
            prepared.scale,
            prepared.gamma_qr,
            prepared.gamma_kv,
            torch.zeros(2, 64),
            torch.empty(0, dtype=torch.int64),
        )
        with mock.patch.object(ops, "_op", side_effect=AssertionError("export")):
            qr, kv = ops.qa_rope(*args)
        self.assertEqual((qr.shape, kv.shape), ((0, 1280), (0, 512)))
        self.assertEqual((qr.dtype, kv.dtype), (torch.bfloat16, torch.bfloat16))
        qr, kv = ops._fake_qa_rope(
            torch.empty(3, 5120, dtype=torch.bfloat16), *args[1:]
        )
        self.assertEqual((qr.shape, kv.shape), ((3, 1280), (3, 512)))
        abstract = jax.eval_shape(
            ops._qa_rope,
            *[
                jax.ShapeDtypeStruct(
                    tuple(t.shape),
                    jnp.bfloat16
                    if t.dtype == torch.bfloat16
                    else jnp.float8_e4m3fn
                    if t.dtype == torch.float8_e4m3fn
                    else jnp.int32
                    if t.dtype == torch.int64
                    else jnp.float32,
                )
                for t in args
            ],
        )
        self.assertEqual(tuple(t.shape for t in abstract), ((0, 1280), (0, 512)))

    def test_real_kernel_independent_abstract_shapes(self):
        """Trace the real kernel with explicit config, not a mocked fused result."""
        for tokens in (1, 16, 17):
            with self.subTest(tokens=tokens):
                qr, kv = jax.eval_shape(
                    lambda x, w, s, gq, gk, rows: qa_rope.fused_qa_kv_norm_rope(
                        x, w, s, gq, gk, rows, config=QARopeConfig(token_block=16)
                    ),
                    jax.ShapeDtypeStruct((tokens, 5120), jnp.bfloat16),
                    jax.ShapeDtypeStruct((5120, 1792), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((1792,), jnp.float32),
                    jax.ShapeDtypeStruct((1280,), jnp.float32),
                    jax.ShapeDtypeStruct((512,), jnp.float32),
                    jax.ShapeDtypeStruct((tokens, 64), jnp.float32),
                )
                self.assertEqual((qr.shape, kv.shape), ((tokens, 1280), (tokens, 512)))
                self.assertEqual((qr.dtype, kv.dtype), (jnp.bfloat16, jnp.bfloat16))


if __name__ == "__main__":
    unittest.main()
