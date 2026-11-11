# SPDX-License-Identifier: Apache-2.0
"""CPU binding/precision boundaries; TPU execution is a separate gate."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import torch
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41 import compressor as binder
from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing import compress_rope
from sglang_tpu.models.deepseek_v41.ops import compressor as ops

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _layer(ratio=1):
    layer = MQALayer.__new__(MQALayer)
    torch.nn.Module.__init__(layer)
    layer.is_dsv41 = True
    layer.qk_rope_head_dim = 64
    layer.compressor = DeepseekV41Compressor(
        5120, 512, ratio, 1e-6, fused_compress=False
    )
    layer.requires_grad_(False)
    return layer


class TestCompressorBinding(CustomTestCase):
    def test_online_updates_reject_before_mutating_bound_transpose(self):
        """Public updates bypass adapter rebinding and must reject stale caches."""
        from types import SimpleNamespace

        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            WeightUpdater,
        )

        from sglang_tpu.models.deepseek_v41.loader import (
            DSV41RunaiModelStreamerLoader,
            RunaiModelStreamerLoader,
        )

        layer = _layer()
        loader = DSV41RunaiModelStreamerLoader.__new__(DSV41RunaiModelStreamerLoader)
        # Model construction/checkpoint I/O is the boundary; real post-load
        # binders and the production update admission/mutation methods run below.
        with mock.patch.object(RunaiModelStreamerLoader, "load_model", return_value=layer):
            model = loader.load_model(model_config=None, device_config=None)
        runner = SimpleNamespace(
            get_model=lambda: model,
            _assert_weight_cache_inactive=lambda op: None,
        )
        before = model.compressor.wkv.weight.detach().clone()
        replacement = torch.full_like(before, 0.25)
        rejection = WeightUpdater.load_rejection(runner, "update_weights_from_tensor")
        self.assertIn("adapters are bound", rejection)
        updates = (
            lambda: WeightUpdater.update_weights_from_tensor(
                runner, [("compressor.wkv.weight", replacement)], load_format="direct"
            ),
            lambda: WeightUpdater.load_weights_from_distributed(
                runner, [("compressor.wkv.weight", replacement)]
            ),
            lambda: WeightUpdater.update_weights_from_disk(runner, "unused", "auto"),
            lambda: WeightUpdater.update_weights_from_ipc(runner, None),
        )
        for update in updates:
            success, reason = update()
            self.assertFalse(success)
            self.assertEqual(reason, rejection)
            torch.testing.assert_close(model.compressor.wkv.weight, before)
        # A complete, explicit offline reload/rebind still refreshes the cache.
        model.compressor.wkv.weight.copy_(replacement)
        binder.bind_model_compressor(model)
        torch.testing.assert_close(model._dsv41_compressor.weight, replacement.t())

    def test_load_preserves_existing_update_rejection_without_blocking_fallback(self):
        from sglang_tpu.models.deepseek_v41.loader import (
            DSV41RunaiModelStreamerLoader,
            RunaiModelStreamerLoader,
        )

        loader = DSV41RunaiModelStreamerLoader.__new__(DSV41RunaiModelStreamerLoader)
        for ratio, existing in ((1, "another derived cache"), (2, None)):
            with self.subTest(ratio=ratio):
                layer = _layer(ratio)
                if existing is not None:
                    layer._derived_weight_cache_error = existing
                with mock.patch.object(
                    RunaiModelStreamerLoader, "load_model", return_value=layer
                ):
                    loader.load_model(model_config=None, device_config=None)
                self.assertEqual(
                    getattr(layer, "_derived_weight_cache_error", None), existing
                )

    def test_real_eval_constructor_grad_metadata_does_not_block_inference(self):
        """eval does not freeze Parameters; binding must detach, not reject them."""
        layer = MQALayer.__new__(MQALayer)
        torch.nn.Module.__init__(layer)
        layer.is_dsv41 = True
        layer.qk_rope_head_dim = 64
        layer.compressor = DeepseekV41Compressor(
            5120, 512, 1, 1e-6, fused_compress=False
        )
        layer.eval()
        before = {
            name: (t.data_ptr(), t.requires_grad)
            for name, t in layer.named_parameters()
        }
        self.assertTrue(all(flag for _, flag in before.values()))
        for _ in range(2):
            self.assertEqual(binder.bind_model_compressor(layer), 1)
            prepared = layer._dsv41_compressor
            self.assertFalse(prepared.weight.requires_grad)
            self.assertFalse(prepared.gamma.requires_grad)
            self.assertEqual(
                prepared.gamma.data_ptr(), layer.compressor.norm.weight.data_ptr()
            )
            torch.testing.assert_close(
                prepared.gamma, layer.compressor.norm.weight, atol=0, rtol=0
            )
            self.assertEqual(
                before,
                {
                    name: (t.data_ptr(), t.requires_grad)
                    for name, t in layer.named_parameters()
                },
            )

    def test_canonical_copy_gamma_alias_and_reload(self):
        """A cached transpose must refresh even on in-place checkpoint reload."""
        layer = _layer()
        layer.compressor.norm.weight = torch.nn.Parameter(
            torch.linspace(0.501, 1.499, 512), False
        )
        before = {k: v.data_ptr() for k, v in layer.state_dict().items()}
        self.assertEqual(binder.bind_model_compressor(layer), 1)
        old = layer._dsv41_compressor
        self.assertEqual(old.gamma.data_ptr(), layer.compressor.norm.weight.data_ptr())
        self.assertEqual(old.gamma.dtype, torch.float32)
        self.assertTrue(old.weight.is_contiguous())
        torch.testing.assert_close(
            old.weight, layer.compressor.wkv.weight.t(), atol=0, rtol=0
        )
        self.assertNotEqual(
            old.weight.data_ptr(), layer.compressor.wkv.weight.data_ptr()
        )
        layer.compressor.wkv.weight.fill_(0.25)
        self.assertEqual(binder.bind_model_compressor(layer), 1)
        torch.testing.assert_close(
            layer._dsv41_compressor.weight, torch.full_like(old.weight, 0.25)
        )
        self.assertNotEqual(
            old.weight.data_ptr(), layer._dsv41_compressor.weight.data_ptr()
        )
        self.assertEqual(
            before, {k: v.data_ptr() for k, v in layer.state_dict().items()}
        )
        layer.compressor.compress_ratio = 2
        self.assertEqual(binder.bind_model_compressor(layer), 0)
        self.assertIsNone(layer._dsv41_compressor)

    def test_finish_boundary_independent_numpy(self):
        """FP32-before-norm and BF16 gamma downcasts both alter core results."""
        layer = _layer()
        core = layer.compressor
        core.norm.weight = torch.nn.Parameter(torch.linspace(0.501, 1.499, 512), False)
        values = torch.linspace(-2.003, 3.017, 512).reshape(1, 512)
        rounded = values.bfloat16().float().numpy()
        expected = rounded / np.sqrt(np.mean(rounded**2, -1, keepdims=True) + 1e-6)
        expected = torch.from_numpy(expected * core.norm.weight.numpy()).bfloat16()
        torch.testing.assert_close(core.finish(values), expected, rtol=0, atol=0)
        wrong = (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
            * core.norm.weight
        ).bfloat16()
        self.assertTrue(torch.any(wrong != expected))
        downcast = (
            torch.from_numpy(rounded)
            * torch.rsqrt(
                torch.from_numpy(rounded).square().mean(-1, keepdim=True) + 1e-6
            )
            * core.norm.weight.bfloat16()
        ).bfloat16()
        self.assertTrue(torch.any(downcast != expected))

    def test_noncore_and_modified_norm_fall_back(self):
        layer = _layer()
        layer.compressor.norm.cast_x_before_out_mul = True
        self.assertIsNone(binder.prepare_compressor_adapter(layer))
        layer = _layer(2)
        before = layer.compressor.wkv.weight.clone()
        self.assertIsNone(binder.prepare_compressor_adapter(layer))
        torch.testing.assert_close(before, layer.compressor.wkv.weight)
        namesake = type("MQALayer", (torch.nn.Module,), {})()
        namesake.is_dsv41 = True
        namesake.compressor = _layer().compressor
        self.assertIsNone(binder.prepare_compressor_adapter(namesake))


class TestCompressorExportContract(CustomTestCase):
    def test_evalshape_actual_geometry_and_mask_contract(self):
        """T=1 padding and exact-row outputs must survive CPU-only export tracing."""
        for tokens in (1, 16, 128):
            with self.subTest(tokens=tokens):
                specs = (
                    jax.ShapeDtypeStruct((tokens, 5120), jnp.bfloat16),
                    jax.ShapeDtypeStruct((5120, 512), jnp.bfloat16),
                    jax.ShapeDtypeStruct((512,), jnp.float32),
                    jax.ShapeDtypeStruct((257, 64), jnp.float32),
                    jax.ShapeDtypeStruct((tokens,), jnp.int32),
                    jax.ShapeDtypeStruct((tokens, 1), jnp.int32),
                )
                result = jax.eval_shape(ops._compress_rope, *specs)
                self.assertEqual(
                    [(s.shape, s.dtype) for s in result],
                    [
                        ((tokens, 512), jnp.bfloat16),
                        ((tokens, 128), jnp.int32),
                        ((tokens, 1), jnp.float32),
                    ],
                )
        with self.assertRaisesRegex(ValueError, "ratio 1 only"):
            compress_rope.fused_compress_rope(
                None,
                None,
                None,
                None,
                None,
                None,
                ratio=2,
                precision="core_finish_bf16",
            )

    def test_validation_and_empty_fullgraph(self):
        inputs = (
            torch.empty(0, 5120, dtype=torch.bfloat16),
            torch.empty(5120, 512, dtype=torch.bfloat16),
            torch.ones(512),
            torch.zeros(257, 64),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, 1, dtype=torch.int32),
        )
        with mock.patch.object(
            ops, "_op", side_effect=AssertionError("empty must not export")
        ):
            outputs = torch.compile(ops.compress_rope, backend="eager", fullgraph=True)(
                *inputs
            )
        self.assertEqual(
            [tuple(t.shape) for t in outputs], [(0, 512), (0, 128), (0, 1)]
        )
        with self.assertRaisesRegex(ValueError, "valid"):
            ops.compress_rope(*inputs[:-1], torch.empty(0, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "positive"):
            ops.compress_rope(*inputs, eps=float("nan"))


if __name__ == "__main__":
    unittest.main()
