# SPDX-License-Identifier: Apache-2.0
"""Independent QA FP8/BF16/core-norm/complex-RoPE reference and TPU checks.

No seam, model weights, server, or mesh required. CPU tests establish reference
boundaries; only the hardware tests establish kernel numerical agreement. The
kernel's tiny-row quantization policy intentionally differs from generic TPU
linear's 1e-12 scale floor; test that discrepancy explicitly, not as equivalence.

The original monolithic NumPy/dense-XLA oracle failed elementwise 1e-2 on
TPU even where fused agreed with production (shape-dependent quantization/dot
lowering). Keep host arithmetic for CPU boundary properties, not as a bitwise
MXU oracle. Device reference stages materialize FP8 codes and BF16 projection
separately, and pad decode to 16 rows, to preserve the production boundaries.
Both the staged independent reference and actual production baseline are gated
at unchanged 1e-2 tolerances; tiny rows are compared only to the kernel-policy
reference, never advertised as generic-minfloor equivalent.
"""

import importlib.util
import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.ops import qa_preprocess as ops

register_tpu_ci(
    est_time=300,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

HAS_TORCH_TPU = importlib.util.find_spec("torch_tpu") is not None
BF16 = ml_dtypes.bfloat16
FP8 = ml_dtypes.float8_e4m3fn
EPS = 1e-6


def _quantize_rows(x, generic_floor=False):
    values = x.astype(np.float32)
    scale = np.max(np.abs(values), axis=-1, keepdims=True) / 448
    scale = (
        np.maximum(scale, 1e-12) if generic_floor else np.where(scale == 0, 1, scale)
    )
    return np.clip(values / scale, -448, 448).astype(FP8), scale


def _normalize(values, gamma, eps=EPS):
    values = values.astype(np.float32)
    variance = np.mean(values * values, axis=-1, keepdims=True)
    return (values * (1 / np.sqrt(variance + eps)) * gamma).astype(BF16)


def _reference(
    x, weight, scale, gamma_qr, gamma_kv, cache, positions, generic_floor=False
):
    encoded, row_scale = _quantize_rows(x, generic_floor)
    product = encoded.astype(np.float32) @ weight.astype(np.float32)
    # Core linear rounds to BF16 BEFORE splitting and either RMSNorm.
    projected = (product * row_scale * scale[None, :]).astype(BF16)
    qr = _normalize(projected[:, :1280], gamma_qr)
    kv = _normalize(projected[:, 1280:], gamma_kv).astype(np.float32)
    rows = cache[np.clip(positions, 0, len(cache) - 1)]
    # Complex multiplication independently specifies adjacent-pair forward RoPE.
    tail = kv[:, -64:].reshape(len(x), 32, 2)
    z = tail[..., 0].astype(np.float64) + 1j * tail[..., 1]
    rotation = rows[:, :32].astype(np.float64) + 1j * rows[:, 32:]
    rotated = z * rotation
    kv[:, -64::2], kv[:, -63::2] = rotated.real, rotated.imag
    return qr, kv.astype(BF16)


def _inputs(tokens):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(tokens, 5120)).astype(BF16)
    if tokens > 1:
        x[0] = 0
        x[1] = (x[1].astype(np.float32) * 1e-13).astype(BF16)
    weight = rng.normal(size=(5120, 1792)).astype(FP8)
    scale = rng.uniform(0.01, 0.1, 1792).astype(np.float32)
    # Original BF16 weights are upcast exactly, never normalized in BF16.
    gq = rng.uniform(0.5, 1.5, 1280).astype(BF16).astype(np.float32)
    gk = rng.uniform(0.5, 1.5, 512).astype(BF16).astype(np.float32)
    angles = rng.uniform(-3, 3, size=(257, 32))
    cache = np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    positions = ((np.arange(tokens) * 37 + 17) % 257).astype(np.int32)
    if tokens >= 16:
        positions[-2:] = [-9, 357]
    return x, weight, scale, gq, gk, cache, positions


def _tensor(array, device="cpu"):
    if array.dtype == BF16:
        tensor = torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    elif array.dtype == FP8:
        tensor = torch.from_numpy(array.view(np.uint8)).view(torch.float8_e4m3fn)
    else:
        tensor = torch.from_numpy(array)
    return tensor.contiguous().to(device)


def _independent_quantize(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    values = x.astype(jnp.float32)
    scale = jnp.max(jnp.abs(values), axis=-1, keepdims=True) / 448
    scale = jnp.where(scale == 0, 1.0, scale)
    return jnp.clip(values / scale, -448, 448).astype(jnp.float8_e4m3fn), scale


def _independent_project(
    encoded: jax.Array, weight: jax.Array, row_scale: jax.Array, scale: jax.Array
) -> jax.Array:
    # Materialized quantization and BF16 projection are separate exported ops.
    # A single dense graph can fuse/reassociate this epilogue differently from
    # the production scaled_mm and Pallas boundaries. Decode uses the same
    # 16-row MXU padding as both production implementations.
    tokens = encoded.shape[0]
    pad = (-tokens) % 16
    encoded = jnp.pad(encoded, ((0, pad), (0, 0)))
    product = jnp.dot(
        encoded,
        weight,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )[:tokens]
    return (product * row_scale * scale).astype(jnp.bfloat16)


def _finish_projection(projected, gamma_qr, gamma_kv, cache, positions):
    qr = _normalize(projected[:, :1280], gamma_qr)
    kv = _normalize(projected[:, 1280:], gamma_kv).astype(np.float32)
    return qr.astype(np.float32), _rotate_kv(kv, cache, positions)


def _rotate_kv(kv, cache, positions):
    kv = kv.copy()
    rows = cache[np.clip(positions, 0, len(cache) - 1)]
    tail = kv[:, -64:].reshape(len(kv), 32, 2)
    rotated = (tail[..., 0].astype(np.float64) + 1j * tail[..., 1]) * (
        rows[:, :32].astype(np.float64) + 1j * rows[:, 32:]
    )
    kv[:, -64::2], kv[:, -63::2] = rotated.real, rotated.imag
    return kv.astype(BF16).astype(np.float32)


class TestQAReferenceBoundaries(CustomTestCase):
    def test_norm_matches_real_core_fp32_multiply_boundary(self):
        """A BF16 cast before gamma multiplication is not the core contract."""
        from sglang.srt.layers.layernorm import RMSNorm

        rng = np.random.default_rng(11)
        values = rng.normal(size=(3, 1280)).astype(BF16)
        gamma = rng.uniform(0.5, 1.5, 1280).astype(BF16)
        norm = RMSNorm(1280, eps=EPS)
        norm.weight = torch.nn.Parameter(_tensor(gamma), requires_grad=False)
        expected = _normalize(values, gamma.astype(np.float32)).astype(np.float32)
        actual = norm.forward_native(_tensor(values)).float().numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_tiny_row_policy_difference_is_explicit(self):
        """Zero rows agree, but nonzero amax below 448e-12 does not share codes."""
        from sglang_tpu.fp8_linear import _quantize_activation_per_row

        x = np.zeros((3, 5120), dtype=BF16)
        x[1] = np.linspace(-1e-13, 1e-13, 5120).astype(BF16)
        x[2] = np.linspace(-1, 1, 5120).astype(BF16)
        encoded, scale = _quantize_rows(x, generic_floor=True)
        actual, actual_scale = _quantize_activation_per_row(
            _tensor(x), torch.float8_e4m3fn
        )
        np.testing.assert_array_equal(
            actual.view(torch.uint8).numpy(), encoded.view(np.uint8)
        )
        np.testing.assert_array_equal(actual_scale.numpy(), scale)
        kernel_encoded, kernel_scale = _quantize_rows(x)
        np.testing.assert_array_equal(kernel_encoded[0].astype(np.float32), 0)
        np.testing.assert_array_equal(kernel_encoded[2], encoded[2])
        self.assertTrue(
            np.any(
                kernel_encoded[1].astype(np.float32) != encoded[1].astype(np.float32)
            )
        )
        self.assertLess(float(kernel_scale[1, 0]), 1e-12)
        self.assertEqual(float(kernel_scale[0, 0]), 1.0)


@unittest.skipUnless(HAS_TORCH_TPU, "requires TorchTPU")
class TestQARopeTPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        ops.register_ops()
        cls.quantize = staticmethod(
            cpu_export_jax_op(
                "sglang_tpu_test::qa_independent_quantize", _independent_quantize
            )
        )
        cls.project = staticmethod(
            cpu_export_jax_op(
                "sglang_tpu_test::qa_independent_project", _independent_project
            )
        )

    def _check(self, tokens, compiled=False):
        arrays = _inputs(tokens)
        inputs = tuple(_tensor(value, "tpu") for value in arrays)
        run = ops.qa_rope
        if compiled:
            run = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        for replay in range(2 if compiled else 1):
            if replay:
                arrays = (*arrays[:-1], arrays[-1][::-1].copy())
                inputs = (*inputs[:-1], _tensor(arrays[-1], "tpu"))
            from sglang.srt.layers.layernorm import RMSNorm

            from sglang_tpu.fp8_linear import apply_tpu_fp8_linear

            encoded, row_scale = self.quantize(inputs[0])
            projected = self.project(encoded, inputs[1], row_scale, inputs[2])
            expected = _finish_projection(projected.cpu().float().numpy(), *arrays[3:])
            # A second gate checks actual model compatibility, not only the
            # independent mathematical variant. Tiny rows intentionally differ
            # under generic's minfloor and are excluded from this gate only.
            generic_projection = apply_tpu_fp8_linear(*inputs[:3])
            generic_norms = []
            for width, gamma, values in (
                (1280, inputs[3], generic_projection[:, :1280]),
                (512, inputs[4], generic_projection[:, 1280:]),
            ):
                norm = RMSNorm(width, eps=EPS)
                norm.weight = torch.nn.Parameter(gamma.bfloat16(), requires_grad=False)
                generic_norms.append(norm.forward_native(values).cpu().float().numpy())
            # Norm is real core; rotate independently via host complex pairs.
            generic = (generic_norms[0], _rotate_kv(generic_norms[1], *arrays[-2:]))
            outputs = run(*inputs)
            for width, output, reference in zip((1280, 512), outputs, expected):
                self.assertEqual(tuple(output.shape), (tokens, width))
                self.assertEqual(output.dtype, torch.bfloat16)
                actual = output.cpu().float().numpy()
                reference = reference.astype(np.float32)
                np.testing.assert_allclose(actual, reference, atol=1e-2, rtol=1e-2)
                baseline = generic[0 if width == 1280 else 1]
                ordinary = (
                    np.arange(tokens) != 1
                    if tokens > 1
                    else np.ones(tokens, dtype=bool)
                )
                np.testing.assert_allclose(
                    actual[ordinary], baseline[ordinary], atol=1e-2, rtol=1e-2
                )
                if tokens > 1:
                    np.testing.assert_array_equal(actual[0], 0)
                    # Tiny rows need their own scaled error: global atol=1e-2
                    # could otherwise accept all zeros and hide a floor change.
                    tiny_error = np.linalg.norm(
                        actual[1] - reference[1]
                    ) / np.linalg.norm(reference[1])
                    self.assertLess(tiny_error, 1e-2)
                relative_error = np.linalg.norm(actual - reference) / np.linalg.norm(
                    reference
                )
                self.assertLess(relative_error, 1e-2)

    def test_decode_t1(self):
        self._check(1)

    def test_decode_t16(self):
        self._check(16)

    def test_prefill_t128(self):
        self._check(128)

    def test_fullgraph_t128(self):
        self._check(128, compiled=True)


if __name__ == "__main__":
    unittest.main()
