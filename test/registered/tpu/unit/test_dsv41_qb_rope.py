# SPDX-License-Identifier: Apache-2.0
"""TorchTPU fused Q-B against independent FP8/BF16/complex-RoPE arithmetic.

Guards row (not tensor) activation quantization, canonical KN weight scales,
the BF16 projection boundary, and forward interleaved RoPE on only the last
64 of each 512-channel head. Small decode/prefill buckets exercise exact-row
output and padding; one Flash-sized projection covers rank1280/H64 without a
cross-product of large shapes. No model weights, server, or mesh is needed.

The arithmetic and 1e-2 absolute/relative tolerances follow the epic source
at vllm-torchtpu dc1cd126e, tests/kernels/attention_preprocessing/test_qb_rope.py
and src/vllm_torchtpu/kernels/attention_preprocessing/reference.py. This test
implements dense JAX arithmetic independently of kernel helpers, executed on
TPU, plus the real generic TorchTPU linear followed by NumPy complex RoPE.
The CPU NumPy oracle checks algebraic boundaries, not bitwise TPU matmul:
XLA reciprocal lowering can change FP8 midpoint codes, and TPU dot rounding
can cross the BF16 projection boundary. Neither tolerance is relaxed for these
platform effects; both TPU references use the original 1e-2 tolerances.
"""

import importlib.util
import os
import unittest

# TorchTPU owns the device; JAX exports custom calls on CPU, as in CSA2 tests.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.ops import preprocess as ops

register_tpu_ci(
    est_time=300,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

HAS_TORCH_TPU = importlib.util.find_spec("torch_tpu") is not None
BF16 = ml_dtypes.bfloat16
FP8 = ml_dtypes.float8_e4m3fn
HEAD_DIM = 512
ROPE_DIM = 64


def _quantize_rows(values):
    """Generic fp8_linear boundary, independently evaluated on the host."""
    values = values.astype(np.float32)
    scale = np.maximum(np.max(np.abs(values), axis=1, keepdims=True) / 448, 1e-12)
    encoded = np.clip(values / scale, -448, 448).astype(FP8)
    # The epic kernel uses scale=1 for an all-zero row, while generic
    # fp8_linear floors at 1e-12. Both encode/dequantize zeros identically.
    # BF16 normalized nonzero rows used here are above that floor.
    return encoded, scale


def _reference(qr, weight, scale, cache, positions):
    encoded, row_scale = _quantize_rows(qr)
    product = encoded.astype(np.float32) @ weight.astype(np.float32)
    projected = (product * row_scale * scale[None, :]).astype(BF16)
    projected = projected.reshape(len(qr), -1, HEAD_DIM)
    return _rotate_reference(projected, cache, positions)


def _rotate_reference(projected, cache, positions):
    result = projected.astype(np.float32)
    rows = cache[np.clip(positions, 0, len(cache) - 1)]
    # Complex multiplication makes the forward sign and adjacent-pair layout
    # independent of the kernel's lane-XOR/sign-folded coefficient machinery.
    tail = result[..., -ROPE_DIM:].reshape(len(projected), -1, ROPE_DIM // 2, 2)
    z = tail[..., 0].astype(np.float64) + 1j * tail[..., 1]
    rotation = rows[:, :32].astype(np.float64) + 1j * rows[:, 32:]
    rotated = z * rotation[:, None, :]
    result[..., -ROPE_DIM::2] = rotated.real
    result[..., -ROPE_DIM + 1 :: 2] = rotated.imag
    return result.astype(BF16)


def _dense_tpu_reference(
    qr: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    cache: jax.Array,
    positions: jax.Array,
) -> jax.Array:
    """Epic's dense arithmetic; no Pallas, lane helpers, or fused op calls."""
    values = qr.astype(jnp.float32)
    maximum = jnp.max(jnp.abs(values), axis=1, keepdims=True)
    row_scale = jnp.where(maximum == 0, 1.0, maximum / 448.0)
    encoded = jnp.clip(values / row_scale, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    product = jax.lax.dot_general(
        encoded,
        weight,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    projected = (product * row_scale * scale).astype(jnp.bfloat16)
    projected = projected.reshape(qr.shape[0], -1, HEAD_DIM)
    rows = cache[jnp.clip(positions, 0, cache.shape[0] - 1)]
    tail = projected[..., -ROPE_DIM:].astype(jnp.float32)
    partner = jnp.stack((-tail[..., 1::2], tail[..., ::2]), axis=-1).reshape(tail.shape)
    cosine = jnp.repeat(rows[:, :32], 2, axis=1)[:, None, :]
    sine = jnp.repeat(rows[:, 32:], 2, axis=1)[:, None, :]
    rotated = (tail * cosine + partner * sine).astype(jnp.bfloat16)
    return jnp.concatenate((projected[..., :-ROPE_DIM], rotated), axis=-1)


def _inputs(tokens, rank=128, heads=8):
    rng = np.random.default_rng(7)
    qr = rng.normal(size=(tokens, rank)).astype(BF16)
    # Keep T=1 nonzero: the decode case must actually exercise the projection.
    if tokens > 1:
        qr[0] = 0
        qr[1] = (qr[1].astype(np.float32) * 1e-6).astype(BF16)
    weight = rng.normal(size=(rank, heads * HEAD_DIM)).astype(FP8)
    scale = rng.uniform(0.01, 0.1, heads * HEAD_DIM).astype(np.float32)
    angles = rng.uniform(-3, 3, size=(257, 32))
    cache = np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    positions = ((np.arange(tokens) * 37 + 17) % len(cache)).astype(np.int32)
    if tokens >= 16:
        # Nonzero rows test clamping independently of zero/padding rows.
        positions[-2:] = [-9, len(cache) + 100]
    return qr, weight, scale, cache, positions


def _tensor(array, device="cpu"):
    # NumPy has no native Torch BF16/FP8 conversion: preserve their raw bits.
    if array.dtype == BF16:
        tensor = torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    elif array.dtype == FP8:
        tensor = torch.from_numpy(array.view(np.uint8)).view(torch.float8_e4m3fn)
    else:
        tensor = torch.from_numpy(array)
    return tensor.contiguous().to(device)


class TestQBRopeReferenceBoundaries(CustomTestCase):
    def test_activation_quantization_matches_generic_linear(self):
        """A per-tensor scale or a BF16 scale silently changes FP8 codes."""
        from sglang_tpu.fp8_linear import _quantize_activation_per_row

        qr = _inputs(16)[0]
        encoded, scale = _quantize_rows(qr)
        actual, actual_scale = _quantize_activation_per_row(
            _tensor(qr), torch.float8_e4m3fn
        )
        np.testing.assert_array_equal(
            actual.view(torch.uint8).numpy(), encoded.view(np.uint8)
        )
        np.testing.assert_array_equal(actual_scale.numpy(), scale)
        np.testing.assert_array_equal(actual[0].float().numpy(), 0)

    def test_rope_coefficients_match_complex_rotation(self):
        """A sign, half-split pairing, or 128-vs-64 tail regression must fail."""
        from sglang_tpu.models.deepseek_v41.kernels.attention_preprocessing import (
            rope_gather,
        )

        qr, weight, scale, cache, positions = _inputs(16)
        encoded, row_scale = _quantize_rows(qr)
        projected = (
            (
                (encoded.astype(np.float32) @ weight.astype(np.float32))
                * row_scale
                * scale
            )
            .astype(BF16)
            .reshape(16, 8, HEAD_DIM)
        )
        rows = rope_gather.gather_rows(jnp.asarray(cache), jnp.asarray(positions))
        cosine, sine = map(np.asarray, rope_gather.coefficients(rows))
        result = projected.astype(np.float32)
        tail = result[..., -128:].copy()
        result[..., -128:] = (
            tail * cosine[:, None, :] + tail[..., np.arange(128) ^ 1] * sine[:, None, :]
        )
        np.testing.assert_allclose(
            result.astype(BF16).astype(np.float32),
            _reference(qr, weight, scale, cache, positions).astype(np.float32),
            atol=1e-2,
            rtol=1e-2,
        )


@unittest.skipUnless(HAS_TORCH_TPU, "requires TorchTPU")
class TestQBRopeTPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        ops.register_ops()
        # Export dense HLO on CPU, execute through TorchTPU. A direct JAX TPU
        # client conflicts with the CPU-only JAX initialized by CSA2 tests in
        # the same process; this reference does not need that second client.
        reference = cpu_export_jax_op(
            "sglang_tpu_test::dsv41_qb_rope_dense_reference", _dense_tpu_reference
        )

        @reference.register_fake
        def fake_reference(qr, weight, scale, cache, positions):
            return qr.new_empty((qr.shape[0], weight.shape[1] // HEAD_DIM, HEAD_DIM))

        cls.reference_op = staticmethod(reference)

    def _check(self, tokens, rank=128, heads=8, compiled=False):
        from sglang_tpu.fp8_linear import apply_tpu_fp8_linear

        arrays = _inputs(tokens, rank, heads)
        inputs = tuple(_tensor(value, "tpu") for value in arrays)
        projected = apply_tpu_fp8_linear(*inputs[:3]).cpu().float().numpy()
        projected = projected.reshape(tokens, heads, HEAD_DIM)
        run = ops.qb_rope
        if compiled:
            run = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        # Reuse the same graph with changed position values, not only trace inputs.
        for replay in range(2 if compiled else 1):
            if replay:
                arrays = (*arrays[:-1], arrays[-1][::-1].copy())
                inputs = (*inputs[:-1], _tensor(arrays[-1], "tpu"))
            expected = self.reference_op(*inputs).cpu().float().numpy()
            generic_expected = _rotate_reference(
                projected, arrays[3], arrays[4]
            ).astype(np.float32)
            output = run(*inputs)
            self.assertEqual(tuple(output.shape), (tokens, heads, HEAD_DIM))
            self.assertEqual(output.dtype, torch.bfloat16)
            actual = output.cpu().float().numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(actual, generic_expected, atol=1e-2, rtol=1e-2)
            if tokens > 1:
                np.testing.assert_array_equal(actual[0], 0)
            # Near-zero elements cannot hide a globally incorrect projection.
            relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(
                expected
            )
            self.assertLess(relative_error, 1e-2)

    def test_decode_t1(self):
        self._check(1)

    def test_decode_t16(self):
        self._check(16)

    def test_prefill_t128(self):
        self._check(128)

    def test_prefill_t256(self):
        self._check(256)

    def test_flash_geometry_t128(self):
        self._check(128, rank=1280, heads=64)

    def test_fullgraph_t128(self):
        self._check(128, compiled=True)


if __name__ == "__main__":
    unittest.main()
