# SPDX-License-Identifier: Apache-2.0
"""Unwired ratio-1 building block: independent dense HLO and real core oracle.

No second JAX TPU client: reference HLO exports on CPU and executes via the
same TorchTPU client. Latent tolerances remain epic's 1e-2 atol/rtol. Quantized
bytes are compared through their decoded values, not blindly bitwise: a small
BF16/dot/RoPE difference can cross an E4M3 midpoint. Scales retain 1e-2;
encoded reconstructions permit exactly one adjacent FP8 code at each value.
"""

import importlib.util
import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import torch
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.ops import compressor as ops

register_tpu_ci(
    est_time=240,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


def _finish_encode(projected, gamma, cache, positions, valid, eps):
    values = projected.astype(jnp.bfloat16).astype(jnp.float32)
    latent = (
        values
        * jax.lax.rsqrt(jnp.mean(values**2, -1, keepdims=True) + eps)
        * gamma.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    latent = jnp.where(valid != 0, latent, 0)
    rows = cache[jnp.clip(positions, 0, cache.shape[0] - 1)]
    pairs = latent[:, -64:].astype(jnp.float32).reshape(-1, 32, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = rows[:, :32], rows[:, 32:]
    tail = (
        jnp.stack((real * cos - imag * sin, real * sin + imag * cos), -1)
        .reshape(-1, 64)
        .astype(jnp.bfloat16)
    )
    rotated = jnp.concatenate((latent[:, :-64], tail), -1).astype(jnp.float32)
    rotated = jnp.where(valid != 0, rotated, 0)
    scales = jnp.maximum(jnp.max(jnp.abs(rotated), -1, keepdims=True), 1e-4) / 448
    codes = (rotated / scales).astype(jnp.float8_e4m3fn)
    # Independently reshape into the four CSA2 byte planes, not kernel helpers.
    planes = (
        jax.lax.bitcast_convert_type(codes, jnp.uint8)
        .reshape(-1, 4, 128)
        .astype(jnp.uint32)
    )
    words = sum(planes[:, i] << (8 * i) for i in range(4))
    return latent, jax.lax.bitcast_convert_type(words, jnp.int32), scales


def _dense_reference(
    x: jax.Array,
    weight: jax.Array,
    gamma: jax.Array,
    cache: jax.Array,
    positions: jax.Array,
    valid: jax.Array,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    projected = jnp.dot(x, weight, preferred_element_type=jnp.float32)
    return _finish_encode(projected, gamma, cache, positions, valid, eps)


def _inputs(tokens):
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(tokens, 5120, generator=generator).bfloat16()
    weight = (torch.randn(5120, 512, generator=generator) / 72).bfloat16()
    gamma = torch.linspace(0.51, 1.49, 512)  # original FP32 must NOT round to BF16
    angles = torch.randn(257, 32, generator=generator)
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    positions = (torch.arange(tokens, dtype=torch.int32) * 37 + 17) % 257
    valid = torch.ones(tokens, 1, dtype=torch.int32)
    if tokens > 1:
        x[0] = 0
        valid[-1] = 0
        positions[-2:] = torch.tensor([-9, 999], dtype=torch.int32)
    return x, weight, gamma, cache, positions, valid


def _decode(words):
    import ml_dtypes

    words = words.cpu().numpy().view(np.uint32)
    planes = np.stack([(words >> (8 * i)).astype(np.uint8) for i in range(4)], 1)
    return planes.reshape(-1, 512).view(ml_dtypes.float8_e4m3fn).astype(np.float32)


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires TorchTPU")
class TestCompressRopeTPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        ops.register_ops()
        reference = cpu_export_jax_op(
            "sglang_tpu_test::dsv41_compressor_reference", _dense_reference
        )
        reference.register_fake(ops._fake)
        cls.reference = staticmethod(reference)

    def _check(self, tokens, compiled=False):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor

        inputs = tuple(t.to("tpu") for t in _inputs(tokens))
        core = DeepseekV41Compressor(5120, 512, 1, 1e-6, fused_compress=False).to("tpu")
        core.wkv.weight = torch.nn.Parameter(inputs[1].t().contiguous(), False)
        core.norm.weight = torch.nn.Parameter(inputs[2], False)
        projected, _ = core.project(inputs[0])
        generic_latent = core.finish(projected)
        run = ops.compress_rope
        if compiled:
            run = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        for replay in range(2 if compiled else 1):
            if replay:
                inputs = (*inputs[:4], inputs[4].flip(0), inputs[5].flip(0))
            actual = run(*inputs)
            expected = self.reference(*inputs)
            generic = torch.where(inputs[5] != 0, generic_latent, 0)
            for oracle in (expected[0], generic):
                torch.testing.assert_close(
                    actual[0].cpu(), oracle.cpu(), atol=1e-2, rtol=1e-2
                )
            torch.testing.assert_close(
                actual[2].cpu(), expected[2].cpu(), atol=1e-2, rtol=1e-2
            )
            # Actual generic core.finish + independent host complex rotation/encode.
            from ml_dtypes import bfloat16, float8_e4m3fn

            latent = generic.cpu().float().numpy()
            cache = inputs[3].cpu().numpy()
            pos = np.clip(inputs[4].cpu().numpy(), 0, len(cache) - 1)
            z = latent[:, -64::2].astype(np.float64) + 1j * latent[:, -63::2]
            z *= cache[pos, :32] + 1j * cache[pos, 32:]
            rotated = latent.copy()
            rotated[:, -64::2], rotated[:, -63::2] = z.real, z.imag
            rotated = rotated.astype(bfloat16).astype(np.float32)
            scale = np.maximum(np.max(np.abs(rotated), -1, keepdims=True), 1e-4) / 448
            generic_codes = (rotated / scale).astype(float8_e4m3fn).astype(np.float32)
            np.testing.assert_allclose(
                actual[2].cpu().numpy(), scale, atol=1e-2, rtol=1e-2
            )
            for oracle in (_decode(expected[1]), generic_codes):
                decoded = _decode(actual[1])
                # One E4M3 step is 12.5% in each bin; min subnormal step 2^-9.
                error = np.abs(decoded - oracle)
                print(
                    f"compressor T={tokens} compiled={compiled} replay={replay}: "
                    f"FP8 differing={np.count_nonzero(error)}/{error.size}, "
                    f"max_code_value_delta={error.max():.6g}"
                )
                step = np.maximum(
                    2.0 ** (np.floor(np.log2(np.maximum(np.abs(oracle), 2**-6))) - 3),
                    2**-9,
                )
                self.assertTrue(
                    np.all(error <= step),
                    f"encoded mismatch beyond one FP8 step: {error.max()}",
                )
            invalid = inputs[5].cpu().numpy()[:, 0] == 0
            np.testing.assert_array_equal(actual[1].cpu().numpy()[invalid], 0)

    def test_actual_geometry(self):
        for tokens in (1, 16, 128):
            with self.subTest(tokens=tokens):
                self._check(tokens)

    def test_fullgraph(self):
        self._check(128, compiled=True)


if __name__ == "__main__":
    unittest.main()
