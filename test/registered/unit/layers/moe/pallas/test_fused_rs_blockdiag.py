"""The fused-RS block-diagonal fp8 dot equals the W4A16 dequant matmul.

Derived property: expanding each LHS row into one masked row per 16-wide scale
group and scaling the per-group partial products is the same contraction as
dequantizing the e2m1 codes with their block scales first. Pins the row / group
bookkeeping (chunking, scale tiling, group sums) that a "looks equivalent"
rewrite can silently break. Pure jnp, so it runs on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest

import numpy as np

from sglang.test.test_utils import CustomTestCase

try:
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.fused_rs.gmm_v2_gather_scatter import (
        fp4_blockdiag_matmul,
    )

    _HAS_JAX = True
except Exception:  # pragma: no cover - depends on JAX availability
    _HAS_JAX = False

GROUP = 16
E2M1 = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def _reference(lhs, codes, scales):
    # codes [K, N] e2m1 values as f32, scales [K // GROUP, N] f32
    w = codes * np.repeat(scales, GROUP, axis=0)
    return lhs.astype(np.float32) @ w


@unittest.skipUnless(_HAS_JAX, "Requires JAX")
class TestFp4BlockdiagMatmul(CustomTestCase):
    def _case(self, rows, size_k, size_n, kc, scale_dtype):
        rng = np.random.default_rng(rows * 1000 + size_k + kc)
        lhs = jnp.asarray(rng.standard_normal((rows, size_k)), dtype=jnp.bfloat16)
        code_idx = rng.integers(0, 16, size=(size_k, size_n))
        codes = jnp.asarray(E2M1[code_idx], dtype=jnp.float32).astype(jnp.float4_e2m1fn)
        scales = jnp.asarray(
            2.0 ** rng.integers(-6, 3, size=(size_k // GROUP, size_n)),
            dtype=scale_dtype,
        )
        got = fp4_blockdiag_matmul(lhs, codes, scales, group=GROUP, kc=kc)
        want = _reference(
            np.asarray(lhs.astype(jnp.float32)),
            np.asarray(codes.astype(jnp.float32)),
            np.asarray(scales.astype(jnp.float32)),
        )
        self.assertEqual(got.shape, (rows, size_n))
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-4)

    def test_rows_and_chunks(self):
        for rows in (1, 2, 4, 8):
            for kc in (128, 256, 512):
                with self.subTest(rows=rows, kc=kc):
                    self._case(rows, 512, 256, kc, jnp.bfloat16)

    def test_e4m3_scales(self):
        self._case(4, 256, 128, 256, jnp.float8_e4m3fn)

    def test_chunk_larger_than_k_is_clamped(self):
        self._case(2, 128, 128, 1024, jnp.bfloat16)


if __name__ == "__main__":
    unittest.main()
