"""Pallas static fp8 decode GEMV vs the XLA path's math (TPU)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

FP8_MAX = 448.0


def _reference(x, w, out_scale, inv_x):
    """apply_tpu_static_fp8_linear: clamp-then-cast quantize, fp8 dot with f32
    accumulation, f32 per-channel scale, cast to x.dtype."""
    xq = jnp.clip(x.astype(jnp.float32) * inv_x[0], -FP8_MAX, FP8_MAX).astype(w.dtype)
    acc = jnp.dot(xq, w, preferred_element_type=jnp.float32)
    return (acc * out_scale[None, :]).astype(x.dtype)


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestStaticFp8GemvPallas(CustomTestCase):
    def _case(self, m, k, n, seed):
        from sglang_tpu.kernels.static_fp8_gemv import block_k, static_fp8_gemv_jax

        rng = np.random.default_rng(seed)
        # An RMSNorm-like activation within the static scale's range.
        x = jnp.asarray(rng.standard_normal((m, k)) * 0.5, jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((k, n)) * 64, jnp.float32).astype(
            jnp.float8_e4m3fn
        )
        out_scale = jnp.asarray(rng.uniform(1e-4, 1e-3, n), jnp.float32)
        inv_x = jnp.asarray([FP8_MAX / 4.0], jnp.float32)
        bk = block_k(k, n)
        self.assertGreater(bk, 0)
        got = jax.jit(static_fp8_gemv_jax, static_argnums=(4,))(
            x, w, out_scale, inv_x, bk
        )
        want = _reference(x, w, out_scale, inv_x)
        self.assertEqual(got.shape, (m, n))
        self.assertEqual(got.dtype, jnp.bfloat16)
        g = np.asarray(got, np.float32)
        r = np.asarray(want, np.float32)
        # Same products, f32 accumulation in a different K order: within a
        # bf16 ulp of the reference.
        err = np.abs(g - r).max() / np.abs(r).max()
        self.assertLess(err, 1e-2, f"m={m} k={k} n={n} bk={bk}")

    def test_kimi_k3_tp32_shapes(self):
        # KDA qkvg, MoE front, latent up (per TP32 rank), and a padded N.
        for k, n in ((7168, 1536), (7168, 3968), (3584, 7168), (7168, 2176)):
            for m in (1, 4, 16):
                self._case(m, k, n, seed=k + n + m)


if __name__ == "__main__":
    unittest.main()
