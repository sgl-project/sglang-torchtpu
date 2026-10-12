"""INT4-resident FP8 GMM: compare with materialized FP8 weights, including EP holes."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:
    _HAS_TPU = False


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestInt4Fp8Gmm(CustomTestCase):
    def test_matches_materialized_fp8_compute(self):
        from sglang_tpu.moe.pallas.gmm_v2 import TileSizes, apply_act_fn, gmm_v2

        rng = np.random.default_rng(91)
        # Two K tiles exercise partial accumulation and final channel restoration.
        k, n = 1024, 256
        q = rng.integers(-8, 8, (2, k, n), dtype=np.int32)
        packed = np.bitwise_or.reduce(
            (q.reshape(2, k // 8, 8, n) & 15)
            << np.arange(0, 32, 4, dtype=np.int32)[None, None, :, None],
            axis=2,
        )
        scales = jnp.asarray(rng.uniform(1, 45, (2, k // 32, 1, n)), jnp.float32)
        channels = jnp.asarray(rng.uniform(0.001, 0.02, (2, 1, n)), jnp.float32)
        weights = jnp.clip(
            jnp.asarray(q, jnp.bfloat16)
            * jnp.repeat(scales[:, :, 0].astype(jnp.bfloat16), 32, axis=1),
            -448,
            448,
        ).astype(jnp.float8_e4m3fn)
        x = jnp.asarray(rng.normal(0, 0.3, (16, k)), jnp.bfloat16)
        for offset, groups in [(0, [4, 12]), (1, [2, 6, 0, 8]), (1, [2, 0, 0, 14])]:
            for fused in (None, "silu"):
                with self.subTest(offset=offset, groups=groups, fused=fused):
                    out = gmm_v2(
                        x,
                        jnp.asarray(packed),
                        jnp.asarray(groups, jnp.int32),
                        rhs_scale=scales,
                        rhs_channel_scale=channels,
                        rhs_quant_dtype=jnp.int4,
                        group_offset=jnp.array([offset], jnp.int32),
                        tile_info=TileSizes(16, 512, 128),
                        fuse_act=fused,
                    )
                    # Quantize activations with the kernel's own FP8 path: eager JAX
                    # bf16 rounds at different boundaries than Pallas.
                    materialized = gmm_v2(
                        x,
                        weights,
                        jnp.asarray(groups, jnp.int32),
                        rhs_scale=jnp.ones((2, k // 512, 1, n), jnp.float32),
                        group_offset=jnp.array([offset], jnp.int32),
                        tile_info=TileSizes(16, 512, 128),
                    )
                    expected = jnp.zeros((16, n // 2 if fused else n), jnp.bfloat16)
                    start = sum(groups[:offset])
                    for e in range(2):
                        size = groups[offset + e]
                        if size:
                            acc = materialized[start : start + size] * channels[
                                e
                            ].astype(jnp.bfloat16)
                            expected = expected.at[start : start + size].set(
                                apply_act_fn(acc, n // 2, fused)
                            )
                        start += size
                    if fused is None:
                        np.testing.assert_array_equal(
                            np.asarray(out), np.asarray(expected)
                        )
                    else:
                        # Fused SiLU moves bf16 rounding: the fused-activation tests'
                        # max-normalized tolerance, plus an L2 bound.
                        actual_f32 = np.asarray(out, np.float32)
                        expected_f32 = np.asarray(expected, np.float32)
                        delta = actual_f32 - expected_f32
                        self.assertLess(
                            float(np.max(np.abs(delta)))
                            / max(float(np.max(np.abs(expected_f32))), 1e-12),
                            0.02,
                        )
                        self.assertLess(
                            float(np.linalg.norm(delta))
                            / max(float(np.linalg.norm(expected_f32)), 1e-12),
                            0.02,
                        )
                    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    def test_rejects_split_activation_quantization_group(self):
        from sglang_tpu.moe.pallas.gmm_v2 import TileSizes, gmm_v2

        with self.assertRaisesRegex(ValueError, "activation quantization"):
            gmm_v2(
                jnp.ones((16, 1024), jnp.bfloat16),
                jnp.zeros((1, 128, 128), jnp.int32),
                jnp.array([16]),
                rhs_scale=jnp.ones((1, 32, 1, 128), jnp.float32),
                rhs_channel_scale=jnp.ones((1, 1, 128)),
                rhs_quant_dtype=jnp.int4,
                tile_info=TileSizes(16, 256, 128),
            )

    def test_channel_scale_cannot_silently_replace_bias(self):
        from sglang_tpu.moe.pallas.gmm_v2 import gmm_v2

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            gmm_v2(
                jnp.ones((16, 512), jnp.bfloat16),
                jnp.zeros((1, 64, 128), jnp.int32),
                jnp.array([16]),
                rhs_bias=jnp.ones((1, 1, 128)),
                rhs_channel_scale=jnp.ones((1, 1, 128)),
                rhs_quant_dtype=jnp.int4,
            )


if __name__ == "__main__":
    unittest.main()
