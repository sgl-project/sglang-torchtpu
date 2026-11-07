"""gmm_v2 vs the dense float32 reference where the tiling splits K or N.

Small shapes fit in one tile under ``calculate_tiling``, so the per-tile scale
window (``rhs_scale_index_map``) and the gate/up pairing of a fused activation
are only exercised at production sizes. Each case pins a ``TileSizes`` that
splits the dimension under test into several tiles and compares against
``sglang_tpu.testing.moe_reference.gmm_reference`` on the dequantized weights.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=90,
    suite="stage-a-test-tpu-kernel",
    labels=["kernel", "moe", "gmm"],
)

import unittest

import numpy as np

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    from sglang_tpu.testing import moe_reference as ref

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

_SKIP = "requires a TPU backend"

# Uneven groups with an empty one; rows sum to M.
_GROUP_SIZES = (96, 0, 112, 48)
_M = sum(_GROUP_SIZES)


def _lhs(k, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((_M, k)), jnp.float32).astype(jnp.bfloat16)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestGmmV2ReferenceModes(CustomTestCase):
    G = len(_GROUP_SIZES)

    def assert_matches(self, out, expected, tol):
        err = ref.max_rel_error(out, expected)
        self.assertLess(err, tol, f"max rel error {err:.4f}")

    def assert_scale_sensitive(self, values, scale, lhs, expected, tol):
        # A kernel that reused the first K block's scales everywhere must fail.
        first_block = jnp.broadcast_to(scale[:, :1], scale.shape)
        wrong = ref.gmm_reference(
            lhs, ref.apply_gmm_scale(values, first_block), jnp.asarray(_GROUP_SIZES)
        )
        self.assertGreater(ref.max_rel_error(wrong, expected), 3 * tol)

    def test_fp8_block_scales_across_k_tiles(self):
        """W8A8 block-128 scales with K split into 4 tiles of 2 scale blocks."""
        from sglang_tpu.moe.pallas.gmm_v2 import TileSizes, gmm_v2

        K, N = 1024, 512
        rng = np.random.default_rng(0)
        values = jnp.clip(rng.standard_normal((self.G, K, N)) * 64, -448, 448)
        rhs = jnp.asarray(values, jnp.float32).astype(jnp.float8_e4m3fn)
        # Per (128-K block, column) powers of two: exact in bf16, 16x spread.
        scale = jnp.exp2(
            rng.integers(-12, -7, (self.G, K // 128, 1, N)).astype(np.float32)
        ).astype(jnp.bfloat16)
        lhs = _lhs(K, seed=1)
        group_sizes = jnp.asarray(_GROUP_SIZES, jnp.int32)

        out = gmm_v2(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=scale,
            tile_info=TileSizes(tile_m=128, tile_k=256, tile_n=256),
            preferred_element_type=jnp.bfloat16,
        )
        q = rhs.astype(jnp.float32)
        expected = ref.gmm_reference(lhs, ref.apply_gmm_scale(q, scale), group_sizes)
        # fp8 activations, bf16 accumulator: a CPU emulation peaks near 0.026.
        tol = 0.06
        self.assert_matches(out, expected, tol)
        self.assert_scale_sensitive(q, scale, lhs, expected, tol)

    def test_mxfp4_vmem_dequant_across_k_tiles(self):
        """Packed MXFP4 (per-32 e8m0 scales, dequantized in VMEM) with K split
        into 4 tiles of 8 scale groups."""
        from sglang_tpu.moe.pallas.gmm_v2 import TileSizes, gmm_v2

        K, N = 1024, 256
        rng = np.random.default_rng(2)
        packed = jnp.asarray(rng.integers(0, 256, (self.G, K // 2, N)), jnp.uint8)
        # f32 e8m0 multipliers, as mxfp4_tpu_util.process_mxfp4_moe_weights_native stores.
        scale = ref.e8m0_to_f32(rng.integers(122, 131, (self.G, K // 32, 1, N)))
        lhs = _lhs(K, seed=3)
        group_sizes = jnp.asarray(_GROUP_SIZES, jnp.int32)

        out = gmm_v2(
            lhs,
            packed,
            group_sizes,
            rhs_scale=scale,
            packed_fp4=True,
            tile_info=TileSizes(tile_m=128, tile_k=256, tile_n=256),
            preferred_element_type=jnp.bfloat16,
        )
        values = ref.unpack_fp4(packed, axis=1)
        expected = ref.gmm_reference(
            lhs, ref.apply_gmm_scale(values, scale), group_sizes
        )
        # bf16 activations; fp4 times a power of two is exact in bf16.
        tol = 0.02
        self.assert_matches(out, expected, tol)
        self.assert_scale_sensitive(values, scale, lhs, expected, tol)

    def test_swigluoai_bias_across_n_tiles(self):
        """Fused swigluoai with gate/up biases, each half split into 3 N tiles:
        tile i of the gate must pair with tile i of the up half and its bias."""
        from sglang_tpu.moe.pallas.gmm_v2 import TileSizes, gmm_v2

        K, N = 256, 384
        rng = np.random.default_rng(4)
        # std(gate) ~ 8 around the swiglu limit of 7.
        rhs = jnp.asarray(rng.standard_normal((self.G, K, 2 * N)) * 0.5, jnp.float32)
        rhs = rhs.astype(jnp.bfloat16)
        bias = jnp.asarray(rng.standard_normal((self.G, 1, 2 * N)) * 2, jnp.float32)
        lhs = _lhs(K, seed=5)
        group_sizes = jnp.asarray(_GROUP_SIZES, jnp.int32)

        out = gmm_v2(
            lhs,
            rhs,
            group_sizes,
            rhs_bias=bias,
            fuse_act="swigluoai",
            tile_info=TileSizes(tile_m=128, tile_k=256, tile_n=128),
            preferred_element_type=jnp.bfloat16,
        )

        def expect(act):
            return ref.gmm_reference(lhs, rhs, group_sizes, rhs_bias=bias, fuse_act=act)

        expected = expect("swigluoai")
        # bf16 activations, f32 accumulation and activation.
        self.assert_matches(out, expected, 0.02)
        # The inputs must drive the clamp, or a wrong limit would pass.
        unclamped = expect(lambda g, u: ref.swiglu_oai(g, u, limit=float("inf")))
        self.assertGreater(ref.max_rel_error(unclamped, expected), 0.1)


if __name__ == "__main__":
    unittest.main()
