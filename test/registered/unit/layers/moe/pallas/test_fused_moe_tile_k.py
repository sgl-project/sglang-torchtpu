"""Unit tests for fused_moe_op._dividing_tile_k (quant-block-safe tile_k).

Pure tile-sizing math, no TPU: ``_dividing_tile_k`` only reads ``dims.size_k``
and ``rhs_cfgs``, so the wrapped tile fn is driven with hand-built dataclasses.
Registered on both lanes and skipped gracefully where JAX / the pallas import
chain is unavailable: the CPU agents have no JAX, so the TPU lane is the one
that actually executes these assertions.

The regression under test: ``calculate_tiling`` rejects a ``tile_k`` that
straddles rhs quantization blocks, but the tuned-lookup and pinned tile paths
bypass it. A straddling tile_k does not fail loudly -- ``IndexMaps``
``rhs_scale_index_map`` just loads the wrong scale window for every
``k_id >= 1``. NVFP4 is the exposed case: ``TUNED_BLOCK_SIZES`` carries no fp4
entries, so every NVFP4 call falls through to the heuristic, and the heuristic
returns ``tile_k`` values (e.g. 1280 for ``size_k=5120``) that are neither a
multiple nor a divisor of the 512 block size.
"""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax.numpy as jnp

    _HAS_GMM = True
except ImportError:  # pragma: no cover - JAX is absent on some CPU CI agents
    _HAS_GMM = False

# Deliberately outside the try: once JAX is present, a failure to import the
# pallas chain is a real breakage and must surface, not silently skip the suite.
if _HAS_GMM:
    from sglang_tpu.layers.moe.pallas import gmm_v2
    from sglang_tpu.layers.moe.pallas.fused_moe_op import (
        _dividing_tile_k,
        _gmm2_tile_source,
        _largest_compatible_tile_k,
        _quant_block_compatible,
        _vmem_safe_packed_tile_n,
    )

_SKIP_REASON = "Requires JAX"

# NVFP4 block size on the TPU path (TPU_NVFP4_GROUP_SIZE).
_NVFP4_BLOCK = 512


@unittest.skipUnless(_HAS_GMM, _SKIP_REASON)
class TestDividingTileK(CustomTestCase):
    @staticmethod
    def _dims(size_k):
        return gmm_v2.Dimensions(
            size_m=4096,
            size_k=size_k,
            size_n=8192,
            size_group=16,
            size_lhs_group=16,
            size_lhs_sublane=8,
        )

    @staticmethod
    def _rhs_cfgs(size_k, quant_block_size):
        """An rhs config carrying a per-block scale, as NVFP4 produces."""
        if quant_block_size is None:
            return gmm_v2.InputConfigs(
                quant_dtype=None,
                quant_block_size=size_k,
                dtype=jnp.bfloat16,
                has_scale=False,
            )
        return gmm_v2.InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=quant_block_size,
            dtype=jnp.uint8,
            has_scale=True,
            packing=8,
            num_quant_blocks=size_k // quant_block_size,
            scale_dtype=jnp.float32,
        )

    def _resolve(self, size_k, tile_k, quant_block_size=_NVFP4_BLOCK):
        """Run a pinned TileSizes through the wrapper, as the tuned lookup does."""
        pinned = gmm_v2.TileSizes(tile_m=512, tile_k=tile_k, tile_n=512)
        wrapped = _dividing_tile_k(pinned)
        return wrapped(
            self._dims(size_k),
            SimpleNamespace(quant_block_size=512, has_scale=False),
            self._rhs_cfgs(size_k, quant_block_size),
            vmem_limit_bytes=128 * 1024 * 1024,
        )

    def test_int4_to_fp8_preserves_activation_quantization_groups(self):
        import dataclasses

        rhs = dataclasses.replace(self._rhs_cfgs(7168, 32), requantize_in_vmem=True)
        for requested_k in (256, 1280, 1792):
            tiles = _dividing_tile_k(gmm_v2.TileSizes(16, requested_k, 128))(
                self._dims(7168),
                SimpleNamespace(quant_block_size=512),
                rhs,
                128 * 1024 * 1024,
            )
            self.assertEqual(tiles.tile_k % 512, 0)
            self.assertEqual(7168 % tiles.tile_k, 0)

    # ---- the predicate itself -------------------------------------------

    def test_predicate_matches_calculate_tiling(self):
        # Mirrors gmm_v2.calculate_tiling._is_tile_k_quant_block_compatible.
        self.assertTrue(_quant_block_compatible(2048, 512))  # multiple
        self.assertTrue(_quant_block_compatible(512, 512))  # equal
        self.assertTrue(_quant_block_compatible(128, 512))  # divisor
        self.assertFalse(_quant_block_compatible(1280, 512))  # straddles
        self.assertFalse(_quant_block_compatible(640, 512))  # straddles
        self.assertTrue(_quant_block_compatible(1280, None))  # unquantized

    # ---- the hazard this change closes ----------------------------------

    def test_incompatible_intermediate_size_is_repaired(self):
        # size_k=5120 is the motivating case: the heuristic hands back
        # tile_k=1280, which is 128-aligned AND divides 5120, so the old
        # divides-only check passed it through. But cdiv(1280, 512) == 3, so at
        # k_id=1 the kernel needs scale blocks 2..4 and loads 0..2 instead:
        # wrong scales over ~75% of the contraction, silently.
        tiles = self._resolve(size_k=5120, tile_k=1280)
        self.assertNotEqual(tiles.tile_k, 1280)
        self.assertEqual(tiles.tile_k, 1024)
        self.assertEqual(5120 % tiles.tile_k, 0)
        self.assertTrue(_quant_block_compatible(tiles.tile_k, _NVFP4_BLOCK))

    def test_all_known_incompatible_widths_are_repaired(self):
        # Every intermediate_size whose heuristic tile_k straddles a 512 block.
        for size_k in (2560, 3584, 5120, 7168):
            with self.subTest(size_k=size_k):
                heuristic_tile_k = min(size_k, 2048)
                # Step down to a 128-aligned divisor, as the heuristic does.
                while size_k % heuristic_tile_k != 0:
                    heuristic_tile_k -= 128
                tiles = self._resolve(size_k, heuristic_tile_k)
                self.assertEqual(size_k % tiles.tile_k, 0)
                self.assertTrue(
                    _quant_block_compatible(tiles.tile_k, _NVFP4_BLOCK),
                    f"{size_k=} resolved a straddling {tiles.tile_k=}",
                )
                self.assertLessEqual(tiles.tile_k, heuristic_tile_k)

    # ---- no-op on everything that already worked ------------------------

    def test_tier5_shape_is_unchanged(self):
        # Tier5: hidden 8192 (GMM1 k) and routed hidden 4096 (GMM2 k) both
        # resolve tile_k=2048, already a multiple of 512. Must not move.
        for size_k in (4096, 8192):
            with self.subTest(size_k=size_k):
                tiles = self._resolve(size_k, tile_k=2048)
                self.assertEqual(tiles.tile_k, 2048)

    def test_unquantized_rhs_keeps_divides_only_behavior(self):
        # No scale -> no block constraint; only the pre-existing divides check
        # applies, so a straddling-looking tile_k is left alone.
        tiles = self._resolve(size_k=5120, tile_k=1280, quant_block_size=None)
        self.assertEqual(tiles.tile_k, 1280)

    def test_channelwise_scale_is_unconstrained(self):
        # Channelwise: quant_block_size == size_k, so any divisor is compatible.
        tiles = self._resolve(size_k=5120, tile_k=1280, quant_block_size=5120)
        self.assertEqual(tiles.tile_k, 1280)

    def test_block_fp8_is_unaffected(self):
        # block-FP8's 128 divides every 128-aligned tile_k.
        tiles = self._resolve(size_k=5120, tile_k=1280, quant_block_size=128)
        self.assertEqual(tiles.tile_k, 1280)

    def test_non_dividing_tile_k_still_repaired(self):
        # The original responsibility of the wrapper survives: a tile_k that
        # does not divide size_k is stepped down even when compatible.
        tiles = self._resolve(size_k=4096, tile_k=1536)
        self.assertEqual(4096 % tiles.tile_k, 0)
        self.assertLessEqual(tiles.tile_k, 1536)

    # ---- the helper -----------------------------------------------------

    def test_largest_compatible_tile_k_picks_the_largest(self):
        # Divisors of 5120 that are 128-aligned and 512-compatible, descending:
        # 5120, 2560, 1024, 512, 256, 128 (640 and 1280 straddle).
        self.assertEqual(_largest_compatible_tile_k(5120, 2048, 512), 1024)
        self.assertEqual(_largest_compatible_tile_k(5120, 1024, 512), 1024)
        self.assertEqual(_largest_compatible_tile_k(5120, 900, 512), 512)

    def test_largest_compatible_tile_k_always_succeeds_for_real_configs(self):
        # tile_k=128 is compatible with every power-of-two block size, and the
        # quantized path already requires size_k % 128 == 0, so the search can
        # always fall back to 128. Real configs never hit the raise.
        for size_k in (2560, 3584, 4096, 5120, 7168, 8192):
            for quant_block_size in (128, 256, 512, 1024, 2048):
                with self.subTest(size_k=size_k, qbs=quant_block_size):
                    tile_k = _largest_compatible_tile_k(
                        size_k, size_k, quant_block_size
                    )
                    self.assertEqual(size_k % tile_k, 0)
                    self.assertTrue(_quant_block_compatible(tile_k, quant_block_size))

    def test_largest_compatible_tile_k_raises_when_impossible(self):
        # Defensive branch, unreachable with power-of-two block sizes. Forced
        # here with a synthetic non-power-of-two: the 128-aligned divisors of
        # 384 are 128 and 384, and neither divides nor is divided by 320.
        with self.assertRaisesRegex(ValueError, "compatible with quant_block_size"):
            _largest_compatible_tile_k(384, 384, 320)


@unittest.skipUnless(_HAS_GMM, _SKIP_REASON)
class TestGmm2TileSource(CustomTestCase):
    """The gmm2 tiling gate must stay narrow.

    Only packed FP4 needs the tuned lookup (``calculate_tiling`` would read the
    halved packed K as the real one). Every other dtype must keep the kernel's
    adaptive ``calculate_tiling``, which is the only source that budgets the rhs
    tile against ``rhs_vmem_target`` -- a pinned ``TileSizes`` is used verbatim
    with no clamp, so widening this gate silently retunes stock TPU MoE.
    """

    @staticmethod
    def _w2(dtype, size_k=4096, size_n=8192, num_experts=16):
        return jnp.zeros((num_experts, size_k, size_n), dtype=dtype)

    def _source(self, dtype, *, w2_packed, w2_scale=None):
        # Real gmm2 K is 4096 (Tier5 routed hidden). Packed FP4 stores 2 codes
        # per byte along K, so its stored K is half that.
        return _gmm2_tile_source(
            jnp.zeros((4096, 4096), dtype=jnp.bfloat16),
            self._w2(dtype, size_k=2048 if w2_packed else 4096),
            w2_scale,
            16,
            w2_packed=w2_packed,
        )

    def test_bf16_keeps_the_adaptive_kernel_tiling(self):
        self.assertIs(
            self._source(jnp.bfloat16, w2_packed=False), gmm_v2.calculate_tiling
        )

    def test_fp8_keeps_the_adaptive_kernel_tiling(self):
        # Channelwise and block-scaled FP8 both stay on calculate_tiling; the
        # packed-FP4 lookup is not a general "quantized w2" path.
        for w2_scale in (
            jnp.ones((16, 1, 1, 8192), dtype=jnp.float32),
            jnp.ones((16, 32, 8192), dtype=jnp.float32),
        ):
            with self.subTest(scale_shape=w2_scale.shape):
                self.assertIs(
                    self._source(jnp.float8_e4m3fn, w2_packed=False, w2_scale=w2_scale),
                    gmm_v2.calculate_tiling,
                )

    def test_packed_fp4_uses_the_tuned_lookup(self):
        # uint8-packed FP4 is the one case that must not reach calculate_tiling.
        tiles = self._source(
            jnp.uint8,
            w2_packed=True,
            w2_scale=jnp.ones((16, 8, 8192), dtype=jnp.float32),
        )
        self.assertIsInstance(tiles, gmm_v2.TileSizes)


@unittest.skipUnless(_HAS_GMM, _SKIP_REASON)
class TestVmemSafePackedTileN(CustomTestCase):
    """``_vmem_safe_packed_tile_n`` keeps room for the unpack staging tile.

    ``calculate_tiling`` sizes the rhs tile from the stored bit width and budgets
    three pipelined buffers, but a W4A16 rhs is also unpacked into a staging tile
    before the matmul. On Kimi-K2.6's GMM1 that gap put the kernel 5.17 MB over
    its own 57.60 MB limit (measured on v7x), which Mosaic reports as
    ``CompileTimeScopedVmemOom``.
    """

    # v7x: 0.9 * 64 MiB, the default gmm_v2 passes to calculate_tiling.
    _VMEM = int(64 * 1024 * 1024 * 0.9)

    @staticmethod
    def _dims(size_k, size_n):
        return gmm_v2.Dimensions(
            size_m=512,
            size_k=size_k,
            size_n=size_n,
            size_group=24,
            size_lhs_group=24,
            size_lhs_sublane=16,
        )

    @staticmethod
    def _packed_int4_cfgs(size_k):
        """Kimi's compressed-tensors rhs: int4 packed 8-per-int32, 32-wide blocks."""
        return gmm_v2.InputConfigs(
            quant_dtype=jnp.int4.dtype,
            quant_block_size=32,
            dtype=jnp.int4,
            has_scale=True,
            packing=8,
            num_quant_blocks=size_k // 32,
            scale_dtype=jnp.bfloat16,
        )

    def _resolve(self, size_k, size_n, tile_n, rhs_cfgs):
        pinned = gmm_v2.TileSizes(tile_m=64, tile_k=size_k, tile_n=tile_n)
        wrapped = _vmem_safe_packed_tile_n(pinned)
        return wrapped(
            self._dims(size_k, size_n),
            SimpleNamespace(quant_block_size=512, has_scale=False),
            rhs_cfgs,
            vmem_limit_bytes=self._VMEM,
        )

    def test_kimi_gmm1_tile_n_is_halved(self):
        # The shape that OOMs: full-n leaves no room for the staging tile.
        tiles = self._resolve(7168, 4096, 4096, self._packed_int4_cfgs(7168))
        self.assertEqual(tiles.tile_n, 2048)
        self.assertEqual(tiles.tile_k, 7168)
        self.assertEqual(tiles.tile_m, 64)

    def test_kimi_gmm2_tile_is_left_alone(self):
        # Already fits, so it must keep the tile it compiles with today.
        tiles = self._resolve(2048, 7168, 7168, self._packed_int4_cfgs(2048))
        self.assertEqual(tiles.tile_n, 7168)

    def test_bf16_weights_are_untouched(self):
        cfgs = gmm_v2.InputConfigs(
            quant_dtype=None,
            quant_block_size=7168,
            dtype=jnp.bfloat16,
            has_scale=False,
        )
        tiles = self._resolve(7168, 4096, 4096, cfgs)
        self.assertEqual(tiles.tile_n, 4096)

    def test_fp8_weights_are_untouched(self):
        # As make_gmm_configs builds them (fp8, four codes per word); fp8 has no
        # unpack tile, so the DeepSeek GMM1 tile keeps the kernel's choice.
        cfgs = gmm_v2.InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=128,
            dtype=jnp.float8_e4m3fn,
            has_scale=True,
            packing=4,
            num_quant_blocks=7168 // 128,
            scale_dtype=jnp.float32,
        )
        tiles = self._resolve(7168, 4096, 4096, cfgs)
        self.assertEqual(tiles.tile_n, 4096)

    def test_result_is_a_fixed_point(self):
        # Re-wrapping an already-safe tile must not shrink it again.
        cfgs = self._packed_int4_cfgs(7168)
        once = self._resolve(7168, 4096, 4096, cfgs)
        twice = self._resolve(7168, 4096, once.tile_n, cfgs)
        self.assertEqual(twice.tile_n, once.tile_n)

    def test_chosen_tile_fits_the_staging_budget(self):
        tiles = self._resolve(7168, 4096, 4096, self._packed_int4_cfgs(7168))
        tile_bytes = tiles.tile_k * tiles.tile_n * 4 // 8  # int4 rhs
        self.assertLessEqual(tile_bytes, self._VMEM // 5)


if __name__ == "__main__":
    unittest.main()
