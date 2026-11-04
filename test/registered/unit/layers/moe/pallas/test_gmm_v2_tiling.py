"""Unit tests for gmm_v2.calculate_tiling (fuse-act VMEM accounting) and
gmm_v2.pad_size_n_to_fewest_tiles (load-time N padding).

These exercise the pure tile-sizing math, so they need no TPU: ``get_tpu_info``
(the only hardware dependency) is stubbed with the tpu7x lane/MXU geometry, and
``vmem_limit_bytes`` is passed explicitly. Registered as a CPU CI test but
skipped gracefully where JAX / the pallas import chain is unavailable.
"""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    from sglang_tpu.layers.moe.pallas import gmm_v2

    _HAS_GMM = True
except Exception:  # pragma: no cover - depends on JAX/pallas availability
    _HAS_GMM = False

_SKIP_REASON = "Requires JAX + the gmm_v2 pallas import chain"

# tpu7x geometry; calculate_tiling only reads mxu_column_size and num_lanes.
_TPU_INFO = SimpleNamespace(mxu_column_size=128, num_lanes=128)

_MiB = 1024 * 1024


@unittest.skipUnless(_HAS_GMM, _SKIP_REASON)
class TestCalculateTilingFuseAct(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            gmm_v2.pltpu, "get_tpu_info", return_value=_TPU_INFO
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _dims(size_m=128, size_k=1024, size_n=1024):
        return gmm_v2.Dimensions(
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            size_group=8,
            size_lhs_group=8,
            size_lhs_sublane=8,
        )

    @staticmethod
    def _bf16_cfgs(size_k=1024):
        # Dense (non-quantized) bf16 lhs/rhs, mirroring make_gmm_configs'
        # non-quant branch (rhs quant_block_size == size_k, lhs == 512).
        lhs = gmm_v2.InputConfigs(
            quant_dtype=None, quant_block_size=512, dtype=jnp.bfloat16
        )
        rhs = gmm_v2.InputConfigs(
            quant_dtype=None, quant_block_size=size_k, dtype=jnp.bfloat16
        )
        return lhs, rhs

    def _tiling(self, vmem_limit_bytes, fuse_act):
        lhs, rhs = self._bf16_cfgs()
        return gmm_v2.calculate_tiling(
            self._dims(),
            lhs,
            rhs,
            vmem_limit_bytes=vmem_limit_bytes,
            fuse_act=fuse_act,
        )

    def test_fuse_act_tiles_valid(self):
        # With ample VMEM the fused path returns full, valid tiles.
        tiles = self._tiling(128 * _MiB, fuse_act="silu")
        self.assertGreater(tiles.tile_m, 0)
        self.assertGreater(tiles.tile_k, 0)
        self.assertGreater(tiles.tile_n, 0)

    def test_fuse_act_scratch_shrinks_tiling_under_pressure(self):
        # The fuse-act scratch reservation lowers the RHS VMEM target, so a
        # tighter budget must yield strictly smaller RHS tiles.
        roomy = self._tiling(128 * _MiB, fuse_act="silu")
        tight = self._tiling(4 * _MiB, fuse_act="silu")
        self.assertLess(tight.tile_n * tight.tile_k, roomy.tile_n * roomy.tile_k)

    def test_fuse_act_shrinks_tile_m_to_fit(self):
        # When the heuristic tile_m's scratch would crowd out the RHS buffers,
        # calculate_tiling shrinks tile_m (aligned down to size_lhs_sublane=8)
        # instead of failing, so a mid-range budget still yields valid tiles with
        # a smaller-than-default tile_m.
        roomy = self._tiling(128 * _MiB, fuse_act="silu")
        tight = self._tiling(2_000_000, fuse_act="silu")
        self.assertLess(tight.tile_m, roomy.tile_m)
        self.assertEqual(tight.tile_m % 8, 0)
        self.assertGreaterEqual(tight.tile_m, 8)
        self.assertGreater(tight.tile_n, 0)
        self.assertGreater(tight.tile_k, 0)

    def test_fuse_act_min_tile_m_too_big_raises(self):
        # Only when even the minimum tile_m (one sublane) can't fit -- its scratch
        # alone exceeds half the budget -- does calculate_tiling give up and raise
        # (rather than spin the tiling loops on a non-positive RHS target).
        with self.assertRaisesRegex(ValueError, "over\\s+half"):
            self._tiling(1024, fuse_act="silu")

    def test_non_fused_no_scratch_reservation(self):
        # The non-fused byte model reserves no scratch, so the same tiny budget
        # that trips the fused guard still produces valid tiles here (behavior is
        # byte-identical to before the fuse-act change).
        tiles = self._tiling(4 * _MiB, fuse_act=None)
        self.assertGreater(tiles.tile_k, 0)
        self.assertGreater(tiles.tile_n, 0)

    def test_weight_only_reconstruction_fits_vmem(self):
        # Kimi's gate/up prefill: the packed INT4 tile fit, its BF16 unpack did not.
        dims = self._dims(size_m=32768, size_k=7168, size_n=4096)
        lhs, _ = self._bf16_cfgs(size_k=dims.size_k)
        rhs = gmm_v2.InputConfigs(
            quant_dtype=jnp.int4,
            quant_block_size=32,
            dtype=jnp.int4,
            has_scale=True,
            packing=8,
            num_quant_blocks=224,
            scale_size_n=4096,
            scale_dtype=jnp.bfloat16,
        )
        budget = int(64 * _MiB * 0.9)
        tiles = gmm_v2.calculate_tiling(dims, lhs, rhs, budget)
        elements = tiles.tile_k * tiles.tile_n
        dma_bytes = 3 * (elements // 2 + tiles.tile_k // 32 * tiles.tile_n * 2)
        reconstructed_bytes = elements * 2
        self.assertLessEqual(dma_bytes + reconstructed_bytes, budget)
        # N tiling can resolve this without changing K accumulation order.
        self.assertEqual(tiles.tile_k, dims.size_k)


# jax's tpu7x metadata. pad_size_n_to_fewest_tiles also reads the VMEM
# capacity (for gmm_v2's default budget) and the bf16 sublane tiling.
_TPU7X_INFO = SimpleNamespace(
    mxu_column_size=256,
    num_lanes=128,
    vmem_capacity_bytes=64 * _MiB,
    get_sublane_tiling=lambda dtype: 16,
)

# GMM1's K in GPT-OSS's MoE: hidden 2880 padded to 128 lanes.
_GPT_OSS_K = 2944
# Routed rows of a prefill chunk; every M from 128 rows up tiles alike.
_PREFILL_M = 16384


@unittest.skipUnless(_HAS_GMM, _SKIP_REASON)
class TestPadSizeNToFewestTiles(CustomTestCase):
    """Shapes default to GPT-OSS's GMM1: bf16, K = 2944, gate/up fused."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            gmm_v2.pltpu, "get_tpu_info", return_value=_TPU7X_INFO
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _padded(size_n, size_k=_GPT_OSS_K, fuse_act="swigluoai", vmem_limit_bytes=None):
        return gmm_v2.pad_size_n_to_fewest_tiles(
            size_k,
            size_n,
            jnp.bfloat16,
            fuse_act=fuse_act,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    @staticmethod
    def _tile_n(size_m, size_n, size_k=_GPT_OSS_K, fuse_act="swigluoai"):
        """The N tile make_gmm_configs runs, on the default VMEM budget, for
        ``size_n`` columns per projection."""
        spec = jax.ShapeDtypeStruct
        rhs_n = 2 * size_n if fuse_act else size_n
        cfgs = gmm_v2.make_gmm_configs(
            spec((size_m, size_k), jnp.bfloat16),
            spec((1, size_k, rhs_n), jnp.bfloat16),
            None,
            None,
            spec((1,), jnp.int32),
            spec((1,), jnp.int32),
            tile_info=gmm_v2.calculate_tiling,
            vmem_limit_bytes=gmm_v2.default_vmem_limit_bytes(),
            out_dtype=None,
            acc_dtype=None,
            maybe_quantize_lhs=True,
            zero_initialize=True,
            fuse_act=fuse_act,
        )
        return cfgs.tiles.tile_n

    def _num_tiles(self, size_n, **shape):
        return size_n // self._tile_n(_PREFILL_M, size_n, **shape)

    def test_gpt_oss_tp1_intermediate_pads_to_3072(self):
        """2880 lane-aligns to 2944 = 23 * 128, which the 1024-wide tile the
        tiler asks for does not divide, so GMM1 falls back to 23 tiles of 128;
        3072 runs three tiles of 1024 (3200 = 25 * 128 would run five)."""
        self.assertEqual(self._tile_n(_PREFILL_M, 2944), 128)
        for size_n in (2880, 2944):
            with self.subTest(size_n=size_n):
                self.assertEqual(self._padded(size_n), 3072)
        self.assertEqual(self._tile_n(_PREFILL_M, 3072), 1024)

    def test_padded_width_keeps_its_prefill_tile_at_every_m(self):
        """Fewer rows leave more VMEM for N: under 128 rows the tile widens
        to 1536, which also divides 3072."""
        padded = self._padded(2880)
        for size_m in (16, 32, 64, 128, 256, 1024):
            with self.subTest(size_m=size_m):
                self.assertGreaterEqual(self._tile_n(size_m, padded), 1024)

    def test_widths_already_at_fewest_tiles_are_kept(self):
        """GPT-OSS per rank at TP2/4/8 (1440, 720, 384) runs one tile at its
        128-aligned width, and so does 2880 with 128 MiB of VMEM."""
        for size_n, expected in ((1440, 1536), (720, 768), (384, 384)):
            with self.subTest(size_n=size_n):
                self.assertEqual(self._padded(size_n), expected)
        self.assertEqual(
            self._padded(2880, vmem_limit_bytes=int(128 * _MiB * 0.9)), 2944
        )

    def test_a_tie_keeps_the_narrower_width(self):
        """2304 runs as few tiles as 2048: two, of 1152 and of 1024."""
        self.assertEqual(self._num_tiles(2048), 2)
        self.assertEqual(self._num_tiles(2304), 2)
        self.assertEqual(self._padded(2048), 2048)

    def test_padding_contract_over_a_shape_sweep(self):
        """The width is lane aligned, at most an eighth past the lane-aligned
        width, and never runs more tiles; it grows only for fewer tiles, and
        no smaller M tiles it narrower than prefill does."""
        violations = []
        for fuse_act in (None, "swigluoai"):
            for size_k in (2944, 4096, 7168):
                shape = dict(size_k=size_k, fuse_act=fuse_act)
                for size_n in range(64, 8192 + 1, 64):
                    aligned = -(-size_n // 128) * 128
                    padded = self._padded(size_n, **shape)
                    tile_n = self._tile_n(_PREFILL_M, padded, **shape)
                    tiles = padded // tile_n
                    aligned_tiles = self._num_tiles(aligned, **shape)
                    ok = (
                        padded % 128 == 0
                        and aligned <= padded <= aligned + aligned // 8
                        and tiles <= aligned_tiles
                        and (padded == aligned or tiles < aligned_tiles)
                        and all(
                            self._tile_n(size_m, padded, **shape) >= tile_n
                            for size_m in (16, 32, 64)
                        )
                    )
                    if not ok:
                        violations.append((fuse_act, size_k, size_n, padded))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
