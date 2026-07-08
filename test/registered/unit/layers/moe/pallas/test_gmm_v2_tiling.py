"""Unit tests for gmm_v2.calculate_tiling (fuse-act VMEM accounting).

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
    import jax.numpy as jnp

    from sglang.srt.layers.moe.pallas import gmm_v2

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


if __name__ == "__main__":
    unittest.main()
