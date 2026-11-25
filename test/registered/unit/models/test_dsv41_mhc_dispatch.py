"""The mHC boundary dispatch: G-B-lite for small rows, shifted mHC above, else original."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import hooks, qa_chain_op
from sglang_tpu.models.deepseek_v41.ops import mhc as tpu_mhc

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _y(rows):
    return SimpleNamespace(device=SimpleNamespace(type="tpu"), shape=(rows, 5120))


class MhcDispatchTest(CustomTestCase):
    def run_dispatch(self, rows, gblite, shifted, max_rows=None):
        env = (
            {}
            if max_rows is None
            else {"SGLANG_TPU_DSV41_MHC_GBLITE_MAX_ROWS": str(max_rows)}
        )
        original = mock.Mock(return_value="original")
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(
                qa_chain_op, "try_boundary_qa", return_value=gblite
            ) as qa,
            mock.patch.object(
                tpu_mhc, "fused_target_boundary", return_value=shifted
            ) as fused,
        ):
            out = hooks.tpu_dsv41_shifted_mhc(
                original, "hc", _y(rows), "r", "c", "next"
            )
        return out, qa.call_count, fused.call_count

    def test_small_rows_take_gblite_first(self):
        self.assertEqual(self.run_dispatch(48, "gb", "sh"), ("gb", 1, 0))

    def test_gblite_decline_falls_to_shifted(self):
        self.assertEqual(self.run_dispatch(96, None, "sh"), ("sh", 1, 1))

    def test_rows_above_threshold_skip_gblite(self):
        self.assertEqual(self.run_dispatch(192, "gb", "sh"), ("sh", 0, 1))
        self.assertEqual(self.run_dispatch(128, "gb", "sh", max_rows=96), ("sh", 0, 1))

    def test_both_decline_runs_original(self):
        self.assertEqual(self.run_dispatch(48, None, None), ("original", 1, 1))

    def test_shifted_env_off_declines_rows(self):
        with mock.patch.dict(os.environ, {"SGLANG_TPU_DSV41_SHIFTED_MHC": "0"}):
            self.assertFalse(tpu_mhc._use_shifted_mhc_for_rows(192))
        self.assertTrue(tpu_mhc._use_shifted_mhc_for_rows(192))

    def test_min_rows_env_opens_small_seams(self):
        # A c4 verify seam (6 rows per rank) declines at the default floor.
        self.assertFalse(tpu_mhc._use_shifted_mhc_for_rows(6))
        env = {"SGLANG_TPU_DSV41_SHIFTED_MHC_MIN_ROWS": "1"}
        with mock.patch.dict(os.environ, env):
            self.assertTrue(tpu_mhc._use_shifted_mhc_for_rows(6))
            self.assertFalse(tpu_mhc._use_shifted_mhc_for_rows(0))

    def test_no_next_runs_original(self):
        original = mock.Mock(return_value="original")
        self.assertEqual(
            hooks.tpu_dsv41_shifted_mhc(original, "hc", _y(48), "r", "c", None),
            "original",
        )


class FlatCollapseTest(CustomTestCase):
    def test_flat_collapse_matches_view_reduce(self):
        import torch

        from sglang_tpu.envs import envs

        g = torch.Generator().manual_seed(0)
        x = torch.randn(6, 4 * 8, generator=g).bfloat16()
        pre = torch.rand(6, 4, generator=g)
        view = hooks.tpu_dsv41_hc_combine(None, x, pre, 4, torch.bfloat16)
        with envs.SGLANG_TPU_DSV41_MHC_FLAT_COLLAPSE.override(True):
            flat = hooks.tpu_dsv41_hc_combine(None, x, pre, 4, torch.bfloat16)
            # Other stream counts keep the view.
            x3 = x[:, :24]
            self.assertTrue(
                torch.equal(
                    hooks.tpu_dsv41_hc_combine(None, x3, pre[:, :3], 3, torch.float32),
                    (pre[:, :3, None] * x3.reshape(6, 3, 8).float()).sum(1),
                )
            )
        # Bitwise on TPU (XLA's reduce order); CPU reduces in another order.
        torch.testing.assert_close(flat, view, rtol=2**-7, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
