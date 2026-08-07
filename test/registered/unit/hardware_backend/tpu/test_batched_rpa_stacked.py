# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the batched-RPA stacked decode schedule."""

import unittest

import numpy as np

from sglang.srt.layers.attention.pallas.batched_rpa.kernel import (
    combine_stacked_partials,
)
from sglang.srt.layers.attention.pallas.batched_rpa.schedule import (
    stacked_lane_has_work,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestStackedRpaFinalization(CustomTestCase):
    BATCH_SIZE = 8
    NUM_K_CASES = (1, 8, 9, 15, 16, 17)

    def test_lane_validity_keeps_every_lane_with_work(self):
        for num_k in self.NUM_K_CASES:
            actual = [
                bool(stacked_lane_has_work(num_k, self.BATCH_SIZE, lane))
                for lane in range(self.BATCH_SIZE)
            ]
            expected = [
                lane < min(num_k, self.BATCH_SIZE) for lane in range(self.BATCH_SIZE)
            ]
            self.assertEqual(actual, expected, f"num_k={num_k}")

    def test_partial_merge_matches_global_softmax(self):
        import jax.numpy as jnp

        for num_k in self.NUM_K_CASES:
            scores = np.linspace(-1.5, 2.0, num_k, dtype=np.float32)
            values = np.linspace(0.25, 1.75, num_k, dtype=np.float32)
            m_all = []
            l_all = []
            acc_all = []

            for lane in range(self.BATCH_SIZE):
                lane_scores = scores[lane :: self.BATCH_SIZE]
                lane_values = values[lane :: self.BATCH_SIZE]
                if bool(stacked_lane_has_work(num_k, self.BATCH_SIZE, lane)):
                    lane_m = lane_scores.max()
                    weights = np.exp(lane_scores - lane_m)
                    lane_l = weights.sum()
                    lane_acc = (weights * lane_values).sum()
                else:
                    lane_m, lane_l, lane_acc = -np.inf, 0.0, 0.0
                m_all.append([[[lane_m]]])
                l_all.append([[[lane_l]]])
                acc_all.append([[[lane_acc]]])

            _, merged_l, merged_acc = combine_stacked_partials(
                jnp.asarray(m_all), jnp.asarray(l_all), jnp.asarray(acc_all)
            )
            expected_weights = np.exp(scores - scores.max())
            expected = np.sum(expected_weights * values) / expected_weights.sum()
            actual = float((merged_acc / merged_l).reshape(()))
            self.assertAlmostEqual(
                actual, float(expected), places=5, msg=f"num_k={num_k}"
            )


if __name__ == "__main__":
    unittest.main()
