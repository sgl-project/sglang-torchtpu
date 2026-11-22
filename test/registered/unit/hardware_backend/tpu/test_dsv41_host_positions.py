# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests: DSV4.1 extend positions built on the host from the extend lists."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.attention_backend import _host_extend_positions


def _batch(mode, prefix, extend):
    return SimpleNamespace(
        forward_mode=mode, extend_prefix_lens_cpu=prefix, extend_seq_lens_cpu=extend
    )


class TestHostExtendPositions(CustomTestCase):
    def test_matches_prefix_plus_arange_with_zero_padding(self):
        out = _host_extend_positions(_batch(ForwardMode.EXTEND, [5, 0], [3, 2]), 8)
        torch.testing.assert_close(out, torch.tensor([5, 6, 7, 0, 1, 0, 0, 0]))

    def test_other_modes_and_missing_lists_fall_back(self):
        self.assertIsNone(
            _host_extend_positions(_batch(ForwardMode.TARGET_VERIFY, [0], [4]), 4)
        )
        self.assertIsNone(
            _host_extend_positions(_batch(ForwardMode.EXTEND, None, [4]), 4)
        )
        self.assertIsNone(
            _host_extend_positions(_batch(ForwardMode.EXTEND, [0], [4, 1]), 8)
        )

    def test_explicit_spec_positions_fall_back(self):
        batch = _batch(ForwardMode.EXTEND, [0], [4])
        batch.spec_info = SimpleNamespace(positions=torch.arange(4))
        self.assertIsNone(_host_extend_positions(batch, 4))

    def test_more_extend_rows_than_tokens_falls_back(self):
        self.assertIsNone(
            _host_extend_positions(_batch(ForwardMode.EXTEND, [0], [9]), 8)
        )


if __name__ == "__main__":
    unittest.main()
