import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_residual
from sglang.srt.layers.attn_residual import (
    _aggregate,
    _supports_attn_res_tma,
    aggregate_stream_torch,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestAttnResidual(unittest.TestCase):
    def test_tma_capability_gate(self):
        self.assertFalse(_supports_attn_res_tma((9, 0)))
        self.assertTrue(_supports_attn_res_tma((10, 0)))
        self.assertTrue(_supports_attn_res_tma((10, 3)))
        self.assertTrue(_supports_attn_res_tma((11, 0)))
        self.assertFalse(_supports_attn_res_tma((12, 0)))
        self.assertTrue(_supports_attn_res_tma((13, 0)))

    def test_aggregate_on_tpu_matches_reference(self):
        """TPU uses the eager reference."""
        torch.manual_seed(0)
        T, H, nvb = 3, 1024, 2
        score_proj = ReplicatedLinear(H, 1, bias=False)
        score_norm, out_norm = RMSNorm(H), RMSNorm(H)
        with torch.no_grad():
            score_proj.weight.normal_()
        prefix_sum, bank = torch.randn(T, H), torch.randn(T, nvb + 1, H)
        with patch.object(attn_residual, "_is_tpu", True):
            out = _aggregate(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)
        expected = out_norm(
            aggregate_stream_torch(prefix_sum, bank, nvb, score_proj, score_norm)
        )
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
