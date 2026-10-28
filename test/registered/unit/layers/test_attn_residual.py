import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_residual
from sglang.srt.layers.attn_residual import (
    _aggregate,
    _aggregate_fused_add,
    _supports_attn_res_tma,
    aggregate_stream_torch,
    register_aggregate_impl,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.platforms import current_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import enter_scope

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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


def _constant_impl(prefix_a, prefix_b, bank, nvb, score_proj, score_norm, out_norm):
    """A platform kernel that takes every call: constant normed, real prefix."""
    prefix = prefix_a if prefix_b is None else prefix_a + prefix_b
    return torch.full_like(prefix, 7.0), prefix


class TestAggregateRegistry(unittest.TestCase):
    T, H, NVB = 3, 1024, 2

    def setUp(self):
        enter_scope(self, patch.object(attn_residual, "_AGGREGATE_IMPLS", {}))
        enter_scope(self, patch.object(attn_residual, "_is_tpu", True))
        # Only the dispatch key: layers resolve their forward method through
        # the rest of the platform.
        enter_scope(
            self,
            patch.object(
                current_platform, "get_dispatch_key_name", lambda: "test_plugin"
            ),
        )
        torch.manual_seed(0)
        score_proj = ReplicatedLinear(self.H, 1, bias=False)
        with torch.no_grad():
            score_proj.weight.normal_()
        self.modules = (score_proj, RMSNorm(self.H), RMSNorm(self.H))
        self.a, self.b = torch.randn(self.T, self.H), torch.randn(self.T, self.H)
        self.bank = torch.randn(self.T, self.NVB + 1, self.H)

    def _builtin(self, prefix):
        score_proj, score_norm, out_norm = self.modules
        mixed = aggregate_stream_torch(
            prefix, self.bank, self.NVB, score_proj, score_norm
        )
        return out_norm(mixed)

    def test_registered_impl_takes_both_entry_points(self):
        register_aggregate_impl("test_plugin", _constant_impl)
        out = _aggregate(self.a, self.bank, self.NVB, *self.modules)
        self.assertTrue(torch.all(out == 7))
        normed, prefix = _aggregate_fused_add(
            self.a, self.b, self.bank, self.NVB, *self.modules
        )
        self.assertTrue(torch.all(normed == 7))
        torch.testing.assert_close(prefix, self.a + self.b)

    def test_unmatched_or_declining_impl_runs_the_builtin_path(self):
        register_aggregate_impl("other_plugin", _constant_impl)
        out = _aggregate(self.a, self.bank, self.NVB, *self.modules)
        torch.testing.assert_close(out, self._builtin(self.a))

        declined = []
        register_aggregate_impl(
            "test_plugin", lambda *args: declined.append(args) or None
        )
        normed, prefix = _aggregate_fused_add(
            self.a, self.b, self.bank, self.NVB, *self.modules
        )
        torch.testing.assert_close(prefix, self.a + self.b)
        torch.testing.assert_close(normed, self._builtin(self.a + self.b))
        # Declined with the addend, then again for the materialized prefix.
        self.assertEqual(len(declined), 2)
        self.assertIs(declined[0][1], self.b)
        self.assertIsNone(declined[1][1])

    def test_fused_bank_write_never_reaches_the_impl(self):
        """The bank write is the CUDA/HIP fast paths' contract."""
        register_aggregate_impl("test_plugin", _constant_impl)
        with self.assertRaisesRegex(AssertionError, "fast-path only"):
            _aggregate(self.a, self.bank, self.NVB, *self.modules, write_bank_row=True)

    def test_registration_rejects_conflicts_without_replacing_impl(self):
        register_aggregate_impl("test_plugin", _constant_impl)
        register_aggregate_impl("test_plugin", _constant_impl)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_aggregate_impl("test_plugin", lambda *args: None)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            register_aggregate_impl("", _constant_impl)
        out = _aggregate(self.a, self.bank, self.NVB, *self.modules)
        self.assertTrue(torch.all(out == 7))

    def test_dispatch_traces_without_graph_breaks(self):
        register_aggregate_impl("test_plugin", _constant_impl)

        def aggregate(prefix_a, prefix_b):
            return _aggregate_fused_add(
                prefix_a, prefix_b, self.bank, self.NVB, *self.modules
            )[0]

        compiled = torch.compile(aggregate, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(self.a, self.b), aggregate(self.a, self.b))


if __name__ == "__main__":
    unittest.main()
