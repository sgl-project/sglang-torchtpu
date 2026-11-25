"""The TPU hook runs GatedResidual's mix/combine eagerly inside the compiled forward.

Core wraps them in torch.compile; on TPU the nested compiled function would run
on the outer trace's placeholders ("cannot Materialize() a placeholder tensor").
Pure CPU: checks the callables the hook installs and that they still compute.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang_tpu.hooks.hyperconnection import eager_gated_residual


def _residual():
    return GatedResidual(
        HyperConnectionConfig(
            hc_count=4,
            hidden_size=64,
            params_dtype=torch.float32,
            hc_lowrank=16,
            rms_norm_eps=1e-6,
            hc_per_branch_norm=True,
        )
    )


def _is_compiled(fn) -> bool:
    return hasattr(fn, "_torchdynamo_orig_callable")


class TestEagerGatedResidualHook(unittest.TestCase):
    def test_hook_unwraps_the_compiled_helpers(self):
        residual = _residual()
        self.assertTrue(_is_compiled(residual._mix_compute))
        self.assertTrue(_is_compiled(residual._combine_compute))
        eager_gated_residual(None, residual)
        self.assertFalse(_is_compiled(residual._mix_compute))
        self.assertFalse(_is_compiled(residual._combine_compute))

    def test_eager_helpers_compute_the_streams(self):
        residual = _residual()
        eager_gated_residual(None, residual)
        torch.manual_seed(0)
        mixed, saved = residual.mix(torch.randn(3, 4 * 64))
        self.assertEqual(tuple(mixed.shape), (3, 64))
        out = residual.combine(torch.randn(3, 64), saved)
        self.assertEqual(tuple(out.shape), (3, 4 * 64))

    def test_hook_is_registered_after_init(self):
        from sglang.srt.plugins.hook_registry import HookType
        from sglang_tpu.hooks import HOOKS

        by_target = {entry[0]: entry[1:] for entry in HOOKS}
        hook, hook_type = by_target[
            "sglang.srt.layers.hyperconnection.GatedResidual.__init__"
        ]
        self.assertIs(hook, eager_gated_residual)
        self.assertIs(hook_type, HookType.AFTER)


if __name__ == "__main__":
    unittest.main()
