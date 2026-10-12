import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.runtime_context import get_context
from sglang_tpu.platform import TpuOOTPlatform
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_TPU_MOD = "sglang_tpu.tpu_platform"
_CHECKER_MOD = "sglang_tpu.compilation.recompilation_checker"


class TestTpuWarmupCompleteStance(unittest.TestCase):
    def _publish_args(self, *, speculative_algorithm, enable_dp_attention=False):
        # on_warmup_complete reads the exec and spec bags; publish both.
        override = get_context().override_server_args(
            enable_torch_compile=True,
            speculative_algorithm=speculative_algorithm,
            enable_dp_attention=enable_dp_attention,
        )
        override.install()
        self.addCleanup(override.restore)

    def _warmup_complete(self, *, allow_spec_recompile):
        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(allow_spec_recompile),
            patch(f"{_TPU_MOD}.torch.compiler.set_stance") as set_stance,
            patch(f"{_CHECKER_MOD}.maybe_enable_site_tracing") as site_tracing,
        ):
            TpuSRTPlatform().on_warmup_complete()
        site_tracing.assert_called_once_with()
        return set_stance

    def test_spec_freezes_recompiles_by_default(self):
        self._publish_args(speculative_algorithm="EAGLE3")
        set_stance = self._warmup_complete(allow_spec_recompile=False)
        set_stance.assert_called_once_with("fail_on_recompile")

    def test_env_allows_spec_recompiles(self):
        self._publish_args(speculative_algorithm="EAGLE3")
        set_stance = self._warmup_complete(allow_spec_recompile=True)
        set_stance.assert_not_called()

    def test_hybrid_dp_attention_freezes_spec_recompiles(self):
        self._publish_args(speculative_algorithm="EAGLE3", enable_dp_attention=True)
        set_stance = self._warmup_complete(allow_spec_recompile=False)
        set_stance.assert_called_once_with("fail_on_recompile")

    def test_env_does_not_affect_non_spec(self):
        self._publish_args(speculative_algorithm=None)
        set_stance = self._warmup_complete(allow_spec_recompile=True)
        set_stance.assert_called_once_with("fail_on_recompile")

    def test_mode_b_inherits_the_override(self):
        self.assertIs(
            TpuOOTPlatform.on_warmup_complete, TpuSRTPlatform.on_warmup_complete
        )

    def test_base_platform_does_nothing(self):
        with (
            patch("torch.compiler.set_stance") as set_stance,
            patch(f"{_CHECKER_MOD}.maybe_enable_site_tracing") as site_tracing,
        ):
            SRTPlatform().on_warmup_complete()
        set_stance.assert_not_called()
        site_tracing.assert_not_called()


if __name__ == "__main__":
    unittest.main()
