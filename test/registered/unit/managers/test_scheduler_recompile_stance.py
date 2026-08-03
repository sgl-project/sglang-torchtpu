import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SCHED_MOD = "sglang.srt.managers.scheduler"


def _make_scheduler(*, speculative_algorithm):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            device="tpu",
            server_args=SimpleNamespace(
                enable_torch_compile=True,
                speculative_algorithm=speculative_algorithm,
            ),
        )
    )
    return scheduler


class TestSchedulerRecompileStance(unittest.TestCase):
    def test_spec_freezes_recompiles_by_default(self):
        scheduler = _make_scheduler(speculative_algorithm="EAGLE3")

        with envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(False), patch(
            f"{_SCHED_MOD}.torch.compiler.set_stance"
        ) as set_stance:
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")

    def test_env_allows_spec_recompiles(self):
        scheduler = _make_scheduler(speculative_algorithm="EAGLE3")

        with envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(True), patch(
            f"{_SCHED_MOD}.torch.compiler.set_stance"
        ) as set_stance:
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_not_called()

    def test_env_does_not_affect_non_spec(self):
        scheduler = _make_scheduler(speculative_algorithm=None)

        with envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(True), patch(
            f"{_SCHED_MOD}.torch.compiler.set_stance"
        ) as set_stance:
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")


if __name__ == "__main__":
    unittest.main()
