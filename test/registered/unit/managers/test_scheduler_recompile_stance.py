import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SCHED_MOD = "sglang.srt.managers.scheduler"


class TestSchedulerRecompileStance(unittest.TestCase):
    def setUp(self):
        self.enterContext(envs.SGLANG_TPU_XLA_CACHE_ONLY.override(False))

    def _make_scheduler(
        self,
        *,
        speculative_algorithm,
        enable_dp_attention=False,
        enable_torch_compile=True,
        device="tpu",
    ):
        # _maybe_arm_fail_on_recompile reads the exec and spec bags; publish both.
        override = get_context().override_server_args(
            enable_torch_compile=enable_torch_compile,
            speculative_algorithm=speculative_algorithm,
            enable_dp_attention=enable_dp_attention,
        )
        server_args = override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(device=device)
        )
        # Built with __new__: wire server_args to the published instance.
        scheduler.server_args = server_args
        return scheduler

    def test_spec_freezes_recompiles_by_default(self):
        scheduler = self._make_scheduler(speculative_algorithm="EAGLE3")

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(False),
            patch(f"{_SCHED_MOD}.torch.compiler.set_stance") as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")

    def test_env_allows_spec_recompiles(self):
        scheduler = self._make_scheduler(speculative_algorithm="EAGLE3")

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(True),
            patch(f"{_SCHED_MOD}.torch.compiler.set_stance") as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_not_called()

    def test_hybrid_dp_attention_freezes_spec_recompiles(self):
        scheduler = self._make_scheduler(
            speculative_algorithm="EAGLE3", enable_dp_attention=True
        )

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(False),
            patch(f"{_SCHED_MOD}.torch.compiler.set_stance") as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")

    def test_env_does_not_affect_non_spec(self):
        scheduler = self._make_scheduler(speculative_algorithm=None)

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(True),
            patch(f"{_SCHED_MOD}.torch.compiler.set_stance") as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")

    def test_xla_cache_only_is_independent_of_graph_compilation(self):
        for enable_torch_compile in (False, True):
            for allow_spec_recompile in (False, True):
                with self.subTest(
                    enable_torch_compile=enable_torch_compile,
                    allow_spec_recompile=allow_spec_recompile,
                ):
                    scheduler = self._make_scheduler(
                        speculative_algorithm="EAGLE3",
                        enable_torch_compile=enable_torch_compile,
                    )
                    device = Mock()
                    with (
                        envs.SGLANG_TPU_XLA_CACHE_ONLY.override(True),
                        envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(
                            allow_spec_recompile
                        ),
                        patch(f"{_SCHED_MOD}.torch.tpu", device, create=True),
                        patch(
                            f"{_SCHED_MOD}.torch.accelerator.synchronize",
                            device.synchronize,
                        ),
                        patch(f"{_SCHED_MOD}.torch.compiler.set_stance") as stance,
                    ):
                        scheduler._maybe_arm_fail_on_recompile()
                    self.assertEqual(
                        device.mock_calls,
                        [call.synchronize(), call._set_cache_only(True)],
                    )
                    if enable_torch_compile and not allow_spec_recompile:
                        stance.assert_called_once_with("fail_on_recompile")
                    else:
                        stance.assert_not_called()

    def test_xla_cache_only_is_opt_in_and_tpu_only(self):
        for enabled, backend in ((False, "tpu"), (True, "cpu"), (True, "cuda")):
            with self.subTest(enabled=enabled, backend=backend):
                scheduler = self._make_scheduler(
                    speculative_algorithm=None, device=backend
                )
                device = Mock()
                with (
                    envs.SGLANG_TPU_XLA_CACHE_ONLY.override(enabled),
                    patch(f"{_SCHED_MOD}.torch.tpu", device, create=True),
                    patch(
                        f"{_SCHED_MOD}.torch.accelerator.synchronize",
                        device.synchronize,
                    ),
                    patch(f"{_SCHED_MOD}.torch.compiler.set_stance"),
                ):
                    scheduler._maybe_arm_fail_on_recompile()
                self.assertEqual(device.mock_calls, [])

    def test_xla_cache_only_requires_runtime_support(self):
        scheduler = self._make_scheduler(speculative_algorithm=None)
        with (
            envs.SGLANG_TPU_XLA_CACHE_ONLY.override(True),
            patch(f"{_SCHED_MOD}.torch.tpu", SimpleNamespace(), create=True),
            patch(f"{_SCHED_MOD}.torch.accelerator.synchronize"),
            self.assertRaisesRegex(AttributeError, "_set_cache_only"),
        ):
            scheduler._maybe_arm_fail_on_recompile()


if __name__ == "__main__":
    unittest.main()
