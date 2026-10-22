"""Which runner shards an active context-parallel extend -- CPU-only.

The eager runner shards it at the model boundary unless the platform's compiled
forward does (TPU), or the model owns the layout under CP-TP group sharing.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner import eager_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagerExtendCpDispatch(CustomTestCase):
    def _shards(self, *, eager_cp, group_sharing, delegates):
        model = SimpleNamespace()
        if delegates:
            model.prepare_cp_inputs = lambda forward_batch, **kwargs: None
        with (
            mock.patch.object(
                eager_runner.current_platform,
                "runs_eager_extend_cp",
                return_value=eager_cp,
            ),
            mock.patch.object(
                eager_runner,
                "get_parallel",
                return_value=SimpleNamespace(enable_cp_tp_group_sharing=group_sharing),
            ),
        ):
            return eager_runner._runner_shards_cp_extend(model)

    def test_platform_compiled_cp_never_takes_the_eager_shard(self):
        for group_sharing in (False, True):
            for delegates in (False, True):
                with self.subTest(group_sharing=group_sharing, delegates=delegates):
                    self.assertFalse(
                        self._shards(
                            eager_cp=False,
                            group_sharing=group_sharing,
                            delegates=delegates,
                        )
                    )

    def test_eager_platform_shards_unless_the_model_owns_the_layout(self):
        self.assertTrue(
            self._shards(eager_cp=True, group_sharing=False, delegates=False)
        )
        self.assertFalse(
            self._shards(eager_cp=True, group_sharing=True, delegates=False)
        )
        self.assertTrue(self._shards(eager_cp=True, group_sharing=True, delegates=True))


if __name__ == "__main__":
    unittest.main()
