"""Kimi-K3 vision hooks (sglang_tpu/hooks/kimi_k3_vision.py): their targets
resolve and the AROUND hooks mirror their targets' signatures, without a TPU."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import inspect
import pkgutil
import unittest

from sglang.srt.plugins.hook_registry import HookType
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import kimi_k3_vision as hooks

_VISION_HOOKS = [h for h in HOOKS if h[1].__module__ == hooks.__name__]


def _parameters(fn, skip: int) -> list[tuple]:
    params = list(inspect.signature(fn).parameters.values())[skip:]
    return [(p.name, p.kind, p.default) for p in params]


class TestKimiK3VisionHooks(CustomTestCase):
    def test_hooks_target_existing_kimi_k3_methods(self):
        # The runner's features skip the projector only when both AROUND hooks
        # apply, and apply_hooks logs and moves on when a target fails.
        self.assertEqual(len(_VISION_HOOKS), 4)
        for target, _, _ in _VISION_HOOKS:
            with self.subTest(target=target):
                self.assertTrue(callable(pkgutil.resolve_name(target)))

    def test_around_hooks_mirror_target_signatures(self):
        """After (original_fn, self) the AROUND hooks take their targets'
        parameters, so a core signature change fails here instead of
        misrouting arguments."""
        arounds = {
            hook: target
            for target, hook, hook_type in _VISION_HOOKS
            if hook_type is HookType.AROUND
        }
        self.assertEqual(
            set(arounds),
            {hooks.encode_with_k3_vision_runner, hooks.pass_through_projected_features},
        )
        for hook, target in arounds.items():
            with self.subTest(target=target):
                self.assertEqual(
                    _parameters(hook, skip=2),
                    _parameters(pkgutil.resolve_name(target), skip=1),
                )


if __name__ == "__main__":
    unittest.main()
