"""Guard the positional calling convention of DeepSeek-V4 TPU hooks."""

import inspect
import pkgutil
import unittest

from sglang.srt.plugins.hook_registry import HookType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import dsv4
from sglang_tpu.models.deepseek_v41 import hooks as dsv41

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsv4HookSignatures(CustomTestCase):
    def test_around_hooks_accept_core_positional_arguments(self):
        positional_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for target_name, hook, hook_type in (*dsv4.hook_specs(), *dsv41.hook_specs()):
            if hook_type is not HookType.AROUND:
                continue
            with self.subTest(target=target_name, hook=hook.__name__):
                target = inspect.signature(pkgutil.resolve_name(target_name))
                wrapper = inspect.signature(hook)
                positional = [
                    p for p in target.parameters.values() if p.kind in positional_kinds
                ]
                if any(
                    p.kind == inspect.Parameter.VAR_POSITIONAL
                    for p in target.parameters.values()
                ):
                    self.assertTrue(
                        any(
                            p.kind == inspect.Parameter.VAR_POSITIONAL
                            for p in wrapper.parameters.values()
                        ),
                        "The target can receive extra positional arguments",
                    )
                # AROUND wrappers receive the original callable first. Partial
                # binding isolates the positional interface from keyword-only
                # arguments and parameter names used by individual hooks.
                wrapper.bind_partial(None, *([None] * len(positional)))


if __name__ == "__main__":
    unittest.main()
