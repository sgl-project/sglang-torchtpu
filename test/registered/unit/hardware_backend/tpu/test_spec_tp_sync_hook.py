"""CPU contract for the SpecTpSync hook: the sync runs with eager deferral off."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks.spec_tp_sync import broadcast_undeferred


def _fake_execution_mode():
    """A stand-in torch_tpu execution_mode that records the mode active in the sync."""
    module = types.SimpleNamespace(
        EagerMode=types.SimpleNamespace(DEFER_NEVER="DEFER_NEVER")
    )
    module.current = "DEFER_AND_FUSE"

    @contextlib.contextmanager
    def set_eager_mode(mode):
        previous, module.current = module.current, mode
        try:
            yield
        finally:
            module.current = previous

    module.set_eager_mode = set_eager_mode
    internal = types.ModuleType("torch_tpu._internal")
    internal.execution_mode = module
    return internal, module


class TestSpecTpSyncHook(CustomTestCase):
    def test_sync_runs_under_defer_never_and_restores_the_mode(self):
        modes = []
        internal, module = _fake_execution_mode()

        def original(sync, site, values):
            modes.append(module.current)
            return values + 1

        with patch.dict(
            sys.modules,
            {
                "torch_tpu": types.ModuleType("torch_tpu"),
                "torch_tpu._internal": internal,
            },
        ):
            out = broadcast_undeferred(original, "sync", "site", torch.zeros(2))
        self.assertEqual(modes, ["DEFER_NEVER"])
        self.assertEqual(module.current, "DEFER_AND_FUSE")
        torch.testing.assert_close(out, torch.ones(2))


if __name__ == "__main__":
    unittest.main()
