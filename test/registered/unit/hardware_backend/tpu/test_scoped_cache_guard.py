# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import unittest
import weakref
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation import scoped_cache_guard
from sglang_tpu.envs import envs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestScopedCacheGuard(unittest.TestCase):
    def test_guard_first_import_can_discover_the_tpu_platform(self):
        # A fresh interpreter reproduces padding -> guard -> SGLang discovery.
        # Stub only hardware discovery so this import check also runs on CPU CI.
        program = """
import importlib.metadata
from types import SimpleNamespace

original_entry_points = importlib.metadata.entry_points
def entry_points(*args, **kwargs):
    if kwargs.get("group") == "sglang.srt.platforms":
        return [SimpleNamespace(
            name="tpu",
            load=lambda: lambda: "sglang_tpu.tpu_platform.TpuSRTPlatform",
        )]
    return original_entry_points(*args, **kwargs)
importlib.metadata.entry_points = entry_points

from sglang_tpu.compilation.scoped_cache_guard import allow_tpu_recompilation
from sglang_tpu.tpu_platform import arm_cache_guard
arm_cache_guard()
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            env={
                **os.environ,
                "SGLANG_PLATFORM": "tpu",
                "SGLANG_TPU_ERROR_ON_RECOMPILES": "0",
                "JAX_PLATFORMS": "cpu",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def setUp(self):
        self.enterContext(envs.SGLANG_TPU_ERROR_ON_RECOMPILES.override(False))
        self.native = SimpleNamespace(
            _set_cache_only=MagicMock(),
            _get_cache_misses=MagicMock(return_value=17),
        )
        self.sync = SimpleNamespace(synchronize=MagicMock())
        self.enterContext(
            patch.object(scoped_cache_guard.torch, "tpu", self.native, create=True)
        )
        self.enterContext(patch.object(scoped_cache_guard, "sync", self.sync))
        self.enterContext(
            patch.object(scoped_cache_guard, "_state", scoped_cache_guard._GuardState())
        )

    def test_arm_flushes_before_enforcement_and_is_opt_in(self):
        scoped_cache_guard.arm_cache_guard()
        self.native._set_cache_only.assert_not_called()
        calls = []
        self.sync.synchronize.side_effect = lambda **kw: calls.append("flush")
        self.native._set_cache_only.side_effect = lambda enabled: calls.append(enabled)
        with envs.SGLANG_TPU_ERROR_ON_RECOMPILES.override(True):
            scoped_cache_guard.arm_cache_guard()
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                scoped_cache_guard.arm_cache_guard()
        self.assertEqual(calls, ["flush", True])
        self.assertEqual(scoped_cache_guard._state.owner, get_ident())

    def test_activation_failure_propagates(self):
        for operation in (self.sync.synchronize, self.native._set_cache_only):
            with (
                self.subTest(operation=operation),
                envs.SGLANG_TPU_ERROR_ON_RECOMPILES.override(True),
                patch.object(
                    operation, "side_effect", RuntimeError("native activation failed")
                ),
                self.assertRaisesRegex(RuntimeError, "native activation failed"),
            ):
                scoped_cache_guard.arm_cache_guard()
            self.assertIsNone(scoped_cache_guard._state.owner)

    def test_allowance_failures_poison_guard_and_preserve_body_error(self):
        for failure in ("entry", "body", "exit", "body_and_exit"):
            with (
                self.subTest(failure=failure),
                patch.object(scoped_cache_guard._state, "owner", get_ident()),
                patch.object(scoped_cache_guard._state, "failed", False),
            ):
                body_error = ValueError("body failed")

                def toggle(enabled):
                    if (failure == "entry" and not enabled) or (
                        failure in ("exit", "body_and_exit") and enabled
                    ):
                        raise RuntimeError("toggle failed")

                self.native._set_cache_only.side_effect = toggle
                expected = ValueError if "body" in failure else RuntimeError
                with self.assertRaises(expected) as captured:
                    with scoped_cache_guard._allow_tpu_recompilation("metadata"):
                        if "body" in failure:
                            raise body_error
                if "body" in failure:
                    self.assertIs(captured.exception, body_error)
                self.assertTrue(scoped_cache_guard._state.failed)
                self.assertIsNone(scoped_cache_guard._state.active_section)
                with self.assertRaisesRegex(RuntimeError, "restart the worker"):
                    with scoped_cache_guard._allow_tpu_recompilation("next"):
                        self.fail("A poisoned worker must not continue")

    def test_decorators_nest_once_and_keep_result_live(self):
        class Result:
            pass

        refs = []

        @scoped_cache_guard.allow_tpu_recompilation
        def inner():
            result = Result()
            refs.append(weakref.ref(result))
            return result

        @scoped_cache_guard.allow_tpu_recompilation
        def outer():
            return inner()

        def flush(*, wait):
            self.assertFalse(wait)
            if refs:
                self.assertIsNotNone(refs[0]())

        self.sync.synchronize.side_effect = flush
        self.native._get_cache_misses.side_effect = [17, 20]
        with (
            patch.object(scoped_cache_guard._state, "owner", get_ident()),
            self.assertLogs(scoped_cache_guard.logger, level="WARNING") as captured,
        ):
            result = outer()
        self.assertIs(result, refs[0]())
        self.assertEqual(outer.__name__, "outer")
        self.assertEqual(self.sync.synchronize.call_count, 2)
        self.assertEqual(
            self.native._set_cache_only.call_args_list, [call(False), call(True)]
        )
        self.assertEqual(len(captured.output), 1)
        self.assertIn(f"{outer.__module__}.{outer.__qualname__}", captured.output[0])
        self.assertIn("process_cache_misses=3", captured.output[0])

    def test_nested_failure_cannot_be_swallowed(self):
        @scoped_cache_guard.allow_tpu_recompilation
        def inner():
            raise ValueError("nested failure")

        @scoped_cache_guard.allow_tpu_recompilation
        def outer():
            try:
                inner()
            except ValueError:
                pass

        with (
            patch.object(scoped_cache_guard._state, "owner", get_ident()),
            self.assertRaisesRegex(RuntimeError, "restart the worker"),
        ):
            outer()
        self.assertTrue(scoped_cache_guard._state.failed)
        self.assertIsNone(scoped_cache_guard._state.active_section)
        self.native._set_cache_only.assert_has_calls([call(False), call(True)])

    def test_allowance_rejects_wrong_thread(self):
        with patch.object(scoped_cache_guard._state, "owner", get_ident() + 1):
            with self.assertRaisesRegex(RuntimeError, "owner thread"):
                with scoped_cache_guard._allow_tpu_recompilation("metadata"):
                    self.fail("Wrong thread entered allowance")
        self.native._set_cache_only.assert_not_called()

    def test_warmup_nesting_uses_the_same_boundaries(self):
        @scoped_cache_guard.allow_tpu_recompilation
        def inner():
            return 7

        @scoped_cache_guard.allow_tpu_recompilation
        def outer():
            return inner()

        with envs.SGLANG_TPU_ERROR_ON_RECOMPILES.override(False):
            self.assertEqual(outer(), 7)
        self.sync.synchronize.assert_not_called()
        with envs.SGLANG_TPU_ERROR_ON_RECOMPILES.override(True):
            self.assertEqual(outer(), 7)
        self.assertEqual(self.sync.synchronize.call_count, 2)
        self.native._set_cache_only.assert_not_called()

    def test_allowance_flush_failures_remain_fatal(self):
        for entry in (True, False):
            with (
                self.subTest(entry=entry),
                patch.object(scoped_cache_guard._state, "owner", get_ident()),
                patch.object(scoped_cache_guard._state, "failed", False),
            ):
                self.native._set_cache_only.reset_mock()
                self.sync.synchronize.side_effect = (
                    [RuntimeError("cold graph")]
                    if entry
                    else [None, RuntimeError("cold graph")]
                )
                with self.assertRaisesRegex(RuntimeError, "cold graph"):
                    with scoped_cache_guard._allow_tpu_recompilation("metadata"):
                        pass
                self.assertEqual(
                    self.native._set_cache_only.call_args_list,
                    [] if entry else [call(False), call(True)],
                )
                self.assertTrue(scoped_cache_guard._state.failed)
                self.assertIsNone(scoped_cache_guard._state.active_section)


if __name__ == "__main__":
    unittest.main()
