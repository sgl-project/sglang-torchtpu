"""Mock-based unit tests for AOTAutogradCache installation and rollback.

Mocks modules in sys.modules to prevent real libtpu initialization and PJRT side
effects. States modified are restored during tearDown to keep tests isolated.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock

_AOT_CACHE_MODULE = "sglang.srt.hardware_backend.tpu.compilation.aot_autograd_cache"


class _FakeTpuBackend:
    """Signature-bound stand-in for ``torch_tpu._internal.compile._backend.TpuBackend``.

    Mirrors the real ``__init__`` so any kwarg drift (rename, removal, new
    required kwarg) in a future torch_tpu release surfaces as a ``TypeError``
    in this test instead of being silently swallowed by a permissive
    ``MagicMock``.
    """

    def __init__(
        self,
        debug: bool = False,
        dynamism: bool = False,
        enable_serialization: bool = False,
        precompile_steps: int = 0,
    ):
        self._debug = debug
        self._dynamism = dynamism
        self._enable_serialization = enable_serialization
        self._precompile_steps = precompile_steps


def _build_fake_torch_tpu_modules():
    """Build the minimum torch_tpu hierarchy the installation imports.

    Includes a fake ``torch_tpu._loader.load`` so torch's
    ``_import_device_backends`` autoload (triggered the first time ``torch``
    is imported in this process) succeeds against our stub instead of trying
    to find the real torch_tpu and pulling in libtpu.
    """
    fake_backend = types.ModuleType("torch_tpu._internal.compile._backend")
    fake_backend.TpuBackend = _FakeTpuBackend

    fake_compile = types.ModuleType("torch_tpu._internal.compile")
    fake_compile._backend = fake_backend

    fake_internal = types.ModuleType("torch_tpu._internal")
    fake_internal.compile = fake_compile

    # Entry-point ``torch_tpu._loader:load`` is what torch's autoload calls.
    # A callable that does nothing is sufficient.
    fake_loader = types.ModuleType("torch_tpu._loader")
    fake_loader.load = MagicMock(name="autoload", return_value=None)

    fake_torch_tpu = types.ModuleType("torch_tpu")
    fake_torch_tpu._internal = fake_internal
    fake_torch_tpu._loader = fake_loader

    modules = {
        "torch_tpu": fake_torch_tpu,
        "torch_tpu._internal": fake_internal,
        "torch_tpu._internal.compile": fake_compile,
        "torch_tpu._internal.compile._backend": fake_backend,
        "torch_tpu._loader": fake_loader,
    }
    return modules


class TestAOTAutogradCache(unittest.TestCase):

    def setUp(self):
        # Use ``addCleanup`` (LIFO) instead of ``tearDown`` so cleanups run
        # even if ``setUp`` raises mid-way — otherwise a partial mutation
        # (e.g. fakes installed in sys.modules) would leak into later tests
        # in the same process. Pattern: snapshot → addCleanup → mutate.

        self._fake_modules = _build_fake_torch_tpu_modules()
        saved_sys_modules = {name: sys.modules.get(name) for name in self._fake_modules}
        self.addCleanup(self._restore_sys_modules, saved_sys_modules)
        sys.modules.update(self._fake_modules)

        # First torch import in this process triggers torch's autoload via
        # ``_import_device_backends``, which looks up the
        # ``torch_tpu._loader:load`` entry_point. Because we already installed
        # a fake ``torch_tpu._loader`` above, autoload finds our stub and
        # libtpu is never initialized.
        import torch._dynamo as _dynamo
        import torch._dynamo.backends.registry as _registry
        import torch._functorch.config as _ft_config

        self._dynamo = _dynamo
        self._registry = _registry
        self._ft_config = _ft_config

        saved_compiler_fns = dict(_registry._COMPILER_FNS)
        self.addCleanup(self._restore_compiler_fns, saved_compiler_fns)
        _registry._COMPILER_FNS.clear()

        saved_enable = _ft_config.enable_autograd_cache
        saved_bundled = _ft_config.bundled_autograd_cache
        self.addCleanup(self._restore_ft_flags, saved_enable, saved_bundled)
        _ft_config.enable_autograd_cache = False
        _ft_config.bundled_autograd_cache = False

        saved_register_backend = _dynamo.register_backend
        self.addCleanup(self._restore_register_backend, saved_register_backend)
        self._mock_register_backend = MagicMock(name="register_backend")
        _dynamo.register_backend = self._mock_register_backend

        # Re-import the module under test fresh each run so its module-level
        # ``_installed`` flag starts at False.
        self.addCleanup(sys.modules.pop, _AOT_CACHE_MODULE, None)
        sys.modules.pop(_AOT_CACHE_MODULE, None)
        self.aot_cache = importlib.import_module(_AOT_CACHE_MODULE)

    # --- cleanup helpers (named for clarity in stack traces) ------------

    @staticmethod
    def _restore_sys_modules(saved):
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def _restore_compiler_fns(self, saved):
        self._registry._COMPILER_FNS.clear()
        self._registry._COMPILER_FNS.update(saved)

    def _restore_ft_flags(self, enable, bundled):
        self._ft_config.enable_autograd_cache = enable
        self._ft_config.bundled_autograd_cache = bundled

    def _restore_register_backend(self, original):
        self._dynamo.register_backend = original

    def test_install_success_forwards_prior_backend_state(self):
        # ``install()`` walks TpuBackend.__init__ and forwards any kwarg
        # whose ``_<name>`` or ``<name>`` attribute exists on the prior
        # backend; here we set _debug, _dynamism, and _precompile_steps
        # on the prior to verify all three are forwarded.
        # ``SimpleNamespace`` (not ``MagicMock``) so any unset attribute
        # raises AttributeError instead of being silently fabricated — a
        # future kwarg on TpuBackend would surface as a forwarding miss
        # rather than a silently-forwarded ``Mock()``.
        old_backend = types.SimpleNamespace(
            _debug=True, _dynamism=False, _precompile_steps=7
        )
        self._registry._COMPILER_FNS["tpu"] = old_backend

        self.aot_cache.install()

        self._mock_register_backend.assert_called_once()
        call_kwargs = self._mock_register_backend.call_args.kwargs
        self.assertEqual(call_kwargs["name"], "tpu")
        registered = call_kwargs["compiler_fn"]
        # Signature-bound fake: catches kwarg drift on a future torch_tpu bump.
        self.assertIsInstance(registered, _FakeTpuBackend)
        self.assertTrue(registered._enable_serialization)
        self.assertTrue(registered._debug)
        self.assertFalse(registered._dynamism)
        self.assertEqual(registered._precompile_steps, 7)
        self.assertTrue(self._ft_config.enable_autograd_cache)
        self.assertTrue(self._ft_config.bundled_autograd_cache)
        self.assertTrue(self.aot_cache._installed)

    def test_forward_backend_kwargs_falls_back_to_bare_attr_name(self):
        # Backends that don't follow the underscore convention should still
        # work, as long as the attribute name matches the param name.
        class _BareAttrBackend:
            def __init__(self, debug: bool = False, dynamism: bool = False):
                self.debug = debug
                self.dynamism = dynamism

        old = _BareAttrBackend(debug=True, dynamism=False)
        kwargs = self.aot_cache._forward_backend_kwargs(
            _BareAttrBackend, old, overrides={}
        )
        self.assertEqual(kwargs, {"debug": True, "dynamism": False})

    def test_forward_backend_kwargs_rejects_override_not_in_signature(self):
        # If a future torch_tpu drops ``enable_serialization``, silently
        # dropping the override would leave the bundled cache disabled
        # without warning. We must fail loudly instead.
        class _NoSerialization:
            def __init__(self, debug: bool = False):
                self._debug = debug

        with self.assertRaises(TypeError) as cm:
            self.aot_cache._forward_backend_kwargs(
                _NoSerialization,
                old_backend=None,
                overrides={"enable_serialization": True},
            )
        self.assertIn("enable_serialization", str(cm.exception))

    def test_forward_backend_kwargs_rejects_uninferrable_kwarg(self):
        # If neither overrides nor the prior backend can supply a value for
        # an accepted kwarg, we must fail before constructing — even if the
        # param has a default. We don't assume old_backend was at the default.
        class _DefaultedKwarg:
            def __init__(self, some_knob: int = 0):
                pass  # no instance attr exposed for ``some_knob``

        with self.assertRaises(TypeError) as cm:
            self.aot_cache._forward_backend_kwargs(
                _DefaultedKwarg, old_backend=_DefaultedKwarg(), overrides={}
            )
        self.assertIn("some_knob", str(cm.exception))

    def test_forward_backend_kwargs_rejects_positional_only_param(self):
        # The helper only returns kwargs, so a positional-only param can't
        # be forwarded. Fail loudly instead of silently dropping it.
        exec_ns: dict = {}
        # Use exec() to construct a class with a positional-only kwarg
        # (the ``/`` syntax requires this in a string for Python 3.12).
        exec(
            "class _PosOnly:\n"
            "    def __init__(self, debug: bool = False, /):\n"
            "        self._debug = debug\n",
            exec_ns,
        )
        pos_only_cls = exec_ns["_PosOnly"]

        with self.assertRaises(TypeError) as cm:
            self.aot_cache._forward_backend_kwargs(
                pos_only_cls, old_backend=None, overrides={}
            )
        self.assertIn("positional-only", str(cm.exception))
        self.assertIn("debug", str(cm.exception))

    def test_install_failure_rolls_back_prior_backend_and_flags(self):
        old_backend = types.SimpleNamespace(
            _debug=False, _dynamism=True, _precompile_steps=0
        )
        self._registry._COMPILER_FNS["tpu"] = old_backend
        self._mock_register_backend.side_effect = RuntimeError(
            "Mock PyTorch Registry Crash"
        )

        with self.assertLogs(self.aot_cache.logger, level="ERROR") as caplog:
            self.aot_cache.install()
        self.assertTrue(
            any("compile-cache MISS" in m for m in caplog.output),
            f"Expected 'compile-cache MISS' in logs, got: {caplog.output}",
        )

        self.assertIs(self._registry._COMPILER_FNS["tpu"], old_backend)
        self.assertFalse(self._ft_config.enable_autograd_cache)
        self.assertFalse(self._ft_config.bundled_autograd_cache)
        self.assertFalse(self.aot_cache._installed)

    def test_install_failure_with_no_prior_backend_leaves_registry_empty(self):
        self.assertNotIn("tpu", self._registry._COMPILER_FNS)
        self._mock_register_backend.side_effect = RuntimeError("boom")

        self.aot_cache.install()

        self.assertNotIn("tpu", self._registry._COMPILER_FNS)
        self.assertFalse(self.aot_cache._installed)

    def test_install_is_idempotent(self):
        self._registry._COMPILER_FNS["tpu"] = types.SimpleNamespace(
            _debug=False, _dynamism=False, _precompile_steps=0
        )

        self.aot_cache.install()
        self.assertTrue(self.aot_cache._installed)
        self.assertEqual(self._mock_register_backend.call_count, 1)

        self._mock_register_backend.reset_mock()
        self.aot_cache.install()
        self._mock_register_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
