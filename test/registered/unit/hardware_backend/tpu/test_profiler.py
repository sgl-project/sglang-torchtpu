"""Unit tests for the TPU xprof profile-options patch."""

import dataclasses
import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.profiler import _install_tpu_profile_options_patch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Pure-logic tests. torch_tpu is faked via sys.modules, so this runs on the
# CPU CI runner (no TPU required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclasses.dataclass
class _BaseProfileOptions:
    host_tracer_level: int = 2


def _fake_torch_tpu(profile_options_cls):
    """Build sys.modules entries faking torch_tpu's private profiler layout."""
    impl = types.ModuleType("torch_tpu._internal.profiler._impl")
    impl.ProfileOptions = profile_options_cls
    profiler = types.ModuleType("torch_tpu._internal.profiler")
    profiler._impl = impl
    internal = types.ModuleType("torch_tpu._internal")
    internal.profiler = profiler
    torch_tpu = types.ModuleType("torch_tpu")
    torch_tpu._internal = internal
    return {
        "torch_tpu": torch_tpu,
        "torch_tpu._internal": internal,
        "torch_tpu._internal.profiler": profiler,
        "torch_tpu._internal.profiler._impl": impl,
    }


def _fake_profiler_api(get_options=lambda activities: _BaseProfileOptions()):
    module = types.ModuleType("fake_profiler_api")
    module._get_profile_options = get_options
    return module


class TestInstallTpuProfileOptionsPatch(CustomTestCase):
    def _install(
        self,
        profiler_api,
        *,
        num_chips,
        num_sparse_cores,
        profile_options_cls=_BaseProfileOptions
    ):
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(num_chips),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(num_sparse_cores),
            patch.dict(sys.modules, _fake_torch_tpu(profile_options_cls)),
        ):
            _install_tpu_profile_options_patch(profiler_api)
            return profiler_api._get_profile_options(["TPU"])

    def test_limits_enabled(self):
        profiler_api = _fake_profiler_api()
        options = self._install(profiler_api, num_chips=1, num_sparse_cores=2)

        self.assertTrue(profiler_api._sglang_tpu_profile_options_patch_installed)
        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_cores_to_trace, 2)
        # Tile cap only kicks in at a single SparseCore; here 2 leaves it off.
        self.assertIsNone(options.tpu_num_sparse_core_tiles_to_trace)
        # Base fields survive the merge.
        self.assertEqual(options.host_tracer_level, 2)

    def test_single_sparse_core_narrows_to_one_tile(self):
        profiler_api = _fake_profiler_api()
        options = self._install(profiler_api, num_chips=1, num_sparse_cores=1)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_cores_to_trace, 1)
        self.assertEqual(options.tpu_num_sparse_core_tiles_to_trace, 1)

    def test_one_limit_disabled(self):
        profiler_api = _fake_profiler_api()
        options = self._install(profiler_api, num_chips=4, num_sparse_cores=0)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 4)
        self.assertIsNone(options.tpu_num_sparse_cores_to_trace)
        self.assertIsNone(options.tpu_num_sparse_core_tiles_to_trace)

    def test_both_limits_disabled_leaves_hook_untouched(self):
        profiler_api = _fake_profiler_api()
        original = profiler_api._get_profile_options
        self._install(profiler_api, num_chips=0, num_sparse_cores=-1)

        self.assertIs(profiler_api._get_profile_options, original)
        self.assertTrue(profiler_api._sglang_tpu_profile_options_patch_installed)

    def test_repeated_installation_is_a_noop(self):
        profiler_api = _fake_profiler_api()
        self._install(profiler_api, num_chips=1, num_sparse_cores=1)
        patched = profiler_api._get_profile_options
        self._install(profiler_api, num_chips=1, num_sparse_cores=1)

        self.assertIs(profiler_api._get_profile_options, patched)

    def test_missing_private_hook_degrades(self):
        profiler_api = types.ModuleType("fake_profiler_api")
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(1),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(1),
            patch.dict(sys.modules, _fake_torch_tpu(_BaseProfileOptions)),
        ):
            _install_tpu_profile_options_patch(profiler_api)

        self.assertTrue(profiler_api._sglang_tpu_profile_options_patch_installed)
        self.assertFalse(hasattr(profiler_api, "_get_profile_options"))

    def test_non_dataclass_profile_options_degrades(self):
        class _NotADataclass:
            pass

        profiler_api = _fake_profiler_api()
        original = profiler_api._get_profile_options
        self._install(
            profiler_api,
            num_chips=1,
            num_sparse_cores=1,
            profile_options_cls=_NotADataclass,
        )

        self.assertIs(profiler_api._get_profile_options, original)
        self.assertTrue(profiler_api._sglang_tpu_profile_options_patch_installed)

    def test_getter_failure_returns_base_and_restores_hook(self):
        base = object()  # asdict on a non-dataclass raises
        original = lambda activities: base  # noqa: E731
        profiler_api = _fake_profiler_api(original)
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(1),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(1),
            patch.dict(sys.modules, _fake_torch_tpu(_BaseProfileOptions)),
        ):
            _install_tpu_profile_options_patch(profiler_api)
            result = profiler_api._get_profile_options(["TPU"])

        self.assertIs(result, base)
        self.assertIs(profiler_api._get_profile_options, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
