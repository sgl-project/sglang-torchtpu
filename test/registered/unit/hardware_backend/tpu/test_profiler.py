"""Unit tests for the TPU torch.profiler wiring."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import (
    profiler_manager as scheduler_profiler,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu import profiler as tpu_profiler
from sglang_tpu.profiler import start_tpu_profile, stop_tpu_profile

# Pure-logic tests. torch_tpu is faked via sys.modules and ProfilerActivity.TPU
# is aliased to PrivateUse1 the way torch_tpu does it, so this runs on the CPU
# CI runner (no TPU required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ProfilerActivity = torch.profiler.ProfilerActivity


def _fake_torch_tpu(calls):
    """Fake torch_tpu's legacy _impl.start_trace/stop_trace, recording calls."""
    impl = types.ModuleType("torch_tpu._internal.profiler._impl")
    impl.start_trace = lambda *args, **kwargs: calls.append("start_trace")
    impl.stop_trace = lambda *args, **kwargs: calls.append("stop_trace")
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


class TestStartUsesNativeTorchProfiler(CustomTestCase):
    """start_tpu_profile builds one torch.profiler with the TPU activity."""

    def _start(self, *, with_cpu, python_tracer=False):
        calls = []
        output_dir = Path(enter_scope(self, tempfile.TemporaryDirectory())) / "r"
        with (
            envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(python_tracer),
            patch.dict(sys.modules, _fake_torch_tpu(calls)),
            patch.object(
                _ProfilerActivity, "TPU", _ProfilerActivity.PrivateUse1, create=True
            ),
            patch("torch.profiler.profile") as profile,
            patch("torch.profiler.tensorboard_trace_handler") as handler,
        ):
            trace = start_tpu_profile(output_dir=output_dir, with_cpu=with_cpu)
        return trace, profile, handler, calls, output_dir

    def test_cpu_and_tpu(self):
        trace, profile, handler, calls, output_dir = self._start(with_cpu=True)

        profile.assert_called_once()
        kwargs = profile.call_args.kwargs
        self.assertEqual(
            kwargs["activities"],
            [_ProfilerActivity.CPU, _ProfilerActivity.PrivateUse1],
        )
        self.assertFalse(kwargs["with_stack"])
        handler.assert_called_once_with(str(output_dir), use_gzip=True)
        self.assertIs(kwargs["on_trace_ready"], handler.return_value)
        profile.return_value.start.assert_called_once_with()
        self.assertIs(trace.profiler, profile.return_value)
        self.assertEqual(trace.output_dir, output_dir)
        self.assertTrue(output_dir.is_dir())
        self.assertEqual(calls, [])  # the legacy _impl.start_trace is not used

    def test_tpu_only(self):
        _, profile, handler, calls, output_dir = self._start(with_cpu=False)

        kwargs = profile.call_args.kwargs
        self.assertEqual(kwargs["activities"], [_ProfilerActivity.PrivateUse1])
        self.assertFalse(kwargs["with_stack"])
        handler.assert_called_once_with(str(output_dir), use_gzip=True)
        self.assertEqual(calls, [])

    def test_python_tracer_on(self):
        _, profile, _, calls, _ = self._start(with_cpu=True, python_tracer=True)

        # with_stack=True is what turns on the plugin's python tracer.
        self.assertTrue(profile.call_args.kwargs["with_stack"])
        self.assertEqual(calls, [])

    def test_missing_tpu_activity_raises(self):
        activities = types.SimpleNamespace(CPU=_ProfilerActivity.CPU)
        with (
            patch("torch.profiler.ProfilerActivity", activities),
            patch("torch.profiler.profile") as profile,
            self.assertRaisesRegex(RuntimeError, "ProfilerActivity.TPU"),
        ):
            start_tpu_profile(output_dir=Path("/nonexistent/unused"))
        profile.assert_not_called()


class TestStop(CustomTestCase):
    def test_stop_stops_the_profiler_not_stop_trace(self):
        calls = []
        profiler = MagicMock()
        trace = tpu_profiler._TpuTrace(output_dir=Path("/unused"), profiler=profiler)
        with patch.dict(sys.modules, _fake_torch_tpu(calls)):
            stop_tpu_profile(trace)

        profiler.stop.assert_called_once_with()
        self.assertEqual(calls, [])  # the legacy _impl.stop_trace is not used


class TestRealTorchProfilerOnCpu(CustomTestCase):
    """End to end through the real torch.profiler on a CPU-only torch.

    Without torch_tpu, PrivateUse1 is dropped, so this checks the API wiring and
    the Kineto trace location, not TPU events.
    """

    def test_cpu_trace_lands_in_output_dir(self):
        tmpdir = Path(enter_scope(self, tempfile.TemporaryDirectory()))
        output_dir = tmpdir / "profile-id-TP-0"
        with patch.object(
            _ProfilerActivity, "TPU", _ProfilerActivity.PrivateUse1, create=True
        ):
            trace = start_tpu_profile(output_dir=output_dir, with_cpu=True)
            torch.randn(8, 8) @ torch.randn(8, 8)
            stop_tpu_profile(trace)

        self.assertEqual(len(list(output_dir.glob("*.pt.trace.json.gz"))), 1)


def _stopping_manager(tmp_path, trace):
    with envs.SGLANG_PROFILE_V2.override(False):
        manager = scheduler_profiler.SchedulerProfilerManager(
            ps=types.SimpleNamespace(),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
        )
    manager.profile_in_progress = True
    manager.tpu_profiler = trace
    manager.torch_profiler_output_dir = tmp_path
    manager.profiler_activities = ["TPU"]
    manager.profile_prefix = ""
    return manager


def test_stage_tpu_profile_synchronizes_before_stopping(tmp_path):
    calls = []
    trace = object()
    manager = _stopping_manager(tmp_path, trace)

    with (
        patch.object(
            scheduler_profiler.current_platform,
            "synchronize",
            side_effect=lambda: calls.append("synchronize"),
        ),
        patch(
            "sglang_tpu.profiler.stop_tpu_profile",
            side_effect=lambda actual: calls.append(("stop", actual)),
        ),
    ):
        manager._stop_profile(stage=ForwardMode.EXTEND)

    assert calls == ["synchronize", ("stop", trace)]


def test_non_stage_tpu_profile_stops_without_synchronizing(tmp_path):
    calls = []
    trace = object()
    manager = _stopping_manager(tmp_path, trace)

    with (
        patch.object(
            scheduler_profiler.current_platform,
            "synchronize",
            side_effect=lambda: calls.append("synchronize"),
        ),
        patch(
            "sglang_tpu.profiler.stop_tpu_profile",
            side_effect=lambda actual: calls.append(("stop", actual)),
        ),
    ):
        manager._stop_profile()

    assert calls == [("stop", trace)]


def test_stage_profile_without_tpu_trace_does_not_synchronize(tmp_path):
    manager = _stopping_manager(tmp_path, None)
    manager.profiler_activities = ["CPU"]

    with patch.object(scheduler_profiler.current_platform, "synchronize") as sync:
        manager._stop_profile(stage=ForwardMode.EXTEND)

    sync.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
