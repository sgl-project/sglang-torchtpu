"""Unit tests for the TPU xprof profiler wiring."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import (
    profiler_manager as scheduler_profiler,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.profiler import start_tpu_profile, stop_tpu_profile

# Pure-logic tests. torch_tpu and torch.profiler.profile are faked, so this
# runs on the CPU CI runner (no TPU required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTpuProfilerConfig:
    """Records the kwargs torch_tpu's TpuProfilerConfig would encode."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeProfile:
    def __init__(self, *, activities, experimental_config):
        self.activities = activities
        self.config = experimental_config
        self.events = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, *exc):
        self.events.append("exit")


def _fake_torch_tpu():
    profiler = types.ModuleType("torch_tpu._internal.profiler")
    profiler.TpuProfilerConfig = _FakeTpuProfilerConfig
    internal = types.ModuleType("torch_tpu._internal")
    internal.profiler = profiler
    torch_tpu = types.ModuleType("torch_tpu")
    torch_tpu._internal = internal
    return {
        "torch_tpu": torch_tpu,
        "torch_tpu._internal": internal,
        "torch_tpu._internal.profiler": profiler,
    }


class TestStartTpuProfile(CustomTestCase):
    """The capture must carry its options through torch.profiler to libtpu.

    torch_tpu's _impl.start_trace forwards only the three tracer levels, so
    the chip/SparseCore scope and the trace mode it was given never reached
    libtpu: every rank traced at backend defaults. TpuProfilerConfig's
    experimental_options is the path that reaches advanced_configuration.
    """

    def _start(
        self,
        *,
        num_chips=1,
        num_sparse_cores=1,
        trace_mode="TRACE_COMPUTE",
        with_cpu=True,
        python_tracer=False,
    ):
        with (
            tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(num_chips),
            tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(num_sparse_cores),
            tpu_envs.SGLANG_TPU_PROFILE_TRACE_MODE.override(trace_mode),
            tpu_envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(python_tracer),
            patch.dict(sys.modules, _fake_torch_tpu()),
            patch.object(torch.profiler, "profile", _FakeProfile),
            tempfile.TemporaryDirectory() as tmp,
        ):
            trace = start_tpu_profile(output_dir=Path(tmp), with_cpu=with_cpu)
            self.assertEqual(trace.profile.config.kwargs["run_dir"], Path(tmp))
        return trace

    def test_scope_and_trace_mode_reach_experimental_options(self):
        trace = self._start(num_chips=1, num_sparse_cores=1)

        self.assertEqual(trace.profile.events, ["enter"])
        self.assertEqual(
            trace.profile.config.kwargs["experimental_options"],
            {
                "tpu_trace_mode": "TRACE_COMPUTE",
                "tpu_num_chips_to_profile_per_task": 1,
                "tpu_num_sparse_cores_to_trace": 1,
                "tpu_num_sparse_core_tiles_to_trace": 1,
            },
        )

    def test_disabled_limits_and_empty_mode_send_no_keys(self):
        # An omitted key is what makes libtpu trace every chip/SparseCore.
        trace = self._start(num_chips=0, num_sparse_cores=-1, trace_mode="")

        self.assertEqual(trace.profile.config.kwargs["experimental_options"], {})

    def test_tile_cap_only_at_one_sparse_core(self):
        trace = self._start(num_chips=0, num_sparse_cores=2)

        self.assertEqual(
            trace.profile.config.kwargs["experimental_options"],
            {"tpu_trace_mode": "TRACE_COMPUTE", "tpu_num_sparse_cores_to_trace": 2},
        )

    def test_without_cpu_drops_host_plane(self):
        trace = self._start(with_cpu=False)

        self.assertEqual(trace.profile.config.kwargs["host_tracer_level"], 0)
        self.assertEqual(trace.profile.config.kwargs["device_tracer_level"], 1)
        self.assertNotIn(torch.profiler.ProfilerActivity.CPU, trace.profile.activities)

    def test_python_tracer_off_by_default(self):
        # It costs 4.9 s of host busy time against 0.36 s without it.
        self.assertEqual(self._start().profile.config.kwargs["python_tracer_level"], 0)
        self.assertEqual(
            self._start(python_tracer=True).profile.config.kwargs[
                "python_tracer_level"
            ],
            1,
        )

    def test_stop_exits_the_profile(self):
        trace = self._start()

        stop_tpu_profile(trace)

        self.assertEqual(trace.profile.events, ["enter", "exit"])


class TestStopProfileSynchronization(CustomTestCase):
    """A staged TPU stop must drain the device before xprof stops.

    Without the synchronize, the trace ends while the profiled step is still
    queued on the device, so the staged capture misses it.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _stopping_manager(self, trace):
        with (
            envs.SGLANG_PROFILE_V2.override(False),
            patch.object(
                scheduler_profiler,
                "get_dp_tp_group",
                return_value=types.SimpleNamespace(cpu_group=None),
            ),
        ):
            manager = scheduler_profiler.SchedulerProfilerManager(
                get_forward_ct=lambda: 0
            )
        manager.profile_in_progress = True
        manager.tpu_profiler = trace
        manager.torch_profiler_output_dir = Path(self._tmp.name)
        manager.profiler_activities = ["TPU"]
        manager.profile_prefix = ""
        return manager

    def _stop_and_record(self, manager, **stop_kwargs):
        calls = []
        with (
            envs.SGLANG_PROFILE_V2.override(False),
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
            manager._stop_profile(**stop_kwargs)
        return calls

    def test_stage_tpu_profile_synchronizes_before_stopping(self):
        trace = object()
        manager = self._stopping_manager(trace)

        calls = self._stop_and_record(manager, stage=ForwardMode.EXTEND)

        self.assertEqual(calls, ["synchronize", ("stop", trace)])

    def test_non_stage_tpu_profile_stops_without_synchronizing(self):
        trace = object()
        manager = self._stopping_manager(trace)

        calls = self._stop_and_record(manager)

        self.assertEqual(calls, [("stop", trace)])

    def test_stage_profile_without_tpu_trace_does_not_synchronize(self):
        manager = self._stopping_manager(None)
        manager.profiler_activities = ["CPU"]

        calls = self._stop_and_record(manager, stage=ForwardMode.EXTEND)

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
