"""Unit tests for the TPU torch.profiler wiring and custom profiler options."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import (
    profiler_manager as scheduler_profiler,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu import profiler as tpu_profiler
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.profiler import (
    TpuProfileOptions,
    resolve_tpu_profile_options,
    start_tpu_profile,
    stop_tpu_profile,
)

# Pure-logic tests. torch_tpu is faked via sys.modules and ProfilerActivity.TPU
# is aliased to PrivateUse1 the way torch_tpu does it, so this runs on the CPU
# CI runner (no TPU required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ProfilerActivity = torch.profiler.ProfilerActivity


class _FakeTpuProfilerConfig(msgspec.Struct, frozen=True):
    host_tracer_level: int = 2
    device_tracer_level: int = 1
    python_tracer_level: int = 0
    run_dir: Path | str | None = None
    experimental_options: dict | None = None
    check_experimental_options: bool = True


def _fake_torch_tpu(calls, *, config_cls=None):
    """Fake torch_tpu's profiler modules to verify legacy _impl paths are unused."""
    impl = types.ModuleType("torch_tpu._internal.profiler._impl")
    impl.start_trace = lambda *args, **kwargs: calls.append("start_trace")
    impl.stop_trace = lambda *args, **kwargs: calls.append("stop_trace")
    profiler = types.ModuleType("torch_tpu._internal.profiler")
    profiler._impl = impl
    if config_cls is not None:
        profiler.TpuProfilerConfig = config_cls
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


def _clean_profile_env(testcase: CustomTestCase) -> None:
    enter_scope(testcase, tpu_envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(False))
    enter_scope(testcase, tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(1))
    enter_scope(testcase, tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(1))
    enter_scope(testcase, tpu_envs.SGLANG_TPU_PROFILE_OPTIONS.override(""))


class TestResolveTpuProfileOptions(CustomTestCase):
    """Option resolution across defaults, SGLANG_TPU_PROFILE_OPTIONS, prefix, and kwargs."""

    def setUp(self):
        super().setUp()
        _clean_profile_env(self)

    def test_resolve_plain_prefix_and_defaults(self):
        name, opts = resolve_tpu_profile_options("my_run_123")
        self.assertEqual(name, "my_run_123")
        self.assertEqual(
            opts,
            TpuProfileOptions(
                host_tracer_level=None,
                device_tracer_level=1,
                python_tracer_level=0,
                check_experimental_options=True,
                experimental_options={
                    "tpu_num_chips_to_profile_per_task": 1,
                    "tpu_num_sparse_cores_to_trace": 1,
                    "tpu_num_sparse_core_tiles_to_trace": 1,
                },
            ),
        )

        # Narrowing keys <= 0 drop the key so libtpu traces all chips/cores.
        with (
            tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(0),
            tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(-1),
        ):
            _, all_opts = resolve_tpu_profile_options(None)
        self.assertEqual(all_opts.experimental_options, {})

    def test_resolve_with_options_and_precedence(self):
        with tpu_envs.SGLANG_TPU_PROFILE_OPTIONS.override(
            "host_tracer_level:1;device_tracer_level:0;tpu_trace_mode:TRACE_ONLY_HOST"
        ):
            # Clean run-name prefix + profiler_kwargs override
            name, opts = resolve_tpu_profile_options(
                "decode_run1",
                profiler_kwargs={
                    "host_tracer_level": 3,
                    "tpu_trace_mode": "TRACE_COMPUTE",
                    "tpu_num_sparse_cores_to_trace": 2,
                },
            )
            self.assertEqual(name, "decode_run1")
            self.assertEqual(opts.host_tracer_level, 3)
            self.assertEqual(opts.device_tracer_level, 0)
            self.assertEqual(
                opts.experimental_options["tpu_trace_mode"], "TRACE_COMPUTE"
            )
            self.assertEqual(
                opts.experimental_options["tpu_num_sparse_cores_to_trace"], 2
            )
            self.assertNotIn(
                "tpu_num_sparse_core_tiles_to_trace", opts.experimental_options
            )

            # Options in both profile_prefix and profiler_kwargs -> profiler_kwargs wins
            name2, opts2 = resolve_tpu_profile_options(
                "host_tracer_level:2;tpu_perf_counters:true",
                profiler_kwargs={
                    "host_tracer_level": 0,
                    "python_tracer_level": 1,
                },
            )
            self.assertIsNone(name2)
            self.assertEqual(opts2.host_tracer_level, 0)
            self.assertEqual(opts2.python_tracer_level, 1)
            self.assertIs(opts2.experimental_options["tpu_perf_counters"], True)

    def test_resolve_invalid_options(self):
        with self.assertRaises(ValueError):
            resolve_tpu_profile_options("host_tracer_level-3;")
        with self.assertRaises(ValueError):
            resolve_tpu_profile_options("tpu_trace_mode:bad$val")
        with self.assertRaises(ValueError):
            resolve_tpu_profile_options(
                "run1", profiler_kwargs={"host_tracer_level": -1}
            )
        # Reserved keys (run_dir, worker_rank) cannot be overridden by callers.
        with self.assertRaisesRegex(ValueError, "run_dir"):
            resolve_tpu_profile_options("run_dir:/tmp/other")
        with self.assertRaisesRegex(ValueError, "worker_rank"):
            resolve_tpu_profile_options("run1", profiler_kwargs={"worker_rank": "0"})
        # Narrowing keys must be integers (not bool or str).
        with self.assertRaisesRegex(ValueError, "tpu_num_chips_to_profile_per_task"):
            resolve_tpu_profile_options(
                "run1",
                profiler_kwargs={"tpu_num_chips_to_profile_per_task": True},
            )
        with self.assertRaisesRegex(ValueError, "tpu_num_sparse_cores_to_trace"):
            resolve_tpu_profile_options("tpu_num_sparse_cores_to_trace:all")


class TestStartUsesNativeTorchProfiler(CustomTestCase):
    """start_tpu_profile builds one torch.profiler with TPU activity and config."""

    def setUp(self):
        super().setUp()
        _clean_profile_env(self)

    def _start(self, *, with_cpu, python_tracer=False, options=None):
        calls = []
        output_dir = Path(enter_scope(self, tempfile.TemporaryDirectory())) / "r"
        fake_tpu = types.SimpleNamespace(
            profiler=types.SimpleNamespace(TpuProfilerConfig=_FakeTpuProfilerConfig)
        )
        with (
            tpu_envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(python_tracer),
            patch.dict(sys.modules, _fake_torch_tpu(calls)),
            patch.object(torch, "tpu", fake_tpu, create=True),
            patch.object(
                _ProfilerActivity, "TPU", _ProfilerActivity.PrivateUse1, create=True
            ),
            patch("torch.profiler.profile") as profile,
            patch("torch.profiler.tensorboard_trace_handler") as handler,
        ):
            trace = start_tpu_profile(
                output_dir=output_dir, with_cpu=with_cpu, options=options
            )
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
        cfg = kwargs["experimental_config"]
        self.assertEqual(cfg.run_dir, output_dir)
        self.assertEqual(cfg.host_tracer_level, 2)
        self.assertEqual(cfg.device_tracer_level, 1)
        self.assertEqual(cfg.python_tracer_level, 0)
        self.assertEqual(
            cfg.experimental_options,
            {
                "tpu_num_chips_to_profile_per_task": 1,
                "tpu_num_sparse_cores_to_trace": 1,
                "tpu_num_sparse_core_tiles_to_trace": 1,
            },
        )
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
        self.assertEqual(kwargs["experimental_config"].run_dir, output_dir)
        self.assertEqual(kwargs["experimental_config"].host_tracer_level, 0)
        handler.assert_called_once_with(str(output_dir), use_gzip=True)
        self.assertEqual(calls, [])

    def test_torch_2_11_privateuse1_and_internal_config_fallback(self):
        # On torch 2.11, ProfilerActivity.TPU and torch.tpu.profiler are absent;
        # fall back to PrivateUse1 and torch_tpu._internal.profiler.TpuProfilerConfig.
        calls = []
        output_dir = Path(enter_scope(self, tempfile.TemporaryDirectory())) / "r211"
        activities = types.SimpleNamespace(
            CPU=_ProfilerActivity.CPU,
            PrivateUse1=_ProfilerActivity.PrivateUse1,
        )
        with (
            patch.dict(
                sys.modules,
                _fake_torch_tpu(calls, config_cls=_FakeTpuProfilerConfig),
            ),
            patch.object(torch, "tpu", None, create=True),
            patch("torch.profiler.ProfilerActivity", activities),
            patch("torch.profiler.profile") as profile,
            patch("torch.profiler.tensorboard_trace_handler"),
        ):
            start_tpu_profile(output_dir=output_dir, with_cpu=True)

        kwargs = profile.call_args.kwargs
        self.assertEqual(
            kwargs["activities"],
            [_ProfilerActivity.CPU, _ProfilerActivity.PrivateUse1],
        )
        self.assertIsInstance(kwargs["experimental_config"], _FakeTpuProfilerConfig)
        self.assertEqual(kwargs["experimental_config"].run_dir, output_dir)
        self.assertEqual(calls, [])

    def test_python_tracer_on(self):
        _, profile, _, calls, _ = self._start(with_cpu=True, python_tracer=True)

        # with_stack=True and python_tracer_level=1 are both set.
        self.assertTrue(profile.call_args.kwargs["with_stack"])
        self.assertEqual(
            profile.call_args.kwargs["experimental_config"].python_tracer_level, 1
        )
        self.assertEqual(calls, [])

    def test_python_tracer_override_off(self):
        # Higher-precedence profiler_kwargs can disable SGLANG_TPU_PROFILE_PYTHON_TRACER=True.
        with tpu_envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(True):
            _, opts = resolve_tpu_profile_options(
                None, profiler_kwargs={"python_tracer_level": 0}
            )
        _, profile, _, _, _ = self._start(
            with_cpu=True, python_tracer=True, options=opts
        )
        self.assertFalse(profile.call_args.kwargs["with_stack"])
        self.assertEqual(
            profile.call_args.kwargs["experimental_config"].python_tracer_level, 0
        )

    def test_output_dir_with_comma_raises(self):
        with self.assertRaisesRegex(ValueError, "output_dir"):
            start_tpu_profile(output_dir=Path("/tmp/bad,dir"))

    def test_profile_option_envs_do_not_key_compile_cache(self):
        from sglang_tpu import compile_cache

        for env_name in (
            "SGLANG_TPU_PROFILE_NUM_CHIPS",
            "SGLANG_TPU_PROFILE_NUM_SPARSE_CORES",
            "SGLANG_TPU_PROFILE_OPTIONS",
            "SGLANG_TPU_PROFILE_PYTHON_TRACER",
        ):
            self.assertIn(env_name, compile_cache._ENV_FIELDS_NOT_KEYED)

    def test_missing_tpu_activity_raises(self):
        activities = types.SimpleNamespace(CPU=_ProfilerActivity.CPU)
        with (
            patch("torch.profiler.ProfilerActivity", activities),
            patch("torch.profiler.profile") as profile,
            self.assertRaisesRegex(RuntimeError, "ProfilerActivity.TPU"),
        ):
            start_tpu_profile(output_dir=Path("/nonexistent/unused"))
        profile.assert_not_called()

    def test_missing_tpu_profiler_config_raises(self):
        calls = []
        with (
            patch.dict(sys.modules, _fake_torch_tpu(calls, config_cls=None)),
            patch.object(torch, "tpu", None, create=True),
            patch.object(
                _ProfilerActivity, "TPU", _ProfilerActivity.PrivateUse1, create=True
            ),
            patch("torch.profiler.profile") as profile,
            self.assertRaisesRegex(RuntimeError, "TpuProfilerConfig"),
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
        _clean_profile_env(self)
        tmpdir = Path(enter_scope(self, tempfile.TemporaryDirectory()))
        output_dir = tmpdir / "profile-id-TP-0"
        with (
            patch.object(
                _ProfilerActivity, "TPU", _ProfilerActivity.PrivateUse1, create=True
            ),
            patch.object(
                tpu_profiler,
                "_tpu_profiler_config_cls",
                return_value=lambda **kwargs: None,
            ),
        ):
            trace = start_tpu_profile(output_dir=output_dir, with_cpu=True)
            torch.randn(8, 8) @ torch.randn(8, 8)
            stop_tpu_profile(trace)

        self.assertEqual(len(list(output_dir.glob("*.pt.trace.json.gz"))), 1)


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
