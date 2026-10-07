"""Unit tests for the TPU torch.profiler wiring."""

import os
import struct
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import profiler as tpu_profiler
from sglang.srt.hardware_backend.tpu.profiler import (
    TpuProfileOptions,
    _find_xplane,
    _has_tpu_device_events,
    resolve_tpu_profile_options,
    start_tpu_profile,
    stop_tpu_profile,
)
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.managers.scheduler_components import profiler_manager
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Pure-logic tests. torch_tpu is faked via sys.modules, ProfilerActivity.TPU is
# aliased to PrivateUse1 the way torch_tpu does it, and torch.tpu.profiler's
# TpuProfilerConfig is faked, so this runs on the CPU CI runner (no TPU
# required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ProfilerActivity = torch.profiler.ProfilerActivity
_XPLANE_ENV_VAR = "TPU_PROFILER_OUTPUT_DIR"
# Defaults from SGLANG_TPU_PROFILE_NUM_CHIPS=1, SGLANG_TPU_PROFILE_NUM_SPARSE_CORES=1.
_DEFAULT_EXPERIMENTAL = {
    "tpu_num_chips_to_profile_per_task": 1,
    "tpu_num_sparse_cores_to_trace": 1,
    "tpu_num_sparse_core_tiles_to_trace": 1,
}


class _FakeTpuProfilerConfig(torch.profiler._ExperimentalConfig):
    """Stands in for torch.tpu.profiler.TpuProfilerConfig; records its kwargs.

    It is a real _ExperimentalConfig, so it also goes through a real
    torch.profiler.profile.
    """

    instances = []

    def __init__(self, **kwargs):
        super().__init__(custom_profiler_config=f"run_dir:{kwargs['run_dir']}")
        self.kwargs = kwargs
        _FakeTpuProfilerConfig.instances.append(self)


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


def _fake_torch_tpu_attrs(test):
    """Patch ProfilerActivity.TPU and torch.tpu.profiler for one test."""
    _FakeTpuProfilerConfig.instances = []
    test.enterContext(
        patch.object(
            _ProfilerActivity, "TPU", _ProfilerActivity.PrivateUse1, create=True
        )
    )
    tpu_module = types.SimpleNamespace(
        profiler=types.SimpleNamespace(TpuProfilerConfig=_FakeTpuProfilerConfig)
    )
    test.enterContext(patch.object(torch, "tpu", tpu_module, create=True))


def _xplane_env(**env_vars):
    """os.environ with the plugin output-dir env var cleared."""
    env = {k: v for k, v in os.environ.items() if k != _XPLANE_ENV_VAR}
    env.update(env_vars)
    return patch.dict(os.environ, env, clear=True)


def _clean_profile_env(test):
    """Clear the SGLANG_TPU_PROFILE_* env vars for one test."""
    test.enterContext(patch.dict(os.environ))
    for env in (
        envs.SGLANG_TPU_PROFILE_NUM_CHIPS,
        envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES,
        envs.SGLANG_TPU_PROFILE_PYTHON_TRACER,
        envs.SGLANG_TPU_PROFILE_OPTIONS,
    ):
        env.clear()


# Minimal protobuf encoders for XSpace test files.
def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _int(field, value):
    return _varint(field << 3) + _varint(value)


def _bytes(field, payload):
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _double(field, value):
    return _varint(field << 3 | 1) + struct.pack("<d", value)


def _line(num_events):
    event = _int(1, 1) + _int(2, 5) + _int(3, 1000)
    return (
        _int(1, 1)
        + _bytes(2, b"XLA Ops")
        + _int(3, 1_700_000_000_000_000_000)
        + b"".join(_bytes(4, event) for _ in range(num_events))
        + _int(9, 10**12)
    )


def _plane(name, *line_event_counts):
    event_metadata_entry = _int(1, 1) + _bytes(2, _int(1, 1) + _bytes(2, b"fusion"))
    stat = _int(1, 2) + _double(2, 1.5)
    return (
        _int(1, 3)
        + _bytes(2, name.encode())
        + b"".join(_bytes(3, _line(n)) for n in line_event_counts)
        + _bytes(4, event_metadata_entry)
        + _bytes(6, stat)
    )


def _xspace(*planes):
    return b"".join(_bytes(1, p) for p in planes) + _bytes(4, b"host-0")


def _write_xplane(run_dir, ts="2026_01_01_00_00_00", name="host.xplane.pb"):
    xplane = Path(run_dir) / "plugins" / "profile" / ts / name
    xplane.parent.mkdir(parents=True, exist_ok=True)
    xplane.write_bytes(_xspace(_plane("/device:TPU:0", 2)))
    return xplane


class TestResolveOptions(CustomTestCase):
    def setUp(self):
        _clean_profile_env(self)

    def test_defaults(self):
        name, options = resolve_tpu_profile_options(None)
        self.assertIsNone(name)
        self.assertEqual(
            options,
            TpuProfileOptions(
                host_tracer_level=None,
                device_tracer_level=1,
                python_tracer_level=0,
                check_experimental_options=True,
                experimental_options=_DEFAULT_EXPERIMENTAL,
            ),
        )

    def test_plain_prefix_is_the_name(self):
        name, options = resolve_tpu_profile_options("llama-c64")
        self.assertEqual(name, "llama-c64")
        self.assertEqual(options, resolve_tpu_profile_options(None)[1])

    def test_empty_prefix(self):
        self.assertIsNone(resolve_tpu_profile_options("")[0])

    def test_options_prefix(self):
        name, options = resolve_tpu_profile_options(
            "host_tracer_level:0; device_tracer_level : 1;python_tracer_level:1;"
            "tpu_trace_mode:TRACE_COMPUTE;tpu_perf_counters:True;"
            "tpu_num_chips_to_profile_per_task:4;"
        )
        self.assertIsNone(name)
        self.assertEqual(options.host_tracer_level, 0)
        self.assertEqual(options.device_tracer_level, 1)
        self.assertEqual(options.python_tracer_level, 1)
        self.assertTrue(options.check_experimental_options)
        self.assertEqual(
            options.experimental_options,
            {
                **_DEFAULT_EXPERIMENTAL,
                "tpu_num_chips_to_profile_per_task": 4,
                "tpu_trace_mode": "TRACE_COMPUTE",
                "tpu_perf_counters": True,
            },
        )

    def test_single_option_without_semicolon(self):
        name, options = resolve_tpu_profile_options("host_tracer_level:3")
        self.assertIsNone(name)
        self.assertEqual(options.host_tracer_level, 3)

    def test_check_experimental_options(self):
        _, options = resolve_tpu_profile_options("check_experimental_options:false")
        self.assertFalse(options.check_experimental_options)
        self.assertNotIn("check_experimental_options", options.experimental_options)

    def test_string_values_keep_commas_and_colons(self):
        _, options = resolve_tpu_profile_options("some_option:a,b:c")
        self.assertEqual(options.experimental_options["some_option"], "a,b:c")

    def test_env_vars(self):
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(0),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(2),
            envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(True),
        ):
            _, options = resolve_tpu_profile_options(None)
        self.assertEqual(options.python_tracer_level, 1)
        # <= 0 chips: trace all of them. > 1 SparseCore: no tile cap.
        self.assertEqual(
            options.experimental_options, {"tpu_num_sparse_cores_to_trace": 2}
        )

    def test_env_vars_trace_everything(self):
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(-1),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(0),
        ):
            _, options = resolve_tpu_profile_options(None)
        self.assertEqual(options.experimental_options, {})

    def test_prefix_can_lift_the_narrowing_defaults(self):
        _, options = resolve_tpu_profile_options(
            "tpu_num_chips_to_profile_per_task:0;tpu_num_sparse_cores_to_trace:-1"
        )
        self.assertEqual(options.experimental_options, {})

    def test_more_sparse_cores_drops_the_tile_default(self):
        _, options = resolve_tpu_profile_options("tpu_num_sparse_cores_to_trace:4")
        self.assertEqual(
            options.experimental_options,
            {
                "tpu_num_chips_to_profile_per_task": 1,
                "tpu_num_sparse_cores_to_trace": 4,
            },
        )

    def test_explicit_tile_count_is_kept(self):
        _, options = resolve_tpu_profile_options("tpu_num_sparse_core_tiles_to_trace:2")
        self.assertEqual(
            options.experimental_options["tpu_num_sparse_core_tiles_to_trace"], 2
        )
        _, options = resolve_tpu_profile_options(
            "tpu_num_sparse_cores_to_trace:4;tpu_num_sparse_core_tiles_to_trace:2"
        )
        self.assertEqual(
            options.experimental_options["tpu_num_sparse_core_tiles_to_trace"], 2
        )

    def test_options_env_var_and_prefix_precedence(self):
        with envs.SGLANG_TPU_PROFILE_OPTIONS.override(
            "host_tracer_level:1;tpu_trace_mode:TRACE_ONLY_HOST;"
            "tpu_num_sparse_cores_to_trace:4"
        ):
            name, from_env = resolve_tpu_profile_options("my-run")
            _, from_prefix = resolve_tpu_profile_options("tpu_trace_mode:TRACE_COMPUTE")
        self.assertEqual(name, "my-run")
        self.assertEqual(from_env.host_tracer_level, 1)
        self.assertEqual(
            from_env.experimental_options["tpu_trace_mode"], "TRACE_ONLY_HOST"
        )
        self.assertEqual(
            from_env.experimental_options["tpu_num_sparse_cores_to_trace"], 4
        )
        # The prefix overrides the env var key by key.
        self.assertEqual(from_prefix.host_tracer_level, 1)
        self.assertEqual(
            from_prefix.experimental_options["tpu_trace_mode"], "TRACE_COMPUTE"
        )

    def test_invalid_options(self):
        cases = {
            "host_tracer_level-3;a:1": "invalid profiler option",
            "host_tracer_level:3$": "invalid characters",
            "tpu_trace_mode:x;9bad:1": "invalid profiler option",
            "host_tracer_level:abc": "non-negative integer",
            "device_tracer_level:-1": "non-negative integer",
            "python_tracer_level:true": "non-negative integer",
            "check_experimental_options:1": "true or false",
            "run_dir:/elsewhere": "SGLang sets 'run_dir'",
            "worker_rank:7": "SGLang sets 'worker_rank'",
            "tpu_trace_mode:\u00fcber": "ASCII",
        }
        for prefix, error in cases.items():
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(ValueError, error):
                    resolve_tpu_profile_options(prefix)

    def test_invalid_env_var_names_the_source(self):
        with (
            envs.SGLANG_TPU_PROFILE_OPTIONS.override("oops"),
            self.assertRaisesRegex(ValueError, "SGLANG_TPU_PROFILE_OPTIONS"),
        ):
            resolve_tpu_profile_options(None)


class TestStartUsesNativeTorchProfiler(CustomTestCase):
    """start_tpu_profile builds one torch.profiler with the TPU activity."""

    def setUp(self):
        _clean_profile_env(self)
        _fake_torch_tpu_attrs(self)

    def _start(self, *, with_cpu, python_tracer=False, **kwargs):
        calls = []
        output_dir = Path(self.enterContext(tempfile.TemporaryDirectory())) / "r"
        with (
            envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(python_tracer),
            patch.dict(sys.modules, _fake_torch_tpu(calls)),
            patch("torch.profiler.profile") as profile,
            patch("torch.profiler.tensorboard_trace_handler") as handler,
        ):
            trace = start_tpu_profile(
                output_dir=output_dir, with_cpu=with_cpu, **kwargs
            )
        return trace, profile, handler, calls, output_dir

    def _config(self, profile):
        config = profile.call_args.kwargs["experimental_config"]
        self.assertIsInstance(config, _FakeTpuProfilerConfig)
        return config.kwargs

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
        self.assertEqual(
            self._config(profile),
            {
                "run_dir": str(output_dir),
                "worker_rank": None,
                "host_tracer_level": 2,
                "device_tracer_level": 1,
                "python_tracer_level": 0,
                "experimental_options": _DEFAULT_EXPERIMENTAL,
                "check_experimental_options": True,
            },
        )
        profile.return_value.start.assert_called_once_with()
        self.assertIs(trace.profiler, profile.return_value)
        self.assertEqual(trace.output_dir, output_dir)
        self.assertTrue(output_dir.is_dir())
        self.assertEqual(calls, [])  # the legacy _impl.start_trace is not used

    def test_tpu_only_turns_the_host_tracer_off(self):
        _, profile, handler, calls, output_dir = self._start(with_cpu=False)

        kwargs = profile.call_args.kwargs
        self.assertEqual(kwargs["activities"], [_ProfilerActivity.PrivateUse1])
        self.assertFalse(kwargs["with_stack"])
        handler.assert_called_once_with(str(output_dir), use_gzip=True)
        self.assertEqual(self._config(profile)["host_tracer_level"], 0)
        self.assertEqual(calls, [])

    def test_python_tracer_on(self):
        _, profile, _, calls, _ = self._start(with_cpu=True, python_tracer=True)

        self.assertTrue(profile.call_args.kwargs["with_stack"])
        self.assertEqual(self._config(profile)["python_tracer_level"], 1)
        self.assertEqual(calls, [])

    def test_python_tracer_on_tpu_only(self):
        _, profile, _, _, _ = self._start(with_cpu=False, python_tracer=True)

        self.assertEqual(
            profile.call_args.kwargs["activities"], [_ProfilerActivity.PrivateUse1]
        )
        self.assertTrue(profile.call_args.kwargs["with_stack"])
        self.assertEqual(self._config(profile)["python_tracer_level"], 1)

    def test_explicit_options_and_worker_rank(self):
        options = TpuProfileOptions(
            host_tracer_level=3,
            device_tracer_level=0,
            python_tracer_level=1,
            check_experimental_options=False,
            experimental_options={"tpu_perf_counters": True},
        )
        _, profile, _, _, output_dir = self._start(
            with_cpu=False, options=options, worker_rank="TP-1-DP-2"
        )

        self.assertEqual(
            self._config(profile),
            {
                "run_dir": str(output_dir),
                "worker_rank": "TP-1-DP-2",
                "host_tracer_level": 3,
                "device_tracer_level": 0,
                "python_tracer_level": 1,
                "experimental_options": {"tpu_perf_counters": True},
                "check_experimental_options": False,
            },
        )

    def test_no_experimental_options_passes_none(self):
        _, profile, _, _, _ = self._start(with_cpu=True, options=TpuProfileOptions())
        self.assertIsNone(self._config(profile)["experimental_options"])

    def test_relative_output_dir_becomes_absolute(self):
        tmpdir = self.enterContext(tempfile.TemporaryDirectory())
        cwd = os.getcwd()
        os.chdir(tmpdir)
        self.addCleanup(os.chdir, cwd)
        with (
            patch("torch.profiler.profile") as profile,
            patch("torch.profiler.tensorboard_trace_handler") as handler,
        ):
            trace = start_tpu_profile(output_dir=Path("rel/r"))
        expected = Path(tmpdir).resolve() / "rel" / "r"
        self.assertEqual(Path(self._config(profile)["run_dir"]).resolve(), expected)
        self.assertEqual(Path(handler.call_args.args[0]).resolve(), expected)
        self.assertEqual(trace.output_dir.resolve(), expected)

    def test_unsupported_output_dir_raises(self):
        for bad in ("a,b", 'a"b', "a\\b", "\u00fcber"):
            with self.subTest(bad=bad):
                with (
                    patch("torch.profiler.profile") as profile,
                    self.assertRaisesRegex(ValueError, "output directory"),
                ):
                    start_tpu_profile(output_dir=Path("/nonexistent") / bad)
                profile.assert_not_called()

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
        with (
            patch.object(torch, "tpu", types.SimpleNamespace(), create=True),
            patch("torch.profiler.profile") as profile,
            self.assertRaisesRegex(RuntimeError, "TpuProfilerConfig"),
        ):
            start_tpu_profile(output_dir=Path("/nonexistent/unused"))
        profile.assert_not_called()

    def test_tpu_profiler_config_import_error_raises_runtime_error(self):
        # torch.tpu.profiler imports TpuProfilerConfig lazily (module
        # __getattr__); a broken import must fail /start_profile cleanly.
        class _BrokenApi:
            def __getattr__(self, name):
                raise ImportError("broken torch_tpu")

        with (
            patch.object(
                torch, "tpu", types.SimpleNamespace(profiler=_BrokenApi()), create=True
            ),
            patch("torch.profiler.profile") as profile,
            self.assertRaisesRegex(RuntimeError, "broken torch_tpu"),
        ):
            start_tpu_profile(output_dir=Path("/nonexistent/unused"))
        profile.assert_not_called()


class TestStop(CustomTestCase):
    def setUp(self):
        self.tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_stop_stops_the_profiler_not_stop_trace(self):
        calls = []
        profiler = MagicMock()
        trace = tpu_profiler._TpuTrace(
            output_dir=self.tmpdir, profiler=profiler, started_at=time.time()
        )
        with (
            patch.dict(sys.modules, _fake_torch_tpu(calls)),
            _xplane_env(),
            self.assertLogs(tpu_profiler.logger, "ERROR"),
        ):
            stop_tpu_profile(trace)

        profiler.stop.assert_called_once_with()
        self.assertEqual(calls, [])

    def test_stop_logs_xplane_path_in_the_run_dir(self):
        run_dir = self.tmpdir / "profile-1-TP-0"
        profiler = MagicMock()
        written = []
        profiler.stop.side_effect = lambda: written.append(
            _write_xplane(run_dir, name="host_TP-0.xplane.pb")
        )
        # An xplane in another rank's directory must not be picked up.
        _write_xplane(self.tmpdir / "profile-1-TP-1", name="host_TP-1.xplane.pb")
        trace = tpu_profiler._TpuTrace(
            output_dir=run_dir, profiler=profiler, started_at=time.time() - 1
        )
        with _xplane_env(), self.assertLogs(tpu_profiler.logger, "INFO") as logs:
            stop_tpu_profile(trace)

        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "INFO")
        self.assertIn(str(written[0]), logs.output[0])

    def test_check_failure_does_not_fail_stop(self):
        profiler = MagicMock()
        trace = tpu_profiler._TpuTrace(
            output_dir=Path("/unused"), profiler=profiler, started_at=time.time()
        )
        with (
            patch.object(tpu_profiler, "_find_xplane", side_effect=OSError("boom")),
            self.assertLogs(tpu_profiler.logger, "WARNING") as logs,
        ):
            stop_tpu_profile(trace)  # must not raise

        profiler.stop.assert_called_once_with()
        self.assertIn("could not check the xplane", logs.output[0])

    def test_stages_keep_separate_xplanes(self):
        # profile_by_stage: EXTEND and DECODE stop within the same second.
        # Each has its own run_dir, so both xplanes survive.
        xplanes = []
        for stage in ("EXTEND", "DECODE"):
            run_dir = self.tmpdir / f"profile-1-{stage}-TP-0"
            profiler = MagicMock()
            profiler.stop.side_effect = lambda d=run_dir: xplanes.append(
                _write_xplane(d, name="host_TP-0.xplane.pb")
            )
            with _xplane_env(), self.assertLogs(tpu_profiler.logger, "INFO") as logs:
                stop_tpu_profile(
                    tpu_profiler._TpuTrace(
                        output_dir=run_dir,
                        profiler=profiler,
                        started_at=time.time() - 1,
                    )
                )
            self.assertEqual([r.levelname for r in logs.records], ["INFO"])
            self.assertIn(str(xplanes[-1]), logs.output[0])
        self.assertNotEqual(xplanes[0], xplanes[1])
        self.assertTrue(all(x.is_file() for x in xplanes))


class TestRealTorchProfilerOnCpu(CustomTestCase):
    """End to end through the real torch.profiler on a CPU-only torch.

    Without torch_tpu, PrivateUse1 is dropped, so this checks the API wiring
    (including experimental_config) and the Kineto trace location, not TPU
    events.
    """

    def setUp(self):
        _clean_profile_env(self)
        _fake_torch_tpu_attrs(self)

    def test_cpu_trace_lands_in_output_dir(self):
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        output_dir = tmpdir / "profile-id-TP-0"
        with (
            _xplane_env(),
            self.assertLogs(tpu_profiler.logger, "INFO") as logs,
        ):
            trace = start_tpu_profile(
                output_dir=output_dir, with_cpu=True, worker_rank="TP-0"
            )
            torch.randn(8, 8) @ torch.randn(8, 8)
            stop_tpu_profile(trace)

        self.assertEqual(len(_FakeTpuProfilerConfig.instances), 1)
        self.assertEqual(len(list(output_dir.glob("*.pt.trace.json.gz"))), 1)
        # No plugin, so no xplane: the empty-trace check must say so, and
        # point at the capture's own directory.
        errors = [r.getMessage() for r in logs.records if r.levelname == "ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertIn("no .xplane.pb", errors[0])
        self.assertIn(str(output_dir / "plugins" / "profile"), errors[0])


class TestFindXplane(CustomTestCase):
    def setUp(self):
        self.tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _write(self, base, ts, mtime=None, name="host.xplane.pb"):
        path = Path(base) / "plugins" / "profile" / ts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_finds_xplane_under_run_dir(self):
        started_at = time.time() - 10
        xplane = self._write(self.tmpdir, "2026_01_01_00_00_01")
        with _xplane_env():
            self.assertEqual(_find_xplane(self.tmpdir, started_at), xplane)

    def test_ignores_tmpdir(self):
        # The TMPDIR fallback only applies without a run_dir.
        self._write(self.tmpdir / "tmp", "2026_01_01_00_00_01")
        with _xplane_env(TMPDIR=str(self.tmpdir / "tmp")):
            self.assertIsNone(_find_xplane(self.tmpdir / "run", time.time() - 10))

    def test_ignores_xplanes_from_before_the_capture(self):
        started_at = time.time()
        self._write(self.tmpdir, "2020_01_01_00_00_00", mtime=started_at - 100)
        with _xplane_env():
            self.assertIsNone(_find_xplane(self.tmpdir, started_at))

    def test_whole_second_mtime_counts(self):
        # Filesystems with whole-second mtimes can round the xplane's mtime
        # down to before started_at.
        second = float(int(time.time()) - 5)
        xplane = self._write(self.tmpdir, "ts", mtime=second)
        with _xplane_env():
            self.assertEqual(_find_xplane(self.tmpdir, second + 0.7), xplane)

    def test_picks_newest(self):
        started_at = time.time() - 100
        self._write(self.tmpdir, "a", mtime=started_at + 1)
        newest = self._write(self.tmpdir, "b", mtime=started_at + 2)
        with _xplane_env():
            self.assertEqual(_find_xplane(self.tmpdir, started_at), newest)

    def test_worker_rank_matches_only_its_own_xplane(self):
        # All ranks share TPU_PROFILER_OUTPUT_DIR; a newer xplane of another
        # rank (including one whose rank starts with this one's) is ignored.
        started_at = time.time() - 100
        base = self.tmpdir / "override"
        own = self._write(base, "a", started_at + 1, name="host_TP-1.xplane.pb")
        self._write(base, "b", started_at + 2, name="host_TP-0.xplane.pb")
        self._write(base, "b", started_at + 3, name="host_TP-11.xplane.pb")
        with _xplane_env(**{_XPLANE_ENV_VAR: str(base)}):
            self.assertEqual(_find_xplane(self.tmpdir / "run", started_at, "TP-1"), own)
            self.assertIsNone(_find_xplane(self.tmpdir / "run", started_at, "TP-2"))

    def test_output_dir_env_var_takes_precedence(self):
        started_at = time.time() - 10
        base = self.tmpdir / "override"
        xplane = self._write(base, "2026_01_01_00_00_02")
        with _xplane_env(**{_XPLANE_ENV_VAR: str(base)}):
            self.assertEqual(_find_xplane(self.tmpdir / "run", started_at), xplane)
            with self.assertLogs(tpu_profiler.logger, "INFO") as logs:
                tpu_profiler._check_xplane(self.tmpdir / "run", started_at)
        # The test file is empty, so the warning is followed by the
        # empty-capture error about the overriding directory's xplane.
        self.assertEqual(
            [r.levelname for r in logs.records], ["WARNING", "ERROR"], logs.output
        )
        self.assertIn(_XPLANE_ENV_VAR, logs.output[0])
        self.assertIn(str(xplane), logs.output[1])

    def test_missing_xplane_logs_error(self):
        with (
            _xplane_env(),
            self.assertLogs(tpu_profiler.logger, "ERROR") as logs,
        ):
            tpu_profiler._check_xplane(self.tmpdir, time.time())
        self.assertIn("no .xplane.pb", logs.output[0])
        self.assertIn(str(self.tmpdir / "plugins" / "profile"), logs.output[0])


class TestHasTpuDeviceEvents(CustomTestCase):
    def setUp(self):
        self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "x.pb"

    def _check(self, data):
        self.path.write_bytes(data)
        return _has_tpu_device_events(self.path)

    def test_tpu_plane_with_events(self):
        data = _xspace(_plane("/host:CPU", 3), _plane("/device:TPU:0", 0, 2))
        self.assertTrue(self._check(data))

    def test_tpu_plane_without_events(self):
        data = _xspace(_plane("/host:CPU", 3), _plane("/device:TPU:0", 0, 0))
        self.assertFalse(self._check(data))

    def test_host_plane_only(self):
        self.assertFalse(self._check(_xspace(_plane("/host:CPU", 5))))

    def test_non_core_plane_does_not_count(self):
        self.assertFalse(self._check(_xspace(_plane("/device:TPU_NON_CORE:0", 4))))

    def test_empty_file(self):
        self.assertFalse(self._check(b""))

    def test_truncated_file_raises(self):
        # Cut inside the only plane, so the walk can't return before it.
        data = _bytes(1, _plane("/device:TPU:0", 2))
        with self.assertRaises((ValueError, IndexError)):
            self._check(data[:-5])

    def test_check_xplane_logs_empty_device_plane(self):
        tmpdir = self.path.parent
        xplane = tmpdir / "plugins/profile/ts/host.xplane.pb"
        xplane.parent.mkdir(parents=True)
        xplane.write_bytes(_xspace(_plane("/host:CPU", 3), _plane("/device:TPU:0")))
        with (
            _xplane_env(),
            self.assertLogs(tpu_profiler.logger, "ERROR") as logs,
        ):
            tpu_profiler._check_xplane(tmpdir, time.time() - 10)
        self.assertIn("no /device:TPU plane with events", logs.output[0])

    def test_check_xplane_survives_unparsable_file(self):
        tmpdir = self.path.parent
        xplane = tmpdir / "plugins/profile/ts/host.xplane.pb"
        xplane.parent.mkdir(parents=True)
        xplane.write_bytes(b"\x0a\xff\xff")
        with (
            _xplane_env(),
            self.assertLogs(tpu_profiler.logger, "INFO") as logs,
        ):
            tpu_profiler._check_xplane(tmpdir, time.time() - 10)
        self.assertEqual(
            [r.levelname for r in logs.records], ["WARNING", "INFO"], logs.output
        )


class TestSchedulerProfilerManagerOnTpu(CustomTestCase):
    """SchedulerProfilerManager's TPU path: directories, options and rank."""

    def setUp(self):
        _clean_profile_env(self)
        self.tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.parallel = types.SimpleNamespace(
            tp_rank=1, dp_size=2, dp_rank=0, pp_size=1, moe_ep_size=1
        )
        self.enterContext(patch.object(profiler_manager, "_is_tpu", True))
        self.enterContext(
            patch.object(profiler_manager, "get_parallel", lambda: self.parallel)
        )
        self.enterContext(envs.SGLANG_PROFILE_V2.override(False))
        self.start = self.enterContext(patch.object(tpu_profiler, "start_tpu_profile"))
        self.stop = self.enterContext(patch.object(tpu_profiler, "stop_tpu_profile"))
        self.manager = profiler_manager.SchedulerProfilerManager(
            dp_tp_cpu_group=None, get_forward_ct=lambda: 0
        )

    def _start(self, profile_prefix=None, **kwargs):
        req = ProfileReq(
            req_type=ProfileReqType.START_PROFILE,
            output_dir=str(self.tmpdir),
            activities=kwargs.pop("activities", ["CPU", "TPU"]),
            profile_id="1700000000.5",
            profile_prefix=profile_prefix,
            **kwargs,
        )
        return self.manager._profile(req)

    def test_name_prefix(self):
        result = self._start("llama")

        self.assertTrue(result.success, result.message)
        kwargs = self.start.call_args.kwargs
        self.assertEqual(
            kwargs["output_dir"], self.tmpdir / "llama-1700000000.5-TP-1-DP-0"
        )
        self.assertTrue(kwargs["with_cpu"])
        self.assertEqual(kwargs["worker_rank"], "TP-1-DP-0")
        self.assertEqual(kwargs["options"], resolve_tpu_profile_options(None)[1])

    def test_options_prefix(self):
        result = self._start(
            "host_tracer_level:0;tpu_perf_counters:true", activities=["TPU"]
        )

        self.assertTrue(result.success, result.message)
        kwargs = self.start.call_args.kwargs
        # The options string doesn't end up in the directory name.
        self.assertEqual(
            kwargs["output_dir"], self.tmpdir / "profile-1700000000.5-TP-1-DP-0"
        )
        self.assertFalse(kwargs["with_cpu"])
        self.assertEqual(kwargs["options"].host_tracer_level, 0)
        self.assertIs(kwargs["options"].experimental_options["tpu_perf_counters"], True)
        self.assertEqual(self.manager.profile_prefix, None)

    def test_bad_options_fail_start_profile_before_capture(self):
        # Deferred start (start_step): the error still reaches the caller.
        result = self._start("host_tracer_level:x", start_step=5, num_steps=2)

        self.assertFalse(result.success)
        self.assertIn("host_tracer_level", result.message)
        self.start.assert_not_called()
        self.assertFalse(self.manager.profile_in_progress)

    def test_start_error_fails_start_profile(self):
        self.start.side_effect = ValueError("output directory 'a,b'")
        result = self._start("llama")

        self.assertFalse(result.success)
        self.assertIn("output directory", result.message)
        self.assertFalse(self.manager.profile_in_progress)

    def test_profile_by_stage_uses_a_directory_per_stage(self):
        result = self._start(
            "tpu_trace_mode:TRACE_COMPUTE", profile_by_stage=True, num_steps=1
        )
        self.assertTrue(result.success, result.message)
        self.manager._start_profile(ForwardMode.EXTEND)
        self.manager._stop_profile(ForwardMode.EXTEND)
        self.manager._start_profile(ForwardMode.DECODE)
        self.manager._stop_profile(ForwardMode.DECODE)

        dirs = [c.kwargs["output_dir"] for c in self.start.call_args_list]
        self.assertEqual(
            dirs,
            [
                self.tmpdir / "profile-1700000000.5-EXTEND-TP-1-DP-0",
                self.tmpdir / "profile-1700000000.5-DECODE-TP-1-DP-0",
            ],
        )
        for call in self.start.call_args_list:
            self.assertEqual(
                call.kwargs["options"].experimental_options["tpu_trace_mode"],
                "TRACE_COMPUTE",
            )
        self.assertEqual(self.stop.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
