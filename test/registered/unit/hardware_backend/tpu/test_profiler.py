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
    _find_xplane,
    _has_tpu_device_events,
    start_tpu_profile,
    stop_tpu_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Pure-logic tests. torch_tpu is faked via sys.modules and ProfilerActivity.TPU
# is aliased to PrivateUse1 the way torch_tpu does it, so this runs on the CPU
# CI runner (no TPU required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ProfilerActivity = torch.profiler.ProfilerActivity
_XPLANE_ENV_VARS = ("TPU_PROFILER_OUTPUT_DIR", "TORCH_TPU_PROFILER_OUTPUT_DIR")


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


def _xplane_env(tmpdir, **env_vars):
    """os.environ with TMPDIR set and the plugin output-dir env vars cleared."""
    env = {k: v for k, v in os.environ.items() if k not in _XPLANE_ENV_VARS}
    env["TMPDIR"] = str(tmpdir)
    env.update(env_vars)
    return patch.dict(os.environ, env, clear=True)


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


class TestStartUsesNativeTorchProfiler(CustomTestCase):
    """start_tpu_profile builds one torch.profiler with the TPU activity."""

    def _start(self, *, with_cpu, python_tracer=False):
        calls = []
        output_dir = Path(self.enterContext(tempfile.TemporaryDirectory())) / "r"
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

    def test_python_tracer_on_tpu_only(self):
        _, profile, _, _, _ = self._start(with_cpu=False, python_tracer=True)

        self.assertEqual(
            profile.call_args.kwargs["activities"], [_ProfilerActivity.PrivateUse1]
        )
        self.assertTrue(profile.call_args.kwargs["with_stack"])

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
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        profiler = MagicMock()
        trace = tpu_profiler._TpuTrace(
            output_dir=tmpdir, profiler=profiler, started_at=time.time()
        )
        with (
            patch.dict(sys.modules, _fake_torch_tpu(calls)),
            _xplane_env(tmpdir),
            self.assertLogs(tpu_profiler.logger, "ERROR"),
        ):
            stop_tpu_profile(trace)

        profiler.stop.assert_called_once_with()
        self.assertEqual(calls, [])

    def test_stop_logs_xplane_path(self):
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        started_at = time.time() - 1
        xplane = tmpdir / "plugins/profile/2026_01_01_00_00_00/host.xplane.pb"
        xplane.parent.mkdir(parents=True)

        def write_xplane():
            xplane.write_bytes(_xspace(_plane("/device:TPU:0", 2)))

        profiler = MagicMock()
        profiler.stop.side_effect = write_xplane
        trace = tpu_profiler._TpuTrace(
            output_dir=tmpdir / "rank", profiler=profiler, started_at=started_at
        )
        with _xplane_env(tmpdir), self.assertLogs(tpu_profiler.logger, "INFO") as logs:
            stop_tpu_profile(trace)

        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "INFO")
        self.assertIn(str(xplane), logs.output[0])

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

    def test_same_second_overwrite_warns(self):
        # profile_by_stage: EXTEND and DECODE stop in the same second, so the
        # plugin writes both to the same path and the second one wins.
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        xplane = tmpdir / "plugins/profile/2026_01_01_00_00_00/host.xplane.pb"
        xplane.parent.mkdir(parents=True)

        def write_xplane():
            xplane.write_bytes(_xspace(_plane("/device:TPU:0", 2)))

        def capture():
            profiler = MagicMock()
            profiler.stop.side_effect = write_xplane
            stop_tpu_profile(
                tpu_profiler._TpuTrace(
                    output_dir=tmpdir, profiler=profiler, started_at=time.time() - 1
                )
            )

        with (
            patch.object(tpu_profiler, "_last_xplane", None),
            _xplane_env(tmpdir),
            self.assertLogs(tpu_profiler.logger, "INFO") as logs,
        ):
            capture()
            capture()

        levels = [r.levelname for r in logs.records]
        self.assertEqual(levels, ["INFO", "WARNING", "INFO"])
        self.assertIn("previous capture's xplane", logs.output[1])


class TestRealTorchProfilerOnCpu(CustomTestCase):
    """End to end through the real torch.profiler on a CPU-only torch.

    Without torch_tpu, PrivateUse1 is dropped, so this checks the API wiring and
    the Kineto trace location, not TPU events.
    """

    def test_cpu_trace_lands_in_output_dir(self):
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        output_dir = tmpdir / "profile-id-TP-0"
        with (
            patch.object(
                _ProfilerActivity, "TPU", _ProfilerActivity.PrivateUse1, create=True
            ),
            _xplane_env(tmpdir / "xplane-base"),
            self.assertLogs(tpu_profiler.logger, "ERROR") as logs,
        ):
            trace = start_tpu_profile(output_dir=output_dir, with_cpu=True)
            torch.randn(8, 8) @ torch.randn(8, 8)
            stop_tpu_profile(trace)

        self.assertEqual(len(list(output_dir.glob("*.pt.trace.json.gz"))), 1)
        # No plugin, so no xplane: the empty-trace check must say so.
        self.assertIn("no .xplane.pb", logs.output[0])


class TestFindXplane(CustomTestCase):
    def setUp(self):
        self.tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _write(self, base, ts, mtime=None):
        path = Path(base) / "plugins" / "profile" / ts / "host.xplane.pb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_finds_xplane_under_tmpdir(self):
        started_at = time.time() - 10
        xplane = self._write(self.tmpdir, "2026_01_01_00_00_01")
        with _xplane_env(self.tmpdir):
            self.assertEqual(_find_xplane(started_at), xplane)

    def test_defaults_to_tmp_without_tmpdir(self):
        with _xplane_env(self.tmpdir):
            del os.environ["TMPDIR"]
            self.assertEqual(tpu_profiler._xplane_base_dirs(), [Path("/tmp")])

    def test_ignores_xplanes_from_before_the_capture(self):
        started_at = time.time()
        self._write(self.tmpdir, "2020_01_01_00_00_00", mtime=started_at - 100)
        with _xplane_env(self.tmpdir):
            self.assertIsNone(_find_xplane(started_at))

    def test_picks_newest(self):
        started_at = time.time() - 100
        self._write(self.tmpdir, "a", mtime=started_at + 1)
        newest = self._write(self.tmpdir, "b", mtime=started_at + 2)
        with _xplane_env(self.tmpdir):
            self.assertEqual(_find_xplane(started_at), newest)

    def test_honors_output_dir_env_vars(self):
        started_at = time.time() - 10
        for env_var in _XPLANE_ENV_VARS:
            with self.subTest(env_var=env_var):
                base = self.tmpdir / env_var
                xplane = self._write(base, "2026_01_01_00_00_02")
                with _xplane_env(self.tmpdir / "unused-tmpdir", **{env_var: str(base)}):
                    self.assertEqual(_find_xplane(started_at), xplane)

    def test_missing_xplane_logs_error(self):
        with (
            _xplane_env(self.tmpdir),
            self.assertLogs(tpu_profiler.logger, "ERROR") as logs,
        ):
            tpu_profiler._check_xplane(time.time())
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
            _xplane_env(tmpdir),
            self.assertLogs(tpu_profiler.logger, "ERROR") as logs,
        ):
            tpu_profiler._check_xplane(time.time() - 10)
        self.assertIn("no /device:TPU plane with events", logs.output[0])

    def test_check_xplane_survives_unparsable_file(self):
        tmpdir = self.path.parent
        xplane = tmpdir / "plugins/profile/ts/host.xplane.pb"
        xplane.parent.mkdir(parents=True)
        xplane.write_bytes(b"\x0a\xff\xff")
        with (
            _xplane_env(tmpdir),
            self.assertLogs(tpu_profiler.logger, "INFO") as logs,
        ):
            tpu_profiler._check_xplane(time.time() - 10)
        self.assertEqual(
            [r.levelname for r in logs.records], ["WARNING", "INFO"], logs.output
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
