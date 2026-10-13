"""Unit tests for the TPU xprof profiler wiring."""

import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.profiler import (
    _tpu_profile_options,
    start_tpu_profile,
    stop_tpu_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Pure-logic tests. torch_tpu is faked via sys.modules, so this runs on the
# CPU CI runner (no TPU required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclasses.dataclass
class _BaseProfileOptions:
    """Mirrors torch_tpu's _impl.ProfileOptions: three levels, all unset."""

    host_tracer_level: int | None = None
    device_tracer_level: int | None = None
    python_tracer_level: int | None = None


def _fake_torch_tpu(profile_options_cls=_BaseProfileOptions, calls=None):
    """Fake torch_tpu's private profiler layout.

    start_trace/stop_trace live in _impl; the package root exports only the
    torch.profiler-facing names. A fake that puts them at the root would let a
    regression to the wrong import path pass here.
    """
    impl = types.ModuleType("torch_tpu._internal.profiler._impl")
    if profile_options_cls is not None:
        impl.ProfileOptions = profile_options_cls
    if calls is not None:
        impl.start_trace = lambda d, o=None: calls.append(("start", d, o))
        impl.stop_trace = lambda: calls.append(("stop",))
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


class TestTpuProfileOptions(CustomTestCase):
    def _options(
        self,
        *,
        num_chips,
        num_sparse_cores,
        with_cpu=True,
        profile_options_cls=_BaseProfileOptions,
    ):
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(num_chips),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(num_sparse_cores),
            patch.dict(sys.modules, _fake_torch_tpu(profile_options_cls)),
        ):
            return _tpu_profile_options(with_cpu=with_cpu)

    def test_limits_enabled(self):
        options = self._options(num_chips=1, num_sparse_cores=2)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_cores_to_trace, 2)
        # Tile cap only kicks in at a single SparseCore; here 2 leaves it off.
        self.assertIsNone(options.tpu_num_sparse_core_tiles_to_trace)
        # Levels survive the narrowing subclass.
        self.assertEqual(options.host_tracer_level, 2)
        self.assertEqual(options.device_tracer_level, 1)

    def test_single_sparse_core_narrows_to_one_tile(self):
        options = self._options(num_chips=1, num_sparse_cores=1)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_cores_to_trace, 1)
        self.assertEqual(options.tpu_num_sparse_core_tiles_to_trace, 1)

    def test_one_limit_disabled(self):
        options = self._options(num_chips=4, num_sparse_cores=0)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 4)
        self.assertIsNone(options.tpu_num_sparse_cores_to_trace)
        self.assertIsNone(options.tpu_num_sparse_core_tiles_to_trace)

    def test_both_limits_disabled_traces_everything(self):
        # No narrowing fields at all, so start_trace sends the levels alone.
        options = self._options(num_chips=0, num_sparse_cores=-1)

        self.assertEqual(
            dataclasses.asdict(options),
            {
                "host_tracer_level": 2,
                "device_tracer_level": 1,
                "python_tracer_level": 0,
            },
        )

    def test_host_tracer_off_without_cpu(self):
        # with_cpu was a no-op on the first cut of this PR: the levels went
        # unset, and the backend default fills the host plane either way.
        options = self._options(num_chips=1, num_sparse_cores=1, with_cpu=False)

        self.assertEqual(options.host_tracer_level, 0)
        self.assertEqual(options.device_tracer_level, 1)

    def test_python_tracer_off_by_default(self):
        # torch_tpu's profile() sets python_tracer_level=1 whenever CPU
        # activity is on, at 4.9 s of host busy time against 0.36 s.
        options = self._options(num_chips=1, num_sparse_cores=1)

        self.assertEqual(options.python_tracer_level, 0)

    def test_python_tracer_opt_in(self):
        with envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(True):
            options = self._options(num_chips=1, num_sparse_cores=1)

        self.assertEqual(options.python_tracer_level, 1)

    def test_non_dataclass_profile_options_degrades(self):
        class _NotADataclass:
            pass

        # A torch_tpu that reshapes ProfileOptions must lose the narrowing, not
        # break capture altogether.
        self.assertIsNone(
            self._options(
                num_chips=1, num_sparse_cores=1, profile_options_cls=_NotADataclass
            )
        )

    def test_missing_profile_options_degrades(self):
        self.assertIsNone(
            self._options(num_chips=1, num_sparse_cores=1, profile_options_cls=None)
        )


class TestStartStopUsesLibtpuTrace(CustomTestCase):
    """The capture must reach _impl.start_trace with its options.

    torch_tpu renamed profiler_api -> _api between the 0817 and 0913 pins,
    which is what broke the import this PR started from. _impl was untouched,
    so these pin the capture to it -- and to the options going along with it,
    since a capture with the wrong ones still succeeds.
    """

    def test_start_calls_start_trace_with_output_dir(self):
        calls = []
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(0),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(0),
            patch.dict(sys.modules, _fake_torch_tpu(calls=calls)),
        ):
            trace = start_tpu_profile(output_dir=Path("/tmp/xprof-unit-test"))

        self.assertEqual([c[0] for c in calls], ["start"])
        self.assertEqual(calls[0][1], "/tmp/xprof-unit-test")
        self.assertEqual(trace.output_dir, Path("/tmp/xprof-unit-test"))

    def test_start_forwards_narrowing_options(self):
        calls = []
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(1),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(1),
            patch.dict(sys.modules, _fake_torch_tpu(calls=calls)),
        ):
            start_tpu_profile(output_dir=Path("/tmp/xprof-unit-test"))

        options = calls[0][2]
        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_core_tiles_to_trace, 1)

    def test_stop_calls_stop_trace(self):
        calls = []
        with patch.dict(sys.modules, _fake_torch_tpu(calls=calls)):
            stop_tpu_profile(None)

        self.assertEqual([c[0] for c in calls], ["stop"])


# A minimal valid XSpace: just the hostname field.
_XSPACE = b"\x22\x0adecode-0-0"


def _fake_libtpu_writing(host="decode-0-0", payload=_XSPACE):
    """A fake torch_tpu whose stop_trace writes the file as libtpu lays it out:
    <dir>/plugins/profile/<timestamp>/<host>.xplane.pb."""
    modules = _fake_torch_tpu()
    impl = modules["torch_tpu._internal.profiler._impl"]
    state = {}

    def _start(output_dir, options=None):
        state["dir"] = Path(output_dir)

    def _stop():
        run = state["dir"] / "plugins" / "profile" / "2026_09_25_20_01_51"
        run.mkdir(parents=True)
        (run / f"{host}.xplane.pb").write_bytes(payload)

    impl.start_trace = _start
    impl.stop_trace = _stop
    return modules


class TestPublishIntoSharedSession(CustomTestCase):
    """Each process traces only its own chip; a combined view needs every
    rank's file in one XProf session folder, under a name no co-hosted rank
    shares."""

    def _capture(self, root, rank_tag, **fake_kwargs):
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(0),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(0),
            patch.dict(sys.modules, _fake_libtpu_writing(**fake_kwargs)),
        ):
            trace = start_tpu_profile(
                output_dir=root / ".staging" / f"DECODE-{rank_tag}",
                session_dir=root / "plugins" / "profile" / "DECODE",
                file_suffix=f"-{rank_tag}",
            )
            return stop_tpu_profile(trace)

    def test_co_hosted_ranks_land_side_by_side(self):
        # Both ranks' files are named after the same host by libtpu.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            published = [self._capture(root, f"TP-{k}") for k in (0, 1)]

            session = root / "plugins" / "profile" / "DECODE"
            self.assertEqual(
                published,
                [
                    session / "decode-0-0-TP-0.xplane.pb",
                    session / "decode-0-0-TP-1.xplane.pb",
                ],
            )
            self.assertEqual(
                sorted(p.name for p in session.iterdir()),
                ["decode-0-0-TP-0.xplane.pb", "decode-0-0-TP-1.xplane.pb"],
            )
            # XProf lists any leftover plugins/profile folder as another run.
            self.assertFalse((root / ".staging").exists())

    def test_published_file_is_rebased(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "sglang.srt.hardware_backend.tpu.profiler.rebase_to_wall_clock",
                return_value=(b"rebased", 1),
            ) as rebase,
        ):
            published = self._capture(Path(tmp), "TP-0")

            rebase.assert_called_once_with(_XSPACE)
            self.assertEqual(published.read_bytes(), b"rebased")

    def test_failed_move_stays_in_staging(self):
        # A failed publish must not lose the capture or take down the scheduler.
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "sglang.srt.hardware_backend.tpu.profiler.os.replace",
                side_effect=PermissionError("permission denied"),
            ),
        ):
            root = Path(tmp)
            published = self._capture(root, "TP-0")

            self.assertIsNone(published)
            staged = list((root / ".staging").glob("*/plugins/profile/*/*.xplane.pb"))
            self.assertEqual(len(staged), 1)
            session = root / "plugins" / "profile" / "DECODE"
            self.assertEqual(list(session.glob("*.xplane.pb")), [])

    def test_unreadable_trace_stays_in_staging(self):
        # A trace the rebase cannot parse is kept, not published half-rewritten.
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "sglang.srt.hardware_backend.tpu.profiler.rebase_to_wall_clock",
                side_effect=ValueError("truncated protobuf message"),
            ),
        ):
            root = Path(tmp)
            published = self._capture(root, "TP-0")

            self.assertIsNone(published)
            staged = list((root / ".staging").glob("*/plugins/profile/*/*.xplane.pb"))
            self.assertEqual(len(staged), 1)
            self.assertFalse((root / "plugins" / "profile" / "DECODE").exists())

    def test_without_session_dir_the_trace_stays_put(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(sys.modules, _fake_libtpu_writing()),
        ):
            trace = start_tpu_profile(output_dir=Path(tmp) / "capture")

            self.assertIsNone(stop_tpu_profile(trace))
            self.assertEqual(
                len(list(Path(tmp).glob("capture/plugins/profile/*/*.xplane.pb"))), 1
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
