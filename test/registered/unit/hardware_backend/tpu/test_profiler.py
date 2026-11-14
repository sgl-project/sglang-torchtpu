"""Unit tests for the TPU xprof profiler wiring."""

import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq
from sglang.srt.managers.scheduler_components import (
    profiler_manager as scheduler_profiler,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import scheduler_profiler as tpu_scheduler_profiler
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.profiler import (
    _tpu_profile_options,
    start_tpu_profile,
    stop_tpu_profile,
)

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
            tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(num_chips),
            tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(num_sparse_cores),
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
        with tpu_envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(True):
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
            tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(0),
            tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(0),
            patch.dict(sys.modules, _fake_torch_tpu(calls=calls)),
        ):
            trace = start_tpu_profile(output_dir=Path("/tmp/xprof-unit-test"))

        self.assertEqual([c[0] for c in calls], ["start"])
        self.assertEqual(calls[0][1], "/tmp/xprof-unit-test")
        self.assertEqual(trace.output_dir, Path("/tmp/xprof-unit-test"))

    def test_start_forwards_narrowing_options(self):
        calls = []
        with (
            tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(1),
            tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(1),
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


class TestContinuousTpuProfile(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        self.parallel = types.SimpleNamespace(
            tp_rank=0,
            dp_rank=0,
            pp_rank=0,
            moe_ep_rank=0,
            num_dp_ranks=2,
            pp_size=1,
            moe_ep_size=4,
        )
        self.disagg = types.SimpleNamespace(disaggregation_mode="null")
        patches = (
            envs.SGLANG_PROFILE_V2.override(False),
            envs.SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS.override(0),
            patch.object(
                scheduler_profiler,
                "get_dp_tp_group",
                return_value=types.SimpleNamespace(cpu_group=None),
            ),
            patch.object(scheduler_profiler, "_is_tpu", True),
            patch.object(
                scheduler_profiler, "get_parallel", return_value=self.parallel
            ),
            patch.object(
                tpu_scheduler_profiler, "get_parallel", return_value=self.parallel
            ),
            patch.object(
                tpu_scheduler_profiler, "get_disagg", return_value=self.disagg
            ),
            patch.object(
                scheduler_profiler.current_platform,
                "synchronize",
                side_effect=lambda: self.calls.append("drain"),
            ),
            patch("sglang_tpu.profiler.start_tpu_profile", side_effect=self._start),
            patch(
                "sglang_tpu.profiler.stop_tpu_profile",
                side_effect=lambda _: self.calls.append("export"),
            ),
        )
        for context in patches:
            self.enterContext(context)
        self.manager = tpu_scheduler_profiler.TpuSchedulerProfilerManager(
            get_forward_ct=lambda: 0
        )

    def _start(self, *, output_dir, with_cpu):
        self.calls.append(("start", output_dir.name))
        return object()

    def _arm(self, steps=2):
        result = self.manager._profile(
            ProfileReq(
                output_dir=self.tmp.name,
                num_steps=steps,
                activities=["CPU", "TPU"],
                profile_by_stage=True,
                profile_id="test",
                profile_prefix="bs-64-il-8192",
            )
        )
        self.assertTrue(result.success)

    def _batch(self, mode, size=64):
        self.manager._profile_batch_predicate(
            types.SimpleNamespace(forward_mode=mode, batch_size=lambda: size)
        )

    def test_prefill_and_decode_share_one_capture_and_drain_before_export(self):
        self._arm()
        for _ in range(4):
            self._batch(ForwardMode.EXTEND)
        self._batch(ForwardMode.DECODE)
        self._batch(ForwardMode.DECODE)
        self.assertEqual(
            self.calls, [("start", "bs-64-il-8192-test-MIXED-TP-0-DP-0-EP-0")]
        )
        self._batch(ForwardMode.DECODE)
        self.assertEqual(self.calls[1:], ["drain", "export"])
        self._batch(ForwardMode.DECODE)
        self.manager._stop_profile()
        self.assertEqual(len(self.calls), 3)

    def test_nonzero_dp_rank_does_not_start_or_export(self):
        self.parallel.dp_rank = 1
        self._arm()
        self._batch(ForwardMode.EXTEND)
        for _ in range(3):
            self._batch(ForwardMode.DECODE)
        self.assertFalse(self.manager.profile_in_progress)
        self.assertEqual(self.calls, [])

    def test_short_capture_manual_stop_and_rearm(self):
        self._arm()
        self._batch(ForwardMode.EXTEND)
        self.manager._stop_profile()
        self.assertEqual(self.calls[1:], ["drain", "export"])
        self._batch(ForwardMode.DECODE)
        self.assertEqual(len(self.calls), 3)
        self._arm()
        self._batch(ForwardMode.EXTEND)
        self.assertEqual(len(self.calls), 4)

    def test_stop_before_first_batch_disarms(self):
        self._arm()
        self.assertTrue(self.manager._stop_profile().success)
        self._batch(ForwardMode.EXTEND)
        self.assertEqual(self.calls, [])

    def test_decode_admission_and_idle_do_not_restart_prefill_capture(self):
        self._arm()
        self._batch(ForwardMode.IDLE)
        self._batch(ForwardMode.PREBUILT)
        self.assertEqual(self.calls, [])
        self._batch(ForwardMode.EXTEND)
        with envs.SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS.override(64):
            self._batch(ForwardMode.DECODE, size=32)
        self.assertEqual(self.manager.profiler_decode_ct, 0)
        self._batch(ForwardMode.DECODE)
        self.assertEqual(len(self.calls), 1)

    def test_pd_prefill_retains_single_stage_bound(self):
        self.disagg.disaggregation_mode = "prefill"
        self._arm()
        for _ in range(3):
            self._batch(ForwardMode.EXTEND)
        self.assertIn("-EXTEND-", self.calls[0][1])
        self.assertEqual(self.calls[1:], ["drain", "export"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
