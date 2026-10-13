"""Unit tests for launch-time (--profile-auto) arming in SchedulerProfilerManager."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakePS:
    tp_rank = dp_rank = pp_rank = moe_ep_rank = 0
    dp_size = pp_size = moe_ep_size = 1
    gpu_id = 0


def _batch(forward_mode: ForwardMode):
    return types.SimpleNamespace(forward_mode=forward_mode)


def _make_manager(output_dir, **auto_kwargs) -> SchedulerProfilerManager:
    """Arm a manager with SGLANG_TORCH_PROFILER_DIR pointed at output_dir.

    The env var is the only way the auto path learns where to write, since it
    takes no output_dir of its own.
    """
    with patch.dict("os.environ", {"SGLANG_TORCH_PROFILER_DIR": str(output_dir)}):
        return SchedulerProfilerManager(
            ps=_FakePS(),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
            **auto_kwargs,
        )


class TestAutoProfileArming(CustomTestCase):
    def test_not_armed_by_default(self):
        # Nothing may start profiling on its own; --profile-auto is the only
        # switch, and a plain launch must behave as it always has.
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(tmp)

        self.assertFalse(mgr.profile_by_stage)
        self.assertIsNone(mgr.pending_stages)
        self.assertIsNone(mgr.profiler_target_prefill_ct)

    def test_arming_configures_a_staged_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(
                tmp,
                auto_profile=True,
                auto_profile_stages=["prefill", "decode"],
                auto_profile_steps=3,
            )

            self.assertTrue(mgr.profile_by_stage)
            self.assertEqual(mgr.pending_stages, {"prefill", "decode"})
            self.assertEqual(mgr.profiler_target_prefill_ct, 3)
            self.assertEqual(mgr.profiler_target_decode_ct, 3)
            self.assertEqual(mgr.torch_profiler_output_dir, Path(tmp))
            # Arming configures only; the first batch of a stage opens it.
            self.assertFalse(mgr.profile_in_progress)

    def test_unselected_stage_is_never_captured(self):
        # Without the stage filter, profile_by_stage captures prefill and
        # decode both, and --profile-auto-stages would be decorative.
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(
                tmp,
                auto_profile=True,
                auto_profile_stages=["decode"],
                auto_profile_steps=1,
            )
            with patch.object(mgr, "_start_profile") as start:
                mgr._profile_batch_predicate(_batch(ForwardMode.EXTEND))

        start.assert_not_called()
        self.assertEqual(mgr.pending_stages, {"decode"})


class TestAutoProfileCompletion(CustomTestCase):
    """The done marker is the CI handshake: it must appear exactly once, after
    the last stage's flush has returned, and never for an API-armed profile."""

    def _run_stage(self, mgr, forward_mode, steps):
        # One batch more than the budget: the capture stops on the step that
        # crosses profiler_target_*_ct, not on the one that reaches it.
        for _ in range(steps + 2):
            mgr._profile_batch_predicate(_batch(forward_mode))

    def _armed_manager(self, tmp, **kwargs):
        mgr = _make_manager(tmp, **kwargs)

        # Stand in for the real capture: _stop_profile still runs for real, so
        # the completion path under test is the production one.
        def _fake_start(stage=None):
            mgr.profile_in_progress = True

        patcher = patch.object(mgr, "_start_profile", side_effect=_fake_start)
        patcher.start()
        self.addCleanup(patcher.stop)
        return mgr

    def test_marker_lands_only_after_the_last_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / ".xprof-done"
            mgr = self._armed_manager(
                tmp,
                auto_profile=True,
                auto_profile_stages=["prefill", "decode"],
                auto_profile_steps=2,
            )

            self._run_stage(mgr, ForwardMode.EXTEND, steps=2)
            self.assertEqual(mgr.pending_stages, {"decode"})
            self.assertFalse(marker.exists())

            self._run_stage(mgr, ForwardMode.DECODE, steps=2)
            self.assertEqual(mgr.pending_stages, set())
            self.assertTrue(marker.exists())

    def test_pd_engine_completes_on_the_only_stage_it_runs(self):
        # A PD engine armed with the default stages never sees the other side's
        # batches; if it waits for them, the marker never lands and CI hangs.
        for mode, forward_mode in (
            ("prefill", ForwardMode.EXTEND),
            ("decode", ForwardMode.DECODE),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                mgr = self._armed_manager(
                    tmp,
                    auto_profile=True,
                    auto_profile_stages=["prefill", "decode"],
                    auto_profile_steps=2,
                    disaggregation_mode=mode,
                )

                self._run_stage(mgr, forward_mode, steps=2)

                self.assertTrue((Path(tmp) / ".xprof-done").exists())

    def test_marker_waits_for_every_rank_and_only_primary_writes(self):
        # A non-primary rank used to write the marker as soon as its own flush
        # returned, while slower ranks were still flushing theirs.
        # Skipping the barrier on any rank would instead deadlock the rest.
        for tp_rank, expect_marker in ((0, True), (1, False)):
            with (
                self.subTest(tp_rank=tp_rank),
                tempfile.TemporaryDirectory() as tmp,
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.broadcast_object_list"),
                patch("torch.distributed.barrier") as barrier,
            ):
                mgr = self._armed_manager(
                    tmp,
                    auto_profile=True,
                    auto_profile_stages=["decode"],
                    auto_profile_steps=2,
                )
                mgr.ps.tp_rank = tp_rank

                self._run_stage(mgr, ForwardMode.DECODE, steps=2)

                barrier.assert_called_once()
                self.assertEqual((Path(tmp) / ".xprof-done").exists(), expect_marker)

    def test_every_rank_adopts_rank0_profile_id(self):
        # Each rank's own clock would give each rank its own id, scattering one
        # engine's traces across unrelated directory names.
        def _receive_rank0_id(object_list, group, group_src):
            self.assertEqual(group_src, 0)
            object_list[0] = "auto-from-rank0"

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.broadcast_object_list",
                side_effect=_receive_rank0_id,
            ) as broadcast,
        ):
            mgr = _make_manager(tmp, auto_profile=True, auto_profile_steps=1)

        broadcast.assert_called_once()
        self.assertEqual(mgr.profile_id, "auto-from-rank0")

    def test_no_marker_for_an_api_armed_profile(self):
        # /start_profile owns its own lifecycle; writing the marker there would
        # tell CI to collect a capture it never asked for.
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._armed_manager(tmp)
            mgr._init_profile(
                tmp, None, 2, None, None, None, True, "api", profile_stages=["prefill"]
            )

            self._run_stage(mgr, ForwardMode.EXTEND, steps=2)

            self.assertFalse((Path(tmp) / ".xprof-done").exists())


class TestTpuSessionLayout(CustomTestCase):
    def test_ranks_share_one_session_under_distinct_names(self):
        # Separate per-rank directories are separate XProf runs, one chip each;
        # a collective can only be read across chips from one session.
        layouts = []
        for tp_rank in (0, 1):
            with (
                self.subTest(tp_rank=tp_rank),
                tempfile.TemporaryDirectory() as tmp,
                patch(
                    "sglang.srt.managers.scheduler_components.profiler_manager._is_tpu",
                    True,
                ),
                patch(
                    "sglang.srt.hardware_backend.tpu.profiler.start_tpu_profile"
                ) as start,
            ):
                mgr = _make_manager(tmp, auto_profile=True, auto_profile_steps=1)
                mgr.ps.tp_rank = tp_rank
                mgr.profile_id = "auto-1"

                mgr._start_profile(ForwardMode.DECODE)

                kwargs = start.call_args.kwargs
                run = Path(tmp) / "profile-auto-1"
                layouts.append(
                    (
                        kwargs["session_dir"].relative_to(run),
                        kwargs["output_dir"].relative_to(run),
                        kwargs["file_suffix"],
                    )
                )

        self.assertEqual(
            layouts,
            [
                (Path("plugins/profile/DECODE"), Path(".staging/DECODE-TP-0"), "-TP-0"),
                (Path("plugins/profile/DECODE"), Path(".staging/DECODE-TP-1"), "-TP-1"),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
