"""Unit tests for the TPU launch-time profile (SGLANG_TPU_ENABLE_PROFILE_AUTO)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import profile_auto as tpu
from sglang_tpu.envs import envs as tpu_envs


@contextlib.contextmanager
def installed_patches():
    """Install the patches and restore the upstream methods afterwards."""
    originals = [
        (owner, name, getattr(owner, name)) for owner, name, _ in tpu._patch_targets()
    ]
    tpu.install()
    try:
        yield originals
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
        tpu._is_installed = False


def _batch(forward_mode: ForwardMode):
    return types.SimpleNamespace(forward_mode=forward_mode)


def _markers(output_dir) -> list[str]:
    return sorted(p.name for p in Path(output_dir).glob(f"{tpu.DONE_MARKER_PREFIX}*"))


class _ProfileAutoTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(installed_patches())
        # A single-rank, non-PD engine unless a test says otherwise.
        stack.enter_context(
            get_parallel().override(
                tp_rank=0,
                dp_size=1,
                pp_size=1,
                moe_ep_size=1,
                # SchedulerProfilerManager reads its group from get_dp_tp_group().
                tp_group=types.SimpleNamespace(cpu_group=None),
                attn_dp_enabled=False,
            )
        )
        stack.enter_context(get_disagg().override(disaggregation_mode="null"))
        self.tmp = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(envs.SGLANG_PROFILE_V2.override(False))
        stack.enter_context(
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_DECODE_STEPS_TO_SKIP.override(0)
        )
        stack.enter_context(
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_DECODE_KV_LEN_THRESHOLD.override(-1)
        )
        stack.enter_context(
            patch.dict("os.environ", {"SGLANG_TORCH_PROFILER_DIR": self.tmp})
        )

    def _make_manager(self, *, enabled=True, steps=2) -> SchedulerProfilerManager:
        with (
            tpu_envs.SGLANG_TPU_ENABLE_PROFILE_AUTO.override(enabled),
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_STEPS.override(steps),
        ):
            return SchedulerProfilerManager(get_forward_ct=lambda: 0)

    def _capturing_manager(self, **kwargs) -> SchedulerProfilerManager:
        """A manager whose captures are faked; _stop_profile still runs for real."""
        mgr = self._make_manager(**kwargs)

        def _fake_start(stage=None):
            mgr.profile_in_progress = True

        patcher = patch.object(mgr, "_start_profile", side_effect=_fake_start)
        patcher.start()
        self.addCleanup(patcher.stop)
        return mgr

    def _run_stage(self, mgr, forward_mode, steps=2):
        # Two batches past the budget: the capture stops on the batch that
        # crosses the target, not on the one that reaches it.
        for _ in range(steps + 2):
            mgr._profile_batch_predicate(_batch(forward_mode))


class TestArming(_ProfileAutoTestCase):
    def test_not_armed_by_default(self):
        # Nothing may start profiling on its own; a plain launch behaves as
        # upstream.
        mgr = self._make_manager(enabled=False)

        self.assertFalse(mgr.profile_by_stage)
        self.assertIsNone(tpu._pending_stages(mgr))
        self.assertIsNone(mgr.profiler_target_prefill_ct)

    def test_arming_configures_an_upstream_staged_profile(self):
        mgr = self._make_manager(steps=3)

        self.assertTrue(mgr.profile_by_stage)
        self.assertEqual(tpu._pending_stages(mgr), {"prefill", "decode"})
        self.assertEqual(mgr.profiler_target_prefill_ct, 3)
        self.assertEqual(mgr.profiler_target_decode_ct, 3)
        self.assertEqual(mgr.torch_profiler_output_dir, Path(self.tmp))
        # Arming only configures; the first batch of a stage opens the capture.
        self.assertFalse(mgr.profile_in_progress)

    def test_pd_engine_arms_only_its_own_stage(self):
        for mode in ("prefill", "decode"):
            with (
                self.subTest(mode=mode),
                get_disagg().override(disaggregation_mode=mode),
            ):
                mgr = self._make_manager()
                self.assertEqual(tpu._pending_stages(mgr), {mode})

    def test_zero_steps_is_rejected(self):
        # It would arm a profile that captures nothing.
        with self.assertRaisesRegex(ValueError, "SGLANG_TPU_PROFILE_AUTO_STEPS"):
            self._make_manager(steps=0)

    def test_not_armed_under_profile_v2(self):
        with (
            envs.SGLANG_PROFILE_V2.override(True),
            patch(
                "sglang.srt.managers.scheduler_components.profiler_manager.ProfileManager"
            ),
        ):
            mgr = self._make_manager()

        self.assertIsNone(tpu._pending_stages(mgr))

    def test_every_rank_adopts_rank0_profile_id(self):
        # Each rank's own clock would scatter one engine's traces across
        # unrelated directory names.
        def _receive_rank0_id(object_list, group, group_src):
            self.assertEqual(group_src, 0)
            object_list[0] = "auto-from-rank0"

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.broadcast_object_list",
                side_effect=_receive_rank0_id,
            ) as broadcast,
        ):
            mgr = self._make_manager()

        broadcast.assert_called_once()
        self.assertEqual(mgr.profile_id, "auto-from-rank0")


class TestCompletion(_ProfileAutoTestCase):
    """The marker is the collector's handshake: one per rank, written after
    that rank's last stage is flushed, and never for an API-armed profile."""

    def test_marker_lands_only_after_the_last_stage(self):
        mgr = self._capturing_manager()

        self._run_stage(mgr, ForwardMode.EXTEND)
        self.assertEqual(tpu._pending_stages(mgr), {"decode"})
        self.assertEqual(_markers(self.tmp), [])

        self._run_stage(mgr, ForwardMode.DECODE)
        self.assertIsNone(tpu._pending_stages(mgr))
        self.assertEqual(_markers(self.tmp), [".xprof-done-TP-0"])

    def test_pd_engine_completes_on_the_only_stage_it_runs(self):
        # Upstream waits for both stages; a PD engine never runs the other
        # side's, so without the role the marker never lands.
        for mode, forward_mode in (
            ("prefill", ForwardMode.EXTEND),
            ("decode", ForwardMode.DECODE),
        ):
            with (
                self.subTest(mode=mode),
                get_disagg().override(disaggregation_mode=mode),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict("os.environ", {"SGLANG_TORCH_PROFILER_DIR": tmp}),
            ):
                mgr = self._capturing_manager()
                self._run_stage(mgr, forward_mode)

                self.assertEqual(_markers(tmp), [".xprof-done-TP-0"])

    def test_decode_engine_ignores_prefill_batches(self):
        with get_disagg().override(disaggregation_mode="decode"):
            mgr = self._capturing_manager()

        mgr._profile_batch_predicate(_batch(ForwardMode.EXTEND))

        mgr._start_profile.assert_not_called()

    def test_prefill_batch_does_not_cut_a_decode_capture_short(self):
        # Upstream stops whatever is running once prefill is past its budget,
        # which would end the decode capture after one step.
        mgr = self._capturing_manager(steps=3)
        self._run_stage(mgr, ForwardMode.EXTEND, steps=3)

        mgr._profile_batch_predicate(_batch(ForwardMode.DECODE))
        mgr._profile_batch_predicate(_batch(ForwardMode.EXTEND))

        self.assertTrue(mgr.profile_in_progress)
        self.assertEqual(tpu._pending_stages(mgr), {"decode"})

    def test_each_rank_writes_its_own_marker_with_the_rank_count(self):
        # The collector waits for one marker per rank, so it needs to know
        # how many there are.
        for tp_rank in (0, 1):
            with (
                self.subTest(tp_rank=tp_rank),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict("os.environ", {"SGLANG_TORCH_PROFILER_DIR": tmp}),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.broadcast_object_list"),
                patch("torch.distributed.get_world_size", return_value=8),
                patch.object(
                    SchedulerProfilerManager,
                    "_rank_parts",
                    return_value=[f"TP-{tp_rank}"],
                ),
                get_disagg().override(disaggregation_mode="decode"),
            ):
                mgr = self._capturing_manager()
                self._run_stage(mgr, ForwardMode.DECODE)

                marker = Path(tmp) / f".xprof-done-TP-{tp_rank}"
                self.assertEqual(_markers(tmp), [marker.name])
                self.assertEqual(marker.read_text(), "8\n")

    def test_no_marker_for_an_api_armed_profile(self):
        # /start_profile owns its own lifecycle; a marker there would tell the
        # collector to copy a capture it never asked for.
        mgr = self._capturing_manager(enabled=False)
        mgr._init_profile(self.tmp, None, 2, None, None, None, True, "api")

        self._run_stage(mgr, ForwardMode.EXTEND)
        self._run_stage(mgr, ForwardMode.DECODE)

        self.assertEqual(_markers(self.tmp), [])


class TestDecodeGatesAndBatchStatsJsonl(_ProfileAutoTestCase):
    def test_invalid_decode_gate_envs_rejected(self):
        with (
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_DECODE_STEPS_TO_SKIP.override(-1),
            self.assertRaisesRegex(
                ValueError, "SGLANG_TPU_PROFILE_AUTO_DECODE_STEPS_TO_SKIP"
            ),
        ):
            self._make_manager()
        with (
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_DECODE_KV_LEN_THRESHOLD.override(-2),
            self.assertRaisesRegex(
                ValueError, "SGLANG_TPU_PROFILE_AUTO_DECODE_KV_LEN_THRESHOLD"
            ),
        ):
            self._make_manager()

    def test_decode_skip_and_kv_threshold_write_matching_jsonl_and_batch_id(self):
        from sglang_tpu import profile_batch_stats

        forward_ct = [10]
        with (
            tpu_envs.SGLANG_TPU_ENABLE_PROFILE_AUTO.override(True),
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_STEPS.override(2),
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_DECODE_STEPS_TO_SKIP.override(2),
            tpu_envs.SGLANG_TPU_PROFILE_AUTO_DECODE_KV_LEN_THRESHOLD.override(16),
            get_disagg().override(disaggregation_mode="decode"),
        ):
            mgr = SchedulerProfilerManager(get_forward_ct=lambda: forward_ct[0])

            def _fake_start(stage=None):
                mgr.profile_in_progress = True

            self.enterContext(
                patch.object(mgr, "_start_profile", side_effect=_fake_start)
            )

            def _step(kv_lens):
                forward_ct[0] += 1
                batch = types.SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=len(kv_lens),
                    seq_lens_cpu=torch.tensor(kv_lens),
                    spec_info=None,
                )
                mgr._profile_batch_predicate(batch)

            # First 2 decode steps skipped by DECODE_STEPS_TO_SKIP (forward_ct 11, 12).
            _step([32, 32])
            self.assertFalse(mgr.profile_in_progress)
            _step([32, 32])
            self.assertFalse(mgr.profile_in_progress)

            # Step 3 skipped because min KV len 8 < threshold 16 (forward_ct 13).
            _step([8, 24])
            self.assertFalse(mgr.profile_in_progress)
            self.assertIsNone(profile_batch_stats.consume_active_batch_id())

            # Step 4 opens decode capture (forward_ct 14) -> JSONL line 1 + batch_id=14.
            _step([16, 32])
            self.assertTrue(mgr.profile_in_progress)
            self.assertEqual(profile_batch_stats.consume_active_batch_id(), 14)

            # Step 5 is step 2/2 of decode capture (forward_ct 15) -> JSONL line 2 + batch_id=15.
            _step([17, 33])
            self.assertTrue(mgr.profile_in_progress)
            self.assertEqual(profile_batch_stats.consume_active_batch_id(), 15)

            # Step 6 closes capture (forward_ct 16) -> no extra JSONL line, active batch_id cleared.
            _step([18, 34])
            self.assertFalse(mgr.profile_in_progress)
            self.assertIsNone(profile_batch_stats.consume_active_batch_id())

        stats_path = Path(self.tmp) / "batch_composition_stats_decode_TP-0.json"
        self.assertTrue(stats_path.exists())
        rows = [
            json.loads(line)
            for line in stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["batch_id"] for r in rows], [14, 15])
        self.assertEqual([r["min_kv_len"] for r in rows], [16, 17])
        self.assertEqual([r["phase"] for r in rows], ["DECODE_ONLY", "DECODE_ONLY"])


class TestInstall(CustomTestCase):
    def test_install_wraps_every_target_once_and_restores(self):
        with installed_patches() as originals:
            tpu.install()
            for owner, name, original in originals:
                with self.subTest(target=f"{owner.__name__}.{name}"):
                    self.assertIs(getattr(owner, name).__wrapped__, original)
        for owner, name, original in originals:
            self.assertIs(getattr(owner, name), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
