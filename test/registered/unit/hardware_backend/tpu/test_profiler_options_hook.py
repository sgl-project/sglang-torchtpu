"""CPU tests for the TPU profiler-options HookRegistry hooks."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.managers.scheduler_components import (
    profiler_manager as scheduler_profiler,
)
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import profiler as tpu_profiler
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import profiler_options as hook_mod

_BASE = (
    "sglang.srt.managers.scheduler_components.profiler_manager.SchedulerProfilerManager"
)
_HOOK_SPECS = (
    (f"{_BASE}._profile", hook_mod.stash_profiler_kwargs, HookType.AROUND),
    (f"{_BASE}._init_profile", hook_mod.init_tpu_profile_options, HookType.AROUND),
    (
        f"{_BASE}._start_profile",
        hook_mod.start_tpu_profile_with_options,
        HookType.AROUND,
    ),
)


@contextlib.contextmanager
def _applied_profiler_options_hooks():
    """Apply the three AROUND hooks via HookRegistry and restore afterward."""
    originals = {
        name: getattr(SchedulerProfilerManager, name)
        for name in ("_profile", "_init_profile", "_start_profile")
    }
    saved_hooks = dict(HookRegistry._hooks)
    saved_patched = set(HookRegistry._patched)
    HookRegistry.reset()
    try:
        for target, fn, htype in _HOOK_SPECS:
            HookRegistry.register(target, fn, htype)
        HookRegistry.apply_hooks()
        yield
    finally:
        for name, fn in originals.items():
            setattr(SchedulerProfilerManager, name, fn)
        HookRegistry.reset()
        HookRegistry._hooks.update(saved_hooks)
        HookRegistry._patched.update(saved_patched)


class TestProfilerOptionsHook(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(_applied_profiler_options_hooks())
        stack.enter_context(get_parallel().override(tp_rank=0))
        stack.enter_context(envs.SGLANG_PROFILE_V2.override(False))
        stack.enter_context(tpu_envs.SGLANG_TPU_PROFILE_PYTHON_TRACER.override(False))
        stack.enter_context(tpu_envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(1))
        stack.enter_context(tpu_envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(1))
        stack.enter_context(tpu_envs.SGLANG_TPU_PROFILE_OPTIONS.override(""))
        self.tmp = stack.enter_context(tempfile.TemporaryDirectory())

    def _make_manager(self, *, get_forward_ct=lambda: 0) -> SchedulerProfilerManager:
        with patch.object(
            scheduler_profiler,
            "get_dp_tp_group",
            return_value=types.SimpleNamespace(cpu_group=None),
        ):
            return SchedulerProfilerManager(get_forward_ct=get_forward_ct)

    def test_hooks_registered_in_tpu_hooks(self):
        for spec in _HOOK_SPECS:
            with self.subTest(target=spec[0]):
                self.assertIn(spec, HOOKS)

    def test_forwards_options_and_preserves_active_options_on_rejection(self):
        captured_configs = []

        def _fake_config_cls(**kwargs):
            captured_configs.append(kwargs)
            return types.SimpleNamespace(**kwargs)

        self.enterContext(patch.object(scheduler_profiler, "_is_tpu", True))
        self.enterContext(
            patch.object(
                torch.profiler.ProfilerActivity,
                "TPU",
                torch.profiler.ProfilerActivity.PrivateUse1,
                create=True,
            )
        )
        self.enterContext(
            patch.object(
                tpu_profiler,
                "_tpu_profiler_config_cls",
                return_value=_fake_config_cls,
            )
        )
        profile_mock = self.enterContext(patch("torch.profiler.profile"))
        self.enterContext(patch("torch.profiler.tensorboard_trace_handler"))
        mgr = self._make_manager()

        def _send(profile_prefix=None, **kwargs):
            req = ProfileReq(
                req_type=ProfileReqType.START_PROFILE,
                output_dir=str(self.tmp),
                activities=kwargs.pop("activities", ["CPU", "TPU"]),
                profile_id="1700000000.5",
                profile_prefix=profile_prefix,
                **kwargs,
            )
            return mgr._profile(req)

        # Malformed options fail upfront before capture starts.
        bad = _send("host_tracer_level:x", start_step=5, num_steps=2)
        self.assertFalse(bad.success)
        profile_mock.assert_not_called()

        # Clean prefix + profiler_kwargs flows into TpuProfilerConfig with per-rank run_dir.
        ok = _send(
            "decode_counters",
            profiler_kwargs={
                "host_tracer_level": 3,
                "tpu_trace_mode": "TRACE_COMPUTE",
            },
        )
        self.assertTrue(ok.success, ok.message)
        self.assertEqual(len(captured_configs), 1)
        cfg = captured_configs[0]
        self.assertEqual(
            cfg["run_dir"],
            Path(self.tmp) / "decode_counters-1700000000.5-TP-0",
        )
        self.assertEqual(cfg["host_tracer_level"], 3)
        self.assertEqual(cfg["experimental_options"]["tpu_trace_mode"], "TRACE_COMPUTE")

        # While profile_in_progress is True, a rejected /start_profile must not
        # clobber the active capture's _tpu_profile_options.
        active_options = getattr(mgr, hook_mod._OPTIONS_ATTR)
        rejected = _send("other_run", profiler_kwargs={"host_tracer_level": 0})
        self.assertFalse(rejected.success)
        self.assertIs(getattr(mgr, hook_mod._OPTIONS_ATTR), active_options)

    def test_start_profile_logs_backend_error_and_disarms_counters(self):
        forward_ct = [0]
        self.enterContext(patch.object(scheduler_profiler, "_is_tpu", True))
        self.enterContext(
            patch.object(
                torch.profiler.ProfilerActivity,
                "TPU",
                torch.profiler.ProfilerActivity.PrivateUse1,
                create=True,
            )
        )
        self.enterContext(
            patch.object(
                tpu_profiler,
                "_tpu_profiler_config_cls",
                return_value=lambda **kwargs: types.SimpleNamespace(**kwargs),
            )
        )
        self.enterContext(patch("torch.profiler.tensorboard_trace_handler"))
        profile_mock = self.enterContext(patch("torch.profiler.profile"))
        profile_mock.return_value.start.side_effect = RuntimeError(
            "Parsing advanced_configuration failed. The following keys were not recognized: tpu_typo"
        )
        mgr = self._make_manager(get_forward_ct=lambda: forward_ct[0])
        req = ProfileReq(
            req_type=ProfileReqType.START_PROFILE,
            output_dir=str(self.tmp),
            num_steps=3,
            activities=["CPU", "TPU"],
            profile_id="1700000000.9",
            profiler_kwargs={"tpu_typo": 1},
        )
        with self.assertLogs("sglang_tpu.hooks.profiler_options", level="ERROR") as cm:
            out = mgr._profile(req)
        self.assertFalse(out.success)
        self.assertIsNone(mgr.tpu_profiler)
        self.assertFalse(mgr.profile_in_progress)
        self.assertIsNone(mgr.profiler_start_forward_ct)
        self.assertIsNone(mgr.profiler_target_forward_ct)
        self.assertFalse(mgr.profile_by_stage)
        self.assertTrue(any("tpu_typo" in line for line in cm.output))

        # Advancing past the failed num_steps target must not trigger _stop_profile().
        forward_ct[0] = 10
        with patch.object(mgr, "_stop_profile") as mock_stop:
            mgr._profile_batch_predicate(
                types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
            )
            mock_stop.assert_not_called()

    def test_v2_rejects_profiler_kwargs_explicitly(self):
        with (
            envs.SGLANG_PROFILE_V2.override(True),
            patch.object(
                scheduler_profiler,
                "get_dp_tp_group",
                return_value=types.SimpleNamespace(cpu_group=None),
            ),
            patch(
                "sglang.srt.observability.profile_utils.get_device",
                return_value=types.SimpleNamespace(gpu_id=0, base_gpu_id=0),
            ),
        ):
            mgr_v2 = SchedulerProfilerManager(get_forward_ct=lambda: 0)
            rejected = mgr_v2._profile(
                ProfileReq(
                    output_dir=str(self.tmp),
                    num_steps=1,
                    profile_by_stage=True,
                    activities=["CPU"],
                    profiler_kwargs={"host_tracer_level": 2},
                )
            )
        self.assertFalse(rejected.success)
        self.assertIn("SGLANG_PROFILE_V2", rejected.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
