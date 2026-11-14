"""A representative xprof capture without an export between serving stages."""

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_disagg, get_parallel


class TpuSchedulerProfilerManager(SchedulerProfilerManager):
    def __post_init__(self) -> None:
        if envs.SGLANG_PROFILE_V2.get():
            raise ValueError("Continuous TPU profiling requires SGLANG_PROFILE_V2=0")
        super().__post_init__()

    def _start_profile(self, stage=None):
        parallel = get_parallel()
        if "TPU" in self.profiler_activities and any(
            (
                parallel.tp_rank,
                parallel.dp_rank,
                parallel.pp_rank,
                parallel.moe_ep_rank,
            )
        ):
            # Keep the control lifecycle on every rank, including auto-profile
            # completion markers, without starting any non-primary collectors.
            self.profile_in_progress = True
            return ProfileReqOutput(success=True, message="Succeeded")
        return super()._start_profile(stage)

    def _stop_profile(self, stage=None):
        if stage is None and not self.profile_in_progress:
            self.profile_by_stage = False
            self.profiler_start_forward_ct = None
            self.profiler_target_forward_ct = None
            return ProfileReqOutput(success=True, message="Succeeded")
        if stage is None and self.tpu_profiler is not None:
            current_platform.synchronize()
        result = super()._stop_profile(stage)
        if stage is None:
            # A manual stop also disarms a capture waiting for its first batch.
            self.profile_by_stage = False
            self.profiler_target_forward_ct = None
        return result

    def _profile_batch_predicate(self, batch):
        if (
            not self.profile_by_stage
            or "TPU" not in self.profiler_activities
            or get_disagg().disaggregation_mode != "null"
        ):
            return super()._profile_batch_predicate(batch)

        mode = batch.forward_mode
        if mode.is_prefill():
            if self.profiler_prefill_ct == 0 and self.profiler_decode_ct == 0:
                self._start_profile(ForwardMode.MIXED)
            self.profiler_prefill_ct += 1
        elif mode.is_decode():
            if self.profiler_decode_ct == 0:
                min_bs = envs.SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS.get()
                if min_bs > 0 and batch.batch_size() < min_bs:
                    return
                if not self.profile_in_progress:
                    self._start_profile(ForwardMode.MIXED)
            self.profiler_decode_ct += 1
            if (
                self.profiler_decode_ct > self.profiler_target_decode_ct
                and self.profile_in_progress
            ):
                # The shared stop drains pending device work before exporting.
                # Prefill never stops the collector: stop_trace is synchronous
                # and would leave other ranks waiting in the next collective.
                self._stop_profile(ForwardMode.MIXED)
        elif not (mode.is_idle() or mode.is_prebuilt()):
            raise RuntimeError(f"unsupported profile stage: {mode}")
