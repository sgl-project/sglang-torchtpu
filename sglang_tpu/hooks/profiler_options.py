# SPDX-License-Identifier: Apache-2.0
"""Resolve and forward TPU profiler options across SchedulerProfilerManager.

Why: Upstream SchedulerProfilerManager._init_profile / _start_profile do not
     yet forward backend-specific ``profiler_kwargs`` or ``profile_prefix``
     options into ``start_tpu_profile`` (or parse ``SGLANG_TPU_PROFILE_OPTIONS``).
How: AROUND hooks on ``SchedulerProfilerManager._profile`` (stash
     ``recv_req.profiler_kwargs``), ``SchedulerProfilerManager._init_profile``
     (parse/validate options and strip ``key:value;...`` from ``profile_prefix``),
     and ``SchedulerProfilerManager._start_profile`` (bind resolved options via
     ``scoped_profile_options`` and disarm armed counters on startup error).
Upstream PR or issue: https://github.com/sgl-project/sglang/pull/53524
     (adds ``ProfileReq.profiler_kwargs`` and
     ``DeviceMixin.get_torch_profiler_experimental_config``).
Remove when: sgl-project/sglang#53524 syncs into ``python/sglang/`` and
     ``TpuOOTPlatform`` implements ``get_torch_profiler_experimental_config``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang_tpu import profiler as tpu_profiler
from sglang_tpu._logging import get_logger

logger = get_logger(__name__)

_OPTIONS_ATTR = "_tpu_profile_options"
_PENDING_KWARGS_ATTR = "_tpu_pending_profiler_kwargs"


def stash_profiler_kwargs(
    upstream: Callable,
    self: SchedulerProfilerManager,
    recv_req: Any,
) -> ProfileReqOutput | None:
    """Bridge ``recv_req.profiler_kwargs`` across ``_profile -> _init_profile``."""
    setattr(self, _PENDING_KWARGS_ATTR, getattr(recv_req, "profiler_kwargs", None))
    try:
        return upstream(self, recv_req)
    finally:
        setattr(self, _PENDING_KWARGS_ATTR, None)


def init_tpu_profile_options(
    upstream: Callable,
    self: SchedulerProfilerManager,
    output_dir: Optional[str],
    start_step: Optional[int],
    num_steps: Optional[int],
    activities: Optional[list[str]],
    with_stack: Optional[bool],
    record_shapes: Optional[bool],
    profile_by_stage: bool,
    profile_id: str,
    merge_profiles: bool = False,
    profile_prefix: str = "",
    detailed_annotations: bool = False,
    profile_stages: Optional[list[str]] = None,
    profiler_kwargs: Optional[dict[str, Any]] = None,
) -> ProfileReqOutput:
    """Parse and validate TPU profiler options before committing manager state."""
    if profiler_kwargs is None:
        profiler_kwargs = getattr(self, _PENDING_KWARGS_ATTR, None)
    resolved_options: tpu_profiler.TpuProfileOptions | None = None
    if envs.SGLANG_PROFILE_V2.get():
        if profiler_kwargs:
            # ProfileManager (V2) does not wire experimental_config or OOT
            # startup errors; reject explicitly rather than ignoring options.
            return ProfileReqOutput(
                success=False,
                message="profiler_kwargs is not supported with SGLANG_PROFILE_V2.",
            )
    else:
        try:
            clean_prefix, resolved_options = tpu_profiler.resolve_tpu_profile_options(
                profile_prefix, profiler_kwargs=profiler_kwargs
            )
        except ValueError as e:
            return ProfileReqOutput(success=False, message=str(e))
        profile_prefix = clean_prefix or ""
    result = upstream(
        self,
        output_dir,
        start_step,
        num_steps,
        activities,
        with_stack,
        record_shapes,
        profile_by_stage,
        profile_id,
        merge_profiles=merge_profiles,
        profile_prefix=profile_prefix,
        detailed_annotations=detailed_annotations,
        profile_stages=profile_stages,
    )
    if resolved_options is not None and (result is None or result.success):
        setattr(self, _OPTIONS_ATTR, resolved_options)
    return result


def start_tpu_profile_with_options(
    upstream: Callable,
    self: SchedulerProfilerManager,
    stage: Optional[ForwardMode] = None,
) -> ProfileReqOutput | None:
    """Bind resolved TPU options around ``_start_profile`` and disarm on failure."""
    options = getattr(self, _OPTIONS_ATTR, None)
    with tpu_profiler.scoped_profile_options(options):
        try:
            return upstream(self, stage=stage)
        except (ValueError, RuntimeError) as e:
            logger.error("Failed to start TPU profiler: %s", e)
            self.tpu_profiler = None
            self.profile_in_progress = False
            self.profiler_start_forward_ct = None
            self.profiler_target_forward_ct = None
            # Keep profiler_target_{prefill,decode}_ct non-None: _profile_batch_predicate
            # compares against them on the current batch right after _start_profile returns.
            self.profile_by_stage = False
            return ProfileReqOutput(success=False, message=str(e))
