"""Launch-time xprof capture, with no /start_profile call.

With SGLANG_TPU_ENABLE_PROFILE_AUTO set, each scheduler arms the upstream
staged profile (``profile_by_stage``) as soon as its profiler manager is built.
From there the upstream lifecycle runs unchanged: the first batch of a stage
opens a capture, and it closes after SGLANG_TPU_PROFILE_AUTO_STEPS forward
passes. Traces go to SGLANG_TORCH_PROFILER_DIR.

- Stages follow the engine's role. A PD prefill engine captures prefill only,
  a PD decode engine decode only, and any other engine both. Upstream waits
  for both stages, so a PD engine would otherwise never finish.
- After its last stage is flushed, each rank writes
  ``.xprof-done-<rank parts>`` beside the traces. The file holds the number of
  ranks in the engine, so a collector knows how many markers to wait for.
- All ranks of an engine share rank 0's profile id, so their trace
  directories group together.

``install()`` wraps three SchedulerProfilerManager methods. Each behaves exactly
like upstream unless the manager was armed here.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_disagg
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs as tpu_envs

logger = get_logger(__name__)

_ALL_STAGES = ("prefill", "decode")

# Keyed by --disaggregation-mode. Any other mode runs both stages.
_STAGES_BY_PD_ROLE = {"prefill": ("prefill",), "decode": ("decode",)}

# Marker name prefix; the rank parts follow, e.g. .xprof-done-TP-0-EP-0.
DONE_MARKER_PREFIX = ".xprof-done"

# Set on a manager armed here: the stages it still has to capture.
_PENDING_ATTR = "_tpu_profile_auto_pending"

# Set by install(), which runs once per scheduler process.
_is_installed = False


def _stage_name(forward_mode: ForwardMode) -> Optional[str]:
    """The stage a batch counts toward, or None for one that never captures."""
    if forward_mode.is_prefill():
        return "prefill"
    if forward_mode.is_decode():
        return "decode"
    return None


def _pending_stages(manager: SchedulerProfilerManager) -> Optional[set[str]]:
    return getattr(manager, _PENDING_ATTR, None)


def _shared_profile_id(manager: SchedulerProfilerManager) -> str:
    """Rank 0's id on every rank.

    /start_profile gets an id from the tokenizer manager; nothing sends one at
    launch. Every rank builds its profiler manager at the same init step, so
    the collective lines up.
    """
    profile_id = [f"auto-{time.time()}"]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(
            profile_id, group=manager.dp_tp_cpu_group, group_src=0
        )
    return profile_id[0]


def _arm(manager: SchedulerProfilerManager) -> None:
    """Configure a staged profile. The first batch of each stage opens it."""
    if envs.SGLANG_PROFILE_V2.get():
        logger.warning(
            "SGLANG_TPU_ENABLE_PROFILE_AUTO is ignored under SGLANG_PROFILE_V2."
        )
        return

    steps = tpu_envs.SGLANG_TPU_PROFILE_AUTO_STEPS.get()
    if steps < 1:
        # Zero steps arms a profile that captures nothing, and the run looks
        # normal until the traces are missing.
        raise ValueError(
            f"SGLANG_TPU_PROFILE_AUTO_STEPS must be at least 1, got {steps}"
        )

    stages = _STAGES_BY_PD_ROLE.get(get_disagg().disaggregation_mode, _ALL_STAGES)
    result = manager._init_profile(
        output_dir=None,
        start_step=None,
        num_steps=steps,
        activities=None,
        with_stack=None,
        record_shapes=None,
        profile_by_stage=True,
        profile_id=_shared_profile_id(manager),
    )
    if result is not None and not result.success:
        logger.warning("Auto profiling failed to arm: %s", result.message)
        return

    setattr(manager, _PENDING_ATTR, set(stages))
    logger.info(
        "Auto profiling armed for %s, %d step(s) each. Traces will be saved to: %s",
        ", ".join(stages),
        steps,
        manager.torch_profiler_output_dir,
    )


def _write_done_marker(manager: SchedulerProfilerManager) -> None:
    """Report this rank's traces as flushed."""
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    name = "-".join([DONE_MARKER_PREFIX, *manager._rank_parts()])
    marker = manager.torch_profiler_output_dir / name
    marker.write_text(f"{world_size}\n")
    logger.info("Auto profiling complete. Wrote %s", marker)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


def _post_init(upstream: Callable, self: SchedulerProfilerManager) -> None:
    upstream(self)
    if tpu_envs.SGLANG_TPU_ENABLE_PROFILE_AUTO.get():
        _arm(self)


def _profile_batch_predicate(
    upstream: Callable, self: SchedulerProfilerManager, batch: Any
) -> None:
    pending = _pending_stages(self)
    stage = _stage_name(batch.forward_mode)
    if pending is not None and stage is not None and stage not in pending:
        # Not ours to capture. Upstream would also end a running decode
        # capture early on the first prefill batch after prefill's capture.
        return
    upstream(self, batch)


def _stop_profile(
    upstream: Callable,
    self: SchedulerProfilerManager,
    stage: Optional[ForwardMode] = None,
):
    result = upstream(self, stage=stage)
    pending = _pending_stages(self)
    if not pending or stage is None or result is None or not result.success:
        return result

    pending.discard(_stage_name(stage))
    if not pending:
        # Disarm, so a later /start_profile runs as upstream.
        setattr(self, _PENDING_ATTR, None)
        _write_done_marker(self)
    return result


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def _patch_targets() -> tuple[tuple[Any, str, Callable], ...]:
    """The upstream methods this module wraps, as (owner, name, wrapper)."""
    return (
        # Arm a staged profile when the manager is built, as /start_profile
        # with profile_by_stage would.
        (SchedulerProfilerManager, "__post_init__", _post_init),
        # Skip stages this engine does not capture or has already captured.
        (
            SchedulerProfilerManager,
            "_profile_batch_predicate",
            _profile_batch_predicate,
        ),
        # Retire a stage once its trace is flushed; after the last one, write
        # this rank's done marker.
        (SchedulerProfilerManager, "_stop_profile", _stop_profile),
    )


def _wrap(owner: Any, name: str, impl: Callable) -> None:
    """Replace ``owner.name`` with ``impl(upstream, *args, **kwargs)``."""
    upstream = getattr(owner, name)

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return impl(upstream, *args, **kwargs)

    setattr(owner, name, wrapper)


def install() -> None:
    """Install every wrapper, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    for owner, name, impl in _patch_targets():
        _wrap(owner, name, impl)
