# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side pieces of TPU pipeline parallelism: the hand-off wave a
scheduler drives, and the decode transfer consensus over it."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.hardware_backend.tpu.pp_wave import PPWave, build_pp_wave

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class PPDecodeTransferStatus(NamedTuple):
    """Transfer terminals shared by every stage and failures seen by any stage."""

    terminal_rids: List[str]
    failed_rids: List[str]


def _pp_slot_config(pp_rank: int, pp_size: int) -> tuple[int, int]:
    """(slot size, this stage's stage-local launches per forward) from
    SGLANG_TPU_PP_SLOT_LAUNCHES and SGLANG_TPU_PP_FORWARD_LAUNCHES (one count,
    or one per stage, comma-separated). Slots are off when the first is unset.
    """
    import os

    slot = int(os.environ.get("SGLANG_TPU_PP_SLOT_LAUNCHES", "0") or 0)
    raw = os.environ.get("SGLANG_TPU_PP_FORWARD_LAUNCHES", "1")
    counts = [int(x) for x in raw.split(",")]
    if len(counts) == 1:
        counts *= pp_size
    if len(counts) != pp_size:
        raise ValueError(
            f"SGLANG_TPU_PP_FORWARD_LAUNCHES={raw!r} needs 1 or {pp_size} counts"
        )
    if slot and max(counts) > slot:
        raise ValueError(
            f"SGLANG_TPU_PP_SLOT_LAUNCHES={slot} is below a stage's "
            f"forward launch count ({raw})"
        )
    return slot, counts[pp_rank]


def build_scheduler_pp_wave(scheduler: Scheduler) -> PPWave:
    """The stage's hand-off wave, sized to the largest batch a compiled
    forward can hand on."""
    from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
        DecoderTpuCompilePolicy,
    )

    model_runner = scheduler.tp_worker.model_runner
    policy = model_runner.compile_policy
    assert isinstance(policy, DecoderTpuCompilePolicy), type(policy)
    measured_slot = int(getattr(policy, "pp_slot_launches", 0) or 0)
    if measured_slot and "SGLANG_TPU_PP_SLOT_LAUNCHES" not in os.environ:
        # Warmup measured every compiled graph's launches on this stage.
        slot, forward_launches = measured_slot, 0
    else:
        slot, forward_launches = _pp_slot_config(
            int(scheduler.pp_group.rank_in_group), int(scheduler.pp_group.world_size)
        )
        measured_slot = 0
    wave = build_pp_wave(
        pp_group=scheduler.pp_group,
        stage_group=scheduler.tp_group,
        world_group=scheduler.world_group,
        rows=policy.max_padded_tokens(),
        hidden=int(model_runner.model_config.hidden_size),
        dtype=model_runner.dtype,
        device=torch.device(model_runner.device),
        slot_size=slot,
    )
    wave.forward_launches = forward_launches if slot else 0
    if measured_slot:
        wave.forward_launch_counter = policy.pp_take_forward_launches
    return wave


def decode_transfer_status(
    queue: Any, previous: Optional[PPDecodeTransferStatus]
) -> PPDecodeTransferStatus:
    """This stage's transfer poll folded into the previous stages' status:
    terminals are shared by every stage and a failure seen by any stage is a
    failure."""
    polls = (
        queue._poll_with_staging()
        if queue.enable_staging
        else queue._poll_with_metadata_gate()
    )
    terminal_rids = [
        item.req.rid
        for item, poll in zip(queue.queue, polls)
        if poll in (KVPoll.Success, KVPoll.Failed)
    ]
    failed_rids = [
        item.req.rid for item, poll in zip(queue.queue, polls) if poll == KVPoll.Failed
    ]
    if previous is not None:
        terminal_rids = sorted(set(previous.terminal_rids) & set(terminal_rids))
        failed_rids = sorted(set(previous.failed_rids) | set(failed_rids))
    return PPDecodeTransferStatus(terminal_rids, failed_rids)


def abort_failed_receivers(queue: Any, failed_rids: List[str]) -> None:
    """Fail this stage's copies of transfers that failed on any stage."""
    failed = set(failed_rids)
    for decode_req in queue.queue:
        if decode_req.req.rid not in failed:
            continue
        receiver = decode_req.kv_receiver
        if getattr(receiver, "group_failure_requires_drain", False):
            receiver.quiesce_group_failure()
        else:
            receiver.abort()
