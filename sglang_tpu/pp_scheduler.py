# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side pieces of TPU pipeline parallelism: the hand-off wave a
scheduler drives, and the decode transfer consensus over it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang_tpu.pp_wave import PPWave, build_pp_wave

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class PPDecodeTransferStatus(NamedTuple):
    """Transfer terminals shared by every stage and failures seen by any stage."""

    terminal_rids: List[str]
    failed_rids: List[str]


def build_scheduler_pp_wave(scheduler: Scheduler) -> PPWave:
    """The stage's hand-off wave, sized to the largest batch a compiled
    forward can hand on."""
    from sglang_tpu.compilation.decoder_compile_policy import (
        DecoderTpuCompilePolicy,
    )

    model_runner = scheduler.tp_worker.model_runner
    policy = model_runner.compile_policy
    assert isinstance(policy, DecoderTpuCompilePolicy), type(policy)
    return build_pp_wave(
        pp_group=scheduler.pp_group,
        stage_group=scheduler.tp_group,
        world_group=scheduler.world_group,
        rows=policy.max_padded_tokens(),
        hidden=int(model_runner.model_config.hidden_size),
        dtype=model_runner.dtype,
        device=torch.device(model_runner.device),
    )


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
