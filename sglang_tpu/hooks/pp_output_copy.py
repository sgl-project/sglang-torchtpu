# SPDX-License-Identifier: Apache-2.0
"""Copy the last pipeline stage's outputs to the host at launch.

Why: Over the wave the last stage sends a forward's outputs to the first stage
     only after it has launched the next forward, and the send copies them to
     the host with a blocking ``.cpu()`` that can wait until that next forward
     ends; the first stage then waits on the relay.
How: AROUND Scheduler._pp_launch_batch starts the copy of the queued outputs'
     device tensors on a TPUAsyncCopyWorker. AROUND
     Scheduler._pp_send_output_to_next_stage and
     Scheduler._pp_pd_send_deferred_output put the host copies in place of the
     outputs the send pops; a skipped send drops them without waiting.
Upstream PR or issue: not filed; the wave and its deferred send are TPU only.
Remove when: the deferred send moves into sglang_tpu with the wave.
"""

from __future__ import annotations

import weakref
from concurrent.futures import Future
from typing import Callable

import torch

from sglang.srt.managers import scheduler_pp_mixin
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


class PPOutputCopies:
    """Queued outputs and the host copies of their device tensors."""

    def __init__(self, copy_to_host: Callable[[torch.Tensor], Future]) -> None:
        self._copy_to_host = copy_to_host
        # PPProxyTensors defines __eq__ without __hash__: key by identity, and
        # drop an entry when its outputs are collected without being sent.
        self._copies: dict[int, tuple[weakref.ref, dict[str, Future]]] = {}

    def start(self, outputs: PPProxyTensors) -> None:
        copies = {
            name: self._copy_to_host(tensor)
            for name, tensor in outputs.tensors.items()
            if isinstance(tensor, torch.Tensor) and not tensor.is_cpu
        }
        if not copies:
            return
        key = id(outputs)
        entries = self._copies
        ref = weakref.ref(outputs, lambda _, key=key: entries.pop(key, None))
        entries[key] = (ref, copies)

    def take(self, outputs: PPProxyTensors, *, sent: bool) -> None:
        """Forget the copies of ``outputs``; if they are about to be sent, wait
        for them and put the host tensors in place."""
        entry = self._copies.get(id(outputs))
        if entry is None or entry[0]() is not outputs:
            return
        del self._copies[id(outputs)]
        if sent:
            for name, copy in entry[1].items():
                outputs.tensors[name] = copy.result()


_copies: PPOutputCopies | None = None


def _active_copies(scheduler: Scheduler) -> PPOutputCopies | None:
    """The copies of a last stage over the wave, whose sends are deferred."""
    global _copies
    if scheduler.device == "cpu" or not scheduler._pp_output_send_deferred():
        return None
    if _copies is None:
        from sglang_tpu.result_copy_worker import TPUAsyncCopyWorker

        _copies = PPOutputCopies(TPUAsyncCopyWorker().copy_to_host)
    return _copies


def _sends(target) -> bool:
    return (
        not target.forward_mode.is_prebuilt()
        and not scheduler_pp_mixin._pp_can_skip_output_comm(target)
    )


def copy_outputs_at_launch(
    original,
    scheduler: Scheduler,
    mb_id,
    cur_batch,
    pp_proxy_tensors,
    mb_metadata,
    last_rank_comm_queue,
):
    queued = len(last_rank_comm_queue)
    launched = original(
        scheduler, mb_id, cur_batch, pp_proxy_tensors, mb_metadata, last_rank_comm_queue
    )
    copies = _active_copies(scheduler)
    if copies is not None and len(last_rank_comm_queue) > queued:
        copies.start(last_rank_comm_queue[-1][1])
    return launched


def send_copied_outputs(
    original,
    scheduler: Scheduler,
    next_first_rank_mb_id,
    mbs,
    last_rank_comm_queue,
    pp_outputs,
):
    copies = _active_copies(scheduler)
    target = mbs[next_first_rank_mb_id]
    if copies is not None and target is not None and last_rank_comm_queue:
        copies.take(last_rank_comm_queue[0][1], sent=_sends(target))
    return original(
        scheduler, next_first_rank_mb_id, mbs, last_rank_comm_queue, pp_outputs
    )


def send_copied_prefill_outputs(original, scheduler: Scheduler, mb_id):
    copies = _active_copies(scheduler)
    target = scheduler.mbs[mb_id] if copies is not None else None
    if target is not None and scheduler.last_rank_comm_queue:
        # The prefill send copies the tokens whenever the send is not skipped.
        sent = not scheduler_pp_mixin._pp_can_skip_output_comm(target)
        copies.take(scheduler.last_rank_comm_queue[0][1], sent=sent)
    return original(scheduler, mb_id)
