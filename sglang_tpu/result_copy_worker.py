from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput


class TpuAsyncCopyEvent:
    """Like torch.cuda.Event(), but for an async copy.

    Call record() on the copy worker's stream after submitting the copies, and
    synchronize() to block until the copy's results are ready.
    """

    def __init__(self, event, submitted: Future[Optional[torch.Tensor]]) -> None:
        self.event = event
        self.submitted = submitted

    def record(self) -> None:
        self.event.record()

    def synchronize(self) -> None:
        """Wait for submission and transfer, propagating worker errors."""
        from sglang.srt.utils.diag_timing import timed as _dt

        with _dt("4a_copy_submit_wait"):
            self.submitted.result()
        with _dt("4b_copy_event_wait"):
            self.event.synchronize()

    def query(self) -> bool:
        """Return whether synchronize() would return without blocking.

        The worker records the event last and an unrecorded event reports
        done, so the event counts only once the submission has finished. A
        failed submission reports done so synchronize() raises its error.
        """
        if not self.submitted.done():
            return False
        if self.submitted.exception() is not None:
            return True
        return self.event.query()


def _draft_tail(result: GenerationBatchResult) -> Optional[torch.Tensor]:
    """Tail: a small tensor produced after the step's last KV write; copying it
    to host makes copy_done cover those writes, so the KV free fence can free pages."""
    draft_input = result.next_draft_input
    if isinstance(draft_input, HostPaddedDraftInput):
        # The experimental worker's commit also writes KV after verify;
        # idle steps write no freeable KV and carry none.
        return draft_input.kv_fence_tail
    if isinstance(draft_input, EagleDraftInput):
        return draft_input.topk_index
    if isinstance(draft_input, DFlashDraftInputV2):
        # The DFLASH draft KV append runs after verify.
        return draft_input.draft_kv_write_tail
    return None


class TPUAsyncCopyWorker:
    """Keep TPU buffer-materialization waits off the scheduler thread.

    This is a workaround for TorchTPU's tensor.to("cpu", non_blocking=True):
    copy submission can still block waiting for the source TPU buffer handle.
    """

    def __init__(self, copy_draft_tail: bool = False) -> None:
        self.copy_draft_tail = copy_draft_tail
        self.device_module = torch.get_device_module("tpu")
        self.device_index = torch.accelerator.current_device_index()
        # TPU events track work by stream. Keep copy events from capturing
        # unrelated work submitted by the scheduler.
        self.stream = self.device_module.Stream(device=self.device_index)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="result-copy"
        )

    def submit(
        self,
        result: GenerationBatchResult,
        return_logprob: bool,
        return_hidden_states: bool,
    ) -> None:
        """Run result.copy_to_cpu(...) asynchronously on the worker.

        Consumers must call result.copy_done.synchronize() before reading the results.
        """
        event = result.copy_done
        tail = _draft_tail(result) if self.copy_draft_tail else None
        # DIAG (remove): SGLANG_TPU_DIAG_SKIP_DRAFT_TAIL=1 skips the tail copy.
        if os.environ.get("SGLANG_TPU_DIAG_SKIP_DRAFT_TAIL") == "1":
            tail = None
        if tail is not None and tail._base is not None:
            # A view of a pool the next step donates: clone it here, on the scheduler's
            # stream ahead of that step, so the copy thread never reads a donated buffer.
            tail = tail.clone()
        submitted = self.executor.submit(
            self._copy, result, tail, return_logprob, return_hidden_states
        )
        result.copy_done = TpuAsyncCopyEvent(event, submitted)

    def close(self) -> None:
        """Drain copy submissions and stop the worker; transfers may still be pending."""
        self.executor.shutdown(wait=True)

    @torch.inference_mode()
    def _copy(
        self,
        result: GenerationBatchResult,
        tail: Optional[torch.Tensor],
        return_logprob: bool,
        return_hidden_states: bool,
    ) -> Optional[torch.Tensor]:
        with (
            torch.accelerator.device_index(self.device_index),
            self.device_module.stream(self.stream),
        ):
            # EAGLE draft-extend runs after verify and writes draft KV; copying
            # one of its outputs makes copy_done cover it. The fence rejects
            # algorithms whose KV writes follow every copied tensor (see
            # supports_kv_free_fence). The returned copy stays referenced by
            # the submission future.
            draft_tail = None if tail is None else tail.to("cpu", non_blocking=True)
            # copy_to_cpu calls result.copy_done.record() after issuing the copies.
            result.copy_to_cpu(return_logprob, return_hidden_states)
            return draft_tail
