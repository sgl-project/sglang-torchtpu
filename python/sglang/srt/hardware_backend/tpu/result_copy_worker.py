from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.utils import GenerationBatchResult


class TpuAsyncCopyEvent:
    """Like torch.cuda.Event(), but for an async copy.

    Call record() on the copy worker's stream after submitting the copies, and
    synchronize() to block until the copy's results are ready.
    """

    def __init__(self, event, submitted: Future[None]) -> None:
        self.event = event
        self.submitted = submitted

    def record(self) -> None:
        self.event.record()

    def synchronize(self) -> None:
        """Wait for submission and transfer, propagating worker errors."""
        self.submitted.result()
        self.event.synchronize()


class TPUAsyncCopyWorker:
    """Keep TPU buffer-materialization waits off the scheduler thread.

    This is a workaround for TorchTPU's tensor.to("cpu", non_blocking=True):
    copy submission can still block waiting for the source TPU buffer handle.
    """

    def __init__(self) -> None:
        self.device_module = torch.get_device_module("tpu")
        self.device_index = self.device_module.current_device()
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
        submitted = self.executor.submit(
            self._copy, result, return_logprob, return_hidden_states
        )
        result.copy_done = TpuAsyncCopyEvent(event, submitted)

    def close(self) -> None:
        """Drain copy submissions and stop the worker; transfers may still be pending."""
        self.executor.shutdown(wait=True)

    @torch.inference_mode()
    def _copy(
        self,
        result: GenerationBatchResult,
        return_logprob: bool,
        return_hidden_states: bool,
    ) -> None:
        with (
            self.device_module.device(self.device_index),
            self.device_module.stream(self.stream),
        ):
            # copy_to_cpu calls result.copy_done.record() after issuing the copies.
            result.copy_to_cpu(return_logprob, return_hidden_states)
