"""TPU compile policy shared by every TpuSpecWorker drafter."""

from __future__ import annotations

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    LogitsOutputMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded
from sglang_tpu.experimental.speculative.adaptive_steps import draft_step_schedule
from sglang_tpu.sampler import negotiated_distributed_top_k_width


class TpuSpecDecodeCompilePolicyMixin(DFlashSpecDecodeCompilePolicyMixin):
    """Target warms TARGET_VERIFY and EXTEND(FULL); subclasses warm their draft forwards.

    Every drafter's verify is TpuSpecWorker's chain verify, whose padded-output
    handling the DFlash policy already owns; the drafters differ only on the draft runner.
    """

    @property
    def algo_label(self) -> str:
        return self.model_runner.spec_algorithm.name

    # TpuSpecWorker never sends the target DECODE (decode is TARGET_VERIFY) or a
    # capture-NULL EXTEND (extend always captures hidden for the draft).
    def _precompile_decode(self) -> list[PendingCall]:
        return []

    def _precompile_extend(self) -> list[PendingCall]:
        return []

    def target_verify_logits_output_modes(self) -> tuple[LogitsOutputMode, ...]:
        # TpuSpecWorker negotiated the width-1 argmax for a verify that reads only it.
        if (
            not self.is_draft_worker
            and negotiated_distributed_top_k_width(self.model_runner.model) == 1
        ):
            return (LogitsOutputMode.DISTRIBUTED_TOP_K,)
        return super().target_verify_logits_output_modes()

    def verify_bucket_widths(self) -> list[tuple[int, int]]:
        """(verify width, bs) pairs the worker runs; a draft-step schedule's when set."""
        assert self.b_bucketer is not None
        buckets = list(self.b_bucketer.buckets)
        schedule = draft_step_schedule()
        if schedule is None:
            return [(self.draft_token_num, bs) for bs in buckets]
        return [
            (width, bs)
            for bs in buckets
            for width in schedule.widths_for_bucket(bs, buckets)
        ]

    def _target_verify_shapes(self) -> list[tuple[int, int]]:
        return self.verify_bucket_widths()

    def _verify_spec_info(self, width: int) -> DFlashVerifyInput:
        if width == self.draft_token_num:
            return self.make_verify_spec_info()
        return DFlashVerifyInput(
            draft_token=torch.empty(
                (0,), dtype=torch.int64, device=self.model_runner.device
            ),
            positions=torch.empty(
                (0,), dtype=torch.int64, device=self.model_runner.device
            ),
            draft_token_num=width,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    def make_null_spec_info(self) -> DFlashVerifyInput:
        return make_draft_block_spec_info(
            draft_token_num=self.draft_token_num, device=self.model_runner.device
        )

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: DFlashVerifyInput | None = None,
    ) -> ForwardBatch:
        batch = super()._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        # The draft embeds input_ids in-graph with the attached embedding.
        if self.is_draft_worker:
            batch.input_embeds = None
        return batch

    def keeps_padded_mm_embeds(self, batch: ForwardBatch) -> bool:
        """Keep multimodal embeds at the token bucket for HOST_PADDED draft extend."""
        return (
            batch.sampling_info is not None
            and is_host_padded(batch.sampling_info)
            and not batch.forward_mode.is_target_verify()
        )
