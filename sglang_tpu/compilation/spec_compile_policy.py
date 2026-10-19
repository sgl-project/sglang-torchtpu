"""TPU compile policy shared by every TpuSpecWorker drafter."""

from __future__ import annotations

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)


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
