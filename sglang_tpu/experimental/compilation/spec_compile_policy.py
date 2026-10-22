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
        # Prompts under prefill CP run the context-parallel program at FULL capture;
        # warmed here, before precompile checks the family. Empty without CP.
        pending = []
        for num_tokens, bs in self._extend_cp_compile_cases():
            for spec_info in (None, self.make_null_spec_info()):
                warmup = self._run_one(
                    num_tokens=num_tokens,
                    bs=bs,
                    mode=ForwardMode.EXTEND,
                    variant="cp",
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    spec_info=spec_info,
                )
                if warmup is not None:
                    pending.append(warmup)
        return pending

    def _seed_dummy_req_to_token(self, forward_batch: ForwardBatch) -> None:
        # A CP extend writes only slots > 0 (padding is slot 0); the spec seed's
        # one repeated slot would make every row its own write run.
        if forward_batch.attn_cp_metadata is not None:
            return
        super()._seed_dummy_req_to_token(forward_batch)

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
