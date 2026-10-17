"""TPU compile policy for TpuEagle3Worker's target and draft runners."""

from __future__ import annotations

from typing import Any

import msgspec
import torch

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.utils.common import ceil_align
from sglang_tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
    _restore_forward_batch_inputs,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.spec_decode_compile_policy_eagle3 import (
    Eagle3SpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded
from sglang_tpu.experimental.speculative.eagle_info import eagle_verify_input


class TpuEagle3SpecDecodeCompilePolicyMixin(Eagle3SpecDecodeCompilePolicyMixin):
    """EAGLE3 warmups; host-padded prefill keeps its outputs at the batch bucket.

    TpuEagle3Worker prepares the draft decode, draft extend and target verify
    batches already at a batch bucket through the EAGLE preparation code, so the
    inherited exact-bucket warmups cover them. Its prefill batches are host
    padded: the target samples, and the draft prefill picks its first draft, on
    the batch-bucket rows instead of the live rows.
    """

    # TpuSpecWorker never sends the target DECODE (decode is TARGET_VERIFY) or a
    # capture-NULL EXTEND (extend always captures hidden for the draft prefill).
    def _precompile_decode(self) -> list[PendingCall]:
        return []

    def _precompile_extend(self) -> list[PendingCall]:
        # A multimodal target still places host-built image rows eagerly per
        # token bucket in its EXTEND(FULL) prefills.
        precompile_mm_embed_placement = getattr(
            self, "_precompile_mm_embed_placement", None
        )
        if precompile_mm_embed_placement is not None and self._uses_input_embeds(
            self.owner
        ):
            precompile_mm_embed_placement()
        return []

    def make_verify_spec_info(self) -> EagleVerifyInput:
        # The dynamo tracer guards on the spec_info structure, so warm exactly what
        # TpuEagle3Worker.make_verify_input builds.
        device = self.model_runner.device
        return eagle_verify_input(
            draft_token=torch.empty(0, device=device, dtype=torch.int64),
            positions=torch.empty(0, device=device, dtype=torch.int64),
            num_drafts=self.draft_token_num - 1,
        )

    def _draft_extend_bs_buckets(self) -> list[int]:
        # Draft extend covers the decode bucket's verify rows, which DP attention
        # aligns to attention TP past the local request capacity.
        buckets = super()._draft_extend_bs_buckets()
        if not is_dp_attention_enabled():
            return buckets
        assert self.b_bucketer is not None
        largest = self.b_bucketer.next_bucket(
            ceil_align(
                int(self.model_runner.req_to_token_pool.size),
                get_parallel().attn_tp_size,
            )
        )
        return [bs for bs in self.b_bucketer.buckets if bs <= largest]

    def _dummy_layout(
        self, mode: ForwardMode, num_tokens: int, bs: int, tokens_per_req: int
    ) -> Any:
        layout = super()._dummy_layout(mode, num_tokens, bs, tokens_per_req)
        pool_rows = self.model_runner.req_to_token_pool.req_to_token.shape[0]
        if not mode.is_draft_extend_v2() or bs < pool_rows:
            return layout
        # A bucket past the local request capacity; serving pads it on pool row 0.
        return msgspec.structs.replace(
            layout,
            req_pool_indices=layout.req_pool_indices % pool_rows,
            req_pool_indices_cpu=layout.req_pool_indices_cpu % pool_rows,
        )

    def keeps_padded_mm_embeds(self, batch: ForwardBatch) -> bool:
        # The draft prefill reads the target's embeds at the token bucket.
        return is_host_padded(batch.sampling_info)

    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        sampling_info = forward_batch.sampling_info
        if (
            sampling_info is None
            or not is_host_padded(sampling_info)
            or forward_batch.forward_mode != ForwardMode.EXTEND
        ):
            return super().finalize_forward_batch(outputs, forward_batch)
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        if real_q is None or real_b is None:
            return super().finalize_forward_batch(outputs, forward_batch)
        if not isinstance(outputs, LogitsProcessorOutput):
            # An inner language model (multimodal wrapper) hands its eager logits
            # processor the [T, ...] rows; keep the token bucket for the draft prefill.
            _restore_forward_batch_inputs(forward_batch, real_q, real_b)
            return outputs
        next_token_logits = outputs.next_token_logits
        hidden_states = outputs.hidden_states

        padded_bs = sampling_info.temperatures.shape[0]
        if next_token_logits is not None and padded_bs < next_token_logits.shape[0]:
            outputs.next_token_logits = next_token_logits[:padded_bs]
        # The draft prefill captures LAST, one row per request: keep the bucket
        # rows like its logits. The target prefill captures FULL; the draft
        # prefill reads that at the token bucket, so a [:real_q] trim would
        # re-pad it per prompt length.
        capture = forward_batch.capture_hidden_mode
        if (
            hidden_states is not None
            and capture is not None
            and capture.is_last()
            and padded_bs < hidden_states.shape[0]
        ):
            outputs.hidden_states = hidden_states[:padded_bs]

        _restore_forward_batch_inputs(forward_batch, real_q, real_b)
        if forward_batch.spec_info is not None:
            forward_batch.spec_info.keep_outputs_padded = False
        return outputs
