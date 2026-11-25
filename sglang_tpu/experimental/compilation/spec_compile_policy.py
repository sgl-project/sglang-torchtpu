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
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded

logger = get_logger(__name__)


def _fused_mixed_steps() -> bool:
    """Whether this server runs fused mixed prefill + verify steps."""
    from sglang.srt.speculative.mixed_verify import mixed_verify_enabled
    from sglang_tpu.experimental.speculative.mixed_step import fused_mixed_forward

    return fused_mixed_forward() and mixed_verify_enabled()


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

    def precompile(self) -> list[PendingCall]:
        pending = super().precompile()
        if not self.is_draft_worker and _fused_mixed_steps():
            pending += self._precompile_mixed_verify()
        return pending

    def _precompile_mixed_verify(self) -> list[PendingCall]:
        """The fused mixed prefill + verify forward (SGLANG_TPU_MIXED_VERIFY_FUSED):
        MIXED with FULL capture and the [ext_cap + R * pad_bv] logits row index,
        at every (token bucket, verify bucket) a step can pick."""
        from sglang.srt.runtime_context import get_parallel
        from sglang_tpu.experimental.speculative import mixed_step

        assert self.t_bucketer is not None and self.b_bucketer is not None
        cases = mixed_step.fused_compile_cases(
            self.t_bucketer, self.b_bucketer, get_parallel().attn_tp_size
        )
        verify_buckets = sorted({pbs for _, _, pbs in cases})
        grid = [
            (num_tokens, ext_cap, pbs, variant)
            for num_tokens, ext_cap, pbs in cases
            for variant in self._mixed_decoder_variants(num_tokens, pbs, verify_buckets)
        ]
        logger.info(
            "[spec compile policy] MIXED (fused prefill + verify): %d shapes "
            "(tokens, prompt cap, verify bucket, decoder variant) = %s",
            len(grid),
            grid,
        )
        pending = []
        for num_tokens, ext_cap, pbs, variant in grid:
            # A fresh upload, as serving's (graph inputs must match in form).
            index = torch.zeros(
                (ext_cap + self.draft_token_num * pbs,), dtype=torch.int64
            ).to(self.model_runner.device)
            warmup = self._run_one(
                num_tokens=num_tokens,
                bs=ext_cap + pbs,
                mode=ForwardMode.MIXED,
                variant=variant,
                logits_output_mode=LogitsOutputMode.FULL_VOCAB,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=mixed_step.make_mixed_verify_input(
                    positions=torch.zeros((0,), dtype=torch.int64),
                    num_verify_rows=self.draft_token_num,
                    logits_select_index=index,
                ),
            )
            if warmup is not None:
                pending.append(warmup)
        return pending

    def _mixed_decoder_variants(
        self, num_tokens: int, pbs: int, verify_buckets: list[int]
    ) -> tuple[str, ...]:
        """Warmup variants of a fused MIXED forward at (num_tokens, pbs): "plain",
        or under DSV4.1 decoder replay the decoder row buckets a step can pick.
        The busiest rank's decoder rows are at least its verify rows, more than
        R x the next smaller verify bucket (pbs is the smallest bucket holding
        the busiest rank's requests), and at most the token bucket."""
        decoder_replay_on = getattr(self, "_decoder_replay_on", None)
        if decoder_replay_on is None or not decoder_replay_on():
            return ("plain",)
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            mixed_decoder_buckets,
        )

        smaller = [b for b in verify_buckets if b < pbs]
        min_rows = self.draft_token_num * ((smaller[-1] if smaller else 0) + 1)
        return tuple(
            "decT" if bucket == num_tokens else f"dec{bucket}"
            for bucket in mixed_decoder_buckets(min_rows, num_tokens)
        )

    def finalize_forward_batch(self, outputs, forward_batch: ForwardBatch):
        """A fused mixed forward keeps its outputs at the bucket: the worker slices
        the fixed-size logits rows (prompt cap + verify rows) and gathers hidden
        rows at fixed offsets; the decoder trim would cut them to the request
        count. Inputs are restored as for any forward."""
        # Keyed on the batch itself (MIXED with its logits row index, or the plan),
        # not on spec_info's keep_outputs_padded: the batch's spec_info here need
        # not be the object the worker flagged (smoke j-2ac8a7c7-8598-4f36-97a2
        # still trimmed to the 64 sampling rows).
        if (
            forward_batch.forward_mode.is_mixed()
            and _fused_mixed_steps()
            and (
                getattr(forward_batch, "logit_rows", None) is not None
                or getattr(forward_batch, "mixed_verify", None) is not None
            )
        ):
            from sglang_tpu.compilation.compiled_forward_shim import (
                _REAL_B_KEY,
                _REAL_Q_KEY,
                _restore_forward_batch_inputs,
            )

            real_q = getattr(forward_batch, _REAL_Q_KEY, None)
            real_b = getattr(forward_batch, _REAL_B_KEY, None)
            if real_q is not None and real_b is not None:
                _restore_forward_batch_inputs(forward_batch, real_q, real_b)
            return outputs
        return super().finalize_forward_batch(outputs, forward_batch)

    def set_dp_attention_metadata(
        self, forward_batch: ForwardBatch, num_tokens: int, bs: int, mode: ForwardMode
    ) -> None:
        super().set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)
        if getattr(forward_batch, "mixed_verify", None) is not None:
            # Serving's mixed step is an extend step on every DP rank.
            forward_batch.is_extend_in_batch = True

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
        index = getattr(spec_info, "logits_select_index", None)
        if mode == ForwardMode.MIXED and index is not None:
            # The fused mixed forward: its logits row index rides on a field the
            # direct tracer lifts, and the plan gives the shim its batch rows.
            from sglang.srt.speculative.mixed_verify import MixedVerifyPlan

            batch.logit_rows = index
            pbs = (index.shape[0] - bs) // (self.draft_token_num - 1)
            batch.mixed_verify = MixedVerifyPlan(
                n_prompt_rows=0,
                n_verify=0,
                num_verify_rows=self.draft_token_num,
                global_prompt_rows=(num_tokens,),
                global_verify_bs=(pbs,),
            )
            spec_info.keep_outputs_padded = True
            _concentrate_dummy_rows(batch, bs)
        return batch

    def keeps_padded_mm_embeds(self, batch: ForwardBatch) -> bool:
        """Keep multimodal embeds at the token bucket for HOST_PADDED draft extend."""
        return (
            batch.sampling_info is not None
            and is_host_padded(batch.sampling_info)
            and not batch.forward_mode.is_target_verify()
        )


def _concentrate_dummy_rows(batch: ForwardBatch, bs: int) -> None:
    """Give a dummy MIXED batch's token rows all to its first request and leave
    the other ``bs - 1`` requests with zero rows, as a serving step's padding
    requests are. Compiled shapes depend on the padded token and batch buckets
    only; one row per dummy request would instead claim one SWA request-window
    workspace page each (sized for a chunk plus real slots), more than a real
    step can."""
    rows = int(batch.extend_num_tokens or bs)
    device = batch.extend_seq_lens.device
    lens = [rows] + [0] * (bs - 1)
    batch.extend_seq_lens_cpu = lens
    batch.extend_seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    batch.extend_start_loc = torch.tensor(
        [0] + [rows] * (bs - 1), dtype=torch.int32, device=device
    )
    seq = torch.tensor(lens, dtype=torch.int64)
    batch.seq_lens_cpu = seq
    batch.seq_lens = seq.to(batch.seq_lens.device)
    batch.orig_seq_lens = batch.seq_lens
    batch.seq_lens_sum = rows
