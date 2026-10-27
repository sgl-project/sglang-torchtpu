# SPDX-License-Identifier: Apache-2.0
"""Greedy decode on the padded logits rows (SGLANG_TPU_PADDED_LOGITS_SAMPLING).

The TPU logits path pads rows to a power of two and hands back the
``logits[:bs]`` view. In eager mode every in-place op on that view rewrites the
whole padded base, so a decode step whose batch is not a power of two pays for
each penalty op twice. Here the repetition-penalty state is kept padded to the
same row bucket (pad rows hold 1.0, a no-op scale), and a plain greedy step runs
the scaling penalty and argmax on the padded base and slices only the ids. The
per-row ops are the stock ones, so the tokens are the same.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.penaltylib import repetition_penalty
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
)
from sglang_tpu.row_pad import pad_rows, row_bucket


class PaddedRepetitionPenalizer(BatchedRepetitionPenalizer):
    """Repetition-penalty state padded with rows of ones to ``row_bucket(n)``."""

    def _pad(self) -> None:
        rows = row_bucket(self._num_rows)
        self.repetition_penalties = pad_rows(self.repetition_penalties, rows, 1.0)
        self.cumulated_repetition_penalties = pad_rows(
            self.cumulated_repetition_penalties, rows, 1.0
        )

    def _prepare(self):
        super()._prepare()
        self._num_rows = len(self.orchestrator.reqs())
        self._pad()

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # Padded output ids (pad rows point at column 0) scatter their own 1.0.
        src = self.repetition_penalties
        if output_ids.shape[0] != src.shape[0]:
            src = src[: output_ids.shape[0]]
        self.cumulated_repetition_penalties.scatter_(
            dim=1, index=output_ids.unsqueeze(1), src=src
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        penalties = self.cumulated_repetition_penalties
        if penalties.shape[0] != logits.shape[0]:
            penalties = penalties[: logits.shape[0]]
        # Looked up at call time: the TPU platform unwraps the module's
        # @torch.compile(dynamic=True) after this module is imported.
        repetition_penalty.apply_scaling_penalties(logits, penalties)
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        super()._filter(keep_indices)
        self._num_rows = keep_indices.shape[0]
        self._pad()

    def _merge(self, their: PaddedRepetitionPenalizer):
        n1, n2 = self._num_rows, their._num_rows
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties[:n1], their.repetition_penalties[:n2]]
        )
        self.cumulated_repetition_penalties = torch.cat(
            [
                self.cumulated_repetition_penalties[:n1],
                their.cumulated_repetition_penalties[:n2],
            ]
        )
        self._num_rows = n1 + n2
        self._pad()


def padded_greedy_sample(
    model_runner, logits_output, forward_batch
) -> Optional[torch.Tensor]:
    """Next-token ids of a plain greedy decode step taken on the padded logits
    rows, or None when the step needs the stock sampling path."""
    logits = logits_output.next_token_logits
    if logits is None or logits_output.has_next_token_global_top_k:
        return None
    if not forward_batch.forward_mode.is_decode() or forward_batch.return_logprob:
        return None
    info = forward_batch.sampling_info
    if (
        info is None
        or not info.is_all_greedy
        or info.has_custom_logit_processor
        or info.logit_bias is not None
        or any(info.grammars or ())
        or info.sampling_mask_batch_indices is not None
        or info.acc_additive_penalties is not None
        or (
            info.penalizer_orchestrator is not None
            and info.penalizer_orchestrator.is_required
        )
        or envs.SGLANG_SANITIZE_NAN_LOGITS.get()
    ):
        return None
    observer = model_runner.sampling_observer
    if observer is not None and observer.is_active(info):
        return None
    bs = logits.shape[0]
    base = logits._base if logits._base is not None else logits
    if (
        logits.dim() != 2
        or base.dim() != 2
        or base.shape[1] != logits.shape[1]
        or base.shape[0] < bs
        or logits.storage_offset() != 0
        or not base.is_contiguous()
        or logits.stride() != base.stride()
    ):
        return None
    scaling = info.acc_scaling_penalties
    padded = base
    if scaling is not None:
        if scaling.shape[0] != base.shape[0]:
            return None
        padded = torch.where(base < 0, base * scaling, base / scaling)
        if padded.dtype != base.dtype:
            padded = padded.to(base.dtype)
    next_token_ids = torch.argmax(padded, -1)
    if base.shape[0] != bs:
        next_token_ids = next_token_ids[:bs]
    next_token_ids = next_token_ids.to(torch.int64)
    logits_output.auxiliary_device_output = None
    model_runner.sampler._sync_token_ids_across_tp(next_token_ids, info)
    model_runner.ngram_embedding_manager.update_after_decode(
        next_token_ids=next_token_ids,
        forward_batch=forward_batch,
    )
    return next_token_ids
