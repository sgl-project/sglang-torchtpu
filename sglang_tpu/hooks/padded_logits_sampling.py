# SPDX-License-Identifier: Apache-2.0
"""Greedy decode on the padded logits rows (SGLANG_TPU_PADDED_LOGITS_SAMPLING).

Registered only when the variable is on; off, none of these hooks exist.

Why: the TPU logits path pads rows to a power of two and returns the
     ``logits[:bs]`` view; penalizing that view in place makes eager TPU rewrite
     the whole padded base per op on every decode step whose batch is not a
     power of two. See sglang_tpu/sampling/padded_logits.py.
How: REPLACE the class BatchedRepetitionPenalizer with PaddedRepetitionPenalizer
     (a subclass whose state is padded with rows of ones). AROUND
     ScheduleBatch.cumulate_penalty_output_tokens: pad the last-token ids on the
     host to the same rows when every prepared penalizer is the padded one.
     AROUND SamplingBatchInfo.update_penalties: skip the all-zeros additive
     accumulator when no additive penalizer is prepared. AROUND
     SamplingBatchInfo._apply_pre_grammar_logits_transforms: apply the padded
     scaling accumulator to the logits' rows only. AROUND ModelRunner.sample:
     plain greedy decode samples on the padded rows, everything else calls the
     original.
Upstream PR or issue: not filed. A seam would let a platform declare the logits
     row bucket so that the penalizer state and the sampler follow it.
Remove when: core sampling follows a platform-declared logits row bucket, or
     the TPU logits path stops returning a padded view.
"""

from __future__ import annotations

import torch

from sglang.srt.utils.common import is_pin_memory_available
from sglang_tpu.envs import envs
from sglang_tpu.row_pad import row_bucket
from sglang_tpu.sampling.padded_logits import (
    PaddedRepetitionPenalizer,
    padded_greedy_sample,
)


def cumulate_penalty_output_tokens(original, batch):
    orchestrator = batch.sampling_info.penalizer_orchestrator
    rows = len(batch.reqs)
    pad = row_bucket(rows) - rows
    if pad == 0 or not all(
        isinstance(p, PaddedRepetitionPenalizer) or not p.is_prepared()
        for p in orchestrator.penalizers.values()
    ):
        return original(batch)
    # The original's ids plus pad ids; pad rows scatter their own 1.0 into column 0.
    last_tokens = [
        req.output_ids[-1] if len(req.output_ids) else req.origin_input_ids[-1]
        for req in batch.reqs
    ] + [0] * pad
    latest_output_ids = torch.tensor(
        last_tokens,
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(batch.device),
    ).to(batch.device, non_blocking=True)
    orchestrator.cumulate_output_tokens(latest_output_ids)


def update_penalties(original, info):
    orchestrator = info.penalizer_orchestrator
    if orchestrator.is_required and not any(
        p.is_prepared() and not p.is_multiplicative
        for p in orchestrator.penalizers.values()
    ):
        info.acc_additive_penalties = None
        info.acc_scaling_penalties = orchestrator.accumulate_scaling_penalties()
        return
    original(info)


def apply_pre_grammar_logits_transforms(original, info, logits):
    scaling = info.acc_scaling_penalties
    if scaling is None or scaling.shape[0] == logits.shape[0]:
        return original(info, logits)
    info.acc_scaling_penalties = scaling[: logits.shape[0]]
    try:
        return original(info, logits)
    finally:
        info.acc_scaling_penalties = scaling


def sample(original, model_runner, logits_output, forward_batch, *args, **kwargs):
    next_token_ids = padded_greedy_sample(model_runner, logits_output, forward_batch)
    if next_token_ids is not None:
        return next_token_ids
    return original(model_runner, logits_output, forward_batch, *args, **kwargs)


def hook_specs():
    if not envs.SGLANG_TPU_PADDED_LOGITS_SAMPLING.get():
        return []

    from sglang.srt.plugins.hook_registry import HookType

    sampling_batch_info = "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo"
    return [
        (
            "sglang.srt.sampling.penaltylib.repetition_penalty."
            "BatchedRepetitionPenalizer",
            PaddedRepetitionPenalizer,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.managers.schedule_batch.ScheduleBatch."
            "cumulate_penalty_output_tokens",
            cumulate_penalty_output_tokens,
            HookType.AROUND,
        ),
        (
            f"{sampling_batch_info}.update_penalties",
            update_penalties,
            HookType.AROUND,
        ),
        (
            f"{sampling_batch_info}._apply_pre_grammar_logits_transforms",
            apply_pre_grammar_logits_transforms,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.model_runner.ModelRunner.sample",
            sample,
            HookType.AROUND,
        ),
    ]
