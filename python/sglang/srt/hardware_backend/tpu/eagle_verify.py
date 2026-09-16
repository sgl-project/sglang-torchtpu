"""Owned verification outputs for the TPU EAGLE top-k-one path."""

from functools import lru_cache

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_verify_tree_greedy_args,
)


@lru_cache(maxsize=32)
def _simulate_count(value: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int32, device=device)


def _make_verify_args(context):
    return (
        *make_verify_tree_greedy_args(context),
        torch.ones(context.physical_bs, dtype=torch.int64, device=context.device),
        _simulate_count(0, context.device),
    )


@compiled_op(warmup=_make_verify_args)
def verify_and_bookkeep_tpu(
    logits, candidates, retrieve_index, seq_lens, simulate_count
):
    from sglang.srt.speculative.eagle_utils import (
        _verify_tree_greedy_torch,
        fused_verify_bookkeeping,
    )

    predict, accept_index, accept_lens = _verify_tree_greedy_torch.__wrapped__(
        logits, candidates, retrieve_index
    )
    bs, width = retrieve_index.shape
    columns = torch.arange(width, device=logits.device)
    # Positive counts force real draft tokens; negative counts use fixed tokens.
    forced = simulate_count.abs()
    simulated_index = torch.where(
        columns[None, :] < forced,
        accept_index[:, :1] + columns[None, :],
        -1,
    ).to(torch.int32)
    draft_tokens = candidates.reshape(bs, width)
    next_columns = (columns + 1).clamp_max(width - 1)
    simulated_predict = torch.where(
        columns[None, :] < forced - 1,
        draft_tokens[:, next_columns].to(predict.dtype),
        predict.reshape(bs, width),
    ).flatten()
    simulated_predict = torch.where(simulate_count < 0, 100, simulated_predict)
    predict = torch.where(forced > 0, simulated_predict, predict)
    accept_lens = torch.where(forced > 0, forced, accept_lens)
    accept_index = torch.where(forced > 0, simulated_index, accept_index)
    new_seq_lens, bonus_tokens = fused_verify_bookkeeping.__wrapped__(
        seq_lens, accept_lens, predict, accept_index
    )
    return predict, accept_lens, accept_index, new_seq_lens, bonus_tokens


def try_verify_and_bookkeep_tpu(*, batch, verify_input, logits_output, grammar_mask):
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        SIMULATE_ACC_METHOD,
        SIMULATE_ACC_TOKEN_MODE,
        sample_simulated_acc_len,
    )
    from sglang.srt.utils.async_probe import sanitize_nan_logits

    sampling = batch.sampling_info
    if (
        batch.forward_mode.is_idle()
        or not sampling.is_all_greedy
        or verify_input.tree_topk != 1
        or get_spec().speculative_use_rejection_sampling
        or grammar_mask is not None
        or batch.return_logprob
        or sampling.acc_additive_penalties is not None
        or sampling.acc_scaling_penalties is not None
        or sampling.logit_bias is not None
    ):
        return None
    state = verify_input.padding_state
    if SIMULATE_ACC_LEN > 0 and (
        SIMULATE_ACC_TOKEN_MODE not in ("fixed", "real-draft-token")
        or state is not None
        and state.logical_batch_size != state.physical_batch_size
    ):
        return None
    logits = logits_output.next_token_logits
    sanitize_nan_logits(logits, "verify: target model logits")
    count = 0
    if SIMULATE_ACC_LEN > 0:
        count = sample_simulated_acc_len(
            SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, verify_input.max_tree_depth
        )
        if SIMULATE_ACC_TOKEN_MODE == "fixed":
            count = -count
    return verify_and_bookkeep_tpu(
        logits,
        verify_input.draft_token,
        verify_input.retrieve_index,
        batch.seq_lens,
        _simulate_count(count, logits.device),
    )
