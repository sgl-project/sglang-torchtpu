"""Owned physical tensors carried between TPU speculative execution phases."""

import msgspec
import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_op_warmup import make_extend_continuation_args


class VerifyContinuation(msgspec.Struct, frozen=True):
    logical_batch_size: int
    physical_batch_size: int
    hidden_states: torch.Tensor
    predict: torch.Tensor
    accept_lens: torch.Tensor
    seq_lens: torch.Tensor
    bonus_tokens: torch.Tensor


class PaddedResultCopy(msgspec.Struct, frozen=True):
    logical_batch_size: int
    tokens_per_request: int
    predict: torch.Tensor
    accept_lens: torch.Tensor

    def copy_to_cpu(self, copy_tensor):
        # Copy producer tensors directly; materializing a TPU slice here can
        # block submission behind the already queued draft extension.
        predict = copy_tensor(self.predict)
        accept_lens = copy_tensor(self.accept_lens)
        return (
            predict[: self.logical_batch_size * self.tokens_per_request],
            accept_lens[: self.logical_batch_size],
        )


@compiled_op(warmup=make_extend_continuation_args)
def prepare_extend_continuation_tpu(
    predict, accept_lens, seq_lens, width, hidden_states
):
    # Real decode requests have a nonempty prefix. Padding has zero KV length.
    valid = seq_lens > 0
    counts = torch.where(valid, accept_lens, 0)
    token_valid = valid[:, None].expand(-1, width).reshape(-1)
    return (
        torch.where(token_valid, predict.to(torch.int64), 0),
        counts,
        torch.where(valid, counts - 1, 0),
        torch.where(valid, seq_lens + width, 0),
        torch.where(token_valid[:, None], hidden_states, 0),
    )


def install_verify_continuation_tpu(forward_batch, continuation, width):
    """Replace execution inputs without changing the scheduler's logical views."""
    predict, counts, correct, seq_lens, hidden_states = prepare_extend_continuation_tpu(
        continuation.predict,
        continuation.accept_lens,
        continuation.seq_lens,
        width,
        continuation.hidden_states,
    )
    forward_batch.input_ids = predict
    forward_batch.seq_lens = seq_lens[: continuation.logical_batch_size]
    forward_batch.spec_info.hidden_states = hidden_states
    forward_batch.spec_info.num_accept_tokens = counts
    forward_batch.spec_info.num_correct_drafts = correct


class DraftContinuation(msgspec.Struct, frozen=True):
    logical_batch_size: int
    physical_batch_size: int
    hidden_states: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    bonus_tokens: torch.Tensor
    accept_lens: torch.Tensor


def install_draft_continuation_tpu(draft_input, physical_bs):
    state = draft_input.tpu_draft_continuation
    draft_input.tpu_draft_continuation = None
    if state is None or state.physical_batch_size != physical_bs:
        return
    draft_input.hidden_states = state.hidden_states
    draft_input.topk_p = state.topk_p
    draft_input.topk_index = state.topk_index
    draft_input.bonus_tokens = state.bonus_tokens.to(draft_input.bonus_tokens.dtype)


def can_continue_padded_spec_tpu(batch):
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.runtime_context import get_spec

    return (
        batch.enable_overlap
        and batch.spec_algorithm.is_eagle3()
        and get_spec().speculative_eagle_topk == 1
        and not get_spec().speculative_use_rejection_sampling
        and not is_dp_attention_enabled()
        and not batch.forward_mode.is_idle()
        and not batch.return_logprob
        and not batch.has_grammar
        and not any(req.lora_id is not None for req in batch.reqs)
    )
