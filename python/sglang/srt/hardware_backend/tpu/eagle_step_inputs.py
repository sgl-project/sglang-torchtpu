"""Physical inputs shared by the three EAGLE execution phases."""

from __future__ import annotations

from typing import NamedTuple

import msgspec
import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_op_warmup import SpecOpWarmupContext


class EagleRelayInputs(msgspec.Struct, frozen=True):
    seq_lens: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    bonus_tokens: torch.Tensor
    hidden_states: torch.Tensor
    cache_locs: torch.Tensor
    logical_seq_lens: torch.Tensor


class EagleStepInputs(NamedTuple):
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    draft_cache_locs: torch.Tensor
    verify_cache_locs: torch.Tensor
    verify_positions: torch.Tensor
    extend_positions: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    bonus_tokens: torch.Tensor
    hidden_states: torch.Tensor
    draft_kv_lens: tuple[torch.Tensor, ...]
    verify_kv_lens: torch.Tensor
    extend_kv_lens: torch.Tensor


def make_eagle_step_inputs_args(context: SpecOpWarmupContext):
    bs, device = context.physical_bs, context.device
    return (
        torch.ones(bs, dtype=torch.int64, device=device),
        torch.ones((bs, 1), dtype=torch.float32, device=device),
        torch.zeros((bs, 1), dtype=torch.int64, device=device),
        torch.zeros(bs, dtype=torch.int64, device=device),
        torch.zeros(
            (bs, context.hidden_size), dtype=context.hidden_dtype, device=device
        ),
        torch.zeros((bs, context.num_verify_tokens), dtype=torch.int64, device=device),
        torch.ones(bs, dtype=torch.int32, device=device),
        torch.tensor(bs, dtype=torch.int32, device=device),
        context.num_verify_tokens,
    )


@compiled_op(warmup=make_eagle_step_inputs_args)
def prepare_eagle_step_inputs(
    seq_lens: torch.Tensor,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    bonus_tokens: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_locs: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    logical_bs: torch.Tensor,
    verify_width: int,
) -> EagleStepInputs:
    active = torch.arange(seq_lens.shape[0], device=seq_lens.device) < logical_bs
    # FutureMap's dummy rows read pool row zero, which need not be initialized.
    seq_lens = torch.where(active, seq_lens, 0)
    cache_locs = torch.where(active[:, None], cache_locs, 0)
    extend_seq_lens = torch.where(active, seq_lens + verify_width, 0)
    offsets = torch.arange(verify_width, dtype=torch.int64, device=seq_lens.device)
    verify_positions = torch.where(
        active[:, None], seq_lens[:, None] + offsets[None, :], 0
    ).flatten()
    extend_positions = torch.where(
        active[:, None],
        extend_prefix_lens.to(torch.int64)[:, None] + offsets[None, :],
        0,
    ).flatten()
    kv_lens = seq_lens.to(torch.int32)
    return EagleStepInputs(
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        draft_cache_locs=cache_locs[:, : verify_width - 1].flatten(),
        verify_cache_locs=cache_locs.flatten(),
        verify_positions=verify_positions,
        extend_positions=extend_positions,
        topk_p=torch.where(active[:, None], topk_p, 0),
        topk_index=torch.where(active[:, None], topk_index, 0),
        bonus_tokens=torch.where(active, bonus_tokens, 0),
        hidden_states=torch.where(active[:, None], hidden_states, 0),
        draft_kv_lens=tuple(kv_lens + i + 1 for i in range(verify_width - 2)),
        verify_kv_lens=(seq_lens + verify_width).to(torch.int32),
        extend_kv_lens=extend_seq_lens.to(torch.int32),
    )
