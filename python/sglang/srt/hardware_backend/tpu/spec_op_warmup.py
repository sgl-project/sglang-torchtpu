"""Warm-up data recipes for compiled speculative-decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpecOpWarmupContext:
    """Shape and dtype metadata used by speculative helper warm-up recipes."""

    device: torch.device
    physical_bs: int
    num_verify_tokens: int
    vocab_size: int
    draft_vocab_size: int
    hidden_size: int
    hidden_dtype: torch.dtype
    req_pool_size: int
    need_hidden_states: bool


def _make_future_map_device_accept_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    candidate_width = 2 * context.num_verify_tokens - 1
    # The topk relay buffers mirror fused_draft_extend_select_argmax_func on
    # TPU (topk == 1 only): [req_pool_size, 1] float32 probs, int64 indices.
    return (
        torch.arange(context.physical_bs, dtype=torch.int64, device=context.device),
        torch.zeros(
            (context.req_pool_size, 1), dtype=torch.float32, device=context.device
        ),
        torch.zeros(
            (context.req_pool_size, 1), dtype=torch.int64, device=context.device
        ),
        torch.zeros((context.req_pool_size,), dtype=torch.int64, device=context.device),
        torch.zeros(
            (context.physical_bs, candidate_width),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.ones((context.physical_bs,), dtype=torch.int32, device=context.device),
        context.num_verify_tokens,
    )


def make_future_map_device_accept_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...] | None:
    if context.need_hidden_states:
        return None
    return _make_future_map_device_accept_args(context)


def make_future_map_device_accept_hidden_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...] | None:
    if not context.need_hidden_states:
        return None
    common = _make_future_map_device_accept_args(context)
    return (
        *common[:4],
        torch.zeros(
            (context.req_pool_size, context.hidden_size),
            dtype=context.hidden_dtype,
            device=context.device,
        ),
        *common[4:],
    )


def make_verify_tree_greedy_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return (
        torch.zeros(
            (num_verify_rows, context.vocab_size),
            dtype=torch.float32,
            device=context.device,
        ),
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.arange(num_verify_rows, dtype=torch.int64, device=context.device).view(
            context.physical_bs, context.num_verify_tokens
        ),
    )


def make_verify_bookkeeping_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return (
        torch.zeros((context.physical_bs,), dtype=torch.int64, device=context.device),
        torch.ones((context.physical_bs,), dtype=torch.int32, device=context.device),
        torch.zeros((num_verify_rows,), dtype=torch.int32, device=context.device),
        torch.arange(num_verify_rows, dtype=torch.int32, device=context.device).view(
            context.physical_bs, context.num_verify_tokens
        ),
    )


def make_draft_extend_select_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    select_index = (
        torch.arange(context.physical_bs, dtype=torch.int64, device=context.device)
        * context.num_verify_tokens
        + context.num_verify_tokens
        - 1
    )
    return (
        torch.zeros(
            (num_verify_rows, context.draft_vocab_size),
            dtype=torch.float32,
            device=context.device,
        ),
        torch.zeros(
            (num_verify_rows, context.hidden_size),
            dtype=context.hidden_dtype,
            device=context.device,
        ),
        select_index,
    )


def make_dflash_verify_bookkeeping_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.zeros(
            (context.physical_bs,),
            dtype=torch.int64,
            device=context.device,
        ),
    )
