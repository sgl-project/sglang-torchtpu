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
    verify_hidden_size: int = 0


def make_future_map_device_accept_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    candidate_width = 2 * context.num_verify_tokens - 1
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


def make_direct_accept_cache_args(context: SpecOpWarmupContext) -> tuple:
    return (
        torch.zeros(
            (context.physical_bs, 2 * context.num_verify_tokens - 1),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.ones(context.physical_bs, dtype=torch.int32, device=context.device),
        context.num_verify_tokens,
    )


def make_future_map_device_accept_hidden_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    common = make_future_map_device_accept_args(context)
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
            (num_verify_rows,),
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
) -> tuple[torch.Tensor | int, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
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
        torch.full(
            (context.physical_bs,),
            context.num_verify_tokens,
            dtype=torch.int32,
            device=context.device,
        ),
        context.num_verify_tokens,
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


def make_extend_continuation_args(context: SpecOpWarmupContext) -> tuple:
    return (
        torch.zeros(
            context.physical_bs * context.num_verify_tokens,
            dtype=torch.int32,
            device=context.device,
        ),
        torch.ones(context.physical_bs, dtype=torch.int32, device=context.device),
        torch.ones(context.physical_bs, dtype=torch.int64, device=context.device),
        context.num_verify_tokens,
        torch.zeros(
            (
                context.physical_bs * context.num_verify_tokens,
                context.verify_hidden_size or context.hidden_size,
            ),
            dtype=context.hidden_dtype,
            device=context.device,
        ),
    )
