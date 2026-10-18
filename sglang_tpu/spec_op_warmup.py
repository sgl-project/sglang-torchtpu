"""Warm-up data recipes for compiled speculative-decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# simulate_mode of eagle_utils_tpu.greedy_verify_chain_tpu: off, fixed tokens,
# real draft tokens.
SIM_OFF, SIM_FIXED, SIM_REAL = 0, 1, 2


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
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int64,
            device=context.device,
        ),
    )


def make_greedy_verify_chain_args(
    context: SpecOpWarmupContext,
) -> tuple:
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        SIMULATE_ACC_TOKEN_MODE,
    )

    if SIMULATE_ACC_LEN <= 0:
        simulate_mode = SIM_OFF
    elif SIMULATE_ACC_TOKEN_MODE == "fixed":
        simulate_mode = SIM_FIXED
    else:
        simulate_mode = SIM_REAL
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return (
        torch.zeros(
            (num_verify_rows, context.vocab_size),
            dtype=torch.float32,
            device=context.device,
        ),
        torch.zeros((num_verify_rows,), dtype=torch.int64, device=context.device),
        # Built on the host: a device view of a 1-D arange keeps the 1-D
        # buffer layout, so every call would relayout it.
        torch.arange(num_verify_rows, dtype=torch.int64)
        .view(context.physical_bs, context.num_verify_tokens)
        .to(context.device),
        torch.ones((1,), dtype=torch.int32, device=context.device),
        torch.zeros((context.physical_bs,), dtype=torch.int64, device=context.device),
        simulate_mode,
    )


def make_device_accept_select_args(
    context: SpecOpWarmupContext,
) -> tuple:
    width = context.num_verify_tokens
    return (
        torch.zeros(
            (context.physical_bs, 2 * width - 1),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.ones((context.physical_bs,), dtype=torch.int32, device=context.device),
        width - 1,  # draft steps of a top-1 chain
    )


def make_verify_kv_lens_args(context: SpecOpWarmupContext) -> tuple:
    return (
        torch.zeros((context.physical_bs,), dtype=torch.int64, device=context.device),
        context.num_verify_tokens,
    )


def make_verify_bookkeeping_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return (
        torch.zeros((context.physical_bs,), dtype=torch.int64, device=context.device),
        torch.ones((context.physical_bs,), dtype=torch.int32, device=context.device),
        torch.zeros((num_verify_rows,), dtype=torch.int32, device=context.device),
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int32,
            device=context.device,
        ),
    )


def make_draft_extend_select_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    # int32 matches `num_correct_drafts` from `eagle_sample`, which the padding
    # helpers preserve; the helper reads the window width off the logits rows.
    num_accept_tokens = torch.full(
        (context.physical_bs,),
        context.num_verify_tokens,
        dtype=torch.int32,
        device=context.device,
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
        num_accept_tokens,
    )


def make_dflash_verify_argmax_args(
    context: SpecOpWarmupContext,
) -> tuple[Any, ...]:
    return (
        torch.zeros(
            (context.physical_bs * context.num_verify_tokens, context.vocab_size),
            dtype=torch.float32,
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


def make_dflash_verify_cache_locs_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor | int, ...]:
    block_size = context.num_verify_tokens
    return (
        torch.zeros(
            (context.physical_bs, 2 * block_size),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.zeros((context.physical_bs,), dtype=torch.int64, device=context.device),
        torch.zeros((context.physical_bs,), dtype=torch.int64, device=context.device),
        block_size,
    )
