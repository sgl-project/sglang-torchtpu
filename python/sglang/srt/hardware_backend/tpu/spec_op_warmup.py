"""Warm-up data recipes for compiled speculative-decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.compilation.compiled_op import CompiledOpWarmup


@dataclass(frozen=True)
class SpecOpWarmupContext:
    """Shape and dtype metadata used by speculative helper warm-up recipes."""

    device: torch.device
    physical_bs: int
    num_verify_tokens: int
    vocab_size: int
    verify_logits_dtype: torch.dtype
    draft_logits_dtype: torch.dtype
    hidden_size: int
    hidden_dtype: torch.dtype


def generate_verify_tree_greedy_warmup(
    context: SpecOpWarmupContext,
) -> CompiledOpWarmup:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return CompiledOpWarmup(
        args=(
            torch.zeros(
                (num_verify_rows, context.vocab_size),
                dtype=context.verify_logits_dtype,
                device=context.device,
            ),
            torch.zeros(
                (context.physical_bs, context.num_verify_tokens),
                dtype=torch.int64,
                device=context.device,
            ),
            torch.arange(
                num_verify_rows, dtype=torch.int64, device=context.device
            ).view(context.physical_bs, context.num_verify_tokens),
            1,
        )
    )


def generate_verify_post_warmup(
    context: SpecOpWarmupContext,
) -> CompiledOpWarmup:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return CompiledOpWarmup(
        args=(
            torch.zeros(
                (context.physical_bs,), dtype=torch.int64, device=context.device
            ),
            torch.ones(
                (context.physical_bs,), dtype=torch.int32, device=context.device
            ),
            torch.zeros((num_verify_rows,), dtype=torch.int32, device=context.device),
            torch.arange(
                num_verify_rows, dtype=torch.int32, device=context.device
            ).view(context.physical_bs, context.num_verify_tokens),
            context.num_verify_tokens,
        )
    )


def generate_draft_extend_select_argmax_warmup(
    context: SpecOpWarmupContext,
) -> CompiledOpWarmup:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    select_index = (
        torch.arange(context.physical_bs, dtype=torch.int64, device=context.device)
        * context.num_verify_tokens
        + context.num_verify_tokens
        - 1
    )
    return CompiledOpWarmup(
        args=(
            torch.zeros(
                (num_verify_rows, context.vocab_size),
                dtype=context.draft_logits_dtype,
                device=context.device,
            ),
            torch.zeros(
                (num_verify_rows, context.hidden_size),
                dtype=context.hidden_dtype,
                device=context.device,
            ),
            select_index,
        )
    )
