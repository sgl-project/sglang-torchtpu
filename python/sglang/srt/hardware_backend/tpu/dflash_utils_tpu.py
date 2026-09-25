"""TPU helpers shared by the DFlash-family draft loops (DFLASH, DSpark)."""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
from torch import nn

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.distributed import get_tp_group, model_parallel_is_initialized
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_dflash_verify_argmax_args,
    make_dflash_verify_bookkeeping_args,
)
from sglang.srt.hardware_backend.tpu.top_k import (
    TpuVocabShard,
    distributed_tpu_top_k,
    supports_distributed_tpu_top_k,
)


class TpuDraftExtendMetadata(msgspec.Struct, frozen=True):
    """Extend split for a fixed-width draft block.

    The Pallas RPA backend plans the block as an extend from the host, so the
    draft forward carries the split explicitly. Off TPU every field is None;
    see EMPTY_TPU_DRAFT_EXTEND.

    Attributes:
      seq_lens: Post-block sequence lengths, [bs] on device, dtype of the
        draft_prefix_lens the builder was handed.
      extend_prefix_lens: Pre-block sequence lengths, [bs] int32 on device.
      extend_seq_lens: Rows added per request, [bs] int32 on device, every
        entry equal to width.
      extend_prefix_lens_cpu: Host mirror of extend_prefix_lens, length bs.
      extend_seq_lens_cpu: Host mirror of extend_seq_lens, length bs.
      extend_num_tokens: Total rows in the block, bs * width.
    """

    seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[list[int]] = None
    extend_seq_lens_cpu: Optional[list[int]] = None
    extend_num_tokens: Optional[int] = None


EMPTY_TPU_DRAFT_EXTEND = TpuDraftExtendMetadata()


def build_tpu_draft_extend_metadata(
    *,
    bs: int,
    width: int,
    draft_prefix_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_seq_lens_buf: Optional[torch.Tensor] = None,
) -> TpuDraftExtendMetadata:
    """Build the draft block's extend metadata.

    Args:
      bs: Number of requests in the block.
      width: Draft rows per request: DFLASH block_size, DSpark query_token_num.
      draft_prefix_lens: Pre-block sequence lengths, [bs] on device.
      seq_lens_cpu: Host mirror of draft_prefix_lens, [bs] on CPU.
      extend_seq_lens_buf: Optional [cap_bs] int32 device buffer with
        cap_bs >= bs, filled in place instead of allocating every step.

    Returns:
      The block's TpuDraftExtendMetadata.

    Raises:
      RuntimeError: seq_lens_cpu is None, which would force a per-step D2H.
    """
    if seq_lens_cpu is None:
        # No Pallas backend opts out of needs_cpu_seq_lens, so FutureMap always
        # publishes the host mirror on TPU. None means that broke; deriving it
        # here would silently add a per-step seq_lens D2H.
        raise RuntimeError(
            "TPU draft extend metadata requires the host seq_lens mirror, got None."
        )
    extend_prefix_lens = draft_prefix_lens.to(torch.int32)
    if extend_seq_lens_buf is not None:
        extend_seq_lens = extend_seq_lens_buf[:bs]
        extend_seq_lens.fill_(width)
    else:
        extend_seq_lens = torch.full(
            (bs,), width, dtype=torch.int32, device=draft_prefix_lens.device
        )
    return TpuDraftExtendMetadata(
        seq_lens=draft_prefix_lens + width,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens_cpu=seq_lens_cpu.tolist(),
        extend_seq_lens_cpu=[width] * bs,
        extend_num_tokens=bs * width,
    )


def correct_drafts_and_bonus_tpu(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy accept rule, written for TPU.

    Same result as the generic path, reached with a masked sum instead of its
    cumprod and its advanced-index gather of bonus.

    Args:
      candidates: Draft-proposed token ids, [bs, block_size] int; column 0 is
        the current token, not a draft.
      target_predict: Target argmax per position, [bs, block_size] int.

    Returns:
      Tuple of correct_len and bonus, both [bs]. correct_len counts accepted
      draft tokens and is int64 when block_size > 1 but int32 when it is 1, so
      callers normalize; bonus carries target_predict's dtype and holds the
      target token at index correct_len.
    """
    bs, block_size = candidates.shape
    if block_size > 1:
        matches = candidates[:, 1:] == target_predict[:, :-1]
        all_match = matches.all(dim=1)
        first_mismatch = (~matches).to(torch.int32).argmax(dim=1)
        correct_len = torch.where(all_match, block_size - 1, first_mismatch)
    else:
        correct_len = torch.zeros((bs,), dtype=torch.int32, device=candidates.device)

    col_idx = torch.arange(block_size, device=candidates.device).unsqueeze(0)
    bonus_mask = col_idx == correct_len.to(torch.int64).unsqueeze(1)
    bonus = torch.where(
        bonus_mask, target_predict, torch.zeros_like(target_predict)
    ).sum(dim=1)
    return correct_len, bonus


@compiled_op(warmup=make_dflash_verify_argmax_args)
def dflash_verify_argmax_torch(
    next_token_logits: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Greedy target prediction for the verify block.

    Reshapes inside the region: a view handed in from outside is materialized by
    an as_strided at the executable boundary, itself JIT-compiled per shape.

    Args:
      next_token_logits: Target logits, [bs * block_size, vocab] float.
      block_size: Draft rows per request.

    Returns:
      Argmax token ids, [bs, block_size] int64.
    """
    return torch.argmax(
        next_token_logits.view(-1, block_size, next_token_logits.shape[-1]), dim=-1
    )


@compiled_op(warmup=make_dflash_verify_bookkeeping_args)
def dflash_verify_bookkeeping_torch(
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused DFlash accept/bonus/out_tokens bookkeeping on TPU.

    Args:
      candidates: Draft-proposed token ids, [bs, block_size] int; column 0 is
        the current token, not a draft.
      target_predict: Target argmax per position, [bs, block_size] int.
      prefix_lens: Pre-block sequence lengths, [bs].

    Returns:
      Tuple of accept_len [bs] int32, commit_lens [bs] int32 (accept_len + 1),
      bonus [bs] in target_predict's dtype, out_tokens [bs, block_size] int64
      zero-padded past the accepted run, and new_seq_lens [bs] in prefix_lens'
      dtype.
    """
    bs, block_size = candidates.shape
    device = candidates.device

    if block_size > 1:
        matches = candidates[:, 1:] == target_predict[:, :-1]
        all_match = matches.all(dim=1)
        first_mismatch = (~matches).to(torch.int32).argmax(dim=1)
        # argmax yields int64; int32 keeps commit_lens matching the eager/Triton
        # paths, whose dtype the compiled draft KV appender guards on.
        accept_len = torch.where(all_match, block_size - 1, first_mismatch).to(
            torch.int32
        )
    else:
        accept_len = torch.zeros((bs,), dtype=torch.int32, device=device)

    commit_lens = accept_len + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)

    col_idx = torch.arange(block_size, device=device).unsqueeze(0)
    accept_len_2d = accept_len.to(torch.int64).unsqueeze(1)
    bonus_mask = col_idx == accept_len_2d
    bonus = torch.where(
        bonus_mask,
        target_predict,
        torch.zeros_like(target_predict),
    ).sum(dim=1)

    draft_tokens = torch.zeros_like(candidates)
    if block_size > 1:
        draft_tokens[:, :-1] = candidates[:, 1:]
    out_tokens = torch.where(
        col_idx < accept_len_2d,
        draft_tokens,
        torch.where(
            col_idx == accept_len_2d,
            target_predict,
            torch.zeros_like(candidates),
        ),
    ).to(torch.int64)

    return accept_len, commit_lens, bonus, out_tokens, new_seq_lens


def argmax_draft_tokens_tpu(
    *,
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Greedy draft tokens from the target lm_head, for the eager draft tail.

    Reads block_size - 1 rows per request: position 0 is the anchor token the
    caller already has.

    Args:
      lm_head: Target head; weight is [vocab_shard, hidden]. Under TP the shard
        is local and shard_indices carries the global offset.
      hidden_states: Draft hidden states, [bs * block_size, hidden].
      block_size: Draft rows per request.

    Returns:
      Argmax token ids in the global vocab, [bs, block_size - 1] int64.
    """
    hs_3d = hidden_states.view(-1, block_size, hidden_states.shape[-1])
    bs = hs_3d.shape[0]
    hs_1d = hs_3d[:, 1:, :].reshape(-1, hs_3d.shape[-1])
    hs = (
        hs_1d if hs_1d.dtype == lm_head.weight.dtype else hs_1d.to(lm_head.weight.dtype)
    )

    tp_size = int(get_tp_group().world_size) if model_parallel_is_initialized() else 1
    if tp_size > 1 and supports_distributed_tpu_top_k(lm_head, top_k=1):
        local_logits = torch.matmul(hs, lm_head.weight.T)
        _, global_token_ids = distributed_tpu_top_k(
            local_logits,
            TpuVocabShard.from_lm_head(lm_head),
            k=1,
            group=get_tp_group(),
            lm_head=lm_head,
        )
        tokens = global_token_ids.squeeze(-1).to(torch.long)
        return tokens.view(bs, block_size - 1)

    shard = getattr(lm_head, "shard_indices", None)
    num_org = int(shard.num_org_elements) if shard is not None else 0
    org_vocab_start = int(shard.org_vocab_start_index) if shard is not None else 0
    if num_org > 0:
        local_logits = torch.matmul(hs, lm_head.weight[:num_org].T)
    else:
        local_logits = torch.matmul(hs, lm_head.weight.T)

    tokens = torch.argmax(local_logits, dim=-1).to(torch.long)
    if org_vocab_start > 0:
        tokens.add_(org_vocab_start)
    return tokens.view(bs, block_size - 1)
