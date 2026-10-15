"""TPU helpers shared by the DFlash-family draft loops (DFLASH, DSpark).

Also hosts device-resident verify-block addressing. The host trails device
acceptance by one commit, so kv_committed_len is an exact base: the true prefix
is base + commit_lens with commit_lens in [0, block_size]. That bound lets the
host pre-gather every slot the next verify block could occupy and leave the
choice to the device. page_aligned_decode_alloc_lens already reserves
2 * block_size for the lag, so the window costs no extra allocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch
from torch import nn

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.distributed import model_parallel_is_initialized
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.experimental import is_experimental_spec
from sglang_tpu.spec_op_warmup import (
    make_dflash_verify_argmax_args,
    make_dflash_verify_bookkeeping_args,
    make_dflash_verify_cache_locs_args,
)
from sglang_tpu.top_k import (
    TpuVocabShard,
    distributed_tpu_top_k,
    supports_distributed_tpu_top_k,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class TpuDraftExtendMetadata(msgspec.Struct, frozen=True):
    """Extend split for a fixed-width draft block.

    The Pallas RPA backend plans the block as an extend from the host, so the
    draft forward carries the split explicitly; it is built on TPU only.

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

    tp_size = (
        int(get_parallel().tp_group.world_size)
        if model_parallel_is_initialized()
        else 1
    )
    if tp_size > 1 and supports_distributed_tpu_top_k(lm_head, top_k=1):
        local_logits = torch.matmul(hs, lm_head.weight.T)
        _, global_token_ids = distributed_tpu_top_k(
            local_logits,
            TpuVocabShard.from_lm_head(lm_head),
            k=1,
            group=get_parallel().tp_group,
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


def _dflash_device_accept_eligible(
    batch: ScheduleBatch, bs: int, *, compact_draft_cache: bool
) -> bool:
    """Whether this batch may address its verify block without a host sync.

    DFLASH (and the experimental TpuSpecWorker chains, same block) commits at
    most one block, which is the single bounded offset the window enumerates.
    Grammar is excluded because grammar_retained_tokens can
    be shorter than commit_lens, which breaks that bound. The compact draft
    cache rebuilds its own windowed mapping from host lengths and is left on
    the synchronizing path.

    Args:
      batch: Batch being prepared for decode.
      bs: Number of live requests in the batch.
      compact_draft_cache: Whether the windowed draft KV cache is enabled.

    Returns:
      True when the device may pick the verify block on its own.
    """
    return (
        bs > 0
        and not compact_draft_cache
        and batch.enable_overlap
        and (
            batch.spec_algorithm.is_dflash()
            or is_experimental_spec(batch.spec_algorithm)
        )
        and batch.req_to_token_pool.host_resident
        and not batch.return_logprob
        and not batch.has_grammar
        and all(req.lora_id is None for req in batch.reqs)
    )


def _select_verify_cache_locs(
    candidates: torch.Tensor,
    seq_lens: torch.Tensor,
    committed_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Slice the block starting at seq_lens - committed_lens out of candidates.

    Args:
      candidates: Slot ids for [base, base + 2 * block_size),
        [bs, 2 * block_size].
      seq_lens: Device prefix lengths, [bs].
      committed_lens: Device mirror of the host base, [bs].
      block_size: Verify block width.

    Returns:
      Slot ids, [bs, block_size].
    """
    device = candidates.device
    # Binds only on the uninitialized bucket-padding rows; live rows stay inside
    # [0, block_size] by the one-commit lag.
    offsets = (seq_lens.to(torch.int64) - committed_lens.to(torch.int64)).clamp_(
        0, block_size
    )
    columns = offsets[:, None] + torch.arange(
        block_size, device=device, dtype=torch.int64
    )
    rows = torch.arange(candidates.shape[0], device=device, dtype=torch.int64)
    return candidates[rows[:, None], columns]


@compiled_op(warmup=make_dflash_verify_cache_locs_args)
def select_dflash_verify_cache_locs_tpu(
    candidates: torch.Tensor,
    seq_lens: torch.Tensor,
    committed_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Pick the verify block's slots out of the pre-gathered candidate window.

    Args:
      candidates: Slot ids for [base, base + 2 * block_size),
        [bs, 2 * block_size].
      seq_lens: Device prefix lengths, [bs], equal to base plus the previous
        step's commit length.
      committed_lens: Device mirror of the host base, [bs].
      block_size: Verify block width.

    Returns:
      Slot ids, [bs, block_size].
    """
    return _select_verify_cache_locs(candidates, seq_lens, committed_lens, block_size)


class DeviceVerifyBlockSelector:
    """Bucket-padded driver for select_dflash_verify_cache_locs_tpu.

    The op is specialized per physical batch bucket, so live rows are staged into
    buffers sized to that bucket. Rows past the live batch stay uninitialized:
    the selection clamps its offsets, so they only ever produce in-bounds garbage
    that the caller trims off.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        block_size: int,
        pad_bs: Callable[[int], int],
    ) -> None:
        """Set up the staging buffers for one decode loop.

        Args:
          device: Device the staging buffers live on.
          block_size: Verify block width.
          pad_bs: Maps a live batch size to its compiled bucket size.
        """
        self.block_size = int(block_size)
        self._device = device
        self._pad_bs = pad_bs
        self._candidates: Optional[torch.Tensor] = None
        self._committed_lens: Optional[torch.Tensor] = None
        self._prefix_lens: Optional[torch.Tensor] = None

    def select(
        self,
        *,
        candidates: torch.Tensor,
        committed_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        bs: int,
    ) -> torch.Tensor:
        """Pick each request's verify block out of its candidate window.

        Args:
          candidates: Slot ids for [base, base + 2 * block_size),
            [bs, 2 * block_size].
          committed_lens: Device mirror of the host base, [bs].
          prefix_lens: Device prefix lengths, [bs].
          bs: Number of live requests.

        Returns:
          Slot ids, [bs, block_size].
        """
        pad_bs = self._pad_bs(bs)
        self._ensure_capacity(pad_bs)
        padded_candidates = self._candidates[:pad_bs]
        padded_committed = self._committed_lens[:pad_bs]
        padded_prefix = self._prefix_lens[:pad_bs]
        padded_candidates[:bs].copy_(candidates)
        padded_committed[:bs].copy_(committed_lens)
        padded_prefix[:bs].copy_(prefix_lens)
        return select_dflash_verify_cache_locs_tpu(
            padded_candidates, padded_prefix, padded_committed, self.block_size
        )[:bs]

    def _ensure_capacity(self, pad_bs: int) -> None:
        """Grow the staging buffers to hold pad_bs rows.

        Args:
          pad_bs: Bucket-padded batch size the next select needs.
        """
        # The capacity is itself a bucket, so pad_bs <= cap holds for every later
        # bs <= cap and buf[:pad_bs] never overruns; the ladder bounds regrowth.
        cap = 0 if self._candidates is None else int(self._candidates.shape[0])
        if cap >= pad_bs:
            return
        self._candidates = torch.empty(
            (pad_bs, 2 * self.block_size), dtype=torch.int64, device=self._device
        )
        self._committed_lens = torch.empty(
            (pad_bs,), dtype=torch.int64, device=self._device
        )
        self._prefix_lens = torch.empty(
            (pad_bs,), dtype=torch.int64, device=self._device
        )


def gather_dflash_candidate_window(
    req_to_token: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    committed_lens_cpu: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Host-gather the slots the next verify block can land on.

    The window spans [base, base + 2 * block_size): offset 0 covers an extend or
    idle predecessor that committed nothing, offset block_size the maximal
    commit, and each start needs block_size slots.

    Args:
      req_to_token: Host req-to-slot mapping, [max_num_reqs, max_context_len].
      req_pool_indices_cpu: Pool row of each request, [bs] on CPU.
      committed_lens_cpu: Host committed prefix lengths, [bs] on CPU.
      block_size: Verify block width.

    Returns:
      Slot ids spanning the window, [bs, 2 * block_size].
    """
    rows = req_pool_indices_cpu.to(torch.int64)[:, None]
    columns = committed_lens_cpu.to(torch.int64)[:, None] + torch.arange(
        2 * block_size, dtype=torch.int64
    )
    return req_to_token[rows, columns]


class DFlashDeviceAcceptPlan(msgspec.Struct, frozen=True):
    """Device-resident verify addressing for one decode step.

    Attributes:
      verify_cache_candidates: Slot ids for [base, base + 2 * block_size),
        [bs, 2 * block_size] int64 on device.
      committed_lens: Device mirror of the host base, [bs] int64.
      req_pool_indices: TpuSpecWorker only, [bs] int64 pool rows. When set, every
        field stays on host; the worker pads and uploads them at the batch
        bucket of the forward, which DP attention fixes only after the plan.
    """

    verify_cache_candidates: torch.Tensor
    committed_lens: torch.Tensor
    req_pool_indices: Optional[torch.Tensor] = None


def plan_dflash_device_accept(
    batch: ScheduleBatch,
    *,
    bs: int,
    block_size: int,
    committed_lens_cpu: torch.Tensor,
    compact_draft_cache: bool,
) -> Optional[DFlashDeviceAcceptPlan]:
    """Pre-gather every slot the next verify block could occupy.

    The published length is base + commit_lens of the previous step, which only
    the device knows. Enumerating the bounded window here lets the worker select
    on device instead of synchronizing on that length.

    Args:
      batch: Batch being prepared for decode.
      bs: Number of live requests in the batch.
      block_size: Verify block width.
      committed_lens_cpu: Host committed prefix lengths, [bs] on CPU.
      compact_draft_cache: Whether the windowed draft KV cache is enabled.

    Returns:
      The window and its base, or None when the batch must stay on the
      synchronizing path.

    Raises:
      AssertionError: A request reserved fewer than two blocks past its base.
    """
    if not _dflash_device_accept_eligible(
        batch, bs, compact_draft_cache=compact_draft_cache
    ):
        return None
    req_pool_indices_cpu = batch.req_pool_indices_cpu
    if not isinstance(req_pool_indices_cpu, torch.Tensor):
        return None
    # page_aligned_decode_alloc_lens reserved 2 * block_size from the same
    # committed base, so the whole window is already mapped.
    assert all(
        int(committed_lens_cpu[i]) + 2 * block_size <= req.kv.kv_allocated_len
        for i, req in enumerate(batch.reqs)
    ), "DFLASH device accept routing requires two blocks of reserved KV"
    candidates = gather_dflash_candidate_window(
        batch.req_to_token_pool.req_to_token,
        req_pool_indices_cpu[:bs],
        committed_lens_cpu[:bs],
        block_size,
    )
    if is_experimental_spec(batch.spec_algorithm):
        # TpuSpecWorker uploads the plan at its forward's batch bucket, which DP
        # attention shares across ranks (TpuSpecVerifyInput.from_batch).
        return DFlashDeviceAcceptPlan(
            verify_cache_candidates=candidates.to(torch.int64),
            committed_lens=committed_lens_cpu[:bs].to(torch.int64),
            req_pool_indices=req_pool_indices_cpu[:bs].to(torch.int64),
        )
    return DFlashDeviceAcceptPlan(
        verify_cache_candidates=candidates.to(
            batch.device, dtype=torch.int64, non_blocking=True
        ),
        committed_lens=committed_lens_cpu[:bs].to(
            batch.device, dtype=torch.int64, non_blocking=True
        ),
    )
