"""TPU-compiled helpers for the DSpark verify loop.

Run eager, the post-verify accept rule fans out into a dozen separately
dispatched kernels, and torch_tpu compiles each one per distinct batch size.
DsparkVerifyEpilogue folds the same work into the target-verify cuda graph, but
only on CUDA; the graphs here are the TPU counterpart. compiled_op is a
dynamic=False torch.compile, so they run on the physical batch bucket the target
verify already produced, and the caller trims the padded rows off afterwards.
"""

from __future__ import annotations

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.dflash_utils_tpu import (
    select_dflash_verify_cache_locs_tpu,
)
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_dspark_accept_greedy_args,
    make_dspark_finalize_accept_args,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import assign_extend_cache_locs_func


@compiled_op(warmup=make_dspark_accept_greedy_args)
def dspark_accept_greedy_tpu(
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_lens: torch.Tensor,
) -> torch.Tensor:
    """Greedy accept for one verify block, packed for a single TP broadcast.

    Args:
      candidates: Proposed ids including the anchor, shape [bs, width] int64.
      target_logits: Target verify logits, shape [bs * width, vocab].
      verify_lens: Per-request verify budget, shape [bs]; width means no cap.

    Returns:
      correct_len, bonus and cap_trim_lens stacked, shape [3, bs] int64.
    """
    bs, width = candidates.shape
    target_predict = torch.argmax(target_logits, dim=-1).view(bs, width)
    if width > 1:
        matches = candidates[:, 1:] == target_predict[:, :-1]
        first_mismatch = (~matches).to(torch.int32).argmax(dim=1).to(torch.int64)
        correct_len = torch.where(matches.all(dim=1), width - 1, first_mismatch)
    else:
        correct_len = torch.zeros((bs,), dtype=torch.int64, device=candidates.device)
    capped = torch.minimum(correct_len, verify_lens.to(torch.int64) - 1)
    bonus = torch.gather(target_predict, 1, capped.unsqueeze(1)).squeeze(1)
    return torch.stack((capped, bonus, correct_len - capped))


@compiled_op(warmup=make_dspark_finalize_accept_args)
def dspark_finalize_accept_tpu(
    packed: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Expand the packed accept decision into the per-request commit outputs.

    Args:
      packed: correct_len, bonus and cap_trim_lens stacked, shape [3, bs] int64.
      prefix_lens: Pre-verify sequence lengths, shape [bs] int64.
      draft_tokens: Proposed ids excluding the anchor, shape [bs, width - 1].

    Returns:
      correct_len, bonus, cap_trim_lens, commit_lens, new_seq_lens, out_tokens.
    """
    correct_len, bonus, cap_trim_lens = packed[0], packed[1], packed[2]
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    out_tokens = torch.cat(
        (draft_tokens, torch.zeros_like(draft_tokens[:, :1])), dim=1
    ).scatter(1, correct_len.unsqueeze(1), bonus.unsqueeze(1))
    return (
        correct_len.to(torch.int32),
        bonus,
        cap_trim_lens.to(torch.int32),
        commit_lens,
        new_seq_lens,
        out_tokens,
    )


class DsparkTpuAcceptor:
    """Driver for the two fused TPU accept graphs.

    Entry padding already put every per-request input on the compile bucket, so
    the bucket is read off the candidate rows and nothing is staged. The TP
    broadcast sits between the two graphs because the accept decision must agree
    bit for bit across ranks before anything derives from it.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        verify_num_draft_tokens: int,
        tp_sync: SpecTpSync,
    ) -> None:
        self._device = device
        self._width = int(verify_num_draft_tokens)
        self._tp_sync = tp_sync
        self._verify_lens: torch.Tensor | None = None

    def accept(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_lens: torch.Tensor | None,
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the fused accept on the bucket shape, outputs still padded.

        Returns correct_len, bonus, cap_trim_lens, commit_lens, new_seq_lens and
        out_tokens, in AcceptOuts field order.
        """
        width = self._width
        pad_bs = int(candidates.shape[0])
        if target_logits.shape[0] != pad_bs * width:
            raise ValueError(
                "DSpark TPU accept expects the verify logits on the candidate "
                f"bucket: candidates={tuple(candidates.shape)}, logits rows="
                f"{int(target_logits.shape[0])}, width={width}."
            )
        capped_verify_lens = (
            self._uncapped_verify_lens(pad_bs)
            if verify_lens is None
            # The graph was warmed on int64; a scheduled int32 would re-key it.
            else verify_lens.to(torch.int64)
        )
        packed = dspark_accept_greedy_tpu(candidates, target_logits, capped_verify_lens)
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GREEDY, packed)
        return dspark_finalize_accept_tpu(packed, prefix_lens, draft_tokens)

    def _uncapped_verify_lens(self, pad_bs: int) -> torch.Tensor:
        # A static schedule never caps the block, so this sentinel is the only
        # value the graph reads. The capacity is itself a bucket, so pad_bs <=
        # cap holds for every later bs <= cap and the slice never overruns.
        cap = 0 if self._verify_lens is None else int(self._verify_lens.shape[0])
        if cap < pad_bs:
            self._verify_lens = torch.full(
                (pad_bs,), self._width, dtype=torch.int64, device=self._device
            )
        return self._verify_lens[:pad_bs]


class DsparkDecodePadding(msgspec.Struct, frozen=True):
    """Live row count plus the per-request fields entry padding replaced."""

    bs: int
    pad_bs: int
    # prepare_for_verify rebinds batch.spec_info mid-step, so the draft input is
    # held here rather than looked up again at restore time.
    draft_input: DFlashDraftInputV2
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    bonus_tokens: torch.Tensor
    verify_cache_candidates: torch.Tensor | None
    committed_lens: torch.Tensor | None


def pad_decode_batch(
    *, batch: ScheduleBatch, draft_input: DFlashDraftInputV2, pad_bs: int
) -> DsparkDecodePadding:
    """Widen the per-request decode inputs to the compile bucket.

    Every downstream shape in the step follows these fields, so widening them
    once here keeps the verify window, the draft prologue and both
    compiled_forward shims off raw batch shapes -- torch_tpu keys a program on
    the whole trace, so a single raw-shaped op re-keys everything beside it.

    Zero is the repo-wide pad sentinel: req pool slot 0 is a reserved all-zero
    row (memory_pool.py) and KV page 0 is the reserved dummy page
    (allocator_tpu.py), so ghost rows read and write there harmlessly.
    """
    bs = int(batch.seq_lens.shape[0])
    saved = DsparkDecodePadding(
        bs=bs,
        pad_bs=pad_bs,
        draft_input=draft_input,
        seq_lens=batch.seq_lens,
        req_pool_indices=batch.req_pool_indices,
        bonus_tokens=draft_input.bonus_tokens,
        verify_cache_candidates=draft_input.tpu_verify_cache_candidates,
        committed_lens=draft_input.tpu_committed_lens,
    )
    if pad_bs <= bs:
        return saved
    pad_rows = pad_bs - bs
    batch.seq_lens = F.pad(batch.seq_lens, (0, pad_rows))
    batch.req_pool_indices = F.pad(batch.req_pool_indices, (0, pad_rows))
    draft_input.bonus_tokens = F.pad(
        draft_input.bonus_tokens.reshape(-1), (0, pad_rows)
    )
    if draft_input.tpu_verify_cache_candidates is not None:
        draft_input.tpu_verify_cache_candidates = F.pad(
            draft_input.tpu_verify_cache_candidates, (0, 0, 0, pad_rows)
        )
        draft_input.tpu_committed_lens = F.pad(
            draft_input.tpu_committed_lens, (0, pad_rows)
        )
    return saved


def restore_decode_batch(*, batch: ScheduleBatch, padding: DsparkDecodePadding) -> None:
    """Put the live-row views back before the step returns to the scheduler."""
    batch.seq_lens = padding.seq_lens
    batch.req_pool_indices = padding.req_pool_indices
    draft_input = padding.draft_input
    draft_input.bonus_tokens = padding.bonus_tokens
    draft_input.tpu_verify_cache_candidates = padding.verify_cache_candidates
    draft_input.tpu_committed_lens = padding.committed_lens


class DsparkTpuVerifyWindow:
    """TPU counterpart of alloc_verify_window.

    Entry padding already put prefix_lens, req_pool_indices and the candidate
    window on the compile bucket, so both branches run on bucket shapes with no
    staging; bs here is that bucket, not the live request count.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        verify_num_draft_tokens: int,
        block_pos_offsets: torch.Tensor,
    ) -> None:
        self._device = device
        self._verify_num_draft_tokens = int(verify_num_draft_tokens)
        self._block_pos_offsets = block_pos_offsets

    def alloc(
        self,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        bs: int,
        candidates: torch.Tensor | None,
        committed_lens: torch.Tensor | None,
    ) -> VerifyWindow:
        width = self._verify_num_draft_tokens
        if candidates is None:
            verify_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                start_offset=prefix_lens,
                end_offset=prefix_lens + width,
                batch_size=bs,
                draft_token_num=width,
                device=self._device,
            )
            verify_cache_loc_2d = verify_cache_loc.view(bs, width)
        else:
            verify_cache_loc_2d = select_dflash_verify_cache_locs_tpu(
                candidates, prefix_lens, committed_lens, width
            )
            verify_cache_loc = verify_cache_loc_2d.reshape(-1)
        return VerifyWindow(
            positions_2d=prefix_lens.unsqueeze(1) + self._block_pos_offsets,
            verify_cache_loc=verify_cache_loc,
            verify_cache_loc_2d=verify_cache_loc_2d,
        )


def physical_verify_candidates(*, verify_ids_2d: torch.Tensor, padding) -> torch.Tensor:
    """Widen the verify chain to the rows the target graph actually produced."""
    if padding is None:
        return verify_ids_2d
    return _pad_rows(verify_ids_2d, int(padding.state.physical_batch_size))


def unpad_verify_outputs(*, compile_policy, logits_output, verify_forward_batch):
    """Release the deferred unpadding once accept has read the padded rows."""
    verify_forward_batch.spec_info.keep_outputs_padded = False
    return compile_policy.finalize_forward_batch(logits_output, verify_forward_batch)


def logical_verify_logits(
    *, next_token_logits: torch.Tensor, bs: int, verify_num_draft_tokens: int
) -> torch.Tensor:
    """The live rows of a padded verify output.

    Penalties, logit bias and grammar apply in place and are sized for the live
    requests, so they must see this view rather than the padded rows.
    """
    return next_token_logits[: bs * verify_num_draft_tokens]


# Keyed by everything that fixes the buffer's shape, so a hit is always usable as
# is. Allocating per call would add a second shape-varying eager op.
_pad_buffers: dict[tuple[int, int, torch.dtype, str], torch.Tensor] = {}


def _pad_rows(rows: torch.Tensor, pad_bs: int) -> torch.Tensor:
    # Rows past the live batch keep the previous step's values; every consumer
    # trims them off, and none reads a row it was not handed.
    key = (pad_bs, int(rows.shape[1]), rows.dtype, str(rows.device))
    padded = _pad_buffers.get(key)
    if padded is None:
        padded = torch.empty(
            (pad_bs, rows.shape[1]), dtype=rows.dtype, device=rows.device
        )
        _pad_buffers[key] = padded
    padded[: rows.shape[0]] = rows
    return padded
