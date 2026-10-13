"""TPU device acceptance and FutureMap gather operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang_tpu.spec_decode_bucket import pad_host_rows

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def _select_device_accept_cache_locs(
    candidate_locs: torch.Tensor,
    seq_lens: torch.Tensor,
    host_base: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Select each row's cache window from its pre-reserved candidates.

    The candidates start right after the host length shadow, which is one
    acceptance behind the device seq_lens, so the window starts at the device
    seq_lens. Padding rows (seq_lens 0, host_base 0) clamp to an all-zero row.
    """
    bs = candidate_locs.shape[0]
    offsets = torch.arange(width, device=candidate_locs.device, dtype=torch.int64)
    starts = (seq_lens.to(torch.int64) - host_base - 1).clamp(min=0)
    cols = starts[:, None] + offsets[None, :]
    rows = torch.arange(bs, device=candidate_locs.device, dtype=torch.int64)[:, None]
    return candidate_locs[rows, cols]


def _decode_rows(future_map: FutureMap, batch: ScheduleBatch) -> torch.Tensor:
    """FutureMap rows for this decode step: bucket-padded if it has a bucket.

    Also tells publish/stash which padded rows stand for the scheduler's
    real-sized future indices of this step.
    """
    draft_input = batch.spec_info
    bucket = getattr(draft_input, "decode_bucket", None)
    if bucket is None:
        future_map.tpu_decode_rows = None
        return draft_input.future_indices
    future_map.tpu_decode_rows = (batch.req_pool_indices, bucket.rows)
    return bucket.rows


def future_rows_tpu(
    future_map: FutureMap, indices: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """Map the scheduler's future indices to the rows a store must write.

    A bucketed decode step stores bucket-sized results; its padding rows write
    the reserved slot 0.
    """
    mapping = future_map.tpu_decode_rows
    if (
        mapping is not None
        and indices is mapping[0]
        and num_rows == mapping[1].shape[0]
        and num_rows != indices.shape[0]
    ):
        return mapping[1]
    return indices


def _host_window_from_lengths(batch: ScheduleBatch, seq_lens: torch.Tensor) -> None:
    """Build a bucketed step's cache window from read-back lengths.

    The candidates start at each row's exact length, so the device selection
    (host_base = length - 1) picks the first columns.
    """
    from sglang_tpu.eagle_utils_tpu import (
        device_accept_candidate_window,
    )

    draft_input = batch.spec_info
    bucket = draft_input.decode_bucket
    bs = bucket.logical_bs
    seq_lens_cpu = seq_lens.cpu()
    seq_lens_cpu[bs:] = 0
    candidates = device_accept_candidate_window(batch, seq_lens_cpu[:bs] - 1)
    draft_input.device_accept_candidate_locs = pad_host_rows(
        candidates, bucket.size
    ).to(batch.device, non_blocking=True)
    draft_input.device_accept_host_base = (seq_lens_cpu - 1).to(
        batch.device, non_blocking=True
    )
    bucket.seq_lens_cpu = seq_lens_cpu
    batch.seq_lens_cpu = seq_lens_cpu[:bs]
    batch.seq_lens_sum = int(seq_lens_cpu.sum())


def _finish_resolve(
    batch: ScheduleBatch, seq_lens: torch.Tensor, candidate_width: int
) -> None:
    draft_input = batch.spec_info
    bucket = getattr(draft_input, "decode_bucket", None)
    if bucket is not None:
        # Padding rows read slot 0, which padding rows of earlier steps wrote.
        seq_lens = seq_lens * bucket.valid
        bucket.seq_lens = seq_lens
    draft_input.device_accept_selected_cache_locs = _select_device_accept_cache_locs(
        draft_input.device_accept_candidate_locs,
        seq_lens,
        draft_input.device_accept_host_base,
        (candidate_width + 1) // 2,
    )
    # A view: the scheduler keeps the real-sized lengths.
    batch.seq_lens = seq_lens[: bucket.logical_bs] if bucket is not None else seq_lens


def device_accept_resolved_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    """True if resolve_seq_lens_cpu already resolved this batch's spec extras
    on device, so _resolve_spec_extras has nothing left to do."""
    draft_input = batch.spec_info
    return (
        # Only the EAGLE relay (need_topk) has a selected window field.
        future_map.need_topk
        and draft_input is not None
        and draft_input.device_accept_selected_cache_locs is not None
    )


def resolve_spec_extras_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    draft_input = batch.spec_info
    if not future_map.need_topk or draft_input is None:
        return False
    assert draft_input.device_accept_selected_cache_locs is None, (
        "device accept routing resolves once per prepared batch"
    )
    indices = draft_input.future_indices
    if (
        indices is None
        or indices.shape[0] == 0
        or future_map.draft_probs_buf is not None
        or draft_input.future_dsa_topk_indices_available
    ):
        return False
    if draft_input.device_accept_candidate_locs is not None:
        rows = _decode_rows(future_map, batch)
        seq_lens = future_map.new_seq_lens_buf[rows]
    elif (
        getattr(draft_input, "decode_bucket", None) is not None
        and future_map.needs_cpu_seq_lens
    ):
        # No acceptance pending: read the lengths back, as the unbucketed path
        # does, and build the window from them on the host.
        rows = _decode_rows(future_map, batch)
        seq_lens = future_map.new_seq_lens_buf[rows]
        _host_window_from_lengths(batch, seq_lens)
    else:
        return False

    candidate_locs = draft_input.device_accept_candidate_locs
    assert candidate_locs.ndim == 2 and candidate_locs.shape[0] == rows.shape[0], (
        "candidate cache locations must match future rows"
    )
    assert candidate_locs.shape[1] >= 3 and candidate_locs.shape[1] % 2 == 1, (
        "candidate cache window must have width 2 * verify_width - 1"
    )

    hidden_buf = future_map.hidden_states_buf if future_map.need_hidden_states else None
    from sglang.srt.speculative.spec_utils import gather_spec_extras_func

    probs, topk, bonus, hidden = gather_spec_extras_func(
        rows,
        future_map.topk_p_buf,
        future_map.topk_index_buf,
        future_map.output_tokens_buf,
        hidden_buf,
    )
    draft_input.topk_p = probs
    draft_input.topk_index = topk
    draft_input.bonus_tokens = bonus
    if hidden is not None:
        draft_input.hidden_states = hidden
    draft_input.dsa_topk_indices = None
    _finish_resolve(batch, seq_lens, candidate_locs.shape[1])

    from sglang.srt.managers.overlap_utils import (
        _DEBUG_ASSERT,
        _assert_nonneg_and_invalidate,
    )

    if _DEBUG_ASSERT:
        _assert_nonneg_and_invalidate(bonus, future_map.output_tokens_buf, rows)
    return True


class DirectHandoff(msgspec.Struct, frozen=True):
    indices: torch.Tensor
    payload: Any
    seq_lens: torch.Tensor
    # FutureMap rows the payload belongs to (bucket-padded for a bucketed step).
    rows: torch.Tensor


class PublishedSeqLens(msgspec.Struct, frozen=True):
    indices: torch.Tensor
    seq_lens: torch.Tensor
    rows: torch.Tensor


def publish_tpu(
    future_map: FutureMap,
    indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    # The overlap forward publishes, then stashes the same indices object;
    # stash_tpu pairs the two so the next step can skip the seq_lens gather.
    future_map.tpu_published_seq_lens = PublishedSeqLens(
        indices=indices, seq_lens=new_seq_lens, rows=rows
    )


def flush_direct_handoff_tpu(future_map: FutureMap) -> None:
    pending = future_map.tpu_direct_handoff
    if pending is None:
        return
    # Clear before calling the normal store path, which also checks for a flush.
    future_map.tpu_direct_handoff = None
    future_map.stash(pending.rows, pending.payload)


def stash_tpu(
    future_map: FutureMap, indices: torch.Tensor, payload: Any, rows: torch.Tensor
) -> bool:
    """Flush any pending handoff, then defer this store when the next step can
    consume the payload directly. Returns True when the store was deferred.

    Only EAGLE3 top-k-1 payloads stashed right after a publish qualify.
    """
    # Take the publish before flushing: the flush re-enters stash and must fall
    # through to the normal store.
    published = future_map.tpu_published_seq_lens
    future_map.tpu_published_seq_lens = None
    # A different batch may still need rows from the previous pending result.
    flush_direct_handoff_tpu(future_map)
    # Only gate what the resolve step itself relies on. Anything about the next
    # batch (device acceptance, filtering, merging) is re-checked there, and a
    # mismatch just flushes the pending payload into the normal store.
    if (
        # No paired publish (prefill seed, beam tail slice): nothing to hand off.
        published is None
        or published.indices is not indices
        or published.rows is not rows
        # Only the EAGLE3 topk-1 decode path keeps acceptance on device.
        or not future_map.spec_algo.is_eagle3()
        or payload.topk_index is None
        or payload.topk_index.shape[1] != 1
        # DP idle stub: an empty payload has nothing to relay.
        or indices.shape[0] == 0
        # Resolve copies these directly. Multi-layer EAGLE3 does not relay
        # hidden states, and draft probabilities are not carried at all.
        or payload.hidden_states is None
        or payload.draft_probs is not None
    ):
        return False
    future_map._maybe_init_forward_bufs(payload)
    future_map.tpu_direct_handoff = DirectHandoff(
        indices=indices,
        payload=payload,
        seq_lens=published.seq_lens,
        rows=rows,
    )
    return True


def resolve_direct_handoff_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    pending = future_map.tpu_direct_handoff
    if pending is None:
        return False
    draft = batch.spec_info
    # filter_batch and merge_batch rebind future_indices and the draft tensors.
    # Never compare device values on the host; the payload tensors are the
    # draft's own, so identity also pins this generation.
    if (
        draft.future_indices is not pending.indices
        or draft.bonus_tokens is not pending.payload.bonus_tokens
        or draft.device_accept_candidate_locs is None
        or draft.accept_lens_device is None
        or draft.future_dsa_topk_indices_available
    ):
        flush_direct_handoff_tpu(future_map)
        return False
    # The payload must have exactly this step's rows (bucketed or not).
    rows = _decode_rows(future_map, batch)
    if (
        pending.rows.shape[0] != rows.shape[0]
        or draft.device_accept_candidate_locs.shape[0] != rows.shape[0]
    ):
        flush_direct_handoff_tpu(future_map)
        return False
    candidates = draft.device_accept_candidate_locs
    assert candidates.shape[1] >= 3 and candidates.shape[1] % 2 == 1
    draft.bonus_tokens = pending.payload.bonus_tokens.to(
        future_map.output_tokens_buf.dtype
    )
    draft.topk_p = pending.payload.topk_p.to(future_map.topk_p_buf.dtype)
    draft.topk_index = pending.payload.topk_index.to(future_map.topk_index_buf.dtype)
    draft.hidden_states = pending.payload.hidden_states.to(
        future_map.hidden_states_buf.dtype
    )
    _finish_resolve(
        batch,
        pending.seq_lens.to(future_map.new_seq_lens_buf.dtype),
        candidates.shape[1],
    )
    # All rows were consumed in order. The batch now owns these references;
    # subsequent filtered/merged generations use their own pending payload.
    future_map.tpu_direct_handoff = None
    return True


def resolve_device_accept_seq_lens_tpu(
    future_map: FutureMap, batch: ScheduleBatch
) -> bool:
    if resolve_direct_handoff_tpu(future_map, batch):
        return True
    # Candidate locations gate device routing. This also resolves the spec
    # extras relay, so FutureMap._resolve_spec_extras only checks
    # device_accept_resolved_tpu afterwards.
    if not resolve_spec_extras_tpu(future_map, batch):
        future_map.tpu_decode_rows = None
        return False
    return True
