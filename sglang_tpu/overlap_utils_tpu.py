"""TPU device acceptance and FutureMap gather operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang_tpu.experimental import uses_host_batch

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def _select_device_accept_cache_locs(
    candidate_locs: torch.Tensor,
    accept_lens: torch.Tensor,
    width: int,
) -> torch.Tensor:
    bs = accept_lens.shape[0]
    offsets = torch.arange(width, device=candidate_locs.device, dtype=torch.int64)
    cols = accept_lens.to(torch.int64)[:, None] - 1 + offsets[None, :]
    rows = torch.arange(bs, device=candidate_locs.device, dtype=torch.int64)[:, None]
    return candidate_locs[rows, cols]


def device_accept_resolved_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    """True if resolve_seq_lens_cpu already resolved this batch's spec extras
    on device, so _resolve_spec_extras has nothing left to do."""
    draft_input = batch.spec_info
    if draft_input is None:
        return False
    if getattr(draft_input, "_tpu_direct_handoff_resolved", False):
        return True
    return (
        # Only the EAGLE relay (need_topk) has a selected window field.
        future_map.need_topk
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
    candidate_locs = draft_input.device_accept_candidate_locs
    accept_lens = draft_input.accept_lens_device
    if (
        indices is None
        or indices.shape[0] == 0
        or candidate_locs is None
        or accept_lens is None
        or future_map.draft_probs_buf is not None
        or draft_input.future_dsa_topk_indices_available
    ):
        return False

    assert candidate_locs.ndim == 2 and candidate_locs.shape[0] == indices.shape[0], (
        "candidate cache locations must match future rows"
    )
    assert accept_lens.shape == (indices.shape[0],), (
        "acceptance lengths must match future rows"
    )
    assert candidate_locs.shape[1] >= 3 and candidate_locs.shape[1] % 2 == 1, (
        "candidate cache window must have width 2 * verify_width - 1"
    )

    hidden_buf = future_map.hidden_states_buf if future_map.need_hidden_states else None
    width = (candidate_locs.shape[1] + 1) // 2
    from sglang.srt.speculative.spec_utils import gather_spec_extras_func

    probs, topk, bonus, hidden = gather_spec_extras_func(
        indices,
        future_map.topk_p_buf,
        future_map.topk_index_buf,
        future_map.output_tokens_buf,
        hidden_buf,
    )
    selected = _select_device_accept_cache_locs(candidate_locs, accept_lens, width)
    draft_input.topk_p = probs
    draft_input.topk_index = topk
    draft_input.bonus_tokens = bonus
    if hidden is not None:
        draft_input.hidden_states = hidden
    draft_input.device_accept_selected_cache_locs = selected
    draft_input.dsa_topk_indices = None

    from sglang.srt.managers.overlap_utils import (
        _DEBUG_ASSERT,
        _assert_nonneg_and_invalidate,
    )

    if _DEBUG_ASSERT:
        _assert_nonneg_and_invalidate(bonus, future_map.output_tokens_buf, indices)
    return True


class DirectHandoff(msgspec.Struct, frozen=True):
    indices: torch.Tensor
    payload: Any
    seq_lens: torch.Tensor
    seq_lens_deferred: bool = False


class PublishedSeqLens(msgspec.Struct, frozen=True):
    indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_deferred: bool = False


def _supports_direct_handoff_algo(spec_algo: Any) -> bool:
    return bool(
        getattr(spec_algo, "is_eagle3", lambda: False)()
        or getattr(spec_algo, "is_dflash_family", lambda: False)()
    )


def _has_dflash_empty_eagle_fields(payload: Any) -> bool:
    topk_empty = payload.topk_index is None or payload.topk_index.numel() == 0
    hidden_empty = payload.hidden_states is None or payload.hidden_states.numel() == 0
    return topk_empty and hidden_empty


def _store_published_seq_lens(
    future_map: FutureMap, indices: torch.Tensor, seq_lens: torch.Tensor
) -> None:
    future_map._store(
        future_map.new_seq_lens_buf,
        indices,
        seq_lens.to(future_map.new_seq_lens_buf.dtype),
    )


def publish_tpu(
    future_map: FutureMap,
    indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    confidence: Any = None,
) -> bool:
    # Flush any unconsumed pending state before arming the next overlap step.
    flush_direct_handoff_tpu(future_map)
    if (
        _supports_direct_handoff_algo(future_map.spec_algo)
        and indices.shape[0] > 0
        and not (future_map.needs_confidence_relay and confidence is not None)
    ):
        if future_map.spec_algo.is_some():
            if future_map.publish_ready is None:
                future_map.publish_ready = torch.get_device_module(
                    future_map.device
                ).Event()
            future_map.publish_ready.record()
            future_map._publish_fresh = True
        future_map.tpu_published_seq_lens = PublishedSeqLens(
            indices=indices, seq_lens=new_seq_lens, seq_lens_deferred=True
        )
        return True
    return False


def flush_direct_handoff_tpu(future_map: FutureMap) -> None:
    published = getattr(future_map, "tpu_published_seq_lens", None)
    if published is not None:
        future_map.tpu_published_seq_lens = None
        if published.seq_lens_deferred:
            _store_published_seq_lens(future_map, published.indices, published.seq_lens)
    pending = getattr(future_map, "tpu_direct_handoff", None)
    if pending is None:
        return
    # Clear before calling the normal store path, which also checks for a flush.
    future_map.tpu_direct_handoff = None
    if pending.seq_lens_deferred:
        _store_published_seq_lens(future_map, pending.indices, pending.seq_lens)
    future_map.stash(pending.indices, pending.payload)


def stash_tpu(future_map: FutureMap, indices: torch.Tensor, payload: Any) -> bool:
    """Flush any pending handoff, then defer this store when the next step can
    consume the payload directly. Returns True when the store was deferred.

    Supports EAGLE3 top-k-1 and DFLASH-family direct handoff payloads paired
    with a preceding publish.
    """
    # Take the publish before flushing: the flush re-enters stash and must fall
    # through to the normal store.
    published = getattr(future_map, "tpu_published_seq_lens", None)
    future_map.tpu_published_seq_lens = None
    # A different batch may still need rows from the previous pending result.
    flush_direct_handoff_tpu(future_map)
    is_eagle3 = bool(getattr(future_map.spec_algo, "is_eagle3", lambda: False)())
    is_dflash = bool(getattr(future_map.spec_algo, "is_dflash_family", lambda: False)())
    algo_ok = (
        is_eagle3
        and payload.topk_index is not None
        and payload.topk_index.ndim == 2
        and payload.topk_index.shape[1] == 1
        and payload.hidden_states is not None
    ) or (
        is_dflash
        and payload.bonus_tokens is not None
        and _has_dflash_empty_eagle_fields(payload)
        and payload.dsa_topk_indices is None
    )
    if (
        # No paired publish (prefill seed, beam tail slice): nothing to hand off.
        published is None
        or published.indices is not indices
        # DP idle stub: an empty payload has nothing to relay.
        or indices.shape[0] == 0
        or payload.draft_probs is not None
        or not algo_ok
    ):
        if published is not None and published.seq_lens_deferred:
            _store_published_seq_lens(future_map, published.indices, published.seq_lens)
        return False
    future_map._maybe_init_forward_bufs(payload)
    future_map.tpu_direct_handoff = DirectHandoff(
        indices=indices,
        payload=payload,
        seq_lens=published.seq_lens,
        seq_lens_deferred=published.seq_lens_deferred,
    )
    return True


def resolve_direct_handoff_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    published = getattr(future_map, "tpu_published_seq_lens", None)
    if published is not None:
        future_map.tpu_published_seq_lens = None
        if published.seq_lens_deferred:
            _store_published_seq_lens(future_map, published.indices, published.seq_lens)
    pending = getattr(future_map, "tpu_direct_handoff", None)
    if pending is None:
        return False
    draft = batch.spec_info
    if getattr(future_map.spec_algo, "is_dflash_family", lambda: False)():
        if (
            draft is None
            or draft.future_indices is not pending.indices
            or draft.bonus_tokens is not pending.payload.bonus_tokens
            or not getattr(draft, "tpu_host_seq_lens_is_committed", False)
            or getattr(draft, "future_dsa_topk_indices_available", False)
        ):
            flush_direct_handoff_tpu(future_map)
            return False
        draft.bonus_tokens = pending.payload.bonus_tokens.to(
            future_map.output_tokens_buf.dtype
        )
        batch.seq_lens = pending.seq_lens.to(future_map.new_seq_lens_buf.dtype)
        draft._tpu_direct_handoff_resolved = True
        future_map.tpu_direct_handoff = None
        return True
    # filter_batch and merge_batch rebind future_indices and the draft tensors.
    # Never compare device values on the host; the payload tensors are the
    # draft's own, so identity also pins this generation.
    if (
        draft is None
        or draft.future_indices is not pending.indices
        or draft.bonus_tokens is not pending.payload.bonus_tokens
        or draft.device_accept_candidate_locs is None
        or draft.accept_lens_device is None
        or draft.device_accept_candidate_locs.shape[0] != pending.indices.shape[0]
        or draft.future_dsa_topk_indices_available
    ):
        flush_direct_handoff_tpu(future_map)
        return False
    candidates = draft.device_accept_candidate_locs
    assert candidates.shape[1] >= 3 and candidates.shape[1] % 2 == 1
    draft.device_accept_selected_cache_locs = _select_device_accept_cache_locs(
        candidates, draft.accept_lens_device, (candidates.shape[1] + 1) // 2
    )
    draft.bonus_tokens = pending.payload.bonus_tokens.to(
        future_map.output_tokens_buf.dtype
    )
    draft.topk_p = pending.payload.topk_p.to(future_map.topk_p_buf.dtype)
    draft.topk_index = pending.payload.topk_index.to(future_map.topk_index_buf.dtype)
    draft.hidden_states = pending.payload.hidden_states.to(
        future_map.hidden_states_buf.dtype
    )
    batch.seq_lens = pending.seq_lens.to(future_map.new_seq_lens_buf.dtype)
    # All rows were consumed in order. The batch now owns these references;
    # subsequent filtered/merged generations use their own pending payload.
    future_map.tpu_direct_handoff = None
    return True


def resolve_device_accept_seq_lens_tpu(
    future_map: FutureMap, batch: ScheduleBatch
) -> bool:
    if resolve_direct_handoff_tpu(future_map, batch):
        return True
    if uses_host_batch(batch) and _keeps_committed_host_lens(batch.spec_info):
        # The verify window was planned from the committed host lengths; reading
        # the published lengths back would reintroduce the sync it avoids.
        batch.seq_lens = future_map.new_seq_lens_buf[batch.spec_info.future_indices]
        return True
    # Candidate locations gate device routing. This also resolves the spec
    # extras relay, so FutureMap._resolve_spec_extras only checks
    # device_accept_resolved_tpu afterwards.
    if not resolve_spec_extras_tpu(future_map, batch):
        flush_direct_handoff_tpu(future_map)
        return False
    batch.seq_lens = future_map.new_seq_lens_buf[batch.spec_info.future_indices]
    return True


def _keeps_committed_host_lens(spec_info: Any) -> bool:
    # Local import: the experimental worker imports this module.
    from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput

    return (
        isinstance(spec_info, HostPaddedDraftInput)
        and spec_info.tpu_host_seq_lens_is_committed
    )
