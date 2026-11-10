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
    if _is_host_padded(draft_input) and draft_input.tpu_relay_resolved:
        draft_input.tpu_relay_resolved = False
        return True
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
    # A host-relay publish skipped its new_seq_lens store; the flush writes it.
    publish_deferred: bool = False


class PublishedSeqLens(msgspec.Struct, frozen=True):
    indices: torch.Tensor
    seq_lens: torch.Tensor
    # False while a deferred publish has not written new_seq_lens_buf yet.
    stored: bool = True


def publish_tpu(
    future_map: FutureMap, indices: torch.Tensor, new_seq_lens: torch.Tensor
) -> None:
    # The overlap forward publishes, then stashes the same indices object;
    # stash_tpu pairs the two so the next step can skip the seq_lens gather.
    future_map.tpu_published_seq_lens = PublishedSeqLens(
        indices=indices, seq_lens=new_seq_lens
    )


def _relays_host_padded(future_map: FutureMap) -> bool:
    """A TpuSpecWorker serves this process and relays only the bonus token and length."""
    from sglang_tpu.experimental import get_host_batch_padder

    return (
        get_host_batch_padder() is not None
        and future_map.spec_algo.is_some()
        and not future_map.need_topk
        and not future_map.need_hidden_states
        and not future_map.needs_confidence_relay
        and future_map.draft_probs_buf is None
        and future_map.dsa_topk_indices_buf is None
    )


def publish_deferred_tpu(
    future_map: FutureMap,
    indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    confidence: torch.Tensor | None,
) -> bool:
    """Record a host-relay publish without its store; the paired stash or a flush writes it.

    The next step of an unchanged batch reads new_seq_lens directly
    (resolve_direct_handoff_tpu), so the store and its gather are two launches
    no one needs. Returns True when the store was deferred.
    """
    if (
        indices.shape[0] == 0
        or confidence is not None
        or not _relays_host_padded(future_map)
    ):
        return False
    # Any earlier pending rows land before this generation's.
    flush_direct_handoff_tpu(future_map)
    # FutureMap.publish's fence: resolve waits on it before reading lengths.
    if future_map.publish_ready is None:
        future_map.publish_ready = torch.get_device_module(future_map.device).Event()
    future_map.publish_ready.record()
    future_map._publish_fresh = True
    future_map.tpu_published_seq_lens = PublishedSeqLens(
        indices=indices, seq_lens=new_seq_lens, stored=False
    )
    return True


def _store_published(future_map: FutureMap, published: PublishedSeqLens) -> None:
    future_map._store(
        future_map.new_seq_lens_buf,
        published.indices,
        published.seq_lens.to(future_map.new_seq_lens_buf.dtype),
    )


def flush_direct_handoff_tpu(future_map: FutureMap) -> None:
    published = future_map.tpu_published_seq_lens
    if published is not None and not published.stored:
        _store_published(future_map, published)
        future_map.tpu_published_seq_lens = PublishedSeqLens(
            indices=published.indices, seq_lens=published.seq_lens
        )
    pending = future_map.tpu_direct_handoff
    if pending is None:
        return
    # Clear before calling the normal store path, which also checks for a flush.
    future_map.tpu_direct_handoff = None
    if pending.publish_deferred:
        _store_published(
            future_map,
            PublishedSeqLens(indices=pending.indices, seq_lens=pending.seq_lens),
        )
    future_map.stash(pending.indices, pending.payload)


def stash_tpu(future_map: FutureMap, indices: torch.Tensor, payload: Any) -> bool:
    """Flush any pending handoff, then defer this store when the next step can
    consume the payload directly. Returns True when the store was deferred.

    EAGLE3 top-k-1 payloads and host-relay (bonus token only) payloads stashed
    right after their publish qualify.
    """
    # Take the publish before flushing: the flush re-enters stash and must fall
    # through to the normal store.
    published = future_map.tpu_published_seq_lens
    future_map.tpu_published_seq_lens = None
    # A different batch may still need rows from the previous pending result.
    flush_direct_handoff_tpu(future_map)
    if published is not None and not published.stored:
        if published.indices is indices and _bonus_only(payload):
            future_map.tpu_direct_handoff = DirectHandoff(
                indices=indices,
                payload=payload,
                seq_lens=published.seq_lens,
                publish_deferred=True,
            )
            return True
        _store_published(future_map, published)
        return False
    # Only gate what the resolve step itself relies on. Anything about the next
    # batch (device acceptance, filtering, merging) is re-checked there, and a
    # mismatch just flushes the pending payload into the normal store.
    if (
        # No paired publish (prefill seed, beam tail slice): nothing to hand off.
        published is None
        or published.indices is not indices
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
        indices=indices, payload=payload, seq_lens=published.seq_lens
    )
    return True


def _bonus_only(payload: Any) -> bool:
    return (
        payload.bonus_tokens is not None
        and payload.topk_p is None
        and payload.topk_index is None
        and payload.hidden_states is None
        and payload.draft_probs is None
        and payload.dsa_topk_indices is None
    )


def resolve_direct_handoff_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    pending = future_map.tpu_direct_handoff
    if pending is None:
        return False
    draft = batch.spec_info
    if pending.publish_deferred:
        return _resolve_host_relay_handoff(future_map, batch, pending)
    # filter_batch and merge_batch rebind future_indices and the draft tensors.
    # Never compare device values on the host; the payload tensors are the
    # draft's own, so identity also pins this generation.
    if (
        draft.future_indices is not pending.indices
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
        return False
    batch.seq_lens = future_map.new_seq_lens_buf[batch.spec_info.future_indices]
    return True


def _resolve_host_relay_handoff(
    future_map: FutureMap, batch: ScheduleBatch, pending: DirectHandoff
) -> bool:
    """Hand an unchanged host-padded batch the previous step's lengths and bonus tokens."""
    draft = batch.spec_info
    # filter_batch / merge_batch rebind future_indices; identity pins the generation.
    if (
        draft.future_indices is not pending.indices
        or draft.bonus_tokens is not pending.payload.bonus_tokens
    ):
        flush_direct_handoff_tpu(future_map)
        return False
    batch.seq_lens = pending.seq_lens
    future_map.tpu_direct_handoff = None
    # _resolve_spec_extras: the bonus tokens are already the draft's own.
    draft.tpu_relay_resolved = True
    return True


def _is_host_padded(spec_info: Any) -> bool:
    # Local import: the experimental worker imports this module.
    from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput

    return isinstance(spec_info, HostPaddedDraftInput)


def _keeps_committed_host_lens(spec_info: Any) -> bool:
    # Local import: the experimental worker imports this module.
    from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput

    return (
        isinstance(spec_info, HostPaddedDraftInput)
        and spec_info.tpu_host_seq_lens_is_committed
    )


def resolve_mixed_spec_tails_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    """Rebind a mixed-verify batch's tails on device when its running part carries the
    decode device-accept plan (verify window candidates from the committed base).

    The block's slots come from the candidates at the published length, so the host
    never waits on the in-flight step; the host lengths stay upper bounds (committed
    base + the block + one block of slack), as for decode. The worker sets the tails'
    positions from the drafted block. False leaves the synchronizing resolve to the
    caller.
    """
    info = batch.mix_tail_spec_info
    if (
        not batch.mix_verify_tails
        or info is None
        or info.tpu_verify_cache_candidates is None
    ):
        return False
    idx = batch.mix_running_indices
    n = int(idx.shape[0])
    width = batch.extend_lens[-1]
    fresh = future_map.new_seq_lens_buf[idx].to(torch.int64)
    seq_lens = batch.seq_lens.clone()
    seq_lens[-n:] = (fresh + width).to(seq_lens.dtype)
    batch.seq_lens = seq_lens
    if batch.out_cache_loc.device == fresh.device:
        offsets = torch.arange(width, device=fresh.device, dtype=torch.int64)
        cols = (fresh - info.tpu_committed_lens)[:, None] + offsets[None, :]
        out_cache_loc = batch.out_cache_loc.clone()
        out_cache_loc[-n * width :] = (
            torch.gather(info.tpu_verify_cache_candidates, 1, cols)
            .reshape(-1)
            .to(out_cache_loc.dtype)
        )
        batch.out_cache_loc = out_cache_loc
    if batch.seq_lens_cpu is not None:
        seq_lens_cpu = batch.seq_lens_cpu.clone()
        seq_lens_cpu[-n:] = batch.mix_tail_committed_cpu + 2 * width
        batch.seq_lens_cpu = seq_lens_cpu
        batch.seq_lens_sum = int(seq_lens_cpu.sum())
    return True
