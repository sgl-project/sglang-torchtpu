"""One-use device state relay for unchanged EAGLE3 batches."""

from typing import Any

import msgspec
import torch
import torch.nn.functional as F


class DirectHandoff(msgspec.Struct, frozen=True):
    draft_input: Any
    indices: torch.Tensor
    payload: Any
    seq_lens: torch.Tensor
    physical: Any = None


def flush_direct_handoff_tpu(future_map) -> None:
    pending = future_map.tpu_direct_handoff
    if pending is None:
        return
    # Clear before calling the normal store path, which also checks for a flush.
    future_map.tpu_direct_handoff = None
    future_map.stash(pending.indices, pending.payload)


def stash_direct_handoff_tpu(future_map, batch, indices, payload, result) -> bool:
    """Defer stores only for owned finalizer outputs in the top-k-1 decode path."""
    draft = result.next_draft_input
    if (
        not batch.enable_overlap
        or not batch.forward_mode.is_draft_extend_v2()
        or not future_map.spec_algo.is_eagle3()
        or draft is None
        or draft.accept_lens_device is None
        or indices.shape[0] == 0
        or payload.topk_p is None
        or payload.topk_index is None
        or payload.topk_index.shape[1] != 1
        or payload.hidden_states is None
        or payload.draft_probs is not None
        or payload.dsa_topk_indices is not None
        or batch.return_logprob
        or batch.has_grammar
        or any(req.lora_id is not None for req in batch.reqs)
    ):
        return False
    physical = draft.tpu_draft_continuation if draft is not None else None
    padded_bucket = (
        physical is not None
        and physical.logical_batch_size == indices.shape[0]
        and physical.physical_batch_size in future_map._tpu_precompiled_batch_sizes
        and physical.physical_batch_size > indices.shape[0]
    )
    if (
        indices.shape[0] not in future_map._tpu_precompiled_batch_sizes
        and not padded_bucket
    ):
        return False
    # A different batch may still need rows from the previous pending result.
    flush_direct_handoff_tpu(future_map)
    future_map._maybe_init_forward_bufs(payload)
    future_map.tpu_direct_handoff = DirectHandoff(
        draft_input=draft,
        indices=indices,
        payload=payload,
        seq_lens=result.new_seq_lens,
        physical=physical,
    )
    draft.tpu_draft_continuation = None
    return True


def resolve_direct_handoff_tpu(future_map, batch) -> bool:
    pending = future_map.tpu_direct_handoff
    if pending is None:
        return False
    draft = batch.spec_info
    # filter_batch and merge_batch rebind future_indices. Never compare device
    # values on the host. The draft object also identifies this generation.
    if (
        draft is not pending.draft_input
        or draft.future_indices is not pending.indices
        or draft.device_accept_candidate_locs is None
        or draft.accept_lens_device is None
        or draft.device_accept_candidate_locs.shape[0] != pending.indices.shape[0]
        or draft.future_dsa_topk_indices_available
    ):
        flush_direct_handoff_tpu(future_map)
        return False
    from sglang.srt.hardware_backend.tpu.overlap_utils_tpu import (
        select_direct_accept_cache_tpu,
    )

    candidates = draft.device_accept_candidate_locs
    physical = pending.physical
    accept_lens = draft.accept_lens_device
    if physical is not None:
        candidates = F.pad(
            candidates, (0, 0, 0, physical.physical_batch_size - candidates.shape[0])
        )
        accept_lens = physical.accept_lens
    assert candidates.shape[1] >= 3 and candidates.shape[1] % 2 == 1
    draft.device_accept_selected_cache_locs = select_direct_accept_cache_tpu(
        candidates, accept_lens, (candidates.shape[1] + 1) // 2
    )
    if physical is not None:
        draft.device_accept_selected_cache_locs = (
            draft.device_accept_selected_cache_locs[: physical.logical_batch_size]
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
    draft.tpu_draft_continuation = physical
    # All rows were consumed in order. The batch now owns these references;
    # subsequent filtered/merged generations use their own pending payload.
    future_map.tpu_direct_handoff = None
    return True
