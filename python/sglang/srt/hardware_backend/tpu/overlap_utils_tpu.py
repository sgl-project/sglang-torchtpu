"""TPU device acceptance and FutureMap gather operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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


def resolve_spec_extras_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    draft_input = batch.spec_info
    if not future_map.need_topk or draft_input is None:
        return False
    if draft_input.device_accept_selected_cache_locs is not None:
        return True
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

    assert (
        candidate_locs.ndim == 2 and candidate_locs.shape[0] == indices.shape[0]
    ), "candidate cache locations must match future rows"
    assert accept_lens.shape == (
        indices.shape[0],
    ), "acceptance lengths must match future rows"
    assert (
        candidate_locs.shape[1] >= 3 and candidate_locs.shape[1] % 2 == 1
    ), "candidate cache window must have width 2 * verify_width - 1"

    hidden_buf = future_map.hidden_states_buf if future_map.need_hidden_states else None
    width = (candidate_locs.shape[1] + 1) // 2
    from sglang.srt.managers.overlap_utils import gather_spec_extras_func

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


def resolve_device_accept_seq_lens_tpu(
    future_map: FutureMap, batch: ScheduleBatch
) -> bool:
    # Candidate locations gate device routing; selected locations let the
    # subsequent relay resolution reuse the tensors resolved here.
    if not resolve_spec_extras_tpu(future_map, batch):
        return False
    batch.seq_lens = future_map.new_seq_lens_buf[batch.spec_info.future_indices]
    return True
