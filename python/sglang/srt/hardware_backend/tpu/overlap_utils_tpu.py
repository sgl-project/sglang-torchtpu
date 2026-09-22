"""TPU device acceptance and FutureMap gather operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_future_map_device_accept_args,
    make_future_map_device_accept_hidden_args,
)

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
    return (
        # Only the EAGLE relay (need_topk) has a selected window field.
        future_map.need_topk
        and draft_input is not None
        and draft_input.device_accept_selected_cache_locs is not None
    )


@compiled_op(warmup=make_future_map_device_accept_args)
def fused_future_map_device_accept_tpu(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    candidate_locs: torch.Tensor,
    accept_lens: torch.Tensor,
    cache_loc_width: int,
):
    rows = indices.to(torch.int64)
    return (
        torch.index_select(topk_p_buf, 0, rows),
        torch.index_select(topk_index_buf, 0, rows),
        torch.index_select(output_tokens_buf, 0, rows),
        _select_device_accept_cache_locs(candidate_locs, accept_lens, cache_loc_width),
    )


@compiled_op(warmup=make_future_map_device_accept_hidden_args)
def fused_future_map_device_accept_hidden_tpu(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: torch.Tensor,
    candidate_locs: torch.Tensor,
    accept_lens: torch.Tensor,
    cache_loc_width: int,
):
    rows = indices.to(torch.int64)
    return (
        torch.index_select(topk_p_buf, 0, rows),
        torch.index_select(topk_index_buf, 0, rows),
        torch.index_select(output_tokens_buf, 0, rows),
        torch.index_select(hidden_states_buf, 0, rows),
        _select_device_accept_cache_locs(candidate_locs, accept_lens, cache_loc_width),
    )


def resolve_spec_extras_tpu(future_map: FutureMap, batch: ScheduleBatch) -> bool:
    draft_input = batch.spec_info
    if not future_map.need_topk or draft_input is None:
        return False
    assert (
        draft_input.device_accept_selected_cache_locs is None
    ), "device accept routing resolves once per prepared batch"
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
    bs = indices.shape[0]
    physical_bs = bs
    if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
        buckets = future_map.tpu_precompiled_batch_sizes
        assert buckets, "TPU device acceptance helpers must be precompiled at startup"
        physical_bs = min((size for size in buckets if size >= bs), default=None)
        assert (
            physical_bs is not None
        ), f"Live batch size {bs} exceeds largest precompiled bucket {max(buckets)}"
    padded_indices = indices
    if physical_bs > bs:
        padding = physical_bs - bs
        # Dummy rows only read pool row zero and select a valid local
        # window. They are discarded before publishing or invalidating.
        padded_indices = F.pad(indices, (0, padding))
        candidate_locs = F.pad(candidate_locs, (0, 0, 0, padding))
        accept_lens = F.pad(accept_lens, (0, padding), value=1)
    common = (
        padded_indices,
        future_map.topk_p_buf,
        future_map.topk_index_buf,
        future_map.output_tokens_buf,
    )
    if hidden_buf is not None:
        probs, topk, bonus, hidden, selected = (
            fused_future_map_device_accept_hidden_tpu(
                *common, hidden_buf, candidate_locs, accept_lens, width
            )
        )
    else:
        probs, topk, bonus, selected = fused_future_map_device_accept_tpu(
            *common, candidate_locs, accept_lens, width
        )
        hidden = None
    if physical_bs > bs:
        probs, topk, bonus, selected = (
            tensor[:bs] for tensor in (probs, topk, bonus, selected)
        )
        if hidden is not None:
            hidden = hidden[:bs]
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
    # Candidate locations gate device routing. This also resolves the spec
    # extras relay, so FutureMap._resolve_spec_extras only checks
    # device_accept_resolved_tpu afterwards.
    if not resolve_spec_extras_tpu(future_map, batch):
        return False
    batch.seq_lens = future_map.new_seq_lens_buf[batch.spec_info.future_indices]
    return True
