"""TPU helpers for EAGLE verify trees and cache routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.runtime_context import get_spec, max_speculative_num_draft_tokens

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput


def build_topk1_verify_chain_torch(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """Build the fixed top-k=1 EAGLE verify tensors with pure torch ops.

    The fused TPU draft loop calls this before returning so tree construction
    stays in the same compiled program. The ordinary path also uses it as its
    torch-native fallback.

    Args:
        bonus_tokens: Bonus token IDs with shape ``[bs]``.
        draft_tokens: Draft token IDs with shape ``[bs, num_draft_tokens]``.
        seq_lens: Prefix lengths with shape ``[bs]``.

    Returns:
        An empty tree mask with shape ``[0]``, followed by positions, retrieve
        indices, next-token indices, next-sibling indices, and verification
        tokens, each with shape ``[bs, num_draft_tokens + 1]``.
    """
    bs = seq_lens.shape[0]
    num_verify_tokens = draft_tokens.shape[1] + 1
    device = seq_lens.device

    verify_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1)
    token_offsets = torch.arange(num_verify_tokens, device=device, dtype=torch.long)
    positions = seq_lens[:, None] + token_offsets[None, :]
    retrieve_index = torch.arange(
        bs * num_verify_tokens, device=device, dtype=torch.long
    ).view(bs, num_verify_tokens)
    next_token_row = torch.cat(
        (
            torch.arange(1, num_verify_tokens, device=device, dtype=torch.long),
            torch.full((1,), -1, device=device, dtype=torch.long),
        )
    )
    retrieve_next_token = next_token_row.expand(bs, -1).contiguous()
    retrieve_next_sibling = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    tree_mask = torch.empty((0,), dtype=torch.bool, device=device)
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        verify_tokens,
    )


def build_topk1_verify_tree_tpu(
    *, bonus_tokens: torch.Tensor, draft_tokens: torch.Tensor, seq_lens: torch.Tensor
):
    tree_mask, positions, retrieve_index, next_token, next_sibling, verify_tokens = (
        build_topk1_verify_chain_torch(bonus_tokens, draft_tokens, seq_lens)
    )
    return (
        tree_mask,
        positions.flatten(),
        retrieve_index,
        next_token,
        next_sibling,
        verify_tokens.flatten(),
    )


def prepare_verify_cache_locs_tpu(
    verify_input: EagleVerifyInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
) -> torch.Tensor:
    bs = len(batch.req_pool_indices)
    precomputed_cache_locs = verify_input.device_accept_cache_locs
    if precomputed_cache_locs is not None:
        assert precomputed_cache_locs.shape == (
            bs * verify_input.draft_token_num,
        ), "precomputed cache locations must match verify tokens"
        # FutureMap selected this window on device while host lengths lagged.
        verify_input.device_accept_cache_locs = None
        return precomputed_cache_locs

    from sglang.srt.speculative.spec_utils import assign_extend_cache_locs_func

    return assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=req_to_token_pool.req_to_token,
        start_offset=batch.seq_lens,
        end_offset=batch.seq_lens + verify_input.draft_token_num,
        batch_size=bs,
        draft_token_num=verify_input.draft_token_num,
        device=batch.device,
        host_req_pool_indices=batch.req_pool_indices_cpu,
        host_seq_lens=batch.seq_lens_cpu,
    )


def prepare_device_accept_tpu(
    batch: ScheduleBatch, *, bs: int, double_alloc: int
) -> None:
    # EAGLE3 topk1 overlap can keep the preceding verify's accepted length
    # on TPU. The host pre-reserves every possible logical start in the bounded
    # next window; FutureMap selects the real window after device acceptance.
    draft_input = batch.spec_info
    if draft_input is not None:
        draft_input.device_accept_candidate_locs = None
        draft_input.device_accept_selected_cache_locs = None

    accept_lens = draft_input.accept_lens_device if draft_input is not None else None
    if accept_lens is not None:
        assert accept_lens.shape == (bs,), "acceptance lengths must match batch rows"
    host_resident = batch.req_to_token_pool.host_resident
    device_accept_kv = bool(
        batch.enable_overlap
        and bs > 0
        and host_resident
        and accept_lens is not None
        and not batch.return_logprob
        and not batch.has_grammar
        and all(req.lora_id is None for req in batch.reqs)
    )
    if device_accept_kv:
        spec = get_spec()
        if spec.speculative_num_steps <= 0:
            return
        # Host result processing is one iteration behind device acceptance.
        # req.seqlen is the logical length immediately before that acceptance.
        host_base = torch.tensor(
            [req.seqlen for req in batch.reqs], dtype=torch.int64, device="cpu"
        )
        assert all(
            int(host_base[i]) + double_alloc <= req.kv.kv_allocated_len
            for i, req in enumerate(batch.reqs)
        ), "TPU acceptance routing requires KV slots reserved for two forwards"
        # Size the window with the same bound as the KV reservation: adaptive
        # spec may switch to a longer state between this prepare and the
        # verify whose acceptance selects the window.
        num_draft_tokens = max_speculative_num_draft_tokens()
        candidate_width = 2 * num_draft_tokens - 1
        offsets = torch.arange(candidate_width, dtype=torch.int64)
        rows = batch.req_pool_indices_cpu.to(dtype=torch.int64)[:, None]
        cols = host_base[:, None] + 1 + offsets[None, :]
        candidates = batch.req_to_token_pool.req_to_token[rows, cols]
        draft_input.device_accept_candidate_locs = candidates.to(
            batch.device, dtype=torch.int64, non_blocking=True
        )
        # The host shadow stays one acceptance behind the device seq_lens.
        # Verify and draft-extend kernel lengths and the draft-extend
        # positions (draft_extend_positions_tpu) come from device seq_lens;
        # Pallas reads only the live-row count and the KV capacity check
        # from seq_lens_cpu, and the RPA backend ignores the draft-extend
        # prefix_lens derived from it. pallas_tpu_flash reads that
        # prefix_lens on host and has no spec support; it is not supported
        # with device accept routing.
        batch.seq_lens_cpu = host_base
        batch.seq_lens_sum = int(host_base.sum())


def draft_extend_positions_tpu(
    seq_lens: torch.Tensor, num_window_tokens: int
) -> torch.Tensor:
    # Under device accept routing seq_lens_cpu is one acceptance behind, so
    # the extend positions come from the device seq_lens, like the widened
    # window's precomputed positions.
    offsets = torch.arange(num_window_tokens, device=seq_lens.device, dtype=torch.int64)
    return (seq_lens.to(torch.int64)[:, None] + offsets[None, :]).reshape(-1)


def prepare_draft_cache_locs_tpu(
    batch: ScheduleBatch,
    draft_input: EagleDraftInput,
    *,
    bs: int,
    topk: int,
    num_steps: int,
    num_verify_tokens: int,
    host_resident: bool,
) -> bool:
    selected = draft_input.device_accept_selected_cache_locs
    draft_width = topk * num_steps
    if not host_resident or topk != 1 or selected is None:
        return False
    assert (
        selected.ndim == 2 and selected.shape[0] == bs
    ), "selected cache locations must match batch rows"
    assert selected.shape[1] >= max(
        draft_width, num_verify_tokens
    ), "selected cache window must cover draft and verify tokens"
    batch.out_cache_loc = selected[:, :draft_width].reshape(-1)
    return True


def device_accept_verify_cache_locs_tpu(
    draft_input: EagleDraftInput, *, bs: int, num_verify_tokens: int
) -> torch.Tensor | None:
    selected = draft_input.device_accept_selected_cache_locs
    if selected is None:
        return None
    assert (
        selected.ndim == 2 and selected.shape[0] == bs
    ), "selected cache locations must match verify rows"
    assert (
        selected.shape[1] >= num_verify_tokens
    ), "selected cache window must cover verify tokens"
    return selected[:, :num_verify_tokens].reshape(-1)


def filter_device_accept_tpu(
    draft_input: EagleDraftInput, new_indices: torch.Tensor
) -> None:
    if draft_input.accept_lens_device is not None:
        draft_input.accept_lens_device = draft_input.accept_lens_device[new_indices]
    if draft_input.device_accept_candidate_locs is not None:
        draft_input.device_accept_candidate_locs = (
            draft_input.device_accept_candidate_locs[new_indices]
        )
    if draft_input.device_accept_selected_cache_locs is not None:
        draft_input.device_accept_selected_cache_locs = (
            draft_input.device_accept_selected_cache_locs[new_indices]
        )


def merge_device_accept_tpu(
    draft_input: EagleDraftInput, spec_info: EagleDraftInput
) -> None:
    if (
        draft_input.accept_lens_device is not None
        and spec_info.accept_lens_device is not None
        and draft_input.device_accept_candidate_locs is not None
        and spec_info.device_accept_candidate_locs is not None
    ):
        draft_input.accept_lens_device = torch.cat(
            [draft_input.accept_lens_device, spec_info.accept_lens_device]
        )
        draft_input.device_accept_candidate_locs = torch.cat(
            [
                draft_input.device_accept_candidate_locs,
                spec_info.device_accept_candidate_locs,
            ]
        )
    else:
        draft_input.accept_lens_device = None
        draft_input.device_accept_candidate_locs = None
    draft_input.device_accept_selected_cache_locs = None
