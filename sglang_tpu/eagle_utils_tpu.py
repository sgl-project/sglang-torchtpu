"""TPU helpers for EAGLE verify trees and cache routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.runtime_context import get_spec, max_speculative_num_draft_tokens
from sglang_tpu.spec_decode_bucket import (
    DecodeBucket,
    bucket_eligible,
    decode_batch_bucket,
    pad_host_rows,
)
from sglang_tpu.spec_op_warmup import (
    make_draft_extend_select_args,
)

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


def _gather_token_window(
    req_to_token: torch.Tensor,
    rows: torch.Tensor,
    starts: torch.Tensor,
    width: int,
    *,
    clamp_columns: bool = False,
) -> torch.Tensor:
    """Gather `width` consecutive page-table slots per row, as [len(rows), width].

    The window is a dense rectangle, so the columns are one broadcast add and
    the read is a single advanced index: no ragged segmentation, no cumsum, and
    no zero-initialized output to mask into. Set `clamp_columns` where the
    consumer ignores the slots past the row (the verify window reads only the
    reserved prefix); leave it off where an out-of-range start is a bug that
    should raise instead of silently folding onto the row edge.
    """
    device = req_to_token.device
    offsets = torch.arange(width, device=device, dtype=torch.int64)
    columns = starts.to(device, torch.int64)[:, None] + offsets[None, :]
    if clamp_columns:
        columns = columns.clamp(min=0, max=req_to_token.shape[1] - 1)
    return req_to_token[rows.to(device, torch.int64)[:, None], columns]


def prepare_verify_cache_locs_tpu(
    verify_input: EagleVerifyInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
) -> torch.Tensor:
    bs = len(batch.req_pool_indices)
    precomputed_cache_locs = verify_input.device_accept_cache_locs
    if precomputed_cache_locs is not None:
        assert precomputed_cache_locs.shape == (bs * verify_input.draft_token_num,), (
            "precomputed cache locations must match verify tokens"
        )
        # FutureMap selected this window on device while host lengths lagged.
        verify_input.device_accept_cache_locs = None
        return precomputed_cache_locs

    # Verification reserves exactly draft_token_num slots per request, so the
    # window is a fixed-width rectangle rather than a ragged range. Read it
    # where the page table lives: host-resident tables use the host mirrors of
    # the row and length tensors so the gather never touches the device.
    req_to_token = req_to_token_pool.req_to_token
    host_table = req_to_token.device.type == "cpu"
    rows = batch.req_pool_indices
    starts = batch.seq_lens
    if host_table:
        if batch.req_pool_indices_cpu is not None:
            rows = batch.req_pool_indices_cpu
        if batch.seq_lens_cpu is not None:
            starts = batch.seq_lens_cpu
    window = _gather_token_window(
        req_to_token,
        rows[:bs],
        starts[:bs],
        verify_input.draft_token_num,
        clamp_columns=True,
    )
    return window.reshape(-1).to(batch.device, dtype=torch.int64)


def device_accept_candidate_window(
    batch: ScheduleBatch, host_base: torch.Tensor
) -> torch.Tensor:
    """Host-reserved cache slots for every acceptance outcome of each row.

    Size the window with the same bound as the KV reservation: adaptive spec
    may switch to a longer state between this prepare and the verify whose
    acceptance selects the window.
    """
    candidate_width = 2 * max_speculative_num_draft_tokens() - 1
    return _gather_token_window(
        batch.req_to_token_pool.req_to_token,
        batch.req_pool_indices_cpu,
        host_base + 1,
        candidate_width,
    ).to(torch.int64)


def prepare_device_accept_tpu(
    batch: ScheduleBatch, *, bs: int, double_alloc: int
) -> None:
    # EAGLE3 topk1 overlap can keep the preceding verify's accepted length
    # on TPU. The host pre-reserves every possible logical start in the bounded
    # next window; FutureMap selects the real window after device acceptance.
    draft_input = batch.spec_info
    if draft_input is not None:
        draft_input.device_accept_candidate_locs = None
        draft_input.device_accept_host_base = None
        draft_input.device_accept_selected_cache_locs = None
        draft_input.decode_bucket = None

    host_resident = batch.req_to_token_pool.host_resident
    routable = bool(
        batch.enable_overlap
        and bs > 0
        and host_resident
        and draft_input is not None
        and not batch.return_logprob
        and not batch.has_grammar
        and all(req.lora_id is None for req in batch.reqs)
    )
    if not routable or get_spec().speculative_num_steps <= 0:
        return
    # Pad on the host so the whole step runs at the batch bucket.
    pad_b = decode_batch_bucket(bs)
    if pad_b is not None and not bucket_eligible(batch):
        pad_b = None
    if draft_input.accept_lens_device is None:
        # No acceptance pending on device (first decode after a prefill or a
        # merge): the FutureMap resolve reads the lengths back and builds the
        # window on the host, so only the rows are needed here.
        if pad_b is not None and draft_input.future_indices is not None:
            draft_input.decode_bucket = _new_decode_bucket(batch, bs, pad_b)
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
    candidates = device_accept_candidate_window(batch, host_base)
    if pad_b is not None:
        # Padding rows select from an all-zero window (KV page 0).
        candidates = pad_host_rows(candidates, pad_b)
        host_base = pad_host_rows(host_base, pad_b)
        draft_input.decode_bucket = _new_decode_bucket(
            batch, bs, pad_b, seq_lens_cpu=host_base
        )
    draft_input.device_accept_candidate_locs = candidates.to(
        batch.device, non_blocking=True
    )
    draft_input.device_accept_host_base = host_base.to(batch.device, non_blocking=True)
    # The host shadow stays one acceptance behind the device seq_lens.
    # Verify and draft-extend kernel lengths and the draft-extend
    # positions (draft_extend_positions_tpu) come from device seq_lens;
    # Pallas reads only the live-row count and the KV capacity check
    # from seq_lens_cpu, and the RPA backend ignores the draft-extend
    # prefix_lens derived from it. pallas_tpu_flash reads that
    # prefix_lens on host and has no spec support; it is not supported
    # with device accept routing.
    batch.seq_lens_cpu = host_base[:bs]
    batch.seq_lens_sum = int(host_base.sum())


def _new_decode_bucket(
    batch: ScheduleBatch,
    bs: int,
    pad_b: int,
    seq_lens_cpu: torch.Tensor | None = None,
) -> DecodeBucket:
    rows_cpu = pad_host_rows(batch.req_pool_indices_cpu.to(torch.int64), pad_b)
    valid = torch.zeros(pad_b, dtype=torch.int64)
    valid[:bs] = 1
    return DecodeBucket(
        logical_bs=bs,
        rows=rows_cpu.to(batch.device, non_blocking=True),
        rows_cpu=rows_cpu,
        valid=valid.to(batch.device, non_blocking=True),
        seq_lens_cpu=seq_lens_cpu,
    )


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
    assert selected.ndim == 2 and selected.shape[0] == bs, (
        "selected cache locations must match batch rows"
    )
    assert selected.shape[1] >= max(draft_width, num_verify_tokens), (
        "selected cache window must cover draft and verify tokens"
    )
    batch.out_cache_loc = selected[:, :draft_width].reshape(-1)
    return True


def device_accept_verify_cache_locs_tpu(
    draft_input: EagleDraftInput, *, bs: int, num_verify_tokens: int
) -> torch.Tensor | None:
    selected = draft_input.device_accept_selected_cache_locs
    if selected is None:
        return None
    assert selected.ndim == 2 and selected.shape[0] == bs, (
        "selected cache locations must match verify rows"
    )
    assert selected.shape[1] >= num_verify_tokens, (
        "selected cache window must cover verify tokens"
    )
    return selected[:, :num_verify_tokens].reshape(-1)


def filter_device_accept_tpu(
    draft_input: EagleDraftInput, new_indices: torch.Tensor
) -> None:
    # The decode prepare rebuilds the windows for the filtered rows, and the
    # acceptance lengths only mark that an acceptance is pending, so nothing is
    # indexed on device (that would compile per pair of batch sizes).
    draft_input.device_accept_candidate_locs = None
    draft_input.device_accept_host_base = None
    draft_input.device_accept_selected_cache_locs = None
    draft_input.decode_bucket = None


def merge_device_accept_tpu(
    draft_input: EagleDraftInput, spec_info: EagleDraftInput
) -> None:
    if draft_input.accept_lens_device is None or spec_info.accept_lens_device is None:
        draft_input.accept_lens_device = None
    draft_input.device_accept_candidate_locs = None
    draft_input.device_accept_host_base = None
    draft_input.device_accept_selected_cache_locs = None
    draft_input.decode_bucket = None


def last_accepted_row_indices(
    num_accept_tokens: torch.Tensor, num_tokens_per_req: int
) -> torch.Tensor:
    """Flat row of each request's last accepted token in a fixed-width window.

    Draft-extend packs every request into `num_tokens_per_req` rows, so request
    `i`'s last accepted row is `i * num_tokens_per_req + num_accept_tokens[i] - 1`.

    Compile-policy padding appends rows with a zero accept count. Clamping to
    one selects those requests' first (inert) row, which keeps every index in
    bounds; `finalize_spec_forward_batch` drops the padded outputs afterwards.

    `fused_draft_extend_select_argmax_func` builds its selection through this
    helper so the clamp rule has exactly one definition.
    """
    assert num_tokens_per_req > 0
    row_starts = (
        torch.arange(
            num_accept_tokens.shape[0],
            device=num_accept_tokens.device,
            dtype=torch.int64,
        )
        * num_tokens_per_req
    )
    return row_starts + num_accept_tokens.to(torch.int64).clamp_min(1) - 1


@compiled_op(warmup=make_draft_extend_select_args)
def fused_draft_extend_select_argmax_func(
    next_token_logits: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    num_accept_tokens: torch.Tensor,
):
    """Select draft-extend rows and compute their greedy next-draft tokens.

    Index construction stays inside this executable: passing the acceptance
    counts instead of pre-built indices keeps one more op inside the compiled
    graph. The window width is read off the logits rather than passed in, so it
    never becomes a second compile-time specialization key beyond the row count
    that already keys this graph.
    """
    # An idle draft-extend batch selects nothing, and its window width is not
    # observable from the logits; any positive width gives the same empty result.
    num_reqs = num_accept_tokens.shape[0]
    num_tokens_per_req = next_token_logits.shape[0] // num_reqs if num_reqs else 1
    select_index = last_accepted_row_indices(num_accept_tokens, num_tokens_per_req)
    selected_logits = next_token_logits[select_index]
    selected_hidden = hidden_states[select_index] if hidden_states is not None else None
    topk_index = torch.argmax(selected_logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    return selected_logits, selected_hidden, topk_p, topk_index


def _renorm_filtered(kept: torch.Tensor) -> torch.Tensor:
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def top_k_renorm_prob_tpu(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    """Sort-free top-k renormalization; keeps every value tied with the k-th."""
    from sglang_tpu.sampling_mask import tpu_filter_probs

    rows = probs.shape[0]
    kept = tpu_filter_probs(
        probs,
        top_ks,
        torch.ones(rows, dtype=torch.float32, device=probs.device),
        torch.zeros(rows, dtype=torch.float32, device=probs.device),
        need_top_k=True,
        need_top_p=False,
        need_min_p=False,
    )
    return _renorm_filtered(kept)


def top_p_renorm_prob_tpu(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Sort-free top-p renormalization; keeps every value tied with the cutoff."""
    from sglang_tpu.sampling_mask import tpu_filter_probs

    rows = probs.shape[0]
    kept = tpu_filter_probs(
        probs,
        torch.ones(rows, dtype=torch.int32, device=probs.device),
        top_ps,
        torch.zeros(rows, dtype=torch.float32, device=probs.device),
        need_top_k=False,
        need_top_p=True,
        need_min_p=False,
    )
    return _renorm_filtered(kept)
