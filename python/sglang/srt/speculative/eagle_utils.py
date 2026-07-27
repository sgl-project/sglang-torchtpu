from __future__ import annotations

import logging
import math
from collections import defaultdict
from enum import IntEnum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    alloc_paged_token_slots_extend_npu,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_build_dsv4_verify_bundle,
)
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_alloc_reserve_per_decode,
    get_last_loc,
)
from sglang.srt.speculative.spec_utils import (
    fill_bonus_tokens_func,
    renorm_draft_probs,
    sample_draft_proposal,
)
from sglang.srt.speculative.triton_ops.spec_tree import (
    sgl_build_tree_kernel_efficient_triton,
    verify_tree_greedy_kernel_triton,
)
from sglang.srt.utils import (
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_tpu,
    is_xpu,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.common import fast_topk

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()
_use_torch_native = is_tpu()

logger = logging.getLogger(__name__)

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


ALLOC_EXTEND_FUNCS = defaultdict(
    lambda: alloc_paged_token_slots_extend,
    {
        "npu": alloc_paged_token_slots_extend_npu,
    },
)


def per_step_draft_out_cache_loc(
    out_cache_loc: torch.Tensor,
    batch_size: int,
    topk: int,
    num_steps: int,
) -> torch.Tensor:
    """Per-step slice of the multi-step EAGLE draft out_cache_loc buffer.

    Single source of truth for the layout shared by EagleWorkerV2.draft_forward
    (per-step write target) and DeepseekV4AttnBackend (per-step compression
    write target baked into metadata).
    """
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


def _eagle_prefill_tail_tokens(
    batch: ScheduleBatch, next_token_ids: torch.Tensor
) -> torch.Tensor:
    """Per-seq tail token for EAGLE prefill rotation; uses next prompt token for
    non-final chunks (chunked-prefill chain consistency, see PR #26329)."""
    tail_tokens = next_token_ids.to(batch.input_ids.dtype)
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is not None:
        for i, r in enumerate(batch.reqs):
            if r is batch.chunked_req:
                tail_tokens = tail_tokens.clone()
                tail_tokens[i] = next_prompt_token
                break
    return tail_tokens


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    maybe_detect_oob(
        top_scores_index,
        0,
        ss_token_list.shape[1],
        "organize_draft_results: top_scores_index OOB for gather on ss_token_list",
    )
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, dtype=torch.long, device=parents_list[0].device
        )

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def _build_tree_kernel_efficient_torch(
    topk: int,
    tree_mask_mode: TreeMaskMode,
    num_verify_tokens: int,
    bs: int,
    device: torch.device,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
):
    """TPU-specific tree construction for EAGLE (topk=1 only)."""
    assert (
        topk == 1
    ), f"TPU EAGLE tree construction currently supports topk=1, got {topk=}."
    assert tree_mask_mode == TreeMaskMode.FULL_MASK
    token_offsets = torch.arange(num_verify_tokens, device=device, dtype=torch.long)
    positions.copy_((seq_lens[:, None] + token_offsets[None, :]).flatten())
    retrieve_index.copy_(
        torch.arange(bs * num_verify_tokens, device=device, dtype=torch.long).view(
            bs, num_verify_tokens
        )
    )
    if num_verify_tokens > 1:
        retrieve_next_token[:, :-1] = torch.arange(
            1, num_verify_tokens, device=device, dtype=torch.long
        )


def build_tree_kernel_efficient(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK and _use_torch_native:
        # TPU causal attention does not use the dense mask.
        tree_mask = torch.empty((0,), dtype=torch.bool, device=device)
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _use_torch_native:
        _build_tree_kernel_efficient_torch(
            topk,
            tree_mask_mode,
            num_verify_tokens,
            bs,
            device,
            seq_lens,
            positions,
            retrieve_index,
            retrieve_next_token,
        )
    elif _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_xpu:
        sgl_build_tree_kernel_triton(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


@compiled_op
def _verify_tree_greedy_torch(
    next_token_logits: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    topk: int = 1,
):
    """TPU greedy tree verification for EAGLE (topk=1).

    Target argmax and verification are compiled into the same dispatch.
    """
    assert (
        topk == 1
    ), f"TPU EAGLE greedy verification currently supports topk=1, got {topk=}."

    bs, num_verify_tokens = candidates.shape
    target_predict = torch.argmax(next_token_logits, dim=-1).reshape(
        bs, num_verify_tokens
    )
    predicts = target_predict.flatten().to(torch.int32)

    # Vectorized accept-length computation (no Python loop, no .item()).
    # Step 1: find first mismatch per row.
    if num_verify_tokens > 1:
        matches = target_predict[:, :-1] == candidates[:, 1:]
        all_match = matches.all(dim=1)  # (bs,)
        # argmax on int32 of ~matches finds the first True (= first mismatch).
        # If all match, argmax returns 0 but all_match guards that case.
        first_mismatch = (~matches).to(torch.int32).argmax(dim=1)  # (bs,)
        num_correct = torch.where(all_match, num_verify_tokens - 1, first_mismatch)
    else:
        # There are no drafts to check, only the bonus token.
        num_correct = torch.zeros((bs,), dtype=torch.long, device=candidates.device)
    accept_len = num_correct + 1  # (bs,)

    # Step 2: clamp at first invalid (-1) retrieve_index entry.
    # retrieve_index: (bs, num_verify_tokens). Build a mask of valid
    # positions (not -1) and find the first invalid one per row.
    ri_valid = retrieve_index != -1  # (bs, num_verify_tokens)
    # For each row, check if any position up to accept_len is invalid.
    col_idx = torch.arange(
        num_verify_tokens, device=candidates.device, dtype=torch.long
    )  # (num_verify_tokens,)
    within_accept = col_idx.unsqueeze(0) < accept_len.unsqueeze(1)  # (bs, N)
    invalid_within = within_accept & ~ri_valid  # (bs, N)
    has_invalid = invalid_within.any(dim=1)  # (bs,)
    first_invalid = invalid_within.to(torch.int32).argmax(dim=1)  # (bs,)
    accept_len = torch.where(has_invalid, first_invalid, accept_len)  # (bs,)

    # Step 3: build accept_index.
    # Copy retrieve_index values where col < accept_len.
    copy_mask = col_idx.unsqueeze(0) < accept_len.unsqueeze(1)  # (bs, N)
    accept_index = torch.where(
        copy_mask,
        retrieve_index.to(torch.int32),
        torch.tensor(-1, dtype=torch.int32, device=candidates.device),
    )
    return predicts, accept_index, accept_len.to(torch.int32)


def sgl_build_tree_kernel_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
):
    """Triton-based implementation."""
    # TODO: Add support for QLEN_ONLY_BITPACKING mode
    if tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        raise NotImplementedError(
            "QLEN_ONLY_BITPACKING is not supported in Triton implementation"
        )

    batch_size = verified_seq_len.shape[0]
    seq_len_prefix_sum = torch.cumsum(verified_seq_len, dim=0) - verified_seq_len

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    sgl_build_tree_kernel_efficient_triton[grid](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=int(tree_mask_mode),
        batch_size=batch_size,
        parent_list_stride=(
            parent_list.stride(0) if parent_list.dim() > 1 else parent_list.shape[0]
        ),
        selected_index_stride=selected_index.stride(0),
    )


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Triton-based implementation."""
    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    verify_tree_greedy_kernel_triton[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: Optional[torch.Tensor],
    topk: int = -1,
    next_token_logits: Optional[torch.Tensor] = None,
):
    if _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    elif _use_torch_native:
        assert next_token_logits is not None
        out_predicts, out_accept_index, accept_lens = _verify_tree_greedy_torch(
            next_token_logits, candidates, retrieve_index, topk
        )
        bs, num_verify_tokens = candidates.shape
        predicts[: bs * num_verify_tokens] = out_predicts.to(predicts.dtype)
        accept_index[:bs] = out_accept_index.to(accept_index.dtype)
        accept_token_num[:bs] = (accept_lens - 1).to(accept_token_num.dtype)
    elif _is_xpu:
        verify_tree_greedy_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num


@compiled_op
def fused_verify_post_func(
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_width: int,
):
    """Fused post-verify bookkeeping: new seq_lens + bonus-token gather."""
    new_seq_lens = seq_lens + accept_lens
    accept_tokens = predict[accept_index]
    bs = accept_lens.shape[0]
    idx = accept_width * torch.arange(bs, device=accept_lens.device) + accept_lens - 1
    bonus_tokens = accept_tokens.reshape(-1)[idx].to(torch.int32)
    return new_seq_lens, bonus_tokens


@compiled_op
def fused_draft_extend_select_argmax_func(
    next_token_logits: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    select_index: torch.Tensor,
):
    """Fused draft-extend row select + greedy next-draft pick."""
    sel_logits = next_token_logits[select_index]
    sel_hidden = hidden_states[select_index] if hidden_states is not None else None
    topk_index = torch.argmax(sel_logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    return sel_logits, sel_hidden, topk_p, topk_index


def verify_post_func(
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_width: int,
    is_idle: bool,
):
    """Route post-verification bookkeeping to the platform implementation."""
    if is_idle:
        return seq_lens + accept_lens, torch.empty(
            (0,), device=seq_lens.device, dtype=torch.int32
        )
    if _use_torch_native:
        return fused_verify_post_func(
            seq_lens, accept_lens, predict, accept_index, accept_width
        )

    new_seq_lens = seq_lens + accept_lens
    accept_tokens = predict[accept_index]
    bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
    fill_bonus_tokens_func(
        accept_tokens,
        accept_lens,
        bonus_tokens,
        accept_width,
    )
    return new_seq_lens, bonus_tokens


def draft_extend_select_func(
    next_token_logits: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    select_index: torch.Tensor,
    sampling_info: Any,
    topk: int,
    use_rejection_sampling: bool,
):
    """Route draft-extend row selection and sampling by platform and mode."""
    if _use_torch_native and topk == 1 and not use_rejection_sampling:
        selected_logits, selected_hidden, topk_p, topk_index = (
            fused_draft_extend_select_argmax_func(
                next_token_logits, hidden_states, select_index
            )
        )
        return selected_logits, selected_hidden, None, topk_p, topk_index

    selected_logits = next_token_logits[select_index]
    selected_hidden = hidden_states[select_index] if hidden_states is not None else None
    if use_rejection_sampling:
        draft_probs, topk_p, topk_index = sample_draft_proposal(
            selected_logits, sampling_info.temperatures
        )
    elif topk == 1 and not _is_hip:
        topk_index = torch.argmax(selected_logits, dim=-1, keepdim=True)
        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
        draft_probs = None
    else:
        probs = renorm_draft_probs(
            selected_logits, sampling_info, use_rejection_sampling
        )
        topk_p, topk_index = fast_topk(probs, topk, dim=-1)
        draft_probs = None
    return selected_logits, selected_hidden, draft_probs, topk_p, topk_index


def get_draft_input_from_target_hidden_dim(model_runner: ModelRunner) -> int:
    """Width of the target hidden states fed into the draft model.

    This is the single source of truth and is derived entirely from config: for
    EAGLE3 aux mode the draft consumes `num_aux` concatenated target layers
    (each `target_hidden_size` wide); every other arch consumes the per-layer
    `spec_hidden_size`.

    Do NOT read this off a draft projection's `in_features` (e.g. an `fc`
    layer): that width is arch-specific.

    Note: read entirely from the *draft* `model_runner`'s config. The non-aux
    branch assumes the draft's `spec_hidden_size` equals the target hidden width
    fed to the draft (true for standard EAGLE, where the draft mirrors the
    target hidden size); aux mode reads the explicit `target_hidden_size`.
    """
    model_config = model_runner.model_config
    hf_config = model_config.hf_config
    eagle_config = getattr(hf_config, "eagle_config", None) or {}
    get_eagle_config = (
        eagle_config.get
        if isinstance(eagle_config, dict)
        else lambda key, default=None: getattr(eagle_config, key, default)
    )
    use_aux = get_eagle_config("use_aux_hidden_state", True)
    spec_algorithm = model_runner.spec_algorithm

    if not (spec_algorithm is not None and spec_algorithm.is_eagle3() and use_aux):
        return model_config.spec_hidden_size

    target_hidden = getattr(hf_config, "target_hidden_size", None)
    if target_hidden is None:
        target_hidden = model_config.hidden_size
    num_aux = getattr(hf_config, "num_aux_hidden_states", None)
    if num_aux is None:
        layer_ids = get_eagle_config("eagle_aux_hidden_state_layer_ids", None)
        if layer_ids is None:
            layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        num_aux = len(layer_ids) if layer_ids else 3
    return target_hidden * num_aux


def get_draft_recurrent_hidden_state_spec(
    model_runner: ModelRunner,
) -> tuple[Optional[int], Optional[torch.dtype]]:
    """Return hidden_states width/dtype carried between draft decode steps."""
    if model_runner.spec_algorithm.is_standalone():
        return None, None
    return model_runner.model_config.spec_hidden_size, model_runner.model_config.dtype


def eagle_prepare_for_verify(
    verify_input: EagleVerifyInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
    target_worker: TpModelWorker,
):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.speculative.spec_utils import (
        assign_extend_cache_locs_func,
        prepare_mamba_track_for_verify,
    )

    if not batch.forward_mode.is_idle():
        # Assign cache locations
        bs = len(batch.req_pool_indices)
        batch.input_ids = verify_input.draft_token
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_verify input_ids",
        )
        device = batch.device
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + verify_input.draft_token_num,
            batch_size=bs,
            draft_token_num=verify_input.draft_token_num,
            device=device,
            host_req_pool_indices=batch.req_pool_indices_cpu,
            host_seq_lens=batch.seq_lens_cpu,
        )

        batch.out_cache_loc_dsv4 = maybe_build_dsv4_verify_bundle(
            batch, verify_input.draft_token_num
        )

        prepare_mamba_track_for_verify(batch)

        # TBO's split_spec_info reads these; no-verify-sync leaves both None.
        verify_input.seq_lens_cpu = batch.seq_lens_cpu
        verify_input.seq_lens_sum = (
            int(batch.seq_lens_cpu.sum()) if batch.seq_lens_cpu is not None else None
        )

    # Get a forward batch
    batch.forward_mode = (
        ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
    )
    capture_mode = (
        CaptureHiddenMode.NULL
        if target_worker.model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )
    batch.capture_hidden_mode = capture_mode
    verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

    # Run attention backend plan and cuda graph preparation
    can_run_cuda_graph = bool(
        target_worker.model_runner.decode_cuda_graph_runner
        and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
            verify_forward_batch
        )
    )
    if can_run_cuda_graph:
        target_worker.model_runner.decode_cuda_graph_runner.load_batch(
            verify_forward_batch
        )
        verify_forward_batch.mark_forward_metadata_ready()
    # Non-cuda-graph: defer init to forward_extend, which runs after
    # `_forward_raw -> prepare_mlp_sync_batch` pads the batch. Initing
    # here would use pre-pad shapes and trip DSv4 indexer shape match.

    return verify_forward_batch, can_run_cuda_graph


def eagle_sample(
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    vocab_mask: torch.Tensor = None,
):
    """
    Verify and find accepted tokens based on logits output and batch
    (which contains spec decoding information).
    """
    import torch.nn.functional as F

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_group,
        is_dp_attention_enabled,
    )
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        SIMULATE_ACC_TOKEN_MODE,
        generate_simulated_accept_index,
    )
    from sglang.srt.utils.async_probe import maybe_detect_nan, sanitize_nan_logits

    device = batch.device
    if batch.forward_mode.is_idle():
        predict = torch.empty(0, dtype=torch.int32, device=device)
        num_correct_drafts = torch.empty(0, dtype=torch.int32, device=device)
        accept_index = torch.empty(0, dtype=torch.int32, device=device)
        return predict, num_correct_drafts, accept_index

    bs = len(batch.seq_lens)
    sampling_info = batch.sampling_info
    next_token_logits = logits_output.next_token_logits

    sanitize_nan_logits(next_token_logits, "verify: target model logits")

    # Apply penalty
    # This is a relaxed version of penalties for speculative decoding.
    if sampling_info.acc_additive_penalties is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.acc_additive_penalties,
                verify_input.draft_token_num,
                dim=0,
            )
        )
    if sampling_info.acc_scaling_penalties is not None:
        apply_scaling_penalties(
            next_token_logits,
            torch.repeat_interleave(
                sampling_info.acc_scaling_penalties, verify_input.draft_token_num, dim=0
            ),
        )
    if sampling_info.logit_bias is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.logit_bias, verify_input.draft_token_num, dim=0
            )
        )

    # Apply grammar mask if provided
    if vocab_mask is not None:
        assert verify_input.grammar is not None
        verify_input.grammar.apply_vocab_mask(
            logits=next_token_logits, vocab_mask=vocab_mask
        )

    candidates = verify_input.draft_token.reshape(bs, verify_input.draft_token_num)
    predict_shape = list(next_token_logits.shape)[:-1]
    predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
    accept_index = torch.full(
        (bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=device
    )
    num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

    # Sample tokens
    target_predict = None
    if sampling_info.is_all_greedy or _is_npu or _is_hip or _is_xpu:
        target_predict = torch.argmax(next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, verify_input.draft_token_num)
        predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            retrieve_index=verify_input.retrieve_index,
            retrieve_next_token=verify_input.retrieve_next_token,
            retrieve_next_sibling=verify_input.retrieve_next_sibling,
            target_predict=target_predict,
            topk=verify_input.tree_topk,
            next_token_logits=next_token_logits,
        )
    else:
        if _use_torch_native:
            top_k_renorm_prob = top_k_renorm_prob_func
            top_p_renorm_prob = top_p_renorm_prob_func
            tree_speculative_sampling_target_only = (
                tree_speculative_sampling_target_only_func
            )
        else:
            from sgl_kernel import (
                top_k_renorm_prob,
                top_p_renorm_prob,
                tree_speculative_sampling_target_only,
            )

            from sglang.srt.speculative.reject_sampling import (
                chain_speculative_sampling_triton,
            )

        use_rejection_sampling = (
            get_global_server_args().speculative_use_rejection_sampling
        )

        # Apply temperature and get target probs
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, verify_input.draft_token_num, dim=0
        )  # (bs * num_draft_tokens, 1)

        target_probs = F.softmax(
            next_token_logits / expanded_temperature, dim=-1
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ks, verify_input.draft_token_num, dim=0
            ),
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after top_k_renorm")
        target_probs = top_p_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ps, verify_input.draft_token_num, dim=0
            ),
        )
        maybe_detect_nan(target_probs, "v2 verify: target_probs after top_p_renorm")
        target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)
        draft_probs = (
            verify_input.draft_probs
            if use_rejection_sampling
            else torch.zeros_like(target_probs)
        )
        # Defense-in-depth behind the spec_hook startup allowlist: validate the
        # actual kernel inputs (catches draft_probs plumbing regressions or a
        # startup guard bypassed by a worker subclass) before the Triton kernel.
        if use_rejection_sampling and (
            draft_probs is None or draft_probs.shape[-1] != target_probs.shape[-1]
        ):
            raise ValueError(
                "Rejection sampling requires a target-vocab draft proposal "
                "distribution; the current speculative algorithm/draft worker "
                "does not produce one (draft_probs missing or vocab-mismatched)."
            )

        # coins for rejection sampling
        coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
        # coins for final sampling
        coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=device)

        if _use_torch_native:
            sampling_fn = tree_speculative_sampling_target_only
        else:
            sampling_fn = (
                chain_speculative_sampling_triton
                if use_rejection_sampling
                else tree_speculative_sampling_target_only
            )
        sampling_fn(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=verify_input.retrieve_index,
            retrive_next_token=verify_input.retrieve_next_token,
            retrive_next_sibling=verify_input.retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_global_server_args().speculative_accept_threshold_single,
            threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
            deterministic=True,
        )

        # Sync sampling results across TP ranks: different GPUs may
        # produce slightly different target_probs due to floating-point
        # non-determinism in softmax/top_k/top_p, causing different
        # sampled tokens. Broadcast from rank 0 to ensure consistency.
        tp_group = (
            get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)

    if SIMULATE_ACC_LEN > 0:
        # Do simulation. The helper builds (and returns) a replacement
        # accept_index of width spec_steps + 1, so pass max_tree_depth - 1
        # to keep the simulated width identical to the real one.
        if SIMULATE_ACC_TOKEN_MODE not in ("fixed", "real-draft-token"):
            raise ValueError(
                "Invalid SGLANG_SIMULATE_ACC_TOKEN_MODE "
                f"{SIMULATE_ACC_TOKEN_MODE!r}; expected 'fixed' or "
                "'real-draft-token'."
            )

        if SIMULATE_ACC_TOKEN_MODE == "real-draft-token":
            if verify_input.tree_topk != 1:
                raise ValueError(
                    "SGLANG_SIMULATE_ACC_LEN with real draft tokens currently "
                    "requires speculative_eagle_topk=1."
                )

            # Use target argmax as the synthetic bonus for non-greedy requests.
            if target_predict is None:
                num_active_tokens = bs * verify_input.draft_token_num
                target_predict = torch.argmax(next_token_logits, dim=-1)
                if target_predict.shape != (bs, verify_input.draft_token_num):
                    # Execution padding retains inactive token rows in logits;
                    # trim them before restoring the active batch/token shape.
                    target_predict = target_predict[:num_active_tokens].reshape(
                        bs, verify_input.draft_token_num
                    )
        accept_index = generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predict,  # mutable
            num_correct_drafts=num_correct_drafts,  # mutable
            candidates=candidates,
            target_predict=target_predict,
            simulate_acc_len=SIMULATE_ACC_LEN,
            simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
            bs=bs,
            spec_steps=verify_input.max_tree_depth - 1,
        )

    # `num_correct_drafts` stays drafts-only inside this function; the returned
    # tensor includes the trailing/bonus token via out-of-place +1 so the
    # name no longer flips semantics mid-function (naming doc C2).
    return predict, num_correct_drafts + 1, accept_index


def eagle_prepare_for_decode(batch: ScheduleBatch):
    batch.maybe_evict_swa()

    from sglang.srt.speculative.spec_utils import (
        allocate_and_assign_decode_cache,
        assign_req_to_token_pool_func,
    )

    bs = batch.batch_size()

    # Accumulate penalty
    # This is a relaxed version of penalties for speculative decoding.
    if batch.sampling_info.penalizer_orchestrator.is_required:
        batch.cumulate_penalty_output_tokens()

    page_size = batch.token_to_kv_pool_allocator.page_size
    double_alloc = get_alloc_reserve_per_decode()

    cur_kv_lens = [0] * bs
    nxt_kv_lens = [0] * bs
    num_needed_tokens = 0
    for i, r in enumerate(batch.reqs):
        cur = r.kv_allocated_len
        # max(cur, ...) clamps so adaptive downswitch cannot make nxt < cur.
        # kv_committed_len is honest (bonus committed in resolve, not here),
        # so it lags batch.seq_lens by ~1 verify in overlap; 2*alloc absorbs.
        nxt = max(cur, r.kv_committed_len + double_alloc)
        cur_kv_lens[i] = cur
        nxt_kv_lens[i] = nxt
        num_needed_tokens += nxt - cur
        r.kv_allocated_len = nxt
        r.decode_batch_idx += 1

    cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
    nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

    # Fail fast if the page>1 + topk>1 draft over-allocation
    # (get_alloc_reserve_per_decode) outgrows the req_to_token row: the write below
    # would OOB and free would leak KV. The row is widened to hold it in _init_pools
    # (PR #26972); fail here with a clear error, not on a later cryptic CUDA assert.
    from sglang.srt.server_args import get_global_server_args

    if page_size > 1 and (get_global_server_args().speculative_eagle_topk or 1) > 1:
        max_alloc_len = int(nxt_kv_lens_cpu.max())
        row_width = batch.req_to_token_pool.req_to_token.shape[1]
        assert max_alloc_len <= row_width, (
            f"spec v2 page>1 topk>1 draft over-allocation ({max_alloc_len}) exceeds "
            f"req_to_token row width ({row_width}); page_size={page_size}. Widen the "
            f"row to hold committed + get_alloc_reserve_per_decode (PR #26972)."
        )

    if _use_torch_native and batch.req_to_token_pool.host_resident:
        out_cache_loc = allocate_and_assign_decode_cache(
            tree_cache=batch.tree_cache,
            req_to_token_pool=batch.req_to_token_pool,
            req_pool_indices=batch.req_pool_indices,
            cur_kv_lens_cpu=cur_kv_lens_cpu,
            nxt_kv_lens_cpu=nxt_kv_lens_cpu,
            num_needed_tokens=num_needed_tokens,
            batch_size=bs,
            device=batch.device,
            host_req_pool_indices=batch.req_pool_indices_cpu,
        )
    else:
        # non_blocking H2D: a blocking .to() syncs the schedule stream, which the WAR
        # barrier has chained to the prev forward -> host stalls a full forward.
        cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens_device,
            )
            device_type = getattr(
                batch.device, "type", str(batch.device).split(":", 1)[0]
            )
            out_cache_loc = ALLOC_EXTEND_FUNCS[device_type](
                batch.tree_cache,
                cur_kv_lens_device,
                cur_kv_lens_cpu,
                nxt_kv_lens_device,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
                req_pool_indices=batch.req_pool_indices,
                batch=batch,
            )
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_device,
            nxt_kv_lens_device,
            out_cache_loc,
            bs,
        )


def _top_k_renorm_prob_torch(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    """Keep the top-k probabilities per row and renormalize (torch fallback)."""
    vocab = probs.shape[-1]
    ks = top_ks.clamp(min=1, max=vocab).to(torch.long).reshape(-1, 1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    kth = sorted_probs.gather(-1, ks - 1)  # k-th largest prob per row
    kept = torch.where(probs >= kth, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _top_p_renorm_prob_torch(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Keep the nucleus (top-p) per row and renormalize (torch fallback)."""
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    ps = top_ps.to(probs.dtype).reshape(-1, 1)
    # Keep a token if the prefix mass *before* it is still below p (so the token
    # that crosses the threshold is included).
    keep_sorted = (cumsum - sorted_probs) < ps
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    kept = torch.where(keep, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def top_k_renorm_prob_func(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    need_top_k_sampling: bool = True,
) -> torch.Tensor:
    if _use_torch_native:
        if not need_top_k_sampling:
            return probs
        return _top_k_renorm_prob_torch(probs, top_ks)
    from sgl_kernel import top_k_renorm_prob

    return top_k_renorm_prob(probs, top_ks)


def top_p_renorm_prob_func(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    if _use_torch_native:
        return _top_p_renorm_prob_torch(probs, top_ps)
    from sgl_kernel import top_p_renorm_prob

    return top_p_renorm_prob(probs, top_ps)


def _tree_speculative_sampling_target_only_torch(
    predicts: torch.Tensor,  # mutable, flat [bs * num_draft_tokens]
    accept_index: torch.Tensor,  # mutable, [bs, num_draft_tokens]
    accept_token_num: torch.Tensor,  # mutable, [bs]
    candidates: torch.Tensor,  # [bs, num_draft_tokens]
    retrive_index: torch.Tensor,  # [bs, num_draft_tokens] (global flat positions)
    uniform_samples: torch.Tensor,  # [bs, num_draft_tokens]
    uniform_samples_for_final_sampling: torch.Tensor,  # [bs]
    target_probs: torch.Tensor,  # [bs, num_draft_tokens, vocab]
    draft_probs: torch.Tensor,  # mutable, [bs, num_draft_tokens, vocab]
    threshold_single: float,
    threshold_acc: float,
    topk: int,
):
    """TPU torch port of ``tree_speculative_sampling_target_only`` for topk=1.

    For topk=1 the draft tree is a chain, so this is sequential rejection
    sampling along the chain, followed by an inverse-CDF sample of the bonus/corrected
    token from the renormalized residual.
    """
    assert topk == 1, (
        "TPU tree speculative sampling currently supports topk=1 only, "
        f"got topk={topk}."
    )
    threshold_acc = max(float(threshold_acc), 1e-9)
    bs, num_draft = candidates.shape
    device = candidates.device
    ar = torch.arange(bs, device=device)
    cand = candidates.to(torch.long)

    coin = uniform_samples[:, 0].to(target_probs.dtype).clone()
    num_acc = torch.zeros(bs, dtype=torch.long, device=device)
    active = torch.ones(bs, dtype=torch.bool, device=device)
    has_rejected = torch.zeros(bs, dtype=torch.bool, device=device)
    rejected_token = torch.zeros(bs, dtype=torch.long, device=device)
    out_tokens = torch.zeros(bs, num_draft, dtype=predicts.dtype, device=device)

    for j in range(1, num_draft):
        cur_pos = num_acc  # parent position whose target row gates candidate j
        tok_j = cand[:, j]
        tp = target_probs[ar, cur_pos, tok_j]
        accept_cond = (coin <= tp / threshold_acc) | (tp >= threshold_single)
        accept_now = active & accept_cond
        reject_now = active & ~accept_cond

        # Write the accepted draft token at column num_acc (pre-increment).
        write_col = torch.where(accept_now, num_acc, torch.zeros_like(num_acc))
        out_tokens[ar, write_col] = torch.where(
            accept_now, tok_j.to(predicts.dtype), out_tokens[ar, write_col]
        )
        rejected_token = torch.where(reject_now, tok_j, rejected_token)
        has_rejected = has_rejected | reject_now

        num_acc = num_acc + accept_now.long()
        coin = torch.where(accept_now, uniform_samples[ar, j].to(coin.dtype), coin)
        active = active & ~reject_now

    # Bonus / corrected token: sample from the renormalized residual of the
    # last accepted position's target row.
    cur_pos = num_acc
    target_row = target_probs[ar, cur_pos].to(torch.float32)
    vocab_idx = torch.arange(target_row.shape[-1], device=device).view(1, -1)
    rejected_mask = has_rejected.view(bs, 1) & (vocab_idx == rejected_token.view(bs, 1))
    resid = torch.where(rejected_mask, torch.zeros_like(target_row), target_row)
    resid_sum = resid.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    u = uniform_samples_for_final_sampling.to(torch.float32) * resid_sum.squeeze(-1)
    cdf = torch.cumsum(resid, dim=-1)
    bonus_id = (cdf < u.unsqueeze(-1)).sum(dim=-1).clamp_(max=resid.shape[-1] - 1)
    out_tokens[ar, num_acc] = bonus_id.to(predicts.dtype)

    # Scatter results: predicts[retrive_index[bx, k]] = out_tokens[bx, k] for
    # the num_acc+1 valid columns; accept_index marks them, rest = -1.
    col = torch.arange(num_draft, device=device)
    valid = col.unsqueeze(0) <= num_acc.unsqueeze(1)  # [bs, num_draft]
    idx = retrive_index.to(torch.long)
    # Static-shape scatter: boolean-mask indexing (predicts[idx[valid]] = ...)
    # lowers to nonzero() on TPU (data-dependent shape -> D2H sync/OOM), so scatter
    # all positions with an index tensor and keep the prior value where not accepted.
    # Safe for topk=1, whose retrieve chain has no -1 padding or duplicate targets.
    flat_idx = idx.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_val = out_tokens.reshape(-1).to(predicts.dtype)
    predicts[flat_idx] = torch.where(flat_valid, flat_val, predicts[flat_idx])
    accept_index[:bs] = torch.where(
        valid,
        retrive_index.to(accept_index.dtype),
        torch.tensor(-1, dtype=accept_index.dtype, device=device),
    )
    accept_token_num[:bs] = num_acc.to(accept_token_num.dtype)


def tree_speculative_sampling_target_only_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
    topk: int = 1,
):
    """Dispatch to the torch port (TPU, topk=1) or the sgl_kernel op."""
    if _use_torch_native:
        _tree_speculative_sampling_target_only_torch(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            topk=topk,
        )
        return
    from sgl_kernel import tree_speculative_sampling_target_only

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )
