import math
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_tpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_use_torch_native = is_tpu()

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


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
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

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


def _verify_tree_greedy_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int,
):
    """TPU-specific greedy tree verification for EAGLE (topk=1 only)."""
    assert (
        topk == 1
    ), f"TPU EAGLE greedy verification currently supports topk=1, got {topk=}."

    bs, num_verify_tokens = candidates.shape
    predicts[: bs * num_verify_tokens] = target_predict.flatten().to(predicts.dtype)

    if num_verify_tokens > 1:
        matches = target_predict[:, :-1] == candidates[:, 1:]
    else:
        matches = torch.empty((bs, 0), dtype=torch.bool, device=candidates.device)

    # Vectorized accept-length computation (no Python loop, no .item()).
    # Step 1: find first mismatch per row.
    all_match = matches.all(dim=1)  # (bs,)
    # argmax on int32 of ~matches finds the first True (= first mismatch).
    # If all match, argmax returns 0 but all_match guards that case.
    first_mismatch = (~matches).to(torch.int32).argmax(dim=1)  # (bs,)
    num_correct = torch.where(all_match, num_verify_tokens - 1, first_mismatch)  # (bs,)
    accept_len = num_correct + 1  # (bs,)

    # Step 2: clamp at first invalid (-1) retrieve_index entry.
    # retrieve_index: (bs, num_verify_tokens).  Build a mask of valid
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
    num_correct = accept_len - 1  # (bs,)

    # Step 3: build accept_index and accept_token_num.
    # Copy retrieve_index values where col < accept_len.
    copy_mask = col_idx.unsqueeze(0) < accept_len.unsqueeze(1)  # (bs, N)
    accept_index[:bs] = torch.where(
        copy_mask,
        retrieve_index.to(accept_index.dtype),
        torch.tensor(-1, dtype=accept_index.dtype, device=candidates.device),
    )
    accept_token_num[:bs] = num_correct.to(accept_token_num.dtype)


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
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
        _verify_tree_greedy_torch(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            target_predict,
            topk,
        )
    return predicts, accept_index, accept_token_num


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
