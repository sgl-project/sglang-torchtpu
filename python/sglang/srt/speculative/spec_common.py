"""Stateless speculative-decoding features shared by every spec worker.

Each function takes explicit tensors and small inputs, never a worker,
scheduler or model runner, so an out-of-tree worker gets the feature by
calling it. Each docstring names its trace class:

- traceable: static shapes, no host sync or data-dependent output size; safe
  inside a compiled region.
- host: runs between forwards on host metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import mamba_track_grid

if TYPE_CHECKING:
    import torch.nn as nn

    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


def adjust_verify_logits(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    draft_token_num: int,
) -> None:
    """Apply accumulated penalties and logit_bias to verify logits in place.

    Traceable. Each request's per-request row is repeated over its
    ``draft_token_num`` verify rows, a relaxed form of penalties for
    speculative decoding. Rows past ``len(sampling_info) * draft_token_num``
    (bucket padding) are left untouched.
    """
    # Looked up at call time: a platform may rebind this module attribute at startup.
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )

    if sampling_info.acc_additive_penalties is not None:
        additive_penalties = torch.repeat_interleave(
            sampling_info.acc_additive_penalties,
            draft_token_num,
            dim=0,
        )
        next_token_logits[: additive_penalties.shape[0]].add_(additive_penalties)
    if sampling_info.acc_scaling_penalties is not None:
        scaling_penalties = torch.repeat_interleave(
            sampling_info.acc_scaling_penalties,
            draft_token_num,
            dim=0,
        )
        apply_scaling_penalties(
            next_token_logits[: scaling_penalties.shape[0]],
            scaling_penalties,
        )
    if sampling_info.logit_bias is not None:
        logit_bias = torch.repeat_interleave(
            sampling_info.logit_bias, draft_token_num, dim=0
        )
        next_token_logits[: logit_bias.shape[0]].add_(logit_bias)


def seeded_verify_coins(
    *,
    sampling_seed: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive deterministic verify-side coins from per-request sampling seeds.

    Traceable.
    Mirrors the main seeded-sampling path: murmur_hash32(seed, seq_lens,
    column) mapped to [0, 1). Columns [0, draft_token_num) drive the
    per-draft rejection coins; column draft_token_num drives the final
    fallback-sampling coin.

    Scope: this seeds only the verify-side RNG. With rejection sampling the
    draft workers still pick candidates via unseeded multinomial
    (fast_sample in eagle_worker_v2), so that mode stays non-deterministic
    until the draft RNG is seeded in a follow-up; top-k/greedy draft
    selection is already deterministic.
    """
    from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

    cols = torch.arange(draft_token_num + 1, device=device, dtype=torch.int64)
    hashed = murmur_hash32(
        sampling_seed.to(torch.uint64), seq_lens.to(torch.uint64), cols
    )
    uniforms = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
    # The float32 cast rounds the top 129 uint32 hashes to exactly 1.0, but
    # the sampling kernels expect half-open [0, 1) coins: a 1.0 coin walks
    # past the last CDF bucket and can return a zero-probability token.
    # Clamp to the largest float32 below one; every other coin value is
    # untouched, so previously verified bitwise baselines stay intact.
    max_coin = 1.0 - 2**-24
    coins = (
        uniforms[:, :draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    coins_for_final_sampling = (
        uniforms[:, draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    return coins, coins_for_final_sampling


def verify_coins(
    *,
    sampling_info: SamplingBatchInfo,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    candidates: torch.Tensor,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rejection and final-sampling coins for verify: deterministic seeded
    coins when sampling_seed is set (see seeded_verify_coins), torch.rand
    otherwise. Traceable.
    """
    if sampling_info.sampling_seed is not None:
        return seeded_verify_coins(
            sampling_seed=sampling_info.sampling_seed,
            seq_lens=seq_lens,
            draft_token_num=draft_token_num,
            device=device,
        )
    # coins for rejection sampling
    coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
    # coins for final sampling
    coins_for_final_sampling = torch.rand(
        (candidates.shape[0],), dtype=torch.float32, device=device
    )
    return coins, coins_for_final_sampling


def build_dp_idle_draft_batch(batch: ScheduleBatch) -> ScheduleBatch:
    """An empty idle batch for the draft pass of a DP-attention prefill step.

    Host. It carries ``batch``'s per-rank token counts, so the draft forward
    enters the same DP collectives as the peer ranks drafting a decode batch.
    """
    draft_batch = ScheduleBatch.init_new(
        [],
        batch.req_to_token_pool,
        batch.token_to_kv_pool_allocator,
        batch.tree_cache,
        batch.model_config,
        batch.enable_overlap,
        batch.spec_algorithm,
    )
    draft_batch.prepare_for_idle()
    draft_batch.global_num_tokens = batch.global_num_tokens
    draft_batch.global_num_tokens_for_logprob = batch.global_num_tokens_for_logprob
    return draft_batch


def build_trivial_verify_input(
    *,
    batch: ScheduleBatch,
    topk: int,
    device,
    target_attn_backend: AttentionBackend,
) -> EagleVerifyInput:
    """Build a 1-node EagleVerifyInput rooted at the previous bonus token.

    Host. Used when ``speculative_num_steps == 0`` to skip drafting while still
    routing through the existing TARGET_VERIFY graph captured at
    ``draft_token_num=1``: the kernel always accepts the root and samples
    one new bonus token from target logits -- functionally a plain decode.
    """
    # Function-local: eagle_info imports eagle_utils, which imports this module.
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    if batch.forward_mode.is_idle():
        return EagleVerifyInput.create_idle_input(
            topk=topk, spec_steps=0, num_verify_tokens=1, device=device
        )

    draft_input = batch.spec_info
    bs = batch.seq_lens.shape[0]

    retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
    retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
    retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

    verify_mask = target_attn_backend.verify_mask
    # Every position in a 1-node tree is visible, so an all-True fill is
    # correct under either layout.
    if verify_mask is not None and verify_mask.fits(bs):
        custom_mask = verify_mask.buffer
        custom_mask.fill_(True)
    else:
        if batch.seq_lens_sum is not None:
            seq_lens_sum = batch.seq_lens_sum
        elif batch.seq_lens_cpu is not None:
            seq_lens_sum = int(batch.seq_lens_cpu.sum())
        else:
            seq_lens_sum = bs * target_attn_backend.max_context_len
        custom_mask = torch.ones(seq_lens_sum + bs, dtype=torch.bool, device=device)

    positions = batch.seq_lens.to(torch.int64)

    return EagleVerifyInput(
        draft_token=draft_input.bonus_tokens,
        custom_mask=custom_mask,
        positions=positions,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_cum_len=None,
        spec_steps=0,
        topk=topk,
        draft_token_num=1,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=None,
        seq_lens_cpu=None,
    )


def update_target_recurrent_state_after_verify(
    *,
    attn_backend: AttentionBackend,
    model: nn.Module,
    mamba_track_indices: Optional[torch.Tensor],
    tree_page_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens_pre_verify: torch.Tensor,
    seq_lens_post_verify: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Commit the target's recurrent (Mamba) state for each request's last accepted step.

    Traceable. TARGET_VERIFY runs the recurrent kernels with
    ``disable_state_update=True`` and caches one intermediate state per verify
    step; this commits the state at step ``commit_lens - 1``. When the radix
    cache tracks checkpoints, the step that crosses a track boundary inside the
    committed span is also saved (-1 where none is crossed).
    """
    last_correct_step_indices = commit_lens.to(torch.int64) - 1
    mamba_steps_to_track = None

    if mamba_track_indices is not None:
        mamba_track_interval = mamba_track_grid(tree_page_size)
        to_track_mask = (
            seq_lens_pre_verify // mamba_track_interval
            != seq_lens_post_verify // mamba_track_interval
        )
        tracking_point = (
            seq_lens_post_verify // mamba_track_interval * mamba_track_interval
        )
        to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
        can_track_mask = to_track_mask & (
            to_track_ith < commit_lens.to(to_track_ith.dtype)
        )
        mamba_steps_to_track = torch.where(
            can_track_mask,
            to_track_ith.to(torch.int64),
            torch.full_like(to_track_ith, -1, dtype=torch.int64),
        )

    if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=model,
            req_pool_indices=req_pool_indices[: commit_lens.shape[0]],
        )
