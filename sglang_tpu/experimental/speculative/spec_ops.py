"""Device ops for TpuSpecWorker: draft sampling and lossless acceptance.

Fixed shapes only; no host reads. The compiled ops take [pbs] bucket rows and are
warmed per batch bucket, by precompile_spec_v2_ops or by the owning worker when
the warmup needs model config.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.speculative.spec_common import (
    VerifyLogitModifiers,
    adjust_verify_logits,
)
from sglang_tpu.dflash_utils_tpu import select_dflash_verify_cache_locs_tpu
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    DraftBlockInputs,
    VerifyRows,
    map_carry,
)
from sglang_tpu.sampling.seeded import sample_categorical, sample_draft_with_seed
from sglang_tpu.sampling_mask import tpu_filter_probs
from sglang_tpu.spec_op_warmup import SpecOpWarmupContext


@compiled_op()
def apply_verify_logit_modifiers(
    logits: torch.Tensor, modifiers: VerifyLogitModifiers, num_verify_rows: int
) -> torch.Tensor:
    result = logits.clone()
    adjust_verify_logits(
        next_token_logits=result,
        sampling_info=modifiers,
        draft_token_num=num_verify_rows,
    )
    return result


@compiled_op()
def apply_token_bitmask(logits: torch.Tensor, bitmask: torch.Tensor) -> torch.Tensor:
    """logits with -inf where a packed int32 grammar bitmask clears a token's bit.

    xgrammar's apply_token_bitmask_inplace rule: bit j of word w allows token
    32 * w + j, and columns past the bitmask keep their logits. Plain ops, so the
    owning worker warms it per verify bucket; the grammar backends' TPU path is a
    torch.compile kernel that nothing warms.
    """
    num_words = bitmask.shape[-1]
    width = min(logits.shape[-1], 32 * num_words)
    shifts = torch.arange(32, dtype=torch.int32, device=bitmask.device)
    allowed = ((bitmask[:, :, None] >> shifts) & 1).reshape(-1, 32 * num_words)
    masked = torch.where(allowed[:, :width] != 0, logits[:, :width], float("-inf"))
    if width == logits.shape[-1]:
        return masked
    return torch.cat([masked, logits[:, width:]], dim=-1)


def sample_tokens(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    coins=None,
    sampling_seed=None,
    positions=None,
) -> torch.Tensor:
    """Sample one draft token per [pbs, V] logits row; rows with top_k <= 1 take argmax.

    Uses the exponential race argmax(p / E), E ~ Exp(1), instead of multinomial
    so the op lowers to a fixed-shape argmax on TPU. Seeded rows draw from the
    draft hash column, never a coin that verify or the target sampler reads.
    """
    greedy = torch.argmax(logits, dim=-1)
    temps = temperatures.float().clamp_min(1e-5)[:, None]
    probs = torch.softmax(logits.float() / temps, dim=-1)
    if sampling_seed is not None:
        sampled = sample_draft_with_seed(probs, sampling_seed, positions)
    else:
        sampled = (
            _exponential_race(probs)
            if coins is None
            else sample_categorical(probs, coins)
        )
    return torch.where(top_ks <= 1, greedy, sampled)


@compiled_op()
def pad_state_rows(
    bonus_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_carry: Any,
    padded_bs: int,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Live [bs] decode state zero-padded to the [padded_bs] bucket in one program.

    seq_lens become int64; draft_carry is a NamedTuple of row-aligned tensors or
    None. The owning worker warms each (bs, padded_bs) pair it can serve.
    """

    def pad(rows: torch.Tensor) -> torch.Tensor:
        pad_rows = padded_bs - rows.shape[0]
        return torch.nn.functional.pad(rows, [0, 0] * (rows.dim() - 1) + [0, pad_rows])

    return (
        pad(bonus_tokens),
        pad(seq_lens.to(torch.int64)),
        map_carry(draft_carry, pad),
    )


@compiled_op(warmup=lambda ctx: _verify_rows_warmup_args(ctx))
def select_verify_rows(
    verify_cache_candidates: torch.Tensor,
    committed_lens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> VerifyRows:
    """Positions and KV slots of the verify rows; candidates [pbs, 2 * (N + 1)], rest [pbs]."""
    num_rows = verify_cache_candidates.shape[1] // 2
    cache_locs = select_dflash_verify_cache_locs_tpu(
        verify_cache_candidates, seq_lens, committed_lens, num_rows
    )
    offsets = torch.arange(num_rows, device=seq_lens.device, dtype=seq_lens.dtype)
    positions = seq_lens[:, None] + offsets
    return VerifyRows(
        positions=positions,
        cache_locs=cache_locs,
        flat_positions=positions.reshape(-1),
        flat_cache_locs=cache_locs.reshape(-1),
    )


@compiled_op(warmup=lambda ctx: _verify_tokens_warmup_args(ctx))
def build_verify_tokens(
    bonus_tokens: torch.Tensor, draft_tokens: torch.Tensor
) -> torch.Tensor:
    """Flat [pbs * (N + 1)] verify input [r, d_1, ..., d_N] per row."""
    return torch.cat([bonus_tokens[:, None], draft_tokens], dim=1).reshape(-1)


@compiled_op()
def build_draft_block_inputs(
    bonus_tokens: torch.Tensor,
    rows: VerifyRows,
    seq_lens: torch.Tensor,
    live_mask: torch.Tensor,
    mask_token_id: int,
    width: int,
) -> DraftBlockInputs:
    """Block-parallel draft inputs [r, m x (width - 1)] over the first width verify rows.

    Model config sets width and mask_token_id, so the owning worker warms this op.
    """
    pbs = bonus_tokens.shape[0]
    mask_tokens = torch.full(
        (pbs, width - 1),
        mask_token_id,
        dtype=bonus_tokens.dtype,
        device=bonus_tokens.device,
    )
    return _block_inputs(
        torch.cat([bonus_tokens[:, None], mask_tokens], dim=1),
        rows,
        seq_lens,
        live_mask,
        offset=0,
    )


@compiled_op()
def build_chain_block_inputs(
    tokens: torch.Tensor,
    rows: VerifyRows,
    seq_lens: torch.Tensor,
    live_mask: torch.Tensor,
    offset: int,
) -> DraftBlockInputs:
    """Draft inputs for tokens [pbs * width] on verify rows offset .. offset + width - 1.

    The chain worker sends offset 0 at full width (draft extend) and each draft
    step at width 1, so it warms this op per offset.
    """
    # Callers pass flat tokens: a [:, None] view input costs an eager as_strided per bucket.
    tokens = tokens.reshape(seq_lens.shape[0], -1)
    inputs = _block_inputs(tokens, rows, seq_lens, live_mask, offset=offset)
    # TARGET_VERIFY RPA adds the row width to seq_lens itself; the post-row length
    # would let each row attend width stale draft KV slots past it. Built anew
    # rather than with _replace, which Dynamo cannot trace.
    return DraftBlockInputs(
        input_ids=inputs.input_ids,
        positions=inputs.positions,
        cache_locs=inputs.cache_locs,
        seq_lens=inputs.extend_prefix_lens.to(seq_lens.dtype),
        extend_prefix_lens=inputs.extend_prefix_lens,
        extend_seq_lens=inputs.extend_seq_lens,
    )


def _block_inputs(
    tokens: torch.Tensor,
    rows: VerifyRows,
    seq_lens: torch.Tensor,
    live_mask: torch.Tensor,
    *,
    offset: int,
) -> DraftBlockInputs:
    width = tokens.shape[1]
    live = live_mask.to(seq_lens.dtype)
    extend_seq_lens = live.to(torch.int32) * width
    end = offset + width
    return DraftBlockInputs(
        input_ids=tokens.reshape(-1),
        positions=rows.positions[:, offset:end].reshape(-1),
        cache_locs=rows.cache_locs[:, offset:end].reshape(-1),
        seq_lens=seq_lens + live * end,
        extend_prefix_lens=(seq_lens + live * offset).to(torch.int32),
        extend_seq_lens=extend_seq_lens,
    )


@compiled_op(warmup=lambda ctx: _accept_warmup_args(ctx, sampled=False))
def accept_drafts_greedy(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_verify_drafts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy acceptance: d_j accepted while d_j == argmax p_j; bonus = argmax p_k.

    target_logits are the flat [pbs * (N + 1), V] verify rows; draft_tokens is
    [pbs, N] and num_verify_drafts [pbs] caps how many drafts are verified.
    Returns num_correct_drafts and bonus_tokens, [pbs] each.
    """
    bs, n = draft_tokens.shape
    target_argmax = torch.argmax(target_logits.view(bs, n + 1, -1), dim=-1)
    ok = (draft_tokens == target_argmax[:, :n]) & _verify_mask(num_verify_drafts, n)
    num_correct_drafts = _leading_true(ok)
    return num_correct_drafts, pick(target_argmax, num_correct_drafts)


@compiled_op(warmup=lambda ctx: _accept_warmup_args(ctx, sampled=True), uses_rng=True)
def accept_drafts_sampled(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_verify_drafts: torch.Tensor,
    draft_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    coins: torch.Tensor | None = None,
    final_sampling_coins: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lossless acceptance of a draft chain (Leviathan et al. 2023).

    Draft j is accepted with prob min(1, p_j(d_j) / q_j(d_j)); at the first
    rejection k the bonus is drawn from norm(max(p_k - q_k, 0)), else from p_k.
    Rows with top_k <= 1 take the greedy rule. draft_logits [bs, N, V] are the
    logits q was sampled from (before temperature); temperatures and top_ks are [pbs].
    Returns num_correct_drafts and bonus_tokens, [pbs] each.
    """
    bs, n = draft_tokens.shape
    target_logits = target_logits.view(bs, n + 1, -1)
    temps = temperatures.view(-1).float().clamp_min(1e-5)[:, None, None]
    p = torch.softmax(target_logits.float() / temps, dim=-1)
    q = torch.softmax(draft_logits.float() / temps, dim=-1)
    return _accept_drafts_rejection_sampling(
        p,
        q,
        torch.argmax(target_logits, dim=-1),
        draft_tokens,
        num_verify_drafts,
        top_ks,
        coins,
        final_sampling_coins,
    )


@compiled_op(uses_rng=True)
def accept_drafts_sampled_filtered(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_verify_drafts: torch.Tensor,
    draft_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    coins: torch.Tensor | None = None,
    final_sampling_coins: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """accept_drafts_sampled against the target filtered like the target sampler.

    Filtering only p keeps the rule lossless for the filtered target: q is the
    distribution the draft actually sampled from. Greedy rows still compare
    against the raw target_logits. The owning worker warms it.
    """
    bs, n = draft_tokens.shape
    temps = temperatures.view(-1).float().clamp_min(1e-5)[:, None, None]
    q = torch.softmax(draft_logits.float() / temps, dim=-1)
    p = _filtered_target_probs(
        target_logits, temperatures, top_ks, top_ps, min_ps, num_verify_rows=n + 1
    )
    return _accept_drafts_rejection_sampling(
        p.view(bs, n + 1, -1),
        q,
        torch.argmax(target_logits.view(bs, n + 1, -1), dim=-1),
        draft_tokens,
        num_verify_drafts,
        top_ks,
        coins,
        final_sampling_coins,
    )


def _accept_drafts_rejection_sampling(
    p: torch.Tensor,
    q: torch.Tensor,
    target_argmax: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_verify_drafts: torch.Tensor,
    top_ks: torch.Tensor,
    coins: torch.Tensor | None = None,
    final_sampling_coins: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sampling acceptance over p [bs, N + 1, V] and q [bs, N, V]; greedy rows where top_k <= 1.

    target_argmax [bs, N + 1] comes from the raw logits: argmax over p can flip
    on probabilities that round equal.
    """
    n = draft_tokens.shape[1]
    verify_mask = _verify_mask(num_verify_drafts, n)

    greedy_ok = (draft_tokens == target_argmax[:, :n]) & verify_mask
    greedy_num_correct = _leading_true(greedy_ok)
    greedy_bonus = pick(target_argmax, greedy_num_correct)

    vocab_onehot = draft_tokens[..., None] == torch.arange(
        p.shape[-1], device=draft_tokens.device
    )
    p_draft = torch.where(vocab_onehot, p[:, :n], 0).sum(-1)
    q_draft = torch.where(vocab_onehot, q, 0).sum(-1)
    u = torch.rand_like(p_draft) if coins is None else coins[:, :n]
    # u < p / q without dividing by q. The coins are half-open, [0, 1), so the
    # strict test accepts with probability p / q and never at p = 0.
    sample_ok = (u * q_draft < p_draft) & verify_mask
    sample_num_correct = _leading_true(sample_ok)

    p_at = pick(p, sample_num_correct)
    q_at = pick(q, sample_num_correct.clamp_max(n - 1))
    residual = (p_at - q_at).clamp_min(0)
    # All-zero residual only arises from rounding when p == q; p is exact there.
    residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, p_at)
    rejected = (sample_num_correct < num_verify_drafts)[:, None]
    weights = torch.where(rejected, residual, p_at)
    sample_bonus = (
        _exponential_race(weights)
        if final_sampling_coins is None
        else sample_categorical(weights, final_sampling_coins)
    )

    greedy = (top_ks <= 1).view(-1)
    return (
        torch.where(greedy, greedy_num_correct, sample_num_correct),
        torch.where(greedy, greedy_bonus, sample_bonus),
    )


def _filtered_target_probs(
    target_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    *,
    num_verify_rows: int,
) -> torch.Tensor:
    """Target p [pbs * (N + 1), V]: temperature, then the target sampler's filter.

    Like TpuSampler, every cutoff is taken on the unfiltered probabilities and
    the survivors are normalized once. Runs inside the accept op's program.
    """
    temps = temperatures.view(-1).float().clamp_min(1e-5)
    temps = temps.repeat_interleave(num_verify_rows)[:, None]
    # Every flag on: disabled rows carry no-op values (TOP_K_ALL, 1.0, 0.0), so
    # this is one program per shape instead of one per flag combination.
    kept = tpu_filter_probs(
        torch.softmax(target_logits.float() / temps, dim=-1),
        top_ks.repeat_interleave(num_verify_rows),
        top_ps.repeat_interleave(num_verify_rows),
        min_ps.repeat_interleave(num_verify_rows),
        need_top_k=True,
        need_top_p=True,
        need_min_p=True,
    )
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@compiled_op(uses_rng=True)
def accept_drafts_target_only(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_verify_drafts: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    coins: torch.Tensor | None = None,
    final_sampling_coins: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lossless acceptance of a deterministic draft chain against the filtered target p.

    With a one-hot draft q, Leviathan's rule reduces to: accept d_j with prob
    p_j(d_j); at the first rejection k the bonus is drawn from p_k with d_k
    zeroed, else from p_k. This is TPU EAGLE's target-only rule, including the
    speculative_accept_threshold_{single,acc} relaxations (1.0 disables them).
    target_logits are the flat [pbs * (N + 1), V] verify rows; p is filtered
    like the target sampler. Rows with top_k <= 1 take the greedy rule.
    Returns num_correct_drafts and bonus_tokens, [pbs] each.
    """
    bs, n = draft_tokens.shape
    p = _filtered_target_probs(
        target_logits, temperatures, top_ks, top_ps, min_ps, num_verify_rows=n + 1
    ).view(bs, n + 1, -1)
    verify_mask = _verify_mask(num_verify_drafts, n)

    target_argmax = torch.argmax(p, dim=-1)
    greedy_ok = (draft_tokens == target_argmax[:, :n]) & verify_mask
    greedy_num_correct = _leading_true(greedy_ok)
    greedy_bonus = pick(target_argmax, greedy_num_correct)

    vocab = torch.arange(p.shape[-1], device=draft_tokens.device)
    p_draft = torch.where(draft_tokens[..., None] == vocab, p[:, :n], 0).sum(-1)
    u = torch.rand_like(p_draft) if coins is None else coins[:, :n]
    # Strict on the half-open coins, as in accept_drafts_sampled: never at p = 0.
    sample_ok = (
        (u * threshold_acc < p_draft) | (p_draft >= threshold_single)
    ) & verify_mask
    sample_num_correct = _leading_true(sample_ok)

    p_at = pick(p, sample_num_correct)
    rejected = sample_num_correct < num_verify_drafts
    rejected_token = pick(draft_tokens, sample_num_correct.clamp_max(n - 1))
    rejected_mask = rejected[:, None] & (vocab[None, :] == rejected_token[:, None])
    residual = torch.where(rejected_mask, 0, p_at)
    # All-zero residual only arises from rounding when p_k(d_k) == 1.
    residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, p_at)
    sample_bonus = (
        _exponential_race(residual)
        if final_sampling_coins is None
        else sample_categorical(residual, final_sampling_coins)
    )

    greedy = (top_ks <= 1).view(-1)
    return (
        torch.where(greedy, greedy_num_correct, sample_num_correct),
        torch.where(greedy, greedy_bonus, sample_bonus),
    )


@compiled_op(warmup=lambda ctx: _finalize_warmup_args(ctx))
def finalize_accept(
    draft_tokens: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    bonus_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    live_mask: torch.Tensor,
) -> AcceptResult:
    """Accepted tokens and new lengths; padded rows commit nothing so their positions stay 0."""
    n = draft_tokens.shape[1]
    rows_idx = torch.arange(n + 1, device=draft_tokens.device)[None, :]
    num_correct_col = num_correct_drafts[:, None]
    padded_drafts = torch.nn.functional.pad(draft_tokens, (0, 1), value=-1)
    accept_tokens = torch.where(
        rows_idx < num_correct_col,
        padded_drafts,
        torch.where(rows_idx == num_correct_col, bonus_tokens[:, None], -1),
    )
    accept_lens = torch.where(live_mask, num_correct_drafts + 1, 0)
    return AcceptResult(
        num_correct_drafts=num_correct_drafts,
        bonus_tokens=bonus_tokens,
        accept_tokens=accept_tokens,
        accept_lens=accept_lens.to(torch.int32),
        next_seq_lens=seq_lens + accept_lens,
        # Rows past the accepted prefix only write KV past the new length, which
        # later steps overwrite before reading, so any valid id serves.
        window_tokens=accept_tokens.clamp_min(0).reshape(-1),
        window_seq_lens=seq_lens + (n + 1),
        live_mask=live_mask,
        last_steps=accept_lens.to(torch.int64) - 1,
    )


@compiled_op()
def live_result_rows(
    accept_tokens: torch.Tensor, accept_lens: torch.Tensor, bs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The scheduler's [bs] accept rows as fresh buffers, flat tokens first.

    Eager [:bs] clones would defer and fuse into the next FutureMap store, one
    program per (bucket, bs, store) that warmup never builds.
    """
    return accept_tokens[:bs].reshape(-1), accept_lens[:bs]


def _verify_mask(num_verify_drafts: torch.Tensor, n: int) -> torch.Tensor:
    steps = torch.arange(n, device=num_verify_drafts.device)
    return steps[None, :] < num_verify_drafts[:, None]


def _accept_warmup_args(
    ctx: SpecOpWarmupContext, *, sampled: bool
) -> tuple[torch.Tensor, ...]:
    # Must match the serving call: float32 logits, int64 drafts, int32 caps.
    bs, rows = ctx.physical_bs, ctx.num_verify_tokens
    n = rows - 1
    args = (
        torch.zeros(
            (bs * rows, ctx.vocab_size), dtype=torch.float32, device=ctx.device
        ),
        torch.zeros((bs, n), dtype=torch.int64, device=ctx.device),
        torch.full((bs,), n, dtype=torch.int32, device=ctx.device),
    )
    if not sampled:
        return args
    return args + (
        torch.zeros((bs, n, ctx.vocab_size), dtype=torch.float32, device=ctx.device),
        torch.ones((bs,), dtype=torch.float32, device=ctx.device),
        torch.ones((bs,), dtype=torch.int32, device=ctx.device),
    )


def _verify_rows_warmup_args(
    ctx: SpecOpWarmupContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, rows = ctx.physical_bs, ctx.num_verify_tokens
    return (
        torch.zeros((bs, 2 * rows), dtype=torch.int64, device=ctx.device),
        torch.zeros((bs,), dtype=torch.int64, device=ctx.device),
        torch.zeros((bs,), dtype=torch.int64, device=ctx.device),
    )


def _verify_tokens_warmup_args(
    ctx: SpecOpWarmupContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, n = ctx.physical_bs, ctx.num_verify_tokens - 1
    return (
        torch.zeros((bs,), dtype=torch.int64, device=ctx.device),
        torch.zeros((bs, n), dtype=torch.int64, device=ctx.device),
    )


def _finalize_warmup_args(ctx: SpecOpWarmupContext) -> tuple[torch.Tensor, ...]:
    bs, n = ctx.physical_bs, ctx.num_verify_tokens - 1
    return (
        torch.zeros((bs, n), dtype=torch.int64, device=ctx.device),
        torch.zeros((bs,), dtype=torch.int64, device=ctx.device),
        torch.zeros((bs,), dtype=torch.int64, device=ctx.device),
        torch.zeros((bs,), dtype=torch.int64, device=ctx.device),
        torch.ones((bs,), dtype=torch.bool, device=ctx.device),
    )


# TPU-safe forms, as in correct_drafts_and_bonus_tpu / sample_exponential_race:
# no cumprod, no advanced-index gather, no exponential_ (U64 lowering).
def _leading_true(mask: torch.Tensor) -> torch.Tensor:
    """Length of the all-True prefix along the last dim, [bs] int64."""
    first_false = (~mask).to(torch.int32).argmax(dim=-1)
    return torch.where(mask.all(dim=-1), mask.shape[-1], first_false)


def pick(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """values[b, index[b], ...] via a one-hot mask, [bs, ...]."""
    onehot = (
        torch.arange(values.shape[1], device=values.device)[None, :] == index[:, None]
    )
    onehot = onehot.view(onehot.shape + (1,) * (values.dim() - 2))
    return torch.where(onehot, values, 0).sum(dim=1)


def _exponential_race(weights: torch.Tensor) -> torch.Tensor:
    """argmax(w / E), E = -log(u) ~ Exp(1)."""
    u = torch.rand_like(weights, dtype=torch.float32)
    noise = torch.log(u.clamp_min(torch.finfo(torch.float32).tiny)).neg_()
    return torch.argmax(weights / noise, dim=-1)
