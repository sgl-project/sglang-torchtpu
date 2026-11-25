"""Mixed prefill + speculative-verify steps under attention DP.

With speculative decoding and DP attention, a step in which any rank prefills is
otherwise a prefill step on every rank, so no rank decodes. With mixed verify
steps (``--enable-mixed-chunk`` on a platform that supports them), such a step
runs one forward per rank holding both the rank's prompt chunk rows and its
decoding requests' verify rows (``num_verify_rows`` each), and every rank runs
the same shapes:

- DECODE: no rank has prompt rows (the plain verify step).
- EXTEND: no rank has verify rows (the plain prefill step).
- MIXED: otherwise, on every rank; a rank with neither runs dummy rows.

This module holds the gate and the shape rules both the scheduler and the
platform's spec worker use; it has no platform dependencies.
"""

import enum
import functools
from typing import Callable, Optional, Sequence

import msgspec


class StepKind(enum.IntEnum):
    DECODE = 0
    EXTEND = 1
    MIXED = 2


class MixedVerifyPlan(msgspec.Struct, frozen=True):
    """A rank's part in a MIXED step, from the scheduler's decision gather.

    The step's batch is a container of native scheduler batches (``mixed_parts``):
    the rank's prompt batch (EXTEND) and its running batch prepared for decode
    (the verify part), either of which may be absent. The platform worker runs
    both in one forward and derives the shapes (verify bucket, prompt cap)
    from the gathered counts, identically on every rank.
    """

    n_prompt_rows: int  # this rank's prompt tokens (folded replay rows excluded)
    n_verify: int  # this rank's verify requests
    num_verify_rows: int  # rows per verify request (draft tokens + 1)
    global_prompt_rows: tuple[int, ...]  # per DP rank, from the decision gather
    global_verify_bs: tuple[int, ...]
    # Whether the verify rows share the prompt rows' forward (one fused forward);
    # otherwise the platform runs a verify forward then an extend forward, and
    # neither the prompt budget nor the MLP-sync count carries the verify rows.
    fused: bool = True

    @property
    def local_rows(self) -> int:
        """This rank's real rows in the MIXED forward (its MLP-sync token count)."""
        return mixed_rows(self.n_prompt_rows, self.n_verify, self.num_verify_rows)


def mixed_parts(batch):
    """(prompt batch or None, verify batch or None) of a MIXED step's container."""
    if batch is None or getattr(batch, "mixed_verify", None) is None:
        return None, None
    if batch.forward_mode.is_extend():
        return batch, batch.mixed_verify_batch
    if batch.forward_mode.is_decode():
        return None, batch
    return None, None


@functools.lru_cache(maxsize=1)
def mixed_verify_enabled() -> bool:
    """Whether this server runs mixed prefill + verify steps under DP attention."""
    from sglang.srt.platforms import current_platform
    from sglang.srt.runtime_context import get_parallel, get_schedule, get_spec

    # Mixed chunk first: off by default, and it keeps the check to one read.
    if not get_schedule().enable_mixed_chunk:
        return False
    algorithm = get_spec().speculative_algorithm
    return bool(
        algorithm is not None
        and get_parallel().attn_dp_enabled
        and current_platform.supports_spec_mixed_verify_dp(str(algorithm))
    )


def global_step_kind(
    global_prompt_rows: Sequence[int], global_verify_bs: Sequence[int]
) -> StepKind:
    """The step kind every rank derives from the same gathered counts."""
    has_prompt = any(n > 0 for n in global_prompt_rows)
    has_verify = any(n > 0 for n in global_verify_bs)
    if not has_prompt:
        return StepKind.DECODE
    if not has_verify:
        return StepKind.EXTEND
    return StepKind.MIXED


def gather_mixed_counts(prompt_rows: int, n_verify: int) -> tuple[list[int], list[int]]:
    """All-gather (prompt rows, verify requests) over the attention-DP ranks.

    One small CPU collective on the TP group's CPU group, as the prefill
    delayer's negotiation; the attention-TP ranks of a DP rank hold the same
    batch, so the first of each is taken."""
    import torch

    from sglang.srt.distributed.utils import all_gather_single
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    group = parallel.tp_group
    inner = parallel.attn_cp_size * parallel.attn_tp_size
    buf = torch.empty((parallel.num_dp_ranks, inner, 2), dtype=torch.int64)
    local = torch.tensor([int(prompt_rows), int(n_verify)], dtype=torch.int64)
    all_gather_single(buf.flatten(), local, group=group.cpu_group)
    per_rank = buf[:, 0, :]
    return per_rank[:, 0].tolist(), per_rank[:, 1].tolist()


def ceil_align(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def verify_bucket(
    global_verify_bs: Sequence[int],
    *,
    attn_tp_size: int,
    b_bucket: Callable[[int], int],
) -> int:
    """pad_bv of a MIXED step: the busiest rank's verify requests, aligned to
    attention TP and bucketed as the decode bucket is; 0 when no rank verifies."""
    max_bv = max(global_verify_bs, default=0)
    return b_bucket(ceil_align(max_bv, attn_tp_size)) if max_bv > 0 else 0


def mixed_rows(prompt_rows: int, n_verify: int, num_verify_rows: int) -> int:
    """A rank's real rows in a MIXED forward (its MLP-sync token count): the
    ragged prompt rows then its verify rows; the token bucket is the max over
    ranks, as for any DP extend."""
    return prompt_rows + n_verify * num_verify_rows


def prompt_token_budget(chunk_budget: int, n_verify: int, num_verify_rows: int) -> int:
    """Prompt tokens a rank may schedule in a MIXED step: its verify rows take
    the first share of the per-rank chunk budget, so prompt + verify rows stay
    within it (and within the extend token grid). check_mixed_verify_limits
    keeps this at least the minimum prompt share for any running batch."""
    return chunk_budget - n_verify * num_verify_rows


def mixed_step_fits(
    global_prompt_rows: Sequence[int],
    global_verify_bs: Sequence[int],
    num_verify_rows: int,
    row_cap: Optional[Callable[[int], int]],
) -> bool:
    """Whether a fused MIXED step adds no compiled shape: every rank's prompt
    rows (with replay rows) plus its verify rows stay within ``row_cap`` of the
    largest prompt row count (the platform's token bucket of that extend, or a
    wider one it allows). The same answer on every rank (gathered counts)."""
    if row_cap is None:
        return True
    cap = row_cap(max(global_prompt_rows))
    return all(
        p + v * num_verify_rows <= cap
        for p, v in zip(global_prompt_rows, global_verify_bs)
    )


def check_mixed_verify_limits(*, chunk_budget: Optional[int], overlap: bool) -> None:
    """Startup rules of mixed prefill + verify DP steps: the verify part's draft
    state relays through the overlap scheduler's future map only, and steps mix
    chunked prompts."""
    if not overlap:
        raise ValueError(
            "Mixed prefill + verify DP steps (--enable-mixed-chunk with this "
            "speculative algorithm under attention DP) require the overlap "
            "scheduler; remove --disable-overlap-schedule"
        )
    if chunk_budget is None:
        raise ValueError(
            "Mixed prefill + verify DP steps require chunked prefill "
            "(--chunked-prefill-size > 0)"
        )
