# SPDX-License-Identifier: Apache-2.0
"""Prefill attention-work cap per DP rank while any rank decodes (opt-in).

Installed on the scheduler by sglang_tpu/hooks/attn_work_cap.py.

Under DP attention every rank steps in lockstep, so the slowest rank's prefill
sets the step time of every rank, decoders included. A chunk of ``n`` tokens
behind ``ctx`` cached tokens attends ``n * (ctx + n / 2)`` token pairs, so one
rank's 4096-token chunk deep in a 100k prompt costs far more than the same
chunk of a fresh prompt. ``SGLANG_DP_ATTN_WORK_CAP = W > 0`` keeps the sum over
a rank's prefill chunks in one pass at or under ``W`` while any DP rank has
running (decoding) requests; with no decoder anywhere nothing is capped.

* A request whose whole remaining prompt fits the room left is not touched.
* Otherwise its chunk is cut to the largest multiple of the legal cut unit
  (page, and the scheduler's truncation alignment) that fits; that request is
  the pass's last prefill (the scheduler tracks one partial request).
* The pass's first prefill chunk always gets at least one cut unit, so a chunk
  whose context alone exceeds ``W`` still advances.
* A request with a pending host-cache load-back is priced at both prefixes it
  may end at (loaded, or resident only if the load delivers nothing).
* A later request with less than one page of room waits for the next pass.
* ``chunked_prefill_size``, ``max_prefill_tokens`` and the memory budget stay
  upper bounds: the cap only lowers a chunk.

Modelled at c256 (cap 400M): p90 interactivity +13% at flat throughput.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional


def chunk_attn_work(n: int, ctx: int) -> float:
    """Token pairs a chunk of ``n`` tokens behind ``ctx`` cached tokens attends."""
    return n * (ctx + n / 2)


def attn_fit(ctx: int, room: float) -> int:
    """The most tokens ``n`` with ``chunk_attn_work(n, ctx) <= room``."""
    if room <= 0:
        return 0
    n = int(math.sqrt(ctx * ctx + 2 * room) - ctx)
    # Guard the float root against landing one token past the room.
    while n > 0 and chunk_attn_work(n, ctx) > room:
        n -= 1
    return n


def pass_attn_work(chunks: Iterable[tuple[int, int]]) -> float:
    """Work of the (prefix_len, extend_len) chunks a pass has admitted."""
    return sum(chunk_attn_work(n, ctx) for ctx, n in chunks)


def capped_chunk_tokens(
    *,
    cap: int,
    work: float,
    ctx: int,
    remaining: int,
    grid: int,
    first: bool,
) -> Optional[int]:
    """The chunk limit the cap sets for one request.

    None: the whole ``remaining`` prompt fits the room (no cap). Otherwise a
    multiple of ``grid`` (the legal cut unit: page, and any truncation
    alignment): the largest that fits, at least one unit for the pass's
    ``first`` prefill, else 0 (the request waits for a later pass).
    """
    fit = attn_fit(ctx, cap - work)
    if fit >= remaining:
        return None
    grid = max(grid, 1)
    fit = fit // grid * grid
    if fit >= grid:
        return fit
    return grid if first else 0


def batch_shows_decode(batch) -> Optional[bool]:
    """Whether some DP rank decoded in the step ``batch`` (this rank's batch,
    after the MLP sync) ran; None when the step does not tell.

    * no batch: under DP attention every rank was idle;
    * a mixed prefill + verify step: its plan holds every rank's verify count;
    * a decode / verify / idle step: ranks with gathered tokens decoded;
    * an extend step: decoders paused, so it does not tell.
    """
    if batch is None:
        return False
    plan = getattr(batch, "mixed_verify", None)
    if plan is not None:
        return any(n > 0 for n in plan.global_verify_bs)
    mode = batch.forward_mode
    if batch.is_extend_in_batch or (mode.is_extend() and not mode.is_target_verify()):
        return None
    tokens = batch.global_num_tokens
    if tokens is None:
        return None
    return any(n > 0 for n in tokens)
