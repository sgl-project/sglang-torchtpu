# SPDX-License-Identifier: Apache-2.0
"""Prefill attention-work cap while any DP rank decodes, and shortest-prefill-first.

Why: under DP attention one rank's long-context prefill chunk sets the step time
     of every rank, so decode on all ranks stalls behind it
     (sglang_tpu/attn_work_cap.py has the rule and the evidence). Ordering the
     waiting prompts shortest first is the other half; core already has it as
     --schedule-policy shortest-prefill-first.
How: pass-through unless SGLANG_DP_ATTN_WORK_CAP > 0 (cap) or
     SGLANG_PREFILL_SPF=1 (order).
  * AFTER Scheduler._finish_next_batch records whether some DP rank decoded in
    the step just formed (its MLP sync, or a mixed step's plan); an extend step
    keeps the previous answer.
  * AROUND Scheduler._get_new_batch_prefill_raw scopes one prefill pass: the cap
    applies when this rank's running batch decodes or (DP attention) some rank
    decoded in the last step that tells.
  * AROUND SchedulePolicy.shortest_prefill_chunk_limit, the per-pass limit of
    the resumed chunked request (core assigns it to chunked_req_limit before
    adding it first), lowers that limit to the cap.
  * AROUND PrefillAdder.add_one_req lowers chunk_tokens_cap to the cap (core
    ends the pass after a capped request is cut), or refuses a request that
    gets no cut unit (page and truncation alignment) of room after the pass's
    first prefill. A pending host-cache load-back is priced at both prefixes
    it may end at; ignore_eos requests without a radix cache (core's
    add_one_req_ignore_eos takes no chunk_tokens_cap) get the cap as a lowered
    chunk budget.
  * AROUND Scheduler.init_schedule_policy builds the policy as
    shortest-prefill-first when SGLANG_PREFILL_SPF=1 and the configured policy
    is fcfs, and refuses the cap together with the hierarchical (host) cache
    (known limitation, see check_cap_config).
Upstream PR or issue: none.
Remove when: core has a per-pass prefill work budget and a decode-presence
     signal across DP ranks.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager

from sglang.srt.plugins.hook_registry import HookType
from sglang_tpu.attn_work_cap import (
    batch_shows_decode,
    capped_chunk_tokens,
    pass_attn_work,
)

logger = logging.getLogger(__name__)

# Set on the scheduler by track_dp_decode.
_DECODE_SEEN = "_tpu_attn_cap_dp_decode_seen"
_LOG_EVERY_PASSES = 5000


class _Pass:
    """The cap of the prefill pass in flight (the scheduler is single-threaded);
    0 outside a pass and in passes the cap does not apply to."""

    cap = 0
    capped_passes = 0
    clipped_chunks = 0


def _cap_env() -> int:
    from sglang_tpu.envs import envs

    return envs.SGLANG_DP_ATTN_WORK_CAP.get()


@contextmanager
def capped_pass(cap: int):
    """Run one prefill pass under ``cap`` token pairs (0: uncapped)."""
    _Pass.cap = cap
    try:
        yield
    finally:
        _Pass.cap = 0


def track_dp_decode(result, scheduler, *args, **kwargs):
    if _cap_env() <= 0:
        return None
    seen = batch_shows_decode(result.batch_to_run)
    if seen is not None:
        setattr(scheduler, _DECODE_SEEN, seen)
    return None


def pass_cap(scheduler, running_batch) -> int:
    """The cap for a prefill pass of ``scheduler`` over ``running_batch``."""
    cap = _cap_env()
    if cap <= 0:
        return 0
    local = not running_batch.is_empty() and not running_batch.is_prefill_only
    if local or (
        scheduler.require_mlp_sync and getattr(scheduler, _DECODE_SEEN, False)
    ):
        return cap
    return 0


def cap_prefill_pass(original, scheduler, *args, **kwargs):
    running_batch = kwargs["running_batch"] if "running_batch" in kwargs else args[1]
    cap = pass_cap(scheduler, running_batch)
    if not cap:
        return original(scheduler, *args, **kwargs)
    _Pass.capped_passes += 1
    if _Pass.capped_passes % _LOG_EVERY_PASSES == 1:
        logger.info(
            "Prefill attention-work cap %d: capped_passes=%d clipped_chunks=%d",
            cap,
            _Pass.capped_passes,
            _Pass.clipped_chunks,
        )
    with capped_pass(cap):
        return original(scheduler, *args, **kwargs)


def cap_resumed_chunk(
    original, policy, chunked_req, waiting_queue, budget, page_size, **kwargs
):
    limit = original(policy, chunked_req, waiting_queue, budget, page_size, **kwargs)
    if not _Pass.cap:
        return limit
    # The resumed request is the pass's first prefill: at least one page.
    capped = capped_chunk_tokens(
        cap=_Pass.cap,
        work=0.0,
        ctx=chunked_req.prefix_len,
        remaining=len(chunked_req.full_untruncated_fill_ids) - chunked_req.prefix_len,
        grid=page_size,
        first=True,
    )
    if capped is None or (limit is not None and limit <= capped):
        return limit
    _Pass.clipped_chunks += 1
    return capped


def _truncation_align(args, kwargs):
    if "truncation_align_size" in kwargs:
        return kwargs["truncation_align_size"]
    return args[1] if len(args) >= 2 else None


def cap_new_request(original, adder, req, *args, **kwargs):
    if not _Pass.cap or adder.dllm_config is not None or adder.rem_chunk_tokens is None:
        return original(adder, req, *args, **kwargs)
    # A cut must land on the page and on the scheduler's truncation alignment,
    # or core rounds the minimum chunk down to nothing.
    align = _truncation_align(args, kwargs)
    grid = math.lcm(adder.page_size, align) if align else adder.page_size
    work = pass_attn_work((r.prefix_len, r.extend_len) for r in adder.can_run_list)
    first = not adder.can_run_list
    total = len(req.full_untruncated_fill_ids)
    # Price every prefix admission may end up at: a host-cache load-back can
    # deliver less than promised (down to nothing), and core then re-selects
    # the chunk behind the resident prefix alone with the same chunk_tokens_cap.
    contexts = {req.prefix_len + req.host_hit_length, req.prefix_len}
    limits = [
        capped_chunk_tokens(
            cap=_Pass.cap,
            work=work,
            ctx=ctx,
            remaining=total - ctx,
            grid=grid,
            first=first,
        )
        for ctx in contexts
    ]
    limits = [n for n in limits if n is not None]
    if not limits:
        return original(adder, req, *args, **kwargs)
    capped = min(limits)
    from sglang.srt.managers.schedule_policy import AddReqResult

    _Pass.clipped_chunks += 1
    if capped == 0:
        return AddReqResult.OTHER
    if req.sampling_params.ignore_eos and getattr(adder.tree_cache, "disable", True):
        # Core admits these through add_one_req_ignore_eos, which cuts at the
        # pass's chunk budget and takes no chunk_tokens_cap.
        return _with_chunk_budget(original, adder, req, capped, args, kwargs)
    if len(args) >= 3:
        args = (*args[:2], capped if args[2] is None else min(args[2], capped))
    else:
        cap = kwargs.get("chunk_tokens_cap")
        kwargs["chunk_tokens_cap"] = capped if cap is None else min(cap, capped)
    return original(adder, req, *args, **kwargs)


def _with_chunk_budget(original, adder, req, capped, args, kwargs):
    """Admit ``req`` with the pass's chunk budget lowered to ``capped``, then
    restore the budget less what the admission spent. A request cut short ends
    the pass, as core does for a capped request."""
    budget = adder.rem_chunk_tokens
    lowered = min(budget, capped)
    adder.rem_chunk_tokens = lowered
    try:
        result = original(adder, req, *args, **kwargs)
    finally:
        spent = lowered - adder.rem_chunk_tokens
        adder.rem_chunk_tokens = budget - spent
    if adder.new_chunked_req is req:
        adder.rem_chunk_tokens = 0
        return adder.budget_state()
    return result


def check_cap_config(scheduler) -> None:
    """Refuse configurations the cap cannot bound.

    Known limitation: with the hierarchical (host) cache, core may restore MORE
    host KV than the radix match promised and keep the selected chunk at that
    deeper prefix; the cap prices the promised prefix, so such a step can exceed
    W (e.g. 452M token-pairs under a 400M cap). Pricing the delivered prefix needs
    a hook inside core's load-back, so the cap and the host cache are not allowed
    together.
    """
    from sglang.srt.runtime_context import get_memory

    if _cap_env() > 0 and get_memory().enable_hierarchical_cache:
        raise ValueError(
            "SGLANG_DP_ATTN_WORK_CAP cannot be combined with the hierarchical "
            "(host) cache (--enable-hierarchical-cache): a host-cache restore can "
            "expose a deeper prefix than the cap priced, so the per-pass attention "
            "work could exceed the cap. Unset SGLANG_DP_ATTN_WORK_CAP or disable "
            "the hierarchical cache."
        )


def shortest_prefill_first(original, scheduler, *args, **kwargs):
    from sglang_tpu.envs import envs

    check_cap_config(scheduler)
    if not envs.SGLANG_PREFILL_SPF.get() or scheduler.schedule_policy != "fcfs":
        return original(scheduler, *args, **kwargs)
    if scheduler.enable_priority_scheduling:
        raise ValueError(
            "SGLANG_PREFILL_SPF=1 orders by uncached prompt tokens and cannot "
            "be combined with priority scheduling."
        )
    logger.info("SGLANG_PREFILL_SPF=1: schedule policy fcfs -> shortest-prefill-first")
    scheduler.schedule_policy = "shortest-prefill-first"
    return original(scheduler, *args, **kwargs)


def hook_specs() -> list[tuple]:
    scheduler = "sglang.srt.managers.scheduler.Scheduler"
    policy = "sglang.srt.managers.schedule_policy"
    return [
        (f"{scheduler}._finish_next_batch", track_dp_decode, HookType.AFTER),
        (f"{scheduler}._get_new_batch_prefill_raw", cap_prefill_pass, HookType.AROUND),
        (
            f"{policy}.SchedulePolicy.shortest_prefill_chunk_limit",
            cap_resumed_chunk,
            HookType.AROUND,
        ),
        (f"{policy}.PrefillAdder.add_one_req", cap_new_request, HookType.AROUND),
        (f"{scheduler}.init_schedule_policy", shortest_prefill_first, HookType.AROUND),
    ]
