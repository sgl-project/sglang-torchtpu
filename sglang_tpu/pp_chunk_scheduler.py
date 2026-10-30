# SPDX-License-Identifier: Apache-2.0
"""Prefill admission using a profiled TPU pipeline step budget."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    MAX_PREFIX_MATCHED_WAITING_REQS,
    AddReqResult,
    PrefillAdder,
    matched_prefix_len,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang_tpu.envs import envs
from sglang_tpu.pp_attention_capacity import (
    attention_schedule_capacity,
)
from sglang_tpu.pp_chunks import (
    StepCostModel,
    chunk_granularity,
    longest_prompt,
    profile_points,
    uses_dynamic_chunks,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def pending_step_limit(scheduler: Scheduler) -> int:
    """Spread outstanding prefill work over stages, rounded up to a bucket.

    Outstanding work is the unscheduled tokens of waiting and partial
    requests plus the chunks launched into the pipeline whose results have
    not come back. A lone prompt still splits into one chunk per stage; under
    load the chunks grow with the work already in flight, since every stage
    advances in lockstep at the pace of the largest chunk in the pipeline.
    """
    cost = scheduler._tpu_pp_chunk_cost
    requests = list(scheduler.waiting_queue)
    if scheduler.chunked_req is not None:
        requests.append(scheduler.chunked_req)
    cached_prefix = scheduler._tpu_pp_cached_prefix
    waiting = {req.rid for req in scheduler.waiting_queue}
    for rid in [rid for rid in cached_prefix if rid not in waiting]:
        del cached_prefix[rid]
    seen = set()
    pending = 0
    matches_left = MAX_PREFIX_MATCHED_WAITING_REQS
    for req in requests:
        if req.rid in seen:
            continue
        seen.add(req.rid)
        # The policy's match counts host hits; a partial request's computed
        # chunks extend prefix_indices past that match.
        cached = max(len(req.prefix_indices), req.num_matched_prefix_tokens)
        if cached == 0 and req.rid in cached_prefix:
            cached = cached_prefix[req.rid]
        elif cached == 0 and matches_left > 0:
            # Matched once per waiting request; its cached prefix is not pending work,
            # and the match refreshes that prefix's recency ahead of admission.
            matches_left -= 1
            cached = matched_prefix_len(scheduler.tree_cache, req)
            cached_prefix[req.rid] = cached
        pending += max(len(req.origin_input_ids) + len(req.output_ids) - cached, 0)
        if pending >= get_parallel().pp_size * cost.buckets[-1]:
            # The limit saturates at a full step; the rest need no match.
            break
    # A micro-batch slot holds its batch until the result is processed, when
    # the same batch is recorded as the slot's last batch.
    for batch, done in zip(scheduler.mbs, scheduler.last_mbs):
        if batch is not None and batch is not done:
            pending += _prefill_tokens(batch)
    granule = cost.granularity
    share = -(-pending // get_parallel().pp_size)
    return min(cost.buckets[-1], max(granule, -(-share // granule) * granule))


def _prefill_tokens(batch: ScheduleBatch) -> int:
    """Prefill tokens of a launched batch. A decode batch keeps the token
    count of the prefill it was built from, and a mixed batch counts one
    token per decode row."""
    if not batch.forward_mode.is_extend():
        return 0
    return (batch.extend_num_tokens or 0) - len(batch.decoding_reqs or ())


class TpuPipelinePrefillAdder(PrefillAdder):
    """Apply time and attention limits alongside the allocator's KV budgets."""

    def __init__(
        self, *args: Any, cost: StepCostModel, step_limit: int, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cost = cost
        self.step_limit = step_limit
        self.step_tokens = 0
        self.step_pairs = 0.0
        self.step_squares = 0.0
        self.step_schedule = 0
        mixed = kwargs.get("num_mixed_decode_tokens", args[7] if len(args) > 7 else 0)
        if self.running_batch is not None and mixed:
            # Mixed decode tokens have already been reserved in rem_input_tokens.
            for req in self.running_batch.reqs[:mixed]:
                self._account(1, len(req.origin_input_ids) + len(req.output_ids) - 1)

    def _account(self, tokens: int, prefix: int) -> None:
        self.step_tokens += tokens
        self.step_pairs += tokens * prefix
        self.step_squares += tokens * tokens
        self.step_schedule += self.cost.pairs(tokens, prefix)

    @contextmanager
    def _request_budget(
        self, req: Req, *, include_pending_host_load: bool = True
    ) -> Iterator[int]:
        original = self.rem_chunk_tokens
        prefix = len(req.prefix_indices)
        if include_pending_host_load and req.needs_host_load_back():
            prefix += req.host_hit_length
        remaining = len(req.full_untruncated_fill_ids) - prefix
        cap = self.cost.chunk(
            prefix,
            remaining,
            self.step_tokens,
            self.step_pairs,
            self.step_squares,
            self.step_limit,
            self.step_schedule,
        )
        if cap == 0 and not self.can_run_list and original > 0:
            cap = min(remaining, self.cost.granularity)
            if (
                self.cost.schedule is not None
                and self.step_schedule + self.cost.pairs(cap, prefix)
                > self.cost.schedule[0]
            ):
                raise ValueError(
                    "The minimum pipeline chunk exceeds the attention schedule capacity; "
                    "reduce --chunked-prefill-size or --context-length."
                )
        # Admission reserves whole KV pages; the cost model counts query tokens.
        budget = (
            self.ceil_paged_tokens(cap)
            if cap >= remaining
            else cap // self.page_size * self.page_size
        )
        budget = min(budget, max(original, 0) // self.page_size * self.page_size)
        cap = min(cap, budget)
        self.rem_chunk_tokens = budget
        before = len(self.can_run_list)
        try:
            yield cap
        finally:
            used_budget = budget - self.rem_chunk_tokens
            self.rem_chunk_tokens = original - used_budget
            if len(self.can_run_list) > before:
                self._account(req.extend_range.length, len(req.prefix_indices))

    def add_chunked_req(self, req: Req) -> Optional[Req]:
        with self._request_budget(req, include_pending_host_load=False) as cap:
            if cap <= 0:
                return req
            return super().add_chunked_req(req)

    def add_one_req(
        self,
        req: Req,
        has_chunked_req: bool,
        truncation_align_size: Optional[int],
        chunk_tokens_cap: Optional[int] = None,
    ) -> AddReqResult:
        before = len(self.can_run_list)
        with self._request_budget(req) as cap:
            if cap <= 0:
                return AddReqResult.OTHER
            remaining = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
            if req.needs_host_load_back():
                remaining -= req.host_hit_length
            if has_chunked_req and cap < remaining:
                return AddReqResult.OTHER
            result = super().add_one_req(
                req=req,
                has_chunked_req=has_chunked_req,
                truncation_align_size=truncation_align_size,
                chunk_tokens_cap=chunk_tokens_cap,
            )
        if self.new_chunked_req is not None:
            return AddReqResult.OTHER
        return self.budget_state() if len(self.can_run_list) > before else result


def profile_and_install_cost_model(scheduler: Scheduler) -> None:
    """Profile every stage after compilation and share the slowest cost curve."""
    runner = scheduler.tp_worker.model_runner
    if not uses_dynamic_chunks():
        return
    if not get_exec().graph.enable_torch_compile:
        raise ValueError("TPU pipeline dynamic chunks require --enable-torch-compile.")
    if envs.SGLANG_TPU_PP_CHUNK_SLACK.get() < 0:
        raise ValueError("SGLANG_TPU_PP_CHUNK_SLACK must be nonnegative.")
    policy = runner.compile_policy
    granularity = chunk_granularity(
        scheduler.page_size, scheduler.chunked_prefill_size, True
    )
    buckets = policy.t_bucketer.buckets
    if buckets[-1] > runner.model_config.context_len:
        raise ValueError(
            "TPU pipeline dynamic chunks require --chunked-prefill-size "
            "not to exceed --context-length."
        )
    schedule = attention_schedule_capacity(runner, policy)
    from sglang_tpu.pp_chunks import chunk_pairs

    if schedule is not None and chunk_pairs(schedule, buckets[-1], 0) > schedule[0]:
        raise ValueError(
            "A full pipeline chunk exceeds the attention schedule capacity."
        )
    from sglang_tpu.pp_profile import profile_cache

    samples: list[tuple[int, int, float]] = []
    with profile_cache(policy, buckets[-1]):
        for tokens, prefix in profile_points(
            buckets, buckets[-1], runner.model_config.context_len, schedule
        ):
            kwargs = dict(
                num_tokens=tokens, bs=1, mode=ForwardMode.EXTEND, profile_prefix=prefix
            )
            policy._run_one(**kwargs)
            elapsed = min(policy._run_one(**kwargs) for _ in range(2))
            samples.append((tokens, prefix, elapsed))
    # Every rank fits one model from every rank's samples, so the devices of a
    # multi-device stage admit the same chunks.
    # TODO(yinlin09): all schedulers of a stage must pick identical chunks each step;
    # let the stage leader decide and broadcast, or check agreement in the stage.
    reports = [None] * scheduler.world_group.world_size
    torch.distributed.all_gather_object(
        reports, (samples, schedule), group=scheduler.world_group.cpu_group
    )
    schedules = [report[1] for report in reports if report[1] is not None]
    cost = StepCostModel.fit(
        [report[0] for report in reports],
        envs.SGLANG_TPU_PP_CHUNK_SLACK.get(),
        granularity,
        min(schedules) if schedules else None,
    )
    context_len = runner.model_config.context_len
    if cost.schedule is not None:
        # The smallest chunk behind the longest prefix must fit, so that no
        # prompt within the context length is refused inside the scheduler.
        longest = longest_prompt(cost.schedule, granularity, context_len)
        if longest < context_len:
            raise ValueError(
                f"The batched attention kernel can schedule the last "
                f"{granularity}-token chunk of prompts up to {longest} tokens; "
                f"lower --context-length from {context_len} to {longest}."
            )
    scheduler._tpu_pp_chunk_cost = cost
    # Cached prefix tokens of each waiting request, matched once.
    scheduler._tpu_pp_cached_prefix = {}
    logger.info("TPU pipeline chunks: %s; stage samples %s", cost.describe(), samples)
