# SPDX-License-Identifier: Apache-2.0
"""Prefill admission using a profiled TPU pipeline step budget."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from statistics import median
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.environ import envs as srt_envs
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
from sglang_tpu.pp_chunk_fit import (
    FitCalibration,
    FitFingerprint,
    SavedFit,
    fit_key,
    guard_points,
    guard_report,
    load_saved_fit,
    read_saved_fits,
    runtime_version,
    saved_cost_model,
    saved_fit,
    slowest_stage_points,
    weak_fingerprint_fields,
)
from sglang_tpu.pp_chunks import (
    StepCostModel,
    chunk_granularity,
    longest_prompt,
    profile_points,
    stage_min_samples,
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
    """Install the stage cost curve: a matching saved fit, else a profile of
    every stage after compilation, keeping the slowest stage's curve."""
    runner = scheduler.tp_worker.model_runner
    if not uses_dynamic_chunks():
        return
    if not get_exec().graph.enable_torch_compile:
        raise ValueError("TPU pipeline dynamic chunks require --enable-torch-compile.")
    slack = envs.SGLANG_TPU_PP_CHUNK_SLACK.get()
    _require_agreed_settings(scheduler, slack)
    policy = runner.compile_policy
    granularity = chunk_granularity(
        scheduler.page_size, scheduler.chunked_prefill_size, True
    )
    buckets = policy.t_bucketer.buckets
    context_len = runner.model_config.context_len
    if buckets[-1] > context_len:
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
    points = profile_points(buckets, buckets[-1], context_len, schedule)
    cost = _agreed_saved_cost_model(
        scheduler,
        points=points,
        schedule=schedule,
        slack=slack,
        granularity=granularity,
    )
    if cost is None:
        stage_samples, schedules = _profile_stages(
            scheduler, points=points, schedule=schedule
        )
        cost = StepCostModel.fit(
            stage_samples, slack, granularity, min(schedules) if schedules else None
        )
        if envs.SGLANG_TPU_PP_CHUNK_FIT_TABLE.get():
            _log_fit_candidate(
                scheduler,
                cost=cost,
                points=points,
                schedule=schedule,
                stage_samples=stage_samples,
            )
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
    if _is_lead():
        # Every rank installs the same model.
        logger.info("TPU pipeline chunks: %s", cost.describe())


def _require_agreed_settings(scheduler: Scheduler, slack: float) -> None:
    """Every rank must time the same number of runs per point and size chunks
    against the same slack: the runs launch collectives, so a rank that runs
    more of them waits in a collective its peers never enter. The settings are
    checked after the gather, so every rank raises on a bad value together."""
    runs = envs.SGLANG_TPU_PP_CHUNK_PROFILE_RUNS.get()
    gathered = _gather(scheduler, ("pp-chunk-settings", runs, slack))
    values = [setting[1:] for setting in gathered]
    if any(value != values[0] for value in values):
        raise ValueError(
            "SGLANG_TPU_PP_CHUNK_PROFILE_RUNS and SGLANG_TPU_PP_CHUNK_SLACK must "
            f"match on every rank; got (runs, slack) {values}."
        )
    if runs < 1:
        raise ValueError("SGLANG_TPU_PP_CHUNK_PROFILE_RUNS must be at least 1.")
    if slack < 0:
        raise ValueError("SGLANG_TPU_PP_CHUNK_SLACK must be nonnegative.")


def _time_points(policy, points: list[tuple[int, int]]) -> list[tuple[int, int, float]]:
    """Median of the configured runs per (tokens, prefix) point, after one warmup."""
    from sglang_tpu.pp_profile import profile_cache

    runs = envs.SGLANG_TPU_PP_CHUNK_PROFILE_RUNS.get()
    samples = []
    with profile_cache(policy, max(tokens for tokens, _ in points)):
        for tokens, prefix in points:
            kwargs = dict(
                num_tokens=tokens, bs=1, mode=ForwardMode.EXTEND, profile_prefix=prefix
            )
            policy._run_one(**kwargs)
            samples.append(
                (tokens, prefix, median(policy._run_one(**kwargs) for _ in range(runs)))
            )
    return samples


def _profile_stages(
    scheduler: Scheduler,
    *,
    points: list[tuple[int, int]],
    schedule: tuple[int, int, int] | None,
) -> tuple[list[list[tuple[int, int, float]]], list[tuple[int, int, int]]]:
    """Every stage's samples, timed after all ranks finish compiling.

    Without the barrier a stage times its steps while another stage on the
    same host is still compiling, and that host contention inflates the fit."""
    group = scheduler.world_group.cpu_group
    torch.distributed.barrier(group=group)
    samples = _time_points(scheduler.tp_worker.model_runner.compile_policy, points)
    # Every rank fits one model from every rank's samples, so the devices of a
    # multi-device stage admit the same chunks.
    # TODO(yinlin09): all schedulers of a stage must pick identical chunks each step;
    # let the stage leader decide and broadcast, or check agreement in the stage.
    reports = _gather(scheduler, (get_parallel().pp_rank, samples, schedule))
    stage_samples = stage_min_samples([(stage, timed) for stage, timed, _ in reports])
    if _is_lead():
        for stage, timed in enumerate(stage_samples):
            logger.info("TPU pipeline chunks: stage %d samples %s", stage, timed)
    return stage_samples, [report[2] for report in reports if report[2] is not None]


def _fit_fingerprint(
    points: list[tuple[int, int]], schedule: tuple[int, int, int] | None
) -> FitFingerprint:
    return FitFingerprint(
        profile_points=[(int(t), int(p)) for t, p in points],
        schedule=tuple(int(n) for n in schedule) if schedule is not None else None,
        kernel_version=envs.SGLANG_TPU_PP_CHUNK_FIT_KERNEL_VERSION.get(),
        runtime_version=runtime_version(),
    )


def _serving_fit_key(scheduler: Scheduler) -> str:
    from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
    from sglang_tpu.pallas_export import get_torch_tpu_device_kind

    runner = scheduler.tp_worker.model_runner
    model_config = runner.model_config
    parallel = get_parallel()
    return fit_key(
        model_type=model_config.hf_config.model_type,
        num_layers=model_config.num_hidden_layers,
        hidden_size=model_config.hidden_size,
        model_shape=_model_shape(
            model_config.hf_text_config, model_config.hf_quant_config
        ),
        dtype=str(model_config.dtype).removeprefix("torch."),
        quantization=model_config.quantization,
        kv_cache_dtype=str(runner.kv_cache_dtype).removeprefix("torch."),
        attention_backend=type(runner.attn_backend).__name__,
        moe_runner_backend=str(get_moe_runner_backend().value),
        moe_a2a_backend=str(get_moe_a2a_backend().value),
        device_kind=get_torch_tpu_device_kind(),
        pp_size=parallel.pp_size,
        tp_size=parallel.tp_size,
        attn_cp_size=parallel.attn_cp_size,
        attn_dcp_size=parallel.attn_dcp_size,
        moe_ep_size=parallel.moe_ep_size,
        moe_tp_size=parallel.moe_tp_size,
        # Unset means the dense layers split like attention, over all of TP.
        moe_dense_tp_size=parallel.moe_dense_tp_size or parallel.tp_size,
        attn_dp_size=parallel.attn_dp_size,
        chunk_tokens=scheduler.chunked_prefill_size,
        # Requests per compiled prefill batch (--prefill-max-requests, or its default).
        extend_batch_cap=runner.compile_policy._extend_compile_cap(),
        page_size=scheduler.page_size,
        layer_partition=os.environ.get("SGLANG_PP_LAYER_PARTITION", ""),
    )


# Text-config fields that name or label the checkpoint rather than shape it.
# Every other field goes into the digest: one that does not change the step
# cost only splits the table key, which costs a profile at launch.
_NOT_MODEL_SHAPE = frozenset(
    {
        "_name_or_path",
        "_commit_hash",
        "transformers_version",
        "architectures",
        "auto_map",
        "id2label",
        "label2id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "tokenizer_class",
        "torch_dtype",
        "dtype",
    }
)


def _model_shape(text_config, quant_config: dict) -> str:
    """A short digest of the text config, so two models with the same type,
    depth and width but different experts, heads or layer types get different
    table keys. ``quant_config`` is the top-level config's quantization
    settings, which wrapper and multimodal checkpoints keep outside the text
    config."""
    import json
    import zlib

    fields = {
        name: value
        for name, value in text_config.to_dict().items()
        if name not in _NOT_MODEL_SHAPE
    }
    if quant_config and quant_config != fields.get("quantization_config"):
        fields["top_level_quantization_config"] = quant_config
    digest = zlib.crc32(json.dumps(fields, sort_keys=True, default=str).encode())
    return f"cfg{digest:08x}"


def _agreed_saved_cost_model(
    scheduler: Scheduler,
    *,
    points: list[tuple[int, int]],
    schedule: tuple[int, int, int] | None,
    slack: float,
    granularity: int,
) -> StepCostModel | None:
    """The saved fit when every rank loaded the same entry and the guard
    passes, else None. Profiling is collective, so every branch here is
    decided from gathered data that all ranks share, including whether a
    table is configured and whether the guard runs."""
    table = envs.SGLANG_TPU_PP_CHUNK_FIT_TABLE.get()
    check = envs.SGLANG_TPU_PP_CHUNK_FIT_CHECK.get()
    key = ""
    fingerprint = _fit_fingerprint(points, schedule)
    if not table:
        entry, reason = None, "SGLANG_TPU_PP_CHUNK_FIT_TABLE is unset on this rank"
    else:
        # A rank that raised here would leave the others waiting in the vote's
        # gather, so a key that cannot be built becomes this rank's vote too.
        try:
            key = _serving_fit_key(scheduler)
        except Exception as error:
            entry, reason = None, f"cannot build the fit key: {error!r}"
        else:
            entry, reason = load_saved_fit(
                table_path=table, key=key, fingerprint=fingerprint, check=check
            )
    votes = _gather(scheduler, (entry, check, bool(table)))
    if table and _is_lead():
        _log_unreadable_entries(table)
    if not any(configured for _, _, configured in votes):
        return None
    if votes[0][0] is None or any(vote != votes[0] for vote in votes):
        if entry is None and table and get_parallel().tp_rank == 0:
            # One rank per stage gives its reason; the lead says how the vote split.
            logger.warning("TPU pipeline chunks: no saved fit on this rank: %s", reason)
        if _is_lead():
            missing = [rank for rank, vote in enumerate(votes) if vote[0] is None]
            logger.warning(
                "TPU pipeline chunks: profiling at launch; %s",
                f"ranks {missing} have no usable saved fit"
                if missing
                else "the ranks did not load one saved fit with one guard setting",
            )
        return None
    agreed: SavedFit = votes[0][0]
    weak = weak_fingerprint_fields(fingerprint) if _is_lead() else []
    if weak:
        logger.warning(
            "TPU pipeline chunks: fit fingerprint is partly off (%s); %s",
            "; ".join(weak),
            "the startup guard is the only drift check for those"
            if check
            else "with SGLANG_TPU_PP_CHUNK_FIT_CHECK off, nothing checks those",
        )
    if check:
        stage_samples, _ = _profile_stages(
            scheduler, points=guard_points(points), schedule=schedule
        )
        report, drift = guard_report(
            agreed.guard_points, slowest_stage_points(stage_samples), slack=slack
        )
        if _is_lead():
            logger.info("TPU pipeline chunks: guard for %s: %s", key, "; ".join(report))
        if drift:
            if _is_lead():
                logger.warning(
                    "TPU pipeline chunks: saved fit for %s drifted (%s); profiling at launch",
                    key,
                    "; ".join(drift),
                )
            return None
    elif _is_lead():
        logger.warning(
            "TPU pipeline chunks: SGLANG_TPU_PP_CHUNK_FIT_CHECK is off; installing the "
            "saved fit for %s without re-timing it",
            key,
        )
    if _is_lead():
        logger.info("TPU pipeline chunks: using %s", reason)
    return saved_cost_model(agreed, slack=slack, granularity=granularity)


def _log_unreadable_entries(table: str) -> None:
    """Name each table entry that does not decode, once, on the lead. Each
    costs only its own key; the vote already reports a table that cannot be
    read at all."""
    try:
        _, unreadable = read_saved_fits(table)
    except Exception:
        # Logging only: the lead must not fail where the other ranks go on.
        return
    for key, error in unreadable.items():
        logger.warning(
            "TPU pipeline chunks: skipping unreadable saved fit %s in %s: %s",
            key,
            table,
            error,
        )


def _gather(scheduler: Scheduler, value: Any) -> list[Any]:
    values = [None] * scheduler.world_group.world_size
    torch.distributed.all_gather_object(
        values, value, group=scheduler.world_group.cpu_group
    )
    return values


def _is_lead() -> bool:
    return get_parallel().pp_rank == 0 and get_parallel().tp_rank == 0


def _expert_routing() -> str:
    """The simulated expert routings this launch enables, or "real"."""
    simulated = [
        env.name
        for env in (
            srt_envs.SGLANG_SIMULATE_UNIFORM_EXPERTS,
            srt_envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS,
            srt_envs.SGLANG_TPU_SIMULATE_PERFECT_BALANCED_EXPERTS,
        )
        if env.get()
    ]
    return ",".join(simulated) or "real"


def _log_fit_candidate(
    scheduler: Scheduler,
    *,
    cost: StepCostModel,
    points: list[tuple[int, int]],
    schedule: tuple[int, int, int] | None,
    stage_samples: list[list[tuple[int, int, float]]],
) -> None:
    """Log the profiled fit as a ready table entry (``key`` and ``entry``, for
    upsert_saved_fit). Its calibration records this launch's weights and expert
    routing, so the loader refuses an entry from dummy weights or simulated routing."""
    if not _is_lead():
        return
    import datetime

    import msgspec

    from sglang.srt.runtime_context import get_server_args

    guard = set(guard_points(points))
    entry = saved_fit(
        cost,
        slack=envs.SGLANG_TPU_PP_CHUNK_SLACK.get(),
        fingerprint=_fit_fingerprint(points, schedule),
        calibration=FitCalibration(
            weights="dummy" if get_server_args().load_format == "dummy" else "real",
            routing=_expert_routing(),
            date=datetime.date.today().isoformat(),
            samples=envs.SGLANG_TPU_PP_CHUNK_PROFILE_RUNS.get(),
        ),
        guard_points=[
            point
            for point in slowest_stage_points(stage_samples)
            if (point[0], point[1]) in guard
        ],
    )
    try:
        key = _serving_fit_key(scheduler)
    except Exception as error:
        # The other ranks have installed the profiled fit and go on without
        # the lead, so a key that cannot be built only skips the candidate.
        logger.warning("TPU pipeline chunks: no fit candidate: %r", error)
        return
    candidate = {"key": key, "entry": entry}
    logger.info(
        "TPU pipeline chunks: fit candidate %s", msgspec.json.encode(candidate).decode()
    )
