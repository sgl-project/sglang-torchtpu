from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import (
    MAX_PREFIX_MATCHED_WAITING_REQS,
    AddReqResult,
    PrefillAdder,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import published_topology
from sglang_tpu import pp_chunk_fit, pp_chunk_scheduler, pp_profile
from sglang_tpu.pp_chunk_scheduler import (
    TpuPipelinePrefillAdder,
    pending_step_limit,
    profile_and_install_cost_model,
)
from sglang_tpu.pp_chunks import StepCostModel


@pytest.fixture(autouse=True)
def pipeline_topology():
    # The scheduler reads its stage count from the published parallel config.
    with published_topology(pp_size=8):
        yield


def cost():
    return StepCostModel(
        [2048, 4096, 8192, 16384],
        {x: 10.0 for x in [2048, 4096, 8192, 16384]},
        0.0,
        11.0,
        2048,
    )


def req(tokens, prefix=0, rid="r"):
    return NS(
        rid=rid,
        origin_input_ids=[0] * tokens,
        output_ids=[],
        prefix_indices=[0] * prefix,
        num_matched_prefix_tokens=0,
        full_untruncated_fill_ids=[0] * tokens,
        needs_host_load_back=lambda: False,
        extra_key=None,
        cache_salt=None,
        _compute_max_prefix_len=lambda length: max(length - 1, 0),
    )


class Cache:
    """Radix cache stand-in: every prompt hits ``matched`` device tokens and
    ``host`` host tokens; records each match's key limit."""

    def __init__(self, matched=0, host=0, tail=0):
        self.matched, self.host, self.tail = matched, host, tail
        self.limits = []

    def swa_reprefill_tail_tokens(self):
        return self.tail

    def match_prefix(self, params):
        self.limits.append(params.key.limit)
        return NS(device_indices=[0] * self.matched, host_hit_length=self.host)


def adder(model=None):
    obj = TpuPipelinePrefillAdder.__new__(TpuPipelinePrefillAdder)
    obj.cost = model or cost()
    obj.step_limit = 16384
    obj.step_tokens = obj.step_pairs = obj.step_squares = obj.step_schedule = 0
    obj.rem_chunk_tokens = 16384
    obj.page_size = 128
    obj.can_run_list = []
    obj.new_chunked_req = None
    obj.budget_state = lambda: (
        AddReqResult.CONTINUE if obj.rem_chunk_tokens > 0 else AddReqResult.OTHER
    )
    return obj


def scheduler(waiting, chunked=None, mbs=(), last_mbs=(), matched=0, host=0, tail=0):
    return NS(
        _tpu_pp_chunk_cost=cost(),
        waiting_queue=waiting,
        chunked_req=chunked,
        mbs=list(mbs),
        last_mbs=list(last_mbs),
        tree_cache=Cache(matched, host, tail),
        _tpu_pp_cached_prefix={},
    )


def batch(tokens, mode=ForwardMode.EXTEND, decoding_reqs=None):
    return NS(extend_num_tokens=tokens, forward_mode=mode, decoding_reqs=decoding_reqs)


def test_pending_work_splits_lone_prompt_and_preserves_busy_steps():
    sched = scheduler([req(32768)])
    assert pending_step_limit(sched) == 4096
    sched.waiting_queue = [req(32768, rid=str(i)) for i in range(8)]
    assert pending_step_limit(sched) == 16384


def test_cached_prefix_of_a_waiting_request_is_not_pending_work():
    """A waiting request's cached prefix is not pending work;
    a 64K-cached + 4K prompt still splits one chunk per stage."""
    assert pending_step_limit(scheduler([req(65536)], matched=61440)) == 2048
    assert pending_step_limit(scheduler([req(65536)], matched=0)) == 8192


def test_pending_match_counts_the_cache_as_admission_does():
    """The estimate uses admission's match: a host-cache hit is cached work,
    and the SWA re-prefill tail limits the match."""
    sched = scheduler([req(65536)], matched=32768, host=28672, tail=4096)
    assert pending_step_limit(sched) == 2048
    assert sched.tree_cache.limits == [65536 - 4096]


def test_waiting_request_matches_are_capped_per_step():
    """Short questions behind a long cached prefix add little pending work
    each, so matching every queued request would walk the prefix hundreds of
    times per step; past the cap they count at full length."""
    waiting = [req(65536 + 256, rid=str(i)) for i in range(600)]
    sched = scheduler(waiting, matched=65536)
    assert pending_step_limit(sched) == 16384
    assert len(sched.tree_cache.limits) == MAX_PREFIX_MATCHED_WAITING_REQS


def test_a_waiting_request_is_matched_once():
    """A match walks the whole cached prefix, so a waiting request keeps its
    first match on later steps; the entry goes when it leaves the queue."""
    sched = scheduler([req(65536 + 256)], matched=65536)
    assert pending_step_limit(sched) == 2048
    assert pending_step_limit(sched) == 2048
    assert len(sched.tree_cache.limits) == 1
    sched.waiting_queue = []
    pending_step_limit(sched)
    assert sched._tpu_pp_cached_prefix == {}


def test_policy_match_counts_host_hits_without_rematching():
    """A request the policy matched counts its host hits as cached without a
    second match; a partial request counts the chunks computed past that match."""
    waiting = req(65536, prefix=32768)
    waiting.num_matched_prefix_tokens = 61440
    sched = scheduler([waiting])
    assert pending_step_limit(sched) == 2048
    assert sched.tree_cache.limits == []
    partial = req(32768, prefix=24576)
    partial.num_matched_prefix_tokens = 4096
    assert pending_step_limit(scheduler([], chunked=partial)) == 2048


def test_pending_continuation_counts_only_unscheduled_tokens_once():
    request = req(32768, 24576)
    assert pending_step_limit(scheduler([request], chunked=request)) == 2048


def test_chunks_in_flight_grow_the_next_chunk():
    request = req(16384)
    assert pending_step_limit(scheduler([request])) == 2048
    flying = [batch(4096) for _ in range(7)]
    sched = scheduler([request], mbs=flying + [None], last_mbs=[None] * 8)
    assert pending_step_limit(sched) == 6144
    sched = scheduler([request], mbs=[batch(16384)] * 7 + [None], last_mbs=[None] * 8)
    assert pending_step_limit(sched) == 16384


def test_processed_and_empty_slots_are_not_in_flight():
    request = req(16384)
    processed, flying = batch(16384), batch(16384)
    decode = batch(None, mode=ForwardMode.DECODE)
    sched = scheduler(
        [request],
        mbs=[processed, flying, None, decode],
        last_mbs=[processed, None, None, None],
    )
    assert pending_step_limit(sched) == 4096


def test_decode_rows_are_not_pending_prefill():
    """A decode batch keeps the token count of the prefill it was built from,
    and a mixed batch counts its decode rows; neither is prefill still in
    flight, so it must not grow the next chunk."""
    request = req(16384)
    stale = batch(16384, mode=ForwardMode.DECODE)
    mixed = batch(4096 + 64, mode=ForwardMode.MIXED, decoding_reqs=[None] * 64)
    sched = scheduler([request], mbs=[stale] * 7 + [mixed], last_mbs=[None] * 8)
    assert pending_step_limit(sched) == 4096


def test_finished_request_leaves_budget_for_next_request(monkeypatch):
    def admit(self, *, req, **_):
        tokens = len(req.full_untruncated_fill_ids)
        assert tokens <= self.rem_chunk_tokens
        req.extend_range = NS(length=tokens)
        self.can_run_list.append(req)
        self.rem_chunk_tokens -= tokens
        return self.budget_state()

    monkeypatch.setattr(PrefillAdder, "add_one_req", admit)
    obj = adder()
    assert obj.add_one_req(req(4096), False, None) == AddReqResult.CONTINUE
    assert obj.rem_chunk_tokens == 12288
    assert obj.add_one_req(req(4096, rid="r2"), False, None) == AddReqResult.CONTINUE
    assert obj.step_tokens == 8192
    assert obj.rem_chunk_tokens == 8192


def test_rejected_request_does_not_charge_time_or_tokens(monkeypatch):
    monkeypatch.setattr(
        PrefillAdder, "add_one_req", lambda *args, **kwargs: AddReqResult.NO_TOKEN
    )
    obj = adder()
    assert obj.add_one_req(req(4096), False, None) == AddReqResult.NO_TOKEN
    assert obj.step_tokens == 0
    assert obj.rem_chunk_tokens == 16384


def test_minimum_progress_never_bypasses_schedule_capacity():
    model = cost()
    model.schedule = (1, 256, 256)
    obj = adder(model)
    with pytest.raises(ValueError, match="minimum pipeline chunk"):
        obj.add_chunked_req(req(32768, 16384))


def test_host_prefix_is_included_before_chunk_is_sized():
    obj = adder()
    request = req(32768)
    request.host_hit_length = 16384
    request.needs_host_load_back = lambda: True
    seen = []
    obj.cost.chunk = lambda *args: seen.append(args[:2]) or 2048
    with obj._request_budget(request):
        pass
    assert seen == [(16384, 16384)]


def test_new_partial_request_stops_admission(monkeypatch):
    def admit(self, *, req, **_):
        req.extend_range = NS(length=self.rem_chunk_tokens)
        self.can_run_list.append(req)
        self.new_chunked_req = req
        self.rem_chunk_tokens = 0
        return AddReqResult.OTHER

    monkeypatch.setattr(PrefillAdder, "add_one_req", admit)
    obj = adder()
    obj.step_limit = 2048
    assert obj.add_one_req(req(32768), False, None) == AddReqResult.OTHER
    assert obj.rem_chunk_tokens == 14336
    assert obj.new_chunked_req is not None


def test_existing_partial_request_cannot_be_replaced(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("A second partial request was admitted")

    monkeypatch.setattr(PrefillAdder, "add_one_req", unexpected)
    obj = adder()
    obj.step_limit = 2048
    assert obj.add_one_req(req(32768), True, None) == AddReqResult.OTHER
    assert not obj.can_run_list
    assert obj.step_tokens == 0


@pytest.mark.parametrize("tokens", [5, 129, 4095])
@pytest.mark.parametrize("has_chunked_req", [False, True])
def test_real_admission_finishes_unaligned_prompt(tokens, has_chunked_req):
    cache = MagicMock()
    cache.disable = False
    cache.supports_mamba.return_value = False
    cache.supports_prefix_sharing.return_value = False
    cache.evictable_size.return_value = 0
    cache.full_evictable_size.return_value = 0
    cache.inc_lock_ref.return_value = IncLockRefResult()
    cache.dec_lock_ref.return_value = DecLockRefResult()
    cache.buffer_pipeline = None
    allocator = MagicMock()
    allocator.page_size = 128
    allocator.available_size.return_value = 65536
    allocator.full_available_size.return_value = 65536
    allocator.swa_req_ring = False
    allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
        PrefillBudget(allocator, tree_cache, **kwargs)
    )
    obj = TpuPipelinePrefillAdder(
        page_size=128,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=16384,
        rem_chunk_tokens=16384,
        cost=cost(),
        step_limit=16384,
    )
    request = MagicMock(spec=Req)
    request.kv = ReqKvInfo()
    request.rid = "r"
    request.cache_request_handle = CacheRequestHandle(request.rid, 0)
    request.origin_input_ids = [0] * tokens
    request.output_ids = []
    request.prefix_indices = []
    request.full_untruncated_fill_ids = [0] * tokens
    request.sampling_params = NS(ignore_eos=False, max_new_tokens=16)
    request.retracted_stain = False
    request.host_hit_length = request.swa_host_hit_length = 0
    request.storage_hit_length = request.host_loaded_length = 0
    request.materialized_host_hit_len.return_value = 0
    request.fulfilled_storage_hit_len.return_value = 0
    request.needs_host_load_back.return_value = False
    request.last_node = None
    request.set_extend_range.side_effect = lambda start, end: setattr(
        request, "extend_range", NS(length=end - start)
    )
    # Called the way the scheduler calls every prefill adder.
    assert (
        obj.add_one_req(
            request,
            has_chunked_req=has_chunked_req,
            truncation_align_size=None,
            chunk_tokens_cap=None,
        )
        == AddReqResult.CONTINUE
    )
    assert obj.can_run_list == [request]
    assert obj.new_chunked_req is None
    assert request.extend_range.length == tokens
    assert obj.step_tokens == tokens
    assert obj.rem_chunk_tokens == 16384 - obj.ceil_paged_tokens(tokens)


def test_page_reservation_cannot_overrun_remaining_budget(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("A second partial request was admitted")

    monkeypatch.setattr(PrefillAdder, "add_one_req", unexpected)
    obj = adder()
    obj.rem_chunk_tokens = 129
    assert obj.add_one_req(req(129), True, None) == AddReqResult.OTHER
    assert obj.rem_chunk_tokens == 129
    assert obj.step_tokens == 0


def test_continuation_does_not_count_loaded_host_prefix_twice(monkeypatch):
    monkeypatch.setattr(PrefillAdder, "add_chunked_req", lambda self, request: request)
    obj = adder()
    request = req(32768, prefix=18432)
    request.host_hit_length = 16384
    request.needs_host_load_back = lambda: True
    seen = []
    obj.cost.chunk = lambda *args: seen.append(args[:2]) or 2048
    assert obj.add_chunked_req(request) is request
    assert seen == [(18432, 14336)]
    assert obj.rem_chunk_tokens == 16384


def test_profile_requires_resolved_compiler_setting(monkeypatch):
    monkeypatch.setenv("SGLANG_TPU_PP_DYNAMIC_CHUNKS", "1")
    monkeypatch.setenv("SGLANG_TPU_PP_CHUNK_SLACK", "-1")
    monkeypatch.setattr(pp_chunk_scheduler, "_gather", lambda scheduler, value: [value])
    with get_context().override_server_args(
        device="tpu",
        pp_size=8,
        disaggregation_mode="null",
        chunked_prefill_size=16384,
        enable_torch_compile=True,
    ) as raw:
        scheduler = NS(tp_worker=NS(model_runner=NS(server_args=raw)))
        assert not raw.enable_torch_compile
        with pytest.raises(ValueError, match="SLACK must be nonnegative"):
            profile_and_install_cost_model(scheduler)
        get_context().override("test.no_compile", enable_torch_compile=False)
        with pytest.raises(ValueError, match="require --enable-torch-compile"):
            profile_and_install_cost_model(scheduler)


def _profile_ms(rank, tokens, prefix):
    # Ranks 0, 1 form stage 0 and ranks 2, 3 stage 1; odd ranks time slower.
    return (1 + rank % 2) * (tokens / 100 + tokens * prefix * 1e-6)


class _RankView:
    """The parallel context as one rank of the simulated world sees it."""

    def __init__(self, base, *, pp_rank, tp_rank):
        self._base = base
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank

    def __getattr__(self, name):
        return getattr(self._base, name)


def _install_profiled_models(
    monkeypatch, *, ms, votes=None, guard_reports=None, rank_env=None, settings=None
):
    """Run profile_and_install_cost_model once per rank of a PP2 x TP2 world
    whose collectives return every rank's precomputed report."""
    points = [(2048, 0), (2048, 2048), (4096, 0), (4096, 4096)]
    reports = [
        (rank // 2, [(t, p, ms(rank, t, p)) for t, p in points], None)
        for rank in range(4)
    ]
    calls = {"gather": 0, "barrier": 0, "ops": {}}
    current = {"rank": 0}

    def all_gather_object(out, obj, group):
        calls["ops"].setdefault(current["rank"], []).append("gather")
        if obj[0] == "pp-chunk-settings":
            out[:] = settings if settings is not None else [obj] * len(group)
            return
        if obj[0] is None or not isinstance(obj[0], int):
            # Every rank's (saved-fit entry or None, guard setting, table set).
            out[:] = votes if votes is not None else [obj] * len(group)
            return
        calls["gather"] += 1
        timed_points = {(t, p) for t, p, _ in obj[1]}
        source = guard_reports if guard_reports is not None else reports
        out[:] = [
            (stage, [s for s in samples if (s[0], s[1]) in timed_points], schedule)
            for stage, samples, schedule in source
        ]

    def barrier(group):
        calls["ops"].setdefault(current["rank"], []).append("barrier")
        calls["barrier"] += 1

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(pp_chunk_scheduler, "uses_dynamic_chunks", lambda: True)
    monkeypatch.setattr(
        pp_chunk_scheduler, "get_exec", lambda: NS(graph=NS(enable_torch_compile=True))
    )
    monkeypatch.setattr(
        pp_chunk_scheduler, "attention_schedule_capacity", lambda runner, policy: None
    )
    monkeypatch.setattr(pp_chunk_scheduler, "profile_points", lambda *args: points)
    monkeypatch.setattr(
        pp_profile, "profile_cache", lambda policy, max_tokens: contextlib.nullcontext()
    )
    parallel = pp_chunk_scheduler.get_parallel()
    models = []
    for rank in range(4):
        current["rank"] = rank
        view = _RankView(parallel, pp_rank=rank // 2, tp_rank=rank % 2)
        monkeypatch.setattr(pp_chunk_scheduler, "get_parallel", lambda view=view: view)
        for name, value in (rank_env or {}).get(rank, {}).items():
            monkeypatch.setenv(name, value)
        policy = NS(
            t_bucketer=NS(buckets=[2048, 4096]),
            _run_one=lambda rank=rank, **kw: ms(
                rank, kw["num_tokens"], kw["profile_prefix"]
            ),
        )
        runner = NS(compile_policy=policy, model_config=NS(context_len=32768))
        scheduler = NS(
            tp_worker=NS(model_runner=runner),
            page_size=256,
            chunked_prefill_size=4096,
            pp_group=NS(cpu_group=(rank % 2, rank % 2 + 2)),
            world_group=NS(world_size=4, cpu_group=(0, 1, 2, 3)),
        )
        profile_and_install_cost_model(scheduler)
        models.append(scheduler._tpu_pp_chunk_cost)
    return models, calls


def test_devices_of_a_stage_fit_one_cost_model(monkeypatch):
    """The devices of a multi-device stage must size chunks alike even when
    they time the profile differently; diverging sizes deadlock the stage."""
    models, calls = _install_profiled_models(monkeypatch, ms=_profile_ms)
    assert all(model == models[0] for model in models)
    assert calls["barrier"] == 4


def test_a_delayed_device_does_not_inflate_its_stage(monkeypatch):
    """A device of a stage timed slow on the host must not set the stage's
    cost: the devices of a stage run each step together."""
    models, _ = _install_profiled_models(monkeypatch, ms=_profile_ms)
    even = lambda rank, t, p: _profile_ms(rank - rank % 2, t, p)
    reference, _ = _install_profiled_models(monkeypatch, ms=even)
    assert models[0].linear_ms == reference[0].linear_ms
    assert models[0].target_ms == reference[0].target_ms


POINTS = [(2048, 0), (2048, 2048), (4096, 0), (4096, 4096)]


def _saved_entry(model, guard_points):
    from sglang_tpu.pp_chunk_fit import FitCalibration, saved_fit

    return saved_fit(
        model,
        slack=0.1,
        fingerprint=pp_chunk_scheduler._fit_fingerprint(POINTS, None),
        calibration=FitCalibration(weights="real", routing="real"),
        guard_points=guard_points,
    )


def _write_saved_fit(monkeypatch, tmp_path, entry, *, check):
    from sglang_tpu.pp_chunk_fit import upsert_saved_fit

    path = str(tmp_path / "fits.json")
    upsert_saved_fit(path, "shape", entry)
    monkeypatch.setenv("SGLANG_TPU_PP_CHUNK_FIT_TABLE", path)
    monkeypatch.setenv("SGLANG_TPU_PP_CHUNK_FIT_CHECK", "1" if check else "0")
    monkeypatch.setattr(
        pp_chunk_scheduler, "_serving_fit_key", lambda scheduler: "shape"
    )


def _profiled_reference(monkeypatch):
    models, _ = _install_profiled_models(monkeypatch, ms=_profile_ms)
    return models[0]


def _skewed_ms(rank, tokens, prefix):
    # Stage 1 (ranks 2, 3) is 1.5x slower than stage 0, so the slowest stage is stage 1.
    return _profile_ms(rank, tokens, prefix) * (1.5 if rank >= 2 else 1.0)


def _slowest_stage_ms(tokens, prefix):
    # Stage 1's fastest device is rank 2.
    return _skewed_ms(2, tokens, prefix)


SAVED = StepCostModel([2048, 4096], {2048: 30.0, 4096: 55.0}, 1e-6, 0.0, 512)


def test_matching_saved_fit_replaces_the_profile(monkeypatch, tmp_path):
    _write_saved_fit(monkeypatch, tmp_path, _saved_entry(SAVED, []), check=False)
    models, calls = _install_profiled_models(monkeypatch, ms=_profile_ms)
    assert calls["gather"] == 0
    assert all(model.linear_ms == SAVED.linear_ms for model in models)


@pytest.mark.parametrize("other", ["missing", "different", "check"])
def test_ranks_without_one_saved_fit_all_profile(monkeypatch, tmp_path, other):
    """Profiling is collective: a rank that lacks the saved fit, or loaded a
    different one, must send every rank to the profile, or the stage hangs."""
    reference = _profiled_reference(monkeypatch)
    entry = _saved_entry(SAVED, [])
    _write_saved_fit(monkeypatch, tmp_path, entry, check=False)
    odd = {
        "missing": (None, False, True),
        "different": (_saved_entry(reference, []), False, True),
        # Same entry, but this rank would run the guard and the others would not.
        "check": (entry, True, True),
    }[other]
    votes = [(entry, False, True), odd] + [(entry, False, True)] * 2
    models, calls = _install_profiled_models(monkeypatch, ms=_profile_ms, votes=votes)
    assert calls["gather"] == 4
    assert all(model == reference for model in models)


@pytest.mark.parametrize(
    "drift, uses_saved",
    [
        ((1.05, 1.05, 1.05), True),
        ((1.5, 1.5, 1.5), False),
        ((1.5, 1.0, 1.0), False),
        ((1.0, 1.5, 1.0), False),
        # Attention drift alone: both prefix-free points still match.
        ((1.0, 1.0, 1.5), False),
    ],
)
def test_startup_guard_compares_against_calibrated_times(
    monkeypatch, tmp_path, drift, uses_saved
):
    """The guard re-times against the calibrated slowest-stage times, not the
    fitted curve, which is an envelope over stages and can sit above them all."""
    reference, _ = _install_profiled_models(monkeypatch, ms=_skewed_ms)
    reference = reference[0]
    # Drift at either end, or behind the longest prefix, alone rejects the fit.
    guard = [
        (t, p, _slowest_stage_ms(t, p) * d)
        for (t, p), d in zip([(2048, 0), (4096, 0), (4096, 4096)], drift)
    ]
    # An envelope well above every stage would fail a curve-based guard.
    envelope = StepCostModel(
        reference.buckets,
        {b: ms * 3 for b, ms in reference.linear_ms.items()},
        reference.attn_ms_per_token_pair * 3,
        0.0,
        reference.granularity,
    )
    _write_saved_fit(monkeypatch, tmp_path, _saved_entry(envelope, guard), check=True)
    models, calls = _install_profiled_models(monkeypatch, ms=_skewed_ms)
    # The guard re-times three points on every rank; a rejection adds the full profile.
    assert calls["gather"] == (4 if uses_saved else 8)
    expected = envelope.linear_ms if uses_saved else reference.linear_ms
    assert all(model.linear_ms == expected for model in models)


def test_an_entry_without_every_guard_point_profiles_without_the_guard(
    monkeypatch, tmp_path
):
    """The guard could only reject an entry that lacks a calibrated time at one
    of its points, so the entry is refused before the guard re-times anything."""
    reference = _profiled_reference(monkeypatch)
    _, profile_only = _install_profiled_models(monkeypatch, ms=_profile_ms)
    partial = [(4096, 0, _profile_ms(0, 4096, 0))]
    _write_saved_fit(monkeypatch, tmp_path, _saved_entry(SAVED, partial), check=True)
    models, calls = _install_profiled_models(monkeypatch, ms=_profile_ms)
    # The settings and the vote, then the profile; no guard barrier.
    assert calls["barrier"] == profile_only["barrier"]
    assert all(model == reference for model in models)


def test_an_unreadable_entry_is_named_once_and_the_good_one_installs(
    monkeypatch, tmp_path, caplog
):
    import json
    import logging

    _write_saved_fit(monkeypatch, tmp_path, _saved_entry(SAVED, []), check=False)
    path = tmp_path / "fits.json"
    table = json.loads(path.read_text())
    table["broken"] = {"linear_ms": "not a ladder"}
    path.write_text(json.dumps(table))
    with caplog.at_level(logging.INFO, logger=pp_chunk_scheduler.logger.name):
        models, _ = _install_profiled_models(monkeypatch, ms=_profile_ms)
    assert all(model.linear_ms == SAVED.linear_ms for model in models)
    skipped = [r for r in caplog.records if "skipping unreadable" in r.getMessage()]
    assert len(skipped) == 1
    assert "broken" in skipped[0].getMessage()


def test_a_rank_without_the_table_setting_takes_the_same_collectives(
    monkeypatch, tmp_path
):
    """A rank that has no table configured must still join the saved-fit vote,
    or it enters the profile barrier while the others wait in the vote."""
    entry = _saved_entry(SAVED, [])
    _write_saved_fit(monkeypatch, tmp_path, entry, check=False)
    table = {"SGLANG_TPU_PP_CHUNK_FIT_TABLE": str(tmp_path / "fits.json")}
    rank_env = {0: table, 1: {"SGLANG_TPU_PP_CHUNK_FIT_TABLE": ""}, 2: table, 3: table}
    # What the vote gathers from these four ranks.
    votes = [(entry, False, True), (None, False, False)] + [(entry, False, True)] * 2
    _, calls = _install_profiled_models(
        monkeypatch, ms=_profile_ms, votes=votes, rank_env=rank_env
    )
    sequences = [calls["ops"][rank] for rank in range(4)]
    assert all(sequence == sequences[0] for sequence in sequences)


def test_a_key_that_cannot_be_built_is_a_vote(monkeypatch, tmp_path):
    """Building the key runs only on ranks with a table set; raising there
    would leave the other ranks waiting in the vote's gather."""

    def broken_key(scheduler):
        raise TypeError("config field is not serializable")

    sent = []
    monkeypatch.setenv("SGLANG_TPU_PP_CHUNK_FIT_TABLE", str(tmp_path / "fits.json"))
    monkeypatch.setattr(pp_chunk_scheduler, "_serving_fit_key", broken_key)
    monkeypatch.setattr(
        pp_chunk_scheduler,
        "_gather",
        lambda scheduler, value: sent.append(value) or [value],
    )
    monkeypatch.setattr(
        pp_chunk_scheduler, "get_parallel", lambda: NS(pp_rank=0, tp_rank=0)
    )
    model = pp_chunk_scheduler._agreed_saved_cost_model(
        None, points=POINTS, schedule=None, slack=0.1, granularity=512
    )
    assert model is None
    assert sent == [(None, True, True)]


@pytest.mark.parametrize(
    "runs, expected",
    [
        ([5, 5, 5, 3], "must match on every rank"),
        ([5, 5, 5, 0], "must match on every rank"),
        ([0, 0, 0, 0], "must be at least 1"),
    ],
)
def test_ranks_must_agree_on_profile_runs_and_slack(monkeypatch, runs, expected):
    """A rank that times more runs per point enters collectives its peers
    never reach, so mismatched or invalid settings fail at launch on every
    rank, after the gather rather than on one rank before it."""
    settings = [("pp-chunk-settings", count, 0.1) for count in runs]
    monkeypatch.setenv("SGLANG_TPU_PP_CHUNK_PROFILE_RUNS", str(runs[0]))
    with pytest.raises(ValueError, match=expected):
        _install_profiled_models(monkeypatch, ms=_profile_ms, settings=settings)


def test_model_shape_tells_apart_models_of_one_depth_and_width():
    """Any shape field splits the key, including model-specific ones; names and
    token ids do not."""
    from transformers import PretrainedConfig

    def shape(**fields):
        return pp_chunk_scheduler._model_shape(PretrainedConfig(**fields), {})

    base = dict(num_experts=128, num_experts_per_tok=4, moe_hidden_size=2048)
    assert shape(**base) != shape(**{**base, "num_experts": 64})
    assert shape(**base) != shape(**{**base, "moe_hidden_size": 1024})
    assert shape(**base) != shape(**base, layer_types=["sliding_attention"])
    assert shape(**base) == shape(**base, name_or_path="/other/copy", eos_token_id=7)


def test_a_candidate_without_a_key_does_not_stop_the_lead(
    monkeypatch, tmp_path, caplog
):
    """The other ranks have installed the profiled fit by then, so the lead
    skips the candidate instead of raising."""
    import logging

    def broken_key(scheduler):
        raise TypeError("config field is not serializable")

    monkeypatch.setenv("SGLANG_TPU_PP_CHUNK_FIT_TABLE", str(tmp_path / "unused.json"))
    monkeypatch.setattr(pp_chunk_scheduler, "_serving_fit_key", broken_key)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_server_args", lambda: NS(load_format="auto")
    )
    reference = _profiled_reference(monkeypatch)
    with caplog.at_level(logging.INFO, logger=pp_chunk_scheduler.logger.name):
        models, _ = _install_profiled_models(monkeypatch, ms=_profile_ms)
    messages = [r.getMessage() for r in caplog.records]
    assert all(model == reference for model in models)
    assert not any("chunks: fit candidate" in m for m in messages)
    assert any("no fit candidate" in m for m in messages)


def test_serving_fit_key_splits_on_shape_and_not_on_names(monkeypatch):
    """The real key builder: a cost-relevant config field splits the key;
    checkpoint names and token ids do not."""
    import enum

    from transformers import PretrainedConfig

    import sglang.srt.layers.moe as moe
    import sglang_tpu.pallas_export as pallas_export

    class Backend(enum.Enum):
        AUTO = "auto"
        NONE = "none"

    monkeypatch.setattr(moe, "get_moe_runner_backend", lambda: Backend.AUTO)
    monkeypatch.setattr(moe, "get_moe_a2a_backend", lambda: Backend.NONE)
    monkeypatch.setattr(pallas_export, "get_torch_tpu_device_kind", lambda: "TPU7x")
    layout = dict(
        pp_size=8,
        tp_size=2,
        attn_cp_size=1,
        attn_dcp_size=1,
        moe_ep_size=2,
        moe_tp_size=1,
        moe_dense_tp_size=None,
        attn_dp_size=1,
        extend_cap=8,
    )
    monkeypatch.setattr(pp_chunk_scheduler, "get_parallel", lambda: NS(**layout))
    monkeypatch.delenv("SGLANG_PP_LAYER_PARTITION", raising=False)

    class RpaBackend:
        pass

    def key(quant_config=None, **fields):
        config = PretrainedConfig(model_type="m", **fields)
        model_config = NS(
            hf_config=config,
            hf_text_config=config,
            hf_quant_config=quant_config or {},
            num_hidden_layers=48,
            hidden_size=5120,
            dtype=torch.bfloat16,
            quantization=None,
        )
        runner = NS(
            model_config=model_config,
            kv_cache_dtype=torch.bfloat16,
            attn_backend=RpaBackend(),
            compile_policy=NS(_extend_compile_cap=lambda: layout["extend_cap"]),
        )
        scheduler = NS(
            tp_worker=NS(model_runner=runner), chunked_prefill_size=8192, page_size=512
        )
        return pp_chunk_scheduler._serving_fit_key(scheduler)

    base = dict(num_local_experts=128, num_experts_per_tok=4)
    assert key(**base).startswith("m/L48xH5120/cfg")
    assert key(**base).endswith(
        "/bfloat16/none/kvbfloat16/RpaBackend/auto/a2anone/TPU7x/"
        "pp8xtp2xcp1xdcp1xep2xmoetp1xdensetp2xdp1/chunk8192/extendbs8/page512/spliteven"
    )
    assert key(**base) != key(**{**base, "num_local_experts": 64})
    assert key(**base) == key(**base, name_or_path="/copy", eos_token_id=7)
    # Quantization settings that a wrapper keeps on its top-level config.
    fp8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    assert key(fp8, **base) != key(**base)
    assert key(fp8, **base) != key({**fp8, "weight_block_size": [64, 64]}, **base)
    # Dense layers split over fewer ranks than TP.
    plain = key(**base)
    layout["moe_dense_tp_size"] = 1
    assert key(**base) != plain
    assert "xdensetp1x" in key(**base)
    # More requests per compiled prefill batch (--prefill-max-requests).
    plain = key(**base)
    layout["extend_cap"] = 64
    assert key(**base) != plain
    assert "/extendbs64/" in key(**base)


@pytest.mark.parametrize("load_format", ["auto", "dummy"])
def test_fit_candidate_is_a_table_entry(monkeypatch, tmp_path, caplog, load_format):
    """The logged candidate saves as-is, and its calibration comes from the
    launch, so an entry from dummy weights is refused by the loader."""
    import json
    import logging

    import msgspec

    from sglang_tpu.pp_chunk_fit import SavedFit, upsert_saved_fit

    monkeypatch.setenv("SGLANG_TPU_PP_CHUNK_FIT_TABLE", str(tmp_path / "unused.json"))
    monkeypatch.setattr(
        pp_chunk_scheduler, "_serving_fit_key", lambda scheduler: "shape"
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_server_args",
        lambda: NS(load_format=load_format),
    )
    with caplog.at_level(logging.INFO, logger=pp_chunk_scheduler.logger.name):
        _install_profiled_models(monkeypatch, ms=_profile_ms)
    line = next(
        r.getMessage() for r in caplog.records if "fit candidate" in r.getMessage()
    )
    candidate = json.loads(line.split("fit candidate ", 1)[1])
    entry = msgspec.json.decode(json.dumps(candidate["entry"]), type=SavedFit)
    path = str(tmp_path / "fits.json")
    upsert_saved_fit(path, candidate["key"], entry)
    loaded, reason = pp_chunk_fit.load_saved_fit(
        table_path=path,
        key="shape",
        fingerprint=pp_chunk_scheduler._fit_fingerprint(POINTS, None),
        check=True,
    )
    assert (loaded is not None) == (load_format == "auto"), reason


@pytest.mark.parametrize("simulated", [None, "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS"])
def test_fit_candidate_routing_is_what_the_table_accepts(monkeypatch, simulated):
    """The candidate records the launch's routing in its entry, so only an
    unsimulated launch may produce the routing the loader accepts."""
    for name in (
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
        "SGLANG_TPU_SIMULATE_PERFECT_BALANCED_EXPERTS",
    ):
        monkeypatch.delenv(name, raising=False)
    if simulated:
        monkeypatch.setenv(simulated, "1")
    accepted = (
        pp_chunk_scheduler._expert_routing() == pp_chunk_fit._REQUIRED_CALIBRATION
    )
    assert accepted == (simulated is None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
