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
from sglang_tpu.managers import pp_chunk_scheduler, pp_profile
from sglang_tpu.managers.pp_chunk_scheduler import (
    TpuPipelinePrefillAdder,
    pending_step_limit,
    profile_and_install_cost_model,
)
from sglang_tpu.managers.pp_chunks import StepCostModel


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


def test_devices_of_a_stage_fit_one_cost_model(monkeypatch):
    """The devices of a multi-device stage must size chunks alike even when
    they time the profile differently; diverging sizes deadlock the stage."""
    points = [(2048, 0), (2048, 2048), (4096, 0), (4096, 4096)]

    def ms(rank, tokens, prefix):
        # Ranks 0, 1 form stage 0 and ranks 2, 3 stage 1; odd ranks time slower.
        return (1 + rank % 2) * (tokens / 100 + tokens * prefix * 1e-6)

    reports = {
        rank: ([(t, p, ms(rank, t, p)) for t, p in points], None) for rank in range(4)
    }

    def all_gather_object(out, obj, group):
        out[:] = [reports[rank] for rank in group]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
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
    models = []
    for rank in range(4):
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
    assert all(model == models[0] for model in models)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
