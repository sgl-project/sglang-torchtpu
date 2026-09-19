from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace as NS

import pytest

from sglang.srt.hardware_backend.tpu.pp_chunk_scheduler import (
    TpuPipelinePrefillAdder,
    pending_step_limit,
    profile_and_install_cost_model,
)
from sglang.srt.hardware_backend.tpu.pp_chunks import StepCostModel
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.runtime_context import get_context


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
        full_untruncated_fill_ids=[0] * tokens,
        needs_host_load_back=lambda: False,
    )


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


def scheduler(waiting, chunked=None, mbs=(), last_mbs=()):
    return NS(
        _tpu_pp_chunk_cost=cost(),
        ps=NS(pp_size=8),
        waiting_queue=waiting,
        chunked_req=chunked,
        mbs=list(mbs),
        last_mbs=list(last_mbs),
    )


def batch(tokens):
    return NS(extend_num_tokens=tokens)


def test_pending_work_splits_lone_prompt_and_preserves_busy_steps():
    sched = scheduler([req(32768)])
    assert pending_step_limit(sched) == 4096
    sched.waiting_queue = [req(32768, rid=str(i)) for i in range(8)]
    assert pending_step_limit(sched) == 16384


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
    processed, flying, decode = batch(16384), batch(16384), NS(extend_num_tokens=None)
    sched = scheduler(
        [request],
        mbs=[processed, flying, None, decode],
        last_mbs=[processed, None, None, None],
    )
    assert pending_step_limit(sched) == 4096


def test_finished_request_leaves_budget_for_next_request(monkeypatch):
    def admit(self, request, *_):
        tokens = len(request.full_untruncated_fill_ids)
        assert tokens <= self.rem_chunk_tokens
        request.extend_range = NS(length=tokens)
        self.can_run_list.append(request)
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
        PrefillAdder, "add_one_req", lambda *args: AddReqResult.NO_TOKEN
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
    def admit(self, request, *_):
        request.extend_range = NS(length=self.rem_chunk_tokens)
        self.can_run_list.append(request)
        self.new_chunked_req = request
        self.rem_chunk_tokens = 0
        return AddReqResult.OTHER

    monkeypatch.setattr(PrefillAdder, "add_one_req", admit)
    obj = adder()
    obj.step_limit = 2048
    assert obj.add_one_req(req(32768), False, None) == AddReqResult.OTHER
    assert obj.rem_chunk_tokens == 14336
    assert obj.new_chunked_req is not None


def test_existing_partial_request_cannot_be_replaced(monkeypatch):
    def unexpected(*args):
        raise AssertionError("A second partial request was admitted")

    monkeypatch.setattr(PrefillAdder, "add_one_req", unexpected)
    obj = adder()
    obj.step_limit = 2048
    assert obj.add_one_req(req(32768), True, None) == AddReqResult.OTHER
    assert not obj.can_run_list
    assert obj.step_tokens == 0


@pytest.mark.parametrize("tokens", [5, 129, 4095])
@pytest.mark.parametrize("has_chunked_req", [False, True])
def test_real_admission_finishes_unaligned_prompt(monkeypatch, tokens, has_chunked_req):
    import sglang.srt.managers.schedule_policy as policy

    monkeypatch.setattr(policy, "is_dsa_prefill_cp_in_seq_split", lambda: False)
    monkeypatch.setattr(policy, "is_prefill_context_parallel_enabled", lambda: False)
    cache = NS(
        disable=False,
        supports_mamba=lambda: False,
        evictable_size=lambda: 0,
        is_tree_cache=lambda: False,
        inc_lock_ref=lambda node: None,
        dec_lock_ref=lambda node: None,
    )
    obj = TpuPipelinePrefillAdder(
        page_size=128,
        tree_cache=cache,
        token_to_kv_pool_allocator=NS(available_size=lambda: 65536),
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=16384,
        rem_chunk_tokens=16384,
        cost=cost(),
        step_limit=16384,
    )
    request = req(tokens)
    request.sampling_params = NS(ignore_eos=False, max_new_tokens=16)
    request.host_hit_length = request.storage_hit_length = 0
    request.retracted_stain = False
    request.last_node = None
    request.set_extend_range = lambda start, end: setattr(
        request, "extend_range", NS(length=end - start)
    )
    assert obj.add_one_req(request, has_chunked_req, None) == AddReqResult.CONTINUE
    assert obj.can_run_list == [request]
    assert obj.new_chunked_req is None
    assert request.extend_range.length == tokens
    assert obj.step_tokens == tokens
    assert obj.rem_chunk_tokens == 16384 - obj.ceil_paged_tokens(tokens)


def test_page_reservation_cannot_overrun_remaining_budget(monkeypatch):
    def unexpected(*args):
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
