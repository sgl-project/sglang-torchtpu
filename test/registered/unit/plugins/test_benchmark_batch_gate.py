"""Host checks for batch admission and scheduler-clock metrics; no TPU claim."""

import sys
from enum import Enum
from types import SimpleNamespace as NS

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import benchmark_batch_gate
from sglang_tpu.benchmark_batch_gate import BenchmarkBatchGate

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Request(NS):
    @property
    def output_ids_through_stop(self):
        return self.output_ids[: self.finished_len]


def request(index, *, cohort="batch", tokens=5):
    req = Request(
        rid=f"hourly-batch:{cohort}_{index}",
        output_ids=[1],
        finished_len=None,
        sampling_params=NS(max_new_tokens=tokens),
        done=False,
        time_stats=NS(completion_time=0),
        retraction_count=0,
    )
    req.finished = lambda: req.done
    return req


def test_wait_for_full_batch_and_use_existing_completion_times(monkeypatch):
    gate = BenchmarkBatchGate(2, 10)
    a, b = request(0), request(1)
    gate.add(a, 1)
    assert not gate.ready([a])
    gate.add(b, 3)
    assert not gate.ready([a])
    assert gate.ready([b, a])
    gate.begin([b, a], 6)
    gate.begin([b, a], 7)
    for req, end in [(a, 8), (b, 10)]:
        req.done, req.output_ids = True, [1] * 5
        req.time_stats.completion_time = end
    # Reading the summary much later must not count RPC/HTTP overhead.
    monkeypatch.setattr(benchmark_batch_gate.time, "perf_counter", lambda: 1000)
    assert gate.report() == {
        "cohort": "batch",
        "batch_size": 2,
        "decode_duration": 4,
        "decode_output_tokens": 8,
        "gate_wait_duration": 5,
        "tpot_ms": [500, 1000],
    }
    gate.add(request(0, cohort="next"), 1001)
    assert not gate.ready([])
    assert gate.start is None


def test_sbd_overshoot_counts_only_emitted_tokens():
    gate = BenchmarkBatchGate(1, 10)
    req = request(0, tokens=1024)
    gate.add(req, 1)
    gate.begin([req], 2)
    req.done, req.output_ids, req.finished_len = True, [1] * 1027, 1024
    req.time_stats.completion_time = 3
    summary = gate.report()
    assert summary["decode_output_tokens"] == 1023
    assert summary["tpot_ms"] == [1000 / 1023]


def test_missing_arrival_or_kv_times_out():
    gate = BenchmarkBatchGate(2, 10)
    gate.add(request(0), 1)
    with pytest.raises(TimeoutError):
        gate.check(12)


def test_duplicate_mixed_and_unrelated_requests_fail():
    gate = BenchmarkBatchGate(2, 10)
    gate.add(request(0), 1)
    for unexpected in [request(0), request(1, cohort="other"), NS(rid="health")]:
        with pytest.raises(RuntimeError):
            gate.add(unexpected, 2)


def test_partial_or_unfinished_prefill_cannot_start_decode():
    gate = BenchmarkBatchGate(2, 10)
    a, b = request(0), request(1)
    gate.add(a, 1)
    gate.add(b, 1)
    with pytest.raises(RuntimeError, match="full prefilled"):
        gate.begin([a], 2)
    b.output_ids = []
    with pytest.raises(RuntimeError, match="full prefilled"):
        gate.begin([a, b], 2)


def test_retraction_invalidates_measurement():
    gate = BenchmarkBatchGate(2, 10)
    req = request(0)
    gate.add(req, 1)
    req.retraction_count = 1
    with pytest.raises(RuntimeError, match="retracted"):
        gate.check(2)


@pytest.mark.parametrize("invalid", ["length", "clock"])
def test_invalid_completions_are_rejected(invalid):
    gate = BenchmarkBatchGate(1, 10)
    req = request(0)
    gate.add(req, 1)
    gate.begin([req], 2)
    req.done, req.output_ids = True, [1] * 5
    req.time_stats.completion_time = 3
    if invalid == "length":
        req.finished_len = 4
    else:
        req.time_stats.completion_time = float("nan")
    with pytest.raises(RuntimeError):
        gate.report()


class Mode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def batch(reqs, mode="decode"):
    return NS(
        reqs=reqs,
        batch_is_full=False,
        forward_mode=NS(is_decode=lambda: mode == "decode"),
        is_empty=lambda: not reqs,
    )


@pytest.fixture
def scheduler_classes(monkeypatch):
    class Mixin:
        def _get_new_prebuilt_batch(self, running):
            result = batch(list(self.waiting_queue))
            self.waiting_queue.clear()
            return result

        def process_decode_queue(self):
            self.waiting_queue.append(self.transfers.pop(0))

        def get_next_disagg_decode_batch_to_run(self, running):
            prebuilt = self._get_new_prebuilt_batch(running)
            if prebuilt.reqs:
                self.output_streamer.stream_output(prebuilt.reqs, False)
                running = batch(running.reqs + prebuilt.reqs)
            if running.reqs:
                running = self.update_running_batch(running)
            return NS(
                batch_to_run=running if running.reqs else None,
                running_batch=running,
            )

    class Scheduler(Mixin):
        def _add_request_to_queue(self, req, is_retracted=False):
            (
                self.transfers
                if self.disaggregation_mode == Mode.DECODE
                else self.waiting_queue
            ).append(req)

        def get_next_batch_to_run(self, running, last):
            self.mixed_seen = self.is_mixed_chunk
            plan = self.next_plan
            if (
                plan.batch_to_run is not None
                and plan.batch_to_run.forward_mode.is_decode()
            ):
                updated = self.update_running_batch(plan.running_batch)
                return NS(
                    batch_to_run=None if not updated.reqs else updated,
                    running_batch=updated,
                )
            return plan

        def update_running_batch(self, running):
            self.decode_preparations += 1
            return running

        def run_batch(self, selected):
            self.forward_sizes.append(len(selected.reqs))

        def get_internal_state(self, recv_req):
            return NS(internal_state={})

    for name, module in {
        "sglang.srt.disaggregation.decode": NS(
            SchedulerDisaggregationDecodeMixin=Mixin
        ),
        "sglang.srt.disaggregation.utils": NS(DisaggregationMode=Mode),
        "sglang.srt.managers.schedule_batch": NS(NextBatchPlan=NS, ScheduleBatch=NS),
        "sglang.srt.managers.scheduler": NS(Scheduler=Scheduler),
        "sglang.srt.runtime_context": NS(
            get_parallel=lambda: NS(
                dp_size=1, pp_size=1, attn_dp_size=1, attn_cp_size=1
            ),
            get_disagg=lambda: NS(disaggregation_decode_host_receive_threshold=0),
        ),
        "sglang_tpu.envs": NS(
            envs=NS(
                SGLANG_TPU_BENCHMARK_BATCH_SIZE=NS(get=lambda: 2),
                SGLANG_TPU_BENCHMARK_TIMEOUT_SEC=NS(get=lambda: 10),
            )
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(benchmark_batch_gate, "_installed", False)
    now = [1.0]
    monkeypatch.setattr(benchmark_batch_gate.time, "perf_counter", lambda: now[0])
    benchmark_batch_gate.install()

    def create(mode):
        scheduler = Scheduler()
        scheduler.disaggregation_mode = mode
        scheduler.waiting_queue, scheduler.transfers, scheduler.forward_sizes = (
            [],
            [],
            [],
        )
        scheduler.enable_decode_hicache = scheduler.enable_hisparse = False
        scheduler.is_fully_idle = lambda: True
        scheduler.is_mixed_chunk = True
        scheduler.decode_preparations = 0
        scheduler.req_to_token_pool, scheduler.max_running_requests = NS(size=2), 2
        scheduler.first_tokens = {}

        def stream_output(reqs, return_logprob):
            for req in reqs:
                if req.output_ids:
                    scheduler.first_tokens.setdefault(
                        int(req.rid.rpartition("_")[2]), now[0]
                    )

        scheduler.output_streamer = NS(stream_output=stream_output)
        return scheduler

    return create, now


@pytest.mark.parametrize("defer_second_admission", [False, True])
def test_pd_emits_first_tokens_before_waiting_for_full_decode(
    scheduler_classes, defer_second_admission
):
    create, now = scheduler_classes
    scheduler = create(Mode.DECODE)
    a, b = request(0), request(1)
    scheduler._add_request_to_queue(a)
    if not defer_second_admission:
        scheduler._add_request_to_queue(b)
    now[0] = 2
    scheduler.process_decode_queue()
    held = scheduler.get_next_disagg_decode_batch_to_run(batch([]))
    assert held.batch_to_run is None
    assert held.running_batch.reqs == [a]
    assert scheduler.first_tokens.get(0) == 2
    assert scheduler.first_tokens.get(1) is None
    assert scheduler.decode_preparations == 0
    assert scheduler.forward_sizes == []
    # Another scheduler iteration preserves the held batch without re-emitting.
    now[0] = 3
    held = scheduler.get_next_disagg_decode_batch_to_run(held.running_batch)
    assert held.batch_to_run is None and held.running_batch.reqs == [a]
    assert scheduler.first_tokens.get(0) == 2
    if defer_second_admission:
        scheduler._add_request_to_queue(b)
    now[0] = 5
    scheduler.process_decode_queue()
    plan = scheduler.get_next_disagg_decode_batch_to_run(held.running_batch)
    assert plan.batch_to_run.reqs == [a, b]
    assert scheduler.first_tokens.get(1) == 5
    assert scheduler.decode_preparations == 1
    now[0] = 6
    scheduler.run_batch(plan.batch_to_run)
    for req in [a, b]:
        req.done, req.output_ids = True, [1] * 5
        req.time_stats.completion_time = 10
    now[0] = 100
    summary = scheduler.get_internal_state(None).internal_state["benchmark_batch_gate"]
    assert summary["decode_duration"] == 4
    assert summary["tpot_ms"] == [1000, 1000]
    assert [scheduler.first_tokens.get(i) for i in range(2)] == [2, 5]
    assert scheduler.forward_sizes == [2]


def test_unified_allows_chunks_but_holds_partial_decode(scheduler_classes):
    create, now = scheduler_classes
    scheduler = create(Mode.NULL)
    a, b = request(0), request(1)
    a.output_ids = b.output_ids = []
    scheduler._add_request_to_queue(a)
    scheduler.next_plan = NS(batch_to_run=batch([a], "extend"), running_batch=batch([]))
    assert scheduler.get_next_batch_to_run(batch([]), None).batch_to_run is None
    scheduler._add_request_to_queue(b)
    prefill = scheduler.get_next_batch_to_run(batch([]), None).batch_to_run
    assert prefill is not None
    assert scheduler.mixed_seen is False and scheduler.is_mixed_chunk is True
    scheduler.run_batch(prefill)
    a.output_ids = [1]
    scheduler.next_plan = NS(batch_to_run=batch([a, b]), running_batch=batch([a, b]))
    held = scheduler.get_next_batch_to_run(batch([a, b]), prefill)
    assert held.batch_to_run is None
    assert held.running_batch.reqs == [a, b]
    assert scheduler.decode_preparations == 0
    # The outstanding final prefill result produces b's first token.
    b.output_ids = [1]
    scheduler.next_plan = NS(batch_to_run=batch([a, b]), running_batch=batch([a, b]))
    decode = scheduler.get_next_batch_to_run(batch([a, b]), prefill).batch_to_run
    now[0] = 2
    scheduler.run_batch(decode)
    assert scheduler.forward_sizes == [1, 2]
    assert scheduler.decode_preparations == 1
    assert scheduler.__dict__["_benchmark_batch_gate"].start == 2


def test_capacity_fails_before_admission(scheduler_classes):
    create, _ = scheduler_classes
    scheduler = create(Mode.NULL)
    scheduler.req_to_token_pool.size = 1
    with pytest.raises(ValueError, match="capacity"):
        scheduler._add_request_to_queue(request(0))


def test_untagged_startup_bypasses_unified_gate(scheduler_classes):
    create, _ = scheduler_classes
    scheduler = create(Mode.NULL)
    scheduler._add_request_to_queue(NS(rid="warmup"))
    scheduler.next_plan = NS(batch_to_run=batch([], "extend"), running_batch=batch([]))
    assert scheduler.get_next_batch_to_run(batch([]), None) is scheduler.next_plan
    assert scheduler.mixed_seen is True
