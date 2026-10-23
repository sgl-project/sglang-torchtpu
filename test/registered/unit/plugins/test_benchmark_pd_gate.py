"""Host tests for the benchmark cohort gate (no TPU runtime required)."""

import sys
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import benchmark_pd_gate
from sglang_tpu.benchmark_pd_gate import BenchmarkPDGate

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def request(index, *, cohort="batch", tokens=5):
    req = SimpleNamespace(
        rid=f"hourly-pd-sync:{cohort}_{index}",
        output_ids=[1],
        sampling_params=SimpleNamespace(max_new_tokens=tokens),
        done=False,
    )
    req.finished = lambda: req.done
    return req


def test_hold_until_every_arrival_and_transfer_then_measure_only_decode():
    gate = BenchmarkPDGate(2, 10)
    a, b = request(0), request(1)
    gate.admit(a, 1)
    assert not gate.ready([a], 0, 2, 2)
    gate.admit(b, 3)
    assert not gate.ready([a], 0, 2, 4)
    assert gate.ready([b, a], 0, 2, 5)
    gate.release([b, a])
    gate.begin_decode(6)
    gate.begin_decode(7)  # Subsequent forwards keep the original start.
    a.output_ids = [1] * 5
    a.done = True
    gate.observe_completion(8)
    assert gate.last_result is None
    b.output_ids = [1] * 5
    b.done = True
    gate.observe_completion(10)
    assert gate.last_result == {
        "cohort": "batch",
        "batch_size": 2,
        "decode_duration": 4,
        "decode_output_tokens": 8,
        "gate_wait_duration": 5,
        "tpot_ms": [500, 1000],
    }
    # A later shape uses a fresh cohort and cannot reuse the previous clock.
    gate.admit(request(0, cohort="next"), 11)
    assert not gate.ready([], 0, 2, 12)


@pytest.mark.parametrize("ready", [[], [request(0)]])
def test_missing_arrival_or_transfer_times_out(ready):
    gate = BenchmarkPDGate(2, 10)
    gate.admit(request(0), 1)
    with pytest.raises(TimeoutError, match="timed out"):
        gate.ready(ready, 0, 2, 12)


@pytest.mark.parametrize("running,capacity", [(0, 1), (1, 2)])
def test_cohort_must_fit_without_other_running_requests(running, capacity):
    gate = BenchmarkPDGate(2, 10)
    gate.admit(request(0), 1)
    with pytest.raises(RuntimeError, match="cannot fit"):
        gate.ready([], running, capacity, 2)


def test_mixed_duplicate_and_unrelated_requests_fail():
    gate = BenchmarkPDGate(2, 10)
    gate.admit(request(0), 1)
    for unexpected in [
        request(0),
        request(1, cohort="other"),
        SimpleNamespace(rid="health"),
    ]:
        with pytest.raises(RuntimeError):
            gate.admit(unexpected, 2)
    with pytest.raises(RuntimeError, match="Unexpected request"):
        gate.ready([request(1, cohort="other")], 0, 2, 2)


def test_partial_admission_or_early_finish_invalidates_measurement():
    gate = BenchmarkPDGate(2, 10)
    a, b = request(0), request(1)
    gate.admit(a, 1)
    gate.admit(b, 1)
    with pytest.raises(RuntimeError, match="split"):
        gate.release([a])
    gate.release([a, b])
    gate.begin_decode(2)
    a.done = True
    with pytest.raises(RuntimeError, match="finish at length"):
        gate.observe_completion(3)


def test_untagged_requests_bypass_gate():
    gate = BenchmarkPDGate(64, 10)
    gate.admit(SimpleNamespace(rid="startup-warmup"), 1)
    assert gate.ready([], 1, 1, 100)
    assert gate.cohort is None


def test_installed_wrappers_keep_transfer_polling_and_release_one_batch(monkeypatch):
    class Mixin:
        def process_decode_queue(self):
            self.waiting_queue.append(self.transfers.pop(0))

        def _get_new_prebuilt_batch(self, running):
            batch = SimpleNamespace(reqs=list(self.waiting_queue))
            self.waiting_queue.clear()
            return batch

    class Scheduler(Mixin):
        def run_batch(self, batch):
            assert len(batch.reqs) == 2
            return "forward-result"

        def process_batch_result(self, batch, result):
            for req in batch.reqs:
                req.output_ids = [1] * 5
                req.done = True

        def get_internal_state(self, recv_req):
            return SimpleNamespace(internal_state={})

    class Queue:
        def add(self, req, is_retracted=False, is_rebootstrap=False):
            self.scheduler.transfers.append(req)

    parallel = SimpleNamespace(dp_size=1, pp_size=1, attn_dp_size=1, attn_cp_size=1)
    for name, module in {
        "sglang.srt.disaggregation.decode": SimpleNamespace(
            DecodePreallocQueue=Queue, SchedulerDisaggregationDecodeMixin=Mixin
        ),
        "sglang.srt.managers.scheduler": SimpleNamespace(Scheduler=Scheduler),
        "sglang.srt.runtime_context": SimpleNamespace(
            get_parallel=lambda: parallel,
            get_disagg=lambda: SimpleNamespace(
                disaggregation_decode_host_receive_threshold=0
            ),
        ),
        "sglang_tpu.envs": SimpleNamespace(
            envs=SimpleNamespace(
                SGLANG_TPU_BENCHMARK_PD_BATCH_SIZE=SimpleNamespace(get=lambda: 2),
                SGLANG_TPU_BENCHMARK_PD_TIMEOUT_SEC=SimpleNamespace(get=lambda: 10),
            )
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(benchmark_pd_gate, "_installed", False)
    now = [1.0]
    monkeypatch.setattr(benchmark_pd_gate.time, "monotonic", lambda: now[0])
    benchmark_pd_gate.install()
    scheduler = Scheduler()
    scheduler.waiting_queue = []
    scheduler.transfers = []
    scheduler.enable_decode_hicache = scheduler.enable_hisparse = False
    scheduler.is_fully_idle = lambda: True
    scheduler.req_to_token_pool = SimpleNamespace(size=2)
    scheduler.max_running_requests = 2
    scheduler.disaggregation_mode = SimpleNamespace(value="decode")
    queue = Queue()
    queue.scheduler = scheduler
    queue.retracted_queue = []
    scheduler.disagg_decode_prealloc_queue = queue
    queue.add(request(0))
    queue.add(request(1))
    running = SimpleNamespace(batch_size=lambda: 0)
    scheduler.process_decode_queue()
    assert scheduler._get_new_prebuilt_batch(running) is None
    scheduler.process_decode_queue()  # Still polls while admission is held.
    batch = scheduler._get_new_prebuilt_batch(running)
    now[0] = 2
    result = scheduler.run_batch(batch)
    now[0] = 6
    scheduler.process_batch_result(batch, result)
    summary = scheduler.get_internal_state(None).internal_state["benchmark_pd_gate"]
    assert summary["decode_duration"] == 4
    assert summary["decode_output_tokens"] == 8
