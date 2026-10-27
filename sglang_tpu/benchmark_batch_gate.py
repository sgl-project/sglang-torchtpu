"""Separate prefill from decode for a tagged, fixed-size benchmark batch.

Why: decode measurements must exclude work overlapped with unfinished prefill.
How: hold decode admission and reuse the scheduler's request completion times.
Upstream PR or issue: https://github.com/sgl-project/sglang-torchtpu/pull/1693
Remove when: the scheduler exposes a benchmark admission/timing policy seam.
"""

from __future__ import annotations

import math
import time

PREFIX = "hourly-batch:"
_installed = False


class BenchmarkBatchGate:
    def __init__(self, size: int, timeout: float):
        if size < 1 or timeout <= 0:
            raise ValueError("Benchmark gate needs positive size and timeout")
        self.size, self.timeout = size, timeout
        self.requests = {}
        self.cohort = None
        self.arrival = self.start = None
        self.last_result = None
        self.held_batch = None

    def add(self, req, now):
        if self.requests and all(r.finished() for r in self.requests.values()):
            self.report()
        if not req.rid.startswith(PREFIX):
            if self.requests:
                raise RuntimeError("Unrelated request entered benchmark batch")
            return
        cohort, _, index = req.rid.removeprefix(PREFIX).rpartition("_")
        if not cohort or not index.isdecimal() or not 0 <= int(index) < self.size:
            raise ValueError(f"Invalid benchmark request ID: {req.rid}")
        if not self.requests:
            self.cohort, self.arrival = cohort, now
        if cohort != self.cohort or req.rid in self.requests:
            raise RuntimeError("Mixed or duplicate benchmark batch")
        if req.sampling_params.max_new_tokens <= 1:
            raise ValueError("Synchronized measurement needs decode tokens")
        self.requests[req.rid] = req

    def check(self, now):
        if any(req.retraction_count for req in self.requests.values()):
            raise RuntimeError("Benchmark request retracted; measurement invalid")
        if self.requests and self.start is None and now - self.arrival > self.timeout:
            raise TimeoutError(
                f"Benchmark {self.cohort} timed out waiting for prefill/KV"
            )

    def ready(self, reqs):
        return (
            len(reqs) == self.size
            and {r.rid for r in reqs} == self.requests.keys()
            and all(len(r.output_ids) == 1 for r in reqs)
        )

    def begin(self, reqs, now):
        if not self.requests or self.start is not None:
            return
        if not self.ready(reqs):
            raise RuntimeError("First decode must contain the full prefilled batch")
        self.start = now

    def report(self):
        if not self.requests or not all(r.finished() for r in self.requests.values()):
            return self.last_result
        self.check(time.perf_counter())
        reqs = list(self.requests.values())
        # SBD can overshoot internally; measure the tokens actually emitted.
        if self.start is None or any(
            len(r.output_ids_through_stop) != r.sampling_params.max_new_tokens
            for r in reqs
        ):
            raise RuntimeError("Benchmark did not complete the requested decode")
        ends = [r.time_stats.completion_time for r in reqs]
        if any(not math.isfinite(end) or end <= self.start for end in ends):
            raise RuntimeError("Invalid scheduler completion time")
        self.last_result = {
            "cohort": self.cohort,
            "batch_size": self.size,
            "decode_duration": max(ends) - self.start,
            "decode_output_tokens": sum(
                len(r.output_ids_through_stop) - 1 for r in reqs
            ),
            "gate_wait_duration": self.start - self.arrival,
            "tpot_ms": [
                (end - self.start) * 1000 / (len(req.output_ids_through_stop) - 1)
                for req, end in zip(reqs, ends)
            ],
        }
        self.requests.clear()
        self.cohort = self.arrival = self.start = None
        return self.last_result


def install():
    global _installed
    if _installed:
        return
    from sglang_tpu.envs import envs

    size = envs.SGLANG_TPU_BENCHMARK_BATCH_SIZE.get()
    if size == 0:
        return
    from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import get_disagg, get_parallel

    def gate(scheduler):
        if "_benchmark_batch_gate" not in scheduler.__dict__:
            scheduler.__dict__["_benchmark_batch_gate"] = BenchmarkBatchGate(
                size, envs.SGLANG_TPU_BENCHMARK_TIMEOUT_SEC.get()
            )
        return scheduler.__dict__["_benchmark_batch_gate"]

    original_add = Scheduler._add_request_to_queue
    original_pd = SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run
    original_unified = Scheduler.get_next_batch_to_run
    original_update = Scheduler.update_running_batch
    original_run = Scheduler.run_batch
    original_state = Scheduler.get_internal_state

    def add(self, req, is_retracted=False):
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            state = gate(self)
            if req.rid.startswith(PREFIX):
                state.report()
                parallel = get_parallel()
                if (
                    any(
                        n != 1
                        for n in (
                            parallel.dp_size,
                            parallel.pp_size,
                            parallel.attn_dp_size,
                            parallel.attn_cp_size,
                        )
                    )
                    or self.enable_decode_hicache
                    or self.enable_hisparse
                    or get_disagg().disaggregation_decode_host_receive_threshold > 0
                    or min(self.req_to_token_pool.size, self.max_running_requests)
                    < size
                ):
                    raise ValueError(
                        "Benchmark gate requires DP1/CP1/PP1 and full batch capacity with device KV"
                    )
                if is_retracted:
                    raise RuntimeError(
                        "Benchmark request retracted; measurement invalid"
                    )
                if not state.requests and not self.is_fully_idle():
                    raise RuntimeError("Benchmark must start on an idle scheduler")
            state.add(req, time.perf_counter())
        return original_add(self, req, is_retracted)

    def wrap_planner(planner, *, unified=False):
        def plan(self, running_batch, *args):
            state = gate(self)
            state.check(time.perf_counter())
            if not state.requests:
                return planner(self, running_batch, *args)
            if unified and len(state.requests) != size:
                return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
            if unified:
                # Preserve chunked prefill without mixing held decode requests.
                mixed = self.is_mixed_chunk
                self.is_mixed_chunk = False
            state.held_batch = None
            try:
                # P/D processes prebuilt results and emits first tokens here.
                result = planner(self, running_batch, *args)
            finally:
                if unified:
                    self.is_mixed_chunk = mixed
            if state.held_batch is not None:
                result.running_batch = state.held_batch
                state.held_batch = None
            return result

        return plan

    def update_running_batch(self, batch):
        state = gate(self)
        if state.requests and state.start is None and not state.ready(batch.reqs):
            # Hold before prepare_for_decode allocates KV and advances positions.
            # Let the planner see an empty batch, then restore the live batch.
            state.held_batch = batch
            return ScheduleBatch(reqs=[], batch_is_full=batch.batch_is_full)
        return original_update(self, batch)

    def run_batch(self, batch, *args, **kwargs):
        if (
            self.disaggregation_mode != DisaggregationMode.PREFILL
            and batch.forward_mode.is_decode()
        ):
            gate(self).begin(batch.reqs, time.perf_counter())
        return original_run(self, batch, *args, **kwargs)

    def internal_state(self, recv_req):
        result = original_state(self, recv_req)
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            result.internal_state["benchmark_batch_gate"] = gate(self).report()
        return result

    Scheduler._add_request_to_queue = add
    SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run = (
        wrap_planner(original_pd)
    )
    Scheduler.get_next_batch_to_run = wrap_planner(original_unified, unified=True)
    Scheduler.update_running_batch = update_running_batch
    Scheduler.run_batch = run_batch
    Scheduler.get_internal_state = internal_state
    _installed = True
