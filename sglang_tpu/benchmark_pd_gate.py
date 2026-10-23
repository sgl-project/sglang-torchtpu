"""A fixed-cohort decode admission gate for isolated P/D benchmarks.

Why: hourly decode timing must exclude prefill and KV transfer overlap.
How: tag benchmark requests, keep polling transfers, then admit the whole cohort.
Upstream PR or issue: pending (local draft).
Remove when: the scheduler exposes a benchmark admission/timing policy seam.

This module's state machine uses only host metadata and the decoder's monotonic
clock. Ordinary requests do not activate it. It is deliberately limited to one
DP group, PP1, device-resident KV, and one benchmark cohort at a time.
"""

from __future__ import annotations

import logging
import time

PREFIX = "hourly-pd-sync:"
_installed = False
logger = logging.getLogger(__name__)


class BenchmarkPDGate:
    def __init__(self, size: int, timeout: float):
        if size < 1 or timeout <= 0:
            raise ValueError("P/D benchmark gate needs positive size and timeout")
        self.size = size
        self.timeout = timeout
        self.cohort = None
        self.arrived = {}
        self.released = False
        self.arrival_time = None
        self.start = None
        self.finished = {}
        self.last_result = None

    def admit(self, req, now: float):
        if not req.rid.startswith(PREFIX):
            if self.cohort is not None:
                raise RuntimeError("Unrelated request entered the P/D benchmark cohort")
            return
        cohort, separator, index = req.rid.removeprefix(PREFIX).rpartition("_")
        if (
            not separator
            or not cohort
            or not index.isdecimal()
            or not 0 <= int(index) < self.size
        ):
            raise ValueError(f"Invalid P/D benchmark request ID: {req.rid}")
        if self.cohort is None:
            self.cohort = cohort
            self.arrival_time = now
        if cohort != self.cohort or req.rid in self.arrived:
            raise RuntimeError("Mixed or duplicate P/D benchmark cohort")
        if req.sampling_params.max_new_tokens <= 1:
            raise ValueError("Synchronized P/D measurement needs decode tokens")
        self.arrived[req.rid] = req

    def ready(self, waiting, running_size: int, capacity: int, now: float) -> bool:
        if self.cohort is None or self.released:
            return True
        if capacity < self.size or running_size:
            raise RuntimeError("P/D benchmark cohort cannot fit as one decode batch")
        if now - self.arrival_time > self.timeout:
            raise TimeoutError(
                f"P/D benchmark {self.cohort} timed out: "
                f"{len(self.arrived)}/{self.size} arrived, {len(waiting)} KV-ready"
            )
        ready_ids = {req.rid for req in waiting}
        if not ready_ids <= self.arrived.keys() or len(ready_ids) != len(waiting):
            raise RuntimeError("Unexpected request in P/D benchmark ready queue")
        return len(self.arrived) == self.size and ready_ids == self.arrived.keys()

    def release(self, reqs):
        if {req.rid for req in reqs} != self.arrived.keys():
            raise RuntimeError("Scheduler split the P/D benchmark cohort")
        if any(len(req.output_ids) != 1 for req in reqs):
            raise RuntimeError("P/D benchmark requires exactly one prefill token")
        self.released = True
        logger.info(
            "P/D benchmark %s released all %d KV-ready requests", self.cohort, self.size
        )

    def begin_decode(self, now: float):
        if self.released and self.start is None:
            self.start = now

    def observe_completion(self, now: float):
        if self.start is None:
            return
        for rid, req in self.arrived.items():
            if req.finished() and rid not in self.finished:
                if len(req.output_ids) != req.sampling_params.max_new_tokens:
                    raise RuntimeError(
                        "Synchronized P/D request did not finish at length"
                    )
                self.finished[rid] = now
        if len(self.finished) != self.size:
            return
        duration = now - self.start
        if duration <= 0:
            raise RuntimeError("Invalid P/D benchmark decode interval")
        tokens = sum(len(req.output_ids) - 1 for req in self.arrived.values())
        self.last_result = {
            "cohort": self.cohort,
            "batch_size": self.size,
            "decode_duration": duration,
            "decode_output_tokens": tokens,
            "gate_wait_duration": self.start - self.arrival_time,
            "tpot_ms": [
                (self.finished[rid] - self.start) * 1000 / (len(req.output_ids) - 1)
                for rid, req in self.arrived.items()
            ],
        }
        self.cohort = None
        self.arrived = {}
        self.finished = {}
        self.released = False
        self.start = None
        self.arrival_time = None


def install():
    """Wrap admission and completed-result boundaries, once per process."""
    global _installed
    if _installed:
        return
    from sglang.srt.disaggregation.decode import (
        DecodePreallocQueue,
        SchedulerDisaggregationDecodeMixin,
    )
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import get_disagg, get_parallel
    from sglang_tpu.envs import envs

    size = envs.SGLANG_TPU_BENCHMARK_PD_BATCH_SIZE.get()
    if size == 0:
        return
    timeout = envs.SGLANG_TPU_BENCHMARK_PD_TIMEOUT_SEC.get()

    def gate(scheduler):
        if "_benchmark_pd_gate" not in scheduler.__dict__:
            scheduler.__dict__["_benchmark_pd_gate"] = BenchmarkPDGate(size, timeout)
        return scheduler.__dict__["_benchmark_pd_gate"]

    original_add = DecodePreallocQueue.add
    original_get = SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch
    original_run = Scheduler.run_batch
    original_result = Scheduler.process_batch_result
    original_state = Scheduler.get_internal_state
    original_queue = SchedulerDisaggregationDecodeMixin.process_decode_queue

    def add(self, req, is_retracted=False, is_rebootstrap=False):
        state = gate(self.scheduler)
        if req.rid.startswith(PREFIX):
            parallel = get_parallel()
            if (
                parallel.dp_size != 1
                or parallel.pp_size != 1
                or parallel.attn_dp_size != 1
                or parallel.attn_cp_size != 1
                or self.scheduler.enable_decode_hicache
                or self.scheduler.enable_hisparse
                or get_disagg().disaggregation_decode_host_receive_threshold > 0
            ):
                raise ValueError("P/D benchmark gate supports DP1/PP1 with device KV")
            if is_retracted or is_rebootstrap:
                raise RuntimeError(
                    "P/D benchmark cohort retracted; measurement invalid"
                )
            if state.cohort is None and not self.scheduler.is_fully_idle():
                raise RuntimeError("P/D benchmark must start on an idle decoder")
        state.admit(req, time.monotonic())
        return original_add(self, req, is_retracted, is_rebootstrap)

    def get_batch(self, running_batch):
        state = gate(self)
        if not state.ready(
            self.waiting_queue,
            running_batch.batch_size(),
            min(self.req_to_token_pool.size, self.max_running_requests),
            time.monotonic(),
        ):
            return None
        batch = original_get(self, running_batch)
        if state.cohort is not None and not state.released and batch is not None:
            state.release(batch.reqs)
        return batch

    def process_queue(self):
        state = gate(self)
        if (
            state.cohort is not None
            and self.disagg_decode_prealloc_queue.retracted_queue
        ):
            raise RuntimeError("P/D benchmark cohort retracted; measurement invalid")
        return original_queue(self)

    def run_batch(self, batch, *args, **kwargs):
        if self.disaggregation_mode.value == "decode":
            gate(self).begin_decode(time.monotonic())
        return original_run(self, batch, *args, **kwargs)

    def process_result(self, batch, result):
        value = original_result(self, batch, result)
        if self.disaggregation_mode.value == "decode":
            gate(self).observe_completion(time.monotonic())
        return value

    def internal_state(self, recv_req):
        result = original_state(self, recv_req)
        if self.disaggregation_mode.value == "decode":
            result.internal_state["benchmark_pd_gate"] = gate(self).last_result
        return result

    DecodePreallocQueue.add = add
    SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch = get_batch
    SchedulerDisaggregationDecodeMixin.process_decode_queue = process_queue
    Scheduler.run_batch = run_batch
    Scheduler.process_batch_result = process_result
    Scheduler.get_internal_state = internal_state
    _installed = True
