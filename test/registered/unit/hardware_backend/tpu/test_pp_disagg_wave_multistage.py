"""PP8 PD loops with blocking host messages and asynchronously queued ICI work."""

import queue
import threading
import time
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers import scheduler_pp_mixin  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang_tpu.pp_wave import PPWave  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Stopped(Exception):
    pass


class Fabric:
    """Separate FIFO host channels; device launches return before peers arrive."""

    def __init__(self, size):
        self.size = size
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.channels = {}
        self.counts = [0] * size
        self.launches = {}
        self.done = set()
        self.errors = []
        self.steps = [-1] * size
        self.waiting = {}
        self.control_sends = [0] * size
        self.control_waits = [0] * size
        self.max_pending_controls = [0] * size

    def describe(self):
        with self.lock:
            return {
                "steps": list(self.steps),
                "launches": list(self.counts),
                "waiting": dict(self.waiting),
                "queued": {
                    str(key): messages.qsize()
                    for key, messages in self.channels.items()
                    if not messages.empty()
                },
            }

    def channel(self, kind, dst):
        with self.lock:
            return self.channels.setdefault((kind, dst), queue.Queue())

    def send(self, rank, kind, value):
        is_control = kind != "output"
        if is_control:
            with self.lock:
                self.control_sends[rank] += 1
                pending = self.control_sends[rank] - self.control_waits[rank]
                self.max_pending_controls[rank] = max(
                    self.max_pending_controls[rank], pending
                )
        received = threading.Event()
        self.channel(kind, (rank + 1) % self.size).put((value, received))

        def wait():
            self.wait_for(received.is_set, rank, f"send {kind}: {value}")
            if is_control:
                with self.lock:
                    self.control_waits[rank] += 1

        return [SimpleNamespace(work=SimpleNamespace(wait=wait))]

    def receive(self, rank, kind):
        messages = self.channel(kind, rank)
        with self.lock:
            self.waiting[rank] = f"receive {kind}"
        try:
            while not self.stop.is_set():
                try:
                    value, received = messages.get(timeout=0.05)
                except queue.Empty:
                    continue
                received.set()
                return value
            raise Stopped
        finally:
            with self.lock:
                self.waiting.pop(rank, None)

    def wait_for(self, predicate, rank, reason):
        deadline = time.monotonic() + 10
        with self.lock:
            self.waiting[rank] = reason
        try:
            while not predicate():
                if self.stop.wait(0.01):
                    raise Stopped
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"PD control/device progress stalled: {self.describe()}"
                    )
        finally:
            with self.lock:
                self.waiting.pop(rank, None)

    def launch(self, rank, value):
        with self.lock:
            index = self.counts[rank]
            self.counts[rank] += 1
            self.launches[index, rank] = value.clone()

        def result():
            def ready():
                with self.lock:
                    return all(count > index for count in self.counts)

            self.wait_for(ready, rank, f"device launch {index}")
            with self.lock:
                return self.launches[index, (rank - 1) % self.size]

        return SimpleNamespace(result=result)


def stage(fabric, rank, mode, schedule):
    scheduler = SchedulerPPMixin()
    # Every stage shares this process, so each thread reads its own topology.
    scheduler.test_parallel = SimpleNamespace(
        pp_size=fabric.size,
        pp_rank=rank,
        pp_async_batch_depth=1,
        enable_dsa_prefill_context_parallel=False,
    )
    scheduler.pp_group = SimpleNamespace(
        rank_in_group=rank,
        world_size=fabric.size,
        is_first_rank=rank == 0,
        is_last_rank=rank == fabric.size - 1,
    )
    step = [-1]
    processed = []
    scheduler.disaggregation_mode = mode
    scheduler.waiting_queue = []
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(queue=[])
    scheduler.device_module = Mock()
    scheduler.forward_stream_ctx = nullcontext()
    scheduler.copy_stream_ctx = nullcontext()
    scheduler.use_overlap_streams = False
    scheduler.enable_staging = False
    scheduler.process_pending_chunked_abort = Mock()
    scheduler.process_prefill_chunk = Mock()
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda x: x
    )
    wave = PPWave(
        rank=rank,
        size=fabric.size,
        rows=8,
        hidden=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        launcher=lambda x: fabric.launch(rank, x),
    )
    scheduler._pp_init_wave = lambda: wave
    scheduler._pp_send_pyobj_to_next_stage = lambda data, async_send=False: fabric.send(
        rank, "raw", data
    )
    scheduler._pp_recv_pyobj_from_prev_stage = lambda: fabric.receive(rank, "raw")

    def send_consensus(data, *, dst, async_send, tag):
        assert dst == (rank + 1) % fabric.size
        assert tag != 0  # Tensor output metadata uses tag zero on this group.
        return fabric.send(rank, ("consensus", tag), data)

    def recv_consensus(*, src, tag):
        assert src == (rank - 1) % fabric.size
        return fabric.receive(rank, ("consensus", tag))

    scheduler.pp_group.send_object = send_consensus
    scheduler.pp_group.recv_object = recv_consensus

    def receive_requests():
        if fabric.stop.is_set():
            raise Stopped
        step[0] += 1
        with fabric.lock:
            fabric.steps[rank] = step[0]
        data = ["requests", step[0]] if rank == 0 else fabric.receive(rank, "raw")
        assert data == ["requests", step[0]], (rank, step[0], data)
        return data

    scheduler.ingest_requests = receive_requests
    scheduler._process_hicache_events = Mock()

    def votes(kind):
        def get(*args):
            data = [kind, step[0]] if rank == 0 else fabric.receive(rank, "raw")
            assert data == [kind, step[0]], (rank, step[0], kind, data)
            return data

        return get

    scheduler._pp_pd_stage_ready = Mock()
    scheduler._pp_pd_get_bootstrapped_ids = votes("bootstrap")
    scheduler._pp_pd_get_prefill_transferred_ids = votes("transfer")
    scheduler._pp_pd_get_retract_ids = votes("retract")
    scheduler._pp_pd_get_prealloc_ids = votes("prealloc")
    scheduler._pp_pd_get_decode_transferred_ids = votes("transfer")

    def released(kind):
        def process(data):
            assert data[0] == kind, (rank, kind, data)
            return data

        return process

    scheduler.process_bootstrapped_queue = released("bootstrap")
    scheduler.process_disagg_prefill_inflight_queue = released("transfer")
    scheduler.process_retract_queue = released("retract")
    scheduler.process_prealloc_queue = released("prealloc")
    scheduler.process_decode_transfer_queue = released("transfer")

    def plan(running_batch):
        item = schedule[step[0]] if step[0] < len(schedule) else None
        current = (
            None
            if item is None
            else SimpleNamespace(
                forward_mode=item,
                reqs=[],
                return_logprob=False,
                ordinal=step[0],
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
        )
        return SimpleNamespace(batch_to_run=current, running_batch=running_batch)

    scheduler.get_new_batch_prefill = plan
    scheduler.get_next_disagg_decode_batch_to_run = plan

    def run_batch(current, proxy):
        if current.forward_mode.is_prebuilt():
            assert proxy is None
            hidden = torch.zeros(8, 4)
        elif rank == 0:
            hidden = torch.full((8, 4), 100.0 + current.ordinal)
        else:
            # A real host read of queued device work. Holding current controls
            # until after this point deadlocks: later ranks cannot queue ICI.
            hidden = proxy["hidden_states"].result()
            assert float(hidden[0, 0]) == 100 + current.ordinal + rank - 1
            hidden = hidden + 1
        return GenerationBatchResult(
            can_run_cuda_graph=False,
            next_token_ids=torch.tensor([current.ordinal]),
            pp_hidden_states_proxy_tensors=PPProxyTensors(
                {"hidden_states": hidden, "residual": torch.zeros_like(hidden)}
            ),
        )

    scheduler.run_batch = run_batch
    scheduler._pp_send_dict_to_next_stage = lambda data, **kwargs: fabric.send(
        rank, "output", data
    )
    scheduler._pp_recv_dict_from_prev_stage = lambda: (
        fabric.receive(rank, "output"),
        None,
    )
    scheduler._pp_prep_batch_result = lambda target, metadata, outputs: SimpleNamespace(
        logits_output=None, next_token_ids=outputs["next_token_ids"]
    )

    def process_batch(target, result):
        assert int(result.next_token_ids[0]) == target.ordinal
        processed.append(target.ordinal)

    scheduler._pp_process_batch_result = process_batch
    expected = [
        i
        for i, item in enumerate(schedule)
        if item is not None and not item.is_prebuilt()
    ]

    def idle():
        if processed == expected and wave.owed == 0 and wave.pending is None:
            with fabric.lock:
                fabric.done.add(rank)
                if len(fabric.done) == fabric.size:
                    fabric.stop.set()

    scheduler.on_idle = idle
    return scheduler, wave


class TestMultiStageDisaggregatedWave(unittest.TestCase):
    def test_pp8_cold_start_lone_prompt_idle_restart_and_prebuilt(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                forward = (
                    ForwardMode.EXTEND
                    if mode == DisaggregationMode.PREFILL
                    else ForwardMode.DECODE
                )
                schedule = [forward] + [None] * 16 + [forward]
                if mode == DisaggregationMode.DECODE:
                    schedule = (
                        [ForwardMode.PREBUILT, forward, ForwardMode.PREBUILT]
                        + [None] * 16
                        + [forward]
                    )
                fabric = Fabric(8)
                stages = [stage(fabric, rank, mode, schedule) for rank in range(8)]

                def run(scheduler):
                    stage_parallel.value = scheduler.test_parallel
                    try:
                        loop = (
                            scheduler.event_loop_pp_disagg_prefill
                            if mode == DisaggregationMode.PREFILL
                            else scheduler.event_loop_pp_disagg_decode
                        )
                        loop()
                    except Stopped:
                        pass
                    except Exception as exc:
                        with fabric.lock:
                            fabric.errors.append(exc)
                        fabric.stop.set()

                stage_parallel = threading.local()
                with (
                    patch.object(scheduler_pp_mixin, "_is_tpu", True),
                    patch.object(
                        scheduler_pp_mixin,
                        "get_parallel",
                        side_effect=lambda: stage_parallel.value,
                    ),
                    patch.object(
                        scheduler_pp_mixin,
                        "get_disagg",
                        return_value=SimpleNamespace(
                            disaggregation_decode_enable_offload_kvcache=False
                        ),
                    ),
                    patch.object(
                        scheduler_pp_mixin,
                        "ScheduleBatch",
                        side_effect=lambda **kw: SimpleNamespace(**kw),
                    ),
                ):
                    threads = [
                        threading.Thread(target=run, args=(scheduler,), daemon=True)
                        for scheduler, _ in stages
                    ]
                    for thread in threads:
                        thread.start()
                    if not fabric.stop.wait(15):
                        fabric.errors.append(
                            TimeoutError(f"PD loops stalled: {fabric.describe()}")
                        )
                    fabric.stop.set()
                    for thread in threads:
                        thread.join(timeout=2)
                self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(fabric.errors, [])
                self.assertEqual(fabric.done, set(range(8)))
                self.assertEqual(len(set(fabric.counts)), 1)
                packets_per_step = 5 if mode == DisaggregationMode.PREFILL else 7
                for rank, (scheduler, _) in enumerate(stages):
                    self.assertGreater(fabric.control_waits[rank], 0)
                    self.assertLessEqual(
                        fabric.max_pending_controls[rank],
                        scheduler.pp_loop_size * packets_per_step,
                    )
                for _, wave in stages:
                    self.assertEqual(wave.forwards, 2)
                    self.assertEqual(wave.owed, 0)


if __name__ == "__main__":
    unittest.main()
