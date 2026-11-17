"""K3_PROGRAM decode under overlap scheduling with sglang's per-step
bookkeeping on the host (SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING,
k3_programs/sglang_host_relay.py).

The device ops each step leaves behind the program's, counted on CPU by
tracking which tensors would live on the TPU, and the scheduler's results with
the switch off and on: real ``Scheduler.run_batch``, ``FutureMap``,
``ScheduleBatch``, ``Req``, decode result processing and decode KV sizing
around stub programs, in the overlap loop's order (a step launches before the
previous one's result is processed)."""

import contextlib
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_colocated
from sglang_tpu.k3_programs import sglang_worker as worker
from sglang_tpu.k3_programs.kimi_k3 import pools
from sglang_tpu.k3_programs.program import StepOutput
from sglang_tpu.k3_programs.sglang_host_relay import (
    HostRowsResult,
    accept_counts,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

WIDTH = worker.WIDTH
STOP = 7
BLOCK = 8192  # tokens per logical block (bf16 layout)
ALLOC_PAGE = 16  # sglang's KV page in the decode sizing below
POOL_ROWS = 6
# The device relay's ops after a step's program ops, in launch order: the
# worker's accept lengths, last tokens and new lengths, the future map's
# length store, the int64 casts, the token store; then the next step's two
# resolves (torch_tpu also lands each store before its gather there).
DEVICE_RELAY_OPS = [
    "ge",
    "sum",
    "sub",
    "clamp",
    "_to_copy",
    "gather",
    "_to_copy",
    "add",
    "index_copy_",
    "_to_copy",
    "_to_copy",
    "index_copy_",
]
RESOLVE_OPS = ["index", "index"]
_FREE = {
    "view",
    "_unsafe_view",
    "unsqueeze",
    "squeeze",
    "reshape",
    "slice",
    "select",
    "detach",
    "alias",
    "expand",
    "t",
    "transpose",
    "permute",
    "lift_fresh",
}


class DeviceOps(TorchDispatchMode):
    """The torch ops sglang would run on the TPU: ops reading a 'device'
    tensor (marked, or produced from one). Views are metadata only; reading a
    device scalar is a sync. Ops inside :meth:`program` are the program's."""

    def __init__(self):
        super().__init__()
        self._device: dict[int, weakref.ref] = {}
        self._muted = 0
        self.ops: list[str] = []
        self.syncs = 0

    def mark(self, *tensors):
        for t in tensors:
            if isinstance(t, torch.Tensor):
                self._device[id(t)] = weakref.ref(t)
        return tensors[0] if len(tensors) == 1 else tensors

    def on_device(self, t) -> bool:
        ref = self._device.get(id(t)) if isinstance(t, torch.Tensor) else None
        return ref is not None and ref() is t

    @contextlib.contextmanager
    def program(self):
        self._muted += 1
        try:
            yield
        finally:
            self._muted -= 1

    def take(self) -> list[str]:
        ops, self.ops = self.ops, []
        return ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        flat = list(args) + list(kwargs.values())
        flat += [x for a in flat if isinstance(a, (list, tuple)) for x in a]
        if not any(self.on_device(a) for a in flat):
            return out
        outs = out if isinstance(out, (list, tuple)) else (out,)
        self.mark(*[o for o in outs if isinstance(o, torch.Tensor)] or [None])
        name = func.overloadpacket.__name__
        if not self._muted:
            if name == "_local_scalar_dense":
                self.syncs += 1
            elif name not in _FREE:
                self.ops.append(name)
        return out


def token(index: int, position: int) -> int:
    """Request ``index``'s output token at ``position`` (never STOP)."""
    return 1000 + 100 * index + position % 89


class StubDecode:
    """A decode program: each request's position held on the 'device' from
    its first step (the host's committed length lags under overlap), accept
    counts from a script, rows padded to four, -1 padded."""

    def __init__(self, ops: DeviceOps, accepts, stops):
        self.ops, self.accepts, self.stops = ops, accepts, stops
        self.position: dict[str, int] = {}
        self.steps: dict[str, int] = {}
        self.inputs: list = []

    def step(self, step_input):
        with self.ops.program():
            self.inputs.append(
                (
                    sorted(step_input.finished),
                    [
                        (
                            r.req_id,
                            r.computed,
                            r.scheduled,
                            r.num_tokens,
                            [t.tolist() for t in r.block_tables],
                        )
                        for r in step_input.requests
                    ],
                )
            )
            for rid in step_input.finished:
                self.position.pop(rid, None)
            n = len(step_input.requests)
            rows = torch.full((max(4, -(-n // 4) * 4), WIDTH), -1, dtype=torch.int32)
            for i, r in enumerate(step_input.requests):
                rid = r.req_id
                if rid not in self.position:
                    self.position[rid] = r.num_tokens - r.num_prompt_tokens
                    self.steps[rid] = 0
                script = self.accepts[rid]
                k = min(script[self.steps[rid] % len(script)], r.scheduled)
                p, index = self.position[rid], ord(rid[0]) - ord("a")
                run = [
                    STOP if p + j in self.stops.get(rid, ()) else token(index, p + j)
                    for j in range(k)
                ]
                rows[i, :k] = torch.tensor(run, dtype=torch.int32)
                self.position[rid] += k
                self.steps[rid] += 1
            return StepOutput(self.ops.mark(rows))


class StubPrefill:
    """A prefill program: a finished prompt's first token (output 0)."""

    def __init__(self, ops: DeviceOps):
        self.ops = ops

    def step(self, step_input):
        with self.ops.program():
            rows = torch.full(
                (max(len(step_input.requests), 1), WIDTH), -1, dtype=torch.int32
            )
            for i, r in enumerate(step_input.requests):
                if r.end >= r.num_prompt_tokens:
                    rows[i, 0] = token(ord(r.req_id[0]) - ord("a"), 0)
            return StepOutput(self.ops.mark(rows))


class _Sampling:
    """The batch's sampling info as the scheduler moves it (no penalties)."""

    def copy_for_forward(self):
        return self

    def filter_batch(self, *args):
        pass

    def merge_batch(self, other):
        pass


class Recorder:
    """What sglang's decode result processing hands on: each finish (whose
    release is not performed here) and each stream call."""

    def __init__(self):
        self.finishes: list = []
        self.streamed: list = []

    def finish(self, req) -> None:
        if req.finished():
            self.finishes.append(
                (req.rid, type(req.finished_reason).__name__, req.finished_len)
            )

    def stream(self, reqs, return_logprob) -> None:
        self.streamed.append(
            [(r.rid, list(r.output_ids_through_stop), r.finished()) for r in reqs]
        )


def _processor(model_worker, recorder: Recorder) -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None, is_encoder_decoder=False),
        token_to_kv_pool_allocator=mock.MagicMock(),
        tree_cache=None,
        hisparse_coordinator=None,
        beam_coordinator=mock.MagicMock(),
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=mock.MagicMock(),
        draft_worker=None,
        model_worker=model_worker,
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(stream_output=recorder.stream),
        abort_request=lambda *args, **kwargs: None,
    )


class _Event:
    def record(self):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True


def _req(rid: str, prompt: int, max_new: int, pool_idx: int) -> Req:
    params = SamplingParams(max_new_tokens=max_new, stop_token_ids=[STOP])
    params.normalize(None)
    req = Req(rid, "", list(range(10, 10 + prompt)), params)
    req.kv.req_pool_idx = pool_idx
    req.kv.mamba_pool_idx = torch.tensor([pool_idx])
    return req


class OverlapRun:
    """One engine: a colocated K3ProgramWorker over stub programs, sglang's
    future map for the switch, a scheduler for run_batch, the decode result
    processor, and the overlap loop's batch handling."""

    def __init__(self, *, accepts, stops):
        self.ops = DeviceOps()
        self.algo = SpeculativeAlgorithm.from_string(worker.ALGORITHM)
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        r2t = torch.zeros((POOL_ROWS, 4 * BLOCK), dtype=torch.int32)
        for row in range(POOL_ROWS):
            r2t[row, :BLOCK] = torch.arange(row * BLOCK, (row + 1) * BLOCK)
        self.pool = SimpleNamespace(req_to_token=r2t, mamba_pool=None)
        self.decode = StubDecode(self.ops, accepts, stops)
        steps = sglang_colocated.ColocatedSteps(
            worker.PrefillSteps(StubPrefill(self.ops), plan, table_width=4),
            worker.DecodeSteps(
                self.decode, plan, table_width=4, schedule=((1, 4, 7), (5, 8, 3))
            ),
        )
        target = SimpleNamespace(
            model_runner=None, get_memory_pool=lambda: (self.pool, None)
        )
        worker.K3ProgramWorker.build_steps = lambda target_worker: steps
        try:
            self.worker = worker.K3ProgramWorker(None, 0, 0, target)
            self.worker.alloc_memory_pool()
        finally:
            worker.K3ProgramWorker.build_steps = None
        self.worker.guard.enabled = False  # torch_tpu only
        self.future_map = self.algo.create_future_map(
            "cpu", self.pool, needs_cpu_seq_lens=False
        )
        # As on the TPU: index_copy_ stores (FutureMap._store).
        self.future_map._use_dynamic_index = True
        self.ops.mark(
            self.future_map.output_tokens_buf, self.future_map.new_seq_lens_buf
        )
        self.scheduler = self._scheduler()
        self.recorder = Recorder()
        self.processor = _processor(self.worker, self.recorder)
        self.alloc: list = []
        self.trace: list = []

    def _scheduler(self) -> Scheduler:
        s = object.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.metrics_reporter = mock.Mock()
        s.forward_ct = 0
        s.processed_tokens_counter = 0
        s._sched_idled = False
        s.scripted_scheduler_hook = None
        s.profiler_manager = SimpleNamespace(_profile_batch_predicate=lambda b: None)
        s.forward_sleep_time = None
        s.disaggregation_mode = None
        s.is_generation = True
        s.enable_overlap = True
        s.enable_pdmux = False
        s.future_map = self.future_map
        s._confidence_budget_prepare = None
        s.forward_stream_ctx = contextlib.nullcontext()
        s.copy_stream_ctx = contextlib.nullcontext()
        s.use_overlap_streams = False
        s.model_worker = self.worker
        s.batch_record_buf = [None, None]
        s.batch_record_ct = 0
        s.enable_unified_memory = False
        s.device_module = SimpleNamespace(Event=_Event)
        s.tpu_async_copy_worker = None
        s.spec_algorithm = self.algo
        s.beam_coordinator = mock.Mock()
        s.chunked_req = None
        s.attn_dp_enabled = False
        return s

    def _batch(self, reqs, mode) -> ScheduleBatch:
        lens = [len(r.origin_input_ids) for r in reqs]
        rows = [r.kv.req_pool_idx for r in reqs]
        batch = ScheduleBatch(
            reqs=reqs,
            enable_overlap=True,
            device="cpu",
            spec_algorithm=self.algo,
            forward_mode=mode,
            model_config=SimpleNamespace(is_encoder_decoder=False),
            req_pool_indices=self.ops.mark(torch.tensor(rows, dtype=torch.int64)),
            req_pool_indices_cpu=torch.tensor(rows, dtype=torch.int64),
            seq_lens=self.ops.mark(torch.tensor(lens, dtype=torch.int64)),
            orig_seq_lens=self.ops.mark(torch.tensor(lens, dtype=torch.int32)),
            sampling_info=_Sampling(),
            return_logprob=False,
        )
        if mode == ForwardMode.EXTEND:
            batch.prefix_lens = [0] * len(reqs)
            batch.extend_lens = lens
            for r, n in zip(reqs, lens):
                r.kv.kv_committed_len = r.kv.kv_allocated_len = n
        return batch

    def _prepare_decode(self, batch: ScheduleBatch) -> None:
        """sglang's spec decode preparation on a host-resident page table
        (eagle_prepare_for_decode): each request's KV sized from its
        committed length; nothing here reads the device."""
        batch.forward_mode = ForwardMode.DECODE
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            batch.reqs, reserve=2 * WIDTH, page_size=ALLOC_PAGE
        )
        for r, n in zip(batch.reqs, nxt):
            r.decode_batch_idx += 1
            r.kv.kv_allocated_len = max(r.kv.kv_allocated_len, n)
        self.alloc.append((tuple(cur), tuple(nxt), needed))

    def _process(self, batch, result) -> None:
        if batch.forward_mode.is_decode():
            with mock.patch.object(
                SchedulerBatchResultProcessor,
                "_handle_finish_state_updated_req",
                lambda processor, req, *args: self.recorder.finish(req),
            ):
                self.processor.process_batch_result_decode(batch, result)
            return
        # A finished prompt's first token (sglang's prefill result processing).
        for req, first in zip(batch.reqs, result.next_token_ids.tolist()):
            req.output_ids.append(first)
            req.update_finish_state()

    def run(self, iterations: int, arrivals: dict) -> list:
        """The overlap loop for ``iterations`` rounds (and one to process the
        last result); ``arrivals``: round -> requests whose prefill runs that
        round. Returns each round's kind and the device ops sglang ran
        outside the programs."""
        running, prefilled, pending, last_rids = None, None, None, None
        rounds = []
        for round_ in range(iterations + 1):
            if prefilled is not None:
                if running is None or running.is_empty():
                    running = prefilled
                else:
                    running.merge_batch(prefilled)
                prefilled = None
            if running is not None:
                running.filter_batch()
            batch, kind = None, "idle"
            if round_ < iterations and round_ in arrivals:
                batch = prefilled = self._batch(arrivals[round_], ForwardMode.EXTEND)
                kind, last_rids = "prefill", None
            elif round_ < iterations and running is not None and not running.is_empty():
                batch = running
                rids = [r.rid for r in batch.reqs]
                kind = "steady" if rids == last_rids else "decode"
                last_rids = rids
                self._prepare_decode(batch)
            queued = None
            if batch is not None:
                result = Scheduler.run_batch(self.scheduler, batch)
                queued = (batch.copy(), result)
            if pending is not None:
                self._process(*pending)
            pending = queued
            rounds.append((kind, self.ops.take()))
            self.trace.append(
                [
                    (
                        r.rid,
                        list(r.output_ids),
                        r.kv.kv_committed_len,
                        r.kv.kv_allocated_len,
                        r.finished(),
                        r.finished_len,
                    )
                    for r in self.reqs
                ]
            )
        return rounds


# Per request: accept counts by step (cycled), outputs that are a stop token,
# prompt length, max new tokens, its row; and the round its prefill runs.
SCRIPT = {
    # A stop token inside an accepted run (output 30), after "b" left and
    # "c" and "d" joined.
    "a": ([3, 2, 5, 1, 4], {30}, 100, 64, 1, 0),
    # Its length cap (13) inside an accepted run.
    "b": ([8, 3], set(), 200, 13, 2, 0),
    # Joins the running "a" (its first decode step before its prefill's
    # token is processed); a stop token inside its third run.
    "c": ([2, 5, 1, 8], {8}, 150, 40, 3, 4),
    # Joins "a" and "c"; its length cap.
    "d": ([1, 6], set(), 120, 20, 4, 7),
}


def _run(host: bool):
    with (
        published_topology(),
        tpu_envs.SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING.override(host),
    ):
        run = OverlapRun(
            accepts={rid: v[0] for rid, v in SCRIPT.items()},
            stops={rid: v[1] for rid, v in SCRIPT.items()},
        )
        run.reqs = [_req(rid, *v[2:5]) for rid, v in SCRIPT.items()]
        arrivals: dict = {}
        for req, v in zip(run.reqs, SCRIPT.values()):
            arrivals.setdefault(v[5], []).append(req)
        with run.ops:
            rounds = run.run(24, arrivals)
    return run, rounds


def setUpModule():
    if overlap_utils._is_tpu:
        # As the TPU plugin runs sglang's future map (its stash/publish
        # handoff for EAGLE3, a no-op here).
        from sglang_tpu.patches import _patch_future_map_direct_handoff

        _patch_future_map_direct_handoff()
    worker.register()


class TestHostBookkeeping(CustomTestCase):
    def test_future_map_follows_the_switch(self):
        algo = SpeculativeAlgorithm.from_string(worker.ALGORITHM)
        pool = SimpleNamespace(req_to_token=torch.zeros((3, 8), dtype=torch.int32))
        with published_topology():
            maps = []
            for host, cpu_lens in ((False, False), (True, False), (True, True)):
                with tpu_envs.SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING.override(host):
                    maps.append(
                        algo.create_future_map("cpu", pool, needs_cpu_seq_lens=cpu_lens)
                    )
        self.assertEqual(
            [type(m).__name__ for m in maps],
            ["FutureMap", "HostRelayFutureMap", "FutureMap"],
        )
        relay = maps[1]
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(
                future_indices=torch.tensor([1]), dsa_topk_indices=0
            ),
            seq_lens=torch.tensor([9]),
            seq_lens_cpu=torch.tensor([9]),
            seq_lens_sum=9,
        )
        relay.publish(torch.tensor([1]), torch.tensor([12]))
        relay.stash(torch.tensor([1]), None)
        relay.resolve_seq_lens_cpu(batch)
        relay._resolve_spec_extras(batch)
        self.assertEqual(batch.seq_lens.tolist(), [9])
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)
        self.assertIsNone(batch.spec_info.dsa_topk_indices)
        with self.assertRaisesRegex(NotImplementedError, "mixed"):
            relay.resolve_mixed_spec_tails(batch)

    def test_rows_result_builds_the_host_fields_from_the_copy(self):
        rows = torch.full((4, WIDTH), -1, dtype=torch.int32)
        rows[0, :3] = torch.tensor([0, 6, 7])  # token 0 counts
        rows[1, :1] = torch.tensor([9])
        rows[2, :] = 1  # a padding row past the batch
        result = HostRowsResult(rows, 2, speculative_output_stride=WIDTH)
        self.assertIsNone(result.accept_lens)
        self.assertIs(result.next_token_ids, rows)
        self.assertTrue(result.has_sampled_token_ids)
        result.copy_done = _Event()
        result.copy_to_cpu(return_logprob=False)
        self.assertEqual(result.accept_lens.tolist(), [3, 1])
        self.assertEqual(result.accept_lens.dtype, torch.int32)
        self.assertEqual(
            result.next_token_ids.tolist(), [0, 6, 7] + [-1] * 5 + [9] + [-1] * 7
        )
        self.assertEqual(accept_counts(np.array([[1, -1], [-1, -1]])).tolist(), [1, 0])
        with self.assertRaisesRegex(NotImplementedError, "tokens only"):
            result.copy_to_cpu(return_logprob=True)

    def test_device_ops_per_round(self):
        """The device relay's fourteen ops a steady decode step with the
        switch off, none with it on; arrivals and departures keep only
        sglang's batch filtering and merging ops."""
        run_off, off = _run(False)
        run_on, on = _run(True)
        # Neither reads a device value on the host (no sync).
        self.assertEqual((run_off.ops.syncs, run_on.ops.syncs), (0, 0))
        self.assertEqual([kind for kind, _ in off], [kind for kind, _ in on])
        steady = [i for i, (kind, _) in enumerate(off) if kind == "steady"]
        self.assertGreaterEqual(len(steady), 7)
        relay = RESOLVE_OPS + DEVICE_RELAY_OPS
        for i in steady:
            self.assertEqual(off[i][1], relay, i)
            self.assertEqual(on[i][1], [], i)
        for (kind, ops_off), (_, ops_on) in zip(off, on):
            # Batch filtering and merging index and concatenate device rows
            # (sglang's, either way); nothing else runs with the switch on.
            self.assertTrue(set(ops_on) <= {"index", "cat"}, (kind, ops_on))
            relayed = list(ops_off)
            for op in ops_on:
                relayed.remove(op)
            expected = {
                "prefill": ["index_copy_", "index_copy_"],
                "steady": relay,
                "decode": relay,
                "idle": [],
            }[kind]
            self.assertEqual(sorted(relayed), sorted(expected), kind)

    def test_same_results_with_and_without(self):
        """Tokens, committed and allocated lengths, finish decisions, the
        stream and what the program saw, round by round."""
        off, _ = _run(False)
        on, _ = _run(True)
        self.assertEqual(off.trace, on.trace)
        self.assertEqual(off.recorder.finishes, on.recorder.finishes)
        self.assertEqual(off.recorder.streamed, on.recorder.streamed)
        self.assertEqual(off.alloc, on.alloc)
        self.assertEqual(off.decode.inputs, on.decode.inputs)
        # The script did what it says.
        self.assertEqual(
            sorted(on.recorder.finishes),
            [
                ("a", "FINISH_MATCHED_TOKEN", 31),
                ("b", "FINISH_LENGTH", 13),
                ("c", "FINISH_MATCHED_TOKEN", 9),
                ("d", "FINISH_LENGTH", 20),
            ],
        )
        final = {rid: out for rid, out, *_ in on.trace[-1]}
        self.assertEqual((final["a"][30], final["c"][8]), (STOP, STOP))
        self.assertGreater(len(final["b"]), 13)  # accepted past its cap
        # Each departure reached the program as finished with the next step
        # ("d" left last: no step followed).
        self.assertEqual(sorted(sum((f for f, _ in on.decode.inputs), [])), list("abc"))
        # "c" and "d" joined running batches.
        batches = [[rid for rid, *_ in reqs] for _, reqs in on.decode.inputs]
        self.assertIn(["a", "c"], batches)
        self.assertIn(["a", "c", "d"], batches)


if __name__ == "__main__":
    unittest.main()
