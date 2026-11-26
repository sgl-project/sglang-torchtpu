"""Finite distributed PD scheduler workload shared by CPU and TPU checks.

Scheduling, batch updates, request rows, KV allocation and retirement use the
real runtime code. The test supplies control traffic and a small collective
forward instead of loading a model; it does not validate native P/D DMA.
"""

from __future__ import annotations

from array import array
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.exit_ledger import record_send
from sglang.srt.managers.scheduler_components.health_monitor import (
    SchedulerHealthMonitor,
)
from sglang.srt.managers.scheduler_components.health_probe import stamp_probe_admission
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import published_topology

PROBE_RID = f"{HEALTH_CHECK_RID_PREFIX}_distributed"


class PDHealthProbeTestScheduler(Scheduler):
    """Keep production scheduling; supply finite tensor I/O and forward work."""

    def __init__(self, overlap, scenario, *, cpu_group=None, forward=None):
        self.cpu_group = cpu_group
        self.forward = forward
        self.enable_overlap = overlap
        self.scenario = scenario
        self.step = -1
        self.gracefully_exit = False
        self.pause_state = SimpleNamespace(paused=False)
        self.invariant_checker = SimpleNamespace(run=lambda *args: None)
        self.enable_skip_finishing_decode = False
        self.uocr_runtime = None
        self.scheduler_stage_metrics = None
        self.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        self.enable_hisparse = False
        self.enable_priority_scheduling = False
        self.max_running_requests = 1
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.use_overlap_streams = False
        self.kv_free_fence = None
        self.decode_offload_manager = None
        self.model_config = SimpleNamespace(vocab_size=32, is_encoder_decoder=False)
        self.req_to_token_pool = ReqToTokenPool(8, 64, "cpu", enable_memory_saver=False)
        kv = MHATokenToKVPool(64, 1, torch.float32, 1, 8, 1, "cpu", False)
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            64, torch.float32, "cpu", kv, need_sort=False
        )
        self.tree_cache = RadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=1,
            )
        )
        self.running_batch = self._batch([])
        self.last_batch = None
        self.future_map = FutureMap(
            torch.device("cpu"), self.spec_algorithm, self.req_to_token_pool
        )
        self.new_token_ratio_tracker = NewTokenRatioTracker.from_config()
        self.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        self.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, **kwargs: batch
        )
        self.dp_attn_adapter = SimpleNamespace(maybe_prepare_mlp_sync_batch=lambda b: b)
        self.disagg_decode_prealloc_queue = SimpleNamespace(
            prefetch_prefill_dp_rank_queries=lambda: None
        )
        self.output_streamer = SimpleNamespace(
            stream_output=self._stream_output,
            release_detokenize_state=lambda req: None,
        )
        self.batch_result_processor = SimpleNamespace(
            process_batch_result_prebuilt=lambda batch: (
                SchedulerBatchResultProcessor.process_batch_result_prebuilt(self, batch)
            )
        )
        self.disaggregation_mode = DisaggregationMode.DECODE
        self.kv_checksum_computer = None
        self.health_monitor = SchedulerHealthMonitor(self._ack, dp_rank=0)
        self.waiting_queue = IndexedReqList(ListKind.WAITING)
        self.running_reqs = KeyedReqList(ListKind.DECODING)
        self.rust_server = None
        self.final_outputs = []
        self.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda output, req: self.final_outputs.append(req.rid)
            )
        )
        self.request_retirer = RequestRetirer(
            tree_cache=self.tree_cache,
            beam_coordinator=SimpleNamespace(retire_group=lambda req: None),
            hisparse_coordinator=None,
            uocr_runtime=None,
            send_abort_output=self.send_abort_output,
            stream_output=self.output_streamer.stream_output,
        )
        self.pending = []
        self.acks = []
        self.forward_batches = []
        self.probe_finished_step = None
        self.requests = []
        self.baseline = self._capacity()
        self.probe = self._req(PROBE_RID, 2)
        self.health_monitor.register(
            SimpleNamespace(rid=PROBE_RID, http_worker_ipc=None), admitted=True
        )
        if scenario == "ack_during_transfer":
            self.pending.append((4, self.probe))
            move([self._req("user-running", 2)], self.waiting_queue)
        elif scenario == "one_token_backlog":
            move(
                [self.probe, *[self._req(f"user-{i}", 1) for i in range(3)]],
                self.waiting_queue,
            )
        else:
            move([self.probe], self.waiting_queue)

    def _batch(self, reqs):
        return ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

    def _capacity(self):
        return (
            self.req_to_token_pool.available_size(),
            self.token_to_kv_pool_allocator.available_size(),
        )

    def _req(self, rid, max_tokens):
        params = SamplingParams(max_new_tokens=max_tokens, ignore_eos=True)
        params.normalize(None)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", [0, 0]),
            sampling_params=params,
        )
        req.output_ids.append(1)
        self.req_to_token_pool.alloc([req])
        indices = self.token_to_kv_pool_allocator.alloc(2)
        self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :2] = indices
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 2
        req.last_node = self.tree_cache.root_node
        req.prefix_len = 0
        req.init_next_round_input(None)
        stamp_probe_admission(req, self.health_monitor.completion_seq)
        self.requests.append(req)
        return req

    def _ack(self, output):
        # The transfer-held probe must be acknowledged before its destination
        # becomes eligible for retirement, never by a PREBUILT-only pass.
        self.acks.append((self.step, output.rid, output.completion_seq))

    def ingest_requests(self):
        self.step += 1
        state = (
            "iteration",
            self.step,
            self.health_monitor.completion_seq,
            tuple(r.rid for r in self.waiting_queue),
            self._capacity(),
        )
        peers = [None] * dist.get_world_size()
        dist.all_gather_object(peers, state, group=self.cpu_group)
        assert all(peer == state for peer in peers), peers
        assert self.step < 30, "PD decode loop did not drain"
        if self.scenario == "finished_probe_waiting_users" and self.step == 1:
            move([self._req("user-after-probe", 2)], self.waiting_queue)
        drained = (
            not self.waiting_queue
            and not self.pending
            and self.running_batch.is_empty()
        )
        if drained and (not self.enable_overlap or not self.result_queue):
            self.gracefully_exit = True

    def process_decode_queue(self, defer_prealloc=False):
        remaining = []
        for arrival, req in self.pending:
            if self.step < arrival:
                remaining.append((arrival, req))
                continue
            # Real CPU transfer on the ranks' process group; no separate clocks
            # or mocked KVPoll result determines arrival.
            payload = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
            dist.broadcast(payload, src=0, group=self.cpu_group)
            indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, :2
            ].long()
            kv = self.token_to_kv_pool_allocator.get_kvcache()
            kv.k_buffer[0][indices] = payload
            kv.v_buffer[0][indices] = payload
            move([req], self.waiting_queue)
        self.pending = remaining
        return False

    def run_batch(self, batch):
        rids = tuple(req.rid for req in batch.reqs)
        peers = [None] * dist.get_world_size()
        dist.all_gather_object(
            peers, ("forward", self.step, rids), group=self.cpu_group
        )
        assert all(peer == ("forward", self.step, rids) for peer in peers), peers
        assert not (PROBE_RID in rids and len(rids) > 1), rids
        self.forward_batches.append((self.step, rids))
        # The collective really runs, and retirement/acks consume its result.
        if self.forward is not None:
            return self.forward(len(rids))
        result = torch.full((len(rids), 8), float(dist.get_rank() + 1))
        work = dist.all_reduce(result, async_op=True)
        if not self.enable_overlap:
            work.wait()
        return work, result

    def process_batch_result(self, batch, result):
        work, values = result
        work.wait()
        world = dist.get_world_size()
        expected = world * (world + 1) // 2
        assert torch.all(values.cpu() == expected), values
        for req in batch.reqs:
            if req.finished():
                continue  # A redundant overlapped forward owns no second release.
            req.output_ids.append(1)
            req.update_finish_state()
            if req.finished():
                self.request_retirer.release(req, checkpoint=False)
                self._stream_output([req], False)
                if req.rid == PROBE_RID:
                    self.probe_finished_step = self.step
        self.health_monitor.on_batch_completed(batch)

    def _stream_output(self, reqs, *_):
        for req in reqs:
            if req.finished():
                record_send(req, "stream")
                self.final_outputs.append(req.rid)

    def should_process_last_batch_early(self, last_batch):
        return False

    def _flush_staging_scatters(self):
        return None

    def _apply_war_barrier(self):
        return None

    def launch_batch_sample_if_needed(self, result, batch):
        return None

    def on_idle(self):
        return None

    def verify(self):
        assert self._capacity() == self.baseline, (self._capacity(), self.baseline)
        assert all(
            req.kv.req_pool_idx is None and req.kv.is_kv_released
            for req in self.requests
        )
        assert not self.waiting_queue and not self.running_reqs
        assert all(
            req.owner_list is None and req.exit_via and req.send_via
            for req in self.requests
        )
        assert sorted(self.final_outputs) == sorted(req.rid for req in self.requests)
        assert self.acks and all(rid == PROBE_RID for _, rid, _ in self.acks)
        if self.scenario == "ack_during_transfer":
            assert self.acks[0][0] < 4, self.acks
            assert all(PROBE_RID not in rids for _, rids in self.forward_batches)
        else:
            assert any(rids == (PROBE_RID,) for _, rids in self.forward_batches)
        if self.scenario == "one_token_backlog":
            assert self.forward_batches[0][1] == (PROBE_RID,), self.forward_batches
        if self.scenario == "finished_probe_waiting_users":
            first_user = next(
                step
                for step, rids in self.forward_batches
                if "user-after-probe" in rids
            )
            assert first_user == self.probe_finished_step + 1, self.forward_batches


def run_pd_health_probe_scenarios(*, cpu_group=None, forward=None):
    rank = dist.get_rank()
    for overlap in (False, True):
        for scenario in (
            "ack_during_transfer",
            "one_token_backlog",
            "finished_probe_waiting_users",
        ):
            with (
                envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True),
                published_topology(
                    role="scheduler",
                    ranks={"world_rank": rank},
                    tp_size=dist.get_world_size(),
                    device="cpu",
                    disaggregation_mode="decode",
                    disaggregation_decode_enable_radix_cache=False,
                    disaggregation_decode_host_receive_threshold=1,
                    disable_overlap_schedule=not overlap,
                ),
            ):
                scheduler = PDHealthProbeTestScheduler(
                    overlap, scenario, cpu_group=cpu_group, forward=forward
                )
                if overlap:
                    scheduler.event_loop_overlap_disagg_decode()
                else:
                    scheduler.event_loop_normal_disagg_decode()
                scheduler.verify()
                # Gloo rendezvous must not initialize the default accelerator.
                dist.monitored_barrier(group=cpu_group, timeout=timedelta(seconds=45))
