"""Regression tests for the SWA chunked-req stash gate (#24252)."""

import unittest
from array import array
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


def _make_req(
    *,
    req_pool_idx: int,
    fill_ids: list,
    prefix_indices: torch.Tensor,
    extend_input_len: int,
    fill_len: int,
) -> Req:
    req = Req.__new__(Req)
    req.rid = "test-req"
    req.origin_input_ids = array("q", fill_ids)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill_ids)
    req.prefix_indices = prefix_indices
    req.extend_range = Range(fill_len - extend_input_len, fill_len)
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
    req.skip_radix_cache_insert = False
    req.finished_reason = None
    req.last_node = None
    req.lock_receipt = DecLockRefParams()
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.cache_salt = None
    req.kv.mamba_pool_idx = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_req_to_token_pool(num_slots: int, max_context: int) -> SimpleNamespace:
    # Slot s contains a recognizable fingerprint [s*1000, s*1000+1, ...]
    # so we can tell a corrupted prefix_indices from a healthy one by content.
    pool = SimpleNamespace()
    pool.req_to_token = (
        torch.arange(max_context, dtype=torch.int32).unsqueeze(0).repeat(num_slots, 1)
        + torch.arange(num_slots, dtype=torch.int32).unsqueeze(1) * 1000
    )
    return pool


def _make_chunk_cache(req_to_token_pool) -> ChunkCache:
    return ChunkCache(
        SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
    )


def _scheduler_for_get_next_batch(*, tree_cache, chunked_req) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.scheduler_stage_metrics = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    # Exercise the unconditional scheduler-loop HiCache event-drain point.
    s.enable_hierarchical_cache = True
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.last_batch = None
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.server_args = MagicMock(speculative_skip_dp_mlp_sync=True)
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.is_prefill_only = False
    s.running_batch.batch_is_full = False
    s.running_batch.reqs = []
    s.prefill_decode_interval = 0
    s._prefill_decode_interval_remaining = 0
    s.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=None, running_batch=s.running_batch)
    )
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.ngram_embedding_manager = MagicMock()
    s.ngram_embedding_manager.prepare_for_forward = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    tree_cache.check_hicache_events = MagicMock()
    s.tree_cache = tree_cache
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    s._deferred_chunked_sender_aborts = []
    return s


class TestStashGatePreservesPrefixIndices(CustomTestCase):
    """Consumer side: real ChunkCache.checkpoint mutates
    req.prefix_indices iff stash actually runs, so prefix_indices content
    is the bug-detection signal. The stash gate is content-based:
    `fill_len > len(prefix_indices)` means there is freshly computed KV to
    cache; otherwise the chunk was parked and stashing must be skipped."""

    POOL_IDX = 4
    INITIAL_PREFIX_LEN = 8  # what was really cached last iter
    POST_RESET_FILL_LEN = 32  # length after init_next_round_input rebuilds
    NUM_SLOTS = 8
    MAX_CONTEXT = 64

    def _build(self, *, fill_len: int):
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        initial_prefix = pool.req_to_token[self.POOL_IDX, : self.INITIAL_PREFIX_LEN].to(
            dtype=torch.int64, copy=True
        )
        req = _make_req(
            req_pool_idx=self.POOL_IDX,
            fill_ids=list(range(self.POST_RESET_FILL_LEN)),
            prefix_indices=initial_prefix,
            extend_input_len=fill_len - self.INITIAL_PREFIX_LEN,
            fill_len=fill_len,
        )
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=req)
        return s, req, initial_prefix, pool

    def test_parked_chunked_req_keeps_real_prefix_indices(self):
        # A parked chunk has fill_len == len(prefix_indices): no new KV was
        # computed, so the gate must skip stash and leave prefix_indices intact.
        s, req, initial_prefix, _ = self._build(fill_len=self.INITIAL_PREFIX_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        self.assertEqual(req.prefix_indices.shape[0], self.INITIAL_PREFIX_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, initial_prefix))

    def test_scheduled_chunked_req_advances_prefix_indices_via_real_stash(self):
        # Symmetric guard against over-gating: when fill_len has advanced past
        # the cached prefix, stash must run and advance prefix_indices.
        s, req, _, pool = self._build(fill_len=self.POST_RESET_FILL_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        expected = pool.req_to_token[self.POOL_IDX, : self.POST_RESET_FILL_LEN].to(
            dtype=torch.int64
        )
        self.assertEqual(req.prefix_indices.shape[0], self.POST_RESET_FILL_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, expected))

    def test_no_chunked_req_never_mutates_state(self):
        # The outer `if chunked_req is not None` guard must hold on the retract
        # path that clears chunked_req.
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=None)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )
        self.assertIsNone(s.chunked_req)


def _run_chunked_abort_rank(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        tp_groups = [
            dist.new_group(ranks, timeout=timedelta(seconds=30))
            for ranks in ([0, 1], [2, 3])
        ]
        cp_groups = [
            dist.new_group(ranks, timeout=timedelta(seconds=30))
            for ranks in ([0, 2], [1, 3])
        ]
        tp_group, cp_group = tp_groups[rank // 2], cp_groups[rank % 2]
        sender = MagicMock(group_failure_requires_drain=True)
        sender.drained.return_value = rank != 0

        def drained(sender):
            assert sender.abort.call_count == 1
            return sender.drained.return_value

        sender.drained.side_effect = lambda: drained(sender)
        req = TestChunkedAbortNativeReadDrain._aborted_chunked_req(sender)
        scheduler = TestChunkedAbortNativeReadDrain._scheduler(req)
        scheduler.attn_tp_cpu_group = tp_group
        scheduler.attn_cp_cpu_group = cp_group

        def release(owner, tree_cache, *, is_insert):
            assert tree_cache is scheduler.tree_cache
            assert not is_insert and owner.kv.holds_kv
            owner.kv.holds_kv = False

        def assert_all_ranks(state, expected):
            tensor = torch.tensor(state, dtype=torch.int64)
            states = [torch.empty_like(tensor) for _ in range(4)]
            dist.all_gather(states, tensor)
            assert [s.tolist() for s in states] == [expected] * 4

        native_all_reduce = dist.all_reduce
        votes = []

        def all_reduce(tensor, op, group):
            assert op == dist.ReduceOp.MIN
            assert group is tp_group or group is cp_group
            votes.append(group)
            native_all_reduce(tensor, op=op, group=group)

        with (
            patch(
                "sglang.srt.managers.scheduler.current_platform",
                SimpleNamespace(is_tpu=lambda: True),
            ),
            patch("torch.distributed.all_reduce", side_effect=all_reduce),
            patch(
                "sglang.srt.managers.scheduler.release_kv_cache", side_effect=release
            ) as free,
            get_context().override_server_args(),
        ):
            scheduler.process_pending_chunked_abort()
            assert votes == [tp_group, cp_group]
            assert_all_ranks(
                [
                    len(scheduler._deferred_chunked_sender_aborts),
                    req.kv.holds_kv,
                    sender.abort.call_count,
                    free.call_count,
                ],
                [1, 1, 1, 0],
            )

            second_sender = MagicMock(group_failure_requires_drain=True)
            second_sender.drained.return_value = rank != 2
            second_sender.drained.side_effect = lambda: drained(second_sender)
            second_req = TestChunkedAbortNativeReadDrain._aborted_chunked_req(
                second_sender, rid="second-abort"
            )
            scheduler.chunked_req = second_req
            scheduler._pending_chunked_abort_req = second_req
            votes.clear()
            scheduler.process_pending_chunked_abort()
            assert votes == [tp_group, cp_group] * 2
            assert_all_ranks(
                [
                    len(scheduler._deferred_chunked_sender_aborts),
                    req.kv.holds_kv,
                    second_req.kv.holds_kv,
                    sender.abort.call_count,
                    second_sender.abort.call_count,
                    free.call_count,
                ],
                [2, 1, 1, 1, 1, 0],
            )

            # Only the second request has drained on every rank. The first
            # still has a reader on rank 0, across both TP and CP subgroups.
            second_sender.drained.return_value = True
            votes.clear()
            scheduler._drain_deferred_chunked_sender_aborts()
            assert votes == [tp_group, cp_group]
            assert_all_ranks(
                [
                    len(scheduler._deferred_chunked_sender_aborts),
                    req.kv.holds_kv,
                    second_req.kv.holds_kv,
                    sender.clear.call_count,
                    second_sender.clear.call_count,
                    free.call_count,
                ],
                [1, 1, 0, 0, 1, 1],
            )

            sender.drained.return_value = True
            votes.clear()
            scheduler._drain_deferred_chunked_sender_aborts()
            assert votes == [tp_group, cp_group]
            assert_all_ranks(
                [
                    len(scheduler._deferred_chunked_sender_aborts),
                    req.kv.holds_kv,
                    second_req.kv.holds_kv,
                    sender.clear.call_count,
                    second_sender.clear.call_count,
                    free.call_count,
                ],
                [0, 0, 0, 1, 1, 2],
            )
    finally:
        dist.destroy_process_group()


class TestChunkedAbortNativeReadDrain(CustomTestCase):
    @staticmethod
    def _scheduler(req):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = req
        scheduler._pending_chunked_abort_req = req
        scheduler._deferred_chunked_sender_aborts = []
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = MagicMock()
        scheduler.ipc_channels = MagicMock()
        scheduler.attn_tp_cpu_group = None
        scheduler.attn_cp_cpu_group = None
        return scheduler

    @staticmethod
    def _aborted_chunked_req(sender, rid="chunked-abort"):
        return SimpleNamespace(
            rid=rid,
            disagg_kv_sender=sender,
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
            to_finish=object(),
            return_logprob=False,
            metadata_buffer_index=7,
            pending_bootstrap=True,
            weight_version_events=[],
            output_ids=array("q"),
            cache_request_handle=object(),
            kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
        )

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    @patch("sglang.srt.managers.scheduler.maybe_release_metadata_buffer")
    @patch("sglang.srt.managers.scheduler.prepare_abort")
    def test_abort_defers_page_release_while_sender_is_not_drained(
        self, _prepare_abort, _release_metadata, release_kv
    ):
        sender = MagicMock(group_failure_requires_drain=True)
        sender.drained.return_value = False
        req = self._aborted_chunked_req(sender)
        scheduler = self._scheduler(req)

        # The same notice reads the live weight version off the serving bag.
        with get_context().override_server_args():
            scheduler.process_pending_chunked_abort()

        sender.abort.assert_called_once_with()
        release_kv.assert_not_called()
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [req])

    def test_non_tpu_abort_and_drain_do_not_use_collectives(self):
        sender = MagicMock(group_failure_requires_drain=True)
        sender.drained.return_value = False
        req = self._aborted_chunked_req(sender)
        scheduler = self._scheduler(req)
        scheduler.attn_tp_cpu_group = object()
        scheduler.attn_cp_cpu_group = object()

        with (
            patch(
                "sglang.srt.managers.scheduler.current_platform",
                SimpleNamespace(is_tpu=lambda: False),
            ),
            patch("torch.distributed.get_world_size", return_value=2) as world_size,
            patch("torch.distributed.all_reduce") as all_reduce,
            patch("sglang.srt.managers.scheduler.release_kv_cache") as free,
            get_context().override_server_args(),
        ):
            scheduler.process_pending_chunked_abort()
            free.assert_not_called()
            self.assertEqual(scheduler._deferred_chunked_sender_aborts, [req])
            sender.drained.return_value = True
            scheduler.process_pending_chunked_abort()

        sender.abort.assert_called_once_with()
        sender.poll.assert_called_once_with()
        sender.clear.assert_called_once_with()
        free.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [])
        world_size.assert_not_called()
        all_reduce.assert_not_called()

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_real_tp_cp_groups_release_requests_in_the_same_step(self):
        with TemporaryDirectory() as directory:
            workers = mp.spawn(
                _run_chunked_abort_rank,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=4,
                join=False,
            )
            deadline = monotonic() + 120
            try:
                while not workers.join(timeout=5):
                    if monotonic() >= deadline:
                        self.fail("chunked-abort drain test exceeded 120 seconds")
            finally:
                for process in workers.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    def _idle_prefill_scheduler(self):
        # is_fully_idle reads the pipeline width from the published config.
        enter_scope(self, published_topology())
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = MagicMock()
        scheduler.running_batch = SimpleNamespace(
            reqs=[], is_empty=lambda: True, batch_is_full=False
        )
        scheduler.chunked_req = None
        scheduler._pending_chunked_abort_req = None
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler.gracefully_exit = False
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[], clear=MagicMock())
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=[],
            pop_bootstrapped=MagicMock(return_value=[]),
            decode_prefix_puller=None,
        )
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False
        scheduler._engine_paused = False
        scheduler._deferred_chunked_sender_aborts = []
        scheduler.attn_tp_cpu_group = None
        scheduler.attn_cp_cpu_group = None
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.draft_worker = None
        return scheduler

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_sender_drain_keeps_prefill_non_idle_until_pages_release(self, release_kv):
        sender = MagicMock()
        sender.drained.side_effect = [False, True]
        req = SimpleNamespace(disagg_kv_sender=sender)
        ready_sender = MagicMock()
        ready_sender.drained.return_value = True
        ready_req = SimpleNamespace(disagg_kv_sender=ready_sender)
        scheduler = self._idle_prefill_scheduler()
        self.assertTrue(scheduler.is_fully_idle())

        scheduler._deferred_chunked_sender_aborts.extend([req, ready_req])
        scheduler._drain_deferred_chunked_sender_aborts()
        self.assertEqual(sender.mock_calls[:2], [call.poll(), call.drained()])
        sender.poll.assert_called_once_with()
        sender.clear.assert_not_called()
        ready_sender.clear.assert_called_once_with()
        release_kv.assert_called_once_with(
            ready_req, scheduler.tree_cache, is_insert=False
        )
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [req])
        self.assertFalse(scheduler.is_fully_idle())
        self.assertTrue(scheduler.is_fully_idle(for_health_check=True))
        self.assertFalse(scheduler.flush_cache(empty_cache=False))
        scheduler.tree_cache.reset.assert_not_called()
        scheduler.token_to_kv_pool_allocator.clear.assert_not_called()

        scheduler._drain_deferred_chunked_sender_aborts()
        self.assertEqual(sender.poll.call_count, 2)
        sender.clear.assert_called_once_with()
        self.assertEqual(
            release_kv.call_args_list,
            [
                call(ready_req, scheduler.tree_cache, is_insert=False),
                call(req, scheduler.tree_cache, is_insert=False),
            ],
        )
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [])
        self.assertTrue(scheduler.is_fully_idle())
        self.assertTrue(scheduler.flush_cache(empty_cache=False))
        scheduler.tree_cache.reset.assert_called_once_with()
        scheduler.token_to_kv_pool_allocator.clear.assert_called_once_with()

    def test_prefill_event_loops_poll_deferred_aborts(self):
        for loop_name in (
            "event_loop_normal_disagg_prefill",
            "event_loop_overlap_disagg_prefill",
        ):
            with self.subTest(loop=loop_name):
                self._check_prefill_loop_drain(loop_name)

    def _check_prefill_loop_drain(self, loop_name):
        class StopLoop(Exception):
            pass

        scheduler = self._idle_prefill_scheduler()
        scheduler.scheduler_stage_metrics = None
        scheduler.enable_overlap = "overlap" in loop_name
        scheduler.result_queue = []
        scheduler._process_hicache_events = MagicMock()
        scheduler.resolve_waiting_queue_bootstrap = MagicMock()
        scheduler.process_prefill_chunk = MagicMock()
        scheduler.get_new_batch_prefill = MagicMock(
            return_value=NextBatchPlan(None, scheduler.running_batch)
        )
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )
        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, **_: batch
        )
        scheduler.process_disagg_prefill_inflight_queue = MagicMock()
        scheduler._deferred_input_requests = []
        scheduler.launch_batch_sample_if_needed = MagicMock()
        scheduler.on_idle = MagicMock()
        scheduler.run_batch = MagicMock()
        scheduler.process_batch_result = MagicMock()
        idle_states = []

        def ingest(**kwargs):
            idle_states.append(scheduler.is_fully_idle())
            if len(idle_states) == 4:
                raise StopLoop

        scheduler.ingest_requests = MagicMock(side_effect=ingest)
        sender = MagicMock()
        sender.drained.side_effect = [False, True]
        req = SimpleNamespace(disagg_kv_sender=sender)
        scheduler._deferred_chunked_sender_aborts = [req]

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release_kv:
            with self.assertRaises(StopLoop):
                getattr(scheduler, loop_name)()

        self.assertEqual(idle_states, [False, False, True, True])
        self.assertEqual(sender.poll.call_count, 2)
        sender.clear.assert_called_once_with()
        release_kv.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [])
        scheduler.run_batch.assert_not_called()
        scheduler.process_batch_result.assert_not_called()


class TestDynamicFirstChunkCap(CustomTestCase):
    def _scheduler(self, predicted: int = 1024):
        s = Scheduler.__new__(Scheduler)
        s.dynamic_chunk_sizer = MagicMock(predict=MagicMock(return_value=predicted))
        return s

    def _req(self, *, prefix_len: int, host_hit_length: int = 0):
        return SimpleNamespace(
            prefix_indices=[0] * prefix_len, host_hit_length=host_hit_length
        )

    def test_no_cap_without_matched_prefix(self):
        # A cold request keeps the static chunk; only a prefix adds attention cost.
        s = self._scheduler()
        self.assertIsNone(s._dynamic_first_chunk_cap(self._req(prefix_len=0)))
        s.dynamic_chunk_sizer.predict.assert_not_called()

    def test_cap_sized_by_prefix_and_host_hit(self):
        # A host-tier hit is loaded back, not recomputed, but it is still KV
        # history the first chunk attends over.
        s = self._scheduler(predicted=1024)
        cap = s._dynamic_first_chunk_cap(
            self._req(prefix_len=4096, host_hit_length=512)
        )
        self.assertEqual(cap, 1024)
        s.dynamic_chunk_sizer.predict.assert_called_once_with(4608)


if __name__ == "__main__":
    unittest.main()
