"""Regression tests for the SWA chunked-req stash gate (#24252)."""

import unittest
from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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
    req.req_pool_idx = req_pool_idx
    req.extend_range = Range(fill_len - extend_input_len, fill_len)
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.cache_protected_len = 0
    req.skip_radix_cache_insert = False
    req.last_node = None
    req.swa_uuid_for_lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.cache_salt = None
    req.mamba_pool_idx = None
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
    s._abort_on_waiting_timeout = MagicMock()
    s._abort_on_running_timeout = MagicMock()
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    s.last_batch = None
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.server_args = MagicMock(speculative_skip_dp_mlp_sync=True)
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.is_prefill_only = False
    s.running_batch.batch_is_full = False
    s.running_batch.reqs = []
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
    s.tree_cache = tree_cache
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    return s


class TestStashGatePreservesPrefixIndices(CustomTestCase):
    """Consumer side: real ChunkCache.cache_unfinished_req mutates
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


class TestChunkedAbortNativeReadDrain(CustomTestCase):
    def _scheduler(self, req):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = req
        scheduler._pending_chunked_abort_req = req
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = MagicMock()
        scheduler.ipc_channels = MagicMock()
        return scheduler

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    @patch("sglang.srt.managers.scheduler.maybe_release_metadata_buffer")
    @patch("sglang.srt.managers.scheduler.prepare_abort")
    def test_abort_defers_page_release_while_sender_is_not_drained(
        self, _prepare_abort, _release_metadata, release_kv
    ):
        sender = MagicMock(group_failure_requires_drain=True)
        sender.drained.return_value = False
        req = SimpleNamespace(
            rid="chunked-abort",
            disagg_kv_sender=sender,
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
            to_finish=object(),
            pending_bootstrap=True,
        )
        scheduler = self._scheduler(req)

        scheduler.process_pending_chunked_abort()

        sender.abort.assert_called_once_with()
        release_kv.assert_not_called()
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [req])

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_deferred_pages_release_only_after_sender_drains(self, release_kv):
        sender = MagicMock()
        sender.drained.side_effect = [False, True]
        req = SimpleNamespace(disagg_kv_sender=sender)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = MagicMock()
        scheduler._deferred_chunked_sender_aborts = [req]

        scheduler._drain_deferred_chunked_sender_aborts()
        release_kv.assert_not_called()
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [req])

        scheduler._drain_deferred_chunked_sender_aborts()
        sender.clear.assert_called_once_with()
        release_kv.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertEqual(scheduler._deferred_chunked_sender_aborts, [])


class TestChunkedReqAdmissionAccounting(CustomTestCase):
    """Admission gates in ``_get_new_batch_prefill_raw``.

    A request that already owns a ``req_pool_idx`` -- a continuing chunked
    prefill -- must not be charged against ``available_size()``:
    ``ReqToTokenPool.alloc`` leaves it out of ``need_size`` and its row is not
    in ``free_slots``, so charging it costs one admission per pass. It must
    still be charged against ``pp_max_micro_batch_size``, which bounds every
    member of the micro-batch and does not see the chunk through ``running_bs``
    (``self.chunked_req`` is deliberately excluded from ``running_batch``).
    """

    NO_CHUNK = "none"
    CHUNK_CONTINUES = "continues"  # add_chunked_req returns the req
    CHUNK_LAST = "last"  # final chunk: returns None, req is still in the batch

    def _admit(
        self,
        *,
        chunk,
        available_size,
        pp_max_micro_batch_size=1024,
        running_bs=0,
        disaggregation_mode=DisaggregationMode.NULL,
        num_waiting=64,
    ):
        """Run one prefill pass and report how many new requests were admitted."""
        adder = MagicMock()
        adder.can_run_list = []
        adder.preempt_list = []
        adder.new_chunked_req = None
        # Use the real predicate rather than restating it: this is the accounting
        # under test, and it must agree with ReqToTokenPool.alloc's need_size.
        adder.num_slot_reusing_reqs = MethodType(
            PrefillAdder.num_slot_reusing_reqs, adder
        )

        def add_chunked_req(req):
            adder.can_run_list.append(req)
            return None if chunk == self.CHUNK_LAST else req

        def add_new_req(req, **_):
            adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        adder.add_chunked_req.side_effect = add_chunked_req
        adder.add_one_req.side_effect = add_new_req

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.min_free_slots_delayer = None
        scheduler.policy = MagicMock()
        scheduler.chunked_prefill_size = 2048
        scheduler.enable_dynamic_chunking = False
        scheduler.tp_worker = MagicMock()
        scheduler.tp_worker.model_runner.attn_backend = SimpleNamespace()
        scheduler.tp_worker.model_runner.prefill_aware_swa = False
        scheduler.page_size = 1
        scheduler.tree_cache = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        # Real available_size() so get_num_allocatable_reqs is exercised, not mocked.
        scheduler.req_to_token_pool = SimpleNamespace(
            mamba_allocator=None,
            available_size=lambda: available_size,
        )
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.5)
        scheduler.max_prefill_tokens = 2048
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 0
        scheduler.max_running_requests = 1024
        scheduler.prefill_delayer = None
        scheduler.dllm_config = None
        scheduler.enable_lora = False
        scheduler.disaggregation_mode = disaggregation_mode
        scheduler.enable_hicache_storage = False
        scheduler.truncation_align_size = None
        scheduler.model_config = MagicMock()
        scheduler.enable_overlap = True
        scheduler.spec_algorithm = MagicMock()
        scheduler.enable_priority_scheduling = False
        scheduler.server_args = MagicMock()
        scheduler.load_inquirer = MagicMock()
        scheduler.load_inquirer._get_num_pending_tokens.return_value = 0

        running_batch = MagicMock()
        running_batch.reqs = [MagicMock() for _ in range(running_bs)]
        running_batch.batch_is_full = False
        running_batch.is_empty.return_value = running_bs == 0

        if chunk == self.NO_CHUNK:
            scheduler.chunked_req = None
        else:
            chunked_req = MagicMock()
            # The row this request already owns -- the whole point of the gate.
            chunked_req.req_pool_idx = 3
            chunked_req.inflight_middle_chunks = 0
            chunked_req.extend_range = SimpleNamespace(length=128)
            scheduler.chunked_req = chunked_req

        waiting_reqs = []
        for _ in range(num_waiting):
            req = MagicMock()
            req.req_pool_idx = None  # a fresh request draws a new row
            waiting_reqs.append(req)
        scheduler.waiting_queue = list(waiting_reqs)

        with (
            patch("sglang.srt.managers.scheduler.PrefillAdder", return_value=adder),
            patch("sglang.srt.managers.scheduler.get_memory") as get_memory,
            patch("sglang.srt.managers.scheduler.get_schedule") as get_schedule,
            patch("sglang.srt.managers.scheduler.get_parallel") as get_parallel,
            patch(
                "sglang.srt.managers.scheduler.ScheduleBatch.init_new",
                return_value=MagicMock(),
            ),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
            patch(
                "sglang.srt.managers.scheduler.PrefillStats.from_adder",
                return_value=MagicMock(),
            ),
        ):
            get_memory.return_value.enable_flexkv = False
            get_schedule.return_value.prefill_max_requests = None
            get_parallel.return_value.pp_max_micro_batch_size = pp_max_micro_batch_size
            Scheduler._get_new_batch_prefill_raw(
                scheduler,
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        return adder.add_one_req.call_count, running_batch, adder

    def test_continuing_chunk_does_not_consume_a_new_request_slot(self):
        # The chunk holds row 3 already, so all 15 free rows stay available.
        admitted, running_batch, adder = self._admit(
            chunk=self.CHUNK_CONTINUES, available_size=15
        )
        self.assertEqual(admitted, 15)
        self.assertIs(adder.can_run_list[0], adder.add_chunked_req.call_args[0][0])
        self.assertTrue(running_batch.batch_is_full)

    def test_final_chunk_still_holds_its_row(self):
        # add_chunked_req returns None on the last chunk, clearing
        # self.chunked_req, but the request is in can_run_list and still owns
        # its row -- accounting must key off req_pool_idx, not chunked_req.
        admitted, _, _ = self._admit(chunk=self.CHUNK_LAST, available_size=15)
        self.assertEqual(admitted, 15)

    def test_no_chunk_charges_every_admitted_request(self):
        # Nothing is reused, so the pool budget is spent entirely on new reqs.
        admitted, _, _ = self._admit(chunk=self.NO_CHUNK, available_size=15)
        self.assertEqual(admitted, 15)

    def test_disagg_prefill_gate_also_excludes_the_chunk(self):
        # The PREFILL-only gate compares against available_size() directly and
        # is strictly tighter than the general gate on a prefill node, so it
        # has to make the same exclusion or the fix is a no-op there.
        admitted, _, _ = self._admit(
            chunk=self.CHUNK_CONTINUES,
            available_size=15,
            disaggregation_mode=DisaggregationMode.PREFILL,
        )
        self.assertEqual(admitted, 15)

    def test_chunk_still_counts_against_the_micro_batch_cap(self):
        # pp_max_micro_batch_size bounds the whole micro-batch. With 7 running
        # plus the chunk, the cap of 8 is already met: admit nothing, even
        # though the request pool has room.
        admitted, running_batch, _ = self._admit(
            chunk=self.CHUNK_CONTINUES,
            available_size=1024,
            pp_max_micro_batch_size=8,
            running_bs=7,
        )
        self.assertEqual(admitted, 0)
        self.assertTrue(running_batch.batch_is_full)

    def test_micro_batch_cap_is_not_relaxed_without_a_chunk(self):
        # Same cap, no chunk: the eighth slot is genuinely free.
        admitted, _, _ = self._admit(
            chunk=self.NO_CHUNK,
            available_size=1024,
            pp_max_micro_batch_size=8,
            running_bs=7,
        )
        self.assertEqual(admitted, 1)


if __name__ == "__main__":
    unittest.main()
