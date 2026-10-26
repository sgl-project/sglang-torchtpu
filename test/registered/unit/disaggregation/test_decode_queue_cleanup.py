import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.disaggregation.fake.conn import (
    FakeKVManager,
)
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver as DisaggFakeKVReceiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.calls = []
        self.conclude_state = None

    def abort(self):
        self.calls.append("abort")

    def clear(self):
        self.calls.append("clear")
        self.clear_called = True

    def failure_exception(self):
        return None


class TestHiCacheReceiverCapabilityProxy(CustomTestCase):
    def test_group_failure_lifecycle_is_forwarded_to_inner_receiver(self):
        inner = SimpleNamespace(
            group_failure_requires_drain=True,
            quiesce_group_failure=MagicMock(),
            drained=MagicMock(return_value=False),
        )
        wrapper = HiCacheRestoreGatedKVReceiver(SimpleNamespace(kv_receiver=inner))

        self.assertTrue(wrapper.group_failure_requires_drain)
        wrapper.quiesce_group_failure()
        self.assertFalse(wrapper.drained())

        inner.quiesce_group_failure.assert_called_once_with()
        inner.drained.assert_called_once_with()

    def test_non_drain_backend_keeps_capability_disabled(self):
        wrapper = HiCacheRestoreGatedKVReceiver(
            SimpleNamespace(kv_receiver=SimpleNamespace())
        )

        self.assertFalse(wrapper.group_failure_requires_drain)

    def test_decode_poll_keeps_hicache_gate_and_reduces_tp_then_cp(self):
        tp_group, cp_group = object(), object()
        inner = SimpleNamespace(group_failure_requires_drain=True)
        decode_req = SimpleNamespace(kv_receiver=inner)
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.gloo_group = tp_group
        queue.attn_cp_cpu_group = cp_group
        queue.metadata_buffers = object()
        queue.scheduler = SimpleNamespace(
            enable_decode_hicache=True,
            server_args=object(),
        )

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Transferring],
        ) as reduce:
            queue._poll_with_metadata_gate()

        pollers = reduce.call_args.args[0]
        self.assertIsInstance(pollers[0], HiCacheRestoreGatedKVReceiver)
        self.assertIs(pollers[0].decode_req, decode_req)
        self.assertIs(reduce.call_args.args[1], tp_group)
        self.assertEqual(reduce.call_args.kwargs["additional_groups"], [cp_group])


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_prealloc_waiting_row_still_enters_tp_cp_vote(self, reduce):
        tp_group, cp_group = object(), object()
        receiver = SimpleNamespace(conclude_state=None)
        decode_req = SimpleNamespace(
            waiting_for_input=True,
            kv_receiver=receiver,
            req=SimpleNamespace(time_stats=MagicMock()),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pp_size = 1
        queue.gloo_group = tp_group
        queue.attn_cp_cpu_group = cp_group
        reduce.return_value = [KVPoll.WaitingForInput]

        queue._update_handshake_waiters()

        reduce.assert_called_once_with(
            [receiver], tp_group, additional_groups=[cp_group]
        )

    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        # resume_retracted_reqs reads the retraction backend off the disagg
        # bag, so the case publishes a config instead of injecting one.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                retraction_backup=None,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=page_size, swa_available_size=lambda: physical_available
        )
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )
        queue._allocatable_token_budgets = MagicMock(
            side_effect=lambda **_: physical_available
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(
            side_effect=lambda **_: physical_available
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)
        queue.tree_cache.evict_for_alloc.assert_not_called()

    def test_resume_reclaims_evictable_kv_before_allocating(self):
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor",
            disaggregation_decode_enable_radix_cache=True,
        )
        override.install()
        self.addCleanup(override.restore)

        for uses_swa, initial_full, initial_swa in (
            (False, 512, 0),
            (True, 512, 2560),
            (True, 8192, 512),
            (True, 512, 512),
            (False, 8192, 0),
            (True, 8192, 2560),
        ):
            has_shortfall = initial_full < 8192 or (uses_swa and initial_swa < 2560)
            for reclaimed_halves in (0, 1, 2) if has_shortfall else (2,):
                with self.subTest(
                    uses_swa=uses_swa,
                    full_available=initial_full,
                    swa_available=initial_swa,
                    reclaimed_halves=reclaimed_halves,
                ):
                    self._check_resume_reclaim(
                        uses_swa, initial_full, initial_swa, reclaimed_halves
                    )

    def _check_resume_reclaim(
        self, uses_swa, full_available, swa_available, reclaimed_halves
    ):
        page_size = 512
        backup = object()
        req = SimpleNamespace(
            rid="retracted",
            origin_input_ids=[0] * 8191,
            output_ids=[],
            is_retracted=True,
            kv=SimpleNamespace(retraction_backup=backup),
            load_kv_cache=MagicMock(),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = [req]
        queue.num_reserved_decode_tokens = 512
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue.token_to_kv_pool_allocator = (
            SimpleNamespace(
                page_size=page_size, swa_available_size=lambda: swa_available
            )
            if uses_swa
            else SimpleNamespace(page_size=page_size)
        )
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False, sliding_window_size=2048
        )
        queue._radix_full_available = MagicMock(side_effect=lambda: full_available)
        queue.tree_cache = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=uses_swa)
        queue._allocatable_token_budgets = MagicMock(return_value=500224)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(500224, 4096)
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(return_value=0)
        full_shortfall = max(8192 - full_available, 0)
        swa_shortfall = max(2560 - swa_available, 0) if uses_swa else 0

        def evict(params):
            nonlocal full_available, swa_available
            self.assertTrue(req.is_retracted)
            full_available += (
                params.num_tokens // page_size * reclaimed_halves // 2
            ) * page_size
            swa_available += (
                params.swa_num_tokens // page_size * reclaimed_halves // 2
            ) * page_size

        def allocate(_req):
            nonlocal full_available, swa_available
            self.assertGreaterEqual(full_available, 8192)
            full_available -= 8192
            if uses_swa:
                self.assertGreaterEqual(swa_available, 2560)
                swa_available -= 2560

        def restore(_req_pool, _allocator):
            queue._pre_alloc.assert_called_once_with(req)
            self.assertFalse(req.is_retracted)
            self.assertIs(req.kv.retraction_backup, backup)
            req.kv.retraction_backup = None

        queue.tree_cache.evict_for_alloc.side_effect = evict
        queue._pre_alloc = MagicMock(side_effect=allocate)
        req.load_kv_cache.side_effect = restore

        resumed = queue.resume_retracted_reqs()

        succeeded = reclaimed_halves == 2
        self.assertEqual(resumed, [req] if succeeded else [])
        self.assertEqual(queue.retracted_queue, [] if succeeded else [req])
        self.assertEqual(req.is_retracted, not succeeded)
        self.assertEqual(queue._pre_alloc.call_count, int(succeeded))
        if succeeded:
            req.load_kv_cache.assert_called_once_with(
                queue.req_to_token_pool, queue.token_to_kv_pool_allocator
            )
            self.assertIsNone(req.kv.retraction_backup)
        else:
            req.load_kv_cache.assert_not_called()
            self.assertIs(req.kv.retraction_backup, backup)
        if full_shortfall or swa_shortfall:
            queue.tree_cache.evict_for_alloc.assert_called_once()
            params = queue.tree_cache.evict_for_alloc.call_args.args[0]
            self.assertEqual(params.num_tokens, full_shortfall)
            self.assertEqual(params.swa_num_tokens, swa_shortfall)
        else:
            queue.tree_cache.evict_for_alloc.assert_not_called()

        if not succeeded:
            # A later scheduler pass can resume the same backup once pages are free.
            full_available, swa_available = 8192, 2560
            self.assertEqual(queue.resume_retracted_reqs(), [req])
            self.assertEqual(queue.retracted_queue, [])
            self.assertFalse(req.is_retracted)
            req.load_kv_cache.assert_called_once_with(
                queue.req_to_token_pool, queue.token_to_kv_pool_allocator
            )
            self.assertIsNone(req.kv.retraction_backup)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            bootstrap_room=42,
            kv=ReqKvInfo(),
            finished_reason=None,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = object()
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.metrics_reporter.enable_metrics = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ) as poll:
            preallocated, failed = queue.pop_preallocated()

        poll.assert_called_once_with(
            [receiver], queue.gloo_group, additional_groups=None
        )
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_reclaims_capacity_consumed_by_request_state(self):
        override = get_context().override_server_args(
            disaggregation_decode_enable_radix_cache=True
        )
        override.install()
        self.addCleanup(override.restore)
        available = 29
        req = SimpleNamespace(
            rid="shared-mamba",
            origin_input_ids=[0] * 29,
            output_ids=[],
            kv=ReqKvInfo(),
            set_extend_range=MagicMock(),
        )

        def alloc_request_slot(reqs):
            nonlocal available
            # A Mamba slot can consume bytes from the shared full-attention pool.
            available -= 3
            reqs[0].kv.req_pool_idx = 0
            return [0]

        def evict(params):
            nonlocal available
            self.assertEqual(req.kv.req_pool_idx, 0)
            self.assertEqual(params.num_tokens, 3)
            available += 3

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = SimpleNamespace(
            alloc=MagicMock(side_effect=alloc_request_slot), write=MagicMock()
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            available_size=lambda: available,
            alloc=MagicMock(
                side_effect=lambda n: torch.arange(n) if available >= n else None
            ),
        )
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False, tp_worker=SimpleNamespace(is_hybrid_swa=False)
        )
        queue.tree_cache = MagicMock()
        queue.tree_cache.evict_for_alloc.side_effect = evict
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)

        self.assertIsNone(
            queue._reclaim_prealloc_capacity(
                full_alloc_tokens=29, swa_tail_len=None, req_id=req.rid
            )
        )
        queue.tree_cache.evict_for_alloc.assert_not_called()
        indices = queue._pre_alloc(req)

        self.assertEqual(len(indices), 29)
        queue.tree_cache.evict_for_alloc.assert_called_once()
        queue.token_to_kv_pool_allocator.alloc.assert_called_once_with(29)

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            kv=ReqKvInfo(),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_kv_reclaim_failure_rejects_only_request(self):
        override = get_context().override_server_args(
            disaggregation_decode_enable_radix_cache=True
        )
        override.install()
        self.addCleanup(override.restore)

        for uses_swa, full_available, swa_available, has_mamba in (
            (False, 512, 0, False),
            (False, 512, 0, True),
            (True, 512, 2560, False),
            (True, 4096, 512, False),
            (True, 512, 512, True),
        ):
            with self.subTest(
                uses_swa=uses_swa,
                full_available=full_available,
                swa_available=swa_available,
                has_mamba=has_mamba,
            ):
                self._check_prealloc_reclaim_failure(
                    uses_swa, full_available, swa_available, has_mamba
                )

    def _check_prealloc_reclaim_failure(
        self, uses_swa, full_available, swa_available, has_mamba
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="kv-reclaim-failed",
            origin_input_ids=[0] * 8191,
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
            last_node=object(),
            lock_receipt=object(),
            swa_prefix_lock_released=False,
            kv=ReqKvInfo(),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )
        waiting_req = SimpleNamespace(
            req=SimpleNamespace(finished_reason=None), waiting_for_input=False
        )
        ready_req = SimpleNamespace(
            req=SimpleNamespace(
                rid="fits-without-eviction",
                origin_input_ids=[0] * 31,
                output_ids=[],
                finished_reason=None,
                sampling_params=SimpleNamespace(max_new_tokens=1),
                kv=ReqKvInfo(),
                time_stats=MagicMock(),
            ),
            kv_receiver=MagicMock(spec=["send_metadata", "abort", "clear"]),
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req, ready_req, waiting_req]
        queue.pending_reqs = [decode_req, waiting_req]
        queue._num_published_destinations = 0
        queue.transfer_queue = SimpleNamespace(enable_staging=False)
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 512
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=uses_swa)
        queue._allocatable_token_budgets = MagicMock(return_value=16384)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(16384, 8192)
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(return_value=8192)
        queue._radix_full_available = MagicMock(return_value=full_available)
        queue.token_to_kv_pool_allocator = (
            SimpleNamespace(page_size=512, swa_available_size=lambda: swa_available)
            if uses_swa
            else SimpleNamespace(page_size=512)
        )
        queue.token_to_kv_pool_allocator.translate_kv_indices_for_transfer = (
            lambda indices: indices
        )
        queue.tree_cache = MagicMock()

        def match_prefix(req):
            # Mamba matching reserves the deferred COW destination before a
            # request slot or full-attention KV has been allocated.
            if has_mamba and req is decode_req.req:
                req.kv.mamba_pool_idx = torch.tensor(1)
            prefix_len = 4096 if req is decode_req.req else 0
            return SimpleNamespace(
                prefix_indices=list(range(prefix_len)),
                l1_prefix_len=prefix_len,
                decode_prefix_len=prefix_len,
                restore_token_count=0,
            )

        def allocate(req, _prefix_indices, prefix_len, total_prefix_len):
            self.assertIs(req, ready_req.req)
            self.assertEqual((prefix_len, total_prefix_len), (0, 0))
            req.kv.req_pool_idx = 0
            return torch.arange(31)

        queue._match_prefix_and_lock = MagicMock(side_effect=match_prefix)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock(side_effect=allocate)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_token_pool.req_to_token = torch.arange(31).reshape(1, -1)
        queue.tree_cache.req_to_token_pool = queue.req_to_token_pool
        queue.tree_cache.supports_mamba.return_value = has_mamba
        # pop_preallocated compares the mamba slot count before admitting.
        queue.req_to_token_pool.mamba_allocator.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.enable_decode_hicache = False
        scheduler.sliding_window_size = 2048
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [ready_req])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [waiting_req])
        self.assertEqual(queue.pending_reqs, [waiting_req])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 503)
        self.assertFalse(req.kv.holds_mamba)
        free_mamba = queue.req_to_token_pool.mamba_allocator.free
        if has_mamba:
            free_mamba.assert_called_once()
            self.assertTrue(
                torch.equal(free_mamba.call_args.args[0], torch.tensor([1]))
            )
        else:
            free_mamba.assert_not_called()
        self.assertEqual(receiver.calls, ["abort", "clear"])
        self.assertIn("eviction insufficient", req.finished_reason.message)
        queue.tree_cache.evict_for_alloc.assert_called_once()
        params = queue.tree_cache.evict_for_alloc.call_args.args[0]
        self.assertEqual(params.num_tokens, max(4096 - full_available, 0))
        self.assertEqual(
            params.swa_num_tokens, max(2560 - swa_available, 0) if uses_swa else 0
        )
        if uses_swa:
            queue.tree_cache.dec_swa_lock_only.assert_called_once_with(
                req.last_node, req.lock_receipt
            )
            queue.tree_cache.dec_lock_ref.assert_called_once_with(
                req.last_node, req.lock_receipt, skip_swa=True
            )
        else:
            queue.tree_cache.dec_lock_ref.assert_called_once_with(
                req.last_node, req.lock_receipt
            )
        self.assertFalse(req.swa_prefix_lock_released)
        queue._pre_alloc.assert_called_once()
        queue.req_to_token_pool.alloc.assert_not_called()
        ready_req.kv_receiver.send_metadata.assert_called_once()
        ready_req.kv_receiver.abort.assert_not_called()
        ready_req.kv_receiver.clear.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_cleanup_respects_deferred_release_gates(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            prefix_offer_sent=True,
            prefix_offer_lock=None,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

        receiver = FakeReceiver()
        receiver.kv_mgr = FakeKVManager.__new__(FakeKVManager)
        decode_req.kv_receiver = receiver
        queue.queue = [decode_req]
        queue.enable_deferred_kv_release = True
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

        receiver = MagicMock()
        decode_req.kv_receiver = receiver
        decode_req.host_staged = True
        queue.enable_host_receive = True
        queue.enable_deferred_kv_release = False
        queue._defer_release = MagicMock()
        queue.queue = [decode_req]
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()
        self.assertEqual(queue.pop_transferred(), [])
        receiver.abort.assert_called_once_with()
        queue._defer_release.assert_called_once_with(decode_req)
        receiver.clear.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        mock_release_kv_cache.assert_not_called()

        queue.queue = [decode_req]
        decode_req.req.finished_reason = FINISH_ABORT("cancelled")
        with (
            patch.object(
                queue, "_poll_with_metadata_gate", return_value=[KVPoll.Success]
            ),
            patch(
                "sglang.srt.disaggregation.decode.discard_kv_cache_backup"
            ) as discard,
        ):
            self.assertEqual(queue.pop_transferred(), [])
            discard.assert_called_once_with(req, queue.tree_cache, "host_pool")
        receiver.clear.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_not_called()

    def test_fake_receiver_initializes_deferred_release_state(self):
        manager = MagicMock()
        receiver = DisaggFakeKVReceiver(manager, "")

        self.assertIs(receiver.kv_mgr, manager)
        self.assertFalse(receiver.abort_notified)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_failed_fake_transfer_releases_at_once_under_deferred_release(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        # A health-check request gets a FakeKVReceiver over the real transfer
        # manager, so a failed one reaches the deferred-release path.
        receiver = DisaggFakeKVReceiver(
            MagicMock(enable_deferred_decode_kv_release=True), ""
        )
        req = SimpleNamespace(
            rid="health-check", bootstrap_room=0, return_logprob=False
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            prefix_offer_sent=True,
            prefix_offer_lock=None,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = True
        queue._defer_release = MagicMock()
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.scheduler = MagicMock(enable_decode_hicache=False, enable_hisparse=False)
        queue.scheduler.metrics_reporter.enable_metrics = False
        mock_poll.return_value = [KVPoll.Failed]

        self.assertEqual(queue.pop_transferred(), [])
        self.assertFalse(receiver.abort_notified)
        queue._defer_release.assert_not_called()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
