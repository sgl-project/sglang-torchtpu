"""Actual scheduler and cache integration for count-only reservation slots."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.reservation_slots import SlotReservations

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Request(SimpleNamespace):
    __eq__ = object.__eq__
    __hash__ = object.__hash__


def request(rid, decode="decode-a"):
    sender = Mock()
    sender.decode_endpoint.return_value = decode
    sender.reservation_busy.return_value = False
    return _Request(
        rid=rid,
        origin_input_ids=range(65536),
        pending_bootstrap=True,
        disagg_kv_sender=sender,
    )


class _Fixture:
    def controller(self, *, limit=8):
        return SlotReservations(limit=limit, rank_groups=[])

    def run_pass(self, controller, reqs, now=0):
        controller.begin(reqs, now=now)
        for req in reqs:
            controller.consider(req)
        return controller.finish(reqs)


class TestBatchReservationHooks(unittest.TestCase):
    def test_idle_scheduler_bypasses_reservation_work(self):
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang_tpu.hooks import prefill_driven as hooks

        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        controller = SlotReservations(limit=8, rank_groups=[])
        hooks._controllers[queue] = controller
        scheduler = SimpleNamespace(
            disagg_prefill_bootstrap_queue=queue,
            waiting_queue=[],
            get_num_allocatable_reqs=Mock(return_value=8),
            req_to_token_pool=Mock(available_size=Mock(return_value=8)),
            max_prefill_bs=0,
        )
        original = Mock(return_value="original result")
        with (
            patch.object(controller, "begin", wraps=controller.begin) as begin,
            patch.object(controller, "finish", wraps=controller.finish) as finish,
            patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
        ):
            self.assertEqual(
                hooks.reservation_pass(
                    original, scheduler, None, SimpleNamespace(reqs=[])
                ),
                "original result",
            )
        scheduler.get_num_allocatable_reqs.assert_not_called()
        begin.assert_not_called()
        finish.assert_not_called()
        self.assertIsNone(hooks._active_pass.get())

    def test_automatic_slots_and_override_are_clamped_to_physical_capacity(self):
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang_tpu.envs import envs
        from sglang_tpu.hooks import prefill_driven as hooks

        for configured, expected in ((0, 8), (32, 8), (3, 3)):
            with self.subTest(configured=configured):
                queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
                queue.tp_rank = 0
                queue.scheduler = SimpleNamespace(
                    max_running_requests=8,
                    req_to_token_pool=SimpleNamespace(size=16),
                    attn_tp_cpu_group=None,
                    attn_cp_cpu_group=None,
                )
                with envs.SGLANG_DISAGG_PREFILL_DRIVEN_SLOTS.override(configured):
                    hooks.attach_batch_reservations(None, queue)
                self.assertEqual(hooks._controllers[queue].limit, expected)

    def test_rejected_old_intake_does_not_replace_live_reservation_owner(self):
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang_tpu.hooks import prefill_driven as hooks

        fixture = _Fixture()
        c = fixture.controller(limit=1)
        replacement, old = request("reused"), request("reused")
        fixture.run_pass(c, [replacement])
        enqueue_id = c.track(replacement)
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = []  # Core add rejected/retired an old grammar request.
        queue.scheduler = SimpleNamespace(processed_tokens_counter=10)
        hooks._controllers[queue] = c
        hooks.stamp_enqueue(None, queue, old)
        self.assertEqual(fixture.run_pass(c, [replacement], now=1), [])
        self.assertEqual(c.track(replacement), enqueue_id)
        replacement.disagg_kv_sender.request_decode_reservation.assert_called_once()
        self.assertFalse(hasattr(old, "arrival_processed_tokens"))

    def test_real_policy_and_adder_select_warm_long_prompt_before_cold_short_prompt(
        self,
    ):
        from array import array

        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.managers.schedule_policy import (
            AddReqResult,
            PrefillAdder,
            SchedulePolicy,
        )
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.prefill_budget import PrefillBudget
        from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
        from sglang_tpu.hooks import prefill_driven as hooks

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        cache = RadixCache.create_simulated(page_size=64)
        prefix = array("q", range(64000))
        cache.insert(InsertParams(key=RadixKey(token_ids=prefix)))
        warm = Req(
            "warm",
            "",
            prefix + array("q", [90000] * 1024),
            SamplingParams(max_new_tokens=1),
        )
        cold = Req(
            "cold", "", array("q", [80000] * 2048), SamplingParams(max_new_tokens=1)
        )
        waiting = [cold, warm]
        controller = _Fixture().controller(limit=1)
        for req in waiting:
            req.pending_bootstrap = True
            req.disagg_kv_sender = Mock(spec=RaidenKVSender)
            req.disagg_kv_sender.reservation_busy.return_value = False
            req.disagg_kv_sender.decode_endpoint.return_value = "decode-a"
            controller.track(req)
        allocator = Mock(page_size=64)
        allocator.available_size.return_value = 1_000_000
        allocator.create_prefill_budget.side_effect = lambda tree, **kwargs: (
            PrefillBudget(allocator, tree, **kwargs)
        )
        policy = SchedulePolicy("shortest-prefill-first", cache, False, False, True)
        adder = PrefillAdder(
            64, cache, allocator, ScheduleBatch(reqs=[]), 1.0, 1024, 1024, 0, 0
        )
        controller.begin(waiting, now=0)
        token = hooks._active_pass.set(SimpleNamespace(controller=controller, slots=8))
        try:
            policy.calc_priority(waiting, processed_tokens=0)
            self.assertEqual(waiting, [warm, cold])
            self.assertEqual(warm.num_matched_prefix_tokens, 64000)
            adder.is_admission_ready = lambda req, work: hooks.gate_admission(
                lambda *args: True, adder, req, work
            )
            for req in waiting:
                req.init_next_round_input(cache)
                self.assertEqual(
                    adder.add_one_req(req, False, None), AddReqResult.DEFERRED
                )
            self.assertEqual(controller.finish(waiting), [warm])
            self.assertEqual(adder.can_run_list, [])
            self.assertEqual(adder.rem_chunk_tokens, 1024)
        finally:
            hooks._active_pass.reset(token)

    def test_transport_ready_is_recorded_when_real_metadata_allocation_fails(self):
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang_tpu.hooks import prefill_driven as hooks

        fixture = _Fixture()
        controller = fixture.controller(limit=1)
        a, b = request("ready"), request("next")
        a.metadata_buffer_index = -1
        fixture.run_pass(controller, [a, b])
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.req_to_metadata_buffer_idx_allocator = Mock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 0
        hooks._controllers[queue] = controller
        self.assertFalse(
            hooks.observe_transport_ready(
                PrefillBootstrapQueue.finalize_bootstrap,
                queue,
                a,
            )
        )
        self.assertTrue(a.pending_bootstrap)
        self.assertEqual(fixture.run_pass(controller, [a, b], now=6), [])
        b.disagg_kv_sender.request_decode_reservation.assert_not_called()

    def test_infeasible_tile_or_scratch_does_not_propose_a_reservation(self):
        from sglang.srt.managers.schedule_policy import AddReqResult
        from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
        from sglang_tpu.hooks import prefill_driven as hooks

        req = request("blocked")
        req.disagg_kv_sender = Mock(spec=RaidenKVSender)
        req.sampling_params = SimpleNamespace(ignore_eos=True)
        req.prefix_indices, req.host_hit_length = [], 0
        for tile, scratch in ((AddReqResult.OTHER, True), (None, False)):
            with self.subTest(tile=tile, scratch=scratch):
                controller = Mock()
                adder = SimpleNamespace(
                    tree_cache=SimpleNamespace(disable=True),
                    _check_prefill_tile_budget=Mock(return_value=tile),
                    _kv_shard_reserve_scratch=Mock(return_value=scratch),
                )
                token = hooks._active_pass.set(SimpleNamespace(controller=controller))
                try:
                    self.assertFalse(
                        hooks.gate_admission(lambda *args: True, adder, req, 1024)
                    )
                    controller.consider.assert_not_called()
                finally:
                    hooks._active_pass.reset(token)

    def test_real_scheduler_with_registered_wrappers_runs_ready_subset_then_late_ready(
        self,
    ):
        import importlib.util
        from contextlib import ExitStack
        from pathlib import Path

        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.plugins.hook_registry import HookType, _wrap_fn
        from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
        from sglang_tpu.hooks import prefill_driven as hooks

        # Reuse the core fixture without relying on collection order or
        # shadowing Python's stdlib `test` package.
        fixture_path = Path(__file__).parents[1] / "managers/test_prefill_adder.py"
        spec = importlib.util.spec_from_file_location(
            "prefill_adder_fixture", fixture_path
        )
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        fixture = fixture_module.TestPrefillAdder()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        scheduler, pending, ready = fixture._deferred_scheduler_fixture()
        scheduler.enable_lora = False
        scheduler.tree_cache.supports_fast_match_prefix.return_value = False
        scheduler.policy = SchedulePolicy(
            "fcfs", scheduler.tree_cache, False, False, True
        )
        queue = Mock()
        scheduler.disagg_prefill_bootstrap_queue = queue
        controller = SlotReservations(limit=8, rank_groups=[])
        hooks._controllers[queue] = controller
        for req in (pending, ready):
            req.disagg_kv_sender = Mock(spec=RaidenKVSender)
            req.disagg_kv_sender.decode_endpoint.return_value = "decode-a"
            req.disagg_kv_sender.reservation_busy.return_value = False
            req.pending_bootstrap = req is pending
            controller.track(req)
        with ExitStack() as stack:
            for cls, name, hook, kind in (
                (
                    Scheduler,
                    "_get_new_batch_prefill_raw",
                    hooks.reservation_pass,
                    HookType.AROUND,
                ),
                (
                    PrefillAdder,
                    "is_admission_ready",
                    hooks.gate_admission,
                    HookType.AROUND,
                ),
                (
                    PrefillAdder,
                    "get_match_group_size",
                    hooks.match_group_size,
                    HookType.AROUND,
                ),
                (
                    SchedulePolicy,
                    "calc_priority",
                    hooks.agree_policy_order,
                    HookType.AFTER,
                ),
            ):
                stack.enter_context(
                    patch.object(cls, name, _wrap_fn(getattr(cls, name), hook, kind))
                )
            stack.enter_context(
                patch.object(ScheduleBatch, "init_new", return_value=Mock())
            )
            running = ScheduleBatch(reqs=[], batch_is_full=False)
            first, _ = scheduler._get_new_batch_prefill_raw(None, running)
            self.assertIsNotNone(first)
            self.assertEqual(scheduler.waiting_queue, [pending])
            pending.disagg_kv_sender.request_decode_reservation.assert_called_once()
            ready.disagg_kv_sender.request_decode_reservation.assert_not_called()
            # Finalization is done by the normal all-reduced bootstrap poll.
            pending.pending_bootstrap = False
            second, _ = scheduler._get_new_batch_prefill_raw(None, running)
            self.assertIsNotNone(second)
            self.assertEqual(scheduler.waiting_queue, [])
            pending.disagg_kv_sender.request_decode_reservation.assert_called_once()
            self.assertIsNone(hooks._active_pass.get())

    def test_pending_bootstrap_reaches_the_waiting_queue_without_metadata(self):
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
        from sglang_tpu.hooks import prefill_driven as hooks

        class Request(SimpleNamespace):
            __hash__ = object.__hash__

        req = Request(
            rid="pending",
            pending_bootstrap=True,
            prefill_attempt_count=0,
            is_retracted=False,
            disagg_kv_sender=Mock(spec=RaidenKVSender),
            time_stats=Mock(),
            arrival_processed_tokens=0,
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler_stage_metrics = None
        queue.scheduler = SimpleNamespace(processed_tokens_counter=12)
        queue.pp_size = 1
        queue.queue = [req]
        queue.ensure_metadata_buffer = Mock()
        queue.finalize_bootstrap = Mock()
        controller = SlotReservations(limit=8, rank_groups=[])
        hooks._controllers[queue] = controller
        hooks.stamp_enqueue(None, queue, req)
        queue.scheduler.processed_tokens_counter = 900
        queue.scheduler.attn_cp_cpu_group = queue.scheduler.attn_tp_cpu_group = None
        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Bootstrapping],
            ),
            patch(
                "sglang.srt.disaggregation.prefill.get_disagg",
                return_value=SimpleNamespace(
                    optimistic_prefill_attempts=0,
                    disaggregation_decode_allocation_policy="early",
                ),
            ),
        ):
            ready = hooks.pop_bootstrapped_with_reservations(
                PrefillBootstrapQueue.pop_bootstrapped,
                queue,
            )
        self.assertEqual(ready, [req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(req.pending_bootstrap)
        self.assertEqual(req.arrival_processed_tokens, 12)
        queue.ensure_metadata_buffer.assert_not_called()
        queue.finalize_bootstrap.assert_not_called()

    def test_gate_records_feasible_candidate_and_waits_for_finalized_bootstrap(self):
        from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
        from sglang_tpu.hooks import prefill_driven as hooks

        req = request("warm")
        req.disagg_kv_sender = Mock(spec=RaidenKVSender)
        req.sampling_params = SimpleNamespace(ignore_eos=False)
        req.prefix_indices = range(64000)
        req.host_hit_length = 512
        controller = Mock()
        adder = SimpleNamespace(
            exact_chunk_fill=True,
            rem_input_tokens=4096,
            rem_chunk_tokens=4096,
            max_prefill_bs=0,
            prefill_max_requests=None,
            tree_cache=SimpleNamespace(disable=False),
            _kv_shard_reserve_scratch=Mock(return_value=True),
        )
        token = hooks._active_pass.set(SimpleNamespace(controller=controller, slots=8))
        try:
            self.assertFalse(hooks.gate_admission(lambda *args: True, adder, req, 1024))
            controller.consider.assert_called_once_with(req)
            adder._kv_shard_reserve_scratch.assert_called_once_with(
                64512, 1024, commit=False
            )
            self.assertEqual(hooks.match_group_size(lambda *args: 123, adder, 123), 0)
            req.pending_bootstrap = False
            self.assertTrue(hooks.gate_admission(lambda *args: True, adder, req, 1024))
        finally:
            hooks._active_pass.reset(token)
        self.assertTrue(hooks.gate_admission(lambda *args: True, adder, req, 1024))


if __name__ == "__main__":
    unittest.main()
