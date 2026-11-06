"""Count-only prefill reservation credits; no token window or charge expiry."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.reservation_slots import (
    SlotReservations,
    effective_slot_limit,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Sender:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.busy = False
        self.asks = 0

    def decode_endpoint(self):
        return self.endpoint

    def reservation_busy(self):
        return self.busy

    def request_decode_reservation(self):
        self.asks += 1
        self.busy = False


def request(rid, endpoint="decode-a"):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=range(65536),
        pending_bootstrap=True,
        disagg_kv_sender=Sender(endpoint),
    )


def run_pass(controller, reqs, now=0, capacity=None, started=()):
    controller.begin(reqs, capacity=capacity, now=now)
    for req in reqs:
        controller.consider(req)
    return controller.finish([r for r in reqs if r.rid not in started])


class TestReservationSlots(unittest.TestCase):
    def controller(self, limit=4):
        return SlotReservations(limit=limit, rank_groups=[])

    def test_no_busy_collective_without_outstanding_reservations(self):
        c = SlotReservations(limit=1024, rank_groups=[object()])
        with (
            patch(
                "sglang_tpu.disaggregation.reservation_slots.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang_tpu.disaggregation.reservation_slots.dist.all_reduce"
            ) as reduce,
        ):
            c.begin([], now=0)
        reduce.assert_not_called()

    def test_empty_queue_has_no_order_or_ask_broadcast(self):
        c = self.controller()
        c.begin([], now=0)
        with patch.object(c, "_broadcast") as broadcast:
            self.assertEqual(c.agree_order([]), [])
            self.assertEqual(c.finish([]), [])
        broadcast.assert_not_called()

    def test_idle_state_waits_for_unregistered_request_cleanup(self):
        c = self.controller()
        self.assertTrue(c.is_idle())
        req = request("unregistered", None)
        run_pass(c, [req])
        self.assertFalse(c.is_idle())
        run_pass(c, [], now=1)
        self.assertTrue(c.is_idle())
        self.assertEqual(c._candidates, [])

    def test_drain_flushes_pass_timing_before_idle_fast_path(self):
        c = SlotReservations(limit=1, rank_groups=[], log_stats=True)
        req = request("a")
        with patch(
            "sglang_tpu.disaggregation.reservation_slots.time.perf_counter",
            side_effect=[10, 10.025, 11, 11.05],
        ):
            run_pass(c, [req])
            with self.assertLogs(
                "sglang_tpu.disaggregation.reservation_slots", level="INFO"
            ) as logs:
                run_pass(c, [], now=0.1)
        self.assertIn("passes_total=2", logs.output[0])
        self.assertIn("pass_ms=37.50", logs.output[0])
        self.assertIn("max_pass_ms=50.00", logs.output[0])
        self.assertIn("max_queue_depth=1", logs.output[0])

    def test_default_and_override_use_effective_request_capacity(self):
        self.assertEqual(effective_slot_limit(0, 8), 8)
        self.assertEqual(effective_slot_limit(32, 8), 8)
        self.assertEqual(effective_slot_limit(2, 8), 2)
        self.assertEqual(effective_slot_limit(0, 0), 0)
        with self.assertRaises(ValueError):
            effective_slot_limit(-1, 8)

    def test_large_prompts_use_one_credit_each_without_a_token_window(self):
        c = self.controller()
        reqs = [request(str(i)) for i in range(8)]
        self.assertEqual(run_pass(c, reqs), reqs[:4])
        self.assertEqual(run_pass(c, reqs, now=10000), [])

    def test_current_capacity_limits_asks_and_retains_existing_ownership(self):
        c = self.controller()
        reqs = [request(str(i)) for i in range(8)]
        self.assertEqual(run_pass(c, reqs, capacity=2), reqs[:2])
        self.assertEqual(run_pass(c, reqs, capacity=1, now=10), [])

    def test_ready_waits_for_first_prefill_batch_before_freeing_credit(self):
        c = self.controller(2)
        reqs = [request(str(i)) for i in range(3)]
        run_pass(c, reqs)
        reqs[0].pending_bootstrap = False
        c.mark_ready(reqs[0])
        self.assertEqual(run_pass(c, reqs, now=20), [])
        self.assertEqual(run_pass(c, reqs, now=21, started=("0",)), [reqs[2]])

    def test_transport_ready_awaiting_metadata_keeps_credit_and_beats_busy(self):
        c = self.controller(1)
        ready, other = request("ready"), request("other")
        run_pass(c, [ready, other])
        c.mark_ready(ready)
        ready.disagg_kv_sender.busy = True
        self.assertEqual(run_pass(c, [ready, other], now=10000), [])

    def test_busy_frees_credit_for_backfill_and_retry_uses_measured_rtt(self):
        c = self.controller(1)
        first, other = request("first"), request("other")
        run_pass(c, [first, other])
        first.disagg_kv_sender.busy = True
        self.assertEqual(run_pass(c, [first, other], now=0.02), [other])
        self.assertEqual(run_pass(c, [first], now=0.059), [])
        self.assertEqual(run_pass(c, [first], now=0.061), [first])

    def test_destination_shares_use_effective_capacity(self):
        c = self.controller(8)
        a = [request(f"a{i}", "a") for i in range(8)]
        b = [request(f"b{i}", "b") for i in range(4)]
        self.assertEqual(run_pass(c, a + b, capacity=4), a[:2] + b[:2])
        b[0].pending_bootstrap = False
        c.mark_ready(b[0])
        self.assertEqual(
            run_pass(c, a + b, capacity=4, now=50, started=("b0",)), [b[2]]
        )

    def test_busy_destination_in_cooldown_does_not_reserve_an_idle_share(self):
        c = self.controller(4)
        a = request("a", "a")
        b = [request(f"b{i}", "b") for i in range(6)]
        self.assertEqual(run_pass(c, [a] + b), [a] + b[:2])
        a.disagg_kv_sender.busy = True
        self.assertEqual(run_pass(c, [a] + b[2:], now=0.02), b[2:])

    def test_cancellation_and_reused_id_start_a_fresh_reservation(self):
        c = self.controller(1)
        old = request("same")
        run_pass(c, [old])
        c.mark_ready(old)
        replacement = request("same")
        self.assertEqual(run_pass(c, [replacement], now=1), [replacement])
        c.mark_ready(old)
        replacement.disagg_kv_sender.busy = True
        other = request("other")
        self.assertEqual(run_pass(c, [replacement, other], now=2), [other])

    def test_missing_endpoint_does_not_hold_a_slot_or_destination_share(self):
        c = self.controller(2)
        unknown = request("unknown", None)
        healthy = [request("a"), request("b")]
        self.assertEqual(run_pass(c, [unknown] + healthy), healthy)

    def test_enqueue_ids_survive_policy_permutations(self):
        c = self.controller()
        reqs = [request("a"), request("b")]
        ids = [c.track(r) for r in reqs]
        self.assertEqual(c.agree_order(list(reversed(reqs))), list(reversed(reqs)))
        self.assertEqual([c.track(r) for r in reqs], ids)
        self.assertEqual(c.canonical_order(list(reversed(reqs))), reqs)

    def test_bounded_statistics_report_busy_rtt_and_final_drain(self):
        c = SlotReservations(limit=1, rank_groups=[], log_stats=True)
        first, other = request("first"), request("other")
        run_pass(c, [first, other])
        first.disagg_kv_sender.busy = True
        run_pass(c, [first, other], now=0.02)
        with self.assertLogs(
            "sglang_tpu.disaggregation.reservation_slots", level="INFO"
        ) as logs:
            run_pass(c, [], now=61)
            run_pass(c, [], now=122)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("busy_total=1", logs.output[0])
        self.assertIn("pending=0", logs.output[0])
        self.assertIn("rtt_ms=20.00", logs.output[0])

    def test_busy_cooldown_keeps_rtt_history_between_attempts(self):
        c = self.controller(1)
        req = request("a")
        run_pass(c, [req])
        req.disagg_kv_sender.busy = True
        run_pass(c, [req], now=0.02)
        run_pass(c, [req], now=0.061)
        req.disagg_kv_sender.busy = True
        run_pass(c, [req], now=0.161)
        self.assertEqual(run_pass(c, [req], now=0.245), [req])


if __name__ == "__main__":
    unittest.main()
