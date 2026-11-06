"""Count-only prefill reservation credits; no token window or charge expiry."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.reservation_slots import SlotReservations

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

    def test_idle_state_waits_for_unregistered_request_cleanup(self):
        c = self.controller()
        self.assertTrue(c.is_idle())
        req = request("unregistered", None)
        run_pass(c, [req])
        self.assertFalse(c.is_idle())
        run_pass(c, [], now=1)
        self.assertTrue(c.is_idle())
        self.assertEqual(c._candidates, [])

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


if __name__ == "__main__":
    unittest.main()
