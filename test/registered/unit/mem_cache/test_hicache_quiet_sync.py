"""QuietSyncGate: the HiCache ack poll may be skipped only in lockstep across ranks."""

import random
import unittest

from sglang.srt.mem_cache.hicache_quiet_sync import QuietSyncGate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def run_ranks(every, ready_steps, num_steps):
    """Simulate per-step polls on several ranks. ready_steps[r][i] is the step at which
    rank r's i-th ack (issued in the same order on every rank) becomes ready. Returns the
    step at which each ack was reaped, asserting the ranks never disagree on polling."""
    num_ranks = len(ready_steps)
    gates = [QuietSyncGate(every) for _ in range(num_ranks)]
    reaped = [0] * num_ranks
    reaped_at = []
    for step in range(num_steps):
        skips = [g.should_skip() for g in gates]
        assert len(set(skips)) == 1, f"ranks disagree on polling at step {step}"
        if skips[0]:
            continue
        local = [
            sum(1 for s in ready_steps[r][reaped[r] :] if s <= step)
            for r in range(num_ranks)
        ]
        agreed = min(local)  # the MIN all-reduce
        for r in range(num_ranks):
            reaped[r] += agreed
            gates[r].record(agreed > 0)
        reaped_at.extend([step] * agreed)
    return reaped_at


class TestQuietSyncGate(unittest.TestCase):
    def test_disabled_never_skips(self):
        for every in (0, 1):
            g = QuietSyncGate(every)
            g.record(False)
            self.assertFalse(any(g.should_skip() for _ in range(10)))

    def test_backs_off_only_after_a_quiet_poll(self):
        g = QuietSyncGate(4)
        self.assertFalse(g.should_skip())  # nothing recorded yet: poll
        g.record(True)
        self.assertFalse(g.should_skip())  # busy: poll every step
        g.record(False)
        self.assertEqual([g.should_skip() for _ in range(4)], [True, True, True, False])

    def test_found_ack_returns_to_every_step(self):
        g = QuietSyncGate(3)
        g.record(False)
        self.assertEqual([g.should_skip() for _ in range(3)], [True, True, False])
        g.record(True)
        self.assertFalse(g.should_skip())

    def test_lockstep_no_ack_lost_bounded_delay(self):
        rng = random.Random(0)
        for every in (2, 4, 8):
            num_ranks, num_acks = 8, 200
            issue = sorted(rng.randrange(0, 2000) for _ in range(num_acks))
            ready = [
                [s + rng.randrange(0, 30) for s in issue] for _ in range(num_ranks)
            ]
            for r in range(num_ranks):
                # an ack is reaped in queue order, so readiness is monotone per rank
                for i in range(1, num_acks):
                    ready[r][i] = max(ready[r][i], ready[r][i - 1])
            reaped_at = run_ranks(every, ready, 2200)
            self.assertEqual(len(reaped_at), num_acks)
            for i, step in enumerate(reaped_at):
                all_ready = max(ready[r][i] for r in range(num_ranks))
                self.assertGreaterEqual(step, all_ready)
                self.assertLessEqual(step - all_ready, every - 1)


if __name__ == "__main__":
    unittest.main()
