import time
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self, conclude_state=None):
        self.conclude_state = conclude_state
        self.aborted = False

    def abort(self):
        self.aborted = True
        self.conclude_state = KVPoll.Failed


def make_req(age_s: float, conclude_state=None, entered=True):
    receiver = FakeReceiver(conclude_state)
    return SimpleNamespace(
        kv_receiver=receiver,
        req=SimpleNamespace(
            rid="r",
            bootstrap_room=1,
            time_stats=SimpleNamespace(
                decode_transfer_queue_entry_time=(
                    time.perf_counter() - age_s if entered else 0.0
                )
            ),
        ),
    )


class TestDecodeTransferDeadline(unittest.TestCase):
    def _run(self, reqs, deadline=120.0):
        # Only the fields the deadline sweep touches; the real __init__ wants a
        # scheduler and process groups.
        q = DecodeTransferQueue.__new__(DecodeTransferQueue)
        q.queue = reqs
        q.tp_rank = 0
        q._transfer_deadline_s = deadline
        q._fail_transfers_past_deadline()
        return [r.kv_receiver.aborted for r in reqs]

    def test_aborts_only_receivers_past_the_deadline(self):
        stale, fresh = make_req(300.0), make_req(5.0)
        self.assertEqual(self._run([stale, fresh]), [True, False])

    def test_leaves_a_receiver_that_already_concluded(self):
        # Success must not be re-aborted, and Failed is already on its way out.
        done = make_req(300.0, conclude_state=KVPoll.Success)
        failed = make_req(300.0, conclude_state=KVPoll.Failed)
        self.assertEqual(self._run([done, failed]), [False, False])

    def test_ignores_a_request_that_never_entered_the_queue(self):
        self.assertEqual(self._run([make_req(300.0, entered=False)]), [False])

    def test_deadline_of_zero_disables_the_sweep(self):
        self.assertEqual(self._run([make_req(300.0)], deadline=0.0), [False])


if __name__ == "__main__":
    unittest.main()
