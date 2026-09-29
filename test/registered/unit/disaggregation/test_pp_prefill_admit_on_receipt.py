"""PP disaggregated prefill admits requests on receipt when the sender needs no handshake."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base.conn import BaseKVSender  # noqa: E402
from sglang.srt.disaggregation.fake.conn import FakeKVSender  # noqa: E402
from sglang.srt.disaggregation.raiden.sender import RaidenKVSender  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAdmitOnReceipt(unittest.TestCase):
    def test_only_consumer_pull_senders_skip_the_handshake(self):
        self.assertTrue(BaseKVSender.requires_bootstrap_handshake)
        self.assertTrue(FakeKVSender.requires_bootstrap_handshake)
        self.assertFalse(RaidenKVSender.requires_bootstrap_handshake)

    def test_aborted_requests_are_flushed_and_the_rest_admitted(self):
        live = SimpleNamespace(rid="live", finished_reason=None)
        aborted = SimpleNamespace(rid="aborted", finished_reason=FINISH_ABORT())
        queue = SimpleNamespace(
            queue=[live, aborted],
            pop_bootstrapped=Mock(return_value=([live], [aborted])),
        )
        scheduler = SchedulerPPMixin()
        scheduler.disagg_prefill_bootstrap_queue = queue
        scheduler.waiting_queue = []

        scheduler._pp_pd_admit_on_receipt()

        queue.pop_bootstrapped.assert_called_once_with(
            return_failed_reqs=True, pp_good_rids=["live"], pp_bad_rids=["aborted"]
        )
        self.assertEqual(scheduler.waiting_queue, [live])


if __name__ == "__main__":
    unittest.main()
