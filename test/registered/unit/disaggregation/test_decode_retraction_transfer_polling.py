"""A retracted decode request must not stop the transfer queue from being polled."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRetractionDefersOnlyNewAdmissions(CustomTestCase):
    def test_transferred_rows_join_while_new_rows_wait(self):
        for threshold in (0.0, 0.5):
            with (
                self.subTest(host_receive_threshold=threshold),
                get_context().override_server_args(
                    disaggregation_decode_host_receive_threshold=threshold
                ),
            ):
                transferred = SimpleNamespace(rid="transferred")
                retracted = SimpleNamespace(rid="retracted")
                prealloc_queue = SimpleNamespace(
                    retracted_queue=[retracted],
                    resume_retracted_reqs=lambda: [],
                    pop_preallocated=Mock(return_value=([], [])),
                )
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.scheduler_stage_metrics = None
                scheduler.enable_decode_hicache = False
                scheduler.enable_hisparse = False
                scheduler.waiting_queue = []
                scheduler.disagg_decode_prealloc_queue = prealloc_queue
                scheduler.disagg_decode_transfer_queue = SimpleNamespace(
                    resolve_deferred_releases=Mock(),
                    pop_transferred=Mock(side_effect=[[transferred], []]),
                    extend=Mock(),
                )

                scheduler.process_decode_queue()
                self.assertEqual(scheduler.waiting_queue, [transferred])
                prealloc_queue.pop_preallocated.assert_not_called()

                # The retracted request now fits. Resume drains retracted_queue
                # inside the call, so admissions reopen in this same iteration.
                prealloc_queue.resume_retracted_reqs = lambda: [
                    prealloc_queue.retracted_queue.pop()
                ]
                scheduler.process_decode_queue()
                self.assertEqual(scheduler.waiting_queue, [transferred, retracted])
                prealloc_queue.pop_preallocated.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
