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
                    move_resumed=lambda dst: [],
                    move_preallocated=Mock(return_value=([], [])),
                    publish_pending_destinations=Mock(),
                )
                arrivals = [[transferred], []]

                def move_transferred(dst):
                    reqs = arrivals.pop(0)
                    dst.extend(reqs)
                    return reqs

                scheduler = Scheduler.__new__(Scheduler)
                scheduler.scheduler_stage_metrics = None
                scheduler.enable_decode_hicache = False
                scheduler.enable_hisparse = False
                scheduler.tree_cache = SimpleNamespace(linker=None)
                scheduler.waiting_queue = []
                scheduler.disagg_decode_prealloc_queue = prealloc_queue
                transfer_queue = SimpleNamespace(
                    queue=[],
                    resolve_deferred_releases=Mock(),
                    publish_pending_metadata=Mock(),
                    move_transferred=move_transferred,
                )
                scheduler.disagg_decode_transfer_queue = transfer_queue

                scheduler.process_decode_queue()
                self.assertEqual(scheduler.waiting_queue, [transferred])
                prealloc_queue.move_preallocated.assert_not_called()

                # The retracted request now fits. Resume drains retracted_queue
                # inside the call, so admissions reopen in this same iteration.
                def move_resumed(dst):
                    resumed = [prealloc_queue.retracted_queue.pop()]
                    dst.extend(resumed)
                    return resumed

                prealloc_queue.move_resumed = move_resumed
                scheduler.process_decode_queue()
                self.assertEqual(scheduler.waiting_queue, [transferred, retracted])
                prealloc_queue.move_preallocated.assert_called_once_with(
                    transfer_queue.queue
                )


if __name__ == "__main__":
    unittest.main()
