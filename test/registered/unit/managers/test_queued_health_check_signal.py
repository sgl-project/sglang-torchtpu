"""Health probes queued behind a chunked prefill are answered after real work.

A probe admitted while the scheduler looks idle can still queue behind a long
prompt (on a PD prefill both leave the bootstrap queue in the same poll). Each
chunk then fills the prefill budget, so the probe only runs after the chain.
After each prefill batch with real work the scheduler answers every queued probe
with a HealthCheckOutput, once per probe, and only if a batch completes (#1325).
"""

import unittest
from collections import deque
from types import SimpleNamespace

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import HealthCheckOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PROBE_RID = f"{HEALTH_CHECK_RID_PREFIX}_probe"
PROBE_IPC = "ipc:///tmp/http_worker_0"


class _QueueThatMustNotBeScanned(list):
    def __iter__(self):
        raise AssertionError("queue scanned after a batch that proves nothing")


def _recv(rid, http_worker_ipc=None):
    return SimpleNamespace(rid=rid, http_worker_ipc=http_worker_ipc)


def _req(rid, http_worker_ipc=None):
    return SimpleNamespace(rid=rid, http_worker_ipc=http_worker_ipc)


def _batch(*reqs, forward_mode=ForwardMode.EXTEND):
    return SimpleNamespace(reqs=list(reqs), forward_mode=forward_mode)


class TestQueuedHealthCheckSignal(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))

    def _scheduler(self, mode=DisaggregationMode.PREFILL):
        scheduler = Scheduler.__new__(Scheduler)
        self.sent = []
        scheduler.scheduler_stage_metrics = None
        scheduler.disaggregation_mode = mode
        scheduler.waiting_queue = []
        scheduler.disagg_prefill_bootstrap_queue = (
            SimpleNamespace(queue=[]) if mode == DisaggregationMode.PREFILL else None
        )
        # State read by the real is_fully_idle(for_health_check=True).
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: True)
        scheduler.chunked_req = None
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._engine_paused = False
        scheduler.return_health_check_ipcs = deque()
        # process_input_requests collaborators outside the code under test.
        scheduler.session_controller = SimpleNamespace(maybe_reap=lambda now: None)
        scheduler.rust_server = None
        scheduler.flush_wrapper = SimpleNamespace(check_pending=lambda: None)
        scheduler.external_corpus_manager = None
        scheduler._request_dispatcher = lambda recv_req: self._enqueue(
            scheduler, recv_req
        )
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda output, recv_obj=None: self.sent.append(output)
            )
        )
        return scheduler

    @staticmethod
    def _enqueue(scheduler, recv_req):
        # handle_generate_request -> _add_request_to_queue: a PD prefill parks
        # every new request in the bootstrap queue first.
        req = _req(recv_req.rid, recv_req.http_worker_ipc)
        if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            scheduler.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            scheduler.waiting_queue.append(req)

    @staticmethod
    def _pop_bootstrapped(scheduler):
        # pop_bootstrapped keeps the bootstrap queue's order.
        scheduler.waiting_queue.extend(scheduler.disagg_prefill_bootstrap_queue.queue)
        scheduler.disagg_prefill_bootstrap_queue.queue.clear()

    def signaled(self):
        for output in self.sent:
            self.assertIsInstance(output, HealthCheckOutput)
        return [(output.rid, output.http_worker_ipc) for output in self.sent]

    def test_probe_behind_chunked_prompt_is_signaled_once(self):
        scheduler = self._scheduler()
        # The long prompt sits in the bootstrap queue, which the health-check
        # idle test ignores, so the probe in the same pass is admitted.
        scheduler.process_input_requests([_recv("long"), _recv(PROBE_RID, PROBE_IPC)])
        self.assertEqual(len(scheduler.return_health_check_ipcs), 0)
        self.assertEqual(self.sent, [])  # admission alone proves nothing

        self._pop_bootstrapped(scheduler)
        long_req, probe = scheduler.waiting_queue
        self.assertEqual(probe.rid, PROBE_RID)
        # First chunk takes the whole budget; the probe stays queued.
        scheduler.waiting_queue.remove(long_req)
        scheduler.maybe_signal_queued_health_checks(_batch(long_req))

        self.assertEqual(self.signaled(), [(PROBE_RID, PROBE_IPC)])
        self.assertEqual(scheduler.waiting_queue, [probe])

        # The rest of the 47-chunk chain does not repeat the signal.
        for _ in range(46):
            scheduler.maybe_signal_queued_health_checks(_batch(long_req))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(scheduler.waiting_queue, [probe])

    def test_probe_still_bootstrapping_is_signaled_on_pd_prefill(self):
        scheduler = self._scheduler()
        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])

        scheduler.maybe_signal_queued_health_checks(_batch(_req("other")))

        self.assertEqual(self.signaled(), [(PROBE_RID, PROBE_IPC)])

    def test_non_disagg_scans_only_waiting_queue(self):
        # disagg_prefill_bootstrap_queue is None here, so touching it raises.
        scheduler = self._scheduler(mode=DisaggregationMode.NULL)
        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])

        scheduler.maybe_signal_queued_health_checks(
            _batch(_req("long"), forward_mode=ForwardMode.MIXED)
        )

        self.assertEqual(self.signaled(), [(PROBE_RID, PROBE_IPC)])

    def test_each_queued_probe_is_signaled_once_by_rid(self):
        scheduler = self._scheduler(mode=DisaggregationMode.NULL)
        first = _req(f"{HEALTH_CHECK_RID_PREFIX}_a", PROBE_IPC)
        second = _req(f"{HEALTH_CHECK_RID_PREFIX}_b", PROBE_IPC)
        other_worker = _req(f"{HEALTH_CHECK_RID_PREFIX}_c", "ipc:///tmp/http_worker_1")
        scheduler.waiting_queue = [first, _req("queued"), second, other_worker]

        scheduler.maybe_signal_queued_health_checks(_batch(_req("long")))
        self.assertEqual(
            self.signaled(),
            [(r.rid, r.http_worker_ipc) for r in (first, second, other_worker)],
        )

        # A probe queued later is answered by the next batch; earlier ones are not
        # signaled again.
        late = _req(f"{HEALTH_CHECK_RID_PREFIX}_d", PROBE_IPC)
        scheduler.waiting_queue.append(late)
        self.sent.clear()
        scheduler.maybe_signal_queued_health_checks(_batch(_req("long")))
        self.assertEqual(self.signaled(), [(late.rid, PROBE_IPC)])

    def test_batches_that_prove_nothing_do_not_scan(self):
        scheduler = self._scheduler(mode=DisaggregationMode.NULL)
        scheduler.waiting_queue = _QueueThatMustNotBeScanned(
            [_req(PROBE_RID, PROBE_IPC)]
        )

        # Decode steps already stream output, speculative verify/draft steps are
        # not prefill work, idle batches did no work, and a probe-only prefill
        # answers its own probes.
        scheduler.maybe_signal_queued_health_checks(
            _batch(_req("running"), forward_mode=ForwardMode.DECODE)
        )
        scheduler.maybe_signal_queued_health_checks(
            _batch(_req("running"), forward_mode=ForwardMode.TARGET_VERIFY)
        )
        scheduler.maybe_signal_queued_health_checks(
            _batch(_req("running"), forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        )
        scheduler.maybe_signal_queued_health_checks(
            _batch(forward_mode=ForwardMode.IDLE)
        )
        scheduler.maybe_signal_queued_health_checks(_batch())
        scheduler.maybe_signal_queued_health_checks(
            _batch(_req(f"{HEALTH_CHECK_RID_PREFIX}_other"))
        )

        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
