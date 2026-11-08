"""CPU checks of the optional prefix-fetcher hook of PrefillBootstrapQueue, on a
fake platform: fake senders and a fake fetcher, single rank and pipeline."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Sender:
    def __init__(self, state=KVPoll.WaitingForInput):
        self.state = state

    def poll(self):
        return self.state


class _Fetcher:
    """Fetches the requests named in `fetch`; records every call."""

    def __init__(self, fetch, recover=False):
        self.fetch = set(fetch)
        self.recover = recover
        self.held = set()
        self.calls = []

    def start(self, req):
        self.calls.append(("start", req.rid))
        if req.rid in self.fetch:
            self.held.add(req.rid)
            # Like a real fetcher, hold the sender until the fetch is ready.
            req.disagg_kv_sender.state = KVPoll.Bootstrapping
            return True
        return False

    def is_fetching(self, req):
        return req.rid in self.held

    def finish(self, req):
        self.calls.append(("finish", req.rid))
        self.held.discard(req.rid)

    def abandon(self, req):
        self.calls.append(("abandon", req.rid))
        self.held.discard(req.rid)
        return self.recover


def _req(rid):
    return SimpleNamespace(
        rid=rid,
        disagg_kv_sender=_Sender(),
        pending_bootstrap=True,
        prefill_attempt_count=0,
        is_retracted=False,
        time_stats=Mock(),
    )


class TestPrefixFetcherHook(unittest.TestCase):
    def _queue(self, reqs, fetcher, pp_size=1):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.pp_size = pp_size
        queue.scheduler_stage_metrics = None
        queue.prefix_fetcher = fetcher
        queue.finalized = []
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            processed_tokens_counter=0,
            handle_bootstrap_failure=lambda req: queue.finalized.append(
                ("failed", req.rid)
            ),
        )
        queue.ensure_metadata_buffer = Mock(return_value=True)

        def finalize(req):
            assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
            req.pending_bootstrap = False
            queue.finalized.append(("finalize", req.rid))
            return True

        queue.finalize_bootstrap = finalize
        queue.queue = list(reqs)
        return queue

    def _pop(self, queue, **pp):
        with patch(
            "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
            side_effect=lambda senders, *_: [s.poll() for s in senders],
        ):
            if queue.pp_size > 1:
                return queue.pop_bootstrapped(return_failed_reqs=True, **pp)
            return queue.pop_bootstrapped()

    def test_unset_fetcher_releases_at_bootstrap(self):
        a, b = _req("a"), _req("b")
        queue = self._queue([a, b], None)
        self.assertEqual(self._pop(queue), [a, b])
        self.assertEqual(queue.finalized, [("finalize", "a"), ("finalize", "b")])

    def test_fetched_request_returns_once_ready_and_finalizes_once(self):
        a, b = _req("a"), _req("b")
        fetcher = _Fetcher(fetch={"a"})
        queue = self._queue([a, b], fetcher)
        self.assertEqual(self._pop(queue), [b])
        self.assertEqual(queue.queue, [a])
        # Still fetching: the sender holds it out of WaitingForInput.
        self.assertEqual(self._pop(queue), [])
        a.disagg_kv_sender.state = KVPoll.WaitingForInput
        self.assertEqual(self._pop(queue), [a])
        self.assertEqual(queue.queue, [])
        self.assertEqual(
            fetcher.calls, [("start", "a"), ("start", "b"), ("finish", "a")]
        )
        self.assertEqual(queue.finalized, [("finalize", "a"), ("finalize", "b")])

    def test_failed_fetch_is_abandoned_before_the_failure_handling(self):
        a = _req("a")
        fetcher = _Fetcher(fetch={"a"})
        queue = self._queue([a], fetcher)
        self._pop(queue)
        a.disagg_kv_sender.state = KVPoll.Failed
        self.assertEqual(self._pop(queue), [])
        self.assertEqual(fetcher.calls[-1], ("abandon", "a"))
        self.assertEqual(queue.finalized[-1], ("failed", "a"))
        self.assertEqual(queue.queue, [])

    def test_recovered_fetch_is_released_without_failing(self):
        a = _req("a")
        fetcher = _Fetcher(fetch={"a"}, recover=True)
        queue = self._queue([a], fetcher)
        self._pop(queue)
        a.disagg_kv_sender.state = KVPoll.Failed
        self.assertEqual(self._pop(queue), [a])
        self.assertEqual(fetcher.calls[-1], ("abandon", "a"))
        self.assertNotIn(("failed", "a"), queue.finalized)

    def test_a_failure_before_any_fetch_skips_the_fetcher(self):
        a = _req("a")
        a.disagg_kv_sender.state = KVPoll.Failed
        fetcher = _Fetcher(fetch=set(), recover=True)
        queue = self._queue([a], fetcher)
        self.assertEqual(self._pop(queue), [])
        self.assertEqual(fetcher.calls, [])
        self.assertEqual(queue.finalized, [("failed", "a")])

    def test_pipeline_consensus_drives_both_passes(self):
        a = _req("a")
        fetcher = _Fetcher(fetch={"a"})
        queue = self._queue([a], fetcher, pp_size=2)
        good, failed = self._pop(queue, pp_good_rids=["a"], pp_bad_rids=[])
        self.assertEqual((good, failed), ([], []))
        # Not every stage is ready: no consensus, the request waits.
        good, _ = self._pop(queue, pp_good_rids=[], pp_bad_rids=[])
        self.assertEqual(good, [])
        good, _ = self._pop(queue, pp_good_rids=["a"], pp_bad_rids=[])
        self.assertEqual(good, [a])
        self.assertEqual(fetcher.calls, [("start", "a"), ("finish", "a")])


if __name__ == "__main__":
    unittest.main()
