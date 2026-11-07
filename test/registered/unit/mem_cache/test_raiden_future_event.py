"""CPU-only unit tests for RaidenFutureEvent's blocking-flush contract.

`synchronize` deliberately polls `is_ready` rather than calling
`RaidenFuture.Await`, because `Await` (PJRT_Event_Await / xla::Future
BlockUntilReady) can deadlock against the live model. The cost of polling is
that the loop needs its own deadline, and that deadline is the only thing
standing between a DMA that never lands and a scheduler that spins forever.

That branch is unreachable from configuration. `SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S`
was lowered from 60s to 2s and then to 0.001s on real TPU hardware and no
timeout ever fired, because the deadline is only consulted *inside* the
not-ready loop: a future that is already ready on the first poll skips the body
entirely. Forcing the branch needs a future that never becomes ready, which is a
stub, not a setting -- hence a unit test rather than an e2e arm.

Covers: a stuck DMA raises TimeoutError instead of hanging; the timeout is
honoured from the env var; a ready future returns promptly and does not consult
the clock; a failed-but-ready future reports its error rather than timing out;
and the no-op event surface (record/wait/None future) stays inert.
"""

import threading
import time
import unittest

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.raiden_transfer import RaidenFutureEvent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_TIMED_OUT = object()


def _run_bounded(fn, limit_s):
    """Call fn() on a daemon thread; return its exception, None, or _TIMED_OUT.

    Lets a "never returns" regression surface as an assertion rather than as a
    hung CI job. The thread is left to die with the process -- it is stuck by
    construction and there is nothing to clean up.
    """
    box = []

    def run():
        try:
            fn()
            box.append(None)
        except BaseException as exc:  # noqa: BLE001 - the exception IS the result
            box.append(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(limit_s)
    return box[0] if box else _TIMED_OUT


class _Future:
    """Minimal stand-in for a raiden future.

    ready_after=None never becomes ready, which is the stuck-DMA case the
    deadline exists for.
    """

    def __init__(self, ready_after=0.0, ok=True, error="stub failure"):
        self._ready_after = ready_after
        self._t0 = time.monotonic()
        self._ok = ok
        self._error = error
        self.poll_count = 0

    def is_ready(self):
        self.poll_count += 1
        if self._ready_after is None:
            return False
        return time.monotonic() - self._t0 >= self._ready_after

    def ok(self):
        return self._ok

    def error_message(self):
        return self._error


class TestRaidenFutureEventSynchronize(unittest.TestCase):
    def test_stuck_dma_raises_instead_of_spinning(self):
        stuck = _Future(ready_after=None)
        event = RaidenFutureEvent(stuck)
        with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(0.05):
            t0 = time.monotonic()
            # Run on a worker and join with a bound. The regression this guards
            # is "spins forever", so calling synchronize() directly would make a
            # broken build HANG the CI job instead of failing it -- verified by
            # deleting the deadline check, which turns a 12 s suite into a
            # timeout kill. The worker converts that into a clean failure.
            raised = _run_bounded(event.synchronize, 10.0)
            elapsed = time.monotonic() - t0
        self.assertIsNot(
            raised,
            _TIMED_OUT,
            "synchronize() never returned: the polling loop has no deadline",
        )
        self.assertIsInstance(raised, TimeoutError)
        self.assertIn("did not complete", str(raised))
        self.assertLess(elapsed, 5.0)
        self.assertGreater(stuck.poll_count, 1)

    def test_timeout_is_taken_from_the_env_var(self):
        """A longer budget must wait longer -- the deadline is not hardcoded.

        Bounded like the stuck-DMA case: every test that calls synchronize() on a
        never-ready future can hang a broken build, not just the one that names
        the hang. Running the file against a deadline-less synchronize() is what
        caught this -- the protected test failed in 19 s and then this one hung.
        """
        durations = {}
        for budget in (0.05, 0.30):
            event = RaidenFutureEvent(_Future(ready_after=None))
            with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(budget):
                t0 = time.monotonic()
                raised = _run_bounded(event.synchronize, 10.0)
                durations[budget] = time.monotonic() - t0
            self.assertIsNot(
                raised,
                _TIMED_OUT,
                f"synchronize() never returned at budget {budget}s: no deadline",
            )
            self.assertIsInstance(raised, TimeoutError)
        self.assertGreater(durations[0.30], durations[0.05])
        self.assertGreaterEqual(durations[0.30], 0.30)

    def test_ready_future_returns_without_waiting(self):
        event = RaidenFutureEvent(_Future(ready_after=0.0))
        with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(0.05):
            t0 = time.monotonic()
            event.synchronize()
            self.assertLess(time.monotonic() - t0, 0.05)

    def test_slow_but_landing_dma_is_not_a_timeout(self):
        event = RaidenFutureEvent(_Future(ready_after=0.10))
        with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(5.0):
            event.synchronize()  # must not raise

    def test_timeout_leaves_the_event_usable_and_uncorrupted(self):
        """After a timeout the event must still be a valid, pollable object.

        Modelled on test_raiden_late_pull.py, which asserts the state a failed
        transfer leaves behind (no pull issued, room marked failed, reason
        recorded, slot not consumed) rather than only that it failed. The
        equivalent here: synchronize() must not consume or mutate the future on
        its way out, because the caller still holds the event and the ack that
        owns it stays on the queue -- writing_check's `ack_write_queue.clear()`
        sits after the synchronize() that raised, so the ack is retained.
        """
        stuck = _Future(ready_after=None)
        event = RaidenFutureEvent(stuck)
        with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(0.05):
            self.assertIsInstance(_run_bounded(event.synchronize, 10.0), TimeoutError)

            # the future is still attached and still pollable, not cleared
            self.assertIs(event._future, stuck)
            self.assertFalse(event.query())

            # and the timeout is repeatable rather than one-shot: a caller that
            # retries gets the same bounded failure, not a hang or a false pass
            again = _run_bounded(event.synchronize, 10.0)
            self.assertIsNot(again, _TIMED_OUT)
            self.assertIsInstance(again, TimeoutError)

    def test_timeout_does_not_mask_itself_as_a_dma_failure(self):
        """A pending DMA must time out, never surface as `raiden DMA failed`.

        `_raise_if_failed` reads `ok()`, which a stub could answer either way
        while still pending. Only the ready path is allowed to consult it, so a
        timeout on a future whose ok() is False must still be a TimeoutError --
        otherwise a slow transfer would be misreported as a corrupt one and
        sent down the wrong recovery path.
        """
        pending_and_unhappy = _Future(ready_after=None, ok=False, error="not reached")
        event = RaidenFutureEvent(pending_and_unhappy)
        with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(0.05):
            raised = _run_bounded(event.synchronize, 10.0)
        self.assertIsInstance(raised, TimeoutError)
        self.assertNotIsInstance(raised, RuntimeError)
        self.assertNotIn("not reached", str(raised))

    def test_failed_future_reports_its_error_not_a_timeout(self):
        event = RaidenFutureEvent(_Future(ready_after=0.0, ok=False, error="boom"))
        with envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.override(5.0):
            with self.assertRaises(RuntimeError) as caught:
                event.synchronize()
        self.assertIn("boom", str(caught.exception))
        self.assertNotIsInstance(caught.exception, TimeoutError)


class TestRaidenFutureEventSurface(unittest.TestCase):
    """The object stands in for a device Event in HiCacheAck, so the rest of the
    event API must be inert rather than absent."""

    def test_query_is_nonblocking_and_tracks_readiness(self):
        pending = _Future(ready_after=None)
        self.assertFalse(RaidenFutureEvent(pending).query())
        self.assertTrue(RaidenFutureEvent(_Future(ready_after=0.0)).query())

    def test_query_surfaces_a_failed_dma(self):
        event = RaidenFutureEvent(_Future(ready_after=0.0, ok=False, error="bad"))
        with self.assertRaises(RuntimeError):
            event.query()

    def test_none_future_is_a_completed_noop(self):
        event = RaidenFutureEvent(None)
        self.assertTrue(event.query())
        event.synchronize()  # must not raise or block

    def test_record_and_wait_are_noops(self):
        event = RaidenFutureEvent(_Future(ready_after=None))
        self.assertIsNone(event.record())
        self.assertIsNone(event.wait())


if __name__ == "__main__":
    unittest.main()
