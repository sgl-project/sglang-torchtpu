"""Transfer-status polling must fail, not hang, when attention ranks diverge.

The prefill scheduler MIN-reduces every queued request's KV-transfer status
across its attention TP/CP ranks on each loop iteration. A rank that never
reaches that collective (blocked in a native poll, holding a lock) or that
reaches it with a different number of requests parks every peer in the gloo
all_reduce until the CPU group's multi-hour timeout; forwards stop, nothing
is logged, and health endpoints that bypass generation keep answering.
"""

import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce_attn_cp_tp_group
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=150, suite="base-a-test-cpu")

WORLD_SIZE = 4
EXIT_COMPLETED = 0
EXIT_RAISED = 7
EXIT_POLL_RAISED = 8


class _Poller:
    def __init__(self, state=KVPoll.WaitingForInput, block=False, raise_=False):
        self._state = state
        self._block = block
        self._raise = raise_

    def poll(self):
        if self._block:
            time.sleep(1e9)
        if self._raise:
            raise RuntimeError("native poll failed")
        return self._state


def _rank_main(
    rank, rendezvous, ready, done, scenario, reduce_timeout_s, check_count, linger
):
    os.environ["SGLANG_DISAGGREGATION_POLL_ALLREDUCE_TIMEOUT"] = str(reduce_timeout_s)
    os.environ["SGLANG_DISAGGREGATION_POLL_CHECK_COUNT"] = "1" if check_count else "0"
    # Mirrors production, where the CPU group's own timeout is hours.
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=600),
    )
    pollers = [_Poller() for _ in range(3)]
    odd_rank = WORLD_SIZE - 1
    if scenario == "blocked" and rank == odd_rank:
        pollers[1] = _Poller(block=True)
    elif scenario == "raise" and rank == odd_rank:
        pollers[1] = _Poller(raise_=True)
    elif scenario == "mismatch" and rank == 0:
        pollers.pop()
    # One CP rank per TP rank, as in the incident topology; new_group is itself
    # a collective, so every rank creates every singleton.
    singletons = [dist.new_group([r]) for r in range(WORLD_SIZE)]
    dist.barrier()
    ready.set()
    try:
        poll_and_all_reduce_attn_cp_tp_group(
            pollers, singletons[rank], dist.group.WORLD
        )
    except RuntimeError as e:
        code = EXIT_POLL_RAISED if "native poll failed" in str(e) else EXIT_RAISED
        print(f"rank {rank}: {e}", file=sys.stderr, flush=True)
        sys.exit(code)
    done.set()
    if linger:
        # A production rank that completes keeps running its loop, so its gloo
        # connections stay open; exiting here would instead make the peers'
        # collective fail with "connection closed by peer".
        time.sleep(1e9)
    sys.exit(EXIT_COMPLETED)


@unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
class TestTransferPollRankDesync(CustomTestCase):
    def _run(self, scenario, *, reduce_timeout_s, check_count, wait_s, linger=False):
        """Per rank (exit code, completed): exit code is None for a rank still
        running wait_s after every rank has initialized and is about to poll."""
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            ready = [context.Event() for _ in range(WORLD_SIZE)]
            done = [context.Event() for _ in range(WORLD_SIZE)]
            processes = [
                context.Process(
                    target=_rank_main,
                    args=(
                        rank,
                        rendezvous,
                        ready[rank],
                        done[rank],
                        scenario,
                        reduce_timeout_s,
                        check_count,
                        linger,
                    ),
                )
                for rank in range(WORLD_SIZE)
            ]
            try:
                for process in processes:
                    process.start()
                for event in ready:
                    self.assertTrue(event.wait(timeout=120), "rank did not start")
                deadline = time.monotonic() + wait_s
                while time.monotonic() < deadline and any(
                    p.is_alive() for p in processes
                ):
                    time.sleep(0.1)
                return [p.exitcode for p in processes], [e.is_set() for e in done]
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)

    def test_agreeing_ranks_complete(self):
        codes, _ = self._run("none", reduce_timeout_s=5, check_count=True, wait_s=30)
        self.assertEqual(codes, [EXIT_COMPLETED] * WORLD_SIZE)

    def test_blocked_rank_hangs_every_peer_without_bound(self):
        """The failure mode: no rank returns, raises or logs."""
        codes, _ = self._run("blocked", reduce_timeout_s=0, check_count=False, wait_s=6)
        self.assertEqual(codes, [None] * WORLD_SIZE)

    def test_blocked_rank_makes_peers_raise_within_bound(self):
        codes, _ = self._run(
            "blocked", reduce_timeout_s=2, check_count=False, wait_s=30
        )
        self.assertEqual(codes[:-1], [EXIT_RAISED] * (WORLD_SIZE - 1))
        self.assertIsNone(codes[-1])

    def test_poll_count_mismatch_hangs_longer_ranks_without_check(self):
        """Gloo completes the shortest rank with a bogus result; peers wait."""
        codes, done = self._run(
            "mismatch", reduce_timeout_s=0, check_count=False, wait_s=6, linger=True
        )
        self.assertEqual(codes, [None] * WORLD_SIZE)
        self.assertEqual(done, [True] + [False] * (WORLD_SIZE - 1))

    def test_poll_count_mismatch_raises_on_every_rank_with_check(self):
        codes, _ = self._run(
            "mismatch", reduce_timeout_s=5, check_count=True, wait_s=30
        )
        self.assertEqual(codes, [EXIT_RAISED] * WORLD_SIZE)

    def test_raising_poll_makes_peers_raise_within_bound(self):
        codes, _ = self._run("raise", reduce_timeout_s=2, check_count=True, wait_s=30)
        self.assertEqual(codes[:-1], [EXIT_RAISED] * (WORLD_SIZE - 1))
        self.assertEqual(codes[-1], EXIT_POLL_RAISED)


if __name__ == "__main__":
    unittest.main()
