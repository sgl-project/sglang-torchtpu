"""Real TP x CP agreement for reservation ownership and delayed BUSY replies."""

import multiprocessing
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.reservation_slots import SlotReservations

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
WORLD_SIZE = 4


class _Sender:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.busy = False

    def decode_endpoint(self):
        return self.endpoint

    def reservation_busy(self):
        return self.busy

    def request_decode_reservation(self):
        self.busy = False


def _req(rid, endpoint="decode-a"):
    return SimpleNamespace(
        rid=rid, pending_bootstrap=True, disagg_kv_sender=_Sender(endpoint)
    )


def _rank(rank, store_path, results):
    dist.init_process_group(
        "gloo",
        init_method="file://" + store_path,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=20),
    )
    tp = cp = None
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            tp = group
    for ranks in ([0, 2], [1, 3]):
        group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            cp = group
    c = SlotReservations(limit=2, rank_groups=[tp, cp])
    a, b, healthy = _req("a"), _req("b"), _req("healthy", "decode-b")
    reqs = [a, b, healthy]
    log = []
    for step in range(7):
        if step == 1:
            for req in reqs:
                c.track(req)  # Intake occurs after the initial empty pass.
        queue = [] if step in (0, 6) else list(reqs)
        if rank % 2:
            queue.reverse()
        queue = c.canonical_order(queue)
        poll_ids = torch.tensor([c.track(r) for r in queue], dtype=torch.int64)
        if queue:
            dist.all_reduce(poll_ids)
            assert poll_ids.tolist() == [c.track(r) * WORLD_SIZE for r in queue]
        if step == 2:
            a.disagg_kv_sender.busy = rank == 0
        if step == 3:
            a.disagg_kv_sender.busy = True
        if step == 5:
            b.pending_bootstrap = False
            c.mark_ready(b)  # All-reduced READY, then first execution.
            queue = [r for r in queue if r is not b]
        c.begin(queue, now=step)
        order = c.agree_order(queue if rank == 0 else list(reversed(queue)))
        for req in order:
            c.consider(req)
        asks = c.finish(queue)
        log.append(
            ([r.rid for r in order], [r.rid for r in asks], sorted(c._outstanding))
        )
        next_poll = torch.tensor([step], dtype=torch.int64)
        dist.all_reduce(next_poll)
        assert next_poll.item() == step * WORLD_SIZE
    previous_id = -1
    for step in range(2):
        req = _req("reused-id")
        enqueue_id = c.track(req)
        assert enqueue_id > previous_id
        c.begin([req], now=10 + step)
        order = c.agree_order([req])
        c.consider(req)
        asks = c.finish([req])
        assert asks == [req]
        c.mark_ready(req)
        log.append(
            ([r.rid for r in order], [r.rid for r in asks], sorted(c._outstanding))
        )
        previous_id = enqueue_id
    results.put((rank, log))
    dist.destroy_process_group()


class TestReservationSlotsCollective(unittest.TestCase):
    def test_tp_cp_agree_through_empty_delayed_busy_retry_and_ready(self):
        ctx = multiprocessing.get_context("spawn")
        results = ctx.Queue()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "store")
            procs = [
                ctx.Process(target=_rank, args=(rank, path, results))
                for rank in range(WORLD_SIZE)
            ]
            for proc in procs:
                proc.start()
            try:
                logs = dict(results.get(timeout=90) for _ in procs)
            finally:
                for proc in procs:
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
            self.assertTrue(all(proc.exitcode == 0 for proc in procs))
        print("reservation rank histories:", logs, flush=True)
        for rank in range(1, WORLD_SIZE):
            self.assertEqual(logs[rank], logs[0])
        self.assertEqual(
            [asks for _, asks, _ in logs[0]],
            [
                [],
                ["a", "healthy"],
                [],
                ["b"],
                [],
                ["a"],
                [],
                ["reused-id"],
                ["reused-id"],
            ],
        )
        self.assertEqual(logs[0][2][2], ["a", "healthy"])
        self.assertEqual(logs[0][3][2], ["b", "healthy"])


if __name__ == "__main__":
    unittest.main()
