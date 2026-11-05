"""The drain gate's reason exchange over real torch.distributed gloo groups.

Spawns separate processes (one per rank) so ``all_gather_object`` runs for
real, including the sequential TP-then-CP legs that let a reason cross from a
rank in another TP group.
"""

import os
import socket
import unittest

import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PLACEHOLDER = "failed on another rank"
REFUSAL = "rank {} refused the room"
JOIN_TIMEOUT_S = 90


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Poller:
    """A drain-gated poller holding one room's reason the way Raiden does."""

    group_failure_requires_drain = True

    def __init__(self, reason):
        self.reason = reason
        self.adopted = False

    def quiesce_group_failure(self):
        if self.reason is None:
            self.reason = PLACEHOLDER

    def drained(self):
        return True

    def group_failure_reason(self):
        return None if self.reason == PLACEHOLDER else self.reason

    def adopt_group_failure_reason(self, reason):
        if self.reason == PLACEHOLDER:
            self.reason = reason
            self.adopted = True


def _worker(rank, world, port, topology, originating, results):
    import torch.distributed as dist

    from sglang.srt.disaggregation.base import KVPoll
    from sglang.srt.disaggregation.utils import _gate_group_failure_on_drain

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    try:
        if topology == "flat":
            groups = [dist.new_group(list(range(world)))]
        else:
            # world 4 as TP 2 x CP 2: TP groups share a CP index, CP groups
            # share a TP index. Every rank creates every group (collective).
            tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
            groups = [tp_groups[rank // 2], cp_groups[rank % 2]]
        # Room 0: one originating rank. Room 1: nobody originated.
        pollers = [
            _Poller(REFUSAL.format(rank) if rank == originating else None),
            _Poller(None),
        ]
        polls = _gate_group_failure_on_drain(
            [KVPoll.Failed, KVPoll.Failed], pollers, groups
        )
        results.put(
            (
                rank,
                [int(p) for p in polls],
                [p.reason for p in pollers],
                [p.adopted for p in pollers],
            )
        )
    finally:
        dist.destroy_process_group()


def _run(world, topology, originating):
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_worker, args=(rank, world, port, topology, originating, results)
        )
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    out = {}
    for _ in procs:
        rank, polls, reasons, adopted = results.get(timeout=JOIN_TIMEOUT_S)
        out[rank] = (polls, reasons, adopted)
    for proc in procs:
        proc.join(timeout=JOIN_TIMEOUT_S)
        if proc.is_alive():
            proc.kill()
    return out


class TestGroupFailureReasonOverGloo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            raise unittest.SkipTest("torch.distributed gloo is unavailable")
        # The spawned ranks only need CPU torch; keep the TPU backend out of
        # them so they start fast and do not contend for the device.
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    def test_two_ranks_share_the_originating_reason(self):
        out = _run(world=2, topology="flat", originating=1)

        for rank in (0, 1):
            polls, reasons, adopted = out[rank]
            self.assertEqual(polls, [0, 0], rank)  # KVPoll.Failed == 0
            self.assertEqual(reasons, [REFUSAL.format(1), PLACEHOLDER], rank)
        self.assertEqual(out[0][2], [True, False])
        self.assertEqual(out[1][2], [False, False])

    def test_reason_crosses_tp_groups_through_the_cp_leg(self):
        # Rank 3 (TP group {2,3}, CP group {1,3}) originates. Rank 0 shares
        # no group with it: its TP leg brings nothing, and its CP peer rank 2
        # carries rank 3's reason out of the TP leg.
        out = _run(world=4, topology="tp_cp", originating=3)

        for rank in range(4):
            polls, reasons, adopted = out[rank]
            self.assertEqual(polls, [0, 0], rank)
            self.assertEqual(reasons, [REFUSAL.format(3), PLACEHOLDER], rank)
            self.assertEqual(adopted, [rank != 3, False], rank)


if __name__ == "__main__":
    unittest.main()
