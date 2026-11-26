"""Real two-process PD normal/overlap scheduler regression on CPU."""

from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.pd_health_probe_harness import run_pd_health_probe_scenarios
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        run_pd_health_probe_scenarios()
    finally:
        dist.destroy_process_group()


class TestPDHealthProbeMultiRank(CustomTestCase):
    def test_normal_and_overlap_decode_loops_agree_and_release_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            rendezvous = "file://" + str(Path(directory) / "rendezvous")
            processes = [
                context.Process(target=_worker, args=(rank, rendezvous))
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(180)
                self.assertEqual([p.exitcode for p in processes], [0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(10)


if __name__ == "__main__":
    unittest.main()
