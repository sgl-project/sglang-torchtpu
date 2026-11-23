"""PD health scheduling over a real eight-rank TPU collective forward.

The shared finite workload checks rank agreement and pool ownership in both
scheduler loops. KV/transfer buffers remain host-resident; this component test
covers TPU collective completion before health acknowledgement, not model
accuracy or native Raiden DMA.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=600,
    suite="stage-a-test-tpu-unit",
    chips=4,
    labels=["singlehost", "infra", "pd", "collectives"],
)

_WORKER = "--pd-health-worker"
_LAUNCHERS = (
    "torch_tpu._internal.distributed.distributed",
    "torch_tpu._internal.distributed.environment",
    "torch_tpu._internal.distributed.launchers.singlehost_wrapper",
)


def _launch_environment():
    environment = os.environ.copy()
    for key in (
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "JOB_COMPLETION_INDEX",
    ):
        environment.pop(key, None)
    for module in _LAUNCHERS:
        try:
            exists = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            exists = False
        if exists:
            output = subprocess.check_output(
                [sys.executable, "-m", module], env=environment, text=True, timeout=60
            )
            break
    else:
        raise RuntimeError("No supported torch_tpu distributed launcher installed")
    keys = {"TORCH_TPU_TOPOLOGY", "TORCH_TPU_SLICEBUILDER_ADDRESSES", "WORLD_SIZE"}
    found = set()
    for line in output.splitlines():
        key, separator, value = line.removeprefix("export ").partition("=")
        if separator and key in keys:
            environment[key] = shlex.split(value)[0]
            found.add(key)
    assert found == keys, found
    assert int(environment["WORLD_SIZE"]) == 8, "Requires one eight-device TPU host"
    return environment


def _run_worker():
    import faulthandler

    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401

    from sglang.test.pd_health_probe_harness import run_pd_health_probe_scenarios

    faulthandler.dump_traceback_later(600, exit=True)
    dist.init_process_group("tpu_dist")
    cpu_group = dist.new_group(backend="gloo")
    try:

        def forward(batch_size):
            values = torch.full(
                (batch_size, 8), float(dist.get_rank() + 1), device="tpu"
            )
            work = dist.all_reduce(values, async_op=True)
            return work, values

        run_pd_health_probe_scenarios(cpu_group=cpu_group, forward=forward)
        dist.barrier(group=cpu_group)
        if dist.get_rank() == 0:
            print(
                "PD_HEALTH_MULTIRANK_PASS: all six scenarios, eight ranks", flush=True
            )
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


class TestPDHealthProbeTPUMultiRank(CustomTestCase):
    def test_normal_and_overlap_loops_agree_and_release_capacity(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=8",
                str(Path(__file__).resolve()),
                _WORKER,
            ],
            env=_launch_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=780)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            self.fail("TPU health regression timed out:\n" + output)
        print(output)
        self.assertEqual(process.returncode, 0, output)
        self.assertIn("PD_HEALTH_MULTIRANK_PASS", output)


if __name__ == "__main__":
    if _WORKER in sys.argv:
        _run_worker()
    else:
        unittest.main()
