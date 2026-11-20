"""The Kimi-K3 programs' host-side unit tests in the TPU nightly, one job.

These modules need jax and torch_tpu, which the CPU runners do not install,
and the TPU unit stage's time budget is spent; each registers itself disabled
with a pointer here. This test runs every module in its own process, as
run_suite runs a test file, with jax on the host's CPU (the modules check host
logic and host meshes, not the TPU the job holds), and fails if one fails or
skips all of its tests.
"""

import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=1200,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "kimi-k3", "k3-programs"],
)

MODULES = (
    "test_k3_decode_compiler_options.py",
    "test_k3_expert_steal_layout.py",
    "test_k3_program_build.py",
    "test_k3_program_colocated.py",
    "test_k3_program_dp.py",
    "test_k3_program_first_drafts.py",
    "test_k3_program_hicache.py",
    "test_k3_program_host_relay.py",
    "test_k3_program_k31.py",
    "test_k3_program_pools.py",
    "test_k3_program_prefill_plan.py",
    "test_k3_program_tails.py",
    "test_k3_program_worker.py",
    "test_k3_programs_bridge.py",
    "test_mla_sparse_reciprocal.py",
)


class TestK3ProgramsNightly(CustomTestCase):
    def test_modules(self):
        here = Path(__file__).resolve().parent
        for name in MODULES:
            with self.subTest(module=name):
                start = time.monotonic()
                result = subprocess.run(
                    [sys.executable, str(here / name)],
                    env=dict(os.environ, JAX_PLATFORMS="cpu"),
                    capture_output=True,
                    text=True,
                    timeout=1500,
                )
                output = (result.stdout + result.stderr)[-6000:]
                print(
                    f"{name}: exit {result.returncode}, "
                    f"{time.monotonic() - start:.0f} s",
                    flush=True,
                )
                self.assertEqual(result.returncode, 0, output)
                ran = re.search(r"^Ran (\d+) tests?", output, re.M)
                skipped = re.search(r"skipped=(\d+)", output)
                self.assertIsNotNone(ran, output)
                self.assertLess(
                    int(skipped.group(1)) if skipped else 0, int(ran.group(1)), output
                )


if __name__ == "__main__":
    unittest.main()
