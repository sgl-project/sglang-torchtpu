# SPDX-License-Identifier: Apache-2.0

"""Compiled TP8 numerical coverage for the FP8 fused-RS kernel."""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=360, suite="stage-a-test-tpu-unit")


class TestFp8FusedRsKernel(CustomTestCase):
    def test_compiled_tp8_kernel_matches_reference(self):
        repo_root = Path(__file__).resolve().parents[3]
        worker = repo_root / "test/manual/tpu/test_fp8_fused_rs_kernel.py"
        wrapper_module = "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
        command = "\n".join(
            (
                f"eval $(python3 -m {wrapper_module})",
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                f'torchrun --nproc_per_node="$WORLD_SIZE" {shlex.quote(str(worker))}',
            )
        )
        env = os.environ.copy()
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            env.pop(name, None)

        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"FP8 fused-RS TP8 test timed out:\n{error.stdout or ''}")

        self.assertEqual(
            result.returncode,
            0,
            f"FP8 fused-RS TP8 test failed:\n{result.stdout}",
        )
        self.assertEqual(result.stdout.count("FP8_FUSED_RS_KERNEL_PASS"), 8)


if __name__ == "__main__":
    unittest.main()
