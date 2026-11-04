# SPDX-License-Identifier: Apache-2.0
"""Compiled TP8/EP8 numerical coverage for the FP8 fused_moe_decode Pallas kernel."""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=4)


class TestFusedMoeDecodeKernel(CustomTestCase):
    def test_compiled_tp8_ep8_glm53_flash_shapes(self):
        repo_root = Path(__file__).resolve().parents[4]
        worker = repo_root / "test/manual/tpu/test_fused_moe_decode_kernel.py"
        command = "\n".join(
            (
                "export TPU_WORKER_HOSTNAMES=localhost",
                "eval $(python3 -m torch.tpu.distributed)",
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
            self.fail(f"fused_moe_decode TP8 test timed out:\n{error.stdout or ''}")

        self.assertEqual(
            result.returncode,
            0,
            f"fused_moe_decode TP8 test failed:\n{result.stdout}",
        )
        self.assertEqual(result.stdout.count("FUSED_MOE_DECODE_KERNEL_PASS"), 8)


if __name__ == "__main__":
    unittest.main()
