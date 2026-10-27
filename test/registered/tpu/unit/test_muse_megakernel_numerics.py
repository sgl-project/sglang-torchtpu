# SPDX-License-Identifier: Apache-2.0

"""TP8 numerics of the Muse Spark megakernel against a teacher-forced CPU reference.

The worker (test/manual/tpu/test_muse_megakernel_numerics.py) runs the compiled op on random
serving-layout weights over fused-bf16, SAL-bf16 and SAL-fp8 KV pools, decode and verify
steps, short and 20k contexts, and checks every layer's residual stream, the KV rows the step
writes, the final norm, routing (near-ties excepted) and that exact-by-construction MKConfig
flags leave the outputs bit-identical.
"""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=420, suite="stage-a-test-tpu-unit", chips=4)


class TestMuseMegakernelNumerics(CustomTestCase):
    def test_kernel_matches_reference_decoder(self):
        repo_root = Path(__file__).resolve().parents[4]
        worker = repo_root / "test/manual/tpu/test_muse_megakernel_numerics.py"
        command = "\n".join(
            (
                "eval $(python3 -m torch.tpu.distributed)",
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                f'torchrun --nproc_per_node="$WORLD_SIZE" {shlex.quote(str(worker))}',
            )
        )
        env = os.environ.copy()
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            env.pop(name, None)
        # The inductor/AOT cache key misses Pallas kernel bodies: a cached executable of an
        # older kernel would be replayed.
        env.pop("SGLANG_CACHE_DIR", None)
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1500,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"Muse megakernel numerics timed out:\n{error.stdout or ''}")
        report = "\n".join(
            line for line in result.stdout.splitlines() if "MK_NUMERICS" in line
        )
        self.assertEqual(
            result.returncode, 0, f"worker failed:\n{report}\n{result.stdout[-5000:]}"
        )
        self.assertIn("MK_NUMERICS_PASS", report, report)
        self.assertEqual(report.count("MK_NUMERICS_DONE"), 8, report)


if __name__ == "__main__":
    unittest.main()
