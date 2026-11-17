# SPDX-License-Identifier: Apache-2.0

"""Compiled TP8 numerics of the expert-TP NVFP4 fused-RS MoE (Muse Spark geometry)."""

import os
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# Each case is its own TP8 torchrun (weight prep plus two compiled token counts, ~100 s).
register_tpu_ci(
    est_time=450,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "muse-spark-open-816b-a42b", "tp"],
)


class TestNvfp4ExpertTpFusedRs(CustomTestCase):
    def _run(self, extra_env: dict[str, str], worker_args: str = ""):
        repo_root = Path(__file__).resolve().parents[4]
        worker = repo_root / "test/manual/tpu/test_nvfp4_expert_tp_fused_rs.py"
        command = "\n".join(
            (
                "eval $(python3 -m torch.tpu.distributed)",
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                f'torchrun --nproc_per_node="$WORLD_SIZE" {worker} {worker_args}',
            )
        )
        env = os.environ.copy()
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            env.pop(name, None)
        env.update(extra_env)
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"expert-TP fused-RS TP8 test timed out:\n{error.stdout or ''}")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("ETP_FUSED_RS_PASS"), 8, result.stdout)

    def test_matches_reference(self):
        self._run({})

    def test_bf16_weights_match_reference(self):
        """bf16 expert-TP weights must fill every output row (the GMM2 per-row store
        covers only quantized weights)."""
        self._run({}, "--weights bf16")

    def test_pair_exchange_matches_reference(self):
        """4096 tokens reach pair exchange (on-chip pair sums, one more bf16 rounding);
        4100 pad to 4104 rows per rank, which the 256-row combine blocks must not drop
        the tail of."""
        self._run(
            {"SGLANG_TPU_FUSED_RS_ETP_PAIR_EXCHANGE": "1"}, "--tokens 64,4096,4100"
        )


if __name__ == "__main__":
    unittest.main()
