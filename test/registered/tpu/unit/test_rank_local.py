"""Rank-lifted direct compile on every rank of a TPU host.

Runs test/manual/tpu/test_rank_local.py under torchrun.
"""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# Four ranks under torchrun hold every chip of the host (266 s in the PR gate),
# which the stage-a budget cannot fit; run it nightly.
register_tpu_ci(
    est_time=270,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=4,
    labels=["singlehost", "infra", "compile"],
)


class TestRankLocal(CustomTestCase):
    def test_every_rank_masks_experts_with_its_traced_ep_rank(self):
        """Under rank lifting each rank must trace the MoE expert mask with its own
        EP rank, derived from the traced TP rank, for EP == TP and EP < TP."""
        from torch_tpu._internal.utils import hardware

        world_size = hardware.get_tpu_device_count()
        repo_root = Path(__file__).resolve().parents[4]
        worker = repo_root / "test/manual/tpu/test_rank_local.py"
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
            self.fail(f"Rank-local test timed out:\n{error.stdout or ''}")

        self.assertEqual(
            result.returncode, 0, f"Rank-local test failed:\n{result.stdout}"
        )
        for rank in range(world_size):
            for moe_tp in (1, 2):
                # The worker's TP group spans the world with moe_dp 1.
                self.assertIn(
                    f"rank-input moe ep rank: rank={rank} moe_tp={moe_tp} "
                    f"ep_rank={rank // moe_tp} PASS",
                    result.stdout,
                )


if __name__ == "__main__":
    unittest.main()
