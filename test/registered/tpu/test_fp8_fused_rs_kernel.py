# SPDX-License-Identifier: Apache-2.0

"""Compiled TP8 numerical coverage for the FP8 fused-RS kernel."""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=600, suite="stage-a-test-tpu-unit")


class TestFp8FusedRsKernel(CustomTestCase):
    def _run_compiled_tp8_kernel(
        self,
        *,
        post_expert_norm: bool,
        tokens: int,
        topk: int,
        blockwise: bool = False,
    ):
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
        env["SGLANG_TPU_FUSED_RS_TEST_POST_EXPERT_NORM"] = str(int(post_expert_norm))
        env["SGLANG_TPU_FUSED_RS_TEST_BLOCKWISE"] = str(int(blockwise))
        env["SGLANG_TPU_FUSED_RS_TEST_TOKENS"] = str(tokens)
        env["SGLANG_TPU_FUSED_RS_TEST_TOPK"] = str(topk)

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
        self.assertEqual(
            result.stdout.count(f"post_norm={post_expert_norm}"),
            8,
        )
        self.assertEqual(result.stdout.count(f"blockwise={blockwise}"), 8)

    def test_compiled_tp8_kernel_matches_reference(self):
        for post_expert_norm in (False, True):
            with self.subTest(post_expert_norm=post_expert_norm):
                # Exercise the M=2048 boundary used by Qwen3 at 256 tokens.
                self._run_compiled_tp8_kernel(
                    post_expert_norm=post_expert_norm, tokens=256, topk=8
                )

    def test_compiled_tp8_kernel_pads_routed_rows(self):
        self._run_compiled_tp8_kernel(post_expert_norm=False, tokens=8, topk=1)

    def test_compiled_tp8_block512_kernel_matches_reference(self):
        self._run_compiled_tp8_kernel(
            post_expert_norm=False,
            tokens=8,
            topk=1,
            blockwise=True,
        )

    def test_compiled_tp8_block512_m16384_matches_reference(self):
        # M = 2,048 padded tokens * top-k 8 = 16,384 routed rows.
        self._run_compiled_tp8_kernel(
            post_expert_norm=True,
            tokens=2048,
            topk=8,
            blockwise=True,
        )


if __name__ == "__main__":
    unittest.main()
