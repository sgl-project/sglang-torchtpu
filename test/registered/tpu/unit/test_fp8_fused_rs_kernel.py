# SPDX-License-Identifier: Apache-2.0

"""Compiled TP8 numerical coverage for the FP8 fused-RS kernel."""

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=155, suite="stage-a-test-tpu-unit", chips=4)


class TestFp8FusedRsKernel(CustomTestCase):
    def _run_compiled_tp8_kernel(
        self,
        *,
        post_expert_norm: bool,
        tokens: int,
        topk: int,
        blockwise: bool = False,
        divergent_ids: bool = False,
        divergent_weights: bool = False,
        fp8_direct_write: bool = False,
        layers: int = 1,
        skewed_routing: bool = False,
    ):
        repo_root = Path(__file__).resolve().parents[4]
        worker = repo_root / "test/manual/tpu/test_fp8_fused_rs_kernel.py"
        wrapper_module = "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
        worker_command = shlex.join(
            [
                str(worker),
                "--layers",
                str(layers),
                *(["--skewed-routing"] if skewed_routing else []),
            ]
        )
        command = "\n".join(
            (
                f"eval $(python3 -m {wrapper_module})",
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                f'torchrun --nproc_per_node="$WORLD_SIZE" {worker_command}',
            )
        )
        env = os.environ.copy()
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            env.pop(name, None)
        env["SGLANG_TPU_FUSED_RS_TEST_POST_EXPERT_NORM"] = str(int(post_expert_norm))
        env["SGLANG_TPU_FUSED_RS_TEST_BLOCKWISE"] = str(int(blockwise))
        env["SGLANG_TPU_FUSED_RS_TEST_TOKENS"] = str(tokens)
        env["SGLANG_TPU_FUSED_RS_TEST_TOPK"] = str(topk)
        env["SGLANG_TPU_FUSED_RS_TEST_DIVERGENT_IDS"] = str(int(divergent_ids))
        env["SGLANG_TPU_FUSED_RS_TEST_DIVERGENT_WEIGHTS"] = str(int(divergent_weights))
        env["SGLANG_TPU_FUSED_RS_TEST_FP8_DIRECT_WRITE"] = str(int(fp8_direct_write))

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
        self.assertEqual(
            result.stdout.count(f"divergent_ids={divergent_ids}"),
            8,
        )
        self.assertEqual(
            result.stdout.count(f"divergent_weights={divergent_weights}"),
            8,
        )
        self.assertEqual(
            result.stdout.count(f"fp8_direct_write={fp8_direct_write}"),
            8,
        )
        self.assertEqual(result.stdout.count(f"layers={layers}"), 8)
        self.assertEqual(
            result.stdout.count(f"skewed_routing={skewed_routing}"),
            8,
        )

    def test_compiled_tp8_kernel_matches_reference(self):
        for post_expert_norm in (False, True):
            with self.subTest(post_expert_norm=post_expert_norm):
                # Exercise the M=2048 boundary used by Qwen3 at 256 tokens.
                self._run_compiled_tp8_kernel(
                    post_expert_norm=post_expert_norm, tokens=256, topk=8
                )

    def test_compiled_tp8_kernel_pads_routed_rows(self):
        self._run_compiled_tp8_kernel(post_expert_norm=False, tokens=8, topk=1)

    def test_compiled_tp8_kernel_synchronizes_divergent_weights(self):
        """Non-source routing weights must not affect fused-RS output."""
        self._run_compiled_tp8_kernel(
            post_expert_norm=False,
            tokens=64,
            topk=2,
            divergent_weights=True,
        )

    def test_compiled_tp8_kernel_synchronizes_divergent_ids(self):
        """Non-source expert IDs must not affect fused-RS output."""
        self._run_compiled_tp8_kernel(
            post_expert_norm=False,
            tokens=64,
            topk=2,
            divergent_ids=True,
        )

    def test_compiled_tp8_block512_kernel_matches_reference(self):
        self._run_compiled_tp8_kernel(
            post_expert_norm=False,
            tokens=8,
            topk=1,
            blockwise=True,
        )

    def test_compiled_tp8_fp8_direct_write_matches_reference(self):
        self._run_compiled_tp8_kernel(
            post_expert_norm=False,
            tokens=8,
            topk=1,
            fp8_direct_write=True,
        )

    def test_compiled_tp8_stacked_skewed_routing_does_not_leak_dma_semaphores(self):
        """Stacked skewed calls must not halt when faster ranks finish first.

        Extreme expert imbalance creates rank skew. Without an exit rendezvous,
        a fast participant can terminate while a peer still owns per-slot DMA
        semaphore state, surfacing as a Mosaic failure or TPU core halt.
        """
        self._run_compiled_tp8_kernel(
            post_expert_norm=False,
            tokens=4096,
            topk=1,
            fp8_direct_write=True,
            layers=10,
            skewed_routing=True,
        )


if __name__ == "__main__":
    unittest.main()
