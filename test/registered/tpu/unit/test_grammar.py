# SPDX-License-Identifier: Apache-2.0
"""Exercise grammar masks and host token broadcasts on a physical TPU host."""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=120,
    suite="stage-a-test-tpu-unit",
    labels=["singlehost", "infra", "agentic", "tp", "compile"],
)

_WORKER_ARG = "--grammar-worker"


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires torch_tpu")
class TestTpuGrammar(unittest.TestCase):
    def test_mask_compilation_and_token_synchronization(self):
        python = shlex.quote(sys.executable)
        worker = shlex.quote(str(Path(__file__).resolve()))
        command = "\n".join(
            (
                f"eval $({python} -m torch.tpu.distributed)",
                "export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE",
                f'{python} -m torch.distributed.run --nproc_per_node="$WORLD_SIZE" '
                f"{worker} {_WORKER_ARG}",
            )
        )
        env = os.environ.copy()
        for name in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            env.pop(name, None)
        result = subprocess.run(
            ["bash", "-lc", command],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("TPU_GRAMMAR_TEST_PASS", output)


def _run_worker():
    from datetime import timedelta
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from torch_tpu._internal import execution_mode
    from torch_tpu._internal.distributed import tpu_distributed  # noqa: F401

    from sglang_tpu.grammar import apply_vocab_mask, synchronize_token_ids

    dist.init_process_group(backend="tpu_dist", timeout=timedelta(seconds=120))
    world_size, rank = dist.get_world_size(), dist.get_rank()
    if world_size < 2:
        raise AssertionError("Token synchronization requires at least two TPU ranks")
    ranks = list(range(world_size))
    cpu_group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=120))
    group = SimpleNamespace(world_size=world_size, ranks=ranks, cpu_group=cpu_group)

    @torch.compile(dynamic=False, fullgraph=True, backend="eager")
    def decoder(value):
        return value + 1

    decoder(torch.zeros(2, device="tpu"))
    try:
        with torch.compiler.set_stance("fail_on_recompile"):
            # Non-power-of-two batches and vocabularies cover both bit 31 and
            # partially used mask words. Different rows admit different tokens.
            for rows in (1, 3, 2):
                mask = torch.zeros(rows, 3, dtype=torch.int32)
                for row in range(rows):
                    mask[row, 0] = -(1 << 31)
                    mask[row, 2] = 1 << row
                for dtype in (torch.float32, torch.bfloat16):
                    logits = torch.arange(70, dtype=dtype).repeat(rows, 1).to("tpu")
                    apply_vocab_mask(logits, mask.to("tpu"))
                    actual = logits.cpu()
                    for row in range(rows):
                        allowed = torch.zeros(70, dtype=torch.bool)
                        allowed[31] = allowed[64 + row] = True
                        assert torch.equal(torch.isfinite(actual[row]), allowed)
                        assert actual[row, 31].item() == 31
                        assert actual[row, 64 + row].item() == 64 + row
            try:
                decoder(torch.zeros(3, device="tpu"))
            except RuntimeError as error:
                assert "recompile" in str(error)
            else:
                raise AssertionError("The decoder compilation stance was not restored")

        for mode in (
            execution_mode.EagerMode.DEFER_NEVER,
            execution_mode.EagerMode.DEFER_AND_FUSE,
        ):
            with execution_mode.set_eager_mode(mode):
                for rows in (1, 3):
                    for dtype in (torch.int32, torch.int64):
                        # Rank-dependent deferred arithmetic must be materialized
                        # before the CPU broadcast, and the result copied in place.
                        token_ids = torch.full((rows,), rank, dtype=dtype, device="tpu")
                        token_ids = token_ids + 163_700
                        synchronize_token_ids(token_ids, group)
                        torch.testing.assert_close(
                            token_ids.cpu(), torch.full((rows,), 163_700, dtype=dtype)
                        )
        dist.barrier(group=cpu_group)
        if rank == 0:
            print("TPU_GRAMMAR_TEST_PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if _WORKER_ARG in sys.argv:
        _run_worker()
    else:
        unittest.main()
