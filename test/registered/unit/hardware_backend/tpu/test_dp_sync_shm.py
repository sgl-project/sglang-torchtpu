"""The shared-memory DP MLP-sync exchange (SGLANG_TPU_DP_SYNC_SHM) gathers the
same rows, in the same rank order, as the stock gloo exchange."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

WORLD = 4
STEPS = 200
WIDTH = 13


def _row(rank: int, step: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(WIDTH, dtype=dtype) * (rank + 1) + step * 1000 + rank


def _worker(rank: int, init_file: str, dtype: torch.dtype, out_dir: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
    try:
        from sglang.srt.managers.scheduler_components.dp_attn import (
            exchange_sync_rows,
        )
        from sglang_tpu.dp_sync_shm import ShmExchangeRegistry

        group = dist.new_group(list(range(WORLD)), backend="gloo")
        registry = ShmExchangeRegistry()
        mismatches = 0
        for step in range(STEPS):
            local = _row(rank, step, dtype)
            if rank == step % WORLD:
                # Skew one rank per step so peers spin on an unpublished row.
                torch.empty(20000).normal_()
            expected = torch.empty(WORLD * WIDTH, dtype=dtype)
            exchange_sync_rows(expected, local, group=group)
            got = torch.full((WORLD * WIDTH,), -1, dtype=dtype)
            registry.get(group=group, local_row=local).exchange(got, local)
            mismatches += int(not torch.equal(got, expected))
        with open(os.path.join(out_dir, f"{rank}.txt"), "w") as f:
            f.write(str(mismatches))
    finally:
        dist.destroy_process_group()


class TestDpSyncShm(CustomTestCase):
    def _run(self, dtype: torch.dtype) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "init")
            mp.start_processes(
                _worker,
                args=(init_file, dtype, tmp),
                nprocs=WORLD,
                join=True,
                start_method="spawn",
            )
            for rank in range(WORLD):
                with open(os.path.join(tmp, f"{rank}.txt")) as f:
                    self.assertEqual(f.read(), "0", f"rank {rank}")

    def test_matches_gloo(self):
        # MLPSyncBatchInfo rows are int64.
        self._run(torch.int64)

    def test_device_rows_keep_stock_exchange(self):
        from sglang_tpu.dp_sync_shm import ShmExchangeRegistry

        row = torch.zeros(WIDTH, dtype=torch.int64, device="meta")
        self.assertIsNone(ShmExchangeRegistry().get(group=None, local_row=row))


if __name__ == "__main__":
    import unittest

    unittest.main()
